"""
스트리밍 가능한 가속 데모 - 단계별로 진행하면서 실시간 업데이트를 지원합니다.

기존 accelerated_demo.py의 배치 처리 대신, 각 타임스텝 단계별로
진행할 수 있도록 리팩토링한 버전입니다.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from app.backtesting.accelerated_demo import (
    DEMO_TICKERS,
    ChartBar,
    SimulatedTrade,
    generate_synthetic_charts,
    load_global_listed_universe,
    _account_from_state,
    _indicators_at,
    _markets_at,
    _prices_at,
    _to_jsonable,
)
from app.backtesting.time_scaler import TimeMode, TimeScaler, TimeScalerConfig
from app.goals import NegotiatedGoal
from app.graph import OntologyReasoner
from app.graph.builders import build_market_graph
from app.graph.npu_classifier import OntologyNpuStatus, get_ontology_npu_classifier
from app.risk import RiskManager
from app.schemas.domain import OrderAction, OrderSide, RiskRules
from app.trading_pipeline import (
    CandidateSelectionResult,
    build_lightweight_market_snapshots,
    ontology_filter_1,
    universe_from_tickers,
)


@dataclass(frozen=True)
class DemoStepResult:
    """데모의 한 스텝 실행 결과"""
    step_index: int
    visible_step: int
    timestamp: datetime
    virtual_time: datetime
    prices: dict[str, float]
    universe_prices: dict[str, float]
    active_ticker_count: int
    universe_scanned_count: int
    candidate_ticker_count: int
    universe_ticker_count: int
    cash: float
    holdings: dict[str, int]
    trades_in_step: tuple[SimulatedTrade, ...]
    cumulative_trades: int
    account_value: float
    return_rate: float
    progress_percent: float
    ontology_npu: OntologyNpuStatus | None = None


@dataclass
class StreamingAcceleratedDemo:
    """스트리밍 방식의 가속 데모 - 단계별로 진행 가능"""
    
    config: TimeScalerConfig
    target_return_rate: float = 0.02
    period_minutes: int = 390
    initial_cash: float = 10_000_000
    output_dir: Path = Path("data/reports")
    seed: int = 42
    tickers: tuple[str, ...] | None = None
    
    # 내부 상태
    _time_scaler: Optional[TimeScaler] = field(default=None, init=False, repr=False)
    _bars_by_ticker: dict[str, tuple[ChartBar, ...]] = field(default_factory=dict, init=False, repr=False)
    _timestamps: tuple[datetime, ...] = field(default_factory=tuple, init=False, repr=False)
    _cash: float = field(default=0.0, init=False, repr=False)
    _holdings: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _trades: list[SimulatedTrade] = field(default_factory=list, init=False, repr=False)
    _current_step: int = field(default=0, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    _final_prices: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _started_at_monotonic: float = field(default=0.0, init=False, repr=False)
    _candidate_selection: CandidateSelectionResult | None = field(default=None, init=False, repr=False)

    def _warmup_steps(self) -> int:
        return min(15, max(0, len(self._timestamps) - 1))
    
    def initialize(self) -> None:
        """데모를 초기화합니다."""
        if self._initialized:
            return
            
        self._time_scaler = TimeScaler(self.config)
        warmup_steps = 15 if self.period_minutes > 1 else 0
        total_minutes = max(1, self.period_minutes) + warmup_steps
        universe_tickers = self.tickers or load_global_listed_universe()
        configured_limit = os.getenv("SIM_STREAMING_UNIVERSE_LIMIT", "").strip()
        try:
            universe_limit = max(1, int(configured_limit)) if configured_limit else 0
        except ValueError:
            universe_limit = 0
        if universe_limit:
            universe_tickers = universe_tickers[:universe_limit]
        target_count = int(os.getenv("ONTOLOGY_FILTER1_TARGET_COUNT", "80"))
        universe = universe_from_tickers(universe_tickers)
        snapshots = build_lightweight_market_snapshots(universe, seed=self.seed)
        self._candidate_selection = ontology_filter_1(
            snapshots,
            target_count=target_count,
            cache_key=f"streaming:{self.seed}:{len(universe_tickers)}:{target_count}",
        )
        tickers = self._candidate_selection.candidate_stocks or tuple(universe_tickers[: min(20, len(universe_tickers))])
        self._bars_by_ticker = generate_synthetic_charts(tickers, total_minutes, self.seed)
        self._timestamps = tuple(bar.timestamp for bar in self._bars_by_ticker[tickers[0]])
        self._cash = self.initial_cash
        self._holdings = {}
        self._trades = []
        self._current_step = 0
        self._initialized = True
        self._started_at_monotonic = time.monotonic()
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_step(self) -> Optional[DemoStepResult]:
        """
        한 스텝을 실행합니다.
        
        Returns:
            DemoStepResult: 스텝 결과 (진행 완료시 None)
        """
        if not self._initialized:
            self.initialize()
        
        step = self._current_step
        warmup_steps = self._warmup_steps()
        
        # 시작 전 일부 스텝은 스킵 (워밍업)
        if step < warmup_steps:
            self._current_step += 1
            # 재귀 호출로 다음 단계 진행
            return self.run_step()
        
        # 종료 조건
        if step >= len(self._timestamps):
            return None
        
        timestamp = self._timestamps[step]
        
        # 가상 시간 업데이트
        virtual_time = self._time_scaler.get_virtual_time()
        
        prices = _prices_at(self._bars_by_ticker, step)
        account = _account_from_state(self._cash, self._holdings, prices, timestamp)
        universe_tickers = tuple(self._bars_by_ticker)
        universe_charts = {ticker: self._bars_by_ticker[ticker] for ticker in universe_tickers}
        universe_markets = _markets_at(universe_charts, step)
        universe_indicators = _indicators_at(universe_charts, step)
        npu_classifier = get_ontology_npu_classifier()
        npu_scores = npu_classifier.classify(universe_markets, universe_indicators)
        candidate_tickers = self._candidate_tickers(universe_markets, npu_scores)
        active_charts = {ticker: self._bars_by_ticker[ticker] for ticker in candidate_tickers}
        markets = tuple(market for market in universe_markets if market.ticker in set(candidate_tickers))
        indicators = {ticker: universe_indicators[ticker] for ticker in candidate_tickers if ticker in universe_indicators}
        
        # 그래프 구축 및 추론
        graph = build_market_graph(markets, indicators, npu_scores=npu_scores)
        OntologyReasoner(graph).infer()
        
        # 전략 실행
        from app.strategy import build_goal_execution_plan
        period_days = max(1, math.ceil(self.period_minutes / 390))
        goal = NegotiatedGoal(
            target_return_rate=self.target_return_rate,
            target_profit_amount=self.initial_cash * self.target_return_rate,
            period_days=period_days,
            feasibility_percent=68,
            label="Accelerated demo target",
        )
        plan = build_goal_execution_plan(goal, account, markets, indicators, graph)
        
        market_by_ticker = {market.ticker: market for market in markets}
        pending: set[str] = set()
        rules = RiskRules(
            max_single_stock_weight=0.06,
            max_sector_weight=0.55,
            minimum_cash_reserve=0.08,
            max_trades_per_day=80,
            min_average_daily_trading_value=max(1_000.0, self.initial_cash * 0.02),
            max_volatility=0.12,
        )
        
        # 이번 스텝의 거래
        step_trades: list[SimulatedTrade] = []
        
        ranked_intents = sorted(
            plan.intents,
            key=lambda intent: (
                0 if intent.action in {OrderAction.SELL, OrderAction.REDUCE} else 1,
                -intent.confidence,
            ),
        )
        
        for intent in ranked_intents[:10]:
            if intent.ticker in pending:
                continue
            market = market_by_ticker[intent.ticker]
            current_account = _account_from_state(self._cash, self._holdings, prices, timestamp)
            result = RiskManager(rules).validate(
                intent,
                current_account,
                market,
                trades_today=0,
                existing_pending_tickers=pending,
            )
            if not result.approved or result.final_order is None:
                continue
            
            order = result.final_order
            value = order.quantity * order.limit_price
            
            if order.side == OrderSide.BUY and self._cash >= value:
                self._cash -= value
                self._holdings[order.ticker] = self._holdings.get(order.ticker, 0) + order.quantity
            elif order.side == OrderSide.SELL:
                owned = self._holdings.get(order.ticker, 0)
                quantity = min(owned, order.quantity)
                if quantity <= 0:
                    continue
                self._cash += quantity * order.limit_price
                self._holdings[order.ticker] = owned - quantity
                if self._holdings[order.ticker] <= 0:
                    del self._holdings[order.ticker]
                value = quantity * order.limit_price
            else:
                continue
            
            pending.add(order.ticker)
            trade = SimulatedTrade(
                timestamp=timestamp,
                ticker=order.ticker,
                side=order.side.value,
                quantity=order.quantity,
                price=order.limit_price,
                value=value,
                reason="; ".join(intent.reasoning_summary),
            )
            step_trades.append(trade)
            self._trades.append(trade)
        
        if step >= len(self._timestamps) - 1:
            step_trades.extend(self._liquidate_holdings(prices, timestamp))

        # 현재 계정 가치 계산
        current_positions = {
            ticker: quantity * prices[ticker]
            for ticker, quantity in self._holdings.items()
            if quantity > 0
        }
        account_value = self._cash + sum(current_positions.values())
        return_rate = (account_value - self.initial_cash) / self.initial_cash
        
        # 진행률 계산
        progress_percent = ((step - warmup_steps + 1) / max(1, len(self._timestamps) - warmup_steps)) * 100
        visible_step = step - warmup_steps + 1
        
        self._current_step += 1
        
        result = DemoStepResult(
            step_index=step,
            visible_step=visible_step,
            timestamp=timestamp,
            virtual_time=virtual_time,
            prices={ticker: round(prices[ticker], 2) for ticker in candidate_tickers if ticker in prices},
            universe_prices={ticker: round(price, 2) for ticker, price in prices.items()},
            active_ticker_count=len(candidate_tickers),
            universe_scanned_count=len(universe_tickers),
            candidate_ticker_count=len(candidate_tickers),
            universe_ticker_count=len(self._bars_by_ticker),
            cash=round(self._cash, 2),
            holdings=dict(self._holdings),
            trades_in_step=tuple(step_trades),
            cumulative_trades=len(self._trades),
            account_value=round(account_value, 2),
            return_rate=round(return_rate, 6),
            progress_percent=round(progress_percent, 1),
            ontology_npu=npu_classifier.status(),
        )
        
        return result

    def _active_tickers(self, step: int, warmup_steps: int) -> tuple[str, ...]:
        tickers = tuple(self._bars_by_ticker)
        if not tickers:
            return ()
        batch_size = max(1, int(os.getenv("SIM_STEP_TICKER_BATCH", "500")))
        if batch_size >= len(tickers):
            return tickers
        visible_index = max(0, step - warmup_steps)
        start = (visible_index * batch_size) % len(tickers)
        selected = [tickers[(start + offset) % len(tickers)] for offset in range(batch_size)]
        for ticker in self._holdings:
            if ticker in self._bars_by_ticker and ticker not in selected:
                selected.append(ticker)
        return tuple(selected)

    def _candidate_tickers(
        self,
        markets: tuple[Any, ...],
        npu_scores: dict[str, tuple[float, ...]],
    ) -> tuple[str, ...]:
        limit = max(100, int(os.getenv("SIM_STRATEGY_CANDIDATES", "1800")))
        ranked = sorted(
            markets,
            key=lambda market: npu_scores.get(market.ticker, (0, 0, 0, 0, 0, 0))[5],
            reverse=True,
        )
        selected = [market.ticker for market in ranked[: min(limit, len(ranked))]]
        for ticker in self._holdings:
            if ticker in self._bars_by_ticker and ticker not in selected:
                selected.append(ticker)
        return tuple(selected)

    def _liquidate_holdings(
        self,
        prices: dict[str, float],
        timestamp: datetime,
    ) -> list[SimulatedTrade]:
        trades: list[SimulatedTrade] = []
        for ticker, quantity in list(self._holdings.items()):
            if quantity <= 0:
                self._holdings.pop(ticker, None)
                continue
            price = float(prices.get(ticker, 0.0) or 0.0)
            if price <= 0:
                continue
            value = quantity * price
            self._cash += value
            del self._holdings[ticker]
            trade = SimulatedTrade(
                timestamp=timestamp,
                ticker=ticker,
                side=OrderSide.SELL.value,
                quantity=quantity,
                price=price,
                value=value,
                reason="mandatory final liquidation",
            )
            trades.append(trade)
            self._trades.append(trade)
        return trades

    
    def run_all_steps(self) -> list[DemoStepResult]:
        """모든 스텝을 실행하고 결과를 반환합니다."""
        results = []
        while True:
            result = self.run_step()
            if result is None:
                break
            results.append(result)
        return results

    def seconds_until_next_step(self) -> float:
        """Return wall-clock seconds until the next visible simulation minute may run."""
        if not self._initialized:
            self.initialize()
        if self.is_complete():
            return 0.0

        warmup_steps = self._warmup_steps()
        visible_steps_completed = max(0, self._current_step - warmup_steps)
        scale_factor = self._time_scaler.get_scale_factor() if self._time_scaler else 1.0
        step_interval_seconds = 60.0 / max(1.0, scale_factor)
        next_due_seconds = (visible_steps_completed + 1) * step_interval_seconds
        elapsed_seconds = time.monotonic() - self._started_at_monotonic
        return max(0.0, next_due_seconds - elapsed_seconds)
    
    def is_complete(self) -> bool:
        """데모 진행이 완료되었는지 확인합니다."""
        if not self._initialized:
            return False
        return self._current_step >= len(self._timestamps)
    
    def get_progress(self) -> float:
        """진행률을 0~100으로 반환합니다."""
        if not self._initialized:
            return 0.0
        warmup_steps = self._warmup_steps()
        if len(self._timestamps) <= warmup_steps:
            return 100.0
        progress = ((self._current_step - warmup_steps) / (len(self._timestamps) - warmup_steps)) * 100
        return min(100.0, max(0.0, progress))
    
    def get_final_results(self) -> Optional[dict]:
        """최종 결과를 반환합니다 (완료 후에만 호출 가능)."""
        if not self.is_complete():
            return None
        
        final_prices = _prices_at(self._bars_by_ticker, len(self._timestamps) - 1)
        if self._holdings:
            self._liquidate_holdings(final_prices, self._timestamps[-1])
        final_positions = {
            ticker: quantity * final_prices[ticker]
            for ticker, quantity in self._holdings.items()
            if quantity > 0
        }
        final_equity = self._cash + sum(final_positions.values())
        profit = final_equity - self.initial_cash
        return_rate = profit / self.initial_cash
        
        return {
            "initial_equity": round(self.initial_cash, 2),
            "final_equity": round(final_equity, 2),
            "profit_amount": round(profit, 2),
            "return_rate": round(return_rate, 6),
            "target_return_rate": self.target_return_rate,
            "target_profit_amount": round(self.initial_cash * self.target_return_rate, 2),
            "target_achieved": return_rate >= self.target_return_rate,
            "simulated_minutes": self.period_minutes,
            "bars_per_ticker": len(self._timestamps),
            "ticker_count": len(self._bars_by_ticker),
            "accelerated_steps": len(self._timestamps),
            "trade_count": len(self._trades),
            "final_cash": round(self._cash, 2),
            "final_positions": {key: round(value, 2) for key, value in sorted(final_positions.items())},
            "sample_trades": [_to_jsonable(t) for t in self._trades[:20]],
        }
    
    def get_time_scaler(self) -> Optional[TimeScaler]:
        """시간 스케일러를 반환합니다."""
        return self._time_scaler

    def get_candidate_selection(self) -> CandidateSelectionResult | None:
        return self._candidate_selection
    
    def pause(self) -> None:
        """가상 시간을 일시 정지합니다."""
        if self._time_scaler:
            self._time_scaler.pause()
    
    def resume(self) -> None:
        """가상 시간을 다시 시작합니다."""
        if self._time_scaler:
            self._time_scaler.resume()

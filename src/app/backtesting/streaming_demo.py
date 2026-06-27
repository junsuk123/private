"""
스트리밍 가능한 가속 데모 - 단계별로 진행하면서 실시간 업데이트를 지원합니다.

기존 accelerated_demo.py의 배치 처리 대신, 각 타임스텝 단계별로
진행할 수 있도록 리팩토링한 버전입니다.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field, replace
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
    _simulated_trade_cost,
    _to_jsonable,
)
from app.cost import TradingCostEngine
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

BASE_CURRENCY = "KRW"
USD_CURRENCY = "USD"
DEFAULT_USD_KRW_RATE = 1350.0


def _usd_krw_rate() -> float:
    raw = os.getenv("SIM_USD_KRW_RATE", "").strip()
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return DEFAULT_USD_KRW_RATE


def _currency_for_ticker(ticker: str, market: str | None = None) -> str:
    normalized_market = (market or "").upper()
    normalized_ticker = ticker.upper()
    if normalized_market in {"KRX", "KOSPI", "KOSDAQ", "KONEX"}:
        return BASE_CURRENCY
    if normalized_ticker.endswith(".KS") or normalized_ticker.endswith(".KQ"):
        return BASE_CURRENCY
    if normalized_ticker.replace(".", "").isdigit():
        return BASE_CURRENCY
    return USD_CURRENCY


def _to_krw(value: float, currency: str, usd_krw_rate: float) -> float:
    return value * usd_krw_rate if currency == USD_CURRENCY else value


@dataclass(frozen=True)
class PrincipalProtectionState:
    enabled: bool
    protected_principal: float
    principal_locked: bool
    cycle_index: int
    cycle_seed: float
    cycle_start_equity: float
    target_profit_amount: float
    target_equity: float
    remaining_to_target: float


@dataclass(frozen=True)
class ProfitGainState:
    target_return_rate: float
    period_minutes: int
    pressure: float
    gain: float
    max_single_stock_weight: float
    minimum_cash_reserve: float
    fast_take_profit: float
    fast_stop_loss: float
    fast_sell_fraction: float
    max_trades_per_day: int


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
    base_currency: str = BASE_CURRENCY
    cash_by_currency: dict[str, float] = field(default_factory=dict)
    account_value_krw: float = 0.0
    usd_krw_rate: float = DEFAULT_USD_KRW_RATE
    currency_by_ticker: dict[str, str] = field(default_factory=dict)
    principal_protection: PrincipalProtectionState | None = None
    profit_gain: ProfitGainState | None = None
    ontology_npu: OntologyNpuStatus | None = None


@dataclass
class StreamingAcceleratedDemo:
    """스트리밍 방식의 가속 데모 - 단계별로 진행 가능"""
    
    config: TimeScalerConfig
    target_return_rate: float = 0.02
    period_minutes: int = 390
    initial_cash: float = 10_000_000
    principal_protection_enabled: bool = True
    profit_gain_multiplier: float = 1.0
    output_dir: Path = Path("data/reports")
    seed: int = 42
    tickers: tuple[str, ...] | None = None
    
    # 내부 상태
    _time_scaler: Optional[TimeScaler] = field(default=None, init=False, repr=False)
    _bars_by_ticker: dict[str, tuple[ChartBar, ...]] = field(default_factory=dict, init=False, repr=False)
    _timestamps: tuple[datetime, ...] = field(default_factory=tuple, init=False, repr=False)
    _cash: float = field(default=0.0, init=False, repr=False)
    _cash_by_currency: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _holdings: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _average_cost_by_ticker: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _holding_currency_by_ticker: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _trades: list[SimulatedTrade] = field(default_factory=list, init=False, repr=False)
    _current_step: int = field(default=0, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    _final_prices: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _started_at_monotonic: float = field(default=0.0, init=False, repr=False)
    _candidate_selection: CandidateSelectionResult | None = field(default=None, init=False, repr=False)
    _principal_locked: bool = field(default=False, init=False, repr=False)
    _capital_cycle_index: int = field(default=1, init=False, repr=False)
    _capital_cycle_seed: float = field(default=0.0, init=False, repr=False)
    _capital_cycle_start_equity: float = field(default=0.0, init=False, repr=False)
    _cost_engine: TradingCostEngine = field(default_factory=TradingCostEngine, init=False, repr=False)

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
        if not self.tickers and len(universe_tickers) < 5:
            universe_tickers = DEMO_TICKERS
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
        tickers = tuple(universe_tickers) if self.tickers else self._candidate_selection.candidate_stocks or tuple(universe_tickers[: min(20, len(universe_tickers))])
        if not self.tickers and len(tickers) < 5:
            seen = set(tickers)
            tickers = tuple(tickers) + tuple(ticker for ticker in DEMO_TICKERS if ticker not in seen)[: 5 - len(tickers)]
        self._bars_by_ticker = generate_synthetic_charts(tickers, total_minutes, self.seed)
        self._timestamps = tuple(bar.timestamp for bar in self._bars_by_ticker[tickers[0]])
        self._cash = self.initial_cash
        self._cash_by_currency = {BASE_CURRENCY: float(self.initial_cash), USD_CURRENCY: 0.0}
        self._holdings = {}
        self._average_cost_by_ticker = {}
        self._holding_currency_by_ticker = {}
        self._trades = []
        self._current_step = 0
        self._principal_locked = False
        self._capital_cycle_index = 1
        self._capital_cycle_seed = float(self.initial_cash)
        self._capital_cycle_start_equity = float(self.initial_cash)
        self._cost_engine = TradingCostEngine()
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
        usd_krw_rate = _usd_krw_rate()
        currency_by_ticker = self._currency_by_ticker()
        self._cash = self._cash_equivalent_krw(usd_krw_rate)
        account = self._account_snapshot(prices, timestamp, usd_krw_rate, currency_by_ticker)
        universe_tickers = tuple(self._bars_by_ticker)
        universe_charts = {ticker: self._bars_by_ticker[ticker] for ticker in universe_tickers}
        universe_markets = _markets_at(universe_charts, step)
        universe_markets_base = self._markets_in_base_currency(universe_markets, usd_krw_rate, currency_by_ticker)
        universe_indicators = _indicators_at(universe_charts, step)
        npu_classifier = get_ontology_npu_classifier()
        npu_scores = npu_classifier.classify(universe_markets, universe_indicators)
        candidate_tickers = self._candidate_tickers(universe_markets, npu_scores)
        active_charts = {ticker: self._bars_by_ticker[ticker] for ticker in candidate_tickers}
        markets = tuple(market for market in universe_markets_base if market.ticker in set(candidate_tickers))
        indicators = {ticker: universe_indicators[ticker] for ticker in candidate_tickers if ticker in universe_indicators}
        
        # 그래프 구축 및 추론
        graph = build_market_graph(markets, indicators, npu_scores=npu_scores)
        OntologyReasoner(graph).infer()
        
        # 전략 실행
        from app.strategy import build_goal_execution_plan
        period_days = max(1, math.ceil(self.period_minutes / 390))
        dynamic_target_profit = self._current_target_profit_amount()
        profit_gain = self._profit_gain_state()
        dynamic_target_rate = min(0.08 + profit_gain.gain * 0.04, dynamic_target_profit / max(1.0, account.equity))
        goal = NegotiatedGoal(
            target_return_rate=dynamic_target_rate,
            target_profit_amount=dynamic_target_profit,
            period_days=period_days,
            feasibility_percent=68,
            label="Principal-preserving compounding target",
        )
        plan = build_goal_execution_plan(goal, account, markets, indicators, graph)
        
        market_by_ticker = {market.ticker: market for market in markets}
        pending: set[str] = set()
        rules = RiskRules(
            max_single_stock_weight=profit_gain.max_single_stock_weight,
            max_sector_weight=1.00,
            minimum_cash_reserve=max(0.05, profit_gain.minimum_cash_reserve) if self._principal_locked else profit_gain.minimum_cash_reserve,
            max_trades_per_day=profit_gain.max_trades_per_day,
            min_average_daily_trading_value=max(1_000.0, self.initial_cash * 0.02),
            max_volatility=0.28,
            max_intraday_position_weight=profit_gain.max_single_stock_weight,
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
        
        intent_limit = max(10, int(12 + profit_gain.gain * 10))
        for intent in ranked_intents[:intent_limit]:
            if intent.ticker in pending:
                continue
            market = market_by_ticker[intent.ticker]
            current_account = self._account_snapshot(prices, timestamp, usd_krw_rate, currency_by_ticker)
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
            currency = _currency_for_ticker(order.ticker, order.market)
            fx_rate = usd_krw_rate if currency == USD_CURRENCY else 1.0
            native_price = order.limit_price / fx_rate
            value = order.quantity * native_price
            trading_cost = _simulated_trade_cost(
                self._cost_engine,
                order.side,
                order.ticker,
                native_price,
                order.quantity,
                currency,
            )
            
            if order.side == OrderSide.BUY and self._ensure_cash_for_buy(currency, value + trading_cost, usd_krw_rate):
                executed_quantity = order.quantity
                self._cash_by_currency[currency] = self._cash_by_currency.get(currency, 0.0) - value - trading_cost
                previous_quantity = self._holdings.get(order.ticker, 0)
                previous_cost = self._average_cost_by_ticker.get(order.ticker, native_price)
                new_quantity = previous_quantity + order.quantity
                self._holdings[order.ticker] = new_quantity
                self._average_cost_by_ticker[order.ticker] = (
                    (previous_cost * previous_quantity) + value
                ) / max(1, new_quantity)
                self._holding_currency_by_ticker[order.ticker] = currency
            elif order.side == OrderSide.SELL:
                owned = self._holdings.get(order.ticker, 0)
                quantity = min(owned, order.quantity)
                if quantity <= 0:
                    continue
                executed_quantity = quantity
                currency = self._holding_currency_by_ticker.get(order.ticker, currency)
                fx_rate = usd_krw_rate if currency == USD_CURRENCY else 1.0
                native_price = order.limit_price / fx_rate
                value = quantity * native_price
                trading_cost = _simulated_trade_cost(
                    self._cost_engine,
                    order.side,
                    order.ticker,
                    native_price,
                    quantity,
                    currency,
                )
                self._cash_by_currency[currency] = self._cash_by_currency.get(currency, 0.0) + value - trading_cost
                self._holdings[order.ticker] = owned - quantity
                if self._holdings[order.ticker] <= 0:
                    del self._holdings[order.ticker]
                    self._average_cost_by_ticker.pop(order.ticker, None)
                    self._holding_currency_by_ticker.pop(order.ticker, None)
            else:
                continue
            self._cash = self._cash_equivalent_krw(usd_krw_rate)
            
            pending.add(order.ticker)
            trade = SimulatedTrade(
                timestamp=timestamp,
                ticker=order.ticker,
                side=order.side.value,
                quantity=executed_quantity,
                price=native_price,
                value=value,
                reason="; ".join(intent.reasoning_summary),
                currency=currency,
                fx_rate=fx_rate,
                value_krw=round(_to_krw(value, currency, usd_krw_rate), 2),
                trading_cost=round(trading_cost, 4),
                net_value=round(value + trading_cost if order.side == OrderSide.BUY else value - trading_cost, 4),
            )
            step_trades.append(trade)
            self._trades.append(trade)

        fast_trades = self._harvest_fast_trading_profits(prices, timestamp, profit_gain)
        if fast_trades:
            step_trades.extend(fast_trades)
        
        if step >= len(self._timestamps) - 1:
            step_trades.extend(self._liquidate_holdings(prices, timestamp))

        # 현재 계정 가치 계산
        current_positions = {
            ticker: _to_krw(
                quantity * prices[ticker],
                self._holding_currency_by_ticker.get(ticker, currency_by_ticker.get(ticker, USD_CURRENCY)),
                usd_krw_rate,
            )
            for ticker, quantity in self._holdings.items()
            if quantity > 0
        }
        self._cash = self._cash_equivalent_krw(usd_krw_rate)
        account_value = self._cash + sum(current_positions.values())
        self._advance_capital_cycle(account_value)
        reserve_trades = self._raise_protected_cash_floor(prices, timestamp)
        if reserve_trades:
            step_trades.extend(reserve_trades)
            current_positions = {
                ticker: _to_krw(
                    quantity * prices[ticker],
                    self._holding_currency_by_ticker.get(ticker, currency_by_ticker.get(ticker, USD_CURRENCY)),
                    usd_krw_rate,
                )
                for ticker, quantity in self._holdings.items()
                if quantity > 0
            }
            self._cash = self._cash_equivalent_krw(usd_krw_rate)
            account_value = self._cash + sum(current_positions.values())
            self._advance_capital_cycle(account_value)
        protection = self._principal_protection_state(account_value)
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
            cash_by_currency={key: round(value, 2) for key, value in sorted(self._cash_by_currency.items())},
            account_value_krw=round(account_value, 2),
            usd_krw_rate=round(usd_krw_rate, 4),
            currency_by_ticker=currency_by_ticker,
            principal_protection=protection,
            profit_gain=profit_gain,
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
            currency = self._holding_currency_by_ticker.get(ticker, _currency_for_ticker(ticker))
            usd_krw_rate = _usd_krw_rate()
            fx_rate = usd_krw_rate if currency == USD_CURRENCY else 1.0
            trading_cost = _simulated_trade_cost(self._cost_engine, OrderSide.SELL, ticker, price, quantity, currency)
            self._cash_by_currency[currency] = self._cash_by_currency.get(currency, 0.0) + value - trading_cost
            self._cash = self._cash_equivalent_krw(usd_krw_rate)
            del self._holdings[ticker]
            self._average_cost_by_ticker.pop(ticker, None)
            self._holding_currency_by_ticker.pop(ticker, None)
            trade = SimulatedTrade(
                timestamp=timestamp,
                ticker=ticker,
                side=OrderSide.SELL.value,
                quantity=quantity,
                price=price,
                value=value,
                reason="mandatory final liquidation",
                currency=currency,
                fx_rate=fx_rate,
                value_krw=round(_to_krw(value, currency, usd_krw_rate), 2),
                trading_cost=round(trading_cost, 4),
                net_value=round(value - trading_cost, 4),
            )
            trades.append(trade)
            self._trades.append(trade)
        return trades

    def _harvest_fast_trading_profits(
        self,
        prices: dict[str, float],
        timestamp: datetime,
        profit_gain: ProfitGainState | None = None,
    ) -> list[SimulatedTrade]:
        profit_gain = profit_gain or self._profit_gain_state()
        take_profit = float(os.getenv("SIM_FAST_TAKE_PROFIT", str(profit_gain.fast_take_profit)))
        stop_loss = float(os.getenv("SIM_FAST_STOP_LOSS", str(profit_gain.fast_stop_loss)))
        sell_fraction = float(os.getenv("SIM_FAST_SELL_FRACTION", str(profit_gain.fast_sell_fraction)))
        max_sales = max(1, int(os.getenv("SIM_FAST_MAX_SALES_PER_STEP", str(max(1, int(2 + profit_gain.gain * 3))))))
        usd_krw_rate = _usd_krw_rate()
        trades: list[SimulatedTrade] = []

        ranked = sorted(
            (
                (
                    (float(prices.get(ticker, 0.0) or 0.0) - self._average_cost_by_ticker.get(ticker, 0.0))
                    / max(0.0001, self._average_cost_by_ticker.get(ticker, 0.0)),
                    ticker,
                    quantity,
                )
                for ticker, quantity in self._holdings.items()
                if quantity > 0 and self._average_cost_by_ticker.get(ticker, 0.0) > 0
            ),
            reverse=True,
        )

        for pnl_rate, ticker, owned in ranked:
            if len(trades) >= max_sales:
                break
            if pnl_rate < take_profit and pnl_rate > -stop_loss:
                continue
            price = float(prices.get(ticker, 0.0) or 0.0)
            if price <= 0:
                continue
            if pnl_rate >= take_profit:
                quantity = min(owned, max(1, math.ceil(owned * sell_fraction)))
                reason = f"fast take-profit {pnl_rate * 100:.2f}%"
            else:
                quantity = min(owned, max(1, math.ceil(owned * max(0.5, sell_fraction))))
                reason = f"fast stop-loss {pnl_rate * 100:.2f}%"

            currency = self._holding_currency_by_ticker.get(ticker, _currency_for_ticker(ticker))
            value = quantity * price
            trading_cost = _simulated_trade_cost(self._cost_engine, OrderSide.SELL, ticker, price, quantity, currency)
            self._cash_by_currency[currency] = self._cash_by_currency.get(currency, 0.0) + value - trading_cost
            remaining = owned - quantity
            if remaining > 0:
                self._holdings[ticker] = remaining
            else:
                self._holdings.pop(ticker, None)
                self._average_cost_by_ticker.pop(ticker, None)
                self._holding_currency_by_ticker.pop(ticker, None)

            trade = SimulatedTrade(
                timestamp=timestamp,
                ticker=ticker,
                side=OrderSide.SELL.value,
                quantity=quantity,
                price=price,
                value=value,
                reason=reason,
                currency=currency,
                fx_rate=usd_krw_rate if currency == USD_CURRENCY else 1.0,
                value_krw=round(_to_krw(value, currency, usd_krw_rate), 2),
                trading_cost=round(trading_cost, 4),
                net_value=round(value - trading_cost, 4),
            )
            trades.append(trade)
            self._trades.append(trade)

        self._cash = self._cash_equivalent_krw(usd_krw_rate)
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
            ticker: _to_krw(
                quantity * final_prices[ticker],
                self._holding_currency_by_ticker.get(ticker, _currency_for_ticker(ticker)),
                _usd_krw_rate(),
            )
            for ticker, quantity in self._holdings.items()
            if quantity > 0
        }
        self._cash = self._cash_equivalent_krw(_usd_krw_rate())
        final_equity = self._cash + sum(final_positions.values())
        self._advance_capital_cycle(final_equity)
        protection = self._principal_protection_state(final_equity)
        profit = final_equity - self.initial_cash
        return_rate = profit / self.initial_cash
        
        return {
            "initial_equity": round(self.initial_cash, 2),
            "final_equity": round(final_equity, 2),
            "profit_amount": round(profit, 2),
            "return_rate": round(return_rate, 6),
            "target_return_rate": self.target_return_rate,
            "target_profit_amount": round(protection.target_profit_amount, 2),
            "target_achieved": final_equity >= protection.target_equity,
            "principal_protection": _to_jsonable(protection),
            "simulated_minutes": self.period_minutes,
            "bars_per_ticker": len(self._timestamps),
            "ticker_count": len(self._bars_by_ticker),
            "accelerated_steps": len(self._timestamps),
            "trade_count": len(self._trades),
            "final_cash": round(self._cash, 2),
            "final_cash_by_currency": {key: round(value, 2) for key, value in sorted(self._cash_by_currency.items())},
            "base_currency": BASE_CURRENCY,
            "usd_krw_rate": round(_usd_krw_rate(), 4),
            "final_positions": {key: round(value, 2) for key, value in sorted(final_positions.items())},
            "sample_trades": [_to_jsonable(t) for t in self._trades[:20]],
        }
    
    def get_time_scaler(self) -> Optional[TimeScaler]:
        """시간 스케일러를 반환합니다."""
        return self._time_scaler

    def get_candidate_selection(self) -> CandidateSelectionResult | None:
        return self._candidate_selection

    def _currency_by_ticker(self) -> dict[str, str]:
        return {
            ticker: self._holding_currency_by_ticker.get(ticker, _currency_for_ticker(ticker))
            for ticker in self._bars_by_ticker
        }

    def _markets_in_base_currency(
        self,
        markets: tuple[Any, ...],
        usd_krw_rate: float,
        currency_by_ticker: dict[str, str],
    ) -> tuple[Any, ...]:
        converted = []
        for market in markets:
            currency = currency_by_ticker.get(market.ticker, _currency_for_ticker(market.ticker, market.market))
            converted.append(
                replace(
                    market,
                    last_price=_to_krw(market.last_price, currency, usd_krw_rate),
                    average_daily_trading_value=_to_krw(
                        market.average_daily_trading_value,
                        currency,
                        usd_krw_rate,
                    ),
                )
            )
        return tuple(converted)

    def _cash_equivalent_krw(self, usd_krw_rate: float) -> float:
        return sum(_to_krw(value, currency, usd_krw_rate) for currency, value in self._cash_by_currency.items())

    def _profit_gain_state(self) -> ProfitGainState:
        period_minutes = max(1, int(self.period_minutes))
        target_rate = max(0.0, float(self.target_return_rate))
        trading_day_fraction = max(1.0 / 390.0, period_minutes / 390.0)
        pressure = target_rate / trading_day_fraction
        raw_gain = 1.0 + pressure * 2.0
        env_multiplier = float(os.getenv("SIM_PROFIT_GAIN", "1.0"))
        user_multiplier = max(0.25, min(4.0, float(self.profit_gain_multiplier)))
        gain = max(0.65, min(3.0, raw_gain * env_multiplier))
        gain = max(0.65, min(4.0, gain * user_multiplier))
        max_weight = max(0.18, min(0.65, 0.18 + gain * 0.14))
        cash_reserve = max(0.005, min(0.08, 0.08 / gain))
        take_profit = max(0.0015, 0.006 / gain)
        stop_loss = max(0.004, min(0.018, 0.006 + gain * 0.003))
        sell_fraction = max(0.25, min(0.85, 0.25 + gain * 0.14))
        max_trades = int(max(60, min(260, 60 + gain * 60)))
        return ProfitGainState(
            target_return_rate=round(target_rate, 6),
            period_minutes=period_minutes,
            pressure=round(pressure, 6),
            gain=round(gain, 4),
            max_single_stock_weight=round(max_weight, 4),
            minimum_cash_reserve=round(cash_reserve, 4),
            fast_take_profit=round(take_profit, 6),
            fast_stop_loss=round(stop_loss, 6),
            fast_sell_fraction=round(sell_fraction, 4),
            max_trades_per_day=max_trades,
        )

    def _ensure_cash_for_buy(self, currency: str, amount: float, usd_krw_rate: float) -> bool:
        protected_floor = self._protected_cash_floor()
        if protected_floor > 0:
            spend_krw = _to_krw(amount, currency, usd_krw_rate)
            if self._cash_equivalent_krw(usd_krw_rate) - spend_krw < protected_floor:
                return False
        available = self._cash_by_currency.get(currency, 0.0)
        if available >= amount:
            return True
        if currency != USD_CURRENCY:
            return False

        needed_usd = amount - available
        needed_krw = needed_usd * usd_krw_rate
        krw_cash = self._cash_by_currency.get(BASE_CURRENCY, 0.0)
        if krw_cash < needed_krw:
            return False
        self._cash_by_currency[BASE_CURRENCY] = krw_cash - needed_krw
        self._cash_by_currency[USD_CURRENCY] = available + needed_usd
        return True

    def _current_target_profit_amount(self) -> float:
        if self.principal_protection_enabled:
            return max(1.0, self._capital_cycle_seed)
        return max(1.0, self.initial_cash * self.target_return_rate)

    def _advance_capital_cycle(self, account_value: float) -> None:
        if not self.principal_protection_enabled:
            return
        while account_value >= self._capital_cycle_start_equity + self._capital_cycle_seed:
            achieved_profit = self._capital_cycle_seed
            self._principal_locked = True
            self._capital_cycle_index += 1
            self._capital_cycle_seed += achieved_profit
            self._capital_cycle_start_equity += achieved_profit

    def _protected_cash_floor(self) -> float:
        if not self.principal_protection_enabled or not self._principal_locked:
            return 0.0
        return float(self.initial_cash)

    def _raise_protected_cash_floor(
        self,
        prices: dict[str, float],
        timestamp: datetime,
    ) -> list[SimulatedTrade]:
        protected_floor = self._protected_cash_floor()
        if protected_floor <= 0:
            return []
        usd_krw_rate = _usd_krw_rate()
        cash_equivalent = self._cash_equivalent_krw(usd_krw_rate)
        if cash_equivalent >= protected_floor:
            return []

        trades: list[SimulatedTrade] = []
        for ticker, owned in sorted(list(self._holdings.items())):
            if owned <= 0:
                continue
            currency = self._holding_currency_by_ticker.get(ticker, _currency_for_ticker(ticker))
            price = float(prices.get(ticker, 0.0) or 0.0)
            if price <= 0:
                continue
            price_krw = _to_krw(price, currency, usd_krw_rate)
            needed_krw = protected_floor - cash_equivalent
            quantity = min(owned, max(1, math.ceil(needed_krw / max(1.0, price_krw))))
            value = quantity * price
            trading_cost = _simulated_trade_cost(self._cost_engine, OrderSide.SELL, ticker, price, quantity, currency)
            net_value_krw = _to_krw(value - trading_cost, currency, usd_krw_rate)
            while quantity < owned and net_value_krw < needed_krw:
                quantity += 1
                value = quantity * price
                trading_cost = _simulated_trade_cost(self._cost_engine, OrderSide.SELL, ticker, price, quantity, currency)
                net_value_krw = _to_krw(value - trading_cost, currency, usd_krw_rate)
            self._cash_by_currency[currency] = self._cash_by_currency.get(currency, 0.0) + value - trading_cost
            remaining = owned - quantity
            if remaining > 0:
                self._holdings[ticker] = remaining
            else:
                self._holdings.pop(ticker, None)
                self._holding_currency_by_ticker.pop(ticker, None)

            trade = SimulatedTrade(
                timestamp=timestamp,
                ticker=ticker,
                side=OrderSide.SELL.value,
                quantity=quantity,
                price=price,
                value=value,
                reason="principal protection reserve",
                currency=currency,
                fx_rate=usd_krw_rate if currency == USD_CURRENCY else 1.0,
                value_krw=round(_to_krw(value, currency, usd_krw_rate), 2),
                trading_cost=round(trading_cost, 4),
                net_value=round(value - trading_cost, 4),
            )
            trades.append(trade)
            self._trades.append(trade)
            cash_equivalent = self._cash_equivalent_krw(usd_krw_rate)
            if cash_equivalent >= protected_floor:
                break
        self._cash = self._cash_equivalent_krw(usd_krw_rate)
        return trades

    def _principal_protection_state(self, account_value: float) -> PrincipalProtectionState:
        target_profit = self._current_target_profit_amount()
        target_equity = self._capital_cycle_start_equity + target_profit
        return PrincipalProtectionState(
            enabled=self.principal_protection_enabled,
            protected_principal=round(self.initial_cash if self._principal_locked else 0.0, 2),
            principal_locked=self._principal_locked,
            cycle_index=self._capital_cycle_index,
            cycle_seed=round(self._capital_cycle_seed, 2),
            cycle_start_equity=round(self._capital_cycle_start_equity, 2),
            target_profit_amount=round(target_profit, 2),
            target_equity=round(target_equity, 2),
            remaining_to_target=round(max(0.0, target_equity - account_value), 2),
        )

    def _account_snapshot(
        self,
        prices: dict[str, float],
        timestamp: datetime,
        usd_krw_rate: float,
        currency_by_ticker: dict[str, str],
    ):
        converted_prices = {
            ticker: _to_krw(price, currency_by_ticker.get(ticker, _currency_for_ticker(ticker)), usd_krw_rate)
            for ticker, price in prices.items()
        }
        return _account_from_state(
            self._cash_equivalent_krw(usd_krw_rate),
            self._holdings,
            converted_prices,
            timestamp,
        )
    
    def pause(self) -> None:
        """가상 시간을 일시 정지합니다."""
        if self._time_scaler:
            self._time_scaler.pause()
    
    def resume(self) -> None:
        """가상 시간을 다시 시작합니다."""
        if self._time_scaler:
            self._time_scaler.resume()

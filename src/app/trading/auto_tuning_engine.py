from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from app.cost import TradingCostEngine
from app.schemas.domain import AccountSnapshot, MarketSnapshot, RiskRules

from .adaptive_exit_policy import AdaptiveExitPolicy, derive_exit_policy
from .decision_logger import DecisionLogger
from .execution_policy import ExecutionPolicy
from .market_regime import estimate_market_regime
from .model_health import assess_model_health


@dataclass(frozen=True)
class MarketStateSnapshot:
    symbol: str
    last_price: float
    spread_bps: float
    liquidity: float
    volatility: float
    quote_age_seconds: float
    orderbook_available: bool
    volume_ratio: float
    recent_performance: float = 0.0
    fallback_score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class AutoTuningEngine:
    def __init__(
        self,
        *,
        cost_engine: TradingCostEngine | None = None,
        decision_logger: DecisionLogger | None = None,
        refresh_quote: Callable[[str, str, datetime], MarketSnapshot | None] | None = None,
    ) -> None:
        self.cost_engine = cost_engine or TradingCostEngine()
        self.decision_logger = decision_logger or DecisionLogger()
        self.refresh_quote = refresh_quote
        self._recent_results: deque[dict[str, Any]] = deque(maxlen=60)

    def snapshot_market_state(
        self,
        *,
        symbol: str,
        market: MarketSnapshot,
        quote_age_seconds: float,
        spread_bps: float,
        orderbook_available: bool,
        volume_ratio: float = 1.0,
        recent_performance: float = 0.0,
        fallback_score: float = 0.0,
    ) -> MarketStateSnapshot:
        return MarketStateSnapshot(
            symbol=symbol,
            last_price=max(0.0, float(market.last_price)),
            spread_bps=max(0.0, float(spread_bps)),
            liquidity=max(0.0, float(market.average_daily_trading_value)),
            volatility=max(0.0, float(market.volatility_20d)),
            quote_age_seconds=max(0.0, float(quote_age_seconds)),
            orderbook_available=bool(orderbook_available),
            volume_ratio=max(0.0, float(volume_ratio)),
            recent_performance=float(recent_performance),
            fallback_score=float(fallback_score),
        )

    def build_buy_policy(
        self,
        *,
        symbol: str,
        account: AccountSnapshot,
        market: MarketSnapshot,
        market_state: MarketStateSnapshot,
        prediction: Any | None,
        fallback_allowed: bool,
        ontology_score: float,
        fallback_score: float,
        prediction_confidence: float,
        prediction_error: Exception | None = None,
        recent_performance: float | None = None,
        decision_time: datetime | None = None,
    ) -> tuple[ExecutionPolicy, dict[str, Any]]:
        decision_time = decision_time or datetime.now(timezone.utc)
        recent_performance = market_state.recent_performance if recent_performance is None else recent_performance
        model_ok = bool(prediction is not None and bool(getattr(prediction, "approved", False)))
        model_health = assess_model_health(
            prediction=prediction,
            prediction_error=prediction_error,
            fallback_allowed=fallback_allowed,
        )
        regime = estimate_market_regime(
            volatility=market_state.volatility,
            liquidity=market_state.liquidity,
            spread_bps=market_state.spread_bps,
            recent_performance=recent_performance,
            model_ok=model_ok,
        )
        base_cost = self.cost_engine.policy_for(
            venue="KRX" if market.market.upper().startswith("K") else "NASD",
            instrument_type="domestic_stock" if market.market.upper().startswith("K") else "overseas_stock",
        )
        equity = max(1.0, float(account.equity or 0.0))
        price = max(0.01, float(market.last_price or market_state.last_price or 0.01))
        risk_budget_ratio = 0.010
        if regime.regime.value == "capital_protection":
            risk_budget_ratio = 0.008
        elif regime.regime.value == "conservative":
            risk_budget_ratio = 0.012
        elif regime.regime.value == "aggressive":
            risk_budget_ratio = 0.015
        if fallback_score > 0:
            risk_budget_ratio *= max(0.70, min(1.15, fallback_score))
        if ontology_score > 0:
            risk_budget_ratio *= min(1.3, 1.0 + ontology_score * 0.15)
        if not model_ok:
            risk_budget_ratio *= 0.8
        available_cash = max(0.0, float(account.cash or 0.0))
        one_share_price = price
        min_cash_for_one_share = one_share_price * 1.05
        equity_based_budget = max(equity * risk_budget_ratio * 10.0, min_cash_for_one_share)
        cash_allocation_ratio = min(0.70, max(0.30, risk_budget_ratio * 40.0))
        cash_budget = max(equity_based_budget, available_cash * cash_allocation_ratio)
        max_position_size = max(1, int(min(available_cash, cash_budget) / price))

        confidence_floor = 0.45 if model_ok else 0.35
        confidence_floor += 0.05 if regime.regime.value == "aggressive" else 0.0
        confidence_floor -= 0.05 if regime.regime.value == "capital_protection" else 0.0
        confidence_floor = max(0.25, min(0.85, confidence_floor - model_health.confidence_penalty))

        buy_threshold = 0.42
        buy_threshold += market_state.spread_bps / 120.0
        buy_threshold += market_state.volatility * 4.0
        buy_threshold -= min(0.2, math.log1p(max(0.0, market_state.liquidity)) / 40.0)
        buy_threshold -= min(0.20, max(0.0, ontology_score) * 0.12)
        buy_threshold -= min(0.12, max(0.0, fallback_score) * 0.12)
        buy_threshold -= 0.08 if model_ok else 0.0
        buy_threshold += 0.06 if not model_ok and not fallback_allowed else 0.0
        buy_threshold = max(0.22, min(0.75, buy_threshold))

        expected_net_return = max(0.0005, float(market_state.fallback_score) * 0.008)
        if model_ok and prediction is not None:
            expected_net_return = max(expected_net_return, float(getattr(prediction, "expected_net_return_bps", 0.0) or 0.0) / 10_000.0)

        max_spread_bps = max(8.0, min(70.0, 22.0 + market_state.volatility * 1_500 + (12.0 if regime.regime.value == "capital_protection" else 0.0)))
        max_slippage_bps = max(8.0, min(80.0, 20.0 + market_state.volatility * 1_700))
        quote_ttl_seconds = max(2.0, min(20.0, 12.0 - market_state.spread_bps / 8.0 - market_state.volatility * 40.0 + market_state.volume_ratio * 1.5))
        time_exit_seconds = max(60, min(7_200, int(300 + market_state.volatility * 12_000 + (market_state.quote_age_seconds * 10))))
        sell_target = max(base_cost.safety_margin_rate, 0.0005 + market_state.volatility * 0.1)
        stop_loss = max(0.002, min(0.05, 0.008 + market_state.volatility * 1.2))
        trailing_stop = max(0.001, min(stop_loss, stop_loss * 0.7 + market_state.volatility * 0.3))
        allowed_fallback_mode = "model"
        if not model_ok and fallback_allowed:
            allowed_fallback_mode = "ontology_only" if ontology_score > 0 else "rule_based"
        if market_state.quote_age_seconds > quote_ttl_seconds * 2:
            allowed_fallback_mode = "no_trade"

        risk_mode = str(regime.regime.value)
        if market_state.quote_age_seconds > quote_ttl_seconds * 2 or market_state.spread_bps > max_spread_bps:
            risk_mode = "capital_protection"

        diagnostics = {
            "symbol": symbol,
            "decision_time": decision_time.isoformat(),
            "regime": regime.as_dict(),
            "model_health": model_health.as_dict(),
            "market_state": market_state.as_dict(),
            "account_equity": round(equity, 4),
            "cash_budget": round(cash_budget, 4),
            "base_cost": asdict(base_cost),
            "recent_performance": round(float(recent_performance), 6),
        }
        policy = ExecutionPolicy(
            buy_threshold=round(buy_threshold, 4),
            sell_target=round(max(0.0, sell_target), 6),
            stop_loss=round(stop_loss, 6),
            trailing_stop=round(trailing_stop, 6),
            max_position_size=max_position_size,
            quote_ttl_seconds=round(quote_ttl_seconds, 3),
            min_expected_net_return=round(max(0.0, expected_net_return), 6),
            max_spread_bps=round(max_spread_bps, 3),
            max_slippage_bps=round(max_slippage_bps, 3),
            allowed_fallback_mode=allowed_fallback_mode,
            time_exit_seconds=time_exit_seconds,
            confidence_floor=round(confidence_floor, 4),
            risk_mode=risk_mode,
            diagnostics=diagnostics,
        )
        return policy, diagnostics

    def build_exit_policy(
        self,
        *,
        symbol: str,
        holding,
        account: AccountSnapshot,
        market: MarketSnapshot,
        market_state: MarketStateSnapshot,
        take_profit: float,
        stop_loss: float,
        ontology_score: float,
        target_net_return: float,
        decision_time: datetime | None = None,
    ) -> tuple[ExecutionPolicy, AdaptiveExitPolicy, dict[str, Any]]:
        decision_time = decision_time or datetime.now(timezone.utc)
        exit_policy, cost_floor = derive_exit_policy(
            holding=holding,
            account=account,
            market=market,
            take_profit=take_profit,
            stop_loss=stop_loss,
            ontology_score=ontology_score,
            decision_time=decision_time,
            target_net_return=target_net_return,
            cost_engine=self.cost_engine,
        )
        policy = ExecutionPolicy(
            buy_threshold=0.0,
            sell_target=round(exit_policy.sell_target, 6),
            stop_loss=round(exit_policy.stop_loss, 6),
            trailing_stop=round(exit_policy.trailing_stop, 6),
            max_position_size=max(1, int(getattr(holding, "quantity", 0) or 0)),
            quote_ttl_seconds=max(2.0, min(20.0, 10.0 - market_state.spread_bps / 10.0)),
            min_expected_net_return=round(exit_policy.min_expected_net_return, 6),
            max_spread_bps=max(8.0, min(80.0, market_state.spread_bps * 2.5 + 20.0)),
            max_slippage_bps=max(8.0, min(80.0, market_state.spread_bps * 2.0 + 16.0)),
            allowed_fallback_mode="rule_based" if exit_policy.allow_loss_exit else "ontology_only",
            time_exit_seconds=exit_policy.time_exit_seconds,
            confidence_floor=exit_policy.confidence_floor,
            risk_mode=exit_policy.exit_mode,
            diagnostics={
                "exit_policy": exit_policy.as_dict(),
                "cost_floor": cost_floor.as_dict(),
                "market_state": market_state.as_dict(),
                "symbol": symbol,
            },
        )
        diagnostics = {"exit_policy": exit_policy.as_dict(), "cost_floor": cost_floor.as_dict(), "market_state": market_state.as_dict()}
        return policy, exit_policy, diagnostics

    def derive_risk_rules(
        self,
        base_rules: RiskRules,
        *,
        policy: ExecutionPolicy,
        account: AccountSnapshot,
        market: MarketSnapshot,
        model_uncertainty: float | None = None,
    ) -> RiskRules:
        equity = max(1.0, float(account.equity or 0.0))
        price = max(0.01, float(market.last_price or 0.01))
        max_single_stock_weight = max(base_rules.max_single_stock_weight, min(0.20, policy.max_position_size * price / equity))
        max_intraday_position_weight = max(base_rules.max_intraday_position_weight, min(0.10, policy.max_position_size * price / equity))
        max_volatility = max(base_rules.max_volatility, max(0.02, market.volatility_20d * 2.5))
        min_average_daily_trading_value = max(1_000.0, min(base_rules.min_average_daily_trading_value, market.average_daily_trading_value * 0.25))
        max_quote_age_seconds = max(base_rules.max_quote_age_seconds, policy.quote_ttl_seconds)
        max_model_uncertainty = base_rules.max_model_uncertainty
        if policy.allowed_fallback_mode != "model":
            max_model_uncertainty = max(max_model_uncertainty, 0.95)
        if model_uncertainty is not None:
            max_model_uncertainty = max(max_model_uncertainty, min(0.95, model_uncertainty + 0.15))
        minimum_cash_reserve = min(base_rules.minimum_cash_reserve, 0.10 if policy.risk_mode != "capital_protection" else 0.20)
        return RiskRules(
            max_single_stock_weight=max_single_stock_weight,
            max_sector_weight=base_rules.max_sector_weight,
            minimum_cash_reserve=minimum_cash_reserve,
            daily_loss_stop=base_rules.daily_loss_stop,
            max_trades_per_day=base_rules.max_trades_per_day,
            min_average_daily_trading_value=min_average_daily_trading_value,
            max_volatility=max_volatility,
            order_type=base_rules.order_type,
            manual_approval_required=base_rules.manual_approval_required,
            live_trading_enabled=base_rules.live_trading_enabled,
            margin_trading_allowed=base_rules.margin_trading_allowed,
            short_selling_allowed=base_rules.short_selling_allowed,
            derivatives_allowed=base_rules.derivatives_allowed,
            leverage_etf_allowed=base_rules.leverage_etf_allowed,
            credit_loan_allowed=base_rules.credit_loan_allowed,
            llm_direct_order_execution_allowed=base_rules.llm_direct_order_execution_allowed,
            max_intraday_position_weight=max_intraday_position_weight,
            max_short_horizon_downside_risk=base_rules.max_short_horizon_downside_risk,
            emergency_exit_loss=base_rules.emergency_exit_loss,
            min_source_trust_level=base_rules.min_source_trust_level,
            min_data_quality_score=base_rules.min_data_quality_score,
            max_quote_age_seconds=max_quote_age_seconds,
            max_model_uncertainty=max_model_uncertainty,
            synthetic_live_data_allowed=base_rules.synthetic_live_data_allowed,
            unknown_source_live_allowed=base_rules.unknown_source_live_allowed,
            principal_protection=base_rules.principal_protection,
        )

    def fallback_buy_score(
        self,
        *,
        ontology_score: float,
        technical_momentum: float,
        liquidity_score: float,
        spread_bps: float,
        volatility: float,
        recent_performance: float,
    ) -> float:
        score = 0.2
        score += max(-1.0, min(1.0, ontology_score)) * 0.45
        score += max(-1.0, min(1.0, technical_momentum)) * 0.15
        score += max(0.0, min(1.0, liquidity_score)) * 0.15
        score += max(-1.0, min(1.0, recent_performance)) * 0.05
        score -= max(0.0, spread_bps) / 300.0
        score -= max(0.0, volatility) * 1.0
        return max(0.0, min(1.0, score))

    def record_feedback(self, payload: dict[str, Any]) -> None:
        self._recent_results.append(payload)
        self.decision_logger.record("decision_feedback", payload)

    def recent_win_rate(self) -> float:
        if not self._recent_results:
            return 0.5
        wins = 0
        count = 0
        for item in self._recent_results:
            pnl = float(item.get("pnl", 0.0) or 0.0)
            count += 1
            if pnl > 0:
                wins += 1
        return wins / count if count else 0.5

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from math import floor
from typing import Any

from app.cost import CostBreakdown
from app.schemas.domain import (
    AccountSnapshot,
    MarketSnapshot,
    OrderAction,
    OrderIntent,
    PrincipalProtectionConfig,
    PrincipalProtectionDecision,
    PrincipalProtectionDecisionAction,
    PrincipalProtectionMode,
    PrincipalProtectionState,
)


class PrincipalProtectionEngine:
    """Deterministic hard gate for principal protection and profit-only sizing."""

    def compute_state(
        self,
        account_state: AccountSnapshot,
        positions: tuple[Any, ...] | None = None,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        config: PrincipalProtectionConfig | None = None,
        high_watermark: float | None = None,
    ) -> PrincipalProtectionState:
        del positions
        config = config or PrincipalProtectionConfig()
        equity = max(0.0, float(account_state.equity))
        initial_principal = max(0.0, float(config.initial_principal))
        if not config.enabled or initial_principal <= 0:
            return PrincipalProtectionState(
                initial_principal=initial_principal,
                current_equity=equity,
                protected_floor=0.0,
                high_watermark=equity,
                locked_profit=0.0,
                cushion=equity,
                risk_budget=equity,
                available_growth_capital=equity,
                current_mode=PrincipalProtectionMode.NOT_CONFIGURED,
                floor_breach_status=False,
                drawdown_from_high_watermark=0.0,
                cost_buffer=0.0,
                gap_risk_buffer=0.0,
                active_risky_exposure=account_state.invested_value,
                reason_codes=("PRINCIPAL_PROTECTION_NOT_CONFIGURED",),
            )

        high = self.update_high_watermark(equity, high_watermark, initial_principal)
        floor_value = self.compute_protected_floor(initial_principal, high, config)
        cost_buffer = equity * max(0.0, config.cost_buffer_ratio)
        gap_risk_buffer = account_state.invested_value * max(0.0, config.max_gap_loss_assumption)
        cushion = self.compute_cushion(equity, floor_value, cost_buffer, gap_risk_buffer)
        risk_budget = self.compute_risk_budget(cushion, equity, initial_principal, config)
        locked_profit = (
            max(0.0, high - initial_principal) * max(0.0, config.profit_lockin_ratio)
            if config.profit_lockin_enabled
            else 0.0
        )
        realized_growth = max(0.0, realized_pnl)
        provisional_growth = max(0.0, unrealized_pnl) if config.count_unrealized_profit_as_growth else 0.0
        profit_growth_capital = max(0.0, realized_growth + provisional_growth - locked_profit)
        available_growth_capital = min(cushion, profit_growth_capital)
        drawdown = (high - equity) / high if high > 0 else 0.0
        floor_breached = equity <= floor_value
        reasons: list[str] = []
        if floor_breached:
            reasons.append("PROTECTED_FLOOR_BREACHED")
        if drawdown >= config.max_total_drawdown:
            reasons.append("MAX_DRAWDOWN_EXCEEDED")
        if cushion <= 0:
            reasons.append("NO_PROTECTION_CUSHION")
        mode = self.get_mode(
            floor_breached=floor_breached,
            drawdown=drawdown,
            cushion=cushion,
            available_growth_capital=available_growth_capital,
            config=config,
        )
        return PrincipalProtectionState(
            initial_principal=round(initial_principal, 4),
            current_equity=round(equity, 4),
            protected_floor=round(floor_value, 4),
            high_watermark=round(high, 4),
            locked_profit=round(locked_profit, 4),
            cushion=round(cushion, 4),
            risk_budget=round(risk_budget, 4),
            available_growth_capital=round(available_growth_capital, 4),
            current_mode=mode,
            floor_breach_status=floor_breached,
            drawdown_from_high_watermark=round(drawdown, 6),
            cost_buffer=round(cost_buffer, 4),
            gap_risk_buffer=round(gap_risk_buffer, 4),
            active_risky_exposure=round(account_state.invested_value, 4),
            reason_codes=tuple(reasons),
        )

    @staticmethod
    def update_high_watermark(
        equity: float,
        previous_high_watermark: float | None = None,
        initial_principal: float = 0.0,
    ) -> float:
        return max(float(equity), float(previous_high_watermark or 0.0), float(initial_principal))

    @staticmethod
    def compute_protected_floor(
        initial_principal: float,
        high_watermark: float,
        config: PrincipalProtectionConfig,
    ) -> float:
        base_floor = initial_principal * max(0.0, config.principal_floor_ratio)
        if not config.principal_floor_enabled:
            base_floor = 0.0
        locked_profit_floor = 0.0
        if config.profit_lockin_enabled:
            locked_profit_floor = max(0.0, high_watermark - initial_principal) * max(0.0, config.profit_lockin_ratio)
        return base_floor + locked_profit_floor

    @staticmethod
    def compute_cushion(equity: float, protected_floor: float, cost_buffer: float, gap_risk_buffer: float) -> float:
        return max(0.0, equity - protected_floor - cost_buffer - gap_risk_buffer)

    @staticmethod
    def compute_risk_budget(cushion: float, equity: float, initial_principal: float, config: PrincipalProtectionConfig) -> float:
        if cushion <= 0:
            return 0.0
        cppi_budget = config.cppi_multiplier * cushion if config.cppi_enabled else cushion
        daily_budget = max(0.0, initial_principal * config.daily_risk_budget_ratio)
        weekly_budget = max(0.0, initial_principal * config.weekly_risk_budget_ratio)
        equity_budget = max(0.0, equity * config.daily_risk_budget_ratio)
        candidates = [cppi_budget, daily_budget, weekly_budget, equity_budget]
        if config.fractional_kelly_enabled:
            candidates.append(cushion * max(0.0, min(1.0, config.fractional_kelly_ratio)))
        if config.cvar_enabled:
            candidates.append(cushion * max(0.0, 1.0 - min(0.999, max(0.0, config.cvar_confidence))))
        return max(0.0, min(value for value in candidates if value >= 0))

    @staticmethod
    def get_mode(
        *,
        floor_breached: bool,
        drawdown: float,
        cushion: float,
        available_growth_capital: float,
        config: PrincipalProtectionConfig,
    ) -> PrincipalProtectionMode:
        if config.principal_lockdown_enabled and (floor_breached or cushion <= 0):
            return PrincipalProtectionMode.PRINCIPAL_LOCKDOWN
        if drawdown >= config.max_total_drawdown:
            return PrincipalProtectionMode.DE_RISK
        if available_growth_capital > 0:
            return PrincipalProtectionMode.PROFIT_ONLY
        return PrincipalProtectionMode.NORMAL_GROWTH

    def validate_order(
        self,
        order_intent: OrderIntent,
        account_state: AccountSnapshot,
        positions: tuple[Any, ...] | None,
        market_snapshot: MarketSnapshot,
        cost_estimate: CostBreakdown | None,
        config: PrincipalProtectionConfig | None = None,
        proposed_quantity: int | None = None,
        high_watermark: float | None = None,
    ) -> PrincipalProtectionDecision:
        config = config or PrincipalProtectionConfig()
        state = self.compute_state(
            account_state,
            positions,
            realized_pnl=account_state.realized_pnl_today,
            unrealized_pnl=account_state.unrealized_pnl_today,
            config=config,
            high_watermark=high_watermark,
        )
        if state.current_mode == PrincipalProtectionMode.NOT_CONFIGURED:
            return _decision(PrincipalProtectionDecisionAction.ALLOW, state, ("PRINCIPAL_PROTECTION_NOT_CONFIGURED",), ("Initial principal is not configured; existing risk rules remain active.",))
        if order_intent.action in {OrderAction.SELL, OrderAction.REDUCE}:
            return _decision(PrincipalProtectionDecisionAction.ALLOW, state, ("RISK_REDUCING_ACTION_ALLOWED",), ("Sell or reduce actions remain available under principal protection.",))
        if order_intent.action != OrderAction.BUY:
            return _decision(PrincipalProtectionDecisionAction.ALLOW, state, ("NON_BUY_ACTION_ALLOWED",), ("No new risky exposure is introduced.",))
        if state.current_mode == PrincipalProtectionMode.PRINCIPAL_LOCKDOWN:
            return _decision(PrincipalProtectionDecisionAction.LOCKDOWN, state, ("PRINCIPAL_LOCKDOWN_BUY_BLOCKED", *state.reason_codes), ("BUY orders are blocked because equity is at or below the protected floor or cushion is zero.",))
        if state.current_mode == PrincipalProtectionMode.DE_RISK:
            return _decision(PrincipalProtectionDecisionAction.SELL_ONLY, state, ("DE_RISK_BUY_BLOCKED", *state.reason_codes), ("BUY orders are restricted while drawdown control is active.",))

        quantity = max(0, int(proposed_quantity or 0))
        stop_price = _stop_loss_price(order_intent, market_snapshot, config)
        estimated_loss = self.estimated_trade_loss(quantity, market_snapshot.last_price, stop_price, cost_estimate)
        per_trade_budget = min(
            state.risk_budget,
            max(0.0, state.initial_principal * config.per_trade_risk_budget_ratio),
            max(0.0, state.available_growth_capital),
        )
        if quantity <= 0:
            return _decision(PrincipalProtectionDecisionAction.BLOCK, state, ("QUANTITY_NOT_POSITIVE",), ("No positive quantity can be validated.",), estimated_loss)
        if estimated_loss <= per_trade_budget and estimated_loss <= state.cushion:
            return _decision(PrincipalProtectionDecisionAction.ALLOW, state, ("PRINCIPAL_PROTECTION_PASSED",), ("Estimated downside loss fits inside the protected risk budget.",), estimated_loss)

        suggested = self.suggest_max_position_size(order_intent, stop_price, per_trade_budget, cost_estimate, config, market_snapshot)
        if suggested > 0 and suggested < quantity:
            return _decision(
                PrincipalProtectionDecisionAction.REDUCE_SIZE,
                state,
                ("PRINCIPAL_PROTECTION_REDUCE_SIZE",),
                ("Proposed quantity is too large for the downside risk budget, but a smaller size is allowed.",),
                estimated_loss,
                suggested,
            )
        return _decision(
            PrincipalProtectionDecisionAction.BLOCK,
            state,
            ("PRINCIPAL_PROTECTION_RISK_BUDGET_EXCEEDED",),
            ("Estimated downside loss exceeds the principal-protection risk budget.",),
            estimated_loss,
        )

    def suggest_max_position_size(
        self,
        order_intent: OrderIntent,
        stop_loss_price: float,
        risk_budget: float,
        cost_estimate: CostBreakdown | None,
        config: PrincipalProtectionConfig,
        market_snapshot: MarketSnapshot,
    ) -> int:
        del order_intent, config
        entry_price = max(0.0, float(market_snapshot.last_price))
        per_share_loss = max(0.0, entry_price - stop_loss_price)
        if cost_estimate and cost_estimate.quantity > 0:
            per_share_loss += cost_estimate.total_cost / max(1, cost_estimate.quantity)
        else:
            per_share_loss += entry_price * 0.001
        if per_share_loss <= 0:
            return 0
        return max(0, floor(max(0.0, risk_budget) / per_share_loss))

    @staticmethod
    def estimated_trade_loss(quantity: int, entry_price: float, stop_price: float, cost_estimate: CostBreakdown | None) -> float:
        quantity = max(0, int(quantity))
        price_loss = quantity * max(0.0, float(entry_price) - float(stop_price))
        cost_loss = cost_estimate.total_cost if cost_estimate is not None else quantity * max(0.0, float(entry_price)) * 0.001
        return max(0.0, price_loss + cost_loss)


def _stop_loss_price(
    order_intent: OrderIntent,
    market_snapshot: MarketSnapshot,
    config: PrincipalProtectionConfig,
) -> float:
    metadata = order_intent.strategy_metadata or {}
    raw = metadata.get("stop_loss_price") or metadata.get("stop_price")
    try:
        stop = float(raw)
    except (TypeError, ValueError):
        stop = 0.0
    entry = max(0.0, float(market_snapshot.last_price))
    if stop <= 0 or stop >= entry:
        stop = entry * (1.0 - max(0.0, min(0.95, config.max_gap_loss_assumption)))
    return max(0.0, stop)


def _decision(
    action: PrincipalProtectionDecisionAction,
    state: PrincipalProtectionState,
    reason_codes: tuple[str, ...],
    explanations: tuple[str, ...],
    estimated_trade_loss: float = 0.0,
    suggested_quantity: int | None = None,
) -> PrincipalProtectionDecision:
    return PrincipalProtectionDecision(
        action=action,
        state=state,
        allowed=action in {PrincipalProtectionDecisionAction.ALLOW, PrincipalProtectionDecisionAction.REDUCE_SIZE},
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        explanations=explanations,
        estimated_trade_loss=round(estimated_trade_loss, 4),
        suggested_quantity=suggested_quantity,
        max_risky_exposure=state.risk_budget,
    )


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from app.strategy.ownership import OwnershipGuard, PositionOwnershipError
from app.trading.contracts import (
    AccountSnapshot,
    IntentAction,
    OrderIntent,
    RiskVerdict,
    RiskVerdictAction,
)


@dataclass(frozen=True)
class CausalRiskLimits:
    maximum_order_notional: float
    maximum_symbol_quantity: int
    minimum_cash_reserve: float = 0.0


class CausalRiskGate:
    def __init__(self, limits: CausalRiskLimits) -> None:
        self.limits = limits
        self.ownership = OwnershipGuard()

    def evaluate(
        self,
        intent: OrderIntent,
        account: AccountSnapshot,
        *,
        price: float,
        timestamp: datetime | None = None,
    ) -> RiskVerdict:
        now = timestamp or datetime.now(timezone.utc)
        reasons: list[str] = []
        checks: dict[str, bool | float | int | str] = {}
        approved = intent.quantity
        if now > intent.expires_at:
            reasons.append("INTENT_EXPIRED")
        if price <= 0:
            reasons.append("INVALID_PRICE")
        if intent.action == IntentAction.BUY:
            currency = next(iter(account.cash_by_currency), "")
            cash = float(account.cash_by_currency.get(currency, 0.0))
            affordable = int(
                max(0.0, cash - self.limits.minimum_cash_reserve) // max(price, 1e-9)
            )
            notional_limit = int(
                self.limits.maximum_order_notional // max(price, 1e-9)
            )
            existing = sum(
                position.quantity
                for position in account.positions
                if position.symbol == intent.symbol
            )
            symbol_limit = max(0, self.limits.maximum_symbol_quantity - existing)
            approved = min(approved, affordable, notional_limit, symbol_limit)
            checks.update(
                {
                    "cash_available": cash,
                    "affordable_quantity": affordable,
                    "notional_limit_quantity": notional_limit,
                    "symbol_limit_quantity": symbol_limit,
                }
            )
            if approved <= 0:
                reasons.append("NO_BUY_CAPACITY")
        elif intent.action == IntentAction.SELL:
            position = next(
                (
                    value
                    for value in account.positions
                    if value.position_id == intent.position_id_if_any
                ),
                None,
            )
            if position is None:
                reasons.append("POSITION_NOT_FOUND")
                approved = 0
            else:
                try:
                    self.ownership.assert_owner(
                        position, intent.strategy_instance_id
                    )
                except PositionOwnershipError:
                    reasons.append("POSITION_OWNER_MISMATCH")
                    approved = 0
                approved = min(approved, max(0, position.quantity))
                checks["broker_position_quantity"] = position.quantity
        else:
            reasons.append("INTENT_ACTION_NOT_RISK_EVALUABLE")
            approved = 0
        action = (
            RiskVerdictAction.REJECT
            if reasons
            else RiskVerdictAction.RESIZE
            if approved < intent.quantity
            else RiskVerdictAction.APPROVE
        )
        return RiskVerdict(
            verdict_id=f"verdict-{uuid4().hex}",
            intent_id=intent.intent_id,
            action=action,
            approved_quantity=0 if action == RiskVerdictAction.REJECT else approved,
            limits_evaluated=checks,
            reason_codes=tuple(reasons),
            account_snapshot_id=account.snapshot_id,
            timestamp=now,
        )

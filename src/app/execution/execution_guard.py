"""ExecutionGuard: can this order physically be sent, and nothing else.

The one rule
------------
**The guard never judges investment quality.** Not the expected return, not the strategy's
confidence, not portfolio concentration, not sector weight, not model uncertainty, not a
volatility or liquidity threshold, not ontology support, not the strategy's rank. Every
one of those was decided before election and is frozen in the
:class:`~app.trading.trade_plan.TradePlan`; re-deciding any of them here would restore the
duplicate-veto stack this refactor exists to remove, at the worst possible place — after
the strategy has committed and while the market is moving.

What it does check is whether the broker can accept the order at all:

* the plan exists, is not expired and is not terminal
* price > 0 and quantity > 0
* the symbol / exchange / product is one this account can trade
* the market is currently orderable
* the quote and book are fresh enough to construct a limit order
* a BUY still has the cash the broker will demand
* a SELL still has the quantity the broker will let go
* a SHORT still has a physical borrow
* no duplicate or in-flight order, idempotency intact
* the kill switch is off and KIS auth / account / order endpoints are healthy

Exits
-----
An exit is subject only to the checks that would make it *unroutable*. A stale feed, an
un-reconciled account or a closed new-entry window never blocks a risk-reducing exit — the
sell policy is that exit speed wins once the strategy's exit condition fires, and a guard
that could trap a losing position would be the most expensive kind of safety.

Failure
-------
Fail closed, with an explicit reason code. Every reason names the check that produced it,
so a blocked order is diagnosable from the code alone.

Relationship to :mod:`app.execution.pre_submit_guard`
-----------------------------------------------------
That module owns the four checks it already does well — session, order state, data
freshness, account reconciliation — and this one composes it rather than reimplementing
it. ExecutionGuard adds the plan, order-shape, affordability and broker-health checks and
is the single object the coordinator holds.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from app.execution.pre_submit_guard import PreSubmitGuard, default_pre_submit_guard

__all__ = [
    "ExecutionGuard",
    "ExecutionGuardDecision",
    "GuardOrder",
    "CASH_UNKNOWN",
    "FORBIDDEN_INVESTMENT_CHECKS",
    "default_execution_guard",
]

#: Named here so the prohibition is executable rather than a comment: a test asserts the
#: guard's decision never carries one of these as a reason code.
FORBIDDEN_INVESTMENT_CHECKS: tuple[str, ...] = (
    "EXPECTED_RETURN",
    "STRATEGY_CONFIDENCE",
    "PORTFOLIO_CONCENTRATION",
    "SECTOR_WEIGHT",
    "MODEL_UNCERTAINTY",
    "VOLATILITY_THRESHOLD",
    "LIQUIDITY_THRESHOLD",
    "ONTOLOGY_SUPPORT",
    "STRATEGY_RANKING",
    "PROFITABILITY",
)

#: Reason codes, all technical.
PLAN_MISSING = "GUARD_PLAN_MISSING"
PLAN_NOT_EXECUTABLE = "GUARD_PLAN_NOT_EXECUTABLE"
PLAN_SYMBOL_MISMATCH = "GUARD_PLAN_SYMBOL_MISMATCH"
INVALID_PRICE = "GUARD_INVALID_PRICE"
INVALID_QUANTITY = "GUARD_INVALID_QUANTITY"
UNSUPPORTED_INSTRUMENT = "GUARD_UNSUPPORTED_INSTRUMENT"
INSUFFICIENT_CASH = "GUARD_INSUFFICIENT_CASH"
CASH_UNKNOWN = "GUARD_ORDERABLE_CASH_UNKNOWN"
INSUFFICIENT_SELLABLE = "GUARD_INSUFFICIENT_SELLABLE_QUANTITY"
BORROW_UNAVAILABLE = "GUARD_BORROW_UNAVAILABLE"
KILL_SWITCH = "GUARD_KILL_SWITCH_ENGAGED"
BROKER_UNHEALTHY = "GUARD_BROKER_UNHEALTHY"
GUARD_INTERNAL_ERROR = "GUARD_INTERNAL_ERROR"

_EXIT_SIDES = {"SELL", "REDUCE", "CLOSE"}


@dataclass(frozen=True)
class GuardOrder:
    """The order as it would be sent. Shape only; no thesis attached."""

    symbol: str
    market: str
    side: str
    quantity: int
    limit_price: float
    direction: str = "LONG"
    position_effect: str = "OPEN"
    execution_product: str = "CASH"
    exchange: str = ""

    @property
    def is_exit(self) -> bool:
        effect = str(self.position_effect or "").strip().upper()
        if effect == "CLOSE":
            return True
        if effect == "OPEN":
            return False
        return str(self.side or "").strip().upper() in _EXIT_SIDES

    @property
    def notional(self) -> float:
        return max(0, int(self.quantity)) * max(0.0, float(self.limit_price))


@dataclass(frozen=True)
class ExecutionGuardDecision:
    allowed: bool
    reason_codes: tuple[str, ...] = ()
    checked: tuple[str, ...] = ()
    #: Quantity the guard will permit. Only ever <= the requested quantity — a reduction
    #: to what the broker will accept, never a resize of the position.
    permitted_quantity: int = 0
    clipped: bool = False
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason_codes": list(self.reason_codes),
            "checked": list(self.checked),
            "permitted_quantity": self.permitted_quantity,
            "clipped": self.clipped,
            "detail": dict(self.detail),
        }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class ExecutionGuard:
    """The only thing between an elected plan and the broker."""

    def __init__(
        self,
        *,
        pre_submit_guard: PreSubmitGuard | None = None,
        broker_health_provider: Any | None = None,
        borrow_provider: Any | None = None,
        kill_switch_provider: Any | None = None,
        require_plan: bool = True,
        #: Extra margin over notional the broker may demand (fees, FX buffer).
        cash_buffer_rate: float = 0.005,
        #: When True an unanswerable affordability question blocks a BUY. Live sets it.
        strict_affordability: bool = False,
    ) -> None:
        self._pre_submit = pre_submit_guard
        self._broker_health = broker_health_provider
        self._borrow = borrow_provider
        self._kill_switch = kill_switch_provider
        self._require_plan = bool(require_plan)
        self._cash_buffer_rate = max(0.0, float(cash_buffer_rate))
        self._strict_affordability = bool(strict_affordability)

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        order: GuardOrder,
        *,
        plan: Any | None = None,
        orderable_cash: float | None = None,
        sellable_quantity: int | None = None,
        now: datetime | None = None,
    ) -> ExecutionGuardDecision:
        """Verdict for one order. Never raises; an internal error blocks."""
        moment = now or _utcnow()
        try:
            return self._evaluate(
                order,
                plan=plan,
                orderable_cash=orderable_cash,
                sellable_quantity=sellable_quantity,
                now=moment,
            )
        except Exception as exc:  # noqa: BLE001 - a guard that can be crashed past is none.
            return ExecutionGuardDecision(
                allowed=False,
                reason_codes=(f"{GUARD_INTERNAL_ERROR}:{type(exc).__name__}",),
                detail={"error": str(exc)},
            )

    # ------------------------------------------------------------------ #
    def _evaluate(
        self,
        order: GuardOrder,
        *,
        plan: Any | None,
        orderable_cash: float | None,
        sellable_quantity: int | None,
        now: datetime,
    ) -> ExecutionGuardDecision:
        reasons: list[str] = []
        checked: list[str] = []
        detail: dict[str, Any] = {"is_exit": order.is_exit}
        permitted = max(0, int(order.quantity))
        clipped = False

        # -- kill switch, first and unconditional ------------------------- #
        checked.append("kill_switch")
        if self._kill_switch_engaged():
            reasons.append(KILL_SWITCH)

        # -- plan validity --------------------------------------------------- #
        checked.append("plan")
        if plan is None:
            if self._require_plan and not order.is_exit:
                reasons.append(PLAN_MISSING)
        else:
            if str(getattr(plan, "symbol", "")).upper() != str(order.symbol).upper():
                reasons.append(PLAN_SYMBOL_MISMATCH)
            executable, why = plan.executable(now)
            detail["plan_id"] = getattr(plan, "plan_id", None)
            detail["plan_status"] = getattr(getattr(plan, "status", None), "value", None)
            if not executable:
                # An EXIT is never blocked by its plan's clock: the position is real
                # whatever the plan's expiry says, and it still needs to be closeable.
                if not order.is_exit:
                    reasons.append(f"{PLAN_NOT_EXECUTABLE}:{why}")
                else:
                    detail["plan_expiry_ignored_for_exit"] = why

        # -- order shape -------------------------------------------------------- #
        checked.append("order_shape")
        price = _finite(order.limit_price)
        if price is None or price <= 0.0:
            reasons.append(INVALID_PRICE)
        if int(order.quantity) <= 0:
            reasons.append(INVALID_QUANTITY)
        if not self._instrument_supported(order):
            reasons.append(UNSUPPORTED_INSTRUMENT)

        # -- session / order state / freshness / account ---------------------------- #
        guard = self._pre_submit
        if guard is not None:
            checked.append("pre_submit")
            pre = guard.evaluate(
                ticker=order.symbol,
                side="SELL" if order.is_exit else "BUY",
                market=order.market,
                now=now,
            )
            detail["pre_submit"] = pre.as_dict()
            if not pre.allowed:
                reasons.extend(pre.reason_codes)

        # -- affordability / deliverability ------------------------------------------ #
        if not order.is_exit:
            checked.append("cash")
            cash = _finite(orderable_cash)
            if cash is None and self._strict_affordability:
                # No cash figure is not "enough cash". Under strict (live) operation an
                # unanswerable affordability question blocks, because the broker's own
                # answer to it is a rejection.
                reasons.append(CASH_UNKNOWN)
            elif cash is not None and price is not None and price > 0.0:
                required_per_share = price * (1.0 + self._cash_buffer_rate)
                affordable = int(cash // required_per_share) if required_per_share > 0 else 0
                detail["orderable_cash"] = cash
                detail["affordable_quantity"] = affordable
                if affordable <= 0:
                    reasons.append(INSUFFICIENT_CASH)
                elif affordable < permitted:
                    # A clip, not a resize. The plan decided how much risk to take; the
                    # broker decides how much of it it will accept right now.
                    permitted = affordable
                    clipped = True
        else:
            checked.append("sellable")
            if sellable_quantity is not None:
                sellable = max(0, int(sellable_quantity))
                detail["sellable_quantity"] = sellable
                if sellable <= 0:
                    reasons.append(INSUFFICIENT_SELLABLE)
                elif sellable < permitted:
                    permitted = sellable
                    clipped = True

        # -- borrow ----------------------------------------------------------------- #
        if str(order.direction).upper() == "SHORT" and not order.is_exit:
            checked.append("borrow")
            if not self._borrow_available(order, now):
                reasons.append(BORROW_UNAVAILABLE)

        # -- broker health ------------------------------------------------------------ #
        checked.append("broker_health")
        health_failures = self._broker_health_failures()
        if health_failures:
            detail["broker_health_failures"] = list(health_failures)
            reasons.append(BROKER_UNHEALTHY)

        deduped = tuple(dict.fromkeys(reasons))
        allowed = not deduped and permitted > 0
        return ExecutionGuardDecision(
            allowed=allowed,
            reason_codes=deduped,
            checked=tuple(dict.fromkeys(checked)),
            permitted_quantity=permitted if allowed else 0,
            clipped=clipped and allowed,
            detail=detail,
        )

    # ------------------------------------------------------------------ #
    def _kill_switch_engaged(self) -> bool:
        provider = self._kill_switch
        if provider is not None:
            return bool(provider())
        return os.getenv("KILL_SWITCH_ENABLED", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _instrument_supported(order: GuardOrder) -> bool:
        """Is this a symbol/exchange/product this account can route at all?

        Shape and routability only. Whether the instrument is a *good* thing to own was
        settled at election by the instrument-eligibility rules inside the risk manager.
        """
        ticker = str(order.symbol or "").strip()
        if not ticker:
            return False
        product = str(order.execution_product or "CASH").strip().upper()
        if product not in {"CASH", "CREDIT", "MARGIN"}:
            return False
        if ticker.isdigit():
            return len(ticker) == 6
        return ticker.replace(".", "").replace("-", "").isalnum()

    def _borrow_available(self, order: GuardOrder, now: datetime) -> bool:
        provider = self._borrow
        if provider is None:
            # No locate source is not evidence of a locate. A short that cannot prove a
            # borrow is not sendable.
            return False
        try:
            snapshot = provider(order.symbol, now)
        except Exception:  # noqa: BLE001 - an unanswered locate is not a locate.
            return False
        if snapshot is None:
            return False
        available = _finite(getattr(snapshot, "available_quantity", None))
        if available is None:
            return bool(getattr(snapshot, "available", False))
        return available >= int(order.quantity)

    def _broker_health_failures(self) -> tuple[str, ...]:
        provider = self._broker_health
        if provider is None:
            return ()
        try:
            health = provider()
        except Exception as exc:  # noqa: BLE001 - unknown health is unhealthy.
            return (f"HEALTH_PROBE_FAILED:{type(exc).__name__}",)
        if health is None:
            return ("HEALTH_UNKNOWN",)
        if isinstance(health, bool):
            return () if health else ("HEALTH_FALSE",)
        if getattr(health, "ok", False):
            return ()
        failures = getattr(health, "failures", ()) or ()
        return tuple(str(item) for item in failures) or ("HEALTH_NOT_OK",)


def default_execution_guard(
    *,
    broker: Any | None = None,
    strict: bool | None = None,
    require_plan: bool = True,
) -> ExecutionGuard:
    """The guard the coordinator builds when none is injected."""

    def _health():
        if broker is None:
            return None
        from app.execution.kis_auth import run_kis_health_check

        return run_kis_health_check(broker, include_account=True, include_websocket=True)

    def _borrow(symbol: str, now: datetime):
        from app.trading.borrow import default_borrow_store

        store = default_borrow_store()
        latest = getattr(store, "latest", None)
        return latest(symbol) if callable(latest) else None

    try:
        from app.config.execution_authority import default_execution_authority_config

        strict_affordability = bool(
            default_execution_authority_config().strict_affordability
        )
    except Exception:  # noqa: BLE001 - an unreadable policy is the strict one.
        strict_affordability = True

    return ExecutionGuard(
        pre_submit_guard=default_pre_submit_guard(strict=strict),
        broker_health_provider=_health if broker is not None else None,
        borrow_provider=_borrow,
        require_plan=require_plan,
        strict_affordability=strict_affordability,
    )

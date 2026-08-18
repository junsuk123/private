"""Reconciliation: make the broker's answer the truth, and say so when it is not.

Three reconciliations, three different failures they catch
-----------------------------------------------------------

:class:`OrderReconciler`
    Every intent the state machine believes is live is queried at the broker. Resolves
    ``UNKNOWN`` back to a real state, applies partial fills, and — the case this exists
    for — finds orders the broker knows about that we do not.

:class:`PositionReconciler`
    Broker positions versus our own. A position we do not know about is unmanaged risk; a
    position we think we have and the broker does not is a phantom that will size the next
    order wrongly. Both are discrepancies, and both block new entries until resolved.

:class:`AccountReconciler`
    Equity and cash. Every position size is a fraction of equity, so an equity figure that
    is stale or disagrees with the broker makes every subsequent size wrong.

The result of each is a :class:`ReconciliationResult` whose ``reconciled`` flag feeds the
gate's ``ACCOUNT_RECONCILIATION_FAIL`` and ``UNKNOWN_ORDER_STATE`` hard gates.

Fail-closed on the query itself
--------------------------------
A broker query that raises produces ``reconciled=False``, not an empty-and-therefore-clean
result. "The broker did not answer" and "the broker says we hold nothing" are the same
bytes to a naive caller and opposite facts; conflating them is how a reconciliation pass
reports success by failing.

No orders are placed here. Reconciliation observes and records; the trading loop decides.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Protocol, Sequence
from uuid import uuid4

from app.execution.order_state_machine import (
    OrderIntentRecord,
    OrderState,
    OrderStateError,
    OrderStateMachine,
)
from app.storage.trading_state_store import (
    TradingStateStore,
    default_trading_state_store,
    iso_column,
    json_column,
)

__all__ = [
    "AccountReconciler",
    "AccountView",
    "BrokerOrderStatus",
    "BrokerPosition",
    "Discrepancy",
    "OrderReconciler",
    "PositionReconciler",
    "ReconciliationResult",
]

#: Quantity difference below which a position is considered matched. Whole shares, so
#: anything at or above 1 is a real discrepancy; the tolerance exists for fractional
#: instruments, not for rounding away a share.
POSITION_QUANTITY_TOLERANCE = 1e-6

#: Relative equity difference tolerated before the account is flagged. 0.1% absorbs FX
#: rounding and mark timing without hiding a missing position.
EQUITY_RELATIVE_TOLERANCE = 0.001


class BrokerOrderSource(Protocol):
    """The minimum a broker client must offer for order reconciliation."""

    def get_order_status(self, broker_order_id: str) -> Any: ...


@dataclass(frozen=True)
class BrokerOrderStatus:
    """Normalised broker answer for one order."""

    broker_order_id: str
    status: str
    filled_quantity: int = 0
    average_price: float | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BrokerPosition:
    ticker: str
    quantity: float
    average_price: float | None = None
    market_value: float | None = None
    currency: str = ""


@dataclass(frozen=True)
class AccountView:
    equity: float | None
    cash: float | None
    currency: str = ""
    observed_at: datetime | None = None


@dataclass(frozen=True)
class Discrepancy:
    kind: str
    key: str
    expected: Any
    observed: Any
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "key": self.key,
            "expected": self.expected,
            "observed": self.observed,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class ReconciliationResult:
    scope: str
    reconciled: bool
    checked_at: datetime
    discrepancies: tuple[Discrepancy, ...] = ()
    reason_codes: tuple[str, ...] = ()
    detail: Mapping[str, Any] = field(default_factory=dict)
    snapshot_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "reconciled": self.reconciled,
            "checked_at": iso_column(self.checked_at),
            "discrepancies": [item.as_dict() for item in self.discrepancies],
            "reason_codes": list(self.reason_codes),
            "detail": dict(self.detail),
            "snapshot_id": self.snapshot_id,
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


#: Broker status strings mapped onto the state machine. Unrecognised strings deliberately
#: map to UNKNOWN rather than to a guess.
_STATUS_MAP: dict[str, OrderState] = {
    "FILLED": OrderState.FILLED,
    "COMPLETE": OrderState.FILLED,
    "COMPLETED": OrderState.FILLED,
    "EXECUTED": OrderState.FILLED,
    "PARTIAL": OrderState.PARTIALLY_FILLED,
    "PARTIALLY_FILLED": OrderState.PARTIALLY_FILLED,
    "PARTIAL_FILL": OrderState.PARTIALLY_FILLED,
    "OPEN": OrderState.SUBMITTED,
    "WORKING": OrderState.SUBMITTED,
    "ACCEPTED": OrderState.SUBMITTED,
    "SUBMITTED": OrderState.SUBMITTED,
    "PENDING": OrderState.SUBMITTED,
    "CANCELLED": OrderState.CANCELLED,
    "CANCELED": OrderState.CANCELLED,
    "REJECTED": OrderState.REJECTED,
    "EXPIRED": OrderState.EXPIRED,
}


def normalise_broker_status(raw: Any) -> BrokerOrderStatus | None:
    """Read a broker response into :class:`BrokerOrderStatus`, or ``None``.

    Attribute-based rather than type-based, because the KIS clients, the mock and the
    paper executor all report the same fields without sharing a base class.
    """
    if raw is None:
        return None
    if isinstance(raw, BrokerOrderStatus):
        return raw
    order_id = (
        getattr(raw, "broker_order_id", None)
        or getattr(raw, "order_id", None)
        or (raw.get("order_id") if isinstance(raw, Mapping) else None)
    )
    status = (
        getattr(raw, "status", None)
        or (raw.get("status") if isinstance(raw, Mapping) else None)
        or "UNKNOWN"
    )
    filled = (
        getattr(raw, "filled_quantity", None)
        or (raw.get("filled_quantity") if isinstance(raw, Mapping) else None)
        or 0
    )
    price = getattr(raw, "average_price", None)
    if price is None and isinstance(raw, Mapping):
        price = raw.get("average_price")
    return BrokerOrderStatus(
        broker_order_id=str(order_id or ""),
        status=str(status).upper(),
        filled_quantity=int(_finite(filled) or 0),
        average_price=_finite(price),
        raw=dict(raw) if isinstance(raw, Mapping) else {},
    )


class OrderReconciler:
    """Resolves every open intent against the broker."""

    def __init__(
        self,
        state_machine: OrderStateMachine,
        broker: BrokerOrderSource | None = None,
    ) -> None:
        self._states = state_machine
        self._broker = broker

    def reconcile(
        self,
        *,
        broker: BrokerOrderSource | None = None,
        now: datetime | None = None,
        broker_open_orders: Sequence[Mapping[str, Any]] | None = None,
    ) -> ReconciliationResult:
        moment = now or _utcnow()
        client = broker or self._broker
        open_intents = self._states.open_intents()
        discrepancies: list[Discrepancy] = []
        reasons: list[str] = []
        resolved = 0
        unresolved: list[str] = []

        if client is None:
            return ReconciliationResult(
                scope="orders",
                reconciled=not open_intents,
                checked_at=moment,
                reason_codes=("NO_BROKER_CLIENT",) if open_intents else (),
                detail={"open_intent_count": len(open_intents)},
            )

        for record in open_intents:
            if record.broker_order_id is None:
                if record.state in {OrderState.CREATED, OrderState.GATED}:
                    # Never reached the broker; nothing to reconcile against.
                    continue
                unresolved.append(record.intent_id)
                discrepancies.append(
                    Discrepancy(
                        kind="ORDER_WITHOUT_BROKER_ID",
                        key=record.intent_id,
                        expected=record.state.value,
                        observed=None,
                    )
                )
                self._to_unknown(record, "NO_BROKER_ORDER_ID", moment)
                continue
            try:
                status = normalise_broker_status(
                    client.get_order_status(record.broker_order_id)
                )
            except Exception as exc:  # noqa: BLE001 - an unanswered query is not a clean one.
                unresolved.append(record.intent_id)
                reasons.append("BROKER_QUERY_FAILED")
                discrepancies.append(
                    Discrepancy(
                        kind="BROKER_QUERY_FAILED",
                        key=record.intent_id,
                        expected=record.state.value,
                        observed=f"{type(exc).__name__}: {exc}",
                    )
                )
                self._to_unknown(record, f"BROKER_QUERY_FAILED:{type(exc).__name__}", moment)
                continue
            if status is None:
                unresolved.append(record.intent_id)
                self._to_unknown(record, "BROKER_STATUS_EMPTY", moment)
                continue
            if self._apply(record, status, moment):
                resolved += 1
            else:
                unresolved.append(record.intent_id)

        # Orders the broker knows about and we do not. This is the reconciliation that
        # actually protects capital: an unmatched working order is exposure no local
        # limit is counting.
        for entry in broker_open_orders or ():
            broker_id = str(entry.get("order_id") or entry.get("broker_order_id") or "")
            if not broker_id:
                continue
            if self._states.by_broker_order_id(broker_id) is None:
                discrepancies.append(
                    Discrepancy(
                        kind="UNTRACKED_BROKER_ORDER",
                        key=broker_id,
                        expected=None,
                        observed=dict(entry),
                    )
                )
                reasons.append("UNTRACKED_BROKER_ORDER")

        if unresolved:
            reasons.append("UNKNOWN_ORDER_STATE")
        return ReconciliationResult(
            scope="orders",
            reconciled=not discrepancies and not unresolved,
            checked_at=moment,
            discrepancies=tuple(discrepancies),
            reason_codes=tuple(dict.fromkeys(reasons)),
            detail={
                "open_intent_count": len(open_intents),
                "resolved_count": resolved,
                "unresolved_intent_ids": unresolved,
            },
        )

    def _to_unknown(
        self, record: OrderIntentRecord, reason: str, moment: datetime
    ) -> None:
        if record.state is OrderState.UNKNOWN:
            return
        try:
            self._states.transition(
                record.intent_id, OrderState.UNKNOWN, reason=reason, now=moment
            )
        except OrderStateError:
            # A terminal intent cannot become UNKNOWN, and does not need to.
            return

    def _apply(
        self, record: OrderIntentRecord, status: BrokerOrderStatus, moment: datetime
    ) -> bool:
        target = _STATUS_MAP.get(status.status)
        if target is None:
            self._to_unknown(record, f"UNMAPPED_BROKER_STATUS:{status.status}", moment)
            return False
        filled = max(record.filled_quantity, int(status.filled_quantity))
        # The COUNT outranks the LABEL. Brokers report "FILLED" on a partially executed
        # order and "OPEN" on one that has already filled some, and taking either label
        # at face value loses a fill or invents one. Wherever the two disagree, the
        # quantity decides.
        if target in {OrderState.FILLED, OrderState.PARTIALLY_FILLED, OrderState.SUBMITTED}:
            if filled >= record.quantity:
                target = OrderState.FILLED
            elif filled > 0:
                target = OrderState.PARTIALLY_FILLED
            elif target is not OrderState.SUBMITTED:
                target = OrderState.SUBMITTED
        if target is record.state and filled == record.filled_quantity:
            return True
        try:
            self._states.transition(
                record.intent_id,
                target,
                event_type="RECONCILED",
                reason=f"BROKER_STATUS:{status.status}",
                broker_order_id=status.broker_order_id or record.broker_order_id,
                filled_quantity=filled if target is not OrderState.SUBMITTED else None,
                fill_price=status.average_price,
                payload=dict(status.raw),
                now=moment,
            )
        except OrderStateError:
            self._to_unknown(record, f"ILLEGAL_RECONCILED_STATE:{target.value}", moment)
            return False
        return True


class PositionReconciler:
    """Broker positions versus the fills this system recorded."""

    def __init__(
        self,
        state_machine: OrderStateMachine,
        store: TradingStateStore | None = None,
    ) -> None:
        self._states = state_machine
        self._store = store or state_machine.store

    def reconcile(
        self,
        broker_positions: Iterable[BrokerPosition] | None,
        *,
        expected_positions: Mapping[str, float] | None = None,
        now: datetime | None = None,
        source: str = "broker",
    ) -> ReconciliationResult:
        moment = now or _utcnow()
        if broker_positions is None:
            return ReconciliationResult(
                scope="positions",
                reconciled=False,
                checked_at=moment,
                reason_codes=("POSITION_QUERY_FAILED",),
            )
        positions = list(broker_positions)
        observed = {
            str(item.ticker).upper(): float(item.quantity) for item in positions
        }
        expected = (
            {str(key).upper(): float(value) for key, value in expected_positions.items()}
            if expected_positions is not None
            else self.expected_from_fills()
        )

        discrepancies: list[Discrepancy] = []
        for ticker in sorted(set(observed) | set(expected)):
            broker_quantity = observed.get(ticker, 0.0)
            local_quantity = expected.get(ticker, 0.0)
            if abs(broker_quantity - local_quantity) <= POSITION_QUANTITY_TOLERANCE:
                continue
            kind = (
                "UNTRACKED_BROKER_POSITION"
                if local_quantity == 0.0
                else "PHANTOM_LOCAL_POSITION"
                if broker_quantity == 0.0
                else "POSITION_QUANTITY_MISMATCH"
            )
            discrepancies.append(
                Discrepancy(
                    kind=kind,
                    key=ticker,
                    expected=local_quantity,
                    observed=broker_quantity,
                )
            )

        self._persist(positions, moment, source)
        return ReconciliationResult(
            scope="positions",
            reconciled=not discrepancies,
            checked_at=moment,
            discrepancies=tuple(discrepancies),
            reason_codes=("POSITION_MISMATCH",) if discrepancies else (),
            detail={
                "broker_position_count": len(positions),
                "expected_position_count": len(expected),
            },
        )

    def expected_from_fills(self) -> dict[str, float]:
        """Net position implied by the fills this system recorded.

        BUY adds, SELL subtracts, computed over every intent with a non-zero fill —
        including terminal ones, because a position is the sum of history rather than of
        what is currently open.
        """
        rows = self._store.fetch_all(
            "select ticker, side, sum(filled_quantity) as filled"
            " from order_intent where filled_quantity > 0 group by ticker, side"
        )
        net: dict[str, float] = {}
        for row in rows:
            ticker = str(row["ticker"]).upper()
            quantity = float(row["filled"] or 0.0)
            sign = -1.0 if str(row["side"]).upper() in {"SELL", "SHORT"} else 1.0
            net[ticker] = net.get(ticker, 0.0) + sign * quantity
        return {ticker: value for ticker, value in net.items() if abs(value) > POSITION_QUANTITY_TOLERANCE}

    def _persist(
        self, positions: Sequence[BrokerPosition], moment: datetime, source: str
    ) -> None:
        if not positions:
            return
        with self._store.transaction() as conn:
            for position in positions:
                conn.execute(
                    "insert into position_snapshot"
                    " (snapshot_id, captured_at, source, ticker, quantity,"
                    "  average_price, market_value, currency, payload_json)"
                    " values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"ps-{uuid4().hex}",
                        iso_column(moment),
                        str(source),
                        str(position.ticker).upper(),
                        float(position.quantity),
                        position.average_price,
                        position.market_value,
                        str(position.currency or ""),
                        json_column(
                            {
                                "ticker": position.ticker,
                                "quantity": position.quantity,
                                "average_price": position.average_price,
                            }
                        ),
                    ),
                )


class AccountReconciler:
    """Broker equity and cash versus what the system is sizing against."""

    def __init__(self, store: TradingStateStore | None = None) -> None:
        self._store = store or default_trading_state_store()

    def reconcile(
        self,
        broker: AccountView | None,
        *,
        local_equity: float | None = None,
        local_cash: float | None = None,
        max_age_seconds: float = 600.0,
        now: datetime | None = None,
        source: str = "broker",
    ) -> ReconciliationResult:
        moment = now or _utcnow()
        if broker is None:
            return ReconciliationResult(
                scope="account",
                reconciled=False,
                checked_at=moment,
                reason_codes=("ACCOUNT_QUERY_FAILED",),
            )
        equity = _finite(broker.equity)
        cash = _finite(broker.cash)
        reasons: list[str] = []
        discrepancies: list[Discrepancy] = []

        if equity is None:
            reasons.append("ACCOUNT_EQUITY_MISSING")
        if broker.observed_at is not None:
            age = (moment - broker.observed_at.astimezone(timezone.utc)).total_seconds()
            if age > max_age_seconds:
                reasons.append("ACCOUNT_SNAPSHOT_STALE")
                discrepancies.append(
                    Discrepancy(
                        kind="ACCOUNT_SNAPSHOT_STALE",
                        key="observed_at",
                        expected=f"<= {max_age_seconds}s",
                        observed=round(age, 3),
                    )
                )

        local = _finite(local_equity)
        if equity is not None and local is not None and equity > 0.0:
            relative = abs(equity - local) / equity
            if relative > EQUITY_RELATIVE_TOLERANCE:
                reasons.append("EQUITY_MISMATCH")
                discrepancies.append(
                    Discrepancy(
                        kind="EQUITY_MISMATCH",
                        key="equity",
                        expected=local,
                        observed=equity,
                        detail={"relative_difference": round(relative, 8)},
                    )
                )
        local_cash_value = _finite(local_cash)
        if cash is not None and local_cash_value is not None and abs(cash) > 0.0:
            relative = abs(cash - local_cash_value) / max(abs(cash), 1.0)
            if relative > EQUITY_RELATIVE_TOLERANCE:
                reasons.append("CASH_MISMATCH")
                discrepancies.append(
                    Discrepancy(
                        kind="CASH_MISMATCH",
                        key="cash",
                        expected=local_cash_value,
                        observed=cash,
                        detail={"relative_difference": round(relative, 8)},
                    )
                )

        reconciled = not reasons and not discrepancies
        snapshot_id = f"as-{uuid4().hex}"
        with self._store.transaction() as conn:
            conn.execute(
                "insert into account_snapshot"
                " (snapshot_id, captured_at, source, equity, cash, currency,"
                "  reconciled, discrepancies_json, payload_json)"
                " values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot_id,
                    iso_column(moment),
                    str(source),
                    equity,
                    cash,
                    str(broker.currency or ""),
                    1 if reconciled else 0,
                    json_column([item.as_dict() for item in discrepancies]),
                    json_column(
                        {
                            "local_equity": local,
                            "local_cash": local_cash_value,
                            "observed_at": iso_column(broker.observed_at),
                        }
                    ),
                ),
            )
        return ReconciliationResult(
            scope="account",
            reconciled=reconciled,
            checked_at=moment,
            discrepancies=tuple(discrepancies),
            reason_codes=tuple(dict.fromkeys(reasons)),
            detail={"equity": equity, "cash": cash},
            snapshot_id=snapshot_id,
        )

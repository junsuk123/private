"""Durable order state machine: one row per intent, every transition appended.

What this adds to the existing execution layer
----------------------------------------------
:class:`~app.execution.live_execution_coordinator.LiveExecutionCoordinator` already owns
idempotent submission, the broker call, amend/cancel and the JSONL journal. What it does
not own is *state that survives a restart*: after a crash there is no single place that
says which intents are still open, how much of each is filled and which have a broker id
whose status was never resolved. That is what this module is.

It does not replace the coordinator and does not talk to a broker. It records what the
coordinator did, inside the same transaction as the gate decision that authorised it, so
an order can never exist without its authorisation and vice versa.

States
------
::

    CREATED -> GATED -> SUBMITTING -> SUBMITTED -> PARTIALLY_FILLED -> FILLED
                  |          |            |               |
                  |          |            +-> CANCELLING -+-> CANCELLED
                  |          |            |
                  |          +-> REJECTED |
                  |                       +-> EXPIRED
                  +-> BLOCKED
    (any non-terminal) -> UNKNOWN -> (resolved back, or CANCELLED after reconciliation)

``UNKNOWN`` is the important one. A submission that timed out is neither live nor dead,
and guessing either way is how a duplicate order or an unmanaged position happens. It is a
real state, it blocks new orders on that symbol through the gate's
``UNKNOWN_ORDER_STATE``, and it is left for :mod:`app.execution.reconciliation` to resolve
against the broker.

Restart recovery
----------------
:meth:`OrderStateMachine.recover` returns every non-terminal intent at startup, so the
first thing the process does after a restart is find out what it left open — rather than
discovering it when a position appears in the account that nothing in memory knows about.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from app.storage.trading_state_store import (
    TradingStateStore,
    default_trading_state_store,
    iso_column,
    json_column,
)

__all__ = [
    "OrderState",
    "OrderStateError",
    "OrderStateMachine",
    "OrderIntentRecord",
    "TERMINAL_STATES",
]


class OrderState(str, Enum):
    CREATED = "CREATED"
    GATED = "GATED"
    BLOCKED = "BLOCKED"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    #: The broker's answer is unavailable. Neither live nor dead; blocks new orders.
    UNKNOWN = "UNKNOWN"


TERMINAL_STATES: frozenset[OrderState] = frozenset(
    {
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.EXPIRED,
        OrderState.BLOCKED,
    }
)

#: Allowed transitions. Anything absent is refused, so a bug that tries to move an order
#: from FILLED back to SUBMITTED raises instead of silently corrupting the record.
_ALLOWED: dict[OrderState, frozenset[OrderState]] = {
    OrderState.CREATED: frozenset({OrderState.GATED, OrderState.BLOCKED}),
    OrderState.GATED: frozenset(
        {OrderState.SUBMITTING, OrderState.BLOCKED, OrderState.EXPIRED}
    ),
    OrderState.BLOCKED: frozenset(),
    OrderState.SUBMITTING: frozenset(
        {
            OrderState.SUBMITTED,
            OrderState.REJECTED,
            OrderState.UNKNOWN,
            # A broker that answers "filled" on the submit call skips SUBMITTED.
            OrderState.FILLED,
            OrderState.PARTIALLY_FILLED,
        }
    ),
    OrderState.SUBMITTED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELLING,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.PARTIALLY_FILLED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELLING,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.CANCELLING: frozenset(
        {
            OrderState.CANCELLED,
            # A cancel that loses the race: the fill already happened.
            OrderState.FILLED,
            OrderState.PARTIALLY_FILLED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.FILLED: frozenset(),
    OrderState.CANCELLED: frozenset(),
    OrderState.REJECTED: frozenset(),
    OrderState.EXPIRED: frozenset(),
    OrderState.UNKNOWN: frozenset(
        {
            OrderState.SUBMITTED,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.UNKNOWN,
        }
    ),
}


class OrderStateError(RuntimeError):
    """An illegal transition, or a write that would break the intent's invariants."""


@dataclass(frozen=True)
class OrderIntentRecord:
    intent_id: str
    ticker: str
    side: str
    quantity: int
    state: OrderState
    created_at: datetime
    state_updated_at: datetime
    idempotency_key: str
    market_group: str = ""
    venue: str = ""
    limit_price: float | None = None
    order_type: str = "LIMIT"
    filled_quantity: int = 0
    average_fill_price: float | None = None
    broker_order_id: str | None = None
    decision_id: str | None = None
    gate_id: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def remaining_quantity(self) -> int:
        return max(0, int(self.quantity) - int(self.filled_quantity))

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "ticker": self.ticker,
            "side": self.side,
            "quantity": self.quantity,
            "state": self.state.value,
            "created_at": iso_column(self.created_at),
            "state_updated_at": iso_column(self.state_updated_at),
            "idempotency_key": self.idempotency_key,
            "market_group": self.market_group,
            "venue": self.venue,
            "limit_price": self.limit_price,
            "order_type": self.order_type,
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "broker_order_id": self.broker_order_id,
            "decision_id": self.decision_id,
            "gate_id": self.gate_id,
            "remaining_quantity": self.remaining_quantity,
            "terminal": self.terminal,
            "payload": dict(self.payload),
        }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return (moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)).astimezone(
        timezone.utc
    )


def _record_from_row(row: Mapping[str, Any]) -> OrderIntentRecord:
    return OrderIntentRecord(
        intent_id=str(row["intent_id"]),
        ticker=str(row["ticker"]),
        side=str(row["side"]),
        quantity=int(row["quantity"]),
        state=OrderState(str(row["state"])),
        created_at=_parse(row["created_at"]) or _utcnow(),
        state_updated_at=_parse(row["state_updated_at"]) or _utcnow(),
        idempotency_key=str(row["idempotency_key"]),
        market_group=str(row["market_group"] or ""),
        venue=str(row["venue"] or ""),
        limit_price=(
            float(row["limit_price"]) if row["limit_price"] is not None else None
        ),
        order_type=str(row["order_type"] or "LIMIT"),
        filled_quantity=int(row["filled_quantity"] or 0),
        average_fill_price=(
            float(row["average_fill_price"])
            if row["average_fill_price"] is not None
            else None
        ),
        broker_order_id=(
            str(row["broker_order_id"]) if row["broker_order_id"] else None
        ),
        decision_id=str(row["decision_id"]) if row["decision_id"] else None,
        gate_id=str(row["gate_id"]) if row["gate_id"] else None,
        payload=json.loads(str(row["payload_json"] or "{}")),
    )


class OrderStateMachine:
    """Persists order intents and their transitions, transactionally."""

    def __init__(self, store: TradingStateStore | None = None) -> None:
        self._store = store or default_trading_state_store()
        self._lock = threading.RLock()

    @property
    def store(self) -> TradingStateStore:
        return self._store

    # ------------------------------------------------------------------ #
    # creation
    # ------------------------------------------------------------------ #
    def create(
        self,
        *,
        ticker: str,
        side: str,
        quantity: int,
        idempotency_key: str,
        limit_price: float | None = None,
        order_type: str = "LIMIT",
        market_group: str = "",
        venue: str = "",
        decision_id: str | None = None,
        gate_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> OrderIntentRecord:
        """Create an intent, or return the existing one for this idempotency key.

        The key is a UNIQUE column, so two threads racing to create the same intent
        resolve to one row rather than to two orders. This is the second line of defence
        behind ``IdempotencyStore``; both exist because they fail differently — the store
        protects the broker call, this protects the record of it.
        """
        moment = now or _utcnow()
        if int(quantity) <= 0:
            raise OrderStateError("order intent quantity must be positive")
        with self._lock, self._store.transaction() as conn:
            existing = conn.execute(
                "select * from order_intent where idempotency_key = ?",
                (str(idempotency_key),),
            ).fetchone()
            if existing is not None:
                return _record_from_row(existing)
            intent_id = f"oi-{uuid4().hex}"
            conn.execute(
                "insert into order_intent"
                " (intent_id, created_at, decision_id, gate_id, ticker, market_group,"
                "  venue, side, quantity, limit_price, order_type, idempotency_key,"
                "  state, state_updated_at, filled_quantity, average_fill_price,"
                "  broker_order_id, payload_json, terminal)"
                " values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, null, null, ?, 0)",
                (
                    intent_id,
                    iso_column(moment),
                    decision_id,
                    gate_id,
                    str(ticker).upper(),
                    str(market_group).upper(),
                    str(venue).upper(),
                    str(side).upper(),
                    int(quantity),
                    limit_price,
                    str(order_type).upper(),
                    str(idempotency_key),
                    OrderState.CREATED.value,
                    iso_column(moment),
                    json_column(dict(payload or {})),
                ),
            )
            self._append_event(
                conn,
                intent_id=intent_id,
                observed_at=moment,
                event_type="CREATED",
                from_state=None,
                to_state=OrderState.CREATED,
                reason="",
                payload=dict(payload or {}),
            )
            row = conn.execute(
                "select * from order_intent where intent_id = ?", (intent_id,)
            ).fetchone()
        return _record_from_row(row)

    # ------------------------------------------------------------------ #
    # transitions
    # ------------------------------------------------------------------ #
    def transition(
        self,
        intent_id: str,
        to_state: OrderState,
        *,
        event_type: str | None = None,
        reason: str = "",
        broker_order_id: str | None = None,
        filled_quantity: int | None = None,
        fill_price: float | None = None,
        payload: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> OrderIntentRecord:
        """Move an intent to ``to_state``, refusing any transition not in the table.

        ``filled_quantity`` is CUMULATIVE, not incremental, because that is what a broker
        status response reports. It may never decrease: a status that shows less filled
        than we already recorded means the two sides disagree about reality, and silently
        accepting the smaller number would lose a fill.
        """
        moment = now or _utcnow()
        with self._lock, self._store.transaction() as conn:
            row = conn.execute(
                "select * from order_intent where intent_id = ?", (str(intent_id),)
            ).fetchone()
            if row is None:
                raise OrderStateError(f"unknown order intent {intent_id}")
            record = _record_from_row(row)
            if to_state not in _ALLOWED.get(record.state, frozenset()):
                raise OrderStateError(
                    f"illegal transition {record.state.value} -> {to_state.value}"
                    f" for {intent_id}"
                )

            new_filled = record.filled_quantity
            if filled_quantity is not None:
                candidate = int(filled_quantity)
                if candidate < record.filled_quantity:
                    raise OrderStateError(
                        f"{intent_id}: filled quantity may not decrease"
                        f" ({record.filled_quantity} -> {candidate})"
                    )
                if candidate > record.quantity:
                    raise OrderStateError(
                        f"{intent_id}: filled quantity {candidate} exceeds"
                        f" ordered {record.quantity}"
                    )
                new_filled = candidate

            if to_state is OrderState.FILLED and new_filled != record.quantity:
                raise OrderStateError(
                    f"{intent_id}: FILLED requires filled == quantity"
                    f" ({new_filled} != {record.quantity})"
                )
            if to_state is OrderState.PARTIALLY_FILLED and not (
                0 < new_filled < record.quantity
            ):
                raise OrderStateError(
                    f"{intent_id}: PARTIALLY_FILLED requires 0 < filled < quantity"
                )

            average = self._blend_average(
                record, previous_filled=record.filled_quantity,
                new_filled=new_filled, fill_price=fill_price,
            )
            conn.execute(
                "update order_intent set state = ?, state_updated_at = ?,"
                " filled_quantity = ?, average_fill_price = ?, broker_order_id = ?,"
                " terminal = ? where intent_id = ?",
                (
                    to_state.value,
                    iso_column(moment),
                    new_filled,
                    average,
                    broker_order_id or record.broker_order_id,
                    1 if to_state in TERMINAL_STATES else 0,
                    record.intent_id,
                ),
            )
            self._append_event(
                conn,
                intent_id=record.intent_id,
                observed_at=moment,
                event_type=event_type or to_state.value,
                from_state=record.state,
                to_state=to_state,
                reason=reason,
                broker_order_id=broker_order_id or record.broker_order_id,
                filled_quantity=new_filled,
                fill_price=fill_price,
                payload=dict(payload or {}),
            )
            updated = conn.execute(
                "select * from order_intent where intent_id = ?", (record.intent_id,)
            ).fetchone()
        return _record_from_row(updated)

    @staticmethod
    def _blend_average(
        record: OrderIntentRecord,
        *,
        previous_filled: int,
        new_filled: int,
        fill_price: float | None,
    ) -> float | None:
        """Quantity-weighted average across partial fills.

        A plain overwrite would report the last partial's price as the position's cost,
        which is wrong in exactly the case partial fills exist for.
        """
        if fill_price is None:
            return record.average_fill_price
        increment = new_filled - previous_filled
        if increment <= 0:
            return record.average_fill_price
        if record.average_fill_price is None or previous_filled <= 0:
            return float(fill_price)
        total = record.average_fill_price * previous_filled + float(fill_price) * increment
        return total / max(1, new_filled)

    def _append_event(
        self,
        conn: Any,
        *,
        intent_id: str,
        observed_at: datetime,
        event_type: str,
        from_state: OrderState | None,
        to_state: OrderState,
        reason: str,
        broker_order_id: str | None = None,
        filled_quantity: int = 0,
        fill_price: float | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        conn.execute(
            "insert into order_execution"
            " (execution_id, intent_id, observed_at, event_type, from_state, to_state,"
            "  filled_quantity, fill_price, broker_order_id, reason, payload_json)"
            " values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"oe-{uuid4().hex}",
                intent_id,
                iso_column(observed_at),
                str(event_type),
                from_state.value if from_state else None,
                to_state.value,
                int(filled_quantity),
                fill_price,
                broker_order_id,
                str(reason),
                json_column(dict(payload or {})),
            ),
        )

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    def get(self, intent_id: str) -> OrderIntentRecord | None:
        row = self._store.fetch_one(
            "select * from order_intent where intent_id = ?", (str(intent_id),)
        )
        return _record_from_row(row) if row else None

    def by_idempotency_key(self, key: str) -> OrderIntentRecord | None:
        row = self._store.fetch_one(
            "select * from order_intent where idempotency_key = ?", (str(key),)
        )
        return _record_from_row(row) if row else None

    def by_broker_order_id(self, broker_order_id: str) -> OrderIntentRecord | None:
        row = self._store.fetch_one(
            "select * from order_intent where broker_order_id = ?"
            " order by created_at desc limit 1",
            (str(broker_order_id),),
        )
        return _record_from_row(row) if row else None

    def open_intents(
        self, *, ticker: str | None = None
    ) -> tuple[OrderIntentRecord, ...]:
        """Every non-terminal intent, optionally for one symbol."""
        sql = "select * from order_intent where terminal = 0"
        params: list[Any] = []
        if ticker:
            sql += " and ticker = ?"
            params.append(str(ticker).upper())
        sql += " order by created_at"
        return tuple(
            _record_from_row(row) for row in self._store.fetch_all(sql, params)
        )

    def unknown_intents(self) -> tuple[OrderIntentRecord, ...]:
        return tuple(
            _record_from_row(row)
            for row in self._store.fetch_all(
                "select * from order_intent where state = ? order by created_at",
                (OrderState.UNKNOWN.value,),
            )
        )

    def has_duplicate_risk(self, ticker: str, side: str) -> bool:
        """Is an equivalent order already live, or in an unresolved state?

        ``UNKNOWN`` counts. An order whose status could not be read might be working at
        the broker, and sending a second one is precisely the duplicate this exists to
        prevent.
        """
        rows = self._store.fetch_all(
            "select count(*) as n from order_intent"
            " where terminal = 0 and ticker = ? and side = ?"
            " and state in (?, ?, ?, ?, ?, ?)",
            (
                str(ticker).upper(),
                str(side).upper(),
                OrderState.GATED.value,
                OrderState.SUBMITTING.value,
                OrderState.SUBMITTED.value,
                OrderState.PARTIALLY_FILLED.value,
                OrderState.CANCELLING.value,
                OrderState.UNKNOWN.value,
            ),
        )
        return bool(rows and int(rows[0]["n"]) > 0)

    def events(self, intent_id: str) -> tuple[dict[str, Any], ...]:
        # Ordered by rowid within a timestamp: several transitions can share an
        # ``observed_at`` (a submit and its immediate fill), and sorting those by the
        # random ``execution_id`` would shuffle the history into an impossible order.
        return self._store.fetch_all(
            "select * from order_execution where intent_id = ?"
            " order by observed_at, rowid",
            (str(intent_id),),
        )

    # ------------------------------------------------------------------ #
    # restart recovery
    # ------------------------------------------------------------------ #
    def recover(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Everything left open by the previous run, classified by what to do next.

        ``needs_broker_query`` are intents that reached the broker and whose outcome must
        be read back before anything else happens on that symbol. ``never_submitted`` are
        intents that did not, and can be expired locally — there is nothing at the broker
        to reconcile against.
        """
        moment = now or _utcnow()
        open_intents = self.open_intents()
        needs_query = [
            record
            for record in open_intents
            if record.state
            in {
                OrderState.SUBMITTING,
                OrderState.SUBMITTED,
                OrderState.PARTIALLY_FILLED,
                OrderState.CANCELLING,
                OrderState.UNKNOWN,
            }
        ]
        never_submitted = [
            record
            for record in open_intents
            if record.state in {OrderState.CREATED, OrderState.GATED}
        ]
        return {
            "recovered_at": iso_column(moment),
            "open_count": len(open_intents),
            "needs_broker_query": [record.as_dict() for record in needs_query],
            "never_submitted": [record.as_dict() for record in never_submitted],
            "blocked_tickers": sorted({record.ticker for record in needs_query}),
        }

    def expire_never_submitted(
        self, records: Iterable[OrderIntentRecord], *, now: datetime | None = None
    ) -> tuple[str, ...]:
        """Expire intents that never reached the broker. Safe by construction."""
        expired: list[str] = []
        for record in records:
            if record.state not in {OrderState.CREATED, OrderState.GATED}:
                continue
            if record.state is OrderState.CREATED:
                self.transition(
                    record.intent_id,
                    OrderState.BLOCKED,
                    reason="RESTART_RECOVERY_NEVER_GATED",
                    now=now,
                )
            else:
                self.transition(
                    record.intent_id,
                    OrderState.EXPIRED,
                    reason="RESTART_RECOVERY_NEVER_SUBMITTED",
                    now=now,
                )
            expired.append(record.intent_id)
        return tuple(expired)

    def summary(self) -> dict[str, Any]:
        rows = self._store.fetch_all(
            "select state, count(*) as n from order_intent group by state"
        )
        by_state = {str(row["state"]): int(row["n"]) for row in rows}
        return {
            "by_state": by_state,
            "open_count": sum(
                count
                for state, count in by_state.items()
                if OrderState(state) not in TERMINAL_STATES
            ),
            "unknown_count": by_state.get(OrderState.UNKNOWN.value, 0),
        }


def allowed_transitions(state: OrderState) -> frozenset[OrderState]:
    """The states ``state`` may move to. Exposed for tests and for the dashboard."""
    return _ALLOWED.get(state, frozenset())


def transitions_table() -> Mapping[str, Sequence[str]]:
    return {
        state.value: sorted(target.value for target in targets)
        for state, targets in _ALLOWED.items()
    }

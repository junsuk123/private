"""The TradePlan: the durable, immutable output of strategy election.

Why this type exists
--------------------
Before it, "the elected strategy" was a scattering of fields on
:class:`~app.trading.strategy_session.StrategySessionState` plus a dict of election
context, and the quantity, the cost basis and the risk verdict were recomputed
*downstream* — after election, by three separate authorities that could each veto what the
election had already decided. That is the duplication this refactor removes.

A ``TradePlan`` is the single artefact that says: **this symbol, this direction, this
strategy, this many shares, entering under this rule, exiting under these rules, with this
cost and risk basis, until this instant.** Everything downstream reads it. Nothing
downstream re-derives it.

What is frozen and why
----------------------
``strategy_id``, the risk-budget basis and the position-size methodology are immutable
after election, and :meth:`TradePlan.with_broker_clip` is the *only* way the quantity may
change — downward, and only for what the broker will actually accept (cash on hand,
sellable quantity). That is a technical clip, not an investment decision, and it is
recorded as one.

Expiry is a hard property, not a hint. A plan past ``expires_at`` cannot be executed; the
strategy must be re-elected against a fresh market. This is what stops a plan built on a
five-minute-old book from being submitted after a restart.

Persistence
-----------
Plans are written to ``trade_plan`` in ``data/store/trading_state.sqlite3`` in the same
transaction discipline as the rest of the decision chain, so a restart recovers the plan
that owns an open position rather than discovering the position with no plan behind it.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4

from app.storage.trading_state_store import (
    TradingStateStore,
    default_trading_state_store,
    iso_column,
    json_column,
)

__all__ = [
    "EntryRule",
    "ExitRules",
    "TradePlan",
    "TradePlanError",
    "TradePlanStore",
    "TradePlanStatus",
    "default_trade_plan_store",
    "reset_default_trade_plan_store",
]

#: How long an elected plan stays executable by default. Long enough to survive a slow
#: cycle and a broker round trip, short enough that a plan can never be acted on against a
#: market the election never saw.
DEFAULT_PLAN_TTL_SECONDS = 300.0


class TradePlanError(RuntimeError):
    """A plan that cannot be constructed or a mutation that is not permitted."""


class TradePlanStatus(str, Enum):
    ARMED = "ARMED"
    ENTERING = "ENTERING"
    OPEN = "OPEN"
    EXITING = "EXITING"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


TERMINAL_PLAN_STATUSES: frozenset[TradePlanStatus] = frozenset(
    {TradePlanStatus.CLOSED, TradePlanStatus.EXPIRED, TradePlanStatus.CANCELLED}
)


@dataclass(frozen=True)
class EntryRule:
    """When the owning strategy may enter, expressed so a tick loop can evaluate it.

    ``trigger`` names the strategy's own entry algorithm; the fast loop calls that
    algorithm rather than reimplementing it. The price band is the envelope the election
    priced against — a fill outside it is a different trade from the one that was
    approved, so the plan simply does not authorise it.
    """

    trigger: str
    min_price: float | None = None
    max_price: float | None = None
    #: Seconds after which an un-triggered entry is abandoned. ``None`` means the plan's
    #: own ``expires_at`` is the only bound.
    max_wait_seconds: float | None = None

    def price_permitted(self, price: float) -> bool:
        value = float(price)
        if not math.isfinite(value) or value <= 0.0:
            return False
        if self.min_price is not None and value < self.min_price:
            return False
        if self.max_price is not None and value > self.max_price:
            return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "trigger": self.trigger,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "max_wait_seconds": self.max_wait_seconds,
        }


@dataclass(frozen=True)
class ExitRules:
    """Take-profit, stop, trailing and time exits, in rates off the entry price.

    Rates rather than absolute prices because the entry fill is not known at election
    time. :meth:`resolve` turns them into prices once the fill is known, so the exit
    levels are derived from what was actually paid, not from what was hoped for.
    """

    take_profit_rate: float
    stop_loss_rate: float
    trailing_rate: float | None = None
    max_holding_seconds: int = 600
    #: Strategy-specific exit signal name. The fast loop asks the owning algorithm.
    strategy_exit_trigger: str | None = None

    def __post_init__(self) -> None:
        if self.take_profit_rate <= 0.0:
            raise TradePlanError("take_profit_rate must be positive")
        if self.stop_loss_rate <= 0.0:
            raise TradePlanError("stop_loss_rate must be positive")
        if self.max_holding_seconds <= 0:
            raise TradePlanError("max_holding_seconds must be positive")

    def resolve(self, entry_price: float, *, direction: str) -> dict[str, float | None]:
        """Absolute exit levels for a realised entry price.

        Direction-aware: a short's take-profit sits *below* its entry. Applying the long
        arithmetic to a short would arm a target that only pays when the position is
        losing.
        """
        price = float(entry_price)
        if price <= 0.0:
            return {"take_profit_price": None, "stop_loss_price": None}
        sign = -1.0 if str(direction).upper() == "SHORT" else 1.0
        return {
            "take_profit_price": price * (1.0 + sign * self.take_profit_rate),
            "stop_loss_price": price * (1.0 - sign * self.stop_loss_rate),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "take_profit_rate": self.take_profit_rate,
            "stop_loss_rate": self.stop_loss_rate,
            "trailing_rate": self.trailing_rate,
            "max_holding_seconds": self.max_holding_seconds,
            "strategy_exit_trigger": self.strategy_exit_trigger,
        }


@dataclass(frozen=True)
class TradePlan:
    """One executable decision, complete before execution begins."""

    plan_id: str
    created_at: datetime
    expires_at: datetime
    symbol: str
    market: str
    direction: str
    strategy_id: str
    quantity: int
    max_notional: float
    entry_rule: EntryRule
    exit_rules: ExitRules
    cancel_rule: str
    expected_net_edge_bps: float
    cost_snapshot: Mapping[str, Any]
    risk_snapshot: Mapping[str, Any]
    weekday_time_context: Mapping[str, Any]
    source_ids: tuple[str, ...]
    status: TradePlanStatus = TradePlanStatus.ARMED
    #: Reference price the election priced against. The entry band is built around it.
    reference_price: float | None = None
    #: Set once the entry actually fills. Exit levels resolve off this.
    entry_fill_price: float | None = None
    filled_quantity: int = 0
    #: Set by :meth:`with_broker_clip` when the broker's real cash or sellable quantity
    #: was smaller than the elected size. Never a re-decision, always a reduction.
    broker_clipped_from: int | None = None
    election_context: Mapping[str, Any] = field(default_factory=dict)
    decision_id: str | None = None
    session_id: str | None = None
    #: The execution product / position effect contract, carried so the order builder
    #: does not re-derive it.
    order_contract: Mapping[str, Any] = field(default_factory=dict)

    # -- invariants --------------------------------------------------------- #
    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise TradePlanError(f"{self.plan_id}: quantity must be positive")
        if self.max_notional <= 0.0:
            raise TradePlanError(f"{self.plan_id}: max_notional must be positive")
        if self.expires_at <= self.created_at:
            raise TradePlanError(f"{self.plan_id}: expires_at must follow created_at")
        if str(self.direction).upper() not in {"LONG", "SHORT"}:
            raise TradePlanError(f"{self.plan_id}: direction must be LONG or SHORT")
        if not self.strategy_id:
            raise TradePlanError(f"{self.plan_id}: a plan must name its strategy")

    # -- lifecycle ---------------------------------------------------------- #
    def is_expired(self, now: datetime) -> bool:
        return _aware(now) >= _aware(self.expires_at)

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_PLAN_STATUSES

    def executable(self, now: datetime) -> tuple[bool, str | None]:
        """May this plan produce an order right now, and if not, why not."""
        if self.terminal:
            return False, f"PLAN_TERMINAL:{self.status.value}"
        if self.is_expired(now):
            return False, "PLAN_EXPIRED"
        if self.remaining_quantity <= 0:
            return False, "PLAN_FULLY_FILLED"
        return True, None

    @property
    def remaining_quantity(self) -> int:
        return max(0, int(self.quantity) - int(self.filled_quantity))

    @property
    def is_short(self) -> bool:
        return str(self.direction).upper() == "SHORT"

    # -- permitted transitions ------------------------------------------------ #
    def with_status(self, status: TradePlanStatus) -> "TradePlan":
        return replace(self, status=status)

    def with_broker_clip(self, quantity: int, *, reason: str) -> "TradePlan":
        """Reduce the quantity to what the broker will accept. Never increase it.

        The single sanctioned quantity mutation. It exists because the elected size is
        computed against an account snapshot that can be seconds old, and the broker's
        actual orderable cash or sellable quantity is the one that decides whether an
        order is accepted at all. Raising the quantity here would be a position-sizing
        decision, which is frozen at election.
        """
        clipped = int(quantity)
        if clipped >= self.quantity:
            raise TradePlanError(
                f"{self.plan_id}: broker clip may only reduce quantity "
                f"({self.quantity} -> {clipped})"
            )
        if clipped <= 0:
            raise TradePlanError(f"{self.plan_id}: broker clip must leave a positive quantity")
        return replace(
            self,
            quantity=clipped,
            broker_clipped_from=self.broker_clipped_from or self.quantity,
            risk_snapshot={**dict(self.risk_snapshot), "broker_clip_reason": reason},
        )

    def with_entry_fill(self, price: float, quantity: int) -> "TradePlan":
        """Record the realised entry. Exit levels resolve off this price."""
        filled = max(0, int(quantity))
        if filled > self.quantity:
            raise TradePlanError(
                f"{self.plan_id}: filled {filled} exceeds planned {self.quantity}"
            )
        fill_price = float(price)
        if fill_price <= 0.0:
            raise TradePlanError(f"{self.plan_id}: entry fill price must be positive")
        blended = fill_price
        if self.entry_fill_price is not None and self.filled_quantity > 0:
            increment = filled - self.filled_quantity
            if increment > 0:
                blended = (
                    self.entry_fill_price * self.filled_quantity + fill_price * increment
                ) / filled
            else:
                blended = self.entry_fill_price
        return replace(
            self,
            entry_fill_price=blended,
            filled_quantity=filled,
            status=(
                TradePlanStatus.OPEN
                if filled >= self.quantity
                else TradePlanStatus.ENTERING
            ),
        )

    def exit_levels(self) -> dict[str, float | None]:
        """Absolute exit prices, from the fill when known and the reference otherwise."""
        anchor = self.entry_fill_price or self.reference_price
        if not anchor:
            return {"take_profit_price": None, "stop_loss_price": None}
        return self.exit_rules.resolve(anchor, direction=self.direction)

    # -- immutability contract -------------------------------------------------- #
    #: Fields the election fixes. A mutation of any of these is a new decision and needs
    #: a new plan, which is why nothing in the execution path can write them.
    IMMUTABLE_FIELDS: tuple[str, ...] = (
        "plan_id",
        "strategy_id",
        "symbol",
        "direction",
        "max_notional",
        "expected_net_edge_bps",
        "cost_snapshot",
        "risk_snapshot",
        "weekday_time_context",
        "entry_rule",
        "exit_rules",
        "created_at",
        "expires_at",
    )

    #: Fields that identify this particular plan rather than the decision it encodes.
    #: Excluded from :meth:`decision_fingerprint` so two elections on identical evidence
    #: can be compared for determinism without their ids and clocks getting in the way.
    _IDENTITY_FIELDS: tuple[str, ...] = ("plan_id", "created_at", "expires_at")

    def _frozen_payload(self, *, include_identity: bool) -> dict[str, Any]:
        names = [
            name
            for name in self.IMMUTABLE_FIELDS
            # risk_snapshot carries the broker clip reason, which is a permitted
            # annotation rather than a change to the risk basis.
            if name != "risk_snapshot"
            and (include_identity or name not in self._IDENTITY_FIELDS)
        ]
        payload = {name: _jsonable(getattr(self, name)) for name in names}
        payload["risk_basis"] = _jsonable(
            {
                key: value
                for key, value in dict(self.risk_snapshot).items()
                if key != "broker_clip_reason"
            }
        )
        return payload

    def _hash(self, payload: Mapping[str, Any]) -> str:
        import hashlib

        encoded = json.dumps(dict(payload), sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def immutable_signature(self) -> str:
        """Hash of this plan's frozen fields, for tamper detection in the trace."""
        return self._hash(self._frozen_payload(include_identity=True))

    def decision_fingerprint(self) -> str:
        """Hash of the DECISION, ignoring the plan's own id and clock.

        Two elections run on the same evidence must produce the same fingerprint. That is
        the property the determinism test asserts, and it would be untestable against
        :meth:`immutable_signature`, which by design differs for every plan.
        """
        return self._hash(self._frozen_payload(include_identity=False))

    # -- serialisation ------------------------------------------------------------ #
    def as_dict(self) -> dict[str, Any]:
        levels = self.exit_levels()
        return {
            "plan_id": self.plan_id,
            "created_at": iso_column(self.created_at),
            "expires_at": iso_column(self.expires_at),
            "symbol": self.symbol,
            "market": self.market,
            "direction": self.direction,
            "strategy_id": self.strategy_id,
            "quantity": self.quantity,
            "max_notional": self.max_notional,
            "entry_rule": self.entry_rule.as_dict(),
            "entry_price_range": [self.entry_rule.min_price, self.entry_rule.max_price],
            "take_profit_rule": {
                "rate": self.exit_rules.take_profit_rate,
                "price": levels["take_profit_price"],
            },
            "stop_loss_rule": {
                "rate": self.exit_rules.stop_loss_rate,
                "price": levels["stop_loss_price"],
            },
            "trailing_rule": {"rate": self.exit_rules.trailing_rate},
            "time_exit": {"max_holding_seconds": self.exit_rules.max_holding_seconds},
            "cancel_rule": self.cancel_rule,
            "expected_net_edge": self.expected_net_edge_bps,
            "cost_snapshot": dict(self.cost_snapshot),
            "risk_snapshot": dict(self.risk_snapshot),
            "weekday_time_context": dict(self.weekday_time_context),
            "source_ids": list(self.source_ids),
            "status": self.status.value,
            "reference_price": self.reference_price,
            "entry_fill_price": self.entry_fill_price,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "broker_clipped_from": self.broker_clipped_from,
            "decision_id": self.decision_id,
            "session_id": self.session_id,
            "order_contract": dict(self.order_contract),
            "immutable_signature": self.immutable_signature(),
            "decision_fingerprint": self.decision_fingerprint(),
        }


def _aware(moment: datetime) -> datetime:
    return (
        moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    ).astimezone(timezone.utc)


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return _aware(value).isoformat()
    if isinstance(value, (EntryRule, ExitRules)):
        return value.as_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def new_plan_id(symbol: str, moment: datetime) -> str:
    slug = "".join(ch for ch in str(symbol).upper() if ch.isalnum())[:12] or "NA"
    return f"plan-{slug}-{_aware(moment).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"


#: Individual statements rather than one script: ``executescript`` issues an implicit
#: COMMIT, which would end the surrounding ``BEGIN IMMEDIATE`` and make the store's
#: transaction contract silently untrue for this table.
_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    create table if not exists trade_plan (
        plan_id text primary key,
        created_at text not null,
        expires_at text not null,
        symbol text not null,
        market text not null default '',
        direction text not null default 'LONG',
        strategy_id text not null,
        quantity integer not null,
        max_notional real not null,
        status text not null,
        reference_price real,
        entry_fill_price real,
        filled_quantity integer not null default 0,
        broker_clipped_from integer,
        expected_net_edge_bps real not null default 0.0,
        immutable_signature text not null default '',
        decision_id text,
        session_id text,
        payload_json text not null,
        updated_at text not null
    )
    """,
    "create index if not exists idx_trade_plan_symbol on trade_plan(symbol, created_at)",
    "create index if not exists idx_trade_plan_status on trade_plan(status, created_at)",
)


class TradePlanStore:
    """Durable plan storage. A restart must find the plan that owns a position."""

    def __init__(self, store: TradingStateStore | None = None) -> None:
        self._store = store or default_trading_state_store()
        with self._store.transaction() as conn:
            for statement in _SCHEMA_STATEMENTS:
                conn.execute(statement)

    @property
    def store(self) -> TradingStateStore:
        return self._store

    def save(self, plan: TradePlan, *, now: datetime | None = None) -> TradePlan:
        moment = _aware(now or datetime.now(timezone.utc))
        with self._store.transaction() as conn:
            conn.execute(
                "insert into trade_plan"
                " (plan_id, created_at, expires_at, symbol, market, direction,"
                "  strategy_id, quantity, max_notional, status, reference_price,"
                "  entry_fill_price, filled_quantity, broker_clipped_from,"
                "  expected_net_edge_bps, immutable_signature, decision_id, session_id,"
                "  payload_json, updated_at)"
                " values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " on conflict(plan_id) do update set"
                "  quantity = excluded.quantity, status = excluded.status,"
                "  entry_fill_price = excluded.entry_fill_price,"
                "  filled_quantity = excluded.filled_quantity,"
                "  broker_clipped_from = excluded.broker_clipped_from,"
                "  payload_json = excluded.payload_json,"
                "  updated_at = excluded.updated_at",
                (
                    plan.plan_id,
                    iso_column(plan.created_at),
                    iso_column(plan.expires_at),
                    plan.symbol,
                    plan.market,
                    plan.direction,
                    plan.strategy_id,
                    plan.quantity,
                    plan.max_notional,
                    plan.status.value,
                    plan.reference_price,
                    plan.entry_fill_price,
                    plan.filled_quantity,
                    plan.broker_clipped_from,
                    plan.expected_net_edge_bps,
                    plan.immutable_signature(),
                    plan.decision_id,
                    plan.session_id,
                    json_column(plan.as_dict()),
                    iso_column(moment),
                ),
            )
        return plan

    def get(self, plan_id: str) -> dict[str, Any] | None:
        return self._store.fetch_one(
            "select * from trade_plan where plan_id = ?", (str(plan_id),)
        )

    def active_for_symbol(self, symbol: str) -> dict[str, Any] | None:
        return self._store.fetch_one(
            "select * from trade_plan where symbol = ? and status not in (?, ?, ?)"
            " order by created_at desc limit 1",
            (
                str(symbol).upper(),
                TradePlanStatus.CLOSED.value,
                TradePlanStatus.EXPIRED.value,
                TradePlanStatus.CANCELLED.value,
            ),
        )

    def open_plans(self) -> tuple[dict[str, Any], ...]:
        return self._store.fetch_all(
            "select * from trade_plan where status not in (?, ?, ?) order by created_at",
            (
                TradePlanStatus.CLOSED.value,
                TradePlanStatus.EXPIRED.value,
                TradePlanStatus.CANCELLED.value,
            ),
        )

    def expire_stale(self, *, now: datetime | None = None) -> tuple[str, ...]:
        """Mark past-expiry plans EXPIRED. Called on restart and each slow cycle.

        A plan that owns a position is NOT expired by the clock: the position still needs
        its exit rules, and orphaning it would leave a live holding with no strategy.
        Only plans that never reached OPEN can expire.
        """
        moment = iso_column(_aware(now or datetime.now(timezone.utc)))
        with self._store.transaction() as conn:
            rows = conn.execute(
                "select plan_id from trade_plan where expires_at < ?"
                " and status in (?, ?)",
                (moment, TradePlanStatus.ARMED.value, TradePlanStatus.ENTERING.value),
            ).fetchall()
            plan_ids = [str(row["plan_id"]) for row in rows]
            for plan_id in plan_ids:
                conn.execute(
                    "update trade_plan set status = ?, updated_at = ? where plan_id = ?",
                    (TradePlanStatus.EXPIRED.value, moment, plan_id),
                )
        return tuple(plan_ids)

    def summary(self) -> dict[str, Any]:
        rows = self._store.fetch_all(
            "select status, count(*) as n from trade_plan group by status"
        )
        by_status = {str(row["status"]): int(row["n"]) for row in rows}
        return {
            "by_status": by_status,
            "active_count": sum(
                count
                for status, count in by_status.items()
                if TradePlanStatus(status) not in TERMINAL_PLAN_STATUSES
            ),
        }


_default_plan_store: TradePlanStore | None = None
_plan_store_lock = threading.Lock()


def default_trade_plan_store() -> TradePlanStore:
    global _default_plan_store
    with _plan_store_lock:
        if _default_plan_store is None:
            _default_plan_store = TradePlanStore()
        return _default_plan_store


def reset_default_trade_plan_store() -> None:
    """Test hook. Never called from the trading path."""
    global _default_plan_store
    with _plan_store_lock:
        _default_plan_store = None

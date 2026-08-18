"""The fast loop: the elected strategy decides entry and exit from the tape itself.

What runs here and what does not
---------------------------------
This is the per-tick critical section. It holds a frozen
:class:`~app.trading.trade_plan.TradePlan` and a few floats of strategy-local state, and
it does exactly one thing per tick: decide whether the plan's entry or exit condition has
fired.

It does **not** rebuild the ontology, run the GNN, recompute portfolio risk, recompute the
position size, or re-evaluate profitability. Those were all settled before election and
are frozen in the plan. The prohibition is structural rather than advisory — this module
has no import path to ``app.ontology``, ``app.graph``, ``app.models``, ``app.cost`` or
``app.risk``, and a test asserts it. There is no database read, no network call and no
model inference in :meth:`StrategyFastExecutor.on_tick`.

The state machine
-----------------
::

    WAIT_ENTRY --entry trigger--> ENTERING --fill--> POSITION_OPEN
         |                            |                    |
         |                     reject/cancel          exit trigger
         +--expiry/cancel--> CLOSED <--fill-- EXITING <-----+

Exit precedence
---------------
Stop loss is evaluated **first**, before the take-profit, the trailing stop, the time exit
and the strategy's own signal. Once a stop level is breached the position is losing and
speed is the only thing that helps; checking the profitable exits first would let a tick
that breached the stop be handled as an ordinary trailing update.

No exit is ever suppressed because the trade is currently unprofitable. That is the whole
point of moving the profitability judgement before election: by the time a position
exists, "is this trade worth doing" has already been answered, and the only remaining
question is when to get out.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol

from app.trading.trade_plan import TradePlan, TradePlanStatus

__all__ = [
    "ExecutionRequest",
    "FastLoopState",
    "StrategyFastExecutor",
    "TickEvent",
    "TriggerContext",
]


class FastLoopState(str, Enum):
    WAIT_ENTRY = "WAIT_ENTRY"
    ENTERING = "ENTERING"
    POSITION_OPEN = "POSITION_OPEN"
    EXITING = "EXITING"
    CLOSED = "CLOSED"


TERMINAL_STATES: frozenset[FastLoopState] = frozenset({FastLoopState.CLOSED})


@dataclass(frozen=True)
class TickEvent:
    """One realtime observation. Carries its own clock so latency is measurable."""

    symbol: str
    price: float
    event_time: datetime
    received_time: datetime | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    volume: float | None = None

    @property
    def mid(self) -> float:
        bid = self.best_bid or 0.0
        ask = self.best_ask or 0.0
        if bid > 0.0 and ask >= bid:
            return (bid + ask) / 2.0
        return float(self.price)


@dataclass(frozen=True)
class TriggerContext:
    """What a strategy's own entry/exit algorithm is handed on each tick.

    Deliberately small. Anything a strategy needs that is not in here was supposed to be
    resolved at election and frozen into ``plan.election_context``; adding a live lookup
    would put IO back in the critical section.
    """

    plan: TradePlan
    tick: TickEvent
    state: FastLoopState
    #: Seconds since the position opened. ``None`` before entry.
    holding_seconds: float | None
    #: Best price seen in the position's favour since entry.
    favourable_extreme: float | None
    unrealised_return: float | None
    election_context: Mapping[str, Any] = field(default_factory=dict)


class EntryTrigger(Protocol):
    def __call__(self, context: TriggerContext) -> bool: ...


class ExitTrigger(Protocol):
    def __call__(self, context: TriggerContext) -> str | None: ...


@dataclass(frozen=True)
class ExecutionRequest:
    """What the fast loop wants done. It does not do it itself.

    The executor decides; the coordinator (through the ExecutionGuard) submits. Keeping
    the decision and the submission apart is what lets the guard be the single technical
    checkpoint without the fast loop having to know about brokers.
    """

    action: str  # SUBMIT_ENTRY | SUBMIT_EXIT | CANCEL_ENTRY
    plan_id: str
    symbol: str
    side: str
    quantity: int
    limit_price: float
    reason: str
    urgent: bool = False
    decided_at: datetime | None = None
    tick_event_time: datetime | None = None
    tick_received_time: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "plan_id": self.plan_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "reason": self.reason,
            "urgent": self.urgent,
            "decided_at": _iso(self.decided_at),
            "tick_event_time": _iso(self.tick_event_time),
            "tick_received_time": _iso(self.tick_received_time),
        }


def _iso(moment: datetime | None) -> str | None:
    if moment is None:
        return None
    aware = moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).isoformat()


def _aware(moment: datetime) -> datetime:
    return (
        moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    ).astimezone(timezone.utc)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


#: Exit reasons, in the order they are evaluated. Stop first, always.
EXIT_STOP_LOSS = "STRATEGY_STOP_LOSS"
EXIT_TRAILING = "STRATEGY_TRAILING_STOP"
EXIT_TAKE_PROFIT = "STRATEGY_TAKE_PROFIT"
EXIT_TIME = "STRATEGY_TIME_EXIT"
EXIT_STRATEGY_SIGNAL = "STRATEGY_SIGNAL_EXIT"

#: Exit reasons that must be routed with urgency (marketable pricing, no chase guard).
URGENT_EXIT_REASONS: frozenset[str] = frozenset({EXIT_STOP_LOSS, EXIT_TRAILING})


class StrategyFastExecutor:
    """Per-plan tick state machine. One instance owns one plan.

    Thread-safe for a single feed thread plus dashboard reads. ``on_tick`` is the hot
    path and takes a lock only long enough to read and write its own few fields.
    """

    def __init__(
        self,
        plan: TradePlan,
        *,
        entry_trigger: EntryTrigger | None = None,
        exit_trigger: ExitTrigger | None = None,
    ) -> None:
        self._plan = plan
        self._entry_trigger = entry_trigger or default_entry_trigger
        self._exit_trigger = exit_trigger
        self._lock = threading.RLock()
        self._state = FastLoopState.WAIT_ENTRY
        self._opened_at: datetime | None = None
        self._entry_price: float | None = None
        self._favourable_extreme: float | None = None
        self._last_tick: TickEvent | None = None
        self._exit_reason: str | None = None
        self._tick_count = 0

    # ------------------------------------------------------------------ #
    # inspection
    # ------------------------------------------------------------------ #
    @property
    def plan(self) -> TradePlan:
        with self._lock:
            return self._plan

    @property
    def state(self) -> FastLoopState:
        with self._lock:
            return self._state

    @property
    def exit_reason(self) -> str | None:
        with self._lock:
            return self._exit_reason

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "plan_id": self._plan.plan_id,
                "symbol": self._plan.symbol,
                "strategy_id": self._plan.strategy_id,
                "direction": self._plan.direction,
                "state": self._state.value,
                "entry_price": self._entry_price,
                "favourable_extreme": self._favourable_extreme,
                "opened_at": _iso(self._opened_at),
                "exit_reason": self._exit_reason,
                "tick_count": self._tick_count,
                "exit_levels": self._plan.exit_levels(),
                "quantity": self._plan.quantity,
                "filled_quantity": self._plan.filled_quantity,
            }

    # ------------------------------------------------------------------ #
    # the hot path
    # ------------------------------------------------------------------ #
    def on_tick(self, tick: TickEvent, *, now: datetime | None = None) -> ExecutionRequest | None:
        """One tick in, at most one execution request out.

        No IO, no inference, no re-decision. The only arithmetic is comparing this price
        against levels the plan already fixed.
        """
        moment = _aware(now or tick.event_time)
        price = _finite(tick.price)
        with self._lock:
            self._tick_count += 1
            self._last_tick = tick
            if self._state in TERMINAL_STATES or price is None or price <= 0.0:
                return None

            if self._state is FastLoopState.WAIT_ENTRY:
                return self._evaluate_entry(tick, price, moment)
            if self._state in {FastLoopState.POSITION_OPEN, FastLoopState.ENTERING}:
                self._update_extreme(price)
                if self._state is FastLoopState.ENTERING:
                    # Partially filled: the position is real, so its exits are live even
                    # though the entry is still working.
                    if self._entry_price is None:
                        return None
                return self._evaluate_exit(tick, price, moment)
        return None

    # ------------------------------------------------------------------ #
    def _evaluate_entry(
        self, tick: TickEvent, price: float, now: datetime
    ) -> ExecutionRequest | None:
        plan = self._plan
        executable, why = plan.executable(now)
        if not executable:
            self._state = FastLoopState.CLOSED
            self._exit_reason = why
            return ExecutionRequest(
                action="CANCEL_ENTRY",
                plan_id=plan.plan_id,
                symbol=plan.symbol,
                side="BUY" if not plan.is_short else "SELL",
                quantity=plan.remaining_quantity,
                limit_price=price,
                reason=why or "PLAN_NOT_EXECUTABLE",
                decided_at=now,
                tick_event_time=tick.event_time,
                tick_received_time=tick.received_time,
            )
        if not plan.entry_rule.price_permitted(price):
            return None
        if not self._entry_trigger(self._context(tick, price, now)):
            return None
        self._state = FastLoopState.ENTERING
        return ExecutionRequest(
            action="SUBMIT_ENTRY",
            plan_id=plan.plan_id,
            symbol=plan.symbol,
            side="SELL" if plan.is_short else "BUY",
            quantity=plan.remaining_quantity,
            limit_price=price,
            reason=f"ENTRY_TRIGGER:{plan.entry_rule.trigger}",
            decided_at=now,
            tick_event_time=tick.event_time,
            tick_received_time=tick.received_time,
        )

    def _evaluate_exit(
        self, tick: TickEvent, price: float, now: datetime
    ) -> ExecutionRequest | None:
        plan = self._plan
        reason = self._exit_reason_for(price, now)
        if reason is None:
            return None
        self._state = FastLoopState.EXITING
        self._exit_reason = reason
        return ExecutionRequest(
            action="SUBMIT_EXIT",
            plan_id=plan.plan_id,
            symbol=plan.symbol,
            side="BUY" if plan.is_short else "SELL",
            quantity=max(1, plan.filled_quantity or plan.quantity),
            limit_price=price,
            reason=reason,
            urgent=reason in URGENT_EXIT_REASONS,
            decided_at=now,
            tick_event_time=tick.event_time,
            tick_received_time=tick.received_time,
        )

    def _exit_reason_for(self, price: float, now: datetime) -> str | None:
        """Exit precedence. Stop first — a losing position is the urgent case."""
        plan = self._plan
        entry = self._entry_price
        if entry is None or entry <= 0.0:
            return None
        sign = -1.0 if plan.is_short else 1.0
        levels = plan.exit_rules.resolve(entry, direction=plan.direction)

        stop = levels.get("stop_loss_price")
        if stop is not None:
            breached = price <= stop if not plan.is_short else price >= stop
            if breached:
                return EXIT_STOP_LOSS

        trailing = plan.exit_rules.trailing_rate
        if trailing and self._favourable_extreme:
            extreme = self._favourable_extreme
            trail_level = extreme * (1.0 - sign * trailing)
            breached = price <= trail_level if not plan.is_short else price >= trail_level
            # Only a position already in profit can trail: otherwise the trailing stop is
            # just a second, tighter stop loss, which is not what it is for.
            in_profit = (price - entry) * sign > 0.0 or (extreme - entry) * sign > 0.0
            if breached and in_profit:
                return EXIT_TRAILING

        target = levels.get("take_profit_price")
        if target is not None:
            reached = price >= target if not plan.is_short else price <= target
            if reached:
                return EXIT_TAKE_PROFIT

        if self._opened_at is not None:
            held = (now - self._opened_at).total_seconds()
            if held >= plan.exit_rules.max_holding_seconds:
                return EXIT_TIME

        if self._exit_trigger is not None and self._last_tick is not None:
            signal = self._exit_trigger(self._context(self._last_tick, price, now))
            if signal:
                return str(signal)
        return None

    def _update_extreme(self, price: float) -> None:
        if self._entry_price is None:
            return
        if self._favourable_extreme is None:
            self._favourable_extreme = price
            return
        if self._plan.is_short:
            self._favourable_extreme = min(self._favourable_extreme, price)
        else:
            self._favourable_extreme = max(self._favourable_extreme, price)

    def _context(self, tick: TickEvent, price: float, now: datetime) -> TriggerContext:
        holding = (
            (now - self._opened_at).total_seconds() if self._opened_at is not None else None
        )
        unrealised = None
        if self._entry_price:
            sign = -1.0 if self._plan.is_short else 1.0
            unrealised = sign * (price - self._entry_price) / self._entry_price
        return TriggerContext(
            plan=self._plan,
            tick=tick,
            state=self._state,
            holding_seconds=holding,
            favourable_extreme=self._favourable_extreme,
            unrealised_return=unrealised,
            election_context=self._plan.election_context,
        )

    # ------------------------------------------------------------------ #
    # lifecycle callbacks (driven by fills, not by ticks)
    # ------------------------------------------------------------------ #
    def on_entry_fill(
        self, price: float, quantity: int, *, now: datetime | None = None
    ) -> None:
        moment = _aware(now or datetime.now(timezone.utc))
        with self._lock:
            self._plan = self._plan.with_entry_fill(price, quantity)
            self._entry_price = self._plan.entry_fill_price
            self._favourable_extreme = self._plan.entry_fill_price
            if self._opened_at is None:
                self._opened_at = moment
            self._state = (
                FastLoopState.POSITION_OPEN
                if self._plan.status is TradePlanStatus.OPEN
                else FastLoopState.ENTERING
            )

    def on_exit_fill(self, *, now: datetime | None = None) -> None:
        with self._lock:
            self._plan = self._plan.with_status(TradePlanStatus.CLOSED)
            self._state = FastLoopState.CLOSED

    def on_entry_rejected(self, reason: str) -> None:
        """A rejected entry returns to WAIT_ENTRY; a rejected plan does not.

        The distinction matters: a broker rejection is a transient routing failure and the
        thesis is still valid until the plan expires, whereas an expired or cancelled plan
        must not be retried at all.
        """
        with self._lock:
            if self._plan.terminal:
                self._state = FastLoopState.CLOSED
                self._exit_reason = reason
                return
            self._state = FastLoopState.WAIT_ENTRY

    def cancel(self, reason: str) -> ExecutionRequest | None:
        """Abandon the plan. Emits a cancel when an entry is working."""
        with self._lock:
            previous = self._state
            self._exit_reason = reason
            self._plan = self._plan.with_status(TradePlanStatus.CANCELLED)
            self._state = FastLoopState.CLOSED
            if previous is not FastLoopState.ENTERING or self._last_tick is None:
                return None
            return ExecutionRequest(
                action="CANCEL_ENTRY",
                plan_id=self._plan.plan_id,
                symbol=self._plan.symbol,
                side="SELL" if self._plan.is_short else "BUY",
                quantity=self._plan.remaining_quantity,
                limit_price=float(self._last_tick.price),
                reason=reason,
                decided_at=datetime.now(timezone.utc),
            )

    def force_exit(self, reason: str, *, now: datetime | None = None) -> ExecutionRequest | None:
        """Emergency exit, bypassing every trigger. Used by the supervisor and kill path.

        Never blocked by profitability or portfolio risk — a risk-reducing exit is exactly
        the order that must always be possible.
        """
        moment = _aware(now or datetime.now(timezone.utc))
        with self._lock:
            if self._state in {FastLoopState.WAIT_ENTRY, FastLoopState.CLOSED}:
                return None
            price = float(self._last_tick.price) if self._last_tick else (
                self._entry_price or self._plan.reference_price or 0.0
            )
            if price <= 0.0:
                return None
            self._state = FastLoopState.EXITING
            self._exit_reason = reason
            return ExecutionRequest(
                action="SUBMIT_EXIT",
                plan_id=self._plan.plan_id,
                symbol=self._plan.symbol,
                side="BUY" if self._plan.is_short else "SELL",
                quantity=max(1, self._plan.filled_quantity or self._plan.quantity),
                limit_price=price,
                reason=reason,
                urgent=True,
                decided_at=moment,
            )


def default_entry_trigger(context: TriggerContext) -> bool:
    """Enter as soon as the price is inside the plan's band.

    The fallback for a plan whose strategy supplies no callable of its own. It is
    deliberately permissive: the band, the expiry and the quantity are all already fixed
    by the election, so the only remaining question is whether the price is one the plan
    authorised.
    """
    return context.plan.entry_rule.price_permitted(context.tick.price)

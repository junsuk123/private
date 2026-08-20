"""Forward shadow evaluation: record a plan now, score it from later data only.

The question this answers
------------------------
"Would this short strategy have made money?" cannot be answered by a backtest
here, and cannot be answered by flipping the sign on the long strategy's results.
Both shortcuts are explicitly rejected:

* A backtest over stored bars has no point-in-time borrow data, so it would short
  names that were unborrowable — and unborrowable names are *precisely* the ones
  that drop hardest, so the bias is large and in the flattering direction.
* Sign-flipping a long result assumes symmetric costs. A short pays the same round
  trip PLUS an accruing borrow fee, sells at the bid and covers at the ask (not
  the other way round), and can be recalled mid-thesis.

So evaluation is FORWARD. A :class:`ShadowTradePlan` is written the moment the
signal fires, with the entry reference, the barriers and the borrow observation
frozen into it. It is scored only from data that arrives afterwards.

The three leak defences
-----------------------
1. **Temporal.** :meth:`ShadowFillSimulator.observe` refuses any tick at or before
   the plan's ``signal_at``. The barrier walk therefore cannot see the bar that
   produced the signal.
2. **Borrow.** Executability is decided from the borrow snapshot stored ON the
   plan, never from a fresh lookup at scoring time.
3. **Pricing.** Entry is at the price a marketable order would actually have got —
   the bid for a short entry, the ask to cover. Mid-price fills are refused
   outright, because a mid fill silently awards half the spread on both legs,
   which on a 20bps KRX spread is 20bps of pure fiction per round trip against a
   target of ~180bps.

A plan whose signal was valid but whose borrow was absent is still recorded, as
``signal_valid_but_unexecutable``. It counts toward signal-quality analysis and
toward ``borrow_availability_rate``, and it is excluded from every promotion
statistic — a strategy may not be promoted on trades it could not have taken.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from app.trading.borrow import (
    DEFAULT_MAX_BORROW_FEE_BPS_ANNUALISED,
    BorrowSnapshot,
    borrow_cost_bps,
    evaluate_borrow,
)
from app.trading.directional import (
    DirectionalStrategyKey,
    PositionDirection,
    ShortReasonCodes,
    gross_return_bps,
    stop_breached,
    stop_price,
    target_price,
    target_reached,
)

DEFAULT_SHADOW_STORE_PATH = "data/store/directional-shadow.sqlite3"

# Terminal states of a shadow plan.
OUTCOME_TARGET = "TARGET"
OUTCOME_STOP = "STOP"
OUTCOME_TIME = "MAX_HOLDING_TIME"
OUTCOME_UNFILLED = "ENTRY_NOT_FILLED"
OUTCOME_UNEXECUTABLE = "SIGNAL_VALID_BUT_UNEXECUTABLE"
OUTCOME_BORROW_RECALLED = "BORROW_RECALLED"
OUTCOME_EXPIRED = "EXPIRED_WITHOUT_RESOLUTION"

_TERMINAL_OUTCOMES: frozenset[str] = frozenset(
    {
        OUTCOME_TARGET,
        OUTCOME_STOP,
        OUTCOME_TIME,
        OUTCOME_UNFILLED,
        OUTCOME_UNEXECUTABLE,
        OUTCOME_BORROW_RECALLED,
        OUTCOME_EXPIRED,
    }
)

# Outcomes that produced a real (simulated) round trip and therefore a net return.
_SCORED_OUTCOMES: frozenset[str] = frozenset(
    {OUTCOME_TARGET, OUTCOME_STOP, OUTCOME_TIME, OUTCOME_BORROW_RECALLED}
)


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return _aware(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class QuoteObservation:
    """One post-signal market observation used to walk the barriers.

    Bid and ask are both required. A "last price" alone cannot price either leg of
    a short honestly: the entry sells into the bid and the cover buys the ask, and
    substituting the last print for either is worth roughly half a spread each way.
    """

    observed_at: datetime
    bid_price: float
    ask_price: float
    last_price: float | None = None
    # Top-of-book sizes, when the feed supplies them. ``None`` means unknown, which is
    # treated as "fills in full" — the alternative (assume zero) would refuse every
    # plan on a feed that omits depth, turning a data gap into a permanent block.
    bid_size: float | None = None
    ask_size: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_at", _aware(self.observed_at))

    @property
    def usable(self) -> bool:
        return self.bid_price > 0 and self.ask_price > 0 and self.ask_price >= self.bid_price

    @property
    def mid(self) -> float:
        return (self.bid_price + self.ask_price) / 2.0

    def entry_reference(self, direction: PositionDirection) -> float:
        """Price an OPENING marketable order would realistically achieve.

        A long buys the ask, a short sells the bid — the unfavourable side in both
        cases. Never the mid.
        """
        return self.ask_price if direction is PositionDirection.LONG else self.bid_price

    def exit_reference(self, direction: PositionDirection) -> float:
        """Price a CLOSING marketable order would realistically achieve.

        A long sells the bid, a short covers at the ask. Again the unfavourable
        side, which is the whole point: the two unfavourable sides are the spread,
        and the spread is the cost being measured.
        """
        return self.bid_price if direction is PositionDirection.LONG else self.ask_price

    def spread_bps(self) -> float:
        mid = self.mid
        if mid <= 0:
            return 0.0
        return (self.ask_price - self.bid_price) / mid * 10_000.0


@dataclass(frozen=True)
class ShadowTradePlan:
    """A short (or long) signal frozen at the moment it fired.

    Everything needed to score it later lives here. In particular the borrow
    snapshot is EMBEDDED rather than referenced by symbol, so a later scoring pass
    physically cannot consult a fresher locate.
    """

    plan_id: str
    key: DirectionalStrategyKey
    symbol: str
    signal_at: datetime
    # Reference price at signal time, on the side the entry would actually pay.
    entry_reference_price: float
    target_rate: float
    stop_rate: float
    max_holding_seconds: int
    # Round-trip cost excluding borrow, in bps, from the cost engine at signal time.
    expected_trading_cost_bps: float
    predicted_gross_edge_bps: float | None = None
    predicted_net_edge_bps: float | None = None
    predicted_success_probability: float | None = None
    regime: str = "UNKNOWN"
    signal_reason_codes: tuple[str, ...] = ()
    # The borrow world AS OBSERVED at signal time. ``None`` means no locate was
    # obtained, which makes the plan unexecutable rather than merely uncertain.
    borrow_snapshot: BorrowSnapshot | None = None
    borrow_reason_codes: tuple[str, ...] = ()
    intended_quantity: int = 1
    spread_bps_at_signal: float | None = None
    liquidity_score_at_signal: float | None = None
    feature_snapshot_id: str = ""
    model_version: str = ""
    deployment_state: str = "SHADOW"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", str(self.symbol or "").strip().upper())
        object.__setattr__(self, "signal_at", _aware(self.signal_at))
        if not self.plan_id:
            object.__setattr__(self, "plan_id", f"shadow-{uuid4().hex}")

    @property
    def direction(self) -> PositionDirection:
        return self.key.direction

    @property
    def target(self) -> float:
        return target_price(self.entry_reference_price, self.target_rate, self.direction)

    @property
    def stop(self) -> float:
        return stop_price(self.entry_reference_price, self.stop_rate, self.direction)

    def executable(self, *, max_fee_bps_annualised: float = DEFAULT_MAX_BORROW_FEE_BPS_ANNUALISED) -> tuple[bool, tuple[str, ...]]:
        """Could this signal have been executed, judged on signal-time evidence?

        Uses the SAME ``evaluate_borrow`` rule the live preflight uses. If the two
        diverged, shadow executability would stop predicting live executability and
        the promotion ladder would be measuring a fiction.
        """
        if self.direction is PositionDirection.LONG:
            return True, ()
        verdict = evaluate_borrow(
            self.borrow_snapshot,
            quantity=max(1, self.intended_quantity),
            # Evaluated AT SIGNAL TIME, not now: the freshness question is whether
            # the locate was fresh when the decision was made.
            now=self.signal_at,
            max_fee_bps_annualised=max_fee_bps_annualised,
            min_hours_before_deadline=None,
        )
        return verdict.allowed, verdict.reason_codes

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            **self.key.as_dict(),
            "symbol": self.symbol,
            "signal_at": self.signal_at.isoformat(),
            "entry_reference_price": self.entry_reference_price,
            "target_price": self.target,
            "stop_price": self.stop,
            "target_rate": self.target_rate,
            "stop_rate": self.stop_rate,
            "max_holding_seconds": self.max_holding_seconds,
            "expected_trading_cost_bps": self.expected_trading_cost_bps,
            "predicted_gross_edge_bps": self.predicted_gross_edge_bps,
            "predicted_net_edge_bps": self.predicted_net_edge_bps,
            "predicted_success_probability": self.predicted_success_probability,
            "regime": self.regime,
            "signal_reason_codes": list(self.signal_reason_codes),
            "borrow_snapshot": (
                self.borrow_snapshot.as_dict() if self.borrow_snapshot else None
            ),
            "borrow_reason_codes": list(self.borrow_reason_codes),
            "intended_quantity": self.intended_quantity,
            "spread_bps_at_signal": self.spread_bps_at_signal,
            "liquidity_score_at_signal": self.liquidity_score_at_signal,
            "feature_snapshot_id": self.feature_snapshot_id,
            "model_version": self.model_version,
            "deployment_state": self.deployment_state,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class ShadowOutcome:
    """The scored result of one shadow plan."""

    plan_id: str
    key: DirectionalStrategyKey
    symbol: str
    signal_at: datetime
    resolved_at: datetime
    outcome: str
    executable: bool
    regime: str = "UNKNOWN"
    entry_price: float | None = None
    exit_price: float | None = None
    holding_seconds: float | None = None
    gross_return_bps: float | None = None
    trading_cost_bps: float | None = None
    borrow_cost_bps: float | None = None
    net_return_bps: float | None = None
    slippage_error_bps: float | None = None
    max_adverse_excursion_bps: float | None = None
    max_favorable_excursion_bps: float | None = None
    fill_ratio: float = 1.0
    reason_codes: tuple[str, ...] = ()
    #: The gross edge the algorithm claimed at signal time. Carried onto the
    #: outcome so the calibrator can pair a claim with what it paid; without it
    #: the pairing would need a second lookup against the plan journal.
    predicted_gross_edge_bps: float | None = None
    #: Did the signal that produced this plan clear its own entry gate?
    #: False for a plan the algorithm layer already refused (below the
    #: cost floor, non-positive edge). Such a plan is still walked and
    #: journalled -- it is how the rejected region stays measurable -- but it
    #: is NOT evidence about trades the executor would place, and the bandit
    #: posterior must not read it as such.
    signal_admissible: bool = True

    @property
    def scored(self) -> bool:
        """Did this produce a usable net return for promotion statistics?"""
        return self.outcome in _SCORED_OUTCOMES and self.net_return_bps is not None

    @property
    def cost_coverage_ratio(self) -> float | None:
        total = (self.trading_cost_bps or 0.0) + (self.borrow_cost_bps or 0.0)
        if total <= 0 or self.gross_return_bps is None:
            return None
        return self.gross_return_bps / total

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            **self.key.as_dict(),
            "symbol": self.symbol,
            "signal_at": self.signal_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat(),
            "outcome": self.outcome,
            "executable": self.executable,
            "regime": self.regime,
            "scored": self.scored,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "holding_seconds": self.holding_seconds,
            "gross_return_bps": self.gross_return_bps,
            "trading_cost_bps": self.trading_cost_bps,
            "borrow_cost_bps": self.borrow_cost_bps,
            "net_return_bps": self.net_return_bps,
            "cost_coverage_ratio": self.cost_coverage_ratio,
            "slippage_error_bps": self.slippage_error_bps,
            "max_adverse_excursion_bps": self.max_adverse_excursion_bps,
            "max_favorable_excursion_bps": self.max_favorable_excursion_bps,
            "fill_ratio": self.fill_ratio,
            "reason_codes": list(self.reason_codes),
        }


@dataclass
class _OpenWalk:
    """Mutable barrier-walk state for one in-flight plan."""

    plan: ShadowTradePlan
    entry_price: float | None = None
    entry_at: datetime | None = None
    best_price: float | None = None
    worst_price: float | None = None
    last_exit_price: float | None = None
    fill_ratio: float = 1.0
    last_observed_at: datetime | None = None


class ShadowFillSimulator:
    """Walks post-signal quotes to a terminal barrier, with realistic pricing.

    One instance owns a set of in-flight plans. Feed it quotes with
    :meth:`observe`; it returns any plans that resolved on that quote.
    """

    def __init__(
        self,
        *,
        max_borrow_fee_bps_annualised: float = DEFAULT_MAX_BORROW_FEE_BPS_ANNUALISED,
        # Extra bps charged against the entry to represent the difference between a
        # passive limit at the reference and what a real order gets. Applied on the
        # unfavourable side only.
        entry_slippage_bps: float = 2.0,
        exit_slippage_bps: float = 3.0,
    ) -> None:
        self.max_borrow_fee_bps_annualised = max_borrow_fee_bps_annualised
        self.entry_slippage_bps = max(0.0, entry_slippage_bps)
        # Exiting is the more urgent leg (a stop-out is not patient), so it is
        # charged more. For a short the exit is a BUY under pressure, which is the
        # single worst execution in this system.
        self.exit_slippage_bps = max(0.0, exit_slippage_bps)
        self._open: dict[str, _OpenWalk] = {}

    # -- lifecycle ---------------------------------------------------------- #
    def submit(self, plan: ShadowTradePlan) -> ShadowOutcome | None:
        """Register a plan. Returns an immediate outcome if it is unexecutable.

        An unexecutable plan resolves at once rather than being walked: there is no
        position to walk, and pretending otherwise would put a fictional trade into
        the promotion statistics.
        """
        executable, reasons = plan.executable(max_fee_bps_annualised=self.max_borrow_fee_bps_annualised)
        if not executable:
            return ShadowOutcome(
                plan_id=plan.plan_id,
                key=plan.key,
                symbol=plan.symbol,
                signal_at=plan.signal_at,
                resolved_at=plan.signal_at,
                outcome=OUTCOME_UNEXECUTABLE,
                executable=False,
                regime=plan.regime,
                reason_codes=tuple(reasons) or (ShortReasonCodes.BORROW_UNAVAILABLE,),
            )
        self._open[plan.plan_id] = _OpenWalk(plan=plan)
        return None

    def observe_symbol(
        self, symbol: str, quote: QuoteObservation
    ) -> tuple[ShadowOutcome, ...]:
        """Advance only the plans on ``symbol``.

        The routing is not a convenience: feeding one symbol's book to another
        symbol's plan would fabricate the entire price path, and every barrier
        decision after that is fiction. :meth:`observe` (no symbol) exists for
        single-symbol tests and callers that manage routing themselves.
        """
        return self.observe(quote, symbol=symbol)

    def observe(
        self, quote: QuoteObservation, *, symbol: str | None = None
    ) -> tuple[ShadowOutcome, ...]:
        """Advance in-flight plans on one post-signal quote.

        With ``symbol`` given, only that symbol's plans advance. Without it, every
        in-flight plan does — correct when the caller owns a single symbol, and
        wrong for a multi-symbol feed, which is why the engine path uses
        :meth:`observe_symbol`.
        """
        if not quote.usable:
            return ()
        wanted = str(symbol or "").strip().upper() or None
        resolved: list[ShadowOutcome] = []
        for plan_id, walk in list(self._open.items()):
            if wanted is not None and walk.plan.symbol != wanted:
                continue
            # THE temporal leak defence. A quote at or before the signal moment is
            # the data the signal was computed from (or older), and using it to
            # decide the entry fill would let the plan enter at a price it already
            # knew. Strictly greater-than, not >=: an identical timestamp is the
            # same instant, not a later one.
            if quote.observed_at <= walk.plan.signal_at:
                continue
            # The engine may reuse its latest cached book across several cycles.
            # Advancing twice on the same timestamp creates a fictional zero-time
            # exit (often an immediate spread-driven stop) from one observation.
            if walk.last_observed_at is not None and quote.observed_at <= walk.last_observed_at:
                continue
            walk.last_observed_at = quote.observed_at
            outcome = self._advance(walk, quote)
            if outcome is not None:
                resolved.append(outcome)
                self._open.pop(plan_id, None)
        return tuple(resolved)

    def expire(self, now: datetime) -> tuple[ShadowOutcome, ...]:
        """Resolve plans that ran out of horizon without another quote arriving.

        A plan that simply stops receiving data is recorded as EXPIRED (unfilled) or
        closed at its last observed mark. Leaving it open forever would silently
        drop every plan whose feed died — and a feed dies most often exactly when
        the market moves, so the survivors would be a biased sample.
        """
        moment = _aware(now)
        resolved: list[ShadowOutcome] = []
        for plan_id, walk in list(self._open.items()):
            plan = walk.plan
            deadline_from = walk.entry_at or plan.signal_at
            if (moment - deadline_from).total_seconds() < plan.max_holding_seconds:
                continue
            if walk.entry_price is None:
                resolved.append(
                    ShadowOutcome(
                        plan_id=plan.plan_id,
                        key=plan.key,
                        symbol=plan.symbol,
                        signal_at=plan.signal_at,
                        resolved_at=moment,
                        outcome=OUTCOME_UNFILLED,
                        executable=True,
                        regime=plan.regime,
                        reason_codes=("SHADOW_ENTRY_NOT_FILLED_IN_HORIZON",),
                    )
                )
            elif walk.last_exit_price is None:
                # An entry observation is not also an executable exit. If no later
                # quote arrived, there is no honest round trip to score; charging
                # cost against the entry price manufactured a loss from missing
                # data and polluted the strategy posterior.
                resolved.append(
                    ShadowOutcome(
                        plan_id=plan.plan_id,
                        key=plan.key,
                        symbol=plan.symbol,
                        signal_at=plan.signal_at,
                        resolved_at=moment,
                        outcome=OUTCOME_EXPIRED,
                        executable=True,
                        regime=plan.regime,
                        entry_price=walk.entry_price,
                        reason_codes=("SHADOW_HORIZON_ELAPSED_NO_EXIT_QUOTE",),
                    )
                )
            else:
                resolved.append(
                    self._settle(
                        walk,
                        # The time-horizon exit is what was executable most recently,
                        # not the most adverse quote seen anywhere in the walk.
                        exit_price=walk.last_exit_price,
                        resolved_at=moment,
                        outcome=OUTCOME_EXPIRED,
                        reasons=("SHADOW_HORIZON_ELAPSED_NO_QUOTE",),
                    )
                )
            self._open.pop(plan_id, None)
        return tuple(resolved)

    @property
    def open_plan_count(self) -> int:
        return len(self._open)

    @property
    def open_symbols(self) -> tuple[str, ...]:
        """Symbols that still need post-signal quotes, in stable order."""

        return tuple(dict.fromkeys(walk.plan.symbol for walk in self._open.values()))

    # -- internals ---------------------------------------------------------- #
    def _advance(self, walk: _OpenWalk, quote: QuoteObservation) -> ShadowOutcome | None:
        plan = walk.plan
        direction = plan.direction
        if walk.entry_price is None:
            # Entry fills at the executable price on the unfavourable side, charged
            # the entry slippage. A short entry that requires selling BELOW its
            # reference is still filled — the reference was a hope, the bid is the
            # fact — and the resulting slippage error is recorded, which is what
            # makes ``mean_slippage_error_bps`` a real calibration measurement
            # rather than a tautology.
            raw = quote.entry_reference(direction)
            walk.entry_price = _apply_slippage(raw, self.entry_slippage_bps, direction, opening=True)
            walk.entry_at = quote.observed_at
            walk.best_price = walk.entry_price
            walk.worst_price = walk.entry_price
            # Fill ratio from the size actually resting on the entry side. A short
            # sells into the BID, so bid size is what limits it. Recording 1.0
            # unconditionally would let a plan claim a full position in a book that
            # could only absorb a fraction of it — and thin books are exactly where
            # these strategies want to fire, so the bias is systematic.
            walk.fill_ratio = _fill_ratio(quote, direction, plan.intended_quantity)
            # A barrier cannot be hit on the entry quote itself: that would mean
            # exiting at the same instant as entering, which no execution path can do.
            return None

        exit_reference = quote.exit_reference(direction)
        walk.last_exit_price = exit_reference
        walk.best_price = _better(walk.best_price, exit_reference, direction)
        walk.worst_price = _worse(walk.worst_price, exit_reference, direction)

        # Ordering matters and is deliberately pessimistic: when a single quote
        # straddles both barriers, the STOP is taken. Within one observation there is
        # no way to know which came first, and assuming the target would
        # systematically flatter every volatile trade — the population these
        # strategies live in.
        if stop_breached(exit_reference, plan.stop, direction):
            return self._settle(
                walk,
                exit_price=exit_reference,
                resolved_at=quote.observed_at,
                outcome=OUTCOME_STOP,
                reasons=("SHADOW_STOP",),
            )
        if target_reached(exit_reference, plan.target, direction):
            return self._settle(
                walk,
                exit_price=exit_reference,
                resolved_at=quote.observed_at,
                outcome=OUTCOME_TARGET,
                reasons=("SHADOW_TARGET",),
            )
        held = (quote.observed_at - (walk.entry_at or plan.signal_at)).total_seconds()
        if held >= plan.max_holding_seconds:
            return self._settle(
                walk,
                exit_price=exit_reference,
                resolved_at=quote.observed_at,
                outcome=OUTCOME_TIME,
                reasons=("SHADOW_MAX_HOLDING_TIME",),
            )
        # A recall during the hold is a forced cover at whatever the book offers.
        deadline = (
            plan.borrow_snapshot.return_deadline if plan.borrow_snapshot else None
        )
        if deadline is not None and quote.observed_at >= deadline:
            return self._settle(
                walk,
                exit_price=exit_reference,
                resolved_at=quote.observed_at,
                outcome=OUTCOME_BORROW_RECALLED,
                reasons=(ShortReasonCodes.RECALL_DEADLINE_NEAR, "SHADOW_FORCED_COVER"),
            )
        return None

    def _settle(
        self,
        walk: _OpenWalk,
        *,
        exit_price: float,
        resolved_at: datetime,
        outcome: str,
        reasons: tuple[str, ...],
    ) -> ShadowOutcome:
        plan = walk.plan
        direction = plan.direction
        entry = walk.entry_price or plan.entry_reference_price
        settled_exit = _apply_slippage(exit_price, self.exit_slippage_bps, direction, opening=False)
        gross = gross_return_bps(entry, settled_exit, direction)
        holding = (resolved_at - (walk.entry_at or plan.signal_at)).total_seconds()
        trading_cost = max(0.0, float(plan.expected_trading_cost_bps))
        borrow = (
            borrow_cost_bps(
                plan.borrow_snapshot.borrow_fee_bps_annualised if plan.borrow_snapshot else None,
                holding,
            )
            if direction is PositionDirection.SHORT
            else 0.0
        )
        # An unpriced borrow on a SHORT cannot be scored as zero cost. Rather than
        # fabricate a number, the round trip is charged the policy ceiling — the
        # conservative direction — and the reason is recorded.
        extra_reasons: list[str] = list(reasons)
        if direction is PositionDirection.SHORT and borrow is None:
            borrow = borrow_cost_bps(self.max_borrow_fee_bps_annualised, holding) or 0.0
            extra_reasons.append("SHADOW_BORROW_FEE_UNKNOWN_CHARGED_AT_CEILING")
        borrow = float(borrow or 0.0)
        net = gross - trading_cost - borrow
        if walk.fill_ratio < 1.0:
            extra_reasons.append("SHADOW_PARTIAL_FILL")
        # Prediction error against what the model claimed at signal time. This is
        # the input to ``prediction_calibration_error``, and it only means anything
        # because the prediction was frozen BEFORE the outcome was known.
        slippage_error = (
            net - float(plan.predicted_net_edge_bps)
            if plan.predicted_net_edge_bps is not None
            else None
        )
        adverse = (
            abs(gross_return_bps(entry, walk.worst_price, direction))
            if walk.worst_price
            else None
        )
        favorable = (
            abs(gross_return_bps(entry, walk.best_price, direction))
            if walk.best_price
            else None
        )
        return ShadowOutcome(
            signal_admissible=plan_signal_admissible(plan),
            predicted_gross_edge_bps=plan.predicted_gross_edge_bps,
            plan_id=plan.plan_id,
            key=plan.key,
            symbol=plan.symbol,
            signal_at=plan.signal_at,
            resolved_at=resolved_at,
            outcome=outcome,
            executable=True,
            regime=plan.regime,
            entry_price=entry,
            exit_price=settled_exit,
            holding_seconds=holding,
            gross_return_bps=gross,
            trading_cost_bps=trading_cost,
            borrow_cost_bps=borrow,
            net_return_bps=net,
            slippage_error_bps=slippage_error,
            max_adverse_excursion_bps=adverse,
            max_favorable_excursion_bps=favorable,
            fill_ratio=walk.fill_ratio,
            reason_codes=tuple(dict.fromkeys(extra_reasons)),
        )


def _fill_ratio(
    quote: QuoteObservation, direction: PositionDirection, intended_quantity: int
) -> float:
    """Fraction of the intended size top-of-book could absorb.

    A short SELLS to open, so the bid side is the constraint; a long buys the ask.
    Unknown depth returns 1.0 — a feed that omits sizes must not silently refuse every
    plan, which would turn a data gap into a permanent block.
    """
    wanted = max(1, int(intended_quantity or 1))
    available = quote.bid_size if direction is PositionDirection.SHORT else quote.ask_size
    if available is None:
        return 1.0
    return max(0.0, min(1.0, float(available) / wanted))


#: Reason codes the algorithm layer emits when it REFUSES a signal. A plan
#: carrying one of these was never a tradeable candidate: ``_fire`` returned a
#: rejection and the executor would not have placed it.
SIGNAL_REJECTION_CODES: frozenset[str] = frozenset(
    {
        "EDGE_BELOW_COST_FLOOR",
        "EDGE_BELOW_ALGORITHM_FLOOR",
        "TECHNICAL_EDGE_NON_POSITIVE",
    }
)


def plan_signal_admissible(plan: "ShadowTradePlan") -> bool:
    """Would the entry gate have let this plan through?

    Two independent ways to fail, because the two layers can disagree: an
    explicit rejection code from ``_fire``, or a predicted net edge that cannot
    pay for the round trip. Either one means the live executor would not have
    taken the trade, so its realized outcome says nothing about the strategy's
    live performance -- it measures the rejected region.
    """
    if set(plan.signal_reason_codes) & SIGNAL_REJECTION_CODES:
        return False
    predicted = plan.predicted_net_edge_bps
    return predicted is None or predicted > 0.0


def _apply_slippage(
    price: float, slippage_bps: float, direction: PositionDirection, *, opening: bool
) -> float:
    """Move ``price`` against the position by ``slippage_bps``.

    Against, always. Working out which way that is: opening a short SELLS, so an
    adverse move is DOWN; closing a short BUYS, so adverse is UP. The long legs are
    the mirror. Getting this sign wrong would award slippage as profit.
    """
    if price <= 0 or slippage_bps <= 0:
        return price
    factor = slippage_bps / 10_000.0
    selling = (direction is PositionDirection.SHORT) == opening
    return price * (1.0 - factor) if selling else price * (1.0 + factor)


def _better(current: float | None, candidate: float, direction: PositionDirection) -> float:
    if current is None:
        return candidate
    return (
        max(current, candidate)
        if direction is PositionDirection.LONG
        else min(current, candidate)
    )


def _worse(current: float | None, candidate: float, direction: PositionDirection) -> float:
    if current is None:
        return candidate
    return (
        min(current, candidate)
        if direction is PositionDirection.LONG
        else max(current, candidate)
    )


class ShadowPlanStore:
    """Durable journal of shadow plans and their resolved outcomes.

    Separate from the realized-outcome store because the two answer different
    questions: this one keeps the FULL plan (including the frozen borrow snapshot
    and the model's prediction) so an audit can reconstruct why a strategy was
    promoted. Scored outcomes are ALSO written to the performance store, which is
    what the bandit reads.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(
            path or os.getenv("DIRECTIONAL_SHADOW_STORE_PATH", DEFAULT_SHADOW_STORE_PATH)
        )
        self._lock = threading.RLock()
        self._available = True
        self._migrate()

    def record_plan(self, plan: ShadowTradePlan) -> bool:
        if not self._available:
            return False
        try:
            with self._lock, closing(self._connect()) as conn:
                conn.execute(
                    """
                    insert or replace into shadow_plans(
                        plan_id, strategy_key, strategy_id, direction, market,
                        execution_product, symbol, signal_at, regime, plan_json
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        plan.plan_id,
                        plan.key.as_text(),
                        plan.key.strategy_id,
                        str(plan.key.direction),
                        plan.key.market,
                        str(plan.key.execution_product),
                        plan.symbol,
                        plan.signal_at.isoformat(),
                        plan.regime,
                        json.dumps(plan.as_dict(), ensure_ascii=False, default=str),
                    ),
                )
                conn.commit()
        except sqlite3.Error:
            return False
        return True

    def has_recent_plan(
        self,
        key: DirectionalStrategyKey,
        symbol: str,
        *,
        since: datetime,
    ) -> bool:
        """Whether this arm already emitted a correlated recent observation."""
        if not self._available:
            return True
        try:
            with self._lock, closing(self._connect()) as conn:
                row = conn.execute(
                    """
                    select 1 from shadow_plans
                    where strategy_key = ? and symbol = ? and signal_at >= ?
                    limit 1
                    """,
                    (key.as_text(), str(symbol).upper(), _aware(since).isoformat()),
                ).fetchone()
        except sqlite3.Error:
            # Fail closed: a journal lookup failure must not manufacture samples.
            return True
        return row is not None

    def record_outcome(self, outcome: ShadowOutcome) -> bool:
        if not self._available:
            return False
        try:
            with self._lock, closing(self._connect()) as conn:
                conn.execute(
                    """
                    insert or replace into shadow_outcomes(
                        plan_id, strategy_key, strategy_id, direction, market,
                        execution_product, symbol, signal_at, resolved_at, outcome,
                        executable, scored, net_return_bps, gross_return_bps,
                        trading_cost_bps, borrow_cost_bps, holding_seconds,
                        slippage_error_bps, outcome_json
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        outcome.plan_id,
                        outcome.key.as_text(),
                        outcome.key.strategy_id,
                        str(outcome.key.direction),
                        outcome.key.market,
                        str(outcome.key.execution_product),
                        outcome.symbol,
                        outcome.signal_at.isoformat(),
                        outcome.resolved_at.isoformat(),
                        outcome.outcome,
                        int(bool(outcome.executable)),
                        int(bool(outcome.scored)),
                        _finite(outcome.net_return_bps),
                        _finite(outcome.gross_return_bps),
                        _finite(outcome.trading_cost_bps),
                        _finite(outcome.borrow_cost_bps),
                        _finite(outcome.holding_seconds),
                        _finite(outcome.slippage_error_bps),
                        json.dumps(outcome.as_dict(), ensure_ascii=False, default=str),
                    ),
                )
                conn.commit()
        except sqlite3.Error:
            return False
        return True

    def outcomes(
        self,
        key: DirectionalStrategyKey | None = None,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        scored_only: bool = False,
        limit: int = 2000,
    ) -> tuple[dict[str, Any], ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if key is not None:
            clauses.append("strategy_key = ?")
            params.append(key.as_text())
        if since is not None:
            clauses.append("signal_at >= ?")
            params.append(_aware(since).isoformat())
        if until is not None:
            clauses.append("signal_at < ?")
            params.append(_aware(until).isoformat())
        if scored_only:
            clauses.append("scored = 1")
        where = f"where {' and '.join(clauses)} " if clauses else ""
        params.append(max(1, int(limit)))
        try:
            with self._lock, closing(self._connect()) as conn:
                rows = conn.execute(
                    f"select outcome_json from shadow_outcomes {where}"
                    "order by signal_at desc, rowid desc limit ?",
                    params,
                ).fetchall()
        except sqlite3.Error:
            return ()
        parsed: list[dict[str, Any]] = []
        for row in rows:
            try:
                parsed.append(json.loads(row[0]))
            except (TypeError, ValueError):
                continue
        return tuple(parsed)

    def plan(self, plan_id: str) -> dict[str, Any] | None:
        try:
            with self._lock, closing(self._connect()) as conn:
                row = conn.execute(
                    "select plan_json from shadow_plans where plan_id = ?", (str(plan_id),)
                ).fetchone()
        except sqlite3.Error:
            return None
        if not row:
            return None
        try:
            return json.loads(row[0])
        except (TypeError, ValueError):
            return None

    def summary(self, *, limit: int = 100) -> dict[str, Any]:
        try:
            with self._lock, closing(self._connect()) as conn:
                rows = conn.execute(
                    """
                    select strategy_key, count(*),
                           sum(case when scored = 1 then 1 else 0 end),
                           sum(case when executable = 1 then 1 else 0 end),
                           avg(case when scored = 1 then net_return_bps else null end),
                           max(resolved_at)
                    from shadow_outcomes group by strategy_key
                    order by max(resolved_at) desc limit ?
                    """,
                    (max(1, int(limit)),),
                ).fetchall()
        except sqlite3.Error:
            rows = []
        return {
            "store_path": str(self.path),
            "available": self._available,
            "arms": [
                {
                    "strategy_key": str(row[0]),
                    "signal_count": int(row[1] or 0),
                    "scored_count": int(row[2] or 0),
                    "executable_count": int(row[3] or 0),
                    "mean_net_return_bps": (
                        round(float(row[4]), 3) if row[4] is not None else None
                    ),
                    "last_resolved_at": row[5],
                }
                for row in rows
            ],
        }

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=15)
        conn.execute("pragma journal_mode = wal")
        return conn

    def _migrate(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, closing(self._connect()) as conn:
                conn.executescript(
                    """
                    create table if not exists shadow_plans (
                        plan_id text primary key,
                        strategy_key text not null,
                        strategy_id text not null,
                        direction text not null,
                        market text not null,
                        execution_product text not null,
                        symbol text not null,
                        signal_at text not null,
                        regime text,
                        plan_json text not null
                    );
                    create index if not exists idx_shadow_plans_key
                        on shadow_plans(strategy_key, signal_at desc);
                    create table if not exists shadow_outcomes (
                        plan_id text primary key,
                        strategy_key text not null,
                        strategy_id text not null,
                        direction text not null,
                        market text not null,
                        execution_product text not null,
                        symbol text not null,
                        signal_at text not null,
                        resolved_at text not null,
                        outcome text not null,
                        executable integer not null,
                        scored integer not null,
                        net_return_bps real,
                        gross_return_bps real,
                        trading_cost_bps real,
                        borrow_cost_bps real,
                        holding_seconds real,
                        slippage_error_bps real,
                        outcome_json text not null
                    );
                    create index if not exists idx_shadow_outcomes_key
                        on shadow_outcomes(strategy_key, signal_at desc);
                    create index if not exists idx_shadow_outcomes_resolved
                        on shadow_outcomes(resolved_at desc);
                    """
                )
                conn.commit()
        except (OSError, sqlite3.Error):
            # No shadow journal means no promotion evidence accumulates, so nothing
            # can be promoted. That is the correct failure direction.
            self._available = False


_DEFAULT_SHADOW_STORE: ShadowPlanStore | None = None
_DEFAULT_SHADOW_LOCK = threading.Lock()


def default_shadow_store() -> ShadowPlanStore:
    global _DEFAULT_SHADOW_STORE
    if _DEFAULT_SHADOW_STORE is None:
        with _DEFAULT_SHADOW_LOCK:
            if _DEFAULT_SHADOW_STORE is None:
                _DEFAULT_SHADOW_STORE = ShadowPlanStore()
    return _DEFAULT_SHADOW_STORE


def reset_default_shadow_store() -> None:
    global _DEFAULT_SHADOW_STORE
    with _DEFAULT_SHADOW_LOCK:
        _DEFAULT_SHADOW_STORE = None

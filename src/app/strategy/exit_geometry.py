"""Single authority for per-strategy exit geometry AND its training labels.

The defect this fixes
---------------------
The live session exited ``intraday_momentum`` at -22bps / +100bps over at most
1800s, while the model that scored it was trained on a triple barrier of
+25bps / -100bps over 600s. Those are different games. A pattern that reliably
rises 25bps and then rolls over is a TRAINING SUCCESS and a LIVE STOP-OUT, so the
model was being rewarded for finding exactly the trades the executor loses on.

Having two independent tables was the root cause, so there is now one table. The
session reads its stop/target/trailing/holding-time from here, and the training
pipeline derives each strategy's triple-barrier labels from the *same* numbers.
Change a strategy's exit rule and its labels move with it.

Label direction note
--------------------
The label barriers are intentionally the executor's own barriers, with the stop
as the *lower* barrier and the profit target as the *upper* one — the opposite
assignment of the previous generic label (tp=25 / sl=100), which tolerated a
100bps drawdown that the executor would never sit through.

Why the numbers below are what they are
---------------------------------------
The first generation of this table set stops of 18-35bps. Measured against the
live tape that is *inside the bid-ask spread*: KRX top-of-book spreads run
materially different by symbol and liquidity. A stop narrower than one spread is
triggered by bid-ask bounce alone, so the trade never gets to express its thesis.

The shadow checkpoint's own simulated outcomes confirmed it exactly. On KRX, with
a modelled round-trip cost of 27.8bps and a 22bps stop, 96.2% of fills stopped
out and the fill-weighted net was -47.3bps — within a bp of the -(22 + 27.8) =
-49.8bps you get if essentially every trade pays stop plus cost. Gross reward:risk
read a healthy 100/22 = 4.5 while the *net* ratio was (100-27.8)/(22+27.8) = 1.45
on the rare winners and irrelevant in practice, because the stop was noise.

So every entry is now derived from one rule, against the market that is actually
funded and trading (KRX):

    stop   >= 3 x typical KRX top-of-book spread (20bps)  ->  60bps floor
    target  = c_ref + R x (stop + c_ref)     with c_ref = 28bps, R = 1.5
    trailing= ~0.5 x stop, and never inside one spread

``R`` is a *net-of-cost* reward:risk target, which is the quantity that actually
compounds; gross ratios flattered the old table precisely because cost was
additive on both sides. ``net_reward_risk_ratio`` exposes it, and
``tests/test_exit_geometry_cost_alignment.py`` asserts the invariant so a future
hand-edit cannot quietly drop a stop back inside the spread.

Two consequences, recorded so they are not rediscovered as bugs:

* Holding times are longer. A 160bps target is not reachable in the 600-1800s the
  old 100bps targets assumed, so each horizon was extended with its target.
* US round-trip cost is 67bps, not 28bps, and one market-agnostic table cannot be
  optimal for both. The table is sized for KRX; for a costlier venue the runtime
  floor in ``strategy_session._cost_aware_profit_bps`` lifts the target further.
  That runtime lift is also the one remaining way labels and execution can
  disagree, since labels are built from the table alone.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping

from app.strategy.catalog import STRATEGY_IDS, is_short_strategy

# Fallback used for an unknown / adopted-position "strategy". Deliberately tight:
# an unidentified thesis gets the least rope, not the most.
FALLBACK_GEOMETRY_KEY = "__fallback__"

# Reference round-trip cost, in bps, of the funded and actively traded venue
# (KRX). Measured from the shadow checkpoint's simulated fills, where it is a
# flat per-fill constant. US fills model at 67bps; see the module docstring.
REFERENCE_ROUND_TRIP_COST_BPS = 28.0

# Reference all-in cost of a SHORT round trip on the same venue: the 28bps cash
# round trip plus the borrow fee accrued over an intraday hold. The borrow
# component is the reference used for *geometry sizing only* — the authoritative
# per-order figure comes from the live borrow snapshot via
# ``app.cost.trading_cost_engine`` and is never replaced by this constant.
#
# 8bps covers a KIS 대주 rate in the high single digits annualised held for a few
# hours (10%/yr ~= 4bps/day), with room for the borrow-uncertainty buffer. A short
# whose expected move cannot clear this is not a trade.
SHORT_BORROW_REFERENCE_BPS = 8.0
SHORT_REFERENCE_ROUND_TRIP_COST_BPS = REFERENCE_ROUND_TRIP_COST_BPS + SHORT_BORROW_REFERENCE_BPS

# Target reward:risk measured *after* cost is paid on both sides.
TARGET_NET_REWARD_RISK = 1.5

# Typical KRX top-of-book spread. A stop must clear several of these or it is
# triggered by bid-ask bounce rather than by the thesis failing.
TYPICAL_SPREAD_BPS = 20.0
STOP_SPREAD_MULTIPLE = 3.0
MINIMUM_STOP_BPS = STOP_SPREAD_MULTIPLE * TYPICAL_SPREAD_BPS

# Strategies whose horizon IS the thesis rather than a leash on it. Their clock is
# session structure (last continuous half-hour; the overnight carry itself), so it
# must never be stretched to make room for a larger target — doing so would carry
# the position past the auction the thesis is defined against.
_TIME_BOXED_STRATEGIES: frozenset[str] = frozenset(
    {
        "market_intraday_momentum",
        "market_intraday_momentum_short",
        "overnight_gap_carry",
    }
)

# A bigger target needs more time to be reachable, but not without bound: past
# this multiple the "trade" is a position, and past the absolute cap it crosses a
# session boundary the intraday theses are not defined across.
_MAX_HOLDING_SCALE = 3.0
_ABSOLUTE_MAX_HOLDING_SECONDS = 21_600


@dataclass(frozen=True)
class StrategyExitGeometry:
    """One strategy's realized trade shape, in bps and seconds.

    ``stop_loss_bps`` and ``take_profit_bps`` are positive magnitudes.
    """

    strategy_id: str
    stop_loss_bps: float
    take_profit_bps: float
    trailing_bps: float
    max_holding_seconds: int
    # Round-trip cost these barriers were actually sized against. Defaults to the
    # KRX reference so a table lookup keeps reading exactly as it always did; a
    # geometry resolved from a live cost estimate carries that estimate instead,
    # which is what makes ``net_reward_risk_ratio()`` answerable per venue rather
    # than only at the reference constant.
    resolved_cost_bps: float = REFERENCE_ROUND_TRIP_COST_BPS
    # True when a measurement (cost and/or spread) shaped these numbers.
    cost_relative: bool = False

    @property
    def reward_risk_ratio(self) -> float:
        """Gross ratio. Flattering and largely meaningless — see ``net_``."""
        return self.take_profit_bps / max(1e-9, self.stop_loss_bps)

    def net_reward_risk_ratio(
        self, cost_bps: float = REFERENCE_ROUND_TRIP_COST_BPS
    ) -> float:
        """Reward:risk once round-trip cost is paid on the win *and* the loss.

        Cost is additive against both barriers, so it compresses the ratio from
        both ends. This is the number that decides whether the geometry can
        compound; the gross ratio read 4.5 while this one was underwater.
        """
        cost = max(0.0, float(cost_bps))
        return (self.take_profit_bps - cost) / max(1e-9, self.stop_loss_bps + cost)

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "stop_loss_bps": self.stop_loss_bps,
            "take_profit_bps": self.take_profit_bps,
            "trailing_bps": self.trailing_bps,
            "max_holding_seconds": self.max_holding_seconds,
            "reward_risk_ratio": round(self.reward_risk_ratio, 3),
            "net_reward_risk_ratio": round(self.net_reward_risk_ratio(), 3),
            "reference_round_trip_cost_bps": REFERENCE_ROUND_TRIP_COST_BPS,
            "resolved_cost_bps": round(self.resolved_cost_bps, 3),
            # The ratio at the cost these barriers were sized against, which is the
            # one that decides whether they compound. Reading only the reference
            # ratio is how a KRX-sized table passed its own test while delivering
            # 0.83 on the US tape it was actually being measured on.
            "resolved_net_reward_risk_ratio": round(
                self.net_reward_risk_ratio(self.resolved_cost_bps), 3
            ),
            "cost_relative": self.cost_relative,
        }

    def as_label_barriers(self) -> dict[str, float]:
        """Triple-barrier parameters for supervised labelling of this strategy."""
        return {
            "take_profit_bps": self.take_profit_bps,
            "stop_loss_bps": self.stop_loss_bps,
            "horizon_seconds": float(self.max_holding_seconds),
        }


# (stop_loss_bps, take_profit_bps, trailing_bps, max_holding_seconds)
# Targets follow c_ref + 1.5 x (stop + c_ref), rounded to 5bps. Relative risk
# ordering between strategies is preserved from the previous table; what changed
# is that the whole range now sits outside the spread.
_GEOMETRY: dict[str, tuple[float, float, float, int]] = {
    "intraday_momentum": (60.0, 160.0, 30.0, 3600),
    "breakout_volume": (60.0, 160.0, 30.0, 4500),
    "vwap_mean_reversion": (60.0, 160.0, 30.0, 3600),
    "bar_confirmed_vwap_recovery": (70.0, 175.0, 35.0, 5400),
    "liquidity_shock_reversal": (65.0, 170.0, 35.0, 2400),
    "event_momentum": (75.0, 185.0, 40.0, 5400),
    "cross_sectional_relative_strength": (60.0, 160.0, 30.0, 5400),
    "gap_context": (70.0, 175.0, 35.0, 4500),
    "rvgi_box_breakout": (60.0, 160.0, 30.0, 3600),
    # --- Strategies added for the current high-volatility, flow-driven tape ---
    # Residual (market/sector-neutral) relative strength holds longer than raw
    # momentum because the thesis is a persistent idiosyncratic bid, not a burst.
    "residual_relative_strength": (65.0, 170.0, 35.0, 5400),
    # Adaptive anchored VWAP reversion targets a normalised displacement, so its
    # stop stays at the wide end of the range.
    "adaptive_anchored_vwap_reversion": (75.0, 185.0, 40.0, 3600),
    # An exhaustion reversal is the shortest-lived thesis here: either liquidity
    # returns within minutes or it was not exhaustion.
    "ofi_microprice_exhaustion_reversal": (60.0, 160.0, 30.0, 1800),
    # An opening-range breakout is a day-long thesis: the published rule holds the
    # position toward the close rather than scalping it, so it gets the longest
    # leash in the table while keeping the same net reward:risk as everything else.
    "opening_range_breakout": (60.0, 160.0, 30.0, 7200),
    # Market intraday momentum is TIME-boxed, not target-boxed: it is entered at the
    # start of the last continuous half-hour and must be flat before the 15:20 KRX
    # closing auction. 1500s covers 14:50->15:15 with margin. The target is kept on
    # the table's standard 60/160 so the net reward:risk invariant still holds if the
    # move arrives early, but the time stop is the intended exit.
    "market_intraday_momentum": (60.0, 160.0, 30.0, 1500),
    # --- SHORT theses -------------------------------------------------------- #
    # The asymmetry here runs the OPPOSITE way to intuition, and it is worth
    # stating plainly because the first draft of this table got it backwards.
    #
    # "A short is riskier, so give it a tighter target" is wrong arithmetic. Cost
    # is additive against BOTH barriers (see ``net_reward_risk_ratio``), and a
    # short's all-in cost is strictly higher than a long's: the same 28bps KRX
    # round trip PLUS an accruing borrow fee. Shrinking the target while the cost
    # rises compresses net reward:risk from both ends — a 130bps target at 60bps
    # stop nets 1.16, below the 1.5 the whole table is built on. The risk would
    # have been paid for with a *worse* payoff.
    #
    # The correct response to a higher cost floor is a LARGER target, so each
    # short target is sized against ``SHORT_REFERENCE_ROUND_TRIP_COST_BPS``
    # (= KRX round trip + intraday borrow) rather than the long reference. Stops
    # stay at the 60bps floor: going tighter would be stopped out by bid-ask
    # bounce, which is the exact mistake the long table already made once.
    #
    # What *does* tighten for a short is TIME. A long's loss is bounded at -100%;
    # a short's is unbounded and accelerates, because the position grows as it
    # moves against you. Borrow fee accrues for as long as it is held, and the
    # lender can recall without notice. So holding time is a cost and a hazard
    # rather than a free option, and every horizon below is shorter than its long
    # counterpart's. None of them may be carried overnight — that is enforced in
    # ``config/short_risk_policy.yaml``, not here.
    #
    # market_intraday_momentum_short keeps its counterpart's 1500s because the
    # horizon is session structure (last continuous half-hour, flat before the
    # 15:20 auction), not a free parameter.
    "market_intraday_momentum_short": (60.0, 180.0, 30.0, 1500),
    "opening_range_breakdown": (60.0, 180.0, 30.0, 3600),
    "residual_relative_weakness": (65.0, 190.0, 32.0, 2700),
    # --- The one thesis that crosses a session boundary ---------------------- #
    # An overnight gap JUMPS; it does not fill through a stop. The stored US tape
    # puts the median absolute overnight move at 69.1bps, so a 60bps stop is
    # inside the typical gap and would be jumped rather than executed — the same
    # mistake the first generation of this table made against the bid-ask spread,
    # one horizon up. 90bps sits outside it, and the target follows the table's
    # own rule so the net reward:risk invariant still holds:
    #     28 + 1.5 x (90 + 28) = 205
    # The 18-hour clock is the carry itself: entered near the close, exited into
    # the next session's opening liquidity.
    "overnight_gap_carry": (90.0, 205.0, 45.0, 64800),
    # Range-floor reversion. The standard 60/160 pair at the 3600s horizon the
    # condition was measured at — deliberately unchanged from the table so the
    # measurement, the training labels and the executor all describe the same trade.
    # Tuning either barrier here would silently make the screened result inapplicable.
    "range_support_reversion": (60.0, 160.0, 30.0, 3600),
    # Multi-hour trend continuation. Same barrier triple as ``event_momentum`` and
    # ``adaptive_anchored_vwap_reversion`` (75/185/40); only the holding clock is longer.
    #
    # This row was first written as 75/265/40, sized so the target would clear a US round
    # trip on its own. That broke the invariant this whole table is built on and
    # ``test_krx_reference_measurement_reproduces_the_table`` caught it: at the KRX
    # reference the rule gives 28 + 1.5 x (75 + 28) = 182.5 -> 185, and 265 corresponds to
    # a ~60bps cost reference instead of 28.
    #
    # Widening for a costlier venue is NOT the table's job — it is
    # ``resolve_exit_geometry``'s, which re-derives the target from the cost actually
    # measured for the trade (a US median of 63.2bps yields 275 here). Baking a US-sized
    # target into the KRX table would have mis-sized every KRX fill of this thesis in the
    # other direction, demanding 265bps where 185 is what compounds at 1.5 net R:R.
    "bar_trend_continuation": (75.0, 185.0, 40.0, 10800),
    # Unknown / adopted position. It gets the tightest admissible stop and the
    # shortest leash — but its target must still clear cost, which the previous
    # 40bps value did not: it was below the 28bps round trip plus any spread.
    FALLBACK_GEOMETRY_KEY: (60.0, 160.0, 30.0, 1200),
}


def _override(strategy_id: str, field_name: str, default: float) -> float:
    """``EXIT_GEOMETRY_<STRATEGY>_<FIELD>`` env override, matching AlgorithmConfig."""
    raw = os.getenv(f"EXIT_GEOMETRY_{strategy_id}_{field_name}".upper())
    if raw in (None, ""):
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def exit_geometry(strategy_id: str | None) -> StrategyExitGeometry:
    """Resolved geometry for a strategy id; unknown ids get the tight fallback."""
    key = str(strategy_id or "").strip().lower()
    stop, target, trailing, holding = _GEOMETRY.get(key, _GEOMETRY[FALLBACK_GEOMETRY_KEY])
    resolved_key = key if key in _GEOMETRY else FALLBACK_GEOMETRY_KEY
    return StrategyExitGeometry(
        strategy_id=key or FALLBACK_GEOMETRY_KEY,
        stop_loss_bps=abs(_override(resolved_key, "stop_loss_bps", stop)),
        take_profit_bps=abs(_override(resolved_key, "take_profit_bps", target)),
        trailing_bps=abs(_override(resolved_key, "trailing_bps", trailing)),
        max_holding_seconds=max(
            30, int(_override(resolved_key, "max_holding_seconds", float(holding)))
        ),
    )


def _measured(value: float | None) -> float | None:
    """A finite, strictly positive measurement, or ``None``.

    Zero is rejected along with None. A zero spread or a zero cost is what an
    absent measurement looks like after a failed parse, and sizing a stop against
    it would produce the tightest possible barrier from the least possible
    information — exactly backwards.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0.0 else None


def _round_up_5(value: float) -> float:
    return math.ceil(max(0.0, value) / 5.0) * 5.0


def resolve_exit_geometry(
    strategy_id: str | None,
    *,
    round_trip_cost_bps: float | None = None,
    spread_bps: float | None = None,
) -> StrategyExitGeometry:
    """Geometry sized against the cost and spread actually measured for this trade.

    Why this exists
    ---------------
    ``_GEOMETRY`` is one market-agnostic table, and the module docstring is explicit
    that it was sized for KRX: stop >= 3 x a *typical KRX* 20bps spread, target =
    c_ref + 1.5 x (stop + c_ref) at c_ref = 28bps. Every consumer then applied it
    unchanged to whatever venue the symbol happened to live on.

    Measured against the stored shadow tape that is the system's entire evidence
    base, that is not a small approximation. US round-trip cost came in at a median
    of 63.2bps (p90 125.2) against the 28bps the table assumes, so the invariant the
    whole table is built on — net reward:risk of 1.5 — delivered 0.83 in practice::

        (170 - 28) / (65 + 28) = 1.53      <- asserted by the unit test
        (170 - 63) / (65 + 63) = 0.83      <- what the tape actually paid

    A geometry with net R:R 0.83 needs a 55% win rate to break even while the table
    was designed to need 40%. Validating a strategy against it cannot return a pass,
    which is why repeated validation on that tape could only ever emit NO_TRADE.

    So the barriers are derived here instead of read:

    * ``stop``   = max(table stop, 3 x the *measured* spread) — the same "outside
      the spread" rule the table states, against this symbol's spread rather than
      against a KRX constant.
    * ``target`` = c + 1.5 x (stop + c) at the *measured* round-trip cost, which is
      the table's own formula with the reference constant replaced by a fact.
    * ``holding`` scales with the target, because a larger target needs more time
      to be reachable — except for the time-boxed theses, whose clock is the thesis.

    Passing neither measurement returns the table verbatim, so every existing call
    site, label and test keeps its current numbers until it opts in.

    Note the honest consequence: on a venue whose cost genuinely cannot be cleared
    by the available move, this returns a target that will not be reached. That is
    the arithmetic becoming visible rather than being absorbed into a stream of
    small losses, and the upstream cost-aware edge floor is what should refuse it.
    """
    base = exit_geometry(strategy_id)
    cost = _measured(round_trip_cost_bps)
    spread = _measured(spread_bps)
    if cost is None and spread is None:
        return base

    key = base.strategy_id
    # Floored at the strategy's own reference, and for a short that reference
    # carries the borrow leg. A measured cost that omitted borrow would size the
    # target as if the position were a long, which is the precise arithmetic error
    # the table's docstring records: cost is additive against BOTH barriers, so
    # under-counting it compresses net reward:risk from both ends. Flooring also
    # keeps the estimate on the conservative side — both terms are estimates, and
    # equality with cost is a loss.
    resolved_cost = max(
        cost if cost is not None else 0.0, reference_round_trip_cost_bps(key)
    )

    stop = base.stop_loss_bps
    if spread is not None:
        stop = max(stop, STOP_SPREAD_MULTIPLE * spread)
    stop = _round_up_5(stop)

    target = _round_up_5(resolved_cost + TARGET_NET_REWARD_RISK * (stop + resolved_cost))

    # Trailing stays at ~half the stop and outside one spread, and must remain
    # strictly inside the stop or it would pre-empt it on every trade.
    spread_floor = (spread if spread is not None else TYPICAL_SPREAD_BPS) * 1.5
    trailing = max(1.0, min(stop - 1.0, max(0.5 * stop, spread_floor)))

    holding = base.max_holding_seconds
    if key not in _TIME_BOXED_STRATEGIES and target > base.take_profit_bps:
        scale = min(_MAX_HOLDING_SCALE, target / max(1e-9, base.take_profit_bps))
        holding = int(min(base.max_holding_seconds * scale, _ABSOLUTE_MAX_HOLDING_SECONDS))
        holding = max(base.max_holding_seconds, holding)

    return StrategyExitGeometry(
        strategy_id=base.strategy_id,
        stop_loss_bps=stop,
        take_profit_bps=target,
        trailing_bps=round(trailing, 1),
        max_holding_seconds=max(30, holding),
        resolved_cost_bps=resolved_cost,
        cost_relative=True,
    )


def reference_round_trip_cost_bps(strategy_id: str | None) -> float:
    """All-in round-trip cost this strategy's geometry was sized against.

    Short theses carry the borrow leg, so asserting the net reward:risk invariant
    at the long reference would understate their cost and let a target through
    that does not actually compound.
    """
    if is_short_strategy(str(strategy_id or "").strip().lower()):
        return SHORT_REFERENCE_ROUND_TRIP_COST_BPS
    return REFERENCE_ROUND_TRIP_COST_BPS


def exit_bps(strategy_id: str | None) -> tuple[float, float, float]:
    """``(stop_bps, take_profit_bps, trailing_bps)`` — the session's tuple shape."""
    geometry = exit_geometry(strategy_id)
    return geometry.stop_loss_bps, geometry.take_profit_bps, geometry.trailing_bps


def max_holding_seconds(strategy_id: str | None) -> int:
    return exit_geometry(strategy_id).max_holding_seconds


def label_geometries() -> Mapping[str, StrategyExitGeometry]:
    """Geometry for every catalogued strategy, for per-strategy labelling."""
    return {strategy_id: exit_geometry(strategy_id) for strategy_id in STRATEGY_IDS}


def all_geometries() -> Mapping[str, StrategyExitGeometry]:
    return {
        key: exit_geometry(key) for key in _GEOMETRY if key != FALLBACK_GEOMETRY_KEY
    }

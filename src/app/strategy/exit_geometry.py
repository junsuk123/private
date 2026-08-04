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
MINIMUM_STOP_BPS = 3.0 * TYPICAL_SPREAD_BPS


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

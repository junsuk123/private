"""What a strategy DECLARES about itself, separately from what it computes.

Before this module every requirement a strategy had was implicit in its ``entry()``
body: ``bar_confirmed_vwap_recovery`` reads ``min_liquidity_score`` from its YAML and
rejects internally, ``opening_range_breakout`` fails closed when the electing layer did
not supply an opening range, and nothing outside the algorithm could see either. Three
consequences followed:

* the eligibility mask could not be computed without running every algorithm,
* the coverage analyzer had no way to say *which* strategies could even apply in a
  bucket, and
* a missing context field showed up as a rejection reason rather than as an
  ineligibility, so "no strategy applies here" and "every strategy said no" were
  indistinguishable.

A :class:`StrategySpec` is the declaration. It is intentionally cheap to evaluate — no
market data, no model, no IO — so the hot path can shrink the candidate set before any
algorithm runs.

Lifecycle
---------
:class:`StrategyLifecycleState` is the research-to-live ladder. It is NOT the same thing
as ``app.trading.directional.StrategyDeploymentState``, and the two must not be merged:
deployment state is a per-(strategy, direction, market, product) *authorisation* held by
the promotion controller and is what gates a broker order; lifecycle state is a
*validation-evidence* claim about the strategy as a whole. A strategy can be ``LIVE`` in
lifecycle and ``SHADOW`` in deployment for one market. :meth:`StrategySpec.submits_orders`
deliberately does not exist — nothing here authorises anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable

__all__ = [
    "StrategyFamily",
    "StrategyLifecycleState",
    "StrategySpec",
]


class StrategyFamily(StrEnum):
    """Thesis family. Coverage is measured across families, not across ids.

    Twenty ids in one family is not broad coverage — that is the
    ``strategy_space_coverage`` failure this whole refactor starts from. Names follow
    the recommended hierarchy; membership is decided in ``app.strategy.registry`` from
    each algorithm's actual thesis, cross-checked against the macro family map that
    already exists in ``app.technical.strategy_algorithms.MACRO_FAMILY_BY_STRATEGY``.
    """

    TREND_FOLLOWING = "TrendFollowing"
    BREAKOUT = "Breakout"
    MEAN_REVERSION = "MeanReversion"
    MICROSTRUCTURE_REVERSAL = "MicrostructureReversal"
    EVENT_DRIVEN = "EventDriven"
    CROSS_SESSION = "CrossSession"
    #: Short-side theses are their own families: a short is not a long with the sign
    #: flipped (it pays borrow, its loss is unbounded, it can be recalled), so pooling
    #: them with their long counterparts would make a family's coverage statistics
    #: describe two different risk objects.
    TREND_FOLLOWING_SHORT = "TrendFollowingShort"
    BREAKDOWN_SHORT = "BreakdownShort"


class StrategyLifecycleState(StrEnum):
    """Validation-evidence ladder, ascending. ``RETIRED`` is terminal."""

    RESEARCH = "RESEARCH"
    VALIDATED = "VALIDATED"
    SHADOW = "SHADOW"
    LIVE_PROBE = "LIVE_PROBE"
    LIVE = "LIVE"
    DEGRADED = "DEGRADED"
    RETIRED = "RETIRED"

    @property
    def rank(self) -> int:
        return _LIFECYCLE_RANK[self]

    @property
    def is_live_candidate(self) -> bool:
        """May a selector consider this strategy for a real order at all?

        ``DEGRADED`` is excluded: it means the drift monitor pulled the strategy, and a
        selector that still ranked it would undo the demotion.
        """
        return self in {StrategyLifecycleState.LIVE_PROBE, StrategyLifecycleState.LIVE}


_LIFECYCLE_RANK: dict[StrategyLifecycleState, int] = {
    StrategyLifecycleState.RETIRED: -1,
    StrategyLifecycleState.RESEARCH: 0,
    StrategyLifecycleState.VALIDATED: 1,
    StrategyLifecycleState.SHADOW: 2,
    StrategyLifecycleState.DEGRADED: 2,
    StrategyLifecycleState.LIVE_PROBE: 3,
    StrategyLifecycleState.LIVE: 4,
}


def _lower_tuple(values: Iterable[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(dict.fromkeys(str(value).strip().lower() for value in values if str(value).strip()))


def _upper_tuple(values: Iterable[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(dict.fromkeys(str(value).strip().upper() for value in values if str(value).strip()))


@dataclass(frozen=True)
class StrategySpec:
    """One strategy's declared identity and requirements.

    Every threshold here is a HARD requirement — a value the strategy cannot function
    without. Preferences ("works well in high volatility") are soft ontology relations,
    not spec fields, because a preference must never zero an eligibility mask.
    """

    strategy_id: str
    family: StrategyFamily
    direction: str  # LONG | SHORT
    horizon_seconds: int
    #: ``TechnicalFeatureSet`` field/property names the ``entry()`` body dereferences.
    required_features: tuple[str, ...] = ()
    #: ``MarketContext`` flat field names the thesis cannot be evaluated without.
    required_context: tuple[str, ...] = ()
    #: ``ElectionContext`` field names only the electing layer can resolve. Absent ->
    #: the algorithm fails closed, so this is a hard eligibility requirement too.
    required_election_inputs: tuple[str, ...] = ()
    minimum_history_bars: int = 0
    #: ``None`` means the strategy declares no floor, NOT that it accepts zero.
    min_liquidity_score: float | None = None
    max_spread_bps: float | None = None
    #: Session phases (``app.data.market_session.MarketPhase`` values) new entries are
    #: admissible in. Empty tuple = the strategy makes no session claim.
    allowed_sessions: tuple[str, ...] = ()
    allowed_markets: tuple[str, ...] = ()
    lifecycle_state: StrategyLifecycleState = StrategyLifecycleState.RESEARCH
    algorithm_version: str = "0.0.0"
    validation_version: str = "unvalidated"
    #: Free-form notes carried into diagnostics, e.g. the measured t-statistic behind a
    #: lifecycle decision. Never parsed.
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy_id", str(self.strategy_id or "").strip().lower())
        object.__setattr__(self, "direction", str(self.direction or "LONG").strip().upper())
        object.__setattr__(self, "horizon_seconds", max(1, int(self.horizon_seconds)))
        object.__setattr__(self, "required_features", _lower_tuple(self.required_features))
        object.__setattr__(self, "required_context", _lower_tuple(self.required_context))
        object.__setattr__(
            self, "required_election_inputs", _lower_tuple(self.required_election_inputs)
        )
        object.__setattr__(self, "allowed_sessions", _lower_tuple(self.allowed_sessions))
        object.__setattr__(self, "allowed_markets", _upper_tuple(self.allowed_markets))
        object.__setattr__(self, "minimum_history_bars", max(0, int(self.minimum_history_bars)))

    @property
    def is_short(self) -> bool:
        return self.direction == "SHORT"

    def permits_market(self, market: str) -> bool:
        if not self.allowed_markets:
            return True
        return str(market or "").strip().upper() in set(self.allowed_markets)

    def permits_session(self, session_phase: str) -> bool:
        if not self.allowed_sessions:
            return True
        return str(session_phase or "").strip().lower() in set(self.allowed_sessions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "family": str(self.family),
            "direction": self.direction,
            "horizon_seconds": self.horizon_seconds,
            "required_features": list(self.required_features),
            "required_context": list(self.required_context),
            "required_election_inputs": list(self.required_election_inputs),
            "minimum_history_bars": self.minimum_history_bars,
            "min_liquidity_score": self.min_liquidity_score,
            "max_spread_bps": self.max_spread_bps,
            "allowed_sessions": list(self.allowed_sessions),
            "allowed_markets": list(self.allowed_markets),
            "lifecycle_state": str(self.lifecycle_state),
            "algorithm_version": self.algorithm_version,
            "validation_version": self.validation_version,
            "notes": list(self.notes),
        }

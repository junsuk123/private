"""Strategy ontology keyed on REAL strategy ids, with hard and soft relations split.

What was wrong before
---------------------
The ontology and the execution layer spoke different vocabularies. The micro reasoner
emitted a generic METHODOLOGY (``momentum`` / ``breakout`` / ``mean_reversion`` /
``vwap_reversion``) and ``app.strategy.catalog.METHODOLOGY_STRATEGY_ALIASES`` translated
it into an executable id. Its own comment records the problem:
``mean_reversion -> vwap_mean_reversion`` is "the loosest fit" — the generic thesis
reverts to a Bollinger midline while the catalogued strategy reverts to VWAP. So an
ontology verdict about one hypothesis authorised a different one, and a catalogue of 19
theses was addressable only through 4 names.

Here every relation names a concrete ``strategy_id``. The methodology enum survives
solely as macro *permission tokens* (``MACRO_FAMILY_BY_STRATEGY``), which is the one job
it can do correctly: a coarse allow/block list over families.

Hard versus soft
----------------
Only a HARD relation may zero an eligibility mask:

===================== ==========================================================
``requires``          a ``MarketContext`` field the thesis is undefined without
``requiresFeature``   a ``TechnicalFeatureSet`` field ``entry()`` dereferences
``requiresLiquidity`` the declared liquidity floor / spread ceiling
``requiresSession``   session phases in which a new entry is defined
``requiresHistory``   completed-bar history the indicators need
``requiresDataQuality`` tick window / orderbook sample / completeness floor
``allowedMarket``     markets the strategy is admissible in
``forbiddenUnder``    market states in which the thesis is invalid outright
===================== ==========================================================

A SOFT relation (``worksWellUnder``, ``prefers``, ``supportedBy``,
``historicallyCompatibleWith``) is *evidence*: it moves the compatibility score, which
becomes the ``O_s`` term of the utility, and it can never block. That separation is the
point — a preference expressed as a block is how a selector loses candidates it should
merely have ranked lower.

Provenance of the soft relations
--------------------------------
They are derived from each thesis and from gating that already exists in the code
(``app.technical.signals`` disables mean reversion in a downtrend and requires volume for
a breakout; ``MACRO_FAMILY_BY_STRATEGY`` records why ``residual_relative_strength`` stays
valid in ``TREND_DOWN`` while ``intraday_momentum`` does not). None is fitted to realized
performance: scoring relations from past results would make the ontology a backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

__all__ = [
    "HARD_RELATION_TYPES",
    "NO_NEW_ENTRY_MARKET_STATES",
    "SOFT_RELATION_TYPES",
    "SoftRelation",
    "StrategyOntology",
    "StrategyRelationType",
    "default_strategy_ontology",
]


class StrategyRelationType(StrEnum):
    # --- hard ------------------------------------------------------------- #
    REQUIRES = "requires"
    FORBIDDEN_UNDER = "forbiddenUnder"
    REQUIRES_FEATURE = "requiresFeature"
    REQUIRES_LIQUIDITY = "requiresLiquidity"
    REQUIRES_SESSION = "requiresSession"
    REQUIRES_HISTORY = "requiresHistory"
    REQUIRES_DATA_QUALITY = "requiresDataQuality"
    ALLOWED_MARKET = "allowedMarket"
    # --- soft ------------------------------------------------------------- #
    WORKS_WELL_UNDER = "worksWellUnder"
    PREFERS = "prefers"
    SUPPORTED_BY = "supportedBy"
    HISTORICALLY_COMPATIBLE_WITH = "historicallyCompatibleWith"

    @property
    def is_hard(self) -> bool:
        return self in HARD_RELATION_TYPES


HARD_RELATION_TYPES: frozenset[StrategyRelationType] = frozenset(
    {
        StrategyRelationType.REQUIRES,
        StrategyRelationType.FORBIDDEN_UNDER,
        StrategyRelationType.REQUIRES_FEATURE,
        StrategyRelationType.REQUIRES_LIQUIDITY,
        StrategyRelationType.REQUIRES_SESSION,
        StrategyRelationType.REQUIRES_HISTORY,
        StrategyRelationType.REQUIRES_DATA_QUALITY,
        StrategyRelationType.ALLOWED_MARKET,
    }
)

SOFT_RELATION_TYPES: frozenset[StrategyRelationType] = frozenset(
    {
        StrategyRelationType.WORKS_WELL_UNDER,
        StrategyRelationType.PREFERS,
        StrategyRelationType.SUPPORTED_BY,
        StrategyRelationType.HISTORICALLY_COMPATIBLE_WITH,
    }
)


#: Market states in which NO new entry is admissible in either direction. Taken from
#: ``app.ontology.trading_domain_ontology.NO_NEW_ENTRY_REGIMES`` and widened with the
#: two blocking labels the technical classifier can emit, so one set answers the
#: question regardless of which classifier produced the label. A dislocated book is not
#: an opportunity — it is a market whose prices are not information.
NO_NEW_ENTRY_MARKET_STATES: frozenset[str] = frozenset(
    {
        "HIGH_VOL_DISLOCATED",
        "DISLOCATED",
        "NEWS_SHOCK",
        "HALTED",
        "NO_TRADE",
        "NO_TRADE_MARKET",
        "NO_TRADE_SYMBOL",
        "LOW_LIQUIDITY_MARKET",
        "LOW_LIQUIDITY_RISK",
    }
)


@dataclass(frozen=True)
class SoftRelation:
    """One piece of compatibility evidence. Never blocks.

    ``field`` is a ``MarketContext`` flat field name; the relation fires when the field's
    value falls inside ``[low, high]`` (either bound may be ``None`` for open-ended), or
    — for a label field — when the value is in ``labels``.

    ``weight`` is a contribution in ``[-1, 1]`` before the utility's ``lambda_o`` scales
    it into bps. Negative weights exist so "this thesis is *known* to be weaker here"
    can be expressed without becoming a block.
    """

    relation: StrategyRelationType
    field: str
    weight: float
    low: float | None = None
    high: float | None = None
    labels: tuple[str, ...] = ()
    rationale: str = ""

    def matches(self, value: Any) -> bool:
        if value is None:
            return False
        if self.labels:
            return str(getattr(value, "value", value)).strip().upper() in {
                label.strip().upper() for label in self.labels
            }
        if isinstance(value, bool):
            # A boolean field with numeric bounds is a contract error, not a match.
            return False
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        if number != number:  # NaN
            return False
        if self.low is not None and number < self.low:
            return False
        if self.high is not None and number > self.high:
            return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "relation": str(self.relation),
            "field": self.field,
            "weight": self.weight,
            "low": self.low,
            "high": self.high,
            "labels": list(self.labels),
            "rationale": self.rationale,
        }


_W = StrategyRelationType.WORKS_WELL_UNDER
_P = StrategyRelationType.PREFERS
_S = StrategyRelationType.SUPPORTED_BY

#: Trending-tape evidence shared by every continuation thesis. Declared once so the
#: momentum family cannot drift apart relation by relation.
_TREND_UP_EVIDENCE: tuple[SoftRelation, ...] = (
    SoftRelation(
        _W, "market_regime", 0.35,
        labels=("TREND_UP", "BREAKOUT_MARKET", "HIGH_VOL_TRENDING"),
        rationale="continuation theses need a trend to continue",
    ),
    SoftRelation(
        _W, "market_regime", -0.35,
        labels=("RANGE_BOUND", "MEAN_REVERSION_CANDIDATE", "HIGH_VOL_MEAN_REVERTING"),
        rationale="a mean-reverting tape punishes continuation entries",
    ),
    SoftRelation(
        _P, "trend_persistence", 0.20, low=0.55,
        rationale="most recent bars up: the trend is a fact, not a hope",
    ),
)

#: Reversion evidence. The negative TREND_DOWN weight mirrors the existing hard rule in
#: ``app.technical.signals`` that disables mean reversion in a downtrend; here it is a
#: penalty rather than a veto, because the reversion thesis is *sometimes* right in a
#: falling tape and a veto removes the evidence needed to find out when.
_REVERSION_EVIDENCE: tuple[SoftRelation, ...] = (
    SoftRelation(
        _W, "market_regime", 0.35,
        labels=("RANGE_BOUND", "MEAN_REVERSION_CANDIDATE", "HIGH_VOL_MEAN_REVERTING"),
        rationale="displacement reverts when the tape has no directional bid",
    ),
    SoftRelation(
        _W, "market_regime", -0.30, labels=("TREND_DOWN",),
        rationale="signals.py disables mean reversion in a downtrend; kept as a penalty",
    ),
    SoftRelation(
        _P, "change_point_probability", -0.25, low=0.5,
        rationale="a structural break voids the level being reverted to",
    ),
)

#: Breakouts need participation. ``signals.py`` already requires volume confirmation for
#: a breakout; expressed here as evidence so a breakout on thin volume ranks below one
#: on real volume instead of being silently discarded.
_BREAKOUT_EVIDENCE: tuple[SoftRelation, ...] = (
    SoftRelation(
        _W, "market_regime", 0.35,
        labels=("BREAKOUT_MARKET", "BREAKOUT_CANDIDATE", "TREND_UP", "HIGH_VOL_TRENDING"),
        rationale="a range break needs a tape willing to extend it",
    ),
    SoftRelation(
        _S, "relative_volume", 0.30, low=1.5,
        rationale="relative volume is the load-bearing filter in the published result",
    ),
    SoftRelation(
        _W, "market_regime", -0.30, labels=("RANGE_BOUND",),
        rationale="a range-bound tape is where breakouts fail",
    ),
)

#: Microstructure reversals live or die on the book, not on the chart.
_MICRO_REVERSAL_EVIDENCE: tuple[SoftRelation, ...] = (
    SoftRelation(
        _P, "liquidity_score", 0.30, low=0.5,
        rationale="the thesis is liquidity RETURNING; it needs a book to return to",
    ),
    SoftRelation(
        _P, "short_term_price_impact", -0.30, low=2.0,
        rationale="spread per unit of capturable volatility above 2 means the "
                  "round trip eats the retrace",
    ),
    SoftRelation(
        _W, "market_regime", -0.25, labels=("TREND_DOWN",),
        rationale="a one-way tape keeps taking liquidity rather than replenishing it",
    ),
)

#: Per-strategy soft relations, on top of the family evidence above.
_SOFT_BY_STRATEGY: dict[str, tuple[SoftRelation, ...]] = {
    "intraday_momentum": (
        *_TREND_UP_EVIDENCE,
        SoftRelation(
            _S, "orderflow_imbalance", 0.25, low=0.15,
            rationale="the thesis IS aggressive buy-side flow",
        ),
    ),
    "breakout_volume": _BREAKOUT_EVIDENCE,
    "rvgi_box_breakout": (
        *_BREAKOUT_EVIDENCE,
        SoftRelation(
            _P, "range_position", 0.20, low=0.8,
            rationale="price at the top of its own box is where the break happens",
        ),
    ),
    "opening_range_breakout": (
        *_BREAKOUT_EVIDENCE,
        SoftRelation(
            _P, "is_opening_window", 0.0,
            labels=("TRUE",),
            rationale="the opening window is the thesis, so it is a HARD session "
                      "requirement rather than evidence; kept at weight 0 for audit",
        ),
    ),
    "vwap_mean_reversion": _REVERSION_EVIDENCE,
    "adaptive_anchored_vwap_reversion": (
        *_REVERSION_EVIDENCE,
        SoftRelation(
            _P, "realized_volatility", 0.20, low=0.001,
            rationale="normalised displacement needs volatility to normalise against",
        ),
    ),
    "bar_confirmed_vwap_recovery": (
        *_REVERSION_EVIDENCE,
        # Hybrid thesis: the LOCATION is reversion but the CLOCK is a momentum turn, so
        # unlike its siblings it stays admissible in a trending tape. This mirrors
        # MACRO_FAMILY_BY_STRATEGY, which lists it under momentum as well.
        SoftRelation(
            _W, "market_regime", 0.20, labels=("TREND_UP", "HIGH_VOL_TRENDING"),
            rationale="entry clock is a completed-bar momentum turn, not a level",
        ),
    ),
    "bar_trend_continuation": (
        *_TREND_UP_EVIDENCE,
        SoftRelation(
            _P, "momentum_persistence", 0.25, low=0.60,
            rationale="completed-bar continuation requires persistent advances",
        ),
        SoftRelation(
            _P, "relative_volume", 0.20, low=1.20,
            rationale="bar trend without participation is not a continuation setup",
        ),
    ),
    "range_support_reversion": (
        *_REVERSION_EVIDENCE,
        SoftRelation(
            _P, "support_distance_bps", 0.30, low=0.0, high=10.0,
            rationale="the measured condition is price within 10bps of the 20-bar floor",
        ),
    ),
    "liquidity_shock_reversal": _MICRO_REVERSAL_EVIDENCE,
    "ofi_microprice_exhaustion_reversal": (
        *_MICRO_REVERSAL_EVIDENCE,
        SoftRelation(
            _S, "bid_ask_depth_imbalance", 0.25, low=0.05,
            rationale="replenishing bid depth is the exhaustion tell",
        ),
    ),
    "event_momentum": (
        SoftRelation(
            _S, "positive_event_score", 0.40, low=0.1,
            rationale="no event, no event thesis",
        ),
        SoftRelation(
            _P, "event_recency_seconds", 0.25, low=0.0, high=600.0,
            rationale="the drift is in the first minutes; an old event is priced",
        ),
        SoftRelation(
            _S, "negative_event_score", -0.35, low=0.1,
            rationale="a negative event is not a long thesis",
        ),
    ),
    "gap_context": (
        SoftRelation(
            _P, "is_opening_window", 0.30, labels=("TRUE",),
            rationale="a gap is an opening-auction dislocation",
        ),
        SoftRelation(
            _S, "relative_volume", 0.25, low=1.5,
            rationale="an unparticipated gap fills rather than extends",
        ),
    ),
    "cross_sectional_relative_strength": (
        *_TREND_UP_EVIDENCE,
        SoftRelation(
            _S, "market_leadership_score", 0.30, low=0.6,
            rationale="the thesis is leadership inside the peer set",
        ),
        SoftRelation(
            _P, "dispersion", 0.20, low=0.0,
            rationale="with no dispersion there is no relative strength to rank",
        ),
    ),
    "residual_relative_strength": (
        # Beta-neutral, so it does NOT inherit the momentum family's TREND_DOWN penalty.
        # MACRO_FAMILY_BY_STRATEGY records exactly this: relative_strength is the family
        # that stays valid in a falling index.
        SoftRelation(
            _S, "market_leadership_score", 0.30, low=0.6,
            rationale="residual leadership inside the sector",
        ),
        SoftRelation(
            _P, "dispersion", 0.25, low=0.0,
            rationale="a residual is only measurable against dispersed peers",
        ),
        SoftRelation(
            _P, "change_point_probability", -0.25, low=0.5,
            rationale="a break invalidates the beta the residual was taken against",
        ),
    ),
    "market_intraday_momentum": (
        SoftRelation(
            _P, "is_closing_window", 0.35, labels=("TRUE",),
            rationale="the published effect is first-half-hour predicting last-half-hour",
        ),
        SoftRelation(
            _P, "volatility_percentile", 0.30, low=0.6,
            rationale="the effect concentrates on volatile days, which are also the "
                      "only days that travel far enough to clear a KRX round trip",
        ),
    ),
    "overnight_gap_carry": (
        SoftRelation(
            _P, "is_closing_window", 0.35, labels=("TRUE",),
            rationale="the carry is a decision about the closing print",
        ),
        SoftRelation(
            _S, "vwap_distance_bps", 0.25, low=10.0,
            rationale="closing above session VWAP is 'buyers ended the day in control'",
        ),
        SoftRelation(
            _P, "market_regime", -0.30, labels=("TREND_DOWN", "NEWS_SHOCK"),
            rationale="carrying a long through a close is what a falling tape punishes",
        ),
    ),
    # --- SHORT theses. Relations are declared so coverage analysis can show what a
    # falling tape WOULD have had available; the account cannot trade them and the
    # eligibility engine hard-blocks them on lifecycle/deployment, not on these.
    "market_intraday_momentum_short": (
        SoftRelation(
            _W, "market_regime", 0.35, labels=("TREND_DOWN",),
            rationale="the short momentum leg is what TREND_DOWN most wants to allow",
        ),
        SoftRelation(
            _P, "market_breadth", 0.25, low=0.0, high=0.55,
            rationale="shorting into a rising broad market is fighting the tape",
        ),
    ),
    "opening_range_breakdown": (
        SoftRelation(
            _W, "market_regime", 0.35, labels=("TREND_DOWN",),
            rationale="a breakdown extends in a falling tape",
        ),
        SoftRelation(
            _S, "relative_volume", 0.25, low=1.5,
            rationale="mirror of the long side's load-bearing RVOL filter",
        ),
    ),
    "residual_relative_weakness": (
        SoftRelation(
            _S, "market_leadership_score", -0.30, high=0.4,
            rationale="the thesis is weakness inside the sector",
        ),
        SoftRelation(
            _P, "dispersion", 0.25, low=0.0,
            rationale="a residual needs dispersed peers, same as its long mirror",
        ),
    ),
}


#: States in which a specific thesis is invalid outright — a HARD block, distinct from
#: the market-wide no-entry set. Only two entries, and both are structural rather than
#: preferential: the label says the market's prices are not usable for THIS thesis.
_FORBIDDEN_UNDER: dict[str, tuple[str, ...]] = {
    # A carry holds through a close. A news shock crossing that boundary is the one
    # scenario where the position cannot be managed at all: the gap is the exposure.
    "overnight_gap_carry": ("NEWS_SHOCK", "HIGH_VOL_DISLOCATED"),
    # The thesis is liquidity returning to the book. In a declared low-liquidity market
    # there is no book to return to, so the entry condition cannot be satisfied.
    "liquidity_shock_reversal": ("LOW_LIQUIDITY_MARKET", "LOW_LIQUIDITY_RISK"),
}


class StrategyOntology:
    """Relations for the catalogued strategies, addressed by real strategy id."""

    def __init__(
        self,
        *,
        soft_relations: Mapping[str, tuple[SoftRelation, ...]] | None = None,
        forbidden_under: Mapping[str, tuple[str, ...]] | None = None,
        no_new_entry_states: frozenset[str] | None = None,
    ) -> None:
        self._soft = dict(soft_relations if soft_relations is not None else _SOFT_BY_STRATEGY)
        self._forbidden = dict(
            forbidden_under if forbidden_under is not None else _FORBIDDEN_UNDER
        )
        self._no_new_entry = (
            no_new_entry_states
            if no_new_entry_states is not None
            else NO_NEW_ENTRY_MARKET_STATES
        )

    def soft_relations(self, strategy_id: str) -> tuple[SoftRelation, ...]:
        return self._soft.get(str(strategy_id or "").strip().lower(), ())

    def forbidden_states(self, strategy_id: str) -> tuple[str, ...]:
        return self._forbidden.get(str(strategy_id or "").strip().lower(), ())

    @property
    def no_new_entry_states(self) -> frozenset[str]:
        return self._no_new_entry

    def as_dict(self) -> dict[str, Any]:
        return {
            "no_new_entry_states": sorted(self._no_new_entry),
            "forbidden_under": {key: list(value) for key, value in self._forbidden.items()},
            "soft_relations": {
                key: [relation.as_dict() for relation in value]
                for key, value in self._soft.items()
            },
        }


_default: StrategyOntology | None = None


def default_strategy_ontology() -> StrategyOntology:
    global _default
    if _default is None:
        _default = StrategyOntology()
    return _default

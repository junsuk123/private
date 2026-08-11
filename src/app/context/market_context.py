"""One market-state snapshot, shared by every module that judges a cycle.

Why this exists
---------------
Before this module a single election cycle computed the market state four times:

* ``web._strategy_session_selection_evidence`` built a ``LiveFeatureFrame`` per symbol,
* the macro/micro observer built its own ``micro_results`` from a different read,
* ``strategy_session._mechanical_entry_verdict`` rebuilt a ``TechnicalFeatureSet``,
* ``strategy_session._bandit_choice`` assembled a ``BanditContext`` from ``macro``.

Each of those reads the store at a slightly different instant, so "the market state"
was never one thing. A strategy could be judged eligible against one snapshot and
scored against another, and no record tied the two together.

A :class:`MarketContext` is therefore identified by a ``context_id``. Every eligibility
verdict, proposal, utility prediction, selection and outcome carries that id, which is
what makes a decision reconstructible after the fact.

Two rules the contract enforces rather than documents:

* **A regime label never stands in for the state.** ``macro.market_regime`` is a derived
  label kept for explanation and branching; the model input is the multi-dimensional
  context. Compressing a tape into one enum is how a selector loses the ability to tell
  "trending and liquid" from "trending and untradeable".
* **Every field can say where it came from and how old it is.** ``sources`` holds a
  :class:`FeatureSource` per field name. A field with no source entry is unattributed,
  which downstream code must treat as unusable rather than as fresh.

Units: every ``*_bps`` field is basis points, every ``*_seconds`` field is seconds, and
percentile/score fields are in ``[0, 1]`` unless the name says otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping
from uuid import uuid4

__all__ = [
    "ContextIdentity",
    "CrossSectionalContext",
    "DataQualityContext",
    "EventContext",
    "FeatureSource",
    "MacroContext",
    "MarketContext",
    "MicrostructureContext",
    "PriceGeometryContext",
    "SymbolContext",
    "TemporalContext",
    "new_context_id",
]


def new_context_id() -> str:
    return f"ctx-{uuid4().hex}"


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _finite(value: Any) -> float | None:
    """A finite float, or ``None``. Never a silent zero."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class FeatureSource:
    """Where one context field came from and how stale it was when read.

    ``age_seconds`` is measured against the context's ``captured_at``, not against
    "now" at read time: a snapshot re-read an hour later must not appear to have aged.
    """

    source: str
    age_seconds: float | None = None
    #: ``None`` when the producer could not answer. Absent is not the same as 0.0.
    confidence: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "age_seconds": self.age_seconds,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ContextIdentity:
    context_id: str
    captured_at: datetime
    symbol: str
    market: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "captured_at": _aware(self.captured_at).isoformat(),
            "symbol": self.symbol,
            "market": self.market,
        }


@dataclass(frozen=True)
class MacroContext:
    """Market-wide state. ``market_regime`` is a LABEL, not the state itself."""

    market_regime: str | None = None
    risk_regime: str | None = None
    index_return: float | None = None
    market_volatility: float | None = None
    fx_regime: str | None = None
    rate_regime: str | None = None
    risk_on_off_score: float | None = None
    #: Structural-break probability from BOCPD. Not a regime name — a number that says
    #: how much any regime label is currently worth.
    change_point_probability: float | None = None
    regime_stability: float | None = None
    volatility_percentile: float | None = None


@dataclass(frozen=True)
class CrossSectionalContext:
    market_breadth: float | None = None
    dispersion: float | None = None
    sector_strength: float | None = None
    relative_strength_rank: int | None = None
    market_leadership_score: float | None = None
    sector_candidate_count: int | None = None


@dataclass(frozen=True)
class SymbolContext:
    trend_strength: float | None = None
    trend_persistence: float | None = None
    realized_volatility: float | None = None
    return_percentile: float | None = None
    volume_percentile: float | None = None
    relative_volume: float | None = None
    price_acceleration: float | None = None
    #: Reference price the geometry and the cost estimate are both taken against.
    #: Absent means no proposal from this context can be priced.
    reference_price: float | None = None


@dataclass(frozen=True)
class PriceGeometryContext:
    vwap_distance_bps: float | None = None
    anchored_vwap_distance_bps: float | None = None
    range_position: float | None = None
    breakout_distance_bps: float | None = None
    support_distance_bps: float | None = None
    session_high_distance_bps: float | None = None
    session_low_distance_bps: float | None = None
    opening_range_position: float | None = None


@dataclass(frozen=True)
class MicrostructureContext:
    spread_bps: float | None = None
    orderflow_imbalance: float | None = None
    microprice_bias: float | None = None
    trade_intensity: float | None = None
    liquidity_score: float | None = None
    bid_ask_depth_imbalance: float | None = None
    short_term_price_impact: float | None = None


@dataclass(frozen=True)
class TemporalContext:
    session_phase: str = "unknown"
    minutes_from_open: float | None = None
    minutes_to_close: float | None = None
    is_opening_window: bool = False
    is_closing_window: bool = False


@dataclass(frozen=True)
class EventContext:
    positive_event_score: float | None = None
    negative_event_score: float | None = None
    event_uncertainty: float | None = None
    event_recency_seconds: float | None = None


@dataclass(frozen=True)
class DataQualityContext:
    """How much of the snapshot is real. Fail-closed inputs live here.

    ``feature_completeness`` is the fraction of *declared* context fields that carry a
    finite value — computed by the builder, never asserted by a caller.
    """

    tick_freshness_sec: float | None = None
    orderbook_freshness_sec: float | None = None
    feature_completeness: float = 0.0
    history_bar_count: int = 0
    second_level_data_ready: bool = False


@dataclass(frozen=True)
class MarketContext:
    """The immutable state one cycle judged one symbol against."""

    identity: ContextIdentity
    macro: MacroContext = field(default_factory=MacroContext)
    cross_sectional: CrossSectionalContext = field(default_factory=CrossSectionalContext)
    symbol: SymbolContext = field(default_factory=SymbolContext)
    price_geometry: PriceGeometryContext = field(default_factory=PriceGeometryContext)
    microstructure: MicrostructureContext = field(default_factory=MicrostructureContext)
    temporal: TemporalContext = field(default_factory=TemporalContext)
    event: EventContext = field(default_factory=EventContext)
    data_quality: DataQualityContext = field(default_factory=DataQualityContext)
    #: Per-field provenance, keyed by the flat field name (``"spread_bps"``).
    sources: Mapping[str, FeatureSource] = field(default_factory=dict)
    #: Raw point-in-time feature snapshot the strategy algorithms consume. Carried by
    #: reference so proposals, shadow plans and training rows all see the SAME numbers
    #: rather than re-deriving them from a later read.
    feature_snapshot: Mapping[str, Any] = field(default_factory=dict)
    #: Reason codes the builder emitted, e.g. ``CONTEXT_NO_ORDERBOOK_SAMPLE``.
    reason_codes: tuple[str, ...] = ()

    # -- identity shortcuts ------------------------------------------------- #
    @property
    def context_id(self) -> str:
        return self.identity.context_id

    @property
    def symbol_id(self) -> str:
        return self.identity.symbol

    @property
    def market(self) -> str:
        return self.identity.market

    @property
    def captured_at(self) -> datetime:
        return _aware(self.identity.captured_at)

    # -- flattened access --------------------------------------------------- #
    def flat(self) -> dict[str, Any]:
        """Every context field by flat name.

        Group names are dropped because no field name repeats across groups; the
        constructor test asserts that, so a future addition that collides fails loudly
        instead of shadowing an existing field.
        """
        flattened: dict[str, Any] = {}
        for group_name in _GROUP_FIELDS:
            group = getattr(self, group_name)
            for member in fields(group):
                flattened[member.name] = getattr(group, member.name)
        return flattened

    def numeric(self) -> dict[str, float]:
        """Finite numeric fields only — the model-input view of the context.

        Booleans are included as 1.0/0.0 because ``second_level_data_ready`` and the
        session-window flags are genuine model inputs; strings (regime labels, session
        phase) are excluded because a label is not a number and encoding it as one is
        how a regime enum ends up dominating a linear head.
        """
        values: dict[str, float] = {}
        for name, raw in self.flat().items():
            if isinstance(raw, bool):
                values[name] = 1.0 if raw else 0.0
                continue
            number = _finite(raw)
            if number is not None:
                values[name] = number
        return values

    def get(self, name: str, default: Any = None) -> Any:
        return self.flat().get(name, default)

    def source_for(self, name: str) -> FeatureSource | None:
        return dict(self.sources).get(name)

    def has(self, *names: str) -> bool:
        """True only when every named field carries a usable value.

        Booleans count as present; ``None`` and non-finite numbers do not. This is the
        primitive every ``required_context`` check is built on, so "absent" resolves the
        same way in the ontology, the proposal engine and the coverage analyzer.
        """
        flat = self.flat()
        for name in names:
            if name not in flat:
                return False
            value = flat[name]
            if value is None:
                return False
            if isinstance(value, bool):
                continue
            if isinstance(value, str):
                # An empty label is an unanswered question, not a value. Treating ""
                # as present would let ``requiresSession`` pass on a context whose
                # session phase was never resolved.
                if not value.strip():
                    return False
                continue
            if _finite(value) is None:
                return False
        return True

    def missing(self, names: Iterable[str]) -> tuple[str, ...]:
        return tuple(name for name in names if not self.has(name))

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"identity": self.identity.as_dict()}
        for group_name in _GROUP_FIELDS:
            group = getattr(self, group_name)
            payload[group_name] = {
                member.name: getattr(group, member.name) for member in fields(group)
            }
        payload["sources"] = {
            name: source.as_dict() for name, source in dict(self.sources).items()
        }
        payload["reason_codes"] = list(self.reason_codes)
        return payload


#: Group attribute names, in the order the contract lists them.
_GROUP_FIELDS: tuple[str, ...] = (
    "macro",
    "cross_sectional",
    "symbol",
    "price_geometry",
    "microstructure",
    "temporal",
    "event",
    "data_quality",
)


def declared_context_fields() -> tuple[str, ...]:
    """Flat names of every declared context field, in group order.

    Used by the builder to compute ``feature_completeness`` and by the coverage
    analyzer to bucket a context without hardcoding a second field list.
    """
    names: list[str] = []
    groups = (
        MacroContext,
        CrossSectionalContext,
        SymbolContext,
        PriceGeometryContext,
        MicrostructureContext,
        TemporalContext,
        EventContext,
        DataQualityContext,
    )
    for group in groups:
        names.extend(member.name for member in fields(group))
    return tuple(names)

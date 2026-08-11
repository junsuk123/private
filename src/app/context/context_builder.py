"""The one place a :class:`MarketContext` is constructed.

Contract
--------
* **Pure.** The builder performs no IO. Every input is passed in, so the same inputs
  always produce the same context (``context_id`` excepted, and injectable for tests).
  This is what lets a stored context be replayed and a decision be reconstructed.
* **One snapshot per (symbol, cycle).** ``build_cycle`` stamps every symbol in a cycle
  with the same ``captured_at`` and the same macro slice, so two strategies on two
  symbols cannot be judged against two different market states within one election.
* **No invented values.** A field the inputs cannot answer stays ``None``. The absence
  is then visible in ``feature_completeness`` and blockable by the ontology's
  ``requiresFeature`` / ``requiresDataQuality`` relations.

The session window comes from ``app.features.session_structure`` and the phase from
``app.data.market_session``, which are already the authorities the live path uses — the
builder does not introduce a third clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from app.context.cross_sectional_context import build_cross_sectional_context
from app.context.macro_context import build_macro_context
from app.context.market_context import (
    ContextIdentity,
    DataQualityContext,
    EventContext,
    FeatureSource,
    MacroContext,
    MarketContext,
    TemporalContext,
    declared_context_fields,
    new_context_id,
)
from app.context.microstructure_context import build_microstructure_context
from app.context.symbol_context import (
    build_price_geometry_context,
    build_symbol_context,
)

__all__ = [
    "CONTEXT_NO_REFERENCE_PRICE",
    "CONTEXT_NO_TICK_WINDOW",
    "CONTEXT_SESSION_UNRESOLVED",
    "MarketContextBuilder",
    "SymbolContextInputs",
]

CONTEXT_NO_REFERENCE_PRICE = "CONTEXT_NO_REFERENCE_PRICE"
CONTEXT_NO_TICK_WINDOW = "CONTEXT_NO_TICK_WINDOW"
CONTEXT_SESSION_UNRESOLVED = "CONTEXT_SESSION_UNRESOLVED"

#: Windows, in minutes from open / to close, that make ``is_opening_window`` and
#: ``is_closing_window`` true. 30 minutes matches the two theses whose definition is a
#: half-hour (``opening_range_breakout``, ``market_intraday_momentum``); using a
#: different number here would make the flag disagree with the strategies that read it.
OPENING_WINDOW_MINUTES = 30.0
CLOSING_WINDOW_MINUTES = 30.0


def _number(raw: Any) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _market_for(symbol: str) -> str:
    text = str(symbol or "").strip().upper()
    return "KR" if text.isdigit() and len(text) == 6 else "US"


@dataclass(frozen=True)
class SymbolContextInputs:
    """Per-symbol inputs for one cycle.

    ``features`` is an ``app.technical.signals.TechnicalFeatureSet`` — the same object
    the strategy algorithms fire on. ``election_inputs`` carries the quantities only the
    electing layer can resolve (sector rank, opening range, anchored VWAP), matching the
    fields of ``app.technical.strategy_algorithms.ElectionContext``.
    """

    symbol: str
    features: Any
    election_inputs: Mapping[str, Any] | None = None
    micro_result: Any = None
    evidence_row: Mapping[str, Any] | None = None
    tick_freshness_sec: float | None = None
    orderbook_freshness_sec: float | None = None
    history_bar_count: int | None = None
    session_high: float | None = None
    session_low: float | None = None
    return_percentile: float | None = None
    volume_percentile: float | None = None
    peer_returns: Sequence[float] | None = None
    #: Injected so a test can assert deterministic construction. Production leaves it
    #: ``None`` and gets a fresh id.
    context_id: str | None = None


class MarketContextBuilder:
    """Builds one :class:`MarketContext` per symbol from shared cycle inputs."""

    def __init__(
        self,
        *,
        session_resolver: Any | None = None,
        phase_resolver: Any | None = None,
    ) -> None:
        # Injected so unit tests need neither the capability service nor a clock, and
        # so a resolver failure degrades to CONTEXT_SESSION_UNRESOLVED rather than
        # raising inside the trading hot path.
        self._session_resolver = session_resolver
        self._phase_resolver = phase_resolver

    # -- public API --------------------------------------------------------- #
    def build_cycle(
        self,
        inputs: Sequence[SymbolContextInputs],
        *,
        captured_at: datetime,
        macro: Any = None,
        macro_age_seconds: float | None = None,
        market_breadth: Any = None,
    ) -> tuple[MarketContext, ...]:
        """One context per symbol, all sharing ``captured_at`` and the macro slice."""
        captured_at = _aware(captured_at)
        macro_context, macro_sources = build_macro_context(
            macro, age_seconds=macro_age_seconds
        )
        breadth = (
            market_breadth
            if market_breadth is not None
            else _macro_breadth(macro)
        )
        return tuple(
            self._build_one(
                item,
                captured_at=captured_at,
                macro_context=macro_context,
                macro_sources=macro_sources,
                market_breadth=breadth,
            )
            for item in inputs
        )

    def build(
        self,
        item: SymbolContextInputs,
        *,
        captured_at: datetime,
        macro: Any = None,
        macro_age_seconds: float | None = None,
        market_breadth: Any = None,
    ) -> MarketContext:
        built = self.build_cycle(
            (item,),
            captured_at=captured_at,
            macro=macro,
            macro_age_seconds=macro_age_seconds,
            market_breadth=market_breadth,
        )
        return built[0]

    # -- internals ---------------------------------------------------------- #
    def _build_one(
        self,
        item: SymbolContextInputs,
        *,
        captured_at: datetime,
        macro_context: MacroContext,
        macro_sources: Mapping[str, FeatureSource],
        market_breadth: Any,
    ) -> MarketContext:
        symbol = str(item.symbol or "").strip().upper()
        features = item.features
        reasons: list[str] = []
        sources: dict[str, FeatureSource] = dict(macro_sources)

        symbol_ctx, symbol_sources = build_symbol_context(
            features,
            return_percentile=item.return_percentile,
            volume_percentile=item.volume_percentile,
            age_seconds=item.tick_freshness_sec,
        )
        sources.update(symbol_sources)
        if symbol_ctx.reference_price is None:
            reasons.append(CONTEXT_NO_REFERENCE_PRICE)

        geometry_ctx, geometry_sources = build_price_geometry_context(
            features,
            election_inputs=item.election_inputs,
            session_high=item.session_high,
            session_low=item.session_low,
            age_seconds=item.tick_freshness_sec,
        )
        sources.update(geometry_sources)

        micro_ctx, micro_sources, micro_reasons = build_microstructure_context(
            features, age_seconds=item.orderbook_freshness_sec
        )
        sources.update(micro_sources)
        reasons.extend(micro_reasons)

        cross_ctx, cross_sources = build_cross_sectional_context(
            election_inputs=item.election_inputs,
            peer_returns=item.peer_returns,
            market_breadth=market_breadth,
            age_seconds=item.tick_freshness_sec,
        )
        sources.update(cross_sources)

        temporal_ctx, temporal_sources, temporal_reasons = self._temporal(
            symbol, captured_at
        )
        sources.update(temporal_sources)
        reasons.extend(temporal_reasons)

        event_ctx, event_sources = _build_event_context(
            micro_result=item.micro_result,
            evidence_row=item.evidence_row,
            election_inputs=item.election_inputs,
        )
        sources.update(event_sources)

        second_ready = bool(
            getattr(features, "tick_data_ready", False)
            or (_number(getattr(features, "second_data_ready", None)) or 0.0) >= 1.0
        )
        if not second_ready:
            reasons.append(CONTEXT_NO_TICK_WINDOW)

        quality_ctx = DataQualityContext(
            tick_freshness_sec=_number(item.tick_freshness_sec),
            orderbook_freshness_sec=_number(item.orderbook_freshness_sec),
            feature_completeness=0.0,  # replaced below, once every group exists.
            history_bar_count=max(0, int(item.history_bar_count or 0)),
            second_level_data_ready=second_ready,
        )

        context = MarketContext(
            identity=ContextIdentity(
                context_id=item.context_id or new_context_id(),
                captured_at=captured_at,
                symbol=symbol,
                market=_market_for(symbol),
            ),
            macro=macro_context,
            cross_sectional=cross_ctx,
            symbol=symbol_ctx,
            price_geometry=geometry_ctx,
            microstructure=micro_ctx,
            temporal=temporal_ctx,
            event=event_ctx,
            data_quality=quality_ctx,
            sources=sources,
            feature_snapshot=_feature_snapshot(features),
            reason_codes=tuple(dict.fromkeys(reasons)),
        )
        # Completeness is measured over the assembled context rather than counted as we
        # go, so it cannot drift from what the context actually holds.
        return _with_completeness(context)

    def _temporal(
        self, symbol: str, captured_at: datetime
    ) -> tuple[TemporalContext, dict[str, FeatureSource], tuple[str, ...]]:
        session = self._resolve_session(symbol)
        phase = self._resolve_phase(symbol, captured_at)
        if session is None:
            return (
                TemporalContext(session_phase=phase or "unknown"),
                {},
                (CONTEXT_SESSION_UNRESOLVED,),
            )
        try:
            opened = session.session_open(captured_at)
            minutes_from_open = (
                captured_at.astimezone(opened.tzinfo) - opened
            ).total_seconds() / 60.0
            minutes_to_close = session.minutes_to_continuous_close(captured_at)
        except Exception:  # noqa: BLE001 - a clock failure must not stop a cycle.
            return (
                TemporalContext(session_phase=phase or "unknown"),
                {},
                (CONTEXT_SESSION_UNRESOLVED,),
            )
        context = TemporalContext(
            session_phase=phase or "unknown",
            minutes_from_open=minutes_from_open,
            minutes_to_close=minutes_to_close,
            is_opening_window=0.0 <= minutes_from_open <= OPENING_WINDOW_MINUTES,
            is_closing_window=0.0 < minutes_to_close <= CLOSING_WINDOW_MINUTES,
        )
        source = FeatureSource(source="session_structure", age_seconds=0.0)
        return (
            context,
            {
                "session_phase": source,
                "minutes_from_open": source,
                "minutes_to_close": source,
                "is_opening_window": source,
                "is_closing_window": source,
            },
            (),
        )

    def _resolve_session(self, symbol: str) -> Any | None:
        resolver = self._session_resolver
        if resolver is None:
            try:
                from app.features.session_structure import regular_session

                resolver = regular_session
            except Exception:  # noqa: BLE001
                return None
        try:
            return resolver(symbol)
        except Exception:  # noqa: BLE001
            return None

    def _resolve_phase(self, symbol: str, captured_at: datetime) -> str | None:
        resolver = self._phase_resolver
        if resolver is None:
            try:
                from app.data.market_session import market_phase

                resolver = market_phase
            except Exception:  # noqa: BLE001
                return None
        try:
            phase = resolver(_market_for(symbol), captured_at)
        except Exception:  # noqa: BLE001
            return None
        value = getattr(phase, "value", phase)
        text = str(value or "").strip().lower()
        return text or None


def _macro_breadth(macro: Any) -> Any:
    diagnostics = getattr(macro, "diagnostics", None)
    if isinstance(diagnostics, Mapping):
        breadth = diagnostics.get("market_breadth")
        if breadth is not None:
            return breadth
    return getattr(macro, "market_breadth", None)


def _build_event_context(
    *,
    micro_result: Any,
    evidence_row: Mapping[str, Any] | None,
    election_inputs: Mapping[str, Any] | None,
) -> tuple[EventContext, dict[str, FeatureSource]]:
    """Event scores from whichever producer answered, without blending them.

    Order of preference is micro result, then the persisted evidence row, then the
    election inputs. They are alternatives rather than terms to average: two producers
    disagreeing about an event means one is stale, and averaging hides which.
    """
    inputs: Mapping[str, Any] = election_inputs or {}
    row: Mapping[str, Any] = evidence_row if isinstance(evidence_row, Mapping) else {}
    diagnostics = getattr(micro_result, "diagnostics", None)
    diag: Mapping[str, Any] = diagnostics if isinstance(diagnostics, Mapping) else {}

    def pick(*names: str) -> tuple[float | None, str | None]:
        for name in names:
            for candidate, origin in (
                (diag, "micro_result"),
                (row, "evidence_row"),
                (inputs, "election_inputs"),
            ):
                value = _number(candidate.get(name))
                if value is not None:
                    return value, origin
        return None, None

    positive, positive_source = pick("positive_event_score", "event_positive_score")
    negative, negative_source = pick("negative_event_score", "event_negative_score")
    uncertainty, uncertainty_source = pick("event_uncertainty")
    recency, recency_source = pick("event_age_seconds", "event_recency_seconds")

    context = EventContext(
        positive_event_score=positive,
        negative_event_score=negative,
        event_uncertainty=uncertainty,
        event_recency_seconds=recency,
    )
    sources = {
        name: FeatureSource(source=origin, age_seconds=recency)
        for name, origin in (
            ("positive_event_score", positive_source),
            ("negative_event_score", negative_source),
            ("event_uncertainty", uncertainty_source),
            ("event_recency_seconds", recency_source),
        )
        if origin is not None
    }
    return context, sources


def _feature_snapshot(features: Any) -> dict[str, Any]:
    """Flat dict of the point-in-time feature object, or ``{}``.

    Kept as the same numbers the algorithms fired on, so a proposal, a shadow plan and a
    training row can all be tied to one observation instead of three reads.
    """
    if features is None:
        return {}
    if isinstance(features, Mapping):
        return dict(features)
    try:
        return {
            member.name: getattr(features, member.name) for member in fields(features)
        }
    except TypeError:  # not a dataclass
        return {}


def _with_completeness(context: MarketContext) -> MarketContext:
    declared = declared_context_fields()
    # ``feature_completeness`` describes the OTHER fields, so it is excluded from its own
    # denominator — otherwise the metric could never reach 1.0.
    measured = tuple(name for name in declared if name != "feature_completeness")
    present = sum(1 for name in measured if context.has(name))
    completeness = present / len(measured) if measured else 0.0
    quality = DataQualityContext(
        tick_freshness_sec=context.data_quality.tick_freshness_sec,
        orderbook_freshness_sec=context.data_quality.orderbook_freshness_sec,
        feature_completeness=round(completeness, 6),
        history_bar_count=context.data_quality.history_bar_count,
        second_level_data_ready=context.data_quality.second_level_data_ready,
    )
    return MarketContext(
        identity=context.identity,
        macro=context.macro,
        cross_sectional=context.cross_sectional,
        symbol=context.symbol,
        price_geometry=context.price_geometry,
        microstructure=context.microstructure,
        temporal=context.temporal,
        event=context.event,
        data_quality=quality,
        sources=context.sources,
        feature_snapshot=context.feature_snapshot,
        reason_codes=context.reason_codes,
    )

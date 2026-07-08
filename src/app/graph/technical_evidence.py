"""Project technical-layer outputs into the semantic graph as EVIDENCE ONLY.

Two projections, both additive and advisory:

* :func:`add_technical_evidence_to_graph` writes ``KnowledgeGraph`` string
  triples (supports/contradicts/increasesRiskOf) whose object names map to the
  existing research-theory framework, so :class:`SemanticPolicyScorer` weights
  technical evidence exactly like any other evidence — it can raise or lower a
  BUY/SELL confidence but can never authorize a trade.
* :func:`attach_technical_evidence_rdf` writes the richer RDF representation
  (a ``tr:TechnicalSignal`` individual with methodology class + data
  properties + provenance ``tr:EvidenceItem``), and optionally marks the symbol
  a ``tr:LiveTechnicalCandidate`` so the live SHACL shape can validate it.

Neither path ever creates a ``tr:FinalOrder`` — OWL/SHACL remain advisory and
the RiskManager stays the sole execution gate.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from app.graph.knowledge_graph import KnowledgeGraph

if TYPE_CHECKING:  # avoid a hard import cycle at module load
    from app.technical.prediction import TechnicalPrediction
    from app.technical.signals import CompositeTechnicalSignal

from app.technical import reason_codes as rc

# methodology -> (KnowledgeGraph support-object name, RDF signal subclass local name)
_METHODOLOGY_MAP: dict[str, tuple[str, str]] = {
    "momentum_trend_following": ("VolumeConfirmedMomentum", "MomentumSignal"),
    "breakout_trading_range_break": ("TechnicalBreakoutBuy", "BreakoutSignal"),
    "mean_reversion": ("RSIOversold", "MeanReversionSignal"),
    "vwap_volume_liquidity": ("OrderFlowPriceConfirmation", "VwapConfirmationSignal"),
    "volatility_band_regime": ("VolatilityRisk", "VolatilityRiskSignal"),
}

_REGIME_CLASS = {
    "TREND_UP": "TrendRegime",
    "TREND_DOWN": "TrendRegime",
    "RANGE_BOUND": "RangeRegime",
    "MEAN_REVERSION_CANDIDATE": "RangeRegime",
    "BREAKOUT_CANDIDATE": "BreakoutRegime",
    "HIGH_VOLATILITY_RISK": "HighRiskRegime",
    "LOW_LIQUIDITY_RISK": "HighRiskRegime",
    "NO_TRADE": "NoTradeRegime",
}


def composite_to_evidence(
    composite: "CompositeTechnicalSignal",
) -> list[tuple[str, str, str, str]]:
    """Return ``(subject, predicate, object, evidence_id)`` evidence tuples."""
    symbol = composite.symbol
    out: list[tuple[str, str, str, str]] = []
    codes = set(composite.reason_codes)

    if composite.blocks_buy:
        if rc.HIGH_VOLATILITY_TECHNICAL_BLOCK in codes:
            out.append((symbol, "increasesRiskOf", "VolatilityRisk", "technical:volatility"))
        if rc.LOW_LIQUIDITY_TECHNICAL_BLOCK in codes:
            out.append((symbol, "increasesRiskOf", "LiquidityRisk", "technical:liquidity"))
        out.append((symbol, "contradictsSignal", "BuyCandidate", "technical:regime-block"))
        return out

    if composite.direction.value == "BUY":
        out.append((symbol, "supportsSignal", "BuyCandidate", "technical:composite"))
        obj, _ = _METHODOLOGY_MAP.get(composite.selected_methodology, ("BuyCandidate", ""))
        if obj != "VolatilityRisk":
            out.append((symbol, "supportsSignal", obj, f"technical:{composite.selected_methodology}"))
        if rc.VWAP_CONFIRMATION_OK in codes:
            out.append((symbol, "supportsSignal", "OrderFlowPriceConfirmation", "technical:vwap"))
        if rc.SPREAD_CONSUMES_TECHNICAL_ALPHA in codes:
            out.append((symbol, "increasesRiskOf", "ThinLiquidityPriceImpactRisk", "technical:spread"))
    else:  # HOLD / unconfirmed
        if rc.VWAP_BREAKDOWN in codes:
            out.append((symbol, "contradictsSignal", "OrderFlowPriceDivergence", "technical:vwap-breakdown"))
    return out


def exit_deterioration_to_evidence(symbol: str, codes: tuple[str, ...]) -> list[tuple[str, str, str, str]]:
    """Map exit-deterioration reason codes to SELL/REDUCE-supporting evidence."""
    mapping = {
        rc.VWAP_BREAKDOWN: ("contradictsSignal", "OrderFlowPriceDivergence"),
        rc.MOMENTUM_WEAKENED: ("contradictsSignal", "BuyCandidate"),
        rc.HIGH_VOLATILITY_TECHNICAL_BLOCK: ("increasesRiskOf", "VolatilityRisk"),
        rc.FALSE_BREAKOUT_RISK_HIGH: ("increasesRiskOf", "VolatilityRisk"),
        rc.LOW_LIQUIDITY_TECHNICAL_BLOCK: ("increasesRiskOf", "LiquidityRisk"),
        rc.TECHNICAL_EXIT_DETERIORATION: ("increasesRiskOf", "OrderFlowDistributionRisk"),
    }
    out: list[tuple[str, str, str, str]] = []
    for code in codes:
        if code in mapping:
            predicate, obj = mapping[code]
            out.append((symbol, predicate, obj, f"technical-exit:{code}"))
    return out


def add_technical_evidence_to_graph(
    graph: KnowledgeGraph,
    composite: "CompositeTechnicalSignal",
) -> int:
    """Add composite-signal evidence triples to a ``KnowledgeGraph``. Returns count."""
    triples = composite_to_evidence(composite)
    for subject, predicate, obj, evidence_id in triples:
        graph.add(subject, predicate, obj, evidence_id)
    return len(triples)


def attach_technical_evidence_rdf(
    rdf,
    composite: "CompositeTechnicalSignal",
    *,
    prediction: "TechnicalPrediction | None" = None,
    live: bool = False,
    decision_time: datetime | None = None,
) -> None:
    """Project the composite signal (and optional prediction) as rich RDF.

    Adds a ``tr:TechnicalSignal`` individual, its methodology subclass, data
    properties, a provenance ``tr:EvidenceItem``, and the regime state. When
    ``live`` is set, marks the symbol ``tr:LiveTradingCandidate`` /
    ``tr:LiveTechnicalCandidate`` and populates the SHACL-required fields.
    """
    from rdflib import Literal
    from rdflib.namespace import RDF, XSD

    from app.graph.rdf_graph import TR, evidence_iri, resource_iri

    symbol = composite.symbol
    subject = resource_iri(symbol)
    methodology = composite.selected_methodology or "composite"
    _, signal_cls = _METHODOLOGY_MAP.get(methodology, ("", "TechnicalSignal"))
    signal_node = resource_iri(f"techsig:{symbol}:{methodology}")

    rdf.add(signal_node, RDF.type, TR.TechnicalSignal)
    if signal_cls and signal_cls != "TechnicalSignal":
        rdf.add(signal_node, RDF.type, TR[signal_cls])
    rdf.add(signal_node, TR.hasMethodologyName, Literal(methodology))
    rdf.add(subject, TR.generatedByMethodology, signal_node)
    rdf.add(signal_node, TR.expectedEdgeBps, Literal(str(round(composite.expected_edge_bps, 4)), datatype=XSD.decimal))
    rdf.add(signal_node, TR.technicalConfidence, Literal(str(round(composite.confidence, 4)), datatype=XSD.decimal))
    rdf.add(signal_node, TR.expectedHorizonSeconds, Literal(int(composite.expected_horizon_seconds), datatype=XSD.integer))
    if prediction is not None and prediction.downside_risk_bps is not None:
        rdf.add(signal_node, TR.downsideRiskBps, Literal(str(round(prediction.downside_risk_bps, 4)), datatype=XSD.decimal))

    # Regime state node.
    regime_cls = _REGIME_CLASS.get(composite.regime.value, "MarketRegimeState")
    regime_node = resource_iri(f"regime:{symbol}")
    rdf.add(regime_node, RDF.type, TR[regime_cls])
    rdf.add(regime_node, TR.hasRegimeName, Literal(composite.regime.value))
    rdf.add(subject, TR.hasRegime, regime_node)

    # Provenance evidence item.
    ev_node = evidence_iri(f"technical:{symbol}:{methodology}")
    rdf.add(subject, TR.hasExecutionQualityEvidence, ev_node)
    rdf.add(ev_node, RDF.type, TR.EvidenceItem)
    rdf.add(ev_node, TR.hasSourceName, Literal("technical_prediction_layer"))
    rdf.add(ev_node, TR.hasSourceType, Literal("technical_methodology"))
    rdf.add(ev_node, TR.hasAnalysisCycleId, Literal(rdf.cycle_id))

    if live:
        rdf.add(subject, RDF.type, TR.LiveTechnicalCandidate)
        rdf.add(subject, TR.hasSymbol, Literal(str(symbol)))
        rdf.add(subject, TR.expectedEdgeBps, Literal(str(round(composite.expected_edge_bps, 4)), datatype=XSD.decimal))
        rdf.add(subject, TR.technicalConfidence, Literal(str(round(composite.confidence, 4)), datatype=XSD.decimal))
        rdf.add(subject, TR.expectedHorizonSeconds, Literal(int(composite.expected_horizon_seconds), datatype=XSD.integer))
        if decision_time is not None:
            rdf.add(subject, TR.hasAsOfTime, Literal(decision_time.isoformat(), datatype=XSD.dateTime))

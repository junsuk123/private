from __future__ import annotations

from app.graph import KnowledgeGraph
from app.schemas.domain import ClassifiedEvent, IndicatorSnapshot, MarketSnapshot
from app.graph.event_mapper import add_events_to_graph
from app.graph.npu_classifier import get_ontology_npu_classifier
from app.strategy.investor_flow import assess_domestic_investor_flow


def build_market_graph(
    markets: tuple[MarketSnapshot, ...],
    indicators: dict[str, IndicatorSnapshot],
    events: tuple[ClassifiedEvent, ...] = (),
    npu_scores: dict[str, tuple[float, ...]] | None = None,
) -> KnowledgeGraph:
    graph = KnowledgeGraph()
    npu_scores = npu_scores or get_ontology_npu_classifier().classify(markets, indicators)

    for market in markets:
        company = market.company_name
        graph.add(company, "hasTicker", market.ticker, market.source.source_id)
        graph.add(company, "belongsToSector", market.sector, market.source.source_id)
        graph.add(market.ticker, "isListedOn", market.market, market.source.source_id)
        _add_investor_flow_to_graph(graph, market)

        indicator = indicators.get(market.ticker)
        if indicator is None:
            continue

        scores = npu_scores.get(market.ticker)
        if scores is None:
            earnings_score = indicator.operating_income_growth or 0.0
            margin_score = indicator.operating_margin or 0.0
            valuation_score = (indicator.per or 0.0) / 100.0
            macro_score = indicator.macro_risk_score
            volatility_score = market.volatility_20d
            combined_score = earnings_score + margin_score - valuation_score - macro_score - volatility_score
        else:
            earnings_score, margin_score, valuation_score, macro_score, volatility_score, combined_score = scores[:6]

        if earnings_score > 0.15:
            graph.add(market.ticker, "supportsSignal", "EarningsGrowth", "npu-indicator-earnings")
        if margin_score > 0.15:
            graph.add(market.ticker, "supportsSignal", "ProfitabilityQuality", "npu-indicator-margin")
        if valuation_score > 0.20:
            graph.add(market.ticker, "contradictsSignal", "ValuationDiscipline", "npu-indicator-valuation")
        if macro_score > 0.40:
            graph.add(market.ticker, "increasesRiskOf", "MacroRateRisk", "npu-indicator-macro")
        if volatility_score > 0.06:
            graph.add(market.ticker, "increasesRiskOf", "VolatilityRisk", market.source.source_id)
        if combined_score > 0.18:
            graph.add(market.ticker, "supportsSignal", "NpuCompositeMomentum", "npu-indicator-composite")

    return add_events_to_graph(graph, events)


def _add_investor_flow_to_graph(graph: KnowledgeGraph, market: MarketSnapshot) -> None:
    assessment = assess_domestic_investor_flow(market)
    if market.investor_flow is None or assessment.metrics is None:
        return
    source_id = market.investor_flow.source.source_id if market.investor_flow.source else market.source.source_id
    graph.add(market.ticker, "hasDominantInvestorFlow", assessment.dominant_group.value, source_id)
    graph.add(market.ticker, "usesFlowModel", "OrderFlowImbalancePriceImpactModel", source_id)
    graph.add("OrderFlowImbalancePriceImpactModel", "basedOnFormula", "normalized_investor_imbalance", "research:ofi")
    graph.add("OrderFlowImbalancePriceImpactModel", "basedOnFormula", "kyle_lambda_proxy", "research:kyle")
    graph.add("OrderFlowImbalancePriceImpactModel", "basedOnFormula", "retail_absorption", "research:investor-type-flow")
    for name, value in assessment.metrics.__dict__.items():
        graph.add(market.ticker, "hasFlowMetric", f"{name}:{value:.6f}", source_id)
    for factor in assessment.supporting_factors:
        graph.add(market.ticker, "supportsSignal", factor, source_id)
    for factor in assessment.contradicting_factors:
        graph.add(market.ticker, "contradictsSignal", factor, source_id)
    for factor in assessment.risk_factors:
        graph.add(market.ticker, "increasesRiskOf", factor, source_id)

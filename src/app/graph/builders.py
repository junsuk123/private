from __future__ import annotations

import os

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
    markets = _scope_markets(markets)
    events = _scope_events(events, {market.ticker for market in markets})
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
            support_score = (indicator.operating_income_growth or 0.0) + (indicator.operating_margin or 0.0)
            risk_score = indicator.macro_risk_score + market.volatility_20d
            momentum_score = (indicator.revenue_growth or 0.0) + max(0.0, (indicator.volume_ratio or 1.0) - 1.0)
            value_score = max(0.0, 0.25 - ((indicator.per or 25.0) / 100.0))
            liquidity_score = min(1.0, market.average_daily_trading_value / 3_000_000_000)
            confidence_score = support_score + momentum_score + value_score + liquidity_score * 0.1 - risk_score
        else:
            (
                support_score,
                risk_score,
                momentum_score,
                value_score,
                liquidity_score,
                confidence_score,
            ) = scores[:6]

        if support_score > 0.10:
            graph.add(market.ticker, "supportsSignal", "EarningsGrowth", "npu-indicator-earnings")
        if support_score > 0.12:
            graph.add(market.ticker, "supportsSignal", "ProfitabilityQuality", "npu-indicator-margin")
        if value_score < -0.05:
            graph.add(market.ticker, "contradictsSignal", "ValuationDiscipline", "npu-indicator-valuation")
        if risk_score > 0.40:
            graph.add(market.ticker, "increasesRiskOf", "MacroRateRisk", "npu-indicator-macro")
        if market.volatility_20d > 0.06:
            graph.add(market.ticker, "increasesRiskOf", "VolatilityRisk", market.source.source_id)
        if momentum_score > 0.18 or confidence_score > 0.18:
            graph.add(market.ticker, "supportsSignal", "NpuCompositeMomentum", "npu-indicator-composite")
        if liquidity_score > 0.20:
            graph.add(market.ticker, "supportsSignal", "LiquiditySupport", "npu-indicator-liquidity")

    return add_events_to_graph(graph, events)


def _scope_markets(markets: tuple[MarketSnapshot, ...]) -> tuple[MarketSnapshot, ...]:
    scope = os.getenv("ONTOLOGY_GRAPH_SCOPE", "candidate_only").strip().lower()
    if scope == "full_debug":
        return markets
    try:
        max_tickers = max(1, int(os.getenv("ONTOLOGY_GRAPH_MAX_TICKERS", "100")))
    except ValueError:
        max_tickers = 100
    return markets[:max_tickers]


def _scope_events(
    events: tuple[ClassifiedEvent, ...],
    candidate_tickers: set[str],
) -> tuple[ClassifiedEvent, ...]:
    scope = os.getenv("ONTOLOGY_GRAPH_SCOPE", "candidate_only").strip().lower()
    if scope == "full_debug":
        return events
    try:
        max_events = max(0, int(os.getenv("ONTOLOGY_GRAPH_MAX_EVENTS_PER_TICKER", "20")))
    except ValueError:
        max_events = 20
    counts: dict[str, int] = {}
    selected: list[ClassifiedEvent] = []
    for event in sorted(events, key=lambda item: item.event_date, reverse=True):
        related = tuple(ticker for ticker in event.tickers if ticker in candidate_tickers)
        if not related:
            continue
        if any(counts.get(ticker, 0) < max_events for ticker in related):
            selected.append(event)
            for ticker in related:
                counts[ticker] = counts.get(ticker, 0) + 1
    return tuple(selected)


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

from __future__ import annotations

from app.graph import KnowledgeGraph
from app.schemas.domain import ClassifiedEvent, IndicatorSnapshot, MarketSnapshot
from app.graph.event_mapper import add_events_to_graph


def build_market_graph(
    markets: tuple[MarketSnapshot, ...],
    indicators: dict[str, IndicatorSnapshot],
    events: tuple[ClassifiedEvent, ...] = (),
) -> KnowledgeGraph:
    graph = KnowledgeGraph()

    for market in markets:
        company = market.company_name
        graph.add(company, "hasTicker", market.ticker, market.source.source_id)
        graph.add(company, "belongsToSector", market.sector, market.source.source_id)
        graph.add(market.ticker, "isListedOn", market.market, market.source.source_id)

        indicator = indicators.get(market.ticker)
        if indicator is None:
            continue

        if indicator.operating_income_growth is not None and indicator.operating_income_growth > 0.15:
            graph.add(market.ticker, "supportsSignal", "EarningsGrowth", "indicator-earnings")
        if indicator.operating_margin is not None and indicator.operating_margin > 0.15:
            graph.add(market.ticker, "supportsSignal", "ProfitabilityQuality", "indicator-margin")
        if indicator.per is not None and indicator.per > 20:
            graph.add(market.ticker, "contradictsSignal", "ValuationDiscipline", "indicator-valuation")
        if indicator.macro_risk_score > 0.40:
            graph.add(market.ticker, "increasesRiskOf", "MacroRateRisk", "indicator-macro")
        if market.volatility_20d > 0.06:
            graph.add(market.ticker, "increasesRiskOf", "VolatilityRisk", market.source.source_id)

    return add_events_to_graph(graph, events)

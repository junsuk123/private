from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass

from app.data.classifier import classify_text_event
from app.data.sample_collectors import collect_sample_account, collect_sample_market
from app.graph import KnowledgeGraph, OntologyReasoner, OntologyRuntime
from app.graph.builders import build_market_graph
from app.indicators import build_sample_indicators
from app.portfolio import build_portfolio_report
from app.risk import RiskManager
from app.schemas.domain import (
    AccountSnapshot,
    ClassifiedEvent,
    IndicatorSnapshot,
    MarketSnapshot,
    OrderIntent,
    PortfolioStatusReport,
    ReasoningPath,
    RiskManagerResult,
    SourceMetadata,
    StrategySignal,
)
from app.strategy import generate_order_intents, generate_strategy_signals
from app.research import ResearchRunResult
from app.storage import StoredResearch


@dataclass(frozen=True)
class AnalysisContext:
    account: AccountSnapshot
    markets: tuple[MarketSnapshot, ...]
    indicators: dict[str, IndicatorSnapshot]
    events: tuple[ClassifiedEvent, ...]
    graph: KnowledgeGraph
    reasoning_paths: tuple[ReasoningPath, ...]
    report: PortfolioStatusReport
    signals: tuple[StrategySignal, ...]
    intents: tuple[OrderIntent, ...]
    risk_results: tuple[RiskManagerResult, ...]
    ontology_runtime: OntologyRuntime


def build_analysis_context(
    research_result: ResearchRunResult | None = None,
    stored_research: StoredResearch | None = None,
) -> AnalysisContext:
    account = collect_sample_account()
    sample_markets = collect_sample_market()
    stored_markets = stored_research.market_snapshots if stored_research else ()
    live_markets = research_result.market_snapshots if research_result else ()
    markets = _merge_markets(sample_markets, _merge_markets(stored_markets, live_markets))
    indicators = build_sample_indicators(markets)
    stored_events = stored_research.events if stored_research else ()
    live_events = research_result.events if research_result else ()
    events = _merge_events(
        _merge_events(stored_events, live_events),
        collect_sample_research_events(),
    )
    graph = build_market_graph(markets, indicators, events)
    reasoner = OntologyReasoner(graph)
    reasoner.infer()
    reasoning_paths = reasoner.build_reasoning_paths(tuple(market.ticker for market in markets))
    report = build_portfolio_report(account)
    signals = generate_strategy_signals(markets, indicators, graph)
    intents = generate_order_intents(markets, indicators, signals)
    market_by_ticker = {market.ticker: market for market in markets}
    risk_results = tuple(
        RiskManager().validate(intent, account, market_by_ticker[intent.ticker]) for intent in intents
    )

    return AnalysisContext(
        account=account,
        markets=markets,
        indicators=indicators,
        events=events,
        graph=graph,
        reasoning_paths=reasoning_paths,
        report=report,
        signals=signals,
        intents=intents,
        risk_results=risk_results,
        ontology_runtime=reasoner.runtime,
    )


def collect_sample_research_events() -> tuple[ClassifiedEvent, ...]:
    source = SourceMetadata(
        source_name="sample_research",
        retrieved_at=datetime.now(timezone.utc),
        raw_url="local://sample-research",
        source_id="sample:event:semiconductor",
    )
    return (
        classify_text_event(
            title="005930 reports memory profit growth and HBM demand strength",
            body=(
                "Samsung Electronics 005930 semiconductor memory profit growth improved. "
                "AI server demand supports HBM and advanced memory outlook."
            ),
            source=source,
            known_tickers={"005930": "Samsung Electronics", "000660": "SK hynix"},
        ),
    )


def _merge_markets(
    primary: tuple[MarketSnapshot, ...],
    secondary: tuple[MarketSnapshot, ...],
) -> tuple[MarketSnapshot, ...]:
    by_ticker = {market.ticker: market for market in primary}
    for market in secondary:
        by_ticker[market.ticker] = market
    return tuple(by_ticker.values())


def _merge_events(
    primary: tuple[ClassifiedEvent, ...],
    secondary: tuple[ClassifiedEvent, ...],
) -> tuple[ClassifiedEvent, ...]:
    by_id = {event.event_id: event for event in primary}
    for event in secondary:
        by_id.setdefault(event.event_id, event)
    return tuple(by_id.values())

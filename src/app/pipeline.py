from __future__ import annotations

import os
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any

from app.data.classifier import classify_text_event
from app.data.sample_collectors import collect_sample_account, collect_sample_market
from app.graph import KnowledgeGraph, OntologyReasoner, OntologyRuntime
from app.graph.builders import build_market_graph
from app.indicators import build_sample_indicators, build_trusted_indicators_from_markets, filter_trusted_indicators
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
    RiskRules,
    SourceMetadata,
    StrategySignal,
    TimeSynchronizedTickerFrame,
)
from app.strategy import generate_order_intents, generate_strategy_signals
from app.research import ResearchRunResult
from app.storage import StoredResearch
from app.time_series import add_time_frames_to_graph, build_time_synchronized_frames
from app.trading_pipeline import (
    CandidateSelectionResult,
    build_lightweight_market_snapshots_from_markets,
    ontology_filter_1,
)


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
    candidate_selection: CandidateSelectionResult | None = None
    parameter_tuning: tuple[dict[str, Any], ...] = ()
    temporal_frames: tuple[TimeSynchronizedTickerFrame, ...] = ()


def build_analysis_context(
    research_result: ResearchRunResult | None = None,
    stored_research: StoredResearch | None = None,
    *,
    allow_sample_indicators: bool = False,
    account_override: AccountSnapshot | None = None,
    risk_rules: RiskRules | None = None,
) -> AnalysisContext:
    account = account_override or collect_sample_account()
    sample_markets = collect_sample_market()
    stored_markets = stored_research.market_snapshots if stored_research else ()
    live_markets = research_result.market_snapshots if research_result else ()
    raw_markets = _merge_markets(sample_markets, _merge_markets(stored_markets, live_markets))
    candidate_selection = _select_analysis_candidates(raw_markets)
    if candidate_selection is not None and candidate_selection.candidate_stocks:
        candidate_set = set(candidate_selection.candidate_stocks)
        candidate_set.update(_priority_tickers(raw_markets))
        markets = tuple(market for market in raw_markets if market.ticker in candidate_set)
    else:
        markets = _limit_markets_for_runtime(raw_markets)
    demo_offline_context = allow_sample_indicators or (research_result is None and stored_research is None)
    indicators = (
        build_sample_indicators(markets)
        if demo_offline_context
        else filter_trusted_indicators(build_trusted_indicators_from_markets(markets))
    )
    stored_events = stored_research.events if stored_research else ()
    live_events = research_result.events if research_result else ()
    events = _merge_events(
        _merge_events(stored_events, live_events),
        collect_sample_research_events(),
    )
    events = _limit_events_for_runtime(events, markets)
    stored_raw_records = getattr(stored_research, "raw_records", ()) if stored_research else ()
    live_raw_records = getattr(research_result, "raw_records", ()) if research_result else ()
    stored_macro_metrics = getattr(stored_research, "macro_metrics", ()) if stored_research else ()
    live_macro_metrics = getattr(research_result, "macro_metrics", ()) if research_result else ()
    realtime_quotes = getattr(stored_research, "realtime_quotes", ()) if stored_research else ()
    realtime_executions = getattr(stored_research, "realtime_executions", ()) if stored_research else ()
    temporal_frames = build_time_synchronized_frames(
        markets=markets,
        events=events,
        raw_records=_merge_raw_records(stored_raw_records, live_raw_records),
        macro_metrics=_merge_macro_metrics(stored_macro_metrics, live_macro_metrics),
        realtime_quotes=realtime_quotes,
        realtime_executions=realtime_executions,
    )
    graph = build_market_graph(markets, indicators, events)
    add_time_frames_to_graph(graph, temporal_frames)
    parameter_tuning = _ontology_parameter_tuning(markets, indicators, events)
    _add_pipeline_metadata_to_graph(graph, candidate_selection, parameter_tuning, events)
    reasoner = OntologyReasoner(graph)
    reasoner.infer()
    reasoning_paths = reasoner.build_reasoning_paths(tuple(market.ticker for market in markets))
    report = build_portfolio_report(account)
    signals = generate_strategy_signals(markets, indicators, graph)
    intents = generate_order_intents(markets, indicators, signals)
    market_by_ticker = {market.ticker: market for market in markets}
    risk_manager = RiskManager(risk_rules) if risk_rules is not None else RiskManager()
    risk_results = tuple(risk_manager.validate(intent, account, market_by_ticker[intent.ticker]) for intent in intents)

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
        candidate_selection=candidate_selection,
        parameter_tuning=parameter_tuning,
        temporal_frames=temporal_frames,
    )


def _select_analysis_candidates(markets: tuple[MarketSnapshot, ...]) -> CandidateSelectionResult | None:
    if not markets:
        return None
    try:
        target_count = max(20, int(os.getenv("ANALYSIS_CANDIDATE_COUNT", "100")))
    except ValueError:
        target_count = 100
    snapshots = build_lightweight_market_snapshots_from_markets(markets)
    return ontology_filter_1(
        snapshots,
        target_count=min(target_count, max(1, len(markets))),
        cache_key=f"analysis:{len(markets)}:{target_count}",
    )


def _priority_tickers(markets: tuple[MarketSnapshot, ...]) -> set[str]:
    priority = {"005930", "005930.KS", "000660", "000660.KS", "AAPL", "MSFT", "NVDA", "SPY", "QQQ"}
    available = {market.ticker for market in markets}
    return priority & available


def _ontology_parameter_tuning(
    markets: tuple[MarketSnapshot, ...],
    indicators: dict[str, IndicatorSnapshot],
    events: tuple[ClassifiedEvent, ...],
) -> tuple[dict[str, Any], ...]:
    if not markets:
        return ()
    avg_volatility = sum(float(market.volatility_20d or 0.0) for market in markets) / len(markets)
    negative_events = sum(1 for event in events if str(event.sentiment) == "NEGATIVE")
    positive_events = sum(1 for event in events if str(event.sentiment) == "POSITIVE")
    risk_bias = min(1.0, avg_volatility * 8 + negative_events / max(1, len(events)) * 0.4)
    max_volume_ratio = max(((indicator.volume_ratio or 0.0) for indicator in indicators.values()), default=0.0)
    momentum_bias = min(1.0, positive_events / max(1, len(events)) * 0.5 + max_volume_ratio * 0.2)
    return (
        {
            "mode": "OntologyTuningMode:RiskAdaptive",
            "target_parameter": "RiskManager.max_single_stock_weight",
            "base_value": 0.06,
            "tuned_value": round(max(0.015, 0.06 * (1 - risk_bias * 0.55)), 4),
            "reason": "Reduce position size when volatility or negative event pressure rises.",
            "risk_bias": round(risk_bias, 4),
        },
        {
            "mode": "OntologyTuningMode:MomentumBreakout",
            "target_parameter": "SemanticMapping.volume_spike_ratio",
            "base_value": 2.0,
            "tuned_value": round(max(1.25, 2.0 - momentum_bias * 0.55), 4),
            "reason": "Lower breakout volume threshold when positive event and flow pressure is strong.",
            "momentum_bias": round(momentum_bias, 4),
        },
        {
            "mode": "OntologyTuningMode:EventRisk",
            "target_parameter": "OntologyReasoner.NegativeEventRiskWeight",
            "base_value": 0.18,
            "tuned_value": round(0.18 + min(0.16, negative_events * 0.01), 4),
            "reason": "Increase event risk weight when fresh negative events are present.",
            "negative_events": negative_events,
        },
    )


def _add_pipeline_metadata_to_graph(
    graph: KnowledgeGraph,
    candidate_selection: CandidateSelectionResult | None,
    parameter_tuning: tuple[dict[str, Any], ...],
    events: tuple[ClassifiedEvent, ...] = (),
) -> None:
    pipeline = "OntologyMultiStagePipeline"
    graph.add(pipeline, "hasStage", "OntologyFilter1:LightweightScreening", "pipeline:stage1")
    graph.add("OntologyFilter1:LightweightScreening", "selects", "CandidateStock", "pipeline:stage1")
    graph.add("CandidateStock", "feedsStage", "SelectiveChartFetching", "pipeline:stage2")
    graph.add("SelectiveChartFetching", "feedsStage", "SemanticFeatureExtraction", "pipeline:stage3")
    graph.add("SemanticFeatureExtraction", "feedsStage", "OntologyFilter2:EntryDecision", "pipeline:stage4")
    graph.add("OntologyFilter2:EntryDecision", "feedsStage", "AIPredictionSmallSet", "pipeline:stage5")
    graph.add("AIPredictionSmallSet", "requiresApprovalFrom", "OntologyFilter3:FinalRiskApproval", "pipeline:stage6")
    graph.add("OntologyFilter3:FinalRiskApproval", "blocksTrade", "NoTradeSignal", "pipeline:risk")
    graph.add("OntologyFilter3:FinalRiskApproval", "usesCostModel", "TradingCost", "pipeline:trading-cost")
    graph.add("TradingCost", "contains", "BrokerageFee", "pipeline:trading-cost")
    graph.add("TradingCost", "contains", "SellTax", "pipeline:trading-cost")
    graph.add("TradingCost", "contains", "Slippage", "pipeline:trading-cost")
    graph.add("TradingCost", "contains", "BidAskSpread", "pipeline:trading-cost")
    graph.add("TradingCost", "contains", "MarketImpact", "pipeline:trading-cost")
    graph.add("TradingCost", "produces", "BreakEvenReturn", "pipeline:trading-cost")
    graph.add("TradingCost", "produces", "RequiredExitPrice", "pipeline:trading-cost")
    graph.add("TradingCost", "produces", "NetExpectedReturn", "pipeline:trading-cost")
    graph.add("BreakEvenReturn", "blocksTradeBelow", "NoTradeSignal", "pipeline:trading-cost")
    graph.add("NetExpectedReturn", "supportsSignal", "NetProfitability", "pipeline:trading-cost")
    graph.add("CostToAlphaRatio", "increasesRiskOf", "CostBurden", "pipeline:trading-cost")
    graph.add("Slippage", "increasesRiskOf", "SlippageRisk", "pipeline:trading-cost")
    graph.add("BidAskSpread", "increasesRiskOf", "SpreadRisk", "pipeline:trading-cost")
    graph.add("NetProfitability", "requiresApprovalFrom", "FinalTradeGate", "pipeline:trading-cost")
    if candidate_selection is not None:
        graph.add("OntologyFilter1:LightweightScreening", "observedUniverseCount", f"UniverseCount:{candidate_selection.full_universe_count}", "pipeline:metrics")
        graph.add("OntologyFilter1:LightweightScreening", "selectedCandidateCount", f"CandidateCount:{len(candidate_selection.candidate_stocks)}", "pipeline:metrics")
        graph.add("SelectiveChartFetching", "fetchesChartsFor", f"CandidateCount:{len(candidate_selection.chart_fetch_scope)}", "pipeline:metrics")
        for ticker in candidate_selection.candidate_stocks[:40]:
            graph.add("OntologyFilter1:LightweightScreening", "selectsCandidate", ticker, "pipeline:filter1")
    for item in parameter_tuning:
        mode = str(item["mode"])
        parameter_node = f"Parameter:{item['target_parameter']}"
        value_node = f"TunedValue:{item['tuned_value']}"
        stage_node = _tuning_stage_for_mode(mode)
        graph.add(mode, "tunesParameter", parameter_node, "pipeline:tuning")
        graph.add(mode, "producesTunedValue", value_node, "pipeline:tuning")
        graph.add(parameter_node, "hasTunedValue", value_node, "pipeline:tuning")
        graph.add(mode, "supportsSignal", "MarketInterpretationParameterTuning", "pipeline:tuning")
        graph.add("OntologyMultiStagePipeline", "hasTuningMode", mode, "pipeline:tuning")
        graph.add(mode, "adjustsStage", stage_node, "pipeline:tuning")
        graph.add(value_node, "appliesToStage", stage_node, "pipeline:tuning")
        for signal_node in _tuning_signal_nodes(mode):
            graph.add(mode, "usesOntologySignal", signal_node, "pipeline:tuning")
            graph.add(parameter_node, "calibratesSignal", signal_node, "pipeline:tuning")
            graph.add(value_node, "calibratesSignal", signal_node, "pipeline:tuning")
        if mode == "OntologyTuningMode:EventRisk":
            for event in events[:20]:
                if str(event.sentiment) == "NEGATIVE":
                    event_node = f"{event.event_type}:{event.event_id}"
                    graph.add(event_node, "raisesTuningPressure", mode, "pipeline:tuning:event")
        if mode == "OntologyTuningMode:MomentumBreakout":
            for event in events[:20]:
                if str(event.sentiment) == "POSITIVE":
                    event_node = f"{event.event_type}:{event.event_id}"
                    graph.add(event_node, "raisesTuningPressure", mode, "pipeline:tuning:event")


def _tuning_stage_for_mode(mode: str) -> str:
    return {
        "OntologyTuningMode:RiskAdaptive": "OntologyFilter3:FinalRiskApproval",
        "OntologyTuningMode:MomentumBreakout": "OntologyFilter2:EntryDecision",
        "OntologyTuningMode:EventRisk": "SemanticFeatureExtraction",
    }.get(mode, "MarketInterpretationParameterTuning")


def _tuning_signal_nodes(mode: str) -> tuple[str, ...]:
    return {
        "OntologyTuningMode:RiskAdaptive": ("VolatilityRisk", "NegativeEventRisk", "RiskAdjustedSizing"),
        "OntologyTuningMode:MomentumBreakout": ("PositiveEventImpact", "BuyCandidate", "EarningsGrowth"),
        "OntologyTuningMode:EventRisk": ("NegativeEventRisk", "PositiveEventImpact"),
    }.get(mode, ("MarketInterpretationParameterTuning",))


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


def _limit_markets_for_runtime(markets: tuple[MarketSnapshot, ...]) -> tuple[MarketSnapshot, ...]:
    try:
        limit = max(100, int(os.getenv("ANALYSIS_MARKET_LIMIT", "300")))
    except ValueError:
        limit = 300
    if len(markets) <= limit:
        return markets
    priority = {"005930", "005930.KS", "000660", "000660.KS", "AAPL", "MSFT", "NVDA", "SPY", "QQQ"}
    prioritized = [market for market in markets if market.ticker in priority]
    remaining = [market for market in markets if market.ticker not in priority]
    remaining.sort(key=lambda market: market.source.retrieved_at, reverse=True)
    selected = prioritized + remaining[: max(0, limit - len(prioritized))]
    return tuple(dict.fromkeys(selected))


def _merge_events(
    primary: tuple[ClassifiedEvent, ...],
    secondary: tuple[ClassifiedEvent, ...],
) -> tuple[ClassifiedEvent, ...]:
    by_id = {event.event_id: event for event in primary}
    for event in secondary:
        by_id.setdefault(event.event_id, event)
    return tuple(by_id.values())


def _merge_raw_records(primary: tuple[Any, ...], secondary: tuple[Any, ...]) -> tuple[Any, ...]:
    by_key: dict[str, Any] = {}
    for record in primary + secondary:
        source = getattr(record, "source", None)
        key = (
            f"{getattr(source, 'source_id', None) or getattr(source, 'raw_url', None) or id(record)}:"
            f"{getattr(source, 'retrieved_at', '')}"
        )
        by_key[key] = record
    return tuple(by_key.values())


def _merge_macro_metrics(primary: tuple[Any, ...], secondary: tuple[Any, ...]) -> tuple[Any, ...]:
    by_key: dict[str, Any] = {}
    for metric in primary + secondary:
        by_key[f"{getattr(metric, 'name', '')}:{getattr(metric, 'observed_at', '')}"] = metric
    return tuple(by_key.values())


def _limit_events_for_runtime(
    events: tuple[ClassifiedEvent, ...],
    markets: tuple[MarketSnapshot, ...],
) -> tuple[ClassifiedEvent, ...]:
    try:
        limit = max(25, int(os.getenv("ANALYSIS_EVENT_LIMIT", "180")))
    except ValueError:
        limit = 180
    if len(events) <= limit:
        return events

    market_tickers = {market.ticker for market in markets}

    def score(event: ClassifiedEvent) -> tuple[float, datetime]:
        related = bool(set(event.tickers) & market_tickers)
        directional = str(event.sentiment) in {"POSITIVE", "NEGATIVE"}
        label_bonus = min(3, len(event.event_labels)) * 0.4
        fact_bonus = min(3, len(event.key_facts)) * 0.2
        confidence = float(event.classification_confidence or 0.0)
        priority = (
            (5.0 if related else 0.0)
            + (3.0 if directional else 0.0)
            + confidence
            + label_bonus
            + fact_bonus
        )
        return priority, event.event_date

    return tuple(sorted(events, key=score, reverse=True)[:limit])

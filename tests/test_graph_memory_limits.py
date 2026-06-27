from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.graph.builders import build_market_graph
from app.schemas.domain import ClassifiedEvent, EventType, IndicatorSnapshot, MarketSnapshot, SentimentDirection, SourceMetadata


def test_graph_candidate_only_limits_tickers_and_event_ttl(monkeypatch) -> None:
    monkeypatch.setenv("ONTOLOGY_GRAPH_SCOPE", "candidate_only")
    monkeypatch.setenv("ONTOLOGY_GRAPH_MAX_TICKERS", "2")
    monkeypatch.setenv("ONTOLOGY_GRAPH_EVENT_TTL_HOURS", "24")
    now = datetime.now(timezone.utc)
    source = SourceMetadata("test", now, source_id="test")
    markets = tuple(
        MarketSnapshot(f"T{i}", "US", f"T{i}", "Tech", 100.0, 2_000_000_000, 0.02, source)
        for i in range(4)
    )
    indicators = {
        market.ticker: IndicatorSnapshot(market.ticker, 0.1, 0.2, 0.2, None, None, 10, None, 60, 1.2, 0.1)
        for market in markets
    }
    events = (
        ClassifiedEvent("fresh", EventType.NEWS, "fresh", "fresh", (), ("T0",), (), SentimentDirection.POSITIVE, now, source),
        ClassifiedEvent("old", EventType.NEWS, "old", "old", (), ("T0",), (), SentimentDirection.POSITIVE, now - timedelta(days=3), source),
    )

    graph = build_market_graph(markets, indicators, events)

    subjects = {triple.subject for triple in graph.triples()}
    assert "T0" in subjects
    assert "T1" in subjects
    assert "T2" not in subjects
    assert all("old" not in triple.object for triple in graph.triples())

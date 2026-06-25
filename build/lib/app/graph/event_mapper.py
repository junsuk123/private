from __future__ import annotations

from app.graph import KnowledgeGraph
from app.schemas.domain import ClassifiedEvent, SentimentDirection


def add_events_to_graph(graph: KnowledgeGraph, events: tuple[ClassifiedEvent, ...]) -> KnowledgeGraph:
    for event in events:
        event_node = f"{event.event_type}:{event.event_id}"
        for ticker in event.tickers:
            predicate = "hasRecentNews" if event.event_type == "NEWS" else "hasRecentDisclosure"
            graph.add(ticker, predicate, event_node, event.source.source_id)

            if event.sentiment == SentimentDirection.POSITIVE:
                graph.add(event_node, "supportsSignal", "PositiveEventImpact", event.source.source_id)
                graph.add(ticker, "supportsSignal", "PositiveEventImpact", event.source.source_id)
            elif event.sentiment == SentimentDirection.NEGATIVE:
                graph.add(event_node, "increasesRiskOf", "NegativeEventRisk", event.source.source_id)
                graph.add(ticker, "increasesRiskOf", "NegativeEventRisk", event.source.source_id)

        for sector in event.sectors:
            graph.add(event_node, "hasExposureTo", sector, event.source.source_id)

    return graph

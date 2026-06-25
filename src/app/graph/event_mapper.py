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
        for label in event.event_labels:
            if _is_risk_label(label):
                graph.add(event_node, "increasesRiskOf", label, event.source.source_id)
            elif _is_negative_label(label):
                graph.add(event_node, "contradictsSignal", label, event.source.source_id)
            else:
                graph.add(event_node, "supportsSignal", label, event.source.source_id)
        for index, fact in enumerate(event.key_facts[:5]):
            graph.add(event_node, "generatesSemanticFeature", f"Fact:{fact}", f"{event.source.source_id}:fact:{index}")

    return graph


def _is_risk_label(label: str) -> bool:
    lowered = label.lower()
    return any(token in lowered for token in ("risk", "penalty", "litigation", "failure", "cut", "lowered"))


def _is_negative_label(label: str) -> bool:
    lowered = label.lower()
    return any(token in lowered for token in ("negative", "downgrade", "miss", "cancellation"))

from __future__ import annotations

from app.features.schemas import RawIndicatorRecord, SemanticFeatureRecord
from app.graph import KnowledgeGraph


def build_semantic_feature_graph(
    raw_indicators: tuple[RawIndicatorRecord, ...],
    semantic_features: tuple[SemanticFeatureRecord, ...],
    base_graph: KnowledgeGraph | None = None,
) -> KnowledgeGraph:
    graph = base_graph or KnowledgeGraph()
    for indicator in raw_indicators:
        indicator_node = f"indicator:{indicator.ticker}:{indicator.indicator_name}:{indicator.as_of.isoformat()}"
        graph.add(indicator.ticker, "hasTechnicalIndicator", indicator_node, indicator.calculation_version)

    for feature in semantic_features:
        feature_node = feature.ontology_node_id or f"semantic:{feature.ticker}:{feature.feature_name}"
        graph.add(feature.ticker, "generatesSemanticFeature", feature_node, feature.feature_name)
        graph.add(feature_node, feature.semantic_relation, feature.target_signal or feature.feature_name, feature.feature_name)
        if feature.semantic_relation == "increasesRiskOf":
            graph.add(feature.ticker, "increasesRiskOf", feature.feature_name, feature.feature_name)
        elif feature.semantic_relation == "contradictsSignal":
            graph.add(feature.ticker, "contradictsSignal", feature.target_signal or feature.feature_name, feature.feature_name)
        elif feature.semantic_relation == "decreasesRiskOf":
            graph.add(feature.ticker, "decreasesRiskOf", feature.feature_name, feature.feature_name)
        else:
            graph.add(feature.ticker, "supportsSignal", feature.target_signal or feature.feature_name, feature.feature_name)
    return graph

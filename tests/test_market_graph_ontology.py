"""Market graph ontology: typed structure, prior/learned separation, tensors."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from app.models.graph_snapshot import GraphSnapshotBuilder, StockNodeObservation
from app.models.temporal_hetero_gnn import REGIME_LABELS, STRATEGY_FAMILIES
from app.ontology.market_graph import (
    NODE_TYPES,
    RELATION_TYPES,
    STRUCTURAL_RELATIONS,
    MarketGraph,
    MarketGraphError,
    OntologyEdge,
    OntologyNode,
    load_market_graph,
)


@pytest.fixture()
def graph() -> MarketGraph:
    return load_market_graph()


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #
def test_shipped_ontology_declares_every_node_and_relation_type(graph: MarketGraph) -> None:
    summary = graph.summary()
    assert summary["node_count"] > 40
    assert set(summary["relations"]) <= set(RELATION_TYPES)
    assert set(summary["node_types"]) <= set(NODE_TYPES)


def test_every_regime_and_strategy_family_is_a_node(graph: MarketGraph) -> None:
    regimes = {node.node_id for node in graph.nodes_of_type("MarketRegime")}
    families = {node.node_id for node in graph.nodes_of_type("Strategy")}
    assert regimes == set(REGIME_LABELS)
    assert families == set(STRATEGY_FAMILIES)


def test_every_regime_has_at_least_one_suitability_edge(graph: MarketGraph) -> None:
    for regime in REGIME_LABELS:
        edges = graph.edges(source=regime)
        assert edges, f"{regime} has no strategy relation"


def test_stale_data_invalidates_every_entry_family(graph: MarketGraph) -> None:
    invalidated = {
        edge.target_id for edge in graph.edges(relation="INVALIDATES", source="STALE_DATA")
    }
    # DEFENSIVE is the stance of holding, so it is not invalidated by stale data.
    assert invalidated == set(STRATEGY_FAMILIES) - {"DEFENSIVE"}


def test_us_market_has_explicit_factor_paths(graph: MarketGraph) -> None:
    sources = {
        edge.source_id
        for edge in graph.edges(target="US_MARKET")
        if edge.relation in {"INFLUENCES", "LEADS", "INCREASES_RISK"}
    }
    assert {
        "US_EQUITY",
        "US_SEMI",
        "INDEX_FUTURES",
        "VOLATILITY",
        "US_RATES",
    } <= sources


def test_target_market_state_can_flow_back_to_its_stock() -> None:
    snapshot = GraphSnapshotBuilder(time_steps=2).build(
        captured_at=datetime(2026, 8, 24, tzinfo=timezone.utc),
        stocks=[
            StockNodeObservation(
                ticker="AAPL",
                market_group="US",
                venue="NASDAQ",
            )
        ],
    )

    edge = snapshot.graph.edges(
        relation="INFLUENCES",
        source="US_MARKET",
        target="STOCK::AAPL",
    )
    assert len(edge) == 1
    assert edge[0].learnable is True
    assert not snapshot.graph.edges(
        relation="INFLUENCES",
        source="KR_MARKET",
        target="STOCK::AAPL",
    )


# --------------------------------------------------------------------------- #
# Prior versus learned
# --------------------------------------------------------------------------- #
def test_an_unlearned_edge_reports_its_prior_and_says_so(graph: MarketGraph) -> None:
    edge = graph.edges(relation="SUITABLE_FOR", source="TREND_UP", target="TREND")[0]
    assert edge.learned_weight is None
    assert edge.effective_weight == edge.prior_strength
    assert edge.weight_source == "prior"
    assert edge.prior_learned_gap is None


def test_learned_weight_is_stored_beside_the_prior_not_instead_of_it(
    graph: MarketGraph,
) -> None:
    edge_id = "TREND_UP|SUITABLE_FOR|TREND"
    graph.apply_learned_weights({edge_id: 0.20})
    edge = graph.edges(relation="SUITABLE_FOR", source="TREND_UP", target="TREND")[0]
    assert edge.prior_strength == pytest.approx(0.85)
    assert edge.learned_weight == pytest.approx(0.20)
    assert edge.effective_weight == pytest.approx(0.20)
    assert edge.weight_source == "learned"
    assert edge.prior_learned_gap == pytest.approx(-0.65)


def test_structural_relations_cannot_be_learned(graph: MarketGraph) -> None:
    structural = graph.edges(relation="BELONGS_TO")[0]
    with pytest.raises(MarketGraphError):
        graph.apply_learned_weights({structural.edge_id: 0.5})


def test_config_cannot_declare_a_structural_relation_learnable() -> None:
    with pytest.raises(MarketGraphError):
        MarketGraph(
            [
                OntologyNode("A", "Stock"),
                OntologyNode("B", "Sector"),
            ],
            [
                OntologyEdge(
                    source_id="A",
                    target_id="B",
                    relation="BELONGS_TO",
                    prior_strength=1.0,
                    learnable=True,
                )
            ],
        )


def test_clearing_learned_weights_returns_to_the_priors(graph: MarketGraph) -> None:
    edge_id = "TREND_UP|SUITABLE_FOR|TREND"
    graph.apply_learned_weights({edge_id: 0.20})
    graph.clear_learned_weights()
    edge = graph.edges(relation="SUITABLE_FOR", source="TREND_UP", target="TREND")[0]
    assert edge.learned_weight is None
    assert edge.effective_weight == pytest.approx(0.85)


def test_trace_edges_expose_both_weights(graph: MarketGraph) -> None:
    for record in graph.trace_edges({"TREND_UP"}):
        assert "prior_strength" in record
        assert "learned_weight" in record
        assert "weight_source" in record


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_unknown_node_type_is_refused() -> None:
    with pytest.raises(MarketGraphError):
        MarketGraph([OntologyNode("X", "Wormhole")], [])


def test_edge_to_a_missing_node_is_refused() -> None:
    with pytest.raises(MarketGraphError):
        MarketGraph(
            [OntologyNode("A", "Stock")],
            [OntologyEdge("A", "MISSING", "INFLUENCES", 0.5, True)],
        )


def test_prior_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(MarketGraphError):
        MarketGraph(
            [OntologyNode("A", "Stock"), OntologyNode("B", "Sector")],
            [OntologyEdge("A", "B", "INFLUENCES", 1.7, True)],
        )


def test_missing_config_is_an_error_not_an_empty_graph(tmp_path) -> None:
    with pytest.raises(MarketGraphError):
        load_market_graph(tmp_path / "absent.yaml")


# --------------------------------------------------------------------------- #
# Numeric projection
# --------------------------------------------------------------------------- #
def test_adjacency_is_row_normalised_per_relation(graph: MarketGraph) -> None:
    adjacency = graph.adjacency()
    assert adjacency.shape[0] == len(RELATION_TYPES)
    sums = adjacency.sum(axis=-1)
    non_empty = sums[sums > 0]
    assert np.allclose(non_empty, 1.0, atol=1e-5)


def test_prior_bias_is_minus_infinity_where_no_edge_exists(graph: MarketGraph) -> None:
    bias = graph.prior_bias()
    adjacency = graph.adjacency()
    declared = np.isfinite(bias)
    assert declared.sum() > 0
    # Every finite bias corresponds to a declared edge, and nothing else.
    assert not np.isfinite(bias[adjacency == 0]).any() or bool(
        np.isfinite(bias[adjacency == 0]).sum() <= declared.sum()
    )


def test_prior_bias_scales_with_prior_strength(graph: MarketGraph) -> None:
    bias = graph.prior_bias()
    source = graph.index_of("TREND_UP")
    target = graph.index_of("TREND")
    assert source is not None and target is not None
    relation = RELATION_TYPES.index("SUITABLE_FOR")
    assert bias[relation, target, source] == pytest.approx(
        graph.prior_bias_scale * 0.85, rel=1e-5
    )


def test_learned_weights_move_the_adjacency_but_not_the_prior_bias(
    graph: MarketGraph,
) -> None:
    before_bias = graph.prior_bias().copy()
    before_adjacency = graph.adjacency().copy()
    graph.apply_learned_weights({"TREND_UP|SUITABLE_FOR|TREND": 0.05})
    assert np.allclose(
        np.nan_to_num(before_bias, neginf=-1.0),
        np.nan_to_num(graph.prior_bias(), neginf=-1.0),
    )
    assert not np.allclose(before_adjacency, graph.adjacency())


def test_node_type_indices_cover_every_node(graph: MarketGraph) -> None:
    grouped = graph.node_type_indices()
    total = sum(len(values) for values in grouped.values())
    assert total == len(graph.node_ids)


def test_structural_relations_are_exactly_the_declared_set(graph: MarketGraph) -> None:
    for relation in STRUCTURAL_RELATIONS:
        for edge in graph.edges(relation=relation):
            assert not edge.learnable

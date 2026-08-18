"""Training the GNN and the ontology's learnable relation weights."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.context.temporal_context import build_temporal_snapshot
from app.models.gnn_runtime import GnnHealthState, GnnRuntime
from app.models.graph_snapshot import (
    FEATURE_DIM,
    GraphSnapshotBuilder,
    StockNodeObservation,
)
from app.models.temporal_hetero_gnn import TemporalHeteroGnn, TemporalHeteroGnnConfig
from app.models.temporal_hetero_gnn_training import (
    RelationOutcome,
    TrainingExample,
    evaluate_loss,
    fit_relation_weights,
    load_relation_weights,
    persist_relation_weights,
    train_temporal_hetero_gnn,
)
from app.ontology.market_graph import load_market_graph
from app.storage.trading_state_store import TradingStateStore

NOW = datetime(2026, 8, 19, 1, 30, tzinfo=timezone.utc)
CONFIG = TemporalHeteroGnnConfig(max_nodes=64, feature_dim=FEATURE_DIM, time_steps=4)


def _examples(count: int = 16, *, seed: int = 0) -> list[TrainingExample]:
    """Examples whose label is a deterministic function of one visible feature."""
    builder = GraphSnapshotBuilder(max_nodes=64, time_steps=4)
    temporal = build_temporal_snapshot("KRX", NOW)
    rng = np.random.default_rng(seed)
    examples: list[TrainingExample] = []
    for index in range(count):
        signal = float(rng.normal())
        snapshot = builder.build(
            captured_at=NOW,
            temporal=temporal,
            stocks=[
                StockNodeObservation(
                    f"{index:06d}",
                    sector="semiconductor",
                    venue="KRX",
                    session_return=signal * 0.01,
                    orderbook_imbalance=signal * 0.3,
                    spread_bps=8.0,
                )
            ],
        )
        examples.append(
            TrainingExample(
                snapshot=snapshot,
                node_id=f"STOCK::{index:06d}",
                regime_labels={
                    "TREND_UP": 1.0 if signal > 0 else 0.0,
                    "TREND_DOWN": 0.0 if signal > 0 else 1.0,
                },
                trade_quality=1.0 if signal > 0 else 0.0,
                realised_return_bps=signal * 30.0,
            )
        )
    return examples


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def test_heads_are_on_a_comparable_scale() -> None:
    """The return head must not dominate the two cross-entropies by two orders."""
    _, per_head = evaluate_loss(TemporalHeteroGnn(CONFIG), _examples())
    values = [value for value in per_head.values() if value > 0]
    assert values
    assert max(values) / min(values) < 100.0


def test_a_head_with_no_labels_contributes_nothing() -> None:
    unlabelled = [
        TrainingExample(snapshot=item.snapshot, node_id=item.node_id)
        for item in _examples(4)
    ]
    loss, per_head = evaluate_loss(TemporalHeteroGnn(CONFIG), unlabelled)
    assert loss == 0.0
    assert all(value == 0.0 for value in per_head.values())


def test_empty_example_set_scores_zero_rather_than_raising() -> None:
    assert evaluate_loss(TemporalHeteroGnn(CONFIG), []) == (0.0, {})


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def test_training_reduces_the_loss(tmp_path) -> None:
    _, report = train_temporal_hetero_gnn(
        _examples(), config=CONFIG, epochs=12, population=10, seed=3
    )
    assert report.improved
    assert report.final_loss < report.initial_loss
    assert report.example_count == 16


def test_training_writes_a_loadable_checkpoint(tmp_path) -> None:
    model, report = train_temporal_hetero_gnn(
        _examples(),
        config=CONFIG,
        epochs=4,
        population=6,
        checkpoint_path=tmp_path / "gnn.npz",
    )
    assert report.checkpoint_path is not None
    reloaded = TemporalHeteroGnn.load_checkpoint(tmp_path / "gnn.npz")
    assert reloaded.config == model.config
    runtime = GnnRuntime(checkpoint_path=tmp_path / "gnn.npz", config=CONFIG)
    assert runtime.health().state is GnnHealthState.HEALTHY


def test_the_returned_model_is_the_best_one_not_the_last(tmp_path) -> None:
    model, report = train_temporal_hetero_gnn(
        _examples(), config=CONFIG, epochs=8, population=6, seed=7
    )
    final, _ = evaluate_loss(model, _examples())
    # The report rounds to eight decimals; the model itself is exact.
    assert final == pytest.approx(report.final_loss, abs=1e-8)
    assert final <= report.initial_loss


def test_training_refuses_an_empty_example_set() -> None:
    with pytest.raises(ValueError):
        train_temporal_hetero_gnn([], config=CONFIG)


def test_training_refuses_a_mismatched_initial_model() -> None:
    other = TemporalHeteroGnn(
        TemporalHeteroGnnConfig(max_nodes=32, feature_dim=FEATURE_DIM, time_steps=4)
    )
    with pytest.raises(ValueError):
        train_temporal_hetero_gnn(_examples(4), config=CONFIG, initial=other)


# --------------------------------------------------------------------------- #
# Relation weights
# --------------------------------------------------------------------------- #
def test_relation_weight_is_the_shrunk_hit_rate() -> None:
    graph = load_market_graph()
    edge_id = "TREND_UP|SUITABLE_FOR|TREND"
    outcomes = [RelationOutcome((edge_id,), profitable=index < 26) for index in range(40)]
    learned = fit_relation_weights(graph, outcomes, minimum_samples=20, shrinkage_k=30.0)
    weight = 40 / (40 + 30)
    assert learned[edge_id] == pytest.approx(
        weight * (26 / 40) + (1 - weight) * 0.85, rel=1e-6
    )


def test_a_thin_sample_is_left_unlearned() -> None:
    graph = load_market_graph()
    outcomes = [
        RelationOutcome(("TREND_UP|SUITABLE_FOR|TREND",), profitable=True)
        for _ in range(5)
    ]
    assert fit_relation_weights(graph, outcomes, minimum_samples=20) == {}


def test_structural_edges_are_never_learned() -> None:
    graph = load_market_graph()
    structural = graph.edges(relation="BELONGS_TO")[0].edge_id
    outcomes = [RelationOutcome((structural,), profitable=True) for _ in range(100)]
    assert structural not in fit_relation_weights(graph, outcomes)


def test_learned_weights_persist_and_reload_beside_their_priors(tmp_path) -> None:
    store = TradingStateStore(tmp_path / "state.sqlite3")
    graph = load_market_graph()
    edge_id = "TREND_UP|SUITABLE_FOR|TREND"
    applied = persist_relation_weights(graph, {edge_id: 0.42}, store=store, updated_at=NOW)
    assert applied == 1

    row = store.fetch_one(
        "select * from ontology_edge where edge_id = ?", (edge_id,)
    )
    assert row is not None
    assert row["prior_strength"] == pytest.approx(0.85)
    assert row["learned_weight"] == pytest.approx(0.42)
    assert row["learned_updated_at"] is not None

    fresh = load_market_graph()
    assert load_relation_weights(fresh, store=store) == 1
    edge = fresh.edges(relation="SUITABLE_FOR", source="TREND_UP", target="TREND")[0]
    assert edge.prior_strength == pytest.approx(0.85)
    assert edge.learned_weight == pytest.approx(0.42)
    assert edge.weight_source == "learned"


def test_persisting_writes_every_edge_so_the_table_is_the_whole_graph(tmp_path) -> None:
    store = TradingStateStore(tmp_path / "state.sqlite3")
    graph = load_market_graph()
    persist_relation_weights(graph, {}, store=store, updated_at=NOW)
    assert store.count("ontology_edge") == len(graph.edges())


def test_reloading_ignores_a_weight_for_an_edge_that_became_structural(tmp_path) -> None:
    store = TradingStateStore(tmp_path / "state.sqlite3")
    graph = load_market_graph()
    persist_relation_weights(graph, {}, store=store, updated_at=NOW)
    structural = graph.edges(relation="BELONGS_TO")[0]
    with store.transaction() as conn:
        conn.execute(
            "update ontology_edge set learned_weight = 0.5 where edge_id = ?",
            (structural.edge_id,),
        )
    fresh = load_market_graph()
    # Applying it would raise; the loader must skip it instead.
    assert load_relation_weights(fresh, store=store) == 0

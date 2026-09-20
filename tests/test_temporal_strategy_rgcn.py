from datetime import datetime, timedelta, timezone
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from app.evaluation.stored_counterfactual import CounterfactualLabel
from app.features.strategy_graph_context import STRATEGY_GRAPH_CONTEXT_DIM, STRATEGY_GRAPH_CONTEXT_SCHEMA, context_index
from app.models.strategy_utility.rgcn import FixedShapeStrategyUtilityModel
from app.models.strategy_utility.strategy_graph import STRATEGY_NODE_COUNT, diagonal_strategy_mask, strategy_ids_for_market
from app.models.strategy_utility.temporal_graph import CausalGraphHistory, GraphObservation, causal_graph_inputs
from app.models.strategy_utility.training import _market_purged_split, train_counterfactual_checkpoint
from app.models.strategy_utility.label_contract import ENTRY_FROZEN_SHADOW_POLICY, LEGACY_BAR_POLICY
from app.routing.shadow_intelligence import ShadowIntelligenceService
from app.strategy.catalog import STRATEGY_IDS

NOW = datetime(2026, 8, 1, 1, tzinfo=timezone.utc)


def observation(seconds=0, symbol="005930", value=0.1):
    values = np.full(STRATEGY_GRAPH_CONTEXT_DIM, value, dtype=np.float32)
    values[context_index("is_krx")] = float(symbol == "005930")
    return GraphObservation(symbol, NOW + timedelta(seconds=seconds), tuple(values))


def test_temporal_inputs_use_elapsed_time_and_exclude_future_stale_other_symbol():
    current = observation(300)
    x, adjacency, weights = causal_graph_inputs(current, (
        observation(60, value=0.2), observation(240, value=0.3),
        observation(301, value=999), observation(-1000, value=999), observation(290, "AAPL", 999),
    ))
    node = STRATEGY_IDS.index("intraday_momentum")
    expected = np.asarray([2 ** -2, 2 ** -.5, 1])
    np.testing.assert_allclose(weights[:, node], expected / expected.sum(), rtol=1e-6)
    assert np.max(x) < 999
    assert x.shape == (3, STRATEGY_NODE_COUNT, STRATEGY_GRAPH_CONTEXT_DIM + STRATEGY_NODE_COUNT)
    us_only = STRATEGY_IDS.index("overnight_gap_carry")
    assert not adjacency[:, :, us_only].any()
    assert not weights[:, us_only].any()


def test_history_is_bounded_and_out_of_order_cannot_poison_newer_clock():
    history = CausalGraphHistory(max_symbols=2)
    history.inputs(observation(180))
    history.inputs(observation(60, value=999))
    x, _, _ = history.inputs(observation(240))
    assert np.max(x) < 999
    history.inputs(observation(240, "AAPL"))
    history.inputs(observation(240, "MSFT"))
    assert len(history._history) == 2


def test_missing_history_normalizes_current_instead_of_diluting_signal():
    _, _, weights = causal_graph_inputs(observation())
    node = STRATEGY_IDS.index("intraday_momentum")
    assert weights[:, node].tolist() == [0, 0, 1]


def test_empty_purged_training_never_restores_overlapping_labels():
    snapshots = [SimpleNamespace(symbol="005930", as_of=NOW + timedelta(minutes=i),
                                 label_end=NOW + timedelta(hours=2)) for i in range(10)]
    train, validation, purged = _market_purged_split(snapshots)
    assert train == [] and len(validation) == 2 and purged == 8


def test_shared_encoder_cannot_train_on_us_future_of_kr_holdout():
    snapshots = [SimpleNamespace(symbol=symbol, as_of=NOW + timedelta(minutes=offset + i * 3),
                                 label_end=NOW + timedelta(minutes=offset + i * 3 + 1))
                 for symbol, offset in (("005930", 0), ("AAPL", 1000)) for i in range(10)]
    train, validation, _ = _market_purged_split(snapshots)
    assert train and validation
    assert max(item.label_end for item in train) + timedelta(seconds=60) < min(item.as_of for item in validation)
    assert not any(item.symbol == "AAPL" for item in train)


def test_rdf_strategy_relation_instances_match_actual_tensor_topology():
    from rdflib import URIRef
    from app.models.strategy_utility.strategy_graph import materialize_strategy_graph, relation_object_properties, strategy_relation_adjacency
    graph = materialize_strategy_graph("KR")
    adjacency = strategy_relation_adjacency(market="KR")
    for index, iri in enumerate(relation_object_properties()):
        assert len(tuple(graph.triples((None, URIRef(iri), None)))) == np.count_nonzero(adjacency[index])


def labels(count=36, *, holdout_only_positive=False):
    for i in range(count):
        current = observation(i * 180, value=float(.15 * np.sin(i)))
        for strategy in STRATEGY_IDS:
            active = strategy == "intraday_momentum"
            net = (20 if i >= int(count * .8) else -25) if holdout_only_positive else (20 if i % 2 else -25)
            yield CounterfactualLabel(as_of=current.as_of, label_end=current.as_of + timedelta(seconds=60),
                                      symbol=current.symbol, strategy_id=strategy,
                                      triggered=active, filled=active, net_return_bps=net if active else 0,
                                      cost_bps=12, exit_reason="TIME", features=current.features)


def test_complete_trained_graph_is_saved_and_holdout_is_not_supervision(tmp_path, monkeypatch):
    monkeypatch.setenv("GNN_TRAINING_MAX_STEPS", "4")
    path = tmp_path / "temporal.npz"
    report = train_counterfactual_checkpoint(labels(holdout_only_positive=True), path,
                                             input_feature_schema=STRATEGY_GRAPH_CONTEXT_SCHEMA)
    model = FixedShapeStrategyUtilityModel.load_checkpoint(path)
    assert model.config.temporal_mode == 1 and model.config.time_steps == 3
    assert report["validation_metrics"]["gradient_steps"] == 4
    assert report["validation_metrics"]["relation_weight_update_l2"] > 0
    assert report["strategy_supervision"]["intraday_momentum"]["upside_rows"] == 0
    assert report["live_authorized"] is False
    assert report["label_execution_policy"] == LEGACY_BAR_POLICY
    assert report["dynamic_live_payoff_validated"] is False
    assert report["bounded_advisory_authorized_markets"] == []
    x, adj, weights = causal_graph_inputs(observation(600), (observation(300), observation(480)))
    mask = diagonal_strategy_mask(strategy_ids_for_market("005930"))
    raw, no_trade = model.infer_raw(x[None], adj[None], weights[None], mask[None])
    assert np.isfinite(raw).all() and np.isfinite(no_trade).all()
    initial = FixedShapeStrategyUtilityModel(model.config)
    assert not np.array_equal(model.relation_weights, initial.relation_weights)
    assert not np.array_equal(model.self_weight, initial.self_weight)


def test_graph_advisory_requires_market_promotion_and_freshness():
    service = ShadowIntelligenceService.__new__(ShadowIntelligenceService)
    service._latest_advisories = {}
    service.live_authorized_markets = ("KRX",)
    service.bounded_advisory_authorized_markets = ("KRX",)
    service.label_execution_policy = ENTRY_FROZEN_SHADOW_POLICY
    service.risk_advisory_strategy_markets = {"intraday_momentum": ("KRX",)}
    service.model = SimpleNamespace(config=SimpleNamespace(temporal_mode=1))
    service.checkpoint_hash = "trained-checkpoint"
    snapshot = SimpleNamespace(symbol="005930", as_of=NOW, valid_until=NOW + timedelta(seconds=5))
    evidence = SimpleNamespace(ontology_allowed=True, strategy_id="intraday_momentum", hard_block_reasons=(),
                               expected_net_return_bps=20, aleatoric_uncertainty=.3,
                               epistemic_uncertainty_or_proxy=.4, expected_adverse_excursion_bps=25,
                               probability_success=.7, ontology_snapshot_id="ontology:test", explanation_paths=())
    untrained = SimpleNamespace(**{**vars(evidence), "strategy_id": "breakout_volume", "expected_net_return_bps": 999})
    service._record_graph_advisory(snapshot, (evidence, untrained), ("NPU",))
    assert service.latest_graph_advisory("005930", NOW)["model_uncertainty"] == .4
    advice = service.latest_graph_advisory("005930", NOW)
    assert advice["authority"] == "bounded_risk_advisory_only"
    assert advice["label_execution_policy"] == ENTRY_FROZEN_SHADOW_POLICY
    assert advice["strategy_id"] == "intraday_momentum"
    assert "expected_net_return_bps" not in advice and "probability_success" not in advice
    assert not service._checkpoint_live_authorized_for("005930")
    assert service.latest_graph_advisory("005930", NOW + timedelta(seconds=6)) is None
    assert service.latest_graph_advisory("005930", NOW - timedelta(seconds=1)) is None
    snapshot.symbol = "AAPL"
    service._record_graph_advisory(snapshot, (evidence,), ("NPU",))
    assert service.latest_graph_advisory("AAPL", NOW) is None


def test_advisory_head_support_cannot_cross_market_or_invent_missing_losses():
    from app.models.strategy_utility.label_contract import BOUNDED_ADVISORY_SCOPE, risk_advisory_strategy_markets
    metadata = dict(label_execution_policy=ENTRY_FROZEN_SHADOW_POLICY, label_policy_provenance_matched=True,
        authorization_scope=BOUNDED_ADVISORY_SCOPE, bounded_advisory_authorized_markets=["KRX", "US"],
        label_outcomes_by_market={"KRX": {
            "intraday_momentum": {"filled": 50, "positive_net": 30, "negative_net": 20},
            "breakout_volume": {"filled": 100, "positive_net": 100},
        }, "US": {"intraday_momentum": {"filled": 30, "positive_net": 25, "negative_net": 5}}})
    assert risk_advisory_strategy_markets(metadata) == {"intraday_momentum": ("KRX",)}


def test_policy_missing_arms_are_wholly_masked_and_do_not_teach_no_trade(tmp_path, monkeypatch):
    from app.models.strategy_utility.training import _target_mask
    monkeypatch.setenv("GNN_TRAINING_MAX_STEPS", "2")
    rows = tuple(replace(row, label_execution_policy=ENTRY_FROZEN_SHADOW_POLICY,
        policy_id=f"policy:{row.as_of}" if row.triggered else "", feature_snapshot_id=f"features:{row.as_of}",
        net_return_bps=-25 if row.triggered else 0,
        exit_reason="TIME" if row.triggered else "FUTURE_WINDOW_CENSORED") for row in labels())
    assert not any(_target_mask(next(row for row in rows if not row.triggered)))
    path = tmp_path/"matched.npz"
    report = train_counterfactual_checkpoint(rows, path, input_feature_schema=STRATEGY_GRAPH_CONTEXT_SCHEMA,
                                            authorize_live_shadow=True)
    model = FixedShapeStrategyUtilityModel.load_checkpoint(path)
    assert not model.no_trade_head.any()
    assert report["label_policy_provenance_matched"] is True
    assert report["training_supervision_rows"] < 36
    assert report["live_authorized"] is False and report["dynamic_live_payoff_validated"] is False
    # A bar-only update cannot replace even an as-yet unpromoted compatible model.
    old = path.read_bytes()
    fallback = train_counterfactual_checkpoint(labels(), path, input_feature_schema=STRATEGY_GRAPH_CONTEXT_SCHEMA)
    assert fallback["retained_incumbent"] is True and path.read_bytes() == old


def test_policy_labels_require_provenance_and_cannot_mix_execution_contracts(tmp_path):
    rows = list(labels(count=3))
    rows[0] = replace(rows[0], label_execution_policy=ENTRY_FROZEN_SHADOW_POLICY)
    with pytest.raises(ValueError, match="mixed execution-policy"):
        train_counterfactual_checkpoint(rows, tmp_path/"mixed.npz", input_feature_schema=STRATEGY_GRAPH_CONTEXT_SCHEMA)
    rows = [replace(row, label_execution_policy=ENTRY_FROZEN_SHADOW_POLICY) for row in rows]
    with pytest.raises(ValueError, match="require frozen policy"):
        train_counterfactual_checkpoint(rows, tmp_path/"missing.npz", input_feature_schema=STRATEGY_GRAPH_CONTEXT_SCHEMA)


def test_actual_forward_journal_join_preserves_only_observed_target_supervision(tmp_path):
    from test_policy_shadow_labels import _record
    from app.models.strategy_utility.policy_labels import load_policy_shadow_labels
    from app.models.strategy_utility.training import _target_mask
    store, _, outcome = _record(tmp_path)
    rows = load_policy_shadow_labels(store.path, now=outcome.resolved_at + timedelta(seconds=1))
    assert len(rows) == len(STRATEGY_IDS)
    observed = [row for row in rows if any(_target_mask(row))]
    assert len(observed) == 1 and observed[0].policy_id == outcome.risk_policy_id
    assert observed[0].net_return_bps == pytest.approx(outcome.net_return_bps)

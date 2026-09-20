from concurrent.futures import Future
from datetime import datetime, timedelta, timezone
import hashlib
import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from app.features.strategy_graph_context import STRATEGY_GRAPH_CONTEXT_DIM, STRATEGY_GRAPH_CONTEXT_SCHEMA
from app.models.strategy_utility.online_training import GraphTrainingScheduler, _snapshot_database
from app.models.strategy_utility.rgcn import FixedShapeStrategyUtilityModel, StrategyUtilityModelConfig
from app.models.strategy_utility.strategy_graph import RELATION_NAMES, STRATEGY_NODE_COUNT
from app.models.strategy_utility.temporal_graph import ARCHITECTURE
from app.models.strategy_utility.label_contract import ENTRY_FROZEN_SHADOW_POLICY, LEGACY_BAR_POLICY, BOUNDED_ADVISORY_SCOPE
from app.routing.shadow_intelligence import _load_graph_replacement, ShadowIntelligenceService
from app.strategy.catalog import STRATEGY_IDS

NOW = datetime(2026, 9, 20, tzinfo=timezone.utc)


def test_scheduler_runs_one_job_and_waits_for_minimum_interval(monkeypatch):
    monkeypatch.setenv("GNN_TRAIN_INTERVAL_SECONDS", "300")
    monkeypatch.setenv("AUTO_TRAIN_TEMPORAL_RGCN", "true")
    clock = [0.0]
    calls = []
    scheduler = GraphTrainingScheduler(lambda: calls.append(True) or {"status": "complete"}, clock=lambda: clock[0])
    try:
        scheduler.tick()
        scheduler._future.result(timeout=3)
        assert scheduler.tick()["status"] == "complete"
        clock[0] = 299
        scheduler.tick()
        assert calls == [True]
        clock[0] = 300
        scheduler.tick()
        scheduler._future.result(timeout=3)
        assert len(calls) == 2
    finally:
        scheduler.close()


def test_scheduler_errors_are_reported_without_propagating_to_training_loop():
    scheduler = GraphTrainingScheduler(lambda: (_ for _ in ()).throw(ValueError("candidate invalid")))
    try:
        scheduler.tick()
        with pytest.raises(ValueError):
            scheduler._future.result(timeout=3)
        result = scheduler.tick()
        assert result["status"] == "error" and "candidate invalid" in result["reason"]
    finally:
        scheduler.close()


def test_database_snapshot_is_bounded_completed_and_preserves_both_markets(tmp_path, monkeypatch):
    monkeypatch.setenv("GNN_TRAIN_BARS_PER_SYMBOL", "120")
    source, target = tmp_path / "source.db", tmp_path / "snapshot.db"
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE TABLE realtime_minute_bars(symbol TEXT, minute_start TEXT, stream_id TEXT, close REAL)")
        conn.executemany("INSERT INTO realtime_minute_bars VALUES(?,?,?,?)", (
            (symbol, (NOW - timedelta(minutes=i)).isoformat(), "verified-feed", 100.0)
            for symbol in ("005930", "AAPL") for i in range(-2, 200)))
    before = source.read_bytes()
    result = _snapshot_database(source, target, NOW)
    assert result["rows"] == 240 and source.read_bytes() == before
    with sqlite3.connect(target) as conn:
        assert {row[0] for row in conn.execute("SELECT DISTINCT symbol FROM realtime_minute_bars")} == {"005930", "AAPL"}
        assert conn.execute("SELECT MAX(minute_start) FROM realtime_minute_bars").fetchone()[0] <= (NOW - timedelta(minutes=1)).isoformat()
        assert conn.execute("SELECT DISTINCT stream_id FROM realtime_minute_bars").fetchall() == [("verified-feed",)]


def checkpoint(tmp_path, markets=("KRX",)):
    config = StrategyUtilityModelConfig(1, 3, STRATEGY_NODE_COUNT, STRATEGY_GRAPH_CONTEXT_DIM + STRATEGY_NODE_COUNT,
                                        len(RELATION_NAMES), len(STRATEGY_IDS), seed=17, temporal_mode=1)
    model = FixedShapeStrategyUtilityModel(config)
    model.relation_weights += .01  # a changed, saved parameter tensor
    path = model.save_checkpoint(tmp_path / "model.npz")
    metadata = {"architecture": ARCHITECTURE, "input_feature_schema": STRATEGY_GRAPH_CONTEXT_SCHEMA,
                "strategy_ids": list(STRATEGY_IDS), "checkpoint_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
                "live_authorized_markets": [], "bounded_advisory_authorized_markets": list(markets),
                "label_execution_policy": ENTRY_FROZEN_SHADOW_POLICY, "label_policy_provenance_matched": True,
                "authorization_scope": BOUNDED_ADVISORY_SCOPE}
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    return path, config, model, metadata


def test_reload_loads_exact_parameters_from_matching_manifest_only(tmp_path):
    path, config, expected, metadata = checkpoint(tmp_path)
    model, card = _load_graph_replacement(path, config, "previous", ())
    np.testing.assert_array_equal(model.relation_weights, expected.relation_weights)
    assert card == metadata
    assert _load_graph_replacement(path, config, metadata["checkpoint_hash"], ()) is None
    metadata["checkpoint_hash"] = "wrong"
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        _load_graph_replacement(path, config, "previous", ())


def test_reload_rejects_authority_regression_and_keeps_incumbent_on_failure(tmp_path):
    path, config, incumbent, _ = checkpoint(tmp_path, markets=())
    with pytest.raises(ValueError, match="loses validated"):
        _load_graph_replacement(path, config, "previous", ("KRX",))
    service = ShadowIntelligenceService.__new__(ShadowIntelligenceService)
    service.model = incumbent
    service._reload_future = Future()
    service._reload_future.set_exception(ValueError("partial write"))
    service._reload_last_check = float("inf")
    service._maybe_reload_graph()
    assert service.model is incumbent and "partial write" in service.reload_error


def test_service_activates_validated_replacement_and_clears_old_advice(tmp_path, monkeypatch):
    monkeypatch.setenv("NPU_DEVICE_PREFERENCE", "CPU_NUMPY")
    path, config, model, metadata = checkpoint(tmp_path)
    candidate = _load_graph_replacement(path, config, "previous", ())
    service = ShadowIntelligenceService.__new__(ShadowIntelligenceService)
    service.model = FixedShapeStrategyUtilityModel(config)
    service.npu = None
    service._latest_advisories = {"005930": {"checkpoint_hash": "previous"}}
    service._reload_future = Future()
    service._reload_future.set_result(candidate)
    service._reload_last_check = float("inf")
    service._maybe_reload_graph()
    assert service.reload_error is None and service.checkpoint_hash == metadata["checkpoint_hash"]
    np.testing.assert_array_equal(service.model.relation_weights, model.relation_weights)
    assert service.cpu.reference is service.model and service.live_authorized_markets == ()
    assert service.bounded_advisory_authorized_markets == ("KRX",)
    assert not service._checkpoint_live_authorized_for("005930")
    assert not service._latest_advisories


def test_malformed_supervision_cannot_partially_replace_incumbent(tmp_path):
    _, config, model, metadata = checkpoint(tmp_path)
    metadata["minimum_upside_supervision_rows"] = "invalid"
    service = ShadowIntelligenceService.__new__(ShadowIntelligenceService)
    incumbent = FixedShapeStrategyUtilityModel(config)
    service.model = incumbent
    service._reload_future = Future()
    service._reload_future.set_result((model, metadata))
    service._reload_last_check = float("inf")
    service._maybe_reload_graph()
    assert service.model is incumbent and "invalid" in service.reload_error


def test_background_update_detects_kr_progress_even_when_us_clock_is_later(tmp_path, monkeypatch):
    from app.evaluation import stored_counterfactual as counterfactual
    from app.models.strategy_utility import online_training, training, policy_labels
    monkeypatch.setattr(policy_labels, "load_policy_shadow_labels", lambda *args, **kwargs: ())
    monkeypatch.setenv("GNN_MIN_NEW_LABELLED_SNAPSHOTS", "16")
    database = tmp_path / "source.db"
    database.touch()
    active = tmp_path / "active.npz"
    latest = (NOW - timedelta(hours=1)).isoformat()
    active.with_suffix(".training-state.json").write_text(json.dumps({
        "latest": latest, "symbol_progress": {"005930": {"latest": "older"}},
        "latest_label_time_by_market": {"KR": (NOW - timedelta(days=2)).isoformat(), "US": latest},
    }), encoding="utf-8")
    labels = [SimpleNamespace(symbol="005930", as_of=NOW-timedelta(days=1, minutes=i),
                              label_end=NOW-timedelta(hours=23), outcome_observed=True) for i in range(16)]
    monkeypatch.setattr(online_training, "_snapshot_database", lambda *args: {
        "rows": 120, "latest": latest, "symbol_progress": {"005930": {"latest": "newer"}, "AAPL": {"latest": latest}}})
    monkeypatch.setattr(counterfactual, "load_minute_bars", lambda path: {})
    monkeypatch.setattr(counterfactual, "load_minute_microstructure", lambda path: {})
    monkeypatch.setattr(counterfactual, "build_labels", lambda *args, **kwargs: labels)
    calls = []
    def train(rows, path, **kwargs):
        calls.append((rows, path, kwargs))
        return {"checkpoint": str(path), "live_authorized": False, "retained_incumbent": True,
                "validation_metrics": {"gradient_steps": 12}}
    monkeypatch.setattr(training, "train_counterfactual_checkpoint", train)
    result = online_training.run_graph_update(database=database, checkpoint=active, now=NOW)
    assert result["status"] == "complete" and result["new_snapshots"] == 16
    assert calls[0][0] == labels and calls[0][2]["authorize_live_shadow"] is False
    assert result["label_execution_policy"] == LEGACY_BAR_POLICY
    state = json.loads(active.with_suffix(".training-state.json").read_text())
    assert state["latest_label_time_by_market"]["KR"] == max(row.as_of.isoformat() for row in labels)
    assert state["latest_label_time_by_market"]["US"] == latest
    assert online_training.run_graph_update(database=database, checkpoint=active, now=NOW)["reason"] == "GRAPH_NO_NEW_COMPLETED_BARS"


def test_forward_labels_are_preferred_without_bar_database_and_new_completions_are_counted(tmp_path, monkeypatch):
    from app.models.strategy_utility import online_training, training, policy_labels
    monkeypatch.setenv("GNN_MIN_NEW_LABELLED_SNAPSHOTS", "16")
    labels = tuple(SimpleNamespace(symbol="005930", strategy_id="intraday_momentum",
        as_of=NOW-timedelta(hours=2, minutes=i), label_end=NOW-timedelta(hours=1),
        outcome_observed=True, policy_id=f"policy-{i}", feature_snapshot_id=f"features-{i}") for i in range(16))
    monkeypatch.setattr(policy_labels, "load_policy_shadow_labels", lambda *args, **kwargs: labels)
    monkeypatch.setattr(online_training, "_snapshot_database", lambda *args: pytest.fail("bar fallback must not run"))
    calls = []
    def train(rows, path, **kwargs):
        calls.append((rows, kwargs))
        return {"checkpoint": str(path), "live_authorized": False, "retained_incumbent": False,
                "validation_metrics": {"gradient_steps": 2}, "bounded_advisory_authorized_markets": []}
    monkeypatch.setattr(training, "train_counterfactual_checkpoint", train)
    kwargs = dict(database=tmp_path/"missing.db", checkpoint=tmp_path/"model.npz", now=NOW)
    result = online_training.run_graph_update(**kwargs)
    assert result["status"] == "complete" and result["label_execution_policy"] == ENTRY_FROZEN_SHADOW_POLICY
    assert calls[0][0] is labels and calls[0][1]["authorize_live_shadow"] is True
    assert online_training.run_graph_update(**kwargs)["reason"] == "GRAPH_INSUFFICIENT_NEW_LABELS"
    assert len(calls) == 1


def test_initial_legacy_flags_cannot_authorize_current_policy_advice_or_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("NPU_DEVICE_PREFERENCE", "CPU_NUMPY")
    path, _, _, metadata = checkpoint(tmp_path)
    metadata.update(label_execution_policy=LEGACY_BAR_POLICY, live_authorized=True, live_authorized_markets=["KRX"])
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setenv("REFACTOR_GNN_CHECKPOINT", str(path))
    service = ShadowIntelligenceService(feature_dim=STRATEGY_GRAPH_CONTEXT_DIM, comparison_path=tmp_path/"comparison.jsonl")
    assert not service.live_authorized and not service.bounded_advisory_authorized_markets
    assert not service._checkpoint_live_authorized_for("005930")
    assert service.latest_graph_advisory("005930", NOW) is None
    with pytest.raises(ValueError, match="research checkpoint"):
        _load_graph_replacement(path, service.model.config, "previous", (), ENTRY_FROZEN_SHADOW_POLICY)

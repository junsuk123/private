from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.features.feature_schema import LIVE_SHORT_HORIZON_SCHEMA
from app.models.adaptive_signal_model import network_logits, network_parameters
from app.models.live_model_trainer import train_live_short_horizon_model
from app.models.live_signal_predictor import LiveSignalPredictor
from app.models.model_artifact_registry import ModelArtifactRegistry
from app.models.model_validation import auc_like_score
from app.npu.runtime_manager import NpuRuntimeManager, _NumpyLinearModel, _NumpyNetwork


def _rows(count=600, seconds=60):
    start = datetime.now(timezone.utc) - timedelta(days=3)
    rows = []
    for index in range(count):
        features = dict.fromkeys(LIVE_SHORT_HORIZON_SCHEMA.feature_names, 0.0)
        # A genuinely nonlinear target: equal signs win; opposite signs lose.
        x, y = ((-1, -1), (-1, 1), (1, -1), (1, 1))[index % 4]
        features["return_1m"], features["orderbook_imbalance"] = x, y
        positive = x == y
        rows.append({"ticker": f"S{index % 5}", "as_of": (start + timedelta(seconds=index * seconds)).isoformat(),
            "features": features, "label": int(positive), "forward_net_return_bps": 80.0 if positive else -40.0})
    return rows


def test_cached_model_tracks_weights_and_real_fallback(monkeypatch):
    manager = NpuRuntimeManager(min_batch_for_npu=1, batch_buckets=(2, 4))
    monkeypatch.setattr(manager, "_requested_device", lambda *_: "NPU")
    compiler = Mock(side_effect=lambda name, requested, bucket, dim, w, b, a:
        (_NumpyLinearModel(w, b, a), "CPU_NUMPY", "NPU driver unavailable"))
    monkeypatch.setattr(manager, "_compile_linear_model", compiler)
    features = np.ones((2, 2), dtype=np.float32)
    old, _ = manager.run_linear(module_name="fitted", features=features, weights=np.ones((2, 1)))
    same, status = manager.run_linear(module_name="fitted", features=features, weights=np.ones((2, 1)))
    new, new_status = manager.run_linear(module_name="fitted", features=features, weights=np.ones((2, 1)) * 3)
    assert compiler.call_count == 2
    np.testing.assert_allclose(old, same)
    np.testing.assert_allclose(new, old * 3)
    assert status.backend == new_status.backend == "CPU_NUMPY"
    assert not status.uses_npu and status.fallback_reason == "NPU driver unavailable"


def test_legacy_scorer_compilation_does_not_block_live_network(monkeypatch):
    manager = NpuRuntimeManager(device_preference="NPU", min_batch_for_npu=1)
    entered, release = Event(), Event()
    parameters = {"mean": np.zeros(2), "scale": np.ones(2), "hidden_weights": np.ones((2, 4)),
        "hidden_bias": np.zeros(4), "output_weights": np.ones((4, 2)), "output_bias": np.zeros(2)}
    monkeypatch.setattr(manager, "_requested_device", lambda *_: "NPU")
    def compile_slow(name, requested, bucket, dim, weights, bias, activation):
        entered.set()
        release.wait(5)
        return _NumpyLinearModel(weights, bias, activation), "CPU_NUMPY", None
    monkeypatch.setattr(manager, "_compile_linear_model", compile_slow)
    monkeypatch.setattr(manager, "_compile_network", lambda *_: (_NumpyNetwork(parameters), "CPU_NUMPY", None))
    with ThreadPoolExecutor(max_workers=2) as executor:
        legacy = executor.submit(manager.run_linear, module_name="legacy", features=np.ones((2, 2)), weights=np.ones((2, 1)))
        try:
            assert entered.wait(1)
            live = executor.submit(manager.run_network, module_name="live", features=np.ones((1, 2)), parameters=parameters)
            output, _ = live.result(timeout=1)
            assert output.shape == (1, 2)
        finally:
            release.set()
            legacy.result(timeout=2)
            manager.close()


def test_static_batches_chunk_without_dropping_large_or_empty_input():
    manager = NpuRuntimeManager(device_preference="CPU", batch_buckets=(2, 4))
    features = np.arange(26, dtype=np.float32).reshape(13, 2)
    weights = np.array([[2.0], [-1.0]], dtype=np.float32)
    output, _ = manager.run_linear(module_name="chunks", features=features, weights=weights)
    np.testing.assert_allclose(output, features @ weights)
    empty, _ = manager.run_linear(module_name="chunks", features=np.empty((0, 2)), weights=weights)
    assert empty.shape == (0, 1)


def test_adaptive_model_learns_nonlinear_target_without_training_on_holdout(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_MODEL_FAMILY", "adaptive_relu")
    model = train_live_short_horizon_model(_rows(800), registry=ModelArtifactRegistry(tmp_path))
    assert model["nonlinear"]["family"] == "fixed_relu_two_head_v1"
    assert model["metrics"]["auc"] > 0.9
    assert model["metrics"]["holdout_evaluated"] == 1.0
    train_end = datetime.fromisoformat(model["nonlinear"]["trained_through"])
    val_start = datetime.fromisoformat(model["nonlinear"]["validation_start"])
    assert (val_start - train_end).total_seconds() > 660.0
    assert model["metrics"]["training_steps"] <= 128
    assert model["metrics"]["training_replay_count"] <= 4096
    assert model["metrics"]["parameter_count"] < 10000
    assert model["deployment"]["promoted"]
    assert ModelArtifactRegistry(tmp_path).load_latest_live_eligible().nonlinear


def test_incremental_update_is_bounded_and_preserves_unseen_validation(tmp_path):
    registry = ModelArtifactRegistry(tmp_path)
    initial_rows = _rows(800)
    initial = train_live_short_horizon_model(initial_rows, registry=registry)
    # Keep observations identical across versions; append genuinely newer data.
    newer_rows = _rows(900)
    initial_start = datetime.fromisoformat(initial_rows[0]["as_of"])
    for index, row in enumerate(newer_rows):
        row["as_of"] = (initial_start + timedelta(minutes=index)).isoformat()
    newer = train_live_short_horizon_model(newer_rows, registry=registry,
        warm_start_artifact=initial, update_rows=newer_rows[800:])
    assert newer["training_state"]["mode"] == "incremental"
    assert newer["metrics"]["training_steps"] == 16
    assert newer["metrics"]["training_replay_count"] <= 1024
    assert datetime.fromisoformat(initial["nonlinear"]["trained_through"]) < datetime.fromisoformat(newer["nonlinear"]["validation_start"]) - timedelta(seconds=660)


def test_future_warm_start_cannot_leak_into_validation(tmp_path):
    registry = ModelArtifactRegistry(tmp_path)
    initial = train_live_short_horizon_model(_rows(800), registry=registry)
    initial["nonlinear"]["trained_through"] = datetime.now(timezone.utc).isoformat()
    newer = train_live_short_horizon_model(_rows(800), registry=registry,
        warm_start_artifact=initial, update_rows=[])
    assert newer["training_state"]["mode"] == "full"


def test_long_plan_labels_are_purged_by_their_actual_horizon(tmp_path):
    rows = _rows(800)
    for row in rows:
        row["label_horizon_seconds"] = 7200.0
    model = train_live_short_horizon_model(rows, registry=ModelArtifactRegistry(tmp_path))
    train_end = datetime.fromisoformat(model["nonlinear"]["trained_through"])
    validation_start = datetime.fromisoformat(model["nonlinear"]["validation_start"])
    assert (validation_start - train_end).total_seconds() > 7260.0
    assert model["training_state"]["purge_seconds"] == 7260.0


def test_positive_alpha_with_negative_actual_cash_returns_never_promotes(tmp_path):
    rows = _rows(800)
    for row in rows:
        row["raw_forward_net_return_bps"] = -15.0 if row["label"] else -40.0
    model = train_live_short_horizon_model(rows, registry=ModelArtifactRegistry(tmp_path))
    assert not model["live_eligible"]
    assert model["metrics"]["positive_labels"] == 0
    assert "INSUFFICIENT_POSITIVE_LABELS" in model["reason_codes"]
    assert not (tmp_path / "latest.json").exists()


def test_horizon_and_cash_return_changes_invalidate_dataset_fingerprint():
    from app.models.live_training_pipeline import _training_rows_fingerprint
    original = _rows(1)
    cash_changed = [dict(original[0], raw_forward_net_return_bps=-10.0)]
    horizon_changed = [dict(original[0], label_horizon_seconds=7200.0)]
    assert len({_training_rows_fingerprint(rows) for rows in (original, cash_changed, horizon_changed)}) == 3


def test_promoted_nonlinear_artifact_is_used_by_live_prediction(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_MODEL_SPLIT_BY_MARKET", "false")
    registry = ModelArtifactRegistry(tmp_path)
    rows = _rows(800)
    artifact = train_live_short_horizon_model(rows, registry=registry)
    predictor = LiveSignalPredictor(registry)
    predictor._runtime = NpuRuntimeManager(device_preference="CPU")
    frame = SimpleNamespace(symbol="US.TEST", feature_schema_hash=LIVE_SHORT_HORIZON_SCHEMA.schema_hash,
        schema=LIVE_SHORT_HORIZON_SCHEMA, values=tuple(rows[0]["features"][name] for name in LIVE_SHORT_HORIZON_SCHEMA.feature_names))
    result = predictor.predict(frame)
    assert result.approved
    assert result.model_artifact_id == artifact["artifact_id"]
    assert result.expected_net_return_bps > 60.0
    assert result.inference_backend == "CPU_NUMPY"
    assert result.provider == "trained_model" and not result.is_fallback
    assert predictor.status()["model_family"] == "fixed_relu_two_head_v1"
    prediction_time = predictor.status()["last_prediction_at"]
    monkeypatch.setattr(registry, "load_latest_live_eligible", Mock(side_effect=RuntimeError("LATEST_MODEL_STALE")))
    with pytest.raises(RuntimeError, match="LATEST_MODEL_STALE"):
        predictor.predict(frame)
    status = predictor.status()
    assert status["available"] is False and status["approved"] is False
    assert status["reason_codes"] == ["LATEST_MODEL_STALE"]
    assert status["last_prediction_at"] == prediction_time
    assert status["artifact_id"] == artifact["artifact_id"]


def test_empty_purge_cannot_be_restored_as_valid_training(tmp_path):
    model = train_live_short_horizon_model(_rows(60, seconds=1), registry=ModelArtifactRegistry(tmp_path))
    assert not model["live_eligible"]
    assert "INSUFFICIENT_PURGED_HOLDOUT" in model["reason_codes"]
    assert not (tmp_path / "latest.json").exists()


def test_future_invalid_and_duplicate_rows_do_not_inflate_evidence(tmp_path):
    rows = _rows()
    rows.extend([dict(rows[0]), dict(rows[0])])
    future = dict(rows[-1], as_of=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat())
    malformed = dict(rows[0], features={})
    model = train_live_short_horizon_model(rows + [future, malformed], registry=ModelArtifactRegistry(tmp_path))
    assert model["metrics"]["example_count"] == 600
    assert model["metrics"]["rejected_rows"] == 2


def test_compile_never_blocks_prediction_and_status_never_probes(tmp_path, monkeypatch):
    model = train_live_short_horizon_model(_rows(), registry=ModelArtifactRegistry(tmp_path))
    p = network_parameters(model["nonlinear"])
    features = np.asarray([[row["features"][name] for name in LIVE_SHORT_HORIZON_SCHEMA.feature_names] for row in _rows(4)], dtype=np.float32)
    manager = NpuRuntimeManager(device_preference="NPU")
    release = Event()
    def compile_later(*_):
        release.wait(5)
        return _NumpyNetwork(p), "NPU", None
    monkeypatch.setattr(manager, "_compile_network", compile_later)
    try:
        output, status = manager.run_network(module_name="live", features=features, parameters=p)
        assert status.backend == "CPU_NUMPY"
        assert manager.snapshot_status()["pending_compilations"] == 1
        np.testing.assert_allclose(output, network_logits(features, model["nonlinear"]), atol=1e-5)
        release.set()
        list(manager._pending.values())[0].result(timeout=5)
        _, ready = manager.run_network(module_name="live", features=features, parameters=p)
        assert ready.backend == "NPU" and ready.uses_npu
    finally:
        release.set()
        manager._compiler.shutdown(wait=True)


def test_rank_auc_matches_pairwise_with_ties():
    labels = [1, 0, 1, 0, 1, 0]
    scores = [0.5, 0.5, 0.8, 0.7, 0.4, 0.2]
    positives = [s for s, y in zip(scores, labels) if y]
    negatives = [s for s, y in zip(scores, labels) if not y]
    expected = sum(float(p > n) + 0.5 * float(p == n) for p in positives for n in negatives) / 9
    assert auc_like_score(labels, scores) == expected
    with pytest.raises(ValueError):
        auc_like_score([1, 0], [float("nan"), 0.0])


def test_artifacts_default_to_machine_local_store(tmp_path, monkeypatch):
    monkeypatch.setenv("REALTIME_STORE_ROOT", str(tmp_path / "local-runtime"))
    monkeypatch.delenv("LIVE_MODEL_ARTIFACT_ROOT", raising=False)
    registry = ModelArtifactRegistry()
    assert registry.root == tmp_path / "local-runtime/models/live_short_horizon"


def test_fused_openvino_graph_executes_on_cpu_when_installed(tmp_path, monkeypatch):
    pytest.importorskip("openvino")
    monkeypatch.setenv("REALTIME_STORE_ROOT", str(tmp_path))
    manager = NpuRuntimeManager(device_preference="NPU")
    monkeypatch.setattr(manager, "_best_openvino_device", lambda: "CPU")
    rng = np.random.default_rng(17)
    parameters = {
        "mean": np.zeros(4, dtype=np.float32), "scale": np.ones(4, dtype=np.float32),
        "hidden_weights": rng.normal(size=(4, 8)).astype(np.float32),
        "hidden_bias": rng.normal(size=8).astype(np.float32),
        "output_weights": rng.normal(size=(8, 2)).astype(np.float32),
        "output_bias": np.zeros(2, dtype=np.float32),
    }
    compiled, backend, reason = manager._compile_network("smoke", 8, parameters)
    assert backend == "CPU", reason
    features = rng.normal(size=(8, 4)).astype(np.float32)
    np.testing.assert_allclose(compiled([features])[0], _NumpyNetwork(parameters)([features])[0], rtol=0.005, atol=0.01)

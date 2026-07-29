from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

from app.routing.shadow_intelligence import (
    ShadowIntelligenceService,
    SlowIntelligenceSnapshot,
)


NOW = datetime(2026, 7, 27, 1, 0, tzinfo=timezone.utc)


def test_slow_intelligence_runs_order_free_comparison_and_throttles(tmp_path) -> None:
    service = ShadowIntelligenceService(
        feature_dim=4,
        minimum_interval_seconds=2,
        comparison_path=tmp_path / "shadow.jsonl",
    )
    snapshot = SlowIntelligenceSnapshot(
        snapshot_id="snapshot-1",
        symbol="005930",
        as_of=NOW,
        valid_until=NOW + timedelta(seconds=5),
        feature_snapshot_id="features-1",
        features=(0.1, 0.2, 0.3, 0.4),
        data_fresh=True,
        tradable=True,
        allowed_strategy_ids=("intraday_momentum", "breakout_volume"),
    )
    result = service.evaluate(snapshot, legacy_action="BUY")
    assert result is not None
    assert len(result.cpu_evidence) == 7
    assert not result.npu_evidence
    assert result.comparison.decisions[0].path == "legacy"
    assert service.evaluate(snapshot, legacy_action="BUY") is None
    assert not hasattr(service, "broker")


def test_stale_data_hard_masks_every_strategy(tmp_path) -> None:
    service = ShadowIntelligenceService(
        feature_dim=2,
        minimum_interval_seconds=0,
        comparison_path=tmp_path / "shadow.jsonl",
    )
    result = service.evaluate(
        SlowIntelligenceSnapshot(
            snapshot_id="snapshot-2",
            symbol="005930",
            as_of=NOW,
            valid_until=NOW + timedelta(seconds=5),
            feature_snapshot_id="features-2",
            features=(0.1, 0.2),
            data_fresh=False,
            tradable=True,
            allowed_strategy_ids=("intraday_momentum",),
        )
    )
    assert result is not None
    assert all(not item.ontology_allowed for item in result.cpu_evidence)
    assert result.comparison.decisions[-1].action == "NO_TRADE"


def test_checkpoint_with_incompatible_live_schema_is_not_scored(tmp_path, monkeypatch) -> None:
    seed_service = ShadowIntelligenceService(
        feature_dim=12,
        minimum_interval_seconds=0,
        comparison_path=tmp_path / "seed-shadow.jsonl",
    )
    checkpoint = seed_service.model.save_checkpoint(tmp_path / "model.npz")
    checkpoint.with_suffix(".json").write_text(
        json.dumps(
            {
                "method": "causal_feature_encoder_plus_ridge_calibrated_heads",
                "input_feature_schema": "counterfactual_quantiles_v1",
                "live_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("REFACTOR_GNN_CHECKPOINT", str(checkpoint))
    service = ShadowIntelligenceService(
        feature_dim=12,
        minimum_interval_seconds=0,
        comparison_path=tmp_path / "shadow.jsonl",
    )

    result = service.evaluate(
        SlowIntelligenceSnapshot(
            snapshot_id="snapshot-schema",
            symbol="396500",
            as_of=NOW,
            valid_until=NOW + timedelta(seconds=5),
            feature_snapshot_id="features-schema",
            features=(0.1,) * 12,
            data_fresh=True,
            tradable=True,
            allowed_strategy_ids=("intraday_momentum",),
            feature_schema_name="realtime_microstructure_v1",
        )
    )

    assert result is not None
    assert result.cpu_evidence == ()
    cpu = result.comparison.decisions[-1]
    assert cpu.action == "NO_TRADE"
    assert any(reason.startswith("MODEL_INPUT_SCHEMA_MISMATCH:") for reason in cpu.reason_codes)
    assert "UTILITY_MODEL_NOT_LIVE_AUTHORIZED" in cpu.reason_codes


def test_authorized_checkpoint_with_matching_schema_emits_live_shadow_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    seed_service = ShadowIntelligenceService(
        feature_dim=12,
        minimum_interval_seconds=0,
        comparison_path=tmp_path / "seed-shadow.jsonl",
    )
    checkpoint = seed_service.model.save_checkpoint(tmp_path / "model.npz")
    checkpoint.with_suffix(".json").write_text(
        json.dumps(
            {
                "input_feature_schema": "realtime_microstructure_v1",
                "live_authorized": True,
                "authorization_scope": "shadow_inference_only",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("REFACTOR_GNN_CHECKPOINT", str(checkpoint))
    service = ShadowIntelligenceService(
        feature_dim=12,
        minimum_interval_seconds=0,
        comparison_path=tmp_path / "shadow.jsonl",
    )

    result = service.evaluate(
        SlowIntelligenceSnapshot(
            snapshot_id="snapshot-live-schema",
            symbol="396500",
            as_of=NOW,
            valid_until=NOW + timedelta(seconds=5),
            feature_snapshot_id="features-live-schema",
            features=(0.1,) * 12,
            data_fresh=True,
            tradable=True,
            allowed_strategy_ids=("intraday_momentum",),
            feature_schema_name="realtime_microstructure_v1",
        )
    )

    assert result is not None
    assert len(result.cpu_evidence) == 7
    assert result.comparison.decisions[-1].path == "cpu_gnn"

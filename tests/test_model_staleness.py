from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from app.models.model_artifact_registry import ModelArtifactRegistry
from app.models.model_staleness import (
    MODEL_AGE_EXCEEDED,
    MODEL_CONSECUTIVE_NEGATIVE_CHALLENGERS,
    MODEL_CREATED_AT_UNKNOWN,
    MODEL_DEPLOYABLE_SAMPLE_TOO_SMALL,
    MODEL_FEATURE_DRIFT,
    MODEL_FRESH,
    MODEL_REGIME_MISMATCH,
    ModelTrustLevel,
    StalenessConfig,
    evaluate_model_staleness,
)


NOW = datetime(2026, 7, 28, 1, 0, tzinfo=timezone.utc)
CONFIG = StalenessConfig(
    max_age_seconds=3_600.0,
    max_feature_drift_score=0.35,
    consecutive_negative_challengers=3,
)


def _payload(*, created_at=NOW, regime=None, **metrics):
    payload = {
        "artifact_id": "live_short_horizon.test",
        "created_at": created_at.isoformat() if created_at else None,
        "metrics": {"auc": 0.84, **metrics},
        "live_eligible": True,
    }
    if regime:
        payload["training_state"] = {"macro_regime": regime}
    return payload


def test_fresh_model_is_live():
    verdict = evaluate_model_staleness(_payload(), now=NOW, config=CONFIG)
    assert not verdict.stale
    assert verdict.trust_level is ModelTrustLevel.LIVE
    assert MODEL_FRESH in verdict.reason_codes


def test_old_model_is_demoted_to_shadow():
    verdict = evaluate_model_staleness(
        _payload(created_at=NOW - timedelta(hours=9)), now=NOW, config=CONFIG
    )
    assert verdict.stale
    assert MODEL_AGE_EXCEEDED in verdict.reason_codes
    assert verdict.trust_level is ModelTrustLevel.SHADOW_ONLY


def test_unknown_age_is_treated_as_stale_not_as_fresh():
    verdict = evaluate_model_staleness(_payload(created_at=None), now=NOW, config=CONFIG)
    assert verdict.stale
    assert MODEL_CREATED_AT_UNKNOWN in verdict.reason_codes


def test_feature_drift_expires_the_model():
    verdict = evaluate_model_staleness(
        _payload(), now=NOW, config=CONFIG, feature_drift_score=0.9
    )
    assert verdict.stale
    assert MODEL_FEATURE_DRIFT in verdict.reason_codes


def test_runtime_aligned_model_with_one_deployable_row_is_shadow_only():
    verdict = evaluate_model_staleness(
        _payload(
            runtime_policy_aligned_evaluation=1.0,
            top_k_count=1.0,
        ),
        now=NOW,
        config=CONFIG,
    )
    assert verdict.stale
    assert MODEL_DEPLOYABLE_SAMPLE_TOO_SMALL in verdict.reason_codes
    assert verdict.trust_level is ModelTrustLevel.SHADOW_ONLY


def test_repeated_negative_challengers_expire_a_confident_incumbent():
    """The measured case: incumbent AUC 0.84, every challenger negative."""
    verdict = evaluate_model_staleness(
        _payload(),
        now=NOW,
        config=CONFIG,
        recent_challenger_return_bps=(-54.2, -12.0, -30.5, 8.0),
    )
    assert verdict.stale
    assert verdict.consecutive_negative_challengers == 3
    assert MODEL_CONSECUTIVE_NEGATIVE_CHALLENGERS in verdict.reason_codes


def test_a_positive_challenger_breaks_the_negative_streak():
    verdict = evaluate_model_staleness(
        _payload(),
        now=NOW,
        config=CONFIG,
        recent_challenger_return_bps=(15.0, -54.2, -12.0, -30.5),
    )
    assert verdict.consecutive_negative_challengers == 0
    assert not verdict.stale


def test_regime_mismatch_expires_the_model():
    verdict = evaluate_model_staleness(
        _payload(regime="TREND_UP"),
        now=NOW,
        config=CONFIG,
        current_regime="HIGH_VOL_DISLOCATED",
    )
    assert verdict.stale
    assert MODEL_REGIME_MISMATCH in verdict.reason_codes


def test_matching_regime_is_not_a_mismatch():
    verdict = evaluate_model_staleness(
        _payload(regime="HIGH_VOL_TRENDING"),
        now=NOW,
        config=CONFIG,
        current_regime="high_vol_trending",
    )
    assert not verdict.stale


def test_registry_counts_consecutive_negative_challengers(tmp_path):
    registry = ModelArtifactRegistry(tmp_path)
    for index in range(3):
        registry.save(
            {
                "artifact_id": f"live_short_horizon.{index}",
                "created_at": NOW.isoformat(),
                "feature_schema_hash": "hash",
                "feature_names": ["a"],
                "classification": {"weights": [0.0], "bias": 0.0},
                "regression": {"weights": [0.0], "bias": 0.0},
                "thresholds": {"minimum_probability_success": 0.51},
                "metrics": {"avg_forward_net_return_bps_top_k": -54.0},
                "live_eligible": False,
                "reason_codes": ["METRICS_BELOW_LIVE_THRESHOLDS"],
            }
        )
    state = json.loads(registry.deployment_state_path.read_text(encoding="utf-8"))
    assert state["consecutive_negative_challengers"] == 3


def test_registry_refuses_to_serve_a_stale_incumbent(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_MODEL_MAX_AGE_SECONDS", "60")
    registry = ModelArtifactRegistry(tmp_path)
    registry.save(
        {
            "artifact_id": "live_short_horizon.old",
            "created_at": (NOW - timedelta(days=2)).isoformat(),
            "feature_schema_hash": "hash",
            "feature_names": ["a"],
            "classification": {"weights": [0.0], "bias": 0.0},
            "regression": {"weights": [0.0], "bias": 0.0},
            "thresholds": {"minimum_probability_success": 0.51},
            "metrics": {"auc": 0.84, "avg_forward_net_return_bps_top_k": 49.9},
            "live_eligible": True,
            "reason_codes": [],
        }
    )
    assert registry.latest_path.exists()
    verdict = registry.staleness()
    assert verdict.stale
    try:
        registry.load_latest_live_eligible()
    except RuntimeError as exc:
        assert "LATEST_MODEL_STALE" in str(exc)
    else:  # pragma: no cover - the raise is the contract
        raise AssertionError("a stale incumbent must not be served live")
    # The artifact itself is preserved for audit, and enforcement is overridable.
    assert registry.load_latest_live_eligible(enforce_staleness=False).live_eligible

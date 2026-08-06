from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import json
from types import SimpleNamespace

from app.routing.shadow_intelligence import (
    ShadowIntelligenceService,
    SlowIntelligenceSnapshot,
    _shadow_route,
    slow_snapshot_from_live_feature_frame,
)
from app.strategy.catalog import STRATEGY_IDS


NOW = datetime(2026, 7, 27, 1, 0, tzinfo=timezone.utc)


def test_live_feature_frame_adapter_emits_trained_28_field_contract() -> None:
    values = {
        "distance_from_vwap": 0.01,
        "spread_bps": 12.0,
        "orderbook_imbalance": 0.2,
        "aggressor_imbalance_5s": 0.4,
        "volume_5s_log": 2.0,
        "realized_volatility_3m": 0.003,
        "rvgi_available": 1.0,
        "rvgi": 0.2,
        "rvgi_signal": 0.1,
        "rvgi_diff": 0.1,
        "rvgi_slope": 0.02,
        "rvgi_bullish_cross": 1.0,
        "box_available": 1.0,
        "box_high": 10.1,
        "box_low": 9.8,
        "box_mid": 9.95,
        "box_width_pct": 0.03,
        "box_position": 0.8,
        "breakout_distance_bps": 5.0,
        "box_previous_close": 9.9,
        "box_context_timestamp_epoch": NOW.timestamp(),
    }
    frame = SimpleNamespace(
        symbol="SOFI",
        decision_time=NOW,
        mark_price=10.0,
        feature_schema_hash="schema",
        provenance=SimpleNamespace(
            orderbook_record_id="book-1",
            tick_record_ids=("tick-1",),
        ),
        validate=lambda: None,
        as_feature_dict=lambda: values,
    )

    snapshot = slow_snapshot_from_live_feature_frame(frame)

    assert len(snapshot.features) == 28
    assert snapshot.symbol == "SOFI"
    assert snapshot.features[0] == 0.0001
    assert snapshot.features[27] == 0.0
    assert snapshot.feature_schema_name == "realtime_strategy_graph_v4_market"
    assert snapshot.data_fresh is True
    assert snapshot.tradable is True


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
    assert result.cpu_evidence == ()
    # Fails closed on a SCHEMA reason. Which schema differs first is not the point of
    # this test, and the shipped checkpoint now differs on two axes: it predates both
    # this service's feature_dim and the head widening that added the borrow channels
    # (8 -> 11). Either code is a correct, actionable answer; "corrupt" would not be,
    # which is why the loader distinguishes them.
    reason_codes = result.comparison.decisions[-1].reason_codes
    assert {"GNN_FEATURE_SCHEMA_MISMATCH", "GNN_HEAD_SCHEMA_MISMATCH"} & set(reason_codes)
    assert "GNN_CHECKPOINT_CORRUPT" not in reason_codes
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
                    "strategy_ids": list(STRATEGY_IDS),
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
    ontology = next(
        decision
        for decision in result.comparison.decisions
        if decision.path == "ontology"
    )
    assert ontology.action == "GATE_ONLY"
    assert ontology.strategy_id is None
    assert "ONTOLOGY_GATE_ONLY" in ontology.reason_codes
    cpu = result.comparison.decisions[-1]
    assert cpu.action == "NO_TRADE"
    assert "GNN_FEATURE_SCHEMA_MISMATCH" in cpu.reason_codes
    assert "GNN_NOT_LIVE_AUTHORIZED" in cpu.reason_codes


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
                    "strategy_ids": list(STRATEGY_IDS),
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
    assert len(result.cpu_evidence) == len(STRATEGY_IDS)
    assert result.comparison.decisions[-1].path == "cpu_gnn"
    assert "GNN_REALTIME_TRUST_NOT_READY" in result.comparison.decisions[-1].reason_codes


def test_zero_compatibility_strategies_are_closed_world_blocked(tmp_path) -> None:
    service = ShadowIntelligenceService(
        feature_dim=12,
        minimum_interval_seconds=0,
        comparison_path=tmp_path / "shadow.jsonl",
    )
    snapshot = SlowIntelligenceSnapshot(
        snapshot_id="compatibility-gate",
        symbol="005930",
        as_of=NOW,
        valid_until=NOW + timedelta(seconds=5),
        feature_snapshot_id="features-compatibility",
        features=(
            1.0,
            1.0,
            0.01,
            1.0,
            0.3,
            0.0,
            0.2,
            1.0,
            0.1,
            1.0,
            0.0,
            1.0,
        ),
        data_fresh=True,
        tradable=True,
        allowed_strategy_ids=STRATEGY_IDS,
    )

    ontology = service._ontology(snapshot)

    assert "intraday_momentum" in ontology.allowed_strategy_ids
    assert ontology.compatibility_scores["intraday_momentum"] > 0.0
    assert "event_momentum" not in ontology.allowed_strategy_ids
    assert ontology.compatibility_scores["event_momentum"] == 0.0
    assert (
        "REQUIRED_TRUE_FAILED:compatible:event_momentum"
        in ontology.blocked_strategy_reasons["event_momentum"]
    )


def test_no_trade_route_does_not_fabricate_zero_compatibility_validation() -> None:
    route = SimpleNamespace(
        action="NO_TRADE",
        selected=None,
        weighted_utility=None,
        reason_codes=("UTILITY_BELOW_THRESHOLD:event_momentum",),
    )
    zero_compatibility_evidence = SimpleNamespace(
        ontology_allowed=True,
        hard_block_reasons=(),
        compatibility_score=0.0,
        expected_net_return_bps=20.0,
        utility=10.0,
    )

    decision = _shadow_route(
        "cpu_gnn",
        route,
        evidence=(zero_compatibility_evidence,),
        checkpoint_hash="checkpoint-v4",
    )

    assert decision.action == "NO_TRADE"
    assert decision.validation_strategy_id is None
    assert decision.ontology_compatibility is None
    assert decision.probability_success is None
    assert (
        "GNN_NO_ONTOLOGY_ADMISSIBLE_VALIDATION_CANDIDATE"
        in decision.reason_codes
    )


def test_checkpoint_with_same_count_but_different_strategy_order_is_rejected(
    tmp_path, monkeypatch
) -> None:
    seed = ShadowIntelligenceService(
        feature_dim=12,
        minimum_interval_seconds=0,
        comparison_path=tmp_path / "seed.jsonl",
    )
    checkpoint = seed.model.save_checkpoint(tmp_path / "model.npz")
    checkpoint.with_suffix(".json").write_text(
        json.dumps(
            {
                "input_feature_schema": "realtime_microstructure_v1",
                "live_authorized": True,
                "strategy_ids": list(reversed(STRATEGY_IDS)),
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
            snapshot_id="order-mismatch",
            symbol="005930",
            as_of=NOW,
            valid_until=NOW + timedelta(seconds=5),
            feature_snapshot_id="features-order",
            features=(0.1,) * 12,
            data_fresh=True,
            tradable=True,
            allowed_strategy_ids=STRATEGY_IDS,
            feature_schema_name="realtime_microstructure_v1",
        )
    )
    assert result is not None
    assert result.cpu_evidence == ()
    assert "GNN_STRATEGY_CATALOG_MISMATCH" in result.comparison.decisions[-1].reason_codes


def _service_with_supervision(tmp_path, monkeypatch, metadata_extra: dict):
    """Build a service whose checkpoint declares the given supervision metadata."""
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
                "strategy_ids": list(STRATEGY_IDS),
                **metadata_extra,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("REFACTOR_GNN_CHECKPOINT", str(checkpoint))
    return ShadowIntelligenceService(
        feature_dim=12,
        minimum_interval_seconds=0,
        comparison_path=tmp_path / "shadow.jsonl",
    )


def _evaluate_all_strategies(service):
    return service.evaluate(
        SlowIntelligenceSnapshot(
            snapshot_id="snapshot-supervision",
            symbol="396500",
            as_of=NOW,
            valid_until=NOW + timedelta(seconds=5),
            feature_snapshot_id="features-supervision",
            features=(0.1,) * 12,
            data_fresh=True,
            tradable=True,
            allowed_strategy_ids=STRATEGY_IDS,
            feature_schema_name="realtime_microstructure_v1",
        )
    )


def test_unsupervised_upside_head_cannot_forecast_a_positive_net_edge(
    tmp_path,
    monkeypatch,
) -> None:
    """A strategy whose MFE channel was never taught must not claim an upside.

    ``net = probability * mfe - (1 - probability) * mae`` makes MFE the only
    positive term in a forecast, and MFE trains only on realized PROFITABLE fills.
    On the 2026-08-03 checkpoint that was 0-21 rows per strategy. Measured
    consequence: every one of 51 positive-edge forecasts in the 2026-08-06T05:30Z
    window came from an under-taught head, and they realized -108bps at a 22% win
    rate. Suppressing the term they have no evidence for is what makes the
    remaining forecasts mean something.
    """
    kept = _service_with_supervision(
        tmp_path / "kept",
        monkeypatch,
        {"upside_supervised_strategy_ids": list(STRATEGY_IDS)},
    )
    suppressed = _service_with_supervision(
        tmp_path / "suppressed",
        monkeypatch,
        {"upside_supervised_strategy_ids": []},
    )

    kept_evidence = {
        item.strategy_id: item
        for item in _evaluate_all_strategies(kept).cpu_evidence
    }
    suppressed_evidence = {
        item.strategy_id: item
        for item in _evaluate_all_strategies(suppressed).cpu_evidence
    }

    assert set(kept_evidence) == set(STRATEGY_IDS)
    for strategy_id, before in kept_evidence.items():
        after = suppressed_evidence[strategy_id]
        # Exactly the unsupported term comes off, nothing else. Asserting the
        # delta rather than the sign keeps this meaningful for any weights: the
        # removed quantity is sigmoid * softplus, so it is always > 0.
        removed = before.probability_success * before.expected_favorable_excursion_bps
        assert removed > 0.0
        assert after.expected_net_return_bps == pytest.approx(
            before.expected_net_return_bps - removed
        )
        # The safety property this exists for.
        assert after.expected_net_return_bps <= 0.0
        # The contract that keeps the three numbers honest together must hold.
        assert after.expected_net_return_bps == pytest.approx(
            after.expected_gross_return_bps - after.expected_cost_bps
        )


def test_supervised_upside_head_keeps_its_forecast(tmp_path, monkeypatch) -> None:
    supervised = "intraday_momentum"
    suppressed = _service_with_supervision(
        tmp_path / "off",
        monkeypatch,
        {"upside_supervised_strategy_ids": []},
    )
    kept = _service_with_supervision(
        tmp_path / "on",
        monkeypatch,
        {"upside_supervised_strategy_ids": [supervised]},
    )

    def net_for(service, strategy_id):
        result = _evaluate_all_strategies(service)
        return next(
            item.expected_net_return_bps
            for item in result.cpu_evidence
            if item.strategy_id == strategy_id
        )

    # Same weights, same snapshot: the only difference is the declared evidence.
    assert net_for(kept, supervised) > net_for(suppressed, supervised)


def test_checkpoint_without_supervision_metadata_fails_closed(
    tmp_path,
    monkeypatch,
) -> None:
    """Silence is not proof. A checkpoint that never reported its supervision
    cannot be read as having demonstrated any."""
    service = _service_with_supervision(tmp_path, monkeypatch, {})

    assert service.upside_supervised_strategy_ids == ()
    result = _evaluate_all_strategies(service)
    assert all(
        item.expected_net_return_bps <= 0.0 for item in result.cpu_evidence
    )


def test_legacy_checkpoint_derives_supervision_from_label_outcomes(
    tmp_path,
    monkeypatch,
) -> None:
    """Pre-existing checkpoints carry the same fact under a different name.

    ``label_outcomes[*].positive_net`` counts realized profitable fills, which are
    exactly the rows that trained MFE — so the safety property is recoverable
    without forcing a retrain.
    """
    service = _service_with_supervision(
        tmp_path,
        monkeypatch,
        {
            "label_outcomes": {
                "intraday_momentum": {"positive_net": 25},
                "breakout_volume": {"positive_net": 4},
                "rvgi_box_breakout": {"positive_net": 1},
            }
        },
    )

    assert service.upside_supervised_strategy_ids == ("intraday_momentum",)

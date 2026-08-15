from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import json
from types import SimpleNamespace

from app.features.strategy_graph_context import (
    STRATEGY_GRAPH_CONTEXT_DIM,
    STRATEGY_GRAPH_CONTEXT_FIELDS,
    STRATEGY_GRAPH_CONTEXT_SCHEMA,
    StrategyGraphContextError,
    as_context_mapping,
    build_strategy_graph_context,
)
from app.routing.shadow_intelligence import (
    ShadowIntelligenceService,
    SlowIntelligenceSnapshot,
    _shadow_route,
    slow_snapshot_from_live_feature_frame,
)
from app.strategy.catalog import STRATEGY_IDS


NOW = datetime(2026, 7, 27, 1, 0, tzinfo=timezone.utc)


def test_live_entry_points_use_current_graph_context_dimension() -> None:
    root = Path(__file__).parents[1]
    for relative_path in ("src/app/web.py", "src/app/data/event_runtime.py"):
        source = (root / relative_path).read_text(encoding="utf-8")
        assert "feature_dim=STRATEGY_GRAPH_CONTEXT_DIM" in source
        assert "feature_dim=28" not in source


def _graph_context_values(**overrides: float) -> dict[str, float]:
    values = {f"graph:{name}": 0.0 for name in STRATEGY_GRAPH_CONTEXT_FIELDS}
    values.update(
        {
            "graph:spread_bps_scaled": 0.12,
            "graph:orderbook_imbalance": 0.2,
            "graph:return_1m_scaled": 0.4,
            "graph:realized_volatility_30m": 0.3,
            "graph:distance_from_vwap": 0.01,
            "graph:liquidity_score": 0.7,
            "graph:volume_spike_ratio": 1.4,
        }
    )
    values.update({f"graph:{name}": value for name, value in overrides.items()})
    return values


def _stub_frame(values: dict[str, float]) -> SimpleNamespace:
    return SimpleNamespace(
        symbol="SOFI",
        decision_time=NOW,
        mark_price=10.0,
        feature_schema_hash="schema",
        provenance=SimpleNamespace(
            orderbook_record_id="book-1",
            tick_record_ids=("tick-1",),
        ),
        validate=lambda: None,
        as_feature_dict=lambda: {},
        as_context_dict=lambda: values,
    )


def test_live_feature_frame_adapter_emits_the_aligned_context() -> None:
    snapshot = slow_snapshot_from_live_feature_frame(
        _stub_frame(_graph_context_values())
    )

    assert len(snapshot.features) == STRATEGY_GRAPH_CONTEXT_DIM
    assert snapshot.symbol == "SOFI"
    assert snapshot.feature_schema_name == STRATEGY_GRAPH_CONTEXT_SCHEMA
    assert snapshot.data_fresh is True
    assert snapshot.tradable is True
    named = as_context_mapping(snapshot.features)
    assert named["orderbook_imbalance"] == pytest.approx(0.2)
    assert named["return_1m_scaled"] == pytest.approx(0.4)


def test_snapshot_carries_the_reference_price_explicitly() -> None:
    """The cost engine used to read ``features[0] * 100_000`` as the price.

    v5 removed the price level from the context (it encodes instrument
    identity), so slot 0 is now ``microstructure_available`` — 0.0 or 1.0. Left
    alone, every cost estimate would have been anchored on 0 or 100,000 KRW
    instead of the mark.
    """
    from app.routing.shadow_intelligence import _reference_price

    snapshot = slow_snapshot_from_live_feature_frame(
        _stub_frame(_graph_context_values())
    )

    assert snapshot.reference_price == pytest.approx(10.0)
    assert _reference_price(snapshot) == pytest.approx(10.0)
    # Slot 0 is emphatically NOT a price any more.
    assert snapshot.features[0] in (0.0, 1.0)


def test_legacy_snapshot_still_derives_its_price_from_slot_zero() -> None:
    legacy = SlowIntelligenceSnapshot(
        snapshot_id="legacy",
        symbol="SOFI",
        as_of=NOW,
        valid_until=NOW + timedelta(seconds=5),
        feature_snapshot_id="legacy-features",
        features=(0.0001,) * 12,
        data_fresh=True,
        tradable=True,
        allowed_strategy_ids=STRATEGY_IDS,
        feature_schema_name="realtime_microstructure_v1",
    )
    from app.routing.shadow_intelligence import _reference_price

    assert _reference_price(legacy) == pytest.approx(10.0)


def test_unknown_book_does_not_score_as_a_perfect_spread() -> None:
    """``spread_bps_scaled == 0`` means "no book sample", not "zero spread".

    ``1 - 0/10 == 1.0`` would make every spread-dependent prior peak on exactly
    the minutes with no microstructure — about nine of ten KRX minutes in the
    current store.
    """
    from app.routing.shadow_intelligence import _strategy_compatibility

    known = as_context_mapping(
        build_strategy_graph_context(
            {
                name.removeprefix("graph:"): value
                for name, value in _graph_context_values(
                    microstructure_available=1.0
                ).items()
            }
        )
    )
    unknown = dict(known)
    unknown.update(
        microstructure_available=0.0,
        spread_bps_scaled=0.0,
        orderbook_imbalance=0.0,
        liquidity_score=0.0,
    )

    known_scores = _strategy_compatibility(
        build_strategy_graph_context(known)
    )
    unknown_scores = _strategy_compatibility(
        build_strategy_graph_context(unknown)
    )

    assert known_scores["intraday_momentum"] > 0.0
    assert known_scores["bar_confirmed_vwap_recovery"] > 0.0
    assert unknown_scores["intraday_momentum"] == 0.0
    assert unknown_scores["breakout_volume"] == 0.0
    assert unknown_scores["vwap_mean_reversion"] == 0.0
    assert unknown_scores["bar_confirmed_vwap_recovery"] == 0.0


def test_adapter_raises_when_the_frame_cannot_supply_the_contract() -> None:
    """The defect this replaces: a dropped column became a silent 0.0.

    ``box_high``, ``box_low``, ``box_mid``, ``box_previous_close`` and
    ``realized_volatility_3m`` were all read out of the live model's feature dict
    with a default after ``LIVE_FEATURE_NAMES`` stopped carrying them, so five of
    twenty-eight slots were served as zero against weights fitted on real values.
    An absent field must now stop the snapshot, not fill it in.
    """
    values = _graph_context_values()
    del values["graph:realized_volatility_30m"]
    del values["graph:box_high_ratio"]

    with pytest.raises(StrategyGraphContextError) as excinfo:
        slow_snapshot_from_live_feature_frame(_stub_frame(values))

    message = str(excinfo.value)
    assert "realized_volatility_30m" in message
    assert "box_high_ratio" in message


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


def test_unpromoted_matching_checkpoint_still_emits_validation_only_evidence(
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
            snapshot_id="snapshot-unpromoted",
            symbol="396500",
            as_of=NOW,
            valid_until=NOW + timedelta(seconds=5),
            feature_snapshot_id="features-unpromoted",
            features=(0.1,) * 12,
            data_fresh=True,
            tradable=True,
            allowed_strategy_ids=("intraday_momentum",),
            feature_schema_name="realtime_microstructure_v1",
        )
    )

    assert result is not None
    assert len(result.cpu_evidence) == len(STRATEGY_IDS)
    cpu = next(item for item in result.comparison.decisions if item.path == "cpu_gnn")
    assert cpu.action == "NO_TRADE"
    assert "GNN_CHECKPOINT_NOT_LIVE_AUTHORIZED" in cpu.reason_codes
    assert result.comparison.validation_candidates
    assert all(
        "GNN_CHECKPOINT_NOT_LIVE_AUTHORIZED" in item.reason_codes
        and "ORDER_PERMISSION_NOT_GRANTED" in item.reason_codes
        and "GNN_REALTIME_TRUST_PASSED" not in item.reason_codes
        for item in result.comparison.validation_candidates
    )


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

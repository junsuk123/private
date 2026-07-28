from __future__ import annotations

from datetime import datetime, timedelta, timezone

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

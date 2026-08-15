from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.trading.directional import StrategyDeploymentState
from app.trading.long_strategy_promotion import (
    LongPromotionConfig,
    deployment_size_cap,
    evaluate_long_promotion,
)
from app.trading.strategy_performance_store import StrategyPerformanceStore


NOW = datetime(2026, 8, 1, tzinfo=timezone.utc)


def _record(store, index, net, *, source="shadow"):
    store.record(
        strategy_id="supertrend_dmi_continuation",
        symbol="INTC",
        market="US",
        regime="TREND_UP",
        realized_net_bps=net,
        expected_net_bps=60.0,
        recorded_at=NOW + timedelta(days=index // 8, minutes=index),
        direction="LONG",
        execution_product="CASH",
        evaluation_source=source,
    )


def test_cold_long_strategy_remains_shadow(tmp_path):
    store = StrategyPerformanceStore(tmp_path / "perf.sqlite3", cache_ttl_seconds=0.0)
    decision = evaluate_long_promotion("supertrend_dmi_continuation", "US", store)
    assert decision.state is StrategyDeploymentState.SHADOW
    assert "LONG_PROMOTION_SAMPLE_INSUFFICIENT" in decision.reason_codes


def test_positive_stable_after_cost_shadow_evidence_auto_promotes_to_probe(tmp_path):
    store = StrategyPerformanceStore(tmp_path / "perf.sqlite3", cache_ttl_seconds=0.0)
    for index in range(30):
        _record(store, index, 70.0 + index % 3)
    decision = evaluate_long_promotion("supertrend_dmi_continuation", "US", store)
    assert decision.state is StrategyDeploymentState.LIVE_PROBE
    assert decision.positive_sample_count == 30
    assert deployment_size_cap(decision.state) == 0.10


def test_live_evidence_advances_ladder_without_manual_flag(tmp_path):
    store = StrategyPerformanceStore(tmp_path / "perf.sqlite3", cache_ttl_seconds=0.0)
    for index in range(70):
        source = "live_probe" if index < 35 else "live"
        store.record(
            strategy_id="supertrend_dmi_continuation",
            symbol="INTC", market="US", regime="TREND_UP",
            realized_net_bps=80.0 + index % 4,
            expected_net_bps=70.0,
            recorded_at=NOW + timedelta(days=index // 3, minutes=index),
            direction="LONG", execution_product="CASH", evaluation_source=source,
        )
    decision = evaluate_long_promotion("supertrend_dmi_continuation", "US", store)
    assert decision.state is StrategyDeploymentState.LIVE_FULL


def test_losses_demote_automatically_even_after_prior_positive_samples(tmp_path):
    store = StrategyPerformanceStore(tmp_path / "perf.sqlite3", cache_ttl_seconds=0.0)
    for index in range(30):
        _record(store, index, 70.0)
    for index in range(30, 33):
        _record(store, index, -150.0)
    decision = evaluate_long_promotion("supertrend_dmi_continuation", "US", store)
    assert decision.state is StrategyDeploymentState.SHADOW
    assert "LONG_PROMOTION_LOSS_STREAK" in decision.reason_codes

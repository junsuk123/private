from datetime import datetime, timedelta, timezone

import pytest

from app.trading.strategy_adaptation import StrategyAdaptation
from app.trading.strategy_performance_store import StrategyPerformanceStore

NOW = datetime(2026, 9, 20, 1, 0, tzinfo=timezone.utc)


def _store(tmp_path):
    return StrategyPerformanceStore(tmp_path / "performance.sqlite3", clock=lambda: NOW, cache_ttl_seconds=0)


def _record(store, *, count=16, net=-120.0, market="KR", regime="TREND_UP", source="shadow", end=NOW, **extra):
    for index in range(count):
        assert store.record(
            strategy_id="intraday_momentum", symbol="005930" if market == "KR" else "AAPL",
            market=market, regime=regime, realized_net_bps=net,
            expected_net_bps=100.0, realized_gross_bps=net + 45.0,
            holding_seconds=30.0, recorded_at=end - timedelta(minutes=2 * index),
            evaluation_source=source, **extra,
        )


def _assess(store, **kwargs):
    return StrategyAdaptation(store=store).assess(
        "intraday_momentum", market=kwargs.pop("market", "KR"),
        regime=kwargs.pop("regime", "TREND_UP"), now=kwargs.pop("now", NOW), **kwargs,
    )


def test_negative_shadow_net_does_not_become_actual_live_performance(tmp_path):
    store = _store(tmp_path)
    _record(store)
    result = _assess(store)
    assert result.performance_state == "SHADOW_ONLY"
    assert result.live_entry_allowed is False
    assert result.live.realized_net_bps is None
    assert result.live.sample_count == 0
    assert result.shadow.sample_count == 16
    assert result.shadow.upper_net_bps < 0
    assert result.shadow.prediction_bias_bps == -220.0
    assert result.shadow.round_trip_cost_bps == 45.0


def test_positive_gross_does_not_hide_negative_cash_after_costs(tmp_path):
    store = _store(tmp_path)
    _record(store, net=-35.0, count=40, source="live")
    result = _assess(store)
    assert result.live.realized_net_bps < 0
    assert result.live.upper_net_bps < 0
    assert result.live_entry_allowed is False


def test_market_and_regime_evidence_do_not_cross_boundaries(tmp_path):
    store = _store(tmp_path)
    _record(store, market="US", regime="TREND_DOWN", net=-500)
    assert _assess(store).performance_state == "COLD"
    assert _assess(store, market="US").performance_state == "COLD"
    assert _assess(store, market="US", regime="TREND_DOWN").live_entry_allowed is False


def test_future_outcomes_cannot_push_mature_evidence_out_of_replay_window(tmp_path):
    store = _store(tmp_path)
    _record(store, count=12, net=-200)
    _record(store, count=140, net=1000, end=NOW + timedelta(days=1))
    result = _assess(store)
    assert result.shadow.sample_count == 12
    assert result.shadow.realized_net_bps == pytest.approx(-200)
    assert result.live_entry_allowed is False


def test_old_semantics_remain_unusable_for_current_strategy(tmp_path):
    store = _store(tmp_path)
    _record(store, algorithm_version="old-version", evaluation_version="old-payoff")
    assert _assess(store).performance_state == "COLD"


def test_same_overlapping_move_does_not_create_sixteen_independent_wins(tmp_path):
    store = _store(tmp_path)
    for index in range(16):
        store.record(
            strategy_id="intraday_momentum", symbol="005930", market="KR", regime="TREND_UP",
            realized_net_bps=100, holding_seconds=300,
            recorded_at=NOW - timedelta(seconds=index), evaluation_source="shadow",
        )
    result = _assess(store)
    assert result.shadow.sample_count == 16
    assert result.shadow.independent_episode_count == 1
    assert result.shadow.effective_sample_count <= 1
    assert result.shadow.lower_net_bps < 0


def test_change_point_increases_uncertainty_without_inventing_positive_history(tmp_path):
    store = _store(tmp_path)
    _record(store)
    ordinary = _assess(store)
    changed = _assess(store, change_point_probability=1.0)
    assert ordinary.shadow.upper_net_bps < 0
    assert changed.shadow.effective_sample_count == 0
    assert changed.shadow.posterior_net_bps == 0
    assert changed.shadow.lower_net_bps < 0 < changed.shadow.upper_net_bps


def test_shadow_recovery_requires_fresh_evidence_and_existing_probe_authority(tmp_path):
    store = _store(tmp_path)
    _record(store, source="live", end=NOW - timedelta(hours=3))
    _record(store, source="shadow", net=200, end=NOW - timedelta(minutes=5))
    result = _assess(store, deployment_state="LIVE_FULL")
    assert result.performance_state == "RECOVERY_READY"
    assert result.live_entry_allowed is False
    probe = _assess(store, deployment_state="LIVE_PROBE")
    assert probe.performance_state == "RECOVERY_READY"
    assert probe.live_entry_allowed is True


def test_old_shadow_wins_cannot_restore_newer_losing_live_evidence(tmp_path):
    store = _store(tmp_path)
    _record(store, source="shadow", net=200, end=NOW - timedelta(hours=3))
    _record(store, source="live", end=NOW)
    result = _assess(store, deployment_state="LIVE_PROBE")
    assert result.performance_state == "SHADOW_ONLY"
    assert result.live_entry_allowed is False


def test_busy_shadow_journal_does_not_evict_live_loss_evidence(tmp_path):
    store = _store(tmp_path)
    _record(store, source="live", end=NOW - timedelta(hours=12))
    _record(store, source="shadow", count=140, net=100, end=NOW)
    result = _assess(store)
    assert result.live.sample_count == 16
    assert result.live.demonstrated_loss
    assert result.live_entry_allowed is False


def test_unreadable_performance_store_is_not_an_unseen_strategy(tmp_path):
    class BrokenStore:
        def recent_outcomes(self, *args, **kwargs):
            raise OSError("database unavailable")

    result = _assess(BrokenStore())
    assert result.performance_state == "UNAVAILABLE"
    assert result.live_entry_allowed is False


def test_point_in_time_replay_ages_against_replay_clock(tmp_path):
    store = _store(tmp_path)
    old = NOW - timedelta(days=100)
    _record(store, end=old)
    assert _assess(store).performance_state == "COLD"
    assert _assess(store, now=old).live_entry_allowed is False


def test_explicit_as_of_query_cache_does_not_leak_later_microsecond(tmp_path):
    store = _store(tmp_path)
    store.cache_ttl_seconds = 30
    _record(store, count=1, end=NOW + timedelta(microseconds=500))
    assert len(store.recent_outcomes("intraday_momentum", as_of=NOW + timedelta(microseconds=600))) == 1
    assert store.recent_outcomes("intraday_momentum", as_of=NOW) == ()


def test_legacy_barrier_wins_cannot_restore_new_policy_after_live_losses(tmp_path):
    store = _store(tmp_path)
    _record(store, source="live", end=NOW - timedelta(hours=3))
    _record(store, source="shadow", net=300)
    result = _assess(store, required_policy_family="ontology-risk-v1", deployment_state="LIVE_PROBE")
    assert result.performance_state == "SHADOW_ONLY"
    assert result.live.demonstrated_loss
    assert not result.live_entry_allowed
    assert "OLD_RISK_POLICY_LOSSES_RETAINED_GAINS_NOT_PROMOTABLE" in result.reason_codes
    _record(store, source="shadow", net=300, risk_policy_family="ontology-risk-v1")
    current = _assess(store, required_policy_family="ontology-risk-v1", deployment_state="LIVE_PROBE")
    assert current.performance_state == "RECOVERY_READY"
    assert current.live_entry_allowed

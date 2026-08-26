from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.trading.strategy_performance_store import (
    POSTERIOR_BELOW_MIN_SAMPLES,
    POSTERIOR_LOSS_STREAK,
    POSTERIOR_NO_SAMPLES,
    POSTERIOR_REGIME_FALLBACK,
    POSTERIOR_REGIME_HISTORY_DISCOUNTED,
    PosteriorConfig,
    StrategyPerformanceStore,
    market_for_symbol,
    normalize_market,
)


NOW = datetime(2026, 7, 28, 1, 0, tzinfo=timezone.utc)


def _store(tmp_path, **overrides) -> StrategyPerformanceStore:
    """A store whose clock is ``NOW``, so the 21-day window stays real but fixed.

    The evidence cutoff is measured against the store's clock. While that clock was
    the wall clock and ``NOW`` was a fixed past date, this file was a time bomb: it
    passed for 21 days and then, on NOW + 21 days, every posterior assertion began
    reading back an empty store while ``record`` still returned True. Pinning the
    store's clock to the same instant the fixtures are dated from removes the coupling
    without disabling the window — production's ``max_age_days`` stays in force here.
    """
    return StrategyPerformanceStore(
        tmp_path / "perf.sqlite3",
        posterior_config=PosteriorConfig(**overrides) if overrides else PosteriorConfig(),
        cache_ttl_seconds=0.0,
        clock=lambda: NOW,
    )


def _record(store, net_bps, *, index=0, strategy="intraday_momentum", regime="HIGH_VOL_TRENDING"):
    return store.record(
        strategy_id=strategy,
        symbol="005930",
        market="KR",
        regime=regime,
        realized_net_bps=net_bps,
        expected_net_bps=25.0,
        holding_seconds=300.0,
        recorded_at=NOW + timedelta(seconds=60 * index),
    )


def test_market_normalisation_matches_the_routing_rule():
    assert market_for_symbol("005930") == "KR"
    assert market_for_symbol("AAPL") == "US"
    assert normalize_market("NASD") == "US"
    assert normalize_market("KOSDAQ") == "KR"
    assert normalize_market("") == "UNKNOWN"


def test_recent_outcomes_are_newest_first(tmp_path):
    store = _store(tmp_path)
    for index, value in enumerate([10.0, -5.0, 30.0]):
        assert _record(store, value, index=index)
    assert store.recent_net_bps("intraday_momentum") == (30.0, -5.0, 10.0)


def test_old_algorithm_or_payoff_versions_are_preserved_but_not_promotable(tmp_path):
    store = _store(tmp_path)
    assert store.record(
        strategy_id="intraday_momentum",
        symbol="005930",
        market="KR",
        regime="HIGH_VOL_TRENDING",
        realized_net_bps=500.0,
        recorded_at=NOW,
        algorithm_version="spec-v1",
        evaluation_version="legacy-payoff",
    )
    assert store.recent_outcomes("intraday_momentum", market="KR") == ()
    historical = store.recent_outcomes(
        "intraday_momentum", market="KR", compatible_only=False
    )
    assert len(historical) == 1
    assert historical[0].algorithm_version == "spec-v1"
    assert historical[0].evaluation_version == "legacy-payoff"


def test_evidence_older_than_the_window_is_dropped(tmp_path):
    """The 21-day cutoff itself, asserted rather than assumed.

    Deterministic because the store's clock is injected: the boundary is measured
    from ``NOW``, not from whenever the suite happens to run.
    """
    store = _store(tmp_path, max_age_days=21.0)
    for label, moment in (
        (11.0, NOW - timedelta(days=30)),  # outside the window
        (22.0, NOW - timedelta(days=1)),  # inside it
    ):
        assert store.record(
            strategy_id="intraday_momentum",
            symbol="005930",
            market="KR",
            regime="HIGH_VOL_TRENDING",
            realized_net_bps=label,
            expected_net_bps=25.0,
            holding_seconds=300.0,
            recorded_at=moment,
        )
    # Both rows are stored; only the recent one counts as evidence.
    assert store.recent_net_bps("intraday_momentum") == (22.0,)


def test_the_store_clock_defaults_to_the_wall_clock(tmp_path):
    """The seam must not change production behaviour when nobody injects one."""
    store = StrategyPerformanceStore(
        tmp_path / "perf.sqlite3",
        posterior_config=PosteriorConfig(),
        cache_ttl_seconds=0.0,
    )
    drift = abs((store._now() - datetime.now(timezone.utc)).total_seconds())
    assert drift < 5.0


def test_loss_streak_counts_only_the_consecutive_tail(tmp_path):
    store = _store(tmp_path)
    for index, value in enumerate([50.0, -10.0, -20.0, -30.0]):
        _record(store, value, index=index)
    assert store.loss_streak("intraday_momentum") == 3
    assert store.had_recent_loss("intraday_momentum") is True

    _record(store, 5.0, index=9)
    assert store.loss_streak("intraday_momentum") == 0
    assert store.had_recent_loss("intraday_momentum") is False


def test_no_history_is_neutral_not_optimistic(tmp_path):
    store = _store(tmp_path)
    assert store.recent_performance_rate("intraday_momentum") == 0.0
    posterior = store.posterior("intraday_momentum", market="KR", regime="TREND_UP")
    assert posterior.sample_count == 0
    assert POSTERIOR_NO_SAMPLES in posterior.reason_codes
    # The prior is break-even and the penalty is large, so a strategy with no
    # evidence cannot present a positive lower bound.
    assert posterior.posterior_mean_net_bps == 0.0
    assert posterior.conservative_edge_bps < 0.0


def test_a_few_lucky_samples_do_not_produce_a_positive_lower_bound(tmp_path):
    """The whole point of selecting on the lower bound rather than the mean."""
    store = _store(tmp_path)
    for index, value in enumerate([120.0, 90.0, 150.0]):
        _record(store, value, index=index)
    posterior = store.posterior("intraday_momentum", market="KR", regime="HIGH_VOL_TRENDING")
    assert posterior.observed_mean_net_bps == 120.0
    assert posterior.sample_count == 3
    assert POSTERIOR_BELOW_MIN_SAMPLES in posterior.reason_codes
    # Shrunk toward the break-even prior, and still penalised for thin evidence.
    assert posterior.posterior_mean_net_bps < posterior.observed_mean_net_bps
    assert posterior.conservative_edge_bps < posterior.posterior_mean_net_bps


def test_consistent_edge_over_many_samples_earns_a_positive_lower_bound(tmp_path):
    store = _store(tmp_path)
    for index in range(40):
        _record(store, 60.0 + (index % 5), index=index)
    posterior = store.posterior("intraday_momentum", market="KR", regime="HIGH_VOL_TRENDING")
    assert posterior.sample_count == 40
    assert posterior.win_rate == 1.0
    assert posterior.conservative_edge_bps > 0.0


def test_loss_streak_widens_the_penalty(tmp_path):
    store = _store(tmp_path)
    for index in range(30):
        _record(store, 60.0, index=index)
    healthy = store.posterior("intraday_momentum", market="KR", regime="HIGH_VOL_TRENDING")
    for index in range(30, 34):
        _record(store, -40.0, index=index)
    bruised = store.posterior("intraday_momentum", market="KR", regime="HIGH_VOL_TRENDING")
    assert bruised.loss_streak == 4
    assert POSTERIOR_LOSS_STREAK in bruised.reason_codes
    assert bruised.uncertainty_penalty_bps > healthy.uncertainty_penalty_bps
    assert bruised.conservative_edge_bps < healthy.conservative_edge_bps


def test_change_point_probability_discounts_history_instead_of_deleting_it(tmp_path):
    store = _store(tmp_path)
    for index in range(40):
        _record(store, 60.0, index=index)
    calm = store.posterior("intraday_momentum", market="KR", regime="HIGH_VOL_TRENDING")
    broken = store.posterior(
        "intraday_momentum",
        market="KR",
        regime="HIGH_VOL_TRENDING",
        change_point_probability=0.9,
    )
    assert broken.sample_count == calm.sample_count  # nothing was deleted
    assert broken.effective_sample_count < calm.effective_sample_count
    assert POSTERIOR_REGIME_HISTORY_DISCOUNTED in broken.reason_codes
    assert broken.uncertainty_penalty_bps > calm.uncertainty_penalty_bps
    assert broken.conservative_edge_bps < calm.conservative_edge_bps


def test_regime_history_falls_back_to_market_wide_and_says_so(tmp_path):
    store = _store(tmp_path)
    for index in range(20):
        _record(store, 60.0, index=index, regime="HIGH_VOL_TRENDING")
    posterior = store.posterior(
        "intraday_momentum", market="KR", regime="HIGH_VOL_MEAN_REVERTING"
    )
    assert posterior.sample_count == 20
    assert POSTERIOR_REGIME_FALLBACK in posterior.reason_codes


def test_recent_performance_rate_is_a_rate_not_bps(tmp_path):
    store = _store(tmp_path)
    for index in range(5):
        _record(store, 100.0, index=index)
    assert store.recent_performance_rate("intraday_momentum", market="KR") == 0.01


def test_summary_and_prune_stay_bounded(tmp_path):
    store = _store(tmp_path)
    for index in range(20):
        _record(store, float(index), index=index)
    summary = store.summary()
    assert summary["strategies"][0]["sample_count"] == 20
    deleted = store.prune(keep_rows=100)
    assert deleted == 0


def test_unwritable_store_degrades_to_priors_instead_of_raising(tmp_path):
    # A directory where the database file should be makes sqlite unusable.
    blocked = tmp_path / "blocked.sqlite3"
    blocked.mkdir()
    store = StrategyPerformanceStore(blocked, cache_ttl_seconds=0.0)
    assert store.record(strategy_id="x", symbol="005930", realized_net_bps=10.0) is False
    assert store.recent_net_bps("x") == ()
    assert store.posterior("x").conservative_edge_bps < 0.0

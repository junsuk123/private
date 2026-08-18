from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.trading.conservative_bandit import (
    BANDIT_ARM_LOSS_STREAK_SUSPENDED,
    BANDIT_ARM_INSUFFICIENT_POSITIVE_SAMPLES,
    BANDIT_ARM_MEASURED_NEGATIVE_EDGE,
    BANDIT_ARM_SELECTED,
    BANDIT_CHANGE_POINT_STAND_DOWN,
    BANDIT_EXPLORATION_ARM_SELECTED,
    BANDIT_NO_CANDIDATE_ARMS,
    BANDIT_NO_POSITIVE_CONSERVATIVE_EDGE,
    ArmCandidate,
    BanditConfig,
    BanditContext,
    ConservativeStrategyBandit,
)
from app.trading.strategy_performance_store import (
    NO_TRADE_ARM,
    PosteriorConfig,
    StrategyPerformanceStore,
)


NOW = datetime(2026, 7, 28, 1, 0, tzinfo=timezone.utc)


def _store(tmp_path) -> StrategyPerformanceStore:
    # The store's clock is pinned to ``NOW`` so the 21-day evidence window is measured
    # from the same instant the fixtures are dated from. While that window ran off the
    # real wall clock these tests expired silently on NOW + 21 days: the store read
    # back empty and every arm looked unmeasured.
    return StrategyPerformanceStore(
        tmp_path / "perf.sqlite3",
        posterior_config=PosteriorConfig(),
        cache_ttl_seconds=0.0,
        clock=lambda: NOW,
    )


def _bandit(tmp_path, **overrides) -> ConservativeStrategyBandit:
    return ConservativeStrategyBandit(
        store=_store(tmp_path), config=BanditConfig(**overrides)
    )


def _context(**overrides) -> BanditContext:
    base = dict(
        market="KR",
        macro_regime="HIGH_VOL_TRENDING",
        change_point_probability=0.0,
        market_breadth=0.5,
        volatility_percentile=0.8,
        spread_percentile=0.4,
        liquidity_score=0.8,
    )
    base.update(overrides)
    return BanditContext(**base)


def _fill(store, strategy, values, *, regime="HIGH_VOL_TRENDING"):
    for index, value in enumerate(values):
        store.record(
            strategy_id=strategy,
            symbol="005930",
            market="KR",
            regime=regime,
            realized_net_bps=value,
            recorded_at=NOW + timedelta(seconds=60 * index),
        )


def test_no_candidates_means_no_trade(tmp_path):
    selection = _bandit(tmp_path).select((), _context(), now=NOW)
    assert selection.selected_arm == NO_TRADE_ARM
    assert selection.is_no_trade
    assert BANDIT_NO_CANDIDATE_ARMS in selection.reason_codes


def test_cold_arm_with_positive_predicted_edge_is_explored_not_exploited(tmp_path):
    bandit = _bandit(tmp_path)
    selection = bandit.select(
        (
            ArmCandidate(
                arm="intraday_momentum",
                symbol="005930",
                predicted_net_edge_bps=40.0,
                predicted_gross_edge_bps=68.0,
                expected_cost_bps=28.0,
            ),
        ),
        _context(),
        now=NOW,
    )
    assert not selection.is_no_trade
    assert selection.selected_arm == "intraday_momentum"
    assert selection.is_exploration
    assert BANDIT_EXPLORATION_ARM_SELECTED in selection.reason_codes


def test_live_execution_config_keeps_cold_arm_shadow_only(tmp_path):
    config = BanditConfig.for_live_execution()
    bandit = ConservativeStrategyBandit(store=_store(tmp_path), config=config)

    selection = bandit.select(
        (
            ArmCandidate(
                arm="intraday_momentum",
                symbol="005930",
                predicted_net_edge_bps=200.0,
                expected_cost_bps=28.0,
            ),
        ),
        _context(),
        now=NOW,
    )

    assert selection.is_no_trade
    reasons = selection.evaluations[0].reason_codes
    assert "BANDIT_ARM_COLD_START_SHADOW_ONLY" in reasons
    assert BANDIT_ARM_INSUFFICIENT_POSITIVE_SAMPLES in reasons


def test_live_execution_config_auto_unlocks_after_positive_net_evidence(tmp_path):
    store = _store(tmp_path)
    _fill(store, "intraday_momentum", [80.0 + index % 3 for index in range(30)])
    bandit = ConservativeStrategyBandit(
        store=store, config=BanditConfig.for_live_execution()
    )

    selection = bandit.select(
        (
            ArmCandidate(
                arm="intraday_momentum",
                symbol="005930",
                predicted_net_edge_bps=60.0,
                expected_cost_bps=28.0,
            ),
        ),
        _context(),
        now=NOW + timedelta(hours=1),
    )

    assert selection.selected_arm == "intraday_momentum"
    assert not selection.is_no_trade
    assert not selection.is_exploration


def test_measured_negative_arm_is_never_explored(tmp_path):
    """The failure mode this module exists to stop: retrying a losing strategy."""
    store = _store(tmp_path)
    _fill(store, "intraday_momentum", [-45.0] * 20)
    bandit = ConservativeStrategyBandit(store=store, config=BanditConfig())
    selection = bandit.select(
        (
            ArmCandidate(
                arm="intraday_momentum",
                symbol="005930",
                predicted_net_edge_bps=80.0,  # the model is still optimistic
                expected_cost_bps=28.0,
            ),
        ),
        _context(),
        now=NOW,
    )
    assert selection.is_no_trade
    assert BANDIT_NO_POSITIVE_CONSERVATIVE_EDGE in selection.reason_codes
    reasons = selection.evaluations[0].reason_codes
    assert BANDIT_ARM_LOSS_STREAK_SUSPENDED in reasons or (
        BANDIT_ARM_MEASURED_NEGATIVE_EDGE in reasons
    )


def test_all_negative_expectancy_arms_produce_no_trade(tmp_path):
    store = _store(tmp_path)
    for strategy in ("intraday_momentum", "breakout_volume", "vwap_mean_reversion"):
        _fill(store, strategy, [-50.0, -40.0, -45.0, -60.0, -30.0] * 4)
    bandit = ConservativeStrategyBandit(store=store, config=BanditConfig())
    selection = bandit.select(
        tuple(
            ArmCandidate(arm=strategy, symbol="005930", predicted_net_edge_bps=30.0, expected_cost_bps=28.0)
            for strategy in ("intraday_momentum", "breakout_volume", "vwap_mean_reversion")
        ),
        _context(),
        now=NOW,
    )
    # This is the headline requirement: when every measured strategy loses money,
    # the answer is nothing, not the least-bad loser.
    assert selection.is_no_trade
    assert selection.selected_arm == NO_TRADE_ARM


def test_demonstrated_edge_outranks_cold_optimism(tmp_path):
    store = _store(tmp_path)
    _fill(store, "vwap_mean_reversion", [70.0 + (i % 4) for i in range(40)])
    bandit = ConservativeStrategyBandit(store=store, config=BanditConfig())
    selection = bandit.select(
        (
            ArmCandidate(
                arm="breakout_volume",
                symbol="000660",
                predicted_net_edge_bps=500.0,  # wildly optimistic, no history
                expected_cost_bps=28.0,
            ),
            ArmCandidate(
                arm="vwap_mean_reversion",
                symbol="005930",
                predicted_net_edge_bps=40.0,
                expected_cost_bps=28.0,
            ),
        ),
        _context(),
        now=NOW,
    )
    assert selection.selected_arm == "vwap_mean_reversion"
    assert not selection.is_exploration
    assert BANDIT_ARM_SELECTED in selection.reason_codes


def test_detected_change_point_stands_down_including_exploration(tmp_path):
    bandit = _bandit(tmp_path)
    selection = bandit.select(
        (
            ArmCandidate(
                arm="intraday_momentum",
                symbol="005930",
                predicted_net_edge_bps=200.0,
                expected_cost_bps=28.0,
            ),
        ),
        _context(change_point_probability=0.8),
        now=NOW,
    )
    assert selection.is_no_trade
    assert BANDIT_CHANGE_POINT_STAND_DOWN in selection.reason_codes


def test_macro_blocked_arm_cannot_be_selected(tmp_path):
    bandit = _bandit(tmp_path)
    selection = bandit.select(
        (
            ArmCandidate(
                arm="intraday_momentum",
                symbol="005930",
                predicted_net_edge_bps=200.0,
                expected_cost_bps=28.0,
                macro_permitted=False,
            ),
        ),
        _context(),
        now=NOW,
    )
    assert selection.is_no_trade


def test_shadow_only_arm_is_reported_but_not_selected(tmp_path):
    bandit = _bandit(tmp_path)
    selection = bandit.select(
        (
            ArmCandidate(
                arm="residual_relative_strength",
                symbol="005930",
                predicted_net_edge_bps=200.0,
                expected_cost_bps=28.0,
                live_authorized=False,
            ),
        ),
        _context(),
        now=NOW,
    )
    assert selection.is_no_trade
    assert "residual_relative_strength" in selection.shadow_arms


def test_dislocated_context_charges_an_explicit_penalty(tmp_path):
    bandit = _bandit(tmp_path)
    calm = bandit.select(
        (ArmCandidate(arm="intraday_momentum", symbol="005930", predicted_net_edge_bps=40.0, expected_cost_bps=28.0),),
        _context(),
        now=NOW,
    )
    wide = bandit.select(
        (ArmCandidate(arm="intraday_momentum", symbol="005930", predicted_net_edge_bps=40.0, expected_cost_bps=28.0),),
        _context(spread_percentile=0.97, liquidity_score=0.2),
        now=NOW,
    )
    assert wide.evaluations[0].context_penalty_bps > calm.evaluations[0].context_penalty_bps
    assert wide.evaluations[0].conservative_edge_bps < calm.evaluations[0].conservative_edge_bps


def test_unreadable_context_is_charged_for(tmp_path):
    bandit = _bandit(tmp_path)
    blind = bandit.select(
        (ArmCandidate(arm="intraday_momentum", symbol="005930", predicted_net_edge_bps=40.0, expected_cost_bps=28.0),),
        _context(market_breadth=None, volatility_percentile=None),
        now=NOW,
    )
    assert blind.evaluations[0].context_penalty_bps > 0.0

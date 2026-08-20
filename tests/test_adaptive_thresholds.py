"""The safety properties of the adaptive threshold layer.

These are not "does the code run" tests. Each one pins an invariant that makes
the layer safe to run against a funded account, and each is a property whose
violation would be invisible in ordinary operation until it cost money.
"""
from __future__ import annotations

import math

import pytest

from app.technical.adaptive_thresholds import (
    AdaptiveConfig,
    AdaptiveThresholds,
    AdaptiveThresholdStore,
    resolve_market,
    default_edge_calibrator,
    reset_default_adaptive_thresholds,
)


@pytest.fixture()
def adaptive(tmp_path):
    return AdaptiveThresholds(
        store=AdaptiveThresholdStore(tmp_path / "adaptive.sqlite3"),
        config=AdaptiveConfig(),
    )


def test_cold_arm_is_exactly_the_configured_policy(adaptive):
    """No evidence must mean no change, or a fresh deployment is a new strategy."""
    multiple, diagnostics = adaptive.cost_multiple("liquidity_shock_reversal", "US")
    assert multiple == 1.0
    assert diagnostics["cost_multiple_basis"] == "insufficient_evidence"


def test_default_calibrator_can_initialize_before_threshold_singleton(
    tmp_path, monkeypatch
):
    """Factory order must not deadlock the live strategy loop."""
    monkeypatch.setenv("ADAPTIVE_THRESHOLDS_STORE_PATH", str(tmp_path / "factory.sqlite3"))
    reset_default_adaptive_thresholds()
    try:
        calibrator = default_edge_calibrator()
        edge, diagnostics = calibrator.calibrate("intraday_momentum", "KR", 50.0)
        assert edge == 50.0
        assert diagnostics["edge_calibration"] == "insufficient_evidence"
    finally:
        reset_default_adaptive_thresholds()


def test_never_looser_than_static_for_a_negative_threshold(adaptive):
    """A shock bound is negative: stricter is MORE negative.

    A quiet tape scales the magnitude down, and taking that value would fire the
    trigger on smaller dislocations than the configuration allows -- in precisely
    the regime where the measured reversion is smallest.
    """
    for _ in range(50):
        adaptive.observe_scale("liquidity_shock_reversal", "US", 20.0)
    quiet, _ = adaptive.adapt_threshold(
        "liquidity_shock_reversal", "US",
        static_value=-40.0, scale_bps=5.0, stricter_is_larger=False,
    )
    assert quiet <= -40.0


def test_never_looser_than_static_for_a_positive_threshold(adaptive):
    for _ in range(50):
        adaptive.observe_scale("vwap_mean_reversion", "US", 20.0)
    quiet, _ = adaptive.adapt_threshold(
        "vwap_mean_reversion", "US",
        static_value=0.55, scale_bps=5.0, stricter_is_larger=True,
    )
    assert quiet >= 0.55


def test_a_wild_tape_makes_the_threshold_stricter(adaptive):
    for _ in range(50):
        adaptive.observe_scale("liquidity_shock_reversal", "US", 20.0)
    wild, diagnostics = adaptive.adapt_threshold(
        "liquidity_shock_reversal", "US",
        static_value=-40.0, scale_bps=60.0, stricter_is_larger=False,
    )
    assert wild < -40.0
    assert diagnostics["scale_multiplier"] > 1.0


def test_losses_raise_strictness_and_stop_at_the_cap(adaptive):
    """Unbounded tightening would be an absorbing state by another name."""
    for _ in range(400):
        adaptive.record_outcome("liquidity_shock_reversal", "US", realized_net_bps=-110.0)
    multiple, _ = adaptive.cost_multiple("liquidity_shock_reversal", "US")
    assert multiple == adaptive.config.maximum_multiple


def test_wins_relax_but_never_below_the_cost_floor(adaptive):
    """1.0 is arithmetic: an edge under its own round trip is a loss."""
    for _ in range(200):
        adaptive.record_outcome("liquidity_shock_reversal", "US", realized_net_bps=-110.0)
    for _ in range(500):
        adaptive.record_outcome("liquidity_shock_reversal", "US", realized_net_bps=+80.0)
    multiple, _ = adaptive.cost_multiple("liquidity_shock_reversal", "US")
    assert multiple == 1.0


def test_inadmissible_outcomes_never_move_the_controller(adaptive):
    """The rejected region is measurable, but it is not evidence about this arm.

    Treating it as such is the defect that let refused plans set the bandit's
    posterior; the same mistake must not be reintroduced one layer down.
    """
    for _ in range(300):
        adaptive.record_outcome("x", "US", realized_net_bps=-150.0, admissible=False)
    multiple, _ = adaptive.cost_multiple("x", "US")
    assert multiple == 1.0


def test_learning_survives_a_restart(tmp_path):
    path = tmp_path / "adaptive.sqlite3"
    first = AdaptiveThresholds(store=AdaptiveThresholdStore(path))
    for _ in range(100):
        first.record_outcome("y", "US", realized_net_bps=-110.0)
    before, _ = first.cost_multiple("y", "US")
    second = AdaptiveThresholds(store=AdaptiveThresholdStore(path))
    after, _ = second.cost_multiple("y", "US")
    assert before == after > 1.0


def test_disabled_config_is_a_no_op(tmp_path):
    off = AdaptiveThresholds(
        store=AdaptiveThresholdStore(tmp_path / "a.sqlite3"),
        config=AdaptiveConfig(enabled=False),
    )
    for _ in range(200):
        off.record_outcome("z", "US", realized_net_bps=-110.0)
    multiple, _ = off.cost_multiple("z", "US")
    assert multiple == 1.0
    value, _ = off.adapt_threshold(
        "z", "US", static_value=-40.0, scale_bps=999.0, stricter_is_larger=False
    )
    assert value == -40.0


def test_missing_or_absurd_scale_degrades_to_the_static_value(adaptive):
    for scale in (None, 0.0, -5.0, float("nan")):
        value, _ = adaptive.adapt_threshold(
            "liquidity_shock_reversal", "US",
            static_value=-40.0, scale_bps=scale, stricter_is_larger=False,
        )
        assert value == -40.0


def test_non_finite_outcome_is_ignored(adaptive):
    adaptive.record_outcome("q", "US", realized_net_bps=float("nan"))
    state = adaptive.state("q", "US")
    assert state.sample_count == 0


def test_resolve_market_matches_the_cost_floor_rule():
    assert resolve_market("005930") == "KR"
    assert resolve_market("INTC") == "US"


# --------------------------------------------------------------------------- #
# Edge calibration                                                             #
# --------------------------------------------------------------------------- #
from app.technical.adaptive_thresholds import EdgeCalibrator, edge_bucket  # noqa: E402


@pytest.fixture()
def calibrator(tmp_path):
    return EdgeCalibrator(AdaptiveThresholdStore(tmp_path / "cal.sqlite3"))


def test_cold_calibrator_returns_the_rule_prediction_unchanged(calibrator):
    value, diagnostics = calibrator.calibrate("liquidity_shock_reversal", "US", 16.0)
    assert value == 16.0
    assert diagnostics["edge_calibration"] == "insufficient_evidence"


def test_calibration_corrects_a_systematically_optimistic_rule(calibrator):
    """The measured case: claimed +16bps gross, paid -32bps."""
    for _ in range(40):
        calibrator.record(
            "liquidity_shock_reversal", "US",
            predicted_edge_bps=16.0, realized_gross_bps=-32.0,
        )
    value, diagnostics = calibrator.calibrate("liquidity_shock_reversal", "US", 16.0)
    assert value < 0.0
    assert diagnostics["measured_mean_gross_bps"] == pytest.approx(-32.0, abs=0.1)


def test_calibration_never_raises_an_edge_above_the_rule_claim(calibrator):
    """Thin lucky evidence must not manufacture optimism through the cost floor."""
    for _ in range(40):
        calibrator.record("x", "US", predicted_edge_bps=20.0, realized_gross_bps=500.0)
    value, _ = calibrator.calibrate("x", "US", 20.0)
    assert value <= 20.0


def test_calibration_is_per_bucket_not_one_global_offset(calibrator):
    """A rule right about small moves and wrong about large ones needs both."""
    for _ in range(40):
        calibrator.record("y", "US", predicted_edge_bps=10.0, realized_gross_bps=9.0)
        calibrator.record("y", "US", predicted_edge_bps=300.0, realized_gross_bps=-100.0)
    small, _ = calibrator.calibrate("y", "US", 10.0)
    large, _ = calibrator.calibrate("y", "US", 300.0)
    assert small == pytest.approx(10.0, abs=2.0)
    assert large < 0.0


def test_markets_are_calibrated_independently(calibrator):
    for _ in range(40):
        calibrator.record("z", "US", predicted_edge_bps=50.0, realized_gross_bps=-80.0)
    us, _ = calibrator.calibrate("z", "US", 50.0)
    kr, _ = calibrator.calibrate("z", "KR", 50.0)
    assert us < 0.0
    assert kr == 50.0


def test_non_finite_observations_are_ignored(calibrator):
    calibrator.record("q", "US", predicted_edge_bps=float("nan"), realized_gross_bps=1.0)
    calibrator.record("q", "US", predicted_edge_bps=1.0, realized_gross_bps=float("inf"))
    value, diagnostics = calibrator.calibrate("q", "US", 1.0)
    assert diagnostics["edge_calibration"] == "insufficient_evidence"


def test_calibration_survives_a_restart(tmp_path):
    store = AdaptiveThresholdStore(tmp_path / "cal.sqlite3")
    first = EdgeCalibrator(store)
    for _ in range(40):
        first.record("r", "US", predicted_edge_bps=16.0, realized_gross_bps=-32.0)
    before, _ = first.calibrate("r", "US", 16.0)
    second = EdgeCalibrator(AdaptiveThresholdStore(tmp_path / "cal.sqlite3"))
    after, _ = second.calibrate("r", "US", 16.0)
    assert before == pytest.approx(after)


def test_edge_buckets_cover_every_magnitude():
    for value in (0.0, 24.9, 25.0, 99.9, 100.0, 199.9, 200.0, 10_000.0, -300.0):
        assert edge_bucket(value)

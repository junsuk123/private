from __future__ import annotations

from app.cost.cost_coverage import (
    REASON_COST_COVERAGE_UNKNOWN,
    REASON_COST_NOT_COVERED,
    CostCoverageBand,
    CostCoverageThresholds,
    cost_coverage_ratio,
    evaluate_cost_coverage,
)


LIMITS = CostCoverageThresholds(covered=1.0, live=1.3, comfortable=1.7)


def test_ratio_is_undefined_without_a_cost_estimate():
    # A missing/zero cost is NOT infinite coverage; it is an unknown.
    assert cost_coverage_ratio(50.0, None) is None
    assert cost_coverage_ratio(50.0, 0.0) is None
    assert cost_coverage_ratio(None, 28.0) is None
    assessment = evaluate_cost_coverage(50.0, 0.0, thresholds=LIMITS)
    assert assessment.band is CostCoverageBand.UNKNOWN
    assert REASON_COST_COVERAGE_UNKNOWN in assessment.reason_codes
    assert not assessment.live_eligible


def test_bands_match_the_documented_policy():
    assert evaluate_cost_coverage(20.0, 28.0, thresholds=LIMITS).band is CostCoverageBand.NOT_COVERED
    assert evaluate_cost_coverage(32.0, 28.0, thresholds=LIMITS).band is CostCoverageBand.INSUFFICIENT
    assert evaluate_cost_coverage(42.0, 28.0, thresholds=LIMITS).band is CostCoverageBand.THIN
    assert evaluate_cost_coverage(56.0, 28.0, thresholds=LIMITS).band is CostCoverageBand.SUFFICIENT


def test_edge_merely_equal_to_cost_is_not_tradeable():
    assessment = evaluate_cost_coverage(28.0, 28.0, thresholds=LIMITS)
    assert assessment.ratio == 1.0
    assert assessment.band is CostCoverageBand.INSUFFICIENT
    assert not assessment.live_eligible


def test_thin_band_is_shadow_or_minimum_size_only():
    assessment = evaluate_cost_coverage(42.0, 28.0, thresholds=LIMITS)
    assert assessment.live_eligible
    assert assessment.shadow_only
    assert not assessment.full_size_eligible


def test_not_covered_is_reported_explicitly():
    assessment = evaluate_cost_coverage(10.0, 28.0, thresholds=LIMITS)
    assert REASON_COST_NOT_COVERED in assessment.reason_codes
    assert not assessment.live_eligible


def test_misordered_thresholds_cannot_invert_the_bands(monkeypatch):
    monkeypatch.setenv("COST_COVERAGE_COVERED_RATIO", "2.0")
    monkeypatch.setenv("COST_COVERAGE_LIVE_RATIO", "1.0")
    monkeypatch.setenv("COST_COVERAGE_COMFORTABLE_RATIO", "0.5")
    limits = CostCoverageThresholds.from_env()
    assert limits.covered <= limits.live <= limits.comfortable

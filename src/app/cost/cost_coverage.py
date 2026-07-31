"""Cost coverage ratio — does the predicted edge actually clear its own cost?

    CostCoverageRatio = predicted_gross_edge_bps / expected_all_in_cost_bps

The existing ``max_cost_to_alpha_ratio`` rule in :mod:`app.cost.profitability_gate`
answers the same question from the other side, but with a live value of 1.05 it
allows a trade whose cost *exceeds* its alpha. That leaves no margin for the
estimation error in either term, and the measured outcome of that setting was a
gross-positive / net-negative population.

This module states the rule the way it should be read, with explicit bands:

    < 1.0    cost is not even covered            -> never
    1.0-1.3  covered but inside the error bars   -> do not trade
    1.3-1.7  real but thin                       -> shadow, or minimum size
    >= 1.7   worth taking                        -> live candidate

The bands are policy, not law: they come from ``config/profitability_policy.yaml``
(``cost_coverage``) or the matching environment variables. The built-in defaults
match the analysis that motivated them.

Everything here is pure arithmetic on two numbers, so it is safe to call from the
tick loop, the trainer, and the dashboard alike.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

REASON_COST_NOT_COVERED = "COST_NOT_COVERED_BY_EDGE"
REASON_COST_COVERAGE_INSUFFICIENT = "COST_COVERAGE_INSUFFICIENT"
REASON_COST_COVERAGE_THIN = "COST_COVERAGE_THIN_SHADOW_OR_MINIMUM"
REASON_COST_COVERAGE_OK = "COST_COVERAGE_SUFFICIENT"
REASON_COST_COVERAGE_UNKNOWN = "COST_COVERAGE_UNKNOWN"


class CostCoverageBand(str, Enum):
    UNKNOWN = "UNKNOWN"
    NOT_COVERED = "NOT_COVERED"
    INSUFFICIENT = "INSUFFICIENT"
    THIN = "THIN"
    SUFFICIENT = "SUFFICIENT"


@dataclass(frozen=True)
class CostCoverageThresholds:
    """Band edges. ``live`` is the only one a gate should enforce."""

    covered: float = 1.0
    live: float = 1.3
    comfortable: float = 1.7

    @classmethod
    def from_env(cls, overrides: dict[str, Any] | None = None) -> "CostCoverageThresholds":
        raw = dict(overrides or {})

        def _value(key: str, env_name: str, default: float) -> float:
            candidate = raw.get(key, default)
            try:
                candidate = float(candidate)
            except (TypeError, ValueError):
                candidate = default
            env_raw = os.getenv(env_name)
            if env_raw not in (None, ""):
                try:
                    candidate = float(env_raw)
                except (TypeError, ValueError):
                    pass
            return candidate

        covered = _value("covered", "COST_COVERAGE_COVERED_RATIO", cls.covered)
        live = _value("live", "COST_COVERAGE_LIVE_RATIO", cls.live)
        comfortable = _value("comfortable", "COST_COVERAGE_COMFORTABLE_RATIO", cls.comfortable)
        # A mis-ordered configuration must not silently invert the bands.
        covered = max(0.0, covered)
        live = max(covered, live)
        comfortable = max(live, comfortable)
        return cls(covered=covered, live=live, comfortable=comfortable)


@dataclass(frozen=True)
class CostCoverageAssessment:
    ratio: float | None
    band: CostCoverageBand
    predicted_gross_edge_bps: float
    expected_all_in_cost_bps: float
    thresholds: CostCoverageThresholds
    reason_codes: tuple[str, ...]

    @property
    def live_eligible(self) -> bool:
        return self.band in (CostCoverageBand.THIN, CostCoverageBand.SUFFICIENT)

    @property
    def full_size_eligible(self) -> bool:
        return self.band is CostCoverageBand.SUFFICIENT

    @property
    def shadow_only(self) -> bool:
        return self.band is CostCoverageBand.THIN

    def as_dict(self) -> dict[str, Any]:
        return {
            "cost_coverage_ratio": None if self.ratio is None else round(self.ratio, 4),
            "band": self.band.value,
            "predicted_gross_edge_bps": round(self.predicted_gross_edge_bps, 3),
            "expected_all_in_cost_bps": round(self.expected_all_in_cost_bps, 3),
            "live_eligible": self.live_eligible,
            "full_size_eligible": self.full_size_eligible,
            "shadow_only": self.shadow_only,
            "thresholds": {
                "covered": self.thresholds.covered,
                "live": self.thresholds.live,
                "comfortable": self.thresholds.comfortable,
            },
            "reason_codes": list(self.reason_codes),
        }


def cost_coverage_ratio(
    predicted_gross_edge_bps: float | None,
    expected_all_in_cost_bps: float | None,
) -> float | None:
    """``None`` when the ratio is undefined — never a fabricated 0.0 or inf.

    A zero (or unknown) cost estimate does NOT mean infinite coverage; it means
    the cost model had nothing to say, which is an unknown.
    """
    edge = _finite(predicted_gross_edge_bps)
    cost = _finite(expected_all_in_cost_bps)
    if edge is None or cost is None or cost <= 0.0:
        return None
    return edge / cost


def evaluate_cost_coverage(
    predicted_gross_edge_bps: float | None,
    expected_all_in_cost_bps: float | None,
    *,
    thresholds: CostCoverageThresholds | None = None,
) -> CostCoverageAssessment:
    limits = thresholds or CostCoverageThresholds.from_env()
    ratio = cost_coverage_ratio(predicted_gross_edge_bps, expected_all_in_cost_bps)
    edge = _finite(predicted_gross_edge_bps) or 0.0
    cost = _finite(expected_all_in_cost_bps) or 0.0
    if ratio is None:
        return CostCoverageAssessment(
            ratio=None,
            band=CostCoverageBand.UNKNOWN,
            predicted_gross_edge_bps=edge,
            expected_all_in_cost_bps=cost,
            thresholds=limits,
            reason_codes=(REASON_COST_COVERAGE_UNKNOWN,),
        )
    if ratio < limits.covered:
        band, reason = CostCoverageBand.NOT_COVERED, REASON_COST_NOT_COVERED
    elif ratio < limits.live:
        band, reason = CostCoverageBand.INSUFFICIENT, REASON_COST_COVERAGE_INSUFFICIENT
    elif ratio < limits.comfortable:
        band, reason = CostCoverageBand.THIN, REASON_COST_COVERAGE_THIN
    else:
        band, reason = CostCoverageBand.SUFFICIENT, REASON_COST_COVERAGE_OK
    return CostCoverageAssessment(
        ratio=ratio,
        band=band,
        predicted_gross_edge_bps=edge,
        expected_all_in_cost_bps=cost,
        thresholds=limits,
        reason_codes=(reason,),
    )


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

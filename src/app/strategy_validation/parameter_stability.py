"""Is the result a property of the thesis, or of one threshold?

The measurement that motivates this
----------------------------------
``range_support_reversion`` was screened at a 10bps distance-from-floor threshold and scored
t=3.01; the 25bps variant scored t=2.91. Those are close, which is reassuring. But adding a
confirmation filter moved the result from -2.3bps to -35.8bps and cut the sample from 207 to
38 — a knob whose small change destroys the result. Distinguishing the two cases is exactly
what parameter stability measures, and the catalogue's own comments show it matters.

Method
------
For each parameter, evaluate the metric over neighbouring values and report:

``sensitivity``   the spread of the metric across the neighbourhood, normalised by the
                  metric at the configured value. Large means the configured value is a
                  cliff edge.
``plateau``       whether the configured value sits inside a run of same-sign results. A
                  parameter on a plateau is a thesis; one on a spike is a fit.

No optimisation happens here, deliberately. Reporting "the best value is X" invites moving
the knob to X, which is curve fitting with extra steps. The output is a stability claim
about the value that is actually configured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "ParameterStabilityResult",
    "ParameterSweep",
    "parameter_stability",
]


@dataclass(frozen=True)
class ParameterSweep:
    """One parameter's neighbourhood and the metric measured at each value."""

    name: str
    configured_value: float
    values: tuple[float, ...]
    metrics: tuple[float | None, ...]

    @property
    def configured_metric(self) -> float | None:
        for value, metric in zip(self.values, self.metrics, strict=True):
            if math.isclose(value, self.configured_value, rel_tol=1e-9, abs_tol=1e-12):
                return metric
        return None

    @property
    def measured(self) -> tuple[float, ...]:
        return tuple(item for item in self.metrics if item is not None)

    @property
    def sensitivity(self) -> float | None:
        """Spread across the neighbourhood over |metric at the configured value|.

        ``None`` when fewer than two values could be measured, or when the configured
        metric is ~0 (dividing by it would report an arbitrary magnitude rather than a
        sensitivity).
        """
        measured = self.measured
        if len(measured) < 2:
            return None
        configured = self.configured_metric
        if configured is None or abs(configured) < 1e-9:
            return None
        return (max(measured) - min(measured)) / abs(configured)

    @property
    def sign_stable(self) -> bool | None:
        """Do all measured neighbours share the sign of the configured value's metric?"""
        measured = self.measured
        configured = self.configured_metric
        if not measured or configured is None:
            return None
        target = configured > 0
        return all((value > 0) == target for value in measured)

    @property
    def on_plateau(self) -> bool | None:
        """Is the configured value NOT a local spike?

        A spike is a configured metric that is more than twice the best of its immediate
        neighbours — the shape of a value that was chosen because it happened to work.
        """
        configured = self.configured_metric
        if configured is None:
            return None
        neighbours = [
            metric
            for value, metric in zip(self.values, self.metrics, strict=True)
            if metric is not None
            and not math.isclose(value, self.configured_value, rel_tol=1e-9, abs_tol=1e-12)
        ]
        if not neighbours:
            return None
        best_neighbour = max(neighbours)
        if configured <= 0:
            return True  # a non-positive configured metric is not a flattering spike
        return configured <= 2.0 * max(best_neighbour, 1e-9)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "configured_value": self.configured_value,
            "configured_metric": _round(self.configured_metric),
            "values": list(self.values),
            "metrics": [_round(item) for item in self.metrics],
            "sensitivity": _round(self.sensitivity, 4),
            "sign_stable": self.sign_stable,
            "on_plateau": self.on_plateau,
        }


@dataclass(frozen=True)
class ParameterStabilityResult:
    strategy_id: str
    sweeps: tuple[ParameterSweep, ...]
    reason_codes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def fragile_parameters(self) -> tuple[str, ...]:
        """Parameters whose neighbourhood changes the sign or spikes at the configured value."""
        return tuple(
            sweep.name
            for sweep in self.sweeps
            if sweep.sign_stable is False or sweep.on_plateau is False
        )

    @property
    def max_sensitivity(self) -> float | None:
        values = [sweep.sensitivity for sweep in self.sweeps if sweep.sensitivity is not None]
        return max(values) if values else None

    @property
    def stable(self) -> bool | None:
        if not self.sweeps:
            return None
        return not self.fragile_parameters

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "stable": self.stable,
            "fragile_parameters": list(self.fragile_parameters),
            "max_sensitivity": _round(self.max_sensitivity, 4),
            "sweeps": [sweep.as_dict() for sweep in self.sweeps],
            "reason_codes": list(self.reason_codes),
        }


def parameter_stability(
    strategy_id: str,
    *,
    parameters: Mapping[str, float],
    evaluate: Callable[[Mapping[str, float]], float | None],
    relative_steps: Sequence[float] = (-0.25, -0.1, 0.1, 0.25),
    parameter_names: Sequence[str] | None = None,
) -> ParameterStabilityResult:
    """Sweep each parameter one at a time and report stability.

    ``evaluate`` receives the full parameter mapping with ONE value replaced and returns the
    metric (net EV in bps, typically) or ``None`` when it cannot be computed for those
    values. One-at-a-time rather than a grid: a full grid over 15 knobs is not runnable, and
    the question here is "is this value a cliff", which is local by nature.
    """
    names = tuple(parameter_names) if parameter_names else tuple(sorted(parameters))
    reasons: list[str] = []
    sweeps: list[ParameterSweep] = []

    for name in names:
        base = parameters.get(name)
        if base is None or not math.isfinite(float(base)) or float(base) == 0.0:
            # A zero or absent base has no meaningful relative neighbourhood. Skipping is
            # reported rather than silently producing an empty sweep.
            reasons.append(f"PARAMETER_NOT_SWEEPABLE:{name}")
            continue
        base = float(base)
        values = [base]
        for step in relative_steps:
            candidate = base * (1.0 + float(step))
            if math.isfinite(candidate) and candidate not in values:
                values.append(candidate)
        values.sort()

        metrics: list[float | None] = []
        for value in values:
            trial = dict(parameters)
            trial[name] = value
            try:
                metric = evaluate(trial)
            except Exception:  # noqa: BLE001 - an unevaluable point is unknown, not zero.
                metric = None
            metrics.append(
                float(metric) if metric is not None and math.isfinite(float(metric)) else None
            )
        sweeps.append(
            ParameterSweep(
                name=name,
                configured_value=base,
                values=tuple(values),
                metrics=tuple(metrics),
            )
        )
    if not sweeps:
        reasons.append("PARAMETER_STABILITY_NO_SWEEPS")
    return ParameterStabilityResult(
        strategy_id=strategy_id,
        sweeps=tuple(sweeps),
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def _round(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None

"""Cross-market relationships: relative strength, rolling beta and lead/lag.

The claim this module makes possible
------------------------------------
"US semis fell 3% and this Korean semi name fell 0.4%" is a different observation from
"this name fell 0.4%", and only the first is tradeable. Relative strength is defined as::

    RS = R_local - beta * R_reference

with ``beta`` estimated over a rolling window rather than assumed to be 1. Assuming 1 is
what makes a low-beta defensive name look permanently strong in a selloff and a
high-beta semi look permanently weak, which is a beta exposure dressed up as stock
selection.

What it is NOT allowed to do
----------------------------
Turn a foreign move into a domestic order. ``RS`` is an input; the requirement that a
weak US tape must be confirmed by domestic breadth, flow and relative strength before it
can reduce a domestic position lives in the domestic context and the strategy selector.
This module only measures.

Lead/lag
--------
:func:`estimate_lead_lag` reports which of two series moves first, by cross-correlating
returns over a bounded lag range and returning the best lag with its correlation. It
reports the *measurement*, including a weak one; deciding whether a correlation of 0.11
is worth acting on is the caller's problem, and the ontology's ``LEADS`` / ``LAGS``
priors are the thing this is compared against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from app.context.global_context import (
    CrossMarketConfig,
    default_global_indicator_config,
)

__all__ = [
    "BetaEstimate",
    "LeadLagEstimate",
    "RelativeStrength",
    "estimate_beta",
    "estimate_lead_lag",
    "relative_strength",
    "resample_returns",
]


@dataclass(frozen=True)
class BetaEstimate:
    """Rolling OLS beta of a local series against a reference series."""

    beta: float
    #: ``False`` when the window was too thin and ``beta`` fell back to 1.0.
    estimated: bool
    sample_count: int
    r_squared: float | None
    clamped: bool
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "estimated": self.estimated,
            "sample_count": self.sample_count,
            "r_squared": self.r_squared,
            "clamped": self.clamped,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class RelativeStrength:
    """``R_local - beta * R_reference`` with the beta that produced it."""

    value: float
    local_return: float
    reference_return: float
    beta: BetaEstimate
    reference: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "relative_strength": self.value,
            "local_return": self.local_return,
            "reference_return": self.reference_return,
            "reference": self.reference,
            "beta": self.beta.as_dict(),
        }


@dataclass(frozen=True)
class LeadLagEstimate:
    """Which series moves first, and by how much."""

    leader: str
    follower: str
    #: Positive means ``leader`` moves ``lag_minutes`` ahead of ``follower``.
    lag_minutes: int
    correlation: float
    sample_count: int
    #: Correlation at zero lag, for comparison — a lead only means something if it beats
    #: contemporaneous correlation.
    contemporaneous_correlation: float

    @property
    def leads(self) -> bool:
        return self.lag_minutes > 0 and abs(self.correlation) > abs(
            self.contemporaneous_correlation
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "leader": self.leader,
            "follower": self.follower,
            "lag_minutes": self.lag_minutes,
            "correlation": self.correlation,
            "contemporaneous_correlation": self.contemporaneous_correlation,
            "sample_count": self.sample_count,
            "leads": self.leads,
        }


BETA_INSUFFICIENT_SAMPLES = "BETA_INSUFFICIENT_SAMPLES"
BETA_REFERENCE_FLAT = "BETA_REFERENCE_FLAT"
BETA_CLAMPED = "BETA_CLAMPED"


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _paired(
    local: Sequence[float], reference: Sequence[float]
) -> tuple[list[float], list[float]]:
    """Positionally aligned finite pairs, newest-last.

    Alignment is by position, so callers must pass series sampled on the same clock.
    :func:`resample_returns` exists to produce exactly that from timestamped points.
    """
    pairs = [
        (left, right)
        for raw_left, raw_right in zip(local, reference)
        if (left := _finite(raw_left)) is not None
        and (right := _finite(raw_right)) is not None
    ]
    if not pairs:
        return [], []
    return [left for left, _ in pairs], [right for _, right in pairs]


def estimate_beta(
    local_returns: Sequence[float],
    reference_returns: Sequence[float],
    *,
    config: CrossMarketConfig | None = None,
) -> BetaEstimate:
    """Rolling OLS beta over the most recent ``beta_window`` paired observations.

    Falls back to ``beta = 1.0`` — flagged, never silently — when the overlap is too
    thin or the reference has no variance. A fabricated beta is worse than an admitted
    fallback: it would be indistinguishable from a measured one downstream.
    """
    settings = config or default_global_indicator_config().cross_market
    left, right = _paired(local_returns, reference_returns)
    window = max(2, int(settings.beta_window))
    left, right = left[-window:], right[-window:]
    count = len(left)
    if count < max(2, int(settings.beta_minimum_samples)):
        return BetaEstimate(
            beta=1.0,
            estimated=False,
            sample_count=count,
            r_squared=None,
            clamped=False,
            reason_codes=(BETA_INSUFFICIENT_SAMPLES,),
        )
    mean_left = sum(left) / count
    mean_right = sum(right) / count
    covariance = sum(
        (a - mean_left) * (b - mean_right) for a, b in zip(left, right)
    )
    variance_right = sum((b - mean_right) ** 2 for b in right)
    if variance_right <= 0.0:
        return BetaEstimate(
            beta=1.0,
            estimated=False,
            sample_count=count,
            r_squared=None,
            clamped=False,
            reason_codes=(BETA_REFERENCE_FLAT,),
        )
    raw_beta = covariance / variance_right
    variance_left = sum((a - mean_left) ** 2 for a in left)
    r_squared = (
        (covariance**2) / (variance_left * variance_right)
        if variance_left > 0.0
        else None
    )
    lower, upper = settings.beta_bounds
    beta = min(upper, max(lower, raw_beta))
    return BetaEstimate(
        beta=round(beta, 6),
        estimated=True,
        sample_count=count,
        r_squared=None if r_squared is None else round(min(1.0, max(0.0, r_squared)), 6),
        clamped=beta != raw_beta,
        reason_codes=(BETA_CLAMPED,) if beta != raw_beta else (),
    )


def relative_strength(
    local_return: float,
    reference_return: float,
    *,
    local_history: Sequence[float] = (),
    reference_history: Sequence[float] = (),
    reference: str = "",
    config: CrossMarketConfig | None = None,
    beta: BetaEstimate | None = None,
) -> RelativeStrength:
    """``RS = R_local - beta * R_reference``, with beta estimated from the histories."""
    resolved = beta or estimate_beta(local_history, reference_history, config=config)
    local = _finite(local_return) or 0.0
    market = _finite(reference_return) or 0.0
    return RelativeStrength(
        value=round(local - resolved.beta * market, 8),
        local_return=local,
        reference_return=market,
        beta=resolved,
        reference=str(reference or ""),
    )


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    count = min(len(left), len(right))
    if count < 3:
        return None
    a, b = list(left[-count:]), list(right[-count:])
    mean_a = sum(a) / count
    mean_b = sum(b) / count
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a <= 0.0 or var_b <= 0.0:
        return None
    return cov / math.sqrt(var_a * var_b)


def estimate_lead_lag(
    candidate_leader: Sequence[float],
    candidate_follower: Sequence[float],
    *,
    leader_name: str = "leader",
    follower_name: str = "follower",
    step_minutes: int = 1,
    config: CrossMarketConfig | None = None,
) -> LeadLagEstimate | None:
    """Best lag at which ``candidate_leader`` explains ``candidate_follower``.

    Series are positionally aligned return series on a common ``step_minutes`` grid. The
    search is bounded by ``max_lag_minutes`` so a long series cannot be mined for a
    spurious lag, and the contemporaneous correlation is returned alongside so a caller
    can see whether the lead is real or an artefact of overall co-movement.
    """
    settings = config or default_global_indicator_config().cross_market
    step = max(1, int(step_minutes))
    max_shift = max(0, int(settings.max_lag_minutes) // step)
    contemporaneous = _correlation(candidate_leader, candidate_follower)
    if contemporaneous is None:
        return None
    best_lag = 0
    best_correlation = contemporaneous
    for shift in range(1, max_shift + 1):
        if shift >= len(candidate_leader) or shift >= len(candidate_follower):
            break
        shifted_leader = candidate_leader[: len(candidate_leader) - shift]
        shifted_follower = candidate_follower[shift:]
        correlation = _correlation(shifted_leader, shifted_follower)
        if correlation is None:
            continue
        if abs(correlation) > abs(best_correlation):
            best_correlation = correlation
            best_lag = shift
    return LeadLagEstimate(
        leader=leader_name,
        follower=follower_name,
        lag_minutes=best_lag * step,
        correlation=round(best_correlation, 6),
        sample_count=min(len(candidate_leader), len(candidate_follower)),
        contemporaneous_correlation=round(contemporaneous, 6),
    )


def resample_returns(
    points: Sequence[tuple[datetime, float]],
    *,
    step_minutes: int = 1,
    window_minutes: int | None = None,
    end: datetime | None = None,
) -> list[float]:
    """Log returns of ``points`` on a fixed ``step_minutes`` grid, oldest-first.

    Two series can only be compared bar-for-bar if they sit on the same clock, and market
    data does not arrive on one. Buckets with no observation carry the previous level
    forward and therefore produce a zero return — the honest reading for "nothing traded",
    and one that cannot manufacture a move the way interpolation would.
    """
    usable = sorted(
        (
            (
                moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc),
                value,
            )
            for moment, raw in points
            if (value := _finite(raw)) is not None and value > 0.0
        ),
        key=lambda item: item[0],
    )
    if len(usable) < 2:
        return []
    step = timedelta(minutes=max(1, int(step_minutes)))
    finish = (end or usable[-1][0]).astimezone(timezone.utc)
    start = (
        finish - timedelta(minutes=int(window_minutes))
        if window_minutes
        else usable[0][0]
    )
    levels: list[float] = []
    index = 0
    last_level: float | None = None
    cursor = start
    while cursor <= finish:
        while index < len(usable) and usable[index][0] <= cursor:
            last_level = usable[index][1]
            index += 1
        if last_level is not None:
            levels.append(last_level)
        cursor += step
    return [
        math.log(levels[position] / levels[position - 1])
        for position in range(1, len(levels))
        if levels[position - 1] > 0.0 and levels[position] > 0.0
    ]

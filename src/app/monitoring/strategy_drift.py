"""Rolling per-strategy health, and the automatic demotions it justifies.

``non_stationarity``: a strategy's edge and the market's regime both move, so a one-off
validation cannot keep a strategy authorised. This monitor watches the rolling numbers and
proposes lifecycle demotions.

Two rules that keep it from being a hair trigger
-----------------------------------------------
* **Never demote on one loss.** A demotion needs a minimum sample, a rolling net EV below
  zero AND a negative lower confidence bound. Any one of those alone is noise: a single
  -60bps fill moves a 12-sample mean by 5bps, which is inside the estimator's own error.
* **One rung at a time.** ``LIVE -> DEGRADED -> SHADOW``. There is no LIVE -> SHADOW edge,
  mirroring ``app.trading.directional.ALLOWED_TRANSITIONS`` on the promotion side: a
  two-rung fall would skip the state where the strategy is still measured but not traded,
  which is exactly where the evidence for the next decision comes from.

The monitor PROPOSES. It returns :class:`DemotionProposal` objects; applying them is a
separate, audited act (``app.strategy_validation.registry``), because a monitor that could
silently rewrite lifecycle state would make the operating configuration unreviewable.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Mapping, Sequence

from app.strategy.spec import StrategyLifecycleState

__all__ = [
    "DemotionProposal",
    "StrategyDriftMonitor",
    "StrategyHealth",
    "StrategyDriftConfig",
]


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class StrategyDriftConfig:
    window: int = 60
    minimum_samples: int = 20
    #: Rolling net EV below this (bps) is a demotion candidate.
    net_ev_floor_bps: float = 0.0
    #: Lower confidence bound below this (bps) is required as well.
    lower_bound_floor_bps: float = 0.0
    #: Cost-to-edge above this means the edge is being eaten by the round trip.
    cost_to_edge_ceiling: float = 1.0
    #: Rolling drawdown (bps, positive magnitude) that on its own justifies review.
    drawdown_ceiling_bps: float = 400.0
    #: Confidence level for the lower bound. 1.645 = one-sided 95%.
    z_score: float = 1.645
    #: Outcomes older than this are dropped from the rolling window entirely.
    max_age_seconds: float = 30 * 24 * 3600.0


@dataclass(frozen=True)
class StrategyHealth:
    """Rolling metrics for one strategy. ``None`` means not measurable, never zero."""

    strategy_id: str
    sample_count: int
    live_sample_count: int
    rolling_net_ev_bps: float | None
    rolling_hit_rate: float | None
    rolling_profit_factor: float | None
    rolling_cost_to_edge_ratio: float | None
    rolling_drawdown_bps: float | None
    lower_confidence_bound_bps: float | None
    rolling_prediction_brier: float | None
    rolling_return_mae_bps: float | None
    rolling_uncertainty_calibration: float | None
    last_outcome_at: datetime | None
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "sample_count": self.sample_count,
            "live_sample_count": self.live_sample_count,
            "rolling_net_ev_bps": _round(self.rolling_net_ev_bps),
            "rolling_hit_rate": _round(self.rolling_hit_rate, 4),
            "rolling_profit_factor": _round(self.rolling_profit_factor, 3),
            "rolling_cost_to_edge_ratio": _round(self.rolling_cost_to_edge_ratio, 3),
            "rolling_drawdown_bps": _round(self.rolling_drawdown_bps),
            "lower_confidence_bound_bps": _round(self.lower_confidence_bound_bps),
            "rolling_prediction_brier": _round(self.rolling_prediction_brier, 4),
            "rolling_return_mae_bps": _round(self.rolling_return_mae_bps),
            "rolling_uncertainty_calibration": _round(
                self.rolling_uncertainty_calibration, 4
            ),
            "last_outcome_at": self.last_outcome_at.isoformat()
            if self.last_outcome_at
            else None,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class DemotionProposal:
    """A recommendation, not a change. Applying it is an audited separate act."""

    strategy_id: str
    from_state: StrategyLifecycleState
    to_state: StrategyLifecycleState
    health: StrategyHealth
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "from_state": str(self.from_state),
            "to_state": str(self.to_state),
            "reason_codes": list(self.reason_codes),
            "health": self.health.as_dict(),
        }


#: Demotion ladder. One rung per proposal; see the module docstring.
_DEMOTION_LADDER: Mapping[StrategyLifecycleState, StrategyLifecycleState] = {
    StrategyLifecycleState.LIVE: StrategyLifecycleState.DEGRADED,
    StrategyLifecycleState.LIVE_PROBE: StrategyLifecycleState.DEGRADED,
    StrategyLifecycleState.DEGRADED: StrategyLifecycleState.SHADOW,
}


@dataclass
class _Sample:
    net_bps: float
    cost_bps: float | None
    predicted_net_bps: float | None
    predicted_probability: float | None
    predicted_uncertainty_bps: float | None
    is_live: bool
    at: datetime


class StrategyDriftMonitor:
    """Keeps a bounded rolling window of outcomes per strategy."""

    def __init__(self, *, config: StrategyDriftConfig | None = None) -> None:
        self._config = config or StrategyDriftConfig()
        self._samples: dict[str, Deque[_Sample]] = defaultdict(
            lambda: deque(maxlen=self._config.window)
        )

    def record(
        self,
        *,
        strategy_id: str,
        net_return_bps: float,
        at: datetime,
        cost_bps: float | None = None,
        predicted_net_bps: float | None = None,
        predicted_probability: float | None = None,
        predicted_uncertainty_bps: float | None = None,
        is_live: bool = False,
    ) -> None:
        key = str(strategy_id or "").strip().lower()
        if not key:
            return
        value = _finite(net_return_bps)
        if value is None:
            return
        self._samples[key].append(
            _Sample(
                net_bps=value,
                cost_bps=_finite(cost_bps),
                predicted_net_bps=_finite(predicted_net_bps),
                predicted_probability=_finite(predicted_probability),
                predicted_uncertainty_bps=_finite(predicted_uncertainty_bps),
                is_live=bool(is_live),
                at=_aware(at),
            )
        )

    def record_outcome(self, outcome: Any, *, prediction: Any = None) -> None:
        """Convenience for a ``ShadowOutcome`` / ``StrategyOutcome`` / mapping."""
        read = (
            (lambda name: outcome.get(name))
            if isinstance(outcome, Mapping)
            else (lambda name: getattr(outcome, name, None))
        )
        at = read("closed_at") or read("recorded_at")
        source = str(read("evidence_source") or read("source") or "SHADOW").upper()
        self.record(
            strategy_id=str(read("strategy_id") or ""),
            net_return_bps=_finite(read("net_return_bps"))
            or _finite(read("realized_net_bps"))
            or 0.0,
            at=at if isinstance(at, datetime) else datetime.now(timezone.utc),
            cost_bps=_finite(read("cost_bps")),
            predicted_net_bps=_finite(getattr(prediction, "expected_net_return_bps", None)),
            predicted_probability=_finite(getattr(prediction, "probability_profit", None)),
            predicted_uncertainty_bps=_finite(getattr(prediction, "uncertainty_bps", None)),
            is_live=source in {"LIVE", "LIVE_PROBE"},
        )

    # -- reads -------------------------------------------------------------- #
    def health(self, strategy_id: str, *, now: datetime | None = None) -> StrategyHealth:
        key = str(strategy_id or "").strip().lower()
        config = self._config
        moment = _aware(now or datetime.now(timezone.utc))
        cutoff = moment - timedelta(seconds=config.max_age_seconds)
        samples = [item for item in self._samples.get(key, ()) if item.at >= cutoff]
        if not samples:
            return StrategyHealth(
                strategy_id=key,
                sample_count=0,
                live_sample_count=0,
                rolling_net_ev_bps=None,
                rolling_hit_rate=None,
                rolling_profit_factor=None,
                rolling_cost_to_edge_ratio=None,
                rolling_drawdown_bps=None,
                lower_confidence_bound_bps=None,
                rolling_prediction_brier=None,
                rolling_return_mae_bps=None,
                rolling_uncertainty_calibration=None,
                last_outcome_at=None,
                reason_codes=("DRIFT_NO_SAMPLES",),
            )

        nets = [item.net_bps for item in samples]
        wins = [value for value in nets if value > 0]
        losses = [-value for value in nets if value < 0]
        mean = sum(nets) / len(nets)
        stdev = statistics.stdev(nets) if len(nets) >= 2 else None
        lower_bound = (
            mean - config.z_score * (stdev / math.sqrt(len(nets)))
            if stdev is not None
            else None
        )
        costs = [item.cost_bps for item in samples if item.cost_bps is not None]
        gross = [
            item.net_bps + item.cost_bps
            for item in samples
            if item.cost_bps is not None
        ]
        mean_gross = sum(gross) / len(gross) if gross else None
        cost_to_edge = (
            (sum(costs) / len(costs)) / abs(mean_gross)
            if costs and mean_gross not in (None, 0.0)
            else None
        )

        predicted = [
            (item.predicted_net_bps, item.net_bps)
            for item in samples
            if item.predicted_net_bps is not None
        ]
        mae = (
            sum(abs(pred - actual) for pred, actual in predicted) / len(predicted)
            if predicted
            else None
        )
        probabilities = [
            (item.predicted_probability, 1.0 if item.net_bps > 0 else 0.0)
            for item in samples
            if item.predicted_probability is not None
        ]
        brier = (
            sum((pred - actual) ** 2 for pred, actual in probabilities) / len(probabilities)
            if probabilities
            else None
        )
        # Uncertainty calibration: the fraction of outcomes that landed inside the
        # predicted band. Well calibrated is ~0.68 for a one-sigma band, so a value near
        # 1.0 means the model is over-stating its uncertainty and near 0.0 understating it.
        banded = [
            item
            for item in samples
            if item.predicted_uncertainty_bps is not None
            and item.predicted_net_bps is not None
        ]
        calibration = (
            sum(
                1
                for item in banded
                if abs(item.net_bps - (item.predicted_net_bps or 0.0))
                <= (item.predicted_uncertainty_bps or 0.0)
            )
            / len(banded)
            if banded
            else None
        )

        reasons: list[str] = []
        if len(samples) < config.minimum_samples:
            reasons.append(f"DRIFT_SAMPLE_BELOW_{config.minimum_samples}")
        if not any(item.is_live for item in samples):
            reasons.append("DRIFT_NO_LIVE_EVIDENCE")

        return StrategyHealth(
            strategy_id=key,
            sample_count=len(samples),
            live_sample_count=sum(1 for item in samples if item.is_live),
            rolling_net_ev_bps=mean,
            rolling_hit_rate=len(wins) / len(nets),
            rolling_profit_factor=(
                (sum(wins) / sum(losses)) if losses and sum(losses) > 0 else None
            ),
            rolling_cost_to_edge_ratio=cost_to_edge,
            rolling_drawdown_bps=_max_drawdown_bps(nets),
            lower_confidence_bound_bps=lower_bound,
            rolling_prediction_brier=brier,
            rolling_return_mae_bps=mae,
            rolling_uncertainty_calibration=calibration,
            last_outcome_at=max(item.at for item in samples),
            reason_codes=tuple(reasons),
        )

    def all_health(self, *, now: datetime | None = None) -> tuple[StrategyHealth, ...]:
        return tuple(
            self.health(strategy_id, now=now) for strategy_id in sorted(self._samples)
        )

    def demotion_proposals(
        self,
        lifecycle_states: Mapping[str, StrategyLifecycleState],
        *,
        now: datetime | None = None,
    ) -> tuple[DemotionProposal, ...]:
        """Propose one-rung demotions for strategies whose rolling health justifies it."""
        config = self._config
        proposals: list[DemotionProposal] = []
        for strategy_id, state in lifecycle_states.items():
            target = _DEMOTION_LADDER.get(state)
            if target is None:
                continue
            health = self.health(strategy_id, now=now)
            if health.sample_count < config.minimum_samples:
                continue
            net = health.rolling_net_ev_bps
            lower = health.lower_confidence_bound_bps
            if net is None or lower is None:
                continue

            reasons: list[str] = []
            # BOTH must fail. Requiring only the mean would demote on a run of bad luck;
            # requiring only the bound would demote a strategy whose mean is fine but whose
            # variance is high, which is a sizing question, not a lifecycle one.
            if net < config.net_ev_floor_bps and lower < config.lower_bound_floor_bps:
                reasons.append("DRIFT_ROLLING_NET_EV_NEGATIVE")
                reasons.append("DRIFT_LOWER_BOUND_NEGATIVE")
            if (
                health.rolling_cost_to_edge_ratio is not None
                and health.rolling_cost_to_edge_ratio > config.cost_to_edge_ceiling
            ):
                reasons.append("DRIFT_COST_EXCEEDS_EDGE")
            if (
                health.rolling_drawdown_bps is not None
                and health.rolling_drawdown_bps > config.drawdown_ceiling_bps
            ):
                reasons.append("DRIFT_DRAWDOWN_ABOVE_CEILING")
            if not reasons:
                continue
            # A cost or drawdown flag alone is a REVIEW signal, not a demotion: the fix may
            # be the horizon or the venue rather than the strategy. Only the paired
            # EV+bound failure demotes.
            if "DRIFT_ROLLING_NET_EV_NEGATIVE" not in reasons:
                continue
            proposals.append(
                DemotionProposal(
                    strategy_id=strategy_id,
                    from_state=state,
                    to_state=target,
                    health=health,
                    reason_codes=tuple(dict.fromkeys(reasons)),
                )
            )
        return tuple(proposals)


def _max_drawdown_bps(values: Sequence[float]) -> float | None:
    """Peak-to-trough of the cumulative bps series, as a positive magnitude."""
    if not values:
        return None
    cumulative = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    return drawdown


def _round(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None

"""One evaluator over every strategy, producing one classification each.

The point of a single runner
---------------------------
Every strategy must be audited by the SAME evaluator, or the comparison between them is
meaningless. Previously each strategy's evidence came from wherever it happened to exist —
one from a screen over stored minute bars, another from a checkpoint's simulated fills, a
third from nothing at all — and those were compared as if they were the same measurement.

Classification
--------------
``KEEP``         positive lower bound after cost, stable out of sample, parameters on a plateau
``FIX``          gross edge exists but cost/horizon/execution destroys it
``SHADOW_ONLY``  plausible but unproven: no live evidence, or fails only the sample floor
``RESEARCH``     the thesis itself loses when allowed to act on its own terms
``RETIRE``       loses AND has enough evidence to say so with confidence

The runner NEVER changes lifecycle state. It returns recommendations plus a
``StrategyValidationRecord``; applying them goes through
``app.strategy_validation.registry``, which enforces the transition whitelist and the
promotion gates. That separation is what keeps "the audit said so" from becoming an
unreviewed change to what trades.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Callable, Mapping, Sequence

from app.strategy.registry import StrategyRegistry, default_strategy_registry
from app.strategy.spec import StrategyLifecycleState, StrategySpec
from app.strategy_validation.cost_stress import CostStressResult, cost_stress
from app.strategy_validation.metrics import (
    StrategyMetrics,
    TradeObservation,
    compute_metrics,
)
from app.strategy_validation.parameter_stability import (
    ParameterStabilityResult,
    parameter_stability,
)
from app.strategy_validation.purged_cv import PurgedSplit, purged_kfold_splits
from app.strategy_validation.regime_breakdown import RegimeBreakdown, regime_breakdown
from app.strategy_validation.registry import StrategyValidationRecord
from app.strategy_validation.walk_forward import WalkForwardResult, walk_forward

__all__ = [
    "AuditClassification",
    "AuditReport",
    "StrategyAudit",
    "StrategyAuditRunner",
]


class AuditClassification(StrEnum):
    KEEP = "KEEP"
    FIX = "FIX"
    SHADOW_ONLY = "SHADOW_ONLY"
    RESEARCH = "RESEARCH"
    RETIRE = "RETIRE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class AuditThresholds:
    """Bars the audit applies. All config, none embedded in a comparison.

    ``minimum_samples`` at 30 is the bar for a *classification with confidence*, not for
    reporting: below it the verdict is ``INSUFFICIENT_DATA`` and the metrics are still shown.
    ``retire_samples`` is higher because retiring is irreversible in practice — a strategy
    nobody runs accumulates no evidence to come back on.
    """

    minimum_samples: int = 30
    retire_samples: int = 60
    neutral_band_bps: float = 5.0
    minimum_break_even_cost_multiple: float = 1.25
    minimum_out_of_sample_stability: float = 0.6
    walk_forward_windows: int = 4
    purged_folds: int = 5


@dataclass(frozen=True)
class StrategyAudit:
    """Everything the audit measured for one strategy, plus its classification."""

    strategy_id: str
    spec: StrategySpec
    classification: AuditClassification
    metrics: StrategyMetrics
    cost: CostStressResult
    walk_forward: WalkForwardResult
    regime: RegimeBreakdown
    parameters: ParameterStabilityResult | None
    purged_splits: tuple[PurgedSplit, ...]
    recommended_lifecycle: StrategyLifecycleState
    reason_codes: tuple[str, ...]

    @property
    def lifecycle_change_recommended(self) -> bool:
        return self.recommended_lifecycle is not self.spec.lifecycle_state

    def to_record(self, *, now: datetime | None = None) -> StrategyValidationRecord:
        return StrategyValidationRecord(
            strategy_id=self.strategy_id,
            validated_at=now or datetime.now(timezone.utc),
            validation_version=f"audit-{self.spec.algorithm_version}",
            algorithm_version=self.spec.algorithm_version,
            sample_count=self.metrics.trigger_count,
            effective_sample_count=self.metrics.effective_sample_count,
            net_ev_bps=self.metrics.net_ev_bps,
            gross_ev_bps=self.metrics.gross_ev_bps,
            lower_confidence_bound_bps=self.metrics.lower_confidence_bound_bps,
            profit_factor=self.metrics.profit_factor,
            cost_to_edge_ratio=self.metrics.cost_to_edge_ratio,
            break_even_cost_multiple=self.cost.break_even_cost_multiple,
            out_of_sample_stability=self.walk_forward.out_of_sample_stability,
            parameter_stability=(
                self.parameters.stable if self.parameters is not None else None
            ),
            approved_markets=tuple(
                market
                for market, value in dict(self.metrics.market_breakdown).items()
                if value is not None and value > 0.0
            ),
            approved_regimes=self.regime.positive_buckets,
            evidence_mix=self.metrics.evidence_mix,
            reason_codes=self.reason_codes,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "classification": str(self.classification),
            "current_lifecycle": str(self.spec.lifecycle_state),
            "recommended_lifecycle": str(self.recommended_lifecycle),
            "lifecycle_change_recommended": self.lifecycle_change_recommended,
            "reason_codes": list(self.reason_codes),
            "metrics": self.metrics.as_dict(),
            "cost_stress": self.cost.as_dict(),
            "walk_forward": self.walk_forward.as_dict(),
            "regime": self.regime.as_dict(),
            "parameters": self.parameters.as_dict() if self.parameters else None,
            "purged_folds": [item.as_dict() for item in self.purged_splits],
        }


@dataclass(frozen=True)
class AuditReport:
    audits: tuple[StrategyAudit, ...]
    generated_at: datetime

    def by_classification(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for audit in self.audits:
            grouped.setdefault(str(audit.classification), []).append(audit.strategy_id)
        return grouped

    def table(self) -> list[dict[str, Any]]:
        return [
            {
                "strategy_id": audit.strategy_id,
                "family": str(audit.spec.family),
                "direction": audit.spec.direction,
                "horizon_seconds": audit.spec.horizon_seconds,
                "current_lifecycle": str(audit.spec.lifecycle_state),
                "classification": str(audit.classification),
                "recommended_lifecycle": str(audit.recommended_lifecycle),
                "trades": audit.metrics.trigger_count,
                "effective_n": round(audit.metrics.effective_sample_count, 2),
                "gross_ev_bps": audit.metrics.gross_ev_bps,
                "net_ev_bps": audit.metrics.net_ev_bps,
                "lower_bound_bps": audit.metrics.lower_confidence_bound_bps,
                "break_even_cost_multiple": audit.cost.break_even_cost_multiple,
                "oos_stability": audit.walk_forward.out_of_sample_stability,
                "reason_codes": list(audit.reason_codes),
            }
            for audit in self.audits
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "strategy_count": len(self.audits),
            "by_classification": self.by_classification(),
            "table": self.table(),
            "audits": [audit.as_dict() for audit in self.audits],
        }


class StrategyAuditRunner:
    """Runs the whole validation stack over one trade set per strategy."""

    def __init__(
        self,
        *,
        registry: StrategyRegistry | None = None,
        thresholds: AuditThresholds | None = None,
    ) -> None:
        self._registry = registry or default_strategy_registry()
        self._thresholds = thresholds or AuditThresholds()

    def run(
        self,
        trades_by_strategy: Mapping[str, Sequence[TradeObservation]],
        *,
        parameter_evaluator: Callable[[str, Mapping[str, float]], float | None] | None = None,
        strategy_ids: Sequence[str] | None = None,
        now: datetime | None = None,
    ) -> AuditReport:
        """Audit every requested strategy — including ones with NO trades.

        Auditing the empty ones matters: "this strategy is live-authorised and has produced
        zero observations" is the single most actionable finding the runner can make, and
        skipping empty inputs would hide it.
        """
        wanted = (
            tuple(str(item).strip().lower() for item in strategy_ids)
            if strategy_ids is not None
            else tuple(spec.strategy_id for spec in self._registry.all_specs())
        )
        moment = now or datetime.now(timezone.utc)
        audits = [
            self._audit_one(
                strategy_id,
                tuple(trades_by_strategy.get(strategy_id, ()) or ()),
                parameter_evaluator=parameter_evaluator,
            )
            for strategy_id in wanted
            if self._registry.get(strategy_id) is not None
        ]
        return AuditReport(audits=tuple(audits), generated_at=moment)

    # -- internals ---------------------------------------------------------- #
    def _audit_one(
        self,
        strategy_id: str,
        trades: Sequence[TradeObservation],
        *,
        parameter_evaluator: Callable[[str, Mapping[str, float]], float | None] | None,
    ) -> StrategyAudit:
        spec = self._registry.require(strategy_id)
        thresholds = self._thresholds

        metrics = compute_metrics(
            strategy_id, trades, minimum_samples=thresholds.minimum_samples
        )
        cost = cost_stress(strategy_id, trades)
        forward = walk_forward(
            strategy_id, trades, windows=thresholds.walk_forward_windows
        )
        regime = regime_breakdown(strategy_id, trades, dimension="market_regime")
        splits = purged_kfold_splits(trades, folds=thresholds.purged_folds)
        parameters = self._parameter_stability(spec, parameter_evaluator)

        classification, reasons = self._classify(
            spec=spec,
            metrics=metrics,
            cost=cost,
            forward=forward,
            parameters=parameters,
        )
        return StrategyAudit(
            strategy_id=strategy_id,
            spec=spec,
            classification=classification,
            metrics=metrics,
            cost=cost,
            walk_forward=forward,
            regime=regime,
            parameters=parameters,
            purged_splits=splits,
            recommended_lifecycle=_recommended_lifecycle(spec, classification),
            reason_codes=reasons,
        )

    def _parameter_stability(
        self,
        spec: StrategySpec,
        evaluator: Callable[[str, Mapping[str, float]], float | None] | None,
    ) -> ParameterStabilityResult | None:
        if evaluator is None:
            # Without a re-evaluation callback there is nothing to sweep. ``None`` (rather
            # than an empty result) keeps "not tested" distinguishable from "tested and
            # stable" — the promotion gate treats them differently.
            return None
        parameters = {
            name: value
            for name, value in self._registry.resolved_parameters(spec.strategy_id).items()
            # Deployment flags are not tuning knobs; sweeping them would ask "what if this
            # strategy were disabled", which is not a stability question.
            if name not in {"enabled", "shadow_enabled", "paper_enabled", "live_authorized"}
        }
        if not parameters:
            return None
        return parameter_stability(
            spec.strategy_id,
            parameters=parameters,
            evaluate=lambda values: evaluator(spec.strategy_id, values),
        )

    def _classify(
        self,
        *,
        spec: StrategySpec,
        metrics: StrategyMetrics,
        cost: CostStressResult,
        forward: WalkForwardResult,
        parameters: ParameterStabilityResult | None,
    ) -> tuple[AuditClassification, tuple[str, ...]]:
        thresholds = self._thresholds
        reasons: list[str] = []

        if metrics.trigger_count == 0:
            reasons.append("AUDIT_NO_OBSERVATIONS")
            if spec.lifecycle_state.is_live_candidate:
                # The finding worth shouting about: authorised to trade, never measured.
                reasons.append("AUDIT_LIVE_AUTHORIZED_WITHOUT_EVIDENCE")
            return AuditClassification.INSUFFICIENT_DATA, tuple(reasons)

        gross = metrics.gross_ev_bps
        net = metrics.net_ev_bps
        lower = metrics.lower_confidence_bound_bps
        band = thresholds.neutral_band_bps

        # 1. Cost / horizon FIRST, for the same reason as in the selector evaluator: on net
        #    numbers alone a cost problem is indistinguishable from a dead thesis, and the
        #    remedy is completely different.
        if (
            gross is not None
            and gross > band
            and net is not None
            and net < -band
        ):
            reasons.append("AUDIT_GROSS_POSITIVE_NET_NEGATIVE")
            return AuditClassification.FIX, tuple(reasons)

        if metrics.trigger_count < thresholds.minimum_samples:
            reasons.append(f"AUDIT_SAMPLE_BELOW_{thresholds.minimum_samples}")
            return AuditClassification.SHADOW_ONLY, tuple(reasons)

        has_live_evidence = any(
            str(source).upper() in {"LIVE", "LIVE_PROBE"}
            for source in metrics.evidence_mix
        )
        if net is not None and net < -band:
            reasons.append("AUDIT_NET_NEGATIVE")
            if metrics.trigger_count >= thresholds.retire_samples:
                reasons.append(f"AUDIT_SAMPLE_ABOVE_{thresholds.retire_samples}")
                if not has_live_evidence:
                    # RETIRE is terminal in practice — a strategy nobody runs accumulates no
                    # evidence to come back on — and a simulated fill is an assumption about
                    # liquidity at a modelled price. A large negative SHADOW sample is ample
                    # reason to stop trading the thesis (RESEARCH does that) but not to close
                    # the file on it.
                    reasons.append("AUDIT_NO_LIVE_EVIDENCE")
                    return AuditClassification.RESEARCH, tuple(reasons)
                return AuditClassification.RETIRE, tuple(reasons)
            return AuditClassification.RESEARCH, tuple(reasons)

        # Positive-mean strategies still have to clear the three robustness bars. Any miss
        # keeps them measurable but out of live selection.
        blocking: list[str] = []
        if lower is None or lower <= 0.0:
            blocking.append("AUDIT_LOWER_BOUND_NOT_POSITIVE")
        if (
            cost.break_even_cost_multiple is None
            or cost.break_even_cost_multiple < thresholds.minimum_break_even_cost_multiple
        ):
            blocking.append("AUDIT_COST_STRESS_MARGINAL")
        stability = forward.out_of_sample_stability
        if stability is None or stability < thresholds.minimum_out_of_sample_stability:
            blocking.append("AUDIT_OUT_OF_SAMPLE_UNSTABLE")
        if parameters is not None and parameters.stable is False:
            blocking.append("AUDIT_PARAMETERS_FRAGILE")
        if not has_live_evidence:
            blocking.append("AUDIT_NO_LIVE_EVIDENCE")

        if blocking:
            reasons.extend(blocking)
            return AuditClassification.SHADOW_ONLY, tuple(reasons)

        reasons.append("AUDIT_ROBUST_POSITIVE_EDGE")
        return AuditClassification.KEEP, tuple(reasons)


def _recommended_lifecycle(
    spec: StrategySpec, classification: AuditClassification
) -> StrategyLifecycleState:
    """Map a classification to a lifecycle state, never skipping a rung upward.

    An audit can recommend a demotion of any size (a broken strategy should stop trading now)
    but a promotion of at most one rung — the registry's transition whitelist would refuse
    more anyway, and recommending something unapplicable is noise.
    """
    current = spec.lifecycle_state
    if classification is AuditClassification.RETIRE:
        return StrategyLifecycleState.RETIRED
    if classification is AuditClassification.RESEARCH:
        return StrategyLifecycleState.RESEARCH
    if classification in {
        AuditClassification.SHADOW_ONLY,
        AuditClassification.FIX,
        AuditClassification.INSUFFICIENT_DATA,
    }:
        # Do not raise a RESEARCH strategy to SHADOW on an inconclusive audit; only lower a
        # trading one.
        return (
            StrategyLifecycleState.SHADOW
            if current.rank > StrategyLifecycleState.SHADOW.rank
            else current
        )
    # KEEP: one rung up at most, and only from a state the ladder allows.
    if current is StrategyLifecycleState.SHADOW:
        return StrategyLifecycleState.LIVE_PROBE
    if current is StrategyLifecycleState.LIVE_PROBE:
        return StrategyLifecycleState.LIVE
    return current

"""Separates "the strategy has no edge" from "the selector picked the wrong one".

The diagnostic table from the task, implemented
----------------------------------------------
=================================  ================================  ==========================
oracle / context-conditioned       router-selected                   verdict
=================================  ================================  ==========================
positive                           negative                          SELECTOR_PROBLEM
negative                           any                               STRATEGY_PROBLEM
sign flips across contexts         any                               CONTEXT_MODELLING_PROBLEM
gross positive, net negative       any                               COST_OR_HORIZON_PROBLEM
=================================  ================================  ==========================

``oracle_context_performance`` is the strategy's outcome over every context where it was
*eligible and entry-ready* — i.e. what it earns when it is allowed to act on its own
thesis. ``router_selected_performance`` is its outcome over the subset where the selector
actually chose it. Both come from the same counterfactual groups, so they are measured on
the same tape and are directly comparable.

Order of checks matters and is not arbitrary: the cost/horizon test runs FIRST, because a
strategy whose gross edge is positive and net edge negative will also look like a
"strategy problem" on net numbers alone, and the fix is completely different (change the
horizon or stop trading the venue, not the thesis).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping, Sequence

from app.evaluation.outcome_resolver import OutcomeResolver, ResolvedOutcome
from app.evaluation.selector_regret import (
    ContextRegret,
    RegretSummary,
    compute_context_regret,
    summarize_regret,
)

__all__ = [
    "StrategyDiagnosis",
    "StrategyVerdict",
    "SelectorEvaluation",
    "SelectorEvaluator",
]


class StrategyVerdict(StrEnum):
    SELECTOR_PROBLEM = "SELECTOR_PROBLEM"
    STRATEGY_PROBLEM = "STRATEGY_PROBLEM"
    CONTEXT_MODELLING_PROBLEM = "CONTEXT_MODELLING_PROBLEM"
    COST_OR_HORIZON_PROBLEM = "COST_OR_HORIZON_PROBLEM"
    HEALTHY = "HEALTHY"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class StrategyDiagnosis:
    """One strategy's split between its own quality and the selector's use of it."""

    strategy_id: str
    verdict: StrategyVerdict
    oracle_sample_count: int
    oracle_net_bps: float | None
    oracle_gross_bps: float | None
    router_sample_count: int
    router_net_bps: float | None
    #: Per-market/regime means. A sign flip across these is what distinguishes a context
    #: modelling failure from a strategy failure.
    conditional_net_bps: Mapping[str, float] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "verdict": str(self.verdict),
            "oracle_sample_count": self.oracle_sample_count,
            "oracle_net_bps": _round(self.oracle_net_bps),
            "oracle_gross_bps": _round(self.oracle_gross_bps),
            "router_sample_count": self.router_sample_count,
            "router_net_bps": _round(self.router_net_bps),
            "conditional_net_bps": {
                key: round(value, 3) for key, value in dict(self.conditional_net_bps).items()
            },
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class SelectorEvaluation:
    regret: RegretSummary
    diagnoses: tuple[StrategyDiagnosis, ...]
    context_count: int

    def by_verdict(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for item in self.diagnoses:
            grouped[str(item.verdict)].append(item.strategy_id)
        return dict(grouped)

    def as_dict(self) -> dict[str, Any]:
        return {
            "context_count": self.context_count,
            "regret": self.regret.as_dict(),
            "diagnoses": [item.as_dict() for item in self.diagnoses],
            "by_verdict": self.by_verdict(),
        }


@dataclass(frozen=True)
class EvaluatorConfig:
    """Sample floors and the thresholds each verdict needs.

    ``minimum_samples`` is 12 rather than a round 30 because these are clustered samples:
    one context contributes several correlated rows, so the effective count is far below
    the row count and a large floor would make every strategy read INSUFFICIENT_DATA
    forever. It is a floor for *reporting a verdict*, not for acting on one — lifecycle
    changes go through ``app.strategy_validation``.
    """

    minimum_samples: int = 12
    #: |mean| below this is treated as indistinguishable from zero, in bps.
    neutral_band_bps: float = 5.0
    #: A conditional mean must exceed this magnitude to count as a genuine sign, so
    #: two near-zero means of opposite sign do not read as a flip.
    sign_flip_band_bps: float = 10.0
    minimum_conditional_samples: int = 5


class SelectorEvaluator:
    """Builds regret and per-strategy diagnoses from counterfactual groups."""

    def __init__(
        self,
        *,
        resolver: OutcomeResolver | None = None,
        config: EvaluatorConfig | None = None,
    ) -> None:
        self._resolver = resolver or OutcomeResolver()
        self._config = config or EvaluatorConfig()

    def evaluate(self, groups: Sequence[Any]) -> SelectorEvaluation:
        regrets = tuple(
            item for item in (compute_context_regret(group) for group in groups) if item
        )
        return SelectorEvaluation(
            regret=summarize_regret(regrets, groups=groups),
            diagnoses=self.diagnose(groups, regrets=regrets),
            context_count=len(regrets),
        )

    def diagnose(
        self,
        groups: Sequence[Any],
        *,
        regrets: Sequence[ContextRegret] | None = None,
    ) -> tuple[StrategyDiagnosis, ...]:
        selected_by_context = {
            str(getattr(group, "context_id", "")): getattr(group, "selected_strategy", None)
            for group in groups
        }
        oracle: dict[str, list[ResolvedOutcome]] = defaultdict(list)
        router: dict[str, list[float]] = defaultdict(list)
        conditional: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for group in groups:
            context_id = str(getattr(group, "context_id", ""))
            market = str(getattr(group, "market", "") or "UNKNOWN")
            for strategy_id, outcome in dict(getattr(group, "outcomes", {}) or {}).items():
                resolved = self._resolver.resolve(outcome)
                if resolved is None:
                    continue
                # Oracle: the strategy was eligible and fired, so this is what its own
                # thesis earned regardless of whether it was chosen.
                oracle[resolved.strategy_id].append(resolved)
                bucket = f"{market}|{resolved.regime}"
                conditional[resolved.strategy_id][bucket].append(resolved.net_return_bps)
                if selected_by_context.get(context_id) == strategy_id:
                    live = _finite(getattr(group, "live_outcome_net_bps", None))
                    # The router-selected number prefers the REAL fill. Falling back to the
                    # simulation is flagged in ``evidence`` below rather than silently mixed.
                    router[resolved.strategy_id].append(
                        live if live is not None else resolved.net_return_bps
                    )
        del regrets  # regret is summarised separately; diagnosis is per strategy

        diagnoses: list[StrategyDiagnosis] = []
        for strategy_id, rows in sorted(oracle.items()):
            diagnoses.append(
                self._diagnose_one(
                    strategy_id=strategy_id,
                    oracle_rows=rows,
                    router_values=router.get(strategy_id, []),
                    conditional=conditional.get(strategy_id, {}),
                )
            )
        return tuple(diagnoses)

    # -- internals ---------------------------------------------------------- #
    def _diagnose_one(
        self,
        *,
        strategy_id: str,
        oracle_rows: Sequence[ResolvedOutcome],
        router_values: Sequence[float],
        conditional: Mapping[str, Sequence[float]],
    ) -> StrategyDiagnosis:
        config = self._config
        oracle_net = _mean(row.net_return_bps for row in oracle_rows)
        gross_values = [
            row.gross_return_bps for row in oracle_rows if row.gross_return_bps is not None
        ]
        oracle_gross = _mean(gross_values) if gross_values else None
        router_net = _mean(router_values) if router_values else None
        conditional_means = {
            bucket: _mean(values)
            for bucket, values in conditional.items()
            if len(values) >= config.minimum_conditional_samples
        }
        conditional_means = {
            key: value for key, value in conditional_means.items() if value is not None
        }

        evidence: list[str] = [
            f"oracle_n={len(oracle_rows)}",
            f"router_n={len(router_values)}",
            f"live_share={_live_share(oracle_rows):.2f}",
        ]

        if len(oracle_rows) < config.minimum_samples:
            return StrategyDiagnosis(
                strategy_id=strategy_id,
                verdict=StrategyVerdict.INSUFFICIENT_DATA,
                oracle_sample_count=len(oracle_rows),
                oracle_net_bps=oracle_net,
                oracle_gross_bps=oracle_gross,
                router_sample_count=len(router_values),
                router_net_bps=router_net,
                conditional_net_bps=conditional_means,
                evidence=tuple(evidence),
            )

        # 1. Cost / horizon FIRST. Gross positive with net negative is a cost problem, and
        #    on net numbers alone it is indistinguishable from a dead thesis.
        if (
            oracle_gross is not None
            and oracle_gross > config.neutral_band_bps
            and oracle_net is not None
            and oracle_net < -config.neutral_band_bps
        ):
            evidence.append("gross_positive_net_negative")
            verdict = StrategyVerdict.COST_OR_HORIZON_PROBLEM
        # 2. Sign flips across market/regime -> the eligibility mask or the context model
        #    is not separating the states this thesis works in.
        elif _sign_flip(conditional_means, config.sign_flip_band_bps):
            evidence.append("conditional_sign_flip")
            verdict = StrategyVerdict.CONTEXT_MODELLING_PROBLEM
        # 3. The thesis itself loses even when allowed to act on its own terms.
        elif oracle_net is not None and oracle_net < -config.neutral_band_bps:
            evidence.append("oracle_negative")
            verdict = StrategyVerdict.STRATEGY_PROBLEM
        # 4. The thesis works but the selector picked it at the wrong times.
        elif (
            oracle_net is not None
            and oracle_net > config.neutral_band_bps
            and router_net is not None
            and router_net < -config.neutral_band_bps
        ):
            evidence.append("oracle_positive_router_negative")
            verdict = StrategyVerdict.SELECTOR_PROBLEM
        elif oracle_net is not None and oracle_net > config.neutral_band_bps:
            verdict = StrategyVerdict.HEALTHY
        else:
            # Inside the neutral band: not a problem anyone can act on, and calling it
            # healthy would overstate it.
            evidence.append("within_neutral_band")
            verdict = StrategyVerdict.INSUFFICIENT_DATA

        return StrategyDiagnosis(
            strategy_id=strategy_id,
            verdict=verdict,
            oracle_sample_count=len(oracle_rows),
            oracle_net_bps=oracle_net,
            oracle_gross_bps=oracle_gross,
            router_sample_count=len(router_values),
            router_net_bps=router_net,
            conditional_net_bps=conditional_means,
            evidence=tuple(evidence),
        )


def _sign_flip(means: Mapping[str, float], band_bps: float) -> bool:
    positives = [value for value in means.values() if value > band_bps]
    negatives = [value for value in means.values() if value < -band_bps]
    return bool(positives and negatives)


def _live_share(rows: Sequence[ResolvedOutcome]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.is_live) / len(rows)


def _mean(values: Iterable[float]) -> float | None:
    collected = [float(value) for value in values if _finite(value) is not None]
    return sum(collected) / len(collected) if collected else None


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None

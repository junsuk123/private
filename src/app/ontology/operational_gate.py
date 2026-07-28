from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping
from uuid import uuid4

from app.trading.contracts import OntologyDecision


@dataclass(frozen=True)
class OperationalFact:
    name: str
    value: bool | float | int | str
    observed_at: datetime
    valid_from: datetime
    valid_until: datetime
    source: str
    confidence: float

    def valid_at(self, as_of: datetime) -> bool:
        return (
            self.observed_at <= as_of
            and self.valid_from <= as_of <= self.valid_until
            and 0 <= self.confidence <= 1
        )


@dataclass(frozen=True)
class StrategyGateRule:
    strategy_id: str
    required_true: tuple[str, ...] = ()
    required_false: tuple[str, ...] = ()
    minimum_confidence: float = 0.0
    compatibility_weights: Mapping[str, float] = ()


@dataclass(frozen=True)
class OperationalOntologySnapshot:
    snapshot_id: str
    symbol: str
    as_of: datetime
    valid_until: datetime
    facts: Mapping[str, OperationalFact]

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        if self.valid_until < self.as_of:
            errors.append("SNAPSHOT_EXPIRED")
        for name, fact in self.facts.items():
            if name != fact.name:
                errors.append(f"FACT_NAME_MISMATCH:{name}")
            if not fact.valid_at(self.as_of):
                errors.append(f"FACT_STALE_OR_NOT_YET_VALID:{name}")
        return tuple(errors)


class ClosedWorldOntologyGate:
    """Deterministic gate over a validated point-in-time operational snapshot."""

    def evaluate(
        self,
        snapshot: OperationalOntologySnapshot,
        rules: tuple[StrategyGateRule, ...],
    ) -> OntologyDecision:
        validation_errors = snapshot.validate()
        allowed: list[str] = []
        blocked: dict[str, tuple[str, ...]] = {}
        compatibility: dict[str, float] = {}
        explanations: dict[str, tuple[str, ...]] = {}

        for rule in rules:
            reasons: list[str] = list(validation_errors)
            paths: list[str] = []
            scores: list[tuple[float, float]] = []
            for fact_name in rule.required_true:
                fact = snapshot.facts.get(fact_name)
                if fact is None:
                    reasons.append(f"MISSING_REQUIRED_FACT:{fact_name}")
                elif not fact.valid_at(snapshot.as_of):
                    reasons.append(f"STALE_REQUIRED_FACT:{fact_name}")
                elif fact.confidence < rule.minimum_confidence:
                    reasons.append(f"LOW_CONFIDENCE_FACT:{fact_name}")
                elif fact.value is not True:
                    reasons.append(f"REQUIRED_TRUE_FAILED:{fact_name}")
                else:
                    paths.append(f"{fact_name}=true@{fact.source}")
            for fact_name in rule.required_false:
                fact = snapshot.facts.get(fact_name)
                if fact is None:
                    reasons.append(f"MISSING_REQUIRED_FACT:{fact_name}")
                elif not fact.valid_at(snapshot.as_of):
                    reasons.append(f"STALE_REQUIRED_FACT:{fact_name}")
                elif fact.value is not False:
                    reasons.append(f"REQUIRED_FALSE_FAILED:{fact_name}")
                else:
                    paths.append(f"{fact_name}=false@{fact.source}")
            for name, weight in dict(rule.compatibility_weights).items():
                fact = snapshot.facts.get(name)
                if fact and fact.valid_at(snapshot.as_of):
                    numeric = (
                        float(fact.value)
                        if isinstance(fact.value, (int, float))
                        else 1.0 if fact.value is True else 0.0
                    )
                    scores.append((max(0.0, min(1.0, numeric)), max(0.0, float(weight))))
            score_weight = sum(weight for _, weight in scores)
            compatibility[rule.strategy_id] = (
                sum(value * weight for value, weight in scores) / score_weight
                if score_weight > 0
                else 1.0 if not reasons else 0.0
            )
            explanations[rule.strategy_id] = tuple(paths)
            if reasons:
                blocked[rule.strategy_id] = tuple(dict.fromkeys(reasons))
            else:
                allowed.append(rule.strategy_id)

        return OntologyDecision(
            snapshot_id=snapshot.snapshot_id,
            as_of=snapshot.as_of,
            symbol=snapshot.symbol,
            allowed_strategy_ids=tuple(allowed),
            blocked_strategy_reasons=blocked,
            compatibility_scores=compatibility,
            explanation_paths=explanations,
            valid_until=snapshot.valid_until,
        )


def new_snapshot_id() -> str:
    return f"ontology-{uuid4().hex}"

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ShadowDecision:
    path: str
    action: str
    strategy_id: str | None
    utility: float | None
    reason_codes: tuple[str, ...]
    probability_success: float | None = None
    expected_net_return_bps: float | None = None
    expected_cost_bps: float | None = None
    total_uncertainty: float | None = None
    ontology_compatibility: float | None = None
    realtime_trust_score: float | None = None
    realtime_trust_samples: int | None = None
    validation_strategy_id: str | None = None
    checkpoint_hash: str | None = None


@dataclass(frozen=True)
class ShadowComparison:
    correlation_id: str
    symbol: str
    as_of: datetime
    decisions: tuple[ShadowDecision, ...]
    action_agreement: bool
    strategy_agreement: bool
    utility_spread: float | None
    validation_candidates: tuple[ShadowDecision, ...] = ()


class ShadowComparisonRecorder:
    """Comparison-only telemetry. It deliberately exposes no broker dependency."""

    def __init__(self, path: str | Path = "logs/refactor-shadow-comparison.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def compare(
        self,
        *,
        correlation_id: str,
        symbol: str,
        as_of: datetime,
        decisions: tuple[ShadowDecision, ...],
        validation_candidates: tuple[ShadowDecision, ...] = (),
    ) -> ShadowComparison:
        actions = {item.action for item in decisions}
        strategies = {item.strategy_id for item in decisions}
        utilities = [item.utility for item in decisions if item.utility is not None]
        comparison = ShadowComparison(
            correlation_id=correlation_id,
            symbol=symbol,
            as_of=as_of,
            decisions=decisions,
            action_agreement=len(actions) <= 1,
            strategy_agreement=len(strategies) <= 1,
            utility_spread=max(utilities) - min(utilities) if utilities else None,
            validation_candidates=validation_candidates,
        )
        payload = asdict(comparison)
        payload["as_of"] = as_of.isoformat()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return comparison

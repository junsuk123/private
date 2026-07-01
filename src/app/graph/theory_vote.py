from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ACTION_NAMES = ("BUY", "SELL", "HOLD", "REDUCE", "WATCH")


@dataclass(frozen=True)
class TheoryVote:
    ticker: str
    theory_id: str
    theory_family: str
    style: str
    action: str
    horizon_bucket: str
    expected_holding_minutes: int
    raw_signal: float
    confidence: float
    expected_net_return: float | None = None
    evidence_cluster_id: str = "unknown_cluster"
    regime_gate: float = 1.0
    data_quality_weight: float = 1.0
    validation_weight: float = 1.0
    horizon_compatibility: float = 1.0
    conflicts: tuple[str, ...] = ()
    explanation: str = ""
    evidence_ids: tuple[str, ...] = ()

    @property
    def normalized_action(self) -> str:
        action = str(self.action).upper()
        return action if action in ACTION_NAMES else "WATCH"

    @property
    def effective_weight(self) -> float:
        weight = (
            self.raw_signal
            * self.confidence
            * self.regime_gate
            * self.data_quality_weight
            * self.validation_weight
            * self.horizon_compatibility
        )
        return max(0.0, float(weight))

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "theory_id": self.theory_id,
            "theory_family": self.theory_family,
            "style": self.style,
            "action": self.normalized_action,
            "horizon_bucket": self.horizon_bucket,
            "expected_holding_minutes": self.expected_holding_minutes,
            "raw_signal": self.raw_signal,
            "confidence": self.confidence,
            "expected_net_return": self.expected_net_return,
            "evidence_cluster_id": self.evidence_cluster_id,
            "regime_gate": self.regime_gate,
            "data_quality_weight": self.data_quality_weight,
            "validation_weight": self.validation_weight,
            "horizon_compatibility": self.horizon_compatibility,
            "effective_weight": self.effective_weight,
            "conflicts": self.conflicts,
            "explanation": self.explanation,
            "evidence_ids": self.evidence_ids,
        }


@dataclass(frozen=True)
class EvidenceClusterVote:
    ticker: str
    cluster_id: str
    action: str
    raw_feature_count: int
    compressed_score: float
    theory_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "cluster_id": self.cluster_id,
            "action": self.action,
            "raw_feature_count": self.raw_feature_count,
            "compressed_score": self.compressed_score,
            "theory_ids": self.theory_ids,
        }


@dataclass(frozen=True)
class ConflictRecord:
    type: str
    theory_a: str
    theory_b: str
    penalty: float
    resolution: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "theory_a": self.theory_a,
            "theory_b": self.theory_b,
            "penalty": self.penalty,
            "resolution": self.resolution,
        }


@dataclass(frozen=True)
class ActionScore:
    action: str
    score: float
    supporting_theories: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "score": self.score,
            "supporting_theories": self.supporting_theories,
        }


@dataclass(frozen=True)
class PositionContext:
    has_position: bool = False
    current_quantity: int = 0
    average_price: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "has_position": self.has_position,
            "current_quantity": self.current_quantity,
            "average_price": self.average_price,
        }


@dataclass(frozen=True)
class FinalActionDecision:
    ticker: str
    selected_action: str
    scores: dict[str, float]
    decision_margin: float
    dominant_theories: tuple[TheoryVote, ...] = ()
    conflicts: tuple[ConflictRecord, ...] = ()
    evidence_clusters: tuple[EvidenceClusterVote, ...] = ()
    position_context: PositionContext = field(default_factory=PositionContext)
    final_explanation: str = ""
    npu_accelerated: bool = False
    npu_profile: dict[str, Any] = field(default_factory=dict)

    @property
    def buy_score(self) -> float:
        return float(self.scores.get("BUY", 0.0))

    @property
    def sell_score(self) -> float:
        return float(self.scores.get("SELL", 0.0))

    @property
    def hold_score(self) -> float:
        return float(self.scores.get("HOLD", 0.0))

    @property
    def reduce_score(self) -> float:
        return float(self.scores.get("REDUCE", 0.0))

    @property
    def watch_score(self) -> float:
        return float(self.scores.get("WATCH", 0.0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "selected_action": self.selected_action,
            "scores": {
                "buy_score": self.buy_score,
                "sell_score": self.sell_score,
                "hold_score": self.hold_score,
                "reduce_score": self.reduce_score,
                "watch_score": self.watch_score,
            },
            "decision_margin": self.decision_margin,
            "dominant_theories": tuple(vote.as_dict() for vote in self.dominant_theories),
            "conflicts": tuple(conflict.as_dict() for conflict in self.conflicts),
            "evidence_clusters": tuple(cluster.as_dict() for cluster in self.evidence_clusters),
            "position_context": self.position_context.as_dict(),
            "final_explanation": self.final_explanation,
            "npu_accelerated": self.npu_accelerated,
            "npu_profile": dict(self.npu_profile),
        }

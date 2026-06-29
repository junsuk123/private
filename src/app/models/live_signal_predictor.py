from __future__ import annotations

import math
from dataclasses import dataclass

from app.features.live_feature_frame import LiveFeatureFrame
from app.models.model_artifact_registry import ModelArtifactRegistry


@dataclass(frozen=True)
class LiveSignalPrediction:
    probability_success: float
    expected_net_return_bps: float
    uncertainty_score: float
    approved: bool
    reason_codes: tuple[str, ...]
    model_artifact_id: str
    feature_schema_hash: str


class LiveSignalPredictor:
    def __init__(self, registry: ModelArtifactRegistry | None = None) -> None:
        self.registry = registry or ModelArtifactRegistry()

    def predict(self, frame: LiveFeatureFrame) -> LiveSignalPrediction:
        artifact = self.registry.load_latest_live_eligible()
        if artifact.feature_schema_hash != frame.feature_schema_hash:
            raise RuntimeError("MODEL_FEATURE_SCHEMA_MISMATCH")
        if artifact.feature_names != frame.schema.feature_names:
            raise RuntimeError("MODEL_FEATURE_ORDER_MISMATCH")
        score = _dot(frame.values, artifact.weights) + artifact.bias
        probability = _sigmoid(score)
        expected = _dot(frame.values, artifact.expected_return_weights) + artifact.expected_return_bias
        uncertainty = 1.0 - abs(probability - 0.5) * 2.0
        reasons: list[str] = []
        if probability < artifact.thresholds["minimum_probability_success"]:
            reasons.append("PROBABILITY_BELOW_THRESHOLD")
        if expected < artifact.thresholds["minimum_expected_net_return_bps"]:
            reasons.append("EXPECTED_NET_RETURN_BELOW_THRESHOLD")
        if uncertainty > artifact.thresholds["maximum_uncertainty"]:
            reasons.append("UNCERTAINTY_TOO_HIGH")
        return LiveSignalPrediction(
            probability_success=probability,
            expected_net_return_bps=expected,
            uncertainty_score=uncertainty,
            approved=not reasons,
            reason_codes=tuple(reasons),
            model_artifact_id=artifact.artifact_id,
            feature_schema_hash=frame.feature_schema_hash,
        )


def _dot(values: tuple[float, ...], weights: tuple[float, ...]) -> float:
    return sum(value * weight for value, weight in zip(values, weights, strict=True))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class ModelHealthStatus(StrEnum):
    READY = "ready"
    FALLBACK_USED = "fallback_used"
    UNAVAILABLE = "unavailable"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class ModelHealthAssessment:
    status: ModelHealthStatus
    fallback_allowed: bool
    reason_codes: tuple[str, ...]
    confidence_penalty: float = 0.0
    model_artifact_id: str | None = None

    @property
    def model_ok(self) -> bool:
        return self.status == ModelHealthStatus.READY

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_model_health(*, prediction: Any | None, prediction_error: Exception | None, fallback_allowed: bool) -> ModelHealthAssessment:
    if prediction is not None and bool(getattr(prediction, "approved", False)):
        uncertainty = float(getattr(prediction, "uncertainty_score", 1.0) or 0.0)
        if uncertainty > 0.7:
            return ModelHealthAssessment(
                status=ModelHealthStatus.UNCERTAIN,
                fallback_allowed=fallback_allowed,
                reason_codes=("MODEL_UNCERTAIN",),
                confidence_penalty=0.2,
                model_artifact_id=str(getattr(prediction, "model_artifact_id", "") or None),
            )
        return ModelHealthAssessment(
            status=ModelHealthStatus.READY,
            fallback_allowed=fallback_allowed,
            reason_codes=("MODEL_READY",),
            confidence_penalty=0.0,
            model_artifact_id=str(getattr(prediction, "model_artifact_id", "") or None),
        )

    if prediction_error is not None:
        return ModelHealthAssessment(
            status=ModelHealthStatus.UNAVAILABLE,
            fallback_allowed=fallback_allowed,
            reason_codes=("MODEL_UNAVAILABLE", str(prediction_error)),
            confidence_penalty=0.35,
        )

    return ModelHealthAssessment(
        status=ModelHealthStatus.UNAVAILABLE,
        fallback_allowed=fallback_allowed,
        reason_codes=("MODEL_UNAVAILABLE",),
        confidence_penalty=0.35,
    )

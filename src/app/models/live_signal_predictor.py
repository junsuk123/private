from __future__ import annotations

import math
import os
from dataclasses import dataclass

from app.config import LiveConfigError, load_live_trading_safety_config
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
    # Provider/backend visibility (Phase 5). `provider` names what produced this
    # prediction ("trained_model" for the fitted live-eligible artifact; other
    # values are reserved for callers that synthesize a placeholder). `is_fallback`
    # is True whenever the prediction did NOT come from the trained model.
    # Both default so existing constructors/tests stay backward compatible.
    provider: str = "trained_model"
    is_fallback: bool = False


class LiveSignalPredictor:
    def __init__(self, registry: ModelArtifactRegistry | None = None) -> None:
        self.registry = registry or ModelArtifactRegistry()
        self._market_registries: dict[str, ModelArtifactRegistry] = {}

    def _registry_for(self, symbol: str) -> ModelArtifactRegistry:
        """Prefer the market-specific artifact, fall back to the combined one.

        KR and US differ by 2-3x in round-trip cost, so one expected-net-return
        head fitted across both is wrong for each. A market artifact is used only
        when it exists and is live-eligible; otherwise the combined model serves,
        so enabling the split never opens a coverage gap.
        """
        if not _market_split_enabled():
            return self.registry
        market = "KR" if str(symbol or "").strip().isdigit() and len(str(symbol).strip()) == 6 else "US"
        cached = self._market_registries.get(market)
        if cached is None:
            cached = ModelArtifactRegistry(self.registry.root / market)
            self._market_registries[market] = cached
        return cached

    def predict(self, frame: LiveFeatureFrame) -> LiveSignalPrediction:
        if not live_signal_model_inference_enabled():
            raise RuntimeError("LIVE_SIGNAL_MODEL_INFERENCE_DISABLED")
        market_registry = self._registry_for(getattr(frame, "symbol", ""))
        try:
            artifact = market_registry.load_latest_live_eligible()
        except RuntimeError:
            if market_registry is self.registry:
                raise
            artifact = self.registry.load_latest_live_eligible()
        if artifact.feature_schema_hash != frame.feature_schema_hash:
            raise RuntimeError("MODEL_FEATURE_SCHEMA_MISMATCH")
        if artifact.feature_names != frame.schema.feature_names:
            raise RuntimeError("MODEL_FEATURE_ORDER_MISMATCH")
        score = _dot(frame.values, artifact.weights) + artifact.bias
        probability = _sigmoid(score)
        expected = _dot(frame.values, artifact.expected_return_weights) + artifact.expected_return_bias
        uncertainty = 1.0 - abs(probability - 0.5) * 2.0
        thresholds = _prediction_thresholds(artifact.thresholds)
        reasons: list[str] = []
        if probability < thresholds["minimum_probability_success"]:
            reasons.append("PROBABILITY_BELOW_THRESHOLD")
        if expected < thresholds["minimum_expected_net_return_bps"]:
            reasons.append("EXPECTED_NET_RETURN_BELOW_THRESHOLD")
        if uncertainty > thresholds["maximum_uncertainty"]:
            reasons.append("UNCERTAINTY_TOO_HIGH")
        return LiveSignalPrediction(
            probability_success=probability,
            expected_net_return_bps=expected,
            uncertainty_score=uncertainty,
            approved=not reasons,
            reason_codes=tuple(reasons),
            model_artifact_id=artifact.artifact_id,
            feature_schema_hash=frame.feature_schema_hash,
            provider="trained_model",
            is_fallback=False,
        )


def _market_split_enabled() -> bool:
    return str(os.getenv("LIVE_MODEL_SPLIT_BY_MARKET", "true")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _dot(values: tuple[float, ...], weights: tuple[float, ...]) -> float:
    return sum(value * weight for value, weight in zip(values, weights, strict=True))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))


def _prediction_thresholds(artifact_thresholds: dict[str, float]) -> dict[str, float]:
    thresholds = dict(artifact_thresholds)
    try:
        safety = load_live_trading_safety_config()
    except LiveConfigError:
        return thresholds
    # Merge artifact thresholds with the safety config by taking the STRICTER bound in
    # each direction, so the safety floor can only tighten (never loosen) the gate:
    #   * minimum floors (probability, expected net return) -> the HIGHER (max)
    #   * maximum ceilings (uncertainty) -> the LOWER (min)
    thresholds["minimum_probability_success"] = max(
        thresholds.get("minimum_probability_success", safety.minimum_probability_success),
        safety.minimum_probability_success,
    )
    thresholds["minimum_expected_net_return_bps"] = max(
        thresholds.get("minimum_expected_net_return_bps", safety.minimum_expected_net_return_bps),
        safety.minimum_expected_net_return_bps,
    )
    thresholds["maximum_uncertainty"] = min(
        thresholds.get("maximum_uncertainty", 0.48),
        1.0 - max(0.0, min(1.0, safety.minimum_model_confidence)),
    )
    return thresholds


def live_signal_model_inference_enabled() -> bool:
    return os.getenv("LIVE_SIGNAL_MODEL_INFERENCE_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

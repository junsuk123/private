from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

FEATURE_SCHEMA = (
    "impact_score",
    "event_count",
    "quote_count",
    "execution_count",
    "macro_count",
    "signal_confidence",
    "signal_score",
    "rsi",
    "volume_ratio",
    "volatility",
    "macro_risk",
    "support_score",
    "risk_score",
    "momentum_score",
    "liquidity_score",
    "confidence_score",
)
OUTPUT_SCHEMA = (
    "expected_return_5s",
    "expected_return_15s",
    "expected_return_60s",
    "downside_risk",
    "prediction_confidence",
)


@dataclass(frozen=True)
class ShortHorizonPrediction:
    expected_return_5s: float
    expected_return_15s: float
    expected_return_60s: float
    downside_risk: float
    prediction_confidence: float
    provider: str
    device: str

    @property
    def is_low_confidence(self) -> bool:
        return self.prediction_confidence < 0.50


class ShortHorizonNpuPredictor:
    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path or os.getenv("SHORT_HORIZON_MODEL_PATH", "models/short_horizon/openvino_model.xml"))
        self.device = device or os.getenv("SHORT_HORIZON_PREDICTOR_DEVICE", "AUTO")
        self._compiled = None
        self._fallback_reason: str | None = None

    def predict(self, features: Mapping[str, float] | Sequence[float]) -> ShortHorizonPrediction:
        vector = normalize_features(features)
        output = self._infer(vector)
        return ShortHorizonPrediction(
            expected_return_5s=float(output[0]),
            expected_return_15s=float(output[1]),
            expected_return_60s=float(output[2]),
            downside_risk=max(0.0, float(output[3])),
            prediction_confidence=max(0.0, min(1.0, float(output[4]))),
            provider="openvino" if self._compiled is not None else "linear_baseline",
            device=self.device if self._compiled is not None else "CPU_NUMPY",
        )

    def _infer(self, vector: np.ndarray) -> np.ndarray:
        if self.model_path.exists():
            try:
                compiled = self._compiled_model()
                output = compiled([vector.reshape(1, -1)])[0]
                return np.asarray(output, dtype=np.float32).reshape(-1)[: len(OUTPUT_SCHEMA)]
            except Exception as exc:
                self._fallback_reason = str(exc)
        return _linear_baseline(vector)

    def _compiled_model(self):
        if self._compiled is not None:
            return self._compiled
        import openvino as ov

        core = ov.Core()
        model = core.read_model(str(self.model_path))
        self._compiled = core.compile_model(model, self.device)
        return self._compiled


def normalize_features(features: Mapping[str, float] | Sequence[float]) -> np.ndarray:
    if isinstance(features, Mapping):
        values = [float(features.get(name, 0.0) or 0.0) for name in FEATURE_SCHEMA]
    else:
        values = [float(value or 0.0) for value in features]
    vector = np.zeros((len(FEATURE_SCHEMA),), dtype=np.float32)
    vector[: min(len(values), len(FEATURE_SCHEMA))] = values[: len(FEATURE_SCHEMA)]
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def predictor_enabled() -> bool:
    return os.getenv("SHORT_HORIZON_PREDICTOR_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def _linear_baseline(vector: np.ndarray) -> np.ndarray:
    signal = vector[5] * 0.004 + vector[6] * 0.002 + vector[11] * 0.003 + vector[13] * 0.002
    risk = max(0.0, vector[9] * 0.03 + vector[10] * 0.02 + vector[12] * 0.015)
    confidence = max(0.0, min(1.0, 0.35 + vector[5] * 0.25 + vector[15] * 0.20 - risk))
    return np.array(
        [
            signal * 0.35,
            signal * 0.70,
            signal,
            risk,
            confidence,
        ],
        dtype=np.float32,
    )

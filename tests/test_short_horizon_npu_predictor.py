from __future__ import annotations

import numpy as np

from app.realtime.short_horizon_npu_predictor import (
    OUTPUT_SCHEMA,
    ShortHorizonNpuPredictor,
    normalize_features,
)


def test_short_horizon_predictor_schema_without_model(tmp_path) -> None:
    predictor = ShortHorizonNpuPredictor(model_path=tmp_path / "missing.xml", device="NPU")

    prediction = predictor.predict({"signal_confidence": 0.7, "signal_score": 1.2, "confidence_score": 0.8})

    assert len(OUTPUT_SCHEMA) == 5
    assert prediction.provider == "linear_baseline"
    assert prediction.device == "CPU_NUMPY"
    assert 0.0 <= prediction.prediction_confidence <= 1.0


def test_normalize_features_handles_missing_nan_and_inf() -> None:
    vector = normalize_features([np.nan, np.inf, -np.inf])

    assert vector.shape == (16,)
    assert np.isfinite(vector).all()

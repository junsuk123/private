from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.npu.runtime_manager import NpuModuleStatus, NpuRuntimeManager, get_npu_runtime_manager


SHORT_HORIZON_OUTPUT_SCHEMA = (
    "forward_return_5m",
    "forward_return_15m",
    "probability_net_positive",
    "downside_risk_proxy",
    "uncertainty",
)


@dataclass(frozen=True)
class ShortHorizonPredictionBatch:
    predictions: np.ndarray
    score_schema: tuple[str, ...]
    status: NpuModuleStatus


class NpuShortHorizonPredictor:
    def __init__(self, runtime: NpuRuntimeManager | None = None, *, enabled: bool = True) -> None:
        self.runtime = runtime or get_npu_runtime_manager()
        self.enabled = enabled

    def predict_matrix(self, feature_matrix: np.ndarray) -> ShortHorizonPredictionBatch:
        matrix = np.asarray(feature_matrix, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("feature_matrix must be 2D")
        weights = np.zeros((matrix.shape[1], len(SHORT_HORIZON_OUTPUT_SCHEMA)), dtype=np.float32)
        weights[: min(matrix.shape[1], 6), :] = np.array(
            [
                [0.25, 0.16, 0.18, -0.02, -0.04],
                [0.16, 0.28, 0.14, -0.02, -0.04],
                [0.04, 0.08, 0.18, 0.00, -0.03],
                [-0.05, -0.08, -0.04, 0.30, 0.18],
                [0.06, 0.06, 0.10, -0.12, -0.04],
                [0.08, 0.10, 0.14, -0.08, -0.04],
            ],
            dtype=np.float32,
        )[: min(matrix.shape[1], 6), :]
        bias = np.array([0.0, 0.0, 0.50, 0.02, 0.20], dtype=np.float32)
        output, status = self.runtime.run_linear(
            module_name="short_horizon_predictor",
            features=matrix,
            weights=weights,
            bias=bias,
            activation="linear",
            enabled=self.enabled,
        )
        output[:, 2] = np.clip(output[:, 2], 0.0, 1.0)
        output[:, 3] = np.maximum(output[:, 3], 0.0)
        output[:, 4] = np.clip(output[:, 4], 0.0, 1.0)
        return ShortHorizonPredictionBatch(output, SHORT_HORIZON_OUTPUT_SCHEMA, status)

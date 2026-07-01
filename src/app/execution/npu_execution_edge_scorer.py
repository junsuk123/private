from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.npu.runtime_manager import NpuModuleStatus, NpuRuntimeManager, get_npu_runtime_manager
from app.npu.tensor_schemas import get_tensor_schema


EXECUTION_EDGE_SCHEMA = (
    "fill_probability",
    "slippage_penalty",
    "adverse_selection_penalty",
    "expected_net_edge",
    "execution_confidence",
)


@dataclass(frozen=True)
class ExecutionEdgeScoreBatch:
    scores: np.ndarray
    score_schema: tuple[str, ...]
    status: NpuModuleStatus


class NpuExecutionEdgeScorer:
    def __init__(self, runtime: NpuRuntimeManager | None = None, *, enabled: bool = True) -> None:
        self.runtime = runtime or get_npu_runtime_manager()
        self.enabled = enabled
        self.schema = get_tensor_schema("execution_edge_features")

    def score_matrix(self, execution_features: np.ndarray) -> ExecutionEdgeScoreBatch:
        matrix = np.asarray(execution_features, dtype=np.float32)
        self.schema.validate(matrix)
        weights = np.array(
            [
                [0.10, 0.00, 0.00, 0.95, 0.20],
                [0.00, 0.10, 0.00, -0.10, 0.00],
                [0.00, 0.05, 0.00, -0.05, 0.00],
                [-0.30, 0.85, 0.10, -0.75, -0.20],
                [-0.20, 0.70, 0.10, -0.55, -0.15],
                [0.45, -0.20, -0.10, 0.15, 0.35],
                [0.20, -0.05, -0.10, 0.05, 0.12],
                [0.15, -0.04, 0.08, 0.05, 0.10],
                [-0.35, 0.15, 0.80, -0.45, -0.25],
                [-0.10, 0.20, 0.10, -0.20, -0.08],
            ],
            dtype=np.float32,
        )
        bias = np.array([0.50, 0.0, 0.0, 0.0, 0.35], dtype=np.float32)
        output, status = self.runtime.run_linear(
            module_name="execution_edge_scorer",
            features=matrix,
            weights=weights,
            bias=bias,
            activation="linear",
            enabled=self.enabled,
        )
        output[:, 0] = np.clip(output[:, 0], 0.0, 1.0)
        output[:, 1] = np.maximum(output[:, 1], 0.0)
        output[:, 2] = np.maximum(output[:, 2], 0.0)
        output[:, 4] = np.clip(output[:, 4], 0.0, 1.0)
        return ExecutionEdgeScoreBatch(output, EXECUTION_EDGE_SCHEMA, status)

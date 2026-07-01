from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.npu.runtime_manager import NpuModuleStatus, NpuRuntimeManager, get_npu_runtime_manager
from app.npu.tensor_schemas import get_tensor_schema


CONFLICT_SCORE_SCHEMA = ("conflict_penalty", "horizon_penalty", "style_penalty", "regime_penalty")


@dataclass(frozen=True)
class ConflictScoreBatch:
    penalties: np.ndarray
    score_schema: tuple[str, ...]
    status: NpuModuleStatus


class NpuConflictScorer:
    def __init__(self, runtime: NpuRuntimeManager | None = None, *, enabled: bool = True) -> None:
        self.runtime = runtime or get_npu_runtime_manager()
        self.enabled = enabled
        self.schema = get_tensor_schema("conflict_features")

    def score_matrix(self, conflict_features: np.ndarray) -> ConflictScoreBatch:
        matrix = np.asarray(conflict_features, dtype=np.float32)
        self.schema.validate(matrix)
        weights = np.array(
            [
                [0.30, 0.00, 0.45, 0.00],
                [0.20, 0.55, 0.00, 0.00],
                [0.30, 0.00, 0.25, 0.00],
                [0.15, 0.00, 0.00, 0.60],
                [0.12, 0.00, 0.00, 0.00],
                [0.10, 0.00, 0.00, 0.20],
                [0.16, 0.00, 0.00, 0.10],
                [0.20, 0.10, 0.00, 0.10],
            ],
            dtype=np.float32,
        )
        output, status = self.runtime.run_linear(
            module_name="conflict_scorer",
            features=matrix,
            weights=weights,
            activation="relu",
            enabled=self.enabled,
        )
        return ConflictScoreBatch(np.minimum(output, 1.0), CONFLICT_SCORE_SCHEMA, status)

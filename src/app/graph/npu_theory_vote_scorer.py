from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app.graph.theory_vote import TheoryVote
from app.npu.runtime_manager import NpuModuleStatus, NpuRuntimeManager, get_npu_runtime_manager
from app.npu.tensor_schemas import get_tensor_schema


THEORY_VOTE_SCORE_SCHEMA = ("buy_vote", "sell_vote", "hold_vote", "reduce_vote", "watch_vote", "confidence", "uncertainty")


@dataclass(frozen=True)
class TheoryVoteScoreBatch:
    scores: np.ndarray
    score_schema: tuple[str, ...]
    status: NpuModuleStatus
    top_indices: tuple[int, ...]


class NpuTheoryVoteScorer:
    def __init__(self, runtime: NpuRuntimeManager | None = None, *, enabled: bool = True) -> None:
        self.runtime = runtime or get_npu_runtime_manager()
        self.enabled = enabled
        self.schema = get_tensor_schema("theory_vote_features")

    def score(self, votes: Sequence[TheoryVote], *, top_k: int | None = None) -> TheoryVoteScoreBatch:
        matrix = matrix_from_theory_votes(votes)
        scores, status = self.score_matrix(matrix)
        if len(scores) == 0:
            indices = ()
        else:
            actionable = np.max(scores[:, :5], axis=1)
            count = len(scores) if top_k is None else min(max(0, int(top_k)), len(scores))
            indices_array = np.argsort(-actionable)[:count]
            indices = tuple(int(index) for index in indices_array)
        return TheoryVoteScoreBatch(scores=scores, score_schema=THEORY_VOTE_SCORE_SCHEMA, status=status, top_indices=indices)

    def score_matrix(self, matrix: np.ndarray) -> tuple[np.ndarray, NpuModuleStatus]:
        self.schema.validate(matrix)
        weights = _weights()
        bias = np.array([0.0, 0.0, 0.05, 0.0, 0.02, 0.0, 0.15], dtype=np.float32)
        return self.runtime.run_linear(
            module_name="theory_vote_scorer",
            features=matrix,
            weights=weights,
            bias=bias,
            activation="relu",
            enabled=self.enabled,
        )


def matrix_from_theory_votes(votes: Sequence[TheoryVote]) -> np.ndarray:
    matrix = np.zeros((len(votes), len(get_tensor_schema("theory_vote_features").feature_names)), dtype=np.float32)
    for row, vote in enumerate(votes):
        action = vote.normalized_action
        position_weight = 1.0 if action in {"SELL", "REDUCE"} else 0.5
        matrix[row, :] = (
            vote.raw_signal,
            vote.validation_weight,
            vote.regime_gate,
            vote.data_quality_weight,
            vote.validation_weight,
            vote.horizon_compatibility,
            vote.effective_weight,
            vote.expected_net_return or 0.0,
            max(0.0, 1.0 - vote.confidence),
            position_weight,
        )
    return matrix


def _weights() -> np.ndarray:
    return np.array(
        [
            [0.30, 0.30, 0.04, 0.18, 0.08, 0.08, -0.04],
            [0.12, 0.12, 0.02, 0.06, 0.05, 0.05, -0.02],
            [0.12, 0.12, 0.04, 0.08, 0.06, 0.06, -0.02],
            [0.08, 0.08, 0.03, 0.06, 0.06, 0.06, -0.02],
            [0.16, 0.16, 0.04, 0.10, 0.06, 0.06, -0.02],
            [0.12, 0.12, 0.04, 0.10, 0.08, 0.08, -0.02],
            [0.30, 0.30, 0.02, 0.20, 0.10, 0.10, -0.03],
            [0.70, 0.20, 0.00, 0.10, 0.00, 0.20, -0.10],
            [-0.06, -0.06, 0.16, 0.03, 0.12, -0.04, 0.40],
            [0.05, 0.24, 0.04, 0.24, 0.04, 0.04, 0.00],
        ],
        dtype=np.float32,
    )

from __future__ import annotations

import unittest

import numpy as np

from app.execution.npu_execution_edge_scorer import NpuExecutionEdgeScorer
from app.graph.npu_conflict_scorer import NpuConflictScorer
from app.graph.npu_evidence_cluster_compressor import NpuEvidenceClusterCompressor
from app.graph.npu_theory_vote_scorer import NpuTheoryVoteScorer
from app.models.npu_short_horizon_predictor import NpuShortHorizonPredictor
from app.npu.runtime_manager import NpuRuntimeManager
from app.npu.tensor_schemas import get_tensor_schema


class ExpandedNpuModulesTest(unittest.TestCase):
    def test_npu_runtime_manager_cpu_fallback_for_small_batch(self) -> None:
        manager = NpuRuntimeManager(min_batch_for_npu=128)
        features = np.ones((2, 3), dtype=np.float32)
        weights = np.ones((3, 2), dtype=np.float32)

        output, status = manager.run_linear(module_name="unit", features=features, weights=weights)

        self.assertEqual(output.shape, (2, 2))
        self.assertEqual(status.backend, "CPU_NUMPY")
        self.assertFalse(status.uses_npu)

    def test_theory_vote_scorer_cpu_consistency(self) -> None:
        manager = NpuRuntimeManager(min_batch_for_npu=9999)
        matrix = np.ones((4, get_tensor_schema("theory_vote_features").feature_dim), dtype=np.float32) * 0.2

        scores, status = NpuTheoryVoteScorer(manager).score_matrix(matrix)

        self.assertEqual(scores.shape, (4, 7))
        self.assertFalse(status.uses_npu)

    def test_evidence_cluster_compressor_reduces_duplicate_votes(self) -> None:
        manager = NpuRuntimeManager(min_batch_for_npu=9999)
        features = np.ones((1, 3), dtype=np.float32)
        result = NpuEvidenceClusterCompressor(manager).compress(
            features,
            feature_names=("PriceAboveMA20", "MA20AboveMA60", "MACDBullishCross"),
        )

        trend = result.clusters[0, result.cluster_schema.index("trend_cluster")]
        self.assertGreater(trend, 1.0)
        self.assertFalse(result.status.uses_npu)

    def test_conflict_scorer_outputs_penalty_matrix(self) -> None:
        manager = NpuRuntimeManager(min_batch_for_npu=9999)
        features = np.zeros((3, get_tensor_schema("conflict_features").feature_dim), dtype=np.float32)
        features[:, 0] = 1.0
        result = NpuConflictScorer(manager).score_matrix(features)

        self.assertEqual(result.penalties.shape, (3, 4))
        self.assertGreater(result.penalties[0, 0], 0.0)

    def test_short_horizon_predictor_npu_fallback(self) -> None:
        manager = NpuRuntimeManager(min_batch_for_npu=9999)
        result = NpuShortHorizonPredictor(manager).predict_matrix(np.ones((2, 8), dtype=np.float32) * 0.1)

        self.assertEqual(result.predictions.shape, (2, 5))
        self.assertFalse(result.status.uses_npu)

    def test_execution_edge_scorer_penalizes_wide_spread(self) -> None:
        manager = NpuRuntimeManager(min_batch_for_npu=9999)
        schema = get_tensor_schema("execution_edge_features")
        low = np.zeros((1, schema.feature_dim), dtype=np.float32)
        high = np.zeros((1, schema.feature_dim), dtype=np.float32)
        high[0, schema.feature_names.index("spread_rate")] = 0.05

        low_score = NpuExecutionEdgeScorer(manager).score_matrix(low).scores[0, 3]
        high_score = NpuExecutionEdgeScorer(manager).score_matrix(high).scores[0, 3]

        self.assertLess(high_score, low_score)


if __name__ == "__main__":
    unittest.main()

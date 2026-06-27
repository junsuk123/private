from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.graph.npu_classifier import OntologyNpuLinearScorer
from app.models.inference_backend import OpenVinoNpuSignalModel


class InferenceBackendTest(unittest.TestCase):
    def test_inference_backend_cpu_fallback_without_openvino(self) -> None:
        weights = np.ones((2, 1), dtype=np.float32)
        model = OpenVinoNpuSignalModel(weights, requested_device="NPU")
        with patch.dict("sys.modules", {"openvino": None}):
            output = model.infer(np.array([[1.0, 2.0]], dtype=np.float32))

        status = model.status()
        self.assertEqual(float(output[0, 0]), 3.0)
        self.assertFalse(status.uses_npu)
        self.assertEqual(status.active_backend, "CPU_NUMPY")
        self.assertIsNotNone(status.fallback_reason)

    def test_ontology_npu_linear_scorer_reports_heuristic_model_kind(self) -> None:
        scorer = OntologyNpuLinearScorer(batch_size=128)
        status = scorer.status()

        self.assertEqual(status.model_kind, "heuristic_linear_scorer")


if __name__ == "__main__":
    unittest.main()

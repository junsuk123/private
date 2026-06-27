from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.evaluation import walk_forward_splits
from app.models.dataset_builder import DatasetRow
from app.models.evaluate_signal_model import evaluate_ranked_signal_rows
from app.models.train_signal_model import build_training_plan


class WalkForwardEvaluationTest(unittest.TestCase):
    def test_walk_forward_split_avoids_lookahead_overlap(self) -> None:
        rows = tuple(range(12))
        splits = walk_forward_splits(rows, train_size=5, test_size=2, step_size=2)

        self.assertTrue(splits)
        for split in splits:
            self.assertLess(max(split.train_indices), min(split.test_indices))

    def test_evaluation_summary_and_training_plan(self) -> None:
        now = datetime.now(timezone.utc)
        rows = tuple(
            DatasetRow(
                ticker="TEST",
                as_of=now + timedelta(minutes=index),
                features={"signal_score": float(index)},
                labels={"future_return_5d": 0.01 if index % 2 else -0.01},
                metadata={"no_lookahead": True, "risk_rejected": index == 0},
            )
            for index in range(6)
        )

        summary = evaluate_ranked_signal_rows(rows, k=3)
        plan = build_training_plan(rows)

        self.assertEqual(summary.row_count, 6)
        self.assertGreaterEqual(summary.precision_at_k, 0.0)
        self.assertTrue(plan.no_lookahead_verified)
        self.assertIn("signal_score", plan.feature_names)


if __name__ == "__main__":
    unittest.main()

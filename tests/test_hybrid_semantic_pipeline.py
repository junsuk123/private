from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.features import (
    CentroidAISemanticModel,
    HybridSemanticFeaturePipeline,
    OHLCVBar,
    default_ai_semantic_targets,
)
from app.models import DatasetBuilder, LabelConfig


class HybridSemanticPipelineTest(unittest.TestCase):
    def test_formula_and_ai_features_keep_separate_provenance(self) -> None:
        bars = _bars(70, 0.006)
        decision_times = tuple(bar.as_of for bar in bars[35:60:5])
        builder = DatasetBuilder(label_config=LabelConfig(horizon_bars=5, profit_taking=0.02, stop_loss=-0.05))
        examples = builder.build_ai_training_examples(
            bars,
            decision_times,
            {
                "AdaptiveBreakoutCandidate": "future_return_5d_above_2pct",
                "AdaptiveRiskOffCandidate": "triple_barrier_negative",
            },
        )
        model = CentroidAISemanticModel(default_ai_semantic_targets())
        model.fit(examples)
        pipeline = HybridSemanticFeaturePipeline(ai_numeric_model=model)

        snapshot = pipeline.build_snapshot(
            bars,
            documents=("The company announced a major supply contract and order backlog growth.",),
        )

        methods = {feature.generation_method for feature in snapshot.semantic_features}
        self.assertIn("formula_rule", methods)
        self.assertIn("ai_model", methods)
        self.assertIn("ai_text_proxy", methods)
        self.assertTrue(any(feature.model_version == model.model_version for feature in snapshot.semantic_features))
        self.assertGreater(len(snapshot.reasoning_paths), 0)

    def test_ai_training_examples_do_not_put_labels_into_inputs(self) -> None:
        bars = _bars(50, 0.003)
        examples = DatasetBuilder().build_ai_training_examples(
            bars,
            tuple(bar.as_of for bar in bars[30:35]),
            {"AdaptiveBreakoutCandidate": "future_return_5d_positive"},
        )

        self.assertGreater(len(examples), 0)
        self.assertIn("AdaptiveBreakoutCandidate", examples[0].labels)
        self.assertNotIn("future_return_5d", examples[0].inputs)
        self.assertNotIn("triple_barrier_label", examples[0].inputs)


def _bars(count: int, drift: float) -> tuple[OHLCVBar, ...]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    price = 100.0
    rows = []
    for i in range(count):
        shock = drift + (0.015 if i % 11 == 0 else 0)
        price *= 1 + shock
        volume = 1_000_000 * (3 if i % 11 == 0 else 1) + i * 10_000
        rows.append(
            OHLCVBar(
                ticker="TEST",
                as_of=start + timedelta(days=i),
                open=price * 0.99,
                high=price * 1.025,
                low=price * 0.985,
                close=price,
                volume=volume,
            )
        )
    return tuple(rows)


if __name__ == "__main__":
    unittest.main()

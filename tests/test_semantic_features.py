from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.features import RawIndicatorRecord, SemanticFeatureEngine
from app.graph.reasoning_rules import build_semantic_reasoning_paths
from app.graph.semantic_builder import build_semantic_feature_graph


class SemanticFeatureEngineTest(unittest.TestCase):
    def test_threshold_mappings_generate_ontology_ready_features(self) -> None:
        records = (
            _raw("TEST", "rsi_14", 75.0),
            _raw("TEST", "return_5d", 0.06),
            _raw("TEST", "volume_spike_ratio", 2.5),
            _raw("TEST", "return_1d", 0.02),
            _raw("TEST", "bollinger_band_width_20", 0.04),
        )

        features = SemanticFeatureEngine().generate(records)
        names = {feature.feature_name for feature in features}

        self.assertIn("RangeOverbought", names)
        self.assertIn("ShortTermMomentumPositive", names)
        self.assertIn("VolumeSpike", names)
        self.assertIn("VolumeBackedRise", names)
        self.assertIn("BollingerSqueeze", names)
        self.assertTrue(all(0 <= feature.confidence <= 1 for feature in features))
        self.assertTrue(all(feature.ontology_node_id for feature in features))

    def test_graph_and_reasoning_records_are_generated(self) -> None:
        records = (
            _raw("TEST", "return_5d", 0.06),
            _raw("TEST", "volume_spike_ratio", 2.5),
            _raw("TEST", "return_1d", 0.02),
        )
        features = SemanticFeatureEngine().generate(records)

        graph = build_semantic_feature_graph(records, features)
        paths = build_semantic_reasoning_paths(features)

        self.assertGreater(len(graph.matching(subject="TEST", predicate="generatesSemanticFeature")), 0)
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].ticker, "TEST")
        self.assertGreaterEqual(paths[0].final_confidence, 0.0)

    def test_missing_volume_ratio_with_positive_return_does_not_crash(self) -> None:
        records = (
            _raw("TEST", "volume_spike_ratio", None),
            _raw("TEST", "return_1d", 0.02),
        )

        features = SemanticFeatureEngine().generate(records)
        names = {feature.feature_name for feature in features}

        self.assertNotIn("VolumeSpike", names)
        self.assertNotIn("VolumeBackedRise", names)

    def test_missing_volume_ratio_with_negative_return_does_not_crash(self) -> None:
        records = (
            _raw("TEST", "volume_spike_ratio", None),
            _raw("TEST", "return_1d", -0.02),
        )

        features = SemanticFeatureEngine().generate(records)
        names = {feature.feature_name for feature in features}

        self.assertNotIn("VolumeSpike", names)
        self.assertNotIn("VolumeBackedFall", names)


def _raw(ticker: str, name: str, value: float | None) -> RawIndicatorRecord:
    return RawIndicatorRecord(
        ticker=ticker,
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        indicator_name=name,
        value=value,
        unit="ratio",
        lookback_window=None,
        source="test",
        calculation_version="test",
    )


if __name__ == "__main__":
    unittest.main()

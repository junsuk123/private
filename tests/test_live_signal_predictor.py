from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.features.feature_provenance import FeatureProvenance
from app.features.feature_schema import LIVE_SHORT_HORIZON_SCHEMA
from app.features.live_feature_frame import LiveFeatureFrame
from app.models.live_model_trainer import train_live_short_horizon_model
from app.models.live_signal_predictor import LiveSignalPredictor, _prediction_thresholds
from app.config.live_config import load_live_trading_safety_config
from tests.test_model_training_artifacts import _rows


class LiveSignalPredictorTest(unittest.TestCase):
    def test_predictor_requires_live_eligible_schema_compatible_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            train_live_short_horizon_model(_rows(), registry=__import__("app.models.model_artifact_registry", fromlist=["ModelArtifactRegistry"]).ModelArtifactRegistry(tmp))
            frame = LiveFeatureFrame(
                "005930",
                datetime.now(timezone.utc),
                LIVE_SHORT_HORIZON_SCHEMA,
                tuple(0.1 for _ in LIVE_SHORT_HORIZON_SCHEMA.feature_names),
                FeatureProvenance("005930", datetime.now(timezone.utc), ("tick",), "book", "kis_realtime_websocket", 1),
            )

            prediction = LiveSignalPredictor(__import__("app.models.model_artifact_registry", fromlist=["ModelArtifactRegistry"]).ModelArtifactRegistry(tmp)).predict(frame)

        self.assertEqual(prediction.feature_schema_hash, LIVE_SHORT_HORIZON_SCHEMA.schema_hash)
        self.assertGreaterEqual(prediction.probability_success, 0.0)

    def test_predictor_can_be_disabled_for_live_inference_while_training_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            train_live_short_horizon_model(_rows(), registry=__import__("app.models.model_artifact_registry", fromlist=["ModelArtifactRegistry"]).ModelArtifactRegistry(tmp))
            frame = LiveFeatureFrame(
                "005930",
                datetime.now(timezone.utc),
                LIVE_SHORT_HORIZON_SCHEMA,
                tuple(0.1 for _ in LIVE_SHORT_HORIZON_SCHEMA.feature_names),
                FeatureProvenance("005930", datetime.now(timezone.utc), ("tick",), "book", "kis_realtime_websocket", 1),
            )

            with patch.dict("os.environ", {"LIVE_SIGNAL_MODEL_INFERENCE_ENABLED": "false"}):
                with self.assertRaisesRegex(RuntimeError, "LIVE_SIGNAL_MODEL_INFERENCE_DISABLED"):
                    LiveSignalPredictor(__import__("app.models.model_artifact_registry", fromlist=["ModelArtifactRegistry"]).ModelArtifactRegistry(tmp)).predict(frame)


class ThresholdMergeSafetyTest(unittest.TestCase):
    """The safety config must only ever TIGHTEN the artifact gate, never loosen it.

    Regression: the merge previously used min() for the probability/return floors and
    max() for the uncertainty ceiling, so a looser artifact could weaken the safety
    floor. Correct behaviour: floors take the higher (max), the ceiling takes the
    lower (min).
    """

    def test_loose_artifact_cannot_weaken_safety_floor(self) -> None:
        safety = load_live_trading_safety_config()
        # Artifact thresholds strictly looser than the safety config in every direction.
        loose = {
            "minimum_probability_success": safety.minimum_probability_success - 0.2,
            "minimum_expected_net_return_bps": safety.minimum_expected_net_return_bps - 50.0,
            "maximum_uncertainty": 0.99,
        }
        merged = _prediction_thresholds(loose)
        self.assertGreaterEqual(
            merged["minimum_probability_success"], safety.minimum_probability_success
        )
        self.assertGreaterEqual(
            merged["minimum_expected_net_return_bps"], safety.minimum_expected_net_return_bps
        )
        ceiling = 1.0 - max(0.0, min(1.0, safety.minimum_model_confidence))
        self.assertLessEqual(merged["maximum_uncertainty"], ceiling)

    def test_strict_artifact_is_preserved(self) -> None:
        safety = load_live_trading_safety_config()
        strict = {
            "minimum_probability_success": safety.minimum_probability_success + 0.1,
            "minimum_expected_net_return_bps": safety.minimum_expected_net_return_bps + 20.0,
            "maximum_uncertainty": 0.10,
        }
        merged = _prediction_thresholds(strict)
        self.assertEqual(
            merged["minimum_probability_success"], safety.minimum_probability_success + 0.1
        )
        self.assertEqual(
            merged["minimum_expected_net_return_bps"], safety.minimum_expected_net_return_bps + 20.0
        )
        self.assertEqual(merged["maximum_uncertainty"], 0.10)


if __name__ == "__main__":
    unittest.main()

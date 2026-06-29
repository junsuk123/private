from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.features.feature_provenance import FeatureProvenance
from app.features.feature_schema import LIVE_SHORT_HORIZON_SCHEMA
from app.features.live_feature_frame import LiveFeatureFrame
from app.models.live_model_trainer import train_live_short_horizon_model
from app.models.live_signal_predictor import LiveSignalPredictor
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


if __name__ == "__main__":
    unittest.main()

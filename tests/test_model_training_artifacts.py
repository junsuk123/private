from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.features.feature_schema import LIVE_SHORT_HORIZON_SCHEMA
from app.models.live_model_trainer import train_live_short_horizon_model
from app.models.model_artifact_registry import ModelArtifactRegistry


class ModelTrainingArtifactsTest(unittest.TestCase):
    def test_trained_artifact_is_live_eligible_with_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = train_live_short_horizon_model(_rows(), registry=ModelArtifactRegistry(tmp))
            latest = ModelArtifactRegistry(tmp).load_latest_live_eligible()

        self.assertTrue(artifact["live_eligible"], artifact["reason_codes"])
        self.assertEqual(latest.feature_schema_hash, LIVE_SHORT_HORIZON_SCHEMA.schema_hash)
        self.assertGreater(artifact["metrics"]["auc"], 0.55)

    def test_zero_positive_labels_never_live_eligible(self) -> None:
        rows = _rows()
        for row in rows:
            row["label"] = 0
            row["forward_net_return_bps"] = -10
        with tempfile.TemporaryDirectory() as tmp:
            artifact = train_live_short_horizon_model(rows, registry=ModelArtifactRegistry(tmp))

        self.assertFalse(artifact["live_eligible"])
        self.assertIn("INSUFFICIENT_POSITIVE_LABELS", artifact["reason_codes"])


def _rows() -> list[dict]:
    names = LIVE_SHORT_HORIZON_SCHEMA.feature_names
    rows = []
    for i in range(60):
        positive = i % 3 != 0
        features = {name: 0.0 for name in names}
        features["return_1m"] = 0.005 if positive else -0.004
        features["return_3m"] = 0.008 if positive else -0.006
        features["spread_bps"] = 8 if positive else 40
        features["orderbook_imbalance"] = 0.3 if positive else -0.4
        features["liquidity_score"] = 0.9 if positive else 0.2
        features["cost_to_volatility_ratio"] = 0.2 if positive else 2.0
        features["bid_depth"] = 200000
        features["ask_depth"] = 150000
        features["depth_ratio"] = 1.3
        features["principal_cushion_ratio"] = 1.0
        rows.append({"features": features, "label": int(positive), "forward_net_return_bps": 50 if positive else -30})
    return rows


if __name__ == "__main__":
    unittest.main()

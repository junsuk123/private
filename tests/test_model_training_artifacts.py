from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
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

    def test_latest_is_cached_and_reloads_on_change(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelArtifactRegistry(tmp)
            train_live_short_horizon_model(_rows(), registry=registry)
            first = registry.load_latest_live_eligible()
            # Second call with an unchanged file must return the SAME cached object.
            second = registry.load_latest_live_eligible()
            self.assertIs(first, second)

            # Rewrite latest.json with a new artifact_id -> mtime changes -> cache reloads.
            payload = json.loads(registry.latest_path.read_text(encoding="utf-8"))
            payload["artifact_id"] = payload["artifact_id"] + "-v2"
            registry.latest_path.write_text(json.dumps(payload), encoding="utf-8")
            reloaded = registry.load_latest_live_eligible()
            self.assertIsNot(reloaded, first)
            self.assertTrue(reloaded.artifact_id.endswith("-v2"))

    def test_zero_positive_labels_never_live_eligible(self) -> None:
        rows = _rows()
        for row in rows:
            row["label"] = 0
            row["forward_net_return_bps"] = -10
        with tempfile.TemporaryDirectory() as tmp:
            artifact = train_live_short_horizon_model(rows, registry=ModelArtifactRegistry(tmp))

        self.assertFalse(artifact["live_eligible"])
        self.assertIn("INSUFFICIENT_POSITIVE_LABELS", artifact["reason_codes"])
        self.assertEqual(artifact["metrics"]["example_count"], float(len(rows)))
        self.assertEqual(artifact["metrics"]["positive_labels"], 0.0)

    def test_weaker_eligible_challenger_does_not_replace_active_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelArtifactRegistry(tmp)
            incumbent = _artifact("incumbent", auc=0.72, precision=0.43, top_k_return=5.0)
            challenger = _artifact("challenger", auc=0.60, precision=0.36, top_k_return=1.0)

            registry.save(incumbent)
            registry.save(challenger)
            active = registry.load_latest_live_eligible()

        self.assertEqual(active.artifact_id, "incumbent")
        self.assertTrue(incumbent["deployment"]["promoted"])
        self.assertFalse(challenger["deployment"]["promoted"])
        self.assertEqual(
            challenger["deployment"]["reason"],
            "CHALLENGER_REGRESSES_ACTIVE_MODEL",
        )


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


def _artifact(
    artifact_id: str,
    *,
    auc: float,
    precision: float,
    top_k_return: float,
) -> dict:
    names = LIVE_SHORT_HORIZON_SCHEMA.feature_names
    return {
        "artifact_id": artifact_id,
        # A real artifact always carries this; without it the staleness check
        # correctly refuses to serve the model (unknown age is not freshness).
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_schema_hash": LIVE_SHORT_HORIZON_SCHEMA.schema_hash,
        "feature_names": list(names),
        "classification": {"weights": [0.0] * len(names), "bias": 0.0},
        "regression": {"weights": [0.0] * len(names), "bias": 0.0},
        "thresholds": {},
        "metrics": {
            "auc": auc,
            "precision_at_k": precision,
            "avg_forward_net_return_bps_top_k": top_k_return,
        },
        "live_eligible": True,
        "reason_codes": [],
    }


if __name__ == "__main__":
    unittest.main()

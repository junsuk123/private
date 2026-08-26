from __future__ import annotations

import sys
import tempfile
import unittest
import json
from unittest.mock import patch
from datetime import datetime, timedelta, timezone
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
        self.assertEqual(artifact["metrics"]["runtime_policy_aligned_evaluation"], 1.0)
        self.assertGreater(artifact["metrics"]["deployable_holdout_count"], 0.0)
        self.assertLessEqual(
            artifact["metrics"]["top_k_count"],
            artifact["metrics"]["top_k_target_count"],
        )

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

    def test_fresh_eligible_challenger_replaces_stale_active_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelArtifactRegistry(tmp)
            incumbent = _artifact("incumbent", auc=0.72, precision=0.43, top_k_return=5.0)
            challenger = _artifact("challenger", auc=0.60, precision=0.36, top_k_return=1.0)
            registry.save(incumbent)
            stale_payload = json.loads(registry.latest_path.read_text(encoding="utf-8"))
            stale_payload["created_at"] = (
                datetime.now(timezone.utc) - timedelta(hours=7)
            ).isoformat()
            registry.latest_path.write_text(json.dumps(stale_payload), encoding="utf-8")

            registry.save(challenger)
            active = registry.load_latest_live_eligible()

        self.assertEqual(active.artifact_id, "challenger")
        self.assertTrue(challenger["deployment"]["promoted"])
        self.assertEqual(
            challenger["deployment"]["reason"],
            "STALE_INCUMBENT_REPLACED",
        )

    def test_registry_prunes_old_challengers_but_keeps_active_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ", {"LIVE_MODEL_ARTIFACT_RETENTION_COUNT": "2"}
        ):
            registry = ModelArtifactRegistry(tmp)
            for index in range(6):
                registry.save(
                    _artifact(
                        f"live_short_horizon.20260101T00000{index}Z",
                        auc=0.72 - index * 0.01,
                        precision=0.43,
                        top_k_return=5.0,
                    )
                )
            saved = sorted(Path(tmp).glob("live_short_horizon.*.json"))
            active = registry.load_latest_live_eligible()

        self.assertLessEqual(len(saved), 3)
        self.assertEqual(active.artifact_id, "live_short_horizon.20260101T000000Z")
        self.assertTrue(any(path.stem == active.artifact_id for path in saved))


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


class AbsoluteEconomicsPromotionTest(unittest.TestCase):
    """Relative merit is not enough to earn live expected-return authority.

    Every other promotion branch is relative -- better than the incumbent, or the
    incumbent is stale/obsolete. That let "first eligible" and "stale incumbent
    replaced" promote a model without ever asking whether its own best decile pays
    for a round trip. A model whose top-k loses money is not usable just because
    the alternative is worse.
    """

    def test_non_positive_top_k_net_return_cannot_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelArtifactRegistry(tmp)
            loser = _artifact("loser", auc=0.95, precision=0.99, top_k_return=-1.0)
            registry.save(loser)

        self.assertFalse(loser["deployment"]["promoted"])
        self.assertEqual(
            loser["deployment"]["reason"], "TOP_K_NET_RETURN_NON_POSITIVE"
        )

    def test_high_auc_but_unprofitable_top_k_is_refused(self):
        # The exact shape observed on 2026-08-05: AUC looked healthy while the
        # top-of-book decile lost money.
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelArtifactRegistry(tmp)
            artifact = _artifact("auc_only", auc=0.99, precision=0.90, top_k_return=-25.8)
            registry.save(artifact)
            self.assertFalse(registry.latest_path.exists())

        self.assertFalse(artifact["deployment"]["promoted"])

    def test_top_k_below_runtime_minimum_cannot_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelArtifactRegistry(tmp)
            weak = _artifact("weak", auc=0.80, precision=0.60, top_k_return=4.0)
            weak["thresholds"] = {"minimum_expected_net_return_bps": 10.0}
            registry.save(weak)

        self.assertFalse(weak["deployment"]["promoted"])
        self.assertEqual(
            weak["deployment"]["reason"], "TOP_K_NET_RETURN_BELOW_RUNTIME_MINIMUM"
        )

    def test_runtime_aligned_policy_requires_enough_deployable_holdout_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelArtifactRegistry(tmp)
            sparse = _artifact("sparse", auc=0.90, precision=1.0, top_k_return=25.0)
            sparse["metrics"].update(
                {
                    "runtime_policy_aligned_evaluation": 1.0,
                    "top_k_count": 1.0,
                    "validation_example_count": 2000.0,
                    "validation_symbol_count": 20.0,
                    "holdout_evaluated": 1.0,
                }
            )
            registry.save(sparse)

        self.assertFalse(sparse["deployment"]["promoted"])
        self.assertEqual(
            sparse["deployment"]["reason"],
            "DEPLOYABLE_HOLDOUT_SAMPLE_TOO_SMALL",
        )

    def test_absolute_floor_also_applies_when_replacing_a_stale_incumbent(self):
        # STALE_INCUMBENT_REPLACED used to bypass economics entirely.
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelArtifactRegistry(tmp)
            registry.save(_artifact("incumbent", auc=0.72, precision=0.43, top_k_return=5.0))
            stale = json.loads(registry.latest_path.read_text(encoding="utf-8"))
            stale["created_at"] = (
                datetime.now(timezone.utc) - timedelta(hours=7)
            ).isoformat()
            registry.latest_path.write_text(json.dumps(stale), encoding="utf-8")

            unprofitable = _artifact("unprofitable", auc=0.90, precision=0.90, top_k_return=-3.0)
            registry.save(unprofitable)
            active = json.loads(registry.latest_path.read_text(encoding="utf-8"))

        self.assertFalse(unprofitable["deployment"]["promoted"])
        self.assertEqual(active["artifact_id"], "incumbent")

    def test_in_sample_only_metrics_cannot_promote(self):
        """A fit is not a measurement.

        The trainer evaluates on the training set when the sample is too small to
        split. Observed 2026-08-06: auc 0.9865 / precision 1.000 / +43.8bps with
        validation_example_count == 0 reached live order authority.
        """
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelArtifactRegistry(tmp)
            overfit = _artifact("overfit", auc=0.9865, precision=1.0, top_k_return=43.8)
            overfit["metrics"]["holdout_evaluated"] = 0.0
            overfit["metrics"]["validation_example_count"] = 0.0
            registry.save(overfit)
            self.assertFalse(registry.latest_path.exists())

        self.assertFalse(overfit["deployment"]["promoted"])
        self.assertEqual(overfit["deployment"]["reason"], "IN_SAMPLE_METRICS_ONLY")

    def test_zero_validation_rows_cannot_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelArtifactRegistry(tmp)
            unvalidated = _artifact("unvalidated", auc=0.95, precision=0.95, top_k_return=40.0)
            unvalidated["metrics"]["validation_example_count"] = 0.0
            registry.save(unvalidated)

        self.assertFalse(unvalidated["deployment"]["promoted"])
        self.assertEqual(unvalidated["deployment"]["reason"], "NO_HOLDOUT_VALIDATION")

    def test_holdout_evaluated_candidate_promotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelArtifactRegistry(tmp)
            good = _artifact("holdout", auc=0.75, precision=0.55, top_k_return=21.4)
            good["metrics"]["holdout_evaluated"] = 1.0
            good["metrics"]["validation_example_count"] = 3601.0
            registry.save(good)
            active = registry.load_latest_live_eligible()

        self.assertTrue(good["deployment"]["promoted"])
        self.assertEqual(active.artifact_id, "holdout")

    def test_in_sample_artifact_cannot_replace_a_stale_incumbent(self):
        # STALE_INCUMBENT_REPLACED was the exact path the unvalidated model took.
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelArtifactRegistry(tmp)
            registry.save(_artifact("incumbent", auc=0.72, precision=0.43, top_k_return=5.0))
            stale = json.loads(registry.latest_path.read_text(encoding="utf-8"))
            stale["created_at"] = (
                datetime.now(timezone.utc) - timedelta(hours=7)
            ).isoformat()
            registry.latest_path.write_text(json.dumps(stale), encoding="utf-8")

            overfit = _artifact("overfit", auc=0.99, precision=1.0, top_k_return=43.8)
            overfit["metrics"]["holdout_evaluated"] = 0.0
            registry.save(overfit)
            active = json.loads(registry.latest_path.read_text(encoding="utf-8"))

        self.assertFalse(overfit["deployment"]["promoted"])
        self.assertEqual(active["artifact_id"], "incumbent")

    def test_tiny_holdout_cannot_promote_at_production_floors(self):
        """16 rows across 3 symbols is a holdout in name only.

        Observed 2026-08-06: exactly that shape produced auc 0.983 / +35bps and
        took live authority because the size floors defaulted off.
        """
        import os
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "LIVE_MODEL_PROMOTION_MIN_VALIDATION_ROWS": "200",
                "LIVE_MODEL_PROMOTION_MIN_VALIDATION_SYMBOLS": "5",
            },
        ):
            registry = ModelArtifactRegistry(tmp)
            tiny = _artifact("tiny", auc=0.983, precision=0.625, top_k_return=35.1)
            tiny["metrics"]["holdout_evaluated"] = 1.0
            tiny["metrics"]["validation_example_count"] = 16.0
            tiny["metrics"]["validation_symbol_count"] = 3.0
            registry.save(tiny)

        self.assertFalse(tiny["deployment"]["promoted"])
        self.assertEqual(tiny["deployment"]["reason"], "VALIDATION_SAMPLE_TOO_SMALL")

    def test_too_few_validation_symbols_cannot_promote(self):
        import os
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "LIVE_MODEL_PROMOTION_MIN_VALIDATION_ROWS": "10",
                "LIVE_MODEL_PROMOTION_MIN_VALIDATION_SYMBOLS": "5",
            },
        ):
            registry = ModelArtifactRegistry(tmp)
            narrow = _artifact("narrow", auc=0.90, precision=0.80, top_k_return=30.0)
            narrow["metrics"]["holdout_evaluated"] = 1.0
            narrow["metrics"]["validation_example_count"] = 4000.0
            narrow["metrics"]["validation_symbol_count"] = 2.0
            registry.save(narrow)

        self.assertFalse(narrow["deployment"]["promoted"])
        self.assertEqual(
            narrow["deployment"]["reason"], "VALIDATION_SYMBOL_COUNT_TOO_SMALL"
        )

    def test_production_scale_holdout_promotes(self):
        import os
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "LIVE_MODEL_PROMOTION_MIN_VALIDATION_ROWS": "200",
                "LIVE_MODEL_PROMOTION_MIN_VALIDATION_SYMBOLS": "5",
            },
        ):
            registry = ModelArtifactRegistry(tmp)
            solid = _artifact("solid", auc=0.75, precision=0.55, top_k_return=21.4)
            solid["metrics"]["holdout_evaluated"] = 1.0
            solid["metrics"]["validation_example_count"] = 3601.0
            solid["metrics"]["validation_symbol_count"] = 44.0
            registry.save(solid)
            active = registry.load_latest_live_eligible()

        self.assertTrue(solid["deployment"]["promoted"])
        self.assertEqual(active.artifact_id, "solid")

    def test_profitable_candidate_still_promotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelArtifactRegistry(tmp)
            good = _artifact("good", auc=0.75, precision=0.55, top_k_return=21.4)
            registry.save(good)
            active = registry.load_latest_live_eligible()

        self.assertTrue(good["deployment"]["promoted"])
        self.assertEqual(active.artifact_id, "good")


class ObsoleteSchemaPromotionTest(unittest.TestCase):
    """A retired-schema incumbent must not veto its own replacement.

    ``live_signal_predictor`` raises MODEL_FEATURE_SCHEMA_MISMATCH for an artifact
    trained on a different feature set, so such an incumbent can never serve a
    prediction. Its metrics are also not comparable — precision_at_k over a
    different feature set is a different measurement. Observed on the 2026-08-05
    v4->v5 change: a v5 candidate (net +19.75bps, live-eligible) was rejected as
    CHALLENGER_REGRESSES_ACTIVE_MODEL against a dead v4 incumbent's higher
    precision, so the migration could never land.
    """

    def test_current_schema_candidate_replaces_obsolete_schema_incumbent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelArtifactRegistry(tmp)
            incumbent = _artifact("v4", auc=0.80, precision=0.79, top_k_return=16.4)
            registry.save(incumbent)
            retired = json.loads(registry.latest_path.read_text(encoding="utf-8"))
            retired["feature_schema_hash"] = "retired0000000000000000f"
            registry.latest_path.write_text(json.dumps(retired), encoding="utf-8")

            # Lower precision than the incumbent: promotion must happen anyway.
            challenger = _artifact("v5", auc=0.74, precision=0.62, top_k_return=19.8)
            registry.save(challenger)
            active = registry.load_latest_live_eligible()

        self.assertEqual(active.artifact_id, "v5")
        self.assertEqual(
            challenger["deployment"]["reason"], "OBSOLETE_SCHEMA_INCUMBENT_REPLACED"
        )

    def test_obsolete_candidate_cannot_displace_a_current_incumbent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelArtifactRegistry(tmp)
            registry.save(_artifact("current", auc=0.80, precision=0.79, top_k_return=16.4))
            stale_schema = _artifact("old", auc=0.74, precision=0.62, top_k_return=19.8)
            stale_schema["feature_schema_hash"] = "retired0000000000000000f"
            registry.save(stale_schema)
            active = registry.load_latest_live_eligible()

        self.assertEqual(active.artifact_id, "current")
        self.assertFalse(stale_schema["deployment"]["promoted"])

    def test_same_schema_regression_is_still_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelArtifactRegistry(tmp)
            registry.save(_artifact("incumbent", auc=0.80, precision=0.79, top_k_return=16.4))
            worse = _artifact("worse", auc=0.74, precision=0.62, top_k_return=19.8)
            registry.save(worse)
            active = registry.load_latest_live_eligible()

        self.assertEqual(active.artifact_id, "incumbent")
        self.assertEqual(
            worse["deployment"]["reason"], "CHALLENGER_REGRESSES_ACTIVE_MODEL"
        )


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

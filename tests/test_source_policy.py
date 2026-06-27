from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.data.source_policy import (
    compute_quality_score,
    default_trust_level,
    infer_source_type,
    is_allowed_for_live_decision,
)
from app.schemas.domain import SourceMetadata


class SourcePolicyTest(unittest.TestCase):
    def test_source_metadata_defaults_are_low_trust_and_clamped(self) -> None:
        metadata = SourceMetadata("legacy", datetime.now(timezone.utc), trust_level=99, quality_score=2.0)

        self.assertEqual(metadata.source_type, "unknown")
        self.assertEqual(metadata.trust_level, 5)
        self.assertEqual(metadata.quality_score, 1.0)

        legacy = SourceMetadata("legacy", datetime.now(timezone.utc))
        self.assertEqual(legacy.trust_level, 0)
        self.assertEqual(legacy.quality_score, 0.0)
        self.assertFalse(legacy.is_synthetic)

    def test_source_policy_live_validation(self) -> None:
        now = datetime.now(timezone.utc)
        synthetic = SourceMetadata(
            "synthetic_demo",
            now,
            source_type="synthetic",
            is_synthetic=True,
            trust_level=0,
            quality_score=0.0,
        )
        official = SourceMetadata(
            "kis_broker_api",
            now,
            source_type="broker_api",
            trust_level=5,
            is_realtime=True,
            quality_score=0.95,
        )

        self.assertEqual(infer_source_type("KIS broker quote"), "broker_api")
        self.assertEqual(default_trust_level("official_exchange_api"), 5)
        self.assertEqual(compute_quality_score(synthetic), 0.0)
        self.assertFalse(is_allowed_for_live_decision(synthetic, 4, 0.8)[0])
        self.assertTrue(is_allowed_for_live_decision(official, 4, 0.8)[0])


if __name__ == "__main__":
    unittest.main()

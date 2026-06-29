from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.data.source_policy import is_allowed_for_live_buy_market_data
from app.schemas.domain import SourceMetadata


class SourcePolicyLiveBlockingTest(unittest.TestCase):
    def test_live_buy_blocks_unknown_delayed_and_stale_sources(self) -> None:
        now = datetime(2026, 6, 29, 9, 30, tzinfo=timezone.utc)
        metadata = SourceMetadata(
            source_name="unknown_chart",
            retrieved_at=now - timedelta(seconds=10),
            source_type="unknown",
            trust_level=0,
            quality_score=0,
            is_realtime=False,
            is_delayed=True,
        )

        allowed, reasons = is_allowed_for_live_buy_market_data(
            metadata,
            max_age_seconds=3,
            min_quality=0.85,
            now=now,
        )

        self.assertFalse(allowed)
        self.assertIn("UNKNOWN_SOURCE_CHECK", reasons)
        self.assertIn("MARKET_DATA_STALE", reasons)
        self.assertIn("MARKET_DATA_NOT_REALTIME", reasons)
        self.assertIn("DELAYED_MARKET_DATA_BLOCKED", reasons)

    def test_live_buy_allows_fresh_realtime_broker_source(self) -> None:
        now = datetime(2026, 6, 29, 9, 30, tzinfo=timezone.utc)
        metadata = SourceMetadata(
            source_name="KIS realtime WebSocket",
            retrieved_at=now,
            observed_at=now,
            source_type="broker_api",
            trust_level=5,
            quality_score=1.0,
            is_realtime=True,
        )

        allowed, reasons = is_allowed_for_live_buy_market_data(
            metadata,
            max_age_seconds=3,
            min_quality=0.85,
            now=now,
        )

        self.assertTrue(allowed, reasons)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.realtime.learning import (
    build_realtime_supervised_examples,
    run_hypothetical_realtime_test,
    update_realtime_model_artifacts,
)
from app.schemas.domain import (
    MarketSnapshot,
    OrderAction,
    SourceMetadata,
    StrategySignal,
)
from app.storage import ModelArtifactStore
from app.time_series import build_time_synchronized_frames


class RealtimeLearningTest(unittest.TestCase):
    def test_realtime_frames_create_supervised_pnl_labels_and_hypothetical_test(self) -> None:
        base = datetime(2026, 6, 17, 0, 0, tzinfo=timezone.utc)
        source1 = SourceMetadata("unit", base, source_id="quote:1")
        source2 = SourceMetadata("unit", base + timedelta(minutes=15), source_id="quote:2")
        frames = build_time_synchronized_frames(
            markets=(
                MarketSnapshot("TEST", "KRX", "Test", "Tech", 100.0, 1_000_000_000, 0.02, source1),
                MarketSnapshot("TEST", "KRX", "Test", "Tech", 104.0, 1_000_000_000, 0.02, source2),
            ),
            bucket_minutes=15,
        )
        signal = StrategySignal(
            ticker="TEST",
            action=OrderAction.BUY,
            confidence=0.8,
            score=0.7,
            supporting_factors=("unit",),
            contradicting_factors=(),
            reasoning_path_ids=(),
        )

        examples = build_realtime_supervised_examples(frames, (signal,))
        result = run_hypothetical_realtime_test(frames, (signal,))

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].label, 1)
        self.assertGreater(examples[0].realized_pnl, 0)
        self.assertEqual(result["orders_submitted"], 0)
        self.assertEqual(result["realized_pnl"], 4.0)

    def test_model_artifacts_are_saved_by_model_family_with_latest_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ModelArtifactStore(Path(tmp), mode="realtime")
            paths = update_realtime_model_artifacts(store, ())

            path = Path(paths["realtime_supervised"])
            self.assertTrue(path.exists())
            self.assertTrue((Path(tmp) / "realtime_supervised" / "realtime_supervised_trade_timing.latest.json").exists())


if __name__ == "__main__":
    unittest.main()

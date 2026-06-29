from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.data.realtime_store import RealtimeMarketDataStore
from app.data.realtime_types import KIS_REALTIME_SOURCE, OrderbookLevel, RealtimeOrderbookSnapshot, RealtimeTradeTick
from app.features.feature_schema import LIVE_SHORT_HORIZON_SCHEMA
from app.features.live_feature_frame import LiveFeatureFrameBuilder


class LiveFeatureFrameTest(unittest.TestCase):
    def test_feature_frame_has_schema_hash_and_provenance(self) -> None:
        now = datetime(2026, 6, 29, 9, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            store = RealtimeMarketDataStore(Path(tmp) / "rt.sqlite3")
            _seed(store, now)
            frame = LiveFeatureFrameBuilder(store, journal_path=Path(tmp) / "features.jsonl").build("005930", decision_time=now)

        self.assertEqual(frame.feature_schema_hash, LIVE_SHORT_HORIZON_SCHEMA.schema_hash)
        self.assertEqual(len(frame.values), len(LIVE_SHORT_HORIZON_SCHEMA.feature_names))
        self.assertGreater(len(frame.provenance.source_record_ids), 0)

    def test_feature_frame_can_use_kis_orderbook_when_trade_ticks_are_sparse(self) -> None:
        now = datetime(2026, 6, 29, 9, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            store = RealtimeMarketDataStore(Path(tmp) / "rt.sqlite3")
            _seed_orderbooks_only(store, now)
            frame = LiveFeatureFrameBuilder(store, journal_path=Path(tmp) / "features.jsonl").build("005930", decision_time=now)

        self.assertEqual(frame.feature_schema_hash, LIVE_SHORT_HORIZON_SCHEMA.schema_hash)
        self.assertGreater(len(frame.provenance.source_record_ids), 0)


def _seed(store: RealtimeMarketDataStore, now: datetime) -> None:
    ticks = tuple(
        RealtimeTradeTick(
            symbol="005930",
            exchange_timestamp=now - timedelta(seconds=120 - i * 10),
            received_at=now - timedelta(seconds=120 - i * 10),
            source=KIS_REALTIME_SOURCE,
            price=70000 + i * 10,
            volume=100 + i,
            sequence_key=f"tick:{i}",
        )
        for i in range(13)
    )
    store.save_ticks(ticks)
    store.save_orderbooks(
        (
            RealtimeOrderbookSnapshot(
                symbol="005930",
                exchange_timestamp=now,
                received_at=now,
                source=KIS_REALTIME_SOURCE,
                levels=(OrderbookLevel(70100, 1000, 70150, 800),),
                sequence_key="book:1",
            ),
        )
    )


def _seed_orderbooks_only(store: RealtimeMarketDataStore, now: datetime) -> None:
    books = tuple(
        RealtimeOrderbookSnapshot(
            symbol="005930",
            exchange_timestamp=now - timedelta(seconds=120 - i * 10),
            received_at=now - timedelta(seconds=120 - i * 10),
            source=KIS_REALTIME_SOURCE,
            levels=(OrderbookLevel(70000 + i * 10, 1000 + i, 70100 + i * 10, 900 + i),),
            sequence_key=f"book:{i}",
        )
        for i in range(13)
    )
    store.save_orderbooks(books)


if __name__ == "__main__":
    unittest.main()

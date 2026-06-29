from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.data.market_data_health import evaluate_market_data_health
from app.data.realtime_store import RealtimeMarketDataStore
from app.data.realtime_types import (
    KIS_REALTIME_SOURCE,
    OrderbookLevel,
    RealtimeOrderbookSnapshot,
    RealtimeTradeTick,
)


class MarketDataFreshnessTest(unittest.TestCase):
    def test_fresh_kis_tick_and_orderbook_allow_live_buy_evidence(self) -> None:
        now = datetime(2026, 6, 29, 9, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            store = RealtimeMarketDataStore(Path(tmp) / "rt.sqlite3")
            store.save_ticks((_tick(now),))
            store.save_orderbooks((_book(now),))

            health = evaluate_market_data_health(store, "005930", now=now + timedelta(seconds=1))

        self.assertTrue(health.ok_for_live_buy, health.reason_codes)

    def test_stale_or_missing_orderbook_blocks_live_buy(self) -> None:
        now = datetime(2026, 6, 29, 9, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            store = RealtimeMarketDataStore(Path(tmp) / "rt.sqlite3")
            store.save_ticks((_tick(now - timedelta(seconds=10)),))

            health = evaluate_market_data_health(store, "005930", now=now)

        self.assertFalse(health.ok_for_live_buy)
        self.assertIn("QUOTE_STALE", health.reason_codes)
        self.assertIn("ORDERBOOK_COUNT_ZERO", health.reason_codes)


def _tick(at: datetime) -> RealtimeTradeTick:
    return RealtimeTradeTick(
        symbol="005930",
        exchange_timestamp=at,
        received_at=at,
        source=KIS_REALTIME_SOURCE,
        price=70000,
        volume=100,
        sequence_key=f"tick:{at.isoformat()}",
    )


def _book(at: datetime) -> RealtimeOrderbookSnapshot:
    return RealtimeOrderbookSnapshot(
        symbol="005930",
        exchange_timestamp=at,
        received_at=at,
        source=KIS_REALTIME_SOURCE,
        levels=(OrderbookLevel(bid_price=70000, bid_size=1000, ask_price=70100, ask_size=900),),
        sequence_key=f"book:{at.isoformat()}",
    )


if __name__ == "__main__":
    unittest.main()

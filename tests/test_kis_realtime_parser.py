from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.data.kis_realtime import (
    KisRealtimeSubscriptionManager,
    QueueMessageSource,
    normalize_symbol,
    parse_kis_realtime_message,
)
from app.data.realtime_store import RealtimeMarketDataStore
from app.data.realtime_types import KIS_REALTIME_SOURCE


class KisRealtimeParserTest(unittest.TestCase):
    def test_trade_tick_parser_normalizes_kis_pipe_payload(self) -> None:
        received_at = datetime(2026, 6, 29, 9, 30, 1, tzinfo=timezone.utc)
        raw = "0|H0STCNT0|001|005930^093000^70000^120^BUY^seq-1"

        parsed = parse_kis_realtime_message(raw, received_at=received_at)

        self.assertEqual(parsed.event_type, "trade")
        tick = parsed.ticks[0]
        self.assertEqual(tick.symbol, "005930")
        self.assertEqual(tick.price, 70000)
        self.assertEqual(tick.volume, 120)
        self.assertEqual(tick.source, KIS_REALTIME_SOURCE)
        self.assertEqual(tick.sequence_key, "seq-1")

    def test_orderbook_parser_computes_spread_and_imbalance(self) -> None:
        received_at = datetime(2026, 6, 29, 9, 30, 1, tzinfo=timezone.utc)
        raw = "0|H0STASP0|001|005930^093000^70100^70000^1000^1500^70200^69900^800^700"

        parsed = parse_kis_realtime_message(raw, received_at=received_at)

        book = parsed.orderbooks[0]
        self.assertEqual(book.symbol, "005930")
        self.assertEqual(book.best_bid, 70000)
        self.assertEqual(book.best_ask, 70100)
        self.assertGreater(book.spread_bps, 0)
        self.assertGreater(book.total_bid_volume, book.total_ask_volume)

    def test_subscription_manager_persists_ticks_orderbooks_and_bar(self) -> None:
        messages = (
            "0|H0STCNT0|001|005930^093000^70000^100^BUY^seq-1",
            "0|H0STCNT0|001|005930^093001^70100^200^BUY^seq-2",
            "0|H0STASP0|001|005930^093001^70150^70100^900^1100",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = RealtimeMarketDataStore(Path(tmp) / "rt.sqlite3")
            manager = KisRealtimeSubscriptionManager(store, QueueMessageSource(messages))
            manager.subscribe(["005930"])

            counts = asyncio.run(manager.run_forever())
            tick = store.latest_tick("005930")
            book = store.latest_orderbook("005930")
            self.assertIsNotNone(tick)
            bar_now = tick.exchange_timestamp.replace(second=30, microsecond=0)
            bar = store.build_latest_minute_bar("005930", now=bar_now)

        self.assertEqual(counts["ticks"], 2)
        self.assertEqual(counts["orderbooks"], 1)
        self.assertIsNotNone(book)
        self.assertIsNotNone(bar)
        self.assertEqual(bar.close, 70100)
        self.assertEqual(bar.volume, 300)

    def test_symbol_normalization_keeps_krx_six_digits(self) -> None:
        self.assertEqual(normalize_symbol("660"), "000660")


if __name__ == "__main__":
    unittest.main()

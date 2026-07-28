from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.data.kis_realtime import (
    DEFAULT_SUBSCRIPTION_TR_IDS,
    KisRealtimeSubscriptionManager,
    QueueMessageSource,
    _is_websocket_connection_closed,
    _kis_realtime_websocket_url,
    _websocket_subscription_delay_seconds,
    _websocket_ping_setting,
    kis_realtime_control_summary,
    kis_realtime_subscription_message,
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
        self.assertEqual(tick.exchange_timestamp.isoformat(), "2026-06-29T00:30:00+00:00")
        self.assertEqual(tick.price, 70000)
        self.assertEqual(tick.volume, 120)
        self.assertEqual(tick.source, KIS_REALTIME_SOURCE)
        self.assertEqual(tick.sequence_key, "seq-1")

    def test_trade_tick_parser_uses_official_h0stcnt0_field_positions(self) -> None:
        received_at = datetime(2026, 7, 28, 0, 30, 1, tzinfo=timezone.utc)
        fields = [
            "005930", "093000", "70000", "5", "-100", "-0.14", "70010.0",
            "70100", "70200", "69900", "70100", "70000", "37", "123456",
            "8641920000", "100", "120", "20", "88.0", "1000", "1200", "1",
            "54.0", "1.2", "090000", "2", "100", "091000", "5", "-200",
            "092000", "2", "100", "20260728", "20", "N", "100", "200",
            "1000", "1200", "0.5", "100000", "123.0", "0", "", "70000",
        ]
        raw = f"0|H0STCNT0|001|{'^'.join(fields)}"

        tick = parse_kis_realtime_message(raw, received_at=received_at).ticks[0]

        self.assertEqual(tick.volume, 37)
        self.assertEqual(tick.trade_direction, "BUY")
        self.assertNotEqual(tick.sequence_key, fields[5])
        self.assertIn("123456", tick.sequence_key)

    def test_trade_parser_expands_multi_record_payload(self) -> None:
        received_at = datetime(2026, 7, 28, 0, 30, 1, tzinfo=timezone.utc)

        def record(symbol: str, hhmmss: str, price: str, volume: str, cumulative: str) -> list[str]:
            fields = [
                symbol, hhmmss, price, "2", "100", "0.14", price,
                price, price, price, price, price, volume, cumulative,
                str(int(price) * int(cumulative)), "1", "1", "0", "100.0",
                "1", "1", "1", "50.0", "1.0", hhmmss, "2", "0", hhmmss,
                "2", "0", hhmmss, "2", "0", "20260728", "20", "N", "1",
                "1", "1", "1", "0.1", cumulative, "100.0", "0", "", price,
            ]
            self.assertEqual(len(fields), 46)
            return fields

        payload = "^".join(
            record("005930", "093000", "70000", "10", "100")
            + record("000660", "093001", "120000", "20", "200")
        )
        parsed = parse_kis_realtime_message(
            f"0|H0STCNT0|002|{payload}",
            received_at=received_at,
        )

        self.assertEqual(len(parsed.ticks), 2)
        self.assertEqual([tick.symbol for tick in parsed.ticks], ["005930", "000660"])
        self.assertEqual([tick.volume for tick in parsed.ticks], [10, 20])

    def test_orderbook_parser_expands_multi_record_payload(self) -> None:
        received_at = datetime(2026, 7, 28, 0, 30, 1, tzinfo=timezone.utc)

        def record(symbol: str, hhmmss: str, ask: int, bid: int) -> list[str]:
            fields = [symbol, hhmmss, "0"]
            fields += [str(ask + index * 100) for index in range(10)]
            fields += [str(bid - index * 100) for index in range(10)]
            fields += [str(1000 + index) for index in range(10)]
            fields += [str(1200 + index) for index in range(10)]
            fields += ["10000", "12000", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"]
            self.assertEqual(len(fields), 59)
            return fields

        payload = "^".join(
            record("005930", "093000", 70100, 70000)
            + record("000660", "093001", 120100, 120000)
        )
        parsed = parse_kis_realtime_message(
            f"0|H0STASP0|002|{payload}",
            received_at=received_at,
        )

        self.assertEqual(len(parsed.orderbooks), 2)
        self.assertEqual([book.symbol for book in parsed.orderbooks], ["005930", "000660"])

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

    def test_orderbook_parser_handles_real_kis_depth_layout(self) -> None:
        received_at = datetime(2026, 6, 29, 5, 33, 55, tzinfo=timezone.utc)
        raw = (
            "0|H0STASP0|001|005930^143354^0^324000^324500^325000^325500^326000^326500^327000^327500^328000^328500^"
            "323500^323000^322500^322000^321500^321000^320500^320000^319500^319000^"
            "34364^26832^32755^16039^24386^14211^16443^19597^23179^13243^"
            "28602^10731^12584^16973^24121^30969^19788^35604^13932^16282^"
            "221049^209586^0^0^0^0^1347146^-339500^5^-100.00^26718921^-79^8^0^0^0^323750^0^0"
        )

        parsed = parse_kis_realtime_message(raw, received_at=received_at)

        book = parsed.orderbooks[0]
        self.assertEqual(book.best_ask, 324000)
        self.assertEqual(book.best_bid, 323500)
        self.assertEqual(book.levels[0].ask_size, 34364)
        self.assertEqual(book.levels[0].bid_size, 28602)

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

    def test_subscription_manager_skips_bad_message_without_stopping_ticks(self) -> None:
        messages = (
            "unexpected-control-message",
            "0|H0STCNT0|001|005930^093000^70000^100^BUY^seq-1",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = RealtimeMarketDataStore(Path(tmp) / "rt.sqlite3")
            manager = KisRealtimeSubscriptionManager(store, QueueMessageSource(messages))
            manager.subscribe(["005930"])

            counts = asyncio.run(manager.run_forever())
            tick = store.latest_tick("005930")

        self.assertEqual(counts["parse_errors"], 1)
        self.assertEqual(counts["ticks"], 1)
        self.assertIsNotNone(tick)

    def test_subscription_manager_can_publish_without_database_io(self) -> None:
        messages = ("0|H0STCNT0|001|005930^093000^70000^100^BUY^seq-1",)
        published = []

        async def sink(event):
            published.append(event)
            return True

        with tempfile.TemporaryDirectory() as tmp:
            store = RealtimeMarketDataStore(Path(tmp) / "rt.sqlite3")
            manager = KisRealtimeSubscriptionManager(
                store,
                QueueMessageSource(messages),
                event_sink=sink,
            )
            manager.subscribe(["005930"])
            counts = asyncio.run(manager.run_forever())
            tick = store.latest_tick("005930")

        self.assertEqual(counts["events_published"], 1)
        self.assertEqual(len(published), 1)
        self.assertIsNone(tick)

    def test_symbol_normalization_keeps_krx_six_digits(self) -> None:
        self.assertEqual(normalize_symbol("660"), "000660")

    def test_subscription_message_uses_approval_key_and_normalized_symbol(self) -> None:
        payload = kis_realtime_subscription_message("approval", "H0STCNT0", "660")

        self.assertIn('"approval_key":"approval"', payload)
        self.assertIn('"tr_id":"H0STCNT0"', payload)
        self.assertIn('"tr_key":"000660"', payload)

    def test_control_summary_keeps_kis_status_without_secrets(self) -> None:
        raw = (
            '{"header":{"tr_id":"H0STASP0","tr_key":"005930","approval_key":"secret"},'
            '"body":{"rt_cd":"1","msg_cd":"OPSP0001","msg1":"subscription failed"}}'
        )

        summary = kis_realtime_control_summary(raw)

        self.assertEqual(summary["tr_id"], "H0STASP0")
        self.assertEqual(summary["tr_key"], "005930")
        self.assertEqual(summary["rt_cd"], "1")
        self.assertEqual(summary["msg_cd"], "OPSP0001")
        self.assertNotIn("approval_key", summary)

    def test_kis_websocket_standard_ping_is_disabled_by_default(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "KIS_REALTIME_WS_PING_INTERVAL_SECONDS": "",
                "KIS_REALTIME_WS_PING_TIMEOUT_SECONDS": "",
            },
            clear=False,
        ):
            self.assertIsNone(_websocket_ping_setting("KIS_REALTIME_WS_PING_INTERVAL_SECONDS", None))
            self.assertIsNone(_websocket_ping_setting("KIS_REALTIME_WS_PING_TIMEOUT_SECONDS", None))

    def test_kis_subscriptions_are_throttled_by_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertGreaterEqual(_websocket_subscription_delay_seconds(), 1.0)

    def test_default_kis_subscriptions_request_trade_first(self) -> None:
        self.assertEqual(DEFAULT_SUBSCRIPTION_TR_IDS[0], "H0STCNT0")
        self.assertEqual(DEFAULT_SUBSCRIPTION_TR_IDS, ("H0STCNT0", "H0STASP0"))

    def test_default_kis_websocket_url_uses_tryitout_path(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(_kis_realtime_websocket_url().endswith("/tryitout"))

    def test_bare_kis_websocket_host_is_normalized(self) -> None:
        with patch.dict("os.environ", {"KIS_WEBSOCKET_URL": "ws://ops.koreainvestment.com:21000"}, clear=True):
            self.assertEqual(_kis_realtime_websocket_url(), "ws://ops.koreainvestment.com:21000/tryitout")

    def test_connection_closed_message_is_recognized(self) -> None:
        exc = RuntimeError("no close frame received or sent")

        self.assertTrue(_is_websocket_connection_closed(exc))


if __name__ == "__main__":
    unittest.main()

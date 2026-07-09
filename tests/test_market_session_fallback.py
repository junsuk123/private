from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.data.market_session import (
    MarketPhase,
    is_market_fully_closed,
    market_has_live_session,
    market_phase,
)
from app.data.realtime_store import RealtimeMarketDataStore
from app.data.realtime_types import KIS_REALTIME_SOURCE, KIS_REST_SNAPSHOT_SOURCE
from app.data.rest_snapshot_fallback import (
    market_snapshot_to_tick,
    refresh_rest_snapshot_into_store,
)


def _utc(y, mo, d, h, mi) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


class _FakeSnapshot:
    def __init__(self, last_price: float) -> None:
        self.last_price = last_price


class MarketPhaseTests(unittest.TestCase):
    # 2026-07-09 is a Thursday (weekday). KST = UTC+9, ET = UTC-4 (EDT).

    def test_krx_regular_session(self) -> None:
        # 10:00 KST -> 01:00 UTC
        self.assertIs(market_phase("KRX", _utc(2026, 7, 9, 1, 0)), MarketPhase.REGULAR)

    def test_krx_pre_market(self) -> None:
        # 08:40 KST -> 23:40 UTC previous day
        self.assertIs(market_phase("KRX", _utc(2026, 7, 8, 23, 40)), MarketPhase.PRE)

    def test_krx_after_market(self) -> None:
        # 16:30 KST -> 07:30 UTC
        self.assertIs(market_phase("KRX", _utc(2026, 7, 9, 7, 30)), MarketPhase.AFTER)

    def test_krx_fully_closed_at_night(self) -> None:
        # 22:00 KST -> 13:00 UTC  (this is exactly the reported symptom window)
        self.assertIs(market_phase("KRX", _utc(2026, 7, 9, 13, 0)), MarketPhase.CLOSED)
        self.assertTrue(is_market_fully_closed("KRX", _utc(2026, 7, 9, 13, 0)))

    def test_krx_weekend_closed(self) -> None:
        # 2026-07-11 is a Saturday, 10:00 KST -> 01:00 UTC
        self.assertIs(market_phase("KRX", _utc(2026, 7, 11, 1, 0)), MarketPhase.CLOSED)

    def test_us_regular_session(self) -> None:
        # 10:00 ET -> 14:00 UTC
        self.assertIs(market_phase("US", _utc(2026, 7, 9, 14, 0)), MarketPhase.REGULAR)

    def test_us_pre_market(self) -> None:
        # 05:00 ET -> 09:00 UTC
        self.assertIs(market_phase("US", _utc(2026, 7, 9, 9, 0)), MarketPhase.PRE)

    def test_us_after_market(self) -> None:
        # 18:00 ET -> 22:00 UTC
        self.assertIs(market_phase("US", _utc(2026, 7, 9, 22, 0)), MarketPhase.AFTER)

    def test_us_fully_closed_overnight(self) -> None:
        # 01:00 ET -> 05:00 UTC
        self.assertTrue(is_market_fully_closed("US", _utc(2026, 7, 9, 5, 0)))

    def test_us_holiday_closed(self) -> None:
        # 2026-07-03 is a listed US market holiday; 10:00 ET -> 14:00 UTC
        self.assertIs(market_phase("US", _utc(2026, 7, 3, 14, 0)), MarketPhase.CLOSED)

    def test_unknown_group_closed(self) -> None:
        self.assertIs(market_phase("MARS", _utc(2026, 7, 9, 1, 0)), MarketPhase.CLOSED)

    def test_market_has_live_session_is_inverse_of_closed(self) -> None:
        when = _utc(2026, 7, 9, 1, 0)
        for group in ("KRX", "US"):
            self.assertEqual(
                market_has_live_session(group, when),
                not is_market_fully_closed(group, when),
            )


class SnapshotToTickTests(unittest.TestCase):
    def test_builds_tick_with_fallback_source(self) -> None:
        tick = market_snapshot_to_tick("005930", _FakeSnapshot(71900.0), now=_utc(2026, 7, 9, 13, 0))
        self.assertIsNotNone(tick)
        assert tick is not None
        self.assertEqual(tick.symbol, "005930")
        self.assertEqual(tick.price, 71900.0)
        self.assertEqual(tick.source, KIS_REST_SNAPSHOT_SOURCE)
        self.assertNotEqual(tick.source, KIS_REALTIME_SOURCE)

    def test_normalizes_numeric_symbol_and_uppercases_alpha(self) -> None:
        self.assertEqual(market_snapshot_to_tick("5930", _FakeSnapshot(1.0)).symbol, "005930")
        self.assertEqual(market_snapshot_to_tick("aapl", _FakeSnapshot(1.0)).symbol, "AAPL")

    def test_non_positive_price_returns_none(self) -> None:
        self.assertIsNone(market_snapshot_to_tick("005930", _FakeSnapshot(0.0)))
        self.assertIsNone(market_snapshot_to_tick("005930", _FakeSnapshot(-5.0)))
        self.assertIsNone(market_snapshot_to_tick("005930", None))


class RefreshIntoStoreTests(unittest.TestCase):
    def _store(self) -> RealtimeMarketDataStore:
        tmp = Path(tempfile.mkdtemp()) / "rt.sqlite3"
        return RealtimeMarketDataStore(db_path=tmp)

    def test_saves_snapshots_and_populates_latest_tick(self) -> None:
        store = self._store()
        prices = {"005930": 71900.0, "000660": 251000.0}
        result = refresh_rest_snapshot_into_store(
            ["005930", "000660"],
            store=store,
            refresher=lambda sym, mkt, when: _FakeSnapshot(prices[sym]),
            market_of=lambda _s: "KRX",
            now=_utc(2026, 7, 9, 13, 0),
        )
        self.assertEqual(result["symbols"], 2)
        self.assertEqual(result["saved"], 2)
        latest = store.latest_tick("005930")
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.price, 71900.0)
        self.assertEqual(latest.source, KIS_REST_SNAPSHOT_SOURCE)

    def test_refresher_exception_is_isolated_per_symbol(self) -> None:
        store = self._store()

        def _refresher(sym, mkt, when):
            if sym == "000660":
                raise RuntimeError("quote fetch failed")
            return _FakeSnapshot(71900.0)

        result = refresh_rest_snapshot_into_store(
            ["005930", "000660"],
            store=store,
            refresher=_refresher,
            market_of=lambda _s: "KRX",
            now=_utc(2026, 7, 9, 13, 0),
        )
        self.assertEqual(result["saved"], 1)
        self.assertEqual(result["errors"], 1)
        self.assertIsNotNone(store.latest_tick("005930"))
        self.assertIsNone(store.latest_tick("000660"))

    def test_closed_market_snapshot_is_not_live_buy_eligible(self) -> None:
        # A closed-market fallback tick must NOT pass live-buy health, because its
        # source is not KIS_REALTIME_SOURCE.
        from app.data.market_data_health import evaluate_market_data_health

        store = self._store()
        now = _utc(2026, 7, 9, 13, 0)
        refresh_rest_snapshot_into_store(
            ["005930"],
            store=store,
            refresher=lambda sym, mkt, when: _FakeSnapshot(71900.0),
            market_of=lambda _s: "KRX",
            now=now,
        )
        health = evaluate_market_data_health(store, "005930", now=now)
        self.assertFalse(health.ok_for_live_buy)
        self.assertIn("QUOTE_SOURCE_NOT_KIS_REALTIME", health.reason_codes)


if __name__ == "__main__":
    unittest.main()

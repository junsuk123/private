from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.data.realtime_store import RealtimeMarketDataStore
from app.data.realtime_types import (
    KIS_REALTIME_SOURCE,
    OrderbookLevel,
    RealtimeOrderbookSnapshot,
    RealtimeTradeTick,
)
from app.features.live_feature_frame import FeatureFrameError, LiveFeatureFrameBuilder


def _seed(store: RealtimeMarketDataStore, symbol: str, age_seconds: float, now: datetime) -> None:
    ts = now - timedelta(seconds=age_seconds)
    store.save_ticks(
        (RealtimeTradeTick(symbol, ts, ts, KIS_REALTIME_SOURCE, 100.0, 10, sequence_key=f"{symbol}:{age_seconds}"),)
    )
    store.save_orderbooks(
        (
            RealtimeOrderbookSnapshot(
                symbol, ts, ts, KIS_REALTIME_SOURCE,
                (OrderbookLevel(99.9, 100, 100.1, 100),),
                sequence_key=f"{symbol}b:{age_seconds}",
            ),
        )
    )


def test_us_freshness_window_is_wider_than_kr() -> None:
    now = datetime(2026, 7, 2, 15, 0, tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory() as tmp:
        store = RealtimeMarketDataStore(Path(tmp) / "rt.sqlite3")
        _seed(store, "AAPL", age_seconds=30, now=now)     # US, REST-polled
        _seed(store, "005930", age_seconds=30, now=now)   # KR, websocket
        builder = LiveFeatureFrameBuilder(store)

        # KR: 30s > 15s -> blocked as stale market data.
        try:
            builder.build("005930", decision_time=now)
            raise AssertionError("KR 30s-old quote should be stale-blocked")
        except FeatureFrameError as exc:
            assert "MARKET_DATA_NOT_LIVE_BUY_ELIGIBLE" in str(exc)
            assert "STALE" in str(exc)

        # US: 30s < 90s -> NOT blocked by the freshness gate (may still fail later for
        # other feature reasons, but never for MARKET_DATA_NOT_LIVE_BUY_ELIGIBLE staleness).
        try:
            builder.build("AAPL", decision_time=now)
        except FeatureFrameError as exc:
            assert "MARKET_DATA_NOT_LIVE_BUY_ELIGIBLE" not in str(exc), f"US wrongly stale-blocked: {exc}"


def test_us_beyond_its_own_window_is_still_blocked() -> None:
    now = datetime(2026, 7, 2, 15, 0, tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory() as tmp:
        store = RealtimeMarketDataStore(Path(tmp) / "rt.sqlite3")
        _seed(store, "AAPL", age_seconds=600, now=now)  # 10 min old -> beyond 90s US window
        builder = LiveFeatureFrameBuilder(store)
        try:
            builder.build("AAPL", decision_time=now)
            raise AssertionError("US 10-min-old quote should be stale-blocked")
        except FeatureFrameError as exc:
            assert "MARKET_DATA_NOT_LIVE_BUY_ELIGIBLE" in str(exc)
            assert "STALE" in str(exc)

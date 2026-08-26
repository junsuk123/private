from datetime import datetime, timedelta, timezone

from app.data.market_capabilities import FeedScope, MarketGroup, SessionId, Venue
from app.data.realtime_store import RealtimeMarketDataStore
from app.data.realtime_types import FeedMetadata, RealtimeMinuteBar


def _bar(symbol, stamp, close, meta):
    return RealtimeMinuteBar(
        symbol=symbol,
        minute_start=stamp,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=10,
        vwap=close,
        trade_count=1,
        spread_bps=0,
        orderbook_imbalance=0,
        liquidity_score=0,
        volatility=0,
        last_update_age_ms=0,
        meta=meta,
    )


def test_reconciled_history_overlays_live_boundary_without_mixing_venues(tmp_path):
    store = RealtimeMarketDataStore(tmp_path / "market.sqlite3")
    start = datetime(2026, 8, 25, 13, 0, tzinfo=timezone.utc)
    historical = FeedMetadata(
        market_group=MarketGroup.US,
        exchange="NAS",
        venue=Venue.NASDAQ,
        session=SessionId.US_REGULAR,
        currency="USD",
        feed_scope=FeedScope.HISTORICAL,
        tr_id="history",
    )
    live = FeedMetadata(
        market_group=MarketGroup.US,
        exchange="NAS",
        venue=Venue.NASDAQ,
        session=SessionId.US_REGULAR,
        currency="USD",
        feed_scope=FeedScope.FREE_REALTIME,
        tr_id="live",
        is_tradeable=True,
    )
    nyse = FeedMetadata(
        market_group=MarketGroup.US,
        exchange="NYS",
        venue=Venue.NYSE,
        session=SessionId.US_REGULAR,
        currency="USD",
        feed_scope=FeedScope.HISTORICAL,
        tr_id="wrong-venue",
    )
    store.save_minute_bars(
        (
            _bar("ABC", start, 10, historical),
            _bar("ABC", start + timedelta(minutes=1), 11, historical),
            _bar("ABC", start + timedelta(minutes=1), 12, live),
            _bar("ABC", start + timedelta(minutes=2), 999, nyse),
        )
    )
    rows = store.reconciled_minute_bars(
        "ABC", start - timedelta(minutes=1), limit=10, market="US"
    )
    assert [row.close for row in rows] == [10, 12]
    assert rows[-1].meta.is_tradeable is True

from __future__ import annotations

from pathlib import Path

from app.data.realtime_store import RealtimeMarketDataStore


def run_realtime_market_data_migrations(
    db_path: str | Path = "data/store/realtime_market_data.sqlite3",
) -> Path:
    store = RealtimeMarketDataStore(db_path)
    return store.db_path

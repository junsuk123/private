from __future__ import annotations

from pathlib import Path

from app.data.realtime_store import RealtimeMarketDataStore
from app.paths import realtime_market_database_path


def run_realtime_market_data_migrations(
    db_path: str | Path | None = None,
) -> Path:
    store = RealtimeMarketDataStore(db_path or realtime_market_database_path())
    return store.db_path

"""REST snapshot fallback for fully-closed markets.

When a market group is fully closed (see :mod:`app.data.market_session`), the KIS
realtime WebSocket delivers only control/PINGPONG frames — zero trade ticks. This
module keeps last-known prices fresh by fetching a REST quote per symbol and
writing it into the realtime store as a :data:`KIS_REST_SNAPSHOT_SOURCE` tick.

That distinct source is deliberate: market-data health requires
``KIS_REALTIME_SOURCE`` for live-buy eligibility, so a closed-market REST snapshot
populates ``latest_tick`` (valuation, dashboard, price history) WITHOUT ever being
mistaken for a live-tradeable quote. It is a fallback, never the primary feed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from app.data.realtime_store import RealtimeMarketDataStore
from app.data.realtime_types import KIS_REST_SNAPSHOT_SOURCE, RealtimeTradeTick

# refresher(symbol, market, when) -> object with a numeric ``last_price`` (or None).
MarketSnapshotRefresher = Callable[[str, str, datetime], Any]


def _normalize_symbol(symbol: str) -> str:
    text = str(symbol or "").strip()
    return text.zfill(6) if text.isdigit() else text.upper()


def market_snapshot_to_tick(
    symbol: str,
    snapshot: Any,
    *,
    now: datetime | None = None,
) -> RealtimeTradeTick | None:
    """Convert a broker ``MarketSnapshot`` (or any object exposing ``last_price``)
    into a fallback :class:`RealtimeTradeTick`. Returns ``None`` for a
    missing/non-positive price so bad quotes never enter the store."""
    if snapshot is None:
        return None
    price = float(getattr(snapshot, "last_price", 0.0) or 0.0)
    if price <= 0:
        return None
    normalized = _normalize_symbol(symbol)
    if not normalized:
        return None
    stamp = now or datetime.now(timezone.utc)
    return RealtimeTradeTick(
        symbol=normalized,
        exchange_timestamp=stamp,
        received_at=stamp,
        source=KIS_REST_SNAPSHOT_SOURCE,
        price=price,
        volume=0,
        trade_direction=None,
        sequence_key=f"kis-rest-snapshot:{normalized}:{stamp.isoformat()}",
    )


def refresh_rest_snapshot_into_store(
    symbols: Iterable[str],
    *,
    store: RealtimeMarketDataStore,
    refresher: MarketSnapshotRefresher,
    market_of: Callable[[str], str],
    now: datetime | None = None,
) -> dict[str, int]:
    """Fetch a REST quote for each symbol and save it as a fallback tick.

    ``refresher`` and ``market_of`` are injected so callers wire in the concrete
    KIS client / market routing (and tests can pass fakes). Every failure is
    isolated per symbol so one bad quote never aborts the batch.
    """
    stamp = now or datetime.now(timezone.utc)
    unique = tuple(dict.fromkeys(_normalize_symbol(s) for s in symbols if str(s or "").strip()))
    saved = 0
    errors = 0
    for symbol in unique:
        try:
            snapshot = refresher(symbol, market_of(symbol), stamp)
        except Exception:  # noqa: BLE001 - one bad quote must not abort the batch.
            errors += 1
            continue
        tick = market_snapshot_to_tick(symbol, snapshot, now=stamp)
        if tick is None:
            continue
        try:
            saved += store.save_ticks((tick,))
        except Exception:  # noqa: BLE001 - store write is best-effort for a fallback.
            errors += 1
    return {"symbols": len(unique), "saved": saved, "errors": errors}

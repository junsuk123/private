from __future__ import annotations

from datetime import datetime, timezone

from app.data.realtime_store import RealtimeMarketDataStore
from app.data.realtime_types import KIS_REALTIME_SOURCE, MarketDataHealth


def evaluate_market_data_health(
    store: RealtimeMarketDataStore,
    symbol: str,
    *,
    max_quote_age_ms: int = 3000,
    max_orderbook_age_ms: int = 3000,
    minimum_source_quality_score: float = 0.85,
    now: datetime | None = None,
) -> MarketDataHealth:
    now = now or datetime.now(timezone.utc)
    tick = store.latest_tick(symbol)
    orderbook = store.latest_orderbook(symbol)
    reasons: list[str] = []
    quote_count = 1 if tick is not None else 0
    orderbook_count = 1 if orderbook is not None else 0

    if tick is None:
        reasons.append("QUOTE_COUNT_ZERO")
    else:
        age_ms = max(0.0, (now - tick.received_at).total_seconds() * 1000)
        if age_ms > max_quote_age_ms:
            reasons.append("QUOTE_STALE")
        if tick.source != KIS_REALTIME_SOURCE:
            reasons.append("QUOTE_SOURCE_NOT_KIS_REALTIME")

    if orderbook is None:
        reasons.append("ORDERBOOK_COUNT_ZERO")
    else:
        age_ms = max(0.0, (now - orderbook.received_at).total_seconds() * 1000)
        if age_ms > max_orderbook_age_ms:
            reasons.append("ORDERBOOK_STALE")
        if orderbook.source != KIS_REALTIME_SOURCE:
            reasons.append("ORDERBOOK_SOURCE_NOT_KIS_REALTIME")

    source_quality_score = 1.0 if not reasons or all("STALE" in reason for reason in reasons) else 0.0
    if source_quality_score < minimum_source_quality_score:
        reasons.append("SOURCE_QUALITY_TOO_LOW")

    health = MarketDataHealth(
        symbol=symbol,
        checked_at=now,
        quote_count=quote_count,
        orderbook_count=orderbook_count,
        latest_tick_at=tick.received_at if tick else None,
        latest_orderbook_at=orderbook.received_at if orderbook else None,
        max_quote_age_ms=max_quote_age_ms,
        max_orderbook_age_ms=max_orderbook_age_ms,
        source=KIS_REALTIME_SOURCE if tick or orderbook else "missing",
        source_quality_score=source_quality_score,
        ok_for_live_buy=not reasons,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )
    store.save_health(health)
    return health


def require_fresh_live_buy_data(health: MarketDataHealth) -> None:
    if not health.ok_for_live_buy:
        raise RuntimeError("LIVE_BUY_MARKET_DATA_BLOCKED:" + ",".join(health.reason_codes))

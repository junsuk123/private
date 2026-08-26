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
    quote_candidates = tuple(item for item in (tick, orderbook) if item is not None)
    quote_source = (
        max(quote_candidates, key=lambda item: item.received_at)
        if quote_candidates
        else None
    )
    quote_count = 1 if quote_source is not None else 0
    orderbook_count = 1 if orderbook is not None else 0

    if quote_source is None:
        reasons.append("QUOTE_COUNT_ZERO")
    else:
        age_ms = max(0.0, (now - quote_source.received_at).total_seconds() * 1000)
        if age_ms > max_quote_age_ms:
            reasons.append("QUOTE_STALE")
        if quote_source.source != KIS_REALTIME_SOURCE:
            reasons.append("QUOTE_SOURCE_NOT_KIS_REALTIME")
        eligible, metadata_reasons = quote_source.meta.is_live_buy_eligible()
        if not eligible:
            reasons.extend(metadata_reasons)

    if orderbook is None:
        reasons.append("ORDERBOOK_COUNT_ZERO")
    else:
        age_ms = max(0.0, (now - orderbook.received_at).total_seconds() * 1000)
        if age_ms > max_orderbook_age_ms:
            reasons.append("ORDERBOOK_STALE")
        if orderbook.source != KIS_REALTIME_SOURCE:
            reasons.append("ORDERBOOK_SOURCE_NOT_KIS_REALTIME")
        eligible, metadata_reasons = orderbook.meta.is_live_buy_eligible()
        if not eligible:
            reasons.extend(metadata_reasons)

    source_quality_score = 1.0 if not reasons or all("STALE" in reason for reason in reasons) else 0.0
    if source_quality_score < minimum_source_quality_score:
        reasons.append("SOURCE_QUALITY_TOO_LOW")

    authoritative = orderbook or quote_source
    meta = authoritative.meta if authoritative is not None else None
    health = MarketDataHealth(
        symbol=symbol,
        checked_at=now,
        quote_count=quote_count,
        orderbook_count=orderbook_count,
        latest_tick_at=quote_source.received_at if quote_source else None,
        latest_orderbook_at=orderbook.received_at if orderbook else None,
        max_quote_age_ms=max_quote_age_ms,
        max_orderbook_age_ms=max_orderbook_age_ms,
        source=KIS_REALTIME_SOURCE if tick or orderbook else "missing",
        source_quality_score=source_quality_score,
        ok_for_live_buy=not reasons,
        reason_codes=tuple(dict.fromkeys(reasons)),
        market_group=(meta.market_group.value if meta and meta.market_group else ""),
        venue=(meta.venue.value if meta else ""),
        session=(meta.session.value if meta else ""),
        feed_scope=(meta.feed_scope.value if meta else ""),
        depth_level_count=(len(orderbook.levels) if orderbook is not None else 0),
        is_consolidated=bool(meta.is_consolidated) if meta else False,
    )
    store.save_health(health)
    return health


def require_fresh_live_buy_data(health: MarketDataHealth) -> None:
    if not health.ok_for_live_buy:
        raise RuntimeError("LIVE_BUY_MARKET_DATA_BLOCKED:" + ",".join(health.reason_codes))

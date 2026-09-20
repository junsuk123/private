from __future__ import annotations

from datetime import datetime, timezone

from app.data.market_capabilities import FeedScope, ReasonCode
from app.data.realtime_store import RealtimeMarketDataStore
from app.data.realtime_types import FeedMetadata, KIS_REALTIME_SOURCE, MarketDataHealth


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

    if orderbook is None:
        reasons.append("ORDERBOOK_COUNT_ZERO")
    else:
        age_ms = max(0.0, (now - orderbook.received_at).total_seconds() * 1000)
        if age_ms > max_orderbook_age_ms:
            reasons.append("ORDERBOOK_STALE")
        if orderbook.source != KIS_REALTIME_SOURCE:
            reasons.append("ORDERBOOK_SOURCE_NOT_KIS_REALTIME")

    # ``source`` alone cannot separate the KIS overseas REST snapshot path from a real
    # WebSocket feed: both carry KIS_REALTIME_SOURCE. The event metadata can, and
    # docs/realtime_session_gap_analysis.md §4.1 measured that path at ~40% of US
    # orderbooks. A cumulative-session snapshot is not evidence a live entry can be
    # priced against, so it is refused here by feed_scope.
    #
    # Only this positively identified scope is refused. The wider switch to
    # FeedMetadata.is_live_buy_eligible() -- which also fails closed on UNKNOWN venue
    # and session -- is Phase 6 in that document and needs operator sign-off, because
    # it removes far more than the REST path.
    meta = _evidence_metadata(quote_source, orderbook)
    if meta.feed_scope is FeedScope.REST_SNAPSHOT:
        reasons.append(ReasonCode.REST_SNAPSHOT_ONLY.value)

    source_quality_score = 1.0 if not reasons or all("STALE" in reason for reason in reasons) else 0.0
    if source_quality_score < minimum_source_quality_score:
        reasons.append("SOURCE_QUALITY_TOO_LOW")

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
        market_group=meta.market_group.value if meta.market_group else "",
        venue=meta.venue.value,
        session=meta.session.value,
        feed_scope=meta.feed_scope.value,
        depth_level_count=len(orderbook.levels) if orderbook is not None else 0,
        is_consolidated=bool(meta.is_consolidated),
    )
    store.save_health(health)
    return health


def _evidence_metadata(quote_source: object | None, orderbook: object | None) -> FeedMetadata:
    """Metadata of the record the live-buy decision would actually rest on.

    The freshest quote wins, because that is what ``quote_source`` above already
    selected; the orderbook stands in when there is no quote at all. An empty default
    is returned when neither exists, so the caller reads UNKNOWN rather than nothing.
    """
    for record in (quote_source, orderbook):
        meta = getattr(record, "meta", None)
        if isinstance(meta, FeedMetadata):
            return meta
    return FeedMetadata()


def require_fresh_live_buy_data(health: MarketDataHealth) -> None:
    if not health.ok_for_live_buy:
        raise RuntimeError("LIVE_BUY_MARKET_DATA_BLOCKED:" + ",".join(health.reason_codes))

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


KIS_REALTIME_SOURCE = "kis_realtime_websocket"


@dataclass(frozen=True)
class RealtimeTradeTick:
    symbol: str
    exchange_timestamp: datetime
    received_at: datetime
    source: str
    price: float
    volume: int
    trade_direction: str | None = None
    sequence_key: str | None = None
    raw_checksum: str | None = None
    latency_ms: float = 0.0

    @property
    def record_id(self) -> str:
        key = self.sequence_key or f"{self.symbol}:{self.exchange_timestamp.isoformat()}:{self.price}:{self.volume}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class OrderbookLevel:
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int


@dataclass(frozen=True)
class RealtimeOrderbookSnapshot:
    symbol: str
    exchange_timestamp: datetime
    received_at: datetime
    source: str
    levels: tuple[OrderbookLevel, ...]
    sequence_key: str | None = None
    raw_checksum: str | None = None
    latency_ms: float = 0.0

    @property
    def best_bid(self) -> float:
        return self.levels[0].bid_price if self.levels else 0.0

    @property
    def best_ask(self) -> float:
        return self.levels[0].ask_price if self.levels else 0.0

    @property
    def spread_bps(self) -> float:
        bid = self.best_bid
        ask = self.best_ask
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
        return ((ask - bid) / mid) * 10_000 if mid > 0 and ask >= bid else 0.0

    @property
    def total_bid_volume(self) -> int:
        return sum(level.bid_size for level in self.levels)

    @property
    def total_ask_volume(self) -> int:
        return sum(level.ask_size for level in self.levels)

    @property
    def imbalance(self) -> float:
        total = self.total_bid_volume + self.total_ask_volume
        if total <= 0:
            return 0.0
        return (self.total_bid_volume - self.total_ask_volume) / total

    @property
    def record_id(self) -> str:
        key = self.sequence_key or f"{self.symbol}:{self.exchange_timestamp.isoformat()}:{self.best_bid}:{self.best_ask}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class RealtimeMinuteBar:
    symbol: str
    minute_start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    trade_count: int
    spread_bps: float
    orderbook_imbalance: float
    liquidity_score: float
    volatility: float
    last_update_age_ms: float
    source_record_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class MarketDataHealth:
    symbol: str
    checked_at: datetime
    quote_count: int
    orderbook_count: int
    latest_tick_at: datetime | None
    latest_orderbook_at: datetime | None
    max_quote_age_ms: int
    max_orderbook_age_ms: int
    source: str
    source_quality_score: float
    ok_for_live_buy: bool
    reason_codes: tuple[str, ...] = field(default_factory=tuple)


def checksum(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def aware_now() -> datetime:
    return datetime.now(timezone.utc)


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value

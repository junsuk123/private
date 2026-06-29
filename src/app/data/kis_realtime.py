from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.data.realtime_store import RealtimeMarketDataStore
from app.data.realtime_types import (
    KIS_REALTIME_SOURCE,
    OrderbookLevel,
    RealtimeOrderbookSnapshot,
    RealtimeTradeTick,
    checksum,
)


TRADE_TR_IDS = {"H0STCNT0", "H0STCNI0"}
ORDERBOOK_TR_IDS = {"H0STASP0"}


@dataclass(frozen=True)
class ParsedKisRealtimeMessage:
    ticks: tuple[RealtimeTradeTick, ...] = ()
    orderbooks: tuple[RealtimeOrderbookSnapshot, ...] = ()
    event_type: str = "unknown"


@dataclass
class KisRealtimeSubscriptionManager:
    store: RealtimeMarketDataStore
    message_source: Callable[[], Awaitable[str | None]]
    symbols: set[str] = field(default_factory=set)
    running: bool = False

    def subscribe(self, symbols: Iterable[str]) -> None:
        for symbol in symbols:
            normalized = normalize_symbol(symbol)
            if normalized:
                self.symbols.add(normalized)

    async def run_forever(self, *, max_messages: int | None = None) -> dict[str, int]:
        self.running = True
        counts = {"messages": 0, "ticks": 0, "orderbooks": 0}
        while self.running:
            raw = await self.message_source()
            if raw is None:
                break
            parsed = parse_kis_realtime_message(raw)
            ticks = tuple(tick for tick in parsed.ticks if not self.symbols or tick.symbol in self.symbols)
            orderbooks = tuple(
                item for item in parsed.orderbooks if not self.symbols or item.symbol in self.symbols
            )
            counts["ticks"] += self.store.save_ticks(ticks)
            counts["orderbooks"] += self.store.save_orderbooks(orderbooks)
            for symbol in {tick.symbol for tick in ticks} | {book.symbol for book in orderbooks}:
                self.store.build_latest_minute_bar(symbol)
            counts["messages"] += 1
            if max_messages is not None and counts["messages"] >= max_messages:
                break
        self.running = False
        return counts

    def shutdown(self) -> None:
        self.running = False


def parse_kis_realtime_message(raw: str, *, received_at: datetime | None = None) -> ParsedKisRealtimeMessage:
    received_at = received_at or datetime.now(timezone.utc)
    raw = raw.strip()
    if not raw:
        return ParsedKisRealtimeMessage(event_type="empty")
    parts = raw.split("|")
    if len(parts) >= 4 and parts[1] in TRADE_TR_IDS:
        tick = _parse_trade_payload(parts[-1], raw=raw, received_at=received_at)
        return ParsedKisRealtimeMessage(ticks=(tick,), event_type="trade")
    if len(parts) >= 4 and parts[1] in ORDERBOOK_TR_IDS:
        orderbook = _parse_orderbook_payload(parts[-1], raw=raw, received_at=received_at)
        return ParsedKisRealtimeMessage(orderbooks=(orderbook,), event_type="orderbook")
    if raw.startswith("{"):
        return ParsedKisRealtimeMessage(event_type="json_control")
    raise ValueError("unsupported KIS realtime message format")


def _parse_trade_payload(payload: str, *, raw: str, received_at: datetime) -> RealtimeTradeTick:
    fields = payload.split("^")
    if len(fields) < 4:
        raise ValueError("KIS trade payload has too few fields")
    symbol = normalize_symbol(fields[0])
    exchange_timestamp = _timestamp_from_hhmmss(fields[1], received_at)
    price = _float(fields[2])
    volume = int(_float(fields[3]))
    direction = fields[4] if len(fields) > 4 and fields[4] else None
    sequence = fields[5] if len(fields) > 5 and fields[5] else f"{symbol}:{fields[1]}:{price}:{volume}"
    return RealtimeTradeTick(
        symbol=symbol,
        exchange_timestamp=exchange_timestamp,
        received_at=received_at,
        source=KIS_REALTIME_SOURCE,
        price=price,
        volume=volume,
        trade_direction=direction,
        sequence_key=sequence,
        raw_checksum=checksum(raw),
        latency_ms=max(0.0, (received_at - exchange_timestamp).total_seconds() * 1000),
    )


def _parse_orderbook_payload(payload: str, *, raw: str, received_at: datetime) -> RealtimeOrderbookSnapshot:
    fields = payload.split("^")
    if len(fields) < 6:
        raise ValueError("KIS orderbook payload has too few fields")
    symbol = normalize_symbol(fields[0])
    exchange_timestamp = _timestamp_from_hhmmss(fields[1], received_at)
    numbers = [_float(item) for item in fields[2:]]
    levels: list[OrderbookLevel] = []
    for index in range(0, len(numbers) - 3, 4):
        ask_price, bid_price, ask_size, bid_size = numbers[index : index + 4]
        if ask_price <= 0 and bid_price <= 0:
            continue
        levels.append(
            OrderbookLevel(
                bid_price=bid_price,
                bid_size=int(bid_size),
                ask_price=ask_price,
                ask_size=int(ask_size),
            )
        )
    if not levels:
        raise ValueError("KIS orderbook payload did not contain any price levels")
    sequence = f"{symbol}:{fields[1]}:{levels[0].bid_price}:{levels[0].ask_price}:{checksum(payload)[:8]}"
    return RealtimeOrderbookSnapshot(
        symbol=symbol,
        exchange_timestamp=exchange_timestamp,
        received_at=received_at,
        source=KIS_REALTIME_SOURCE,
        levels=tuple(levels),
        sequence_key=sequence,
        raw_checksum=checksum(raw),
        latency_ms=max(0.0, (received_at - exchange_timestamp).total_seconds() * 1000),
    )


def normalize_symbol(symbol: str) -> str:
    text = str(symbol).strip()
    return text.zfill(6) if text.isdigit() else text


def _timestamp_from_hhmmss(value: str, received_at: datetime) -> datetime:
    text = value.strip()
    if len(text) < 6 or not text[:6].isdigit():
        return received_at
    hour = int(text[:2])
    minute = int(text[2:4])
    second = int(text[4:6])
    microsecond = int((text[6:] or "0").ljust(6, "0")[:6])
    return received_at.replace(hour=hour, minute=minute, second=second, microsecond=microsecond)


def _float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(str(value).replace(",", ""))


class QueueMessageSource:
    def __init__(self, messages: Iterable[str]) -> None:
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        for message in messages:
            self._queue.put_nowait(message)
        self._queue.put_nowait(None)

    async def __call__(self) -> str | None:
        return await self._queue.get()

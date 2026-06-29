from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from typing import Any
from zoneinfo import ZoneInfo

from app.data.realtime_store import RealtimeMarketDataStore
from app.data.realtime_types import (
    KIS_REALTIME_SOURCE,
    OrderbookLevel,
    RealtimeOrderbookSnapshot,
    RealtimeTradeTick,
    checksum,
)
from app.execution.kis_auth import build_kis_client, issue_websocket_approval_key
from app.features.live_feature_frame import FeatureFrameError, LiveFeatureFrameBuilder

if TYPE_CHECKING:
    from app.execution.kis_real import KisDevelopersApiClient


TRADE_TR_IDS = {"H0STCNT0", "H0STCNI0"}
ORDERBOOK_TR_IDS = {"H0STASP0"}
DEFAULT_SUBSCRIPTION_TR_IDS = ("H0STCNT0", "H0STASP0")
KIS_REALTIME_LIVE_WS_URL = "ws://ops.koreainvestment.com:21000"
KIS_REALTIME_PAPER_WS_URL = "ws://ops.koreainvestment.com:31000"
KIS_EXCHANGE_TIMEZONE = ZoneInfo("Asia/Seoul")


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


async def run_kis_realtime_websocket_collector(
    *,
    symbols: Iterable[str],
    store: RealtimeMarketDataStore | None = None,
    client: "KisDevelopersApiClient | None" = None,
    url: str | None = None,
    stop_event: Any | None = None,
    max_messages: int | None = None,
) -> dict[str, int]:
    websockets = _load_websockets()
    client = client or build_kis_client(enabled=True)
    approval_key = issue_websocket_approval_key(client)
    target_url = url or _kis_realtime_websocket_url(paper=client.paper)
    normalized_symbols = tuple(symbol for symbol in (normalize_symbol(item) for item in symbols) if symbol)
    store = store or RealtimeMarketDataStore()
    counts = {"messages": 0, "ticks": 0, "orderbooks": 0, "subscriptions": 0}
    feature_builder = LiveFeatureFrameBuilder(store)
    async with websockets.connect(target_url, ping_interval=20, ping_timeout=20) as websocket:
        for symbol in normalized_symbols:
            for tr_id in DEFAULT_SUBSCRIPTION_TR_IDS:
                await websocket.send(kis_realtime_subscription_message(approval_key, tr_id, symbol))
                counts["subscriptions"] += 1
        while stop_event is None or not stop_event.is_set():
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=1.0)
            except TimeoutError:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            parsed = parse_kis_realtime_message(str(raw))
            ticks = tuple(tick for tick in parsed.ticks if tick.symbol in normalized_symbols)
            orderbooks = tuple(book for book in parsed.orderbooks if book.symbol in normalized_symbols)
            counts["ticks"] += store.save_ticks(ticks)
            counts["orderbooks"] += store.save_orderbooks(orderbooks)
            for symbol in {tick.symbol for tick in ticks} | {book.symbol for book in orderbooks}:
                store.build_latest_minute_bar(symbol)
                try:
                    feature_builder.build(symbol)
                    counts["feature_frames"] = counts.get("feature_frames", 0) + 1
                except (FeatureFrameError, RuntimeError, ValueError) as exc:
                    counts["feature_frame_errors"] = counts.get("feature_frame_errors", 0) + 1
                    counts["last_feature_frame_error"] = str(exc) or exc.__class__.__name__
            counts["messages"] += 1
            if max_messages is not None and counts["messages"] >= max_messages:
                break
    return counts


def kis_realtime_subscription_message(approval_key: str, tr_id: str, symbol: str) -> str:
    return json.dumps(
        {
            "header": {
                "approval_key": approval_key,
                "custtype": os.getenv("KIS_CUSTTYPE", "P"),
                "tr_type": "1",
                "content-type": "utf-8",
            },
            "body": {"input": {"tr_id": tr_id, "tr_key": normalize_symbol(symbol)}},
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _kis_realtime_websocket_url(*, paper: bool) -> str:
    explicit = os.getenv("KIS_WEBSOCKET_URL", "").strip()
    if explicit:
        return explicit
    return os.getenv(
        "KIS_PAPER_WEBSOCKET_URL" if paper else "KIS_LIVE_WEBSOCKET_URL",
        KIS_REALTIME_PAPER_WS_URL if paper else KIS_REALTIME_LIVE_WS_URL,
    ).strip()


def _load_websockets() -> Any:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("KIS_REALTIME_WEBSOCKETS_DEPENDENCY_MISSING") from exc
    return websockets


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
    levels: list[OrderbookLevel] = []
    if len(fields) >= 43:
        ask_prices = [_float(item) for item in fields[3:13]]
        bid_prices = [_float(item) for item in fields[13:23]]
        ask_sizes = [_float(item) for item in fields[23:33]]
        bid_sizes = [_float(item) for item in fields[33:43]]
        for ask_price, bid_price, ask_size, bid_size in zip(
            ask_prices,
            bid_prices,
            ask_sizes,
            bid_sizes,
            strict=True,
        ):
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
    else:
        numbers = [_float(item) for item in fields[2:]]
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
    received_local = received_at.astimezone(KIS_EXCHANGE_TIMEZONE)
    exchange_local = received_local.replace(
        hour=hour,
        minute=minute,
        second=second,
        microsecond=microsecond,
    )
    if exchange_local > received_local.replace(microsecond=microsecond) and (
        exchange_local - received_local
    ).total_seconds() > 5 * 60:
        exchange_local -= timedelta(days=1)
    return exchange_local.astimezone(timezone.utc)


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

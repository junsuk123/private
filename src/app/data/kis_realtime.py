from __future__ import annotations

import asyncio
import json
import os
import time
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
DEFAULT_SUBSCRIPTION_TR_IDS = ("H0STASP0",)
KIS_REALTIME_LIVE_WS_URL = "ws://ops.koreainvestment.com:21000/tryitout"
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
            try:
                parsed = parse_kis_realtime_message(raw)
            except ValueError:
                counts["parse_errors"] = counts.get("parse_errors", 0) + 1
                counts["messages"] += 1
                continue
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


def _websocket_ping_setting(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"", "0", "false", "no", "off", "none", "null"}:
        return None
    try:
        return max(1.0, float(value))
    except ValueError:
        return default


def _websocket_subscription_delay_seconds() -> float:
    raw = os.getenv("KIS_REALTIME_SUBSCRIBE_DELAY_SEC", "1.0")
    try:
        return max(0.0, min(5.0, float(raw)))
    except (TypeError, ValueError):
        return 1.0


def _websocket_post_subscribe_drain_seconds() -> float:
    raw = os.getenv("KIS_REALTIME_POST_SUBSCRIBE_DRAIN_SEC", "0.6")
    try:
        return max(0.0, min(3.0, float(raw)))
    except (TypeError, ValueError):
        return 0.6


async def run_kis_realtime_websocket_collector(
    *,
    symbols: Iterable[str],
    store: RealtimeMarketDataStore | None = None,
    client: "KisDevelopersApiClient | None" = None,
    url: str | None = None,
    stop_event: Any | None = None,
    resubscribe_event: Any | None = None,
    skip_subscriptions: Iterable[tuple[str, str]] | None = None,
    max_messages: int | None = None,
    max_runtime_seconds: float | None = None,
) -> dict[str, int]:
    websockets = _load_websockets()
    client = client or build_kis_client(enabled=True)
    approval_key = issue_websocket_approval_key(client)
    target_url = url or _kis_realtime_websocket_url()
    normalized_symbols = tuple(symbol for symbol in (normalize_symbol(item) for item in symbols) if symbol)
    skipped_pairs = {
        (normalize_symbol(symbol), str(tr_id))
        for symbol, tr_id in (skip_subscriptions or ())
        if normalize_symbol(symbol) and tr_id
    }
    store = store or RealtimeMarketDataStore()
    counts: dict[str, Any] = {
        "messages": 0,
        "ticks": 0,
        "orderbooks": 0,
        "subscriptions": 0,
        "mode": "live",
        "url": target_url,
    }
    feature_builder = LiveFeatureFrameBuilder(store)
    # KIS realtime sends application-level PINGPONG frames and doesn't reliably
    # respond to RFC websocket pings from the client library. Keep the standard
    # ping disabled by default; operators can opt in with env vars if needed.
    ping_interval = _websocket_ping_setting("KIS_REALTIME_WS_PING_INTERVAL_SECONDS", None)
    ping_timeout = _websocket_ping_setting("KIS_REALTIME_WS_PING_TIMEOUT_SECONDS", None)
    subscribe_delay = _websocket_subscription_delay_seconds()
    post_subscribe_drain = _websocket_post_subscribe_drain_seconds()
    async with websockets.connect(target_url, ping_interval=ping_interval, ping_timeout=ping_timeout) as websocket:
        for symbol in normalized_symbols:
            for tr_id in DEFAULT_SUBSCRIPTION_TR_IDS:
                if (symbol, tr_id) in skipped_pairs:
                    counts["subscriptions_skipped"] = counts.get("subscriptions_skipped", 0) + 1
                    continue
                counts["last_subscription_symbol"] = symbol
                counts["last_subscription_tr_id"] = tr_id
                try:
                    await websocket.send(kis_realtime_subscription_message(approval_key, tr_id, symbol))
                except Exception as exc:
                    if _is_websocket_connection_closed(exc):
                        counts["connection_closed"] = 1
                        counts["last_close_error"] = str(exc) or exc.__class__.__name__
                        return counts
                    raise
                counts["subscriptions"] += 1
                if post_subscribe_drain > 0.0:
                    closed = await _drain_kis_realtime_messages(
                        websocket=websocket,
                        symbols=normalized_symbols,
                        store=store,
                        feature_builder=feature_builder,
                        counts=counts,
                        seconds=post_subscribe_drain,
                        max_messages=max_messages,
                    )
                    if closed:
                        return counts
                if subscribe_delay > 0.0:
                    await asyncio.sleep(subscribe_delay)
        # Optional soft deadline so the caller can periodically reconnect with a
        # refreshed symbol set (e.g. today's affordable candidates), not just the
        # static config list.
        deadline = time.monotonic() + max_runtime_seconds if max_runtime_seconds else None
        while stop_event is None or not stop_event.is_set():
            if resubscribe_event is not None and resubscribe_event.is_set():
                counts["resubscribe_requested"] = 1
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=1.0)
            except TimeoutError:
                continue
            except Exception as exc:
                if _is_websocket_connection_closed(exc):
                    counts["connection_closed"] = 1
                    counts["last_close_error"] = str(exc) or exc.__class__.__name__
                    break
                raise
            closed = await _process_kis_realtime_raw(
                raw=raw,
                websocket=websocket,
                symbols=normalized_symbols,
                store=store,
                feature_builder=feature_builder,
                counts=counts,
            )
            if closed:
                break
            if max_messages is not None and counts["messages"] >= max_messages:
                break
    return counts


async def _drain_kis_realtime_messages(
    *,
    websocket: Any,
    symbols: tuple[str, ...],
    store: RealtimeMarketDataStore,
    feature_builder: LiveFeatureFrameBuilder,
    counts: dict[str, Any],
    seconds: float,
    max_messages: int | None,
) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if max_messages is not None and counts.get("messages", 0) >= max_messages:
            return False
        try:
            raw = await asyncio.wait_for(
                websocket.recv(),
                timeout=min(0.2, max(0.01, deadline - time.monotonic())),
            )
        except TimeoutError:
            return False
        except Exception as exc:
            if _is_websocket_connection_closed(exc):
                counts["connection_closed"] = 1
                counts["last_close_error"] = str(exc) or exc.__class__.__name__
                return True
            raise
        closed = await _process_kis_realtime_raw(
            raw=raw,
            websocket=websocket,
            symbols=symbols,
            store=store,
            feature_builder=feature_builder,
            counts=counts,
        )
        if closed:
            return True
    return False


async def _process_kis_realtime_raw(
    *,
    raw: str | bytes,
    websocket: Any,
    symbols: tuple[str, ...],
    store: RealtimeMarketDataStore,
    feature_builder: LiveFeatureFrameBuilder,
    counts: dict[str, Any],
) -> bool:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    raw = str(raw)
    # KIS uses application-level PINGPONG keepalive frames. The official
    # sample answers those frames with a websocket pong before parsing.
    if raw.startswith("{") and "PINGPONG" in raw:
        try:
            pong = getattr(websocket, "pong", None)
            if callable(pong):
                await pong(raw)
            else:
                await websocket.send(raw)
        except Exception as exc:
            if _is_websocket_connection_closed(exc):
                counts["connection_closed"] = 1
                counts["last_close_error"] = str(exc) or exc.__class__.__name__
                return True
            raise
        counts["pingpongs"] = counts.get("pingpongs", 0) + 1
        return False
    if raw.startswith("{"):
        control = kis_realtime_control_summary(raw)
        counts["control_messages"] = counts.get("control_messages", 0) + 1
        counts["last_control_message"] = control
        if _kis_realtime_control_is_appkey_in_use(control):
            counts["appkey_already_in_use"] = 1
        if _kis_realtime_control_is_error(control):
            counts["control_errors"] = counts.get("control_errors", 0) + 1
        counts["messages"] += 1
        return False
    try:
        parsed = parse_kis_realtime_message(raw)
    except ValueError as exc:
        counts["parse_errors"] = counts.get("parse_errors", 0) + 1
        counts["last_parse_error"] = str(exc) or exc.__class__.__name__
        counts["messages"] += 1
        return False
    ticks = tuple(tick for tick in parsed.ticks if tick.symbol in symbols)
    orderbooks = tuple(book for book in parsed.orderbooks if book.symbol in symbols)
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
    return False


def _is_websocket_connection_closed(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    return (
        "connectionclosed" in name
        or "connection closed" in message
        or "no close frame received or sent" in message
        or "keepalive ping timeout" in message
    )


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


def kis_realtime_control_summary(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return {"raw": str(raw)[:200]}
    if not isinstance(payload, dict):
        return {"payload_type": type(payload).__name__}
    header = payload.get("header") if isinstance(payload.get("header"), dict) else {}
    body = payload.get("body") if isinstance(payload.get("body"), dict) else {}
    output = body.get("output") if isinstance(body.get("output"), dict) else {}
    summary: dict[str, Any] = {}
    for key in ("tr_id", "tr_key", "rt_cd", "msg_cd", "msg1"):
        value = payload.get(key)
        if value in (None, ""):
            value = header.get(key)
        if value in (None, ""):
            value = body.get(key)
        if value in (None, ""):
            value = output.get(key)
        if value not in (None, ""):
            summary[key] = _scrub_kis_control_value(key, value)
    if not summary:
        for key, value in payload.items():
            if isinstance(value, (dict, list, tuple)):
                continue
            summary[key] = _scrub_kis_control_value(str(key), value)
            if len(summary) >= 6:
                break
    return summary or {"control": "empty"}


def _kis_realtime_control_is_error(summary: dict[str, Any]) -> bool:
    code = str(summary.get("rt_cd") or summary.get("msg_cd") or "").strip().lower()
    message = str(summary.get("msg1") or "").strip().lower()
    if code and code not in {"0", "success", "ok"}:
        return True
    return any(marker in message for marker in ("error", "fail", "invalid", "거부", "실패", "오류"))


def _kis_realtime_control_is_appkey_in_use(summary: dict[str, Any]) -> bool:
    code = str(summary.get("msg_cd") or "").strip().upper()
    message = str(summary.get("msg1") or "").strip().lower()
    return code == "OPSP8996" or "already in use appkey" in message


def _scrub_kis_control_value(key: str, value: Any) -> Any:
    lowered = key.lower()
    if any(token in lowered for token in ("approval", "secret", "token", "key")) and lowered not in {
        "tr_key",
    }:
        return "***"
    if isinstance(value, str):
        return value[:300]
    return value


def _kis_realtime_websocket_url() -> str:
    explicit = os.getenv("KIS_WEBSOCKET_URL", "").strip()
    if explicit:
        return _normalize_kis_realtime_websocket_url(explicit)
    return _normalize_kis_realtime_websocket_url(
        os.getenv("KIS_LIVE_WEBSOCKET_URL", KIS_REALTIME_LIVE_WS_URL).strip()
    )


def _normalize_kis_realtime_websocket_url(url: str) -> str:
    if not url:
        return KIS_REALTIME_LIVE_WS_URL
    if url.rstrip("/") in {"ws://ops.koreainvestment.com:21000", "wss://ops.koreainvestment.com:21000"}:
        return f"{url.rstrip('/')}/tryitout"
    return url


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

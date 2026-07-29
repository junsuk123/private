from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable, Container, Iterable
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


# Domestic realtime feeds. KRX-only TRs cover the 09:00-15:30 regular session;
# the 통합 (KRX+NXT) TRs additionally carry NXT, whose session runs 08:00-20:00,
# so they are what makes pre/after-hours data possible at all.
TRADE_TR_IDS = {"H0STCNT0", "H0STCNI0", "H0UNCNT0", "H0NXCNT0", "H0STOUP0"}
ORDERBOOK_TR_IDS = {"H0STASP0", "H0UNASP0", "H0NXASP0"}
OVERSEAS_TRADE_TR_IDS = {"HDFSCNT0"}
OVERSEAS_ORDERBOOK_TR_IDS = {"HDFSASP0"}

# Records are split positionally, so the field count per TR must be exact.
# Verified against the official KIS workbook (research_notes/*.xlsx):
#   trade     H0STCNT0/H0UNCNT0/H0NXCNT0 = 46 fields, identical order
#             (index 21 is named CCLD_DVSN on KRX, CNTG_CLS_CODE on 통합/NXT)
#             H0STOUP0 (시간외) = 43 fields
#   orderbook H0STASP0 = 59; H0UNASP0/H0NXASP0 = 65 (same first 59, then
#             KMID_*/NMID_* mid-price fields for KRX and NXT)
KIS_TRADE_FIELDS_PER_RECORD = 46
KIS_ORDERBOOK_FIELDS_PER_RECORD = 59
KIS_OVERSEAS_TRADE_FIELDS_PER_RECORD = 26
# Highest positional index the official trade layout needs ([21] 체결구분) plus
# one. Gating on this instead of the full record length lets the shorter 시간외
# record (43 fields) parse with the same code, while still separating it from
# the compact 6-field fixture format used by local adapters.
KIS_TRADE_MIN_OFFICIAL_FIELDS = 22
TRADE_FIELDS_BY_TR_ID = {
    "H0STCNT0": 46,
    "H0STCNI0": 46,
    "H0UNCNT0": 46,
    "H0NXCNT0": 46,
    "H0STOUP0": 43,
}
ORDERBOOK_FIELDS_BY_TR_ID = {
    "H0STASP0": 59,
    "H0UNASP0": 65,
    "H0NXASP0": 65,
}


def _domestic_subscription_tr_ids() -> tuple[str, str]:
    """Trade/orderbook TR pair for the domestic collector.

    Defaults to the 통합 feed so one subscription covers KRX plus NXT, which is
    what extends coverage from 09:00-15:30 to 08:00-20:00. Set
    ``KIS_REALTIME_FEED=krx`` to fall back to the KRX-only TRs.
    """
    feed = os.getenv("KIS_REALTIME_FEED", "unified").strip().lower()
    if feed in {"krx", "kospi", "kosdaq", "legacy"}:
        return ("H0STCNT0", "H0STASP0")
    if feed in {"nxt", "next", "nextrade"}:
        return ("H0NXCNT0", "H0NXASP0")
    return ("H0UNCNT0", "H0UNASP0")


# Trades come first so a constrained KIS subscription budget still produces
# candles and activity signals. A complete symbol receives trade + orderbook
# before the collector advances to the next symbol.
DEFAULT_SUBSCRIPTION_TR_IDS = _domestic_subscription_tr_ids()
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
    event_sink: Callable[[RealtimeTradeTick | RealtimeOrderbookSnapshot], Awaitable[bool]] | None = None
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
            if self.event_sink is not None:
                for event in (*ticks, *orderbooks):
                    await self.event_sink(event)
                counts["ticks"] += len(ticks)
                counts["orderbooks"] += len(orderbooks)
                counts["events_published"] = counts.get("events_published", 0) + len(ticks) + len(orderbooks)
            else:
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
    event_sink: Callable[
        [RealtimeTradeTick | RealtimeOrderbookSnapshot], Awaitable[bool]
    ]
    | None = None,
    subscription_tr_ids: tuple[str, ...] = DEFAULT_SUBSCRIPTION_TR_IDS,
    subscription_key_factory: Callable[[str], str] | None = None,
    symbols_provider: Callable[[], Iterable[str]] | None = None,
) -> dict[str, Any]:
    """Stream KIS realtime data over ONE persistent websocket session.

    When ``symbols_provider`` is supplied the session is long-lived: a
    resubscribe request re-reads the desired symbols and applies the difference
    in place with ``tr_type`` 1/2, instead of dropping the connection and
    reconnecting. That matters because every reconnect used to mint a new
    approval key, and KIS bills realtime registrations per session — the old
    behaviour drained the account's budget until a single symbol fit.
    """
    websockets = _load_websockets()
    client = client or build_kis_client(enabled=True)
    approval_key = issue_websocket_approval_key(client)
    target_url = url or _kis_realtime_websocket_url()
    # Mutable so an in-place resubscribe immediately changes which inbound
    # ticks are accepted; the message handlers only do membership tests.
    active_symbols: set[str] = {
        symbol for symbol in (normalize_symbol(item) for item in symbols) if symbol
    }
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
        "subscription_requests": 0,
        "subscriptions_accepted": 0,
        "subscriptions_rejected": 0,
        "accepted_subscription_pairs": [],
        "rejected_subscription_pairs": [],
        "mode": "live",
        "url": target_url,
        "subscription_tr_ids": list(subscription_tr_ids),
    }
    feature_builder = LiveFeatureFrameBuilder(store) if event_sink is None else None
    # KIS realtime sends application-level PINGPONG frames and doesn't reliably
    # respond to RFC websocket pings from the client library. Keep the standard
    # ping disabled by default; operators can opt in with env vars if needed.
    ping_interval = _websocket_ping_setting("KIS_REALTIME_WS_PING_INTERVAL_SECONDS", None)
    ping_timeout = _websocket_ping_setting("KIS_REALTIME_WS_PING_TIMEOUT_SECONDS", None)
    subscribe_delay = _websocket_subscription_delay_seconds()
    post_subscribe_drain = _websocket_post_subscribe_drain_seconds()
    subscription_limit_reached = False
    # Registrations this session currently holds. KIS bills them against the
    # account, so they must be released with tr_type=2 before the socket goes
    # away — otherwise the next session starts with a smaller budget until only
    # one symbol fits.
    held_registrations: set[tuple[str, str]] = set()

    async def _release_registrations(websocket) -> None:
        if not held_registrations:
            return
        released = int(counts.get("subscriptions_released") or 0)
        for tr_key, tr_id in sorted(held_registrations):
            try:
                await websocket.send(
                    kis_realtime_unsubscribe_message(approval_key, tr_id, tr_key)
                )
                released += 1
            except Exception:  # noqa: BLE001 - a closing socket cannot be unsubscribed.
                break
        held_registrations.clear()
        counts["subscriptions_released"] = released

    def _registration_key(symbol: str) -> str:
        return (
            str(subscription_key_factory(symbol)).upper().strip()
            if subscription_key_factory is not None
            else symbol
        )

    def _desired_registrations(symbol_set: Iterable[str]) -> list[tuple[str, str, str]]:
        """(symbol, tr_key, tr_id) triples the session should currently hold."""
        wanted: list[tuple[str, str, str]] = []
        for symbol in symbol_set:
            tr_key = _registration_key(symbol)
            for tr_id in subscription_tr_ids:
                if (symbol, tr_id) in skipped_pairs or (tr_key, tr_id) in skipped_pairs:
                    counts["subscriptions_skipped"] = counts.get("subscriptions_skipped", 0) + 1
                    continue
                wanted.append((symbol, tr_key, tr_id))
        return wanted

    async def _apply_registrations(websocket, wanted) -> bool:
        """Bring the live session to ``wanted`` with tr_type 2 then 1. True == closed."""
        nonlocal subscription_limit_reached
        target = {(tr_key, tr_id) for _s, tr_key, tr_id in wanted}
        for tr_key, tr_id in sorted(held_registrations - target):
            try:
                await websocket.send(
                    kis_realtime_unsubscribe_message(approval_key, tr_id, tr_key)
                )
            except Exception as exc:
                if _is_websocket_connection_closed(exc):
                    counts["connection_closed"] = 1
                    counts["last_close_error"] = str(exc) or exc.__class__.__name__
                    return True
                raise
            held_registrations.discard((tr_key, tr_id))
            counts["subscriptions_released"] = int(counts.get("subscriptions_released") or 0) + 1
        for symbol, tr_key, tr_id in wanted:
            if (tr_key, tr_id) in held_registrations:
                continue
            counts["last_subscription_symbol"] = symbol
            counts["last_subscription_tr_id"] = tr_id
            counts["last_subscription_key"] = tr_key
            try:
                await websocket.send(
                    kis_realtime_subscription_message(approval_key, tr_id, tr_key)
                )
            except Exception as exc:
                if _is_websocket_connection_closed(exc):
                    counts["connection_closed"] = 1
                    counts["last_close_error"] = str(exc) or exc.__class__.__name__
                    return True
                raise
            held_registrations.add((tr_key, tr_id))
            counts["subscriptions"] += 1
            counts["subscription_requests"] += 1
            if post_subscribe_drain > 0.0:
                closed = await _drain_kis_realtime_messages(
                    websocket=websocket,
                    symbols=active_symbols,
                    store=store,
                    feature_builder=feature_builder,
                    counts=counts,
                    seconds=post_subscribe_drain,
                    max_messages=max_messages,
                    event_sink=event_sink,
                )
                if closed:
                    return True
                if counts.get("subscription_limit_reached"):
                    # The account is full. Stop asking for more on this session
                    # and keep what was accepted — reconnecting would only cost
                    # another approval key without freeing anything.
                    subscription_limit_reached = True
                    return False
            if subscribe_delay > 0.0:
                await asyncio.sleep(subscribe_delay)
        return False

    async with websockets.connect(target_url, ping_interval=ping_interval, ping_timeout=ping_timeout) as websocket:
        if await _apply_registrations(websocket, _desired_registrations(sorted(active_symbols))):
            return counts
        # A soft deadline only applies to one-shot callers. With a provider the
        # session persists and the same interval becomes a re-diff cadence, so
        # session-derived subscription keys (e.g. the US daytime RBAQ* family
        # replacing DNAS*) are picked up without dropping the connection.
        deadline = (
            time.monotonic() + max_runtime_seconds
            if max_runtime_seconds and symbols_provider is None
            else None
        )
        resync_interval = (
            max_runtime_seconds if max_runtime_seconds and symbols_provider is not None else None
        )
        next_resync = time.monotonic() + resync_interval if resync_interval else None

        async def _resync() -> bool:
            """Recompute desired registrations and apply the difference."""
            nonlocal subscription_limit_reached
            try:
                refreshed = {
                    symbol
                    for symbol in (normalize_symbol(item) for item in symbols_provider())
                    if symbol
                }
            except Exception:  # noqa: BLE001 - keep streaming on a provider failure.
                refreshed = set(active_symbols)
            if refreshed:
                active_symbols.clear()
                active_symbols.update(refreshed)
            wanted = _desired_registrations(sorted(active_symbols))
            # Diff on registration keys, not on the symbol set: an unchanged
            # symbol can still need a new key when the session flips.
            if {(k, t) for _s, k, t in wanted} == held_registrations:
                return False
            subscription_limit_reached = False
            counts["in_place_resubscribes"] = int(counts.get("in_place_resubscribes") or 0) + 1
            return await _apply_registrations(websocket, wanted)

        while stop_event is None or not stop_event.is_set():
            if resubscribe_event is not None and resubscribe_event.is_set():
                counts["resubscribe_requested"] = int(counts.get("resubscribe_requested") or 0) + 1
                if symbols_provider is None:
                    break
                resubscribe_event.clear()
                if await _resync():
                    break
                if resync_interval:
                    next_resync = time.monotonic() + resync_interval
            if next_resync is not None and time.monotonic() >= next_resync:
                next_resync = time.monotonic() + resync_interval
                if await _resync():
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
                symbols=active_symbols,
                store=store,
                feature_builder=feature_builder,
                counts=counts,
                event_sink=event_sink,
            )
            if closed:
                break
            if max_messages is not None and counts["messages"] >= max_messages:
                break
        # Normal end of run (resubscribe, deadline, stop). Release the account's
        # registrations so the next connection gets its full budget back.
        if not counts.get("connection_closed"):
            await _release_registrations(websocket)
    return counts


# KIS routes the US daytime (주간거래) session through different realtime keys:
# the prefix becomes R and the exchange code switches to the BA* family. The
# night/regular session keeps D + NAS/NYS/AMS. Subscribing with the wrong pair
# is silent — KIS accepts it and simply never sends data for that session.
US_DAYTIME_EXCHANGE_CODES = {"NAS": "BAQ", "NYS": "BAY", "AMS": "BAA"}


def is_us_daytime_quote_session(now: datetime | None = None) -> bool:
    """True during the KIS US daytime quote window (10:00-16:00 KST, weekdays).

    Window per the HDFSCNT0 documentation. This is the *quote* window; the order
    routing window in ``app.execution.kis_real`` is deliberately separate.
    """
    override = os.getenv("KIS_FORCE_US_DAYTIME_QUOTES", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    local = current.astimezone(KIS_EXCHANGE_TIMEZONE)
    if local.weekday() >= 5:
        return False
    minutes = local.hour * 60 + local.minute
    return 10 * 60 <= minutes < 16 * 60


def overseas_realtime_subscription_key(
    symbol: str,
    market_hint: str = "",
    *,
    now: datetime | None = None,
) -> str:
    """Build the official KIS overseas realtime key for the current session.

    ``DNASAAPL`` during the US night/regular session, ``RBAQAAPL`` during the US
    daytime session — the exchange code differs, not just the prefix.
    """
    from app.trading.us_realtime_bridge import _exchange_code

    ticker = normalize_symbol(symbol).upper()
    if not ticker or ticker.isdigit():
        raise ValueError(f"invalid overseas realtime symbol: {symbol}")
    exchange = _exchange_code(ticker, market_hint)
    if is_us_daytime_quote_session(now) and exchange in US_DAYTIME_EXCHANGE_CODES:
        return f"R{US_DAYTIME_EXCHANGE_CODES[exchange]}{ticker}"
    return f"D{exchange}{ticker}"


async def run_kis_overseas_realtime_websocket_collector(
    *,
    symbols: Iterable[str],
    store: RealtimeMarketDataStore | None = None,
    client: "KisDevelopersApiClient | None" = None,
    url: str | None = None,
    stop_event: Any | None = None,
    resubscribe_event: Any | None = None,
    max_messages: int | None = None,
    max_runtime_seconds: float | None = None,
    symbols_provider: Callable[[], Iterable[str]] | None = None,
) -> dict[str, Any]:
    """Collect official KIS US tick trades and best bid/ask WebSocket events.

    KIS HDFSCNT0 provides trade events (0-minute delay for US, so effectively
    realtime) and also serves the US daytime session. HDFSASP0 officially
    provides one best-bid/best-ask level for US equities; deeper levels must not
    be invented.

    The subscription key is session-aware, so a resubscribe across the daytime
    boundary swaps ``DNASAAPL`` for ``RBAQAAPL`` without reconnecting.
    """
    return await run_kis_realtime_websocket_collector(
        symbols=symbols,
        store=store,
        client=client,
        url=url,
        stop_event=stop_event,
        resubscribe_event=resubscribe_event,
        max_messages=max_messages,
        max_runtime_seconds=max_runtime_seconds,
        subscription_tr_ids=("HDFSCNT0", "HDFSASP0"),
        subscription_key_factory=overseas_realtime_subscription_key,
        symbols_provider=symbols_provider,
    )


async def _drain_kis_realtime_messages(
    *,
    websocket: Any,
    symbols: Container[str],
    store: RealtimeMarketDataStore,
    feature_builder: LiveFeatureFrameBuilder,
    counts: dict[str, Any],
    seconds: float,
    max_messages: int | None,
    event_sink: Callable[
        [RealtimeTradeTick | RealtimeOrderbookSnapshot], Awaitable[bool]
    ]
    | None = None,
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
            event_sink=event_sink,
        )
        if closed:
            return True
    return False


async def _process_kis_realtime_raw(
    *,
    raw: str | bytes,
    websocket: Any,
    symbols: Container[str],
    store: RealtimeMarketDataStore,
    feature_builder: LiveFeatureFrameBuilder | None,
    counts: dict[str, Any],
    event_sink: Callable[
        [RealtimeTradeTick | RealtimeOrderbookSnapshot], Awaitable[bool]
    ]
    | None = None,
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
            counts["subscriptions_rejected"] = counts.get("subscriptions_rejected", 0) + 1
            _append_subscription_control_pair(counts, "rejected_subscription_pairs", control)
            code = str(control.get("msg_cd") or control.get("rt_cd") or "UNKNOWN")
            by_code = counts.setdefault("subscription_errors_by_code", {})
            by_code[code] = int(by_code.get(code, 0) or 0) + 1
            if code.upper() == "OPSP0008" or "MAX SUBSCRIBE" in str(control.get("msg1") or "").upper():
                counts["subscription_limit_reached"] = 1
        elif control.get("tr_id") and control.get("tr_key"):
            counts["subscriptions_accepted"] = counts.get("subscriptions_accepted", 0) + 1
            _append_subscription_control_pair(counts, "accepted_subscription_pairs", control)
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
    if event_sink is not None:
        for event in (*ticks, *orderbooks):
            await event_sink(event)
        counts["ticks"] += len(ticks)
        counts["orderbooks"] += len(orderbooks)
        counts["events_published"] = counts.get("events_published", 0) + len(ticks) + len(orderbooks)
        counts["messages"] += 1
        return False
    counts["ticks"] += store.save_ticks(ticks)
    counts["orderbooks"] += store.save_orderbooks(orderbooks)
    for symbol in {tick.symbol for tick in ticks} | {book.symbol for book in orderbooks}:
        store.build_latest_minute_bar(symbol)
        try:
            if feature_builder is not None:
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


def kis_realtime_subscription_message(
    approval_key: str,
    tr_id: str,
    symbol: str,
    *,
    tr_type: str = "1",
) -> str:
    """Build a KIS realtime control frame.

    ``tr_type`` is ``"1"`` to subscribe and ``"2"`` to unsubscribe. Releasing a
    subscription matters: KIS counts registrations against the account, so a
    connection dropped without unsubscribing leaves its slots occupied and the
    next session gets fewer — eventually only enough for a single symbol.
    """
    return json.dumps(
        {
            "header": {
                "approval_key": approval_key,
                "custtype": os.getenv("KIS_CUSTTYPE", "P"),
                "tr_type": str(tr_type),
                "content-type": "utf-8",
            },
            "body": {"input": {"tr_id": tr_id, "tr_key": normalize_symbol(symbol)}},
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )


def kis_realtime_unsubscribe_message(approval_key: str, tr_id: str, symbol: str) -> str:
    return kis_realtime_subscription_message(approval_key, tr_id, symbol, tr_type="2")


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


def _append_subscription_control_pair(
    counts: dict[str, Any],
    key: str,
    control: dict[str, Any],
) -> None:
    pair = {
        "symbol": str(control.get("tr_key") or "").upper().strip(),
        "tr_id": str(control.get("tr_id") or "").upper().strip(),
        "msg_cd": str(control.get("msg_cd") or "").strip(),
        "message": str(control.get("msg1") or "").strip(),
    }
    if not pair["symbol"] or not pair["tr_id"]:
        return
    rows = counts.setdefault(key, [])
    if pair not in rows and len(rows) < 200:
        rows.append(pair)


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
    if len(parts) >= 4 and parts[1] in OVERSEAS_TRADE_TR_IDS:
        payloads = _split_kis_payload_records(
            parts[-1],
            _kis_payload_count(parts[2]),
            KIS_OVERSEAS_TRADE_FIELDS_PER_RECORD,
        )
        ticks = tuple(
            _parse_overseas_trade_payload(
                payload,
                raw=f"{parts[0]}|{parts[1]}|001|{payload}",
                received_at=received_at,
            )
            for payload in payloads
        )
        return ParsedKisRealtimeMessage(ticks=ticks, event_type="overseas_trade")
    if len(parts) >= 4 and parts[1] in OVERSEAS_ORDERBOOK_TR_IDS:
        orderbook = _parse_overseas_orderbook_payload(
            parts[-1],
            raw=f"{parts[0]}|{parts[1]}|{parts[2]}|{parts[-1]}",
            received_at=received_at,
        )
        return ParsedKisRealtimeMessage(
            orderbooks=(orderbook,),
            event_type="overseas_orderbook",
        )
    if len(parts) >= 4 and parts[1] in TRADE_TR_IDS:
        payloads = _split_kis_payload_records(
            parts[-1],
            _kis_payload_count(parts[2]),
            TRADE_FIELDS_BY_TR_ID.get(parts[1], KIS_TRADE_FIELDS_PER_RECORD),
        )
        ticks = tuple(
            _parse_trade_payload(
                payload,
                raw=f"{parts[0]}|{parts[1]}|001|{payload}",
                received_at=received_at,
            )
            for payload in payloads
        )
        return ParsedKisRealtimeMessage(ticks=ticks, event_type="trade")
    if len(parts) >= 4 and parts[1] in ORDERBOOK_TR_IDS:
        payloads = _split_kis_payload_records(
            parts[-1],
            _kis_payload_count(parts[2]),
            ORDERBOOK_FIELDS_BY_TR_ID.get(parts[1], KIS_ORDERBOOK_FIELDS_PER_RECORD),
        )
        orderbooks = tuple(
            _parse_orderbook_payload(
                payload,
                raw=f"{parts[0]}|{parts[1]}|001|{payload}",
                received_at=received_at,
            )
            for payload in payloads
        )
        return ParsedKisRealtimeMessage(orderbooks=orderbooks, event_type="orderbook")
    if raw.startswith("{"):
        return ParsedKisRealtimeMessage(event_type="json_control")
    raise ValueError("unsupported KIS realtime message format")


def _normalize_overseas_realtime_symbol(value: str, fallback: str = "") -> str:
    text = str(value or "").upper().strip()
    if len(text) >= 5 and text[0] in {"D", "R"} and text[1:4] in {
        "NAS",
        "NYS",
        "AMS",
        "BAQ",
        "BAY",
        "BAA",
    }:
        text = text[4:]
    return text or str(fallback or "").upper().strip()


def _overseas_exchange_timestamp(
    date_value: str,
    time_value: str,
    received_at: datetime,
) -> datetime:
    date_text = str(date_value or "").strip()
    time_text = str(time_value or "").strip()
    if len(date_text) == 8 and date_text.isdigit() and len(time_text) >= 6 and time_text[:6].isdigit():
        try:
            local = datetime.strptime(
                f"{date_text}{time_text[:6]}",
                "%Y%m%d%H%M%S",
            ).replace(tzinfo=ZoneInfo("America/New_York"))
            return local.astimezone(timezone.utc)
        except ValueError:
            pass
    return received_at


def _parse_overseas_trade_payload(
    payload: str,
    *,
    raw: str,
    received_at: datetime,
) -> RealtimeTradeTick:
    fields = payload.split("^")
    if len(fields) < KIS_OVERSEAS_TRADE_FIELDS_PER_RECORD:
        raise ValueError(
            f"KIS overseas trade payload has {len(fields)} fields; expected 26"
        )
    symbol = _normalize_overseas_realtime_symbol(fields[1], fields[0])
    exchange_timestamp = _overseas_exchange_timestamp(fields[4], fields[5], received_at)
    price = _float(fields[11])
    bid = _float(fields[15])
    ask = _float(fields[16])
    volume = max(0, int(_float(fields[19])))
    direction = "BUY" if ask > 0 and price >= ask else "SELL" if bid > 0 and price <= bid else None
    sequence = (
        f"us-kis-ws:{symbol}:{fields[4]}:{fields[5]}:{fields[20]}:"
        f"{price}:{volume}:{checksum(payload)[:10]}"
    )
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


def _parse_overseas_orderbook_payload(
    payload: str,
    *,
    raw: str,
    received_at: datetime,
) -> RealtimeOrderbookSnapshot:
    fields = payload.split("^")
    # The current official schema has 16 fields. Older official samples include
    # RSYM before SYMB and therefore contain 17; support both wire layouts.
    if len(fields) >= 17:
        symbol = _normalize_overseas_realtime_symbol(fields[1], fields[0])
        exchange_timestamp = _overseas_exchange_timestamp(fields[3], fields[4], received_at)
        bid, ask, bid_size, ask_size = map(_float, fields[11:15])
        sequence_time = fields[4]
    elif len(fields) >= 16:
        symbol = _normalize_overseas_realtime_symbol(fields[0])
        exchange_timestamp = _overseas_exchange_timestamp(fields[2], fields[3], received_at)
        bid, ask, bid_size, ask_size = map(_float, fields[10:14])
        sequence_time = fields[3]
    else:
        raise ValueError(
            f"KIS overseas orderbook payload has {len(fields)} fields; expected 16 or 17"
        )
    if not symbol or bid <= 0 or ask <= 0 or ask < bid:
        raise ValueError(
            f"KIS overseas orderbook invalid symbol/bid/ask: {symbol} {bid} {ask}"
        )
    level = OrderbookLevel(
        bid_price=bid,
        bid_size=max(0, int(bid_size)),
        ask_price=ask,
        ask_size=max(0, int(ask_size)),
    )
    sequence = (
        f"us-kis-ws:{symbol}:{sequence_time}:{bid}:{ask}:"
        f"{level.bid_size}:{level.ask_size}:{checksum(payload)[:10]}"
    )
    return RealtimeOrderbookSnapshot(
        symbol=symbol,
        exchange_timestamp=exchange_timestamp,
        received_at=received_at,
        source=KIS_REALTIME_SOURCE,
        levels=(level,),
        sequence_key=sequence,
        raw_checksum=checksum(raw),
        latency_ms=max(0.0, (received_at - exchange_timestamp).total_seconds() * 1000),
    )


def _parse_trade_payload(payload: str, *, raw: str, received_at: datetime) -> RealtimeTradeTick:
    fields = payload.split("^")
    if len(fields) < 4:
        raise ValueError("KIS trade payload has too few fields")
    symbol = normalize_symbol(fields[0])
    exchange_timestamp = _timestamp_from_hhmmss(fields[1], received_at)
    price = _float(fields[2])
    if len(fields) >= KIS_TRADE_MIN_OFFICIAL_FIELDS:
        # Official layout, shared by H0STCNT0 (KRX, 46), H0UNCNT0/H0NXCNT0
        # (통합/NXT, 46) and H0STOUP0 (시간외, 43) — verified field-by-field
        # against the KIS workbook. Only the name at [21] differs
        # (CCLD_DVSN vs CNTG_CLS_CODE); the position and meaning are the same.
        # [3] PRDY_VRSS_SIGN, [5] PRDY_CTRT, [12] CNTG_VOL,
        # [13] ACML_VOL, [14] ACML_TR_PBMN, [21] 체결구분.
        volume = int(_float(fields[12]))
        direction = _kis_trade_direction(fields[21])
        sequence = (
            f"{symbol}:{fields[1]}:{fields[13]}:{fields[14]}:"
            f"{price}:{volume}:{checksum(payload)[:12]}"
        )
    else:
        # Retain the compact test/fixture format used by local adapters.
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


def _kis_payload_count(value: str) -> int:
    try:
        return max(1, int(str(value).strip()))
    except (TypeError, ValueError):
        return 1


def _split_kis_payload_records(
    payload: str,
    count: int,
    fields_per_record: int,
) -> tuple[str, ...]:
    fields = payload.split("^")
    expected = count * fields_per_record
    if count <= 1 or len(fields) < expected:
        return (payload,)
    return tuple(
        "^".join(fields[index : index + fields_per_record])
        for index in range(0, expected, fields_per_record)
    )


def _kis_trade_direction(value: str) -> str | None:
    code = str(value or "").strip().upper()
    if code == "1":
        return "BUY"
    if code == "5":
        return "SELL"
    return code or None


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

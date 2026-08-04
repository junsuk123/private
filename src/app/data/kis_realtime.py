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
                # sink 경로에서도 분 bar 를 만든다. 이 분기가 bar 를 만들지 않아서
                # macro reasoner 가 연속 bar 를 얻지 못했다 (자세한 배경은
                # ``_build_minute_bars_throttled`` 참조).
                _build_minute_bars_throttled(
                    self.store,
                    {tick.symbol for tick in ticks}
                    | {book.symbol for book in orderbooks},
                    counts,
                )
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
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    session_active_provider: Callable[[], bool] | None = None,
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

    def _notify_progress() -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(dict(counts))
        except Exception:
            pass

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
                    _notify_progress()
                    return True
                if counts.get("subscription_limit_reached"):
                    # The account is full. Stop asking for more on this session
                    # and keep what was accepted — reconnecting would only cost
                    # another approval key without freeing anything.
                    subscription_limit_reached = True
                    _notify_progress()
                    return False
            if subscribe_delay > 0.0:
                await asyncio.sleep(subscribe_delay)
        _notify_progress()
        return False

    async with websockets.connect(target_url, ping_interval=ping_interval, ping_timeout=ping_timeout) as websocket:
        if await _apply_registrations(websocket, _desired_registrations(sorted(active_symbols))):
            _notify_progress()
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
            if session_active_provider is not None:
                try:
                    session_active = bool(session_active_provider())
                except Exception:
                    session_active = True
                if not session_active:
                    counts["session_relinquished"] = 1
                    break
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
            _notify_progress()
            if max_messages is not None and counts["messages"] >= max_messages:
                break
        # Normal end of run (resubscribe, deadline, stop). Release the account's
        # registrations so the next connection gets its full budget back.
        if not counts.get("connection_closed"):
            await _release_registrations(websocket)
        _notify_progress()
    return counts


# KIS routes the US daytime (주간거래) session through different realtime keys:
# the prefix becomes R and the exchange code switches to the BA* family. The
# night/regular session keeps D + NAS/NYS/AMS. Subscribing with the wrong pair
# is silent — KIS accepts it and simply never sends data for that session.
US_DAYTIME_EXCHANGE_CODES = {"NAS": "BAQ", "NYS": "BAY", "AMS": "BAA"}


def is_us_daytime_quote_session(now: datetime | None = None) -> bool:
    """True during the KIS US daytime QUOTE window (10:00-16:00 KST, weekdays).

    공식 문서는 두 창을 다르게 명시한다:

    * 주문창  10:00~18:00 KST ("해외주식 미국주간주문")
    * 시세창  10:00~16:00 KST ("해당 API로 미국주간거래(10:00~16:00) 시세 조회도 가능")

    이 함수는 **시세창** 이다. 주문 route 창은 capability service 의 세션 창이며,
    16:00-18:00 KST 구간은 "주문은 되지만 공식 시세 근거가 없는" 상태로
    ``DAYTIME_QUOTE_WINDOW_ENDED`` 사유코드가 붙는다.

    판정은 ``config/market_sessions.yaml`` 의 ``US_DAYTIME.data_window_end`` 를 읽는
    canonical service 에 위임한다.
    """
    override = os.getenv("KIS_FORCE_US_DAYTIME_QUOTES", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    from app.data.market_capabilities import MarketGroup, SessionId, default_service

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    service = default_service()
    window = service.window(SessionId.US_DAYTIME)
    if window is None or not window.data_contains(current):
        return False
    # 주간거래는 한국시간 창이므로 요일 판정도 한국시간으로 해야 한다. 토요일 12:00 KST
    # 는 뉴욕 기준 금요일 밤이지만 주간거래는 열리지 않는다.
    if current.astimezone(window.zone).weekday() >= 5:
        return False
    return service.is_trading_day(MarketGroup.US, current)


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
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    session_active_provider: Callable[[], bool] | None = None,
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
        progress_callback=progress_callback,
        session_active_provider=session_active_provider,
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
        control_message = str(control.get("msg1") or "").upper()
        control_code = str(control.get("msg_cd") or "").upper()
        if "UNSUBSCRIBE" in control_message or control_code in {"OPSP0001", "OPSP0003"}:
            key = (
                "unsubscribe_errors"
                if _kis_realtime_control_is_error(control)
                else "unsubscribe_confirmations"
            )
            counts[key] = int(counts.get(key) or 0) + 1
            counts["messages"] += 1
            return False
        if _kis_realtime_control_is_appkey_in_use(control):
            counts["appkey_already_in_use"] = 1
        if _kis_realtime_control_is_error(control):
            counts["control_errors"] = counts.get("control_errors", 0) + 1
            _append_subscription_control_pair(counts, "rejected_subscription_pairs", control)
            counts["subscriptions_rejected"] = len(
                counts.get("rejected_subscription_pairs") or ()
            )
            code = str(control.get("msg_cd") or control.get("rt_cd") or "UNKNOWN")
            by_code = counts.setdefault("subscription_errors_by_code", {})
            by_code[code] = int(by_code.get(code, 0) or 0) + 1
            if code.upper() == "OPSP0008" or "MAX SUBSCRIBE" in str(control.get("msg1") or "").upper():
                counts["subscription_limit_reached"] = 1
        elif control.get("tr_id") and control.get("tr_key"):
            _append_subscription_control_pair(counts, "accepted_subscription_pairs", control)
            # KIS also acknowledges tr_type=2 unsubscribe frames. The response
            # does not reliably echo tr_type, so count unique successful pairs
            # instead of incrementing every control acknowledgement.
            counts["subscriptions_accepted"] = len(
                counts.get("accepted_subscription_pairs") or ()
            )
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
        # 분 bar 는 이 분기에서도 만들어야 한다.
        #
        # 이전에는 event_sink 경로가 여기서 바로 반환했고, 분 bar 를 만드는 곳은
        # (a) sink 를 쓰지 않는 경로, (b) 미국 REST 브릿지, (c) 대시보드 차트
        # 엔드포인트 뿐이었다. 라이브 서버는 sink 경로를 쓰므로 **웹소켓 체결로부터
        # 분 bar 가 전혀 생성되지 않았고**, 누군가 UI 차트를 열 때만 산발적으로 생겼다.
        #
        # 그 결과 macro reasoner 가 rolling return 을 계산할 만큼의 연속 bar 를 얻지
        # 못해 MACRO_INSUFFICIENT_DATA → NO_TRADE_MARKET 이 되고, NO_TRADE_MARKET 은
        # new_buy 를 전면 차단한다. 즉 "전략 채택 불가"의 최상위 원인이었다.
        #
        # 분당 심볼당 한 번으로 제한한다 (메시지마다 DB 왕복을 하지 않는다).
        _build_minute_bars_throttled(
            store,
            {tick.symbol for tick in ticks} | {book.symbol for book in orderbooks},
            counts,
        )
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


# TR ID → 피드 identity. 값의 근거는 공식 문서이며
# ``docs/kis_market_session_capability_matrix.md`` §1·§4 에 정리되어 있다.
# (market_group, venue, feed_scope, exchange, currency, is_consolidated)
_FEED_IDENTITY_BY_TR: dict[str, tuple[str, str, str, str, str, bool]] = {
    "H0STCNT0": ("KR", "KRX", "VENUE_SPECIFIC", "KRX", "KRW", False),
    "H0STASP0": ("KR", "KRX", "VENUE_SPECIFIC", "KRX", "KRW", False),
    "H0STOUP0": ("KR", "KRX", "VENUE_SPECIFIC", "KRX", "KRW", False),
    "H0STOAA0": ("KR", "KRX", "VENUE_SPECIFIC", "KRX", "KRW", False),
    "H0STCNI0": ("KR", "KRX", "VENUE_SPECIFIC", "KRX", "KRW", False),
    "H0NXCNT0": ("KR", "NXT", "VENUE_SPECIFIC", "NXT", "KRW", False),
    "H0NXASP0": ("KR", "NXT", "VENUE_SPECIFIC", "NXT", "KRW", False),
    # 통합(KRX+NXT) 피드는 consolidated 이며 주문 venue 가 될 수 없다 (DATA_ONLY).
    "H0UNCNT0": ("KR", "KRX_NXT_UNIFIED", "UNIFIED", "KRX", "KRW", True),
    "H0UNASP0": ("KR", "KRX_NXT_UNIFIED", "UNIFIED", "KRX", "KRW", True),
    # 미국 무료 실시간. 나스닥 마켓센터 단일 시장 호가이므로 consolidated 가 아니다.
    "HDFSCNT0": ("US", "NASDAQ", "FREE_REALTIME", "NASD", "USD", False),
    "HDFSASP0": ("US", "NASDAQ", "FREE_REALTIME", "NASD", "USD", False),
}

#: 미국 subscription key 의 시장구분(3자리) → venue / OVRS_EXCG_CD.
_US_FEED_CODE_TO_VENUE = {
    "NAS": ("NASDAQ", "NASD"), "BAQ": ("NASDAQ", "NASD"),
    "NYS": ("NYSE", "NYSE"), "BAY": ("NYSE", "NYSE"),
    "AMS": ("AMEX", "AMEX"), "BAA": ("AMEX", "AMEX"),
}
#: ``R`` + ``BA*`` 는 주간거래 키다 (공식: BAY 뉴욕(주간), BAQ 나스닥(주간), BAA 아멕스(주간)).
_US_DAYTIME_FEED_CODES = frozenset({"BAQ", "BAY", "BAA"})


def feed_metadata_for_tr(
    tr_id: str,
    *,
    subscription_key: str = "",
    now: datetime | None = None,
):
    """이 TR/subscription key 로 들어온 이벤트의 :class:`FeedMetadata`.

    세션과 ``is_tradeable`` 은 canonical capability service 에서 가져온다. 알 수 없으면
    ``UNKNOWN`` 을 남기고, ``UNKNOWN`` 은 실시간 신규매수 적격을 통과하지 못한다.
    """
    from app.data.market_capabilities import (
        FeedScope,
        MarketGroup,
        SessionId,
        Venue,
        default_service,
    )
    from app.data.realtime_types import FeedMetadata

    identity = _FEED_IDENTITY_BY_TR.get(str(tr_id or "").upper().strip())
    if identity is None:
        return FeedMetadata(tr_id=str(tr_id or ""), subscription_key=str(subscription_key or ""))
    group_name, venue_name, scope_name, exchange, currency, consolidated = identity
    key = str(subscription_key or "").upper().strip()

    # 미국은 subscription key 로 거래소와 주간/야간을 구별한다 (DNASAAPL vs RBAQAAPL).
    daytime = False
    if group_name == "US" and len(key) >= 4 and key[0] in {"D", "R"}:
        code = key[1:4]
        mapped = _US_FEED_CODE_TO_VENUE.get(code)
        if mapped is not None:
            venue_name, exchange = mapped
        daytime = key[0] == "R" and code in _US_DAYTIME_FEED_CODES

    group = MarketGroup(group_name)
    venue = Venue(venue_name)
    service = default_service()
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)

    session = SessionId.UNKNOWN
    is_tradeable = False
    active = service.active_capabilities(group, current)
    if group is MarketGroup.US:
        wanted = SessionId.US_DAYTIME if daytime else None
        for capability in active:
            if wanted is not None and capability.session is not wanted:
                continue
            if wanted is None and capability.session is SessionId.US_DAYTIME:
                continue
            session = capability.session
            is_tradeable = capability.trade_available
            break
        if daytime:
            venue = Venue.US_DAYTIME_VENUE if session is SessionId.US_DAYTIME else venue
    elif venue is Venue.KRX_NXT_UNIFIED:
        # 통합 피드는 세션 라벨만 붙이고 주문 근거로는 쓰지 않는다.
        unified = service.unified_feed_capability(current)
        session = unified.session
        is_tradeable = False
    else:
        for capability in active:
            if capability.venue is venue:
                session = capability.session
                is_tradeable = capability.trade_available
                break

    return FeedMetadata(
        market_group=group,
        exchange=exchange,
        venue=venue,
        session=session,
        currency=currency,
        feed_scope=FeedScope(scope_name),
        tr_id=str(tr_id or "").upper().strip(),
        subscription_key=key,
        is_consolidated=consolidated,
        is_tradeable=is_tradeable,
        metadata_inferred=False,
    )


def _apply_feed_metadata(
    parsed: ParsedKisRealtimeMessage,
    tr_id: str,
    *,
    subscription_key: str,
    received_at: datetime,
) -> ParsedKisRealtimeMessage:
    """파싱된 이벤트에 출처 metadata 를 붙인다.

    metadata 가 없으면 저장소에서 KRX 체결과 NXT 체결을 구분할 수 없고, 통합 피드와
    venue 피드가 같은 분 bar 행을 다투게 된다. 그래서 파서 단계에서 바로 붙인다.
    """
    meta = feed_metadata_for_tr(
        tr_id, subscription_key=subscription_key, now=received_at
    )
    return ParsedKisRealtimeMessage(
        ticks=tuple(tick.with_meta(meta) for tick in parsed.ticks),
        orderbooks=tuple(book.with_meta(meta) for book in parsed.orderbooks),
        event_type=parsed.event_type,
    )


#: (symbol) -> 마지막으로 분 bar 를 만든 monotonic 시각. DB 왕복 제한용.
_LAST_MINUTE_BAR_BUILT: dict[str, float] = {}


def _minute_bar_rebuild_interval_seconds() -> float:
    raw = os.getenv("REALTIME_MINUTE_BAR_REBUILD_SEC", "2.0")
    try:
        return max(0.0, min(30.0, float(raw)))
    except (TypeError, ValueError):
        return 2.0


def _build_minute_bars_throttled(
    store: RealtimeMarketDataStore,
    symbols: set[str],
    counts: dict[str, Any],
    *,
    now: datetime | None = None,
) -> None:
    """관측된 심볼의 분 bar 를 만든다.

    ``build_latest_minute_bar`` 은 **이번 분에 관측된 스트림별로** bar 를 만들므로
    venue 별 bar 와 통합/REST bar 가 섞이지 않는다.

    **분당 1회로 제한하면 안 된다.** bar 는 "이번 분에 지금까지 도착한 체결"의 집계이므로,
    분이 시작될 때 한 번만 만들면 그 분의 첫 체결만 담긴 미완성 bar 로 굳는다 (실제로
    그렇게 만들어 놓고 2~3개짜리 시계열을 얻었다). 기존 non-sink 경로는 메시지마다
    다시 만들어 bar 가 수렴하게 했다. 여기서는 심볼당 최소 간격만 두어 DB 왕복을
    제한하면서 같은 수렴 성질을 유지한다.
    """
    current = now or datetime.now(timezone.utc)
    interval = _minute_bar_rebuild_interval_seconds()
    stamp = time.monotonic()
    for symbol in symbols:
        previous = _LAST_MINUTE_BAR_BUILT.get(symbol)
        if previous is not None and (stamp - previous) < interval:
            continue
        try:
            store.build_latest_minute_bar(symbol, now=current)
        except Exception as exc:  # noqa: BLE001 - bar 집계 실패가 수집을 멈추면 안 된다.
            counts["minute_bar_errors"] = counts.get("minute_bar_errors", 0) + 1
            counts["last_minute_bar_error"] = str(exc) or exc.__class__.__name__
            continue
        _LAST_MINUTE_BAR_BUILT[symbol] = stamp
        counts["minute_bars_built"] = counts.get("minute_bars_built", 0) + 1
    # 구독이 바뀌며 사라진 심볼의 항목이 무한히 쌓이지 않게 한다.
    if len(_LAST_MINUTE_BAR_BUILT) > 512:
        for stale in [
            key
            for key, value in _LAST_MINUTE_BAR_BUILT.items()
            if (stamp - value) > 600.0
        ]:
            _LAST_MINUTE_BAR_BUILT.pop(stale, None)


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
        return _apply_feed_metadata(
            ParsedKisRealtimeMessage(ticks=ticks, event_type="overseas_trade"),
            parts[1],
            subscription_key=_subscription_key_from_payload(payloads[0] if payloads else ""),
            received_at=received_at,
        )
    if len(parts) >= 4 and parts[1] in OVERSEAS_ORDERBOOK_TR_IDS:
        orderbook = _parse_overseas_orderbook_payload(
            parts[-1],
            raw=f"{parts[0]}|{parts[1]}|{parts[2]}|{parts[-1]}",
            received_at=received_at,
        )
        return _apply_feed_metadata(
            ParsedKisRealtimeMessage(orderbooks=(orderbook,), event_type="overseas_orderbook"),
            parts[1],
            subscription_key=_subscription_key_from_payload(parts[-1]),
            received_at=received_at,
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
        return _apply_feed_metadata(
            ParsedKisRealtimeMessage(ticks=ticks, event_type="trade"),
            parts[1],
            subscription_key=ticks[0].symbol if ticks else "",
            received_at=received_at,
        )
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
        return _apply_feed_metadata(
            ParsedKisRealtimeMessage(orderbooks=orderbooks, event_type="orderbook"),
            parts[1],
            subscription_key=orderbooks[0].symbol if orderbooks else "",
            received_at=received_at,
        )
    if raw.startswith("{"):
        return ParsedKisRealtimeMessage(event_type="json_control")
    raise ValueError("unsupported KIS realtime message format")


def _subscription_key_from_payload(payload: str) -> str:
    """해외 payload 첫 필드가 원래의 ``D``/``R`` + 시장구분 + 종목 키다.

    이 값이 주간(``RBAQ*``)/야간(``DNAS*``) 세션과 거래소를 구별하는 유일한 단서이므로
    정규화 전 원문을 metadata 로 보존한다.
    """
    head = str(payload or "").split("^", 1)[0].strip().upper()
    return head


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

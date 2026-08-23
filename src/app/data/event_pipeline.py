from __future__ import annotations

import asyncio
import math
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypeAlias

from app.data.realtime_types import (
    FeedMetadata,
    RealtimeMinuteBar,
    RealtimeOrderbookSnapshot,
    RealtimeTradeTick,
)


MarketEvent: TypeAlias = RealtimeTradeTick | RealtimeOrderbookSnapshot


def _paired_feed_key(meta: FeedMetadata) -> tuple[object, ...]:
    """Identity shared by the trade and orderbook TRs for one logical feed.

    ``FeedMetadata.stream_id`` deliberately contains ``tr_id`` because persisted
    trade and book rows are separate physical streams (H0STCNT0 vs H0STASP0).
    A minute bar, however, needs the matching book from the same venue/feed.
    Everything except the transport TR must agree before the two are paired.
    """
    return (
        meta.market_group,
        str(meta.exchange or "").upper(),
        meta.venue,
        meta.session,
        str(meta.currency or "").upper(),
        meta.feed_scope,
        str(meta.subscription_key or "").upper(),
        bool(meta.is_consolidated),
        bool(meta.is_tradeable),
        bool(meta.metadata_inferred),
    )


def _minute_bar_refresh_interval_seconds() -> float:
    raw = os.getenv("REALTIME_MINUTE_BAR_REBUILD_SEC", "2.0")
    try:
        return max(0.0, min(30.0, float(raw)))
    except (TypeError, ValueError):
        return 2.0


@dataclass(frozen=True)
class EventBusStats:
    published: int
    consumed: int
    coalesced: int
    dropped: int
    depth: int
    capacity: int


class BoundedMarketEventBus:
    """In-process bounded bus; market events coalesce by symbol and kind."""

    def __init__(self, capacity: int = 4096) -> None:
        if capacity <= 0:
            raise ValueError("event bus capacity must be positive")
        self.capacity = capacity
        self._items: deque[MarketEvent] = deque()
        self._condition = asyncio.Condition()
        self._published = 0
        self._consumed = 0
        self._coalesced = 0
        self._dropped = 0

    async def publish(self, event: MarketEvent) -> bool:
        async with self._condition:
            self._published += 1
            if len(self._items) >= self.capacity:
                key = (event.symbol, type(event))
                for index in range(len(self._items) - 1, -1, -1):
                    queued = self._items[index]
                    if (queued.symbol, type(queued)) == key:
                        self._items[index] = event
                        self._coalesced += 1
                        self._condition.notify()
                        return True
                self._items.popleft()
                self._dropped += 1
            self._items.append(event)
            self._condition.notify()
            return True

    async def get(self) -> MarketEvent:
        async with self._condition:
            while not self._items:
                await self._condition.wait()
            self._consumed += 1
            return self._items.popleft()

    async def wait_for_depth(self, timeout: float) -> bool:
        """Sleep until an event is queued, or ``timeout`` elapses.

        Exists so an idle consumer can WAIT instead of polling ``stats().depth``
        in a ``sleep(0)`` loop. That loop shared its event loop with the KIS
        websocket reader, and burned a full core spinning while the reader was
        starved of it — the socket stayed ESTABLISHED, the kernel receive queue
        grew past 3MB, and US market data stopped arriving with no error anywhere.

        A missed ``notify`` cannot lose data: producers append before notifying
        and the caller re-checks depth, so the worst case is one extra timeout.
        """
        async with self._condition:
            if self._items:
                return True
            try:
                await asyncio.wait_for(self._condition.wait(), timeout)
            except (TimeoutError, asyncio.TimeoutError):
                return False
            return bool(self._items)

    def stats(self) -> EventBusStats:
        return EventBusStats(
            published=self._published,
            consumed=self._consumed,
            coalesced=self._coalesced,
            dropped=self._dropped,
            depth=len(self._items),
            capacity=self.capacity,
        )


@dataclass(frozen=True)
class IncrementalFeatureState:
    symbol: str
    as_of: datetime
    last_price: float
    mid: float | None
    spread_bps: float | None
    microprice: float | None
    orderbook_imbalance: float | None
    order_flow_imbalance: float
    aggressor_trade_imbalance: float
    vwap: float
    realized_volatility: float
    fresh: bool
    sequence_uncertain: bool


@dataclass
class SymbolMarketState:
    symbol: str
    latest_tick: RealtimeTradeTick | None = None
    latest_book: RealtimeOrderbookSnapshot | None = None
    duplicate_count: int = 0
    out_of_order_count: int = 0
    gap_count: int = 0
    sequence_uncertain: bool = False
    _seen_ids: deque[str] = field(default_factory=lambda: deque(maxlen=2048))
    _seen_set: set[str] = field(default_factory=set)
    _last_numeric_sequence: dict[str, int] = field(default_factory=dict)
    _vwap_numerator: float = 0.0
    _vwap_denominator: float = 0.0
    _returns: deque[float] = field(default_factory=lambda: deque(maxlen=300))
    _buy_volume: float = 0.0
    _sell_volume: float = 0.0
    _ofi: float = 0.0

    def apply(self, event: MarketEvent) -> bool:
        record_id = event.record_id
        if record_id in self._seen_set:
            self.duplicate_count += 1
            return False
        if len(self._seen_ids) == self._seen_ids.maxlen:
            self._seen_set.discard(self._seen_ids[0])
        self._seen_ids.append(record_id)
        self._seen_set.add(record_id)

        kind = "trade" if isinstance(event, RealtimeTradeTick) else "book"
        previous = self.latest_tick if kind == "trade" else self.latest_book
        if previous is not None and event.exchange_timestamp < previous.exchange_timestamp:
            self.out_of_order_count += 1
            return False
        self._check_sequence(kind, event.sequence_key)
        if isinstance(event, RealtimeTradeTick):
            self._apply_tick(event)
        else:
            self._apply_book(event)
        return True

    def mark_reconnected(self) -> None:
        self.sequence_uncertain = True
        self._last_numeric_sequence.clear()
        self._ofi = 0.0

    def features(self, now: datetime, staleness_ms: int) -> IncrementalFeatureState | None:
        if self.latest_tick is None:
            return None
        book = self.latest_book
        bid = book.best_bid if book else 0.0
        ask = book.best_ask if book else 0.0
        mid = (bid + ask) / 2 if bid > 0 and ask >= bid else None
        spread = book.spread_bps if book and mid else None
        microprice = None
        imbalance = None
        if book and book.levels:
            level = book.levels[0]
            total = level.bid_size + level.ask_size
            if total > 0:
                microprice = (
                    level.ask_price * level.bid_size + level.bid_price * level.ask_size
                ) / total
            imbalance = book.imbalance
        total_trade = self._buy_volume + self._sell_volume
        trade_imbalance = (
            (self._buy_volume - self._sell_volume) / total_trade if total_trade > 0 else 0.0
        )
        vwap = (
            self._vwap_numerator / self._vwap_denominator
            if self._vwap_denominator > 0
            else self.latest_tick.price
        )
        rv = math.sqrt(sum(value * value for value in self._returns))
        latest_receive = max(
            self.latest_tick.received_at,
            book.received_at if book else self.latest_tick.received_at,
        )
        age_ms = max(0.0, (now - latest_receive).total_seconds() * 1000)
        return IncrementalFeatureState(
            symbol=self.symbol,
            as_of=max(
                self.latest_tick.exchange_timestamp,
                book.exchange_timestamp if book else self.latest_tick.exchange_timestamp,
            ),
            last_price=self.latest_tick.price,
            mid=mid,
            spread_bps=spread,
            microprice=microprice,
            orderbook_imbalance=imbalance,
            order_flow_imbalance=self._ofi,
            aggressor_trade_imbalance=trade_imbalance,
            vwap=vwap,
            realized_volatility=rv,
            fresh=age_ms <= staleness_ms,
            sequence_uncertain=self.sequence_uncertain,
        )

    def _apply_tick(self, tick: RealtimeTradeTick) -> None:
        if self.latest_tick and self.latest_tick.price > 0:
            self._returns.append(math.log(tick.price / self.latest_tick.price))
        quantity = max(0.0, float(tick.volume))
        self._vwap_numerator += tick.price * quantity
        self._vwap_denominator += quantity
        direction = str(tick.trade_direction or "").upper()
        if direction in {"BUY", "B"}:
            self._buy_volume += quantity
        elif direction in {"SELL", "S"}:
            self._sell_volume += quantity
        self.latest_tick = tick

    def _apply_book(self, book: RealtimeOrderbookSnapshot) -> None:
        previous = self.latest_book
        if previous and previous.levels and book.levels:
            old = previous.levels[0]
            new = book.levels[0]
            self._ofi += (
                (new.bid_size if new.bid_price >= old.bid_price else 0)
                - (old.bid_size if new.bid_price <= old.bid_price else 0)
                - (new.ask_size if new.ask_price <= old.ask_price else 0)
                + (old.ask_size if new.ask_price >= old.ask_price else 0)
            )
        self.latest_book = book
        # A full snapshot after reconnect establishes a new trustworthy baseline.
        self.sequence_uncertain = False

    def _check_sequence(self, kind: str, sequence_key: str | None) -> None:
        if not sequence_key:
            return
        match = re.search(r"(\d+)$", sequence_key)
        if not match:
            return
        current = int(match.group(1))
        previous = self._last_numeric_sequence.get(kind)
        if previous is not None and current > previous + 1:
            self.gap_count += 1
            self.sequence_uncertain = True
        if previous is None or current > previous:
            self._last_numeric_sequence[kind] = current


class MarketState:
    def __init__(self) -> None:
        self._symbols: dict[str, SymbolMarketState] = {}

    def apply(self, event: MarketEvent) -> bool:
        state = self._symbols.setdefault(event.symbol, SymbolMarketState(event.symbol))
        return state.apply(event)

    def symbol(self, symbol: str) -> SymbolMarketState | None:
        return self._symbols.get(symbol)

    def mark_reconnected(self) -> None:
        for state in self._symbols.values():
            state.mark_reconnected()


@dataclass
class IncrementalMinuteBarBuilder:
    """한 (symbol, stream) 의 분 bar 를 증분 집계한다.

    **스트림별로 하나씩** 두어야 한다. 분 bar 의 identity 는
    ``(stream_id, symbol, minute_start)`` 이므로, 여러 피드(웹소켓 / REST 스냅샷 /
    KRX+NXT 통합)의 체결을 한 builder 에 섞으면 서로 다른 시장의 체결이 한 bar 로
    합산되고, 저장 시에는 metadata 없는 ``stream_id=''`` 행 하나로 뭉개진다.
    """

    symbol: str
    _minute_start: datetime | None = None
    #: 이 builder 가 집계 중인 체결의 출처. bar 에 그대로 실어 보낸다.
    _meta: FeedMetadata = field(default_factory=FeedMetadata)
    _open: float = 0.0
    _high: float = 0.0
    _low: float = 0.0
    _close: float = 0.0
    _volume: int = 0
    _notional: float = 0.0
    _count: int = 0
    _price_sum: float = 0.0
    _price_square_sum: float = 0.0
    _source_ids: list[str] = field(default_factory=list)
    _latest_book: RealtimeOrderbookSnapshot | None = None
    _latest_received_at: datetime | None = None

    def update(self, tick: RealtimeTradeTick) -> RealtimeMinuteBar | None:
        minute = tick.exchange_timestamp.replace(second=0, microsecond=0)
        completed = self.current_bar() if self._minute_start and minute > self._minute_start else None
        if self._minute_start is None or minute > self._minute_start:
            self._reset(minute, tick)
        elif minute == self._minute_start:
            self._high = max(self._high, tick.price)
            self._low = min(self._low, tick.price)
            self._close = tick.price
            self._volume += max(0, tick.volume)
            self._notional += tick.price * max(0, tick.volume)
            self._count += 1
            self._price_sum += tick.price
            self._price_square_sum += tick.price * tick.price
            self._source_ids.append(tick.record_id)
            self._latest_received_at = max(
                self._latest_received_at or tick.received_at,
                tick.received_at,
            )
        return completed

    def update_orderbook(self, book: RealtimeOrderbookSnapshot) -> bool:
        """Attach a same-minute book from the paired feed without crossing venues."""
        if self._minute_start is None:
            return False
        if _paired_feed_key(book.meta) != _paired_feed_key(self._meta):
            return False
        minute = book.exchange_timestamp.replace(second=0, microsecond=0)
        if minute != self._minute_start:
            return False
        if (
            self._latest_book is not None
            and book.exchange_timestamp < self._latest_book.exchange_timestamp
        ):
            return False
        self._latest_book = book
        self._latest_received_at = max(
            self._latest_received_at or book.received_at,
            book.received_at,
        )
        return True

    def current_bar(self) -> RealtimeMinuteBar | None:
        if self._minute_start is None:
            return None
        variance = max(
            0.0,
            self._price_square_sum / max(1, self._count)
            - (self._price_sum / max(1, self._count)) ** 2,
        )
        now = datetime.now(timezone.utc)
        last_update_age_ms = (
            max(0.0, (now - self._latest_received_at).total_seconds() * 1000)
            if self._latest_received_at is not None
            else 0.0
        )
        return RealtimeMinuteBar(
            symbol=self.symbol,
            minute_start=self._minute_start,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
            vwap=self._notional / self._volume if self._volume else self._close,
            trade_count=self._count,
            spread_bps=self._latest_book.spread_bps if self._latest_book else 0.0,
            orderbook_imbalance=self._latest_book.imbalance if self._latest_book else 0.0,
            # Keep the established minute-bar definition used by the historical
            # rebuild path: traded volume, capped to [0, 1].
            liquidity_score=min(1.0, self._volume / 100_000.0),
            volatility=variance**0.5,
            last_update_age_ms=last_update_age_ms,
            source_record_ids=tuple(self._source_ids),
            meta=self._meta,
        )

    def _reset(self, minute: datetime, tick: RealtimeTradeTick) -> None:
        self._minute_start = minute
        self._meta = tick.meta
        self._open = self._high = self._low = self._close = tick.price
        self._volume = max(0, tick.volume)
        self._notional = tick.price * self._volume
        self._count = 1
        self._price_sum = tick.price
        self._price_square_sum = tick.price * tick.price
        self._source_ids = [tick.record_id]
        self._latest_book = None
        self._latest_received_at = tick.received_at


@dataclass(frozen=True)
class RuntimeStats:
    fast_path_events: int
    rejected_events: int
    persistence_enqueued: int
    persistence_dropped: int
    persistence_completed: int
    persistence_errors: int
    last_persistence_success_at: str | None
    last_persistence_error_at: str | None
    last_persistence_error_type: str | None


class EventDrivenMarketRuntime:
    """Separates in-memory fast-path updates from blocking persistence."""

    def __init__(
        self,
        bus: BoundedMarketEventBus,
        *,
        store: object | None = None,
        persistence_capacity: int = 8192,
    ) -> None:
        self.bus = bus
        self.store = store
        self.state = MarketState()
        self._bars: dict[tuple[str, str], IncrementalMinuteBarBuilder] = {}
        self._pending_books: dict[tuple[str, tuple[object, ...]], RealtimeOrderbookSnapshot] = {}
        self._persistence: asyncio.Queue[
            tuple[MarketEvent, RealtimeMinuteBar | None]
        ] = asyncio.Queue(maxsize=persistence_capacity)
        self._fast_path_events = 0
        self._rejected_events = 0
        self._persistence_enqueued = 0
        self._persistence_dropped = 0
        self._persistence_completed = 0
        self._persistence_errors = 0
        self._last_persistence_success_at: datetime | None = None
        self._last_persistence_error_at: datetime | None = None
        self._last_persistence_error_type: str | None = None
        self.last_processed_event: MarketEvent | None = None
        #: (symbol, stream_id) -> 진행 중인 분 bar 를 마지막으로 내보낸 monotonic 시각.
        self._minute_bar_flushed: dict[tuple[str, str], float] = {}

    async def process_one(self) -> bool:
        event = await self.bus.get()
        self.last_processed_event = event
        accepted = self.state.apply(event)
        self._fast_path_events += 1
        if not accepted:
            self._rejected_events += 1
            return False
        completed_bar = None
        if isinstance(event, RealtimeTradeTick):
            # builder 는 (symbol, stream) 단위다. 스트림을 섞으면 venue 가 다른 체결이
            # 한 bar 로 합산되고 거래량이 이중 계산된다.
            builder_key = (event.symbol, event.meta.stream_id)
            builder = self._bars.setdefault(
                builder_key,
                IncrementalMinuteBarBuilder(event.symbol),
            )
            completed_bar = builder.update(event)
            pending = self._pending_books.get(
                (event.symbol, _paired_feed_key(event.meta))
            )
            if pending is not None:
                builder.update_orderbook(pending)
            if completed_bar is None:
                # 진행 중인 분도 주기적으로 저장한다.
                #
                # 완료 bar 는 **다음 분의 첫 체결**이 와야 emit 되므로, 그것만 저장하면
                # (a) 현재 분이 저장소에 없고 (b) 조용해진 심볼의 마지막 분은 영구 결손이
                # 된다. macro reasoner 는 연속 분 bar 를 요구하므로 그 결손이 곧
                # MACRO_INSUFFICIENT_DATA → NO_TRADE_MARKET → new_buy 전면 차단이었다.
                #
                # 값은 **메모리 집계기**에서 가져온다. 저장소를 다시 읽어 재집계하면
                # (초기 구현이 그랬다) 6GB 규모 DB 에서 심볼당 2초마다 조회+upsert 가
                # 발생해 lock 경합으로 조용히 실패하고, 틱은 계속 쌓이는데 bar 만
                # 사라지는 상태가 된다.
                completed_bar = self._current_bar_if_due(builder_key, builder)
        else:
            pending_key = (event.symbol, _paired_feed_key(event.meta))
            previous = self._pending_books.get(pending_key)
            if previous is None or event.exchange_timestamp >= previous.exchange_timestamp:
                self._pending_books[pending_key] = event
            for builder_key, builder in self._bars.items():
                if builder_key[0] != event.symbol or not builder.update_orderbook(event):
                    continue
                # Persist the enriched in-progress bar even when the next trade
                # is quiet. The same debounce used for trade flushes prevents a
                # busy orderbook from turning every snapshot into a DB upsert.
                if completed_bar is None:
                    completed_bar = self._current_bar_if_due(builder_key, builder)
        if self.store is not None:
            try:
                self._persistence.put_nowait((event, completed_bar))
                self._persistence_enqueued += 1
            except asyncio.QueueFull:
                # Market-state truth remains in memory; persistence loss is
                # explicit telemetry and replay can recover from the raw feed.
                self._persistence_dropped += 1
        return True

    async def persist_one(self) -> None:
        batch = [await self._persistence.get()]
        # One SQLite transaction per market event cannot keep up with an active
        # symbol and makes fresh quotes appear stale while thousands of writes
        # wait in FIFO order. Drain a bounded batch and persist each event type
        # together; ordering within each type is preserved.
        while len(batch) < 128:
            try:
                batch.append(self._persistence.get_nowait())
            except asyncio.QueueEmpty:
                break
        try:
            await asyncio.to_thread(self._persist_batch, batch)
            self._persistence_completed += len(batch)
            self._last_persistence_success_at = datetime.now(timezone.utc)
        except Exception as exc:
            self._persistence_errors += 1
            self._last_persistence_error_at = datetime.now(timezone.utc)
            self._last_persistence_error_type = exc.__class__.__name__
            raise
        finally:
            for _ in batch:
                self._persistence.task_done()

    async def wait_for_persistence(self) -> None:
        await self._persistence.join()

    def stats(self) -> RuntimeStats:
        return RuntimeStats(
            fast_path_events=self._fast_path_events,
            rejected_events=self._rejected_events,
            persistence_enqueued=self._persistence_enqueued,
            persistence_dropped=self._persistence_dropped,
            persistence_completed=self._persistence_completed,
            persistence_errors=self._persistence_errors,
            last_persistence_success_at=(
                self._last_persistence_success_at.isoformat()
                if self._last_persistence_success_at is not None
                else None
            ),
            last_persistence_error_at=(
                self._last_persistence_error_at.isoformat()
                if self._last_persistence_error_at is not None
                else None
            ),
            last_persistence_error_type=self._last_persistence_error_type,
        )

    def _current_bar_if_due(
        self,
        key: tuple[str, str],
        builder: "IncrementalMinuteBarBuilder",
    ) -> RealtimeMinuteBar | None:
        """진행 중인 분 bar 를 최소 간격마다 한 번씩 내보낸다 (메모리에서, DB 조회 없음)."""
        interval = _minute_bar_refresh_interval_seconds()
        stamp = time.monotonic()
        previous = self._minute_bar_flushed.get(key)
        if previous is not None and (stamp - previous) < interval:
            return None
        self._minute_bar_flushed[key] = stamp
        if len(self._minute_bar_flushed) > 1024:
            for stale in [
                item
                for item, value in self._minute_bar_flushed.items()
                if (stamp - value) > 600.0
            ]:
                self._minute_bar_flushed.pop(stale, None)
        return builder.current_bar()

    def _persist(
        self,
        event: MarketEvent,
        completed_bar: RealtimeMinuteBar | None,
    ) -> None:
        if isinstance(event, RealtimeTradeTick):
            self.store.save_ticks((event,))  # type: ignore[attr-defined]
        else:
            self.store.save_orderbooks((event,))  # type: ignore[attr-defined]
        if completed_bar is not None:
            self.store.save_minute_bars((completed_bar,))  # type: ignore[attr-defined]

    def _persist_batch(
        self,
        batch: list[tuple[MarketEvent, RealtimeMinuteBar | None]],
    ) -> None:
        ticks = tuple(
            event for event, _bar in batch if isinstance(event, RealtimeTradeTick)
        )
        books = tuple(
            event
            for event, _bar in batch
            if isinstance(event, RealtimeOrderbookSnapshot)
        )
        bars = tuple(bar for _event, bar in batch if bar is not None)
        if ticks:
            self.store.save_ticks(ticks)  # type: ignore[attr-defined]
        if books:
            self.store.save_orderbooks(books)  # type: ignore[attr-defined]
        if bars:
            self.store.save_minute_bars(bars)  # type: ignore[attr-defined]

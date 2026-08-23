from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from app.data.event_pipeline import (
    BoundedMarketEventBus,
    EventDrivenMarketRuntime,
    IncrementalMinuteBarBuilder,
    MarketState,
)
from app.data.market_capabilities import FeedScope, MarketGroup, SessionId, Venue
from app.data.realtime_types import (
    FeedMetadata,
    KIS_REALTIME_SOURCE,
    OrderbookLevel,
    RealtimeOrderbookSnapshot,
    RealtimeTradeTick,
)


NOW = datetime(2026, 7, 27, 0, 0, tzinfo=timezone.utc)


def _tick(second: int, price: float, sequence: int, direction: str = "BUY") -> RealtimeTradeTick:
    at = NOW + timedelta(seconds=second)
    return RealtimeTradeTick(
        symbol="005930",
        exchange_timestamp=at,
        received_at=at,
        source=KIS_REALTIME_SOURCE,
        price=price,
        volume=10,
        trade_direction=direction,
        sequence_key=f"trade-{sequence}",
    )


def _book(second: int, bid: float, ask: float, sequence: int) -> RealtimeOrderbookSnapshot:
    at = NOW + timedelta(seconds=second)
    return RealtimeOrderbookSnapshot(
        symbol="005930",
        exchange_timestamp=at,
        received_at=at,
        source=KIS_REALTIME_SOURCE,
        levels=(OrderbookLevel(bid, 120, ask, 80),),
        sequence_key=f"book-{sequence}",
    )


def _krx_meta(tr_id: str, *, venue: Venue = Venue.KRX) -> FeedMetadata:
    return FeedMetadata(
        market_group=MarketGroup.KR,
        exchange=venue.value,
        venue=venue,
        session=SessionId.KRX_REGULAR if venue is Venue.KRX else SessionId.NXT_REGULAR,
        currency="KRW",
        feed_scope=FeedScope.VENUE_SPECIFIC,
        tr_id=tr_id,
        subscription_key="005930",
        is_tradeable=True,
    )


def test_bounded_bus_coalesces_same_symbol_and_kind() -> None:
    async def scenario() -> None:
        bus = BoundedMarketEventBus(capacity=1)
        await bus.publish(_tick(0, 70000, 1))
        await bus.publish(_tick(1, 70100, 2))
        event = await bus.get()
        assert isinstance(event, RealtimeTradeTick)
        assert event.price == 70100
        assert bus.stats().coalesced == 1
        assert bus.stats().dropped == 0

    asyncio.run(scenario())


def test_market_state_rejects_duplicate_and_out_of_order_and_marks_gap() -> None:
    state = MarketState()
    assert state.apply(_tick(0, 70000, 1))
    assert not state.apply(_tick(0, 70000, 1))
    assert state.apply(_tick(2, 70200, 3))
    assert not state.apply(_tick(1, 70100, 2))
    symbol = state.symbol("005930")
    assert symbol is not None
    assert symbol.duplicate_count == 1
    assert symbol.out_of_order_count == 1
    assert symbol.gap_count == 1
    assert symbol.sequence_uncertain


def test_incremental_features_match_microstructure_formulas_and_freshness() -> None:
    state = MarketState()
    state.apply(_tick(0, 70000, 1, "BUY"))
    state.apply(_tick(1, 70100, 2, "SELL"))
    state.apply(_book(1, 70050, 70150, 1))
    features = state.symbol("005930").features(NOW + timedelta(seconds=2), 1500)  # type: ignore[union-attr]
    assert features is not None
    assert features.mid == 70100
    assert features.microprice == 70110
    assert features.orderbook_imbalance == 0.2
    assert features.vwap == 70050
    assert features.aggressor_trade_imbalance == 0
    assert features.realized_volatility > 0
    assert features.fresh
    stale = state.symbol("005930").features(NOW + timedelta(seconds=10), 1500)  # type: ignore[union-attr]
    assert stale is not None and not stale.fresh


def test_reconnect_requires_new_book_snapshot_for_sequence_confidence() -> None:
    state = MarketState()
    state.apply(_tick(0, 70000, 1))
    state.apply(_book(0, 69900, 70100, 1))
    state.mark_reconnected()
    assert state.symbol("005930").sequence_uncertain  # type: ignore[union-attr]
    state.apply(_book(1, 70000, 70200, 2))
    assert not state.symbol("005930").sequence_uncertain  # type: ignore[union-attr]


def test_incremental_bar_emits_closed_minute_without_history_recompute() -> None:
    builder = IncrementalMinuteBarBuilder("005930")
    assert builder.update(_tick(0, 70000, 1)) is None
    assert builder.update(_tick(30, 70200, 2)) is None
    completed = builder.update(_tick(60, 70100, 3))
    assert completed is not None
    assert (completed.open, completed.high, completed.low, completed.close) == (
        70000,
        70200,
        70000,
        70200,
    )
    assert completed.volume == 20
    assert completed.trade_count == 2


def test_incremental_bar_attaches_paired_orderbook_microstructure() -> None:
    builder = IncrementalMinuteBarBuilder("005930")
    tick = _tick(0, 70000, 1).with_meta(_krx_meta("H0STCNT0"))
    book = _book(1, 69900, 70100, 1).with_meta(_krx_meta("H0STASP0"))

    assert builder.update(tick) is None
    assert builder.update_orderbook(book)
    bar = builder.current_bar()

    assert bar is not None
    assert bar.stream_id == tick.meta.stream_id
    assert bar.spread_bps == book.spread_bps
    assert bar.orderbook_imbalance == book.imbalance
    assert bar.liquidity_score == 10 / 100_000


def test_incremental_bar_rejects_orderbook_from_another_venue() -> None:
    builder = IncrementalMinuteBarBuilder("005930")
    tick = _tick(0, 70000, 1).with_meta(_krx_meta("H0STCNT0"))
    nxt_book = _book(1, 69900, 70100, 1).with_meta(
        _krx_meta("H0NXASP0", venue=Venue.NXT)
    )

    builder.update(tick)

    assert not builder.update_orderbook(nxt_book)
    assert builder.current_bar().spread_bps == 0.0  # type: ignore[union-attr]


def test_runtime_pairs_book_arriving_after_trade_despite_different_tr_ids(
    monkeypatch,
) -> None:
    class Store:
        def __init__(self) -> None:
            self.bars = []

        def save_ticks(self, _values):
            pass

        def save_orderbooks(self, _values):
            pass

        def save_minute_bars(self, values):
            self.bars.extend(values)

    async def scenario() -> Store:
        monkeypatch.setenv("REALTIME_MINUTE_BAR_REBUILD_SEC", "0")
        bus = BoundedMarketEventBus(capacity=4)
        store = Store()
        runtime = EventDrivenMarketRuntime(bus, store=store)
        await bus.publish(_tick(0, 70000, 1).with_meta(_krx_meta("H0STCNT0")))
        await runtime.process_one()
        await bus.publish(_book(1, 69900, 70100, 1).with_meta(_krx_meta("H0STASP0")))
        await runtime.process_one()
        await runtime.persist_one()
        return store

    store = asyncio.run(scenario())

    assert store.bars[-1].spread_bps > 0.0
    assert store.bars[-1].orderbook_imbalance == 0.2


def test_runtime_retains_book_that_arrives_before_first_trade() -> None:
    async def scenario():
        bus = BoundedMarketEventBus(capacity=4)
        runtime = EventDrivenMarketRuntime(bus)
        await bus.publish(_book(0, 69900, 70100, 1).with_meta(_krx_meta("H0STASP0")))
        await runtime.process_one()
        await bus.publish(_tick(1, 70000, 1).with_meta(_krx_meta("H0STCNT0")))
        await runtime.process_one()
        return next(iter(runtime._bars.values())).current_bar()

    bar = asyncio.run(scenario())

    assert bar is not None
    assert bar.spread_bps > 0.0
    assert bar.orderbook_imbalance == 0.2


def test_fast_path_does_not_wait_for_persistence_and_reports_overflow() -> None:
    class Store:
        def save_ticks(self, values):
            raise AssertionError("persistence must not run in process_one")

    async def scenario() -> None:
        bus = BoundedMarketEventBus(capacity=4)
        runtime = EventDrivenMarketRuntime(bus, store=Store(), persistence_capacity=1)
        await bus.publish(_tick(0, 70000, 1))
        await runtime.process_one()
        await bus.publish(_tick(1, 70100, 2))
        await runtime.process_one()
        symbol = runtime.state.symbol("005930")
        assert symbol is not None and symbol.latest_tick.price == 70100
        assert runtime.stats().persistence_enqueued == 1
        assert runtime.stats().persistence_dropped == 1

    asyncio.run(scenario())


def test_event_driven_collector_drains_async_persistence(monkeypatch) -> None:
    from app.data import event_runtime

    class Store:
        def __init__(self):
            self.saved = 0

        def save_ticks(self, values):
            self.saved += len(values)

        def save_orderbooks(self, values):
            self.saved += len(values)

        def save_minute_bars(self, values):
            self.saved += len(values)

    async def fake_collector(*, event_sink, **kwargs):
        await event_sink(_tick(0, 70000, 1))
        await event_sink(_book(0, 69900, 70100, 1))
        return {"messages": 2}

    store = Store()
    monkeypatch.setattr(
        event_runtime, "run_kis_realtime_websocket_collector", fake_collector
    )
    result = asyncio.run(
        event_runtime.run_event_driven_kis_websocket_collector(store=store)
    )
    assert result["mode"] == "event_driven"
    assert result["event_runtime"]["fast_path_events"] == 2
    assert result["event_runtime"]["persistence_completed"] == 2
    # tick 1건 + 호가 1건 + **진행 중인 분 bar** 1건 = 3.
    #
    # 진행 중 bar 를 함께 저장하는 것이 의도된 동작이다. 완료 bar 는 다음 분의 첫 체결이
    # 와야 emit 되므로, 그것만 저장하면 현재 분이 저장소에 없고 조용해진 심볼의 마지막
    # 분은 영구 결손이 된다. macro reasoner 는 연속 분 bar 를 요구하며, 결손 시
    # MACRO_INSUFFICIENT_DATA → NO_TRADE_MARKET 으로 신규 매수가 전면 차단된다.
    assert store.saved == 3


def test_persistence_worker_survives_one_store_failure() -> None:
    from app.data import event_runtime

    class Store:
        def __init__(self) -> None:
            self.attempts = 0
            self.saved = 0

        def save_ticks(self, values) -> None:
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("transient sqlite write failure")
            self.saved += len(values)

        def save_orderbooks(self, values) -> None:
            self.saved += len(values)

        def save_minute_bars(self, values) -> None:
            self.saved += len(values)

    async def scenario() -> tuple[int, object]:
        bus = BoundedMarketEventBus(capacity=4)
        store = Store()
        runtime = EventDrivenMarketRuntime(bus, store=store)
        worker = asyncio.create_task(event_runtime.runtime_persistence_worker(runtime))
        try:
            await bus.publish(_tick(0, 70000, 1))
            await runtime.process_one()
            for _ in range(100):
                if store.attempts:
                    break
                await asyncio.sleep(0.001)
            await bus.publish(_tick(1, 70100, 2))
            await runtime.process_one()
            for _ in range(100):
                if store.saved:
                    break
                await asyncio.sleep(0.001)
            return store.saved, runtime.stats()
        finally:
            if worker.done():
                try:
                    worker.result()
                except OSError:
                    pass
            else:
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    pass

    saved, stats = asyncio.run(scenario())

    assert saved == 1
    assert stats.persistence_errors == 1
    assert stats.persistence_completed == 1
    assert stats.last_persistence_error_type == "OSError"
    assert stats.last_persistence_error_at is not None
    assert stats.last_persistence_success_at is not None


def test_event_runtime_can_dispatch_slow_shadow_off_fast_path(monkeypatch, tmp_path) -> None:
    from app.data import event_runtime

    class Store:
        def save_ticks(self, values):
            pass

        def save_orderbooks(self, values):
            pass

        def save_minute_bars(self, values):
            pass

    async def fake_collector(*, event_sink, **kwargs):
        await event_sink(_tick(0, 70000, 1))
        await event_sink(_book(0, 69900, 70100, 1))
        return {"messages": 2}

    monkeypatch.setattr(
        event_runtime, "run_kis_realtime_websocket_collector", fake_collector
    )
    with patch.dict(
        "os.environ",
        {
            "REFACTOR_ONTOLOGY_ROUTER": "true",
            "REFACTOR_GNN_SHADOW": "true",
        },
        clear=True,
    ):
        result = asyncio.run(
            event_runtime.run_event_driven_kis_websocket_collector(store=Store())
        )
    assert result["slow_intelligence_enabled"] is True


def _slow_request(symbol: str):
    from app.data.event_runtime import _SlowSnapshotRequest

    now = datetime.now(timezone.utc)
    return _SlowSnapshotRequest(
        symbol=symbol,
        record_id=f"{symbol}-1",
        now=now,
        as_of=now,
        last_price=100.0,
        data_fresh=True,
        sequence_uncertain=False,
    )


def test_slow_shadow_worker_continues_after_one_inference_failure(monkeypatch) -> None:
    from app.data import event_runtime
    from app.data.event_runtime import slow_intelligence_worker

    class FlakyService:
        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, _snapshot) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient inference failure")

    monkeypatch.setattr(
        event_runtime,
        "_build_slow_snapshot",
        lambda runtime, request: SimpleNamespace(symbol=request.symbol),
    )

    async def scenario() -> int:
        service = FlakyService()
        queue: asyncio.Queue = asyncio.Queue()
        worker = asyncio.create_task(
            slow_intelligence_worker(service, queue, SimpleNamespace(store=object()))
        )
        await queue.put(_slow_request("SOFI"))
        await queue.put(_slow_request("PFE"))
        await asyncio.wait_for(queue.join(), timeout=2.0)
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        return service.calls

    assert asyncio.run(scenario()) == 2


def test_slow_snapshot_build_never_runs_on_the_event_loop_thread(monkeypatch) -> None:
    """The store-backed half must be executed in a worker thread.

    Building the snapshot issues several SQLite queries against an 8GB store
    (``_slow_context_bars`` -> ``recent_minute_bars``). Running that on the
    collector's event loop, once per second per symbol, starved the websocket
    reader sharing it: the socket stayed ESTABLISHED with the kernel receive
    queue climbing and US market data stopped ~70s after every reconnect. A
    SIGUSR1 thread dump caught the loop parked in ``recent_minute_bars``.
    """
    import threading

    from app.data import event_runtime
    from app.data.event_runtime import slow_intelligence_worker

    build_threads: list[str] = []

    def _fake_build(runtime, request):
        build_threads.append(threading.current_thread().name)
        return SimpleNamespace(symbol=request.symbol)

    monkeypatch.setattr(event_runtime, "_build_slow_snapshot", _fake_build)

    async def scenario() -> str:
        queue: asyncio.Queue = asyncio.Queue()
        worker = asyncio.create_task(
            slow_intelligence_worker(
                SimpleNamespace(evaluate=lambda _s: None),
                queue,
                SimpleNamespace(store=object()),
            )
        )
        await queue.put(_slow_request("INTC"))
        await asyncio.wait_for(queue.join(), timeout=2.0)
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        return threading.current_thread().name

    loop_thread = asyncio.run(scenario())

    assert build_threads, "the snapshot was never built"
    assert loop_thread not in build_threads


def test_event_runtime_builds_current_strategy_graph_context(monkeypatch) -> None:
    from app.data import event_runtime
    from app.features import live_feature_frame
    from app.features.strategy_graph_context import (
        STRATEGY_GRAPH_CONTEXT_DIM,
        STRATEGY_GRAPH_CONTEXT_FIELDS,
    )

    runtime = SimpleNamespace(store=object())
    monkeypatch.setattr(live_feature_frame, "_rvgi_box_columns", lambda *a: {})
    monkeypatch.setattr(live_feature_frame, "_slow_context_bars", lambda *a: (object(), ()))
    monkeypatch.setattr(
        live_feature_frame,
        "_strategy_graph_context_columns",
        lambda *a, **k: {name: 0.0 for name in STRATEGY_GRAPH_CONTEXT_FIELDS},
    )

    context = event_runtime._runtime_strategy_graph_context(
        runtime,
        "005930",
        datetime.now(timezone.utc),
        70_000.0,
    )

    assert context is not None
    assert len(context) == STRATEGY_GRAPH_CONTEXT_DIM


def test_bus_idle_wait_sleeps_until_published_instead_of_spinning() -> None:
    """An idle consumer must WAIT, not poll ``stats().depth`` in a bare yield.

    The consumer loop used ``await asyncio.sleep(0)`` when the bus was empty.
    That is a yield, not a sleep, so the loop ran as fast as the interpreter
    could execute it and starved the KIS websocket reader sharing its event
    loop: on 2026-08-21 the socket stayed ESTABLISHED with a growing kernel
    receive queue while the process burned 1.6 cores and read nothing.
    """

    async def scenario() -> tuple[bool, bool, int]:
        bus = BoundedMarketEventBus(4)
        loop = asyncio.get_running_loop()

        # Empty bus: the wait must actually block for the whole timeout.
        started = loop.time()
        timed_out = await bus.wait_for_depth(0.05)
        idle_elapsed = loop.time() - started

        polls = 0

        async def _count_polls() -> None:
            nonlocal polls
            while True:
                polls += 1
                await asyncio.sleep(0)

        counter = asyncio.create_task(_count_polls())
        waiter = asyncio.create_task(bus.wait_for_depth(1.0))
        await asyncio.sleep(0.01)
        await bus.publish(_tick(1, 100.0, 1))
        woke = await waiter
        counter.cancel()
        try:
            await counter
        except asyncio.CancelledError:
            pass
        return timed_out, woke, int(idle_elapsed >= 0.04)

    timed_out, woke, blocked_for_the_timeout = asyncio.run(scenario())

    # Empty bus -> reports nothing available, and did not return immediately.
    assert timed_out is False
    assert blocked_for_the_timeout == 1
    # A publish wakes the waiter rather than it discovering the item by polling.
    assert woke is True

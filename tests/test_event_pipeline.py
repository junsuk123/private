from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from app.data.event_pipeline import (
    BoundedMarketEventBus,
    EventDrivenMarketRuntime,
    IncrementalMinuteBarBuilder,
    MarketState,
)
from app.data.realtime_types import (
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


def test_slow_shadow_worker_continues_after_one_inference_failure() -> None:
    from app.data.event_runtime import slow_intelligence_worker

    class FlakyService:
        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, _snapshot) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient inference failure")

    async def scenario() -> int:
        service = FlakyService()
        queue: asyncio.Queue = asyncio.Queue()
        worker = asyncio.create_task(slow_intelligence_worker(service, queue))
        await queue.put(type("Snapshot", (), {"symbol": "SOFI"})())
        await queue.put(type("Snapshot", (), {"symbol": "PFE"})())
        await asyncio.wait_for(queue.join(), timeout=2.0)
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        return service.calls

    assert asyncio.run(scenario()) == 2

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from typing import Any

from app.config.refactor_flags import RefactorFeatureFlags
from app.data.event_pipeline import BoundedMarketEventBus, EventDrivenMarketRuntime
from app.data.kis_realtime import run_kis_realtime_websocket_collector
from app.data.realtime_store import RealtimeMarketDataStore
from app.features.strategy_graph_context import (
    STRATEGY_GRAPH_CONTEXT_DIM,
    STRATEGY_GRAPH_CONTEXT_SCHEMA,
    StrategyGraphContextError,
    build_strategy_graph_context,
)
from app.routing.shadow_intelligence import (
    STRATEGY_IDS,
    ShadowIntelligenceService,
    SlowIntelligenceSnapshot,
)

logger = logging.getLogger(__name__)


#: 해외(미국) 실시간 TR 쌍과 세션 인식 subscription key. 값의 근거는
#: ``docs/kis_market_session_capability_matrix.md`` §4.
_OVERSEAS_SUBSCRIPTION_TR_IDS = ("HDFSCNT0", "HDFSASP0")


async def run_event_driven_kis_overseas_websocket_collector(
    **collector_kwargs: Any,
) -> dict[str, Any]:
    """미국 수집을 국내와 **같은** event-driven 런타임으로 돌린다.

    이전에는 미국만 ``run_kis_overseas_realtime_websocket_collector`` 를 직접 호출해
    ``event_sink`` 없이 동작했다. 그 경로는 메시지마다 ``build_latest_minute_bar`` 로
    저장소를 다시 읽어 분 bar 를 재집계하는데, 6GB 규모 tick 테이블에서는 lock 경합으로
    실패하고 (그 호출은 try 로 감싸여 있지도 않아) 수집기 자체를 죽였다. 결과적으로
    틱은 쌓이는데 분 bar 만 사라지는 상태가 되고, macro reasoner 가 연속 분 bar 를
    얻지 못해 ``MACRO_INSUFFICIENT_DATA`` → ``NO_TRADE_MARKET`` → 신규 매수 전면 차단이
    됐다.

    event-driven 런타임은 (a) 분 bar 를 메모리 집계기로 만들고, (b) 영속화를 WebSocket
    콜백 밖의 워커로 옮기며, (c) 국내와 동일한 metadata 파이프라인을 쓴다.
    """
    collector_kwargs.setdefault(
        "subscription_tr_ids", _OVERSEAS_SUBSCRIPTION_TR_IDS
    )
    if "subscription_key_factory" not in collector_kwargs:
        from app.data.kis_realtime import overseas_realtime_subscription_key

        collector_kwargs["subscription_key_factory"] = (
            overseas_realtime_subscription_key
        )
    return await run_event_driven_kis_websocket_collector(**collector_kwargs)


async def run_event_driven_kis_websocket_collector(
    *,
    event_bus_capacity: int = 4096,
    persistence_capacity: int = 8192,
    persistence_workers: int = 1,
    **collector_kwargs: Any,
) -> dict[str, Any]:
    """Run KIS ingestion with DB work outside the WebSocket callback."""
    store = collector_kwargs.pop("store", None) or RealtimeMarketDataStore()
    bus = BoundedMarketEventBus(event_bus_capacity)
    runtime = EventDrivenMarketRuntime(
        bus,
        store=store,
        persistence_capacity=persistence_capacity,
    )
    workers = [
        asyncio.create_task(runtime_persistence_worker(runtime))
        for _ in range(max(1, persistence_workers))
    ]
    flags = RefactorFeatureFlags.from_env()
    slow_service = (
        ShadowIntelligenceService(
            feature_dim=STRATEGY_GRAPH_CONTEXT_DIM,
            enable_npu_comparison=flags.npu_inference,
        )
        if flags.ontology_router or flags.gnn_shadow
        else None
    )
    slow_queue: asyncio.Queue[SlowIntelligenceSnapshot] | None = (
        asyncio.Queue(maxsize=64) if slow_service is not None else None
    )
    slow_worker = (
        asyncio.create_task(slow_intelligence_worker(slow_service, slow_queue))
        if slow_service is not None and slow_queue is not None
        else None
    )
    slow_snapshot_last_enqueued: dict[str, float] = {}
    try:
        slow_snapshot_interval = max(
            0.1,
            float(os.getenv("REALTIME_SLOW_INTELLIGENCE_INTERVAL_SECONDS", "1.0")),
        )
    except (TypeError, ValueError):
        slow_snapshot_interval = 1.0
    collector = asyncio.create_task(
        run_kis_realtime_websocket_collector(
            store=store,
            event_sink=bus.publish,
            **collector_kwargs,
        )
    )
    try:
        while not collector.done() or bus.stats().depth:
            if bus.stats().depth:
                accepted = await runtime.process_one()
                if accepted and slow_queue is not None:
                    event = runtime.last_processed_event
                    symbol = str(getattr(event, "symbol", "") or "")
                    now_monotonic = time.monotonic()
                    last_enqueued = slow_snapshot_last_enqueued.get(symbol, 0.0)
                    snapshot = (
                        _slow_snapshot(runtime)
                        if symbol
                        and now_monotonic - last_enqueued >= slow_snapshot_interval
                        else None
                    )
                    if snapshot is not None:
                        slow_snapshot_last_enqueued[symbol] = now_monotonic
                        if slow_queue.full():
                            with suppress(asyncio.QueueEmpty):
                                slow_queue.get_nowait()
                                slow_queue.task_done()
                        slow_queue.put_nowait(snapshot)
            else:
                await asyncio.sleep(0)
        counts = await collector
        await runtime.wait_for_persistence()
        if slow_queue is not None:
            await slow_queue.join()
    finally:
        for worker in workers:
            worker.cancel()
        for worker in workers:
            with suppress(asyncio.CancelledError):
                await worker
        if slow_worker is not None:
            slow_worker.cancel()
            with suppress(asyncio.CancelledError):
                await slow_worker
    return {
        **counts,
        "event_bus": vars(bus.stats()),
        "event_runtime": vars(runtime.stats()),
        "mode": "event_driven",
        "slow_intelligence_enabled": slow_service is not None,
    }


async def runtime_persistence_worker(runtime: EventDrivenMarketRuntime) -> None:
    while True:
        try:
            await runtime.persist_one()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - one transient store failure must not kill the worker.
            logger.exception("realtime market persistence batch failed; worker will continue")


async def slow_intelligence_worker(
    service: ShadowIntelligenceService,
    queue: asyncio.Queue[SlowIntelligenceSnapshot],
) -> None:
    while True:
        snapshot = await queue.get()
        try:
            try:
                await asyncio.to_thread(service.evaluate, snapshot)
            except Exception:  # noqa: BLE001 - one bad inference must not kill all future samples.
                logger.exception(
                    "slow intelligence inference failed for %s",
                    snapshot.symbol,
                )
        finally:
            queue.task_done()


def _slow_snapshot(runtime: EventDrivenMarketRuntime) -> SlowIntelligenceSnapshot | None:
    event = runtime.last_processed_event
    if event is None:
        return None
    state = runtime.state.symbol(event.symbol)
    if state is None:
        return None
    now = datetime.now(timezone.utc)
    features = state.features(now, staleness_ms=5_000)
    if features is None:
        return None
    values = _runtime_strategy_graph_context(
        runtime,
        event.symbol,
        now,
        features.last_price,
    )
    if values is None:
        return None
    return SlowIntelligenceSnapshot(
        snapshot_id=f"{event.symbol}:{event.record_id}",
        symbol=event.symbol,
        as_of=features.as_of,
        valid_until=features.as_of + timedelta(seconds=5),
        feature_snapshot_id=f"event:{event.record_id}",
        features=values,
        data_fresh=features.fresh,
        tradable=features.fresh and not features.sequence_uncertain,
        allowed_strategy_ids=STRATEGY_IDS,
        feature_schema_name=STRATEGY_GRAPH_CONTEXT_SCHEMA,
        reference_price=max(0.0, float(features.last_price)),
    )


def _runtime_strategy_graph_context(
    runtime: EventDrivenMarketRuntime,
    symbol: str,
    now: datetime,
    last_price: float,
) -> tuple[float, ...] | None:
    """Build the same completed-minute-bar context used in training and web inference."""
    if runtime.store is None or last_price <= 0:
        return None
    try:
        from app.features.live_feature_frame import (
            _rvgi_box_columns,
            _slow_context_bars,
            _strategy_graph_context_columns,
        )

        rvgi_box = _rvgi_box_columns(runtime.store, symbol, now, last_price)
        bar_set, rows = _slow_context_bars(runtime.store, symbol, now)
        columns = _strategy_graph_context_columns(
            bar_set,
            rows,
            symbol=symbol,
            rvgi_box=rvgi_box,
        )
        if not columns:
            return None
        return build_strategy_graph_context(columns)
    except (StrategyGraphContextError, TypeError, ValueError):
        return None

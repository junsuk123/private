from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import suppress
from dataclasses import dataclass
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


#: Idle wait between bus polls. Long enough that an empty bus costs nothing,
#: short enough that ``collector.done()`` is still noticed promptly.
_BUS_IDLE_WAIT_SECONDS = 0.25


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
    quant_sink = collector_kwargs.pop("quant_sink", None)
    quant_activation: dict[str, Any]
    if quant_sink is None:
        try:
            from app.quant.runtime import build_quant_sink

            quant_sink, decision = build_quant_sink(market_store=store)
            quant_activation = decision.as_dict()
        except Exception as exc:  # noqa: BLE001 - optional evidence cannot stop KIS ingestion.
            quant_activation = {
                "enabled": False,
                "mode": "auto",
                "conditions": [],
                "unavailable_reason": f"{type(exc).__name__}:{exc}",
            }
            logger.exception("quant reference layer initialization failed; KIS ingestion continues")
    else:
        quant_activation = {
            "enabled": True,
            "mode": "injected",
            "conditions": ["injected_sink"],
            "unavailable_reason": None,
        }
    bus = BoundedMarketEventBus(event_bus_capacity)
    runtime = EventDrivenMarketRuntime(
        bus,
        store=store,
        persistence_capacity=persistence_capacity,
        completed_bar_sink=quant_sink,
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
    slow_queue: "asyncio.Queue[_SlowSnapshotRequest] | None" = (
        asyncio.Queue(maxsize=64) if slow_service is not None else None
    )
    slow_worker = (
        asyncio.create_task(slow_intelligence_worker(slow_service, slow_queue, runtime))
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
                    # Only the cheap in-memory part happens here. Building the
                    # full snapshot used to run on this event loop, and it issues
                    # several SQLite queries against an 8GB store
                    # (``_slow_context_bars`` -> ``recent_minute_bars``). At one
                    # per second per symbol across six subscribed symbols that
                    # consumed the loop, so the websocket reader sharing it never
                    # got to call ``recv()``: the socket stayed ESTABLISHED with
                    # the kernel receive queue climbing and US market data
                    # stopped ~70s after every reconnect, with nothing raised.
                    # Confirmed by a SIGUSR1 thread dump showing this loop parked
                    # in ``recent_minute_bars``. The store work now happens in the
                    # slow worker's thread. See
                    # [[obaits-collector-busywait-starves-websocket]].
                    request = (
                        _slow_snapshot_request(runtime)
                        if symbol
                        and now_monotonic - last_enqueued >= slow_snapshot_interval
                        else None
                    )
                    if request is not None:
                        slow_snapshot_last_enqueued[symbol] = now_monotonic
                        if slow_queue.full():
                            with suppress(asyncio.QueueEmpty):
                                slow_queue.get_nowait()
                                slow_queue.task_done()
                        slow_queue.put_nowait(request)
            else:
                # Not ``sleep(0)``: that is a bare yield, so this loop spun as
                # fast as the interpreter could run it and starved the websocket
                # reader sharing this event loop. Wait for the producer instead.
                await bus.wait_for_depth(_BUS_IDLE_WAIT_SECONDS)
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
        "quant_reference": (
            {**quant_activation, **quant_sink.health()}
            if quant_sink is not None and callable(getattr(quant_sink, "health", None))
            else quant_activation
        ),
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
    queue: "asyncio.Queue[_SlowSnapshotRequest]",
    runtime: EventDrivenMarketRuntime,
) -> None:
    """Build and evaluate slow snapshots entirely off the event loop.

    Both halves run in a worker thread: assembling the snapshot queries the store
    (which is what stalled the websocket reader when it ran on the loop), and the
    inference is CPU work. ``_build_slow_snapshot`` only reads
    ``runtime.store``, which opens its own connection per call, so it is safe
    here — the mutable runtime state was already sampled on the loop.
    """
    while True:
        request = await queue.get()
        try:
            try:
                await asyncio.to_thread(
                    _build_and_evaluate_slow_snapshot, service, runtime, request
                )
            except Exception:  # noqa: BLE001 - one bad inference must not kill all future samples.
                logger.exception(
                    "slow intelligence inference failed for %s",
                    request.symbol,
                )
        finally:
            queue.task_done()


def _build_and_evaluate_slow_snapshot(
    service: ShadowIntelligenceService,
    runtime: EventDrivenMarketRuntime,
    request: "_SlowSnapshotRequest",
) -> None:
    snapshot = _build_slow_snapshot(runtime, request)
    if snapshot is None:
        return
    service.evaluate(snapshot)


@dataclass(frozen=True)
class _SlowSnapshotRequest:
    """Everything sampled from mutable runtime state, frozen on the event loop.

    Split out so the loop touches ONLY in-memory state. The store-backed half of
    the old ``_slow_snapshot`` (``_runtime_strategy_graph_context``) now runs in
    the slow worker's thread, where a slow SQLite query costs a delayed shadow
    evaluation instead of a stalled market-data socket.
    """

    symbol: str
    record_id: str
    now: datetime
    as_of: datetime
    last_price: float
    data_fresh: bool
    sequence_uncertain: bool


def _slow_snapshot_request(
    runtime: EventDrivenMarketRuntime,
) -> _SlowSnapshotRequest | None:
    """Cheap, in-memory sample of the just-processed event. No store access."""
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
    return _SlowSnapshotRequest(
        symbol=event.symbol,
        record_id=str(event.record_id),
        now=now,
        as_of=features.as_of,
        last_price=float(features.last_price),
        data_fresh=bool(features.fresh),
        sequence_uncertain=bool(features.sequence_uncertain),
    )


def _build_slow_snapshot(
    runtime: EventDrivenMarketRuntime, request: _SlowSnapshotRequest
) -> SlowIntelligenceSnapshot | None:
    """Store-backed half. Must run off the event loop — it queries SQLite."""
    values = _runtime_strategy_graph_context(
        runtime,
        request.symbol,
        request.now,
        request.last_price,
    )
    if values is None:
        return None
    return SlowIntelligenceSnapshot(
        snapshot_id=f"{request.symbol}:{request.record_id}",
        symbol=request.symbol,
        as_of=request.as_of,
        valid_until=request.as_of + timedelta(seconds=5),
        feature_snapshot_id=f"event:{request.record_id}",
        features=values,
        data_fresh=request.data_fresh,
        tradable=request.data_fresh and not request.sequence_uncertain,
        allowed_strategy_ids=STRATEGY_IDS,
        feature_schema_name=STRATEGY_GRAPH_CONTEXT_SCHEMA,
        reference_price=max(0.0, request.last_price),
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

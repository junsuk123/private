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
from app.routing.shadow_intelligence import (
    STRATEGY_IDS,
    ShadowIntelligenceService,
    SlowIntelligenceSnapshot,
)

logger = logging.getLogger(__name__)


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
            feature_dim=28,
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
        await runtime.persist_one()


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
    values = (
        features.last_price / 100_000,
        (features.mid or features.last_price) / 100_000,
        (features.spread_bps or 0) / 100,
        (features.microprice or features.last_price) / 100_000,
        features.orderbook_imbalance or 0,
        features.order_flow_imbalance / 10_000,
        features.aggressor_trade_imbalance,
        features.vwap / 100_000,
        features.realized_volatility * 100,
        1.0 if features.fresh else 0.0,
        1.0 if features.sequence_uncertain else 0.0,
        1.0,
    )
    rvgi_box = _runtime_rvgi_box_features(runtime, event.symbol, now, features.last_price)
    values = (*values, *rvgi_box)
    is_krx = event.symbol.isdigit() and len(event.symbol) == 6
    values = (*values, 1.0 if is_krx else 0.0)
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
        feature_schema_name="realtime_strategy_graph_v4_market",
    )


def _runtime_rvgi_box_features(
    runtime: EventDrivenMarketRuntime,
    symbol: str,
    now: datetime,
    last_price: float,
) -> tuple[float, ...]:
    """Normalized completed-bar descriptors appended to the GNN snapshot."""
    if runtime.store is None or last_price <= 0:
        return (0.0,) * 15
    try:
        from app.features.live_feature_frame import _rvgi_box_columns

        row = _rvgi_box_columns(runtime.store, symbol, now, last_price)
    except Exception:  # noqa: BLE001 - missing bar history is an explicit mask.
        return (0.0,) * 15
    scale = max(float(last_price), 1e-12)
    return (
        float(row["rvgi_available"]),
        float(row["rvgi"]),
        float(row["rvgi_signal"]),
        float(row["rvgi_diff"]),
        float(row["rvgi_slope"]),
        float(row["rvgi_bullish_cross"]),
        float(row["box_available"]),
        float(row["box_high"]) / scale,
        float(row["box_low"]) / scale,
        float(row["box_mid"]) / scale,
        float(row["box_width_pct"]),
        float(row["box_position"]),
        float(row["breakout_distance_bps"]) / 100.0,
        float(row["box_previous_close"]) / scale,
        1.0 if float(row["box_context_timestamp_epoch"]) > 0 else 0.0,
    )

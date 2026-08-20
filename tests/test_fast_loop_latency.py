"""The fast path: tick -> strategy -> guard -> broker, and what it is not allowed to do.

The latency assertions here are the executable form of the constraint "no ontology, GNN,
DB or model inference inside the per-tick critical section". They are deliberately
generous in absolute terms — the point is to catch a stage that has become
*qualitatively* heavy (a database read, a model call), not to police microseconds on a
loaded CI box.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from app.execution.execution_guard import ExecutionGuard, GuardOrder
from app.monitoring.execution_latency import (
    FAST_PATH_STAGES,
    ExecutionLatencyRecorder,
)
from app.trading.fast_loop_runner import FastLoopRunner
from app.trading.strategy_fast_executor import (
    FastLoopState,
    StrategyFastExecutor,
    TickEvent,
)
from app.trading.trade_plan import EntryRule, ExitRules, TradePlan

NOW = datetime(2026, 8, 19, 1, 30, tzinfo=timezone.utc)

#: A decision must not take longer than this. A SQLite read on a warm page cache is
#: ~0.1-1ms and a model inference tens of ms, so 5ms separates "compared some floats"
#: from "did real work" without being flaky.
MAX_DECISION_LATENCY_MS = 5.0


def _plan(**overrides) -> TradePlan:
    base = dict(
        plan_id="plan-FAST-1",
        created_at=NOW,
        expires_at=NOW + timedelta(minutes=10),
        symbol="000660",
        market="KR",
        direction="LONG",
        strategy_id="intraday_momentum",
        quantity=10,
        max_notional=700_000.0,
        entry_rule=EntryRule(trigger="intraday_momentum", min_price=69_000.0, max_price=71_000.0),
        exit_rules=ExitRules(
            take_profit_rate=0.006,
            stop_loss_rate=0.003,
            trailing_rate=0.0015,
            max_holding_seconds=900,
        ),
        cancel_rule="PLAN_EXPIRY",
        expected_net_edge_bps=67.0,
        cost_snapshot={},
        risk_snapshot={},
        weekday_time_context={},
        source_ids=(),
        reference_price=70_000.0,
    )
    base.update(overrides)
    return TradePlan(**base)


class _Recorder:
    def __init__(self) -> None:
        self.orders: list[tuple[object, object]] = []

    def submit(self, order, request):  # noqa: ANN001
        self.orders.append((order, request))
        return {"status": "SUBMITTED"}


def _runner(**overrides) -> tuple[FastLoopRunner, _Recorder, ExecutionLatencyRecorder]:
    sink = _Recorder()
    latency = ExecutionLatencyRecorder()
    base = dict(
        guard=ExecutionGuard(pre_submit_guard=None, require_plan=False),
        submit=sink.submit,
        order_factory=lambda plan, request: GuardOrder(
            symbol=plan.symbol,
            market=plan.market,
            side=request.side,
            quantity=request.quantity,
            limit_price=request.limit_price,
        ),
        orderable_cash_provider=lambda symbol: 1e9,
        sellable_quantity_provider=lambda symbol: 10,
        latency=latency,
    )
    base.update(overrides)
    return FastLoopRunner(**base), sink, latency


def _tick(
    price: float, *, at: datetime = NOW, received_at: datetime | None = None
) -> TickEvent:
    """A tick. ``received_at`` separates feed-receipt time from the frozen event time.

    Needed because ``decision_latency_ms`` is ``strategy_decision - tick_received``,
    and the ``strategy_decision`` stage is stamped from the wall clock inside the
    runner (correctly -- a latency cannot be measured on an injected clock). With
    ``received_time`` also frozen to ``NOW``, that delta measured the gap between the
    hardcoded date and the real clock, so it grew by a second every second: the
    assertion held only when run near 2026-08-19 01:30 UTC and failed by ~36,500,000ms
    ten hours later. Tests that care about the decision COST pass a live
    ``received_at``; the rest keep the frozen clock, which is what makes them
    deterministic.
    """
    return TickEvent(
        symbol="000660",
        price=price,
        event_time=at,
        received_time=received_at if received_at is not None else at,
    )


def _live_tick(price: float) -> TickEvent:
    """A tick whose receipt is stamped now, so latency spans measure real elapsed work."""
    return _tick(price, received_at=datetime.now(timezone.utc))


# --------------------------------------------------------------------------- #
# The path
# --------------------------------------------------------------------------- #
def test_a_tick_drives_an_entry_all_the_way_to_the_broker() -> None:
    runner, sink, _ = _runner()
    runner.adopt(_plan())
    result = runner.on_tick(_tick(70_000.0), now=NOW)
    assert result.submitted
    assert result.request is not None and result.request.action == "SUBMIT_ENTRY"
    assert len(sink.orders) == 1


def test_a_tick_outside_the_band_submits_nothing() -> None:
    runner, sink, _ = _runner()
    runner.adopt(_plan())
    result = runner.on_tick(_tick(80_000.0), now=NOW)
    assert not result.submitted
    assert result.request is None
    assert sink.orders == []


def test_a_stop_loss_tick_reaches_the_broker_from_an_open_position() -> None:
    runner, sink, _ = _runner()
    runner.adopt(_plan())
    runner.on_tick(_tick(70_000.0), now=NOW)
    runner.on_entry_fill("000660", 70_000.0, 10, now=NOW)
    result = runner.on_tick(_tick(69_000.0), now=NOW)
    assert result.submitted
    assert result.request is not None
    assert result.request.reason == "STRATEGY_STOP_LOSS"
    assert result.request.urgent
    assert sink.orders[-1][1].action == "SUBMIT_EXIT"


def test_the_guard_can_block_and_the_executor_recovers() -> None:
    runner, sink, _ = _runner(
        guard=ExecutionGuard(
            pre_submit_guard=None, require_plan=False, kill_switch_provider=lambda: True
        )
    )
    runner.adopt(_plan())
    result = runner.on_tick(_tick(70_000.0), now=NOW)
    assert not result.submitted
    assert "GUARD_KILL_SWITCH_ENGAGED" in result.blocked_reasons
    assert sink.orders == []
    # A blocked entry returns to WAIT_ENTRY so the next tick can try again.
    assert runner.executor_for("000660").state is FastLoopState.WAIT_ENTRY


def test_a_broker_clip_reduces_the_submitted_quantity() -> None:
    runner, sink, _ = _runner(orderable_cash_provider=lambda symbol: 300_000.0)
    runner.adopt(_plan())
    result = runner.on_tick(_tick(70_000.0), now=NOW)
    assert result.submitted
    assert result.request.quantity < 10
    assert "BROKER_CLIP" in result.request.reason


def test_a_broker_exception_is_recorded_not_raised() -> None:
    def _explode(order, request):  # noqa: ANN001
        raise RuntimeError("connection reset")

    runner, _, _ = _runner(submit=_explode)
    runner.adopt(_plan())
    result = runner.on_tick(_tick(70_000.0), now=NOW)
    assert not result.submitted
    assert result.blocked_reasons[0].startswith("SUBMIT_FAILED")


def test_force_exit_all_flattens_every_open_plan() -> None:
    runner, sink, _ = _runner()
    runner.adopt(_plan())
    runner.on_tick(_tick(70_000.0), now=NOW)
    runner.on_entry_fill("000660", 70_000.0, 10, now=NOW)
    results = runner.force_exit_all("EMERGENCY")
    assert len(results) == 1 and results[0].submitted
    assert sink.orders[-1][1].reason == "EMERGENCY"


def test_ticks_for_unowned_symbols_are_ignored() -> None:
    runner, sink, _ = _runner()
    runner.adopt(_plan())
    runner.on_ticks(
        [
            type("T", (), {"symbol": "005380", "price": 70_000.0,
                           "exchange_timestamp": NOW, "received_at": NOW, "volume": 1})()
        ]
    )
    assert sink.orders == []


# --------------------------------------------------------------------------- #
# Latency
# --------------------------------------------------------------------------- #
def test_every_stage_is_timestamped() -> None:
    runner, _, latency = _runner()
    runner.adopt(_plan())
    runner.on_tick(_tick(70_000.0), now=NOW)
    span = latency.recent(1)[0]
    for stage in FAST_PATH_STAGES:
        assert stage in span["stages"], stage
    assert span["decision_latency_ms"] is not None
    assert span["guard_latency_ms"] is not None
    assert span["submit_latency_ms"] is not None


def test_the_decision_stage_stays_light() -> None:
    """The constraint made executable: deciding must not do real work."""
    runner, _, latency = _runner()
    runner.adopt(_plan())
    # Every tick here carries a live receipt time, so each span measures the work the
    # decision actually did rather than the distance to a frozen date.
    runner.on_tick(_live_tick(70_000.0), now=NOW)
    runner.on_entry_fill("000660", 70_000.0, 10, now=NOW)
    for index in range(200):
        runner.on_tick(_live_tick(70_000.0 + (index % 5)), now=NOW)
    summary = latency.summary()
    assert summary["sample_count"] >= 200
    assert summary["decision_latency_ms"]["p99"] < MAX_DECISION_LATENCY_MS


def test_the_executor_itself_is_fast_in_wall_clock_terms() -> None:
    executor = StrategyFastExecutor(_plan())
    executor.on_tick(_tick(70_000.0), now=NOW)
    executor.on_entry_fill(70_000.0, 10, now=NOW)
    started = time.perf_counter()
    for index in range(2_000):
        executor.on_tick(_tick(70_000.0 + (index % 3)), now=NOW)
    per_tick_ms = (time.perf_counter() - started) * 1000.0 / 2_000
    assert per_tick_ms < MAX_DECISION_LATENCY_MS


def test_the_runner_has_no_import_path_to_the_slow_layers() -> None:
    import app.trading.fast_loop_runner as module

    text = open(module.__file__, encoding="utf-8").read()  # type: ignore[arg-type]
    for forbidden in (
        "app.ontology",
        "app.graph",
        "app.models.temporal_hetero_gnn",
        "app.cost",
        "app.risk",
        "RealtimeMarketDataStore",
    ):
        assert f"import {forbidden}" not in text, forbidden
        assert f"from {forbidden}" not in text, forbidden


def test_latency_summary_reports_percentiles_and_outcomes() -> None:
    runner, _, latency = _runner()
    runner.adopt(_plan())
    runner.on_tick(_tick(70_000.0), now=NOW)
    runner.on_tick(_tick(80_000.0), now=NOW)
    summary = latency.summary()
    assert summary["sample_count"] == 2
    assert set(summary["outcomes"]) >= {"SUBMIT_ENTRY"}
    assert summary["total_ms"]["count"] >= 1


def test_a_missing_stage_reads_as_missing_rather_than_zero() -> None:
    latency = ExecutionLatencyRecorder()
    span = latency.begin("000660", tick_event_time=NOW, received_time=NOW)
    latency.finish(span, outcome="NO_ACTION")
    payload = latency.recent(1)[0]
    assert payload["decision_latency_ms"] is None
    assert payload["total_ms"] is None

"""Drives the strategy fast loop from live ticks, straight to the guard and the broker.

The path this module owns
--------------------------
::

    websocket tick -> StrategyFastExecutor.on_tick -> ExecutionGuard -> broker

Four stages, no database read, no model inference, no ontology, no risk recomputation.
The tick arrives as a parsed object from :mod:`app.data.kis_realtime`'s ``tick_observer``
hook — deliberately *before* persistence, because reading the tick back out of SQLite
would put the write, the read and the contention between them inside every decision.

What the runner adds over the executor
--------------------------------------
The executor is a pure state machine over one plan. The runner is the part that has to
exist in the real system: it owns the executor registry, converts the executor's
:class:`~app.trading.strategy_fast_executor.ExecutionRequest` into a broker order, runs it
past the :class:`~app.execution.execution_guard.ExecutionGuard`, and records the latency
of every stage so the "no heavy work in the fast path" claim stays measurable.

It makes no investment decision of its own. Its only arithmetic is the broker clip the
guard returns.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from app.execution.execution_guard import ExecutionGuard, GuardOrder
from app.monitoring.execution_latency import (
    ExecutionLatencyRecorder,
    default_latency_recorder,
)
from app.trading.strategy_fast_executor import (
    ExecutionRequest,
    FastLoopState,
    StrategyFastExecutor,
    TickEvent,
)
from app.trading.trade_plan import TradePlan

__all__ = ["FastLoopRunner", "FastLoopResult"]


@dataclass(frozen=True)
class FastLoopResult:
    """What one tick produced. ``None`` request means the tick changed nothing."""

    request: ExecutionRequest | None = None
    submitted: bool = False
    blocked_reasons: tuple[str, ...] = ()
    permitted_quantity: int = 0
    span_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.as_dict() if self.request else None,
            "submitted": self.submitted,
            "blocked_reasons": list(self.blocked_reasons),
            "permitted_quantity": self.permitted_quantity,
            "span_id": self.span_id,
        }


def _aware(moment: datetime) -> datetime:
    return (
        moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    ).astimezone(timezone.utc)


class FastLoopRunner:
    """Registry of per-plan executors, plus the tick -> guard -> submit path."""

    def __init__(
        self,
        *,
        guard: ExecutionGuard,
        submit: Callable[[Any, ExecutionRequest], Any],
        order_factory: Callable[[TradePlan, ExecutionRequest], Any],
        orderable_cash_provider: Callable[[str], float | None] | None = None,
        sellable_quantity_provider: Callable[[str], int | None] | None = None,
        latency: ExecutionLatencyRecorder | None = None,
        on_state_change: Callable[[str, FastLoopState], None] | None = None,
    ) -> None:
        self._guard = guard
        self._submit = submit
        self._order_factory = order_factory
        self._cash = orderable_cash_provider
        self._sellable = sellable_quantity_provider
        self._latency = latency or default_latency_recorder()
        self._on_state_change = on_state_change
        self._lock = threading.RLock()
        self._executors: dict[str, StrategyFastExecutor] = {}

    # ------------------------------------------------------------------ #
    # registry
    # ------------------------------------------------------------------ #
    def adopt(
        self,
        plan: TradePlan,
        *,
        entry_trigger: Any | None = None,
        exit_trigger: Any | None = None,
    ) -> StrategyFastExecutor:
        """Take ownership of a plan. Replacing an existing plan for the symbol is
        refused while that plan owns a position — a second executor on one holding would
        produce two exits."""
        symbol = str(plan.symbol).upper()
        with self._lock:
            existing = self._executors.get(symbol)
            if existing is not None and existing.state not in {
                FastLoopState.CLOSED,
                FastLoopState.WAIT_ENTRY,
            }:
                return existing
            executor = StrategyFastExecutor(
                plan, entry_trigger=entry_trigger, exit_trigger=exit_trigger
            )
            self._executors[symbol] = executor
            return executor

    def release(self, symbol: str) -> None:
        with self._lock:
            self._executors.pop(str(symbol).upper(), None)

    def executor_for(self, symbol: str) -> StrategyFastExecutor | None:
        with self._lock:
            return self._executors.get(str(symbol).upper())

    def active_symbols(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                symbol
                for symbol, executor in self._executors.items()
                if executor.state is not FastLoopState.CLOSED
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            executors = list(self._executors.values())
        return {
            "active_count": sum(
                1 for item in executors if item.state is not FastLoopState.CLOSED
            ),
            "executors": [item.snapshot() for item in executors],
            "latency": self._latency.summary(),
        }

    # ------------------------------------------------------------------ #
    # the fast path
    # ------------------------------------------------------------------ #
    def on_ticks(self, ticks: Iterable[Any], orderbooks: Iterable[Any] = ()) -> None:
        """The ``tick_observer`` entry point. Never raises into the feed thread."""
        books = {
            str(getattr(item, "symbol", "")).upper(): item for item in (orderbooks or ())
        }
        for raw in ticks or ():
            symbol = str(getattr(raw, "symbol", "")).upper()
            if not symbol:
                continue
            with self._lock:
                executor = self._executors.get(symbol)
            if executor is None or executor.state is FastLoopState.CLOSED:
                continue
            try:
                self.on_tick(_to_tick_event(raw, books.get(symbol)))
            except Exception:  # noqa: BLE001 - one symbol must not stop the feed.
                continue

    def on_tick(self, tick: TickEvent, *, now: datetime | None = None) -> FastLoopResult:
        """One tick through the whole path. This is the measured critical section."""
        symbol = str(tick.symbol).upper()
        with self._lock:
            executor = self._executors.get(symbol)
        if executor is None:
            return FastLoopResult()

        moment = _aware(now or datetime.now(timezone.utc))
        span = self._latency.begin(
            symbol,
            plan_id=executor.plan.plan_id,
            tick_event_time=tick.event_time,
            received_time=tick.received_time or moment,
        )

        request = executor.on_tick(tick, now=moment)
        # Stamped from the wall clock, NOT from ``moment``: a latency is elapsed real
        # time, and marking it with an injected clock would report 0.0 for every span
        # whenever a caller supplies ``now``. ``moment`` governs decision logic only.
        # The corollary for callers: a span is only meaningful if the tick's
        # ``received_time`` came from the same real clock.
        self._latency.mark(span, "strategy_decision")
        if request is None:
            self._latency.finish(span, outcome="NO_ACTION")
            return FastLoopResult(span_id=span.span_id)

        if self._on_state_change is not None:
            try:
                self._on_state_change(symbol, executor.state)
            except Exception:  # noqa: BLE001 - telemetry must not break execution.
                pass

        if request.action == "CANCEL_ENTRY":
            self._latency.mark(span, "execution_guard")
            outcome = self._dispatch(executor, request, span)
            return outcome

        return self._guarded_submit(executor, request, span)

    # ------------------------------------------------------------------ #
    def _guarded_submit(
        self,
        executor: StrategyFastExecutor,
        request: ExecutionRequest,
        span: Any,
    ) -> FastLoopResult:
        plan = executor.plan
        is_exit = request.action == "SUBMIT_EXIT"
        order = GuardOrder(
            symbol=plan.symbol,
            market=plan.market,
            side=request.side,
            quantity=request.quantity,
            limit_price=request.limit_price,
            direction=plan.direction,
            position_effect="CLOSE" if is_exit else "OPEN",
            execution_product=str(
                dict(plan.order_contract).get("execution_product") or "CASH"
            ),
        )
        decision = self._guard.evaluate(
            order,
            plan=plan,
            orderable_cash=self._lookup(self._cash, plan.symbol),
            sellable_quantity=self._lookup(self._sellable, plan.symbol),
            now=request.decided_at,
        )
        self._latency.mark(span, "execution_guard")
        if not decision.allowed:
            self._latency.finish(
                span,
                outcome="GUARD_BLOCKED",
                reason=",".join(decision.reason_codes),
            )
            if not is_exit:
                executor.on_entry_rejected("GUARD_BLOCKED")
            return FastLoopResult(
                request=request,
                blocked_reasons=decision.reason_codes,
                span_id=span.span_id,
            )

        if decision.permitted_quantity < request.quantity:
            # The broker will not take the full size. A clip, applied to the ORDER; the
            # plan's own quantity stays as elected unless the clip is durable.
            request = ExecutionRequest(
                **{
                    **request.__dict__,
                    "quantity": decision.permitted_quantity,
                    "reason": f"{request.reason}+BROKER_CLIP",
                }
            )
        return self._dispatch(executor, request, span)

    def _dispatch(
        self,
        executor: StrategyFastExecutor,
        request: ExecutionRequest,
        span: Any,
    ) -> FastLoopResult:
        try:
            self._submit(self._order_factory(executor.plan, request), request)
        except Exception as exc:  # noqa: BLE001 - a broker error is not a crash.
            self._latency.finish(
                span, outcome="SUBMIT_FAILED", reason=f"{type(exc).__name__}: {exc}"
            )
            if request.action == "SUBMIT_ENTRY":
                executor.on_entry_rejected(f"SUBMIT_FAILED:{type(exc).__name__}")
            return FastLoopResult(
                request=request,
                blocked_reasons=(f"SUBMIT_FAILED:{type(exc).__name__}",),
                span_id=span.span_id,
            )
        self._latency.mark(span, "broker_submitted")
        self._latency.finish(span, outcome=request.action, reason=request.reason)
        return FastLoopResult(
            request=request,
            submitted=True,
            permitted_quantity=request.quantity,
            span_id=span.span_id,
        )

    @staticmethod
    def _lookup(provider: Callable[[str], Any] | None, symbol: str) -> Any | None:
        if provider is None:
            return None
        try:
            return provider(symbol)
        except Exception:  # noqa: BLE001 - the guard treats absence as absence.
            return None

    # ------------------------------------------------------------------ #
    # fill callbacks
    # ------------------------------------------------------------------ #
    def on_entry_fill(
        self, symbol: str, price: float, quantity: int, *, now: datetime | None = None
    ) -> None:
        executor = self.executor_for(symbol)
        if executor is not None:
            executor.on_entry_fill(price, quantity, now=now)

    def on_exit_fill(self, symbol: str, *, now: datetime | None = None) -> None:
        executor = self.executor_for(symbol)
        if executor is not None:
            executor.on_exit_fill(now=now)

    def force_exit_all(self, reason: str) -> tuple[FastLoopResult, ...]:
        """Emergency flatten. Never blocked by anything a plan says."""
        results: list[FastLoopResult] = []
        with self._lock:
            executors = list(self._executors.items())
        for symbol, executor in executors:
            request = executor.force_exit(reason)
            if request is None:
                continue
            span = self._latency.begin(symbol, plan_id=executor.plan.plan_id)
            self._latency.mark(span, "strategy_decision")
            results.append(self._guarded_submit(executor, request, span))
        return tuple(results)


def _to_tick_event(raw: Any, orderbook: Any | None) -> TickEvent:
    """Adapt a ``RealtimeTradeTick`` (and its book, when present) to a TickEvent."""
    return TickEvent(
        symbol=str(getattr(raw, "symbol", "")).upper(),
        price=float(getattr(raw, "price", 0.0) or 0.0),
        event_time=_aware(getattr(raw, "exchange_timestamp", None) or datetime.now(timezone.utc)),
        received_time=(
            _aware(getattr(raw, "received_at", None))
            if getattr(raw, "received_at", None) is not None
            else None
        ),
        best_bid=(
            float(getattr(orderbook, "best_bid", 0.0) or 0.0) if orderbook else None
        ),
        best_ask=(
            float(getattr(orderbook, "best_ask", 0.0) or 0.0) if orderbook else None
        ),
        volume=float(getattr(raw, "volume", 0.0) or 0.0),
    )

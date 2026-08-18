"""Stage timestamps from tick receipt to broker submission.

What this measures and why it is not optional
----------------------------------------------
The refactor's claim is that the fast path contains no ontology, no GNN, no portfolio risk
and no database work. A claim like that decays silently: someone adds one lookup, it costs
40ms, and nothing fails — the orders just arrive later than the decision that produced
them. This module makes the claim measurable, and
``tests/test_fast_path_latency.py`` makes it enforced.

Five stages, four intervals::

    tick_event -> tick_received -> strategy_decision -> guard -> submitted
                 (feed lag)      (decision latency)  (guard) (broker)

``decision_latency_ms`` is the one the constraint is about: the time from having the tick
in hand to knowing what to do with it. It should be sub-millisecond, because at that point
the only work left is comparing a price against levels the plan already fixed.

Every span carries the stages it actually recorded, so a missing stage reads as missing
rather than as zero.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

__all__ = [
    "FAST_PATH_STAGES",
    "ExecutionLatencyRecorder",
    "LatencySpan",
    "default_latency_recorder",
    "reset_default_latency_recorder",
]

#: In order. A span missing an earlier stage cannot compute the intervals that need it.
FAST_PATH_STAGES: tuple[str, ...] = (
    "tick_event",
    "tick_received",
    "strategy_decision",
    "execution_guard",
    "broker_submitted",
)

#: Ring size. Enough to characterise a session's distribution without unbounded growth on
#: a process that runs for days.
DEFAULT_CAPACITY = 2048


def _aware(moment: datetime) -> datetime:
    return (
        moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    ).astimezone(timezone.utc)


@dataclass
class LatencySpan:
    """One tick's journey. Mutable while open, read-only in practice once submitted."""

    span_id: str
    symbol: str
    plan_id: str | None = None
    stages: dict[str, datetime] = field(default_factory=dict)
    reason: str | None = None
    outcome: str | None = None

    def mark(self, stage: str, moment: datetime | None = None) -> "LatencySpan":
        if stage not in FAST_PATH_STAGES:
            raise ValueError(f"unknown fast-path stage {stage!r}")
        self.stages[stage] = _aware(moment or datetime.now(timezone.utc))
        return self

    def _delta_ms(self, start: str, end: str) -> float | None:
        first, second = self.stages.get(start), self.stages.get(end)
        if first is None or second is None:
            return None
        return round((second - first).total_seconds() * 1000.0, 4)

    @property
    def feed_lag_ms(self) -> float | None:
        return self._delta_ms("tick_event", "tick_received")

    @property
    def decision_latency_ms(self) -> float | None:
        """Tick in hand to decision made. The number the fast-path constraint is about."""
        return self._delta_ms("tick_received", "strategy_decision")

    @property
    def guard_latency_ms(self) -> float | None:
        return self._delta_ms("strategy_decision", "execution_guard")

    @property
    def submit_latency_ms(self) -> float | None:
        return self._delta_ms("execution_guard", "broker_submitted")

    @property
    def total_ms(self) -> float | None:
        return self._delta_ms("tick_received", "broker_submitted")

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "symbol": self.symbol,
            "plan_id": self.plan_id,
            "reason": self.reason,
            "outcome": self.outcome,
            "stages": {
                name: _aware(value).isoformat() for name, value in self.stages.items()
            },
            "feed_lag_ms": self.feed_lag_ms,
            "decision_latency_ms": self.decision_latency_ms,
            "guard_latency_ms": self.guard_latency_ms,
            "submit_latency_ms": self.submit_latency_ms,
            "total_ms": self.total_ms,
        }


class ExecutionLatencyRecorder:
    """Bounded ring of recent spans, plus percentile summaries."""

    def __init__(self, *, capacity: int = DEFAULT_CAPACITY) -> None:
        self._lock = threading.RLock()
        self._spans: deque[LatencySpan] = deque(maxlen=max(16, int(capacity)))
        self._open: dict[str, LatencySpan] = {}

    def begin(
        self,
        symbol: str,
        *,
        span_id: str | None = None,
        plan_id: str | None = None,
        tick_event_time: datetime | None = None,
        received_time: datetime | None = None,
    ) -> LatencySpan:
        from uuid import uuid4

        identifier = span_id or f"span-{uuid4().hex[:12]}"
        span = LatencySpan(span_id=identifier, symbol=str(symbol), plan_id=plan_id)
        if tick_event_time is not None:
            span.mark("tick_event", tick_event_time)
        span.mark("tick_received", received_time)
        with self._lock:
            self._open[identifier] = span
        return span

    def mark(
        self, span: LatencySpan | str, stage: str, moment: datetime | None = None
    ) -> LatencySpan | None:
        target = self._resolve(span)
        if target is None:
            return None
        target.mark(stage, moment)
        return target

    def finish(
        self,
        span: LatencySpan | str,
        *,
        outcome: str,
        reason: str | None = None,
    ) -> LatencySpan | None:
        target = self._resolve(span)
        if target is None:
            return None
        target.outcome = outcome
        target.reason = reason
        with self._lock:
            self._open.pop(target.span_id, None)
            self._spans.append(target)
        return target

    def _resolve(self, span: LatencySpan | str) -> LatencySpan | None:
        if isinstance(span, LatencySpan):
            return span
        with self._lock:
            return self._open.get(str(span))

    # ------------------------------------------------------------------ #
    def recent(self, limit: int = 50) -> tuple[dict[str, Any], ...]:
        with self._lock:
            items = list(self._spans)[-max(1, int(limit)) :]
        return tuple(span.as_dict() for span in reversed(items))

    def summary(self) -> dict[str, Any]:
        with self._lock:
            spans = list(self._spans)
        if not spans:
            return {"sample_count": 0, "stages": list(FAST_PATH_STAGES)}
        return {
            "sample_count": len(spans),
            "stages": list(FAST_PATH_STAGES),
            "decision_latency_ms": _percentiles(
                [span.decision_latency_ms for span in spans]
            ),
            "guard_latency_ms": _percentiles([span.guard_latency_ms for span in spans]),
            "submit_latency_ms": _percentiles([span.submit_latency_ms for span in spans]),
            "total_ms": _percentiles([span.total_ms for span in spans]),
            "feed_lag_ms": _percentiles([span.feed_lag_ms for span in spans]),
            "outcomes": _counts(span.outcome for span in spans),
        }

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()
            self._open.clear()


def _percentiles(values: Iterable[float | None]) -> dict[str, float] | None:
    usable = sorted(value for value in values if value is not None)
    if not usable:
        return None

    def at(fraction: float) -> float:
        index = min(len(usable) - 1, max(0, int(round(fraction * (len(usable) - 1)))))
        return usable[index]

    return {
        "count": float(len(usable)),
        "p50": at(0.50),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": usable[-1],
    }


def _counts(values: Iterable[str | None]) -> dict[str, int]:
    tally: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        tally[key] = tally.get(key, 0) + 1
    return dict(sorted(tally.items()))


_default_recorder: ExecutionLatencyRecorder | None = None
_recorder_lock = threading.Lock()


def default_latency_recorder() -> ExecutionLatencyRecorder:
    global _default_recorder
    with _recorder_lock:
        if _default_recorder is None:
            _default_recorder = ExecutionLatencyRecorder()
        return _default_recorder


def reset_default_latency_recorder() -> None:
    """Test hook. Never called from the trading path."""
    global _default_recorder
    with _recorder_lock:
        _default_recorder = None

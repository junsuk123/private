"""Demand-driven minute-bar preparation for live trading candidates.

The coordinator in this module deliberately owns *preparation*, not trade
authority.  Historical bars can make rolling features computable, but a fresh
tradeable quote/book, RiskManager and FinalTradeGate remain mandatory downstream.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import itertools
import math
import os
from queue import PriorityQueue
import random
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


class SymbolReadiness(str, Enum):
    DISCOVERED = "DISCOVERED"
    MONITORING = "MONITORING"
    CANDIDATE = "CANDIDATE"
    DATA_REQUIREMENTS_RESOLVED = "DATA_REQUIREMENTS_RESOLVED"
    CACHE_CHECK = "CACHE_CHECK"
    BACKFILLING = "BACKFILLING"
    DATA_READY = "DATA_READY"
    FEATURE_READY = "FEATURE_READY"
    STRATEGY_READY = "STRATEGY_READY"
    MODEL_READY = "MODEL_READY"
    TRADE_READY = "TRADE_READY"
    INELIGIBLE = "INELIGIBLE"
    STALE = "STALE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class HistoricalDependency:
    component: str
    timeframe_minutes: int
    minimum_observations: int
    preferred_observations: int | None = None
    required_fields: tuple[str, ...] = ("open", "high", "low", "close", "volume")
    market_constraints: tuple[str, ...] = ()

    def applies_to(self, market: str) -> bool:
        return not self.market_constraints or market.upper() in self.market_constraints


class DependencyProvider(Protocol):
    def historical_dependencies(self, *, market: str) -> Iterable[HistoricalDependency]: ...


@dataclass(frozen=True)
class ResolvedHistoryRequirement:
    symbol: str
    market: str
    timeframe_minutes: int
    minimum_observations: int
    preferred_observations: int
    required_fields: tuple[str, ...]
    components: tuple[str, ...]
    resolved_at: datetime

    @property
    def key(self) -> tuple[str, str, int]:
        # Consumers of overlapping ranges for the same symbol/timeframe share one
        # preparation. RequirementResolver already aggregates every active consumer,
        # so encoding a lookback count here would recreate duplicate fetches whenever
        # a regime change adjusts that aggregate while work is in flight.
        return (self.market, self.symbol, self.timeframe_minutes)


class RequirementResolver:
    """Aggregate component metadata; no universal lookback table is maintained here."""

    def __init__(self, strategy_registry: Any, providers: Iterable[DependencyProvider] = ()) -> None:
        self.strategy_registry = strategy_registry
        self.providers = tuple(providers)

    def resolve(
        self,
        symbol: str,
        market: str,
        *,
        applicable_strategy_ids: Iterable[str],
        include_model_dependencies: bool = True,
    ) -> ResolvedHistoryRequirement:
        dependencies: list[HistoricalDependency] = []
        for strategy_id in dict.fromkeys(str(item).strip().lower() for item in applicable_strategy_ids):
            if not strategy_id:
                continue
            try:
                spec = self.strategy_registry.require(strategy_id)
            except (KeyError, ValueError):
                continue
            if not spec.permits_market(market):
                continue
            count = int(getattr(spec, "minimum_history_bars", 0) or 0)
            if count:
                dependencies.append(
                    HistoricalDependency(
                        component=f"strategy:{strategy_id}",
                        timeframe_minutes=1,
                        minimum_observations=count,
                    )
                )
        if include_model_dependencies:
            for provider in self.providers:
                dependencies.extend(
                    item for item in provider.historical_dependencies(market=market) if item.applies_to(market)
                )
        if not dependencies:
            # A candidate with only tick-driven strategies needs no completed-bar
            # warmup.  One observation represents the live anchor, not a global floor.
            dependencies.append(HistoricalDependency("live_anchor", 1, 1))
        timeframe = min(max(1, item.timeframe_minutes) for item in dependencies)
        normalized = [
            math.ceil(item.minimum_observations * item.timeframe_minutes / timeframe)
            for item in dependencies
        ]
        preferred = [
            math.ceil((item.preferred_observations or item.minimum_observations) * item.timeframe_minutes / timeframe)
            for item in dependencies
        ]
        return ResolvedHistoryRequirement(
            symbol=str(symbol).upper().strip(),
            market=str(market).upper().strip(),
            timeframe_minutes=timeframe,
            minimum_observations=max(normalized),
            preferred_observations=max(preferred),
            required_fields=tuple(sorted({field for item in dependencies for field in item.required_fields})),
            components=tuple(item.component for item in dependencies),
            resolved_at=datetime.now(timezone.utc),
        )


@dataclass(frozen=True)
class MissingRange:
    start: datetime
    end: datetime
    expected_observations: int


def detect_missing_ranges(
    requirement: ResolvedHistoryRequirement,
    bars: Sequence[Any],
    *,
    as_of: datetime,
    is_expected_bar: Callable[[datetime, str], bool] | None = None,
) -> tuple[MissingRange, ...]:
    """Return exact expected intervals absent from cache.

    ``is_expected_bar`` is the exchange-calendar/session authority.  Closed
    minutes are skipped, so holidays and breaks are not misclassified as gaps.
    """

    step = timedelta(minutes=requirement.timeframe_minutes)
    end = as_of.astimezone(timezone.utc).replace(second=0, microsecond=0)
    # Walk backwards until the required number of actual session observations is
    # represented. The safety bound avoids an invalid calendar looping forever.
    expected: list[datetime] = []
    cursor = end - step
    bound = max(requirement.minimum_observations * 20, 2_000)
    while len(expected) < requirement.minimum_observations and bound > 0:
        if is_expected_bar is None or is_expected_bar(cursor, requirement.market):
            expected.append(cursor)
        cursor -= step
        bound -= 1
    expected.reverse()
    available = {
        getattr(bar, "minute_start", None).astimezone(timezone.utc).replace(second=0, microsecond=0)
        for bar in bars
        if isinstance(getattr(bar, "minute_start", None), datetime)
    }
    missing = [stamp for stamp in expected if stamp not in available]
    if not missing:
        return ()
    ranges: list[MissingRange] = []
    start = previous = missing[0]
    count = 1
    for stamp in missing[1:]:
        if stamp - previous == step:
            previous = stamp
            count += 1
            continue
        ranges.append(MissingRange(start, previous + step, count))
        start = previous = stamp
        count = 1
    ranges.append(MissingRange(start, previous + step, count))
    return tuple(ranges)


@dataclass
class WarmupMetrics:
    warmup_requests_total: int = 0
    warmup_requests_deduplicated: int = 0
    warmup_requests_suppressed: int = 0
    historical_bars_downloaded: int = 0
    historical_bars_reused_from_cache: int = 0
    missing_bars_backfilled: int = 0
    provider_throttle_events: int = 0
    incremental_backfill_count: int = 0
    full_rewarm_count: int = 0
    websocket_reconnect_count: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    total_latency_seconds: float = 0.0


@dataclass
class _RetryState:
    no_progress_streak: int = 0
    next_attempt_monotonic: float = 0.0
    latest_observation: datetime | None = None
    suppression_logged: bool = False


@dataclass
class _SymbolState:
    state: SymbolReadiness
    requirement: ResolvedHistoryRequirement | None = None
    reasons: tuple[str, ...] = ()
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    candidate_since: datetime | None = None
    data_ready_at: datetime | None = None


class ReadinessManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._states: dict[tuple[str, str], _SymbolState] = {}

    def transition(
        self,
        market: str,
        symbol: str,
        state: SymbolReadiness,
        *,
        requirement: ResolvedHistoryRequirement | None = None,
        reasons: Iterable[str] = (),
    ) -> None:
        key = (market.upper(), symbol.upper())
        now = datetime.now(timezone.utc)
        with self._lock:
            old = self._states.get(key)
            candidate_since = old.candidate_since if old else None
            if state is SymbolReadiness.CANDIDATE and candidate_since is None:
                candidate_since = now
            self._states[key] = _SymbolState(
                state=state,
                requirement=requirement or (old.requirement if old else None),
                reasons=tuple(dict.fromkeys(str(reason) for reason in reasons if reason)),
                updated_at=now,
                candidate_since=candidate_since,
                data_ready_at=now if state is SymbolReadiness.DATA_READY else (old.data_ready_at if old else None),
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            rows = {
                f"{market}:{symbol}": {
                    "market": market,
                    "symbol": symbol,
                    "state": value.state.value,
                    "reasons": list(value.reasons),
                    "updated_at": value.updated_at.isoformat(),
                    "minimum_observations": (
                        value.requirement.minimum_observations if value.requirement else None
                    ),
                    "components": list(value.requirement.components) if value.requirement else [],
                }
                for (market, symbol), value in self._states.items()
            }
        counts: dict[str, int] = {}
        for row in rows.values():
            counts[row["state"]] = counts.get(row["state"], 0) + 1
        return {"counts": counts, "symbols": rows}


class AdaptiveConcurrencyController:
    """AIMD controller bounded by provider policy and detected machine capacity."""

    def __init__(self, hard_limit: int | None = None) -> None:
        cpu = max(1, os.cpu_count() or 1)
        provider_limit = hard_limit or _positive_int_env("MINUTE_WARMUP_PROVIDER_MAX_CONCURRENCY", cpu)
        self.maximum = max(1, min(provider_limit, cpu * 2))
        self.minimum = 1
        self.current = min(self.maximum, max(1, int(math.sqrt(cpu))))
        self._successes = 0
        self._lock = threading.Lock()

    def record(self, *, latency_seconds: float, throttled: bool, failed: bool) -> None:
        with self._lock:
            if throttled or failed or latency_seconds > _float_env("MINUTE_WARMUP_SLOW_REQUEST_SEC", 3.0):
                self.current = max(self.minimum, self.current // 2)
                self._successes = 0
                return
            self._successes += 1
            if self._successes >= max(2, self.current * 2) and self.current < self.maximum:
                self.current += 1
                self._successes = 0

    def refresh_resource_pressure(self) -> None:
        """Reduce capacity under observed CPU or memory pressure.

        This is intentionally fail-open to the existing bounded level when the
        host does not expose Linux pressure files (for example on macOS tests).
        """
        pressured = False
        try:
            pressured = os.getloadavg()[0] / max(1, os.cpu_count() or 1) >= 0.90
        except (AttributeError, OSError):
            pass
        try:
            values: dict[str, float] = {}
            with open("/proc/meminfo", encoding="ascii") as handle:
                for line in handle:
                    key, _, raw = line.partition(":")
                    if key in {"MemTotal", "MemAvailable"}:
                        values[key] = float(raw.strip().split()[0])
            if values.get("MemTotal", 0) > 0:
                pressured = pressured or values.get("MemAvailable", 0) / values["MemTotal"] < 0.10
        except (OSError, ValueError):
            pass
        if pressured:
            with self._lock:
                self.current = max(self.minimum, self.current // 2)
                self._successes = 0


FetchBars = Callable[[ResolvedHistoryRequirement, tuple[MissingRange, ...]], Sequence[Any]]


class HistoricalDataCoordinator:
    """Priority, coalescing and failure-isolated historical preparation."""

    def __init__(
        self,
        *,
        repository: Any,
        fetch_bars: FetchBars,
        resolver: RequirementResolver,
        readiness: ReadinessManager | None = None,
        concurrency: AdaptiveConcurrencyController | None = None,
        expected_bar: Callable[[datetime, str], bool] | None = None,
        event_sink: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.repository = repository
        self.fetch_bars = fetch_bars
        self.resolver = resolver
        self.readiness = readiness or ReadinessManager()
        self.concurrency = concurrency or AdaptiveConcurrencyController()
        self.expected_bar = expected_bar
        self.event_sink = event_sink
        self.metrics = WarmupMetrics()
        self._metrics_lock = threading.Lock()
        self._condition = threading.Condition()
        self._queue: PriorityQueue[tuple[int, int, tuple[Any, ...], ResolvedHistoryRequirement, Future[Any]]] = PriorityQueue()
        self._sequence = itertools.count()
        self._inflight: dict[tuple[Any, ...], Future[Any]] = {}
        self._retry_state: dict[tuple[Any, ...], _RetryState] = {}
        # Candidate admission can prove a cache hit synchronously so a ready symbol
        # is not delayed by the coordinator queue. Remember the observed cache
        # generation to publish that readiness exactly once rather than once per
        # one-second trading cycle.
        self._observed_ready_cache: dict[tuple[Any, ...], tuple[int, datetime | None, int]] = {}
        self._active = 0
        self._stopped = False
        self._workers = [
            threading.Thread(target=self._worker, name=f"minute-warmup-{index}", daemon=True)
            for index in range(self.concurrency.maximum)
        ]
        for worker in self._workers:
            worker.start()

    def request(
        self,
        symbol: str,
        market: str,
        *,
        applicable_strategy_ids: Iterable[str],
        priority: int = 50,
    ) -> Future[Any]:
        requirement = self.resolver.resolve(
            symbol,
            market,
            applicable_strategy_ids=applicable_strategy_ids,
        )
        key = requirement.key
        self.readiness.transition(market, symbol, SymbolReadiness.DATA_REQUIREMENTS_RESOLVED, requirement=requirement)
        with self._condition:
            current = self._inflight.get(key)
            if current is not None and not current.done():
                self.metrics.warmup_requests_deduplicated += 1
                self._emit({"event": "request_coalesced", "symbol": requirement.symbol, "market": requirement.market, "components": requirement.components})
                return current
            retry = self._retry_state.get(key)
            now_monotonic = time.monotonic()
            if retry is not None and retry.next_attempt_monotonic > now_monotonic:
                self.metrics.warmup_requests_suppressed += 1
                remaining = retry.next_attempt_monotonic - now_monotonic
                self.readiness.transition(
                    market,
                    symbol,
                    SymbolReadiness.STALE,
                    requirement=requirement,
                    reasons=(f"BACKFILL_RETRY_COOLDOWN_SECONDS:{math.ceil(remaining)}",),
                )
                future = Future()
                future.set_result(
                    {
                        "symbol": requirement.symbol,
                        "market": requirement.market,
                        "ready": False,
                        "suppressed": True,
                        "retry_after_seconds": remaining,
                    }
                )
                if not retry.suppression_logged:
                    retry.suppression_logged = True
                    self._emit(
                        {
                            "event": "warmup_retry_suppressed",
                            "symbol": requirement.symbol,
                            "market": requirement.market,
                            "retry_after_seconds": round(remaining, 3),
                            "no_progress_streak": retry.no_progress_streak,
                        }
                    )
                return future
            self.metrics.warmup_requests_total += 1
            future: Future[Any] = Future()
            self._inflight[key] = future
            self._queue.put((int(priority), next(self._sequence), key, requirement, future))
            self._emit({"event": "warmup_requested", "symbol": requirement.symbol, "market": requirement.market, "priority": int(priority), "minimum_observations": requirement.minimum_observations, "components": requirement.components})
            self._condition.notify_all()
            return future

    def cancel_if_irrelevant(self, symbol: str, market: str) -> bool:
        prefix = (market.upper(), symbol.upper())
        with self._condition:
            matches = [future for key, future in self._inflight.items() if key[:2] == prefix and not future.running()]
            for future in matches:
                future.cancel()
            for key in tuple(self._retry_state):
                if key[:2] == prefix:
                    self._retry_state.pop(key, None)
            if matches:
                self.readiness.transition(market, symbol, SymbolReadiness.MONITORING, reasons=("CANDIDATE_NO_LONGER_RELEVANT",))
            return bool(matches)

    def observe_ready_cache(
        self,
        requirement: ResolvedHistoryRequirement,
        bars: Sequence[Any],
    ) -> None:
        """Publish a synchronous cache hit without scheduling provider work.

        The live candidate filter already has the reconciled rows in hand. Enqueuing
        another cache read merely to populate readiness telemetry would add latency
        and inflate request counts, while omitting the transition makes a genuinely
        ready candidate invisible on the status API. The cache-generation signature
        keeps this operation idempotent across rapid engine cycles.
        """
        latest = getattr(bars[-1], "minute_start", None) if bars else None
        signature = (len(bars), latest if isinstance(latest, datetime) else None,
                     requirement.minimum_observations)
        with self._condition:
            if self._observed_ready_cache.get(requirement.key) == signature:
                return
            self._observed_ready_cache[requirement.key] = signature
            self.metrics.historical_bars_reused_from_cache += min(
                len(bars), requirement.preferred_observations
            )
        self.readiness.transition(
            requirement.market,
            requirement.symbol,
            SymbolReadiness.DATA_READY,
            requirement=requirement,
        )
        self._emit(
            {
                "event": "readiness_transition",
                "symbol": requirement.symbol,
                "market": requirement.market,
                "state": SymbolReadiness.DATA_READY.value,
                "available_observations": len(bars),
                "required_observations": requirement.minimum_observations,
                "cache_reused": min(len(bars), requirement.preferred_observations),
                "provider_request": False,
            }
        )

    def reconnect(self, market: str, symbols: Iterable[str], disconnected_at: datetime) -> None:
        # Cache is intentionally untouched. Candidate requests will gap-detect from
        # the disconnect boundary and fetch only missing session minutes.
        with self._metrics_lock:
            self.metrics.websocket_reconnect_count += 1
        for symbol in symbols:
            self.readiness.transition(market, symbol, SymbolReadiness.STALE, reasons=(f"STREAM_GAP_SINCE:{disconnected_at.isoformat()}",))

    def status(self) -> dict[str, Any]:
        with self._condition:
            pending = self._queue.qsize()
            active = self._active
            metrics = dict(vars(self.metrics))
            now_monotonic = time.monotonic()
            retry_cooldowns = {
                f"{key[0]}:{key[1]}:{key[2]}m": {
                    "no_progress_streak": value.no_progress_streak,
                    "retry_after_seconds": round(
                        max(0.0, value.next_attempt_monotonic - now_monotonic), 3
                    ),
                    "latest_observation": (
                        value.latest_observation.isoformat()
                        if value.latest_observation is not None else None
                    ),
                }
                for key, value in self._retry_state.items()
                if value.next_attempt_monotonic > now_monotonic
            }
        completed = max(1, metrics["completed_requests"])
        metrics["cache_hit_ratio"] = (
            metrics["historical_bars_reused_from_cache"]
            / max(1, metrics["historical_bars_reused_from_cache"] + metrics["historical_bars_downloaded"])
        )
        metrics["mean_warmup_latency_seconds"] = metrics["total_latency_seconds"] / completed
        return {
            "global_state": "SYSTEM_OPERATIONAL",
            "active_warmup_tasks": active,
            "pending_warmup_tasks": pending,
            "adaptive_concurrency_level": self.concurrency.current,
            "adaptive_concurrency_maximum": self.concurrency.maximum,
            "backfill_retry_cooldowns": retry_cooldowns,
            "metrics": metrics,
            "readiness": self.readiness.snapshot(),
        }

    def _worker(self) -> None:
        while not self._stopped:
            _priority, _sequence, key, requirement, future = self._queue.get()
            if future.cancelled():
                self._finish_key(key)
                continue
            self.concurrency.refresh_resource_pressure()
            with self._condition:
                while self._active >= self.concurrency.current and not self._stopped:
                    self._condition.wait(timeout=0.25)
                self._active += 1
            started = time.monotonic()
            throttled = False
            failed = False
            result: Mapping[str, Any] | None = None
            try:
                if not future.set_running_or_notify_cancel():
                    continue
                result = self._prepare(requirement)
                self._record_completion(key, result)
                future.set_result(result)
            except Exception as exc:  # noqa: BLE001 - isolate provider failures per symbol.
                failed = True
                self._record_completion(key, None)
                if "429" in str(exc) or "thrott" in str(exc).lower():
                    throttled = True
                self.readiness.transition(requirement.market, requirement.symbol, SymbolReadiness.FAILED, requirement=requirement, reasons=(f"{type(exc).__name__}:{exc}",))
                self._emit({"event": "warmup_failed", "symbol": requirement.symbol, "market": requirement.market, "error_type": type(exc).__name__, "error": str(exc)[:240]})
                if not future.done():
                    future.set_exception(exc)
            finally:
                latency = time.monotonic() - started
                if failed or bool((result or {}).get("provider_requested")):
                    self.concurrency.record(
                        latency_seconds=latency,
                        throttled=throttled,
                        failed=failed,
                    )
                with self._condition:
                    self._active -= 1
                    self.metrics.total_latency_seconds += latency
                    self.metrics.failed_requests += int(failed)
                    self.metrics.completed_requests += int(not failed)
                    self.metrics.provider_throttle_events += int(throttled)
                    self._inflight.pop(key, None)
                    self._condition.notify_all()

    def _finish_key(self, key: tuple[Any, ...]) -> None:
        with self._condition:
            self._inflight.pop(key, None)
            self._condition.notify_all()

    def _record_completion(
        self,
        key: tuple[Any, ...],
        result: Mapping[str, Any] | None,
    ) -> None:
        """Back off only when a completed provider cycle made no useful progress."""
        with self._condition:
            previous = self._retry_state.get(key, _RetryState())
            if result is not None and bool(result.get("ready")):
                self._retry_state.pop(key, None)
                return
            latest = result.get("latest_observation") if result is not None else None
            if isinstance(latest, str):
                try:
                    latest = datetime.fromisoformat(latest.replace("Z", "+00:00"))
                except ValueError:
                    latest = None
            progressed = bool(
                isinstance(latest, datetime)
                and (
                    previous.latest_observation is None
                    or latest > previous.latest_observation
                )
            )
            streak = 0 if progressed else previous.no_progress_streak + 1
            base = max(
                float(key[2]) * 60.0,
                _float_env("MINUTE_WARMUP_RETRY_BASE_SEC", 60.0),
            )
            maximum = max(base, _float_env("MINUTE_WARMUP_RETRY_MAX_SEC", 900.0))
            delay = min(maximum, base * (2 ** min(streak, 8)))
            self._retry_state[key] = _RetryState(
                no_progress_streak=streak,
                next_attempt_monotonic=time.monotonic() + delay,
                latest_observation=latest if isinstance(latest, datetime) else previous.latest_observation,
            )

    def _prepare(self, requirement: ResolvedHistoryRequirement) -> dict[str, Any]:
        self.readiness.transition(requirement.market, requirement.symbol, SymbolReadiness.CACHE_CHECK, requirement=requirement)
        now = datetime.now(timezone.utc)
        bars = tuple(self.repository.bars_for_requirement(requirement, as_of=now))
        reused = min(len(bars), requirement.preferred_observations)
        with self._metrics_lock:
            self.metrics.historical_bars_reused_from_cache += reused
        latest = getattr(bars[-1], "minute_start", None) if bars else None
        maximum_age = max(60.0, _float_env("REALTIME_STRATEGY_HISTORY_MAX_AGE_SEC", 180.0))
        fresh = bool(
            isinstance(latest, datetime)
            and max(0.0, (now - latest).total_seconds()) <= maximum_age
        )
        # Strategies require observations, not a print in every wall-clock minute.
        # KIS omits zero-trade minutes, so scanning internal clock gaps after 64 fresh
        # observations are present creates provider calls that can never fill them.
        if len(bars) >= requirement.minimum_observations and fresh:
            missing: tuple[MissingRange, ...] = ()
        else:
            exact = detect_missing_ranges(
                requirement, bars, as_of=now, is_expected_bar=self.expected_bar
            )
            # KIS returns a bounded page rather than accepting arbitrary disjoint
            # intervals. One envelope obtains every recoverable row in the tail and
            # avoids serially requesting each no-trade minute.
            missing = (
                MissingRange(
                    exact[0].start,
                    exact[-1].end,
                    sum(item.expected_observations for item in exact),
                ),
            ) if exact else ()
        if missing:
            self._emit({"event": "missing_ranges_detected", "symbol": requirement.symbol, "market": requirement.market, "cache_bars": len(bars), "ranges": tuple({"start": item.start.isoformat(), "end": item.end.isoformat(), "expected_observations": item.expected_observations} for item in missing)})
            self.readiness.transition(requirement.market, requirement.symbol, SymbolReadiness.BACKFILLING, requirement=requirement, reasons=tuple(f"MISSING_RANGE:{item.start.isoformat()}:{item.end.isoformat()}" for item in missing))
            downloaded = tuple(self.fetch_bars(requirement, missing))
            if downloaded:
                self.repository.merge_bars(downloaded)
                with self._metrics_lock:
                    self.metrics.historical_bars_downloaded += len(downloaded)
                    self.metrics.missing_bars_backfilled += len(downloaded)
                    self.metrics.incremental_backfill_count += 1
                bars = tuple(self.repository.bars_for_requirement(requirement, as_of=now))
        latest = getattr(bars[-1], "minute_start", None) if bars else None
        fresh = bool(
            isinstance(latest, datetime)
            and max(0.0, (now - latest).total_seconds()) <= maximum_age
        )
        ready = len(bars) >= requirement.minimum_observations and fresh
        state = SymbolReadiness.DATA_READY if ready else SymbolReadiness.STALE
        reasons = () if ready else tuple(
            reason
            for reason in (
                (
                    f"HISTORY_INCOMPLETE:{len(bars)}/{requirement.minimum_observations}"
                    if len(bars) < requirement.minimum_observations else ""
                ),
                "HISTORY_STALE" if not fresh else "",
            )
            if reason
        )
        self.readiness.transition(requirement.market, requirement.symbol, state, requirement=requirement, reasons=reasons)
        self._emit({"event": "readiness_transition", "symbol": requirement.symbol, "market": requirement.market, "state": state.value, "available_observations": len(bars), "required_observations": requirement.minimum_observations, "cache_reused": reused})
        return {
            "symbol": requirement.symbol,
            "market": requirement.market,
            "ready": ready,
            "available_observations": len(bars),
            "required_observations": requirement.minimum_observations,
            "missing_ranges": [vars(item) for item in missing],
            "latest_observation": latest.isoformat() if isinstance(latest, datetime) else None,
            "provider_requested": bool(missing),
        }

    def _emit(self, payload: Mapping[str, Any]) -> None:
        if self.event_sink is None:
            return
        try:
            self.event_sink(dict(payload))
        except Exception:
            # Observability must never become a market-data or order dependency.
            pass


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return max(1, default)


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def retry_with_backoff(call: Callable[[], Sequence[Any]], *, attempts: int = 3) -> Sequence[Any]:
    """Bounded transient retry with jitter; permanent errors are surfaced."""
    delay = 0.15
    for attempt in range(max(1, attempts)):
        try:
            return call()
        except Exception as exc:
            transient = any(token in str(exc).lower() for token in ("429", "timeout", "tempor", "rate", "connection"))
            if not transient or attempt + 1 >= attempts:
                raise
            time.sleep(delay * (0.75 + random.random() * 0.5))
            delay = min(delay * 2.0, 2.0)
    return ()

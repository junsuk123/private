"""Demand-driven historical minute-bar warmup and request coalescing."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Iterable, Sequence


@dataclass(frozen=True)
class HistoricalDependency:
    component: str
    timeframe_minutes: int
    minimum_observations: int
    preferred_observations: int
    required_fields: tuple[str, ...] = ()
    market_constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.timeframe_minutes <= 0:
            raise ValueError("timeframe_minutes must be positive")
        if self.minimum_observations < 0:
            raise ValueError("minimum_observations must be non-negative")
        if self.preferred_observations < self.minimum_observations:
            raise ValueError("preferred_observations must cover the minimum")


@dataclass(frozen=True)
class HistoricalRequirement:
    symbol: str
    market: str
    timeframe_minutes: int
    minimum_observations: int
    preferred_observations: int
    components: tuple[str, ...]
    required_fields: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.market}:{self.symbol}:{self.timeframe_minutes}m"

    @property
    def display_key(self) -> str:
        return f"{self.market}:{self.symbol}"


@dataclass(frozen=True)
class MissingRange:
    start: datetime
    end: datetime


class SymbolReadiness(str, Enum):
    DATA_READY = "DATA_READY"
    WARMING_UP = "WARMING_UP"
    STALE = "STALE"
    FAILED = "FAILED"


class RequirementResolver:
    def __init__(self, strategy_registry: Any, *, providers: Sequence[Any] = ()) -> None:
        self.strategy_registry = strategy_registry
        self.providers = tuple(providers)

    def resolve(
        self,
        symbol: str,
        market: str,
        *,
        applicable_strategy_ids: Sequence[str] = (),
    ) -> HistoricalRequirement:
        normalized_symbol = str(symbol or "").strip().upper()
        normalized_market = str(market or "").strip().upper()
        dependencies: list[HistoricalDependency] = []

        strategy_ids = tuple(applicable_strategy_ids)
        if not strategy_ids and hasattr(self.strategy_registry, "all_specs"):
            strategy_ids = tuple(
                str(spec.strategy_id) for spec in self.strategy_registry.all_specs()
            )
        for strategy_id in strategy_ids:
            try:
                spec = self.strategy_registry.require(strategy_id)
            except (KeyError, AttributeError):
                continue
            permits = getattr(spec, "permits_market", None)
            if callable(permits) and not permits(normalized_market):
                continue
            bars = max(0, int(getattr(spec, "minimum_history_bars", 0) or 0))
            if bars:
                dependencies.append(
                    HistoricalDependency(f"strategy:{strategy_id}", 1, bars, bars)
                )

        for provider in self.providers:
            factory = getattr(provider, "historical_dependencies", None)
            if not callable(factory):
                continue
            for dependency in factory(market=normalized_market) or ():
                constraints = {
                    str(item).strip().upper() for item in dependency.market_constraints
                }
                if constraints and normalized_market not in constraints:
                    continue
                dependencies.append(dependency)

        one_minute = [item for item in dependencies if item.timeframe_minutes == 1]
        minimum = max((item.minimum_observations for item in one_minute), default=1)
        preferred = max(
            minimum,
            max((item.preferred_observations for item in one_minute), default=minimum),
        )
        return HistoricalRequirement(
            symbol=normalized_symbol,
            market=normalized_market,
            timeframe_minutes=1,
            minimum_observations=minimum,
            preferred_observations=preferred,
            components=tuple(dict.fromkeys(item.component for item in one_minute)),
            required_fields=tuple(
                dict.fromkeys(
                    field for item in one_minute for field in item.required_fields
                )
            ),
        )


def _minute(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(second=0, microsecond=0)


def detect_missing_ranges(
    requirement: HistoricalRequirement,
    bars: Iterable[Any],
    *,
    as_of: datetime,
    is_expected_bar: Callable[[datetime, str], bool],
) -> tuple[MissingRange, ...]:
    """Return coalesced gaps in the preferred completed-bar window."""
    step = timedelta(minutes=requirement.timeframe_minutes)
    last_complete = _minute(as_of) - step
    start = last_complete - step * max(0, requirement.preferred_observations - 1)
    present = {
        _minute(stamp)
        for bar in bars
        if (stamp := getattr(bar, "minute_start", None)) is not None
        and _minute(stamp) <= last_complete
    }
    missing: list[datetime] = []
    cursor = start
    while cursor <= last_complete:
        if is_expected_bar(cursor, requirement.market) and cursor not in present:
            missing.append(cursor)
        cursor += step
    if not missing:
        return ()

    ranges: list[MissingRange] = []
    range_start = previous = missing[0]
    for stamp in missing[1:]:
        if stamp - previous != step:
            ranges.append(MissingRange(range_start, previous))
            range_start = stamp
        previous = stamp
    ranges.append(MissingRange(range_start, previous))
    return tuple(ranges)


class AdaptiveConcurrencyController:
    """Small additive-increase/multiplicative-decrease controller."""

    def __init__(self, *, hard_limit: int = 8, initial: int | None = None) -> None:
        self.maximum = max(1, int(hard_limit))
        self.current = max(1, min(self.maximum, int(initial or min(4, self.maximum))))
        self._successes = 0
        self._lock = threading.Lock()

    def record(self, *, latency_seconds: float, throttled: bool, failed: bool) -> None:
        with self._lock:
            if throttled or failed or latency_seconds >= 5.0:
                self.current = max(1, self.current // 2)
                self._successes = 0
                return
            self._successes += 1
            if self._successes >= self.current:
                self.current = min(self.maximum, self.current + 1)
                self._successes = 0


class HistoricalDataCoordinator:
    def __init__(
        self,
        *,
        repository: Any,
        fetch_bars: Callable[[HistoricalRequirement, tuple[MissingRange, ...]], Iterable[Any]],
        resolver: RequirementResolver,
        expected_bar: Callable[[datetime, str], bool],
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        maximum_workers: int = 8,
        retry_cooldown_seconds: float = 60.0,
    ) -> None:
        self.repository = repository
        self.fetch_bars = fetch_bars
        self.resolver = resolver
        self.expected_bar = expected_bar
        self.event_sink = event_sink
        self.control = AdaptiveConcurrencyController(hard_limit=maximum_workers)
        self.retry_cooldown_seconds = max(1.0, float(retry_cooldown_seconds))
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(maximum_workers)),
            thread_name_prefix="minute-warmup",
        )
        self._lock = threading.RLock()
        self._inflight: dict[str, Future] = {}
        self._cooldowns: dict[str, float] = {}
        self._readiness: dict[str, dict[str, Any]] = {}
        self._cache_reuse: dict[str, int] = {}
        self._metrics = {
            "warmup_requests_total": 0,
            "warmup_requests_deduplicated": 0,
            "warmup_requests_suppressed": 0,
            "warmup_failures_total": 0,
            "historical_bars_downloaded": 0,
            "historical_bars_reused_from_cache": 0,
        }

    def request(
        self,
        symbol: str,
        market: str,
        *,
        applicable_strategy_ids: Sequence[str] = (),
        priority: int = 50,
    ) -> Future:
        del priority  # ThreadPoolExecutor has no priority queue; retained in the API contract.
        requirement = self.resolver.resolve(
            symbol, market, applicable_strategy_ids=applicable_strategy_ids
        )
        key = requirement.key
        now_mono = time.monotonic()
        with self._lock:
            existing = self._inflight.get(key)
            if existing is not None and not existing.done():
                self._metrics["warmup_requests_deduplicated"] += 1
                return existing
            if self._cooldowns.get(key, 0.0) > now_mono:
                self._metrics["warmup_requests_suppressed"] += 1
                future: Future = Future()
                future.set_result({"ready": False, "suppressed": True, "key": key})
                return future
            future = Future()
            self._inflight[key] = future
            self._metrics["warmup_requests_total"] += 1
            self._readiness[requirement.display_key] = self._row(
                requirement, SymbolReadiness.WARMING_UP, ("HISTORY_LOADING",), 0
            )
        self._executor.submit(self._execute, future, requirement)
        return future

    def _execute(self, future: Future, requirement: HistoricalRequirement) -> None:
        started = time.monotonic()
        throttled = False
        try:
            bars = tuple(
                self.repository.bars_for_requirement(
                    requirement, as_of=datetime.now(timezone.utc)
                )
            )
            self._record_cache_reuse(requirement.key, len(bars))
            missing = detect_missing_ranges(
                requirement,
                bars,
                as_of=datetime.now(timezone.utc),
                is_expected_bar=self.expected_bar,
            )
            if len(bars) >= requirement.minimum_observations and not missing:
                self.observe_ready_cache(requirement, bars)
                result = {"ready": True, "suppressed": False, "key": requirement.key}
                future.set_result(result)
                return

            downloaded = tuple(self.fetch_bars(requirement, missing) or ())
            if downloaded:
                self.repository.merge_bars(downloaded)
            with self._lock:
                self._metrics["historical_bars_downloaded"] += len(downloaded)
            refreshed = tuple(
                self.repository.bars_for_requirement(
                    requirement, as_of=datetime.now(timezone.utc)
                )
            )
            refreshed_missing = detect_missing_ranges(
                requirement,
                refreshed,
                as_of=datetime.now(timezone.utc),
                is_expected_bar=self.expected_bar,
            )
            ready = len(refreshed) >= requirement.minimum_observations and not refreshed_missing
            if ready:
                self.observe_ready_cache(requirement, refreshed)
            else:
                reasons = (
                    ("HISTORY_STALE",)
                    if len(refreshed) >= requirement.minimum_observations
                    else ("HISTORY_INSUFFICIENT",)
                )
                with self._lock:
                    self._readiness[requirement.display_key] = self._row(
                        requirement, SymbolReadiness.STALE, reasons, len(refreshed)
                    )
                    self._cooldowns[requirement.key] = (
                        time.monotonic() + self.retry_cooldown_seconds
                    )
            future.set_result(
                {"ready": ready, "suppressed": False, "key": requirement.key}
            )
        except BaseException as exc:  # propagate to callers while isolating other symbols.
            with self._lock:
                self._metrics["warmup_failures_total"] += 1
                self._readiness[requirement.display_key] = self._row(
                    requirement,
                    SymbolReadiness.FAILED,
                    (f"HISTORY_FETCH_FAILED:{exc.__class__.__name__}",),
                    0,
                )
                self._cooldowns[requirement.key] = (
                    time.monotonic() + self.retry_cooldown_seconds
                )
            future.set_exception(exc)
        finally:
            self.control.record(
                latency_seconds=time.monotonic() - started,
                throttled=throttled,
                failed=future.exception() is not None if future.done() and not future.cancelled() else False,
            )
            with self._lock:
                if self._inflight.get(requirement.key) is future:
                    self._inflight.pop(requirement.key, None)
            self._emit({
                "key": requirement.key,
                "readiness": self._readiness.get(requirement.display_key),
            })

    def observe_ready_cache(
        self, requirement: HistoricalRequirement, bars: Iterable[Any]
    ) -> None:
        materialized = tuple(bars)
        self._record_cache_reuse(requirement.key, len(materialized))
        with self._lock:
            self._readiness[requirement.display_key] = self._row(
                requirement, SymbolReadiness.DATA_READY, (), len(materialized)
            )
            self._cooldowns.pop(requirement.key, None)

    def reconnect(self, market: str, symbols: Sequence[str], at: datetime) -> None:
        del at
        normalized_market = str(market or "").strip().upper()
        with self._lock:
            for symbol in symbols:
                display_key = f"{normalized_market}:{str(symbol).strip().upper()}"
                row = dict(self._readiness.get(display_key) or {})
                row.update(
                    {
                        "symbol": str(symbol).strip().upper(),
                        "market": normalized_market,
                        "state": SymbolReadiness.STALE.value,
                        "reasons": ["STREAM_RECONNECTED"],
                    }
                )
                self._readiness[display_key] = row
                self._cooldowns.pop(f"{display_key}:1m", None)

    def cancel_if_irrelevant(self, symbol: str, market: str) -> None:
        key = f"{str(market).strip().upper()}:{str(symbol).strip().upper()}:1m"
        with self._lock:
            future = self._inflight.get(key)
            if future is not None and future.cancel():
                self._inflight.pop(key, None)

    def status(self) -> dict[str, Any]:
        now_mono = time.monotonic()
        with self._lock:
            self._cooldowns = {
                key: deadline for key, deadline in self._cooldowns.items() if deadline > now_mono
            }
            rows = {key: dict(value) for key, value in sorted(self._readiness.items())}
            counts: dict[str, int] = {}
            for row in rows.values():
                state = str(row.get("state") or "UNKNOWN")
                counts[state] = counts.get(state, 0) + 1
            return {
                "global_state": "SYSTEM_OPERATIONAL",
                "active_warmup_tasks": sum(
                    1 for item in self._inflight.values() if item.running()
                ),
                "pending_warmup_tasks": sum(
                    1 for item in self._inflight.values() if not item.running() and not item.done()
                ),
                "adaptive_concurrency": self.control.current,
                "metrics": dict(self._metrics),
                "readiness": {"counts": counts, "symbols": rows},
                "backfill_retry_cooldowns": {
                    key: max(0.0, deadline - now_mono)
                    for key, deadline in self._cooldowns.items()
                },
            }

    def _record_cache_reuse(self, key: str, count: int) -> None:
        with self._lock:
            self._cache_reuse[key] = max(0, int(count))
            self._metrics["historical_bars_reused_from_cache"] = sum(
                self._cache_reuse.values()
            )

    @staticmethod
    def _row(
        requirement: HistoricalRequirement,
        state: SymbolReadiness,
        reasons: Sequence[str],
        observations: int,
    ) -> dict[str, Any]:
        return {
            "symbol": requirement.symbol,
            "market": requirement.market,
            "timeframe_minutes": requirement.timeframe_minutes,
            "state": state.value,
            "reasons": list(reasons),
            "observations": int(observations),
            "minimum_observations": requirement.minimum_observations,
            "preferred_observations": requirement.preferred_observations,
            "components": list(requirement.components),
        }

    def _emit(self, payload: dict[str, Any]) -> None:
        if self.event_sink is None:
            return
        try:
            self.event_sink(payload)
        except Exception:
            pass

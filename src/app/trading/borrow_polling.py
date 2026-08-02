"""Keeps the borrow journal populated, so a short signal has a locate to check.

Why this exists
---------------
`app.trading.borrow.BorrowSnapshotStore` is the authority on "could this have been
borrowed at that instant", and every short gate reads it. Nothing wrote to it, so the
store stayed empty, every locate lookup returned ``None``, and every short candidate was
dropped before it became a proposal.

The answers come from ``app.trading.borrow_source.BorrowDataSource``, NOT from a KIS
method — three guessed 대주 endpoints were disproven by read-only probes against the
live account (see that module). With no source configured this service polls nothing.

The subsystem was therefore fail-closed AND inert — safe, but structurally unable to
accumulate the forward evidence its own promotion ladder requires. This service is
what makes the bottom rung of the ladder have an exit.

What it does NOT do
-------------------
It does not authorise anything. A snapshot recording ``available=True`` is one
precondition among many; the deployment state, profitability gate and risk checks all
still apply. It also never *infers* availability — a symbol it failed to poll simply
has no snapshot, which every consumer reads as no-locate.

Cost discipline
---------------
A borrow source may rate-limit (a broker query certainly does), and polling every
candidate every tick would spend the whole budget on names no short will ever look at.
So the poller is:

* **demand-driven** — only symbols the short strategies could actually elect;
* **staggered** — a per-symbol interval, not a full sweep per tick;
* **budgeted** — a hard cap on lookups per cycle.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

from app.trading.borrow import (
    DEFAULT_MAX_SNAPSHOT_AGE_SECONDS,
    BorrowSnapshot,
    BorrowSnapshotStore,
    default_borrow_store,
)

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.getenv(name)
        return float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.getenv(name)
        return int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class BorrowPollingConfig:
    # ON by default, because the SOURCE is now the gate rather than this flag.
    #
    # Polling no longer calls KIS directly. Three guessed 대주 endpoints were disproven
    # by read-only probes on 2026-08-02 (wrong semantic / no such TR / HTTP 404), so
    # availability moved behind ``app.trading.borrow_source.BorrowDataSource``. When no
    # source is configured the provider is ``None`` and this service polls nothing —
    # inert without needing a flag.
    #
    # Leaving it on therefore means "poll IF a source exists", which is what an
    # operator who has just configured one expects. Set ``BORROW_POLLING_ENABLED=false``
    # to suppress polling even with a source present.
    enabled: bool = True
    # Re-poll a symbol at most this often. Shorter than the snapshot freshness
    # ceiling so a symbol under active consideration always has a usable locate.
    interval_seconds: float = 20.0
    # Hard cap on broker lookups per cycle. KIS rate-limits, and an unbounded sweep
    # would starve the account/quote calls that the LONG path depends on.
    max_lookups_per_cycle: int = 4
    # Symbols to poll at all. Empty means "whatever the caller passes in".
    max_tracked_symbols: int = 40
    # After this many consecutive failures a symbol is parked and stops consuming
    # budget. It is retried after ``failure_backoff_seconds``.
    failure_threshold: int = 3
    failure_backoff_seconds: float = 300.0

    @classmethod
    def from_env(cls) -> "BorrowPollingConfig":
        return cls(
            enabled=os.getenv("BORROW_POLLING_ENABLED", "true").strip().lower()
            not in {"0", "false", "no", "off"},
            interval_seconds=max(
                5.0, _env_float("BORROW_POLL_INTERVAL_SECONDS", cls.interval_seconds)
            ),
            max_lookups_per_cycle=max(
                1, _env_int("BORROW_POLL_MAX_LOOKUPS_PER_CYCLE", cls.max_lookups_per_cycle)
            ),
            max_tracked_symbols=max(
                1, _env_int("BORROW_POLL_MAX_TRACKED_SYMBOLS", cls.max_tracked_symbols)
            ),
            failure_threshold=max(
                1, _env_int("BORROW_POLL_FAILURE_THRESHOLD", cls.failure_threshold)
            ),
            failure_backoff_seconds=max(
                30.0,
                _env_float("BORROW_POLL_FAILURE_BACKOFF_SECONDS", cls.failure_backoff_seconds),
            ),
        )


@dataclass
class _SymbolState:
    last_polled_at: datetime | None = None
    consecutive_failures: int = 0
    parked_until: datetime | None = None


@dataclass(frozen=True)
class BorrowPollingStats:
    polled: int = 0
    recorded: int = 0
    failed: int = 0
    skipped_fresh: int = 0
    parked: int = 0
    tracked: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "polled": self.polled,
            "recorded": self.recorded,
            "failed": self.failed,
            "skipped_fresh": self.skipped_fresh,
            "parked": self.parked,
            "tracked": self.tracked,
        }


class BorrowPollingService:
    """Polls 대주 availability for short-eligible symbols and journals the answers."""

    def __init__(
        self,
        *,
        availability_provider: Callable[[str], BorrowSnapshot] | None = None,
        shortable_provider: Callable[[], Sequence[str]] | None = None,
        store: BorrowSnapshotStore | None = None,
        config: BorrowPollingConfig | None = None,
    ) -> None:
        # Injected rather than resolved internally, so this runs in tests and on a Pi
        # with no credentials without reaching for the network.
        self.availability_provider = availability_provider
        self.shortable_provider = shortable_provider
        self.store = store or default_borrow_store()
        self.config = config or BorrowPollingConfig.from_env()
        self._lock = threading.RLock()
        self._symbols: dict[str, _SymbolState] = {}
        self._shortable: frozenset[str] | None = None
        self._shortable_at: datetime | None = None
        self._last_stats = BorrowPollingStats()

    # -- symbol tracking ----------------------------------------------------- #
    def track(self, symbols: Iterable[str]) -> None:
        """Register symbols a short strategy could elect this cycle.

        Demand-driven: the caller passes the current short-eligible candidates, so the
        poller never spends budget on names no short will look at.
        """
        with self._lock:
            for raw in symbols:
                symbol = str(raw or "").strip().upper()
                if not symbol:
                    continue
                if symbol not in self._symbols:
                    if len(self._symbols) >= self.config.max_tracked_symbols:
                        continue
                    self._symbols[symbol] = _SymbolState()

    def forget(self, symbols: Iterable[str]) -> None:
        with self._lock:
            for raw in symbols:
                self._symbols.pop(str(raw or "").strip().upper(), None)

    # -- polling ------------------------------------------------------------- #
    def poll_once(self, *, now: datetime | None = None) -> BorrowPollingStats:
        """Poll up to ``max_lookups_per_cycle`` due symbols. Safe to call per tick."""
        moment = _aware(now or datetime.now(timezone.utc))
        if not self.config.enabled or self.availability_provider is None:
            with self._lock:
                self._last_stats = BorrowPollingStats(tracked=len(self._symbols))
            return self._last_stats

        polled = recorded = failed = skipped = parked = 0
        for symbol in self._due_symbols(moment):
            if polled >= self.config.max_lookups_per_cycle:
                break
            polled += 1
            try:
                snapshot = self.availability_provider(symbol)
            except Exception as exc:  # noqa: BLE001 - a source error is not a no-locate.
                failed += 1
                self._record_failure(symbol, moment)
                # Deliberately NOT written as ``available=False``. "The broker said no"
                # and "we could not ask" need different operator responses, and
                # conflating them hides a credentials or endpoint outage behind a
                # normal-looking refusal.
                logger.debug("borrow lookup unanswered for %s: %s", symbol, exc)
                continue
            if snapshot is None:
                failed += 1
                self._record_failure(symbol, moment)
                continue
            if self.store.record(snapshot):
                recorded += 1
            self._record_success(symbol, moment)

        with self._lock:
            parked = sum(
                1
                for state in self._symbols.values()
                if state.parked_until is not None and state.parked_until > moment
            )
            skipped = max(0, len(self._symbols) - polled - parked)
            self._last_stats = BorrowPollingStats(
                polled=polled,
                recorded=recorded,
                failed=failed,
                skipped_fresh=skipped,
                parked=parked,
                tracked=len(self._symbols),
            )
        return self._last_stats

    def refresh_shortable_universe(self, *, now: datetime | None = None) -> frozenset[str]:
        """Which symbols the broker permits 대주 on at all.

        Cached for an hour: the permitted list changes daily, not per tick. A failure
        leaves the previous answer in place rather than clearing it, because an empty
        set would read as "nothing is shortable" and silently stop all short evaluation
        on a transient error.
        """
        moment = _aware(now or datetime.now(timezone.utc))
        with self._lock:
            fresh = (
                self._shortable is not None
                and self._shortable_at is not None
                and (moment - self._shortable_at).total_seconds() < 3600.0
            )
            if fresh:
                return self._shortable  # type: ignore[return-value]
        if self.shortable_provider is None:
            return frozenset()
        try:
            resolved = frozenset(
                str(item).strip().upper() for item in (self.shortable_provider() or ())
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("shortable universe lookup failed: %s", exc)
            with self._lock:
                return self._shortable if self._shortable is not None else frozenset()
        with self._lock:
            self._shortable = resolved
            self._shortable_at = moment
            return resolved

    def is_shortable(self, symbol: str) -> bool | None:
        """``None`` when the universe has never been resolved (unknown, not false)."""
        with self._lock:
            if self._shortable is None:
                return None
            return str(symbol or "").strip().upper() in self._shortable

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.config.enabled,
                "provider_available": self.availability_provider is not None,
                "interval_seconds": self.config.interval_seconds,
                "max_lookups_per_cycle": self.config.max_lookups_per_cycle,
                "tracked_symbols": sorted(self._symbols),
                "parked_symbols": sorted(
                    symbol
                    for symbol, state in self._symbols.items()
                    if state.parked_until is not None
                ),
                "shortable_universe_size": (
                    len(self._shortable) if self._shortable is not None else None
                ),
                "last_cycle": self._last_stats.as_dict(),
            }

    # -- internals ----------------------------------------------------------- #
    def _due_symbols(self, now: datetime) -> list[str]:
        """Symbols whose snapshot is stale enough to re-poll, oldest first."""
        with self._lock:
            due: list[tuple[datetime, str]] = []
            for symbol, state in self._symbols.items():
                if state.parked_until is not None:
                    if state.parked_until > now:
                        continue
                    # Backoff elapsed: give it another chance.
                    state.parked_until = None
                    state.consecutive_failures = 0
                last = state.last_polled_at
                if (
                    last is not None
                    and (now - last).total_seconds() < self.config.interval_seconds
                ):
                    continue
                # Never-polled symbols sort first via the epoch sentinel.
                due.append((last or datetime.min.replace(tzinfo=timezone.utc), symbol))
            due.sort()
            return [symbol for _, symbol in due]

    def _record_success(self, symbol: str, now: datetime) -> None:
        with self._lock:
            state = self._symbols.setdefault(symbol, _SymbolState())
            state.last_polled_at = now
            state.consecutive_failures = 0
            state.parked_until = None

    def _record_failure(self, symbol: str, now: datetime) -> None:
        with self._lock:
            state = self._symbols.setdefault(symbol, _SymbolState())
            state.last_polled_at = now
            state.consecutive_failures += 1
            if state.consecutive_failures >= self.config.failure_threshold:
                # Park it so one broken symbol cannot consume the whole per-cycle
                # budget and starve the symbols that still answer.
                state.parked_until = now + timedelta(
                    seconds=self.config.failure_backoff_seconds
                )


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


_DEFAULT_POLLER: BorrowPollingService | None = None
_DEFAULT_POLLER_LOCK = threading.Lock()


def default_borrow_poller() -> BorrowPollingService:
    """Process-wide poller, bound to the configured borrow source.

    Falls back to a provider-less instance (which polls nothing) rather than raising:
    with no borrow source the short path should be inert, not broken.
    """
    global _DEFAULT_POLLER
    if _DEFAULT_POLLER is None:
        with _DEFAULT_POLLER_LOCK:
            if _DEFAULT_POLLER is None:
                _DEFAULT_POLLER = BorrowPollingService(
                    availability_provider=_source_availability_provider(),
                    shortable_provider=_source_universe_provider(),
                )
    return _DEFAULT_POLLER


def reset_default_borrow_poller() -> None:
    global _DEFAULT_POLLER
    with _DEFAULT_POLLER_LOCK:
        _DEFAULT_POLLER = None


def _source_availability_provider() -> Callable[[str], BorrowSnapshot] | None:
    """Bind the poller to whatever borrow source is configured.

    Returns ``None`` when no source exists, which makes the poller inert rather than
    erroring — a machine with no borrow data should have a quiet short subsystem, not a
    broken one.
    """
    from app.trading.borrow_source import default_borrow_source

    source = default_borrow_source()
    if not source.available():
        return None

    def _lookup(symbol: str) -> BorrowSnapshot:
        snapshot = source.snapshot(symbol, now=datetime.now(timezone.utc))
        if snapshot is None:
            # Raised, not returned as unavailable: the poller records a FAILURE, which
            # leaves no snapshot at all. "We could not ask" must stay distinguishable
            # from "the answer was no".
            raise LookupError(f"borrow source has no answer for {symbol}")
        return snapshot

    return _lookup


def _source_universe_provider() -> Callable[[], Sequence[str]] | None:
    from app.trading.borrow_source import default_borrow_source

    source = default_borrow_source()
    if not source.available():
        return None
    return source.universe

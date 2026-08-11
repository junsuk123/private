"""In-memory retention of recent contexts, so a decision stays reconstructible.

Bounded on purpose. The realtime loop runs every few seconds over a rotating candidate
set, so an unbounded map would grow for as long as the process lives — and this store
exists to serve dashboards and outcome resolution, both of which only ever want the
recent past.

Deliberately NOT a SQLite table. The realtime writer already owns the sqlite files, and
the project has measured that large reads there interfere with it (see
``docs/`` notes on read-only/query-only separation). A context is worth keeping for
minutes, not months; the durable record of a decision is the selection row, which
carries the ``context_id`` and the term decomposition.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Iterable

from app.context.market_context import MarketContext

__all__ = ["MarketContextStore", "default_context_store", "reset_default_context_store"]


class MarketContextStore:
    """Thread-safe LRU of recent contexts, keyed by ``context_id``."""

    def __init__(self, *, max_contexts: int = 512, retention_seconds: float = 3600.0) -> None:
        self._max_contexts = max(1, int(max_contexts))
        self._retention = max(1.0, float(retention_seconds))
        self._lock = threading.RLock()
        self._by_id: "OrderedDict[str, MarketContext]" = OrderedDict()
        self._latest_by_symbol: dict[str, str] = {}

    def put(self, context: MarketContext) -> MarketContext:
        with self._lock:
            self._by_id[context.context_id] = context
            self._by_id.move_to_end(context.context_id)
            self._latest_by_symbol[context.symbol_id] = context.context_id
            self._evict_locked(now=context.captured_at)
        return context

    def put_all(self, contexts: Iterable[MarketContext]) -> tuple[MarketContext, ...]:
        return tuple(self.put(context) for context in contexts)

    def get(self, context_id: str) -> MarketContext | None:
        with self._lock:
            context = self._by_id.get(str(context_id or ""))
            if context is not None:
                self._by_id.move_to_end(context.context_id)
            return context

    def latest_for_symbol(self, symbol: str) -> MarketContext | None:
        key = str(symbol or "").strip().upper()
        with self._lock:
            context_id = self._latest_by_symbol.get(key)
            return self._by_id.get(context_id) if context_id else None

    def recent(self, *, limit: int = 50) -> tuple[MarketContext, ...]:
        with self._lock:
            values = list(self._by_id.values())
        return tuple(reversed(values[-max(0, int(limit)) :]))

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_id)

    def _evict_locked(self, *, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self._retention)
        # Age first, then size. Both are needed: a quiet market ages entries out while a
        # busy one hits the size cap long before the retention window.
        for context_id, context in list(self._by_id.items()):
            if context.captured_at < cutoff:
                self._by_id.pop(context_id, None)
        while len(self._by_id) > self._max_contexts:
            self._by_id.popitem(last=False)
        live_ids = set(self._by_id)
        self._latest_by_symbol = {
            symbol: context_id
            for symbol, context_id in self._latest_by_symbol.items()
            if context_id in live_ids
        }


_default_store: MarketContextStore | None = None
_default_lock = threading.Lock()


def default_context_store() -> MarketContextStore:
    global _default_store
    with _default_lock:
        if _default_store is None:
            _default_store = MarketContextStore()
        return _default_store


def reset_default_context_store() -> None:
    """Test hook. Never called from the trading path."""
    global _default_store
    with _default_lock:
        _default_store = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

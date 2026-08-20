"""Persistent per-symbol / per-strategy realized-slippage & execution-quality store.

Append-only JSONL log plus an in-memory rolling cache of recent realized slippage,
keyed by (symbol, strategy_family) and by time-of-day bucket. Feeds
:class:`app.execution.execution_quality.ExecutionQualityEngine` so a symbol with
persistently worse-than-expected fills is down-scored or blocked.

Kept deliberately lightweight (JSONL + bounded deque) so the tick loop never blocks on
heavy IO; the recent-average query is served from memory.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque
from app.audit import log_path


class ExecutionQualityStore:
    def __init__(
        self,
        path: str | Path | None = None,
        *,
        window: int = 50,
    ) -> None:
        self.path = Path(path) if path is not None else log_path("execution-quality.jsonl")
        self.window = max(1, int(window))
        self._recent: dict[tuple[str, str], Deque[float]] = defaultdict(lambda: deque(maxlen=self.window))
        self._by_time_bucket: dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=self.window))
        self._loaded = False

    def record(
        self,
        *,
        symbol: str,
        strategy_family: str,
        realized_slippage_rate: float,
        side: str = "BUY",
        time_bucket: str | None = None,
        recorded_at: str | None = None,
    ) -> None:
        self._ensure_loaded()
        rate = max(0.0, float(realized_slippage_rate))
        bucket = time_bucket or _default_time_bucket(recorded_at)
        entry = {
            "symbol": symbol,
            "strategy_family": strategy_family,
            "realized_slippage_rate": rate,
            "side": side,
            "time_bucket": bucket,
            "recorded_at": recorded_at or _now_iso(),
        }
        self._recent[(symbol, strategy_family)].append(rate)
        self._by_time_bucket[bucket].append(rate)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            # Persistence is best-effort; the in-memory cache still serves queries.
            pass

    def recent_average(self, *, symbol: str, strategy_family: str) -> float | None:
        self._ensure_loaded()
        samples = self._recent.get((symbol, strategy_family))
        if not samples:
            return None
        return sum(samples) / len(samples)

    def time_bucket_average(self, *, time_bucket: str) -> float | None:
        self._ensure_loaded()
        samples = self._by_time_bucket.get(time_bucket)
        if not samples:
            return None
        return sum(samples) / len(samples)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = (str(entry.get("symbol", "")), str(entry.get("strategy_family", "")))
                    rate = float(entry.get("realized_slippage_rate", 0.0) or 0.0)
                    self._recent[key].append(rate)
                    bucket = str(entry.get("time_bucket", "") or _default_time_bucket())
                    self._by_time_bucket[bucket].append(rate)
        except OSError:
            pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_time_bucket(recorded_at: str | None = None) -> str:
    """Coarse time-of-day bucket (KST-agnostic hour) for slippage seasonality."""
    try:
        if recorded_at:
            hour = datetime.fromisoformat(recorded_at).hour
        else:
            hour = datetime.now(timezone.utc).hour
    except (ValueError, TypeError):
        hour = datetime.now(timezone.utc).hour
    return f"h{hour:02d}"


# Optional override for tests / alternate deployments.
def default_store() -> ExecutionQualityStore:
    return ExecutionQualityStore(os.getenv("EXECUTION_QUALITY_STORE_PATH", "logs/execution-quality.jsonl"))

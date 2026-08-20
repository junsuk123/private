"""Cross-process liveness for workers that live in one process's memory.

The problem
-----------
``/api/system-diagnostics`` decides whether the trading engine is running by
asking ``_realtime_trading_worker.is_alive()`` -- a question about THIS process,
not about the system. That is correct for the live instance, which owns the
worker, and wrong for any other observer of the same system. The read-only
instance published behind Tailscale Funnel deliberately starts no workers,
because the live instance owns the SQLite stores and running the same writers
twice risks corrupting them. It therefore reported "거래 엔진 중지" and a degraded
readiness score while the engine was running normally a port away.

Three of the five workers already avoid this: ``krx_realtime``, ``us_realtime``
and ``research_collection`` infer liveness from the freshness of data in the
SHARED stores, so any process reading those stores reaches the same conclusion.
This module gives the remaining two -- ``trading_engine`` and ``model_training``
-- the same property, by having whichever process runs a worker leave a dated
mark that every process can read.

Why a file and not a table
--------------------------
A heartbeat is written on every engine cycle, which is several times a minute
for the life of the process. The SQLite stores in this system are already the
contended resource -- the live tick loop, the training row store and the
performance store share a disk -- and a heartbeat has none of the properties
that would justify joining them: it is not queried in aggregate, it is worthless
after a minute, and losing it costs a stale flag rather than data. One small
JSON file per worker, replaced atomically, keeps the write off the databases
that matter.

Staleness, not liveness
-----------------------
A mark says a worker was alive at a time, never that it is alive now. A process
killed with SIGKILL leaves its last mark behind, so readers must treat the mark
as expired once it is older than ``ttl_seconds``; the default is generous enough
to survive a slow cycle and short enough that a dead worker stops being reported
as running within a couple of minutes.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_HEARTBEAT_DIR = Path("data/runtime/worker-heartbeats")

#: How long a mark counts for. Longer than the slowest expected cycle of the
#: workers this covers, and short enough that a killed process is not reported as
#: running for more than about two minutes.
DEFAULT_TTL_SECONDS = 150.0


def _heartbeat_path(worker: str, directory: Path | None = None) -> Path:
    safe = "".join(ch for ch in str(worker) if ch.isalnum() or ch in "-_")
    return (directory or DEFAULT_HEARTBEAT_DIR) / f"{safe}.json"


def record(worker: str, *, detail: Any = None, directory: Path | None = None) -> None:
    """Mark ``worker`` alive as of now. Never raises.

    Written to a temporary file and renamed, so a reader can never observe a
    half-written mark: on POSIX the rename is atomic, and a torn read here would
    turn a running worker into a stopped one on the dashboard.
    """
    path = _heartbeat_path(worker, directory)
    payload = {
        "worker": str(worker),
        "at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "detail": detail,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=str(path.parent), delete=False, suffix=".tmp"
        )
        try:
            json.dump(payload, handle, ensure_ascii=False, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        os.replace(handle.name, path)
    except Exception:  # noqa: BLE001 - a heartbeat must never break a worker loop.
        try:
            os.unlink(handle.name)  # type: ignore[possibly-undefined]
        except Exception:  # noqa: BLE001
            pass


def read(worker: str, *, directory: Path | None = None) -> dict[str, Any] | None:
    path = _heartbeat_path(worker, directory)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - absent or unreadable is simply "no mark".
        return None


def is_alive(
    worker: str,
    *,
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    directory: Path | None = None,
    now: datetime | None = None,
) -> bool:
    """Did some process mark ``worker`` alive recently enough to count?"""
    mark = read(worker, directory=directory)
    if not mark:
        return False
    raw = mark.get("at")
    if not raw:
        return False
    try:
        stamped = datetime.fromisoformat(str(raw))
    except ValueError:
        return False
    if stamped.tzinfo is None:
        stamped = stamped.replace(tzinfo=timezone.utc)
    reference = now or datetime.now(timezone.utc)
    return (reference - stamped).total_seconds() <= max(1.0, float(ttl_seconds))


def running(
    worker: str,
    local_thread_alive: bool,
    *,
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    directory: Path | None = None,
) -> bool:
    """Is this worker running ANYWHERE, as far as this process can tell?

    The local thread wins when it is alive, because that is direct knowledge and
    needs no freshness argument. Otherwise a recent mark from another process
    answers -- which is the whole point: an observer that runs no workers should
    report the system, not itself.
    """
    if local_thread_alive:
        return True
    return is_alive(worker, ttl_seconds=ttl_seconds, directory=directory)


#: Worker keys, kept here so producer and consumer cannot drift apart.
TRADING_ENGINE = "trading_engine"
MODEL_TRAINING = "model_training"

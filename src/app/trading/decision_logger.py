from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.audit import log_path


class DecisionLogger:
    """Append-only journal of trading decisions, bounded by size.

    Rotation is not optional here. This journal is written on every decision
    cycle of a server that runs for weeks, and without a bound it was the only
    log in the tree that grew forever -- 230 MB and climbing, while every other
    writer (AuditLogger, ShadowComparisonRecorder, the feature journal) already
    capped itself. The scheme mirrors AuditLogger's deliberately: same numbered
    suffixes, same "drop the oldest" rule, so anything that reads or prunes
    these files can treat both the same way.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        max_bytes: int | None = None,
        backup_count: int = 3,
    ) -> None:
        # See app.audit.logger.log_path: resolved on call, not bound at import.
        self.path = Path(path) if path is not None else log_path("decision-log.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max(
            1,
            int(
                max_bytes
                if max_bytes is not None
                else os.getenv("DECISION_LOG_MAX_BYTES", str(128 * 1024 * 1024))
            ),
        )
        self.backup_count = max(1, int(backup_count))
        self._write_lock = threading.Lock()

    def _rotate_if_needed(self, incoming_bytes: int) -> None:
        try:
            current_bytes = self.path.stat().st_size
        except FileNotFoundError:
            return
        if current_bytes + incoming_bytes <= self.max_bytes:
            return
        oldest = self.path.with_name(f"{self.path.name}.{self.backup_count}")
        oldest.unlink(missing_ok=True)
        for index in range(self.backup_count - 1, 0, -1):
            source = self.path.with_name(f"{self.path.name}.{index}")
            if source.exists():
                source.replace(self.path.with_name(f"{self.path.name}.{index + 1}"))
        self.path.replace(self.path.with_name(f"{self.path.name}.1"))

    def record(self, event_type: str, payload: dict[str, Any]) -> None:
        data = {
            "event_type": event_type,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **_jsonable(payload),
        }
        line = json.dumps(data, ensure_ascii=True, sort_keys=True) + "\n"
        # Two engines share the default journal; without the lock one can append
        # while the other is renaming the file out from under it.
        with self._write_lock:
            self._rotate_if_needed(len(line.encode("utf-8")))
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value

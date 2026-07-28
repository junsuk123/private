from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}


@dataclass(frozen=True)
class IdempotencyRecord:
    key: str
    created_at: datetime
    payload_hash: str
    status: str
    result: dict[str, Any]


class IdempotencyStore:
    def __init__(self, path: str | Path = "data/store/live_order_idempotency.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(self.path.resolve())
        with _LOCKS_GUARD:
            self._lock = _PATH_LOCKS.setdefault(resolved, threading.RLock())

    def get(self, key: str, *, ttl_seconds: int) -> IdempotencyRecord | None:
        with self._lock:
            return self._get_unlocked(key, ttl_seconds=ttl_seconds)

    def reserve(
        self,
        key: str,
        payload_hash: str,
        result: dict[str, Any],
        *,
        ttl_seconds: int,
    ) -> tuple[bool, IdempotencyRecord]:
        """Atomically reserve an idempotency key before broker submission."""
        with self._lock:
            existing = self._get_unlocked(key, ttl_seconds=ttl_seconds)
            if existing is not None:
                return False, existing
            self._put_unlocked(key, payload_hash, "PENDING_SUBMISSION", result)
            created = self._get_unlocked(key, ttl_seconds=ttl_seconds)
            if created is None:  # pragma: no cover - defensive filesystem failure
                raise RuntimeError("failed to persist idempotency reservation")
            return True, created

    def _get_unlocked(self, key: str, *, ttl_seconds: int) -> IdempotencyRecord | None:
        now = datetime.now(timezone.utc)
        record = None
        for item in self._read_all():
            if item.key == key:
                record = item
        if record is None:
            return None
        if record.created_at + timedelta(seconds=ttl_seconds) < now:
            return None
        return record

    def put(self, key: str, payload_hash: str, status: str, result: dict[str, Any]) -> None:
        with self._lock:
            self._put_unlocked(key, payload_hash, status, result)

    def _put_unlocked(
        self,
        key: str,
        payload_hash: str,
        status: str,
        result: dict[str, Any],
    ) -> None:
        record = {
            "key": key,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "payload_hash": payload_hash,
            "status": status,
            "result": result,
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
            file.flush()
            os.fsync(file.fileno())

    def _read_all(self) -> list[IdempotencyRecord]:
        if not self.path.exists():
            return []
        records: list[IdempotencyRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                created_at = datetime.fromisoformat(str(raw["created_at"]))
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                records.append(
                    IdempotencyRecord(
                        key=str(raw["key"]),
                        created_at=created_at.astimezone(timezone.utc),
                        payload_hash=str(raw["payload_hash"]),
                        status=str(raw["status"]),
                        result=dict(raw.get("result") or {}),
                    )
                )
            except (KeyError, ValueError, json.JSONDecodeError, TypeError):
                continue
        return records

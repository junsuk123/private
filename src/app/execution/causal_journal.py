from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from app.trading.contracts import OrderIntent, RiskVerdict


class CausalJournalError(RuntimeError):
    pass


class CausalOrderJournal:
    """Append-only lifecycle journal for the strategy-owned execution path.

    Critical records are flushed and fsynced before this method returns. The
    existing live path remains separate until strategy_owned_execution is enabled.
    """

    def __init__(self, path: str | Path = "data/store/causal-order-journal.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def persist_intent(self, intent: OrderIntent) -> None:
        with self._lock:
            existing = self.intent_for_idempotency_key(intent.idempotency_key)
            if existing is not None:
                if existing != _jsonable(intent):
                    raise CausalJournalError("IDEMPOTENCY_KEY_PAYLOAD_MISMATCH")
                return
            self._append("order_intent_persisted", _jsonable(intent))

    def persist_risk_verdict(self, verdict: RiskVerdict) -> None:
        with self._lock:
            if self.intent_by_id(verdict.intent_id) is None:
                raise CausalJournalError("RISK_VERDICT_WITHOUT_PERSISTED_INTENT")
            self._append("risk_verdict_persisted", _jsonable(verdict))

    def record(self, event_type: str, payload: Any) -> None:
        self._append(event_type, _jsonable(payload))

    def intent_for_idempotency_key(self, key: str) -> dict[str, Any] | None:
        for row in reversed(self.read_all()):
            payload = row.get("payload")
            if (
                row.get("event_type") == "order_intent_persisted"
                and isinstance(payload, dict)
                and payload.get("idempotency_key") == key
            ):
                return payload
        return None

    def intent_by_id(self, intent_id: str) -> dict[str, Any] | None:
        for row in reversed(self.read_all()):
            payload = row.get("payload")
            if (
                row.get("event_type") == "order_intent_persisted"
                and isinstance(payload, dict)
                and payload.get("intent_id") == intent_id
            ):
                return payload
        return None

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows

    def _append(self, event_type: str, payload: Any) -> None:
        row = {"event_type": event_type, "payload": payload}
        encoded = json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    return value

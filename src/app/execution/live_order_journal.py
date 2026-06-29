from __future__ import annotations

from pathlib import Path
from typing import Any

from app.audit import AuditLogger


class LiveOrderJournal:
    def __init__(self, path: str | Path = "logs/live-orders.jsonl") -> None:
        self.audit = AuditLogger(Path(path))

    def record(self, event_type: str, payload: Any) -> None:
        self.audit.record(event_type, payload)

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.audit import AuditLogger, log_path


class LiveOrderJournal:
    def __init__(self, path: str | Path | None = None) -> None:
        # Resolved here rather than in the signature: a default argument is bound at
        # import time, which is what made this journal impossible to redirect.
        self.audit = AuditLogger(Path(path) if path is not None else log_path("live-orders.jsonl"))

    def record(self, event_type: str, payload: Any) -> None:
        self.audit.record(event_type, payload)

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class KisMode:
    paper: bool
    live_enabled: bool
    base_url: str

    @property
    def name(self) -> str:
        return "paper" if self.paper else "live"


@dataclass(frozen=True)
class KisHealthCheck:
    ok: bool
    mode: str
    checked_at: datetime
    gates: dict[str, bool]
    failures: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LiveOrderSubmission:
    execution_id: str
    idempotency_key: str
    status: str
    broker_order_id: str | None
    submitted_at: datetime
    message: str

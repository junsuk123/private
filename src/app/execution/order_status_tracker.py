from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class OrderStatusSnapshot:
    order_id: str
    status: str
    observed_at: datetime
    raw: Any


class OrderStatusTracker:
    terminal_statuses = {"FILLED", "REJECTED", "CANCELED", "EXPIRED"}

    def __init__(self, broker: Any) -> None:
        self.broker = broker

    def poll(self, order_id: str, *, interval_seconds: int, timeout_seconds: int) -> OrderStatusSnapshot:
        deadline = time.monotonic() + timeout_seconds
        last = None
        while True:
            execution = self.broker.get_order_status(order_id)
            status = str(getattr(execution, "status", "UNKNOWN")).upper()
            last = OrderStatusSnapshot(
                order_id=order_id,
                status=status,
                observed_at=datetime.now(timezone.utc),
                raw=execution,
            )
            if status in self.terminal_statuses or time.monotonic() >= deadline:
                return last
            time.sleep(max(1, interval_seconds))

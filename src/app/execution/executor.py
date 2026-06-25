from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from app.schemas.domain import FinalOrder


@dataclass(frozen=True)
class ExecutionReceipt:
    mode: str
    accepted: bool
    message: str
    order: FinalOrder
    recorded_at: datetime


class PaperOrderExecutor:
    def submit(self, order: FinalOrder) -> ExecutionReceipt:
        return ExecutionReceipt(
            mode="paper",
            accepted=True,
            message="Paper order recorded; no brokerage API was called.",
            order=order,
            recorded_at=datetime.now(timezone.utc),
        )


class DisabledLiveOrderExecutor:
    def submit(self, order: FinalOrder) -> ExecutionReceipt:
        raise RuntimeError(
            f"Live trading is disabled. Refused to submit {order.side} {order.ticker}."
        )

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class FeatureProvenance:
    symbol: str
    decision_time: datetime
    tick_record_ids: tuple[str, ...]
    orderbook_record_id: str | None
    source: str
    max_input_age_ms: float

    @property
    def source_record_ids(self) -> tuple[str, ...]:
        return (*self.tick_record_ids, *((self.orderbook_record_id,) if self.orderbook_record_id else ()))

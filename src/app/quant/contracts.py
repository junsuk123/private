from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


class DataQuality(str, Enum):
    GOOD = "good"
    DEGRADED = "degraded"
    INVALID = "invalid"


class ValidationStatus(str, Enum):
    UNVALIDATED = "unvalidated"
    PASSED = "passed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


def aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("quant timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class QuantBar:
    """A completed market bar, with event and knowledge time kept separate."""

    symbol: str
    market: str
    interval: str
    start_time: datetime
    end_time: datetime
    received_at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        for field_name in ("start_time", "end_time", "received_at"):
            aware_utc(getattr(self, field_name))
        if self.end_time <= self.start_time:
            raise ValueError("bar end_time must be after start_time")
        values = (self.open, self.high, self.low, self.close, self.volume)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("bar values must be finite")
        if min(self.open, self.high, self.low, self.close) <= 0 or self.volume < 0:
            raise ValueError("bar price must be positive and volume non-negative")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("invalid OHLC ordering")


@dataclass(frozen=True)
class QuantEvidence:
    symbol: str
    market: str
    timestamp: datetime
    bar_interval: str
    metric: str
    value: float | None
    window: int | None
    input_start: datetime
    input_end: datetime
    freshness_ms: float
    data_quality: DataQuality
    implementation: str
    method_reference: str
    validation_status: ValidationStatus
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        aware_utc(self.timestamp)
        aware_utc(self.input_start)
        aware_utc(self.input_end)
        if self.input_end > self.timestamp:
            raise ValueError("quant evidence cannot use future input")
        if self.value is not None and not math.isfinite(float(self.value)):
            raise ValueError("evidence value must be finite or None")
        if self.value is None and not self.unavailable_reason:
            raise ValueError("unavailable evidence requires a reason")
        if self.freshness_ms < 0:
            raise ValueError("freshness_ms cannot be negative")

    @property
    def usable(self) -> bool:
        return (
            self.value is not None
            and self.data_quality is not DataQuality.INVALID
            and self.validation_status is not ValidationStatus.FAILED
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in ("timestamp", "input_start", "input_end"):
            result[name] = getattr(self, name).isoformat()
        result["data_quality"] = self.data_quality.value
        result["validation_status"] = self.validation_status.value
        return result


@runtime_checkable
class QuantKnowledgeProvider(Protocol):
    def compute_features(self, bars: Sequence[QuantBar], *, as_of: datetime | None = None) -> tuple[QuantEvidence, ...]: ...
    def compute_risk_metrics(self, context: Mapping[str, Any]) -> tuple[QuantEvidence, ...]: ...
    def evaluate_portfolio(self, context: Mapping[str, Any]) -> tuple[QuantEvidence, ...]: ...
    def run_scenario(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def health(self) -> Mapping[str, Any]: ...

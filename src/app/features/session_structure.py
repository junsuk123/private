"""Causal KRX session features shared by training and live serving.

The functions in this module are the single calculation authority for the
price-derived fields used by session-boxed strategies.  Callers may adapt their
bar objects, but must not reimplement the calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from statistics import fmean
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class OpeningRange:
    high: float
    low: float
    close: float
    volatility: float
    minutes: int
    samples: int


def _timestamp(bar: Any) -> datetime | None:
    for name in ("end_time", "as_of"):
        value = getattr(bar, name, None)
        if isinstance(value, datetime):
            return value
    start = getattr(bar, "minute_start", None)
    if isinstance(start, datetime):
        return start + timedelta(minutes=1)
    if isinstance(bar, dict):
        for name in ("end_time", "as_of", "time", "minute_start"):
            value = bar.get(name)
            if not value:
                continue
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except (TypeError, ValueError):
                continue
            return parsed + timedelta(minutes=1) if name in {"minute_start", "time"} else parsed
    return None


def _number(bar: Any, name: str) -> float | None:
    value = bar.get(name) if isinstance(bar, dict) else getattr(bar, name, None)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0.0 and math.isfinite(result) else None


def opening_range(
    bars: Sequence[Any],
    *,
    session_open: datetime,
    minutes: int = 30,
    now: datetime,
) -> OpeningRange | None:
    """Return a completed opening range using only observations visible at ``now``."""
    if minutes <= 0 or now < session_open + timedelta(minutes=minutes):
        return None
    window_end = session_open + timedelta(minutes=minutes)
    visible: list[tuple[datetime, Any]] = []
    for bar in bars:
        timestamp = _timestamp(bar)
        if timestamp is None or timestamp > now:
            continue
        if session_open < timestamp <= window_end:
            visible.append((timestamp, bar))
    visible.sort(key=lambda item: item[0])
    if not visible:
        return None

    # A last observation before the boundary is a partial range.  Also reject a
    # feed whose first bar starts materially after the open or has a hole larger
    # than its normal cadence.
    timestamps = [item[0] for item in visible]
    if timestamps[-1] < window_end:
        return None
    deltas = [
        (timestamps[index] - timestamps[index - 1]).total_seconds()
        for index in range(1, len(timestamps))
        if timestamps[index] > timestamps[index - 1]
    ]
    cadence = min(deltas, default=float(minutes * 60))
    if (timestamps[0] - session_open).total_seconds() > cadence * 1.5:
        return None
    if any(delta > cadence * 1.5 for delta in deltas):
        return None

    highs = [_number(bar, "high") for _, bar in visible]
    lows = [_number(bar, "low") for _, bar in visible]
    closes = [_number(bar, "close") for _, bar in visible]
    if any(value is None for value in (*highs, *lows, *closes)):
        return None
    numeric_highs = [float(value) for value in highs if value is not None]
    numeric_lows = [float(value) for value in lows if value is not None]
    numeric_closes = [float(value) for value in closes if value is not None]
    mean_close = fmean(numeric_closes)
    if mean_close <= 0.0:
        return None
    return OpeningRange(
        high=max(numeric_highs),
        low=min(numeric_lows),
        close=numeric_closes[-1],
        volatility=(max(numeric_highs) - min(numeric_lows)) / mean_close,
        minutes=minutes,
        samples=len(visible),
    )


def first_half_hour_return_bps(
    bars: Sequence[Any],
    *,
    previous_close: float,
    session_open: datetime,
    minutes: int = 30,
    now: datetime,
) -> float | None:
    """First-half-hour return relative to the prior day's close (gap included)."""
    try:
        reference = float(previous_close)
    except (TypeError, ValueError):
        return None
    if reference <= 0.0:
        return None
    observed = opening_range(
        bars, session_open=session_open, minutes=minutes, now=now
    )
    if observed is None:
        return None
    return (observed.close / reference - 1.0) * 10_000.0


def first_half_hour_volatility_percentile(
    history: Iterable[float],
    current: float,
    *,
    minimum_samples: int,
) -> float | None:
    """Empirical percentile against strictly historical opening volatilities."""
    try:
        value = float(current)
        observations = [float(item) for item in history]
    except (TypeError, ValueError):
        return None
    observations = [item for item in observations if item >= 0.0 and math.isfinite(item)]
    if value < 0.0 or not math.isfinite(value) or len(observations) < max(1, int(minimum_samples)):
        return None
    return sum(item <= value for item in observations) / len(observations)

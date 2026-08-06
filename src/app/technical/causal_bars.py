"""Completed fixed-time bars for slow indicators. One builder, live and replay.

Slow indicators were computed over the last N irregular ticks. "RSI(14)" then
meant fourteen *prints*, which is fourteen seconds on a busy name and forty
minutes on a quiet one -- the same number described two unrelated things, and the
period had no time meaning at all. Worse, the in-progress minute was included, so
a value recomputed a second later could change retroactively.

This module produces bars that are:

    * COMPLETED -- the minute containing ``as_of`` is excluded, always. A bar is
      only emitted once its window has closed, so a value computed at
      ``decision_time`` never moves afterwards.
    * FIXED-TIME -- 1m is the base; 3m/5m are causal aggregations of completed 1m
      bars, so an indicator period means the same wall-clock span in KR (websocket
      cadence) and US (REST cadence).
    * SHARED -- live and replay call this same function, so a replay number and a
      live number are comparable by construction rather than by hope.

The in-progress minute is still available via :func:`forming_bar` for entry
timing, which is a different question from "what is the trend": timing wants the
freshest possible print, indicators want a value that will not be revised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

from app.features.schemas import OHLCVBar

#: Aggregations that are exact multiples of the 1m base.
SUPPORTED_TIMEFRAMES: tuple[int, ...] = (1, 3, 5)


@dataclass(frozen=True)
class CausalBarSet:
    """Completed bars plus the provenance needed to audit an indicator value."""

    symbol: str
    timeframe_minutes: int
    as_of: datetime
    bars: tuple[OHLCVBar, ...]
    #: Bars requested vs. actually available; warmup_complete answers "may a
    #: period-N indicator be trusted here" without the caller re-deriving it.
    warmup_required: int = 0
    source_record_ids: tuple[str, ...] = ()
    dropped_forming_bar: bool = False

    @property
    def bar_count(self) -> int:
        return len(self.bars)

    @property
    def warmup_complete(self) -> bool:
        return self.bar_count >= max(1, self.warmup_required)

    @property
    def last_bar_start(self) -> datetime | None:
        return self.bars[-1].as_of if self.bars else None

    def as_provenance(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe_minutes": self.timeframe_minutes,
            "as_of": self.as_of.isoformat(),
            "bar_count": self.bar_count,
            "warmup_required": self.warmup_required,
            "warmup_complete": self.warmup_complete,
            "last_bar_start": (
                self.last_bar_start.isoformat() if self.last_bar_start else None
            ),
            "dropped_forming_bar": self.dropped_forming_bar,
            "source_record_ids": list(self.source_record_ids[-64:]),
        }


def floor_to_minute(moment: datetime, minutes: int = 1) -> datetime:
    """Start of the fixed window containing ``moment`` (UTC, minute-aligned)."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)
    step = max(1, int(minutes))
    floored_minute = (moment.minute // step) * step
    return moment.replace(minute=floored_minute, second=0, microsecond=0)


def _as_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _to_ohlcv(row: Any, symbol: str) -> OHLCVBar | None:
    """RealtimeMinuteBar (or any row exposing the same names) -> OHLCVBar."""
    start = _as_utc(getattr(row, "minute_start", None))
    if start is None:
        return None
    try:
        open_ = float(getattr(row, "open"))
        high = float(getattr(row, "high"))
        low = float(getattr(row, "low"))
        close = float(getattr(row, "close"))
        volume = float(getattr(row, "volume", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    # A malformed bar is dropped rather than repaired: silently widening the range
    # to make it "valid" would invent a high or low that never traded.
    if not (high >= low and high >= close >= low and high >= open_ >= low):
        return None
    return OHLCVBar(
        ticker=str(getattr(row, "symbol", symbol) or symbol),
        as_of=start,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def completed_bars(
    rows: Iterable[Any],
    *,
    symbol: str,
    as_of: datetime,
    timeframe_minutes: int = 1,
    warmup_required: int = 0,
    limit: int | None = None,
) -> CausalBarSet:
    """Completed, de-duplicated, time-ordered bars strictly before ``as_of``.

    ``rows`` are minute rows (``minute_start``/OHLCV). The window containing
    ``as_of`` is dropped even when the store has already persisted it: the store
    writes the forming minute periodically so a quiet symbol does not lose its
    last bar, which is right for storage and wrong as an indicator input.
    """
    moment = _as_utc(as_of) or datetime.now(timezone.utc)
    step = int(timeframe_minutes)
    if step not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"unsupported timeframe: {timeframe_minutes}")

    current_minute = floor_to_minute(moment, 1)
    by_start: dict[datetime, OHLCVBar] = {}
    source_ids: list[str] = []
    dropped_forming = False
    for row in rows or ():
        bar = _to_ohlcv(row, symbol)
        if bar is None:
            continue
        if bar.as_of >= current_minute:
            # The minute containing as_of has not closed yet.
            dropped_forming = True
            continue
        by_start[bar.as_of] = bar
        for record_id in getattr(row, "source_record_ids", ()) or ():
            source_ids.append(str(record_id))

    ordered = [by_start[key] for key in sorted(by_start)]
    if step > 1:
        ordered = _aggregate(ordered, step, symbol)
    if limit is not None and limit > 0:
        ordered = ordered[-limit:]
    return CausalBarSet(
        symbol=symbol,
        timeframe_minutes=step,
        as_of=moment,
        bars=tuple(ordered),
        warmup_required=max(0, int(warmup_required)),
        source_record_ids=tuple(source_ids[-256:]),
        dropped_forming_bar=dropped_forming,
    )


def _aggregate(bars: Sequence[OHLCVBar], step: int, symbol: str) -> list[OHLCVBar]:
    """Causally aggregate completed 1m bars into ``step``-minute bars.

    A window is emitted only when the NEXT window has started, because a window
    whose final minute has not yet elapsed is itself still forming. Partial
    trailing windows are therefore dropped rather than emitted short -- an
    aggregate built from 2 of 5 minutes is not a 5m bar.
    """
    if not bars:
        return []
    grouped: dict[datetime, list[OHLCVBar]] = {}
    for bar in bars:
        grouped.setdefault(floor_to_minute(bar.as_of, step), []).append(bar)

    out: list[OHLCVBar] = []
    for start in sorted(grouped):
        window = sorted(grouped[start], key=lambda item: item.as_of)
        # Require the window to be closed: its last possible minute must be in the
        # past relative to the newest completed 1m bar we hold.
        window_end = start + timedelta(minutes=step)
        if bars[-1].as_of + timedelta(minutes=1) < window_end:
            continue
        out.append(
            OHLCVBar(
                ticker=symbol,
                as_of=start,
                open=window[0].open,
                high=max(item.high for item in window),
                low=min(item.low for item in window),
                close=window[-1].close,
                volume=sum(item.volume for item in window),
            )
        )
    return out


def forming_bar(rows: Iterable[Any], *, symbol: str, as_of: datetime) -> OHLCVBar | None:
    """The in-progress minute. Entry timing only -- never an indicator input."""
    moment = _as_utc(as_of) or datetime.now(timezone.utc)
    current_minute = floor_to_minute(moment, 1)
    for row in rows or ():
        bar = _to_ohlcv(row, symbol)
        if bar is not None and bar.as_of == current_minute:
            return bar
    return None


def load_causal_bars(
    store: Any,
    symbol: str,
    *,
    as_of: datetime | None = None,
    timeframe_minutes: int = 1,
    lookback_minutes: int = 240,
    warmup_required: int = 0,
) -> CausalBarSet:
    """Store-backed convenience wrapper. Returns an empty set on any store error.

    Failing soft is deliberate: an indicator that cannot be computed must degrade
    to "unavailable" (and be reported as such by the availability mask), never
    take down the trading loop.
    """
    moment = _as_utc(as_of) or datetime.now(timezone.utc)
    since = moment - timedelta(minutes=max(1, int(lookback_minutes)))
    try:
        rows = store.recent_minute_bars(symbol, since, limit=max(1, lookback_minutes))
    except Exception:  # noqa: BLE001 - see docstring.
        rows = ()
    return completed_bars(
        rows,
        symbol=symbol,
        as_of=moment,
        timeframe_minutes=timeframe_minutes,
        warmup_required=warmup_required,
    )

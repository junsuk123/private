"""Causal session features shared by training and live serving.

The functions in this module are the single calculation authority for the
price-derived fields used by session-boxed strategies.  Callers may adapt their
bar objects, but must not reimplement the calculations — including the session
clock itself.

Why the clock lives here
------------------------
Every caller used to carry its own ``09:00 Asia/Seoul`` constant.  That is right
for KRX and silently wrong for every US symbol: the US regular session opens at
09:30 America/New_York, which is 22:30 Seoul, so a US symbol's opening range was
searched for in a window where the US market is shut.  ``opening_range`` then
returned ``None`` for every US symbol on every day, and the five session-anchored
strategies — ``market_intraday_momentum`` and its short twin,
``opening_range_breakout``/``_breakdown`` and ``gap_context`` — reported
``STRUCTURALLY_UNREACHABLE:CONTEXT_UNAVAILABLE`` in every strategy-utility
checkpoint: 0 triggers out of 2,330 US labels.  Those are precisely the
low-turnover theses whose one-round-trip-per-day cost profile is the only shape
that can survive a ~51bp US round trip, so the hardcoded constant was disabling
the only family the US cost structure permits.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import math
from statistics import fmean
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class OpeningRange:
    high: float
    low: float
    close: float
    volatility: float
    minutes: int
    samples: int


#: Continuous-trading windows, used when the capability service cannot be read.
#: Kept identical to ``SessionId.KRX_REGULAR`` / ``SessionId.US_REGULAR`` in
#: ``app.data.market_capabilities``, which remains the authority when available.
#: KRX ends at 15:20 because 15:20-15:30 is a closing single-price auction — a
#: different matching mechanism that nothing here models.
_FALLBACK_REGULAR_WINDOWS: dict[str, tuple[str, time, time]] = {
    "KR": ("Asia/Seoul", time(9, 0), time(15, 20)),
    "US": ("America/New_York", time(9, 30), time(16, 0)),
}


@dataclass(frozen=True)
class RegularSession:
    """The continuous-trading window a symbol's session features are boxed by."""

    market_group: str
    tz: str
    open_time: time
    close_time: time

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.tz)

    def session_open(self, moment: datetime) -> datetime:
        """Open of the session ``moment`` belongs to, in exchange-local time.

        Before today's open the current session is still yesterday's, so a US
        symbol observed at 02:00 Seoul (13:00 New York, previous day) is anchored
        to that day's 09:30 open rather than to a window that has not happened.
        """
        local = moment.astimezone(self.zone)
        opened = local.replace(
            hour=self.open_time.hour,
            minute=self.open_time.minute,
            second=0,
            microsecond=0,
        )
        if local < opened:
            opened = (local - timedelta(days=1)).replace(
                hour=self.open_time.hour,
                minute=self.open_time.minute,
                second=0,
                microsecond=0,
            )
        return opened

    def trading_day(self, moment: datetime) -> date:
        """Exchange-local calendar day of the session ``moment`` belongs to.

        Not the UTC day and not the Seoul day: a US session runs 22:30-05:00
        Seoul time and therefore straddles two Seoul dates, which is what made
        "bars from the same session" an unusable grouping key for US symbols.
        """
        return self.session_open(moment).date()

    def continuous_close(self, moment: datetime) -> datetime:
        opened = self.session_open(moment)
        return opened.replace(
            hour=self.close_time.hour,
            minute=self.close_time.minute,
            second=0,
            microsecond=0,
        )

    def minutes_to_continuous_close(self, moment: datetime) -> float:
        delta = self.continuous_close(moment) - moment.astimezone(self.zone)
        return delta.total_seconds() / 60.0

    def in_last_continuous_half_hour(self, moment: datetime, *, minutes: int = 30) -> bool:
        remaining = self.minutes_to_continuous_close(moment)
        return 0.0 < remaining <= float(max(1, int(minutes)))


def market_group_for_symbol(symbol: Any) -> str:
    """``"KR"`` for six-digit KRX codes, ``"US"`` otherwise."""
    text = str(symbol or "").strip().upper()
    return "KR" if text.isdigit() and len(text) == 6 else "US"


def regular_session(symbol: Any) -> RegularSession:
    """Continuous-trading window for the market this symbol trades on."""
    group = market_group_for_symbol(symbol)
    tz, open_time, close_time = _FALLBACK_REGULAR_WINDOWS[group]
    window = _capability_window(group)
    if window is not None:
        tz, open_time, close_time = window
    return RegularSession(
        market_group=group,
        tz=tz,
        open_time=open_time,
        close_time=close_time,
    )


def _capability_window(group: str) -> tuple[str, time, time] | None:
    """Read the configured window, or ``None`` when the service is unusable.

    Imported lazily and defensively: session features must keep building from
    the fallback constants above even if the capability config is missing, and a
    module-level import would tie this leaf calculation module to config loading.
    """
    try:
        from app.data.market_capabilities import SessionId, default_service

        session = SessionId.KRX_REGULAR if group == "KR" else SessionId.US_REGULAR
        window = default_service().window(session)
    except Exception:  # noqa: BLE001 - config problems must not disable features.
        return None
    if window is None:
        return None
    return str(window.tz), window.start, window.end


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

"""The top of the context hierarchy: what *kind of time* it is right now.

``Calendar/Session -> Global -> Domestic -> Sector -> Stock``. Everything below this
module is conditioned on the answer it produces, because the same order-flow imbalance
means different things at 09:01 on the Monday after a long weekend and at 13:30 on an
ordinary Wednesday.

The one rule this module exists to enforce
------------------------------------------
**Day-of-week and clock position never become a trading rule.** No consumer may branch on
"it is Friday" or "it is 14:00" to buy or sell. They are emitted here as *features*, and
the only path from a feature to a decision runs through
:mod:`app.features.seasonality`, which normalises each one against a rolling baseline
conditioned on (day, session phase, regime). A fixed weekday rule cannot survive a regime
change; a z-score against a rolling baseline reports when the current reading is unusual
*for this kind of time*, which is the claim actually worth acting on.

Everything is derived from the exchange calendar in
:class:`~app.data.market_capabilities.MarketSessionService`: holidays, early closes, DST
and the session windows. No time literal appears in this file.
"""

from __future__ import annotations

import calendar as _calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

from app.config.temporal_config import TemporalConfig, default_temporal_config
from app.context.session_phase import (
    SessionPhase,
    SessionPhaseState,
    resolve_session_phase,
)
from app.data.market_capabilities import (
    MarketGroup,
    MarketSessionService,
    ReasonCode,
    default_service,
    normalize_market_group,
)

__all__ = [
    "DISPLAY_TIMEZONE",
    "ExpiryContext",
    "TemporalSnapshot",
    "build_temporal_snapshot",
    "expiry_date_for",
    "next_trading_day",
    "previous_trading_day",
]

#: Storage is UTC everywhere; this is the operator-facing rendering timezone only.
DISPLAY_TIMEZONE = "Asia/Seoul"

_EXCHANGE_TIMEZONE = {MarketGroup.KR: "Asia/Seoul", MarketGroup.US: "America/New_York"}

_DAY_NAMES = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")


class ExpiryContext(str, Enum):
    """Proximity to a derivatives expiry, as context rather than as a signal."""

    NONE = "NONE"
    #: Within ``adjacent_trading_days`` trading days of a monthly expiry.
    EXPIRY_ADJACENT = "EXPIRY_ADJACENT"
    MONTHLY_EXPIRY = "MONTHLY_EXPIRY"
    QUARTERLY_EXPIRY = "QUARTERLY_EXPIRY"


@dataclass(frozen=True)
class TemporalSnapshot:
    """One market group's temporal state at an instant.

    Timestamps are UTC. ``local_time`` and ``display_time`` are renderings, never the
    stored value — the storage timezone rule is "UTC in the row, timezone at the edge".
    """

    market_group: str
    as_of: datetime
    exchange_timezone: str
    local_time: datetime
    display_time: datetime
    trading_day: date | None

    # -- clock position -------------------------------------------------- #
    day_of_week: int
    day_of_week_name: str
    session_phase: SessionPhase
    session_progress: float | None
    minutes_from_open: float | None
    minutes_to_close: float | None

    # -- calendar position ----------------------------------------------- #
    is_trading_day: bool
    days_since_last_session: int | None
    days_to_next_session: int | None
    holiday_adjacent: bool
    month_end: bool
    quarter_end: bool
    expiry_context: ExpiryContext

    phase_state: SessionPhaseState
    calendar_reasons: tuple[str, ...] = ()

    @property
    def seasonality_bucket_day(self) -> str:
        """Day component of the seasonality bucket key."""
        return self.day_of_week_name

    def numeric_features(self) -> dict[str, float]:
        """The model-input view. Absent values are omitted, never defaulted to zero."""
        values: dict[str, float] = {
            "day_of_week": float(self.day_of_week),
            "holiday_adjacent": 1.0 if self.holiday_adjacent else 0.0,
            "month_end": 1.0 if self.month_end else 0.0,
            "quarter_end": 1.0 if self.quarter_end else 0.0,
            "is_trading_day": 1.0 if self.is_trading_day else 0.0,
            "expiry_monthly": 1.0
            if self.expiry_context
            in (ExpiryContext.MONTHLY_EXPIRY, ExpiryContext.QUARTERLY_EXPIRY)
            else 0.0,
            "expiry_quarterly": 1.0
            if self.expiry_context is ExpiryContext.QUARTERLY_EXPIRY
            else 0.0,
            "expiry_adjacent": 1.0
            if self.expiry_context is ExpiryContext.EXPIRY_ADJACENT
            else 0.0,
        }
        for name, value in (
            ("session_progress", self.session_progress),
            ("minutes_from_open", self.minutes_from_open),
            ("minutes_to_close", self.minutes_to_close),
            ("days_since_last_session", self.days_since_last_session),
            ("days_to_next_session", self.days_to_next_session),
        ):
            if value is not None:
                values[name] = float(value)
        return values

    def as_dict(self) -> dict[str, Any]:
        return {
            "market_group": self.market_group,
            "as_of": self.as_of.isoformat(),
            "exchange_timezone": self.exchange_timezone,
            "local_time": self.local_time.isoformat(),
            "display_time": self.display_time.isoformat(),
            "display_timezone": DISPLAY_TIMEZONE,
            "trading_day": self.trading_day.isoformat() if self.trading_day else None,
            "day_of_week": self.day_of_week,
            "day_of_week_name": self.day_of_week_name,
            "session_phase": self.session_phase.value,
            "session_progress": self.session_progress,
            "minutes_from_open": self.minutes_from_open,
            "minutes_to_close": self.minutes_to_close,
            "is_trading_day": self.is_trading_day,
            "days_since_last_session": self.days_since_last_session,
            "days_to_next_session": self.days_to_next_session,
            "holiday_adjacent": self.holiday_adjacent,
            "month_end": self.month_end,
            "quarter_end": self.quarter_end,
            "expiry_context": self.expiry_context.value,
            "calendar_reasons": list(self.calendar_reasons),
            "session": self.phase_state.as_dict(),
        }


def _as_utc(moment: datetime | None) -> datetime:
    current = moment or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


def _noon(day: date, zone: ZoneInfo) -> datetime:
    """Midday on ``day`` in ``zone``, as UTC.

    Trading-day questions are asked at midday so that a DST transition or a UTC-offset
    rollover cannot move the probe into the neighbouring local date.
    """
    return datetime(day.year, day.month, day.day, 12, 0, tzinfo=zone).astimezone(
        timezone.utc
    )


def previous_trading_day(
    market: MarketGroup,
    day: date,
    *,
    service: MarketSessionService | None = None,
    max_lookback_days: int = 21,
) -> date | None:
    """Latest trading day strictly before ``day``, or ``None`` within the bound."""
    resolver = service or default_service()
    zone = ZoneInfo(_EXCHANGE_TIMEZONE[market])
    for offset in range(1, max_lookback_days + 1):
        candidate = day - timedelta(days=offset)
        if resolver.is_trading_day(market, _noon(candidate, zone)):
            return candidate
    return None


def next_trading_day(
    market: MarketGroup,
    day: date,
    *,
    service: MarketSessionService | None = None,
    max_lookahead_days: int = 21,
) -> date | None:
    """Earliest trading day strictly after ``day``, or ``None`` within the bound."""
    resolver = service or default_service()
    zone = ZoneInfo(_EXCHANGE_TIMEZONE[market])
    for offset in range(1, max_lookahead_days + 1):
        candidate = day + timedelta(days=offset)
        if resolver.is_trading_day(market, _noon(candidate, zone)):
            return candidate
    return None


def expiry_date_for(
    market: MarketGroup,
    year: int,
    month: int,
    *,
    config: TemporalConfig | None = None,
) -> date | None:
    """The Nth-weekday-of-month derivatives expiry date, or ``None`` when unconfigured.

    The calendar date is returned unadjusted. Whether the exchange rolls a holiday expiry
    forward or back is a market-specific rule this project has no verified source for, so
    the *proximity* classification below treats an expiry that falls on a non-trading day
    as adjacent to its neighbouring sessions rather than inventing a rolled date.
    """
    rule = (config or default_temporal_config()).expiry.rule_for(market.value)
    if rule is None:
        return None
    first = date(year, month, 1)
    offset = (rule.weekday - first.weekday()) % 7
    day_number = 1 + offset + (rule.ordinal - 1) * 7
    if day_number > _calendar.monthrange(year, month)[1]:
        return None
    return date(year, month, day_number)


def _expiry_context(
    market: MarketGroup,
    trading_day: date,
    *,
    service: MarketSessionService,
    config: TemporalConfig,
) -> ExpiryContext:
    rule = config.expiry.rule_for(market.value)
    if rule is None:
        return ExpiryContext.NONE
    window = max(0, int(config.expiry.adjacent_trading_days))
    # Neighbouring months are included so an expiry on the 1st or the 31st is still seen
    # from the adjacent side of the month boundary.
    candidates: list[date] = []
    for month_offset in (-1, 0, 1):
        anchor = date(trading_day.year, trading_day.month, 15) + timedelta(
            days=31 * month_offset
        )
        expiry = expiry_date_for(market, anchor.year, anchor.month, config=config)
        if expiry is not None:
            candidates.append(expiry)
    if not candidates:
        return ExpiryContext.NONE

    for expiry in candidates:
        if expiry != trading_day:
            continue
        quarterly = trading_day.month in rule.quarterly_months
        return (
            ExpiryContext.QUARTERLY_EXPIRY if quarterly else ExpiryContext.MONTHLY_EXPIRY
        )

    if window == 0:
        return ExpiryContext.NONE
    lower, upper = _adjacent_span(
        market,
        trading_day,
        window,
        service=service,
        max_span_days=config.calendar.max_lookback_days,
    )
    for expiry in candidates:
        if lower <= expiry <= upper:
            return ExpiryContext.EXPIRY_ADJACENT
    return ExpiryContext.NONE


def _adjacent_span(
    market: MarketGroup,
    day: date,
    count: int,
    *,
    service: MarketSessionService,
    max_span_days: int,
) -> tuple[date, date]:
    """Inclusive calendar span covering ``count`` trading days either side of ``day``.

    Returned as a date range rather than a set of trading days so that a holiday inside
    the span counts as adjacent: an expiry that lands on a non-trading day still shapes
    the sessions on both sides of it.
    """
    lower = day
    for _ in range(count):
        found = previous_trading_day(
            market, lower, service=service, max_lookback_days=max_span_days
        )
        if found is None:
            break
        lower = found
    upper = day
    for _ in range(count):
        found = next_trading_day(
            market, upper, service=service, max_lookahead_days=max_span_days
        )
        if found is None:
            break
        upper = found
    return lower, upper


def _is_last_trading_day_of(
    market: MarketGroup,
    trading_day: date,
    *,
    service: MarketSessionService,
    quarter: bool,
    max_lookahead_days: int,
) -> bool:
    upcoming = next_trading_day(
        market, trading_day, service=service, max_lookahead_days=max_lookahead_days
    )
    if upcoming is None:
        return False
    if quarter:
        return _quarter_of(upcoming) != _quarter_of(trading_day)
    return (upcoming.year, upcoming.month) != (trading_day.year, trading_day.month)


def _quarter_of(day: date) -> tuple[int, int]:
    return day.year, (day.month - 1) // 3


def _closed_weekdays_between(start: date | None, end: date | None) -> int:
    """Weekdays strictly between two trading days that the market was shut.

    A weekend is NOT a holiday. Counting raw calendar days would flag every Friday and
    every Monday as holiday-adjacent, which makes the feature a restatement of
    ``day_of_week`` and gives a seasonality bucket nothing to condition on. Only Mon-Fri
    dates on which no session ran are counted, so the flag fires on a genuine market
    closure — a national holiday, a 대체공휴일, an exchange suspension.
    """
    if start is None or end is None or end <= start:
        return 0
    closed = 0
    cursor = start + timedelta(days=1)
    while cursor < end:
        if cursor.weekday() < 5:
            closed += 1
        cursor += timedelta(days=1)
    return closed


def build_temporal_snapshot(
    group: str | MarketGroup,
    now_utc: datetime | None = None,
    *,
    service: MarketSessionService | None = None,
    config: TemporalConfig | None = None,
) -> TemporalSnapshot:
    """Full temporal state for one market group.

    Fail-soft by construction: an unknown group or an out-of-coverage calendar yields a
    snapshot whose ``session_phase`` is ``CLOSED`` and whose ``calendar_reasons`` say why.
    Downstream gates reject on those reasons; nothing here guesses a session.
    """
    current = _as_utc(now_utc)
    settings = config or default_temporal_config()
    resolver = service or default_service()
    market = group if isinstance(group, MarketGroup) else normalize_market_group(str(group))
    display_zone = ZoneInfo(DISPLAY_TIMEZONE)

    if market is None:
        phase_state = resolve_session_phase(group, current, service=resolver)
        return TemporalSnapshot(
            market_group=str(group or "").upper(),
            as_of=current,
            exchange_timezone="UTC",
            local_time=current,
            display_time=current.astimezone(display_zone),
            trading_day=None,
            day_of_week=current.weekday(),
            day_of_week_name=_DAY_NAMES[current.weekday()],
            session_phase=SessionPhase.CLOSED,
            session_progress=None,
            minutes_from_open=None,
            minutes_to_close=None,
            is_trading_day=False,
            days_since_last_session=None,
            days_to_next_session=None,
            holiday_adjacent=False,
            month_end=False,
            quarter_end=False,
            expiry_context=ExpiryContext.NONE,
            phase_state=phase_state,
            calendar_reasons=(ReasonCode.MARKET_SESSION_UNKNOWN.value,),
        )

    exchange_tz = _EXCHANGE_TIMEZONE[market]
    zone = ZoneInfo(exchange_tz)
    local = current.astimezone(zone)
    phase_state = resolve_session_phase(
        market, current, service=resolver, config=settings.session_phase
    )

    # The trading day is the exchange-local date of the session the phase is anchored to,
    # not the local date of "now". A US session runs 22:30-05:00 Seoul and therefore
    # straddles two Seoul dates; anchoring to the session keeps one arc on one key.
    anchor = phase_state.continuous_open or current
    trading_day = anchor.astimezone(zone).date()
    # ``is_trading_day`` answers "does the market run today", asked of NOW — not of the
    # anchored session, which is a trading day by construction and would therefore report
    # True right through a Saturday.
    day_is_tradeable = phase_state.is_trading_day

    lookback = settings.calendar.max_lookback_days
    previous = previous_trading_day(
        market, trading_day, service=resolver, max_lookback_days=lookback
    )
    upcoming = next_trading_day(
        market, trading_day, service=resolver, max_lookahead_days=lookback
    )
    days_since = (trading_day - previous).days if previous is not None else None
    days_to = (upcoming - trading_day).days if upcoming is not None else None
    threshold = max(1, int(settings.calendar.holiday_gap_days))
    closed_weekdays = max(
        _closed_weekdays_between(previous, trading_day),
        _closed_weekdays_between(trading_day, upcoming),
    )
    holiday_adjacent = closed_weekdays >= threshold

    return TemporalSnapshot(
        market_group=market.value,
        as_of=current,
        exchange_timezone=exchange_tz,
        local_time=local,
        display_time=current.astimezone(display_zone),
        trading_day=trading_day,
        day_of_week=trading_day.weekday(),
        day_of_week_name=_DAY_NAMES[trading_day.weekday()],
        session_phase=phase_state.phase,
        session_progress=phase_state.session_progress,
        minutes_from_open=phase_state.minutes_from_open,
        minutes_to_close=phase_state.minutes_to_close,
        is_trading_day=day_is_tradeable,
        days_since_last_session=days_since,
        days_to_next_session=days_to,
        holiday_adjacent=holiday_adjacent,
        month_end=_is_last_trading_day_of(
            market,
            trading_day,
            service=resolver,
            quarter=False,
            max_lookahead_days=lookback,
        ),
        quarter_end=_is_last_trading_day_of(
            market,
            trading_day,
            service=resolver,
            quarter=True,
            max_lookahead_days=lookback,
        ),
        expiry_context=_expiry_context(
            market, trading_day, service=resolver, config=settings
        ),
        phase_state=phase_state,
        calendar_reasons=phase_state.calendar_reasons,
    )

"""Nine-phase intraday session taxonomy, resolved from the exchange calendar.

Why nine and not four
---------------------
``app.data.market_session.MarketPhase`` collapses a trading day into PRE / REGULAR /
AFTER / CLOSED. That is enough to decide "may an order be routed", which is what it was
built for, and useless as a *model input*: the first two minutes after the KRX opening
auction and 13:00 on the same day are both ``REGULAR``, and nothing about the spread, the
trade rate or the flow composition is comparable between them. A seasonality baseline
conditioned on ``REGULAR`` averages those two tapes together and reports the mean of two
distributions that never overlap.

The phases here are therefore about *where in the session's own arc* the market is:

============================  ==============================================
``PRE_MARKET``                a pre-open / pre-market / daytime session is
                              open; the continuous session has not started
``OPEN_TRANSITION``           first minutes of continuous trading, while the
                              auction imbalance is still resolving
``OPENING``                   opening range, out to ``opening_minutes``
``MORNING_TREND``             out to ``morning_end_fraction`` of the session
``MIDDAY``                    out to ``midday_end_fraction``
``AFTERNOON``                 until the closing window begins
``CLOSING``                   last ``closing_minutes``, plus the closing
                              auction session
``POST_MARKET``               after-hours / single-price / NXT post session
``CLOSED``                    no session of any kind is open
============================  ==============================================

Boundaries are minutes and fractions **relative to the calendar-resolved open and close**
(``config/temporal_context.yaml``), never wall-clock literals, so an early close, a DST
shift or a config change to the session windows moves them without a code change.

Authority
---------
Session membership comes from :class:`~app.data.market_capabilities.MarketSessionService`
— the same service the order router gates on. This module adds no second clock and no
second holiday list; it only projects that service's answer onto the phase arc.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any

from app.config.temporal_config import SessionPhaseConfig, default_temporal_config
from app.data.market_capabilities import (
    MarketGroup,
    MarketSessionService,
    ReasonCode,
    SessionId,
    SessionWindow,
    default_service,
    normalize_market_group,
)

__all__ = [
    "CONTINUOUS_SESSIONS",
    "SessionPhase",
    "SessionPhaseState",
    "resolve_session_phase",
]


class SessionPhase(str, Enum):
    PRE_MARKET = "PRE_MARKET"
    OPEN_TRANSITION = "OPEN_TRANSITION"
    OPENING = "OPENING"
    MORNING_TREND = "MORNING_TREND"
    MIDDAY = "MIDDAY"
    AFTERNOON = "AFTERNOON"
    CLOSING = "CLOSING"
    POST_MARKET = "POST_MARKET"
    CLOSED = "CLOSED"


#: Phases in which the continuous order book is matching. Callers that need "is the tape
#: continuous right now" ask this rather than re-listing phases.
CONTINUOUS_PHASES: frozenset[SessionPhase] = frozenset(
    {
        SessionPhase.OPEN_TRANSITION,
        SessionPhase.OPENING,
        SessionPhase.MORNING_TREND,
        SessionPhase.MIDDAY,
        SessionPhase.AFTERNOON,
        SessionPhase.CLOSING,
    }
)

#: The continuous-trading session per market group — the arc every phase is measured on.
#: NXT_REGULAR shares KRX_REGULAR's window, so KRX is the domestic reference; using both
#: would make the phase depend on which venue happened to be queried.
CONTINUOUS_SESSIONS: dict[MarketGroup, SessionId] = {
    MarketGroup.KR: SessionId.KRX_REGULAR,
    MarketGroup.US: SessionId.US_REGULAR,
}

_CONTINUOUS_SESSION_IDS: frozenset[SessionId] = frozenset(
    {SessionId.KRX_REGULAR, SessionId.NXT_REGULAR, SessionId.US_REGULAR}
)
_CLOSING_AUCTION_SESSION_IDS: frozenset[SessionId] = frozenset(
    {SessionId.KRX_CLOSING_AUCTION}
)
_PRE_SESSION_IDS: frozenset[SessionId] = frozenset(
    {
        SessionId.KRX_PREOPEN,
        SessionId.KRX_OPENING_AUCTION,
        SessionId.NXT_PRE,
        SessionId.US_PREMARKET,
        SessionId.US_DAYTIME,
    }
)
_POST_SESSION_IDS: frozenset[SessionId] = frozenset(
    {
        SessionId.KRX_AFTER_CLOSE,
        SessionId.KRX_AFTER_SINGLE_PRICE,
        SessionId.NXT_POST,
        SessionId.US_AFTERMARKET,
    }
)


@dataclass(frozen=True)
class SessionPhaseState:
    """Where the clock sits on one market group's session arc.

    ``continuous_open`` / ``continuous_close`` are the UTC bounds of the session the
    phase is measured against: the live one while the tape is continuous, the *next* one
    while in ``PRE_MARKET``, and the one that just ended afterwards. Anchoring pre-market
    to the next open is what makes ``minutes_from_open`` negative there rather than a
    large positive number left over from yesterday.
    """

    market_group: str
    phase: SessionPhase
    as_of: datetime
    primary_session: str
    active_sessions: tuple[str, ...]
    is_trading_day: bool
    continuous_open: datetime | None = None
    continuous_close: datetime | None = None
    minutes_from_open: float | None = None
    minutes_to_close: float | None = None
    #: Position on the continuous arc in [0, 1]. ``None`` outside a resolvable session.
    session_progress: float | None = None
    calendar_reasons: tuple[str, ...] = ()

    @property
    def is_continuous(self) -> bool:
        return self.phase in CONTINUOUS_PHASES

    def as_dict(self) -> dict[str, Any]:
        return {
            "market_group": self.market_group,
            "phase": self.phase.value,
            "as_of": self.as_of.isoformat(),
            "primary_session": self.primary_session,
            "active_sessions": list(self.active_sessions),
            "is_trading_day": self.is_trading_day,
            "continuous_open": (
                self.continuous_open.isoformat() if self.continuous_open else None
            ),
            "continuous_close": (
                self.continuous_close.isoformat() if self.continuous_close else None
            ),
            "minutes_from_open": self.minutes_from_open,
            "minutes_to_close": self.minutes_to_close,
            "session_progress": self.session_progress,
            "calendar_reasons": list(self.calendar_reasons),
            "is_continuous": self.is_continuous,
        }


def _as_utc(moment: datetime | None) -> datetime:
    current = moment or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


def _window_bounds_on(
    window: SessionWindow,
    day: date,
    *,
    service: MarketSessionService,
    market: MarketGroup,
) -> tuple[datetime, datetime]:
    """UTC bounds of ``window`` on the given window-local calendar day.

    The end time is read through :meth:`SessionWindow.end_at`, so a window whose close
    moves with DST (the US after-market) resolves against that day's offset rather than
    against today's. An early close on that day shortens the window here for the same
    reason it does in ``MarketSessionService._build_capability``: a half-day whose phases
    were laid out over a full session would report MIDDAY through the closing auction.
    """
    zone = window.zone
    start_local = datetime.combine(day, window.start, tzinfo=zone)
    end_time = window.end_at(start_local)
    early = service.calendar.early_close_time(market, day)
    if early is not None and early < end_time:
        end_time = early
    end_local = datetime.combine(day, end_time, tzinfo=zone)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def _continuous_bounds(
    service: MarketSessionService,
    market: MarketGroup,
    now_utc: datetime,
    *,
    prefer_next: bool,
    max_scan_days: int,
) -> tuple[datetime, datetime] | None:
    """Bounds of the continuous session the phase is measured against.

    ``prefer_next`` picks the upcoming session (pre-market anchoring); otherwise the
    current one, falling back to the most recent past one. Returns ``None`` when the
    calendar cannot answer within ``max_scan_days`` — an unresolved anchor is reported
    as absent rather than guessed.
    """
    session = CONTINUOUS_SESSIONS.get(market)
    if session is None:
        return None
    window = service.window(session)
    if window is None:
        return None
    anchor_day = now_utc.astimezone(window.zone).date()
    step = 1 if prefer_next else -1
    # Today first, then outward in the requested direction.
    for offset in range(0, max_scan_days + 1):
        day = anchor_day + timedelta(days=offset * step)
        start_utc, end_utc = _window_bounds_on(
            window, day, service=service, market=market
        )
        if not service.is_trading_day(market, start_utc):
            continue
        if prefer_next and end_utc <= now_utc:
            continue
        if not prefer_next and start_utc > now_utc:
            continue
        return start_utc, end_utc
    return None


def _phase_from_progress(
    minutes_from_open: float,
    minutes_to_close: float,
    session_minutes: float,
    config: SessionPhaseConfig,
) -> SessionPhase:
    if minutes_to_close <= config.closing_minutes:
        return SessionPhase.CLOSING
    if minutes_from_open < config.open_transition_minutes:
        return SessionPhase.OPEN_TRANSITION
    if minutes_from_open <= config.opening_minutes:
        return SessionPhase.OPENING
    progress = minutes_from_open / session_minutes if session_minutes > 0 else 1.0
    if progress <= config.morning_end_fraction:
        return SessionPhase.MORNING_TREND
    if progress <= config.midday_end_fraction:
        return SessionPhase.MIDDAY
    return SessionPhase.AFTERNOON


def resolve_session_phase(
    group: str | MarketGroup,
    now_utc: datetime | None = None,
    *,
    service: MarketSessionService | None = None,
    config: SessionPhaseConfig | None = None,
) -> SessionPhaseState:
    """Current phase for a market group.

    Never raises for an unknown group: an unrecognised name resolves to ``CLOSED`` with
    ``MARKET_SESSION_UNKNOWN``, which is the fail-closed answer every downstream gate
    already knows how to reject.
    """
    current = _as_utc(now_utc)
    phase_config = config or default_temporal_config().session_phase
    max_scan_days = default_temporal_config().calendar.max_lookback_days
    resolver = service or default_service()
    market = group if isinstance(group, MarketGroup) else normalize_market_group(str(group))
    if market is None:
        return SessionPhaseState(
            market_group=str(group or "").upper(),
            phase=SessionPhase.CLOSED,
            as_of=current,
            primary_session=SessionId.UNKNOWN.value,
            active_sessions=(),
            is_trading_day=False,
            calendar_reasons=(ReasonCode.MARKET_SESSION_UNKNOWN.value,),
        )

    active = resolver.active_capabilities(market, current)
    active_ids = tuple(item.session for item in active)
    calendar_reasons = resolver.calendar_state(market, current)
    trading_day = resolver.is_trading_day(market, current)

    continuous = [item for item in active if item.session in _CONTINUOUS_SESSION_IDS]
    closing_auction = [
        item for item in active if item.session in _CLOSING_AUCTION_SESSION_IDS
    ]
    pre = [item for item in active if item.session in _PRE_SESSION_IDS]
    post = [item for item in active if item.session in _POST_SESSION_IDS]

    if continuous:
        primary = max(continuous, key=lambda item: item.source_quality)
        prefer_next = False
    elif closing_auction:
        primary = closing_auction[0]
        prefer_next = False
    elif pre:
        primary = max(pre, key=lambda item: item.source_quality)
        prefer_next = True
    elif post:
        primary = max(post, key=lambda item: item.source_quality)
        prefer_next = False
    else:
        primary = None
        prefer_next = False

    bounds = _continuous_bounds(
        resolver,
        market,
        current,
        prefer_next=prefer_next,
        max_scan_days=max_scan_days,
    )
    minutes_from_open: float | None = None
    minutes_to_close: float | None = None
    progress: float | None = None
    open_utc: datetime | None = None
    close_utc: datetime | None = None
    session_minutes = 0.0
    if bounds is not None:
        open_utc, close_utc = bounds
        session_minutes = (close_utc - open_utc).total_seconds() / 60.0
        minutes_from_open = (current - open_utc).total_seconds() / 60.0
        minutes_to_close = (close_utc - current).total_seconds() / 60.0

    if primary is None:
        phase = SessionPhase.CLOSED
    elif continuous and minutes_from_open is not None and minutes_to_close is not None:
        phase = _phase_from_progress(
            minutes_from_open, minutes_to_close, session_minutes, phase_config
        )
    elif continuous:
        # The tape is continuous but the calendar could not anchor it. Reporting a
        # specific arc position here would be an invention; OPEN_TRANSITION is the
        # conservative label (widest spreads, least trusted micro state).
        phase = SessionPhase.OPEN_TRANSITION
    elif closing_auction:
        phase = SessionPhase.CLOSING
    elif pre:
        phase = SessionPhase.PRE_MARKET
    else:
        phase = SessionPhase.POST_MARKET

    # ``session_progress`` measures position on the continuous arc, so it is only
    # defined while that arc is running. Reporting 1.0 through every after-hours and
    # overnight observation would hand a model a saturated constant and let a
    # seasonality bucket built on it average the close together with 03:00.
    if (
        phase in CONTINUOUS_PHASES
        and minutes_from_open is not None
        and session_minutes > 0
    ):
        progress = min(1.0, max(0.0, minutes_from_open / session_minutes))

    return SessionPhaseState(
        market_group=market.value,
        phase=phase,
        as_of=current,
        primary_session=(primary.session.value if primary else _closed_session(market)),
        active_sessions=tuple(item.value for item in active_ids),
        is_trading_day=trading_day,
        continuous_open=open_utc,
        continuous_close=close_utc,
        minutes_from_open=minutes_from_open,
        minutes_to_close=minutes_to_close,
        session_progress=progress,
        calendar_reasons=calendar_reasons,
    )


def _closed_session(market: MarketGroup) -> str:
    return (
        SessionId.KR_CLOSED.value
        if market is MarketGroup.KR
        else SessionId.US_CLOSED.value
    )

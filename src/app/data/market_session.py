"""Canonical intraday market-phase classifier.

Prior to this module, session state was only expressed as scattered boolean
helpers (``_is_live_market_core_open`` / ``_is_live_market_extended_open`` in
``web.py``, ``_is_krx_core_buy_session`` in the realtime engine, etc.), each with
slightly different boundaries and none of which distinguished *pre-market* from
*after-market* from *fully closed*.

This module gives a single, pure, fully-testable classifier that maps a market
group to one of four phases — PRE / REGULAR / AFTER / CLOSED — using
exchange-local time. It is the source of truth for the "fully closed" fallback:
when a group is CLOSED the realtime WebSocket delivers no ticks, so callers fall
back to REST snapshot polling (see ``app.data.rest_snapshot_fallback``).

Boundaries intentionally mirror the extended-session windows already used by
``web.py`` so behavior stays consistent:

* KRX  — regular 09:00-15:30 KST; pre 08:30-09:00; after 15:30-18:00 (weekdays).
* US   — regular 09:30-16:00 ET;  pre 04:00-09:30; after 16:00-20:00 (weekdays,
         excluding NYSE/Nasdaq full holidays).
"""

from __future__ import annotations

from datetime import datetime, time, timezone
from enum import Enum
from zoneinfo import ZoneInfo

_SEOUL = ZoneInfo("Asia/Seoul")
_NEW_YORK = ZoneInfo("America/New_York")

_KRX_GROUP_NAMES = {"KRX", "KR", "KOSPI", "KOSDAQ", "KONEX"}
_US_GROUP_NAMES = {"US", "USA", "NASDAQ", "NASD", "NAS", "NYSE", "NYS", "AMEX", "AMS", "OVERSEAS"}

# NYSE/Nasdaq full-market holidays. Kept in sync with web.py::_is_us_market_holiday.
_US_MARKET_HOLIDAYS = frozenset(
    {
        "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
        "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
        "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
        "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
        "2028-01-17", "2028-02-21", "2028-04-14", "2028-05-29", "2028-07-04",
        "2028-09-04", "2028-11-23", "2028-12-25",
    }
)


class MarketPhase(str, Enum):
    """Intraday session phase for a market group."""

    PRE = "pre"          # 프리마켓 / 장전
    REGULAR = "regular"  # 정규장
    AFTER = "after"      # 애프터마켓 / 장후
    CLOSED = "closed"    # 완전 마감 (프장·정규장·애프터 모두 아님)


def _normalize_group(group: str) -> str:
    name = str(group or "").upper().strip()
    if name in _KRX_GROUP_NAMES:
        return "KRX"
    if name in _US_GROUP_NAMES:
        return "US"
    return name


def _as_utc(now_utc: datetime | None) -> datetime:
    current = now_utc or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current


def _phase_from_windows(
    local_time: time,
    *,
    regular: tuple[time, time],
    pre: tuple[time, time],
    after: tuple[time, time],
) -> MarketPhase:
    if regular[0] <= local_time <= regular[1]:
        return MarketPhase.REGULAR
    if pre[0] <= local_time < pre[1]:
        return MarketPhase.PRE
    if after[0] < local_time <= after[1]:
        return MarketPhase.AFTER
    return MarketPhase.CLOSED


def _krx_phase(now_utc: datetime) -> MarketPhase:
    local = now_utc.astimezone(_SEOUL)
    if local.weekday() >= 5:
        return MarketPhase.CLOSED
    return _phase_from_windows(
        local.time(),
        regular=(time(9, 0), time(15, 30)),
        pre=(time(8, 30), time(9, 0)),
        after=(time(15, 30), time(18, 0)),
    )


def _us_phase(now_utc: datetime) -> MarketPhase:
    eastern = now_utc.astimezone(_NEW_YORK)
    if eastern.weekday() >= 5 or eastern.date().isoformat() in _US_MARKET_HOLIDAYS:
        return MarketPhase.CLOSED
    return _phase_from_windows(
        eastern.time(),
        regular=(time(9, 30), time(16, 0)),
        pre=(time(4, 0), time(9, 30)),
        after=(time(16, 0), time(20, 0)),
    )


def market_phase(group: str, now_utc: datetime | None = None) -> MarketPhase:
    """Classify the current intraday phase for a market group ("KRX" or "US")."""
    normalized = _normalize_group(group)
    current = _as_utc(now_utc)
    if normalized == "KRX":
        return _krx_phase(current)
    if normalized == "US":
        return _us_phase(current)
    return MarketPhase.CLOSED


def is_market_fully_closed(group: str, now_utc: datetime | None = None) -> bool:
    """True when the group has no session at all (not pre, regular, or after)."""
    return market_phase(group, now_utc) is MarketPhase.CLOSED


def streaming_phase(
    group: str,
    now_utc: datetime | None = None,
    *,
    include_nxt: bool = True,
) -> MarketPhase:
    """Phase for DATA STREAMING, which is wider than the order-gating phase.

    NXT (넥스트레이드, the domestic ATS) quotes and trades 08:00-20:00 KST, so
    the 통합 (KRX+NXT) realtime feed delivers ticks well outside the KRX
    09:00-15:30 window. Order gating deliberately does NOT use this: routing an
    order to KRX at 19:00 would simply be rejected, and NXT order routing is a
    separate capability.
    """
    normalized = _normalize_group(group)
    current = _as_utc(now_utc)
    if normalized != "KRX" or not include_nxt:
        return market_phase(group, current)
    local = current.astimezone(_SEOUL)
    if local.weekday() >= 5:
        return MarketPhase.CLOSED
    return _phase_from_windows(
        local.time(),
        regular=(time(9, 0), time(15, 30)),
        pre=(time(8, 0), time(9, 0)),
        after=(time(15, 30), time(20, 0)),
    )


def market_has_live_session(group: str, now_utc: datetime | None = None) -> bool:
    """True when the group is in any tradeable/quotable session (pre/regular/after)."""
    return not is_market_fully_closed(group, now_utc)

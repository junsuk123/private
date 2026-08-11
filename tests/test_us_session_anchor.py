"""The session clock must follow the symbol's exchange, not a hardcoded KRX one.

Every session-boxed strategy read ``09:00 Asia/Seoul`` regardless of market. For
a US symbol that is 20:00 the previous evening in New York, so the opening range
was searched for in a window where the US market is shut and
``market_intraday_momentum``, ``market_intraday_momentum_short``,
``opening_range_breakout``, ``opening_range_breakdown`` and ``gap_context`` all
reported ``STRUCTURALLY_UNREACHABLE:CONTEXT_UNAVAILABLE`` — 0 triggers out of
2,330 US labels in the strategy-utility checkpoint.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.features import session_structure
from app.features.session_structure import market_group_for_symbol, regular_session
from app.trading.strategy_session import _session_structure_context

UTC = timezone.utc


def _bars(start: datetime, count: int, *, price: float = 100.0) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            minute_start=start + timedelta(minutes=index),
            high=price + 0.10,
            low=price - 0.10,
            close=price + 0.01 * (index % 3),
        )
        for index in range(count)
    ]


def test_market_group_is_read_from_the_symbol_shape() -> None:
    assert market_group_for_symbol("005930") == "KR"
    assert market_group_for_symbol("PFE") == "US"
    assert market_group_for_symbol("") == "US"


def test_each_market_gets_its_own_continuous_window() -> None:
    krx = regular_session("005930")
    us = regular_session("PFE")

    assert (krx.tz, krx.open_time.hour, krx.open_time.minute) == ("Asia/Seoul", 9, 0)
    # KRX continuous trading stops at 15:20; 15:20-15:30 is a closing auction.
    assert (krx.close_time.hour, krx.close_time.minute) == (15, 20)
    assert (us.tz, us.open_time.hour, us.open_time.minute) == ("America/New_York", 9, 30)
    assert (us.close_time.hour, us.close_time.minute) == (16, 0)


def test_us_session_open_is_the_new_york_open_not_the_seoul_one() -> None:
    # 13:17 UTC = 09:17 New York (same day) = 22:17 Seoul (same day).
    moment = datetime(2026, 8, 10, 14, 17, tzinfo=UTC)
    session = regular_session("PFE")

    assert session.session_open(moment).isoformat() == "2026-08-10T09:30:00-04:00"
    assert session.trading_day(moment).isoformat() == "2026-08-10"


def test_a_us_session_does_not_split_across_seoul_midnight() -> None:
    """22:30-05:00 Seoul is one New York session, and must group as one."""
    session = regular_session("PFE")
    afternoon = datetime(2026, 8, 10, 14, 0, tzinfo=UTC)   # 23:00 Seoul, 10:00 NY
    after_midnight = datetime(2026, 8, 10, 19, 0, tzinfo=UTC)  # 04:00 Seoul +1d, 15:00 NY

    assert session.trading_day(afternoon) == session.trading_day(after_midnight)


def test_before_the_open_the_current_session_is_the_previous_one() -> None:
    session = regular_session("PFE")
    # 08:00 New York, before the 09:30 open.
    moment = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)

    assert session.trading_day(moment).isoformat() == "2026-08-10"


def test_last_continuous_half_hour_follows_the_symbols_market() -> None:
    us_close = datetime(2026, 8, 10, 19, 45, tzinfo=UTC)   # 15:45 New York
    krx_close = datetime(2026, 8, 10, 6, 0, tzinfo=UTC)    # 15:00 Seoul

    assert regular_session("PFE").in_last_continuous_half_hour(us_close) is True
    assert regular_session("005930").in_last_continuous_half_hour(us_close) is False
    assert regular_session("005930").in_last_continuous_half_hour(krx_close) is True
    assert regular_session("PFE").in_last_continuous_half_hour(krx_close) is False


def test_election_context_window_follows_the_symbols_market() -> None:
    us_close = datetime(2026, 8, 10, 19, 45, tzinfo=UTC)

    us_context = _session_structure_context(us_close, "PFE")
    krx_context = _session_structure_context(us_close, "005930")

    assert us_context["in_last_continuous_half_hour"] is True
    assert us_context["minutes_to_continuous_close"] == 15.0
    assert krx_context["in_last_continuous_half_hour"] is False


def test_us_opening_range_resolves_against_the_new_york_open() -> None:
    """The regression: a US opening range under the KRX anchor was always absent."""
    session = regular_session("PFE")
    open_utc = datetime(2026, 8, 10, 13, 30, tzinfo=UTC)
    bars = _bars(open_utc, 30)
    now = open_utc + timedelta(minutes=45)

    observed = session_structure.opening_range(
        bars, session_open=session.session_open(now), minutes=30, now=now
    )
    assert observed is not None
    assert observed.samples == 30

    krx_anchor = regular_session("005930").session_open(now)
    assert session_structure.opening_range(
        bars, session_open=krx_anchor, minutes=30, now=now
    ) is None

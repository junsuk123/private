"""Temporal context: session phases, calendar position, DST and early closes.

Every expectation below is derived from the exchange calendar in
``config/market_sessions.yaml``, not from wall-clock literals in the code under test.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from app.config.temporal_config import (
    SessionPhaseConfig,
    TemporalConfigError,
    load_temporal_config,
)
from app.context.session_phase import SessionPhase, resolve_session_phase
from app.context.temporal_context import (
    ExpiryContext,
    build_temporal_snapshot,
    expiry_date_for,
    next_trading_day,
    previous_trading_day,
)
from app.data.market_capabilities import MarketGroup

KST = ZoneInfo("Asia/Seoul")
NEW_YORK = ZoneInfo("America/New_York")


def _utc(moment: datetime, zone: ZoneInfo) -> datetime:
    return moment.replace(tzinfo=zone).astimezone(timezone.utc)


def _phase(group: str, moment: datetime, zone: ZoneInfo) -> SessionPhase:
    return resolve_session_phase(group, _utc(moment, zone)).phase


# --------------------------------------------------------------------------- #
# Phase arc
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("local", "expected"),
    [
        (datetime(2026, 8, 19, 7, 0), SessionPhase.CLOSED),
        (datetime(2026, 8, 19, 8, 10), SessionPhase.PRE_MARKET),
        (datetime(2026, 8, 19, 8, 45), SessionPhase.PRE_MARKET),
        (datetime(2026, 8, 19, 9, 1), SessionPhase.OPEN_TRANSITION),
        (datetime(2026, 8, 19, 9, 15), SessionPhase.OPENING),
        (datetime(2026, 8, 19, 10, 30), SessionPhase.MORNING_TREND),
        (datetime(2026, 8, 19, 12, 30), SessionPhase.MIDDAY),
        (datetime(2026, 8, 19, 14, 0), SessionPhase.AFTERNOON),
        (datetime(2026, 8, 19, 15, 10), SessionPhase.CLOSING),
        (datetime(2026, 8, 19, 15, 25), SessionPhase.CLOSING),
        (datetime(2026, 8, 19, 16, 30), SessionPhase.POST_MARKET),
        (datetime(2026, 8, 19, 19, 0), SessionPhase.POST_MARKET),
        (datetime(2026, 8, 19, 21, 0), SessionPhase.CLOSED),
    ],
)
def test_krx_phase_arc(local: datetime, expected: SessionPhase) -> None:
    assert _phase("KRX", local, KST) is expected


@pytest.mark.parametrize(
    ("local", "expected"),
    [
        (datetime(2026, 8, 19, 3, 0), SessionPhase.PRE_MARKET),
        (datetime(2026, 8, 19, 9, 31), SessionPhase.OPEN_TRANSITION),
        (datetime(2026, 8, 19, 9, 45), SessionPhase.OPENING),
        (datetime(2026, 8, 19, 11, 0), SessionPhase.MORNING_TREND),
        (datetime(2026, 8, 19, 13, 0), SessionPhase.MIDDAY),
        (datetime(2026, 8, 19, 15, 0), SessionPhase.AFTERNOON),
        (datetime(2026, 8, 19, 15, 45), SessionPhase.CLOSING),
        (datetime(2026, 8, 19, 16, 30), SessionPhase.POST_MARKET),
        (datetime(2026, 8, 19, 20, 0), SessionPhase.CLOSED),
    ],
)
def test_us_phase_arc(local: datetime, expected: SessionPhase) -> None:
    assert _phase("US", local, NEW_YORK) is expected


def test_saturday_is_closed_in_both_groups() -> None:
    saturday = datetime(2026, 8, 22, 11, 0)
    assert _phase("KRX", saturday, KST) is SessionPhase.CLOSED
    assert _phase("US", saturday, NEW_YORK) is SessionPhase.CLOSED


def test_krx_holiday_has_no_session() -> None:
    # 2026-10-09 한글날 is a Friday holiday in the shipped calendar.
    assert _phase("KRX", datetime(2026, 10, 9, 10, 0), KST) is SessionPhase.CLOSED


def test_unknown_group_fails_closed() -> None:
    state = resolve_session_phase("LSE", datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc))
    assert state.phase is SessionPhase.CLOSED
    assert "MARKET_SESSION_UNKNOWN" in state.calendar_reasons


def test_session_progress_only_defined_on_the_continuous_arc() -> None:
    midday = resolve_session_phase("KRX", _utc(datetime(2026, 8, 19, 12, 30), KST))
    overnight = resolve_session_phase("KRX", _utc(datetime(2026, 8, 19, 21, 0), KST))
    premarket = resolve_session_phase("KRX", _utc(datetime(2026, 8, 19, 8, 10), KST))
    assert midday.session_progress is not None
    assert 0.0 < midday.session_progress < 1.0
    assert overnight.session_progress is None
    assert premarket.session_progress is None


def test_pre_market_anchors_to_the_upcoming_open() -> None:
    state = resolve_session_phase("KRX", _utc(datetime(2026, 8, 19, 8, 10), KST))
    assert state.minutes_from_open is not None
    assert state.minutes_from_open == pytest.approx(-50.0)


# --------------------------------------------------------------------------- #
# DST and early close
# --------------------------------------------------------------------------- #
def test_us_open_tracks_dst_rather_than_a_fixed_utc_hour() -> None:
    """09:30 New York is 13:30 UTC in summer and 14:30 UTC in winter."""
    summer = resolve_session_phase("US", _utc(datetime(2026, 7, 15, 9, 31), NEW_YORK))
    winter = resolve_session_phase("US", _utc(datetime(2026, 1, 15, 9, 31), NEW_YORK))
    assert summer.continuous_open is not None and winter.continuous_open is not None
    assert summer.continuous_open.hour == 13
    assert winter.continuous_open.hour == 14
    assert summer.phase is SessionPhase.OPEN_TRANSITION
    assert winter.phase is SessionPhase.OPEN_TRANSITION


def test_early_close_shortens_the_phase_arc() -> None:
    """2026-11-27 closes at 13:00 ET, so 11:00 is past the halfway point."""
    early = resolve_session_phase("US", _utc(datetime(2026, 11, 27, 11, 0), NEW_YORK))
    ordinary = resolve_session_phase("US", _utc(datetime(2026, 11, 20, 11, 0), NEW_YORK))
    assert early.continuous_close is not None
    assert early.continuous_close.astimezone(NEW_YORK).hour == 13
    assert early.session_progress is not None and ordinary.session_progress is not None
    assert early.session_progress > ordinary.session_progress


def test_early_close_afternoon_becomes_closing() -> None:
    assert _phase("US", datetime(2026, 11, 27, 12, 45), NEW_YORK) is SessionPhase.CLOSING


# --------------------------------------------------------------------------- #
# Calendar position
# --------------------------------------------------------------------------- #
def test_weekend_gap_is_not_holiday_adjacent() -> None:
    friday = build_temporal_snapshot("KRX", _utc(datetime(2026, 8, 21, 10, 0), KST))
    monday = build_temporal_snapshot("KRX", _utc(datetime(2026, 8, 24, 10, 0), KST))
    assert friday.days_to_next_session == 3
    assert monday.days_since_last_session == 3
    assert not friday.holiday_adjacent
    assert not monday.holiday_adjacent


def test_weekday_holiday_makes_both_neighbours_adjacent() -> None:
    before = build_temporal_snapshot("KRX", _utc(datetime(2026, 10, 8, 10, 0), KST))
    after = build_temporal_snapshot("KRX", _utc(datetime(2026, 10, 12, 10, 0), KST))
    assert before.holiday_adjacent
    assert after.holiday_adjacent


def test_month_and_quarter_end_are_last_trading_sessions() -> None:
    quarter_end = build_temporal_snapshot("KRX", _utc(datetime(2026, 9, 30, 10, 0), KST))
    month_end = build_temporal_snapshot("KRX", _utc(datetime(2026, 8, 31, 10, 0), KST))
    mid_month = build_temporal_snapshot("KRX", _utc(datetime(2026, 9, 15, 10, 0), KST))
    assert quarter_end.month_end and quarter_end.quarter_end
    assert month_end.month_end and not month_end.quarter_end
    assert not mid_month.month_end and not mid_month.quarter_end


def test_trading_day_navigation_skips_holidays() -> None:
    assert previous_trading_day(MarketGroup.KR, date(2026, 10, 12)) == date(2026, 10, 8)
    assert next_trading_day(MarketGroup.KR, date(2026, 10, 8)) == date(2026, 10, 12)


# --------------------------------------------------------------------------- #
# Expiry context
# --------------------------------------------------------------------------- #
def test_krx_expiry_is_the_second_thursday() -> None:
    assert expiry_date_for(MarketGroup.KR, 2026, 9) == date(2026, 9, 10)
    assert expiry_date_for(MarketGroup.KR, 2026, 10) == date(2026, 10, 8)


def test_us_expiry_is_the_third_friday() -> None:
    assert expiry_date_for(MarketGroup.US, 2026, 9) == date(2026, 9, 18)
    assert expiry_date_for(MarketGroup.US, 2026, 11) == date(2026, 11, 20)


def test_quarterly_expiry_is_distinguished_from_monthly() -> None:
    quarterly = build_temporal_snapshot("KRX", _utc(datetime(2026, 9, 10, 10, 0), KST))
    monthly = build_temporal_snapshot("KRX", _utc(datetime(2026, 8, 13, 10, 0), KST))
    adjacent = build_temporal_snapshot("KRX", _utc(datetime(2026, 9, 9, 10, 0), KST))
    quiet = build_temporal_snapshot("KRX", _utc(datetime(2026, 9, 8, 10, 0), KST))
    assert quarterly.expiry_context is ExpiryContext.QUARTERLY_EXPIRY
    assert monthly.expiry_context is ExpiryContext.MONTHLY_EXPIRY
    assert adjacent.expiry_context is ExpiryContext.EXPIRY_ADJACENT
    assert quiet.expiry_context is ExpiryContext.NONE


# --------------------------------------------------------------------------- #
# Snapshot contract
# --------------------------------------------------------------------------- #
def test_snapshot_is_utc_stored_and_seoul_rendered() -> None:
    snapshot = build_temporal_snapshot("US", _utc(datetime(2026, 8, 19, 11, 0), NEW_YORK))
    assert snapshot.as_of.tzinfo == timezone.utc
    assert str(snapshot.display_time.tzinfo) == "Asia/Seoul"
    assert snapshot.exchange_timezone == "America/New_York"
    assert snapshot.trading_day == date(2026, 8, 19)


def test_us_session_trading_day_does_not_straddle_seoul_dates() -> None:
    """23:30 Seoul is 10:30 New York on the previous US calendar day."""
    snapshot = build_temporal_snapshot("US", _utc(datetime(2026, 8, 19, 23, 30), KST))
    assert snapshot.trading_day == date(2026, 8, 19)
    assert snapshot.session_phase is SessionPhase.MORNING_TREND


def test_numeric_features_omit_absent_values() -> None:
    closed = build_temporal_snapshot("KRX", _utc(datetime(2026, 8, 22, 11, 0), KST))
    values = closed.numeric_features()
    assert "session_progress" not in values
    assert values["is_trading_day"] == 0.0
    assert set(values) >= {"day_of_week", "month_end", "quarter_end", "holiday_adjacent"}


def test_snapshot_serialises_to_json_safe_primitives() -> None:
    import json

    payload = build_temporal_snapshot(
        "KRX", _utc(datetime(2026, 8, 19, 10, 0), KST)
    ).as_dict()
    assert json.loads(json.dumps(payload))["session_phase"] == "MORNING_TREND"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_shipped_config_parses() -> None:
    config = load_temporal_config()
    assert config.source_path is not None
    assert config.session_phase.opening_minutes == 30.0
    assert config.seasonality.shrinkage_k == 30.0


def test_absent_config_file_falls_back_to_defaults(tmp_path) -> None:
    config = load_temporal_config(tmp_path / "missing.yaml")
    assert config.source_path is None
    assert config.session_phase.opening_minutes == 30.0


def test_inconsistent_phase_boundaries_are_rejected() -> None:
    with pytest.raises(TemporalConfigError):
        SessionPhaseConfig(morning_end_fraction=0.8, midday_end_fraction=0.5)


def test_malformed_config_raises_rather_than_defaulting(tmp_path) -> None:
    path = tmp_path / "temporal.yaml"
    path.write_text("session_phase:\n  opening_minutes: not-a-number\n", encoding="utf-8")
    with pytest.raises(TemporalConfigError):
        load_temporal_config(path)

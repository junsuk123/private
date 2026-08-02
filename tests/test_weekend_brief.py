"""Weekend research must produce a claim that can be proven wrong.

The repository has already paid for unfalsifiable work: six strategies sat in the
catalogue producing zero training labels because nothing scored them. So the weekend
pipeline commits a direction/magnitude/confidence BEFORE the Monday open and is
graded against the realized gap afterwards.

These tests pin the properties that keep it honest — coverage-limited confidence, no
fabrication from missing inputs, and an immutable record once scored.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from app.research.weekend_brief import (
    DOWN,
    KST,
    NEUTRAL,
    UP,
    MondayOpenPrior,
    WeekendBriefStore,
    WeekendSignals,
    build_monday_prior,
    collect_weekend_signals,
    score_prior,
    weekend_window,
)

NOW_SAT = datetime(2026, 8, 1, 12, 0, tzinfo=KST)  # Saturday


# --------------------------------------------------------------------------- #
# Window                                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "moment,expected",
    [
        (datetime(2026, 7, 31, 14, 0, tzinfo=KST), False),  # Friday, still trading
        (datetime(2026, 7, 31, 15, 40, tzinfo=KST), True),  # Friday after close
        (datetime(2026, 8, 1, 3, 0, tzinfo=KST), True),     # Saturday
        (datetime(2026, 8, 2, 22, 0, tzinfo=KST), True),    # Sunday
        (datetime(2026, 8, 3, 8, 30, tzinfo=KST), True),    # Monday pre-open
        (datetime(2026, 8, 3, 9, 30, tzinfo=KST), False),   # Monday, open
        (datetime(2026, 8, 5, 11, 0, tzinfo=KST), False),   # Wednesday
    ],
)
def test_weekend_window_covers_exactly_the_closed_span(moment, expected) -> None:
    assert (weekend_window(moment) is not None) is expected


def test_window_runs_friday_close_to_monday_open() -> None:
    window = weekend_window(NOW_SAT)
    assert window is not None
    assert window.start == datetime(2026, 7, 31, 15, 30, tzinfo=KST)
    assert window.end == datetime(2026, 8, 3, 9, 0, tzinfo=KST)
    # Identity is the Monday it describes, so Friday/Sat/Sun runs agree.
    assert window.key == "2026-08-03"


def test_all_weekend_moments_agree_on_the_same_window() -> None:
    keys = {
        weekend_window(m).key
        for m in (
            datetime(2026, 7, 31, 16, 0, tzinfo=KST),
            datetime(2026, 8, 1, 12, 0, tzinfo=KST),
            datetime(2026, 8, 2, 23, 0, tzinfo=KST),
            datetime(2026, 8, 3, 8, 0, tzinfo=KST),
        )
    }
    assert keys == {"2026-08-03"}


# --------------------------------------------------------------------------- #
# Prior construction                                                           #
# --------------------------------------------------------------------------- #
def _signals(**overrides) -> WeekendSignals:
    values = {
        "window_key": "2026-08-03",
        "us_session_move_bps": 100.0,
        "vix_change": None,
        "treasury_10y_change": None,
        "dollar_index_change": None,
        "event_count": 100,
        "negative_event_count": 2,
        "macro_event_count": 5,
        "top_sectors": ("Semiconductor",),
        "inputs_available": 1,
        "inputs_expected": 4,
    }
    values.update(overrides)
    return WeekendSignals(**values)


def test_us_move_drives_direction() -> None:
    up = build_monday_prior(_signals(us_session_move_bps=200.0), computed_at=NOW_SAT)
    down = build_monday_prior(_signals(us_session_move_bps=-200.0), computed_at=NOW_SAT)
    assert up.direction == UP
    assert down.direction == DOWN
    assert "US_SESSION_SPILLOVER" in up.reason_codes


def test_pass_through_is_less_than_one() -> None:
    """KRX does not reproduce the whole US move; asserting it would overstate."""
    prior = build_monday_prior(_signals(us_session_move_bps=100.0), computed_at=NOW_SAT)
    assert 0 < prior.magnitude_bps < 100.0


def test_small_move_is_neutral_not_a_forced_call() -> None:
    prior = build_monday_prior(_signals(us_session_move_bps=2.0), computed_at=NOW_SAT)
    assert prior.direction == NEUTRAL
    assert "BELOW_DIRECTIONAL_THRESHOLD" in prior.reason_codes


def test_macro_deltas_act_as_drags() -> None:
    """Risk-off inputs must pull the estimate down, not add conviction."""
    base = build_monday_prior(_signals(us_session_move_bps=100.0), computed_at=NOW_SAT)
    risk_off = build_monday_prior(
        _signals(us_session_move_bps=100.0, vix_change=3.0, treasury_10y_change=0.10),
        computed_at=NOW_SAT,
    )
    assert risk_off.magnitude_bps < base.magnitude_bps
    assert "VIX_RISK_REPRICING" in risk_off.reason_codes
    assert "RATES_REPRICING" in risk_off.reason_codes


def test_confidence_scales_with_input_coverage() -> None:
    """One input of four must not look as certain as four of four."""
    thin = build_monday_prior(
        _signals(inputs_available=1, inputs_expected=4), computed_at=NOW_SAT
    )
    full = build_monday_prior(
        _signals(
            inputs_available=4,
            inputs_expected=4,
            vix_change=-0.5,
            treasury_10y_change=-0.02,
            dollar_index_change=-0.2,
        ),
        computed_at=NOW_SAT,
    )
    assert full.confidence > thin.confidence
    assert full.confidence <= 0.75, "an untested prior must never look certain"


def test_missing_us_move_is_reported_not_assumed_zero() -> None:
    prior = build_monday_prior(_signals(us_session_move_bps=None), computed_at=NOW_SAT)
    assert "US_SESSION_MOVE_UNAVAILABLE" in prior.reason_codes
    assert prior.direction == NEUTRAL


def test_negative_event_concentration_dampens_confidence() -> None:
    calm = build_monday_prior(
        _signals(event_count=100, negative_event_count=1), computed_at=NOW_SAT
    )
    stressed = build_monday_prior(
        _signals(event_count=100, negative_event_count=40), computed_at=NOW_SAT
    )
    assert stressed.confidence < calm.confidence
    assert "ELEVATED_NEGATIVE_EVENT_SHARE" in stressed.reason_codes


# --------------------------------------------------------------------------- #
# Signal collection                                                            #
# --------------------------------------------------------------------------- #
def _research_db(tmp_path, events, macro):
    path = tmp_path / "research.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute(
        "create table records (kind text, record_key text, observed_at text,"
        " inserted_at text, payload text)"
    )
    for at, payload in events:
        conn.execute(
            "insert into records values ('events','k',?,?,?)",
            (at, at, json.dumps(payload)),
        )
    for at, payload in macro:
        conn.execute(
            "insert into records values ('macro_metrics','k',?,?,?)",
            (at, at, json.dumps(payload)),
        )
    conn.commit()
    conn.close()
    return path


def test_signals_only_count_events_inside_the_window(tmp_path) -> None:
    window = weekend_window(NOW_SAT)
    assert window is not None
    inside = (window.start + timedelta(hours=5)).astimezone(timezone.utc).isoformat()
    before = (window.start - timedelta(days=2)).astimezone(timezone.utc).isoformat()
    db = _research_db(
        tmp_path,
        events=[
            (inside, {"sentiment": "POSITIVE", "sectors": ["Semiconductor"]}),
            (inside, {"sentiment": "NEGATIVE", "sectors": ["Finance"]}),
            (before, {"sentiment": "NEGATIVE", "sectors": ["Finance"]}),
        ],
        macro=[],
    )
    signals = collect_weekend_signals(window, research_db=db)
    assert signals.event_count == 2, "an event before Friday's close is not weekend news"
    assert signals.negative_event_count == 1


def test_single_macro_observation_yields_no_change(tmp_path) -> None:
    """One print cannot express a change; reporting 0.0 would assert 'no move'."""
    window = weekend_window(NOW_SAT)
    assert window is not None
    at = (window.start - timedelta(days=1)).astimezone(timezone.utc).isoformat()
    db = _research_db(
        tmp_path, events=[], macro=[(at, {"name": "us_vix_close", "value": 17.0})]
    )
    signals = collect_weekend_signals(window, research_db=db)
    assert signals.vix_change is None


def test_two_macro_observations_yield_the_change(tmp_path) -> None:
    window = weekend_window(NOW_SAT)
    assert window is not None
    older = (window.start - timedelta(days=3)).astimezone(timezone.utc).isoformat()
    newer = (window.start - timedelta(days=1)).astimezone(timezone.utc).isoformat()
    db = _research_db(
        tmp_path,
        events=[],
        macro=[
            (older, {"name": "us_vix_close", "value": 15.0}),
            (newer, {"name": "us_vix_close", "value": 18.5}),
        ],
    )
    signals = collect_weekend_signals(window, research_db=db)
    assert signals.vix_change == pytest.approx(3.5)


def test_missing_research_db_degrades_to_empty_not_crash(tmp_path) -> None:
    window = weekend_window(NOW_SAT)
    assert window is not None
    signals = collect_weekend_signals(window, research_db=tmp_path / "nope.sqlite3")
    assert signals.event_count == 0
    assert signals.coverage == 0.0


# --------------------------------------------------------------------------- #
# Scoring and the record                                                       #
# --------------------------------------------------------------------------- #
def test_score_marks_direction_and_error() -> None:
    prior = build_monday_prior(_signals(us_session_move_bps=200.0), computed_at=NOW_SAT)
    hit = score_prior(prior, realized_gap_bps=90.0)
    miss = score_prior(prior, realized_gap_bps=-90.0)
    assert hit["direction_correct"] is True
    assert miss["direction_correct"] is False
    assert miss["absolute_error_bps"] > hit["absolute_error_bps"]


def test_store_round_trip_and_track_record(tmp_path) -> None:
    store = WeekendBriefStore(tmp_path / "brief.sqlite3")
    prior = build_monday_prior(_signals(us_session_move_bps=200.0), computed_at=NOW_SAT)
    store.save_prior(prior)

    latest = store.latest_prior()
    assert latest is not None and latest["direction"] == UP
    assert store.track_record()["scored"] == 0

    score = store.record_score(prior.window_key, realized_gap_bps=80.0)
    assert score is not None and score["direction_correct"] is True
    record = store.track_record()
    assert record["scored"] == 1
    assert record["direction_accuracy"] == 1.0


def test_a_scored_prior_can_never_be_rewritten(tmp_path) -> None:
    """Otherwise a later run could quietly improve its own track record."""
    store = WeekendBriefStore(tmp_path / "brief.sqlite3")
    original = build_monday_prior(_signals(us_session_move_bps=200.0), computed_at=NOW_SAT)
    store.save_prior(original)
    store.record_score(original.window_key, realized_gap_bps=-150.0)

    revised = build_monday_prior(_signals(us_session_move_bps=-200.0), computed_at=NOW_SAT)
    store.save_prior(revised)  # must be ignored

    latest = store.latest_prior()
    assert latest["direction"] == UP, "the committed claim must stand"
    assert store.track_record()["direction_accuracy"] == 0.0


def test_scoring_twice_is_rejected(tmp_path) -> None:
    store = WeekendBriefStore(tmp_path / "brief.sqlite3")
    prior = build_monday_prior(_signals(us_session_move_bps=200.0), computed_at=NOW_SAT)
    store.save_prior(prior)
    assert store.record_score(prior.window_key, 80.0) is not None
    assert store.record_score(prior.window_key, 999.0) is None

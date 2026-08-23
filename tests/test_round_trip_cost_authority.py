"""One round-trip cost, read the same way by every layer that charges it.

The three layers used to disagree: the trigger floor read the fee policy (33.8bps on
KRX), the session election read a 28bps constant, and the training labels read the
policy floored by the realized tape (51.9). A trigger therefore fired on a 40bps edge
that the labels scored as a loss and the executor paid ~53bps to take.
"""

from __future__ import annotations

import sqlite3

import pytest

from app.cost import round_trip
from app.technical.strategy_algorithms import (
    reset_cost_floor_cache,
    round_trip_cost_bps,
)


@pytest.fixture(autouse=True)
def _clean_caches():
    round_trip.reset_caches()
    yield
    round_trip.reset_caches()


def _tape(path, market: str, costs: list[float]) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "create table if not exists shadow_outcomes "
            "(market text, trading_cost_bps real)"
        )
        conn.executemany(
            "insert into shadow_outcomes (market, trading_cost_bps) values (?, ?)",
            [(market, cost) for cost in costs],
        )


def test_policy_excludes_the_spread_because_the_schedule_cannot_know_it():
    krx = round_trip.policy_round_trip_bps("005930")
    us = round_trip.policy_round_trip_bps("AAPL")
    assert krx is not None and us is not None
    # 20bps KRX transfer tax dominates; US is commission-heavy instead.
    assert 28.0 <= krx <= 40.0
    assert 45.0 <= us <= 60.0


def test_symbol_spread_is_charged_once_over_the_round_trip():
    policy = round_trip.policy_round_trip_bps("064260")
    assert round_trip.all_in_round_trip_bps("064260", spread_bps=19.3) == pytest.approx(
        policy + 19.3
    )
    # A wider book on the same venue must produce a strictly higher cost; collapsing
    # every KR name onto one constant is what hid the difference between a 6.5bps
    # name and a 24.7bps one.
    assert round_trip.all_in_round_trip_bps(
        "064260", spread_bps=24.7
    ) > round_trip.all_in_round_trip_bps("064260", spread_bps=6.5)


def test_realized_tape_lifts_the_estimate_without_double_counting_the_spread(
    tmp_path, monkeypatch
):
    """The tape total already contains a spread; it is an alternative, not an addend."""
    tape = tmp_path / "directional-shadow.sqlite3"
    _tape(tape, "KR", [51.9] * 60)
    monkeypatch.setenv("DIRECTIONAL_SHADOW_STORE_PATH", str(tape))
    round_trip.reset_caches()

    policy = round_trip.policy_round_trip_bps("005930")
    assert policy is not None and policy < 51.9

    # No spread reading: the tape is the only estimate that has seen one.
    assert round_trip.all_in_round_trip_bps("005930") == pytest.approx(51.9)
    # A narrow book still cannot price the round trip below what the venue charged.
    assert round_trip.all_in_round_trip_bps("005930", spread_bps=2.0) == pytest.approx(51.9)
    # A wide book overtakes it — and the result is policy + spread, NOT tape + spread.
    wide = round_trip.all_in_round_trip_bps("005930", spread_bps=30.0)
    assert wide == pytest.approx(policy + 30.0)
    assert wide < 51.9 + 30.0


def test_a_thin_tape_is_not_allowed_to_speak(tmp_path, monkeypatch):
    tape = tmp_path / "directional-shadow.sqlite3"
    _tape(tape, "KR", [900.0] * (round_trip.MINIMUM_RESOLVED_ROUND_TRIPS - 1))
    monkeypatch.setenv("DIRECTIONAL_SHADOW_STORE_PATH", str(tape))
    round_trip.reset_caches()

    policy = round_trip.policy_round_trip_bps("005930")
    assert round_trip.all_in_round_trip_bps("005930") == pytest.approx(policy)


def test_configured_fallback_is_a_floor_never_a_replacement():
    assert round_trip.all_in_round_trip_bps("005930", fallback_bps=999.0) == 999.0


def test_trigger_floor_reads_the_same_authority_as_the_session(tmp_path, monkeypatch):
    """This is the disagreement itself, pinned.

    ``round_trip_cost_bps`` is what ``AlgorithmBase.entry_floor_bps`` holds a trigger
    to. It must not be able to clear a bar the election will then reject.
    """
    tape = tmp_path / "directional-shadow.sqlite3"
    _tape(tape, "KR", [51.9] * 60)
    monkeypatch.setenv("DIRECTIONAL_SHADOW_STORE_PATH", str(tape))
    round_trip.reset_caches()
    reset_cost_floor_cache()

    trigger = round_trip_cost_bps("005930")
    assert trigger == pytest.approx(round_trip.all_in_round_trip_bps("005930"))
    assert trigger == pytest.approx(51.9)

    reset_cost_floor_cache()


def test_unreadable_cost_config_is_unknown_not_free(monkeypatch):
    monkeypatch.setattr(
        round_trip, "policy_round_trip_bps", lambda symbol: None
    )
    monkeypatch.setattr(
        round_trip, "measured_round_trip_bps", lambda market: None
    )
    # Nothing can answer, so the caller's own configured reference is all that is
    # left — and it is a floor, so the answer is never 0.
    assert round_trip.all_in_round_trip_bps("005930", fallback_bps=28.0) == 28.0

"""Three integration defects that made registered strategies lie about themselves.

Each was a case of a declaration existing while nothing honoured it:

1. ``live_authorized: 0.0`` was ignored unless the strategy also appeared in a
   hand-maintained ``_DEPLOYMENT_GATED_STRATEGIES`` literal. Two shadow-only
   strategies were missing from it and were therefore treated as live-tradable.
2. ``_apply_owned_exit_geometry`` special-cased ``rvgi_box_breakout`` while its
   docstring claimed it resolved every elected algorithm's rule, so every other
   algorithm's structural stop/target was discarded.
3. The session-boxed strategies declared context fields that nothing populated, so
   they rejected every tick with ``*_CONTEXT_ABSENT`` — registered but inert.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.strategy.catalog import STRATEGY_IDS
from app.technical.strategy_algorithms import (
    _DEFAULTS,
    _DEPLOYMENT_GATED_STRATEGIES,
    strategy_live_authorized,
    strategy_shadow_authorized,
)
from app.trading.strategy_session import _session_structure_context

KST = timezone(timedelta(hours=9))
#: The session clock is resolved from the symbol, so a KRX assertion must name one.
KRX_SYMBOL = "005930"


# --------------------------------------------------------------------------- #
# 1. The deployment gate must be derived, not hand-listed.                     #
# --------------------------------------------------------------------------- #
def test_every_strategy_declaring_the_knob_is_actually_gated() -> None:
    """Declaring ``live_authorized`` IS the gate; the two cannot disagree."""
    declared = {
        strategy_id
        for strategy_id, values in _DEFAULTS.items()
        if strategy_id != "shared" and "live_authorized" in values
    }
    assert declared, "expected at least one deployment-gated strategy"
    assert _DEPLOYMENT_GATED_STRATEGIES == frozenset(declared)


@pytest.mark.parametrize(
    "strategy_id",
    sorted(
        strategy_id
        for strategy_id, values in _DEFAULTS.items()
        if strategy_id != "shared" and values.get("live_authorized") == 0.0
    ),
)
def test_shadow_only_strategies_are_not_live_authorized(strategy_id: str) -> None:
    """The regression: opening_range_breakout and market_intraday_momentum both
    declared 0.0 and both returned True because the literal set omitted them."""
    assert strategy_live_authorized(strategy_id) is False
    # ...but they must still run in shadow, which is how they earn promotion.
    assert strategy_shadow_authorized(strategy_id) is True


def test_established_strategies_stay_live_by_default() -> None:
    """Strategies that never declared the knob must not become gated by this fix."""
    for strategy_id in ("intraday_momentum", "breakout_volume", "vwap_mean_reversion"):
        assert strategy_id not in _DEPLOYMENT_GATED_STRATEGIES
        assert strategy_live_authorized(strategy_id) is True


def test_unknown_strategy_is_never_live_authorized() -> None:
    assert strategy_live_authorized("no_such_strategy") is False


# --------------------------------------------------------------------------- #
# 3. Session structure is supplied from the clock.                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "hour,minute,expected_in_window",
    [
        (9, 30, False),    # morning
        (14, 49, False),   # one minute early
        (14, 50, True),    # window opens
        (15, 10, True),    # inside
        (15, 19, True),    # last minute of continuous trading
        (15, 20, False),   # closing auction begins
        (15, 25, False),   # inside the auction
    ],
)
def test_last_continuous_half_hour_window(hour, minute, expected_in_window) -> None:
    """14:50-15:20 for a KRX symbol, never into the 15:20 auction."""
    moment = datetime(2026, 8, 3, hour, minute, tzinfo=KST)
    context = _session_structure_context(moment, KRX_SYMBOL)
    assert context["in_last_continuous_half_hour"] is expected_in_window


def test_minutes_to_continuous_close_counts_down_and_goes_negative() -> None:
    early = _session_structure_context(datetime(2026, 8, 3, 14, 50, tzinfo=KST), KRX_SYMBOL)
    late = _session_structure_context(datetime(2026, 8, 3, 15, 15, tzinfo=KST), KRX_SYMBOL)
    after = _session_structure_context(datetime(2026, 8, 3, 15, 25, tzinfo=KST), KRX_SYMBOL)
    assert early["minutes_to_continuous_close"] == pytest.approx(30.0)
    assert late["minutes_to_continuous_close"] == pytest.approx(5.0)
    # Must not wrap to a large positive number after the close, which would read as
    # "plenty of time left" to every consumer.
    assert after["minutes_to_continuous_close"] < 0


def test_context_is_timezone_correct_from_utc() -> None:
    """Bars and clocks are UTC; the window is a Korean-calendar fact."""
    utc = datetime(2026, 8, 3, 6, 0, tzinfo=timezone.utc)  # 15:00 KST
    assert _session_structure_context(utc, KRX_SYMBOL)["in_last_continuous_half_hour"] is True


def test_the_window_is_the_symbols_market_not_a_fixed_krx_one() -> None:
    """A US name's last continuous half hour is 15:30-16:00 New York.

    Reading the KRX window for every symbol put it at 14:50-15:20 Seoul — the
    middle of the New York night — so every session-boxed strategy rejected every
    US tick with ``*_OUTSIDE_ENTRY_WINDOW``.
    """
    us_close = datetime(2026, 8, 10, 19, 45, tzinfo=timezone.utc)  # 15:45 New York

    assert _session_structure_context(us_close, "PFE")["in_last_continuous_half_hour"] is True
    assert _session_structure_context(us_close, KRX_SYMBOL)["in_last_continuous_half_hour"] is False


# --------------------------------------------------------------------------- #
# 2. Structural exits apply to every strategy, without deleting stops.         #
# --------------------------------------------------------------------------- #
def _manager(tmp_path):
    from app.trading.strategy_session import StrategySessionConfig, StrategySessionManager

    return StrategySessionManager(
        config=StrategySessionConfig(state_path=str(tmp_path / "session.json"))
    )


def test_exit_geometry_applies_beyond_rvgi_without_deleting_the_stop(tmp_path) -> None:
    """A rule that cannot resolve a stop must leave the table's stop in place.

    The rule is built from a bare feature set at fill time, so volatility-derived
    stops come back as None for most algorithms. Assigning that blindly would strip
    the position's protection.
    """
    manager = _manager(tmp_path)
    state = manager._state  # noqa: SLF001
    state.selected_strategy = "intraday_momentum"
    state.selected_symbol = "005930"
    state.stop_price = 79_000.0
    state.target_return_rate = 0.016
    state.max_holding_seconds = 3600

    manager._apply_owned_exit_geometry(80_000.0)  # noqa: SLF001

    assert state.stop_price == 79_000.0, "an unresolved rule stop must not clear it"
    assert state.target_price is not None and state.target_price > 80_000.0


def test_horizon_may_only_shorten(tmp_path) -> None:
    """Shortening enforces a session deadline; lengthening would overrule the table."""
    manager = _manager(tmp_path)
    state = manager._state  # noqa: SLF001
    state.selected_strategy = "market_intraday_momentum"
    state.selected_symbol = "005930"
    state.max_holding_seconds = 1500
    state.target_return_rate = 0.016
    # Only eight minutes of continuous trading left.
    state.election_context = {
        "in_last_continuous_half_hour": True,
        "minutes_to_continuous_close": 8.0,
        "first_half_hour_return_bps": 80.0,
        "first_half_hour_volatility_percentile": 0.9,
    }

    manager._apply_owned_exit_geometry(80_000.0)  # noqa: SLF001

    assert state.max_holding_seconds < 1500, "must be flat before the auction"
    assert state.max_holding_seconds <= 6 * 60


def test_no_selected_strategy_falls_back_to_the_rate_target(tmp_path) -> None:
    manager = _manager(tmp_path)
    state = manager._state  # noqa: SLF001
    state.selected_strategy = None
    state.target_return_rate = 0.016
    manager._apply_owned_exit_geometry(80_000.0)  # noqa: SLF001
    assert state.target_price == pytest.approx(80_000.0 * 1.016)


def test_every_catalogued_strategy_has_an_algorithm_and_a_gate_decision() -> None:
    """No strategy may be registered without a resolvable deployment decision."""
    for strategy_id in STRATEGY_IDS:
        assert isinstance(strategy_live_authorized(strategy_id), bool)
        assert isinstance(strategy_shadow_authorized(strategy_id), bool)

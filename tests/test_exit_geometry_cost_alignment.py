"""Exit geometry must survive contact with the spread and the round-trip cost.

The defect this guards: stops of 18-35bps were set against a KRX tape whose
top-of-book spread is 13-50bps. A stop inside one spread is hit by bid-ask
bounce, not by the thesis failing, so 96.2% of simulated KRX fills stopped out
and the fill-weighted net was -47.3bps -- almost exactly -(stop + cost). Gross
reward:risk read 4.5 the whole time, which is why the table looked healthy.

These tests pin the two properties that made the old numbers wrong, so a future
hand-edit of the table cannot quietly reintroduce either one.
"""

from __future__ import annotations

import pytest

from app.strategy.catalog import STRATEGY_IDS
from app.strategy.exit_geometry import (
    FALLBACK_GEOMETRY_KEY,
    MINIMUM_STOP_BPS,
    REFERENCE_ROUND_TRIP_COST_BPS,
    TARGET_NET_REWARD_RISK,
    TYPICAL_SPREAD_BPS,
    all_geometries,
    exit_geometry,
    label_geometries,
    resolve_exit_geometry,
)

# Every catalogued strategy plus the adopted-position fallback.
ALL_KEYS = (*STRATEGY_IDS, FALLBACK_GEOMETRY_KEY)


@pytest.mark.parametrize("strategy_id", ALL_KEYS)
def test_stop_clears_the_spread(strategy_id: str) -> None:
    """A stop inside the spread measures bid-ask bounce, not a failed thesis."""
    geometry = exit_geometry(strategy_id)
    assert geometry.stop_loss_bps >= MINIMUM_STOP_BPS, (
        f"{strategy_id} stops at {geometry.stop_loss_bps}bps, inside "
        f"{MINIMUM_STOP_BPS / TYPICAL_SPREAD_BPS:.0f}x the typical "
        f"{TYPICAL_SPREAD_BPS}bps spread"
    )


@pytest.mark.parametrize("strategy_id", ALL_KEYS)
def test_net_reward_risk_meets_the_target(strategy_id: str) -> None:
    """Cost is additive on both barriers; the gross ratio hides that."""
    geometry = exit_geometry(strategy_id)
    net = geometry.net_reward_risk_ratio()
    assert net >= TARGET_NET_REWARD_RISK - 0.05, (
        f"{strategy_id} nets {net:.2f} reward:risk at "
        f"{REFERENCE_ROUND_TRIP_COST_BPS}bps cost, below the "
        f"{TARGET_NET_REWARD_RISK} target"
    )


@pytest.mark.parametrize("strategy_id", ALL_KEYS)
def test_target_clears_round_trip_cost_with_margin(strategy_id: str) -> None:
    """A target below cost loses money on a *winning* trade."""
    geometry = exit_geometry(strategy_id)
    assert geometry.take_profit_bps > REFERENCE_ROUND_TRIP_COST_BPS * 2.0


@pytest.mark.parametrize("strategy_id", ALL_KEYS)
def test_trailing_stop_is_not_inside_the_spread(strategy_id: str) -> None:
    """A trailing stop tighter than the spread exits on the first bounce."""
    geometry = exit_geometry(strategy_id)
    assert geometry.trailing_bps > TYPICAL_SPREAD_BPS
    # It must also be tighter than the hard stop, or it can never bind.
    assert geometry.trailing_bps < geometry.stop_loss_bps


@pytest.mark.parametrize("strategy_id", ALL_KEYS)
def test_holding_time_is_plausible_for_the_target(strategy_id: str) -> None:
    """Targets grew ~1.6x; a horizon that assumed 100bps cannot express 160bps.

    The bound is deliberately loose -- it catches a target raised without its
    horizon, not a specific tuning choice.
    """
    geometry = exit_geometry(strategy_id)
    assert geometry.max_holding_seconds >= 1200


def test_labels_and_executor_read_the_same_table() -> None:
    """One table, so a geometry change moves the training labels with it."""
    for strategy_id, geometry in label_geometries().items():
        barriers = geometry.as_label_barriers()
        live = exit_geometry(strategy_id)
        assert barriers["take_profit_bps"] == live.take_profit_bps
        assert barriers["stop_loss_bps"] == live.stop_loss_bps
        assert barriers["horizon_seconds"] == float(live.max_holding_seconds)


def test_every_catalogued_strategy_has_its_own_entry() -> None:
    """A strategy silently falling back would be labelled on the wrong barriers."""
    explicit = set(all_geometries())
    missing = [s for s in STRATEGY_IDS if s not in explicit]
    assert not missing, f"strategies with no geometry of their own: {missing}"


def test_net_ratio_degrades_on_a_costlier_venue() -> None:
    """US models at 67bps, so the same table nets far less there.

    Asserted rather than assumed: it is the reason the runtime cost floor in
    ``strategy_session._cost_aware_profit_bps`` still has to exist.
    """
    geometry = exit_geometry("intraday_momentum")
    assert geometry.net_reward_risk_ratio(67.0) < geometry.net_reward_risk_ratio(28.0)


# --- Cost-relative resolution ---------------------------------------------- #
# The gap the tests above could not see: every assertion in this file is made at
# REFERENCE_ROUND_TRIP_COST_BPS, which is KRX's 28. The table passed all of them
# while being applied, unchanged, to a US tape whose measured median round trip is
# 63.2bps (p90 125.2) across the 775 stored shadow fills. There it delivered a net
# reward:risk of 0.83 against the 1.5 asserted above — a break-even win rate of 55%
# instead of 40%, which is why no arm evaluated on that tape could ever pass.


@pytest.mark.parametrize("strategy_id", ALL_KEYS)
@pytest.mark.parametrize("cost_bps", (28.0, 52.0, 63.2, 79.6, 125.2))
def test_resolved_geometry_holds_the_invariant_at_the_measured_cost(
    strategy_id: str, cost_bps: float
) -> None:
    """net R:R must be ~1.5 at the cost actually paid, not only at the reference."""
    geometry = resolve_exit_geometry(strategy_id, round_trip_cost_bps=cost_bps)
    net = geometry.net_reward_risk_ratio(cost_bps)
    assert net >= TARGET_NET_REWARD_RISK - 0.05, (
        f"{strategy_id} nets {net:.2f} reward:risk at a measured {cost_bps}bps "
        f"round trip, below the {TARGET_NET_REWARD_RISK} the table is built on"
    )


@pytest.mark.parametrize("spread_bps", (5.0, 20.0, 35.0, 60.0))
def test_resolved_stop_clears_three_measured_spreads(spread_bps: float) -> None:
    """The 'outside the spread' rule, against THIS symbol's spread.

    ``MINIMUM_STOP_BPS`` encodes 3 x a *typical KRX* spread. On a wider name that
    constant is back inside the spread, which is the original defect one venue over.
    """
    geometry = resolve_exit_geometry(
        "intraday_momentum", round_trip_cost_bps=63.2, spread_bps=spread_bps
    )
    assert geometry.stop_loss_bps >= min(MINIMUM_STOP_BPS, 3.0 * spread_bps)
    assert geometry.stop_loss_bps >= 3.0 * spread_bps or spread_bps * 3.0 <= MINIMUM_STOP_BPS
    assert geometry.trailing_bps < geometry.stop_loss_bps


@pytest.mark.parametrize("strategy_id", ALL_KEYS)
def test_unmeasured_resolution_returns_the_table_verbatim(strategy_id: str) -> None:
    """No measurement means no change: every existing caller keeps its numbers."""
    assert resolve_exit_geometry(strategy_id) == exit_geometry(strategy_id)


def test_krx_reference_measurement_reproduces_the_table() -> None:
    """The table IS the formula at the KRX reference, so this must be an identity.

    If it ever stops being one, either the table was hand-edited off its own rule or
    the resolver drifted from it.

    Note the shorts pass here while being handed the LONG reference: the resolver
    floors every cost at ``reference_round_trip_cost_bps``, which carries the borrow
    leg for a short. That floor is load-bearing — a measured cost that omitted borrow
    would otherwise size a short's target as if it were a long.
    """
    for strategy_id, table in all_geometries().items():
        resolved = resolve_exit_geometry(
            strategy_id,
            round_trip_cost_bps=REFERENCE_ROUND_TRIP_COST_BPS,
            spread_bps=TYPICAL_SPREAD_BPS,
        )
        assert resolved.stop_loss_bps == table.stop_loss_bps, strategy_id
        assert resolved.take_profit_bps == table.take_profit_bps, strategy_id
        assert resolved.max_holding_seconds == table.max_holding_seconds, strategy_id


def test_time_boxed_horizons_are_never_stretched() -> None:
    """A session-structure clock is the thesis, not a leash on it.

    Stretching it to make room for a larger target would carry the position past the
    auction the thesis is defined against.
    """
    for strategy_id in ("market_intraday_momentum", "overnight_gap_carry"):
        table = exit_geometry(strategy_id)
        resolved = resolve_exit_geometry(
            strategy_id, round_trip_cost_bps=125.2, spread_bps=35.0
        )
        assert resolved.take_profit_bps > table.take_profit_bps
        assert resolved.max_holding_seconds == table.max_holding_seconds


def test_idle_session_placeholders_match_the_fallback_geometry() -> None:
    """The SCANNING panel must not advertise numbers no code path applies.

    These were hardcoded to 0.004 / 0.0022 / 0.0015 / 600s. Once the geometry
    table moved they described nothing real, and an operator reading a 40bps
    target against a ~28bps round trip reasonably concluded the target could not
    clear costs -- a misdiagnosis caused purely by a stale default.
    """
    from app.trading.strategy_session import StrategySessionState

    state = StrategySessionState()
    fallback = exit_geometry(FALLBACK_GEOMETRY_KEY)
    assert state.target_return_rate == fallback.take_profit_bps / 10_000.0
    assert state.stop_loss_rate == fallback.stop_loss_bps / 10_000.0
    assert state.trailing_stop_rate == fallback.trailing_bps / 10_000.0
    assert state.max_holding_seconds == fallback.max_holding_seconds


def test_stale_persisted_placeholders_are_refreshed_on_load(tmp_path) -> None:
    """A months-old state file must not outrank the geometry table."""
    import json

    from app.trading.strategy_session import (
        StrategySessionConfig,
        StrategySessionManager,
    )

    state_path = tmp_path / "strategy-session.json"
    state_path.write_text(
        json.dumps(
            {
                "phase": "SCANNING",
                "selected_strategy": None,
                "target_return_rate": 0.004,  # the stale 40bps placeholder
                "stop_loss_rate": 0.0022,
                "trailing_stop_rate": 0.0015,
                "max_holding_seconds": 600,
            }
        ),
        encoding="utf-8",
    )
    manager = StrategySessionManager(
        config=StrategySessionConfig(state_path=str(state_path))
    )
    fallback = exit_geometry(FALLBACK_GEOMETRY_KEY)
    state = manager._state  # noqa: SLF001 - asserting the restored state itself
    assert state.target_return_rate == fallback.take_profit_bps / 10_000.0
    assert state.stop_loss_rate == fallback.stop_loss_bps / 10_000.0
    assert state.max_holding_seconds == fallback.max_holding_seconds


def test_open_position_keeps_the_geometry_it_was_armed_with(tmp_path) -> None:
    """With a thesis active those fields are real state, not placeholders.

    Refreshing them on restart would re-arm a live position against different
    barriers than the ones it was entered on.
    """
    import json

    from app.trading.strategy_session import (
        StrategySessionConfig,
        StrategySessionManager,
    )

    state_path = tmp_path / "strategy-session.json"
    state_path.write_text(
        json.dumps(
            {
                "phase": "OWNED",
                "selected_strategy": "intraday_momentum",
                "selected_symbol": "005930",
                "target_return_rate": 0.0123,
                "stop_loss_rate": 0.0045,
                "trailing_stop_rate": 0.0021,
                "max_holding_seconds": 999,
            }
        ),
        encoding="utf-8",
    )
    manager = StrategySessionManager(
        config=StrategySessionConfig(state_path=str(state_path))
    )
    state = manager._state  # noqa: SLF001
    assert state.target_return_rate == 0.0123
    assert state.stop_loss_rate == 0.0045
    assert state.trailing_stop_rate == 0.0021
    assert state.max_holding_seconds == 999


def test_election_target_floor_tracks_the_fallback_geometry(monkeypatch) -> None:
    """The floor must not sit below the table it backstops."""
    from app.trading.strategy_session import StrategySessionConfig

    monkeypatch.delenv("STRATEGY_SESSION_TARGET_RETURN_RATE", raising=False)
    config = StrategySessionConfig()
    fallback = exit_geometry(FALLBACK_GEOMETRY_KEY)
    assert config.fallback_target_return_rate == fallback.take_profit_bps / 10_000.0


def test_env_override_still_wins() -> None:
    """Operators can retune without a deploy; the table is the default, not a law."""
    import os

    key = "EXIT_GEOMETRY_INTRADAY_MOMENTUM_STOP_LOSS_BPS"
    previous = os.environ.get(key)
    os.environ[key] = "88"
    try:
        assert exit_geometry("intraday_momentum").stop_loss_bps == 88.0
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous

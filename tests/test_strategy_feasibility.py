"""Discovery must not hand the joint election charts no strategy can work on.

Turnover chose the universe; turnover says nothing about whether a strategy's exit
target is reachable. Once ``exit_geometry`` sizes that target against the symbol's own
cost and spread, the comparison becomes arithmetic, and the arithmetic reproduces the
measured outcome without using any outcome data.

Validation, 2026-08-11: the six US names that produced 775 of the 776 stored realized
outcomes at a -123bps mean scored the WORST headroom in the set (INTC -64, SOFI -113,
T -116, F -133, PFE -141, BAC -155). The three defects the tests below pin were all
found by running the metric against the live store rather than by reading it.
"""

from __future__ import annotations

import pytest

from app.trading.strategy_feasibility import (
    VERDICT_FEASIBLE,
    VERDICT_INFEASIBLE,
    VERDICT_UNKNOWN,
    SymbolFeasibility,
    attainable_move_bps,
    evaluate_symbol,
    rank_by_feasibility,
    sigma_1m_bps,
)

# 40-minute and 120-minute horizons, in seconds.
STRATEGIES = {"liquidity_shock_reversal": 2400.0, "opening_range_breakout": 7200.0}


def _closes(count: int, *, step_bps: float, start: float = 100.0) -> list[float]:
    """A price path whose mean absolute per-bar return is exactly ``step_bps``."""
    prices = [start]
    for index in range(count - 1):
        direction = 1.0 if index % 2 == 0 else -1.0
        prices.append(prices[-1] * (1.0 + direction * step_bps / 10_000.0))
    return prices


def test_sigma_uses_the_mean_so_a_thin_name_is_still_measurable() -> None:
    """Half-zero bars make the MEDIAN absolute return 0 and the estimator useless.

    That failure is not hypothetical: it silently reinstated the fixed 160/60 barrier
    pair on every US name the last time a median was used here.
    """
    closes = [100.0, 100.0, 100.0, 100.0, 101.0, 101.0, 101.0]
    assert sigma_1m_bps(closes) is not None
    assert sigma_1m_bps(closes) > 0.0


def test_attainable_scales_with_the_square_root_of_time() -> None:
    assert attainable_move_bps(10.0, 3600) == pytest.approx(10.0 * 60**0.5)


def test_a_chart_that_cannot_reach_the_target_is_infeasible() -> None:
    feasibility = evaluate_symbol(
        "PFE",
        closes=_closes(1200, step_bps=2.0),
        spread_bps=4.0,
        round_trip_cost_bps=45.0,
        strategies=STRATEGIES,
        market="US",
    )
    assert feasibility.verdict == VERDICT_INFEASIBLE
    assert feasibility.headroom_bps is not None and feasibility.headroom_bps < 0


def test_a_chart_that_can_reach_the_target_is_feasible() -> None:
    feasibility = evaluate_symbol(
        "010170",
        closes=_closes(1200, step_bps=40.0),
        spread_bps=8.0,
        round_trip_cost_bps=28.0,
        strategies=STRATEGIES,
        market="KR",
    )
    assert feasibility.verdict == VERDICT_FEASIBLE
    assert feasibility.headroom_bps is not None and feasibility.headroom_bps > 0


# --- The three defects found by running it against the live store -------------- #


def test_a_horizon_longer_than_the_history_is_not_extrapolated_into() -> None:
    """sqrt-scaling 32 bars up to a 120-minute horizon is a projection, not an estimate.

    Left unguarded it ranked two thin names with 32 and 40 bars at +607 and +164
    headroom — above every symbol with real coverage. A wild microcap topping the
    universe is exactly the outcome to avoid, and it arrived as a data artefact.
    """
    thin = evaluate_symbol(
        "900340",
        closes=_closes(32, step_bps=140.0),
        spread_bps=16.0,
        round_trip_cost_bps=28.0,
        strategies=STRATEGIES,
        market="KR",
    )
    assert thin.verdict == VERDICT_UNKNOWN
    assert thin.headroom_bps is None
    assert thin.score == 0.0, "an unjudgeable symbol must rank neutral, not first"


def test_enough_history_admits_the_short_horizon_strategy_only() -> None:
    """The window rule selects WHICH strategy is judgeable, it does not just refuse.

    The bar counts come from the resolved geometry, not from the base horizon: at a
    45bps cost and a 30bps spread both strategies stretch their holding time to make
    room for the larger target, so liquidity_shock_reversal needs 294 bars (58.8min x
    5) and opening_range_breakout needs 938 (187.5min x 5). 400 admits exactly one.
    """
    feasibility = evaluate_symbol(
        "AXTI",
        closes=_closes(400, step_bps=35.0),
        spread_bps=30.0,
        round_trip_cost_bps=45.0,
        strategies=STRATEGIES,
        market="US",
    )
    assert feasibility.strategy_id == "liquidity_shock_reversal"
    assert [name for name, _ in feasibility.per_strategy] == ["liquidity_shock_reversal"]


def test_an_unmeasured_spread_is_unknown_rather_than_feasible() -> None:
    """The target is spread-sized, so a missing spread makes it unknowable.

    Measured while building this: LCID scored +55 with its spread column empty and
    -315 once the orderbook supplied the real 70bps quote.
    """
    feasibility = evaluate_symbol(
        "LCID",
        closes=_closes(1200, step_bps=20.0),
        spread_bps=None,
        round_trip_cost_bps=45.0,
        strategies=STRATEGIES,
        market="US",
    )
    assert feasibility.verdict == VERDICT_UNKNOWN
    assert feasibility.headroom_bps is None


def test_too_few_bars_to_estimate_dispersion_is_unknown() -> None:
    feasibility = evaluate_symbol(
        "003010",
        closes=_closes(12, step_bps=20.0),
        spread_bps=10.0,
        round_trip_cost_bps=28.0,
        strategies=STRATEGIES,
        market="KR",
    )
    assert feasibility.verdict == VERDICT_UNKNOWN


# --- Ranking ------------------------------------------------------------------- #


def _entry(symbol: str, headroom: float | None, verdict: str) -> SymbolFeasibility:
    return SymbolFeasibility(
        symbol=symbol,
        market="KR",
        verdict=verdict,
        bar_count=500,
        spread_bps=10.0,
        sigma_1m_bps=20.0,
        attainable_bps=None,
        required_bps=None,
        headroom_bps=headroom,
    )


def test_ranking_puts_reachable_charts_first_and_unknown_in_the_middle() -> None:
    """UNKNOWN scores neutral: a data gap must not read as either verdict."""
    symbols = ("BAD", "UNSEEN", "GOOD")
    feasibility = {
        "BAD": _entry("BAD", -150.0, VERDICT_INFEASIBLE),
        "UNSEEN": _entry("UNSEEN", None, VERDICT_UNKNOWN),
        "GOOD": _entry("GOOD", 90.0, VERDICT_FEASIBLE),
    }
    assert rank_by_feasibility(symbols, feasibility) == ("GOOD", "UNSEEN", "BAD")


def test_ranking_is_stable_so_turnover_still_breaks_ties() -> None:
    """Turnover is not discarded — it is the liquidity prerequisite that built the list."""
    symbols = ("FIRST", "SECOND", "THIRD")
    feasibility = {name: _entry(name, 10.0, VERDICT_FEASIBLE) for name in symbols}
    assert rank_by_feasibility(symbols, feasibility) == symbols


def test_ranking_keeps_every_symbol() -> None:
    """This reorders; truncation to the universe size is what drops the tail."""
    symbols = ("A", "B", "C", "D")
    feasibility = {"A": _entry("A", -10.0, VERDICT_INFEASIBLE)}
    assert set(rank_by_feasibility(symbols, feasibility)) == set(symbols)

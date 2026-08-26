"""Scenario tests: the market conditions where a short actually loses money.

The unit tests in ``test_directional_short_ladder.py`` pin contracts. These pin
BEHAVIOUR under the specific tapes that make shorting dangerous — a squeeze, a
vanishing locate, a broker outage, a partial fill into a rally, a restart with an
open position. Each one is a way the system could look correct in isolation and
still lose money in practice.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
from datetime import datetime, timedelta, timezone

import pytest

from app.trading.borrow import (
    BorrowSnapshot,
    BorrowSnapshotStore,
    evaluate_borrow,
)
from app.trading.borrow_polling import BorrowPollingConfig, BorrowPollingService
from app.trading.directional import (
    DirectionalStrategyKey,
    PositionDirection,
    ShortReasonCodes,
    StrategyDeploymentState,
)
from app.trading.directional_shadow import (
    OUTCOME_BORROW_RECALLED,
    OUTCOME_EXPIRED,
    OUTCOME_STOP,
    OUTCOME_UNFILLED,
    QuoteObservation,
    ShadowFillSimulator,
    ShadowPlanStore,
    ShadowTradePlan,
)
from app.trading.shadow_evaluation_service import ShadowEvaluationService
from app.trading.short_strategy_promotion import (
    DeploymentStateStore,
    RuntimeHealth,
    ShortStrategyPromotionConfig,
    ShortStrategyPromotionController,
    evaluate_immediate_suspension,
)
from app.trading.strategy_performance_store import (
    EVALUATION_SOURCE_SHADOW,
    StrategyPerformanceStore,
)

NOW = datetime(2026, 8, 1, 5, 0, tzinfo=timezone.utc)
KEY = DirectionalStrategyKey.for_short("opening_range_breakdown", "KR")


def _locate(**overrides) -> BorrowSnapshot:
    payload = dict(
        symbol="005930",
        observed_at=NOW - timedelta(seconds=30),
        available=True,
        available_quantity=500,
        borrow_fee_bps_annualised=800.0,
    )
    payload.update(overrides)
    return BorrowSnapshot(**payload)


def _plan(**overrides) -> ShadowTradePlan:
    payload = dict(
        plan_id="plan-1",
        key=KEY,
        symbol="005930",
        signal_at=NOW,
        entry_reference_price=1000.0,
        target_rate=0.018,
        stop_rate=0.006,
        max_holding_seconds=3600,
        expected_trading_cost_bps=28.0,
        predicted_net_edge_bps=100.0,
        borrow_snapshot=_locate(),
        intended_quantity=10,
    )
    payload.update(overrides)
    return ShadowTradePlan(**payload)


def _simulator() -> ShadowFillSimulator:
    return ShadowFillSimulator(entry_slippage_bps=0.0, exit_slippage_bps=0.0)


# --------------------------------------------------------------------------- #
# Squeeze                                                                      #
# --------------------------------------------------------------------------- #
def test_short_squeeze_stops_out_and_records_a_real_loss() -> None:
    """A squeeze is where a short's unbounded loss materialises.

    The stop must fire on the ASK (what a cover actually pays), not the bid — using
    the bid would report an exit price the position could never have achieved and
    understate every squeeze loss.
    """
    simulator = _simulator()
    simulator.submit(_plan())
    simulator.observe_symbol(
        "005930", QuoteObservation(NOW + timedelta(seconds=1), 999.0, 1001.0)
    )
    # Violent rally: stop is 1006, the ask gaps to 1080.
    outcomes = simulator.observe_symbol(
        "005930", QuoteObservation(NOW + timedelta(seconds=120), 1078.0, 1080.0)
    )
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.outcome == OUTCOME_STOP
    assert outcome.exit_price == pytest.approx(1080.0)
    # A short covering higher LOST money: the sign must be negative.
    assert outcome.gross_return_bps < 0
    assert outcome.net_return_bps < outcome.gross_return_bps  # costs make it worse
    # The realized loss exceeds the nominal 60bps stop, because the gap jumped it.
    assert outcome.net_return_bps < -600


def test_squeeze_loss_is_not_capped_by_the_stop_distance() -> None:
    """A stop is a trigger, not a guarantee. Modelling it as a cap would make every
    backtest of a short look bounded when the real risk is not."""
    simulator = _simulator()
    simulator.submit(_plan())
    simulator.observe_symbol(
        "005930", QuoteObservation(NOW + timedelta(seconds=1), 999.0, 1001.0)
    )
    outcomes = simulator.observe_symbol(
        "005930", QuoteObservation(NOW + timedelta(seconds=60), 1490.0, 1500.0)
    )
    # Entered ~999, covered 1500: roughly -5000bps. Nothing clamps it.
    assert outcomes[0].gross_return_bps < -4_500


# --------------------------------------------------------------------------- #
# Borrow supply                                                                #
# --------------------------------------------------------------------------- #
def test_borrow_exhausted_between_signal_and_order_blocks_the_entry() -> None:
    """Availability is re-checked at order time, not trusted from the signal."""
    fresh = _locate(observed_at=NOW, available_quantity=500)
    assert evaluate_borrow(fresh, quantity=10, now=NOW).allowed
    exhausted = _locate(observed_at=NOW, available=True, available_quantity=3)
    verdict = evaluate_borrow(exhausted, quantity=10, now=NOW)
    assert not verdict.allowed
    assert ShortReasonCodes.BORROW_QUANTITY_INSUFFICIENT in verdict.reason_codes


def test_recall_mid_hold_forces_a_cover_at_whatever_the_book_offers() -> None:
    """The lender, not the strategy, decides the exit time."""
    simulator = _simulator()
    recalled = _plan(
        borrow_snapshot=_locate(return_deadline=NOW + timedelta(seconds=300))
    )
    simulator.submit(recalled)
    simulator.observe_symbol(
        "005930", QuoteObservation(NOW + timedelta(seconds=1), 999.0, 1001.0)
    )
    # Past the deadline, and the price has moved AGAINST the short.
    outcomes = simulator.observe_symbol(
        "005930", QuoteObservation(NOW + timedelta(seconds=301), 1002.0, 1004.0)
    )
    assert outcomes[0].outcome == OUTCOME_BORROW_RECALLED
    assert ShortReasonCodes.RECALL_DEADLINE_NEAR in outcomes[0].reason_codes
    # Forced covers still count as real outcomes — excluding them would hide the
    # cost of recall risk from every promotion statistic.
    assert outcomes[0].scored


def test_borrow_api_outage_never_becomes_a_false_no_locate(tmp_path) -> None:
    """"The broker said no" and "we could not ask" need different responses.

    An outage written as ``available=False`` would look like normal market state and
    hide a credentials or endpoint failure.
    """
    store = BorrowSnapshotStore(tmp_path / "borrow.sqlite3")

    def _broken(symbol: str) -> BorrowSnapshot:
        raise RuntimeError("KIS 500")

    poller = BorrowPollingService(
        availability_provider=_broken,
        store=store,
        config=BorrowPollingConfig(enabled=True, interval_seconds=1.0, max_lookups_per_cycle=5),
    )
    poller.track(["005930"])
    stats = poller.poll_once(now=NOW)
    assert stats.failed == 1
    assert stats.recorded == 0
    # No snapshot at all — not a False one.
    assert store.latest("005930", as_of=NOW) is None


def test_repeatedly_failing_symbol_is_parked_so_it_cannot_starve_the_budget(
    tmp_path,
) -> None:
    store = BorrowSnapshotStore(tmp_path / "borrow.sqlite3")

    def _broken(symbol: str) -> BorrowSnapshot:
        raise RuntimeError("KIS 500")

    poller = BorrowPollingService(
        availability_provider=_broken,
        store=store,
        config=BorrowPollingConfig(
            enabled=True,
            interval_seconds=0.0,
            max_lookups_per_cycle=5,
            failure_threshold=2,
            failure_backoff_seconds=300.0,
        ),
    )
    poller.track(["BAD"])
    poller.poll_once(now=NOW)
    poller.poll_once(now=NOW + timedelta(seconds=1))
    assert "BAD" in poller.status()["parked_symbols"]
    # Parked: consumes no further budget...
    assert poller.poll_once(now=NOW + timedelta(seconds=2)).polled == 0
    # ...until the backoff elapses.
    assert poller.poll_once(now=NOW + timedelta(seconds=400)).polled == 1


def test_polling_is_demand_driven_and_budgeted(tmp_path) -> None:
    """KIS rate-limits. A full sweep per tick would starve the LONG path's calls."""
    store = BorrowSnapshotStore(tmp_path / "borrow.sqlite3")
    seen: list[str] = []

    def _ok(symbol: str) -> BorrowSnapshot:
        seen.append(symbol)
        return _locate(symbol=symbol, observed_at=NOW)

    poller = BorrowPollingService(
        availability_provider=_ok,
        store=store,
        config=BorrowPollingConfig(enabled=True, interval_seconds=60.0, max_lookups_per_cycle=2),
    )
    poller.track(["A", "B", "C", "D", "E"])
    assert poller.poll_once(now=NOW).polled == 2
    assert poller.poll_once(now=NOW).polled == 2
    # Five symbols, two per cycle — never all at once.
    assert len(seen) == 4


# --------------------------------------------------------------------------- #
# Execution quality                                                            #
# --------------------------------------------------------------------------- #
def test_thin_book_produces_a_partial_fill_not_a_full_position() -> None:
    """Thin books are exactly where these strategies fire, so assuming a full fill is
    a systematic bias, not an edge case."""
    simulator = _simulator()
    simulator.submit(_plan(intended_quantity=100))
    simulator.observe_symbol(
        "005930",
        QuoteObservation(
            NOW + timedelta(seconds=1), 999.0, 1001.0, bid_size=25.0, ask_size=500.0
        ),
    )
    outcomes = simulator.observe_symbol(
        "005930", QuoteObservation(NOW + timedelta(seconds=600), 979.0, 981.0)
    )
    assert outcomes[0].fill_ratio == pytest.approx(0.25)
    assert "SHADOW_PARTIAL_FILL" in outcomes[0].reason_codes


def test_unknown_depth_does_not_block_the_plan() -> None:
    """A feed that omits sizes must not turn a data gap into a permanent block."""
    simulator = _simulator()
    simulator.submit(_plan())
    simulator.observe_symbol(
        "005930", QuoteObservation(NOW + timedelta(seconds=1), 999.0, 1001.0)
    )
    outcomes = simulator.observe_symbol(
        "005930", QuoteObservation(NOW + timedelta(seconds=600), 979.0, 981.0)
    )
    assert outcomes[0].fill_ratio == pytest.approx(1.0)


def test_wide_spread_is_paid_on_both_legs() -> None:
    """A short sells the bid and covers the ask; a mid fill would invent the spread
    back as profit.

    Both plans walk the identical MID path (1000 -> 980) and exit on time, so the only
    difference is the book. Exiting on the time barrier rather than the target is
    deliberate: on a wide book the ask may never reach the target at all, which is
    correct behaviour but would leave nothing to compare.
    """

    def _run(plan_id: str, entry: tuple[float, float], exit_book: tuple[float, float]):
        simulator = _simulator()
        simulator.submit(_plan(plan_id=plan_id, max_holding_seconds=300))
        simulator.observe_symbol(
            "005930", QuoteObservation(NOW + timedelta(seconds=1), *entry)
        )
        return simulator.observe_symbol(
            "005930", QuoteObservation(NOW + timedelta(seconds=400), *exit_book)
        )

    tight = _run("tight", (999.5, 1000.5), (979.5, 980.5))
    wide = _run("wide", (995.0, 1005.0), (975.0, 985.0))
    assert tight and wide
    # Same mid path, worse book: the wide-spread trade must earn strictly less.
    assert wide[0].gross_return_bps < tight[0].gross_return_bps
    # And the difference is roughly the extra spread paid on the two legs.
    assert tight[0].gross_return_bps - wide[0].gross_return_bps > 80.0


def test_wide_book_can_prevent_the_target_from_ever_being_reached() -> None:
    """The mid can print through the target while the ask never does.

    A mid-price simulator would report a winner here. The cover has to actually buy
    the ask, and on a 100bps-wide book that price never gets there.
    """
    simulator = _simulator()
    simulator.submit(_plan())  # target = 982
    simulator.observe_symbol(
        "005930", QuoteObservation(NOW + timedelta(seconds=1), 995.0, 1005.0)
    )
    # Mid is 980 (through the target), but the ask is 985.
    assert (
        simulator.observe_symbol(
            "005930", QuoteObservation(NOW + timedelta(seconds=600), 975.0, 985.0)
        )
        == ()
    )
    assert simulator.open_plan_count == 1


def test_entry_that_never_fills_is_recorded_as_unfilled_not_as_a_loss() -> None:
    """A plan whose feed dies must not vanish — feeds die when markets move, so the
    survivors would be a biased sample."""
    simulator = _simulator()
    simulator.submit(_plan(max_holding_seconds=60))
    outcomes = simulator.expire(NOW + timedelta(seconds=120))
    assert outcomes[0].outcome == OUTCOME_UNFILLED
    assert not outcomes[0].scored


def test_expiry_without_post_entry_quote_is_not_scored_as_a_loss() -> None:
    simulator = _simulator()
    simulator.submit(_plan(max_holding_seconds=60))
    simulator.observe_symbol(
        "005930", QuoteObservation(NOW + timedelta(seconds=1), 999.0, 1001.0)
    )

    outcome = simulator.expire(NOW + timedelta(seconds=120))[0]

    assert outcome.outcome == OUTCOME_EXPIRED
    assert outcome.net_return_bps is None
    assert outcome.scored is False


def test_expiry_uses_latest_executable_quote_not_worst_excursion() -> None:
    simulator = _simulator()
    simulator.submit(_plan(max_holding_seconds=60))
    simulator.observe_symbol(
        "005930", QuoteObservation(NOW + timedelta(seconds=1), 999.0, 1001.0)
    )
    simulator.observe_symbol(
        "005930", QuoteObservation(NOW + timedelta(seconds=10), 1001.0, 1003.0)
    )
    simulator.observe_symbol(
        "005930", QuoteObservation(NOW + timedelta(seconds=20), 997.0, 999.0)
    )

    outcome = simulator.expire(NOW + timedelta(seconds=90))[0]

    assert outcome.outcome == OUTCOME_EXPIRED
    assert outcome.exit_price == pytest.approx(999.0)
    assert outcome.max_adverse_excursion_bps > 0.0


# --------------------------------------------------------------------------- #
# Restart and reconciliation                                                   #
# --------------------------------------------------------------------------- #
def test_restart_without_restored_borrow_state_suspends_the_arm() -> None:
    """New entries stop; existing shorts stay manageable."""
    from tests.test_directional_short_ladder import _passing_snapshot

    failures = evaluate_immediate_suspension(
        _passing_snapshot(broker_state_restored=False), ShortStrategyPromotionConfig()
    )
    assert ShortReasonCodes.BROKER_STATE_UNRESTORED in failures


def test_in_flight_walks_are_lost_on_restart_rather_than_replayed(tmp_path) -> None:
    """Reconstructing a barrier walk after a restart would replay quotes we did not
    observe at the time — the exact leak this subsystem exists to prevent. Losing the
    sample is the correct trade."""
    first = ShadowEvaluationService(
        simulator=_simulator(),
        shadow_store=ShadowPlanStore(tmp_path / "shadow.sqlite3"),
        performance_store=StrategyPerformanceStore(
            tmp_path / "perf.sqlite3", clock=lambda: NOW
        ),
    )
    first.evaluate_tick({}, now=NOW, new_plans=[_plan()])
    assert first.status()["open_plans"] == 1
    # A fresh process reading the same stores adopts nothing automatically.
    second = ShadowEvaluationService(
        simulator=_simulator(),
        shadow_store=ShadowPlanStore(tmp_path / "shadow.sqlite3"),
        performance_store=StrategyPerformanceStore(tmp_path / "perf.sqlite3"),
    )
    assert second.status()["open_plans"] == 0


def test_capacity_overflow_is_deferred_instead_of_orphaned(tmp_path) -> None:
    service = ShadowEvaluationService(
        simulator=_simulator(),
        shadow_store=ShadowPlanStore(tmp_path / "shadow.sqlite3"),
        performance_store=StrategyPerformanceStore(tmp_path / "perf.sqlite3"),
        max_open_plans=1,
    )
    second = dataclasses.replace(_plan(), plan_id="deferred-plan")

    first_stats = service.evaluate_tick({}, now=NOW, new_plans=[_plan(), second])
    assert first_stats.adopted == 1
    assert first_stats.open_plans == 1

    # The active walk expires first. On the following cycle the overflow plan is
    # still owned by the service and is adopted without being re-supplied by the
    # session manager.
    service.evaluate_tick({}, now=NOW + timedelta(seconds=4000))
    third_stats = service.evaluate_tick({}, now=NOW + timedelta(seconds=4001))
    assert third_stats.adopted == 1
    assert third_stats.resolved == 1
    assert any(
        item["plan_id"] == "deferred-plan"
        for item in service.shadow_store.outcomes(limit=10)
    )


# --------------------------------------------------------------------------- #
# Regime                                                                       #
# --------------------------------------------------------------------------- #
def test_dislocated_book_blocks_both_directions_not_just_longs() -> None:
    """A dislocated book is not a short opportunity — it is a market whose prices are
    not information."""
    from app.ontology.short_rules import blocked_directions, permits_new_entry

    assert not permits_new_entry("HIGH_VOL_DISLOCATED")
    assert set(blocked_directions("HIGH_VOL_DISLOCATED")) == {"LONG", "SHORT"}


def test_change_point_suspends_a_live_short_within_one_cycle(tmp_path) -> None:
    from tests.test_directional_short_ladder import _passing_snapshot

    controller = ShortStrategyPromotionController(
        config=ShortStrategyPromotionConfig(
            strategies={"opening_range_breakdown": {"enabled": True}}
        ),
        state_store=DeploymentStateStore(tmp_path / "dep.sqlite3"),
        performance_store=StrategyPerformanceStore(tmp_path / "perf.sqlite3"),
        shadow_store=ShadowPlanStore(tmp_path / "shadow.sqlite3"),
        borrow_store=BorrowSnapshotStore(tmp_path / "borrow.sqlite3"),
    )
    store = controller.state_store
    store.ensure(KEY, StrategyDeploymentState.SHADOW)
    store.force_state(KEY, StrategyDeploymentState.LIVE_PROBE, actor="t", reason="setup")
    breaking = _passing_snapshot(
        state=StrategyDeploymentState.LIVE_PROBE, change_point_probability=0.85
    )
    decision = controller.decide(breaking, store.get(KEY))
    assert decision.to_state is StrategyDeploymentState.SUSPENDED
    assert ShortReasonCodes.REGIME_UNSTABLE in decision.reason_codes


def test_sector_weakness_rank_mirrors_the_strength_rank() -> None:
    """A symbol must never be simultaneously the strongest and the weakest."""
    from app.graph.macro_reasoner import build_sector_rank_table

    table = build_sector_rank_table(
        sector_of={"A": "semi", "B": "semi", "C": "semi"},
        residual_returns={"A": -0.004, "B": 0.002, "C": 0.006},
    )
    assert table.rank_for("C") == (1, 3)
    assert table.weakness_rank_for("C") == (3, 3)
    assert table.rank_for("A") == (3, 3)
    assert table.weakness_rank_for("A") == (1, 3)
    # Unanswerable stays unanswerable in both views.
    assert table.rank_for("ZZ") is None
    assert table.weakness_rank_for("ZZ") is None


def test_short_election_context_populates_the_borrow_facts() -> None:
    """Without this the short algorithms are registered, evaluated, and permanently
    inert — the exact defect the long session-boxed strategies already hit once."""
    from app.graph.macro_reasoner import build_sector_rank_table
    from app.trading.strategy_session import _short_election_context

    table = build_sector_rank_table(
        sector_of={"005930": "semi", "000660": "semi"},
        residual_returns={"005930": -0.004, "000660": 0.002},
        long_residual_returns={"005930": -0.003, "000660": 0.001},
    )
    context = _short_election_context(
        symbol="005930",
        borrow_snapshot=_locate(),
        macro=None,
        micro_diagnostics={"liquidity_score": 0.8, "spread_bps": 12.0},
        table=table,
        now=NOW,
    )
    assert context["borrow_available"] is True
    assert context["short_sale_permitted"] is True
    assert context["borrow_fee_bps_annualised"] == pytest.approx(800.0)
    assert context["borrow_available_quantity"] == 500
    assert "borrow_observed_at" in context
    # Weak-end rank, and the residuals under SHORT field names.
    assert context["sector_rank"] == 1
    assert context["residual_short_bps"] == pytest.approx(-40.0)
    # No snapshot -> no borrow facts at all (omitted, not defaulted to a passing value).
    empty = _short_election_context(
        symbol="005930",
        borrow_snapshot=None,
        macro=None,
        micro_diagnostics=None,
        table=table,
        now=NOW,
    )
    assert "borrow_available" not in empty
    assert "short_sale_permitted" not in empty


def test_short_algorithm_fires_only_with_a_complete_context() -> None:
    """End-to-end: the populated context makes the thesis reachable, and each borrow
    gate individually makes it unreachable again."""
    from app.graph.macro_reasoner import build_sector_rank_table
    from app.technical.signals import TechnicalFeatureSet
    from app.technical.strategy_algorithms import ElectionContext, get_algorithm
    from app.trading.strategy_session import _short_election_context

    table = build_sector_rank_table(
        sector_of={"005930": "semi", "000660": "semi", "035420": "semi"},
        residual_returns={"005930": -0.004, "000660": 0.002, "035420": 0.006},
        long_residual_returns={"005930": -0.003, "000660": 0.001, "035420": 0.005},
    )
    raw = _short_election_context(
        symbol="005930",
        borrow_snapshot=_locate(),
        macro=None,
        micro_diagnostics={
            "liquidity_score": 0.8,
            "spread_bps": 12.0,
            "days_to_cover": 2.0,
            "short_interest_ratio": 0.05,
        },
        table=table,
        now=NOW,
    )
    allowed = ElectionContext.__dataclass_fields__.keys()
    payload = {key: value for key, value in raw.items() if key in allowed}
    payload.update(
        strategy_id="residual_relative_weakness",
        elected_at=NOW,
        foreign_flow_zscore=-1.2,
        change_point_probability=0.1,
    )
    context = ElectionContext(**payload)
    features = TechnicalFeatureSet(
        symbol="005930",
        price=70_000,
        second_data_ready=1.0,
        tick_count_5s=10,
        realized_volatility_10s=0.004,
        relative_volume=1.5,
        aggressor_imbalance_5s=-0.3,
        spread_bps=12.0,
        orderbook_imbalance=-0.4,
        short_return=-0.01,
        return_5s=-0.002,
    )
    algorithm = get_algorithm("residual_relative_weakness")
    assert algorithm.entry(features, context).triggered

    # The stop sits ABOVE entry and the target BELOW it.
    rule = algorithm.exit_rule(70_000, features, context)
    assert rule.stop_price > 70_000
    assert rule.target_price < 70_000

    for override, expected in [
        ({"borrow_available": False}, ShortReasonCodes.BORROW_UNAVAILABLE),
        ({"short_sale_permitted": False}, ShortReasonCodes.SHORT_SALE_NOT_PERMITTED),
        ({"borrow_fee_bps_annualised": None}, ShortReasonCodes.BORROW_COST_TOO_HIGH),
        ({"borrow_available_quantity": 0}, ShortReasonCodes.BORROW_QUANTITY_INSUFFICIENT),
        ({"days_to_cover": 9.0}, "SHORT_CROWDED_DAYS_TO_COVER"),
        ({"liquidity_score": 0.1}, "SHORT_EXECUTION_LIQUIDITY_INSUFFICIENT"),
        (
            {"borrow_observed_at": (NOW - timedelta(hours=2)).isoformat()},
            ShortReasonCodes.BORROW_SNAPSHOT_STALE,
        ),
    ]:
        blocked = algorithm.entry(features, dataclasses.replace(context, **override))
        assert not blocked.triggered, override
        assert expected in blocked.reason_codes, override


def test_borrow_observed_after_the_signal_is_refused() -> None:
    """Future information relative to the decision is a leak, not freshness."""
    from app.technical.strategy_algorithms import ElectionContext, get_algorithm
    from app.technical.signals import TechnicalFeatureSet

    context = ElectionContext(
        strategy_id="opening_range_breakdown",
        elected_at=NOW,
        short_sale_permitted=True,
        borrow_available=True,
        borrow_available_quantity=500,
        borrow_fee_bps_annualised=800.0,
        # Observed AFTER the signal.
        borrow_observed_at=(NOW + timedelta(seconds=60)).isoformat(),
    )
    features = TechnicalFeatureSet(
        symbol="005930", price=70_000, second_data_ready=1.0, tick_count_5s=10
    )
    decision = get_algorithm("opening_range_breakdown").entry(features, context)
    assert ShortReasonCodes.BORROW_SNAPSHOT_STALE in decision.reason_codes


# --------------------------------------------------------------------------- #
# The full loop                                                                #
# --------------------------------------------------------------------------- #
def test_signal_to_promotion_evidence_in_one_pass(tmp_path) -> None:
    """Journal a plan, walk it, score it, and see it as promotion evidence.

    This is the loop that was missing: without it plans accumulate and nothing can
    ever leave SHADOW.
    """
    service = ShadowEvaluationService(
        simulator=_simulator(),
        shadow_store=ShadowPlanStore(tmp_path / "shadow.sqlite3"),
        performance_store=StrategyPerformanceStore(
            tmp_path / "perf.sqlite3", clock=lambda: NOW
        ),
    )
    service.evaluate_tick({}, now=NOW, new_plans=[_plan(regime="TREND_DOWN")])
    service.evaluate_tick(
        {
            "005930": {
                "bid_price": 999.0,
                "ask_price": 1001.0,
                "observed_at": NOW + timedelta(seconds=1),
            }
        },
        now=NOW + timedelta(seconds=1),
    )
    service.evaluate_tick(
        {
            "005930": {
                "bid_price": 979.0,
                "ask_price": 981.0,
                "observed_at": NOW + timedelta(seconds=600),
            }
        },
        now=NOW + timedelta(seconds=600),
    )
    metrics = service.performance_store.directional_metrics(
        KEY, evaluation_sources=(EVALUATION_SOURCE_SHADOW,)
    )
    assert metrics["filled_trade_count"] == 1
    assert metrics["mean_net_return_bps"] > 0
    outcomes = service.performance_store.recent_outcomes(
        KEY.strategy_id,
        market=KEY.market,
        regime="TREND_DOWN",
        direction=str(KEY.direction),
        execution_product=str(KEY.execution_product),
        evaluation_sources=(EVALUATION_SOURCE_SHADOW,),
    )
    assert len(outcomes) == 1
    assert outcomes[0].regime == "TREND_DOWN"

    controller = ShortStrategyPromotionController(
        config=ShortStrategyPromotionConfig(
            strategies={"opening_range_breakdown": {"enabled": True}}
        ),
        state_store=DeploymentStateStore(tmp_path / "dep.sqlite3"),
        performance_store=service.performance_store,
        shadow_store=service.shadow_store,
        borrow_store=BorrowSnapshotStore(tmp_path / "borrow.sqlite3"),
    )
    snapshot = controller.build_snapshot(KEY, StrategyDeploymentState.SHADOW)
    # One sample is real evidence and is counted...
    assert snapshot.filled_trade_count == 1
    # ...and nowhere near enough to promote.
    decision = controller.evaluate(KEY, health=RuntimeHealth())
    assert decision.to_state is StrategyDeploymentState.SHADOW
    assert ShortReasonCodes.PROMOTION_SAMPLE_INSUFFICIENT in decision.failed_gates


def test_polling_is_inert_without_a_configured_source() -> None:
    """The SOURCE is the gate, not a flag.

    Polling is enabled by default so that configuring a source is all an operator has
    to do. With no source the provider is ``None`` and nothing is called — inert
    without needing a second switch to remember.
    """
    poller = BorrowPollingService(availability_provider=None)
    poller.track(["005930"])
    assert poller.poll_once(now=NOW).polled == 0
    assert poller.status()["provider_available"] is False
    # And an explicit off-switch still suppresses a configured source.
    calls: list[str] = []

    def _provider(symbol: str) -> BorrowSnapshot:
        calls.append(symbol)
        return _locate(symbol=symbol)

    off = BorrowPollingService(
        availability_provider=_provider, config=BorrowPollingConfig(enabled=False)
    )
    off.track(["005930"])
    assert off.poll_once(now=NOW).polled == 0
    assert calls == []


# --------------------------------------------------------------------------- #
# Borrow data source                                                           #
# --------------------------------------------------------------------------- #
# Three KIS 대주 endpoints were guessed and all three disproven by read-only probes
# against the live account (wrong semantic / no such TR / HTTP 404). Availability now
# lives behind a source interface whose default is an explicit "not configured".
def test_no_borrow_source_reports_absence_rather_than_unavailability() -> None:
    """"No source configured" must not look like "nothing is borrowable".

    Returning ``available=False`` would be indistinguishable from a normal market
    state, and the system would appear to work while having no data path at all.
    """
    from app.trading.borrow_source import REASON_NO_SOURCE, NullBorrowSource

    source = NullBorrowSource()
    assert not source.available()
    assert source.snapshot("005930", now=NOW) is None
    status = source.status()
    assert status["reason"] == REASON_NO_SOURCE
    assert "no shadow evidence accumulates" in status["detail"]


def test_file_borrow_source_round_trips_a_locate(tmp_path) -> None:
    from app.trading.borrow_source import FileBorrowSource

    path = tmp_path / "borrow.json"
    path.write_text(
        json.dumps(
            {
                "observed_at": (NOW - timedelta(seconds=10)).isoformat(),
                "source": "manual",
                "symbols": {
                    "005930": {
                        "available": True,
                        "quantity": 500,
                        "fee_bps_annualised": 800,
                    },
                    "000660": {"available": False, "reason": "no inventory"},
                },
            }
        ),
        encoding="utf-8",
    )
    source = FileBorrowSource(path)
    assert source.available()
    snapshot = source.snapshot("005930", now=NOW)
    assert snapshot is not None and snapshot.available
    assert snapshot.available_quantity == 500
    assert snapshot.borrow_fee_bps_annualised == pytest.approx(800.0)
    # The file's own timestamp is used, NOT `now` — that is what lets the standard
    # freshness rule catch a stale file.
    assert snapshot.observed_at == NOW - timedelta(seconds=10)
    assert evaluate_borrow(snapshot, quantity=10, now=NOW).allowed

    # An explicit refusal is a real answer.
    refused = source.snapshot("000660", now=NOW)
    assert refused is not None and not refused.available
    # ...but an unlisted symbol is UNANSWERED, not refused.
    assert source.snapshot("035420", now=NOW) is None
    assert source.universe() == ("005930",)


def test_file_without_observed_at_is_refused(tmp_path) -> None:
    """An undated locate cannot be point-in-time evaluated.

    Stamping it "now" would make a week-old file look fresh — the same look-ahead leak
    the shadow evaluator exists to prevent.
    """
    from app.trading.borrow_source import FileBorrowSource

    path = tmp_path / "borrow.json"
    path.write_text(
        json.dumps({"symbols": {"005930": {"available": True, "quantity": 10}}}),
        encoding="utf-8",
    )
    source = FileBorrowSource(path)
    assert not source.available()
    assert source.snapshot("005930", now=NOW) is None
    assert source.status()["reason"] == "BORROW_SOURCE_MISSING_OBSERVED_AT"


def test_stale_file_fails_the_normal_freshness_rule(tmp_path) -> None:
    """A hand-maintained file is not exempt from staleness."""
    from app.trading.borrow_source import FileBorrowSource

    path = tmp_path / "borrow.json"
    path.write_text(
        json.dumps(
            {
                "observed_at": (NOW - timedelta(days=1)).isoformat(),
                "symbols": {
                    "005930": {
                        "available": True,
                        "quantity": 500,
                        "fee_bps_annualised": 800,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    snapshot = FileBorrowSource(path).snapshot("005930", now=NOW)
    assert snapshot is not None
    verdict = evaluate_borrow(snapshot, quantity=10, now=NOW)
    assert not verdict.allowed
    assert ShortReasonCodes.BORROW_SNAPSHOT_STALE in verdict.reason_codes


def test_file_source_reloads_on_change(tmp_path) -> None:
    """An operator must be able to update the locate without a restart."""
    from app.trading.borrow_source import FileBorrowSource

    path = tmp_path / "borrow.json"

    def _write(available: bool, mtime_bump: int) -> None:
        path.write_text(
            json.dumps(
                {
                    "observed_at": NOW.isoformat(),
                    "symbols": {
                        "005930": {
                            "available": available,
                            "quantity": 500,
                            "fee_bps_annualised": 800,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        os.utime(path, (1_800_000_000 + mtime_bump, 1_800_000_000 + mtime_bump))

    source = FileBorrowSource(path)
    _write(True, 0)
    first = source.snapshot("005930", now=NOW)
    assert first is not None and first.available
    _write(False, 60)
    second = source.snapshot("005930", now=NOW)
    assert second is not None and not second.available


def test_unreadable_file_never_becomes_a_false_locate(tmp_path) -> None:
    from app.trading.borrow_source import FileBorrowSource

    path = tmp_path / "borrow.json"
    path.write_text("{ not json", encoding="utf-8")
    source = FileBorrowSource(path)
    assert not source.available()
    assert source.snapshot("005930", now=NOW) is None
    assert source.status()["reason"].startswith("BORROW_SOURCE_UNREADABLE")


def test_callable_source_treats_an_error_as_unanswered() -> None:
    """A raising provider yields ``None`` (unanswered), never ``available=False``."""
    from app.trading.borrow_source import CallableBorrowSource

    def _broken(symbol: str):
        raise RuntimeError("broker down")

    source = CallableBorrowSource(_broken, label="kis")
    assert source.available()
    assert source.snapshot("005930", now=NOW) is None
    assert "broker down" in source.status()["reason"]


def test_kis_borrow_source_uses_live_inventory_and_conservative_fee() -> None:
    from app.trading.borrow_source import KisBorrowSource

    class _Client:
        def get_lendable_by_company(self, symbol):
            assert symbol == "005930"
            return {
                "available": True,
                "available_quantity": 321,
                "raw": {"psbl_yn": "Y", "rqst_psbl_qty": "321"},
            }

    source = KisBorrowSource(client=_Client())
    snapshot = source.snapshot("005930", now=NOW)
    assert snapshot is not None and snapshot.available
    assert snapshot.available_quantity == 321
    # Unknown index membership uses KIS's higher "other" policy rate, never the
    # cheaper KOSPI200 rate by assumption.
    assert snapshot.borrow_fee_bps_annualised == pytest.approx(600.0)
    assert snapshot.source.startswith("kis:CTSC2702R")
    assert source.universe() == ("005930",)


def test_kis_lendable_query_normalizes_official_response() -> None:
    from app.execution.kis_real import KisDevelopersApiClient

    calls = []

    class _Transport:
        def request(self, method, url, headers=None, params=None, body=None, timeout=None):
            calls.append((method, url, headers, params))
            return {
                "rt_cd": "0",
                "output1": {"psbl_yn": "Y"},
                "output2": [
                    {
                        "pdno": "005930",
                        "rqst_psbl_qty": "1,234",
                        "trad_psbl_qty2": "1,500",
                    }
                ],
            }

    client = KisDevelopersApiClient(
        app_key="key",
        app_secret="secret",
        account_no="12345678",
        enabled=False,
        transport=_Transport(),
        access_token="token",
    )
    answer = client.get_lendable_by_company("005930")
    assert answer["available"] is True
    assert answer["available_quantity"] == 1234
    method, url, headers, params = calls[-1]
    assert method == "GET"
    assert url.endswith("/uapi/domestic-stock/v1/quotations/lendable-by-company")
    assert headers["tr_id"] == "CTSC2702R"
    assert params["PDNO"] == "005930"


def test_kis_lendable_query_treats_verified_empty_contract_as_unavailable() -> None:
    from app.execution.kis_real import KisDevelopersApiClient

    class _Transport:
        def request(self, method, url, headers=None, params=None, body=None, timeout=None):
            return {
                "rt_cd": "0",
                "msg_cd": "KIOK0560",
                "msg1": "조회할 내용이 없습니다",
                "output1": [],
                "output2": {
                    "tot_stup_lmt_qty": "0",
                    "brch_lmt_qty": "0",
                    "rqst_psbl_qty": "0",
                },
            }

    client = KisDevelopersApiClient(
        app_key="key",
        app_secret="secret",
        account_no="12345678",
        enabled=False,
        transport=_Transport(),
        access_token="token",
    )
    answer = client.get_lendable_by_company("005930")
    assert answer["available"] is False
    assert answer["available_quantity"] == 0
    assert answer["reject_reason"] == "KIS_BORROW_SYMBOL_NOT_LISTED"


def test_credit_order_contract_uses_official_kis_side_and_product_codes() -> None:
    from app.execution.kis_real import KisDevelopersApiClient, KisEndpointSet
    from app.schemas.domain import FinalOrder, OrderSide, OrderType

    endpoints = KisEndpointSet.for_mode(False)
    assert endpoints.credit_tr_id_for_order(OrderSide.SELL) == "TTTC0051U"
    assert endpoints.credit_tr_id_for_order(OrderSide.BUY) == "TTTC0052U"

    client = KisDevelopersApiClient(
        app_key="key",
        app_secret="secret",
        account_no="12345678",
        enabled=False,
        transport=object(),
        access_token="token",
    )
    common = dict(
        ticker="005930",
        market="KR",
        order_type=OrderType.LIMIT,
        quantity=1,
        limit_price=70000,
        position_direction="SHORT",
        execution_product="CREDIT_BORROW",
    )
    opening = FinalOrder(
        **common,
        side=OrderSide.SELL,
        position_effect="OPEN",
        credit_type="22",
    )
    closing = FinalOrder(
        **common,
        side=OrderSide.BUY,
        position_effect="CLOSE",
        credit_type="26",
        loan_date="20260810",
    )
    assert client._credit_order_body(opening, repay=False)["CRDT_TYPE"] == "22"
    close_body = client._credit_order_body(closing, repay=True)
    assert close_body["CRDT_TYPE"] == "26"
    assert close_body["LOAN_DT"] == "20260810"


def test_borrow_balance_reads_the_verified_endpoint_and_skips_cash_rows() -> None:
    """Reconciliation uses the SAME inquire-balance call the portfolio path already
    makes in production — the separate credit-balance endpoint returned 404."""
    from app.execution.kis_real import KisDevelopersApiClient

    class _Transport:
        def request(self, method, url, headers=None, params=None, body=None, timeout=None):
            if "inquire-balance" in url:
                return {
                    "rt_cd": "0",
                    "output1": [
                        # Plain cash long: no credit metadata at all -> not a lot.
                        {"pdno": "005930", "hldg_qty": "10", "pchs_avg_pric": "70000"},
                        # 대주 short.
                        {
                            "pdno": "000660",
                            "hldg_qty": "5",
                            "pchs_avg_pric": "150000",
                            "loan_dt": "20260801",
                            "loan_amt": "750000",
                            "crdt_type": "05",
                        },
                        # 융자 = leveraged LONG, must not count as short exposure.
                        {
                            "pdno": "035420",
                            "hldg_qty": "3",
                            "pchs_avg_pric": "200000",
                            "loan_dt": "20260801",
                            "loan_amt": "600000",
                            "crdt_type": "01",
                        },
                    ],
                }
            return {"rt_cd": "0", "output": {}}

    client = KisDevelopersApiClient(
        app_key="k",
        app_secret="s",
        account_no="1",
        account_product_code="01",
        transport=_Transport(),
        access_token="t",
        token_expires_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
        enabled=True,
    )
    lots = client.get_borrow_balance()
    assert {lot["symbol"] for lot in lots} == {"000660", "035420"}
    by_symbol = {lot["symbol"]: lot for lot in lots}
    assert by_symbol["000660"]["direction"] == "SHORT"
    assert by_symbol["035420"]["direction"] == "LONG"
    assert client.reconcile_credit_positions(internal_lots=())["broker_short_lot_count"] == 1


def test_shipped_example_borrow_file_is_valid_and_not_active() -> None:
    """The example must parse, and must NOT be the live file.

    A committed ``borrow_availability.json`` would silently make every listed symbol
    borrowable on someone else's machine.
    """
    example = pathlib.Path("config/borrow_availability.example.json")
    assert example.exists()
    payload = json.loads(example.read_text(encoding="utf-8"))
    assert payload.get("observed_at")
    assert payload.get("symbols")
    assert not pathlib.Path("config/borrow_availability.json").exists()


def _balance_client(rows: list[dict[str, str]]):
    """KIS client whose verified inquire-balance returns ``rows``."""
    from app.execution.kis_real import KisDevelopersApiClient

    class _Transport:
        def request(self, method, url, headers=None, params=None, body=None, timeout=None):
            if "inquire-balance" in url:
                return {"rt_cd": "0", "output1": rows}
            return {"rt_cd": "0", "output": {}}

    return KisDevelopersApiClient(
        app_key="k",
        app_secret="s",
        account_no="1",
        account_product_code="01",
        transport=_Transport(),
        access_token="t",
        token_expires_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
        enabled=True,
    )


def test_orphan_broker_short_blocks_new_entries_but_allows_closing() -> None:
    """We cannot manage an exit for a position whose thesis we do not know — but the
    position still has to be closeable, so it goes to close-only rather than frozen."""
    client = _balance_client(
        [
            {
                "pdno": "005930",
                "hldg_qty": "10",
                "pchs_avg_pric": "70000",
                "loan_dt": "20260801",
                "crdt_type": "05",
            }
        ]
    )
    verdict = client.reconcile_credit_positions(internal_lots=())
    assert verdict["orphan_lots"]
    assert verdict["new_short_entries_blocked"]
    assert verdict["close_only_mode"]
    assert verdict["position_direction_mismatch"]


def test_phantom_internal_short_blocks_new_entries() -> None:
    """A buy-to-cover for stock we do not owe would OPEN a long."""
    client = _balance_client([])
    verdict = client.reconcile_credit_positions(
        internal_lots=(
            {"symbol": "005930", "loan_date": "20260801", "direction": "SHORT", "quantity": 10},
        )
    )
    assert verdict["phantom_lots"]
    assert verdict["new_short_entries_blocked"]
    # Not close-only: there is nothing at the broker to close.
    assert not verdict["close_only_mode"]


def test_lot_without_a_loan_date_blocks_new_entries() -> None:
    """It cannot be repaid through the 매수상환 contract, which requires 대출일."""
    client = _balance_client(
        [{"pdno": "005930", "hldg_qty": "10", "pchs_avg_pric": "70000",
          "loan_amt": "700000", "crdt_type": "05"}]
    )
    verdict = client.reconcile_credit_positions(internal_lots=())
    assert verdict["loan_date_missing"]
    assert verdict["new_short_entries_blocked"]


def test_reconciliation_failure_is_not_a_clean_reconciliation() -> None:
    """An unanswered check must not read as "no discrepancies"."""
    from app.execution.kis_real import KisApiError, KisDevelopersApiClient

    class _Transport:
        def request(self, *args, **kwargs):
            raise KisApiError("KIS 500")

    client = KisDevelopersApiClient(
        app_key="k", app_secret="s", account_no="1", account_product_code="01",
        transport=_Transport(), access_token="t",
        token_expires_at=datetime(2030, 1, 1, tzinfo=timezone.utc), enabled=True,
    )
    verdict = client.reconcile_credit_positions(internal_lots=())
    assert not verdict["broker_state_restored"]
    assert verdict["new_short_entries_blocked"]


# --------------------------------------------------------------------------- #
# GNN borrow channels                                                          #
# --------------------------------------------------------------------------- #
def _gnn_model(strategy_count: int = 16):
    import numpy as np

    from app.models.strategy_utility.rgcn import (
        FixedShapeStrategyUtilityModel,
        StrategyUtilityModelConfig,
    )

    config = StrategyUtilityModelConfig(
        batch_size=1,
        time_steps=2,
        max_nodes=3,
        feature_dim=4,
        relation_count=2,
        strategy_count=strategy_count,
        hidden_dim=8,
        seed=17,
    )
    model = FixedShapeStrategyUtilityModel(config)
    rng = np.random.default_rng(0)
    inputs = (
        rng.normal(0, 1, (1, 2, 3, 4)).astype(np.float32),
        rng.random((1, 2, 2, 3, 3)).astype(np.float32),
        np.ones((1, 2, 3), dtype=np.float32),
        np.ones((1, 3, strategy_count), dtype=np.float32),
    )
    return model, inputs


def test_long_strategies_cannot_have_a_borrow_cost() -> None:
    """A cash long has no borrow leg, and the model must not be able to invent one.

    The mask is built from the catalogue rather than learned, so no training data can
    teach a long head to charge — or discount — a borrow.
    """
    from app.models.strategy_utility.rgcn import _SHORT_STRATEGY_INDICES

    model, inputs = _gnn_model()
    out = model.infer(*inputs)
    for index in range(16):
        if index in _SHORT_STRATEGY_INDICES:
            continue
        assert out.borrow_cost_bps[0, 0, index] == 0.0, index
        assert out.borrow_probability[0, 0, index] == 1.0, index


def test_short_strategies_get_real_borrow_channels() -> None:
    from app.models.strategy_utility.rgcn import _SHORT_STRATEGY_INDICES

    model, inputs = _gnn_model()
    out = model.infer(*inputs)
    assert _SHORT_STRATEGY_INDICES, "expected catalogued short strategies"
    for index in _SHORT_STRATEGY_INDICES:
        assert out.borrow_cost_bps[0, 0, index] > 0.0
        assert 0.0 <= out.borrow_probability[0, 0, index] <= 1.0
    assert out.epistemic_uncertainty is not None


def test_borrow_probability_scales_the_utility() -> None:
    """An edge that only exists on names you cannot borrow is not an edge."""
    import numpy as np

    from app.models.strategy_utility.rgcn import _SHORT_STRATEGY_INDICES, output_from_raw

    index = _SHORT_STRATEGY_INDICES[0]
    node_mask = np.ones((1, 1, 1), dtype=np.float32)
    strategy_mask = np.ones((1, 1, 16), dtype=np.float32)

    def _utility(borrow_logit: float) -> float:
        raw = np.zeros((1, 1, 16, 11), dtype=np.float32)
        raw[..., 0] = 2.0   # high success probability
        raw[..., 4] = 2.0   # meaningful MFE
        raw[0, 0, index, 9] = borrow_logit
        out = output_from_raw(raw, np.zeros((1, 1), dtype=np.float32), node_mask, strategy_mask)
        return float(out.utility[0, 0, index])

    assert _utility(4.0) > _utility(-4.0)


def test_widening_the_head_invalidates_old_checkpoints(tmp_path) -> None:
    """A schema change must fail closed, not load silently with missing channels."""
    import numpy as np

    from app.models.strategy_utility.rgcn import FixedShapeStrategyUtilityModel

    model, _ = _gnn_model()
    path = tmp_path / "ck.npz"
    model.save_checkpoint(path)
    # Round-trips at the current width.
    assert FixedShapeStrategyUtilityModel.load_checkpoint(path) is not None
    # An old 8-channel checkpoint is refused.
    data = dict(np.load(path, allow_pickle=False))
    data["strategy_heads"] = data["strategy_heads"][..., :8]
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="strategy_heads"):
        FixedShapeStrategyUtilityModel.load_checkpoint(path)


# --------------------------------------------------------------------------- #
# Short indicators                                                             #
# --------------------------------------------------------------------------- #
def test_unsourced_crowding_metrics_are_reported_not_silently_absent() -> None:
    """A silently-absent crowding metric looks identical to a check that passed."""
    from app.features.short_indicators import (
        UNSOURCED_SHORT_INDICATORS,
        ShortIndicators,
        short_indicator_gaps,
    )

    gaps = short_indicator_gaps(ShortIndicators())
    assert set(gaps["unsourced"]) == set(UNSOURCED_SHORT_INDICATORS)
    assert gaps["squeeze_filter_active"] is False
    assert "스퀴즈" in gaps["detail"]
    # With a source supplied, the filter reports active.
    supplied = ShortIndicators(short_interest_ratio=0.04, days_to_cover=1.5)
    assert short_indicator_gaps(supplied)["squeeze_filter_active"] is True


def test_computed_indicators_omit_what_they_cannot_measure() -> None:
    from app.features.short_indicators import compute_short_indicators

    class _Book:
        best_bid = 999.0
        best_ask = 1001.0

    indicators = compute_short_indicators(
        orderbook=_Book(), average_daily_trading_value=5_000_000_000
    )
    context = indicators.as_context()
    assert context["spread_bps"] == pytest.approx(20.0, rel=1e-3)
    assert 0.0 < context["liquidity_score"] <= 1.0
    # Unmeasured fields are OMITTED, not emitted as None — an absent key means
    # "unresolved" to ElectionContext, whereas None would look like an empty result.
    assert "market_alignment" not in context
    assert "days_to_cover" not in context


def test_crossed_book_yields_no_spread_rather_than_a_huge_one() -> None:
    from app.features.short_indicators import spread_bps_from_book

    class _Crossed:
        best_bid = 1001.0
        best_ask = 999.0

    assert spread_bps_from_book(_Crossed()) is None
    assert spread_bps_from_book(None) is None


def test_market_alignment_is_none_when_either_leg_is_flat() -> None:
    """Undefined agreement must not read as "perfectly neutral"."""
    from app.features.short_indicators import market_alignment

    assert market_alignment(-0.01, -0.02) == pytest.approx(0.5)
    assert market_alignment(-0.01, 0.02) == pytest.approx(-0.5)
    assert market_alignment(0.0, -0.02) is None
    assert market_alignment(None, -0.02) is None


def test_head_width_change_reports_schema_not_corruption(tmp_path) -> None:
    """"Corrupt" and "schema" send an operator in different directions.

    Corrupt means hunt for a damaged file; schema means retrain against the current
    contract. Widening the head for the borrow channels invalidates every pre-existing
    checkpoint, and reporting those as corruption would send everyone chasing files
    that are perfectly intact.
    """
    import numpy as np

    from app.routing.shadow_intelligence import ShadowIntelligenceService

    model, _ = _gnn_model()
    checkpoint = tmp_path / "rgcn.npz"
    model.save_checkpoint(checkpoint)
    data = dict(np.load(checkpoint, allow_pickle=False))
    data["strategy_heads"] = data["strategy_heads"][..., :8]
    np.savez_compressed(checkpoint, **data)

    previous = os.environ.get("REFACTOR_GNN_CHECKPOINT")
    os.environ["REFACTOR_GNN_CHECKPOINT"] = str(checkpoint)
    try:
        service = ShadowIntelligenceService(
            feature_dim=4,
            minimum_interval_seconds=2,
            comparison_path=tmp_path / "shadow.jsonl",
        )
    finally:
        if previous is None:
            os.environ.pop("REFACTOR_GNN_CHECKPOINT", None)
        else:
            os.environ["REFACTOR_GNN_CHECKPOINT"] = previous
    assert service.checkpoint_error == "GNN_HEAD_SCHEMA_MISMATCH"
    assert not service.checkpoint_loaded

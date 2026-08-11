"""The safety properties of the short-strategy deployment ladder.

Every test here pins a rule whose violation would put real money behind an
unvalidated short. They are grouped by the claim they defend:

1. direction arithmetic — a wrong sign inverts stop and target;
2. order semantics — SELL is ambiguous between "close a long" and "open a short";
3. deployment gating — a SHADOW arm must be structurally unable to trade;
4. transition legality — SHADOW -> LIVE_FULL must be unreachable by any path;
5. promotion / demotion asymmetry — demotion is faster than promotion;
6. leak defences — forward evaluation may not see the future;
7. regression — the existing LONG path is unchanged.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

import pytest

from app.strategy.catalog import SHORT_STRATEGY_IDS, STRATEGY_IDS, is_short_strategy
from app.strategy.exit_geometry import exit_geometry, reference_round_trip_cost_bps
from app.trading.borrow import (
    BorrowSnapshot,
    BorrowSnapshotStore,
    borrow_cost_bps,
    evaluate_borrow,
)
from app.trading.conservative_bandit import (
    ArmCandidate,
    BanditContext,
    ConservativeStrategyBandit,
)
from app.trading.directional import (
    ALLOWED_TRANSITIONS,
    DirectionalStrategyKey,
    ExecutionProduct,
    PositionDirection,
    PositionEffect,
    ShortReasonCodes,
    StrategyDeploymentState,
    broker_side,
    favourable_watermark,
    gross_return_bps,
    next_promotion_state,
    stop_breached,
    stop_price,
    target_price,
    target_reached,
    trailing_price,
    transition_allowed,
)
from app.trading.directional_shadow import (
    OUTCOME_STOP,
    OUTCOME_TARGET,
    OUTCOME_UNEXECUTABLE,
    QuoteObservation,
    ShadowFillSimulator,
    ShadowPlanStore,
    ShadowTradePlan,
)
from app.trading.short_strategy_promotion import (
    DemotionThresholds,
    DeploymentStateStore,
    DirectionalValidationSnapshot,
    RuntimeHealth,
    ShortStrategyPromotionConfig,
    ShortStrategyPromotionController,
    compute_confidence_components,
    compute_confidence_score,
    evaluate_demotion,
    evaluate_hard_gates,
    evaluate_immediate_suspension,
)
from app.trading.strategy_performance_store import (
    EVALUATION_SOURCE_LIVE,
    EVALUATION_SOURCE_SHADOW,
    StrategyPerformanceStore,
)
from app.trading.strategy_session import _ElectionProposal

NOW = datetime(2026, 8, 1, 5, 0, tzinfo=timezone.utc)
KEY = DirectionalStrategyKey.for_short("opening_range_breakdown", "KR")
LONG_KEY = DirectionalStrategyKey.for_long("opening_range_breakout", "KR")


def _proposal(
    *,
    direction: PositionDirection,
    deployment_state: StrategyDeploymentState,
    borrow_snapshot=None,
) -> _ElectionProposal:
    return _ElectionProposal(
        symbol="005930",
        strategy_id=(
            "opening_range_breakdown"
            if direction is PositionDirection.SHORT
            else "opening_range_breakout"
        ),
        source="test",
        entry_price=70_000.0,
        target_return_rate=0.018,
        stop_loss_rate=0.006,
        trailing_stop_rate=0.003,
        max_holding_seconds=3_600,
        score=0.0,
        confidence=0.0,
        expected_net_return_bps=None,
        expected_cost_bps=None,
        gnn_actionable=False,
        gnn_action="UNAVAILABLE",
        gnn_reason_codes=[],
        ontology_reason_codes=[],
        macro_regime="UNKNOWN",
        micro_regime="",
        explanation_paths=[],
        intent=None,
        candidate_count=None,
        micro_result=None,
        evidence_row=None,
        last_reason="test",
        direction=direction,
        deployment_state=deployment_state,
        borrow_snapshot=borrow_snapshot,
    )


def test_live_short_without_point_in_time_locate_cannot_submit_order() -> None:
    proposal = _proposal(
        direction=PositionDirection.SHORT,
        deployment_state=StrategyDeploymentState.LIVE_PROBE,
    )

    assert not proposal.submits_orders


def test_shadow_proposal_remains_non_ordering_with_or_without_locate() -> None:
    proposal = _proposal(
        direction=PositionDirection.SHORT,
        deployment_state=StrategyDeploymentState.SHADOW,
        borrow_snapshot=object(),
    )

    assert not proposal.submits_orders


# --------------------------------------------------------------------------- #
# 1. Direction arithmetic                                                      #
# --------------------------------------------------------------------------- #
def test_short_target_is_below_entry_and_stop_is_above() -> None:
    """Getting this pair backwards arms a position that exits on the winning side."""
    entry = 1000.0
    assert target_price(entry, 0.02, PositionDirection.SHORT) == pytest.approx(980.0)
    assert stop_price(entry, 0.006, PositionDirection.SHORT) == pytest.approx(1006.0)
    # The long side is the mirror, and unchanged.
    assert target_price(entry, 0.02, PositionDirection.LONG) == pytest.approx(1020.0)
    assert stop_price(entry, 0.006, PositionDirection.LONG) == pytest.approx(994.0)


def test_short_gross_return_is_positive_when_price_falls() -> None:
    """A short that covered lower made money. The sign lives in ONE function."""
    assert gross_return_bps(1000.0, 950.0, PositionDirection.SHORT) == pytest.approx(500.0)
    assert gross_return_bps(1000.0, 1050.0, PositionDirection.SHORT) == pytest.approx(-500.0)
    assert gross_return_bps(1000.0, 1050.0, PositionDirection.LONG) == pytest.approx(500.0)


def test_short_barrier_comparisons_are_mirrored() -> None:
    """``price <= stop`` is a stop-out for a long and a PROFIT for a short."""
    # Short entered at 1000, target 980, stop 1006.
    assert target_reached(979.0, 980.0, PositionDirection.SHORT)
    assert not target_reached(981.0, 980.0, PositionDirection.SHORT)
    assert stop_breached(1007.0, 1006.0, PositionDirection.SHORT)
    assert not stop_breached(1005.0, 1006.0, PositionDirection.SHORT)


def test_short_watermark_tracks_the_low_not_the_high() -> None:
    """A short's favourable extreme is its LOWEST price.

    Seeding a short's trailing stop from a high watermark would arm a stop that is
    already triggered.
    """
    mark = favourable_watermark(None, 1000.0, PositionDirection.SHORT)
    mark = favourable_watermark(mark, 980.0, PositionDirection.SHORT)
    mark = favourable_watermark(mark, 995.0, PositionDirection.SHORT)
    assert mark == pytest.approx(980.0)
    # Trailing sits ABOVE the low for a short.
    assert trailing_price(980.0, 0.003, PositionDirection.SHORT) == pytest.approx(982.94)


def test_short_geometry_targets_are_larger_than_long_counterparts() -> None:
    """A higher cost floor demands a LARGER target, not a tighter one.

    Cost is additive against both barriers, so shrinking the target while the cost
    rises compresses net reward:risk from both ends. The first draft of the geometry
    table got this backwards.
    """
    for short_id in SHORT_STRATEGY_IDS:
        geometry = exit_geometry(short_id)
        cost = reference_round_trip_cost_bps(short_id)
        # Short cost reference must exceed the long one (it includes borrow).
        assert cost > reference_round_trip_cost_bps("opening_range_breakout")
        # And the net reward:risk must still clear the table's target at THAT cost.
        assert geometry.net_reward_risk_ratio(cost) >= 1.45, short_id


def test_no_short_thesis_outlives_its_long_counterpart() -> None:
    """Holding time is a cost AND a hazard for a short (accrual + recall)."""
    from app.strategy.catalog import SHORT_LONG_COUNTERPART

    for short_id, long_id in SHORT_LONG_COUNTERPART.items():
        short_holding = exit_geometry(short_id).max_holding_seconds
        long_holding = exit_geometry(long_id).max_holding_seconds
        assert short_holding <= long_holding, (short_id, long_id)


# --------------------------------------------------------------------------- #
# 2. Order semantics                                                           #
# --------------------------------------------------------------------------- #
def test_open_short_and_close_long_are_different_orders() -> None:
    """Both are broker-side SELL. Conflating them is the core ambiguity."""
    assert broker_side(PositionDirection.LONG, PositionEffect.CLOSE) == "SELL"
    assert broker_side(PositionDirection.SHORT, PositionEffect.OPEN) == "SELL"
    # ...and both covers are BUY.
    assert broker_side(PositionDirection.LONG, PositionEffect.OPEN) == "BUY"
    assert broker_side(PositionDirection.SHORT, PositionEffect.CLOSE) == "BUY"


def test_trade_plan_rejects_a_side_that_contradicts_its_direction() -> None:
    from app.trading.contracts import TradePlan

    def _plan(**overrides):
        payload = dict(
            strategy_id="opening_range_breakdown",
            strategy_instance_id="i-1",
            symbol="005930",
            side="SELL",
            thesis="t",
            entry_trigger={},
            entry_price_policy={},
            proposed_quantity=1,
            initial_stop={},
            profit_policy={},
            trailing_policy={},
            max_holding_seconds=600,
            invalidation_conditions=(),
            max_entry_slippage_bps=5.0,
            expires_at=NOW + timedelta(seconds=5),
            feature_snapshot_id="f-1",
            utility_evidence_id="u-1",
            position_direction="SHORT",
            position_effect="OPEN",
        )
        payload.update(overrides)
        return TradePlan(**payload)

    _plan()  # consistent: SHORT/OPEN == SELL
    with pytest.raises(ValueError, match="contradicts"):
        _plan(side="BUY")  # would open a LONG against a short plan


def test_final_order_infers_effect_rather_than_defaulting_to_open() -> None:
    """A long SELL exit must not be relabelled as an entry.

    A hard ``position_effect="OPEN"`` default made every legacy exit contradict its own
    broker side, which the contract-consistency check then rejected — blocking every
    de-risking order in the system.
    """
    from app.schemas.domain import FinalOrder, OrderSide, OrderType

    long_exit = FinalOrder(
        ticker="005930",
        market="KR",
        order_type=OrderType.LIMIT,
        side=OrderSide.SELL,
        quantity=1,
        limit_price=70000,
    )
    assert long_exit.resolved_position_effect == "CLOSE"
    short_entry = FinalOrder(
        ticker="005930",
        market="KR",
        order_type=OrderType.LIMIT,
        side=OrderSide.SELL,
        quantity=1,
        limit_price=70000,
        position_direction="SHORT",
    )
    assert short_entry.resolved_position_effect == "OPEN"


def test_credit_contract_refuses_every_mismatch() -> None:
    from app.execution.kis_real import _require_credit_contract
    from app.schemas.domain import FinalOrder, OrderSide, OrderType

    def _order(**overrides):
        payload = dict(
            ticker="005930",
            market="KR",
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            quantity=1,
            limit_price=70000,
            position_direction="SHORT",
            position_effect="OPEN",
            execution_product="CREDIT_BORROW",
            credit_type="22",
        )
        payload.update(overrides)
        return FinalOrder(**payload)

    # The valid contract passes.
    _require_credit_contract(
        _order(), direction="SHORT", effect="OPEN", side=OrderSide.SELL
    )
    # A cash product cannot deliver a borrow.
    with pytest.raises(ValueError, match="CREDIT_BORROW"):
        _require_credit_contract(
            _order(execution_product="CASH"),
            direction="SHORT",
            effect="OPEN",
            side=OrderSide.SELL,
        )
    # A cover carrying SELL would sell stock the account does not hold.
    with pytest.raises(ValueError, match="broker side"):
        _require_credit_contract(
            _order(position_effect="CLOSE"),
            direction="SHORT",
            effect="CLOSE",
            side=OrderSide.BUY,
        )
    with pytest.raises(ValueError, match="CRDT_TYPE 22"):
        _require_credit_contract(
            _order(credit_type="05"),
            direction="SHORT",
            effect="OPEN",
            side=OrderSide.SELL,
        )


# --------------------------------------------------------------------------- #
# 3. Deployment gating                                                         #
# --------------------------------------------------------------------------- #
def _store(tmp_path, name="perf.sqlite3") -> StrategyPerformanceStore:
    return StrategyPerformanceStore(tmp_path / name)


def test_long_and_short_posteriors_never_pool(tmp_path) -> None:
    """Pooling would make a +60/-60 pair read as break-even in BOTH directions."""
    store = _store(tmp_path)
    for _ in range(20):
        store.record_directional(LONG_KEY, symbol="005930", realized_net_bps=60.0)
    for _ in range(20):
        store.record_directional(
            KEY,
            symbol="005930",
            realized_net_bps=-60.0,
            evaluation_source=EVALUATION_SOURCE_SHADOW,
        )
    assert store.posterior_for_key(LONG_KEY).observed_mean_net_bps == pytest.approx(60.0)
    assert store.posterior_for_key(KEY).observed_mean_net_bps == pytest.approx(-60.0)


def test_short_arm_with_no_history_falls_back_to_prior_not_to_the_long_arm(tmp_path) -> None:
    """Borrowing the long side's evidence would gift an unearned positive bound."""
    store = _store(tmp_path)
    for _ in range(40):
        store.record_directional(
            DirectionalStrategyKey.for_long("opening_range_breakdown", "KR"),
            symbol="005930",
            realized_net_bps=200.0,
        )
    posterior = store.posterior_for_key(KEY)
    assert posterior.sample_count == 0
    # Prior-centred (0.0), NOT the long arm's +200.
    assert posterior.posterior_mean_net_bps == pytest.approx(0.0)
    assert posterior.conservative_edge_bps < 0.0


def test_unexecutable_signals_are_excluded_from_promotion_statistics(tmp_path) -> None:
    """A strategy may not be promoted on trades it could not have taken."""
    store = _store(tmp_path)
    for _ in range(5):
        store.record_directional(
            KEY,
            symbol="005930",
            realized_net_bps=500.0,
            evaluation_source=EVALUATION_SOURCE_SHADOW,
            borrow_available=False,
            signal_executable=False,
        )
    metrics = store.directional_metrics(
        KEY, evaluation_sources=(EVALUATION_SOURCE_SHADOW,)
    )
    assert metrics["filled_trade_count"] == 0
    # ...but they DO count toward the borrow-availability measurement.
    assert metrics["executable_signal_count"] == 5
    assert metrics["borrow_availability_rate"] == pytest.approx(0.0)


def test_shadow_short_is_not_selectable_however_large_its_edge(tmp_path) -> None:
    """The whole point of SHADOW: evaluated and reported, structurally untradable."""
    store = _store(tmp_path)
    for _ in range(40):
        store.record_directional(
            KEY,
            symbol="005930",
            realized_net_bps=300.0,
            evaluation_source=EVALUATION_SOURCE_SHADOW,
        )
    bandit = ConservativeStrategyBandit(store=store)
    candidate = ArmCandidate(
        arm="opening_range_breakdown",
        symbol="005930",
        predicted_net_edge_bps=250.0,
        direction=PositionDirection.SHORT,
        execution_product=ExecutionProduct.CREDIT_BORROW,
        deployment_state=StrategyDeploymentState.SHADOW,
        borrow_available=True,
    )
    selection = bandit.select(
        (candidate,), BanditContext(market="KR", macro_regime="TREND_DOWN"), now=NOW
    )
    assert selection.is_no_trade
    assert "opening_range_breakdown:SHORT" in selection.shadow_arms
    # The edge is still MEASURED and reported — that is the promotion evidence.
    assert selection.best_short_edge_bps is not None
    assert selection.best_short_edge_bps > 0
    assert selection.short_rescued


def test_promoted_short_becomes_selectable(tmp_path) -> None:
    """The same arm, same evidence, one rung up: now it trades."""
    store = _store(tmp_path)
    for _ in range(40):
        store.record_directional(
            KEY,
            symbol="005930",
            realized_net_bps=300.0,
            evaluation_source=EVALUATION_SOURCE_SHADOW,
        )
    bandit = ConservativeStrategyBandit(store=store)
    selection = bandit.select(
        (
            ArmCandidate(
                arm="opening_range_breakdown",
                symbol="005930",
                predicted_net_edge_bps=250.0,
                direction=PositionDirection.SHORT,
                execution_product=ExecutionProduct.CREDIT_BORROW,
                deployment_state=StrategyDeploymentState.LIVE_PROBE,
                borrow_available=True,
            ),
        ),
        BanditContext(market="KR", macro_regime="TREND_DOWN"),
        now=NOW,
    )
    assert not selection.is_no_trade
    assert selection.selected_direction == "SHORT"


def test_authorized_short_without_a_locate_is_still_refused(tmp_path) -> None:
    """No borrow, no trade — regardless of deployment state or edge."""
    store = _store(tmp_path)
    for _ in range(40):
        store.record_directional(
            KEY,
            symbol="005930",
            realized_net_bps=300.0,
            evaluation_source=EVALUATION_SOURCE_SHADOW,
        )
    bandit = ConservativeStrategyBandit(store=store)
    for borrow in (None, False):
        selection = bandit.select(
            (
                ArmCandidate(
                    arm="opening_range_breakdown",
                    symbol="005930",
                    predicted_net_edge_bps=250.0,
                    direction=PositionDirection.SHORT,
                    execution_product=ExecutionProduct.CREDIT_BORROW,
                    deployment_state=StrategyDeploymentState.LIVE_FULL,
                    borrow_available=borrow,
                ),
            ),
            BanditContext(market="KR", macro_regime="TREND_DOWN"),
            now=NOW,
        )
        assert selection.is_no_trade, borrow


def test_arm_candidate_default_state_is_direction_dependent() -> None:
    """LONG defaults open (its gate is ``live_authorized``); SHORT defaults closed.

    A flat SHADOW default would have made every existing LONG caller unselectable; a
    flat LIVE_FULL default would have made an unevaluated SHORT tradable.
    """
    assert (
        ArmCandidate(arm="intraday_momentum").deployment_state
        is StrategyDeploymentState.LIVE_FULL
    )
    assert (
        ArmCandidate(
            arm="opening_range_breakdown", direction=PositionDirection.SHORT
        ).deployment_state
        is StrategyDeploymentState.SHADOW
    )


# --------------------------------------------------------------------------- #
# 4. Transition legality                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "current,target",
    [
        ("DISABLED", "LIVE_PROBE"),
        ("DISABLED", "LIVE_LIMITED"),
        ("DISABLED", "LIVE_FULL"),
        ("SHADOW", "LIVE_LIMITED"),
        ("SHADOW", "LIVE_FULL"),
        ("LIVE_PROBE", "LIVE_FULL"),
        ("SUSPENDED", "LIVE_PROBE"),
        ("SUSPENDED", "LIVE_LIMITED"),
        ("SUSPENDED", "LIVE_FULL"),
    ],
)
def test_forbidden_transitions_are_unreachable(current: str, target: str) -> None:
    """No config value, env var or operator action may make these reachable."""
    assert not transition_allowed(
        StrategyDeploymentState(current), StrategyDeploymentState(target)
    )


def test_every_promotion_is_exactly_one_rung() -> None:
    for state in StrategyDeploymentState:
        target = next_promotion_state(state)
        if target is None:
            continue
        assert transition_allowed(state, target)
        assert target.rank == state.rank + 1, state


def test_suspended_recovers_only_to_shadow() -> None:
    reachable = {
        target
        for source, target in ALLOWED_TRANSITIONS
        if source is StrategyDeploymentState.SUSPENDED
    }
    assert reachable == {StrategyDeploymentState.SHADOW}


def test_config_cannot_seed_an_arm_into_a_live_state(tmp_path) -> None:
    """``initial_state`` is clamped. Editing YAML must not skip the ladder."""
    config = ShortStrategyPromotionConfig(
        strategies={"opening_range_breakdown": {"enabled": True, "initial_state": "LIVE_FULL"}}
    )
    assert config.initial_state_for("opening_range_breakdown") is StrategyDeploymentState.SHADOW


def test_store_refuses_to_persist_a_forbidden_transition(tmp_path) -> None:
    """Defence in depth: the whitelist is enforced at the persistence boundary too.

    So a future caller that hand-builds a decision still cannot write
    SHADOW -> LIVE_FULL.
    """
    store = DeploymentStateStore(tmp_path / "dep.sqlite3")
    store.ensure(KEY, StrategyDeploymentState.SHADOW)
    assert not store.force_state(
        KEY, StrategyDeploymentState.LIVE_FULL, actor="test", reason="attempt"
    )
    assert store.state_of(KEY) is StrategyDeploymentState.SHADOW


def test_unknown_arm_defaults_to_shadow(tmp_path) -> None:
    """Never evaluated must read as not authorised."""
    store = DeploymentStateStore(tmp_path / "dep.sqlite3")
    unknown = DirectionalStrategyKey.for_short("residual_relative_weakness", "US")
    assert store.state_of(unknown) is StrategyDeploymentState.SHADOW
    assert store.get(unknown) is None


# --------------------------------------------------------------------------- #
# 5. Promotion / demotion asymmetry                                            #
# --------------------------------------------------------------------------- #
def _passing_snapshot(**overrides) -> DirectionalValidationSnapshot:
    payload = dict(
        key=KEY,
        evaluated_at=NOW,
        state=StrategyDeploymentState.SHADOW,
        executable_signal_count=200,
        filled_trade_count=90,
        distinct_trading_days=25,
        distinct_symbols=14,
        distinct_regimes=3,
        mean_net_return_bps=22.0,
        median_net_return_bps=18.0,
        win_rate=0.55,
        profit_factor=1.4,
        conservative_edge_bps=12.0,
        lower_confidence_bound_net_bps=6.0,
        cost_coverage_ratio=2.1,
        maximum_drawdown_bps=180.0,
        expected_shortfall_95_bps=95.0,
        loss_streak=2,
        prediction_calibration_error=0.05,
        mean_slippage_error_bps=7.0,
        borrow_availability_rate=0.85,
        data_freshness_pass_rate=0.995,
        strategy_regime_stability=0.8,
        change_point_probability=0.1,
        short_rescue_rate=0.09,
        holdout_windows_passed=3,
        holdout_windows_evaluated=3,
        model_calibrated=True,
    )
    payload.update(overrides)
    snapshot = DirectionalValidationSnapshot(**payload)
    thresholds = ShortStrategyPromotionConfig().shadow_to_live_probe
    components = compute_confidence_components(snapshot, thresholds)
    return dataclasses.replace(
        snapshot,
        confidence_score=compute_confidence_score(components),
        confidence_components=components,
    )


def _controller(tmp_path) -> ShortStrategyPromotionController:
    return ShortStrategyPromotionController(
        config=ShortStrategyPromotionConfig(
            strategies={
                strategy_id: {"enabled": True, "initial_state": "SHADOW"}
                for strategy_id in SHORT_STRATEGY_IDS
            }
        ),
        state_store=DeploymentStateStore(tmp_path / "dep.sqlite3"),
        performance_store=StrategyPerformanceStore(tmp_path / "perf.sqlite3"),
        shadow_store=ShadowPlanStore(tmp_path / "shadow.sqlite3"),
        borrow_store=BorrowSnapshotStore(tmp_path / "borrow.sqlite3"),
    )


def test_promotion_requires_the_full_consecutive_cycle_count(tmp_path) -> None:
    controller = _controller(tmp_path)
    snapshot = _passing_snapshot()
    required = controller.config.shadow_to_live_probe.required_consecutive_evaluation_cycles
    record = controller.state_store.ensure(KEY, StrategyDeploymentState.SHADOW)
    for cycle in range(1, required):
        decision = controller.decide(snapshot, record)
        controller.state_store.apply(decision)
        record = controller.state_store.get(KEY)
        assert record.state is StrategyDeploymentState.SHADOW, cycle
        assert record.consecutive_passes == cycle
    decision = controller.decide(snapshot, record)
    controller.state_store.apply(decision)
    assert controller.state_store.state_of(KEY) is StrategyDeploymentState.LIVE_PROBE


def test_one_failing_cycle_resets_the_streak_it_does_not_decrement_it(tmp_path) -> None:
    """"5 passes, 1 fail, 1 pass" is not 5 consecutive passes.

    Decrementing would let an arm that passes 60% of cycles eventually accumulate its
    way to live.
    """
    controller = _controller(tmp_path)
    record = controller.state_store.ensure(KEY, StrategyDeploymentState.SHADOW)
    for _ in range(4):
        controller.state_store.apply(controller.decide(_passing_snapshot(), record))
        record = controller.state_store.get(KEY)
    assert record.consecutive_passes == 4
    failing = _passing_snapshot(filled_trade_count=1)
    controller.state_store.apply(controller.decide(failing, record))
    record = controller.state_store.get(KEY)
    assert record.consecutive_passes == 0
    assert record.state is StrategyDeploymentState.SHADOW


def test_any_single_hard_gate_failure_blocks_promotion() -> None:
    """A high confidence score never overrides a failing gate.

    ``confidence_score`` is a weighted blend, so it can sit at 0.85 while one
    component is catastrophic. A blend is right for ranking and wrong for permission.
    """
    thresholds = ShortStrategyPromotionConfig().shadow_to_live_probe
    assert evaluate_hard_gates(_passing_snapshot(), thresholds) == ()
    for override, expected in [
        ({"filled_trade_count": 1}, ShortReasonCodes.PROMOTION_SAMPLE_INSUFFICIENT),
        ({"distinct_symbols": 1}, "SHORT_PROMOTION_SYMBOL_BREADTH_INSUFFICIENT"),
        ({"conservative_edge_bps": 0.0}, ShortReasonCodes.CONSERVATIVE_EDGE_NON_POSITIVE),
        ({"cost_coverage_ratio": 1.0}, ShortReasonCodes.COST_COVERAGE_INSUFFICIENT),
        ({"borrow_availability_rate": 0.1}, ShortReasonCodes.BORROW_AVAILABILITY_RATE_LOW),
        ({"short_rescue_rate": 0.0}, ShortReasonCodes.RESCUE_RATE_INSUFFICIENT),
        ({"holdout_windows_passed": 0}, ShortReasonCodes.HOLDOUT_NOT_PASSED),
        ({"model_calibrated": False}, ShortReasonCodes.MODEL_NOT_CALIBRATED),
        ({"maximum_drawdown_bps": 9_999.0}, ShortReasonCodes.DRAWDOWN_EXCEEDED),
    ]:
        failures = evaluate_hard_gates(_passing_snapshot(**override), thresholds)
        assert expected in failures, override


def test_unmeasurable_profit_factor_fails_rather_than_passes() -> None:
    """An all-winners sample has an undefined profit factor, not an infinite one."""
    thresholds = ShortStrategyPromotionConfig().shadow_to_live_probe
    failures = evaluate_hard_gates(_passing_snapshot(profit_factor=None), thresholds)
    assert "SHORT_PROFIT_FACTOR_INSUFFICIENT" in failures


def test_unmeasured_calibration_and_borrow_score_zero_not_neutral() -> None:
    """Absence of evidence must not carry half the weight of evidence."""
    thresholds = ShortStrategyPromotionConfig().shadow_to_live_probe
    components = compute_confidence_components(
        _passing_snapshot(
            prediction_calibration_error=None,
            mean_slippage_error_bps=None,
            borrow_availability_rate=None,
        ),
        thresholds,
    )
    assert components["calibration_quality"] == 0.0
    assert components["execution_quality"] == 0.0
    assert components["borrow_reliability"] == 0.0


def test_demotion_acts_on_the_first_failing_cycle(tmp_path) -> None:
    """Demotion must be faster than promotion.

    A slow promotion costs a missed trade; a slow demotion costs a compounding loss on
    an unbounded downside.
    """
    controller = _controller(tmp_path)
    store = controller.state_store
    store.ensure(KEY, StrategyDeploymentState.SHADOW)
    store.force_state(KEY, StrategyDeploymentState.LIVE_PROBE, actor="t", reason="setup")
    record = store.get(KEY)
    degraded = _passing_snapshot(
        state=StrategyDeploymentState.LIVE_PROBE,
        conservative_edge_bps=-5.0,
    )
    degraded = dataclasses.replace(degraded, confidence_score=0.40)
    decision = controller.decide(degraded, record)
    assert decision.demoted
    assert decision.to_state is StrategyDeploymentState.SHADOW


@pytest.mark.parametrize(
    "flag,expected",
    [
        ("loan_date_missing", ShortReasonCodes.LOAN_DATE_MISSING),
        ("position_direction_mismatch", ShortReasonCodes.POSITION_DIRECTION_MISMATCH),
        ("credit_contract_failure", ShortReasonCodes.ORDER_CONTRACT_INCOMPLETE),
        ("data_quality_hard_fail", ShortReasonCodes.DATA_QUALITY_FAILED),
        ("regime_dislocated", ShortReasonCodes.HIGH_VOL_DISLOCATED),
        ("daily_loss_limit_breached", ShortReasonCodes.DAILY_LOSS_LIMIT),
        ("stop_order_submission_failed", ShortReasonCodes.STOP_ORDER_CAPABILITY_MISSING),
        ("duplicate_short_order", "SHORT_DUPLICATE_ORDER_DETECTED"),
    ],
)
def test_immediate_suspension_has_no_grace_period(flag: str, expected: str) -> None:
    """Each of these says internal state and broker state disagree.

    Averaging that over several cycles means trading on a position we cannot describe.
    """
    config = ShortStrategyPromotionConfig()
    assert evaluate_immediate_suspension(_passing_snapshot(), config) == ()
    failures = evaluate_immediate_suspension(_passing_snapshot(**{flag: True}), config)
    assert expected in failures


def test_broker_state_not_restored_suspends(tmp_path) -> None:
    """Restart without restored borrow state blocks new entries."""
    config = ShortStrategyPromotionConfig()
    failures = evaluate_immediate_suspension(
        _passing_snapshot(broker_state_restored=False), config
    )
    assert ShortReasonCodes.BROKER_STATE_UNRESTORED in failures


def test_change_point_probability_suspends_at_the_threshold() -> None:
    config = ShortStrategyPromotionConfig()
    assert (
        ShortReasonCodes.REGIME_UNSTABLE
        in evaluate_immediate_suspension(
            _passing_snapshot(change_point_probability=0.7), config
        )
    )
    assert (
        ShortReasonCodes.REGIME_UNSTABLE
        not in evaluate_immediate_suspension(
            _passing_snapshot(change_point_probability=0.5), config
        )
    )


def test_demotion_thresholds_are_looser_than_promotion_thresholds() -> None:
    """The hysteresis band. Without it an arm on its boundary oscillates every cycle,
    and each oscillation is a real change in live position limits."""
    config = ShortStrategyPromotionConfig()
    demotion = config.demotion
    assert demotion.live_probe_confidence < config.shadow_to_live_probe.minimum_confidence_score
    assert (
        demotion.live_limited_confidence
        < config.live_probe_to_live_limited.minimum_confidence_score
    )
    assert demotion.live_full_confidence < config.live_limited_to_live_full.minimum_confidence_score


def test_a_new_short_strategy_starts_in_shadow_and_submits_no_orders(tmp_path) -> None:
    """The headline acceptance criterion."""
    controller = _controller(tmp_path)
    for key in controller.managed_keys():
        decision = controller.evaluate(key, health=RuntimeHealth())
        assert decision.to_state is StrategyDeploymentState.SHADOW
        assert not decision.to_state.submits_orders
        authorized, reasons = controller.may_submit_orders(key)
        assert not authorized
        assert ShortReasonCodes.SHADOW_ONLY in reasons


def test_yaml_overlay_tunes_one_threshold_without_zeroing_the_others() -> None:
    """A partial YAML block must not silently disable every gate it omits."""
    from app.trading.short_strategy_promotion import (
        DEFAULT_SHADOW_TO_LIVE_PROBE,
        _merge_thresholds,
    )

    merged = _merge_thresholds(
        DEFAULT_SHADOW_TO_LIVE_PROBE, {"minimum_filled_trades": 5}
    )
    assert merged.minimum_filled_trades == 5
    assert (
        merged.minimum_confidence_score
        == DEFAULT_SHADOW_TO_LIVE_PROBE.minimum_confidence_score
    )
    assert (
        merged.required_consecutive_evaluation_cycles
        == DEFAULT_SHADOW_TO_LIVE_PROBE.required_consecutive_evaluation_cycles
    )


def test_shipped_config_operator_override_authorizes_every_short_arm(tmp_path) -> None:
    config = ShortStrategyPromotionConfig.load("config/short_strategy_deployment.yaml")
    assert config.operator_live_full_override is True
    controller = ShortStrategyPromotionController(
        config=config,
        state_store=DeploymentStateStore(tmp_path / "dep.sqlite3"),
        performance_store=StrategyPerformanceStore(tmp_path / "perf.sqlite3"),
        shadow_store=ShadowPlanStore(tmp_path / "shadow.sqlite3"),
        borrow_store=BorrowSnapshotStore(tmp_path / "borrow.sqlite3"),
    )
    for strategy_id in SHORT_STRATEGY_IDS:
        assert config.strategy_enabled(strategy_id), strategy_id
        key = DirectionalStrategyKey.for_short(strategy_id, "US")
        assert controller.authorized_state(key) is StrategyDeploymentState.LIVE_FULL
        assert controller.may_submit_orders(key)[0] is True


# --------------------------------------------------------------------------- #
# 6. Leak defences                                                             #
# --------------------------------------------------------------------------- #
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
        borrow_snapshot=BorrowSnapshot(
            symbol="005930",
            observed_at=NOW - timedelta(seconds=30),
            available=True,
            available_quantity=500,
            borrow_fee_bps_annualised=800.0,
        ),
        intended_quantity=10,
    )
    payload.update(overrides)
    return ShadowTradePlan(**payload)


def test_simulator_refuses_quotes_at_or_before_the_signal() -> None:
    """The barrier walk must not see the bar that produced the signal."""
    simulator = ShadowFillSimulator()
    assert simulator.submit(_plan()) is None
    # Same instant, and earlier: both ignored.
    assert simulator.observe(QuoteObservation(NOW, 900.0, 901.0)) == ()
    assert (
        simulator.observe(QuoteObservation(NOW - timedelta(seconds=5), 900.0, 901.0)) == ()
    )
    assert simulator.open_plan_count == 1


def test_shadow_store_deduplicates_correlated_recent_signals(tmp_path) -> None:
    store = ShadowPlanStore(tmp_path / "shadow.sqlite3")
    plan = _plan()

    assert store.record_plan(plan)
    assert store.has_recent_plan(KEY, "005930", since=NOW - timedelta(minutes=5))
    assert not store.has_recent_plan(KEY, "005930", since=NOW + timedelta(seconds=1))
    assert not store.has_recent_plan(KEY, "000660", since=NOW - timedelta(minutes=5))


def test_simulator_exposes_symbols_that_still_require_quotes() -> None:
    simulator = ShadowFillSimulator()
    simulator.submit(_plan(plan_id="a", symbol="AAA"))
    simulator.submit(_plan(plan_id="b", symbol="BBB"))
    simulator.submit(_plan(plan_id="c", symbol="AAA"))

    assert simulator.open_symbols == ("AAA", "BBB")
    assert simulator.open_plan_count == 3


def test_simulator_does_not_reuse_the_entry_quote_as_an_exit() -> None:
    simulator = ShadowFillSimulator(entry_slippage_bps=0.0, exit_slippage_bps=0.0)
    simulator.submit(_plan())
    entry_quote = QuoteObservation(NOW + timedelta(seconds=1), 1000.0, 1010.0)

    assert simulator.observe(entry_quote) == ()
    assert simulator.observe(entry_quote) == ()
    assert simulator.open_plan_count == 1

    outcome = simulator.observe(
        QuoteObservation(NOW + timedelta(seconds=2), 1000.0, 1010.0)
    )
    assert outcome[0].outcome == OUTCOME_STOP
    assert outcome[0].holding_seconds == 1.0


def test_short_fills_at_the_bid_and_covers_at_the_ask() -> None:
    """Mid-price fills award half the spread on both legs — 20bps of fiction per
    round trip on a KRX book, against a ~180bps target."""
    simulator = ShadowFillSimulator(entry_slippage_bps=0.0, exit_slippage_bps=0.0)
    simulator.submit(_plan())
    simulator.observe(QuoteObservation(NOW + timedelta(seconds=1), 999.0, 1001.0))
    outcomes = simulator.observe(
        QuoteObservation(NOW + timedelta(seconds=600), 979.0, 981.0)
    )
    assert len(outcomes) == 1
    outcome = outcomes[0]
    # Entered at the BID (999), covered at the ASK (981) — the unfavourable side both
    # times, which is what the spread costs.
    assert outcome.entry_price == pytest.approx(999.0)
    assert outcome.exit_price == pytest.approx(981.0)
    assert outcome.outcome == OUTCOME_TARGET
    assert outcome.gross_return_bps > 0


def test_a_quote_straddling_both_barriers_resolves_to_the_stop() -> None:
    """Within one observation there is no way to know which came first, and assuming
    the target would systematically flatter every volatile trade."""
    simulator = ShadowFillSimulator(entry_slippage_bps=0.0, exit_slippage_bps=0.0)
    simulator.submit(_plan())
    simulator.observe(QuoteObservation(NOW + timedelta(seconds=1), 999.0, 1001.0))
    # Ask 1010 is past the 1006 stop; bid 975 is past the 982 target.
    outcomes = simulator.observe(
        QuoteObservation(NOW + timedelta(seconds=60), 975.0, 1010.0)
    )
    assert outcomes[0].outcome == OUTCOME_STOP


def test_signal_without_a_locate_is_recorded_but_never_scored() -> None:
    simulator = ShadowFillSimulator()
    outcome = simulator.submit(_plan(borrow_snapshot=None))
    assert outcome is not None
    assert outcome.outcome == OUTCOME_UNEXECUTABLE
    assert not outcome.executable
    assert not outcome.scored
    assert simulator.open_plan_count == 0


def test_a_borrow_snapshot_from_the_future_is_refused_not_clamped() -> None:
    """Future information relative to the decision is a leak, not freshness."""
    future = BorrowSnapshot(
        symbol="005930",
        observed_at=NOW + timedelta(seconds=60),
        available=True,
        available_quantity=500,
        borrow_fee_bps_annualised=800.0,
    )
    assert not future.is_fresh(NOW)
    executable, reasons = _plan(borrow_snapshot=future).executable()
    assert not executable
    assert ShortReasonCodes.BORROW_SNAPSHOT_STALE in reasons


def test_unpriced_borrow_is_charged_at_the_ceiling_not_at_zero() -> None:
    """Pricing an unpriced borrow at zero is how a losing short passes a cost gate."""
    simulator = ShadowFillSimulator(entry_slippage_bps=0.0, exit_slippage_bps=0.0)
    unpriced = BorrowSnapshot(
        symbol="005930",
        observed_at=NOW - timedelta(seconds=30),
        available=True,
        available_quantity=500,
        borrow_fee_bps_annualised=None,
    )
    # The plan is unexecutable on an unpriced borrow, which is the primary defence.
    executable, _ = _plan(borrow_snapshot=unpriced).executable()
    assert not executable


def test_holdout_windows_are_disjoint_and_silence_does_not_pass(tmp_path) -> None:
    """A window with no data does not pass. Silence is not evidence."""
    controller = _controller(tmp_path)
    passed, evaluated = controller._holdout_result(
        KEY, NOW, controller.config.shadow_to_live_probe
    )
    assert evaluated == controller.config.shadow_to_live_probe.required_holdout_windows_passed
    assert passed == 0


def test_borrow_fee_prorating_is_annualised() -> None:
    """The unit conversion everything short-side depends on."""
    # 8%/yr held one full day is ~2.19bps, not 800bps and not 0.
    assert borrow_cost_bps(800.0, 86_400) == pytest.approx(2.1918, rel=1e-3)
    assert borrow_cost_bps(800.0, 1_800) == pytest.approx(0.0457, rel=1e-2)
    # Unknown stays unknown.
    assert borrow_cost_bps(None, 1_800) is None


def test_borrow_store_answers_point_in_time_not_latest(tmp_path) -> None:
    """A signal from 14:03 must see the borrow world as of 14:03.

    Handing it today's locate would let it short names that were unborrowable at the
    time — which is exactly the population these strategies target.
    """
    store = BorrowSnapshotStore(tmp_path / "borrow.sqlite3")
    store.record(
        BorrowSnapshot(
            symbol="005930",
            observed_at=NOW - timedelta(hours=2),
            available=False,
            available_quantity=0,
        )
    )
    store.record(
        BorrowSnapshot(
            symbol="005930",
            observed_at=NOW,
            available=True,
            available_quantity=500,
            borrow_fee_bps_annualised=800.0,
        )
    )
    past = store.latest("005930", as_of=NOW - timedelta(hours=1))
    assert past is not None and not past.available
    now = store.latest("005930", as_of=NOW)
    assert now is not None and now.available


def test_missing_borrow_observation_is_not_availability(tmp_path) -> None:
    store = BorrowSnapshotStore(tmp_path / "borrow.sqlite3")
    assert store.latest("999999") is None
    verdict = evaluate_borrow(None, quantity=1, now=NOW)
    assert not verdict.allowed
    assert ShortReasonCodes.BORROW_LOOKUP_FAILED in verdict.reason_codes


def test_borrow_health_reports_null_rates_when_nothing_was_asked(tmp_path) -> None:
    """"We asked nothing" must not read as "nothing was available" — that would demote
    a strategy for an outage in the polling loop."""
    store = BorrowSnapshotStore(tmp_path / "borrow.sqlite3")
    health = store.health()
    assert health["lookup_count"] == 0
    assert health["availability_rate"] is None
    assert health["rejection_rate"] is None


# --------------------------------------------------------------------------- #
# 7. Regression: the LONG path is unchanged                                    #
# --------------------------------------------------------------------------- #
def test_short_strategies_are_appended_never_inserted() -> None:
    """Model output indices and persisted masks depend on this order."""
    expected_long_prefix = (
        "intraday_momentum",
        "breakout_volume",
        "vwap_mean_reversion",
        "liquidity_shock_reversal",
        "event_momentum",
        "cross_sectional_relative_strength",
        "gap_context",
        "rvgi_box_breakout",
        "residual_relative_strength",
        "adaptive_anchored_vwap_reversion",
        "ofi_microprice_exhaustion_reversal",
        "opening_range_breakout",
        "market_intraday_momentum",
    )
    assert STRATEGY_IDS[: len(expected_long_prefix)] == expected_long_prefix
    short_start = len(expected_long_prefix)
    short_end = short_start + len(SHORT_STRATEGY_IDS)
    assert STRATEGY_IDS[short_start:short_end] == SHORT_STRATEGY_IDS
    # The tail is append-only, so it is pinned by PREFIX rather than by equality:
    # a later thesis must be able to land after these without rewriting a literal,
    # while still proving nothing was inserted before or among the shorts.
    later_appends = ("bar_confirmed_vwap_recovery", "overnight_gap_carry")
    assert STRATEGY_IDS[short_end : short_end + len(later_appends)] == later_appends
    assert len(STRATEGY_IDS) == short_end + len(later_appends)


def test_every_short_strategy_is_flagged_as_short() -> None:
    for strategy_id in STRATEGY_IDS:
        assert is_short_strategy(strategy_id) == (strategy_id in SHORT_STRATEGY_IDS)


def test_no_trade_arm_survives(tmp_path) -> None:
    """NO_TRADE remains a real, selectable outcome."""
    store = _store(tmp_path)
    bandit = ConservativeStrategyBandit(store=store)
    selection = bandit.select((), BanditContext(market="KR"), now=NOW)
    assert selection.is_no_trade


def test_both_directions_negative_is_distinguished_from_no_coverage(tmp_path) -> None:
    """"We looked both ways and neither paid" is a finding; "we only had one
    direction" is a coverage gap."""
    store = _store(tmp_path)
    for _ in range(30):
        store.record_directional(LONG_KEY, symbol="005930", realized_net_bps=-80.0)
        store.record_directional(
            KEY,
            symbol="005930",
            realized_net_bps=-80.0,
            evaluation_source=EVALUATION_SOURCE_SHADOW,
        )
    bandit = ConservativeStrategyBandit(store=store)
    selection = bandit.select(
        (
            ArmCandidate(
                arm="opening_range_breakout",
                symbol="005930",
                direction=PositionDirection.LONG,
                deployment_state=StrategyDeploymentState.LIVE_FULL,
            ),
            ArmCandidate(
                arm="opening_range_breakdown",
                symbol="005930",
                direction=PositionDirection.SHORT,
                execution_product=ExecutionProduct.CREDIT_BORROW,
                deployment_state=StrategyDeploymentState.LIVE_PROBE,
                borrow_available=True,
            ),
        ),
        BanditContext(market="KR", macro_regime="RANGE"),
        now=NOW,
    )
    assert selection.is_no_trade
    assert selection.both_directions_negative
    assert ShortReasonCodes.BOTH_DIRECTIONS_NEGATIVE in selection.reason_codes


def test_long_profitability_decision_is_unchanged_by_short_support() -> None:
    """The existing LONG cost gate must behave identically."""
    from app.cost.profitability_gate import ProfitabilityGate, ProfitabilityInput

    gate = ProfitabilityGate()
    decision = gate.evaluate(
        ProfitabilityInput(
            symbol="005930",
            action="BUY",
            entry_price=70_000,
            expected_exit_price=71_500,
            quantity=10,
            liquidity_score=0.9,
            account_equity_krw=10_000_000,
            orderbook_snapshot={"bid_price": 69_990, "ask_price": 70_010},
        )
    )
    assert decision.allowed
    assert decision.position_direction == "LONG"
    assert decision.borrow_cost_rate == 0.0


def test_long_exit_is_never_gated_by_the_entry_side_rule() -> None:
    """De-risking must never be blocked. A SELL that closes a long is informational."""
    from app.cost.profitability_gate import ProfitabilityGate, ProfitabilityInput

    gate = ProfitabilityGate()
    decision = gate.evaluate(
        ProfitabilityInput(
            symbol="005930",
            action="SELL",
            entry_price=70_000,
            expected_exit_price=60_000,  # a loss
            quantity=10,
        )
    )
    assert decision.allowed


def test_short_cover_is_never_gated_either() -> None:
    """The one exit whose loss is unbounded if it is blocked."""
    from app.cost.profitability_gate import ProfitabilityGate, ProfitabilityInput

    gate = ProfitabilityGate()
    decision = gate.evaluate(
        ProfitabilityInput(
            symbol="005930",
            action="BUY",
            position_direction="SHORT",
            position_effect="CLOSE",
            execution_product="CREDIT_BORROW",
            entry_price=70_000,
            expected_exit_price=80_000,  # covering at a loss
            quantity=10,
        )
    )
    assert decision.allowed


def test_short_entry_with_exit_on_the_wrong_side_is_rejected() -> None:
    """A short predicting a higher exit is a trade whose own thesis loses money."""
    from app.cost.profitability_gate import (
        REASON_EXIT_WRONG_SIDE,
        ProfitabilityGate,
        ProfitabilityInput,
    )

    gate = ProfitabilityGate()
    decision = gate.evaluate(
        ProfitabilityInput(
            symbol="005930",
            action="SELL",
            position_direction="SHORT",
            position_effect="OPEN",
            execution_product="CREDIT_BORROW",
            entry_price=70_000,
            expected_exit_price=71_500,
            quantity=10,
            borrow_fee_bps_annualised=800.0,
            expected_holding_seconds=1_800,
        )
    )
    assert not decision.allowed
    assert REASON_EXIT_WRONG_SIDE in decision.rejection_reasons


def test_short_entry_with_unknown_borrow_cost_is_rejected() -> None:
    from app.cost.profitability_gate import (
        REASON_BORROW_COST_UNKNOWN,
        ProfitabilityGate,
        ProfitabilityInput,
    )

    gate = ProfitabilityGate()
    decision = gate.evaluate(
        ProfitabilityInput(
            symbol="005930",
            action="SELL",
            position_direction="SHORT",
            position_effect="OPEN",
            execution_product="CREDIT_BORROW",
            entry_price=70_000,
            expected_exit_price=68_500,
            quantity=10,
            expected_holding_seconds=1_800,
        )
    )
    assert not decision.allowed
    assert REASON_BORROW_COST_UNKNOWN in decision.rejection_reasons


def test_short_risk_checks_are_off_by_default_at_the_account_level() -> None:
    """``short_selling_allowed`` / ``credit_loan_allowed`` still default False."""
    from app.schemas.domain import RiskRules

    rules = RiskRules()
    assert not rules.short_selling_allowed
    assert not rules.credit_loan_allowed
    assert not rules.overnight_short_allowed
    # And the short risk budget is capped at half a long's.
    assert rules.short_risk_budget_ratio_of_long <= 0.5


def test_dislocated_regime_blocks_new_entries_in_both_directions() -> None:
    from app.ontology.short_rules import blocked_directions, permits_new_entry, permitted_arms

    assert not permits_new_entry("HIGH_VOL_DISLOCATED")
    assert permitted_arms("HIGH_VOL_DISLOCATED") == ()
    assert set(blocked_directions("HIGH_VOL_DISLOCATED")) == {"LONG", "SHORT"}


def test_unknown_regime_permits_no_arm() -> None:
    """A typo in the regime classifier must not open every arm in both directions."""
    from app.ontology.short_rules import arm_permitted, permitted_arms

    assert permitted_arms("TOTALLY_MADE_UP") == ()
    # ...but the claim is reported as unanswerable rather than as a refusal, matching
    # the long side's existing convention.
    assert arm_permitted("opening_range_breakdown", "SHORT", "TOTALLY_MADE_UP") is None


def test_trend_down_still_permits_a_long_thesis() -> None:
    """A falling index is not a reason to be structurally short-only."""
    from app.ontology.short_rules import permitted_arms

    arms = permitted_arms("TREND_DOWN")
    assert any(arm.endswith(":LONG") for arm in arms)
    assert any(arm.endswith(":SHORT") for arm in arms)


def test_short_ontology_facts_are_closed_world() -> None:
    """Every required fact is individually load-bearing; unstated is false."""
    from app.ontology.short_rules import evaluate_short_facts
    from app.ontology.trading_fact_builder import TradingFacts

    full = TradingFacts(
        symbol="005930",
        position_direction="SHORT",
        position_effect="OPEN",
        execution_product="CREDIT_BORROW",
        has_orderbook=True,
        orderbook_fresh=True,
        best_bid=69_990,
        best_ask=70_010,
        short_sale_permitted=True,
        borrow_available=True,
        borrow_available_quantity=500,
        borrow_quantity_required=10,
        borrow_fee_bps_annualised=800.0,
        borrow_snapshot_age_seconds=15.0,
        hours_to_recall_deadline=72.0,
        days_to_cover=2.0,
        short_strategy_shadow_validated=True,
        short_strategy_live_probe_authorized=True,
    )
    assert evaluate_short_facts(full).executable
    # A bare fact set establishes nothing.
    bare = TradingFacts(symbol="005930", position_direction="SHORT", position_effect="OPEN")
    assert not evaluate_short_facts(bare).executable
    # Each fact removed individually blocks execution.
    for field in (
        "short_sale_permitted",
        "borrow_available",
        "short_strategy_shadow_validated",
        "short_strategy_live_probe_authorized",
    ):
        assert not evaluate_short_facts(dataclasses.replace(full, **{field: False})).executable, field
    assert not evaluate_short_facts(
        dataclasses.replace(full, borrow_fee_bps_annualised=None)
    ).executable
    assert not evaluate_short_facts(
        dataclasses.replace(full, borrow_snapshot_age_seconds=900.0)
    ).executable


# --------------------------------------------------------------------------- #
# 8. The evidence loop closes                                                  #
# --------------------------------------------------------------------------- #
# Without this service, plans are journaled and never scored: no forward evidence
# accumulates and no arm can ever leave SHADOW. That failure is SAFE but not correct —
# a ladder whose bottom rung has no exit is a permanent block dressed up as validation.
def _service(tmp_path) -> "ShadowEvaluationService":
    from app.trading.shadow_evaluation_service import ShadowEvaluationService

    return ShadowEvaluationService(
        simulator=ShadowFillSimulator(entry_slippage_bps=0.0, exit_slippage_bps=0.0),
        shadow_store=ShadowPlanStore(tmp_path / "shadow.sqlite3"),
        performance_store=StrategyPerformanceStore(tmp_path / "perf.sqlite3"),
    )


def _quote(ts: datetime, bid: float, ask: float) -> dict[str, object]:
    return {"bid_price": bid, "ask_price": ask, "observed_at": ts}


def test_resolved_shadow_outcomes_become_promotion_evidence(tmp_path) -> None:
    service = _service(tmp_path)
    service.evaluate_tick({}, now=NOW, new_plans=[_plan()])
    service.evaluate_tick(
        {"005930": _quote(NOW + timedelta(seconds=1), 999.0, 1001.0)},
        now=NOW + timedelta(seconds=1),
    )
    stats = service.evaluate_tick(
        {"005930": _quote(NOW + timedelta(seconds=600), 979.0, 981.0)},
        now=NOW + timedelta(seconds=600),
    )
    assert stats.scored == 1
    metrics = service.performance_store.directional_metrics(
        KEY, evaluation_sources=(EVALUATION_SOURCE_SHADOW,)
    )
    assert metrics["filled_trade_count"] == 1
    # ...and the bandit's posterior can now see it.
    assert service.performance_store.posterior_for_key(KEY).sample_count == 1


def test_quotes_are_routed_by_symbol(tmp_path) -> None:
    """Feeding one symbol's book to another symbol's plan would fabricate the whole
    price path, and every barrier decision after that is fiction."""
    service = _service(tmp_path)
    other = dataclasses.replace(
        _plan(),
        plan_id="plan-2",
        symbol="000660",
        entry_reference_price=2000.0,
        borrow_snapshot=BorrowSnapshot(
            symbol="000660",
            observed_at=NOW - timedelta(seconds=30),
            available=True,
            available_quantity=500,
            borrow_fee_bps_annualised=800.0,
        ),
    )
    service.evaluate_tick({}, now=NOW, new_plans=[_plan(), other])
    # A 005930 book at 005930's target must not resolve the 000660 plan.
    service.evaluate_tick(
        {"005930": _quote(NOW + timedelta(seconds=1), 999.0, 1001.0)},
        now=NOW + timedelta(seconds=1),
    )
    stats = service.evaluate_tick(
        {"005930": _quote(NOW + timedelta(seconds=600), 979.0, 981.0)},
        now=NOW + timedelta(seconds=600),
    )
    assert stats.resolved == 1
    assert stats.open_plans == 1


def test_quote_without_a_real_observation_time_is_skipped(tmp_path) -> None:
    """Stamping a quote with ``now`` would defeat the temporal leak check."""
    service = _service(tmp_path)
    service.evaluate_tick({}, now=NOW, new_plans=[_plan()])
    stats = service.evaluate_tick(
        {"005930": {"bid_price": 979.0, "ask_price": 981.0}},
        now=NOW + timedelta(seconds=600),
    )
    # Not resolved by the quote; only the horizon expiry could act, and 600s < 3600s.
    assert stats.resolved == 0
    assert stats.open_plans == 1


def test_unexecutable_plan_is_journaled_but_excluded_from_metrics(tmp_path) -> None:
    service = _service(tmp_path)
    stats = service.evaluate_tick(
        {}, now=NOW, new_plans=[_plan(plan_id="plan-x", borrow_snapshot=None)]
    )
    assert stats.unexecutable == 1
    metrics = service.performance_store.directional_metrics(
        KEY, evaluation_sources=(EVALUATION_SOURCE_SHADOW,)
    )
    assert metrics["filled_trade_count"] == 0
    # But it IS counted in the borrow denominator.
    assert metrics["executable_signal_count"] == 1
    assert metrics["borrow_availability_rate"] == pytest.approx(0.0)


def test_short_rescue_rate_is_none_until_observed(tmp_path) -> None:
    """``None`` must FAIL the promotion gate rather than pass as "no rescues"."""
    service = _service(tmp_path)
    assert service.short_rescue_rate is None
    service.record_directional_comparison({"short_rescued": True})
    service.record_directional_comparison({"short_rescued": False})
    assert service.short_rescue_rate == pytest.approx(0.5)


def test_unmeasured_rescue_rate_blocks_promotion() -> None:
    """The failure direction of the un-wired metric is safe."""
    thresholds = ShortStrategyPromotionConfig().shadow_to_live_probe
    failures = evaluate_hard_gates(_passing_snapshot(short_rescue_rate=None), thresholds)
    assert ShortReasonCodes.RESCUE_RATE_INSUFFICIENT in failures


def test_session_hands_journaled_plans_over_exactly_once(tmp_path) -> None:
    """Drained, not accumulated: a cycle whose plans nobody collects must not grow."""
    from app.trading.strategy_session import StrategySessionManager, StrategySessionConfig

    manager = StrategySessionManager(
        config=StrategySessionConfig(state_path=str(tmp_path / "session.json"))
    )
    assert manager.drain_shadow_plans() == ()
    manager._pending_shadow_plans = [_plan()]
    assert len(manager.drain_shadow_plans()) == 1
    assert manager.drain_shadow_plans() == ()

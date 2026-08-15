from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.strategy.catalog import STRATEGY_IDS
from app.strategy.experts import (
    ALL_EXPERT_TYPES,
    BreakoutVolumeExpert,
    ExpertContext,
    LiquidityShockReversalExpert,
    OwnedStrategyLifecycle,
    RvgiBoxBreakoutExpert,
)
from app.trading.contracts import IntentAction
from app.trading.directional import PositionDirection


NOW = datetime(2026, 7, 27, 1, 0, tzinfo=timezone.utc)


def _expert_context(**quantiles: float) -> ExpertContext:
    return ExpertContext(
        symbol="005930",
        as_of=NOW,
        price=80000,
        proposed_quantity=1,
        feature_snapshot_id="features-policy-test",
        utility_evidence_id="utility-policy-test",
        quantiles=quantiles,
    )


def test_breakout_requires_follow_through_and_executable_liquidity() -> None:
    expert = BreakoutVolumeExpert()
    assert expert.admissible(
        _expert_context(**{"breakout": 0.9, "breakout_acceptance": 1.0, "volume": 0.9, "return": 0.8, "liquidity": 0.8})
    )
    assert not expert.admissible(
        _expert_context(**{"breakout": 0.9, "breakout_acceptance": 1.0, "volume": 0.9, "return": 0.5, "liquidity": 0.8})
    )
    assert not expert.admissible(
        _expert_context(**{"breakout": 0.9, "breakout_acceptance": 1.0, "volume": 0.9, "return": 0.8, "liquidity": 0.5})
    )


def test_liquidity_shock_reversal_waits_for_actual_normalization() -> None:
    expert = LiquidityShockReversalExpert()
    common = dict(liquidity_shock=0.9, price_drop=0.9, recovery=0.8, liquidity=0.8)
    assert expert.admissible(_expert_context(**common, liquidity_recovery=0.8))
    assert not expert.admissible(_expert_context(**common, liquidity_recovery=0.2))


def test_rvgi_box_requires_cross_timing_and_executable_liquidity() -> None:
    expert = RvgiBoxBreakoutExpert()
    context = _expert_context(
        rvgi_diff=0.9,
        rvgi_cross=0.9,
        box_position=0.9,
        volume=0.8,
        liquidity=0.8,
        false_breakout_risk=0.1,
    )
    assert expert.admissible(context)
    assert not expert.admissible(
        _expert_context(
            rvgi_diff=0.9,
            rvgi_cross=0.0,
            box_position=0.9,
            volume=0.8,
            liquidity=0.8,
            false_breakout_risk=0.1,
        )
    )


def test_every_catalogued_expert_creates_an_independent_trade_plan() -> None:
    quantiles = {
        "return": 0.9,
        "volume": 0.9,
        "breakout": 0.9,
        "breakout_acceptance": 1.0,
        "vwap_deviation": 0.1,
        "reversion": 0.9,
        "liquidity_shock": 0.9,
        "price_drop": 0.9,
        "recovery": 0.9,
        "event_relevance": 0.9,
        "event_direction": 0.9,
        "relative_strength": 0.9,
        "liquidity": 0.9,
        "gap": 0.9,
        "opening_confirmation": 0.9,
        "gap_entry_window": 1.0,
        "rvgi_diff": 0.9,
        "rvgi_cross": 0.9,
        "box_position": 0.9,
        "false_breakout_risk": 0.1,
        # Residual / microstructure strategies. Low values mean "strong" for the
        # displacement and toxicity keys, the same convention as vwap_deviation
        # and false_breakout_risk above.
        "residual_strength_short": 0.9,
        "residual_strength_long": 0.9,
        "investor_flow": 0.9,
        "vwap_zscore": 0.1,
        "liquidity_recovery": 0.9,
        "microprice_edge": 0.9,
        "ofi_slope": 0.9,
        "depth_recovery": 0.9,
        "flow_toxicity": 0.1,
        # Opening-range breakout. Relative volume is a hard precondition of that
        # thesis ("stocks in play"), so it belongs in the all-inputs-strong fixture.
        "opening_range_breakout": 0.9,
        "relative_volume": 0.9,
        # Market intraday momentum. The window flag is session structure, so an
        # all-inputs-strong fixture must place the moment INSIDE the entry window.
        # Overnight gap carry reads the same session clock one window later. Its
        # directional input is ``breakout`` (closing at the top of the recent
        # range), NOT ``vwap_deviation``: this fixture holds the latter at 0.1
        # because "strong" means low for the reversion theses, and one fixture
        # cannot mean both directions at once.
        "overnight_carry_window": 1.0,
        "intraday_momentum_signal": 0.9,
        "intraday_momentum_window": 1.0,
        "first_half_hour_volatility": 0.9,
        "momentum_persistence_short": 0.9,
        "momentum_persistence_long": 0.9,
    }
    # A single "all inputs strong" fixture cannot make both directions fire, and it
    # should not: the inputs that make a long thesis strong are precisely the ones that
    # make its short counterpart wrong. So each direction gets the fixture that is
    # maximally favourable TO IT, and the assertion below is that every catalogued
    # expert proposes under its own favourable conditions.
    #
    # For shorts the directional signals are mirrored (low == weak == strong short
    # case) and two inputs with no long-side analogue are supplied: ``borrow_available``
    # (a hard precondition — no locate, no trade) and ``squeeze_risk``.
    short_quantiles = {
        **quantiles,
        "intraday_momentum_signal": 0.1,
        "opening_range_breakdown": 0.9,
        "aggressor_imbalance": 0.1,
        "residual_strength_short": 0.1,
        "residual_strength_long": 0.1,
        "investor_flow": 0.1,
        "squeeze_risk": 0.1,
        "borrow_available": 0.95,
    }
    # Same reasoning one step finer: ``box_position`` is the raw position inside the
    # 20-bar box, and the two theses that read it want OPPOSITE ends. The fixture holds
    # it at 0.9 for rvgi_box_breakout ("at the ceiling"), so the range-floor thesis —
    # whose favourable fact is "at the floor" — needs its own value on that one key.
    # Inverting it globally would simply move the failure to the breakout expert.
    per_strategy_quantiles = {
        "range_support_reversion": {**quantiles, "box_position": 0.05},
        # The CHOP thesis wants the opposite persistence regime from every trend
        # expert. Keep the shared strong fixture and override only that fact.
        "choppiness_range_reversion": {
            **quantiles,
            "momentum_persistence_long": 0.1,
        },
    }

    def _context(values: dict[str, float]) -> ExpertContext:
        return ExpertContext(
            symbol="005930",
            as_of=NOW,
            price=80000,
            proposed_quantity=2,
            feature_snapshot_id="features-1",
            utility_evidence_id="utility-1",
            quantiles=values,
        )

    plans = tuple(
        expert_type().propose(
            _context(
                per_strategy_quantiles.get(
                    expert_type.strategy_id,
                    short_quantiles
                    if expert_type.direction is PositionDirection.SHORT
                    else quantiles,
                )
            )
        )
        for expert_type in ALL_EXPERT_TYPES
    )
    expected = len(STRATEGY_IDS)
    assert len(plans) == expected
    assert all(plan is not None for plan in plans)
    assert len({plan.strategy_id for plan in plans if plan}) == expected
    assert len({plan.strategy_instance_id for plan in plans if plan}) == expected
    # Each plan's broker side must match its declared direction/effect: an OPEN LONG is
    # a BUY, an OPEN SHORT is a SELL. TradePlan validates this itself, so a mismatch
    # would have raised — this pins that the experts actually produce both.
    sides = {
        (plan.position_direction, plan.side) for plan in plans if plan is not None
    }
    assert ("LONG", "BUY") in sides
    assert ("SHORT", "SELL") in sides
    # Every short plan ships SHADOW. A proposal is not an authorisation, and nothing in
    # the expert layer may hand out a live short.
    assert all(
        plan.deployment_state == "SHADOW"
        for plan in plans
        if plan is not None and plan.position_direction == "SHORT"
    )


def test_short_experts_refuse_without_a_borrow_locate() -> None:
    """No locate, no trade — regardless of how good the price thesis looks.

    ``ExpertContext.q`` defaults an absent quantile to 0.5, which is below the entry
    bar, so "we never asked about borrow" reads as "not available". That is the
    fail-closed direction, and it is the one input whose absence cannot be recovered
    from at execution time.
    """
    from app.strategy.experts import ShortStrategyExpert

    short_types = [
        kind for kind in ALL_EXPERT_TYPES if kind.direction is PositionDirection.SHORT
    ]
    assert short_types, "expected at least one catalogued short expert"
    ideal = {
        "intraday_momentum_signal": 0.1,
        "intraday_momentum_window": 1.0,
        "first_half_hour_volatility": 0.9,
        "opening_range_breakdown": 0.9,
        "relative_volume": 0.9,
        "aggressor_imbalance": 0.1,
        "residual_strength_short": 0.1,
        "residual_strength_long": 0.1,
        "investor_flow": 0.1,
        "squeeze_risk": 0.1,
        "liquidity": 0.9,
        "volume": 0.9,
    }
    for kind in short_types:
        assert issubclass(kind, ShortStrategyExpert), kind
        # Borrow present -> proposes. Borrow absent -> refuses, same thesis inputs.
        with_borrow = ExpertContext(
            symbol="005930",
            as_of=NOW,
            price=80000,
            proposed_quantity=2,
            feature_snapshot_id="features-1",
            utility_evidence_id="utility-1",
            quantiles={**ideal, "borrow_available": 0.95},
        )
        without_borrow = ExpertContext(
            symbol="005930",
            as_of=NOW,
            price=80000,
            proposed_quantity=2,
            feature_snapshot_id="features-1",
            utility_evidence_id="utility-1",
            quantiles=dict(ideal),
        )
        assert kind().propose(with_borrow) is not None, kind.strategy_id
        assert kind().propose(without_borrow) is None, kind.strategy_id


def test_opening_range_breakout_requires_stocks_in_play() -> None:
    """The relative-volume gate is the thesis, not a tuning knob.

    In the published result the unrestricted opening-range breakout does not pay;
    restricting it to the highest relative-volume names is what produces the edge.
    A breakout that fires without relative volume is therefore a different and
    unprofitable strategy wearing the same name, so this pins the precondition.
    """
    from app.strategy.experts import OpeningRangeBreakoutExpert

    def _context(**overrides: float) -> ExpertContext:
        quantiles = {
            "opening_range_breakout": 0.9,
            "relative_volume": 0.9,
            "volume": 0.9,
            "liquidity": 0.9,
        }
        quantiles.update(overrides)
        return ExpertContext(
            symbol="005930",
            as_of=NOW,
            price=80000,
            proposed_quantity=2,
            feature_snapshot_id="features-1",
            utility_evidence_id="utility-1",
            quantiles=quantiles,
        )

    expert = OpeningRangeBreakoutExpert()
    assert expert.propose(_context()) is not None
    # No relative volume -> not in play -> must not fire, even on a clean break.
    assert expert.propose(_context(relative_volume=0.4)) is None
    # In play but price never cleared the opening range -> nothing to trade.
    assert expert.propose(_context(opening_range_breakout=0.4)) is None


def test_strategy_instance_owns_entry_and_mechanical_exit() -> None:
    context = ExpertContext(
        symbol="005930",
        as_of=NOW,
        price=80000,
        proposed_quantity=2,
        feature_snapshot_id="features-1",
        utility_evidence_id="utility-1",
        quantiles={
            "return": 0.9,
            "volume": 0.9,
            "momentum_persistence_short": 0.9,
            "momentum_persistence_long": 0.9,
        },
    )
    plan = ALL_EXPERT_TYPES[0]().propose(context)
    assert plan is not None
    lifecycle = OwnedStrategyLifecycle(plan)
    entry = lifecycle.entry_intent(NOW)
    assert entry.action == IntentAction.BUY
    assert entry.strategy_instance_id == plan.strategy_instance_id
    exit_intent = lifecycle.exit_intent(
        position_id="position-1",
        quantity=2,
        price=float(plan.initial_stop["price"]) - 1,
        opened_at=NOW,
        as_of=NOW + timedelta(seconds=1),
    )
    assert exit_intent is not None
    assert exit_intent.action == IntentAction.SELL
    assert exit_intent.strategy_instance_id == plan.strategy_instance_id
    assert exit_intent.reason_code == "INITIAL_STOP"

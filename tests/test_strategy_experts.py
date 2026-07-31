from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.strategy.catalog import STRATEGY_IDS
from app.strategy.experts import ALL_EXPERT_TYPES, ExpertContext, OwnedStrategyLifecycle
from app.trading.contracts import IntentAction


NOW = datetime(2026, 7, 27, 1, 0, tzinfo=timezone.utc)


def test_every_catalogued_expert_creates_an_independent_trade_plan() -> None:
    quantiles = {
        "return": 0.9,
        "volume": 0.9,
        "breakout": 0.9,
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
        "intraday_momentum_signal": 0.9,
        "intraday_momentum_window": 1.0,
        "first_half_hour_volatility": 0.9,
    }
    context = ExpertContext(
        symbol="005930",
        as_of=NOW,
        price=80000,
        proposed_quantity=2,
        feature_snapshot_id="features-1",
        utility_evidence_id="utility-1",
        quantiles=quantiles,
    )
    plans = tuple(expert_type().propose(context) for expert_type in ALL_EXPERT_TYPES)
    expected = len(STRATEGY_IDS)
    assert len(plans) == expected
    assert all(plan is not None for plan in plans)
    assert len({plan.strategy_id for plan in plans if plan}) == expected
    assert len({plan.strategy_instance_id for plan in plans if plan}) == expected


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
        quantiles={"return": 0.9, "volume": 0.9},
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

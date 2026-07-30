from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.strategy.experts import ALL_EXPERT_TYPES, ExpertContext, OwnedStrategyLifecycle
from app.trading.contracts import IntentAction


NOW = datetime(2026, 7, 27, 1, 0, tzinfo=timezone.utc)


def test_all_eight_experts_create_independent_trade_plans() -> None:
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
    assert len(plans) == 8
    assert all(plan is not None for plan in plans)
    assert len({plan.strategy_id for plan in plans if plan}) == 8
    assert len({plan.strategy_instance_id for plan in plans if plan}) == 8


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

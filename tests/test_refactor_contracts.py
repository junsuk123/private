from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.strategy.ownership import OwnershipGuard, PositionOwnershipError
from app.trading.contracts import (
    IntentAction,
    LatencyTrace,
    OrderIntent,
    Position,
    RiskVerdict,
    RiskVerdictAction,
    StrategyUtilityEvidence,
)


NOW = datetime(2026, 7, 27, 1, 0, tzinfo=timezone.utc)


def _intent(**overrides: object) -> OrderIntent:
    values: dict[str, object] = {
        "intent_id": "intent-1",
        "idempotency_key": "strategy-1:decision-1",
        "strategy_instance_id": "strategy-instance-1",
        "position_id_if_any": None,
        "symbol": "005930",
        "action": IntentAction.BUY,
        "quantity": 3,
        "limit_or_price_policy": {"kind": "passive_limit", "limit": 80000},
        "urgency": "NORMAL",
        "reason_code": "ENTRY_TRIGGERED",
        "created_at": NOW,
        "expires_at": NOW + timedelta(seconds=5),
    }
    values.update(overrides)
    return OrderIntent(**values)  # type: ignore[arg-type]


def test_order_intent_requires_aware_ordered_timestamps() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _intent(created_at=NOW.replace(tzinfo=None))
    with pytest.raises(ValueError, match="expire after"):
        _intent(expires_at=NOW)


def test_utility_is_explicitly_net_of_costs() -> None:
    evidence = StrategyUtilityEvidence(
        evidence_id="evidence-1",
        as_of=NOW,
        symbol="005930",
        strategy_id="intraday_momentum",
        ontology_allowed=True,
        hard_block_reasons=(),
        compatibility_score=0.8,
        probability_success=0.6,
        expected_gross_return_bps=25.0,
        expected_cost_bps=8.0,
        expected_net_return_bps=17.0,
        expected_adverse_excursion_bps=10.0,
        expected_favorable_excursion_bps=30.0,
        fill_probability=0.7,
        expected_holding_seconds=120.0,
        aleatoric_uncertainty=0.2,
        epistemic_uncertainty_or_proxy=0.1,
        utility=7.0,
        model_version="shadow-none",
        feature_snapshot_id="features-1",
        ontology_snapshot_id="ontology-1",
        explanation_paths=("fresh-data",),
    )
    assert evidence.expected_net_return_bps == 17.0
    with pytest.raises(ValueError, match="gross return minus costs"):
        StrategyUtilityEvidence(
            **{**evidence.__dict__, "expected_net_return_bps": 18.0}
        )


def test_risk_rejection_cannot_approve_quantity() -> None:
    with pytest.raises(ValueError, match="zero quantity"):
        RiskVerdict(
            verdict_id="verdict-1",
            intent_id="intent-1",
            action=RiskVerdictAction.REJECT,
            approved_quantity=1,
            limits_evaluated={},
            reason_codes=("STALE_DATA",),
            account_snapshot_id="account-1",
            timestamp=NOW,
        )


def test_latency_trace_uses_monotonic_order_and_is_immutable() -> None:
    trace = LatencyTrace(
        trace_id="trace-1",
        correlation_id="decision-1",
        symbol="005930",
        exchange_timestamp=NOW,
    )
    received = trace.mark("feed_receive_monotonic_ns", 100)
    normalized = received.mark("normalized_monotonic_ns", 140)
    assert trace.feed_receive_monotonic_ns is None
    assert normalized.duration_ns(
        "feed_receive_monotonic_ns", "normalized_monotonic_ns"
    ) == 40
    with pytest.raises(ValueError, match="monotonic"):
        normalized.mark("feature_ready_monotonic_ns", 120)


def test_non_origin_strategy_cannot_manage_position() -> None:
    position = Position(
        position_id="position-1",
        symbol="005930",
        quantity=3,
        average_price=80000,
        origin_strategy_id="intraday_momentum",
        strategy_instance_id="strategy-instance-1",
        opened_at=NOW,
    )
    OwnershipGuard().assert_owner(position, "strategy-instance-1")
    with pytest.raises(PositionOwnershipError, match="DIFFERENT_STRATEGY"):
        OwnershipGuard().assert_owner(position, "strategy-instance-2")

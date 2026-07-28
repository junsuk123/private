from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.routing.orchestrator import StrategyActivation, StrategyOrchestrator
from app.storage.lifecycle_store import LifecycleStore
from app.strategy.experts import ExpertContext
from app.trading.contracts import (
    IntentAction,
    OntologyDecision,
    Position,
    StrategyUtilityEvidence,
)


NOW = datetime(2026, 7, 27, 1, 0, tzinfo=timezone.utc)


def _ontology() -> OntologyDecision:
    return OntologyDecision(
        snapshot_id="ontology-1",
        as_of=NOW,
        symbol="005930",
        allowed_strategy_ids=("intraday_momentum",),
        blocked_strategy_reasons={},
        compatibility_scores={"intraday_momentum": 1},
        explanation_paths={"intraday_momentum": ("unit",)},
        valid_until=NOW + timedelta(seconds=5),
    )


def _evidence() -> StrategyUtilityEvidence:
    return StrategyUtilityEvidence(
        evidence_id="utility-1",
        as_of=NOW,
        symbol="005930",
        strategy_id="intraday_momentum",
        ontology_allowed=True,
        hard_block_reasons=(),
        compatibility_score=1,
        probability_success=0.6,
        expected_gross_return_bps=15,
        expected_cost_bps=5,
        expected_net_return_bps=10,
        expected_adverse_excursion_bps=5,
        expected_favorable_excursion_bps=15,
        fill_probability=0.8,
        expected_holding_seconds=60,
        aleatoric_uncertainty=0.1,
        epistemic_uncertainty_or_proxy=0.1,
        utility=2,
        model_version="unit",
        feature_snapshot_id="features-1",
        ontology_snapshot_id="ontology-1",
        explanation_paths=("unit",),
    )


def test_activation_persists_plan_and_owner_then_restart_manages_exit(tmp_path) -> None:
    path = tmp_path / "lifecycle.sqlite3"
    orchestrator = StrategyOrchestrator(LifecycleStore(path))
    context = ExpertContext(
        symbol="005930",
        as_of=NOW,
        price=80000,
        proposed_quantity=2,
        feature_snapshot_id="features-1",
        utility_evidence_id="utility-1",
        quantiles={"return": 0.9, "volume": 0.9},
    )
    activation = orchestrator.activate(
        context=context, ontology=_ontology(), evidence=(_evidence(),)
    )
    assert isinstance(activation, StrategyActivation)
    position = Position(
        position_id="position-1",
        symbol="005930",
        quantity=2,
        average_price=80000,
        origin_strategy_id=activation.plan.strategy_id,
        strategy_instance_id=activation.plan.strategy_instance_id,
        opened_at=NOW,
    )
    orchestrator.record_open_position(position, NOW)

    restarted = StrategyOrchestrator(LifecycleStore(path))
    exit_intent = restarted.manage_position(
        position,
        price=float(activation.plan.initial_stop["price"]) - 1,
        as_of=NOW + timedelta(seconds=1),
    )
    assert exit_intent is not None
    assert exit_intent.action == IntentAction.SELL
    assert exit_intent.strategy_instance_id == activation.plan.strategy_instance_id

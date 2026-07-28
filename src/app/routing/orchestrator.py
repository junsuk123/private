from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.routing.strategy_router import RoutingDecision, StrategyRouter
from app.storage.lifecycle_store import LifecycleStore
from app.strategy.experts import (
    ALL_EXPERT_TYPES,
    ExpertContext,
    OwnedStrategyLifecycle,
    StrategyExpert,
)
from app.strategy.ownership import OwnershipGuard
from app.trading.contracts import (
    OntologyDecision,
    OrderIntent,
    Position,
    StrategyInstanceState,
    StrategyLifecycleStatus,
    StrategyUtilityEvidence,
    TradePlan,
)


@dataclass(frozen=True)
class StrategyActivation:
    routing: RoutingDecision
    plan: TradePlan
    entry_intent: OrderIntent


class StrategyOrchestrator:
    def __init__(
        self,
        store: LifecycleStore,
        *,
        router: StrategyRouter | None = None,
        experts: tuple[StrategyExpert, ...] | None = None,
    ) -> None:
        self.store = store
        self.router = router or StrategyRouter()
        configured = experts or tuple(expert_type() for expert_type in ALL_EXPERT_TYPES)
        self.experts = {expert.strategy_id: expert for expert in configured}
        self.ownership = OwnershipGuard()

    def activate(
        self,
        *,
        context: ExpertContext,
        ontology: OntologyDecision,
        evidence: tuple[StrategyUtilityEvidence, ...],
    ) -> StrategyActivation | RoutingDecision:
        open_positions = self.store.load_open_positions()
        routing = self.router.route(
            as_of=context.as_of,
            symbol=context.symbol,
            ontology=ontology,
            evidence=evidence,
            open_positions=open_positions,
        )
        if routing.is_no_trade or routing.selected is None:
            return routing
        expert = self.experts.get(routing.selected.strategy_id)
        if expert is None:
            return RoutingDecision(
                as_of=context.as_of,
                symbol=context.symbol,
                action="NO_TRADE",
                selected=None,
                reason_codes=("STRATEGY_IMPLEMENTATION_MISSING",),
                ranked_evidence_ids=routing.ranked_evidence_ids,
            )
        selected_context = ExpertContext(
            symbol=context.symbol,
            as_of=context.as_of,
            price=context.price,
            proposed_quantity=context.proposed_quantity,
            feature_snapshot_id=context.feature_snapshot_id,
            utility_evidence_id=routing.selected.evidence_id,
            quantiles=context.quantiles,
        )
        plan = expert.propose(selected_context)
        if plan is None:
            return RoutingDecision(
                as_of=context.as_of,
                symbol=context.symbol,
                action="NO_TRADE",
                selected=None,
                reason_codes=("STRATEGY_ENTRY_TRIGGER_NOT_MET",),
                ranked_evidence_ids=routing.ranked_evidence_ids,
            )
        state = StrategyInstanceState(
            strategy_instance_id=plan.strategy_instance_id,
            strategy_id=plan.strategy_id,
            symbol=plan.symbol,
            status=StrategyLifecycleStatus.ARMED,
            created_at=context.as_of,
            updated_at=context.as_of,
        )
        self.store.save_strategy_instance(state)
        self.store.save_trade_plan(plan)
        return StrategyActivation(
            routing=routing,
            plan=plan,
            entry_intent=OwnedStrategyLifecycle(plan).entry_intent(context.as_of),
        )

    def record_open_position(self, position: Position, as_of: datetime) -> None:
        state = self.store.load_strategy_instance(position.strategy_instance_id)
        if state is None:
            raise ValueError("cannot open position without strategy instance")
        self.ownership.assert_strategy(position, state.strategy_id)
        self.ownership.assert_owner(position, state.strategy_instance_id)
        self.store.save_position(position)
        self.store.save_strategy_instance(
            StrategyInstanceState(
                strategy_instance_id=state.strategy_instance_id,
                strategy_id=state.strategy_id,
                symbol=state.symbol,
                status=StrategyLifecycleStatus.OPEN,
                created_at=state.created_at,
                updated_at=as_of,
                position_id=position.position_id,
                state_version=state.state_version + 1,
            )
        )

    def manage_position(
        self,
        position: Position,
        *,
        price: float,
        as_of: datetime,
        invalidated: bool = False,
        data_stale: bool = False,
    ) -> OrderIntent | None:
        state = self.store.load_strategy_instance(position.strategy_instance_id)
        plan = self.store.load_trade_plan(position.strategy_instance_id)
        if state is None or plan is None:
            raise ValueError("position lifecycle state is incomplete")
        self.ownership.assert_owner(position, state.strategy_instance_id)
        self.ownership.assert_strategy(position, state.strategy_id)
        return OwnedStrategyLifecycle(plan).exit_intent(
            position_id=position.position_id,
            quantity=position.quantity,
            price=price,
            opened_at=position.opened_at,
            as_of=as_of,
            invalidated=invalidated,
            data_stale=data_stale,
        )

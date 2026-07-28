from __future__ import annotations

from dataclasses import dataclass

from app.execution.causal_journal import CausalOrderJournal
from app.execution.kis_types import LiveOrderSubmission
from app.execution.live_execution_coordinator import LiveExecutionCoordinator
from app.risk.causal_gate import CausalRiskGate
from app.schemas.domain import FinalOrder, OrderSide, OrderType
from app.trading.contracts import AccountSnapshot, IntentAction, OrderIntent, RiskVerdict


@dataclass(frozen=True)
class StrategyOwnedExecutionResult:
    intent: OrderIntent
    verdict: RiskVerdict
    submission: LiveOrderSubmission | None


class StrategyOwnedExecutionWorkflow:
    def __init__(
        self,
        risk_gate: CausalRiskGate,
        coordinator: LiveExecutionCoordinator,
    ) -> None:
        self.risk_gate = risk_gate
        self.coordinator = coordinator

    def execute(
        self,
        intent: OrderIntent,
        account: AccountSnapshot,
        *,
        market: str,
        limit_price: float,
    ) -> StrategyOwnedExecutionResult:
        verdict = self.risk_gate.evaluate(intent, account, price=limit_price)
        if verdict.approved_quantity <= 0:
            # Rejections are still durable causal outcomes.
            causal = self.coordinator.causal_journal or CausalOrderJournal()
            self.coordinator.causal_journal = causal
            causal.persist_intent(intent)
            causal.persist_risk_verdict(verdict)
            return StrategyOwnedExecutionResult(intent, verdict, None)
        order = FinalOrder(
            ticker=intent.symbol,
            market=market,
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY if intent.action == IntentAction.BUY else OrderSide.SELL,
            quantity=verdict.approved_quantity,
            limit_price=limit_price,
            manual_approval_required=False,
        )
        submission = self.coordinator.submit_approved_intent(
            intent, verdict, order
        )
        return StrategyOwnedExecutionResult(intent, verdict, submission)

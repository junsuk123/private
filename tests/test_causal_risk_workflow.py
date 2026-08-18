from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from app.execution.causal_journal import CausalOrderJournal
from app.execution.idempotency_store import IdempotencyStore
from app.execution.live_execution_coordinator import LiveExecutionCoordinator
from app.execution.live_order_journal import LiveOrderJournal
from app.execution.strategy_owned_workflow import StrategyOwnedExecutionWorkflow
from app.risk.causal_gate import CausalRiskGate, CausalRiskLimits
from app.trading.contracts import (
    AccountSnapshot,
    IntentAction,
    OrderIntent,
    Position,
    RiskVerdictAction,
)


NOW = datetime(2026, 7, 27, 1, 0, tzinfo=timezone.utc)


def _intent(action=IntentAction.BUY, position_id=None, owner="owner-1"):
    return OrderIntent(
        intent_id="intent-1",
        idempotency_key="owner-1:decision-1",
        strategy_instance_id=owner,
        position_id_if_any=position_id,
        symbol="005930",
        action=action,
        quantity=10,
        limit_or_price_policy={"kind": "limit"},
        urgency="NORMAL",
        reason_code="UNIT",
        created_at=NOW,
        expires_at=NOW + timedelta(seconds=5),
    )


def test_buy_is_resized_by_cash_notional_and_symbol_limits() -> None:
    gate = CausalRiskGate(CausalRiskLimits(500, 4))
    account = AccountSnapshot("account-1", NOW, {"KRW": 1000}, ())
    verdict = gate.evaluate(_intent(), account, price=150, timestamp=NOW)
    assert verdict.action == RiskVerdictAction.RESIZE
    assert verdict.approved_quantity == 3


def test_non_owner_cannot_sell_position() -> None:
    position = Position(
        "position-1", "005930", 5, 100, "strategy", "owner-1", NOW
    )
    account = AccountSnapshot("account-1", NOW, {"KRW": 0}, (position,))
    verdict = CausalRiskGate(CausalRiskLimits(1000, 10)).evaluate(
        _intent(IntentAction.SELL, "position-1", "owner-2"),
        account,
        price=100,
        timestamp=NOW,
    )
    assert verdict.action == RiskVerdictAction.REJECT
    assert verdict.reason_codes == ("POSITION_OWNER_MISMATCH",)


def test_full_strategy_owned_chain_reaches_mock_broker_once(tmp_path) -> None:
    class Broker:
        calls = 0

        def place_limit_order(self, order):
            self.calls += 1
            return SimpleNamespace(order_id="broker-1", status="ACCEPTED", message="ok")

    broker = Broker()
    coordinator = LiveExecutionCoordinator(
        broker,
        idempotency_store=IdempotencyStore(tmp_path / "idempotency.jsonl"),
        journal=LiveOrderJournal(tmp_path / "legacy.jsonl"),
        causal_journal=CausalOrderJournal(tmp_path / "causal.jsonl"),
    )
    coordinator._preflight_failures = lambda: []  # type: ignore[method-assign]
    # Same reasoning as the process-level gates above: this test is about the
    # causal chain, not about whether a session is open right now.
    coordinator.execution_guard = None
    workflow = StrategyOwnedExecutionWorkflow(
        CausalRiskGate(CausalRiskLimits(1000, 10)), coordinator
    )
    account = AccountSnapshot("account-1", NOW, {"KRW": 1000}, ())
    current = datetime.now(timezone.utc)
    intent = replace(
        _intent(),
        created_at=current,
        expires_at=current + timedelta(seconds=5),
    )
    with patch.dict(
        "os.environ",
        {"REFACTOR_STRATEGY_OWNED_EXECUTION": "true"},
        clear=True,
    ):
        first = workflow.execute(intent, account, market="KRX", limit_price=100)
        second = workflow.execute(intent, account, market="KRX", limit_price=100)
    assert first.submission is not None
    assert second.submission is not None
    assert second.submission.message == "idempotent replay"
    assert broker.calls == 1

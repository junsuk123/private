from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.execution.causal_journal import CausalOrderJournal
from app.execution.idempotency_store import IdempotencyStore
from app.execution.kis_errors import LiveExecutionBlocked
from app.execution.live_execution_coordinator import LiveExecutionCoordinator
from app.execution.live_order_journal import LiveOrderJournal
from app.schemas.domain import FinalOrder, OrderSide, OrderType
from app.trading.contracts import (
    IntentAction,
    OrderIntent,
    RiskVerdict,
    RiskVerdictAction,
)


NOW = datetime(2026, 7, 27, 1, 0, tzinfo=timezone.utc)


class _Broker:
    def __init__(self) -> None:
        self.calls = 0

    def place_limit_order(self, order: FinalOrder) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(order_id="broker-1", status="ACCEPTED", message="ok")


def _intent(action: IntentAction = IntentAction.BUY) -> OrderIntent:
    return OrderIntent(
        intent_id="intent-1",
        idempotency_key="strategy-instance-1:decision-1",
        strategy_instance_id="strategy-instance-1",
        position_id_if_any=None,
        symbol="005930",
        action=action,
        quantity=2,
        limit_or_price_policy={"kind": "limit", "price": 80000},
        urgency="NORMAL",
        reason_code="ENTRY_TRIGGERED",
        created_at=NOW,
        expires_at=NOW + timedelta(seconds=5),
    )


def _verdict() -> RiskVerdict:
    return RiskVerdict(
        verdict_id="verdict-1",
        intent_id="intent-1",
        action=RiskVerdictAction.APPROVE,
        approved_quantity=2,
        limits_evaluated={"position_limit": True},
        reason_codes=(),
        account_snapshot_id="account-1",
        timestamp=NOW,
    )


def _order(side: OrderSide = OrderSide.BUY) -> FinalOrder:
    return FinalOrder(
        ticker="005930",
        market="KRX",
        order_type=OrderType.LIMIT,
        side=side,
        quantity=2,
        limit_price=80000,
        manual_approval_required=False,
    )


def _coordinator(tmp_path: Path, broker: _Broker) -> LiveExecutionCoordinator:
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
    return coordinator


def test_strategy_owned_path_is_disabled_by_default(tmp_path) -> None:
    coordinator = _coordinator(tmp_path, _Broker())
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(LiveExecutionBlocked, match="STRATEGY_OWNED_EXECUTION_DISABLED"):
            coordinator.submit_approved_intent(_intent(), _verdict(), _order())


def test_strategy_owned_path_persists_causal_chain_before_broker_link(tmp_path) -> None:
    broker = _Broker()
    coordinator = _coordinator(tmp_path, broker)
    with patch.dict(
        "os.environ",
        {"REFACTOR_STRATEGY_OWNED_EXECUTION": "true"},
        clear=True,
    ):
        first = coordinator.submit_approved_intent(_intent(), _verdict(), _order())
        second = coordinator.submit_approved_intent(_intent(), _verdict(), _order())

    assert first.broker_order_id == "broker-1"
    assert second.message == "idempotent replay"
    assert broker.calls == 1
    rows = coordinator.causal_journal.read_all()  # type: ignore[union-attr]
    event_types = [row["event_type"] for row in rows]
    assert event_types[:3] == [
        "order_intent_persisted",
        "risk_verdict_persisted",
        "broker_submission_authorized",
    ]
    assert event_types[-1] == "broker_order_linked"


def test_execution_cannot_flip_strategy_direction(tmp_path) -> None:
    broker = _Broker()
    coordinator = _coordinator(tmp_path, broker)
    with patch.dict(
        "os.environ",
        {"REFACTOR_STRATEGY_OWNED_EXECUTION": "true"},
        clear=True,
    ):
        with pytest.raises(LiveExecutionBlocked, match="SIDE_DIFFERS_FROM_INTENT"):
            coordinator.submit_approved_intent(_intent(), _verdict(), _order(OrderSide.SELL))
    assert broker.calls == 0
    assert not (tmp_path / "causal.jsonl").exists()

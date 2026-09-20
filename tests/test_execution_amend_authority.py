"""Amendments consume the original order's residual and frozen grant."""
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.execution.execution_guard import ExecutionGuard
from app.execution.idempotency_store import IdempotencyStore
from app.execution.kis_errors import LiveExecutionBlocked
from app.execution.live_execution_coordinator import LiveExecutionCoordinator
from app.execution.order_state_machine import OrderState, OrderStateMachine
from app.execution.pre_submit_guard import PreSubmitGuard
from app.schemas.domain import FinalOrder, OrderSide, OrderType
from app.storage.trading_state_store import TradingStateStore
from test_ontology_decision_receipt import NOW, _approved_plan


def _coordinator(tmp_path, monkeypatch, *, side=OrderSide.BUY):
    plan = _approved_plan(tmp_path)
    order = FinalOrder(ticker=plan.symbol, market=plan.market, side=side,
                       order_type=OrderType.LIMIT, quantity=plan.quantity, limit_price=1000,
                       position_effect="CLOSE" if side == OrderSide.SELL else "OPEN")
    broker = SimpleNamespace(
        place_limit_order=Mock(return_value=SimpleNamespace(order_id="original", status="ACCEPTED")),
        amend_limit_order=Mock(return_value=SimpleNamespace(order_id="amended", status="ACCEPTED")),
        cancel_order=Mock(return_value=SimpleNamespace(order_id="canceled", status="CANCELED")),
    )
    machine = OrderStateMachine(TradingStateStore(tmp_path / "orders.sqlite3"))
    pre = PreSubmitGuard(
        state_machine=machine, strict=True,
        freshness=SimpleNamespace(blocking_reasons=lambda **_: ()),
        store=SimpleNamespace(fetch_one=lambda _: {"captured_at": NOW.isoformat(), "reconciled": 1}),
        session_service=SimpleNamespace(
            new_entry_allowed=lambda *args: True,
            primary_capability=lambda *args: SimpleNamespace(session=SimpleNamespace(value="OPEN")),
        ),
    )
    coordinator = LiveExecutionCoordinator(
        broker, idempotency_store=IdempotencyStore(tmp_path / "idempotency.jsonl"),
        journal=SimpleNamespace(record=Mock()),
        execution_config=SimpleNamespace(idempotency_ttl_seconds=60,
                                         poll_order_status_interval_seconds=1,
                                         max_order_status_poll_seconds=0),
        execution_guard=ExecutionGuard(pre_submit_guard=pre, strict_affordability=True,
                                        kill_switch_provider=lambda: False),
        plan_provider=lambda _: plan, orderable_cash_provider=lambda _: 1e9,
        sellable_quantity_provider=lambda _: order.quantity,
    )
    monkeypatch.setattr(coordinator, "_preflight_failures", lambda: [])
    monkeypatch.setattr("app.execution.execution_guard._utcnow", lambda: NOW)
    coordinator.submit_final_order(order, idempotency_key="original-key")
    return coordinator, broker, machine, plan, order


def _pending(machine, order, broker_id="original", *, unknown=False):
    record = machine.create(ticker=order.ticker, side=order.side.value, quantity=order.quantity,
                            idempotency_key=broker_id, limit_price=order.limit_price, now=NOW)
    for state in (OrderState.GATED, OrderState.SUBMITTING, OrderState.SUBMITTED):
        record = machine.transition(record.intent_id, state, broker_order_id=broker_id, now=NOW)
    if unknown:
        record = machine.transition(record.intent_id, OrderState.UNKNOWN, now=NOW)
    return record


def _status(coordinator, order, *, filled=0, status="OPEN", broker_id="original"):
    snapshot = SimpleNamespace(status=status, raw=SimpleNamespace(
        quantity=filled, order_id=broker_id, ticker=order.ticker, side=order.side,
    ))
    coordinator.status_tracker = SimpleNamespace(poll=lambda *args, **kwargs: snapshot)
    coordinator.poll_status(broker_id)


@pytest.mark.parametrize("persisted", [False, True])
def test_exact_origin_can_be_amended_with_reserved_cash_and_no_duplicate(tmp_path, monkeypatch, persisted):
    coordinator, broker, machine, plan, order = _coordinator(tmp_path, monkeypatch)
    if persisted:
        _pending(machine, order)
    coordinator.orderable_cash_provider = lambda _: 0.0
    result = coordinator.amend_final_order("original", replace(order, limit_price=995))
    assert result.broker_order_id == "amended"
    broker.amend_limit_order.assert_called_once()
    assert coordinator._last_guard_decision.detail["amendment_remaining_quantity"] == plan.quantity
    assert "original" not in coordinator._accepted_orders
    assert "amended" in coordinator._accepted_orders
    event = next(call.args for call in coordinator.journal.record.call_args_list if call.args[0] == "live_order_amended")
    assert event[1]["previous_broker_order_id"] == "original"


@pytest.mark.parametrize("mode,reason", [
    ("outside_band", "ONTOLOGY_AUTHORITY_PRICE_OUTSIDE_APPROVAL"),
    ("wrong_id", "GUARD_AMEND_ORIGIN_UNKNOWN"),
    ("other_pending", "DUPLICATE_ORDER_RISK"),
    ("unknown_origin", "UNKNOWN_ORDER_STATE"),
    ("unknown_other", "UNKNOWN_ORDER_STATE"),
    ("changed_contract", "GUARD_AMEND_CONTRACT_MISMATCH"),
])
def test_amendment_never_escapes_bounds_origin_or_unknown_order_checks(tmp_path, monkeypatch, mode, reason):
    coordinator, broker, machine, _, order = _coordinator(tmp_path, monkeypatch)
    _pending(machine, order, unknown=mode == "unknown_origin")
    replacement = replace(order, limit_price=1001) if mode == "outside_band" else order
    if mode == "other_pending":
        _pending(machine, order, "another")
    if mode == "unknown_other":
        _pending(machine, replace(order, side=OrderSide.SELL), "another", unknown=True)
    if mode == "changed_contract":
        replacement = replace(order, side=OrderSide.SELL, position_effect="CLOSE")
    with pytest.raises(LiveExecutionBlocked) as caught:
        coordinator.amend_final_order("wrong" if mode == "wrong_id" else "original", replacement)
    assert reason in caught.value.reason_codes
    broker.amend_limit_order.assert_not_called()


def test_partial_fill_credits_only_confirmed_unfilled_quantity(tmp_path, monkeypatch):
    coordinator, broker, _, plan, order = _coordinator(tmp_path, monkeypatch)
    _status(coordinator, order, filled=10, status="PARTIALLY_FILLED")
    coordinator.plan_provider = lambda _: plan.with_entry_fill(1000, 10)
    coordinator.orderable_cash_provider = lambda _: 0.0
    with pytest.raises(LiveExecutionBlocked) as caught:
        coordinator.amend_final_order("original", order)
    assert "GUARD_AMEND_QUANTITY_EXCEEDS_REMAINING" in caught.value.reason_codes
    residual = replace(order, quantity=order.quantity - 10, limit_price=995)
    coordinator.amend_final_order("original", residual)
    broker.amend_limit_order.assert_called_once_with("original", residual)


def test_real_exit_can_amend_reserved_sell_quantity_after_entry_expiry(tmp_path, monkeypatch):
    coordinator, broker, machine, plan, order = _coordinator(tmp_path, monkeypatch, side=OrderSide.SELL)
    _pending(machine, order)
    coordinator.sellable_quantity_provider = lambda _: 0
    monkeypatch.setattr("app.execution.execution_guard._utcnow", lambda: plan.expires_at + timedelta(hours=1))
    coordinator.amend_final_order("original", replace(order, limit_price=800))
    broker.amend_limit_order.assert_called_once()


@pytest.mark.parametrize("status,filled", [("UNKNOWN", 0), ("FILLED", 100), ("CANCELED", 0)])
def test_uncertain_or_terminal_status_removes_amendable_origin(tmp_path, monkeypatch, status, filled):
    coordinator, broker, _, _, order = _coordinator(tmp_path, monkeypatch)
    _status(coordinator, order, status=status, filled=filled)
    with pytest.raises(LiveExecutionBlocked) as caught:
        coordinator.amend_final_order("original", order)
    assert "GUARD_AMEND_ORIGIN_UNKNOWN" in caught.value.reason_codes
    broker.amend_limit_order.assert_not_called()


def test_cancel_still_works_without_origin_or_valid_entry_receipt(tmp_path, monkeypatch):
    coordinator, broker, machine, plan, order = _coordinator(tmp_path, monkeypatch)
    _pending(machine, order, unknown=True)
    coordinator._accepted_orders.clear()
    monkeypatch.setattr("app.execution.execution_guard._utcnow", lambda: plan.expires_at + timedelta(hours=1))
    coordinator.cancel_final_order("original", order)
    broker.cancel_order.assert_called_once_with("original", order)
    broker.amend_limit_order.assert_not_called()


def test_durable_replay_does_not_invent_an_unfilled_residual_after_restart(tmp_path, monkeypatch):
    coordinator, broker, _, _, order = _coordinator(tmp_path, monkeypatch)
    coordinator._accepted_orders.clear()
    coordinator.submit_final_order(order, idempotency_key="original-key")
    broker.place_limit_order.assert_called_once()
    with pytest.raises(LiveExecutionBlocked) as caught:
        coordinator.amend_final_order("original", order)
    assert "GUARD_AMEND_ORIGIN_UNKNOWN" in caught.value.reason_codes
    _status(coordinator, order)
    coordinator.amend_final_order("original", order)
    broker.amend_limit_order.assert_called_once()


def test_amend_timeout_cannot_reuse_old_reservation_but_cancel_remains_possible(tmp_path, monkeypatch):
    coordinator, broker, _, _, order = _coordinator(tmp_path, monkeypatch)
    broker.amend_limit_order.side_effect = TimeoutError("unknown broker result")
    with pytest.raises(TimeoutError):
        coordinator.amend_final_order("original", order)
    with pytest.raises(LiveExecutionBlocked) as caught:
        coordinator.amend_final_order("original", order)
    assert "GUARD_AMEND_ORIGIN_UNKNOWN" in caught.value.reason_codes
    coordinator.cancel_final_order("original", order)
    broker.amend_limit_order.assert_called_once()
    broker.cancel_order.assert_called_once()


def test_acknowledged_amendment_ancestry_is_the_same_pending_order(tmp_path, monkeypatch):
    coordinator, broker, machine, _, order = _coordinator(tmp_path, monkeypatch)
    _pending(machine, order)
    broker.amend_limit_order.side_effect = [
        SimpleNamespace(order_id="amended", status="ACCEPTED"),
        SimpleNamespace(order_id="amended-again", status="ACCEPTED"),
    ]
    coordinator.amend_final_order("original", replace(order, limit_price=998))
    # The SQLite row still carries the old ID; both acknowledgements identify
    # one order, while a genuinely separate pending order remains a blocker.
    coordinator.amend_final_order("amended", replace(order, limit_price=995))
    assert broker.amend_limit_order.call_count == 2
    _pending(machine, order, "unrelated")
    with pytest.raises(LiveExecutionBlocked) as caught:
        coordinator.amend_final_order("amended-again", replace(order, limit_price=994))
    assert "DUPLICATE_ORDER_RISK" in caught.value.reason_codes
    assert broker.amend_limit_order.call_count == 2


def test_residual_cannot_grow_when_broker_status_regresses(tmp_path, monkeypatch):
    coordinator, broker, _, _, order = _coordinator(tmp_path, monkeypatch)
    _status(coordinator, order, filled=10, status="PARTIALLY_FILLED")
    _status(coordinator, order, filled=0, status="OPEN")
    with pytest.raises(LiveExecutionBlocked) as caught:
        coordinator.amend_final_order("original", replace(order, quantity=10))
    assert "GUARD_AMEND_ORIGIN_UNKNOWN" in caught.value.reason_codes
    broker.amend_limit_order.assert_not_called()


def test_same_id_amend_keeps_cumulative_fill_baseline_without_double_subtraction(tmp_path, monkeypatch):
    coordinator, broker, _, plan, order = _coordinator(tmp_path, monkeypatch)
    broker.amend_limit_order.return_value = SimpleNamespace(order_id="original", status="ACCEPTED")
    _status(coordinator, order, filled=10, status="PARTIALLY_FILLED")
    coordinator.plan_provider = lambda _: plan.with_entry_fill(1000, 10)
    coordinator.orderable_cash_provider = lambda _: 0.0
    coordinator.amend_final_order("original", replace(order, quantity=order.quantity - 10, limit_price=995))
    _status(coordinator, order, filled=10, status="PARTIALLY_FILLED")
    assert coordinator._accepted_orders["original"].remaining_quantity == order.quantity - 10
    _status(coordinator, order, filled=20, status="PARTIALLY_FILLED")
    assert coordinator._accepted_orders["original"].remaining_quantity == order.quantity - 20
    coordinator.plan_provider = lambda _: plan.with_entry_fill(1000, 20)
    coordinator.amend_final_order("original", replace(order, quantity=order.quantity - 20, limit_price=994))
    assert coordinator._last_guard_decision.detail["amendment_remaining_quantity"] == order.quantity - 20
    assert broker.amend_limit_order.call_count == 2


def test_new_id_amend_starts_a_new_fill_counter(tmp_path, monkeypatch):
    coordinator, broker, _, plan, order = _coordinator(tmp_path, monkeypatch)
    _status(coordinator, order, filled=10, status="PARTIALLY_FILLED")
    coordinator.plan_provider = lambda _: plan.with_entry_fill(1000, 10)
    coordinator.amend_final_order("original", replace(order, quantity=order.quantity - 10, limit_price=995))
    _status(coordinator, order, filled=5, status="PARTIALLY_FILLED", broker_id="amended")
    assert coordinator._accepted_orders["amended"].remaining_quantity == order.quantity - 15
    coordinator.plan_provider = lambda _: plan.with_entry_fill(1000, 15)
    coordinator.amend_final_order("amended", replace(order, quantity=order.quantity - 15, limit_price=994))
    assert coordinator._last_guard_decision.detail["amendment_remaining_quantity"] == order.quantity - 15


@pytest.mark.parametrize("side", [OrderSide.BUY, OrderSide.SELL])
def test_unknown_status_keeps_identity_for_later_confirmed_residual_recovery(tmp_path, monkeypatch, side):
    coordinator, broker, _, plan, order = _coordinator(tmp_path, monkeypatch, side=side)
    _status(coordinator, order, filled=10, status="PARTIALLY_FILLED")
    _status(coordinator, order, status="UNKNOWN")
    assert coordinator._accepted_orders["original"].remaining_quantity is None
    replacement = replace(order, quantity=order.quantity - 20, limit_price=995)
    with pytest.raises(LiveExecutionBlocked):
        coordinator.amend_final_order("original", replacement)
    broker.amend_limit_order.assert_not_called()
    _status(coordinator, order, filled=20, status="PARTIALLY_FILLED")
    assert coordinator._accepted_orders["original"].remaining_quantity == order.quantity - 20
    coordinator.plan_provider = lambda _: plan.with_entry_fill(1000, 20) if side == OrderSide.BUY else plan
    coordinator.amend_final_order("original", replacement)
    broker.amend_limit_order.assert_called_once_with("original", replacement)

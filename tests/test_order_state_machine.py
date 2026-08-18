"""Order state machine and reconciliation: duplicates, partial fills, restart, unknowns."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.execution.order_state_machine import (
    TERMINAL_STATES,
    OrderState,
    OrderStateError,
    OrderStateMachine,
    allowed_transitions,
)
from app.execution.reconciliation import (
    AccountReconciler,
    AccountView,
    BrokerOrderStatus,
    BrokerPosition,
    OrderReconciler,
    PositionReconciler,
)
from app.storage.trading_state_store import TradingStateStore

NOW = datetime(2026, 8, 19, 1, 30, tzinfo=timezone.utc)


@pytest.fixture()
def machine(tmp_path) -> OrderStateMachine:
    return OrderStateMachine(TradingStateStore(tmp_path / "state.sqlite3"))


def _intent(machine: OrderStateMachine, *, key: str = "k1", quantity: int = 100, **kwargs):
    return machine.create(
        ticker="005930",
        side="BUY",
        quantity=quantity,
        idempotency_key=key,
        limit_price=70_000.0,
        market_group="KR",
        venue="KRX",
        now=NOW,
        **kwargs,
    )


def _submit(machine: OrderStateMachine, record, *, broker_order_id="B-1"):
    machine.transition(record.intent_id, OrderState.GATED, now=NOW)
    machine.transition(record.intent_id, OrderState.SUBMITTING, now=NOW)
    return machine.transition(
        record.intent_id,
        OrderState.SUBMITTED,
        broker_order_id=broker_order_id,
        now=NOW,
    )


# --------------------------------------------------------------------------- #
# Creation and idempotency
# --------------------------------------------------------------------------- #
def test_create_is_idempotent_on_the_key(machine: OrderStateMachine) -> None:
    first = _intent(machine)
    second = _intent(machine)
    assert first.intent_id == second.intent_id
    assert machine.store.count("order_intent") == 1


def test_zero_quantity_is_refused(machine: OrderStateMachine) -> None:
    with pytest.raises(OrderStateError):
        machine.create(
            ticker="005930", side="BUY", quantity=0, idempotency_key="z", now=NOW
        )


def test_creation_records_an_event(machine: OrderStateMachine) -> None:
    record = _intent(machine)
    events = machine.events(record.intent_id)
    assert [event["to_state"] for event in events] == ["CREATED"]


# --------------------------------------------------------------------------- #
# Transitions
# --------------------------------------------------------------------------- #
def test_illegal_transition_is_refused(machine: OrderStateMachine) -> None:
    record = _intent(machine)
    with pytest.raises(OrderStateError):
        machine.transition(record.intent_id, OrderState.FILLED, now=NOW)


def test_terminal_states_accept_nothing_further(machine: OrderStateMachine) -> None:
    for state in TERMINAL_STATES:
        assert allowed_transitions(state) == frozenset()


def test_filled_requires_the_full_quantity(machine: OrderStateMachine) -> None:
    record = _submit(machine, _intent(machine))
    with pytest.raises(OrderStateError):
        machine.transition(
            record.intent_id, OrderState.FILLED, filled_quantity=40, now=NOW
        )


def test_filled_quantity_may_not_decrease(machine: OrderStateMachine) -> None:
    record = _submit(machine, _intent(machine))
    machine.transition(
        record.intent_id, OrderState.PARTIALLY_FILLED, filled_quantity=60, now=NOW
    )
    with pytest.raises(OrderStateError):
        machine.transition(
            record.intent_id, OrderState.PARTIALLY_FILLED, filled_quantity=30, now=NOW
        )


def test_overfill_is_refused(machine: OrderStateMachine) -> None:
    record = _submit(machine, _intent(machine))
    with pytest.raises(OrderStateError):
        machine.transition(
            record.intent_id, OrderState.PARTIALLY_FILLED, filled_quantity=500, now=NOW
        )


def test_partial_fills_average_by_quantity(machine: OrderStateMachine) -> None:
    record = _submit(machine, _intent(machine, quantity=100))
    machine.transition(
        record.intent_id,
        OrderState.PARTIALLY_FILLED,
        filled_quantity=40,
        fill_price=70_000.0,
        now=NOW,
    )
    final = machine.transition(
        record.intent_id,
        OrderState.FILLED,
        filled_quantity=100,
        fill_price=71_000.0,
        now=NOW,
    )
    assert final.state is OrderState.FILLED
    assert final.remaining_quantity == 0
    # 40 @ 70,000 then 60 @ 71,000 -> 70,600, not the last print.
    assert final.average_fill_price == pytest.approx(70_600.0)


def test_every_transition_is_appended_to_the_event_log(machine: OrderStateMachine) -> None:
    record = _submit(machine, _intent(machine))
    machine.transition(
        record.intent_id, OrderState.PARTIALLY_FILLED, filled_quantity=10, now=NOW
    )
    machine.transition(record.intent_id, OrderState.CANCELLING, now=NOW)
    machine.transition(record.intent_id, OrderState.CANCELLED, now=NOW)
    states = [event["to_state"] for event in machine.events(record.intent_id)]
    assert states == [
        "CREATED",
        "GATED",
        "SUBMITTING",
        "SUBMITTED",
        "PARTIALLY_FILLED",
        "CANCELLING",
        "CANCELLED",
    ]


def test_cancel_can_lose_the_race_to_a_fill(machine: OrderStateMachine) -> None:
    record = _submit(machine, _intent(machine))
    machine.transition(record.intent_id, OrderState.CANCELLING, now=NOW)
    filled = machine.transition(
        record.intent_id, OrderState.FILLED, filled_quantity=100, now=NOW
    )
    assert filled.state is OrderState.FILLED


# --------------------------------------------------------------------------- #
# Duplicate prevention
# --------------------------------------------------------------------------- #
def test_live_order_creates_duplicate_risk(machine: OrderStateMachine) -> None:
    assert not machine.has_duplicate_risk("005930", "BUY")
    _submit(machine, _intent(machine))
    assert machine.has_duplicate_risk("005930", "BUY")
    assert not machine.has_duplicate_risk("005930", "SELL")


def test_unknown_state_still_counts_as_duplicate_risk(machine: OrderStateMachine) -> None:
    record = _submit(machine, _intent(machine))
    machine.transition(record.intent_id, OrderState.UNKNOWN, now=NOW)
    assert machine.has_duplicate_risk("005930", "BUY")


def test_terminal_order_clears_duplicate_risk(machine: OrderStateMachine) -> None:
    record = _submit(machine, _intent(machine))
    machine.transition(record.intent_id, OrderState.CANCELLED, now=NOW)
    assert not machine.has_duplicate_risk("005930", "BUY")


# --------------------------------------------------------------------------- #
# Restart recovery
# --------------------------------------------------------------------------- #
def test_recovery_separates_submitted_from_never_submitted(tmp_path) -> None:
    store = TradingStateStore(tmp_path / "state.sqlite3")
    first = OrderStateMachine(store)
    submitted = _submit(first, _intent(first, key="live"))
    _intent(first, key="stillborn")

    # A fresh process over the same file.
    second = OrderStateMachine(TradingStateStore(tmp_path / "state.sqlite3"))
    recovery = second.recover(now=NOW)
    assert recovery["open_count"] == 2
    assert [item["intent_id"] for item in recovery["needs_broker_query"]] == [
        submitted.intent_id
    ]
    assert len(recovery["never_submitted"]) == 1
    assert recovery["blocked_tickers"] == ["005930"]


def test_never_submitted_intents_can_be_expired_locally(machine: OrderStateMachine) -> None:
    record = _intent(machine, key="stillborn")
    machine.transition(record.intent_id, OrderState.GATED, now=NOW)
    expired = machine.expire_never_submitted(machine.open_intents(), now=NOW)
    assert expired == (record.intent_id,)
    assert machine.get(record.intent_id).state is OrderState.EXPIRED  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# Order reconciliation
# --------------------------------------------------------------------------- #
class _Broker:
    def __init__(self, answers: dict[str, object]) -> None:
        self.answers = answers
        self.calls: list[str] = []

    def get_order_status(self, broker_order_id: str):
        self.calls.append(broker_order_id)
        answer = self.answers[broker_order_id]
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_reconciliation_resolves_a_filled_order(machine: OrderStateMachine) -> None:
    record = _submit(machine, _intent(machine))
    broker = _Broker(
        {"B-1": BrokerOrderStatus("B-1", "FILLED", filled_quantity=100, average_price=70_500.0)}
    )
    result = OrderReconciler(machine, broker).reconcile(now=NOW)
    assert result.reconciled
    assert machine.get(record.intent_id).state is OrderState.FILLED  # type: ignore[union-attr]


def test_reconciliation_reads_a_partial_fill_as_partial(machine: OrderStateMachine) -> None:
    record = _submit(machine, _intent(machine))
    broker = _Broker({"B-1": BrokerOrderStatus("B-1", "OPEN", filled_quantity=30)})
    OrderReconciler(machine, broker).reconcile(now=NOW)
    updated = machine.get(record.intent_id)
    assert updated is not None
    assert updated.state is OrderState.PARTIALLY_FILLED
    assert updated.filled_quantity == 30


def test_broker_saying_filled_with_a_partial_count_is_treated_as_partial(
    machine: OrderStateMachine,
) -> None:
    record = _submit(machine, _intent(machine, quantity=100))
    broker = _Broker({"B-1": BrokerOrderStatus("B-1", "FILLED", filled_quantity=40)})
    OrderReconciler(machine, broker).reconcile(now=NOW)
    updated = machine.get(record.intent_id)
    assert updated is not None
    assert updated.state is OrderState.PARTIALLY_FILLED
    assert updated.filled_quantity == 40


def test_broker_failure_becomes_unknown_not_clean(machine: OrderStateMachine) -> None:
    record = _submit(machine, _intent(machine))
    broker = _Broker({"B-1": RuntimeError("timeout")})
    result = OrderReconciler(machine, broker).reconcile(now=NOW)
    assert not result.reconciled
    assert "UNKNOWN_ORDER_STATE" in result.reason_codes
    assert machine.get(record.intent_id).state is OrderState.UNKNOWN  # type: ignore[union-attr]


def test_unmapped_broker_status_is_unknown_rather_than_guessed(
    machine: OrderStateMachine,
) -> None:
    record = _submit(machine, _intent(machine))
    broker = _Broker({"B-1": BrokerOrderStatus("B-1", "SOMETHING_NEW")})
    OrderReconciler(machine, broker).reconcile(now=NOW)
    assert machine.get(record.intent_id).state is OrderState.UNKNOWN  # type: ignore[union-attr]


def test_untracked_broker_order_is_a_discrepancy(machine: OrderStateMachine) -> None:
    result = OrderReconciler(machine, _Broker({})).reconcile(
        now=NOW, broker_open_orders=[{"order_id": "GHOST-1", "ticker": "000660"}]
    )
    assert not result.reconciled
    assert result.discrepancies[0].kind == "UNTRACKED_BROKER_ORDER"


def test_no_broker_client_with_open_orders_is_not_reconciled(
    machine: OrderStateMachine,
) -> None:
    _submit(machine, _intent(machine))
    result = OrderReconciler(machine, None).reconcile(now=NOW)
    assert not result.reconciled
    assert "NO_BROKER_CLIENT" in result.reason_codes


# --------------------------------------------------------------------------- #
# Position reconciliation
# --------------------------------------------------------------------------- #
def test_positions_match_the_recorded_fills(machine: OrderStateMachine) -> None:
    record = _submit(machine, _intent(machine, quantity=100))
    machine.transition(
        record.intent_id, OrderState.FILLED, filled_quantity=100, fill_price=70_000.0, now=NOW
    )
    result = PositionReconciler(machine).reconcile(
        [BrokerPosition("005930", 100.0, 70_000.0)], now=NOW
    )
    assert result.reconciled
    assert machine.store.count("position_snapshot") == 1


def test_untracked_broker_position_is_flagged(machine: OrderStateMachine) -> None:
    result = PositionReconciler(machine).reconcile(
        [BrokerPosition("000660", 50.0)], now=NOW
    )
    assert not result.reconciled
    assert result.discrepancies[0].kind == "UNTRACKED_BROKER_POSITION"


def test_phantom_local_position_is_flagged(machine: OrderStateMachine) -> None:
    record = _submit(machine, _intent(machine, quantity=100))
    machine.transition(
        record.intent_id, OrderState.FILLED, filled_quantity=100, fill_price=70_000.0, now=NOW
    )
    result = PositionReconciler(machine).reconcile([], now=NOW)
    assert not result.reconciled
    assert result.discrepancies[0].kind == "PHANTOM_LOCAL_POSITION"


def test_sells_net_against_buys(machine: OrderStateMachine) -> None:
    buy = _submit(machine, _intent(machine, key="b", quantity=100), broker_order_id="B-1")
    machine.transition(buy.intent_id, OrderState.FILLED, filled_quantity=100, now=NOW)
    sell = machine.create(
        ticker="005930", side="SELL", quantity=40, idempotency_key="s", now=NOW
    )
    _submit(machine, sell, broker_order_id="B-2")
    machine.transition(sell.intent_id, OrderState.FILLED, filled_quantity=40, now=NOW)
    assert PositionReconciler(machine).expected_from_fills() == {"005930": 60.0}


def test_failed_position_query_is_not_reconciled(machine: OrderStateMachine) -> None:
    result = PositionReconciler(machine).reconcile(None, now=NOW)
    assert not result.reconciled
    assert "POSITION_QUERY_FAILED" in result.reason_codes


# --------------------------------------------------------------------------- #
# Account reconciliation
# --------------------------------------------------------------------------- #
def test_matching_equity_reconciles(tmp_path) -> None:
    store = TradingStateStore(tmp_path / "state.sqlite3")
    result = AccountReconciler(store).reconcile(
        AccountView(equity=100_000_000.0, cash=20_000_000.0, currency="KRW", observed_at=NOW),
        local_equity=100_000_000.0,
        local_cash=20_000_000.0,
        now=NOW,
    )
    assert result.reconciled
    assert store.count("account_snapshot") == 1
    row = store.fetch_one("select * from account_snapshot")
    assert row is not None and row["reconciled"] == 1


def test_equity_mismatch_fails_reconciliation(tmp_path) -> None:
    result = AccountReconciler(TradingStateStore(tmp_path / "s.sqlite3")).reconcile(
        AccountView(equity=100_000_000.0, cash=None, observed_at=NOW),
        local_equity=90_000_000.0,
        now=NOW,
    )
    assert not result.reconciled
    assert "EQUITY_MISMATCH" in result.reason_codes


def test_stale_account_snapshot_fails_reconciliation(tmp_path) -> None:
    result = AccountReconciler(TradingStateStore(tmp_path / "s.sqlite3")).reconcile(
        AccountView(equity=1.0, cash=1.0, observed_at=NOW - timedelta(hours=3)),
        local_equity=1.0,
        max_age_seconds=600.0,
        now=NOW,
    )
    assert not result.reconciled
    assert "ACCOUNT_SNAPSHOT_STALE" in result.reason_codes


def test_absent_account_view_is_not_reconciled(tmp_path) -> None:
    result = AccountReconciler(TradingStateStore(tmp_path / "s.sqlite3")).reconcile(
        None, now=NOW
    )
    assert not result.reconciled
    assert "ACCOUNT_QUERY_FAILED" in result.reason_codes

"""The last-line guard: what it re-checks, and what it refuses to block."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.data.freshness import DataFreshnessRegistry
from app.execution.order_state_machine import OrderState, OrderStateMachine
from app.execution.pre_submit_guard import PreSubmitGuard
from app.storage.trading_state_store import TradingStateStore, iso_column

NOW = datetime(2026, 8, 19, 1, 30, tzinfo=timezone.utc)  # 10:30 KST, KRX continuous


class _Session:
    """A session service stub. Its answer is the guard's only session input."""

    def __init__(self, *, allowed: bool) -> None:
        self.allowed = allowed

    def new_entry_allowed(self, group, now) -> bool:  # noqa: ANN001
        return self.allowed

    def new_entry_block_reasons(self, group, now):  # noqa: ANN001
        return () if self.allowed else ("SESSION_CLOSED",)

    def primary_capability(self, group, now):  # noqa: ANN001
        class _Capability:
            session = type("S", (), {"value": "KRX_REGULAR" if self.allowed else "KR_CLOSED"})()

        return _Capability()


@pytest.fixture()
def store(tmp_path) -> TradingStateStore:
    return TradingStateStore(tmp_path / "state.sqlite3")


@pytest.fixture()
def machine(store: TradingStateStore) -> OrderStateMachine:
    return OrderStateMachine(store)


@pytest.fixture()
def registry() -> DataFreshnessRegistry:
    fresh = DataFreshnessRegistry()
    for source, data_type in (
        ("kis_realtime", "trade"),
        ("kis_realtime", "orderbook"),
        ("kis_rest", "account"),
        ("kis_rest", "positions"),
        ("kis_rest", "order_status"),
        ("internal", "domestic_context"),
    ):
        fresh.record_event(source, data_type, NOW, received_time=NOW, processed_time=NOW, now=NOW)
    return fresh


def _reconciled(store: TradingStateStore, *, at: datetime = NOW, ok: bool = True) -> None:
    with store.transaction() as conn:
        conn.execute(
            "insert into account_snapshot"
            " (snapshot_id, captured_at, source, equity, cash, currency, reconciled,"
            "  discrepancies_json, payload_json) values (?, ?, ?, ?, ?, ?, ?, '[]', '{}')",
            ("as-1", iso_column(at), "broker", 1e8, 3e7, "KRW", 1 if ok else 0),
        )


def _guard(store, machine, registry, *, allowed: bool = True, strict: bool = True) -> PreSubmitGuard:
    return PreSubmitGuard(
        state_machine=machine,
        freshness=registry,
        store=store,
        strict=strict,
        session_service=_Session(allowed=allowed),
    )


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #
def test_a_clean_buy_is_allowed(store, machine, registry) -> None:
    _reconciled(store)
    decision = _guard(store, machine, registry).evaluate(
        ticker="005930", side="BUY", market="KR", now=NOW
    )
    assert decision.allowed
    assert decision.reason_codes == ()
    assert set(decision.checked) == {
        "order_state",
        "session",
        "data_freshness",
        "account_reconciliation",
    }


# --------------------------------------------------------------------------- #
# Each check
# --------------------------------------------------------------------------- #
def test_a_closed_session_blocks_a_buy(store, machine, registry) -> None:
    _reconciled(store)
    decision = _guard(store, machine, registry, allowed=False).evaluate(
        ticker="005930", side="BUY", market="KR", now=NOW
    )
    assert not decision.allowed
    assert "SESSION_NOT_TRADEABLE" in decision.reason_codes


def test_an_unrecognised_market_blocks(store, machine, registry) -> None:
    _reconciled(store)
    decision = _guard(store, machine, registry).evaluate(
        ticker="XYZ", side="BUY", market="LSE", now=NOW
    )
    assert not decision.allowed
    assert "UNKNOWN_SESSION" in decision.reason_codes


def test_stale_critical_data_blocks_a_buy(store, machine) -> None:
    _reconciled(store)
    empty = DataFreshnessRegistry()
    empty.expect_all()
    decision = _guard(store, machine, empty).evaluate(
        ticker="005930", side="BUY", market="KR", now=NOW
    )
    assert not decision.allowed
    assert "STALE_DATA" in decision.reason_codes
    assert decision.detail["stale_streams"]


def test_an_unreconciled_account_blocks_a_buy(store, machine, registry) -> None:
    _reconciled(store, ok=False)
    decision = _guard(store, machine, registry).evaluate(
        ticker="005930", side="BUY", market="KR", now=NOW
    )
    assert not decision.allowed
    assert "ACCOUNT_RECONCILIATION_FAIL" in decision.reason_codes


def test_an_aged_account_snapshot_blocks_a_buy(store, machine, registry) -> None:
    _reconciled(store, at=NOW - timedelta(hours=3))
    decision = _guard(store, machine, registry).evaluate(
        ticker="005930", side="BUY", market="KR", now=NOW
    )
    assert not decision.allowed
    assert "ACCOUNT_RECONCILIATION_FAIL" in decision.reason_codes


def test_no_account_snapshot_at_all_blocks_a_buy(store, machine, registry) -> None:
    decision = _guard(store, machine, registry).evaluate(
        ticker="005930", side="BUY", market="KR", now=NOW
    )
    assert not decision.allowed
    assert "ACCOUNT_RECONCILIATION_FAIL" in decision.reason_codes


def test_a_working_order_blocks_a_second_one(store, machine, registry) -> None:
    _reconciled(store)
    record = machine.create(
        ticker="005930", side="BUY", quantity=10, idempotency_key="k", now=NOW
    )
    machine.transition(record.intent_id, OrderState.GATED, now=NOW)
    machine.transition(record.intent_id, OrderState.SUBMITTING, now=NOW)
    machine.transition(record.intent_id, OrderState.SUBMITTED, broker_order_id="B", now=NOW)

    decision = _guard(store, machine, registry).evaluate(
        ticker="005930", side="BUY", market="KR", now=NOW
    )
    assert not decision.allowed
    assert "DUPLICATE_ORDER_RISK" in decision.reason_codes


def test_an_unknown_order_blocks_its_symbol(store, machine, registry) -> None:
    _reconciled(store)
    record = machine.create(
        ticker="005930", side="SELL", quantity=10, idempotency_key="k", now=NOW
    )
    machine.transition(record.intent_id, OrderState.GATED, now=NOW)
    machine.transition(record.intent_id, OrderState.SUBMITTING, now=NOW)
    machine.transition(record.intent_id, OrderState.UNKNOWN, now=NOW)

    decision = _guard(store, machine, registry).evaluate(
        ticker="005930", side="BUY", market="KR", now=NOW
    )
    assert not decision.allowed
    assert "UNKNOWN_ORDER_STATE" in decision.reason_codes


# --------------------------------------------------------------------------- #
# Exits
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("side", ["SELL", "REDUCE", "CLOSE"])
def test_an_exit_is_exempt_from_session_data_and_account(
    store, machine, side: str
) -> None:
    """The asymmetry: never unable to close a position because the feed went quiet."""
    empty = DataFreshnessRegistry()
    empty.expect_all()
    decision = _guard(store, machine, empty, allowed=False).evaluate(
        ticker="005930", side=side, market="KR", now=NOW
    )
    assert decision.allowed
    assert decision.checked == ("order_state",)


def test_an_exit_is_still_blocked_when_it_cannot_be_routed(store, machine, registry) -> None:
    record = machine.create(
        ticker="005930", side="SELL", quantity=10, idempotency_key="k", now=NOW
    )
    machine.transition(record.intent_id, OrderState.GATED, now=NOW)
    machine.transition(record.intent_id, OrderState.SUBMITTING, now=NOW)
    machine.transition(record.intent_id, OrderState.SUBMITTED, broker_order_id="B", now=NOW)

    decision = _guard(store, machine, registry).evaluate(
        ticker="005930", side="SELL", market="KR", now=NOW
    )
    assert not decision.allowed
    assert "DUPLICATE_ORDER_RISK" in decision.reason_codes


# --------------------------------------------------------------------------- #
# Absent evidence
# --------------------------------------------------------------------------- #
def test_strict_mode_refuses_when_a_source_is_missing() -> None:
    guard = PreSubmitGuard(strict=True, session_service=_Session(allowed=True))
    decision = guard.evaluate(ticker="005930", side="BUY", market="KR", now=NOW)
    assert not decision.allowed
    assert "PRESUBMIT_NO_EVIDENCE:ORDER_STATE" in decision.reason_codes
    assert "PRESUBMIT_NO_EVIDENCE:DATA_FRESHNESS" in decision.reason_codes
    assert "PRESUBMIT_NO_EVIDENCE:ACCOUNT_RECONCILIATION" in decision.reason_codes


def test_permissive_mode_records_the_absence_without_blocking() -> None:
    guard = PreSubmitGuard(strict=False, session_service=_Session(allowed=True))
    decision = guard.evaluate(ticker="005930", side="BUY", market="KR", now=NOW)
    assert decision.allowed
    assert any(
        code.startswith("PRESUBMIT_NO_EVIDENCE") for code in decision.reason_codes
    )


def test_permissive_mode_still_enforces_a_real_finding() -> None:
    """Missing evidence is forgiven; an actual closed session is not."""
    guard = PreSubmitGuard(strict=False, session_service=_Session(allowed=False))
    decision = guard.evaluate(ticker="005930", side="BUY", market="KR", now=NOW)
    assert not decision.allowed
    assert "SESSION_NOT_TRADEABLE" in decision.reason_codes


def test_a_throwing_check_blocks_rather_than_escaping(store, machine, registry) -> None:
    class _Exploding:
        def new_entry_allowed(self, group, now):  # noqa: ANN001
            raise RuntimeError("calendar unreadable")

    guard = PreSubmitGuard(
        state_machine=machine,
        freshness=registry,
        store=store,
        strict=True,
        session_service=_Exploding(),
    )
    _reconciled(store)
    decision = guard.evaluate(ticker="005930", side="BUY", market="KR", now=NOW)
    assert not decision.allowed
    assert any(
        code.startswith("PRESUBMIT_CHECK_FAILED:session") for code in decision.reason_codes
    )


# --------------------------------------------------------------------------- #
# Coordinator wiring
# --------------------------------------------------------------------------- #
def test_the_coordinator_refuses_an_order_the_guard_blocks(store, machine, registry) -> None:
    from app.execution.kis_errors import LiveExecutionBlocked
    from app.execution.live_execution_coordinator import LiveExecutionCoordinator
    from app.schemas.domain import FinalOrder, OrderSide, OrderType

    class _Broker:
        def place_limit_order(self, order):  # noqa: ANN001 - must never be reached
            raise AssertionError("the guard should have blocked this order")

    from app.execution.execution_guard import ExecutionGuard

    coordinator = LiveExecutionCoordinator(
        _Broker(),
        execution_guard=ExecutionGuard(
            pre_submit_guard=_guard(store, machine, registry, allowed=False),
            require_plan=False,
        ),
    )
    # The process-level gates (arming, kill switch, broker health) are a separate concern
    # with their own tests; disabling them here isolates the per-order guard.
    coordinator._preflight_failures = lambda: []  # type: ignore[method-assign]
    order = FinalOrder(
        ticker="005930",
        market="KR",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=1,
        limit_price=70_000,
    )
    with pytest.raises(LiveExecutionBlocked) as excinfo:
        coordinator.submit_final_order(order)
    assert "SESSION_NOT_TRADEABLE" in str(excinfo.value)

from __future__ import annotations

import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from app.execution.order_status_tracker import OrderStatusSnapshot
from app.execution.kis_real import KisDevelopersApiClient
from app.schemas.domain import AccountSnapshot, FinalOrder, OrderSide, OrderType
from app.trading.realtime_trading_engine import RealtimeTradingEngine


def test_terminal_sell_fill_tombstones_stale_holding_and_notifies_session():
    filled_at = datetime(2026, 8, 11, 15, 33, 21, tzinfo=timezone.utc)
    snapshot = OrderStatusSnapshot(
        order_id="sell-1",
        status="FILLED",
        observed_at=filled_at,
        raw=SimpleNamespace(
            ticker="AXTI",
            side=OrderSide.SELL,
            quantity=1,
            price=74.68,
            executed_at=filled_at,
        ),
    )
    coordinator = SimpleNamespace(poll_status=lambda order_id: snapshot)
    fills: list[tuple[str, float, datetime]] = []
    manager = SimpleNamespace(
        mark_exit_filled=lambda symbol, price, at: fills.append((symbol, price, at))
    )
    engine = RealtimeTradingEngine(
        decision_engine=SimpleNamespace(),
        coordinator=coordinator,
        account_provider=lambda: None,
        candidate_symbols_provider=lambda: (),
        session_open_provider=lambda: True,
        strategy_session_manager=manager,
    )
    engine._open_sell_orders["AXTI"] = {"broker_order_id": "sell-1"}

    engine._poll_submitted_order_status("sell-1", "AXTI")

    assert "AXTI" in engine._terminal_sell_fills
    assert "AXTI" not in engine._open_sell_orders
    assert fills == [("AXTI", 74.68, filled_at)]


def test_stale_open_buy_is_canceled_and_releases_global_entry_lock():
    observed_at = datetime.now(timezone.utc)
    order = FinalOrder(
        ticker="001510",
        market="KR",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=1,
        limit_price=3000.0,
        manual_approval_required=False,
    )
    snapshot = OrderStatusSnapshot(
        order_id="buy-1",
        status="OPEN",
        observed_at=observed_at,
        raw=SimpleNamespace(
            ticker="001510",
            side=OrderSide.BUY,
            quantity=0,
            price=0.0,
            executed_at=observed_at,
        ),
    )
    cancel = Mock(return_value=SimpleNamespace(status="CANCELED"))
    coordinator = SimpleNamespace(
        poll_status=lambda order_id: snapshot,
        cancel_final_order=cancel,
        execution_config=SimpleNamespace(
            cancel_stale_unfilled_orders=True,
            max_unfilled_order_age_seconds=120,
        ),
    )
    engine = RealtimeTradingEngine(
        decision_engine=SimpleNamespace(),
        coordinator=coordinator,
        account_provider=lambda: None,
        candidate_symbols_provider=lambda: (),
        session_open_provider=lambda: True,
    )
    engine._open_buy_orders["001510"] = {
        "broker_order_id": "buy-1",
        "order": order,
        "submitted_monotonic": time.monotonic() - 121,
    }

    engine._poll_submitted_order_status("buy-1", "001510", order)

    cancel.assert_called_once_with("buy-1", order)
    assert engine._open_buy_orders == {}
    assert any(
        event.get("outcome") == "stale_buy_canceled"
        for event in engine.get_status()["recent_events"]
    )


def test_any_open_buy_blocks_election_of_additional_entry_risk():
    decision_engine = SimpleNamespace(
        evaluate_buy=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("another BUY must not be evaluated")
        )
    )
    engine = RealtimeTradingEngine(
        decision_engine=decision_engine,
        coordinator=SimpleNamespace(),
        account_provider=lambda: AccountSnapshot(cash=100_000.0, holdings=()),
        candidate_symbols_provider=lambda: ("SECOND",),
        session_open_provider=lambda: True,
    )
    engine._open_buy_orders["FIRST"] = {
        "broker_order_id": "buy-1",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }

    result = engine.run_once()

    assert result["buy_disabled"] is True
    assert result["buy_disabled_reason"] == "OPEN_BUY_ORDER_PENDING"
    assert result["open_buy_order_symbols"] == ["FIRST"]

    engine._open_buy_orders.clear()
    engine.run_once()

    assert engine.get_status()["buy_disabled_reason"] is None


def test_later_cycles_reconcile_a_submit_time_open_order():
    order = FinalOrder(
        ticker="001510",
        market="KR",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=1,
        limit_price=3000.0,
        manual_approval_required=False,
    )
    engine = RealtimeTradingEngine(
        decision_engine=SimpleNamespace(),
        coordinator=SimpleNamespace(),
        account_provider=lambda: None,
        candidate_symbols_provider=lambda: (),
        session_open_provider=lambda: True,
    )
    engine._open_buy_orders["001510"] = {
        "broker_order_id": "buy-1",
        "order": order,
        "submitted_monotonic": time.monotonic(),
    }
    scheduled = Mock()
    engine._poll_submitted_order_status_async = scheduled

    engine._reconcile_open_orders_async()

    scheduled.assert_called_once_with("buy-1", "001510", order)


def test_status_poll_scheduler_throttles_duplicate_threads():
    engine = RealtimeTradingEngine(
        decision_engine=SimpleNamespace(),
        coordinator=SimpleNamespace(),
        account_provider=lambda: None,
        candidate_symbols_provider=lambda: (),
        session_open_provider=lambda: True,
    )
    engine._open_buy_orders["001510"] = {
        "broker_order_id": "buy-1",
        "order": None,
        "status_poll_inflight": True,
    }

    engine._poll_submitted_order_status_async("buy-1", "001510")

    assert engine._open_buy_orders["001510"]["status_poll_inflight"] is True


def test_domestic_zero_fill_and_zero_remaining_is_terminal_not_open():
    order = FinalOrder(
        ticker="175250",
        market="KR",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=1,
        limit_price=2625.0,
        manual_approval_required=False,
    )
    client = object.__new__(KisDevelopersApiClient)

    execution = client._execution_from_status(
        "buy-1",
        {
            "pdno": "175250",
            "ord_qty": "1",
            "tot_ccld_qty": "0",
            "rmn_qty": "0",
            "cncl_cfrm_qty": "0",
            "rjct_qty": "0",
            "ord_unpr": "2625",
        },
        order,
    )

    assert execution.status == "EXPIRED"
    assert execution.quantity == 0

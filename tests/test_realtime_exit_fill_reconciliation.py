from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from app.execution.order_status_tracker import OrderStatusSnapshot
from app.schemas.domain import OrderSide
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

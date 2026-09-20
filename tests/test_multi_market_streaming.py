from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.data.multi_market_stream import fair_market_symbols, subscription_symbols, subscription_tr_ids
from app.schemas.domain import AccountSnapshot, FinalOrder, OrderSide, OrderType
from app.trading.realtime_trading_engine import RealtimeTradingConfig, RealtimeTradingEngine


def test_fair_budget_preserves_local_rank_and_deduplicates():
    assert fair_market_symbols(("005930", "000660", "005930", "AAPL", "MSFT"), 4) == (
        "005930", "AAPL", "000660", "MSFT"
    )
    assert fair_market_symbols(("005930", "AAPL"), 1, first="US") == ("AAPL",)


def test_shared_subscription_budget_pins_holdings_and_keeps_both_markets():
    assert subscription_symbols(
        ("005930", "000660", "035420"), ("AAPL", "MSFT", "AMZN"),
        held=("035420", "AMZN"), max_subscriptions=8,
    ) == ("035420", "AMZN", "005930", "AAPL")


def test_each_market_uses_its_own_tr_protocol(monkeypatch):
    monkeypatch.setenv("KIS_REALTIME_FEED", "KRX")
    assert subscription_tr_ids("005930") == ("H0STCNT0", "H0STASP0")
    assert subscription_tr_ids("AAPL") == ("HDFSCNT0", "HDFSASP0")


def _engine(account, candidates=(), budget=4):
    return RealtimeTradingEngine(
        decision_engine=SimpleNamespace(), coordinator=SimpleNamespace(),
        account_provider=lambda: account, candidate_symbols_provider=lambda: candidates,
        session_open_provider=lambda: True, market_open_provider=lambda *_: True,
        new_entries_authorized_provider=lambda: False,
        config=RealtimeTradingConfig(max_buy_evaluations_per_cycle=budget),
    )


def test_engine_keeps_simultaneous_market_candidates_with_one_slot(monkeypatch):
    monkeypatch.setenv("REALTIME_IGNORE_SYMBOLS", "")
    engine = _engine(AccountSnapshot(cash=100_000, holdings=()), ("005930", "AAPL"), 1)
    when = datetime(2026, 9, 18, 4, tzinfo=timezone.utc)
    assert engine.run_once(when)["buy_candidate_sample"] == ["005930"]
    assert engine.run_once(when)["buy_candidate_sample"] == ["AAPL"]


def test_stricter_daily_loss_budget_wins(monkeypatch):
    monkeypatch.setenv("REALTIME_DAILY_REALIZED_LOSS_BUY_STOP_KRW", "1000")
    monkeypatch.setenv("REALTIME_DAILY_REALIZED_LOSS_BUY_STOP_RATE", "0.004")
    engine = _engine(AccountSnapshot(cash=100_000, holdings=(), realized_pnl_today=-500))
    engine.new_entries_authorized_provider = lambda: True
    summary = engine.run_once(datetime(2026, 9, 18, 4, tzinfo=timezone.utc))
    assert summary["daily_loss_budget_krw"] == 400
    assert summary["buy_disabled"] is True
    assert summary["buy_disabled_reason"].startswith("DAILY_REALIZED_LOSS_BUY_STOP")


def test_cash_account_cannot_submit_short_or_credit_entry(tmp_path):
    from app.execution.live_execution_coordinator import LiveExecutionCoordinator
    from app.execution.kis_errors import LiveExecutionBlocked
    coordinator = LiveExecutionCoordinator(SimpleNamespace(), cash_equity_only=True)
    for direction, product, side in (("SHORT", "CREDIT_BORROW", OrderSide.SELL), ("LONG", "MARGIN", OrderSide.BUY)):
        order = FinalOrder("AAPL", "NASDAQ", OrderType.LIMIT, side, 1, 100.0,
                           position_direction=direction, execution_product=product)
        with pytest.raises(LiveExecutionBlocked, match="CASH_LONG_ENTRY_ONLY"):
            coordinator.submit_final_order(order)


def test_cash_clip_cannot_send_original_oversized_order():
    from app.execution.live_execution_coordinator import LiveExecutionCoordinator
    from app.execution.execution_guard import ExecutionGuard
    coordinator = LiveExecutionCoordinator(
        SimpleNamespace(), execution_guard=ExecutionGuard(require_plan=False),
        orderable_cash_provider=lambda order: 101.0,
    )
    order = FinalOrder("AAPL", "NASDAQ", OrderType.LIMIT, OrderSide.BUY, 2, 100.0)
    assert coordinator._pre_submit_failures(order) == ["GUARD_QUANTITY_EXCEEDS_ORDERABLE"]


def test_market_risk_caps_and_bad_inputs_do_not_increase_size(monkeypatch):
    from app.risk.position_sizing import PositionSizer, SizingInputs
    monkeypatch.setenv("REALTIME_BUY_WEIGHT", "1.0")
    monkeypatch.setenv("REALTIME_SMALL_ACCOUNT_MAX_POSITION_WEIGHT", "1.25")
    sizer = PositionSizer()
    kr = sizer.size(SizingInputs(0.02, 0.01, confidence_score=1.0, market="KRX"))
    us = sizer.size(SizingInputs(0.02, 0.01, confidence_score=1.0, market="NASDAQ"))
    assert kr.position_weight <= 0.15
    assert us.position_weight <= 0.12
    assert sizer.size(SizingInputs(float("nan"), 0.01)).position_weight == 0.0
    assert sizer.size(SizingInputs(0.02, 0.01, p_win=0.3)).position_weight == 0.0

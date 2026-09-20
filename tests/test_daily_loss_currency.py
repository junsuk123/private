from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.schemas.domain import AccountSnapshot, Holding
from app.trading.realtime_trading_engine import RealtimeTradingEngine, _daily_loss_budget
from app.trading.strategy_supervisor import StrategySupervisor


def _caps(monkeypatch, absolute="1000", rate="0.01"):
    monkeypatch.setenv("REALTIME_DAILY_REALIZED_LOSS_BUY_STOP_KRW", absolute)
    monkeypatch.setenv("REALTIME_DAILY_REALIZED_LOSS_BUY_STOP_RATE", rate)
    monkeypatch.setenv("REALTIME_BUY_ENABLED", "true")


def _supervise(account):
    engine = RealtimeTradingEngine.__new__(RealtimeTradingEngine)
    engine.strategy_session_manager = SimpleNamespace(
        snapshot=lambda: {"selected_symbol": "AAPL", "phase": "ARMED"}, request_halt=Mock())
    engine.strategy_supervisor = StrategySupervisor()
    engine.decision_engine = SimpleNamespace(store=None)
    engine.ontology_policy_resolver = None
    engine.ontology_policy_snapshot_provider = None
    engine.market_open_provider = lambda *_: True
    engine._record = Mock()
    return engine._supervise_session(account, None, datetime.now(timezone.utc))


def test_cycle_and_supervisor_share_tighter_daily_loss_limit(monkeypatch):
    _caps(monkeypatch)
    account = AccountSnapshot(cash=1_000_000, holdings=(), realized_pnl_today=-1500)
    engine = RealtimeTradingEngine(
        decision_engine=SimpleNamespace(store=None, evaluate_buy=Mock()), coordinator=SimpleNamespace(),
        account_provider=lambda: account, candidate_symbols_provider=lambda: (), session_open_provider=lambda: True)
    summary = engine.run_once()
    assert summary["buy_disabled"]
    assert summary["daily_loss_budget_krw"] == 1000
    assert summary["daily_loss_budget_remaining_krw"] == 0
    verdict = _supervise(account)
    assert verdict.forces_exit
    assert "DAILY_LOSS_LIMIT_BREACHED" in verdict.hard_reason_codes


def test_mixed_market_holdings_use_measured_krw_equity(monkeypatch):
    _caps(monkeypatch, absolute="20000", rate="0.01")
    holding = Holding("AAPL", "NASDAQ", "Apple", "Technology", 10, 100, 100)
    account = AccountSnapshot(cash=100_000, holdings=(holding,), realized_pnl_today=-5000,
        fx_rate_by_currency={"USD": 1400})
    budget = _daily_loss_budget(account)
    assert budget.threshold_krw == 15000
    assert not budget.blocked
    assert not _supervise(account).blocks_new_entries


@pytest.mark.parametrize("explicit_equity", [None, 1_400_000.0])
def test_usd_realized_loss_and_krw_ceiling_share_one_denomination(monkeypatch, explicit_equity):
    _caps(monkeypatch, absolute="14000", rate="0.02")
    account = AccountSnapshot(cash=1000, holdings=(), base_currency="USD", realized_pnl_today=-11,
        fx_rate_by_currency={"USD": 1400}, total_equity_krw=explicit_equity)
    budget = _daily_loss_budget(account)
    assert budget.blocked
    assert budget.threshold_krw == 14000
    assert budget.realized_pnl_krw == -15400
    assert _supervise(account).forces_exit


def test_missing_fx_blocks_exposure_without_forcing_liquidation(monkeypatch):
    _caps(monkeypatch, absolute="1000", rate="0")
    account = AccountSnapshot(cash=1000, holdings=(), base_currency="USD", realized_pnl_today=-1)
    budget = _daily_loss_budget(account)
    assert budget.blocked and not budget.valuation_complete
    assert budget.reason == "DAILY_LOSS_LIMIT_FX_UNKNOWN"
    verdict = _supervise(account)
    assert verdict.blocks_new_entries and not verdict.forces_exit


def test_unvalued_foreign_holding_blocks_even_with_daily_caps_disabled(monkeypatch):
    _caps(monkeypatch, absolute="0", rate="0")
    holding = Holding("AAPL", "NASDAQ", "Apple", "Technology", 1, 100, 100)
    account = AccountSnapshot(cash=100_000, holdings=(holding,))
    assert _daily_loss_budget(account).reason == "ACCOUNT_VALUATION_INCOMPLETE"
    verdict = _supervise(account)
    assert verdict.blocks_new_entries and not verdict.forces_exit


def test_rate_only_usd_limit_needs_no_invented_krw_rate(monkeypatch):
    _caps(monkeypatch, absolute="0", rate="0.01")
    account = AccountSnapshot(cash=1000, holdings=(), base_currency="USD", realized_pnl_today=-11)
    budget = _daily_loss_budget(account)
    assert budget.valuation_complete and budget.blocked
    assert budget.currency == "USD" and budget.threshold == 10
    assert budget.realized_pnl_krw is None and budget.threshold_krw is None


@pytest.mark.parametrize("loss", [float("nan"), float("inf")])
def test_nonfinite_realized_pnl_cannot_disable_the_stop(monkeypatch, loss):
    _caps(monkeypatch)
    account = AccountSnapshot(cash=100_000, holdings=(), realized_pnl_today=loss)
    assert _daily_loss_budget(account).blocked
    assert not _daily_loss_budget(account).valuation_complete


def test_zero_disables_only_its_own_limit(monkeypatch):
    _caps(monkeypatch, absolute="0", rate="0.01")
    account = AccountSnapshot(cash=100_000, holdings=(), realized_pnl_today=-1001)
    assert _daily_loss_budget(account).threshold == 1000
    assert _daily_loss_budget(account).blocked

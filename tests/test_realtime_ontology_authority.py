from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock
import time

import pytest

from app.ontology.decision_receipt import validate_plan_authority
from app.schemas.domain import AccountSnapshot, Holding
from app.trading.realtime_trading_engine import RealtimeTradingEngine
from app.trading.strategy_supervisor import StrategySupervisor
from app.trading.trade_plan_builder import TradePlanBuilder
from test_ontology_policy_integration import _request
from test_ontology_thresholds import NOW, _policy


def _plan():
    def resolver(**kwargs):
        return _policy(symbol=kwargs["symbol"], all_in_cost_rate=kwargs["all_in_cost_rate"],
                       forecast_gross_bps=kwargs.get("forecast_gross_bps", 600))

    outcome = TradePlanBuilder(ontology_policy_resolver=resolver).build(_request(), now=NOW)
    assert outcome.plan is not None, outcome.as_dict()
    assert validate_plan_authority(outcome.plan, NOW) == ()
    return outcome.plan


def _engine(monkeypatch, *, plan=None, phase="ARMED", legacy=False, account=None, source_age=0):
    monkeypatch.setenv("REALTIME_BUY_ENABLED", "true")
    monkeypatch.setenv("REALTIME_DAILY_REALIZED_LOSS_BUY_STOP_KRW", "1000")
    monkeypatch.setenv("REALTIME_DAILY_REALIZED_LOSS_BUY_STOP_RATE", "0")
    account = account or AccountSnapshot(cash=50_000_000, holdings=(), realized_pnl_today=-1500)
    resolver = Mock(side_effect=AssertionError("Supervision must not make a second ontology assessment"))
    snapshot = {"selected_symbol": "000660", "selected_strategy": "intraday_momentum", "phase": phase}
    session = SimpleNamespace(
        ontology_policy_resolver=None if legacy else resolver,
        snapshot=lambda: snapshot, trade_plan_for=lambda symbol: plan, request_halt=Mock(),
        evaluate=lambda *args: snapshot, allowed_buy_candidates=lambda *args: (),
    )
    quote = SimpleNamespace(received_at=NOW - timedelta(seconds=source_age), spread_bps=10000.)
    decision = SimpleNamespace(store=SimpleNamespace(latest_tick=lambda symbol: quote,
        latest_orderbook=lambda symbol: quote), _symbol_realtime_volatility=lambda *args: 9.)
    engine = RealtimeTradingEngine(
        decision_engine=decision, coordinator=SimpleNamespace(), account_provider=lambda: account,
        candidate_symbols_provider=lambda: (), session_open_provider=lambda: True,
        market_open_provider=lambda *args: True, strategy_session_manager=session,
        strategy_supervisor=StrategySupervisor(), ontology_policy_resolver=None if legacy else resolver,
    )
    return engine, account, resolver


def _other_market_risk():
    return SimpleNamespace(macro_result=SimpleNamespace(
        blocks_buy=True, risk_level=SimpleNamespace(value="CRITICAL"),
        allowed_micro_strategies=(), blocked_micro_strategies=("intraday_momentum",),
    ))


def test_valid_plan_receipt_is_not_vetoed_by_second_macro_spread_or_daily_risk_vote(monkeypatch):
    engine, account, resolver = _engine(monkeypatch, plan=_plan())
    verdict = engine._supervise_session(account, _other_market_risk(), NOW)
    assert not verdict.blocks_new_entries, verdict.as_dict()
    assert not verdict.forces_exit
    assert verdict.diagnostics["supervision_scope"] == "execution_integrity"
    resolver.assert_not_called()
    engine.strategy_session_manager.request_halt.assert_not_called()


@pytest.mark.parametrize("invalid", ["missing", "expired", "wrong_symbol", "unapproved"])
def test_missing_or_invalid_receipt_never_gets_legacy_entry_authority(monkeypatch, invalid):
    plan = _plan()
    if invalid == "missing":
        plan = replace(plan, risk_snapshot={})
    elif invalid == "expired":
        plan = replace(plan, created_at=NOW - timedelta(seconds=2), expires_at=NOW - timedelta(seconds=1))
    elif invalid == "wrong_symbol":
        plan = replace(plan, symbol="005930")
    else:
        snapshot = dict(plan.risk_snapshot)
        snapshot["ontology_authority"] = {**snapshot["ontology_authority"], "approved": False}
        plan = replace(plan, risk_snapshot=snapshot)
    engine, account, resolver = _engine(monkeypatch, plan=plan)
    verdict = engine._supervise_session(account, None, NOW)
    assert verdict.blocks_new_entries
    assert any(code.startswith("ONTOLOGY_AUTHORITY_") for code in verdict.reason_codes)
    engine.strategy_session_manager.request_halt.assert_called_once()
    resolver.assert_not_called()


@pytest.mark.parametrize("phase", ["ENTERING", "OWNED", "EXITING"])
def test_committed_position_exits_remain_with_owning_session(monkeypatch, phase):
    engine, account, resolver = _engine(monkeypatch, phase=phase)
    verdict = engine._supervise_session(account, _other_market_risk(), NOW)
    assert not verdict.forces_exit
    assert not verdict.blocks_new_entries
    resolver.assert_not_called()
    engine.strategy_session_manager.request_halt.assert_not_called()


def test_frozen_approval_does_not_bypass_stale_feed_guard(monkeypatch):
    engine, account, _ = _engine(monkeypatch, plan=_plan(), source_age=120)
    verdict = engine._supervise_session(account, None, NOW)
    assert verdict.blocks_new_entries
    assert any(code.startswith("DATA_STALE:") for code in verdict.reason_codes)


def test_account_valuation_still_blocks_new_exposure_without_forcing_position_exit(monkeypatch):
    account = AccountSnapshot(cash=100_000, holdings=(Holding("AAPL", "NASDAQ", "Apple", "Technology", 1, 100, 100),))
    engine, _, resolver = _engine(monkeypatch, phase="OWNED", account=account)
    verdict = engine._supervise_session(account, None, NOW)
    assert verdict.blocks_new_entries and not verdict.forces_exit
    assert not verdict.diagnostics["daily_loss_valuation_complete"]
    resolver.assert_not_called()


def test_legacy_supervision_retains_its_existing_risk_checks(monkeypatch):
    engine, account, _ = _engine(monkeypatch, legacy=True)
    verdict = engine._supervise_session(account, _other_market_risk(), NOW)
    assert verdict.blocks_new_entries and verdict.forces_exit
    assert "DAILY_LOSS_LIMIT_BREACHED" in verdict.reason_codes


def test_cycle_does_not_preempt_per_market_ontology_with_global_risk_vote(monkeypatch):
    engine, _, resolver = _engine(monkeypatch, plan=_plan())
    engine.macro_micro_observer = lambda *args: _other_market_risk()
    summary = engine.run_once(NOW)
    assert not summary.get("buy_disabled"), summary
    assert summary["realized_pnl_today_krw"] == -1500
    assert summary["daily_loss_budget_krw"] is None, "A legacy cap is not the approved ontology budget"
    assert summary["strategy_supervisor"]["diagnostics"]["supervision_scope"] == "execution_integrity"
    resolver.assert_not_called()


def test_admin_operation_mode_still_blocks_an_ontology_approved_cycle(monkeypatch):
    engine, _, _ = _engine(monkeypatch, plan=_plan())
    engine.new_entries_authorized_provider = lambda: False
    summary = engine.run_once(NOW)
    assert summary["buy_disabled"]
    assert summary["buy_disabled_reason"] == "OPERATION_MODE_BLOCKS_NEW_ENTRIES"


def test_invalid_legacy_daily_risk_setting_cannot_revoke_current_ontology_approval(monkeypatch):
    engine, account, _ = _engine(monkeypatch, plan=_plan())
    monkeypatch.setenv("REALTIME_DAILY_REALIZED_LOSS_BUY_STOP_KRW", "nan")
    verdict = engine._supervise_session(account, None, NOW)
    assert not verdict.blocks_new_entries
    assert verdict.diagnostics["daily_loss_valuation_complete"]


@pytest.mark.parametrize("cooldown", ["loss", "reentry", "broker"])
def test_approved_plan_defers_risk_cooldowns_but_keeps_broker_backoff(monkeypatch, cooldown):
    engine, _, _ = _engine(monkeypatch, plan=_plan())
    engine.candidate_symbols_provider = lambda: ("000660",)
    session = engine.strategy_session_manager
    session.allowed_buy_candidates = lambda candidates, account: candidates
    session.selected_strategy_for = lambda symbol: "intraday_momentum"
    session.election_context_for = lambda symbol: {}
    evaluate = Mock(return_value=SimpleNamespace(approved=False, final_order=None,
        reason_codes=("TEST_CAPTURE_ONLY",), diagnostics={}))
    engine.decision_engine.evaluate_buy = evaluate
    if cooldown == "loss":
        engine._loss_cooldown_until["000660"] = time.monotonic() + 3600
    elif cooldown == "reentry":
        engine._recent_sell_monotonic["000660"] = time.monotonic()
    else:
        engine._error_backoff_until["000660"] = time.monotonic() + 3600
    summary = engine.run_once(NOW)
    if cooldown == "broker":
        evaluate.assert_not_called()
        assert summary["backoff_candidate_excluded"] == ["000660"]
    else:
        evaluate.assert_called_once()
        assert not summary.get("loss_cooldown_candidate_excluded")
        assert not summary.get("rebuy_cooldown_candidate_excluded")
        assert summary["buy_evaluated"] == 1

from dataclasses import replace
from datetime import timedelta
from unittest.mock import Mock

import pytest

from app.audit import AuditLogger
from app.risk import RiskManager
from app.schemas.domain import (
    AccountSnapshot, Holding, MarketSnapshot, OrderAction, OrderIntent, RiskRules,
    SourceMetadata, PrincipalProtectionConfig,
)
from test_ontology_thresholds import NOW, _policy


def _case(tmp_path, **policy_changes):
    policy = replace(_policy(symbol="000660"), position_cap=.15, sector_cap=.30,
        minimum_cash_reserve=.25, trade_loss_budget_rate=.002, all_in_cost_rate=.003,
        hard_stop_rate=.01, soft_stop_rate=.005, net_profit_floor_rate=.001, minimum_reward_risk=1.2,
        **policy_changes)
    market = MarketSnapshot("000660", "KR", "Example", "Technology", 1000, 1e10, .05,
        SourceMetadata("KIS", NOW, observed_at=NOW, source_type="broker", trust_level=5, quality_score=1, is_realtime=True))
    intent = OrderIntent("000660", "KR", OrderAction.BUY, .10, .9, NOW + timedelta(minutes=5),
        ("entry",), (), (), ("actual-quote",), strategy_family="example", expected_exit_price=1100,
        validation_id="validated", model_uncertainty=.99)
    account = AccountSnapshot(cash=1_000_000, holdings=(), total_equity_krw=1_000_000)
    manager = RiskManager(RiskRules(live_trading_enabled=True), AuditLogger(tmp_path / "audit.jsonl"), ontology_policy_required=True)
    return manager, policy, intent, account, market


def test_policy_is_the_only_economic_gate_and_static_models_cannot_veto_it(tmp_path):
    manager, policy, intent, account, market = _case(tmp_path)
    manager.rules = replace(manager.rules, max_single_stock_weight=.0001, max_intraday_position_weight=.0001,
        max_trades_per_day=1, min_average_daily_trading_value=1e15, max_volatility=.00001,
        max_model_uncertainty=.01, principal_protection=PrincipalProtectionConfig(initial_principal=1_000_000))
    manager._profitability_gate.evaluate = Mock(side_effect=AssertionError("duplicate profitability decision"))
    manager.principal_protection.validate_order = Mock(side_effect=AssertionError("duplicate principal decision"))
    result = manager.validate(intent, account, market, trades_today=2, ontology_policy=policy, now=NOW)
    assert result.approved, result.rejection_reasons
    assert result.final_order.quantity == 100
    assert result.adjusted_weight == pytest.approx(.10)
    receipt = result.metadata["ontology_risk_authority"]
    assert receipt["approved"] and receipt["authority_id"] == "ontology-risk-authority-v1"
    assert receipt["actual_quantity"] == result.metadata["cost_breakdown"]["quantity"] == 100
    assert receipt["actual_notional"] == 100_000
    assert "profitability_gate" not in result.checks
    assert "principal_protection_gate" not in result.checks


def test_actual_cost_clips_integer_quantity_and_reserve_uses_final_spend(tmp_path):
    manager, policy, intent, account, market = _case(tmp_path)
    policy = replace(policy, minimum_cash_reserve=.90, trade_loss_budget_rate=.1)
    account = replace(account, cash=100_000, total_equity_krw=100_000)
    result = manager.validate(replace(intent, suggested_weight=.15), account, market, ontology_policy=policy, now=NOW)
    assert result.approved, result.rejection_reasons
    assert result.final_order.quantity == 9
    assert result.metadata["cost_breakdown"]["quantity"] == 9
    cost = result.metadata["cost_breakdown"]
    entry_cost = cost["buy_fee"] + cost["slippage_cost"] + cost["spread_cost"] + cost["market_impact_cost"]
    assert 100_000 - 9000 - entry_cost >= 90_000


def test_market_cash_and_portfolio_reserve_use_measured_fx_without_usd_krw_mixing(tmp_path):
    manager, policy, intent, _, market = _case(tmp_path)
    policy = replace(policy, symbol="AAPL", market="US", minimum_cash_reserve=.80, trade_loss_budget_rate=.1)
    market = replace(market, ticker="AAPL", market="NASDAQ", last_price=50)
    intent = replace(intent, ticker="AAPL", market="US", suggested_weight=.15, expected_exit_price=55)
    account = AccountSnapshot(cash=1_260_000, holdings=(), total_equity_krw=1_400_000,
        cash_by_currency={"KRW": 1_260_000, "USD": 100}, fx_rate_by_currency={"USD": 1400})
    result = manager.validate(intent, account, market, ontology_policy=policy, now=NOW)
    assert result.approved, result.rejection_reasons
    assert result.final_order.quantity == 1
    assert result.metadata["equity_for_sizing"] == pytest.approx(1000)
    assert result.metadata["cash_available_for_market"] == 100
    blocked = manager.validate(intent, replace(account, orderable_cash_by_currency={"USD": 0}), market, ontology_policy=policy, now=NOW)
    assert not blocked.approved


def test_add_on_never_uses_one_share_rounding_to_exceed_total_policy_cap(tmp_path):
    manager, policy, intent, _, market = _case(tmp_path)
    policy = replace(policy, position_cap=.105, trade_loss_budget_rate=.1)
    holding = Holding("000660", "KR", "Example", "Technology", 5, 1000, 1000)
    account = AccountSnapshot(cash=90_000, holdings=(holding, holding), total_equity_krw=100_000)
    result = manager.validate(replace(intent, suggested_weight=.105), account, market, ontology_policy=policy, now=NOW)
    assert not result.approved
    assert result.metadata["current_position_value"] == 10_000
    assert "ONTOLOGY_POSITION_BUDGET_EXCEEDED" in result.rejection_reasons
    assert not result.metadata["ontology_risk_authority"]["approved"]


def test_actual_round_trip_cost_can_reject_an_optimistic_frozen_forecast(tmp_path):
    manager, policy, intent, account, market = _case(tmp_path)
    real_estimate = manager.cost_engine.estimate
    manager.cost_engine.estimate = lambda **kwargs: replace(real_estimate(**kwargs), total_cost_rate=.04)
    result = manager.validate(replace(intent, expected_exit_price=1040), account, market, ontology_policy=policy, now=NOW)
    assert not result.approved
    assert "POLICY_NET_REWARD_INSUFFICIENT" in result.rejection_reasons
    assert result.metadata["ontology_risk_authority"]["all_in_cost_rate"] == .04


@pytest.mark.parametrize("timestamp", [NOW + timedelta(seconds=1), NOW.replace(tzinfo=None), NOW - timedelta(minutes=1)])
def test_noncausal_undated_and_stale_execution_quotes_fail_closed(tmp_path, timestamp):
    manager, policy, intent, account, market = _case(tmp_path)
    market = replace(market, source=replace(market.source, observed_at=timestamp))
    result = manager.validate(intent, account, market, ontology_policy=policy, now=NOW)
    assert not result.approved and not result.checks["quote_freshness_check"]


@pytest.mark.parametrize("price", [float("nan"), float("inf"), 0, -1])
def test_invalid_quote_has_no_order_and_no_numeric_crash(tmp_path, price):
    manager, policy, intent, account, market = _case(tmp_path)
    result = manager.validate(intent, account, replace(market, last_price=price), ontology_policy=policy, now=NOW)
    assert not result.approved and not result.checks["data_integrity_check"]


def test_expired_signal_and_wrong_instrument_cannot_reuse_a_current_policy(tmp_path):
    manager, policy, intent, account, market = _case(tmp_path)
    expired = manager.validate(replace(intent, valid_until=NOW - timedelta(seconds=1)), account, market, ontology_policy=policy, now=NOW)
    assert "ORDER_INTENT_EXPIRED_OR_INVALID" in expired.rejection_reasons
    wrong = manager.validate(intent, account, replace(market, ticker="005930"), ontology_policy=policy, now=NOW)
    assert not wrong.checks["market_identity_check"]
    boundary = manager.validate(intent, account, market, ontology_policy=replace(policy, expires_at=NOW), now=NOW)
    assert not boundary.approved and not boundary.checks["ontology_policy_current"]


def test_policy_reduces_total_position_loss_budget_when_actual_cost_is_higher(tmp_path):
    manager, policy, intent, account, market = _case(tmp_path)
    real_estimate = manager.cost_engine.estimate
    manager.cost_engine.estimate = lambda **kwargs: replace(real_estimate(**kwargs), total_cost_rate=.04)
    result = manager.validate(intent, account, market, ontology_policy=policy, now=NOW)
    assert result.approved, result.rejection_reasons
    assert result.final_order.quantity == 40
    assert result.final_order.quantity * market.last_price * (.01 + .04) <= account.equity * policy.trade_loss_budget_rate


@pytest.mark.parametrize("authority_required", [True, False])
def test_long_exit_contract_is_close_even_without_entry_evidence(tmp_path, authority_required):
    manager, _, intent, account, market = _case(tmp_path)
    manager.ontology_policy_required = authority_required
    holding = Holding("000660", "KR", "Example", "Technology", 3, 1000, 1000, sellable_quantity=2)
    account = replace(account, cash=0, holdings=(holding,), realized_pnl_today=-50_000)
    intent = replace(intent, action=OrderAction.SELL, suggested_weight=0)
    result = manager.validate(intent, account, market, trades_today=100, now=NOW)
    assert result.approved, result.rejection_reasons
    assert result.final_order.quantity == 2
    assert result.final_order.position_effect == "CLOSE"


def test_partial_reduction_uses_current_quote_and_integer_target(tmp_path):
    manager, _, intent, _, market = _case(tmp_path)
    holding = Holding("000660", "KR", "Example", "Technology", 5, 1000, 1000)
    account = AccountSnapshot(cash=0, holdings=(holding,), total_equity_krw=10_000)
    intent = replace(intent, action=OrderAction.REDUCE, suggested_weight=.4)
    result = manager.validate(intent, account, replace(market, last_price=2000), now=NOW)
    assert result.approved, result.rejection_reasons
    assert result.final_order.quantity == 3
    assert result.final_order.position_effect == "CLOSE"


def test_static_entry_product_permission_cannot_trap_a_held_cash_etf(tmp_path):
    manager, _, intent, _, market = _case(tmp_path)
    market = replace(market, ticker="069500", company_name="KODEX 200 ETF")
    holding = Holding("069500", "KR", "KODEX 200 ETF", "Technology", 1, 1000, 1000)
    account = AccountSnapshot(cash=0, holdings=(holding,))
    result = manager.validate(replace(intent, ticker="069500", action=OrderAction.SELL), account, market, now=NOW)
    assert result.approved, result.rejection_reasons
    assert result.final_order.position_effect == "CLOSE"


def test_single_authority_preserves_explicit_warrant_permission(tmp_path):
    manager, policy, intent, _, market = _case(tmp_path)
    policy = replace(policy, market="US", symbol="LCFYW")
    market = replace(market, market="NASDAQ", ticker="LCFYW")
    intent = replace(intent, market="NASDAQ", ticker="LCFYW")
    account = AccountSnapshot(cash=1_000_000, holdings=(), base_currency="USD")
    result = manager.validate(intent, account, market, ontology_policy=policy, now=NOW)
    assert not result.approved and "NON_COMMON_INSTRUMENT_BUY_BLOCKED" in result.rejection_reasons


@pytest.mark.parametrize("count", [None, float("nan"), float("inf"), -1, 1.5, True])
def test_missing_or_malformed_trade_count_cannot_create_daily_capacity(tmp_path, count):
    manager, policy, intent, account, market = _case(tmp_path)
    result = manager.validate(intent, account, market, ontology_policy=policy, now=NOW, trades_today=count)
    assert not result.approved
    assert not result.checks["trade_count_known"]
    assert "ONTOLOGY_TRADE_COUNT_UNAVAILABLE" in result.rejection_reasons

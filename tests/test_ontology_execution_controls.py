from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest

from app.cost import ProfitabilityGate, ProfitabilityInput
from app.schemas.domain import AccountSnapshot, FinalOrder, Holding, OrderSide, OrderType
from app.trading.realtime_trading_engine import RealtimeTradingEngine, _daily_loss_budget
from app.trading.strategy_supervisor import StrategySupervisor, SupervisorConfig, SupervisorObservation
from test_ontology_thresholds import NOW, _policy


def _engine(policy):
    return RealtimeTradingEngine(
        decision_engine=SimpleNamespace(), coordinator=SimpleNamespace(),
        account_provider=lambda: None, candidate_symbols_provider=lambda: (),
        session_open_provider=lambda: False,
        ontology_policy_resolver=lambda **kwargs: policy,
        ontology_policy_snapshot_provider=lambda: {"policies": {"KR:005930": {"policy": policy.as_dict()}}},
    )


def test_generated_daily_budget_uses_actual_currency_and_stricter_operator_ceiling(monkeypatch):
    monkeypatch.setenv("REALTIME_DAILY_REALIZED_LOSS_BUY_STOP_KRW", "14000")
    monkeypatch.setenv("REALTIME_DAILY_REALIZED_LOSS_BUY_STOP_RATE", "0.02")
    account = AccountSnapshot(cash=1000, holdings=(), base_currency="USD",
        realized_pnl_today=-6, fx_rate_by_currency={"USD": 1400})
    budget = _daily_loss_budget(account, policy_rate=.005)
    assert budget.blocked
    assert budget.threshold == 5
    assert budget.threshold_krw == 7000
    assert budget.realized_pnl_krw == -8400
    assert not _daily_loss_budget(account).blocked


@pytest.mark.parametrize("invalid", ["expired", "future", "no_entry"])
def test_stale_or_unusable_cached_policy_cannot_change_account_budget(monkeypatch, invalid):
    monkeypatch.setenv("REALTIME_DAILY_REALIZED_LOSS_BUY_STOP_KRW", "0")
    monkeypatch.setenv("REALTIME_DAILY_REALIZED_LOSS_BUY_STOP_RATE", "0")
    policy = replace(_policy(), daily_loss_budget_rate=.001)
    policy = {"expired": replace(policy, expires_at=NOW - timedelta(seconds=1)),
              "future": replace(policy, as_of=NOW + timedelta(seconds=1)),
              "no_entry": replace(policy, valid_for_entry=False)}[invalid]
    account = AccountSnapshot(cash=100_000, holdings=(), realized_pnl_today=-500)
    budget = _engine(policy)._account_loss_budget(account, NOW)
    assert budget.threshold == 1000
    assert not budget.blocked


def test_current_generated_budget_reaches_account_cycle(monkeypatch):
    monkeypatch.setenv("REALTIME_DAILY_REALIZED_LOSS_BUY_STOP_KRW", "0")
    monkeypatch.setenv("REALTIME_DAILY_REALIZED_LOSS_BUY_STOP_RATE", "0")
    policy = replace(_policy(), daily_loss_budget_rate=.001)
    account = AccountSnapshot(cash=100_000, holdings=(), realized_pnl_today=-101)
    assert _engine(policy)._account_loss_budget(account, NOW).blocked


def test_supervisor_uses_generated_spread_and_missing_policy_cannot_force_position_exit():
    policy = replace(_policy(), max_spread_rate=.0002)
    supervisor = StrategySupervisor(SupervisorConfig(max_spread_bps=100))
    observed = SupervisorObservation(symbol=policy.symbol, as_of=NOW, position_open=True,
        spread_bps=3, ontology_policy=policy, ontology_policy_required=True)
    verdict = supervisor.evaluate(observed)
    assert verdict.blocks_new_entries and not verdict.forces_exit
    assert "SPREAD_WIDENED:3.0bps" in verdict.reason_codes
    missing = supervisor.evaluate(replace(observed, ontology_policy=None))
    assert missing.blocks_new_entries and not missing.forces_exit
    assert "ONTOLOGY_POLICY_UNAVAILABLE" in missing.reason_codes


def test_disabling_legacy_supervisor_does_not_disable_required_policy():
    verdict = StrategySupervisor(SupervisorConfig(enabled=False)).evaluate(
        SupervisorObservation(symbol="005930", as_of=NOW, position_open=True, ontology_policy_required=True)
    )
    assert verdict.blocks_new_entries and not verdict.forces_exit


def test_generated_cost_gate_overrides_fixed_return_floor_without_mutating_shared_gate():
    gate = ProfitabilityGate()
    gate.policy = replace(gate.policy, min_required_net_return={"default": .5, "KR": .5})
    request = ProfitabilityInput(symbol="005930", market="KR", venue="KRX", entry_price=70000,
        expected_exit_price=73500, quantity=1, liquidity_score=.9, spread_rate=.00015,
        average_daily_trading_value=5e11)
    assert not gate.evaluate(request).allowed
    policy = _policy(all_in_cost_rate=gate.evaluate(request).all_in_cost_rate)
    decision = gate.evaluate(request, ontology_policy=policy, now=NOW)
    assert decision.allowed, decision.rejection_reasons
    assert decision.policy_version == policy.policy_id
    assert decision.required_min_net_return == max(policy.net_profit_floor_rate, policy.soft_stop_rate * policy.minimum_reward_risk)
    assert gate.policy.min_required_net_return["KR"] == .5
    assert not gate.evaluate(request, ontology_policy=replace(policy, expires_at=NOW - timedelta(seconds=1)), now=NOW).allowed


def test_submitted_prices_do_not_become_realized_loss_evidence():
    engine = _engine(_policy())
    order = FinalOrder(ticker="005930", market="KR", side=OrderSide.BUY,
                       order_type=OrderType.LIMIT, quantity=1, limit_price=100)
    engine._record_submitted_order_for_performance(order, "BUY")
    engine._record_submitted_order_for_performance(replace(order, side=OrderSide.SELL, limit_price=90), "SELL")
    assert not engine._loss_cooldown_until
    assert not engine._recent_buy_orders


def test_forced_strategy_exit_preserves_broker_zero_sellable_quantity():
    holding = Holding("005930", "KR", "Example", "Technology", 5, 100, 99, sellable_quantity=0)
    result = _engine(_policy())._forced_strategy_session_exit_result(holding, SimpleNamespace(), "STRATEGY_STOP_LOSS")
    assert result.final_order.quantity == 0

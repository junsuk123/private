from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest

from app.risk.manager import RiskManager
from app.schemas.domain import AccountSnapshot, Holding, MarketSnapshot, OrderAction, OrderIntent, RiskRules, SourceMetadata
from app.trading.directional import PositionDirection
from app.trading.strategy_session import StrategySessionConfig, StrategySessionManager, _ElectionProposal
from app.trading.trade_plan_builder import PlanRequest, TradePlanBuilder

from test_ontology_thresholds import NOW, _policy, _projection


def _market(price=70_000):
    return MarketSnapshot(
        ticker="000660", market="KR", company_name="Example", sector="semiconductor",
        last_price=price, average_daily_trading_value=5e11, volatility_20d=.02,
        source=SourceMetadata(source_name="kis_realtime", retrieved_at=NOW,
                              observed_at=NOW, source_type="broker_api", trust_level=5, is_realtime=True),
    )


def _request(**changes):
    values = dict(
        symbol="000660", strategy_id="intraday_momentum", market="KR",
        account=AccountSnapshot(cash=50_000_000, holdings=(), total_equity_krw=100_000_000),
        market_snapshot=_market(), reference_price=70_000, take_profit_rate=.06,
        stop_loss_rate=.04, trailing_rate=.02, max_holding_seconds=1800,
        gross_edge_bps=600, confidence=.9, liquidity_score=.9, spread_bps=1.5,
        realized_volatility=.002, source_ids=("tick:fixture",),
    )
    values.update(changes)
    return PlanRequest(**values)


def _intent():
    return OrderIntent(
        ticker="000660", market="KR", action=OrderAction.BUY, suggested_weight=.02,
        confidence=.9, valid_until=NOW + timedelta(minutes=5),
        reasoning_summary=("fixture",), supporting_factors=("fixture",),
        contradicting_factors=(), source_data_ids=("tick:fixture",),
        expected_exit_price=74_200, target_net_return=.01,
    )


def test_frozen_trade_plan_uses_generated_geometry_cost_basis_and_expiry():
    captured = []
    generated = []

    def resolver(**kwargs):
        captured.append(kwargs)
        policy = _policy(symbol=kwargs["symbol"], all_in_cost_rate=kwargs["all_in_cost_rate"],
                         forecast_gross_bps=kwargs["forecast_gross_bps"])
        generated.append(policy)
        return policy

    outcome = TradePlanBuilder(ontology_policy_resolver=resolver).build(_request(), now=NOW)
    assert outcome.plan is not None, outcome.as_dict()
    plan, policy = outcome.plan, generated[0]
    assert plan.exit_rules.take_profit_rate == policy.target_return_rate
    assert plan.exit_rules.stop_loss_rate == policy.soft_stop_rate
    assert plan.exit_rules.trailing_rate == policy.trailing_stop_rate
    assert plan.exit_rules.max_holding_seconds == policy.maximum_holding_seconds
    assert plan.expires_at <= policy.expires_at
    assert plan.risk_snapshot["ontology_risk_policy"]["policy_id"] == policy.policy_id
    assert plan.election_context["ontology_risk_policy"]["evidence_id"] == policy.evidence_id
    assert policy.policy_id in plan.source_ids
    assert captured[0]["all_in_cost_rate"] <= plan.cost_snapshot["all_in_cost_rate"]
    assert plan.cost_snapshot["quantity"] == plan.quantity
    assert plan.max_notional <= 100_000_000 * policy.position_cap


@pytest.mark.parametrize("invalid", ["missing", "expired", "future", "wrong_market", "wrong_symbol"])
def test_builder_cannot_freeze_a_plan_with_missing_or_mismatched_policy(invalid):
    policy = _policy(symbol="000660")
    policy = {
        "missing": None,
        "expired": replace(policy, expires_at=NOW - timedelta(seconds=1)),
        "future": replace(policy, as_of=NOW + timedelta(seconds=1)),
        "wrong_market": replace(policy, market="US"),
        "wrong_symbol": replace(policy, symbol="005930"),
    }[invalid]
    outcome = TradePlanBuilder(ontology_policy_resolver=lambda **kwargs: policy).build(_request(), now=NOW)
    assert outcome.plan is None
    assert outcome.no_trade.stage == "ontology_policy"


@pytest.mark.parametrize("invalid", ["missing", "expired", "wrong_market", "wrong_symbol"])
def test_required_risk_policy_rejects_new_entry_even_with_good_cash_and_price(invalid):
    policy = _policy(symbol="000660")
    policy = {
        "missing": None,
        "expired": replace(policy, expires_at=NOW - timedelta(seconds=1)),
        "wrong_market": replace(policy, market="US"),
        "wrong_symbol": replace(policy, symbol="005930"),
    }[invalid]
    result = RiskManager(ontology_policy_required=True).validate(
        _intent(), _request().account, _market(), ontology_policy=policy, now=NOW,
    )
    assert result.approved is False
    assert result.final_order is None
    assert result.checks["ontology_policy_current"] is False


def test_one_share_rounding_cannot_exceed_generated_position_budget():
    policy = replace(_policy(symbol="000660"), position_cap=.01)
    # The single authority must return NO_TRADE when one share exceeds the
    # generated 1k position budget; no downstream sizing veto is needed.
    request = _request(account=AccountSnapshot(cash=100_000, holdings=(), total_equity_krw=100_000))
    outcome = TradePlanBuilder(ontology_policy_resolver=lambda **kwargs: policy).build(request, now=NOW)
    assert outcome.plan is None
    assert "ONTOLOGY_POSITION_BUDGET_EXCEEDED" in outcome.no_trade.reason_codes
    assert outcome.no_trade.stage == "ontology_authority"
    assert outcome.no_trade.risk_snapshot["ontology_authority"]["approved"] is False


def test_dynamic_risk_budget_tightens_current_order_without_mutating_shared_rules():
    rules = RiskRules(max_trades_per_day=24, daily_loss_stop=.05)
    manager = RiskManager(rules, ontology_policy_required=True)
    policy = replace(_policy(symbol="000660"), max_trades_per_day=2, daily_loss_budget_rate=.001)
    result = manager.validate(_intent(), _request().account, _market(), trades_today=3,
                              ontology_policy=policy, now=NOW)
    assert result.checks["ontology_policy_current"]
    assert not result.checks["trade_count_limit"]
    assert not result.approved
    assert manager.rules.max_trades_per_day == 24
    assert manager.rules.daily_loss_stop == .05


def test_krx_alias_uses_the_same_kr_ontology_policy():
    policy = _policy(symbol="000660")
    outcome = TradePlanBuilder(ontology_policy_resolver=lambda **kwargs: policy).build(
        _request(market="KRX", market_snapshot=replace(_market(), market="KRX")), now=NOW,
    )
    assert outcome.plan is not None, outcome.as_dict()


def test_risk_reducing_exit_does_not_need_a_fresh_entry_policy_or_trade_budget():
    holding = Holding(ticker="000660", market="KR", company_name="Example", quantity=1, average_price=70_000, last_price=70_000, sector="semiconductor")
    account = AccountSnapshot(cash=1_000_000, holdings=(holding,), total_equity_krw=10_000_000, realized_pnl_today=-1_000_000)
    intent = replace(_intent(), action=OrderAction.SELL, position_effect="CLOSE", suggested_weight=1.0)
    result = RiskManager(RiskRules(max_trades_per_day=1), ontology_policy_required=True).validate(
        intent, account, _market(), trades_today=5, now=NOW,
    )
    assert result.approved, result.rejection_reasons
    assert "ontology_policy_current" not in result.checks
    assert result.final_order.quantity == 1


def _owned_manager(tmp_path, resolver):
    manager = StrategySessionManager(
        config=StrategySessionConfig(state_path=str(tmp_path / "session.json"), invalidation_confirm_cycles=1),
        ontology_policy_resolver=resolver, selector_v2_runner=False,
    )
    state = manager._state
    state.session_id = "session-1"
    state.phase = "OWNED"
    state.selected_symbol = "005930"
    state.selected_strategy = "intraday_momentum"
    state.entry_price = 100
    state.stop_loss_rate = .02
    state.stop_price = 98
    state.trailing_stop_rate = .01
    state.target_return_rate = .20
    state.target_price = 120
    state.max_holding_seconds = 900
    state.expected_cost_bps = 10
    state.position_opened_at = (NOW - timedelta(seconds=300)).isoformat()
    return manager


def test_owned_position_uses_generated_stop_on_same_exit_cycle(tmp_path):
    policy = replace(_policy(), soft_stop_rate=.005, target_return_rate=.2,
                     hard_stop_rate=.007, trailing_stop_rate=.003)
    manager = _owned_manager(tmp_path, lambda **kwargs: policy)
    holding = SimpleNamespace(average_price=100, last_price=99.2, quantity=1)
    manager._evaluate_exit(holding, None, NOW)
    assert manager._state.stop_price == pytest.approx(99.5)
    assert manager._state.phase == "EXITING"
    assert manager._state.exit_reason == "STRATEGY_STOP_LOSS"


def test_refresh_never_extends_holding_or_recursively_reduces_requested_horizon(tmp_path):
    requests = []
    policies = [replace(_policy(), maximum_holding_seconds=400), replace(_policy(), maximum_holding_seconds=800)]

    def resolver(**kwargs):
        requests.append(kwargs["requested_horizon_seconds"])
        return policies[len(requests) - 1]

    manager = _owned_manager(tmp_path, resolver)
    holding = SimpleNamespace(average_price=100, last_price=100, quantity=1)
    manager._refresh_position_ontology_policy(holding, NOW)
    manager._refresh_position_ontology_policy(holding, NOW + timedelta(seconds=1))
    assert requests == [900, 900]
    assert manager._state.max_holding_seconds == 400


@pytest.mark.parametrize("invalid", ["expired", "future", "wrong_market"])
def test_invalid_refresh_preserves_last_owned_policy_and_stop(tmp_path, invalid):
    good = replace(_policy(), soft_stop_rate=.005, target_return_rate=.2)
    bad = replace(good, soft_stop_rate=.001, target_return_rate=.5)
    bad = {
        "expired": replace(bad, expires_at=NOW - timedelta(seconds=1)),
        "future": replace(bad, as_of=NOW + timedelta(minutes=1)),
        "wrong_market": replace(bad, market="US"),
    }[invalid]
    responses = iter((good, bad))
    manager = _owned_manager(tmp_path, lambda **kwargs: next(responses))
    holding = SimpleNamespace(average_price=100, last_price=100, quantity=1)
    manager._refresh_position_ontology_policy(holding, NOW)
    stop, target = manager._state.stop_price, manager._state.target_price
    manager._refresh_position_ontology_policy(holding, NOW + timedelta(seconds=1))
    assert manager._state.stop_price == stop
    assert manager._state.target_price == target
    assert manager._exit_policy_values()["soft_stop_rate"] == good.soft_stop_rate


def test_new_session_on_same_symbol_does_not_inherit_previous_positions_tight_stop(tmp_path):
    first = replace(_policy(), soft_stop_rate=.003)
    second = replace(_policy(), soft_stop_rate=.015)
    responses = iter((first, second))
    manager = _owned_manager(tmp_path, lambda **kwargs: next(responses))
    holding = SimpleNamespace(average_price=100, last_price=100, quantity=1)
    manager._refresh_position_ontology_policy(holding, NOW)
    manager._state.session_id = "session-2"
    manager._state.election_context = {}
    manager._state.stop_loss_rate = .02
    manager._state.stop_price = 98
    manager._refresh_position_ontology_policy(holding, NOW + timedelta(seconds=1))
    assert manager._state.stop_loss_rate == .015
    assert manager._state.stop_price == pytest.approx(98.5)


def test_generated_early_exit_confirmation_count_replaces_legacy_fixed_count(tmp_path):
    policy = replace(_policy(), early_exit_confirmations=3, minimum_holding_seconds=0,
                     noise_band_rate=.00001)
    manager = _owned_manager(tmp_path, lambda **kwargs: policy)
    manager._state.election_context["ontology_risk_policy"] = policy.as_dict()
    ids = iter((NOW.isoformat(), (NOW + timedelta(seconds=1)).isoformat(), (NOW + timedelta(seconds=2)).isoformat()))
    manager._continuation_invalidation_evidence = lambda *args: (next(ids), ["ONTOLOGY_THESIS_CHANGED"])
    holding = SimpleNamespace(average_price=100, last_price=99.5, quantity=1)
    manager._refresh_position_ontology_policy(holding, NOW)
    assert manager._confirmed_continuation_exit_reason(holding, None, direction=PositionDirection.LONG, now=NOW) is None
    assert manager._confirmed_continuation_exit_reason(holding, None, direction=PositionDirection.LONG, now=NOW + timedelta(seconds=1)) is None
    assert manager._confirmed_continuation_exit_reason(holding, None, direction=PositionDirection.LONG, now=NOW + timedelta(seconds=2)) == "STRATEGY_EDGE_DECAY_LOSS_LIMIT"


def test_algorithm_static_economic_floor_is_advisory_to_ontology(tmp_path, monkeypatch):
    manager = _owned_manager(tmp_path, lambda **kwargs: _policy())
    raw = {"triggered": True, "expected_edge_bps": 35., "cost_viable": False,
           "score": .8, "confidence": .8, "diagnostics": {"minimum_edge_bps": 100.}}
    algorithm = SimpleNamespace(entry=lambda *args: SimpleNamespace(as_dict=lambda: dict(raw)))
    monkeypatch.setattr("app.technical.strategy_algorithms.get_algorithm", lambda *args, **kwargs: algorithm)
    decision = manager._mechanical_entry_verdict(symbol="005930", strategy_id="breakout_volume",
        evidence_row={"technical_features": {"symbol": "005930"}}, now=NOW, macro=None,
        intent=None, micro_result=None, candidate_count=1, borrow_snapshot=None, record=False)
    assert decision["triggered"]
    assert decision["expected_edge_bps"] == 35., "Do not inflate the forecast to clear a floor"
    assert decision["cost_viable"] is None, "The ontology must decide actual economics"
    assert decision["diagnostics"]["legacy_cost_viable"] is False


@pytest.mark.parametrize("mark", [float("nan"), float("inf"), 0.0])
def test_invalid_mark_cannot_manufacture_a_dynamic_thesis_exit(tmp_path, mark):
    policy = replace(_policy(), early_exit_confirmations=1, minimum_holding_seconds=0,
                     noise_band_rate=.00001)
    manager = _owned_manager(tmp_path, lambda **kwargs: policy)
    manager._state.election_context["ontology_risk_policy"] = policy.as_dict()
    manager._continuation_invalidation_evidence = lambda *args: (NOW.isoformat(), ["ONTOLOGY_THESIS_CHANGED"])
    holding = SimpleNamespace(average_price=100, last_price=mark, quantity=1)
    manager._refresh_position_ontology_policy(holding, NOW)
    assert manager._confirmed_continuation_exit_reason(holding, None, direction=PositionDirection.LONG, now=NOW) is None


def test_entry_policy_expiry_does_not_block_informational_exit_profitability():
    from app.cost.profitability_gate import ProfitabilityGate, ProfitabilityInput
    expired = replace(_policy(), valid_for_entry=False, expires_at=NOW - timedelta(seconds=1))
    request = ProfitabilityInput(symbol="005930", action="SELL", market="KR", venue="KRX",
                                 instrument_type="EQUITY", entry_price=100, expected_exit_price=99,
                                 quantity=1, position_effect="CLOSE")
    verdict = ProfitabilityGate().evaluate(request, ontology_policy=expired, now=NOW)
    assert verdict.allowed, verdict.rejection_reasons


def test_mandatory_plan_failure_cannot_leave_a_legacy_armed_session(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGO_INTRADAY_MOMENTUM_LIVE_AUTHORIZED", "1")
    manager = StrategySessionManager(
        config=StrategySessionConfig(state_path=str(tmp_path / "session.json"), require_live_gnn=False,
                                     algorithm_primary_election=False, bandit_enabled=False),
        ontology_policy_resolver=lambda **kwargs: None,
        selector_v2_runner=False,
    )
    calls = []
    manager._build_trade_plan = lambda *args, **kwargs: calls.append(True)
    proposal = _ElectionProposal(
        symbol="005930", strategy_id="intraday_momentum", source="TEST", entry_price=70_000,
        target_return_rate=.06, stop_loss_rate=.01, trailing_stop_rate=.005, max_holding_seconds=900,
        score=.9, confidence=.9, expected_net_return_bps=500, expected_cost_bps=30,
        gnn_actionable=True, gnn_action="ACTIVATE_STRATEGY", gnn_reason_codes=[], ontology_reason_codes=[],
        macro_regime="TREND_UP", micro_regime="MOMENTUM", explanation_paths=[], intent=None,
        candidate_count=1, micro_result=None, evidence_row={}, last_reason="TEST",
        gnn_required_for_edge=False, algorithm_triggered=True,
    )
    assert manager._arm(proposal, NOW, account=_request().account) is False
    assert calls == [True]
    state = manager.snapshot()
    assert state["phase"] == "SCANNING"
    assert state["selected_symbol"] is None
    assert state["selected_strategy"] is None

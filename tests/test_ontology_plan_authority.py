from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest

from app.cost import ProfitabilityGate, ProfitabilityInput
from app.risk.manager import RiskManager
from app.trading.trade_plan_builder import TradePlanBuilder
from test_ontology_policy_integration import _request
from test_ontology_thresholds import NOW, _policy


class ForbiddenLegacyGate:
    def evaluate(self, *args, **kwargs):
        raise AssertionError("ontology plan must not invoke a second profitability gate")

    def size(self, *args, **kwargs):
        raise AssertionError("ontology plan must not invoke a second position sizer")


class RecordingAuthority(RiskManager):
    def __init__(self):
        super().__init__()
        self.calls = []

    def validate(self, intent, account, market, **kwargs):
        result = super().validate(intent, account, market, **kwargs)
        self.calls.append((intent, market, kwargs, result))
        return result


def test_one_authority_prices_entire_entry_band_without_inflating_forecast():
    manager = RecordingAuthority()
    policies = []

    def resolve(**kwargs):
        policies.append(_policy(symbol=kwargs["symbol"], all_in_cost_rate=kwargs["all_in_cost_rate"],
                                forecast_gross_bps=kwargs["forecast_gross_bps"]))
        return policies[-1]

    request = _request()
    outcome = TradePlanBuilder(risk_manager=manager, ontology_policy_resolver=resolve,
        profitability_gate=ForbiddenLegacyGate(), position_sizer=ForbiddenLegacyGate()).build(request, now=NOW)
    assert outcome.plan is not None, outcome.as_dict()
    assert len(policies) == len(manager.calls) == 1
    plan, policy = outcome.plan, policies[0]
    intent, market, kwargs, result = manager.calls[0]
    worst = request.reference_price * (1 + policy.entry_band_rate)
    assert market.last_price == worst
    assert intent.expected_exit_price == request.reference_price * (1 + request.gross_edge_bps / 10000)
    assert plan.entry_rule.max_price == worst
    assert plan.reference_price == worst
    assert plan.entry_rule.min_price == request.reference_price * (1 - policy.entry_band_rate)
    assert plan.quantity == result.final_order.quantity
    assert plan.max_notional == plan.quantity * worst
    assert plan.cost_snapshot["quantity"] == plan.quantity
    assert plan.cost_snapshot["entry_price"] == worst
    assert plan.cost_snapshot["all_in_cost_rate"] == result.metadata["ontology_risk_authority"]["all_in_cost_rate"]
    assert plan.risk_snapshot["ontology_authority"] == result.metadata["ontology_risk_authority"]
    assert plan.expected_net_edge_bps == pytest.approx(
        (intent.expected_exit_price / worst - 1 - plan.cost_snapshot["all_in_cost_rate"]) * 10000)
    assert plan.expires_at <= policy.expires_at
    assert plan.order_contract == {"direction": "LONG", "position_direction": "LONG", "position_effect": "OPEN", "execution_product": "CASH"}
    from app.ontology.decision_receipt import validate_plan_authority
    assert validate_plan_authority(plan, NOW) == ()


@pytest.mark.parametrize("contract", [dict(direction="SHORT"), dict(position_direction="SHORT"), dict(position_effect="CLOSE"),
                                    dict(execution_product="CREDIT_BORROW")])
def test_conflicting_order_contract_is_not_silently_changed(contract):
    outcome = TradePlanBuilder(ontology_policy_resolver=lambda **kw: _policy(symbol=kw["symbol"])).build(
        _request(order_contract=contract), now=NOW)
    assert outcome.plan is None
    assert outcome.no_trade.reason_codes == ("ONTOLOGY_PLAN_REQUIRES_CASH_LONG",)


def test_plan_expiry_respects_authority_shorter_lease():
    class ShortLeaseAuthority(RecordingAuthority):
        def validate(self, *args, **kwargs):
            result = super().validate(*args, **kwargs)
            metadata = dict(result.metadata)
            metadata["ontology_risk_authority"] = {**metadata["ontology_risk_authority"],
                "expires_at": (NOW + timedelta(seconds=1)).isoformat()}
            return replace(result, metadata=metadata)

    outcome = TradePlanBuilder(risk_manager=ShortLeaseAuthority(),
        ontology_policy_resolver=lambda **kw: _policy(symbol=kw["symbol"])).build(_request(), now=NOW)
    assert outcome.plan is not None, outcome.as_dict()
    assert outcome.plan.expires_at == NOW + timedelta(seconds=1)


def test_confirmed_market_day_entries_reach_the_single_authority():
    manager = RecordingAuthority()
    policy = replace(_policy(symbol="000660"), max_trades_per_day=2)
    outcome = TradePlanBuilder(risk_manager=manager, ontology_policy_resolver=lambda **kw: policy).build(
        _request(trades_today=2), now=NOW)
    assert len(manager.calls) == 1
    assert manager.calls[0][2]["trades_today"] == 2
    assert outcome.plan is None
    assert "trade_count_limit" in outcome.no_trade.reason_codes


@pytest.mark.parametrize("count", [None, -1, 1.5, True])
def test_unavailable_or_invalid_daily_count_cannot_default_to_zero(count):
    manager = RecordingAuthority()
    outcome = TradePlanBuilder(risk_manager=manager, ontology_policy_resolver=lambda **kw: _policy(symbol=kw["symbol"])).build(
        _request(trades_today=count), now=NOW)
    assert outcome.plan is None
    assert outcome.no_trade.reason_codes == ("ONTOLOGY_TRADE_COUNT_UNAVAILABLE",)
    assert not manager.calls


@pytest.mark.parametrize("change", [dict(approved=False), dict(symbol="MSFT"), dict(actual_quantity=999),
                                    dict(authorized_price=1), dict(actual_notional=1)])
def test_plan_cannot_freeze_mismatched_authority_receipt(change):
    class BadReceiptAuthority(RecordingAuthority):
        def validate(self, *args, **kwargs):
            result = super().validate(*args, **kwargs)
            metadata = dict(result.metadata)
            metadata["ontology_risk_authority"] = {**metadata["ontology_risk_authority"], **change}
            return replace(result, metadata=metadata)

    outcome = TradePlanBuilder(risk_manager=BadReceiptAuthority(),
        ontology_policy_resolver=lambda **kw: _policy(symbol=kw["symbol"])).build(_request(), now=NOW)
    assert outcome.plan is None
    assert outcome.no_trade.reason_codes == ("ONTOLOGY_AUTHORITY_RECEIPT_INVALID",)


def test_approved_legacy_adapter_cannot_invent_ontology_authority():
    class LegacyAdapter:
        def validate(self, *args, **kwargs):
            return SimpleNamespace(approved=True, rejection_reasons=(), metadata={},
                                   final_order=SimpleNamespace(quantity=1))

    outcome = TradePlanBuilder(risk_manager=LegacyAdapter(),
        ontology_policy_resolver=lambda **kw: _policy(symbol=kw["symbol"])).build(_request(), now=NOW)
    assert outcome.plan is None
    assert outcome.no_trade.reason_codes == ("ONTOLOGY_AUTHORITY_RECEIPT_MISSING",)


@pytest.mark.parametrize("changes", [dict(liquidity_score=float("nan")), dict(spread_rate=float("nan")),
    dict(realized_volatility=float("inf")), dict(liquidity_score="invalid")])
def test_profitability_rejects_invalid_measurements_instead_of_clamping_to_good_quality(changes):
    request = ProfitabilityInput(symbol="005930", entry_price=10000, expected_exit_price=10600, **changes)
    for policy in (None, _policy()):
        result = ProfitabilityGate().evaluate(request, ontology_policy=policy, now=NOW)
        assert not result.allowed
        assert "INVALID_ORDER_SIZE_OR_PRICE" in result.rejection_reasons


def test_policy_cannot_erase_current_worse_spread_liquidity_or_requested_floor():
    request = ProfitabilityInput(symbol="005930", entry_price=10000, expected_exit_price=10600,
                                 spread_rate=.02, liquidity_score=.01, target_net_return=.08)
    result = ProfitabilityGate().evaluate(request, ontology_policy=_policy(), now=NOW)
    assert not result.allowed
    assert result.spread_rate >= .02
    assert result.required_min_net_return >= .08

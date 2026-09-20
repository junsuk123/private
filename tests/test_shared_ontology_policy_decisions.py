from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.ontology.policy_evidence import PolicyObservation, project_market_evidence
from app.risk.ontology_thresholds import resolve_ontology_policy
from app.schemas.domain import AccountSnapshot, Holding, OrderSide
from app.trading.dynamic_exit_policy import DynamicExitPolicy
from app.trading.shared_decision_engine import SharedLiveDecisionEngine

NOW = datetime(2026, 9, 20, 1, tzinfo=timezone.utc)


def policy(now=NOW, **overrides):
    values = dict(realized_volatility=.003, spread_rate=.0001, liquidity_score=.9,
                  quote_age_seconds=.1, regime_confidence=.9, data_quality_score=.9,
                  market_breadth=.4, trend_strength=.5)
    obs = tuple(PolicyObservation(metric=k, value=v, market="KR", observed_at=now,
                source="kis_realtime", max_age_seconds=30, unit="seconds" if k == "quote_age_seconds" else "ratio",
                horizon_seconds=60 if k == "realized_volatility" else None) for k, v in values.items())
    projection = project_market_evidence("KR", as_of=now, observations=obs, regime="TREND_UP", context_id="typed-market")
    p = resolve_ontology_policy(projection, symbol="005930", all_in_cost_rate=.001)
    assert p.valid_for_entry, p.reason_codes
    return replace(p, **overrides)


class Store:
    def __init__(self, price):
        self.tick = SimpleNamespace(price=price, received_at=NOW, exchange_timestamp=NOW, sequence_key="tick-1")

    def latest_tick(self, symbol):
        return self.tick


def engine(price, resolver):
    result = SharedLiveDecisionEngine(Store(price), ontology_policy_resolver=resolver)
    result._technical_exit_deterioration = lambda *_: ((), 0.0)
    return result


def position(price=100_000.0, opened_at=NOW - timedelta(seconds=60)):
    return Holding(ticker="005930", market="KR", company_name="Samsung", sector="Tech", quantity=1,
                   average_price=100_000.0, last_price=price, opened_at=opened_at)


def test_policy_levels_override_legacy_loss_disable_and_absolute_profit_settings(monkeypatch):
    monkeypatch.setenv("REALTIME_HARD_STOP_LOSS", "0.9")
    monkeypatch.setenv("REALTIME_ALLOW_LOSS_EXIT", "false")
    monkeypatch.setenv("REALTIME_BLOCK_SELL_BELOW_BREAKEVEN", "true")
    p = policy(hard_stop_rate=.012, soft_stop_rate=.008, emergency_stop_rate=.02)
    resolved = DynamicExitPolicy().resolve(all_in_cost_rate=.001, ontology_policy=p)
    assert resolved.hard_stop_rate == .012
    assert resolved.allow_loss_exit and not resolved.block_sell_below_breakeven


def test_dynamic_hard_stop_closes_one_share_even_with_legacy_loss_blocks(monkeypatch):
    monkeypatch.setenv("REALTIME_ALLOW_LOSS_EXIT", "false")
    monkeypatch.setenv("REALTIME_BLOCK_SELL_BELOW_BREAKEVEN", "true")
    monkeypatch.setenv("REALTIME_SMALL_ACCOUNT_MODE", "true")
    monkeypatch.setenv("REALTIME_BLOCK_ONE_SHARE_LOSS_REDUCE", "true")
    p = policy(hard_stop_rate=.012, emergency_stop_rate=.025)
    e = engine(98_500, lambda **_: p)
    h = position(98_500)
    result = e.evaluate_exit_for_holding(h, AccountSnapshot(cash=0, holdings=(h,)), decision_time=NOW)
    assert result.approved, result.reason_codes
    assert result.final_order.side is OrderSide.SELL and result.final_order.quantity == 1
    assert result.diagnostics["exit_reason"] == "ontology_hard_stop"


def test_tiny_currency_profit_does_not_override_market_generated_target(monkeypatch):
    monkeypatch.setenv("REALTIME_TAKE_PROFIT_AMOUNT_KRW", "1")
    p = policy(target_return_rate=.03, net_profit_floor_rate=.01, maximum_holding_seconds=900)
    e = engine(100_700, lambda **_: p)
    h = position(100_700)
    result = e.evaluate_exit_for_holding(h, AccountSnapshot(cash=0, holdings=(h,)), decision_time=NOW)
    assert not result.approved and result.reason_codes == ("HOLD_ONTOLOGY_POLICY",)


def test_policy_staleness_retains_tighter_barrier_and_flat_cycle_resets_it():
    p = policy(hard_stop_rate=.01, emergency_stop_rate=.015, maximum_holding_seconds=900)
    holder = [p]
    e = engine(100_000, lambda **_: holder[0])
    h = position()
    account = AccountSnapshot(cash=0, holdings=(h,))
    e.evaluate_exit_for_holding(h, account, decision_time=NOW)
    holder[0] = replace(p, valid_for_entry=False, hard_stop_rate=.08, emergency_stop_rate=.10)
    e.store.tick.price = 98_800
    result = e.evaluate_exit_for_holding(h, account, decision_time=NOW + timedelta(seconds=1))
    assert result.approved, result.reason_codes
    assert result.diagnostics["ontology_risk_policy"]["hard_stop_rate"] == .01
    assert not result.diagnostics["ontology_policy_current_evidence"]
    e.sync_position_policy_state(AccountSnapshot(cash=0, holdings=()))
    assert not e._holding_ontology_policies and not e._ontology_peak_net


def test_resolver_failure_keeps_exit_only_safety_ceiling():
    def broken(**_):
        raise RuntimeError("unavailable")
    e = engine(96_000, broken)
    h = position(96_000)
    result = e.evaluate_exit_for_holding(h, AccountSnapshot(cash=0, holdings=(h,)), decision_time=NOW)
    assert result.approved, result.reason_codes
    assert result.diagnostics["ontology_risk_policy"]["valid_for_entry"] is False


@pytest.mark.parametrize("bad_price", [float("nan"), float("inf"), -1.0])
def test_exit_rejects_nonfinite_or_nonpositive_quote_before_cost_calculation(bad_price):
    e = engine(bad_price, lambda **_: policy())
    e._exit_price_source = lambda *_: (bad_price, NOW, NOW, "invalid")
    h = position()
    result = e.evaluate_exit_for_holding(h, AccountSnapshot(cash=0, holdings=(h,)), decision_time=NOW)
    assert not result.approved and result.reason_codes == ("MISSING_MARKET_DATA",)


def test_repeated_quote_does_not_count_as_multiple_early_exit_confirmations():
    p = policy(soft_stop_rate=.005, hard_stop_rate=.03, emergency_stop_rate=.04,
               early_exit_confirmations=2, minimum_holding_seconds=0, maximum_holding_seconds=900)
    e = engine(99_000, lambda **_: p)
    h = position(99_000)
    account = AccountSnapshot(cash=0, holdings=(h,))
    first = e.evaluate_exit_for_holding(h, account, decision_time=NOW)
    second = e.evaluate_exit_for_holding(h, account, decision_time=NOW + timedelta(seconds=1))
    assert not first.approved and not second.approved
    e.store.tick.received_at = e.store.tick.exchange_timestamp = NOW + timedelta(seconds=2)
    third = e.evaluate_exit_for_holding(h, account, decision_time=NOW + timedelta(seconds=2))
    assert third.approved, third.reason_codes
    assert third.diagnostics["exit_reason"] == "ontology_soft_stop"


def test_fresh_entry_rejection_still_tightens_owned_position():
    calm = policy(hard_stop_rate=.02, emergency_stop_rate=.03, maximum_holding_seconds=900)
    holder = [calm]
    e = engine(100_000, lambda **_: holder[0])
    h = position()
    account = AccountSnapshot(cash=0, holdings=(h,))
    e.evaluate_exit_for_holding(h, account, decision_time=NOW)
    holder[0] = replace(calm, policy_id="adverse-market", valid_for_entry=False,
                        reason_codes=("POLICY_SPREAD_TOO_WIDE",), hard_stop_rate=.005)
    e.store.tick.price = 99_000
    result = e.evaluate_exit_for_holding(h, account, decision_time=NOW + timedelta(seconds=1))
    assert result.approved, result.reason_codes
    assert result.diagnostics["ontology_policy_current_evidence"]
    assert result.diagnostics["ontology_risk_policy"]["policy_id"] == "adverse-market"
    assert result.diagnostics["exit_reason"] == "ontology_hard_stop"


def test_frozen_plan_cannot_bypass_missing_current_policy():
    e = engine(100_000, lambda **_: None)
    plan = SimpleNamespace(executable=lambda _: (True, None),
                           entry_rule=SimpleNamespace(price_permitted=lambda _: True))
    result = e._plan_driven_buy(symbol="005930", plan=plan, price=100_000, market_name="KR",
        prediction=None, technical_prediction=None, quote_refresh_status="ok", quote_age_seconds=0,
        spread_bps=1, orderbook=None, decision_time=NOW)
    assert not result.approved and result.reason_codes == ("ONTOLOGY_AUTHORITY_RECEIPT_MISSING",)


def test_frozen_plan_cannot_recreate_approval_by_running_risk_again():
    p = policy(position_cap=.15)
    e = engine(10_000, lambda **_: p)
    seen = []
    def reject(*args, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(approved=False, final_order=None, rejection_reasons=("current_risk_rejected",), metadata={})
    e.risk_manager.validate = reject
    plan = SimpleNamespace(executable=lambda _: (True, None), quantity=1,
                           entry_rule=SimpleNamespace(price_permitted=lambda _: True))
    h = position(10_000)
    market = e._exit_market_snapshot(h, 10_000, NOW, NOW)
    result = e._plan_driven_buy(symbol="005930", plan=plan, price=10_000, market_name="KR",
        prediction=None, technical_prediction=None, quote_refresh_status="ok", quote_age_seconds=0,
        spread_bps=1, orderbook=None, decision_time=NOW, ontology_policy=p,
        account=AccountSnapshot(cash=1_000_000, holdings=()), market=market)
    assert not result.approved and result.reason_codes == ("ONTOLOGY_AUTHORITY_RECEIPT_MISSING",)
    assert seen == []


def _approved_plan():
    from app.trading.trade_plan_builder import PlanRequest, TradePlanBuilder
    from app.schemas.domain import MarketSnapshot, SourceMetadata
    p = policy()
    market = MarketSnapshot("005930", "KR", "Samsung", "Tech", 10_000., 1e10, .02,
        SourceMetadata("KIS realtime WebSocket", NOW, observed_at=NOW, source_type="broker_api",
                       trust_level=5, is_realtime=True, quality_score=1.))
    account = AccountSnapshot(cash=10_000_000., holdings=())
    builder = TradePlanBuilder(ontology_policy_resolver=lambda **_: p)
    outcome = builder.build(PlanRequest(symbol="005930", strategy_id="breakout_volume", market="KR",
        account=account, market_snapshot=market, reference_price=10_000., take_profit_rate=.04,
        stop_loss_rate=.01, trailing_rate=.005, max_holding_seconds=900, gross_edge_bps=400,
        confidence=.9, source_ids=("live-quote",)), now=NOW)
    assert outcome.plan is not None, outcome.no_trade
    return outcome.plan, account


def test_approved_plan_is_consumed_without_resolver_profitability_sizing_or_risk_recheck(monkeypatch):
    plan, account = _approved_plan()
    def forbidden(*_, **__):
        pytest.fail("A frozen ontology decision must not be economically assessed again")
    e = engine(10_000, forbidden)
    e.risk_manager.validate = forbidden
    e.profitability_gate.evaluate = forbidden
    e.position_sizer.size = forbidden
    e.auto_tuner.build_buy_policy = forbidden
    e.feature_builder.build = forbidden
    e.predictor.predict = forbidden
    result = e.evaluate_buy("005930", account, selected_strategy="breakout_volume",
                            trade_plan=plan, decision_time=NOW)
    assert result.approved, result.reason_codes
    assert result.final_order.quantity == plan.quantity
    assert result.diagnostics["execution_authority"] == "ONTOLOGY"
    assert result.diagnostics["post_selection_gates"] == []
    # A partial fill consumes only the unfilled part of the same approval.
    partial = replace(plan, filled_quantity=1)
    result = e.evaluate_buy("005930", account, selected_strategy="breakout_volume",
                            trade_plan=partial, decision_time=NOW)
    assert result.approved and result.final_order.quantity == plan.quantity - 1


def test_expired_ontology_plan_cannot_reach_execution():
    plan, account = _approved_plan()
    e = engine(10_000, lambda **_: pytest.fail("Expired plan must return to election"))
    result = e.evaluate_buy("005930", account, trade_plan=plan,
                            decision_time=plan.expires_at + timedelta(microseconds=1))
    assert not result.approved and "PLAN_EXPIRED" in result.reason_codes


@pytest.mark.parametrize("trade_count,approved", [(0, True), (100, False), (None, False)])
def test_unplanned_entry_uses_one_ontology_assessment_with_real_activity(trade_count, approved):
    p = policy()
    e = engine(10_000, lambda **_: p)
    e.entry_activity_provider = lambda market, now: trade_count
    e.feature_builder.build = lambda *args, **kwargs: None
    e.predictor.predict = lambda frame: SimpleNamespace(approved=True, expected_net_return_bps=500.,
                                                        probability_success=.9)
    e._technical_prediction = lambda *args, **kwargs: None
    def forbidden(*args, **kwargs):
        pytest.fail("Ontology entry must not invoke a legacy economic gate")
    e.profitability_gate.evaluate = forbidden
    e.position_sizer.size = forbidden
    e.auto_tuner.build_buy_policy = forbidden
    original = e.risk_manager.validate
    calls = []
    def assess(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    e.risk_manager.validate = assess
    result = e.evaluate_buy("005930", AccountSnapshot(cash=1_000_000., holdings=()), decision_time=NOW)
    assert result.approved is approved, result.reason_codes
    assert len(calls) == 1 and calls[0]["trades_today"] == trade_count
    assert result.diagnostics["risk_metadata"]["ontology_risk_authority"]["approved"] is approved

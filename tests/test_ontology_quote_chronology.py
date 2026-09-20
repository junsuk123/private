from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest

from app.schemas.domain import AccountSnapshot, MarketSnapshot, SourceMetadata
from test_shared_ontology_policy_decisions import NOW, engine, policy, position, _approved_plan


def quote(price, when):
    return MarketSnapshot(ticker="005930", market="KR", company_name="Samsung", sector="Tech",
        last_price=price, average_daily_trading_value=1e10, volatility_20d=.02,
        source=SourceMetadata("KIS REST quote", when, observed_at=when,
            source_type="broker_api", trust_level=5, is_realtime=True, quality_score=1.))


def prepare_buy(e):
    e.store.tick = None
    e.feature_builder.build = lambda *args, **kwargs: None
    e.predictor.predict = lambda _: SimpleNamespace(approved=True, expected_net_return_bps=500., probability_success=.9)
    e._technical_prediction = lambda *args, **kwargs: None


def test_post_fetch_quote_uses_trusted_completion_time_for_single_authority():
    observed = NOW + timedelta(seconds=1)
    completed = observed + timedelta(milliseconds=100)
    resolved_times = []

    def resolve(**kwargs):
        resolved_times.append(kwargs["now"])
        return policy(now=kwargs["now"])

    e = engine(10_000, resolve)
    prepare_buy(e)
    e.market_refresher = lambda *args: quote(10_000, observed)
    e.decision_clock = lambda: completed
    result = e.evaluate_buy("005930", AccountSnapshot(cash=1_000_000, holdings=(), captured_at=NOW), decision_time=NOW)
    assert result.approved, result.reason_codes
    assert resolved_times == [completed]
    assert result.diagnostics["risk_metadata"]["ontology_risk_authority"]["evaluated_at"] == completed.isoformat()


def test_incoming_future_quote_cannot_set_its_own_evaluation_clock():
    completed = NOW + timedelta(seconds=1)
    e = engine(10_000, lambda **kwargs: pytest.fail("Future source must not reach economic assessment"))
    prepare_buy(e)
    e.market_refresher = lambda *args: quote(10_000, completed + timedelta(seconds=1))
    e.decision_clock = lambda: completed
    result = e.evaluate_buy("005930", AccountSnapshot(cash=1_000_000, holdings=(), captured_at=NOW), decision_time=NOW)
    assert not result.approved
    assert result.reason_codes == ("QUOTE_FROM_FUTURE",)


def test_refresh_cannot_execute_a_plan_that_expired_during_fetch():
    plan, account = _approved_plan()
    completed = plan.expires_at + timedelta(seconds=1)
    e = engine(10_000, lambda **kwargs: pytest.fail("Frozen plan must not be reassessed"))
    e.store.tick = None
    e.market_refresher = lambda *args: quote(10_000, completed)
    e.decision_clock = lambda: completed
    result = e.evaluate_buy("005930", account, trade_plan=plan, decision_time=NOW)
    assert not result.approved
    assert result.reason_codes == ("PLAN_EXPIRED",)


def test_stale_balance_mark_retains_account_timestamp_and_cannot_authorize_exit():
    h = position(97_000)
    captured = NOW - timedelta(minutes=5)
    account = AccountSnapshot(cash=0, holdings=(h,), captured_at=captured)
    e = engine(97_000, lambda **kwargs: policy(now=kwargs["now"]))
    e.store.tick = None
    assert e._exit_price_source(h.ticker, h, NOW, account) == (97_000, captured, captured, "balance:005930")
    result = e.evaluate_exit_for_holding(h, account, decision_time=NOW)
    assert not result.approved
    assert "quote_freshness_check" in result.reason_codes
    assert result.diagnostics["risk_metadata"]["quote_age_seconds"] == 300


@pytest.mark.parametrize("balance_price", [0, 97_000])
def test_exit_missing_price_and_stale_price_refresh_both_advance_clock(balance_price):
    h = position(balance_price)
    observed = NOW + timedelta(seconds=1)
    completed = observed + timedelta(milliseconds=100)
    account = AccountSnapshot(cash=0, holdings=(h,), captured_at=NOW-timedelta(minutes=5))
    e = engine(balance_price, lambda **kwargs: policy(now=kwargs["now"]))
    e.store.tick = None
    e.market_refresher = lambda *args: quote(97_000, observed)
    e.decision_clock = lambda: completed
    result = e.evaluate_exit_for_holding(h, account, decision_time=NOW)
    assert result.approved, result.reason_codes
    receipt = result.diagnostics["risk_metadata"]["ontology_risk_authority"]
    assert receipt["evaluated_at"] == completed.isoformat()
    assert result.diagnostics["risk_metadata"]["quote_age_seconds"] == pytest.approx(.1)


def test_exit_refresh_with_genuinely_future_source_still_fails():
    h = position(97_000)
    completed = NOW+timedelta(seconds=1)
    account = AccountSnapshot(cash=0, holdings=(h,), captured_at=NOW-timedelta(minutes=5))
    e = engine(97_000, lambda **kwargs: policy(now=kwargs["now"]))
    e.store.tick = None
    e.market_refresher = lambda *args: quote(97_000, completed+timedelta(seconds=1))
    e.decision_clock = lambda: completed
    result = e.evaluate_exit_for_holding(h, account, decision_time=NOW)
    assert not result.approved
    assert "quote_freshness_check" in result.reason_codes


def test_replay_without_wall_clock_advances_only_by_monotonic_elapsed(monkeypatch):
    from app.trading import shared_decision_engine
    e = engine(10_000, lambda **kwargs: policy())
    monkeypatch.setattr(shared_decision_engine.time, "monotonic", lambda: 102.)
    assert e._post_refresh_time(NOW, 100.) == NOW + timedelta(seconds=2)


def test_async_tick_arriving_during_buy_cycle_uses_post_read_live_clock():
    observed = NOW + timedelta(milliseconds=100)
    completed = NOW + timedelta(milliseconds=200)
    resolved = []

    def resolve(**kwargs):
        resolved.append(kwargs["now"])
        return policy(now=kwargs["now"])

    e = engine(10_000, resolve)
    prepare_buy(e)
    e.store.tick = SimpleNamespace(price=10_000, received_at=observed, exchange_timestamp=observed)
    e.decision_clock = lambda: completed
    result = e.evaluate_buy("005930", AccountSnapshot(cash=1_000_000, holdings=(), captured_at=NOW), decision_time=NOW)
    assert result.approved, result.reason_codes
    assert resolved == [completed]
    e.store.tick.exchange_timestamp = completed+timedelta(seconds=1)
    future = e.evaluate_buy("005930", AccountSnapshot(cash=1_000_000, holdings=(), captured_at=NOW), decision_time=NOW)
    assert future.reason_codes == ("QUOTE_FROM_FUTURE",)


def test_async_tick_arriving_during_exit_cycle_uses_post_read_live_clock():
    observed = NOW + timedelta(milliseconds=100)
    completed = NOW + timedelta(milliseconds=200)
    h = position(97_000)
    account = AccountSnapshot(cash=0, holdings=(h,), captured_at=NOW)
    e = engine(97_000, lambda **kwargs: policy(now=kwargs["now"]))
    e.store.tick.received_at = e.store.tick.exchange_timestamp = observed
    e.decision_clock = lambda: completed
    result = e.evaluate_exit_for_holding(h, account, decision_time=NOW)
    assert result.approved, result.reason_codes
    assert result.diagnostics["risk_metadata"]["ontology_risk_authority"]["evaluated_at"] == completed.isoformat()
    assert result.diagnostics["quote_age_seconds"] == pytest.approx(.1)


def test_no_injected_live_clock_does_not_accept_tick_after_explicit_replay_asof():
    e = engine(10_000, lambda **kwargs: pytest.fail("A replay cannot use future ticks"))
    prepare_buy(e)
    observed = NOW+timedelta(milliseconds=100)
    e.store.tick = SimpleNamespace(price=10_000, received_at=observed, exchange_timestamp=observed)
    result = e.evaluate_buy("005930", AccountSnapshot(cash=1_000_000, holdings=(), captured_at=NOW), decision_time=NOW)
    assert result.reason_codes == ("QUOTE_FROM_FUTURE",)

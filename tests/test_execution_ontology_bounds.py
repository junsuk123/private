"""Final broker pricing consumes a grant; it cannot enlarge that grant."""
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.execution.execution_guard import ExecutionGuard, GuardOrder
from app.ontology.decision_receipt import validate_plan_authority
from test_ontology_decision_receipt import NOW, _approved_plan


def _order(plan, **overrides):
    values = dict(symbol=plan.symbol, market=plan.market, side="BUY",
                  quantity=plan.quantity, limit_price=1000, direction="LONG",
                  position_effect="OPEN", execution_product="CASH")
    values.update(overrides)
    return GuardOrder(**values)


def _guard():
    return ExecutionGuard(kill_switch_provider=lambda: False)


def test_final_approval_bounds_allow_broker_clip_and_remaining_fill(tmp_path):
    plan = _approved_plan(tmp_path)
    clipped = _guard().evaluate(_order(plan), plan=plan, now=NOW, orderable_cash=20_000)
    assert clipped.allowed and clipped.clipped
    assert 0 < clipped.permitted_quantity < plan.quantity
    partial = plan.with_entry_fill(995, 10)
    allowed = _guard().evaluate(
        _order(partial, quantity=partial.remaining_quantity), plan=partial, now=NOW,
        orderable_cash=1e9,
    )
    assert allowed.allowed
    assert "ontology_authority_bounds" in allowed.checked


@pytest.mark.parametrize("price", [1001, 989, float("nan"), float("inf")])
def test_final_price_must_remain_in_original_band(tmp_path, price):
    plan = _approved_plan(tmp_path)
    assert validate_plan_authority(plan, NOW, price=1000) == ()
    decision = _guard().evaluate(_order(plan, limit_price=price), plan=plan, now=NOW,
                                 orderable_cash=1e9)
    assert not decision.allowed
    assert "ONTOLOGY_AUTHORITY_PRICE_OUTSIDE_APPROVAL" in decision.reason_codes


@pytest.mark.parametrize("cash", [1e9, 20_000])
def test_requested_quantity_cannot_exceed_remaining_even_when_cash_would_clip(tmp_path, cash):
    plan = _approved_plan(tmp_path).with_entry_fill(1000, 10)
    decision = _guard().evaluate(_order(plan), plan=plan, now=NOW, orderable_cash=cash)
    assert not decision.allowed
    assert "ONTOLOGY_AUTHORITY_QUANTITY_EXCEEDED" in decision.reason_codes


def test_quantity_and_cumulative_notional_cannot_expand_approval(tmp_path):
    plan = _approved_plan(tmp_path)
    decision = _guard().evaluate(_order(plan, quantity=plan.quantity + 1), plan=plan,
                                 now=NOW, orderable_cash=1e9)
    assert "ONTOLOGY_AUTHORITY_QUANTITY_EXCEEDED" in decision.reason_codes
    narrow = replace(plan, max_notional=plan.max_notional - 1).with_entry_fill(1000, 10)
    decision = _guard().evaluate(_order(narrow, quantity=narrow.remaining_quantity),
                                 plan=narrow, now=NOW, orderable_cash=1e9)
    assert not decision.allowed
    assert "ONTOLOGY_AUTHORITY_PRICE_OR_NOTIONAL_EXCEEDED" in decision.reason_codes


@pytest.mark.parametrize("change", [
    {"side": "SELL"}, {"direction": "SHORT"}, {"execution_product": "MARGIN"},
    {"position_effect": "CLOSE"}, {"position_effect": "INVALID"},
])
def test_final_order_contract_cannot_differ_or_fake_an_exit(tmp_path, change):
    plan = _approved_plan(tmp_path)
    decision = _guard().evaluate(_order(plan, **change), plan=plan, now=NOW,
                                 orderable_cash=1e9, sellable_quantity=plan.quantity)
    assert not decision.allowed
    assert "ONTOLOGY_AUTHORITY_CONTRACT_MISMATCH" in decision.reason_codes


def test_policy_without_receipt_fails_closed_but_legacy_plan_is_preserved(tmp_path):
    plan = _approved_plan(tmp_path)
    policy_only = replace(plan, risk_snapshot={"ontology_risk_policy": plan.risk_snapshot["ontology_risk_policy"]})
    decision = _guard().evaluate(_order(plan), plan=policy_only, now=NOW, orderable_cash=1e9)
    assert "ONTOLOGY_AUTHORITY_RECEIPT_MISSING" in decision.reason_codes
    legacy = replace(plan, risk_snapshot={})
    assert _guard().evaluate(_order(legacy, limit_price=1001), plan=legacy, now=NOW,
                             orderable_cash=1e9).allowed


@pytest.mark.parametrize("sellable", [None, float("nan"), float("inf"), 0, -1])
def test_expired_grant_does_not_prove_there_is_a_position_to_close(tmp_path, sellable):
    plan = _approved_plan(tmp_path)
    decision = _guard().evaluate(_order(plan, side="SELL", position_effect="CLOSE"),
                                 plan=plan, now=plan.expires_at + timedelta(hours=1),
                                 sellable_quantity=sellable)
    assert not decision.allowed
    assert any("SELLABLE" in reason for reason in decision.reason_codes)
    assert "ONTOLOGY_AUTHORITY_EXPIRED_OR_FUTURE" not in decision.reason_codes


def test_held_position_can_exit_after_entry_receipt_expires_at_any_exit_price(tmp_path):
    plan = _approved_plan(tmp_path)
    decision = _guard().evaluate(_order(plan, side="SELL", position_effect="CLOSE", limit_price=800),
                                 plan=plan, now=plan.expires_at + timedelta(hours=1),
                                 sellable_quantity=plan.quantity)
    assert decision.allowed


def test_quote_approved_then_ask_repriced_outside_band_never_reaches_broker(tmp_path, monkeypatch):
    from app.data.realtime_types import KIS_REALTIME_SOURCE
    from app.execution.idempotency_store import IdempotencyStore
    from app.execution.kis_errors import LiveExecutionBlocked
    from app.execution.live_execution_coordinator import LiveExecutionCoordinator
    from app.execution.order_pricing_policy import ExecutionPricingPolicy
    from app.schemas.domain import FinalOrder, OrderSide, OrderType
    from app.trading.realtime_trading_engine import RealtimeTradingEngine

    plan = _approved_plan(tmp_path)
    assert validate_plan_authority(plan, NOW, price=1000) == ()
    order = FinalOrder(ticker=plan.symbol, market=plan.market, order_type=OrderType.LIMIT,
                       side=OrderSide.BUY, quantity=plan.quantity, limit_price=1000,
                       position_effect="OPEN")
    monkeypatch.setenv("EXEC_PASSIVE_ENTRY", "false")
    monkeypatch.setenv("EXEC_BUY_MAX_CHASE_BPS", "20")
    book = SimpleNamespace(best_bid=1000, best_ask=1001, total_bid_volume=10000,
                           total_ask_volume=10000, received_at=NOW, source=KIS_REALTIME_SOURCE)
    stub = SimpleNamespace(
        decision_engine=SimpleNamespace(store=SimpleNamespace(latest_orderbook=lambda _: book)),
        _latest_orderbook=lambda _: book, _last_failed_entry_price={}, _live_mode=lambda: False,
        exchange_resolver=SimpleNamespace(resolve=lambda *args, **kwargs: SimpleNamespace(
            allowed=True, source="fixture", exchange="KR", confidence=1,
        )), pricing_policy=ExecutionPricingPolicy(), _record=Mock(),
    )
    priced, okay, reason, _ = RealtimeTradingEngine._prepare_order_for_execution(
        stub, plan.symbol, "BUY", order, {}, (), None, NOW, plan_owned=True,
    )
    assert okay and reason == "EXEC_OK" and priced.limit_price == 1001
    broker = SimpleNamespace(place_limit_order=Mock())
    coordinator = LiveExecutionCoordinator(
        broker, idempotency_store=IdempotencyStore(tmp_path / "idempotency.jsonl"),
        journal=SimpleNamespace(record=Mock()),
        execution_config=SimpleNamespace(idempotency_ttl_seconds=60), execution_guard=_guard(),
        plan_provider=lambda _: plan, orderable_cash_provider=lambda _: 1e9,
    )
    monkeypatch.setattr(coordinator, "_preflight_failures", lambda: [])
    monkeypatch.setattr("app.execution.execution_guard._utcnow", lambda: NOW)
    with pytest.raises(LiveExecutionBlocked) as caught:
        coordinator.submit_final_order(priced, idempotency_key="offline-reprice")
    assert "ONTOLOGY_AUTHORITY_PRICE_OUTSIDE_APPROVAL" in caught.value.reason_codes
    broker.place_limit_order.assert_not_called()
    assert not coordinator.idempotency_store.path.exists()

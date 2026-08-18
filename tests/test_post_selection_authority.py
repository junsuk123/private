"""The acceptance criteria: what decides before election, and what may decide after.

The single claim under test is that **after a TradePlan exists, no authority re-judges
investment quality**. Each test below picks one of the three that used to, and shows it
either runs before election or does not run at all.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone

import pytest

from app.execution.execution_guard import (
    FORBIDDEN_INVESTMENT_CHECKS,
    ExecutionGuard,
    GuardOrder,
)
from app.schemas.domain import (
    AccountSnapshot,
    MarketSnapshot,
    OrderSide,
    OrderType,
    SourceMetadata,
)
from app.storage.trading_state_store import TradingStateStore
from app.trading.strategy_fast_executor import (
    EXIT_STOP_LOSS,
    EXIT_TAKE_PROFIT,
    FastLoopState,
    StrategyFastExecutor,
    TickEvent,
)
from app.trading.trade_plan import EntryRule, ExitRules, TradePlan, TradePlanStatus
from app.trading.trade_plan_builder import PlanRequest, TradePlanBuilder

NOW = datetime(2026, 8, 19, 1, 30, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _account(cash: float = 50_000_000.0, equity: float = 100_000_000.0) -> AccountSnapshot:
    return AccountSnapshot(
        cash=cash,
        holdings=(),
        realized_pnl_today=0.0,
        unrealized_pnl_today=0.0,
        total_equity_krw=equity,
        captured_at=NOW,
    )


def _market(price: float = 70_000.0, adtv: float = 5e11) -> MarketSnapshot:
    return MarketSnapshot(
        ticker="000660",
        market="KR",
        company_name="Example",
        sector="semiconductor",
        last_price=price,
        average_daily_trading_value=adtv,
        volatility_20d=0.02,
        source=SourceMetadata(
            source_name="kis_realtime",
            retrieved_at=NOW,
            source_type="broker_api",
            trust_level=5,
            is_realtime=True,
        ),
    )


def _request(**overrides) -> PlanRequest:
    base = dict(
        symbol="000660",
        strategy_id="intraday_momentum",
        market="KR",
        account=_account(),
        market_snapshot=_market(),
        reference_price=70_000.0,
        take_profit_rate=0.006,
        stop_loss_rate=0.003,
        trailing_rate=0.0015,
        max_holding_seconds=900,
        gross_edge_bps=95.0,
        confidence=0.7,
        liquidity_score=0.85,
        spread_bps=8.0,
        realized_volatility=0.0012,
        weekday_time_context={"day_of_week": "WED", "session_phase": "MORNING_TREND"},
        source_ids=("tick:1",),
    )
    base.update(overrides)
    return PlanRequest(**base)


def _plan(**overrides) -> TradePlan:
    outcome = TradePlanBuilder().build(_request(**overrides), now=NOW)
    assert outcome.plan is not None, outcome.as_dict()
    return outcome.plan


# --------------------------------------------------------------------------- #
# Everything decides before election
# --------------------------------------------------------------------------- #
def test_cost_size_and_risk_are_all_resolved_into_the_plan() -> None:
    plan = _plan()
    assert plan.cost_snapshot, "the profitability verdict must be frozen into the plan"
    assert plan.risk_snapshot["approved"] is True
    assert plan.risk_snapshot["sizing"], "sizing must be frozen into the plan"
    assert plan.quantity > 0
    assert plan.expected_net_edge_bps > 0.0
    assert plan.weekday_time_context["day_of_week"] == "WED"


def test_the_election_is_deterministic_on_identical_inputs() -> None:
    first = _plan()
    second = _plan()
    assert first.decision_fingerprint() == second.decision_fingerprint()
    assert first.quantity == second.quantity
    assert first.expected_net_edge_bps == second.expected_net_edge_bps


def test_a_candidate_whose_edge_does_not_clear_cost_never_becomes_a_plan() -> None:
    outcome = TradePlanBuilder().build(_request(gross_edge_bps=3.0), now=NOW)
    assert outcome.plan is None
    assert outcome.no_trade is not None
    assert outcome.no_trade.stage == "profitability"
    assert "PRE_ELECTION_NET_EDGE_INSUFFICIENT" in outcome.no_trade.reason_codes
    # NO_TRADE carries the same evidence a plan would.
    assert outcome.no_trade.cost_snapshot


def test_a_candidate_the_risk_layer_refuses_never_becomes_a_plan() -> None:
    """An account with no cash cannot fund the elected size, and that is settled here."""
    outcome = TradePlanBuilder().build(
        _request(account=_account(cash=0.0, equity=1000.0)), now=NOW
    )
    assert outcome.plan is None
    assert outcome.no_trade is not None
    assert outcome.no_trade.stage in {"risk", "sizing"}


def test_an_unauthorised_arm_never_becomes_a_plan() -> None:
    outcome = TradePlanBuilder().build(_request(authority_size_fraction=0.0), now=NOW)
    assert outcome.plan is None
    assert outcome.no_trade.reason_codes == ("AUTHORITY_NOT_ORDERABLE",)


def test_the_authority_cap_is_applied_once_at_election() -> None:
    full = _plan(authority_size_fraction=1.0)
    half = _plan(authority_size_fraction=0.5)
    assert half.quantity <= full.quantity
    assert half.risk_snapshot["authority_size_fraction"] == 0.5


# --------------------------------------------------------------------------- #
# Nothing re-judges afterwards
# --------------------------------------------------------------------------- #
def test_the_plan_driven_buy_path_calls_no_post_selection_authority() -> None:
    """``_plan_driven_buy`` is the whole post-election decision path. It must not
    reference the three authorities the refactor moved upstream."""
    from app.trading.shared_decision_engine import SharedLiveDecisionEngine

    source = inspect.getsource(SharedLiveDecisionEngine._plan_driven_buy)
    for forbidden in (
        "profitability_gate.evaluate",
        "position_sizer.size",
        "risk_manager.validate",
        "RiskManager(",
    ):
        assert forbidden not in source, forbidden


def test_the_fast_executor_cannot_reach_a_decision_authority() -> None:
    """Structural, not advisory: no import path from the fast loop to the slow layers."""
    import app.trading.strategy_fast_executor as module

    text = open(module.__file__, encoding="utf-8").read()  # type: ignore[arg-type]
    for forbidden in (
        "app.ontology",
        "app.graph",
        "app.models",
        "app.cost",
        "app.risk",
        "ProfitabilityGate",
        "PositionSizer",
        "RiskManager",
    ):
        assert f"import {forbidden}" not in text, forbidden
        assert f"from {forbidden}" not in text, forbidden


def test_the_fast_executor_does_no_io() -> None:
    import app.trading.strategy_fast_executor as module

    text = open(module.__file__, encoding="utf-8").read()  # type: ignore[arg-type]
    for forbidden in ("sqlite3", "requests", "httpx", "RealtimeMarketDataStore", "fetch_all"):
        assert forbidden not in text, forbidden


def test_an_elected_entry_reaches_execution_prep_without_re_review() -> None:
    """The end-state: entry trigger fires, an order request comes out, and the only
    thing between it and the broker is the technical guard."""
    plan = _plan()
    executor = StrategyFastExecutor(plan)
    request = executor.on_tick(
        TickEvent(symbol="000660", price=70_010.0, event_time=NOW), now=NOW
    )
    assert request is not None
    assert request.action == "SUBMIT_ENTRY"
    assert request.quantity == plan.quantity, "the elected size is submitted unchanged"
    assert executor.state is FastLoopState.ENTERING


# --------------------------------------------------------------------------- #
# The sell policy
# --------------------------------------------------------------------------- #
def _opened(plan: TradePlan) -> StrategyFastExecutor:
    executor = StrategyFastExecutor(plan)
    executor.on_tick(TickEvent(symbol=plan.symbol, price=70_000.0, event_time=NOW), now=NOW)
    executor.on_entry_fill(70_000.0, plan.quantity, now=NOW)
    return executor


def test_a_stop_loss_exits_immediately_while_deeply_unprofitable() -> None:
    """The core sell-policy guarantee: a losing position is never held because closing
    it would realise a loss."""
    plan = _plan()
    executor = _opened(plan)
    request = executor.on_tick(
        TickEvent(symbol="000660", price=69_000.0, event_time=NOW), now=NOW
    )
    assert request is not None
    assert request.action == "SUBMIT_EXIT"
    assert request.reason == EXIT_STOP_LOSS
    assert request.urgent


def test_the_stop_is_evaluated_before_the_profitable_exits() -> None:
    """A tick that breaches the stop must not be handled as a trailing update."""
    plan = _plan()
    executor = _opened(plan)
    # A price below the stop is also below any trailing level; the stop must win.
    request = executor.on_tick(
        TickEvent(symbol="000660", price=60_000.0, event_time=NOW), now=NOW
    )
    assert request is not None and request.reason == EXIT_STOP_LOSS


def test_a_take_profit_exits_without_re_checking_profitability() -> None:
    plan = _plan()
    executor = _opened(plan)
    request = executor.on_tick(
        TickEvent(symbol="000660", price=70_500.0, event_time=NOW), now=NOW
    )
    assert request is not None
    assert request.reason == EXIT_TAKE_PROFIT
    assert not request.urgent


def test_a_time_exit_fires_on_the_plan_clock() -> None:
    plan = _plan()
    executor = _opened(plan)
    later = NOW + timedelta(seconds=plan.exit_rules.max_holding_seconds + 1)
    request = executor.on_tick(
        TickEvent(symbol="000660", price=70_050.0, event_time=later), now=later
    )
    assert request is not None and request.reason == "STRATEGY_TIME_EXIT"


def test_a_trailing_stop_only_arms_once_in_profit() -> None:
    plan = _plan()
    executor = _opened(plan)
    # Straight down from entry: this is the stop's job, not the trailing stop's.
    request = executor.on_tick(
        TickEvent(symbol="000660", price=69_950.0, event_time=NOW), now=NOW
    )
    assert request is None
    # Run up, then give back more than the trailing rate.
    executor.on_tick(TickEvent(symbol="000660", price=70_400.0, event_time=NOW), now=NOW)
    request = executor.on_tick(
        TickEvent(symbol="000660", price=70_250.0, event_time=NOW), now=NOW
    )
    assert request is not None and request.reason == "STRATEGY_TRAILING_STOP"
    assert request.urgent


def test_force_exit_bypasses_every_trigger() -> None:
    plan = _plan()
    executor = _opened(plan)
    request = executor.force_exit("EMERGENCY_FLATTEN", now=NOW)
    assert request is not None
    assert request.action == "SUBMIT_EXIT"
    assert request.urgent


# --------------------------------------------------------------------------- #
# The guard checks order-ability only
# --------------------------------------------------------------------------- #
def _guard(**overrides) -> ExecutionGuard:
    base = dict(pre_submit_guard=None, require_plan=False)
    base.update(overrides)
    return ExecutionGuard(**base)


def _order(**overrides) -> GuardOrder:
    base = dict(
        symbol="000660", market="KR", side="BUY", quantity=10, limit_price=70_000.0
    )
    base.update(overrides)
    return GuardOrder(**base)


def test_the_guard_never_emits_an_investment_reason() -> None:
    plan = _plan()
    for kwargs in (
        {},
        {"orderable_cash": 0.0},
        {"sellable_quantity": 0},
    ):
        decision = _guard().evaluate(_order(), plan=plan, **kwargs)
        for reason in decision.reason_codes:
            for forbidden in FORBIDDEN_INVESTMENT_CHECKS:
                assert forbidden not in reason.upper(), reason


def test_the_guard_source_contains_no_investment_vocabulary() -> None:
    import app.execution.execution_guard as module

    text = open(module.__file__, encoding="utf-8").read()  # type: ignore[arg-type]
    # The words appear in the module docstring and the FORBIDDEN list by design; what
    # must not appear is a call into the layers that compute them.
    for forbidden in (
        "ProfitabilityGate",
        "PositionSizer",
        "RiskManager",
        "profitability_gate",
        "position_sizer",
    ):
        assert forbidden not in text, forbidden


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"limit_price": 0.0}, "GUARD_INVALID_PRICE"),
        ({"quantity": 0}, "GUARD_INVALID_QUANTITY"),
        ({"symbol": "12345"}, "GUARD_UNSUPPORTED_INSTRUMENT"),
        ({"execution_product": "SWAP"}, "GUARD_UNSUPPORTED_INSTRUMENT"),
    ],
)
def test_technical_defects_block(kwargs: dict, expected: str) -> None:
    decision = _guard().evaluate(_order(**kwargs), plan=_plan())
    assert not decision.allowed
    assert expected in decision.reason_codes


def test_insufficient_cash_blocks_a_buy() -> None:
    decision = _guard().evaluate(_order(), plan=_plan(), orderable_cash=100.0)
    assert not decision.allowed
    assert "GUARD_INSUFFICIENT_CASH" in decision.reason_codes


def test_partial_cash_clips_rather_than_blocks() -> None:
    decision = _guard().evaluate(_order(quantity=10), plan=_plan(), orderable_cash=300_000.0)
    assert decision.allowed
    assert decision.clipped
    assert 0 < decision.permitted_quantity < 10


def test_unsellable_quantity_blocks_an_exit() -> None:
    decision = _guard().evaluate(
        _order(side="SELL", position_effect="CLOSE"), plan=_plan(), sellable_quantity=0
    )
    assert not decision.allowed
    assert "GUARD_INSUFFICIENT_SELLABLE_QUANTITY" in decision.reason_codes


def test_the_kill_switch_blocks_everything() -> None:
    decision = ExecutionGuard(
        pre_submit_guard=None, require_plan=False, kill_switch_provider=lambda: True
    ).evaluate(_order(), plan=_plan())
    assert not decision.allowed
    assert "GUARD_KILL_SWITCH_ENGAGED" in decision.reason_codes


def test_an_unhealthy_broker_blocks() -> None:
    class _Health:
        ok = False
        failures = ("credentials",)

    decision = ExecutionGuard(
        pre_submit_guard=None, require_plan=False, broker_health_provider=lambda: _Health()
    ).evaluate(_order(), plan=_plan())
    assert not decision.allowed
    assert "GUARD_BROKER_UNHEALTHY" in decision.reason_codes


def test_an_expired_plan_blocks_an_entry_but_not_an_exit() -> None:
    plan = _plan()
    late = NOW + timedelta(hours=2)
    entry = _guard().evaluate(_order(), plan=plan, orderable_cash=1e9, now=late)
    assert not entry.allowed
    assert any(code.startswith("GUARD_PLAN_NOT_EXECUTABLE") for code in entry.reason_codes)

    exit_decision = _guard().evaluate(
        _order(side="SELL", position_effect="CLOSE"),
        plan=plan,
        sellable_quantity=10,
        now=late,
    )
    assert exit_decision.allowed, "a real position must always be closeable"


def test_a_missing_plan_blocks_an_entry_when_plans_are_required() -> None:
    decision = ExecutionGuard(pre_submit_guard=None, require_plan=True).evaluate(
        _order(), orderable_cash=1e9
    )
    assert not decision.allowed
    assert "GUARD_PLAN_MISSING" in decision.reason_codes


def test_the_builder_refuses_a_short_on_this_long_only_account() -> None:
    """Short selling is disallowed by the risk rules, and that is settled at election."""
    outcome = TradePlanBuilder().build(_request(direction="SHORT"), now=NOW)
    assert outcome.plan is None
    assert outcome.no_trade is not None
    # A short's expected exit sits below its entry, so the cost model sizes it to nothing
    # long before the risk rules would have to refuse it. Either stage is a refusal at
    # election; what matters is that no short plan reaches the execution path.
    assert outcome.no_trade.stage in {"sizing", "risk", "profitability"}


def test_a_short_without_a_locate_cannot_be_sent() -> None:
    """Constructed directly: the builder will not produce a short on this account, and
    the check under test belongs to the guard rather than to the builder."""
    short_plan = TradePlan(
        plan_id="plan-SHORT-1",
        created_at=NOW,
        expires_at=NOW + timedelta(minutes=5),
        symbol="000660",
        market="KR",
        direction="SHORT",
        strategy_id="residual_relative_weakness",
        quantity=10,
        max_notional=700_000.0,
        entry_rule=EntryRule(trigger="x", min_price=69_000.0, max_price=71_000.0),
        exit_rules=ExitRules(take_profit_rate=0.006, stop_loss_rate=0.003),
        cancel_rule="PLAN_EXPIRY",
        expected_net_edge_bps=40.0,
        cost_snapshot={},
        risk_snapshot={},
        weekday_time_context={},
        source_ids=(),
        reference_price=70_000.0,
    )
    decision = _guard().evaluate(
        _order(side="SELL", direction="SHORT", position_effect="OPEN"),
        plan=short_plan,
        orderable_cash=1e9,
    )
    assert not decision.allowed
    assert "GUARD_BORROW_UNAVAILABLE" in decision.reason_codes


def test_a_guard_failure_blocks_rather_than_escaping() -> None:
    class _Exploding(ExecutionGuard):
        def _evaluate(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("boom")

    decision = _Exploding(pre_submit_guard=None, require_plan=False).evaluate(_order())
    assert not decision.allowed
    assert decision.reason_codes[0].startswith("GUARD_INTERNAL_ERROR")

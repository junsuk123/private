"""Integration: with a TradePlan in hand, ``evaluate_buy`` runs no post-selection gate.

``tests/test_post_selection_authority.py`` proves the property by reading the source.
This proves it by running the real engine with spies wrapped around the three authorities
and asserting none of them was called.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.data.realtime_store import RealtimeMarketDataStore
from app.data.realtime_types import (
    KIS_REALTIME_SOURCE,
    OrderbookLevel,
    RealtimeOrderbookSnapshot,
    RealtimeTradeTick,
)
from app.schemas.domain import AccountSnapshot, OrderSide
from app.trading.shared_decision_engine import SharedLiveDecisionEngine
from app.trading.trade_plan import EntryRule, ExitRules, TradePlan, TradePlanStatus

NOW = datetime(2026, 6, 29, 9, 30, tzinfo=timezone.utc)
SYMBOL = "000660"


def _seed(store: RealtimeMarketDataStore) -> None:
    store.save_ticks(
        tuple(
            RealtimeTradeTick(
                SYMBOL,
                NOW - timedelta(seconds=120 - index * 10),
                NOW - timedelta(seconds=120 - index * 10),
                KIS_REALTIME_SOURCE,
                70_000 + index * 30,
                1000,
                sequence_key=f"t{index}",
            )
            for index in range(13)
        )
    )
    store.save_orderbooks(
        (
            RealtimeOrderbookSnapshot(
                SYMBOL,
                NOW,
                NOW,
                KIS_REALTIME_SOURCE,
                (OrderbookLevel(70_380, 500_000, 70_400, 100_000),),
                sequence_key="b",
            ),
        )
    )


def _plan(**overrides) -> TradePlan:
    base = dict(
        plan_id="plan-INTEGRATION-1",
        created_at=NOW,
        expires_at=NOW + timedelta(minutes=10),
        symbol=SYMBOL,
        market="KR",
        direction="LONG",
        strategy_id="intraday_momentum",
        quantity=7,
        max_notional=500_000.0,
        entry_rule=EntryRule(trigger="intraday_momentum", min_price=60_000.0, max_price=80_000.0),
        exit_rules=ExitRules(
            take_profit_rate=0.006, stop_loss_rate=0.003, max_holding_seconds=900
        ),
        cancel_rule="PLAN_EXPIRY",
        expected_net_edge_bps=67.0,
        cost_snapshot={"net_expected_return": 0.0067, "required_min_net_return": 0.003},
        risk_snapshot={"approved": True, "sizing": {"position_weight": 0.02}},
        weekday_time_context={"day_of_week": "MON", "session_phase": "OPENING"},
        source_ids=("tick:t12",),
        reference_price=70_360.0,
        order_contract={"position_effect": "OPEN", "execution_product": "CASH"},
    )
    base.update(overrides)
    return TradePlan(**base)


class _Spy:
    """Records calls and re-raises, so a hit is a loud failure rather than a wrong number."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.calls += 1
        raise AssertionError(
            "a post-selection authority ran on the plan-driven path"
        )


@pytest.fixture()
def engine(tmp_path):
    store = RealtimeMarketDataStore(tmp_path / "rt.sqlite3")
    _seed(store)
    return SharedLiveDecisionEngine(store)


@pytest.fixture()
def account() -> AccountSnapshot:
    return AccountSnapshot(cash=10_000_000, holdings=(), realized_pnl_today=0.0)


class _FiredTrigger:
    """A tradable owned-strategy prediction.

    Stubbed because the entry TRIGGER is a separate subject with its own tests: the
    strategy's algorithm decides *when*, and what is under test here is what happens
    afterwards. Seeding enough real ticks to fire a specific algorithm would couple these
    assertions to that algorithm's parameters.
    """

    tradable = True
    confidence = 0.72
    regime = "TREND_UP"
    methodology = "intraday_momentum"
    expected_net_return_bps = 67.0
    expected_gross_return = 0.0095
    reason_codes = ("MOMENTUM_CONFIRMED",)

    def as_dict(self) -> dict:
        return {"tradable": True, "methodology": self.methodology}


def _disarm(engine: SharedLiveDecisionEngine) -> dict[str, _Spy]:
    spies = {
        "profitability": _Spy(),
        "sizing": _Spy(),
        "risk": _Spy(),
    }
    engine.profitability_gate.evaluate = spies["profitability"]  # type: ignore[method-assign]
    engine.position_sizer.size = spies["sizing"]  # type: ignore[method-assign]
    engine.risk_manager.validate = spies["risk"]  # type: ignore[method-assign]
    engine._owned_strategy_prediction = (  # type: ignore[method-assign]
        lambda frame, symbol, strategy_id, election_context=None: _FiredTrigger()
    )
    return spies


# --------------------------------------------------------------------------- #
def test_a_plan_reaches_a_final_order_without_touching_any_gate(engine, account) -> None:
    spies = _disarm(engine)
    plan = _plan()
    result = engine.evaluate_buy(
        SYMBOL,
        account,
        decision_time=NOW,
        selected_strategy=plan.strategy_id,
        trade_plan=plan,
    )
    assert result.approved, result.reason_codes
    assert result.final_order is not None
    assert all(spy.calls == 0 for spy in spies.values())


def test_a_frozen_plan_does_not_require_the_entry_trigger_to_fire_twice(engine, account) -> None:
    spies = _disarm(engine)

    class LaterFrameNoLongerFires:
        tradable = False
        confidence = 0.0
        regime = "NO_TRADE"
        methodology = "intraday_momentum"
        expected_net_return_bps = 0.0
        expected_gross_return = 0.0
        reason_codes = ("TECHNICAL_CONFIDENCE_TOO_LOW",)

        def as_dict(self) -> dict:
            return {"tradable": False, "reason_codes": list(self.reason_codes)}

    engine._owned_strategy_prediction = (  # type: ignore[method-assign]
        lambda frame, symbol, strategy_id, election_context=None: LaterFrameNoLongerFires()
    )
    plan = _plan()

    result = engine.evaluate_buy(
        SYMBOL,
        account,
        decision_time=NOW,
        selected_strategy=plan.strategy_id,
        trade_plan=plan,
    )

    assert result.approved, result.reason_codes
    assert result.final_order is not None
    assert "SELECTED_STRATEGY_ENTRY_NOT_READY" not in result.reason_codes
    assert all(spy.calls == 0 for spy in spies.values())


def test_the_submitted_quantity_is_the_elected_quantity(engine, account) -> None:
    _disarm(engine)
    plan = _plan(quantity=7)
    result = engine.evaluate_buy(
        SYMBOL, account, decision_time=NOW, selected_strategy=plan.strategy_id, trade_plan=plan
    )
    assert result.final_order.quantity == 7
    assert result.final_order.side is OrderSide.BUY
    assert result.final_order.manual_approval_required is False


def test_the_diagnostics_name_the_plan_as_the_authority(engine, account) -> None:
    _disarm(engine)
    plan = _plan()
    result = engine.evaluate_buy(
        SYMBOL, account, decision_time=NOW, selected_strategy=plan.strategy_id, trade_plan=plan
    )
    diagnostics = result.diagnostics
    assert diagnostics["execution_authority"] == "TRADE_PLAN"
    assert diagnostics["post_selection_gates"] == []
    assert diagnostics["trade_plan"]["plan_id"] == plan.plan_id
    # The frozen numbers are still published — as telemetry, not as a gate.
    assert diagnostics["profitability_decision"] == dict(plan.cost_snapshot)
    assert diagnostics["risk_snapshot"] == dict(plan.risk_snapshot)
    assert diagnostics["weekday_time_context"]["day_of_week"] == "MON"


def test_the_reason_codes_identify_the_owning_plan(engine, account) -> None:
    _disarm(engine)
    plan = _plan()
    result = engine.evaluate_buy(
        SYMBOL, account, decision_time=NOW, selected_strategy=plan.strategy_id, trade_plan=plan
    )
    assert f"TRADE_PLAN:{plan.plan_id}" in result.reason_codes
    assert f"STRATEGY_OWNED:{plan.strategy_id}" in result.reason_codes


def test_an_expired_plan_is_refused_by_the_plan_itself(engine, account) -> None:
    _disarm(engine)
    plan = _plan(expires_at=NOW - timedelta(seconds=1), created_at=NOW - timedelta(minutes=5))
    result = engine.evaluate_buy(
        SYMBOL, account, decision_time=NOW, selected_strategy=plan.strategy_id, trade_plan=plan
    )
    assert not result.approved
    assert "PLAN_EXPIRED" in result.reason_codes


def test_a_price_outside_the_band_is_refused(engine, account) -> None:
    _disarm(engine)
    plan = _plan(entry_rule=EntryRule(trigger="x", min_price=10.0, max_price=20.0))
    result = engine.evaluate_buy(
        SYMBOL, account, decision_time=NOW, selected_strategy=plan.strategy_id, trade_plan=plan
    )
    assert not result.approved
    assert "PLAN_ENTRY_PRICE_OUT_OF_BAND" in result.reason_codes


def test_a_terminal_plan_is_refused(engine, account) -> None:
    _disarm(engine)
    plan = _plan().with_status(TradePlanStatus.CANCELLED)
    result = engine.evaluate_buy(
        SYMBOL, account, decision_time=NOW, selected_strategy=plan.strategy_id, trade_plan=plan
    )
    assert not result.approved
    assert any("TERMINAL" in code for code in result.reason_codes)


def test_without_a_plan_the_legacy_gated_path_still_runs(engine, account) -> None:
    """The fallback is intact: no plan means the old authorities still apply, so a
    deployment that cannot build a plan is degraded rather than ungated."""
    calls = {"profitability": 0}
    original = engine.profitability_gate.evaluate

    def _counting(*args, **kwargs):  # noqa: ANN002, ANN003
        calls["profitability"] += 1
        return original(*args, **kwargs)

    engine.profitability_gate.evaluate = _counting  # type: ignore[method-assign]
    engine.evaluate_buy(SYMBOL, account, decision_time=NOW)
    assert calls["profitability"] == 1

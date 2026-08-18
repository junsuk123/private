"""TradePlan: what the election freezes, and what may still change afterwards."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.storage.trading_state_store import TradingStateStore
from app.trading.trade_plan import (
    EntryRule,
    ExitRules,
    TradePlan,
    TradePlanError,
    TradePlanStatus,
    TradePlanStore,
)

NOW = datetime(2026, 8, 19, 1, 30, tzinfo=timezone.utc)


def _plan(**overrides) -> TradePlan:
    base = dict(
        plan_id="plan-TEST-1",
        created_at=NOW,
        expires_at=NOW + timedelta(minutes=5),
        symbol="000660",
        market="KR",
        direction="LONG",
        strategy_id="intraday_momentum",
        quantity=10,
        max_notional=700_000.0,
        entry_rule=EntryRule(trigger="intraday_momentum", min_price=69_720.0, max_price=70_280.0),
        exit_rules=ExitRules(
            take_profit_rate=0.006,
            stop_loss_rate=0.003,
            trailing_rate=0.0015,
            max_holding_seconds=900,
        ),
        cancel_rule="PLAN_EXPIRY",
        expected_net_edge_bps=67.0,
        cost_snapshot={"net_expected_return": 0.0067},
        risk_snapshot={"approved": True},
        weekday_time_context={"day_of_week": "WED", "session_phase": "MORNING_TREND"},
        source_ids=("tick:1",),
        reference_price=70_000.0,
    )
    base.update(overrides)
    return TradePlan(**base)


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "overrides",
    [
        {"quantity": 0},
        {"max_notional": 0.0},
        {"expires_at": NOW - timedelta(seconds=1)},
        {"direction": "SIDEWAYS"},
        {"strategy_id": ""},
    ],
)
def test_an_invalid_plan_cannot_be_constructed(overrides: dict) -> None:
    with pytest.raises(TradePlanError):
        _plan(**overrides)


def test_exit_rules_reject_non_positive_geometry() -> None:
    with pytest.raises(TradePlanError):
        ExitRules(take_profit_rate=0.0, stop_loss_rate=0.003)
    with pytest.raises(TradePlanError):
        ExitRules(take_profit_rate=0.006, stop_loss_rate=-0.001)


def test_the_required_fields_are_all_present_in_the_payload() -> None:
    payload = _plan().as_dict()
    for field in (
        "plan_id",
        "created_at",
        "expires_at",
        "symbol",
        "market",
        "direction",
        "strategy_id",
        "quantity",
        "max_notional",
        "entry_rule",
        "entry_price_range",
        "take_profit_rule",
        "stop_loss_rule",
        "trailing_rule",
        "time_exit",
        "cancel_rule",
        "expected_net_edge",
        "cost_snapshot",
        "risk_snapshot",
        "weekday_time_context",
        "source_ids",
    ):
        assert field in payload, field


# --------------------------------------------------------------------------- #
# Expiry and executability
# --------------------------------------------------------------------------- #
def test_a_plan_expires() -> None:
    plan = _plan()
    assert plan.executable(NOW) == (True, None)
    assert plan.executable(NOW + timedelta(minutes=6)) == (False, "PLAN_EXPIRED")


def test_a_terminal_plan_is_not_executable() -> None:
    plan = _plan().with_status(TradePlanStatus.CANCELLED)
    allowed, why = plan.executable(NOW)
    assert not allowed
    assert why == "PLAN_TERMINAL:CANCELLED"


def test_a_fully_filled_plan_authorises_nothing_further() -> None:
    plan = _plan().with_entry_fill(70_000.0, 10)
    assert plan.remaining_quantity == 0
    assert plan.executable(NOW) == (False, "PLAN_FULLY_FILLED")


# --------------------------------------------------------------------------- #
# Immutability
# --------------------------------------------------------------------------- #
def test_the_broker_clip_may_only_reduce() -> None:
    plan = _plan()
    with pytest.raises(TradePlanError):
        plan.with_broker_clip(plan.quantity + 1, reason="CASH")
    with pytest.raises(TradePlanError):
        plan.with_broker_clip(plan.quantity, reason="CASH")
    with pytest.raises(TradePlanError):
        plan.with_broker_clip(0, reason="CASH")
    clipped = plan.with_broker_clip(4, reason="INSUFFICIENT_CASH")
    assert clipped.quantity == 4
    assert clipped.broker_clipped_from == 10


def test_the_clip_does_not_change_the_frozen_decision() -> None:
    """A quantity clip is a broker fact, not a re-decision, and the hash must say so."""
    plan = _plan()
    clipped = plan.with_broker_clip(4, reason="INSUFFICIENT_CASH")
    assert clipped.immutable_signature() == plan.immutable_signature()
    assert clipped.strategy_id == plan.strategy_id
    assert clipped.risk_snapshot["broker_clip_reason"] == "INSUFFICIENT_CASH"


def test_changing_the_strategy_changes_the_signature() -> None:
    assert _plan().immutable_signature() != _plan(strategy_id="breakout_volume").immutable_signature()


def test_changing_the_risk_basis_changes_the_signature() -> None:
    assert (
        _plan().immutable_signature()
        != _plan(risk_snapshot={"approved": True, "position_weight": 0.9}).immutable_signature()
    )


def test_the_decision_fingerprint_ignores_identity() -> None:
    first = _plan(plan_id="plan-A", created_at=NOW, expires_at=NOW + timedelta(minutes=5))
    second = _plan(
        plan_id="plan-B",
        created_at=NOW + timedelta(seconds=30),
        expires_at=NOW + timedelta(minutes=6),
    )
    assert first.decision_fingerprint() == second.decision_fingerprint()
    assert first.immutable_signature() != second.immutable_signature()


def test_overfill_is_refused() -> None:
    with pytest.raises(TradePlanError):
        _plan().with_entry_fill(70_000.0, 11)


# --------------------------------------------------------------------------- #
# Exit levels
# --------------------------------------------------------------------------- #
def test_long_exit_levels_bracket_the_entry() -> None:
    levels = _plan().with_entry_fill(70_000.0, 10).exit_levels()
    assert levels["take_profit_price"] == pytest.approx(70_420.0)
    assert levels["stop_loss_price"] == pytest.approx(69_790.0)


def test_short_exit_levels_are_mirrored() -> None:
    """A short's target sits BELOW its entry; the long arithmetic would arm a target
    that only pays when the position is losing."""
    levels = _plan(direction="SHORT").with_entry_fill(70_000.0, 10).exit_levels()
    assert levels["take_profit_price"] == pytest.approx(69_580.0)
    assert levels["stop_loss_price"] == pytest.approx(70_210.0)


def test_exit_levels_bind_to_the_fill_not_the_reference() -> None:
    plan = _plan(reference_price=70_000.0).with_entry_fill(69_800.0, 10)
    assert plan.exit_levels()["take_profit_price"] == pytest.approx(69_800.0 * 1.006)


def test_partial_fills_average_the_entry_price() -> None:
    plan = _plan(quantity=10).with_entry_fill(70_000.0, 4)
    assert plan.status is TradePlanStatus.ENTERING
    plan = plan.with_entry_fill(71_000.0, 10)
    assert plan.status is TradePlanStatus.OPEN
    assert plan.entry_fill_price == pytest.approx(70_600.0)


# --------------------------------------------------------------------------- #
# Entry band
# --------------------------------------------------------------------------- #
def test_the_entry_band_is_enforced() -> None:
    rule = _plan().entry_rule
    assert rule.price_permitted(70_000.0)
    assert not rule.price_permitted(69_000.0)
    assert not rule.price_permitted(71_000.0)
    assert not rule.price_permitted(0.0)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_a_plan_survives_a_restart(tmp_path) -> None:
    store = TradingStateStore(tmp_path / "state.sqlite3")
    TradePlanStore(store).save(_plan())

    reopened = TradePlanStore(TradingStateStore(tmp_path / "state.sqlite3"))
    row = reopened.active_for_symbol("000660")
    assert row is not None
    assert row["plan_id"] == "plan-TEST-1"
    assert row["strategy_id"] == "intraday_momentum"
    assert row["quantity"] == 10


def test_saving_the_same_plan_twice_updates_rather_than_duplicates(tmp_path) -> None:
    store = TradePlanStore(TradingStateStore(tmp_path / "state.sqlite3"))
    plan = _plan()
    store.save(plan)
    store.save(plan.with_entry_fill(70_000.0, 10))
    assert store.summary()["by_status"] == {"OPEN": 1}


def test_stale_unfilled_plans_expire_but_open_ones_do_not(tmp_path) -> None:
    """An open position still needs its exit rules; orphaning it would be worse than
    keeping an expired plan alive."""
    store = TradePlanStore(TradingStateStore(tmp_path / "state.sqlite3"))
    store.save(_plan(plan_id="plan-armed"))
    store.save(_plan(plan_id="plan-open", symbol="005380").with_entry_fill(70_000.0, 10))

    expired = store.expire_stale(now=NOW + timedelta(hours=1))
    assert expired == ("plan-armed",)
    assert store.summary()["by_status"] == {"EXPIRED": 1, "OPEN": 1}


def test_terminal_plans_are_not_returned_as_active(tmp_path) -> None:
    store = TradePlanStore(TradingStateStore(tmp_path / "state.sqlite3"))
    store.save(_plan().with_status(TradePlanStatus.CLOSED))
    assert store.active_for_symbol("000660") is None

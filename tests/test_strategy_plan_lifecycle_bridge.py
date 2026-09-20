from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.trading.strategy_session import StrategySessionConfig, StrategySessionManager
from app.trading.trade_plan import (
    EntryRule,
    ExitRules,
    TradePlan,
    TradePlanStatus,
)


NOW = datetime(2026, 9, 2, tzinfo=timezone.utc)


def _plan() -> TradePlan:
    return TradePlan(
        plan_id="plan-lifecycle",
        created_at=NOW,
        expires_at=NOW + timedelta(minutes=5),
        symbol="005930",
        market="KR",
        direction="LONG",
        strategy_id="breakout_volume",
        quantity=1,
        max_notional=70_000.0,
        entry_rule=EntryRule(
            trigger="breakout_volume",
            min_price=69_000.0,
            max_price=71_000.0,
            max_wait_seconds=300.0,
        ),
        exit_rules=ExitRules(
            take_profit_rate=0.016,
            stop_loss_rate=0.006,
            trailing_rate=0.003,
            max_holding_seconds=4500,
        ),
        cancel_rule="expiry",
        expected_net_edge_bps=30.0,
        cost_snapshot={},
        risk_snapshot={},
        weekday_time_context={},
        source_ids=(),
        reference_price=70_000.0,
    )


def _manager(tmp_path) -> tuple[StrategySessionManager, list[TradePlan]]:
    manager = StrategySessionManager(
        config=StrategySessionConfig(state_path=str(tmp_path / "session.json")),
        selector_v2_runner=False,
    )
    saved: list[TradePlan] = []
    manager._save_trade_plan = saved.append  # type: ignore[method-assign]  # noqa: SLF001
    manager._trade_plan = _plan()  # noqa: SLF001
    manager._state.phase = "ARMED"  # noqa: SLF001
    manager._state.selected_symbol = "005930"  # noqa: SLF001
    return manager, saved


def test_broker_entry_lifecycle_is_persisted_on_the_trade_plan(tmp_path) -> None:
    manager, saved = _manager(tmp_path)

    manager.mark_entry_submitted("005930", NOW)
    manager.note_plan_entry_fill("005930", 70_100.0, 1)

    assert [item.status for item in saved] == [
        TradePlanStatus.ENTERING,
        TradePlanStatus.OPEN,
    ]
    assert saved[-1].filled_quantity == 1


def test_terminal_unfilled_order_cancels_plan_and_releases_session(tmp_path) -> None:
    manager, saved = _manager(tmp_path)

    manager.mark_entry_submitted("005930", NOW)
    manager.mark_entry_terminal("005930", "REJECTED", NOW + timedelta(seconds=1))

    assert saved[-1].status is TradePlanStatus.CANCELLED
    assert manager.snapshot()["phase"] == "SCANNING"
    assert manager.snapshot()["last_reason"] == "ENTRY_ORDER_REJECTED"


def test_expired_armed_plan_returns_to_same_election_authority_without_cooldown(tmp_path):
    from app.schemas.domain import AccountSnapshot
    manager, saved = _manager(tmp_path)
    selected = []
    manager._select = lambda *args, **kwargs: selected.append((args, kwargs))
    manager.evaluate(AccountSnapshot(cash=100_000., holdings=()), ("005930",), None,
                     NOW + timedelta(minutes=5))
    assert manager._trade_plan is None
    assert manager._state.phase == "SCANNING"
    assert manager._state.last_reason == "ONTOLOGY_PLAN_EXPIRED_RESELECT"
    assert len(selected) == 1
    assert saved[-1].status is TradePlanStatus.CANCELLED

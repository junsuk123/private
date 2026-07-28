from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from app.storage.lifecycle_store import LifecycleStore
from app.trading.contracts import (
    Position,
    StrategyInstanceState,
    StrategyLifecycleStatus,
    TradePlan,
)


NOW = datetime(2026, 7, 27, 1, 0, tzinfo=timezone.utc)


def _state() -> StrategyInstanceState:
    return StrategyInstanceState(
        strategy_instance_id="instance-1",
        strategy_id="intraday_momentum",
        symbol="005930",
        status=StrategyLifecycleStatus.OPEN,
        created_at=NOW,
        updated_at=NOW,
        position_id="position-1",
    )


def _position(strategy_id: str = "intraday_momentum") -> Position:
    return Position(
        position_id="position-1",
        symbol="005930",
        quantity=2,
        average_price=80000,
        origin_strategy_id=strategy_id,
        strategy_instance_id="instance-1",
        opened_at=NOW,
    )


def test_restart_restores_strategy_owned_position(tmp_path) -> None:
    path = tmp_path / "lifecycle.sqlite3"
    first = LifecycleStore(path)
    first.save_strategy_instance(_state())
    first.save_trade_plan(_plan())
    first.save_position(_position())

    restarted = LifecycleStore(path)
    restored = restarted.load_open_positions()
    assert restored == (_position(),)
    assert restarted.load_strategy_instance("instance-1") == _state()
    assert restarted.load_trade_plan("instance-1") == _plan()


def test_position_without_matching_owner_fails_closed(tmp_path) -> None:
    store = LifecycleStore(tmp_path / "lifecycle.sqlite3")
    with pytest.raises(ValueError, match="durable strategy instance"):
        store.save_position(_position())
    store.save_strategy_instance(_state())
    with pytest.raises(ValueError, match="ownership does not match"):
        store.save_position(_position("different-strategy"))


def test_migration_is_idempotent_and_reversible(tmp_path) -> None:
    path = tmp_path / "lifecycle.sqlite3"
    store = LifecycleStore(path)
    store.migrate()
    assert store.migration_versions() == (1, 2)
    store.rollback()
    with sqlite3.connect(path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "select name from sqlite_master where type = 'table'"
            ).fetchall()
        }
    assert "positions" not in tables
    assert "strategy_instances" not in tables
    assert "trade_plans" not in tables


def _plan() -> TradePlan:
    return TradePlan(
        strategy_id="intraday_momentum",
        strategy_instance_id="instance-1",
        symbol="005930",
        side="BUY",
        thesis="unit",
        entry_trigger={"kind": "unit"},
        entry_price_policy={"kind": "limit", "price": 80000},
        proposed_quantity=2,
        initial_stop={"price": 79000},
        profit_policy={"price": 81000},
        trailing_policy={"bps": 10},
        max_holding_seconds=60,
        invalidation_conditions=("DATA_STALE",),
        max_entry_slippage_bps=5,
        expires_at=NOW.replace(minute=1),
        feature_snapshot_id="features-1",
        utility_evidence_id="utility-1",
    )

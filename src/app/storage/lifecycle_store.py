from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from app.trading.contracts import (
    Position,
    StrategyInstanceState,
    StrategyLifecycleStatus,
    TradePlan,
)


MIGRATION_VERSION = 2


class LifecycleStore:
    """Durable strategy ownership state used during broker reconciliation."""

    def __init__(self, path: str | Path = "data/store/trading-lifecycle.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    def migrate(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "create table if not exists schema_migrations "
                "(version integer primary key, applied_at text not null)"
            )
            applied = {
                int(row[0])
                for row in conn.execute("select version from schema_migrations").fetchall()
            }
            if 1 not in applied:
                conn.executescript(
                """
                create table strategy_instances (
                    strategy_instance_id text primary key,
                    strategy_id text not null,
                    symbol text not null,
                    status text not null,
                    position_id text,
                    state_version integer not null,
                    created_at text not null,
                    updated_at text not null,
                    payload text not null
                );
                create table positions (
                    position_id text primary key,
                    symbol text not null,
                    quantity integer not null,
                    origin_strategy_id text not null,
                    strategy_instance_id text not null,
                    opened_at text not null,
                    payload text not null,
                    foreign key(strategy_instance_id)
                        references strategy_instances(strategy_instance_id)
                );
                create index idx_positions_symbol on positions(symbol);
                create index idx_strategy_instances_status on strategy_instances(status);
                """
                )
                conn.execute(
                    "insert into schema_migrations(version, applied_at) values (?, ?)",
                    (1, datetime.now().astimezone().isoformat()),
                )
            if 2 not in applied:
                conn.executescript(
                    """
                    create table trade_plans (
                        strategy_instance_id text primary key,
                        strategy_id text not null,
                        symbol text not null,
                        expires_at text not null,
                        payload text not null,
                        foreign key(strategy_instance_id)
                            references strategy_instances(strategy_instance_id)
                    );
                    """
                )
                conn.execute(
                    "insert into schema_migrations(version, applied_at) values (?, ?)",
                    (2, datetime.now().astimezone().isoformat()),
                )
            conn.commit()

    def rollback(self, version: int = MIGRATION_VERSION) -> None:
        if version != MIGRATION_VERSION:
            raise ValueError(f"unsupported lifecycle migration version: {version}")
        with closing(self._connect()) as conn:
            conn.executescript(
                """
                drop table if exists trade_plans;
                drop table if exists positions;
                drop table if exists strategy_instances;
                delete from schema_migrations where version in (1, 2);
                """
            )
            conn.commit()

    def save_strategy_instance(self, state: StrategyInstanceState) -> None:
        payload = json.dumps(_jsonable(asdict(state)), sort_keys=True, separators=(",", ":"))
        with closing(self._connect()) as conn:
            existing = conn.execute(
                "select state_version from strategy_instances where strategy_instance_id = ?",
                (state.strategy_instance_id,),
            ).fetchone()
            if existing and int(existing[0]) > state.state_version:
                raise ValueError("strategy state version cannot move backwards")
            conn.execute(
                """
                insert into strategy_instances(
                    strategy_instance_id, strategy_id, symbol, status, position_id,
                    state_version, created_at, updated_at, payload
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(strategy_instance_id) do update set
                    status=excluded.status,
                    position_id=excluded.position_id,
                    state_version=excluded.state_version,
                    updated_at=excluded.updated_at,
                    payload=excluded.payload
                """,
                (
                    state.strategy_instance_id,
                    state.strategy_id,
                    state.symbol,
                    state.status.value,
                    state.position_id,
                    state.state_version,
                    state.created_at.isoformat(),
                    state.updated_at.isoformat(),
                    payload,
                ),
            )
            conn.commit()

    def save_position(self, position: Position) -> None:
        owner = self.load_strategy_instance(position.strategy_instance_id)
        if owner is None:
            raise ValueError("position requires a durable strategy instance")
        if owner.strategy_id != position.origin_strategy_id or owner.symbol != position.symbol:
            raise ValueError("position ownership does not match strategy instance")
        payload = json.dumps(_jsonable(asdict(position)), sort_keys=True, separators=(",", ":"))
        with closing(self._connect()) as conn:
            conn.execute(
                """
                insert into positions(
                    position_id, symbol, quantity, origin_strategy_id,
                    strategy_instance_id, opened_at, payload
                ) values (?, ?, ?, ?, ?, ?, ?)
                on conflict(position_id) do update set
                    quantity=excluded.quantity, payload=excluded.payload
                """,
                (
                    position.position_id,
                    position.symbol,
                    position.quantity,
                    position.origin_strategy_id,
                    position.strategy_instance_id,
                    position.opened_at.isoformat(),
                    payload,
                ),
            )
            conn.commit()

    def save_trade_plan(self, plan: TradePlan) -> None:
        owner = self.load_strategy_instance(plan.strategy_instance_id)
        if owner is None:
            raise ValueError("trade plan requires a durable strategy instance")
        if owner.strategy_id != plan.strategy_id or owner.symbol != plan.symbol:
            raise ValueError("trade plan does not match strategy instance")
        payload = json.dumps(_jsonable(asdict(plan)), sort_keys=True, separators=(",", ":"))
        with closing(self._connect()) as conn:
            conn.execute(
                """
                insert into trade_plans(
                    strategy_instance_id, strategy_id, symbol, expires_at, payload
                ) values (?, ?, ?, ?, ?)
                on conflict(strategy_instance_id) do update set
                    expires_at=excluded.expires_at, payload=excluded.payload
                """,
                (
                    plan.strategy_instance_id,
                    plan.strategy_id,
                    plan.symbol,
                    plan.expires_at.isoformat(),
                    payload,
                ),
            )
            conn.commit()

    def load_trade_plan(self, strategy_instance_id: str) -> TradePlan | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "select payload from trade_plans where strategy_instance_id = ?",
                (strategy_instance_id,),
            ).fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        return TradePlan(
            strategy_id=value["strategy_id"],
            strategy_instance_id=value["strategy_instance_id"],
            symbol=value["symbol"],
            side=value["side"],
            thesis=value["thesis"],
            entry_trigger=value["entry_trigger"],
            entry_price_policy=value["entry_price_policy"],
            proposed_quantity=int(value["proposed_quantity"]),
            initial_stop=value["initial_stop"],
            profit_policy=value["profit_policy"],
            trailing_policy=value["trailing_policy"],
            max_holding_seconds=int(value["max_holding_seconds"]),
            invalidation_conditions=tuple(value["invalidation_conditions"]),
            max_entry_slippage_bps=float(value["max_entry_slippage_bps"]),
            expires_at=datetime.fromisoformat(value["expires_at"]),
            feature_snapshot_id=value["feature_snapshot_id"],
            utility_evidence_id=value["utility_evidence_id"],
        )

    def load_strategy_instance(self, strategy_instance_id: str) -> StrategyInstanceState | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "select payload from strategy_instances where strategy_instance_id = ?",
                (strategy_instance_id,),
            ).fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        return StrategyInstanceState(
            strategy_instance_id=value["strategy_instance_id"],
            strategy_id=value["strategy_id"],
            symbol=value["symbol"],
            status=StrategyLifecycleStatus(value["status"]),
            created_at=datetime.fromisoformat(value["created_at"]),
            updated_at=datetime.fromisoformat(value["updated_at"]),
            position_id=value.get("position_id"),
            state_version=int(value["state_version"]),
        )

    def load_open_positions(self) -> tuple[Position, ...]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "select payload from positions where quantity != 0 order by opened_at, position_id"
            ).fetchall()
        return tuple(_position_from_payload(json.loads(row[0])) for row in rows)

    def migration_versions(self) -> tuple[int, ...]:
        with closing(self._connect()) as conn:
            rows = conn.execute("select version from schema_migrations order by version").fetchall()
        return tuple(int(row[0]) for row in rows)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=15)
        conn.execute("pragma foreign_keys = on")
        return conn


def _position_from_payload(value: dict[str, Any]) -> Position:
    return Position(
        position_id=value["position_id"],
        symbol=value["symbol"],
        quantity=int(value["quantity"]),
        average_price=float(value["average_price"]),
        origin_strategy_id=value["origin_strategy_id"],
        strategy_instance_id=value["strategy_instance_id"],
        opened_at=datetime.fromisoformat(value["opened_at"]),
        realized_pnl=float(value.get("realized_pnl", 0.0)),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    return value

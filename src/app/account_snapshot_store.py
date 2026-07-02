from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class AccountSnapshotStore:
    def __init__(self, db_path: str | Path = "data/store/account_dashboard.sqlite3") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    def migrate(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.executescript(
                """
                create table if not exists account_snapshots (
                    id integer primary key autoincrement,
                    created_at text not null,
                    source text not null,
                    total_asset_krw real not null default 0,
                    net_asset_krw real not null default 0,
                    cash_equivalent_krw real not null default 0,
                    krw_cash real not null default 0,
                    foreign_cash_krw real not null default 0,
                    domestic_stock_value_krw real not null default 0,
                    overseas_stock_value_krw real not null default 0,
                    realized_pnl_krw real not null default 0,
                    unrealized_pnl_krw real not null default 0,
                    total_pnl_krw real not null default 0,
                    total_pnl_rate real not null default 0,
                    raw_payload_json text
                );
                create table if not exists holding_snapshots (
                    id integer primary key autoincrement,
                    snapshot_id integer not null,
                    created_at text not null,
                    market_group text not null,
                    market text,
                    exchange text,
                    ticker text not null,
                    name text,
                    currency text,
                    quantity real not null default 0,
                    available_quantity real not null default 0,
                    average_price real not null default 0,
                    current_price real not null default 0,
                    purchase_amount_krw real not null default 0,
                    evaluation_amount_krw real not null default 0,
                    unrealized_pnl_krw real not null default 0,
                    unrealized_pnl_rate real not null default 0,
                    realized_pnl_krw real not null default 0,
                    raw_payload_json text
                );
                create table if not exists trade_events (
                    id integer primary key autoincrement,
                    occurred_at text not null,
                    market_group text not null,
                    market text,
                    exchange text,
                    ticker text not null,
                    name text,
                    side text,
                    order_type text,
                    order_id text,
                    order_status text,
                    ordered_quantity real not null default 0,
                    filled_quantity real not null default 0,
                    average_fill_price real not null default 0,
                    amount_krw real not null default 0,
                    fee_krw real not null default 0,
                    tax_krw real not null default 0,
                    realized_pnl_krw real not null default 0,
                    currency text,
                    source text,
                    raw_payload_json text
                );
                create table if not exists cash_currency_snapshots (
                    id integer primary key autoincrement,
                    snapshot_id integer not null,
                    created_at text not null,
                    currency text not null,
                    cash_balance real not null default 0,
                    orderable_amount real not null default 0,
                    withdrawable_amount real not null default 0,
                    fx_rate_to_krw real not null default 0,
                    krw_equivalent real not null default 0,
                    source text,
                    raw_payload_json text
                );
                create index if not exists idx_account_snapshots_created_at on account_snapshots(created_at);
                create index if not exists idx_holding_snapshots_snapshot_id on holding_snapshots(snapshot_id);
                create index if not exists idx_trade_events_occurred_at on trade_events(occurred_at);
                create index if not exists idx_cash_currency_snapshot_id on cash_currency_snapshots(snapshot_id);
                """
            )
            conn.commit()

    def save_dashboard(self, dashboard: dict[str, Any]) -> int:
        snapshot = dict(dashboard.get("snapshot") or {})
        created_at = str(snapshot.get("created_at") or datetime.now(timezone.utc).isoformat())
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute(
                """
                insert into account_snapshots
                (created_at, source, total_asset_krw, net_asset_krw, cash_equivalent_krw,
                 krw_cash, foreign_cash_krw, domestic_stock_value_krw, overseas_stock_value_krw,
                 realized_pnl_krw, unrealized_pnl_krw, total_pnl_krw, total_pnl_rate, raw_payload_json)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    str(snapshot.get("source") or "unknown"),
                    _num(snapshot.get("total_asset_krw")),
                    _num(snapshot.get("net_asset_krw")),
                    _num(snapshot.get("cash_equivalent_krw")),
                    _num(snapshot.get("krw_cash")),
                    _num(snapshot.get("foreign_cash_krw")),
                    _num(snapshot.get("domestic_stock_value_krw")),
                    _num(snapshot.get("overseas_stock_value_krw")),
                    _num(snapshot.get("realized_pnl_period_krw") or snapshot.get("realized_pnl_today_krw")),
                    _num(snapshot.get("unrealized_pnl_krw")),
                    _num(snapshot.get("total_pnl_krw")),
                    _num(snapshot.get("total_pnl_rate")),
                    json.dumps(dashboard, ensure_ascii=True, sort_keys=True),
                ),
            )
            snapshot_id = int(cursor.lastrowid)
            self._insert_holdings(conn, snapshot_id, created_at, dashboard.get("holdings") or ())
            self._insert_cash(conn, snapshot_id, created_at, dashboard.get("cash") or ())
            self._insert_trades(conn, dashboard.get("trades") or ())
            conn.commit()
        return snapshot_id

    def latest_dashboard(self) -> dict[str, Any] | None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                "select raw_payload_json from account_snapshots order by created_at desc, id desc limit 1"
            ).fetchone()
        if not row:
            return None
        try:
            payload = json.loads(str(row[0] or "{}"))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def asset_history(self, range_name: str = "1D") -> list[dict[str, Any]]:
        start = datetime.now(timezone.utc) - _range_delta(range_name)
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute(
                """
                select created_at, total_asset_krw, cash_equivalent_krw, domestic_stock_value_krw,
                       overseas_stock_value_krw, unrealized_pnl_krw, realized_pnl_krw, total_pnl_krw
                from account_snapshots
                where created_at >= ?
                order by created_at
                """,
                (start.isoformat(),),
            ).fetchall()
        return [
            {
                "created_at": row[0],
                "total_asset_krw": float(row[1] or 0),
                "cash_equivalent_krw": float(row[2] or 0),
                "domestic_stock_value_krw": float(row[3] or 0),
                "overseas_stock_value_krw": float(row[4] or 0),
                "unrealized_pnl_krw": float(row[5] or 0),
                "realized_pnl_krw": float(row[6] or 0),
                "total_pnl_krw": float(row[7] or 0),
            }
            for row in rows
        ]

    def _insert_holdings(self, conn: sqlite3.Connection, snapshot_id: int, created_at: str, rows: Any) -> None:
        for item in rows if isinstance(rows, list) else []:
            conn.execute(
                """
                insert into holding_snapshots
                (snapshot_id, created_at, market_group, market, exchange, ticker, name, currency,
                 quantity, available_quantity, average_price, current_price, purchase_amount_krw,
                 evaluation_amount_krw, unrealized_pnl_krw, unrealized_pnl_rate, realized_pnl_krw, raw_payload_json)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    created_at,
                    str(item.get("market_group") or ""),
                    str(item.get("market") or ""),
                    str(item.get("exchange") or ""),
                    str(item.get("ticker") or ""),
                    str(item.get("name") or ""),
                    str(item.get("currency") or ""),
                    _num(item.get("quantity")),
                    _num(item.get("available_quantity")),
                    _num(item.get("average_price")),
                    _num(item.get("current_price")),
                    _num(item.get("purchase_amount_krw")),
                    _num(item.get("evaluation_amount_krw")),
                    _num(item.get("unrealized_pnl_krw")),
                    _num(item.get("unrealized_pnl_rate")),
                    _num(item.get("realized_pnl_krw")),
                    json.dumps(item, ensure_ascii=True, sort_keys=True),
                ),
            )

    def _insert_cash(self, conn: sqlite3.Connection, snapshot_id: int, created_at: str, rows: Any) -> None:
        for item in rows if isinstance(rows, list) else []:
            conn.execute(
                """
                insert into cash_currency_snapshots
                (snapshot_id, created_at, currency, cash_balance, orderable_amount, withdrawable_amount,
                 fx_rate_to_krw, krw_equivalent, source, raw_payload_json)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    created_at,
                    str(item.get("currency") or ""),
                    _num(item.get("cash_balance")),
                    _num(item.get("orderable_amount")),
                    _num(item.get("withdrawable_amount")),
                    _num(item.get("fx_rate_to_krw")),
                    _num(item.get("krw_equivalent")),
                    str(item.get("source") or ""),
                    json.dumps(item, ensure_ascii=True, sort_keys=True),
                ),
            )

    def _insert_trades(self, conn: sqlite3.Connection, rows: Any) -> None:
        for item in rows if isinstance(rows, list) else []:
            conn.execute(
                """
                insert into trade_events
                (occurred_at, market_group, market, exchange, ticker, name, side, order_type,
                 order_id, order_status, ordered_quantity, filled_quantity, average_fill_price,
                 amount_krw, fee_krw, tax_krw, realized_pnl_krw, currency, source, raw_payload_json)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(item.get("occurred_at") or datetime.now(timezone.utc).isoformat()),
                    str(item.get("market_group") or ""),
                    str(item.get("market") or ""),
                    str(item.get("exchange") or ""),
                    str(item.get("ticker") or ""),
                    str(item.get("name") or ""),
                    str(item.get("side") or ""),
                    str(item.get("order_type") or ""),
                    str(item.get("order_id") or ""),
                    str(item.get("order_status") or ""),
                    _num(item.get("ordered_quantity")),
                    _num(item.get("filled_quantity")),
                    _num(item.get("average_fill_price")),
                    _num(item.get("amount_krw")),
                    _num(item.get("fee_krw")),
                    _num(item.get("tax_krw")),
                    _num(item.get("realized_pnl_krw")),
                    str(item.get("currency") or ""),
                    str(item.get("source") or ""),
                    json.dumps(item, ensure_ascii=True, sort_keys=True),
                ),
            )


def _num(value: Any) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _range_delta(range_name: str) -> timedelta:
    name = str(range_name or "1D").upper()
    if name == "1W":
        return timedelta(days=7)
    if name == "1M":
        return timedelta(days=31)
    if name == "3M":
        return timedelta(days=93)
    return timedelta(days=1)

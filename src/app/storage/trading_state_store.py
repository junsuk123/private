"""Transactional store for the context -> decision -> order chain.

One SQLite file (``data/store/trading_state.sqlite3``) holding the tables that have to
agree with each other inside a single transaction: a decision's contexts, the regime and
model predictions it rested on, the gate verdict, the order intent it produced and the
execution that followed. Splitting those across stores is what makes a decision
unreconstructible after a crash — the intent commits, the gate row does not, and there is
no way afterwards to say what the gate saw.

Relationship to the stores that already exist
---------------------------------------------
Nothing here replaces them and nothing is copied into them:

``data/store/realtime_market_data.sqlite3``
    High-frequency ticks, order books, minute bars. Written by the feed threads at a rate
    this store is not designed for, and deliberately left alone — the project has already
    measured that large reads there interfere with the writer.
``data/store/account_dashboard.sqlite3``
    Operator-facing account/holding history for the dashboard.
    ``account_snapshot`` here is a *different thing*: the broker truth captured at
    decision time and used for reconciliation, keyed by ``snapshot_id`` and referenced by
    gate decisions.

Durability
----------
WAL, ``synchronous=FULL`` and a real transaction around every multi-table write. This
file is on the order path; a torn write here is a live order whose authorisation cannot
be produced. Migrations are versioned and recorded in ``schema_migrations`` with row
counts before and after, matching the convention in
:mod:`app.data.realtime_store` — additive columns always carry a DEFAULT so a writer
that predates the migration cannot start failing its INSERTs.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

__all__ = [
    "SCHEMA_VERSION",
    "TradingStateStore",
    "default_trading_state_store",
    "reset_default_trading_state_store",
]

#: v1 = initial context / decision / execution schema.
SCHEMA_VERSION = 1

DEFAULT_DB_PATH = Path("data/store/trading_state.sqlite3")

_SCHEMA = """
create table if not exists schema_version (
    id integer primary key check (id = 1),
    version integer not null
);
create table if not exists schema_migrations (
    version integer primary key,
    applied_at text not null,
    description text not null,
    rows_before integer not null,
    rows_after integer not null
);

-- ---------------------------------------------------------------- calendar --
create table if not exists exchange_calendar (
    market_group text not null,
    trading_day text not null,
    is_trading_day integer not null,
    early_close_local text,
    calendar_version text not null,
    calendar_source text not null,
    completeness text not null default 'unknown',
    recorded_at text not null,
    primary key (market_group, trading_day)
);

create table if not exists market_session (
    session_key text primary key,
    market_group text not null,
    trading_day text not null,
    session_id text not null,
    venue text not null,
    phase text not null,
    session_start text,
    session_end text,
    data_available integer not null default 0,
    trade_available integer not null default 0,
    new_entry_allowed integer not null default 0,
    exit_allowed integer not null default 0,
    reason_codes_json text not null default '[]',
    recorded_at text not null
);
create index if not exists idx_market_session_day
    on market_session(market_group, trading_day);

-- ---------------------------------------------------------------- contexts --
create table if not exists global_context (
    context_id text primary key,
    captured_at text not null,
    trading_day text not null,
    session_phase text not null,
    direction real,
    momentum real,
    risk_sentiment real,
    volatility real,
    rates_pressure real,
    fx_pressure real,
    global_alignment real,
    confidence real not null default 0.0,
    group_scores_json text not null default '{}',
    sources_json text not null default '{}',
    reason_codes_json text not null default '[]'
);
create index if not exists idx_global_context_captured on global_context(captured_at);

create table if not exists domestic_context (
    context_id text primary key,
    captured_at text not null,
    trading_day text not null,
    session_phase text not null,
    global_context_id text,
    direction real,
    breadth real,
    liquidity real,
    volatility real,
    flow real,
    leadership real,
    venue_divergence real,
    confidence real not null default 0.0,
    components_json text not null default '{}',
    sources_json text not null default '{}',
    reason_codes_json text not null default '[]'
);
create index if not exists idx_domestic_context_captured on domestic_context(captured_at);

create table if not exists sector_context (
    context_id text primary key,
    captured_at text not null,
    sector text not null,
    market_group text not null,
    domestic_context_id text,
    return_value real,
    breadth real,
    volume_z real,
    volatility real,
    relative_strength real,
    foreign_flow real,
    leader_strength real,
    leader_concentration real,
    global_alignment real,
    confidence real not null default 0.0,
    member_count integer not null default 0,
    sources_json text not null default '{}',
    reason_codes_json text not null default '[]'
);
create index if not exists idx_sector_context_captured on sector_context(captured_at, sector);

create table if not exists stock_context (
    context_id text primary key,
    captured_at text not null,
    ticker text not null,
    market_group text not null,
    sector text,
    sector_context_id text,
    features_json text not null default '{}',
    confidence real not null default 0.0,
    reason_codes_json text not null default '[]'
);
create index if not exists idx_stock_context_captured on stock_context(captured_at, ticker);

create table if not exists feature_snapshot (
    snapshot_id text primary key,
    captured_at text not null,
    scope text not null,
    scope_key text not null,
    payload_json text not null,
    source text not null default '',
    event_time text,
    received_time text,
    processed_time text
);
create index if not exists idx_feature_snapshot_scope
    on feature_snapshot(scope, scope_key, captured_at);

-- ------------------------------------------------------------- seasonality --
create table if not exists seasonality_baseline (
    metric text not null,
    market_group text not null,
    day_of_week text not null,
    session_phase text not null,
    regime text not null,
    weight real not null,
    mean real not null,
    m2 real not null,
    sample_count integer not null,
    confidence real not null,
    baseline_window integer not null,
    baseline_updated_at text not null,
    last_observed_at text not null,
    out_of_order_rejected integer not null default 0,
    primary key (metric, market_group, day_of_week, session_phase, regime)
);

-- ---------------------------------------------------------------- ontology --
create table if not exists ontology_node (
    node_id text primary key,
    node_type text not null,
    label text not null default '',
    attributes_json text not null default '{}',
    updated_at text not null
);
create index if not exists idx_ontology_node_type on ontology_node(node_type);

create table if not exists ontology_edge (
    edge_id text primary key,
    source_id text not null,
    target_id text not null,
    relation text not null,
    direction text not null default 'FORWARD',
    prior_strength real not null default 0.0,
    learnable integer not null default 1,
    learned_weight real,
    learned_updated_at text,
    lag_min integer not null default 0,
    lag_max integer not null default 0,
    attributes_json text not null default '{}',
    updated_at text not null
);
create index if not exists idx_ontology_edge_source on ontology_edge(source_id, relation);
create index if not exists idx_ontology_edge_target on ontology_edge(target_id, relation);

-- ------------------------------------------------------------- predictions --
create table if not exists regime_prediction (
    prediction_id text primary key,
    predicted_at text not null,
    scope text not null,
    scope_key text not null,
    probabilities_json text not null,
    dominant_regime text not null,
    entropy real,
    source text not null,
    model_version text not null default '',
    confidence real not null default 0.0
);
create index if not exists idx_regime_prediction_scope
    on regime_prediction(scope, scope_key, predicted_at);

create table if not exists model_prediction (
    prediction_id text primary key,
    predicted_at text not null,
    model_name text not null,
    model_version text not null default '',
    ticker text not null default '',
    heads_json text not null,
    uncertainty real,
    confidence real,
    health text not null default 'UNKNOWN',
    input_context_ids_json text not null default '[]'
);
create index if not exists idx_model_prediction_ticker
    on model_prediction(ticker, predicted_at);

-- ---------------------------------------------------------------- decision --
create table if not exists strategy_decision (
    decision_id text primary key,
    decided_at text not null,
    ticker text not null,
    strategy text,
    strategy_family text,
    action text not null,
    regime_prediction_id text,
    model_prediction_id text,
    global_context_id text,
    domestic_context_id text,
    sector_context_id text,
    stock_context_id text,
    supporting_factors_json text not null default '[]',
    conflicting_factors_json text not null default '[]',
    ontology_relations_json text not null default '[]',
    learned_relation_weights_json text not null default '{}',
    model_confidence real,
    uncertainty real,
    trace_json text not null default '{}'
);
create index if not exists idx_strategy_decision_ticker
    on strategy_decision(ticker, decided_at);
create index if not exists idx_strategy_decision_action_time
    on strategy_decision(action, decided_at);

create table if not exists gate_decision (
    gate_id text primary key,
    decision_id text not null,
    evaluated_at text not null,
    ticker text not null,
    approved integer not null,
    hard_failures_json text not null default '[]',
    soft_failures_json text not null default '[]',
    reasons_json text not null default '[]',
    position_multiplier real not null default 0.0,
    account_snapshot_id text,
    data_health_id text,
    detail_json text not null default '{}'
);
create index if not exists idx_gate_decision_decision on gate_decision(decision_id);
create index if not exists idx_gate_decision_ticker on gate_decision(ticker, evaluated_at);

-- --------------------------------------------------------------- execution --
create table if not exists order_intent (
    intent_id text primary key,
    created_at text not null,
    decision_id text,
    gate_id text,
    ticker text not null,
    market_group text not null default '',
    venue text not null default '',
    side text not null,
    quantity integer not null,
    limit_price real,
    order_type text not null default 'LIMIT',
    idempotency_key text not null unique,
    state text not null,
    state_updated_at text not null,
    filled_quantity integer not null default 0,
    average_fill_price real,
    broker_order_id text,
    payload_json text not null default '{}',
    terminal integer not null default 0
);
create index if not exists idx_order_intent_state on order_intent(state, created_at);
create index if not exists idx_order_intent_broker on order_intent(broker_order_id);
create index if not exists idx_order_intent_ticker on order_intent(ticker, created_at);

create table if not exists order_execution (
    execution_id text primary key,
    intent_id text not null,
    observed_at text not null,
    event_type text not null,
    from_state text,
    to_state text not null,
    filled_quantity integer not null default 0,
    fill_price real,
    broker_order_id text,
    reason text not null default '',
    payload_json text not null default '{}'
);
create index if not exists idx_order_execution_intent
    on order_execution(intent_id, observed_at);

create table if not exists position_snapshot (
    snapshot_id text primary key,
    captured_at text not null,
    source text not null,
    ticker text not null,
    quantity real not null,
    average_price real,
    market_value real,
    currency text not null default '',
    payload_json text not null default '{}'
);
create index if not exists idx_position_snapshot_ticker
    on position_snapshot(ticker, captured_at);

create table if not exists account_snapshot (
    snapshot_id text primary key,
    captured_at text not null,
    source text not null,
    equity real,
    cash real,
    currency text not null default '',
    reconciled integer not null default 0,
    discrepancies_json text not null default '[]',
    payload_json text not null default '{}'
);
create index if not exists idx_account_snapshot_captured on account_snapshot(captured_at);

-- ------------------------------------------------------------------ health --
create table if not exists model_health (
    health_id text primary key,
    observed_at text not null,
    model_name text not null,
    state text not null,
    reason_codes_json text not null default '[]',
    detail_json text not null default '{}'
);
create index if not exists idx_model_health_model on model_health(model_name, observed_at);

create table if not exists data_health (
    health_id text primary key,
    observed_at text not null,
    source text not null,
    data_type text not null,
    scope_key text not null default '',
    state text not null,
    age_seconds real,
    event_time text,
    received_time text,
    processed_time text,
    reason_codes_json text not null default '[]',
    detail_json text not null default '{}'
);
create index if not exists idx_data_health_source
    on data_health(source, data_type, observed_at);
"""

#: Tables pruned by ``prune_history``: append-only observation logs whose value decays.
#: The baselines, the ontology and the calendar are NOT pruned — they are state, not log.
_PRUNABLE_TABLES: tuple[tuple[str, str], ...] = (
    ("feature_snapshot", "captured_at"),
    ("global_context", "captured_at"),
    ("domestic_context", "captured_at"),
    ("sector_context", "captured_at"),
    ("stock_context", "captured_at"),
    ("regime_prediction", "predicted_at"),
    ("model_prediction", "predicted_at"),
    ("position_snapshot", "captured_at"),
    ("account_snapshot", "captured_at"),
    ("model_health", "observed_at"),
    ("data_health", "observed_at"),
    ("market_session", "recorded_at"),
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime | None) -> str | None:
    if moment is None:
        return None
    aware = moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).isoformat()


def _json(payload: Any) -> str:
    return json.dumps(payload if payload is not None else {}, ensure_ascii=False, sort_keys=True, default=str)


class TradingStateStore:
    """SQLite store for the context/decision/execution chain.

    Thread-safe: every public method opens its own connection, and writes that span
    tables run inside :meth:`transaction`.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self.init_db()

    # ------------------------------------------------------------------ #
    # connection / schema
    # ------------------------------------------------------------------ #
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.execute("pragma busy_timeout=30000")
        conn.execute("pragma foreign_keys=on")
        return conn

    def init_db(self) -> None:
        with self._write_lock, closing(self._connect()) as conn:
            conn.execute("pragma journal_mode=wal")
            # An order authorisation that survives a power loss is worth the fsync.
            conn.execute("pragma synchronous=full")
            conn.executescript(_SCHEMA)
            self._record_version(conn)

    def _record_version(self, conn: sqlite3.Connection) -> None:
        row = conn.execute("select version from schema_version where id = 1").fetchone()
        current = int(row[0]) if row else 0
        if current == SCHEMA_VERSION:
            return
        rows_before = self._total_rows(conn)
        conn.execute(
            "insert into schema_version (id, version) values (1, ?)"
            " on conflict(id) do update set version = excluded.version",
            (SCHEMA_VERSION,),
        )
        conn.execute(
            "insert or replace into schema_migrations"
            " (version, applied_at, description, rows_before, rows_after)"
            " values (?, ?, ?, ?, ?)",
            (
                SCHEMA_VERSION,
                _iso(_utcnow()),
                f"trading state schema v{SCHEMA_VERSION}",
                rows_before,
                self._total_rows(conn),
            ),
        )

    @staticmethod
    def _total_rows(conn: sqlite3.Connection) -> int:
        total = 0
        names = [
            str(row[0])
            for row in conn.execute(
                "select name from sqlite_master where type = 'table'"
                " and name not like 'sqlite_%'"
            ).fetchall()
        ]
        for name in names:
            total += int(conn.execute(f"select count(*) from {name}").fetchone()[0])
        return total

    def schema_version(self) -> int:
        with closing(self._connect()) as conn:
            row = conn.execute("select version from schema_version where id = 1").fetchone()
            return int(row[0]) if row else 0

    def migration_history(self) -> tuple[dict[str, Any], ...]:
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "select version, applied_at, description, rows_before, rows_after"
                " from schema_migrations order by version"
            ).fetchall()
        return tuple(dict(row) for row in rows)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """One atomic unit across tables.

        ``BEGIN IMMEDIATE`` so two writers cannot both read-then-write the same intent
        row and produce a duplicate order.
        """
        with self._write_lock, closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("begin immediate")
            try:
                yield conn
            except BaseException:
                conn.execute("rollback")
                raise
            conn.execute("commit")

    @contextmanager
    def reader(self) -> Iterator[sqlite3.Connection]:
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            yield conn

    # ------------------------------------------------------------------ #
    # generic helpers
    # ------------------------------------------------------------------ #
    def upsert(self, table: str, row: Mapping[str, Any], *, keys: Sequence[str]) -> None:
        with self.transaction() as conn:
            upsert_row(conn, table, row, keys=keys)

    def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        with self.reader() as conn:
            row = conn.execute(sql, tuple(params)).fetchone()
        return dict(row) if row is not None else None

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> tuple[dict[str, Any], ...]:
        with self.reader() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return tuple(dict(row) for row in rows)

    def count(self, table: str) -> int:
        with self.reader() as conn:
            return int(conn.execute(f"select count(*) from {table}").fetchone()[0])

    # ------------------------------------------------------------------ #
    # retention
    # ------------------------------------------------------------------ #
    def prune_history(self, *, retention_days: int = 90, now: datetime | None = None) -> dict[str, int]:
        """Drop observation rows older than ``retention_days``.

        Order intents and their execution events are never pruned here: they are the
        audit trail for a real order, and the reconciliation path may need to reach back
        past any retention window to explain a broker position.
        """
        cutoff = _iso((now or _utcnow()) - timedelta(days=max(1, int(retention_days))))
        removed: dict[str, int] = {}
        with self.transaction() as conn:
            for table, column in _PRUNABLE_TABLES:
                cursor = conn.execute(f"delete from {table} where {column} < ?", (cutoff,))
                if cursor.rowcount > 0:
                    removed[table] = int(cursor.rowcount)
        return removed

    def prune_unexecuted_decisions(
        self,
        *,
        retention_hours: int = 24,
        batch_size: int = 25_000,
        now: datetime | None = None,
    ) -> dict[str, int]:
        """Bound high-volume non-order traces without touching execution audit.

        Only terminal non-entry decisions with no linked order intent are eligible.
        Any decision or gate referenced by an order is retained indefinitely.
        """
        cutoff = _iso(
            (now or _utcnow()) - timedelta(hours=max(1, int(retention_hours)))
        )
        limit = max(1, min(int(batch_size), 250_000))
        with self.transaction() as conn:
            conn.execute(
                "create temporary table prune_decision_ids"
                " (decision_id text primary key)"
            )
            conn.execute(
                "insert into prune_decision_ids(decision_id)"
                " select d.decision_id from strategy_decision d"
                " where d.decided_at < ?"
                " and upper(d.action) in ('WAIT', 'NO_TRADE', 'HOLD')"
                " and not exists ("
                "   select 1 from order_intent oi where oi.decision_id = d.decision_id"
                " ) order by d.decided_at limit ?",
                (cutoff, limit),
            )
            gates = conn.execute(
                "delete from gate_decision"
                " where decision_id in (select decision_id from prune_decision_ids)"
                " and not exists ("
                "   select 1 from order_intent oi where oi.gate_id = gate_decision.gate_id"
                " )"
            ).rowcount
            decisions = conn.execute(
                "delete from strategy_decision"
                " where decision_id in (select decision_id from prune_decision_ids)"
                " and not exists ("
                "   select 1 from order_intent oi"
                "   where oi.decision_id = strategy_decision.decision_id"
                " )"
            ).rowcount
            conn.execute("drop table prune_decision_ids")
        return {
            "strategy_decision": max(0, int(decisions)),
            "gate_decision": max(0, int(gates)),
        }


def upsert_row(
    conn: sqlite3.Connection,
    table: str,
    row: Mapping[str, Any],
    *,
    keys: Sequence[str],
) -> None:
    """INSERT ... ON CONFLICT UPDATE for one row, inside the caller's transaction.

    Column names come from ``row``'s keys, which are code-controlled table columns; the
    values always go through parameter binding.
    """
    columns = list(row)
    if not columns:
        raise ValueError("upsert_row requires at least one column")
    placeholders = ", ".join("?" for _ in columns)
    assignments = ", ".join(
        f"{name} = excluded.{name}" for name in columns if name not in set(keys)
    )
    conflict = ", ".join(keys)
    sql = (
        f"insert into {table} ({', '.join(columns)}) values ({placeholders})"
        f" on conflict({conflict}) do update set {assignments}"
        if assignments
        else f"insert into {table} ({', '.join(columns)}) values ({placeholders})"
        f" on conflict({conflict}) do nothing"
    )
    conn.execute(sql, tuple(row[name] for name in columns))


def json_column(payload: Any) -> str:
    """Serialise a payload for a ``*_json`` column."""
    return _json(payload)


def iso_column(moment: datetime | None) -> str | None:
    """Serialise a timestamp for a text column, always UTC."""
    return _iso(moment)


_default_store: TradingStateStore | None = None
_default_lock = threading.Lock()


#: Env override for the store path. Exists so a test run cannot write decision and order
#: history into the operator's real ``data/store``; the same mechanism the strategy
#: performance store and the change-point detector already use.
DB_PATH_ENV = "TRADING_STATE_DB_PATH"


def default_trading_state_store(db_path: str | Path | None = None) -> TradingStateStore:
    global _default_store
    with _default_lock:
        if _default_store is None:
            resolved = db_path or os.environ.get(DB_PATH_ENV) or DEFAULT_DB_PATH
            _default_store = TradingStateStore(resolved)
        return _default_store


def reset_default_trading_state_store() -> None:
    """Test hook. Never called from the trading path."""
    global _default_store
    with _default_lock:
        _default_store = None

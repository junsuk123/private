from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.data.realtime_types import (
    FeedMetadata,
    MarketDataHealth,
    RealtimeMinuteBar,
    RealtimeOrderbookSnapshot,
    RealtimeTradeTick,
    to_jsonable,
)

#: 현재 스키마 버전.
#:   v1 = venue/session metadata 가 없던 원본 스키마
#:   v2 = market/venue/session/feed metadata + stream 단위 분 bar identity
#:   v3 = ``realtime_minute_bars.stream_id`` 에 DEFAULT '' 추가
#:
#: v3 이 필요한 이유 (실제로 운영에서 터진 사고):
#: v2 는 ``stream_id text not null`` 을 **DEFAULT 없이** 만들었다. 그래서 이 컬럼을 모르는
#: writer — 특히 마이그레이션 시점에 이미 실행 중이던 구버전 서버 프로세스 — 의 INSERT 가
#: 전부 ``NOT NULL constraint failed: realtime_minute_bars.stream_id`` 로 죽었다.
#: additive 마이그레이션이 하위호환되려면 새 NOT NULL 컬럼에는 반드시 DEFAULT 가 있어야
#: 한다. v3 은 그 DEFAULT 를 채운다.
SCHEMA_VERSION = 3

#: 체결·호가·health 테이블에 공통으로 붙는 metadata 컬럼 (컬럼명, DDL 타입, 기본값).
_METADATA_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("market_group", "text", "''"),
    ("exchange", "text", "''"),
    ("venue", "text", "'UNKNOWN'"),
    ("session", "text", "'UNKNOWN'"),
    ("currency", "text", "''"),
    ("feed_scope", "text", "'UNKNOWN'"),
    ("tr_id", "text", "''"),
    ("subscription_key", "text", "''"),
    ("is_consolidated", "integer", "0"),
    ("is_tradeable", "integer", "0"),
    ("metadata_inferred", "integer", "1"),
    ("stream_id", "text", "''"),
)

_METADATA_COLUMN_NAMES = tuple(name for name, _, _ in _METADATA_COLUMNS)


class RealtimeStoreMigrationError(RuntimeError):
    """마이그레이션이 데이터를 잃을 수 있는 상태에서 중단됐다."""


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _metadata_values(meta: FeedMetadata, *, include_stream_id: bool = True) -> tuple[Any, ...]:
    row = meta.to_row()
    return tuple(
        row[name]
        for name in _METADATA_COLUMN_NAMES
        if include_stream_id or name != "stream_id"
    )


_METADATA_SELECT = ", ".join(_METADATA_COLUMN_NAMES)


def _metadata_from_tail(row: Any, start: int) -> FeedMetadata:
    """SELECT 결과 뒤쪽에 붙인 metadata 컬럼들을 :class:`FeedMetadata` 로 되돌린다."""
    try:
        mapping = {
            name: row[start + index] for index, name in enumerate(_METADATA_COLUMN_NAMES)
        }
    except (IndexError, TypeError):
        return FeedMetadata()
    return FeedMetadata.from_row(mapping)


class RealtimeMarketDataStore:
    _last_prune_monotonic = 0.0

    def __init__(self, db_path: str | Path = "data/store/realtime_market_data.sqlite3") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def init_db(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute("pragma journal_mode=wal")
            conn.executescript(
                """
                create table if not exists realtime_ticks (
                    record_id text primary key,
                    symbol text not null,
                    exchange_timestamp text not null,
                    received_at text not null,
                    source text not null,
                    price real not null,
                    volume integer not null,
                    trade_direction text,
                    sequence_key text,
                    raw_checksum text,
                    latency_ms real not null
                );
                create index if not exists idx_realtime_ticks_symbol_time
                    on realtime_ticks(symbol, exchange_timestamp);
                create index if not exists idx_realtime_ticks_symbol_received
                    on realtime_ticks(symbol, received_at);
                create index if not exists idx_realtime_ticks_received
                    on realtime_ticks(received_at);

                create table if not exists realtime_orderbook (
                    record_id text primary key,
                    symbol text not null,
                    exchange_timestamp text not null,
                    received_at text not null,
                    source text not null,
                    best_bid real not null,
                    best_ask real not null,
                    spread_bps real not null,
                    total_bid_volume integer not null,
                    total_ask_volume integer not null,
                    imbalance real not null,
                    levels_json text not null,
                    sequence_key text,
                    raw_checksum text,
                    latency_ms real not null
                );
                create index if not exists idx_realtime_orderbook_symbol_time
                    on realtime_orderbook(symbol, exchange_timestamp);
                create index if not exists idx_realtime_orderbook_symbol_received
                    on realtime_orderbook(symbol, received_at);

                create table if not exists realtime_minute_bars (
                    symbol text not null,
                    minute_start text not null,
                    open real not null,
                    high real not null,
                    low real not null,
                    close real not null,
                    volume integer not null,
                    vwap real not null,
                    trade_count integer not null,
                    spread_bps real not null,
                    orderbook_imbalance real not null,
                    liquidity_score real not null,
                    volatility real not null,
                    last_update_age_ms real not null,
                    source_record_ids_json text not null,
                    primary key(symbol, minute_start)
                );

                create table if not exists market_data_health (
                    symbol text not null,
                    checked_at text not null,
                    quote_count integer not null,
                    orderbook_count integer not null,
                    latest_tick_at text,
                    latest_orderbook_at text,
                    max_quote_age_ms integer not null,
                    max_orderbook_age_ms integer not null,
                    source text not null,
                    source_quality_score real not null,
                    ok_for_live_buy integer not null,
                    reason_codes_json text not null,
                    primary key(symbol, checked_at)
                );

                create table if not exists data_source_events (
                    event_id text primary key,
                    event_type text not null,
                    symbol text,
                    observed_at text not null,
                    source text not null,
                    payload_json text not null
                );
                create index if not exists idx_data_source_events_observed
                    on data_source_events(observed_at);

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
                """
            )
            conn.commit()
            self._migrate(conn)
        now = time.monotonic()
        if now - self.__class__._last_prune_monotonic >= 3600:
            self.prune_operational_history()
            self.__class__._last_prune_monotonic = now

    # ------------------------------------------------------------------ #
    # 스키마 마이그레이션
    # ------------------------------------------------------------------ #
    def schema_version(self) -> int:
        with closing(self._connect()) as conn:
            return self._current_version(conn)

    def migration_history(self) -> tuple[dict[str, Any], ...]:
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "select version, applied_at, description, rows_before, rows_after"
                " from schema_migrations order by version"
            ).fetchall()
        return tuple(dict(row) for row in rows)

    @staticmethod
    def _current_version(conn: sqlite3.Connection) -> int:
        row = conn.execute("select version from schema_version where id = 1").fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
        rows = conn.execute(f"pragma table_info({table})").fetchall()
        return tuple(str(row[1]) for row in rows)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """기존 SQLite 파일을 삭제하지 않고 버전 단위로 올린다.

        v1 → v2 는 (a) 체결/호가/health 에 metadata 컬럼 추가 — 순수 additive,
        (b) ``realtime_minute_bars`` PK 를 ``(symbol, minute_start)`` 에서
        ``(stream_id, symbol, minute_start)`` 로 교체 — SQLite 는 PK 변경을 지원하지
        않으므로 v2 테이블을 만들어 트랜잭션 안에서 옮기고 이름을 바꾼다.

        기존 행은 어느 venue/session 에서 왔는지 알 수 없으므로 ``metadata_inferred=1``
        로 표시된다. 그 행들은 신규 진입 근거와 high-trust 학습 표본에서 제외된다.
        """
        current = self._current_version(conn)
        if current >= SCHEMA_VERSION:
            return
        if current == 0:
            # 새로 만든 파일인지, metadata 가 없는 v1 파일인지 구분한다.
            has_rows = any(
                conn.execute(f"select 1 from {table} limit 1").fetchone() is not None
                for table in ("realtime_ticks", "realtime_orderbook", "realtime_minute_bars")
            )
            current = 1 if has_rows else 0
        if current <= 1:
            self._apply_v2(conn, fresh=current == 0)
        if current <= 2:
            self._apply_v3(conn)
        conn.execute(
            "insert or replace into schema_version (id, version) values (1, ?)",
            (SCHEMA_VERSION,),
        )
        conn.commit()

    def _apply_v3(self, conn: sqlite3.Connection) -> None:
        """``realtime_minute_bars.stream_id`` 에 ``DEFAULT ''`` 를 부여한다.

        SQLite 는 컬럼 DEFAULT 를 ALTER 로 바꿀 수 없으므로 테이블을 재작성한다.
        데이터는 트랜잭션 안에서 그대로 옮기고, 행 수가 줄면 중단한다.
        """
        if self._minute_bar_stream_id_has_default(conn):
            return
        before = self._row_counts(conn).get("realtime_minute_bars", 0)
        try:
            conn.execute("begin")
            conn.execute(
                """
                create table realtime_minute_bars_v3 (
                    stream_id text not null default '',
                    symbol text not null,
                    minute_start text not null,
                    open real not null,
                    high real not null,
                    low real not null,
                    close real not null,
                    volume integer not null,
                    vwap real not null,
                    trade_count integer not null,
                    spread_bps real not null,
                    orderbook_imbalance real not null,
                    liquidity_score real not null,
                    volatility real not null,
                    last_update_age_ms real not null,
                    source_record_ids_json text not null,
                    market_group text not null default '',
                    exchange text not null default '',
                    venue text not null default 'UNKNOWN',
                    session text not null default 'UNKNOWN',
                    currency text not null default '',
                    feed_scope text not null default 'UNKNOWN',
                    tr_id text not null default '',
                    subscription_key text not null default '',
                    is_consolidated integer not null default 0,
                    is_tradeable integer not null default 0,
                    metadata_inferred integer not null default 1,
                    primary key(stream_id, symbol, minute_start)
                )
                """
            )
            columns = [
                name
                for name in self._table_columns(conn, "realtime_minute_bars")
                if name in set(self._table_columns(conn, "realtime_minute_bars_v3"))
            ]
            joined = ", ".join(columns)
            conn.execute(
                f"insert into realtime_minute_bars_v3 ({joined})"
                f" select {joined} from realtime_minute_bars"
            )
            conn.execute("drop table realtime_minute_bars")
            conn.execute(
                "alter table realtime_minute_bars_v3 rename to realtime_minute_bars"
            )
            conn.execute(
                """
                create index if not exists idx_realtime_minute_bars_symbol_time
                    on realtime_minute_bars(symbol, minute_start)
                """
            )
            conn.execute("commit")
        except Exception:
            conn.execute("rollback")
            raise
        after = self._row_counts(conn).get("realtime_minute_bars", -1)
        if after < before:
            raise RealtimeStoreMigrationError(
                f"migration v3 lost minute bars: {before} -> {after}"
            )
        conn.execute(
            """
            insert or replace into schema_migrations
              (version, applied_at, description, rows_before, rows_after)
            values (?, ?, ?, ?, ?)
            """,
            (
                3,
                datetime.now(timezone.utc).isoformat(),
                "give realtime_minute_bars.stream_id a DEFAULT so writers "
                "unaware of the column keep working",
                before,
                after,
            ),
        )
        conn.commit()

    @staticmethod
    def _minute_bar_stream_id_has_default(conn: sqlite3.Connection) -> bool:
        for row in conn.execute("pragma table_info(realtime_minute_bars)").fetchall():
            if str(row[1]) == "stream_id":
                # row[4] = dflt_value
                return row[4] is not None
        # 컬럼 자체가 없으면 v2 가 아직 적용되지 않은 상태다.
        return True

    def _apply_v2(self, conn: sqlite3.Connection, *, fresh: bool) -> None:
        counts_before = self._row_counts(conn)
        try:
            conn.execute("begin")
            for table in ("realtime_ticks", "realtime_orderbook", "market_data_health"):
                existing = set(self._table_columns(conn, table))
                for name, ddl_type, default in _METADATA_COLUMNS:
                    if name in existing:
                        continue
                    # 새 파일은 metadata_inferred 기본값이 0 이어야 한다 (추정이 아니라 미지정).
                    effective = "0" if (fresh and name == "metadata_inferred") else default
                    conn.execute(
                        f"alter table {table} add column {name} {ddl_type} not null default {effective}"
                    )
            if "dedup_key" not in set(self._table_columns(conn, "realtime_ticks")):
                conn.execute(
                    "alter table realtime_ticks add column dedup_key text not null default ''"
                )
            for extra, ddl_type, default in (
                ("depth_level_count", "integer", "0"),
            ):
                if extra not in set(self._table_columns(conn, "market_data_health")):
                    conn.execute(
                        f"alter table market_data_health add column {extra} {ddl_type}"
                        f" not null default {default}"
                    )
            self._rebuild_minute_bars_with_stream_identity(conn, fresh=fresh)
            conn.execute(
                """
                create index if not exists idx_realtime_ticks_stream
                    on realtime_ticks(stream_id, symbol, received_at);
                """
            )
            conn.execute(
                """
                create index if not exists idx_realtime_ticks_dedup
                    on realtime_ticks(dedup_key);
                """
            )
            conn.execute(
                """
                create index if not exists idx_realtime_orderbook_stream
                    on realtime_orderbook(stream_id, symbol, received_at);
                """
            )
            conn.execute("commit")
        except Exception:
            conn.execute("rollback")
            raise
        counts_after = self._row_counts(conn)
        for table, before in counts_before.items():
            after = counts_after.get(table, -1)
            if after < before:
                raise RealtimeStoreMigrationError(
                    f"migration v2 lost rows in {table}: {before} -> {after}"
                )
        conn.execute(
            """
            insert or replace into schema_migrations
              (version, applied_at, description, rows_before, rows_after)
            values (?, ?, ?, ?, ?)
            """,
            (
                2,
                datetime.now(timezone.utc).isoformat(),
                "add market/venue/session/feed metadata; stream-scoped minute bar identity",
                sum(counts_before.values()),
                sum(counts_after.values()),
            ),
        )
        conn.commit()

    @staticmethod
    def _row_counts(conn: sqlite3.Connection) -> dict[str, int]:
        counts: dict[str, int] = {}
        for table in ("realtime_ticks", "realtime_orderbook", "realtime_minute_bars"):
            try:
                counts[table] = int(
                    conn.execute(f"select count(*) from {table}").fetchone()[0]
                )
            except sqlite3.Error:
                counts[table] = 0
        return counts

    def _rebuild_minute_bars_with_stream_identity(
        self, conn: sqlite3.Connection, *, fresh: bool
    ) -> None:
        """``realtime_minute_bars`` 의 PK 에 ``stream_id`` 를 포함시킨다.

        이전 PK 는 ``(symbol, minute_start)`` 뿐이어서 KRX·NXT·통합 피드가 같은 행을
        다투었다. ``insert or replace`` 였으므로 마지막에 도착한 피드가 앞선 피드의
        거래량을 덮어썼고, 반대로 합산 로직이 있었다면 이중 계산이 됐다. venue 별 bar 와
        통합 bar 를 **애초에 다른 행**으로 만들어 구조적으로 막는다.

        테이블 이름은 유지한다 — 기존 SQL 을 그대로 쓸 수 있게 하는 compatibility 전략.
        """
        columns = set(self._table_columns(conn, "realtime_minute_bars"))
        if "stream_id" in columns:
            return
        conn.execute(
            """
            create table realtime_minute_bars_v2 (
                -- DEFAULT 는 필수다. 이 컬럼을 모르는 writer(마이그레이션 시점에 실행 중인
                -- 구버전 프로세스 포함)의 INSERT 가 NOT NULL 로 죽지 않게 한다.
                stream_id text not null default '',
                symbol text not null,
                minute_start text not null,
                open real not null,
                high real not null,
                low real not null,
                close real not null,
                volume integer not null,
                vwap real not null,
                trade_count integer not null,
                spread_bps real not null,
                orderbook_imbalance real not null,
                liquidity_score real not null,
                volatility real not null,
                last_update_age_ms real not null,
                source_record_ids_json text not null,
                market_group text not null default '',
                exchange text not null default '',
                venue text not null default 'UNKNOWN',
                session text not null default 'UNKNOWN',
                currency text not null default '',
                feed_scope text not null default 'UNKNOWN',
                tr_id text not null default '',
                subscription_key text not null default '',
                is_consolidated integer not null default 0,
                is_tradeable integer not null default 0,
                metadata_inferred integer not null default 1,
                primary key(stream_id, symbol, minute_start)
            )
            """
        )
        conn.execute(
            f"""
            insert into realtime_minute_bars_v2
                (stream_id, symbol, minute_start, open, high, low, close, volume, vwap,
                 trade_count, spread_bps, orderbook_imbalance, liquidity_score, volatility,
                 last_update_age_ms, source_record_ids_json, metadata_inferred)
            select '', symbol, minute_start, open, high, low, close, volume, vwap,
                   trade_count, spread_bps, orderbook_imbalance, liquidity_score, volatility,
                   last_update_age_ms, source_record_ids_json, {0 if fresh else 1}
            from realtime_minute_bars
            """
        )
        conn.execute("drop table realtime_minute_bars")
        conn.execute("alter table realtime_minute_bars_v2 rename to realtime_minute_bars")
        conn.execute(
            """
            create index if not exists idx_realtime_minute_bars_symbol_time
                on realtime_minute_bars(symbol, minute_start)
            """
        )

    def save_ticks(self, ticks: tuple[RealtimeTradeTick, ...]) -> int:
        rows = [
            (
                tick.record_id,
                tick.symbol,
                tick.exchange_timestamp.isoformat(),
                tick.received_at.isoformat(),
                tick.source,
                tick.price,
                tick.volume,
                tick.trade_direction,
                tick.sequence_key,
                tick.raw_checksum,
                tick.latency_ms,
                *_metadata_values(tick.meta),
                tick.dedup_key,
            )
            for tick in ticks
        ]
        inserted = self._insert_many(
            f"""
            insert or ignore into realtime_ticks
            (record_id, symbol, exchange_timestamp, received_at, source, price, volume,
             trade_direction, sequence_key, raw_checksum, latency_ms,
             {", ".join(_METADATA_COLUMN_NAMES)}, dedup_key)
            values ({", ".join("?" * (11 + len(_METADATA_COLUMN_NAMES) + 1))})
            """,
            rows,
        )
        self._save_source_events(
            [
                (
                    f"trade:{tick.record_id}",
                    "trade",
                    tick.symbol,
                    tick.exchange_timestamp.isoformat(),
                    tick.source,
                    json.dumps(
                        {"price": tick.price, "volume": tick.volume, "record_id": tick.record_id},
                        ensure_ascii=True,
                    ),
                )
                for tick in ticks
            ]
        )
        return inserted

    def save_orderbooks(self, snapshots: tuple[RealtimeOrderbookSnapshot, ...]) -> int:
        rows = [
            (
                snapshot.record_id,
                snapshot.symbol,
                snapshot.exchange_timestamp.isoformat(),
                snapshot.received_at.isoformat(),
                snapshot.source,
                snapshot.best_bid,
                snapshot.best_ask,
                snapshot.spread_bps,
                snapshot.total_bid_volume,
                snapshot.total_ask_volume,
                snapshot.imbalance,
                json.dumps(to_jsonable(snapshot.levels), ensure_ascii=True, sort_keys=True),
                snapshot.sequence_key,
                snapshot.raw_checksum,
                snapshot.latency_ms,
                *_metadata_values(snapshot.meta),
            )
            for snapshot in snapshots
        ]
        inserted = self._insert_many(
            f"""
            insert or ignore into realtime_orderbook
            (record_id, symbol, exchange_timestamp, received_at, source, best_bid, best_ask,
             spread_bps, total_bid_volume, total_ask_volume, imbalance, levels_json,
             sequence_key, raw_checksum, latency_ms, {", ".join(_METADATA_COLUMN_NAMES)})
            values ({", ".join("?" * (15 + len(_METADATA_COLUMN_NAMES)))})
            """,
            rows,
        )
        self._save_source_events(
            [
                (
                    f"orderbook:{snapshot.record_id}",
                    "orderbook",
                    snapshot.symbol,
                    snapshot.exchange_timestamp.isoformat(),
                    snapshot.source,
                    json.dumps(
                        {
                            "best_bid": snapshot.best_bid,
                            "best_ask": snapshot.best_ask,
                            "spread_bps": snapshot.spread_bps,
                            "record_id": snapshot.record_id,
                        },
                        ensure_ascii=True,
                    ),
                )
                for snapshot in snapshots
            ]
        )
        return inserted

    def save_minute_bars(self, bars: tuple[RealtimeMinuteBar, ...]) -> int:
        rows = [
            (
                bar.stream_id,
                bar.symbol,
                bar.minute_start.isoformat(),
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
                bar.vwap,
                bar.trade_count,
                bar.spread_bps,
                bar.orderbook_imbalance,
                bar.liquidity_score,
                bar.volatility,
                bar.last_update_age_ms,
                json.dumps(list(bar.source_record_ids), ensure_ascii=True),
                *_metadata_values(bar.meta, include_stream_id=False),
            )
            for bar in bars
        ]
        inserted = self._insert_many(
            f"""
            insert or replace into realtime_minute_bars
            (stream_id, symbol, minute_start, open, high, low, close, volume, vwap, trade_count,
             spread_bps, orderbook_imbalance, liquidity_score, volatility, last_update_age_ms,
             source_record_ids_json,
             {", ".join(n for n in _METADATA_COLUMN_NAMES if n != "stream_id")})
            values ({", ".join("?" * (16 + len(_METADATA_COLUMN_NAMES) - 1))})
            """,
            rows,
        )
        self._save_source_events(
            [
                (
                    f"minute_bar:{bar.stream_id}:{bar.symbol}:{bar.minute_start.isoformat()}",
                    "minute_bar",
                    bar.symbol,
                    bar.minute_start.isoformat(),
                    "realtime_aggregation",
                    json.dumps(
                        {"close": bar.close, "volume": bar.volume, "trade_count": bar.trade_count},
                        ensure_ascii=True,
                    ),
                )
                for bar in bars
            ]
        )
        return inserted

    def save_health(self, health: MarketDataHealth) -> None:
        # Health is a state sample, not a tick feed. A minute bucket prevents a
        # tight polling loop from producing millions of duplicate rows per day.
        #
        # health 는 symbol 단위가 아니라 (symbol, market, venue, session, feed_scope)
        # 단위 상태다. 같은 종목이 KRX 정규장과 NXT 애프터마켓에서 전혀 다른 건전성을
        # 가질 수 있기 때문이다. PK 는 (symbol, checked_at) 그대로 두되 metadata 를
        # 함께 기록해 조회 시 분해할 수 있게 한다.
        checked_at = health.checked_at.replace(second=0, microsecond=0)
        self._execute_with_retry(
            """
            insert or replace into market_data_health
            (symbol, checked_at, quote_count, orderbook_count, latest_tick_at, latest_orderbook_at,
             max_quote_age_ms, max_orderbook_age_ms, source, source_quality_score,
             ok_for_live_buy, reason_codes_json,
             market_group, venue, session, feed_scope, is_consolidated, depth_level_count)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                health.symbol,
                checked_at.isoformat(),
                health.quote_count,
                health.orderbook_count,
                health.latest_tick_at.isoformat() if health.latest_tick_at else None,
                health.latest_orderbook_at.isoformat() if health.latest_orderbook_at else None,
                health.max_quote_age_ms,
                health.max_orderbook_age_ms,
                health.source,
                health.source_quality_score,
                1 if health.ok_for_live_buy else 0,
                json.dumps(list(health.reason_codes), ensure_ascii=True),
                health.market_group,
                health.venue or "UNKNOWN",
                health.session or "UNKNOWN",
                health.feed_scope or "UNKNOWN",
                1 if health.is_consolidated else 0,
                int(health.depth_level_count),
            ),
        )

    def prune_operational_history(self, *, retention_hours: int | None = None) -> int:
        hours = (
            max(1, int(retention_hours))
            if retention_hours is not None
            else max(1, int(os.getenv("REALTIME_OPERATIONAL_RETENTION_HOURS", "24")))
        )
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with closing(self._connect()) as conn:
            before = conn.total_changes
            conn.execute("delete from market_data_health where checked_at < ?", (cutoff,))
            conn.execute("delete from data_source_events where observed_at < ?", (cutoff,))
            conn.commit()
            return int(conn.total_changes - before)

    def backfill_source_events(self, *, retention_hours: int | None = None) -> int:
        hours = max(
            1,
            int(
                retention_hours
                if retention_hours is not None
                else os.getenv("REALTIME_OPERATIONAL_RETENTION_HOURS", "24")
            ),
        )
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with closing(self._connect()) as conn:
            before = conn.total_changes
            conn.execute(
                """
                insert or ignore into data_source_events
                  (event_id, event_type, symbol, observed_at, source, payload_json)
                select 'trade:' || record_id, 'trade', symbol, exchange_timestamp, source,
                       json_object('price', price, 'volume', volume, 'record_id', record_id)
                from realtime_ticks where exchange_timestamp >= ?
                """,
                (cutoff,),
            )
            conn.execute(
                """
                insert or ignore into data_source_events
                  (event_id, event_type, symbol, observed_at, source, payload_json)
                select 'orderbook:' || record_id, 'orderbook', symbol, exchange_timestamp, source,
                       json_object('best_bid', best_bid, 'best_ask', best_ask,
                                   'spread_bps', spread_bps, 'record_id', record_id)
                from realtime_orderbook where exchange_timestamp >= ?
                """,
                (cutoff,),
            )
            conn.commit()
            return int(conn.total_changes - before)

    def latest_tick(self, symbol: str) -> RealtimeTradeTick | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                select symbol, exchange_timestamp, received_at, source, price, volume,
                       trade_direction, sequence_key, raw_checksum, latency_ms,
                       """ + _METADATA_SELECT + """
                from realtime_ticks
                where symbol = ?
                order by received_at desc, exchange_timestamp desc
                limit 1
                """,
                (symbol,),
            ).fetchone()
        if row is None:
            return None
        return RealtimeTradeTick(
            symbol=row[0],
            exchange_timestamp=_parse_dt(row[1]),
            received_at=_parse_dt(row[2]),
            source=row[3],
            price=float(row[4]),
            volume=int(row[5]),
            trade_direction=row[6],
            sequence_key=row[7],
            raw_checksum=row[8],
            latency_ms=float(row[9]),
            meta=_metadata_from_tail(row, 10),
        )

    def latest_orderbook(self, symbol: str) -> RealtimeOrderbookSnapshot | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                select symbol, exchange_timestamp, received_at, source, levels_json,
                       sequence_key, raw_checksum, latency_ms,
                       """ + _METADATA_SELECT + """
                from realtime_orderbook
                where symbol = ?
                order by received_at desc, exchange_timestamp desc
                limit 1
                """,
                (symbol,),
            ).fetchone()
        if row is None:
            return None
        from app.data.realtime_types import OrderbookLevel

        levels = tuple(OrderbookLevel(**item) for item in json.loads(row[4]))
        return RealtimeOrderbookSnapshot(
            symbol=row[0],
            exchange_timestamp=_parse_dt(row[1]),
            received_at=_parse_dt(row[2]),
            source=row[3],
            levels=levels,
            sequence_key=row[5],
            raw_checksum=row[6],
            latency_ms=float(row[7]),
            meta=_metadata_from_tail(row, 8),
        )

    def active_symbols(self, since: datetime, *, limit: int = 200) -> tuple[str, ...]:
        """Symbols with at least one tick since the cutoff, most-recent first."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select symbol, max(received_at) as last_at
                from realtime_ticks
                where received_at >= ?
                group by symbol
                order by last_at desc
                limit ?
                """,
                (since.isoformat(), int(limit)),
            ).fetchall()
        return tuple(str(row[0]) for row in rows if row and row[0])

    def recent_ticks(
        self,
        symbol: str,
        since: datetime,
        *,
        until: datetime | None = None,
    ) -> tuple[RealtimeTradeTick, ...]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select symbol, exchange_timestamp, received_at, source, price, volume,
                       trade_direction, sequence_key, raw_checksum, latency_ms,
                       """ + _METADATA_SELECT + """
                from realtime_ticks
                where symbol = ? and exchange_timestamp >= ?
                  and (? is null or exchange_timestamp <= ?)
                  and (? is null or received_at <= ?)
                order by exchange_timestamp asc, received_at asc
                """,
                (
                    symbol,
                    since.isoformat(),
                    until.isoformat() if until is not None else None,
                    until.isoformat() if until is not None else None,
                    until.isoformat() if until is not None else None,
                    until.isoformat() if until is not None else None,
                ),
            ).fetchall()
        return tuple(
            RealtimeTradeTick(
                symbol=row[0],
                exchange_timestamp=_parse_dt(row[1]),
                received_at=_parse_dt(row[2]),
                source=row[3],
                price=float(row[4]),
                volume=int(row[5]),
                trade_direction=row[6],
                sequence_key=row[7],
                raw_checksum=row[8],
                latency_ms=float(row[9]),
                meta=_metadata_from_tail(row, 10),
            )
            for row in rows
        )

    def recent_orderbooks(
        self,
        symbol: str,
        since: datetime,
        *,
        until: datetime | None = None,
    ) -> tuple[RealtimeOrderbookSnapshot, ...]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select symbol, exchange_timestamp, received_at, source, levels_json,
                       sequence_key, raw_checksum, latency_ms,
                       """ + _METADATA_SELECT + """
                from realtime_orderbook
                where symbol = ? and exchange_timestamp >= ?
                  and (? is null or exchange_timestamp <= ?)
                  and (? is null or received_at <= ?)
                order by exchange_timestamp asc, received_at asc
                """,
                (
                    symbol,
                    since.isoformat(),
                    until.isoformat() if until is not None else None,
                    until.isoformat() if until is not None else None,
                    until.isoformat() if until is not None else None,
                    until.isoformat() if until is not None else None,
                ),
            ).fetchall()
        from app.data.realtime_types import OrderbookLevel

        return tuple(
            RealtimeOrderbookSnapshot(
                symbol=row[0],
                exchange_timestamp=_parse_dt(row[1]),
                received_at=_parse_dt(row[2]),
                source=row[3],
                levels=tuple(OrderbookLevel(**item) for item in json.loads(row[4])),
                sequence_key=row[5],
                raw_checksum=row[6],
                latency_ms=float(row[7]),
                meta=_metadata_from_tail(row, 8),
            )
            for row in rows
        )

    def preferred_minute_bar_stream(
        self, symbol: str, since: datetime
    ) -> str | None:
        """이 종목의 분 bar 시계열로 쓸 **단일** 스트림.

        bar 는 스트림별로 저장되므로 한 종목·한 분에 여러 행이 있을 수 있다 (예: 웹소켓
        ``FREE_REALTIME`` 과 REST ``REST_SNAPSHOT``). 그것을 한 시계열로 이어 붙이면
        같은 분이 두 번 등장해 "N개 관측 = N분"이라는 가정이 깨지고, 수익률·변동성이
        서로 다른 피드에서 계산된 값으로 오염된다.

        선택 규칙 (결정론적):

        1. 창 안에서 **분 커버리지가 가장 넓은** 스트림 — 시계열 연속성이 우선이다.
        2. 동수면 tradeable 한 스트림 (REST 스냅샷보다 실시간 피드).
        3. 그래도 동수면 체결 수, 마지막으로 ``stream_id`` 사전순.
        """
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select stream_id,
                       count(distinct minute_start) as minutes,
                       max(is_tradeable) as tradeable,
                       sum(trade_count) as trades
                from realtime_minute_bars
                where symbol = ? and minute_start >= ?
                group by stream_id
                """,
                (symbol, since.isoformat()),
            ).fetchall()
        if not rows:
            return None
        ranked = sorted(
            rows,
            key=lambda row: (
                int(row[1] or 0),
                int(row[2] or 0),
                int(row[3] or 0),
                str(row[0] or ""),
            ),
            reverse=True,
        )
        return str(ranked[0][0] or "")

    def recent_minute_bars(
        self,
        symbol: str,
        since: datetime,
        *,
        limit: int = 120,
        stream_id: str | None = None,
    ) -> tuple[RealtimeMinuteBar, ...]:
        """한 종목의 분 bar 시계열. **단일 스트림만** 반환한다.

        ``stream_id`` 를 주면 그 스트림, 생략하면
        :meth:`preferred_minute_bar_stream` 이 고른 스트림이다. 여러 스트림을 섞어
        반환하면 같은 ``minute_start`` 가 중복되어 하류의 rolling return / 변동성 계산이
        조용히 잘못된 값을 낸다 (실제로 macro frame 이 그 때문에 오염됐다).
        """
        selected = (
            stream_id
            if stream_id is not None
            else self.preferred_minute_bar_stream(symbol, since)
        )
        if selected is None:
            return ()
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select symbol, minute_start, open, high, low, close, volume, vwap,
                       trade_count, spread_bps, orderbook_imbalance, liquidity_score,
                       volatility, last_update_age_ms, source_record_ids_json,
                       """ + _METADATA_SELECT + """
                from realtime_minute_bars
                where symbol = ? and minute_start >= ? and stream_id = ?
                order by minute_start desc
                limit ?
                """,
                (symbol, since.isoformat(), selected, int(limit)),
            ).fetchall()
        bars = tuple(
            RealtimeMinuteBar(
                symbol=row[0],
                minute_start=_parse_dt(row[1]),
                open=float(row[2]),
                high=float(row[3]),
                low=float(row[4]),
                close=float(row[5]),
                volume=int(row[6]),
                vwap=float(row[7]),
                trade_count=int(row[8]),
                spread_bps=float(row[9]),
                orderbook_imbalance=float(row[10]),
                liquidity_score=float(row[11]),
                volatility=float(row[12]),
                last_update_age_ms=float(row[13]),
                source_record_ids=tuple(json.loads(row[14] or "[]")),
                meta=_metadata_from_tail(row, 15),
            )
            for row in rows
        )
        return tuple(reversed(bars))

    def counts_since(self, symbol: str, since: datetime) -> tuple[int, int]:
        with closing(self._connect()) as conn:
            tick_count = conn.execute(
                "select count(*) from realtime_ticks where symbol = ? and exchange_timestamp >= ?",
                (symbol, since.isoformat()),
            ).fetchone()[0]
            orderbook_count = conn.execute(
                "select count(*) from realtime_orderbook where symbol = ? and exchange_timestamp >= ?",
                (symbol, since.isoformat()),
            ).fetchone()[0]
        return int(tick_count), int(orderbook_count)

    def build_latest_minute_bar(
        self,
        symbol: str,
        *,
        now: datetime | None = None,
        stream_id: str | None = None,
    ) -> RealtimeMinuteBar | None:
        """이번 분의 bar 를 만든다. ``stream_id`` 를 주면 그 스트림만 집계한다.

        ``stream_id`` 를 생략하면 이번 분에 관측된 **모든 스트림에 대해 각각** bar 를
        만들고 가장 활발한 것을 반환한다. venue 별 bar 와 통합 bar 를 절대 합산하지
        않는다 — 같은 체결이 두 피드로 오면 거래량이 두 배가 되기 때문이다.
        """
        now = now or datetime.now(timezone.utc)
        minute_start = now.replace(second=0, microsecond=0)
        all_ticks = self.recent_ticks(symbol, minute_start)
        if not all_ticks:
            return None
        streams: dict[str, list[RealtimeTradeTick]] = {}
        for tick in all_ticks:
            streams.setdefault(tick.meta.stream_id, []).append(tick)
        if stream_id is not None:
            selected = {stream_id: streams.get(stream_id, [])}
            if not selected[stream_id]:
                return None
        else:
            selected = streams
        built: list[RealtimeMinuteBar] = []
        for key, ticks in selected.items():
            bar = self._build_minute_bar_for_stream(symbol, minute_start, ticks, now, key)
            if bar is not None:
                built.append(bar)
        if not built:
            return None
        self.save_minute_bars(tuple(built))
        # 가장 체결이 많은 스트림을 대표 bar 로 돌려준다.
        return max(built, key=lambda bar: (bar.trade_count, bar.volume))

    def _build_minute_bar_for_stream(
        self,
        symbol: str,
        minute_start: datetime,
        ticks: list[RealtimeTradeTick],
        now: datetime,
        stream_id: str,
    ) -> RealtimeMinuteBar | None:
        if not ticks:
            return None
        orderbook = self.latest_orderbook_for_stream(symbol, stream_id)
        prices = [tick.price for tick in ticks]
        volumes = [max(0, tick.volume) for tick in ticks]
        total_volume = sum(volumes)
        vwap = (
            sum(price * volume for price, volume in zip(prices, volumes, strict=True)) / total_volume
            if total_volume > 0
            else prices[-1]
        )
        mean = sum(prices) / len(prices)
        variance = sum((price - mean) ** 2 for price in prices) / max(1, len(prices))
        last_update_age_ms = max(0.0, (now - ticks[-1].received_at).total_seconds() * 1000)
        liquidity_score = min(1.0, total_volume / 100_000.0)
        bar = RealtimeMinuteBar(
            symbol=symbol,
            minute_start=minute_start,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=total_volume,
            vwap=vwap,
            trade_count=len(ticks),
            spread_bps=orderbook.spread_bps if orderbook else 0.0,
            orderbook_imbalance=orderbook.imbalance if orderbook else 0.0,
            liquidity_score=liquidity_score,
            volatility=variance**0.5,
            last_update_age_ms=last_update_age_ms,
            source_record_ids=tuple(tick.record_id for tick in ticks),
            meta=ticks[-1].meta,
        )
        return bar

    def latest_orderbook_for_stream(
        self, symbol: str, stream_id: str
    ) -> RealtimeOrderbookSnapshot | None:
        """같은 스트림(venue+feed_scope+TR)의 최신 호가.

        분 bar 의 spread/imbalance 를 다른 거래소 호가로 채우면 안 된다. 스트림 안에
        호가가 없으면(체결만 구독한 경우) ``None`` 을 돌려주고, 호출자는 spread 0 이
        아니라 "호가 없음"으로 다뤄야 한다.
        """
        if not stream_id:
            return self.latest_orderbook(symbol)
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                select symbol, exchange_timestamp, received_at, source, levels_json,
                       sequence_key, raw_checksum, latency_ms,
                       """ + _METADATA_SELECT + """
                from realtime_orderbook
                where symbol = ? and stream_id = ?
                order by received_at desc, exchange_timestamp desc
                limit 1
                """,
                (symbol, stream_id),
            ).fetchone()
        if row is None:
            return None
        from app.data.realtime_types import OrderbookLevel

        return RealtimeOrderbookSnapshot(
            symbol=row[0],
            exchange_timestamp=_parse_dt(row[1]),
            received_at=_parse_dt(row[2]),
            source=row[3],
            levels=tuple(OrderbookLevel(**item) for item in json.loads(row[4])),
            sequence_key=row[5],
            raw_checksum=row[6],
            latency_ms=float(row[7]),
            meta=_metadata_from_tail(row, 8),
        )

    def stream_inventory(self, since: datetime) -> tuple[dict[str, Any], ...]:
        """스트림별 수집 현황 (관측·진단용).

        같은 종목이 여러 스트림으로 들어오는지, 어느 스트림이 조용한지를 한 번에 본다.
        """
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select stream_id, market_group, venue, session, feed_scope, tr_id,
                       count(distinct symbol) as symbols, count(*) as ticks,
                       max(received_at) as last_at,
                       sum(metadata_inferred) as inferred
                from realtime_ticks
                where received_at >= ?
                group by stream_id, market_group, venue, session, feed_scope, tr_id
                order by ticks desc
                """,
                (since.isoformat(),),
            ).fetchall()
        return tuple(
            {
                "stream_id": row[0],
                "market_group": row[1],
                "venue": row[2],
                "session": row[3],
                "feed_scope": row[4],
                "tr_id": row[5],
                "symbols": int(row[6]),
                "ticks": int(row[7]),
                "last_at": row[8],
                "inferred_rows": int(row[9] or 0),
            }
            for row in rows
        )

    def cross_stream_duplicate_count(self, since: datetime) -> int:
        """서로 다른 스트림에 동일 체결(``dedup_key``)이 존재하는 건수.

        통합 피드와 venue 별 피드를 동시에 구독하면 0 보다 커진다. bar 는 스트림별로
        따로 만들어지므로 거래량이 이중 계산되지는 않지만, 구독 예산이 낭비되고 있다는
        신호이므로 운영 화면에 노출한다.
        """
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                select count(*) from (
                    select dedup_key
                    from realtime_ticks
                    where received_at >= ? and dedup_key != ''
                    group by dedup_key
                    having count(distinct stream_id) > 1
                )
                """,
                (since.isoformat(),),
            ).fetchone()
        return int(row[0]) if row else 0

    def _insert_many(self, sql: str, rows: list[tuple[Any, ...]]) -> int:
        if not rows:
            return 0

        def run(conn: sqlite3.Connection) -> int:
            before = conn.total_changes
            conn.executemany(sql, rows)
            conn.commit()
            return int(conn.total_changes - before)

        return self._with_retry(run)

    def _execute_with_retry(self, sql: str, params: tuple[Any, ...]) -> None:
        def run(conn: sqlite3.Connection) -> int:
            conn.execute(sql, params)
            conn.commit()
            return 0

        self._with_retry(run)

    def _with_retry(self, action: Any) -> int:
        """SQLite lock 에 대해 bounded retry + backoff.

        학습기가 같은 파일을 읽는 동안 쓰기가 잠길 수 있다. 여기서 무한 대기하면 수집·주문
        루프가 함께 멈추므로 시도 횟수를 제한하고, 끝내 실패하면 예외를 올려 호출자가
        fail-closed 하게 한다.
        """
        attempts = max(1, _env_int("REALTIME_STORE_LOCK_RETRIES", 4))
        delay = max(0.01, _env_float("REALTIME_STORE_LOCK_BACKOFF_SEC", 0.05))
        last: sqlite3.OperationalError | None = None
        for attempt in range(attempts):
            try:
                with closing(self._connect()) as conn:
                    return action(conn)
            except sqlite3.OperationalError as exc:
                if "lock" not in str(exc).lower():
                    raise
                last = exc
                if attempt == attempts - 1:
                    break
                time.sleep(delay * (2**attempt))
        raise last if last is not None else RuntimeError("sqlite retry failed")

    def _save_source_events(self, rows: list[tuple[Any, ...]]) -> int:
        if not rows:
            return 0
        return self._insert_many(
            """
            insert or ignore into data_source_events
              (event_id, event_type, symbol, observed_at, source, payload_json)
            values (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30)


def _parse_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def cutoff_ms(now: datetime, max_age_ms: int) -> datetime:
    return now - timedelta(milliseconds=max_age_ms)

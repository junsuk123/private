from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import closing
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from app.graph import Triple
from app.runtime import DataMode, default_environment
from app.schemas.domain import (
    ClassifiedEvent,
    MacroMetricRecord,
    MarketSnapshot,
    RawSourceRecord,
    RealtimeExecution,
    RealtimeQuote,
    ReasoningPath,
)


_T = TypeVar("_T")


@dataclass(frozen=True)
class StoredResearch:
    events: tuple[ClassifiedEvent, ...]
    raw_records: tuple[RawSourceRecord, ...]
    market_snapshots: tuple[MarketSnapshot, ...]
    macro_metrics: tuple[MacroMetricRecord, ...]
    realtime_quotes: tuple[RealtimeQuote, ...]
    realtime_executions: tuple[RealtimeExecution, ...]
    graph_triples: tuple[Triple, ...]
    reasoning_paths: tuple[ReasoningPath, ...]


class LocalResearchStore:
    def __init__(
        self,
        root: Path | None = None,
        retention_days: int | None = None,
        mode: DataMode | None = None,
    ) -> None:
        if root is None:
            environment = default_environment()
            self.root = environment.store_dir
            self.mode = environment.mode
        else:
            self.root = root
            self.mode = mode or "custom"
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "research.sqlite3"
        self.retention_days = (
            retention_days
            if retention_days is not None
            else max(1, int(os.getenv("RESEARCH_RETENTION_DAYS", "30")))
        )
        self._init_db()

    def save_research_result(self, result: Any) -> dict[str, int]:
        self.prune_stale()
        saved = {
            "events": self._insert_unique("events", result.events, _event_key, _event_observed_at),
            "raw_records": self._insert_unique(
                "raw_records", result.raw_records, _raw_key, _raw_observed_at
            ),
            "market_snapshots": self._insert_unique(
                "market_snapshots", result.market_snapshots, _market_key, _market_observed_at
            ),
            "macro_metrics": self._insert_unique(
                "macro_metrics", result.macro_metrics, _macro_key, _macro_observed_at
            ),
        }
        self.save_typed_market_snapshots(tuple(result.market_snapshots))
        return saved

    def save_macro_metrics(self, records: Any) -> int:
        """Persist macro observations on their own, outside a full research run.

        The weekend backfill replays FRED history so the store holds a series rather
        than a single latest print. Deduplication is the same ``_macro_key`` used by
        ``save_research_result``, so replaying an overlapping range is idempotent and
        safe to repeat.
        """
        return self._insert_unique(
            "macro_metrics", tuple(records), _macro_key, _macro_observed_at
        )

    def save_graph_and_reasoning(
        self,
        triples: tuple[Triple, ...],
        reasoning_paths: tuple[ReasoningPath, ...],
    ) -> dict[str, int]:
        self.prune_stale()
        return {
            "graph_triples": self._insert_unique(
                "graph_triples", triples, _triple_key, _now_observed_at
            ),
            "reasoning_paths": self._insert_unique(
                "reasoning_paths", reasoning_paths, _reasoning_key, _now_observed_at
            ),
        }

    def save_realtime_records(
        self,
        quotes: tuple[RealtimeQuote, ...] = (),
        executions: tuple[RealtimeExecution, ...] = (),
    ) -> dict[str, int]:
        self.prune_stale()
        saved = {
            "realtime_quotes": self._insert_unique(
                "realtime_quotes", quotes, _realtime_quote_key, _realtime_quote_observed_at
            ),
            "realtime_executions": self._insert_unique(
                "realtime_executions",
                executions,
                _realtime_execution_key,
                _realtime_execution_observed_at,
            ),
        }
        self.save_typed_realtime_quotes(quotes)
        return saved

    def save_typed_market_snapshots(self, snapshots: tuple[MarketSnapshot, ...]) -> int:
        rows = []
        for snapshot in snapshots:
            source = snapshot.source
            observed_at = source.observed_at or source.retrieved_at
            if source.source_name in {"listed_universe_reference", "listed_universe_catalog"}:
                # Static universe rows are refreshed every collection cycle, but
                # they are not intraday observations. One row per UTC day keeps
                # recency semantics without adding ~1,500 rows every 15 seconds.
                observed_at = _as_aware(observed_at).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
            source_id = source.source_id or source.source_name
            rows.append(
                (
                    snapshot.ticker,
                    _as_aware(observed_at).isoformat(),
                    snapshot.market,
                    snapshot.last_price,
                    snapshot.average_daily_trading_value,
                    snapshot.volatility_20d,
                    source_id,
                )
            )
        return self._insert_typed(
            """
            insert or replace into typed_market_snapshots
              (ticker, observed_at, market, last_price, average_daily_trading_value,
               volatility_20d, source_id)
            values (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def save_typed_realtime_quotes(self, quotes: tuple[RealtimeQuote, ...]) -> int:
        rows = []
        for quote in quotes:
            source_id = (
                (quote.source.source_id or quote.source.source_name)
                if quote.source is not None
                else "unknown"
            )
            rows.append(
                (
                    quote.ticker,
                    _as_aware(quote.observed_at).isoformat(),
                    quote.last_price,
                    quote.bid_price,
                    quote.ask_price,
                    quote.volume,
                    source_id,
                )
            )
        return self._insert_typed(
            """
            insert or replace into typed_realtime_quotes
              (ticker, observed_at, price, bid, ask, volume, source_id)
            values (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def save_typed_ohlcv_bars(self, bars: tuple[Any, ...], *, source_id: str = "realtime") -> int:
        rows = []
        for bar in bars:
            ticker = getattr(bar, "ticker", None) or getattr(bar, "symbol", None)
            observed_at = getattr(bar, "as_of", None) or getattr(bar, "minute_start", None)
            if not ticker or observed_at is None:
                continue
            rows.append(
                (
                    str(ticker),
                    _as_aware(observed_at).isoformat(),
                    float(bar.open),
                    float(bar.high),
                    float(bar.low),
                    float(bar.close),
                    float(bar.volume),
                    source_id,
                )
            )
        return self._insert_typed(
            """
            insert or replace into typed_ohlcv_bars
              (ticker, observed_at, open, high, low, close, volume, source_id)
            values (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def save_typed_candidate_scores(self, scores: tuple[dict[str, Any], ...]) -> int:
        rows = []
        now = datetime.now(timezone.utc)
        for score in scores:
            ticker = str(score.get("ticker") or score.get("symbol") or "").strip()
            if not ticker:
                continue
            observed_at = score.get("observed_at") or score.get("as_of") or now
            if isinstance(observed_at, str):
                observed_at = datetime.fromisoformat(observed_at)
            rows.append(
                (
                    ticker,
                    _as_aware(observed_at).isoformat(),
                    str(score.get("stage") or "candidate_selection"),
                    float(score.get("score") or 0.0),
                    int(score.get("reason_mask") or 0),
                    str(score.get("backend") or "rule"),
                )
            )
        return self._insert_typed(
            """
            insert or replace into typed_candidate_scores
              (ticker, observed_at, stage, score, reason_mask, backend)
            values (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _sync_from_realtime(
        self,
        realtime_db_path: str | Path,
        *,
        source_table: str,
        columns: str,
        insert_sql: str,
        limit: int,
    ) -> int:
        """Project rows appended to a realtime table since the last call.

        Both projections used to read ``order by <timestamp> desc limit 20000``.
        Neither table has an index on that column alone -- only ``(symbol, time)``
        -- so SQLite full-scanned the source and sorted it in a temp b-tree, then
        re-wrote the same 20,000 rows the previous cycle had already written.
        Measured on the live store (4.5M ticks): 33.5s per call, repeated every
        refresh cycle, which is why a cycle configured for 180s took ~224s and the
        UI never left "refreshing".

        ``rowid`` is monotonic for both tables -- ticks are appended, and the bar
        table's ``insert or replace`` re-inserts an updated bar at a higher rowid
        -- so a stored watermark turns the same work into an indexed seek over
        only what is new (measured: 0.10s). The watermark is advanced only after
        the insert commits, so a failed cycle re-copies rather than skips.
        """
        path = Path(realtime_db_path)
        if not path.exists():
            return 0
        batch = max(1, int(limit))
        watermark_key = f"{path.as_posix()}::{source_table}"
        with closing(sqlite3.connect(path, timeout=30)) as conn:
            highest = conn.execute(f"select max(rowid) from {source_table}").fetchone()[0]
            if highest is None:
                return 0
            watermark = self._read_sync_watermark(watermark_key)
            if watermark <= 0 or watermark > int(highest):
                # First projection, or a rebuilt source database whose rowids
                # restarted below the stored mark. Seed from the tail so a cold
                # start copies the recent window instead of replaying history.
                watermark = max(0, int(highest) - batch)
            rows = conn.execute(
                f"""
                select rowid, {columns}
                from {source_table}
                where rowid > ?
                order by rowid
                limit ?
                """,
                (watermark, batch),
            ).fetchall()
        if not rows:
            return 0
        inserted = self._insert_typed(insert_sql, [tuple(row[1:]) for row in rows])
        self._write_sync_watermark(watermark_key, int(rows[-1][0]))
        return inserted

    def sync_realtime_ohlcv(
        self,
        realtime_db_path: str | Path = "data/store/realtime_market_data.sqlite3",
        *,
        limit: int = 20_000,
    ) -> int:
        """Idempotently project recent realtime bars into the research typed schema."""
        return self._sync_from_realtime(
            realtime_db_path,
            source_table="realtime_minute_bars",
            columns="symbol, minute_start, open, high, low, close, volume",
            insert_sql="""
            insert or replace into typed_ohlcv_bars
              (ticker, observed_at, open, high, low, close, volume, source_id)
            values (?, ?, ?, ?, ?, ?, ?, 'realtime_market_data')
            """,
            limit=limit,
        )

    def sync_realtime_quotes(
        self,
        realtime_db_path: str | Path = "data/store/realtime_market_data.sqlite3",
        *,
        limit: int = 20_000,
    ) -> int:
        """Idempotently project trade ticks into the typed quote schema."""
        return self._sync_from_realtime(
            realtime_db_path,
            source_table="realtime_ticks",
            columns="symbol, exchange_timestamp, price, null, null, volume, source",
            insert_sql="""
            insert or replace into typed_realtime_quotes
              (ticker, observed_at, price, bid, ask, volume, source_id)
            values (?, ?, ?, ?, ?, ?, ?)
            """,
            limit=limit,
        )

    def _read_sync_watermark(self, source_key: str) -> int:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "select last_rowid from sync_watermarks where source_key = ?",
                (source_key,),
            ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def _write_sync_watermark(self, source_key: str, last_rowid: int) -> None:
        recorded_at = datetime.now(timezone.utc).isoformat()

        def write(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                insert or replace into sync_watermarks (source_key, last_rowid, updated_at)
                values (?, ?, ?)
                """,
                (source_key, int(last_rowid), recorded_at),
            )
            conn.commit()

        self._write_with_retry(write)

    def load(self) -> StoredResearch:
        self.prune_stale()
        return StoredResearch(
            events=tuple(_event_from_dict(item) for item in self._read_kind("events")),
            raw_records=tuple(_raw_from_dict(item) for item in self._read_kind("raw_records")),
            market_snapshots=tuple(
                _market_from_dict(item) for item in self._read_kind("market_snapshots")
            ),
            macro_metrics=tuple(_macro_from_dict(item) for item in self._read_kind("macro_metrics")),
            realtime_quotes=tuple(
                _realtime_quote_from_dict(item) for item in self._read_kind("realtime_quotes")
            ),
            realtime_executions=tuple(
                _realtime_execution_from_dict(item)
                for item in self._read_kind("realtime_executions")
            ),
            graph_triples=tuple(_triple_from_dict(item) for item in self._read_kind("graph_triples")),
            reasoning_paths=tuple(
                _reasoning_from_dict(item) for item in self._read_kind("reasoning_paths")
            ),
        )

    def load_analysis_inputs(self, *, prune: bool = True) -> StoredResearch:
        if prune:
            self.prune_stale()
        return StoredResearch(
            events=tuple(_event_from_dict(item) for item in self._read_kind("events")),
            raw_records=tuple(_raw_from_dict(item) for item in self._read_kind("raw_records")),
            market_snapshots=tuple(
                _market_from_dict(item) for item in self._read_kind("market_snapshots")
            ),
            macro_metrics=tuple(_macro_from_dict(item) for item in self._read_kind("macro_metrics")),
            realtime_quotes=tuple(
                _realtime_quote_from_dict(item) for item in self._read_kind("realtime_quotes")
            ),
            realtime_executions=tuple(
                _realtime_execution_from_dict(item)
                for item in self._read_kind("realtime_executions")
            ),
            graph_triples=(),
            reasoning_paths=(),
        )

    def load_live_analysis_inputs(self, *, prune: bool = True) -> StoredResearch:
        """Load a bounded, recent research window for the live intelligence path.

        The durable store can contain millions of historical market snapshots.
        Loading all of them every live refresh both stalls the worker and was the
        reason the live path previously discarded events altogether.  This
        bounded view keeps current event/macro provenance without putting the
        historical corpus on the latency-sensitive path.
        """

        if prune:
            self.prune_stale()

        def _limit(name: str, default: int) -> int:
            try:
                return max(0, int(os.getenv(name, str(default))))
            except (TypeError, ValueError):
                return default

        return StoredResearch(
            events=tuple(
                _event_from_dict(item)
                for item in self._read_kind("events", _limit("LIVE_RESEARCH_EVENT_LIMIT", 1000))
            ),
            raw_records=tuple(
                _raw_from_dict(item)
                for item in self._read_kind("raw_records", _limit("LIVE_RESEARCH_RAW_LIMIT", 1000))
            ),
            market_snapshots=tuple(
                _market_from_dict(item)
                for item in self._read_kind(
                    "market_snapshots",
                    _limit("LIVE_RESEARCH_MARKET_SNAPSHOT_LIMIT", 5000),
                )
            ),
            macro_metrics=tuple(
                _macro_from_dict(item)
                for item in self._read_kind("macro_metrics", _limit("LIVE_RESEARCH_MACRO_LIMIT", 500))
            ),
            realtime_quotes=tuple(
                _realtime_quote_from_dict(item)
                for item in self._read_kind("realtime_quotes", _limit("LIVE_RESEARCH_QUOTE_LIMIT", 5000))
            ),
            realtime_executions=tuple(
                _realtime_execution_from_dict(item)
                for item in self._read_kind(
                    "realtime_executions",
                    _limit("LIVE_RESEARCH_EXECUTION_LIMIT", 5000),
                )
            ),
            graph_triples=(),
            reasoning_paths=(),
        )

    def load_recent_events(self, *, limit: int = 500) -> tuple[ClassifiedEvent, ...]:
        """Read only the newest classified events for latency-sensitive consumers."""

        return tuple(
            _event_from_dict(item)
            for item in self._read_kind("events", max(1, int(limit)))
        )

    def summary(self, *, prune: bool = True) -> dict[str, int | str]:
        if prune:
            self.prune_stale()
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "select kind, count(*) from records group by kind order by kind"
            ).fetchall()
        counts = {kind: count for kind, count in rows}
        return {
            "events": int(counts.get("events", 0)),
            "raw_records": int(counts.get("raw_records", 0)),
            "market_snapshots": int(counts.get("market_snapshots", 0)),
            "macro_metrics": int(counts.get("macro_metrics", 0)),
            "realtime_quotes": int(counts.get("realtime_quotes", 0)),
            "realtime_executions": int(counts.get("realtime_executions", 0)),
            "graph_triples": int(counts.get("graph_triples", 0)),
            "reasoning_paths": int(counts.get("reasoning_paths", 0)),
            "database_path": str(self.db_path),
            "retention_days": self.retention_days,
        }

    def data_volume(self, *, prune: bool = True) -> dict[str, Any]:
        if prune:
            self.prune_stale()
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select kind, observed_at, inserted_at, payload
                from records
                where kind in (
                    'events',
                    'raw_records',
                    'market_snapshots',
                    'macro_metrics',
                    'realtime_quotes',
                    'realtime_executions'
                )
                order by observed_at asc
                """
            ).fetchall()

        by_kind: dict[str, int] = {}
        by_source: dict[tuple[str, str], int] = {}
        by_day: dict[tuple[str, str], int] = {}
        market_sources: dict[str, int] = {}
        ticker_counts: dict[str, int] = {}
        for kind, observed_at, _inserted_at, payload in rows:
            by_kind[kind] = by_kind.get(kind, 0) + 1
            day = str(observed_at)[:10] if observed_at else "unknown"
            by_day[(day, kind)] = by_day.get((day, kind), 0) + 1
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                data = {}
            source = data.get("source") if isinstance(data.get("source"), dict) else {}
            source_name = str(source.get("source_name") or kind)
            by_source[(kind, source_name)] = by_source.get((kind, source_name), 0) + 1
            if kind == "market_snapshots":
                market_sources[source_name] = market_sources.get(source_name, 0) + 1
                ticker = str(data.get("ticker") or "-")
                ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
            if kind in {"realtime_quotes", "realtime_executions"}:
                ticker = str(data.get("ticker") or "-")
                ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1

        return {
            "by_kind": dict(sorted(by_kind.items())),
            "by_source": [
                {"kind": kind, "source_name": source_name, "count": count}
                for (kind, source_name), count in sorted(
                    by_source.items(), key=lambda item: (item[0][0], -item[1], item[0][1])
                )
            ],
            "by_day": [
                {"date": day, "kind": kind, "count": count}
                for (day, kind), count in sorted(by_day.items())
            ],
            "market_snapshot_sources": dict(sorted(market_sources.items(), key=lambda item: -item[1])),
            "top_market_tickers": [
                {"ticker": ticker, "count": count}
                for ticker, count in sorted(ticker_counts.items(), key=lambda item: (-item[1], item[0]))[:20]
            ],
        }

    def prune_stale(self) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        macro_cutoff = datetime.now(timezone.utc) - timedelta(days=_macro_retention_days())
        cutoff_text = cutoff.isoformat()
        macro_cutoff_text = macro_cutoff.isoformat()
        try:
            batch_size = max(1_000, int(os.getenv("RESEARCH_PRUNE_BATCH_SIZE", "10000")))
        except (TypeError, ValueError):
            batch_size = 10_000
        def _retention_days(name: str, default: int) -> int:
            try:
                return max(1, int(os.getenv(name, str(default))))
            except (TypeError, ValueError):
                return default

        typed_cutoffs = {
            "typed_market_snapshots": datetime.now(timezone.utc) - timedelta(
                days=_retention_days("RESEARCH_TYPED_MARKET_RETENTION_DAYS", 7)
            ),
            "typed_realtime_quotes": datetime.now(timezone.utc) - timedelta(
                days=_retention_days("RESEARCH_TYPED_QUOTE_RETENTION_DAYS", 7)
            ),
            "typed_ohlcv_bars": datetime.now(timezone.utc) - timedelta(
                days=_retention_days("RESEARCH_TYPED_BAR_RETENTION_DAYS", 45)
            ),
            "typed_candidate_scores": datetime.now(timezone.utc) - timedelta(
                days=_retention_days("RESEARCH_TYPED_SCORE_RETENTION_DAYS", 30)
            ),
        }

        def write(conn: sqlite3.Connection) -> int:
            before = conn.total_changes
            conn.execute(
                """
                delete from records where rowid in (
                    select rowid from records
                    where (kind = 'macro_metrics' and observed_at < ?)
                       or (kind <> 'macro_metrics' and observed_at < ?)
                    limit ?
                )
                """,
                (macro_cutoff_text, cutoff_text, batch_size),
            )
            for table, typed_cutoff in typed_cutoffs.items():
                conn.execute(
                    f"delete from {table} where rowid in ("
                    f"select rowid from {table} where observed_at < ? limit ?)",
                    (typed_cutoff.isoformat(), batch_size),
                )
            conn.commit()
            return int(conn.total_changes - before)

        return self._write_with_retry(write)

    def _init_db(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute("pragma journal_mode=wal")
            conn.execute(
                """
                create table if not exists records (
                    kind text not null,
                    record_key text not null,
                    observed_at text not null,
                    inserted_at text not null,
                    payload text not null,
                    primary key (kind, record_key)
                )
                """
            )
            conn.execute(
                "create index if not exists idx_records_kind_observed on records(kind, observed_at)"
            )
            conn.execute(
                """
                create table if not exists typed_ohlcv_bars (
                    ticker text not null,
                    observed_at text not null,
                    open real,
                    high real,
                    low real,
                    close real,
                    volume real,
                    source_id text,
                    primary key (ticker, observed_at, source_id)
                )
                """
            )
            conn.execute(
                """
                create table if not exists typed_realtime_quotes (
                    ticker text not null,
                    observed_at text not null,
                    price real,
                    bid real,
                    ask real,
                    volume real,
                    source_id text,
                    primary key (ticker, observed_at, source_id)
                )
                """
            )
            conn.execute(
                """
                create table if not exists typed_market_snapshots (
                    ticker text not null,
                    observed_at text not null,
                    market text,
                    last_price real,
                    average_daily_trading_value real,
                    volatility_20d real,
                    source_id text,
                    primary key (ticker, observed_at, source_id)
                )
                """
            )
            conn.execute(
                """
                create table if not exists typed_candidate_scores (
                    ticker text not null,
                    observed_at text not null,
                    stage text not null,
                    score real,
                    reason_mask integer,
                    backend text,
                    primary key (ticker, observed_at, stage, backend)
                )
                """
            )
            conn.execute(
                """
                create table if not exists sync_watermarks (
                    source_key text primary key,
                    last_rowid integer not null,
                    updated_at text not null
                )
                """
            )
            conn.execute("create index if not exists idx_typed_ohlcv_ticker_time on typed_ohlcv_bars(ticker, observed_at)")
            conn.execute("create index if not exists idx_typed_quotes_ticker_time on typed_realtime_quotes(ticker, observed_at)")
            conn.execute("create index if not exists idx_typed_market_ticker_time on typed_market_snapshots(ticker, observed_at)")
            conn.execute("create index if not exists idx_typed_scores_ticker_time on typed_candidate_scores(ticker, observed_at)")
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=60)
        conn.execute("pragma busy_timeout=60000")
        return conn

    def _write_with_retry(
        self,
        operation: Callable[[sqlite3.Connection], _T],
        *,
        attempts: int = 3,
    ) -> _T:
        """Retry transient SQLite writer contention without hiding real errors."""

        maximum = max(1, int(attempts))
        for attempt in range(maximum):
            try:
                with closing(self._connect()) as conn:
                    return operation(conn)
            except sqlite3.OperationalError as exc:
                locked = "locked" in str(exc).lower() or "busy" in str(exc).lower()
                if not locked or attempt + 1 >= maximum:
                    raise
                time.sleep(0.1 * (2**attempt))
        raise RuntimeError("unreachable sqlite retry state")

    def _insert_unique(
        self,
        kind: str,
        records: tuple[Any, ...],
        key_fn: Any,
        observed_at_fn: Any,
    ) -> int:
        inserted = 0
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for record in records:
            row = _to_jsonable(record)
            if _is_simulated_row(kind, row):
                raise ValueError(f"Refusing to save simulated {kind} record into realtime store: {key_fn(row)}")
            observed_at = _as_aware(observed_at_fn(row))
            retention_days = _macro_retention_days() if kind == "macro_metrics" else self.retention_days
            cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
            if observed_at < cutoff:
                continue
            rows.append(
                (
                    kind,
                    key_fn(row),
                    observed_at.isoformat(),
                    now,
                    json.dumps(row, ensure_ascii=False, sort_keys=True),
                )
            )

        def write(conn: sqlite3.Connection) -> int:
            before = conn.total_changes
            conflict = "replace" if kind == "market_snapshots" else "ignore"
            conn.executemany(
                f"""
                insert or {conflict} into records
                  (kind, record_key, observed_at, inserted_at, payload)
                values (?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            return int(conn.total_changes - before)

        return self._write_with_retry(write)

    def _insert_typed(self, sql: str, rows: list[tuple[Any, ...]]) -> int:
        if not rows:
            return 0
        def write(conn: sqlite3.Connection) -> int:
            before = conn.total_changes
            conn.executemany(sql, rows)
            conn.commit()
            return int(conn.total_changes - before)

        return self._write_with_retry(write)

    def _read_kind(self, kind: str, limit: int | None = None) -> tuple[dict[str, Any], ...]:
        if limit is not None and limit <= 0:
            return ()
        limit_sql = " limit ?" if limit is not None else ""
        params: tuple[Any, ...] = (kind, int(limit)) if limit is not None else (kind,)
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"""
                select payload
                from records
                where kind = ?
                order by observed_at desc, inserted_at desc
                {limit_sql}
                """,
                params,
            ).fetchall()
        return tuple(json.loads(row[0]) for row in rows)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _source(data: dict[str, Any]):
    from app.schemas.domain import SourceMetadata

    observed_at = data.get("observed_at")
    return SourceMetadata(
        source_name=data["source_name"],
        retrieved_at=datetime.fromisoformat(data["retrieved_at"]),
        raw_url=data.get("raw_url"),
        source_id=data.get("source_id"),
        source_type=str(data.get("source_type", "unknown")),
        trust_level=int(data.get("trust_level", 0) or 0),
        observed_at=datetime.fromisoformat(observed_at) if observed_at else None,
        latency_sec=_optional_float(data.get("latency_sec")),
        is_realtime=bool(data.get("is_realtime", False)),
        is_delayed=bool(data.get("is_delayed", False)),
        is_synthetic=bool(data.get("is_synthetic", False)),
        is_backfilled=bool(data.get("is_backfilled", False)),
        license_policy=str(data.get("license_policy", "unknown")),
        quality_score=float(data.get("quality_score", 0.0) or 0.0),
    )


def _event_from_dict(data: dict[str, Any]) -> ClassifiedEvent:
    from app.schemas.domain import EventType, SentimentDirection

    return ClassifiedEvent(
        event_id=data["event_id"],
        event_type=EventType(data["event_type"]),
        title=data["title"],
        summary=data["summary"],
        companies=tuple(data["companies"]),
        tickers=tuple(data["tickers"]),
        sectors=tuple(data["sectors"]),
        sentiment=SentimentDirection(data["sentiment"]),
        event_date=datetime.fromisoformat(data["event_date"]),
        source=_source(data["source"]),
        key_facts=tuple(data.get("key_facts", ())),
        event_labels=tuple(data.get("event_labels", ())),
        classification_confidence=float(data.get("classification_confidence", 0.0)),
        classification_model=str(data.get("classification_model", "keyword_v1")),
    )


def _raw_from_dict(data: dict[str, Any]) -> RawSourceRecord:
    return RawSourceRecord(source=_source(data["source"]), content_type=data["content_type"], payload=data["payload"])


def _market_from_dict(data: dict[str, Any]) -> MarketSnapshot:
    return MarketSnapshot(
        ticker=data["ticker"],
        market=data["market"],
        company_name=data["company_name"],
        sector=data["sector"],
        last_price=float(data["last_price"]),
        average_daily_trading_value=float(data["average_daily_trading_value"]),
        volatility_20d=float(data["volatility_20d"]),
        source=_source(data["source"]),
    )


def _macro_from_dict(data: dict[str, Any]) -> MacroMetricRecord:
    return MacroMetricRecord(
        name=data["name"],
        value=float(data["value"]),
        observed_at=datetime.fromisoformat(data["observed_at"]),
        source=_source(data["source"]),
    )


def _realtime_quote_from_dict(data: dict[str, Any]) -> RealtimeQuote:
    source = data.get("source")
    return RealtimeQuote(
        ticker=data["ticker"],
        market=data["market"],
        observed_at=datetime.fromisoformat(data["observed_at"]),
        last_price=float(data["last_price"]),
        bid_price=_optional_float(data.get("bid_price")),
        ask_price=_optional_float(data.get("ask_price")),
        bid_size=_optional_float(data.get("bid_size")),
        ask_size=_optional_float(data.get("ask_size")),
        volume=_optional_float(data.get("volume")),
        change=_optional_float(data.get("change")),
        change_rate=_optional_float(data.get("change_rate")),
        source=_source(source) if isinstance(source, dict) else None,
    )


def _realtime_execution_from_dict(data: dict[str, Any]) -> RealtimeExecution:
    source = data.get("source")
    return RealtimeExecution(
        ticker=data["ticker"],
        market=data["market"],
        executed_at=datetime.fromisoformat(data["executed_at"]),
        price=float(data["price"]),
        quantity=int(data["quantity"]),
        side=data.get("side"),
        trade_id=data.get("trade_id"),
        source=_source(source) if isinstance(source, dict) else None,
    )


def _triple_from_dict(data: dict[str, Any]) -> Triple:
    return Triple(
        subject=data["subject"],
        predicate=data["predicate"],
        object=data["object"],
        evidence_id=data.get("evidence_id"),
    )


def _reasoning_from_dict(data: dict[str, Any]) -> ReasoningPath:
    return ReasoningPath(
        path_id=data["path_id"],
        ticker=data["ticker"],
        conclusion=data["conclusion"],
        confidence=float(data["confidence"]),
        supporting_triples=tuple(data["supporting_triples"]),
        contradicting_triples=tuple(data["contradicting_triples"]),
        risk_triples=tuple(data["risk_triples"]),
        explanation=data["explanation"],
    )


def _as_aware(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _now_observed_at(row: dict[str, Any]) -> datetime:
    return datetime.now(timezone.utc)


def _event_observed_at(row: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(row["event_date"])


def _raw_observed_at(row: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(row["source"]["retrieved_at"])


def _market_observed_at(row: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(row["source"]["retrieved_at"])


def _macro_observed_at(row: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(row["observed_at"])


def _realtime_quote_observed_at(row: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(row["observed_at"])


def _realtime_execution_observed_at(row: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(row["executed_at"])


def _event_key(row: dict[str, Any]) -> str:
    return row["event_id"]


def _raw_key(row: dict[str, Any]) -> str:
    source = row["source"]
    source_id = source.get("source_id") or source.get("raw_url") or row["payload"][:80]
    return f"{source_id}:{source.get('retrieved_at')}"


def _market_key(row: dict[str, Any]) -> str:
    source = row["source"]
    if source.get("source_name") in {"listed_universe_reference", "listed_universe_catalog"}:
        return f"{row['ticker']}:{source.get('source_id') or source.get('source_name')}"
    return f"{row['ticker']}:{source.get('source_id')}:{source.get('retrieved_at')}"


def _macro_key(row: dict[str, Any]) -> str:
    return f"{row['name']}:{row['observed_at']}"


def _realtime_quote_key(row: dict[str, Any]) -> str:
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    source_id = source.get("source_id") or source.get("raw_url") or source.get("source_name") or "quote"
    return f"{row['ticker']}:{row['market']}:{source_id}:{row['observed_at']}"


def _realtime_execution_key(row: dict[str, Any]) -> str:
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    source_id = source.get("source_id") or source.get("raw_url") or source.get("source_name") or "execution"
    trade_id = row.get("trade_id")
    if trade_id:
        return f"{row['ticker']}:{row['market']}:{source_id}:{trade_id}"
    return (
        f"{row['ticker']}:{row['market']}:{source_id}:{row['executed_at']}:"
        f"{row['price']}:{row['quantity']}:{row.get('side')}"
    )


def _triple_key(row: dict[str, Any]) -> str:
    return f"{row['subject']}|{row['predicate']}|{row['object']}|{row.get('evidence_id')}"


def _reasoning_key(row: dict[str, Any]) -> str:
    return row["path_id"]


def _macro_retention_days() -> int:
    try:
        return max(365, int(os.getenv("MACRO_METRIC_RETENTION_DAYS", "730")))
    except (TypeError, ValueError):
        return 730


def _is_simulated_row(kind: str, row: dict[str, Any]) -> bool:
    if kind in {"market_snapshots", "realtime_quotes", "realtime_executions"} and str(
        row.get("market", "")
    ).upper() == "SIM":
        return True
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    source_name = str(source.get("source_name", "")).lower()
    raw_url = str(source.get("raw_url", "")).lower()
    source_id = str(source.get("source_id", "")).lower()
    if source_name.startswith(("sim", "synthetic", "accelerated_demo")):
        return True
    if raw_url.startswith(("local://sim", "local://synthetic", "local://accelerated-demo")):
        return True
    if source_id.startswith(("sim:", "synthetic:", "demo-chart:")):
        return True
    if kind == "graph_triples":
        evidence = str(row.get("evidence_id", "")).lower()
        return evidence.startswith(("sim:", "synthetic:", "demo-chart:", "reasoner:sim"))
    return False


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)

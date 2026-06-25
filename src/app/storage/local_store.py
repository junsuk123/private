from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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
        return {
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
        return {
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
        cutoff_text = cutoff.isoformat()
        with closing(self._connect()) as conn:
            cursor = conn.execute("delete from records where observed_at < ?", (cutoff_text,))
            conn.commit()
            return int(cursor.rowcount)

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
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30)

    def _insert_unique(
        self,
        kind: str,
        records: tuple[Any, ...],
        key_fn: Any,
        observed_at_fn: Any,
    ) -> int:
        inserted = 0
        now = datetime.now(timezone.utc).isoformat()
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        rows = []
        for record in records:
            row = _to_jsonable(record)
            if _is_simulated_row(kind, row):
                raise ValueError(f"Refusing to save simulated {kind} record into realtime store: {key_fn(row)}")
            observed_at = _as_aware(observed_at_fn(row))
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

        with closing(self._connect()) as conn:
            for row in rows:
                cursor = conn.execute(
                    """
                    insert or ignore into records
                      (kind, record_key, observed_at, inserted_at, payload)
                    values (?, ?, ?, ?, ?)
                    """,
                    row,
                )
                inserted += int(cursor.rowcount)
            conn.commit()
        return inserted

    def _read_kind(self, kind: str) -> tuple[dict[str, Any], ...]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select payload
                from records
                where kind = ?
                order by observed_at desc, inserted_at desc
                """,
                (kind,),
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

    return SourceMetadata(
        source_name=data["source_name"],
        retrieved_at=datetime.fromisoformat(data["retrieved_at"]),
        raw_url=data.get("raw_url"),
        source_id=data.get("source_id"),
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

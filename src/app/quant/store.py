from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.quant.contracts import QuantEvidence


class QuantEvidenceStore:
    """Isolated additive SQLite store using the repository's WAL convention."""

    def __init__(self, path: str | Path = "data/store/quant_reference.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute("pragma journal_mode=wal")
            connection.executescript(
                """
                create table if not exists quant_evidence (
                    evidence_id integer primary key autoincrement,
                    symbol text not null, market text not null, timestamp text not null,
                    metric text not null, validation_status text not null,
                    payload_json text not null
                );
                create index if not exists idx_quant_evidence_symbol_time on quant_evidence(symbol, timestamp);
                create index if not exists idx_quant_evidence_metric_time on quant_evidence(metric, timestamp);
                create index if not exists idx_quant_evidence_validation on quant_evidence(validation_status);
                create table if not exists quant_provider_health (
                    provider text not null, checked_at text not null, payload_json text not null,
                    primary key(provider, checked_at)
                );
                create table if not exists quant_validation_result (
                    metric text not null, checked_at text not null, status text not null,
                    payload_json text not null, primary key(metric, checked_at)
                );
                """
            )
            connection.commit()

    def append(self, evidence: Iterable[QuantEvidence]) -> int:
        rows = tuple(evidence)
        with closing(self._connect()) as connection:
            connection.executemany(
                "insert into quant_evidence(symbol, market, timestamp, metric, validation_status, payload_json) values (?, ?, ?, ?, ?, ?)",
                tuple((item.symbol, item.market, item.timestamp.isoformat(), item.metric, item.validation_status.value, json.dumps(item.as_dict(), ensure_ascii=False, sort_keys=True)) for item in rows),
            )
            connection.commit()
        return len(rows)

    def record_health(self, provider: str, payload: dict[str, Any], *, checked_at: datetime | None = None) -> None:
        moment = checked_at or datetime.now(timezone.utc)
        with closing(self._connect()) as connection:
            connection.execute(
                "insert or replace into quant_provider_health values (?, ?, ?)",
                (provider, moment.isoformat(), json.dumps(payload, ensure_ascii=False, sort_keys=True)),
            )
            connection.commit()

    def record_validation(self, metric: str, status: str, payload: dict[str, Any], *, checked_at: datetime | None = None) -> None:
        moment = checked_at or datetime.now(timezone.utc)
        with closing(self._connect()) as connection:
            connection.execute(
                "insert or replace into quant_validation_result values (?, ?, ?, ?)",
                (metric, moment.isoformat(), status, json.dumps(payload, ensure_ascii=False, sort_keys=True)),
            )
            connection.commit()

    def latest(self, symbol: str, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "select payload_json from quant_evidence where symbol = ? order by timestamp desc, evidence_id desc limit ?",
                (symbol, max(1, int(limit))),
            ).fetchall()
        return tuple(json.loads(row[0]) for row in rows)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30.0)

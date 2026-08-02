"""Weekend-only enrichment: deepen macro history and de-saturate event sentiment.

Both jobs exist because the weekend brief's inputs were measured and found thin:

* FRED macro held 12 rows across 6 series, because the collector fetched a batch and
  returned only its newest row. Two observations is the minimum that can express a
  change at all, and several series had one.
* Stored event sentiment came from a keyword classifier and was saturated: 155,959
  of 158,338 scores were exactly +1.0 (98.5%). A feed that says "positive" to
  almost everything cannot discriminate, so the brief refuses to use it.

Both run only while the market is shut. That is not merely convenient — the LLM pass
is the heaviest thing this process does, and running it during a session would
compete with the trading loop for the same CPU.

Re-classification never overwrites the original record. Corrections are written to a
separate table with their model and confidence, so the keyword verdict and the LLM
verdict stay independently inspectable and a bad model run can be discarded.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_RESEARCH_DB = "data/store/research.sqlite3"
DEFAULT_ENRICHMENT_DB = "data/store/weekend_enrichment.sqlite3"

# The classifier whose output is saturated and therefore worth revisiting.
SATURATED_MODEL_PREFIX = "keyword"


@dataclass
class MacroBackfillResult:
    series_attempted: int = 0
    records_written: int = 0
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "series_attempted": self.series_attempted,
            "records_written": self.records_written,
            "failed_series": len(self.failures),
            "failures": self.failures[:10],
        }


@dataclass
class ReclassifyResult:
    considered: int = 0
    reclassified: int = 0
    changed_sentiment: int = 0
    failures: int = 0
    model: str | None = None
    # Distinguishes "the LLM is not configured" from "there was nothing to do".
    # Without it both report considered=0 / failures=0, and a silently disabled
    # classifier looks exactly like a clean run with no work.
    classifier_available: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "considered": self.considered,
            "reclassified": self.reclassified,
            "changed_sentiment": self.changed_sentiment,
            "failures": self.failures,
            "model": self.model,
            "classifier_available": self.classifier_available,
        }


class EventReclassificationStore:
    """LLM verdicts, kept separate from the keyword verdicts they revisit."""

    def __init__(self, path: str | Path = DEFAULT_ENRICHMENT_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute(
                """
                create table if not exists event_reclassification (
                    event_id text primary key,
                    observed_at text,
                    original_sentiment text,
                    llm_sentiment text not null,
                    llm_confidence real,
                    model text,
                    reclassified_at text not null
                )
                """
            )
            conn.commit()

    def save(
        self,
        *,
        event_id: str,
        observed_at: str | None,
        original_sentiment: str | None,
        llm_sentiment: str,
        llm_confidence: float,
        model: str | None,
    ) -> None:
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute(
                """
                insert into event_reclassification (
                    event_id, observed_at, original_sentiment, llm_sentiment,
                    llm_confidence, model, reclassified_at
                ) values (?, ?, ?, ?, ?, ?, ?)
                on conflict(event_id) do update set
                    llm_sentiment = excluded.llm_sentiment,
                    llm_confidence = excluded.llm_confidence,
                    model = excluded.model,
                    reclassified_at = excluded.reclassified_at
                """,
                (
                    event_id,
                    observed_at,
                    original_sentiment,
                    llm_sentiment,
                    float(llm_confidence),
                    model,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def sentiment_overrides(self, since_iso: str, until_iso: str) -> dict[str, str]:
        with closing(sqlite3.connect(self.path)) as conn:
            rows = conn.execute(
                "select event_id, llm_sentiment from event_reclassification"
                " where observed_at >= ? and observed_at < ?",
                (since_iso, until_iso),
            ).fetchall()
        return {str(row[0]): str(row[1]) for row in rows}

    def distribution(self) -> dict[str, int]:
        """Sentiment spread of the LLM pass — the point is that it is NOT flat."""
        with closing(sqlite3.connect(self.path)) as conn:
            rows = conn.execute(
                "select llm_sentiment, count(*) from event_reclassification group by 1"
            ).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}


def backfill_macro_history(
    *,
    series: Sequence[tuple[str, str]],
    store: Any,
    collector: Any | None = None,
    limit: int = 120,
) -> MacroBackfillResult:
    """Persist full recent history for each ``(series_id, name)`` pair."""
    result = MacroBackfillResult()
    if collector is None:
        from app.data.public_collectors import FredMacroCollector

        collector = FredMacroCollector()
    for series_id, name in series:
        result.series_attempted += 1
        try:
            records = collector.collect_history(series_id, name, limit=limit)
        except Exception as exc:  # noqa: BLE001 - one bad series must not stop the sweep
            result.failures.append(f"{series_id}:{type(exc).__name__}")
            continue
        if not records:
            continue
        try:
            # Batched, so the store's own dedup decides what is genuinely new.
            result.records_written += int(store.save_macro_metrics(records) or 0)
        except Exception as exc:  # noqa: BLE001
            result.failures.append(f"{series_id}:save:{type(exc).__name__}")
    return result


def load_saturated_events(
    *,
    since_iso: str,
    until_iso: str,
    research_db: str | Path = DEFAULT_RESEARCH_DB,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Events still carrying a keyword-classifier verdict, newest first."""
    path = Path(research_db)
    if not path.exists():
        return []
    try:
        with closing(sqlite3.connect(path)) as conn:
            rows = conn.execute(
                "select payload, observed_at from records where kind='events'"
                " and observed_at >= ? and observed_at < ? order by observed_at desc",
                (since_iso, until_iso),
            ).fetchall()
    except sqlite3.Error:
        return []
    out: list[dict[str, Any]] = []
    for payload, observed_at in rows:
        try:
            data = json.loads(payload)
        except (TypeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        model = str(data.get("classification_model") or "")
        if not model.lower().startswith(SATURATED_MODEL_PREFIX):
            continue
        data["_observed_at"] = str(observed_at)
        out.append(data)
        if len(out) >= max(1, int(limit)):
            break
    return out


def reclassify_events(
    events: Iterable[dict[str, Any]],
    *,
    classifier: Any | None = None,
    store: EventReclassificationStore | None = None,
    known_tickers: dict[str, str] | None = None,
) -> ReclassifyResult:
    """Re-score events with the LLM, recording where it disagrees.

    A failure on one headline is skipped rather than aborting the batch: a partial
    re-classification is still an improvement over a saturated feed, and the counts
    make the partiality visible.
    """
    result = ReclassifyResult()
    if classifier is None:
        from app.data.llm_classifier import build_event_llm_classifier_from_env

        classifier = build_event_llm_classifier_from_env()
    if classifier is None:
        # Report it. A disabled classifier is an operational fact worth surfacing,
        # not an empty success.
        result.classifier_available = False
        return result
    target = store or EventReclassificationStore()

    for event in events:
        result.considered += 1
        title = str(event.get("title") or "").strip()
        body = str(event.get("summary") or "").strip()
        if not title and not body:
            continue
        try:
            verdict = classifier.classify(title, body, known_tickers or {})
        except Exception:  # noqa: BLE001 - one headline must not end the batch
            result.failures += 1
            continue
        original = str(event.get("sentiment") or "").upper()
        llm_sentiment = str(getattr(verdict, "sentiment", "") or "NEUTRAL").upper()
        target.save(
            event_id=str(event.get("event_id") or ""),
            observed_at=str(event.get("_observed_at") or event.get("event_date") or ""),
            original_sentiment=original or None,
            llm_sentiment=llm_sentiment,
            llm_confidence=float(getattr(verdict, "confidence", 0.0) or 0.0),
            model=str(getattr(verdict, "model", "") or "") or None,
        )
        result.reclassified += 1
        result.model = result.model or str(getattr(verdict, "model", "") or "") or None
        if original and original != llm_sentiment:
            result.changed_sentiment += 1
    return result

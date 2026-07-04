"""Lightweight, memory-safe store of per-ticker news sentiment over time.

This is the bridge that lets the (text) news LLM feed the real-time *numeric*
short-horizon learner: classified news sentiment is recorded here with the news
observation time, and the live feature builder reads a recency-decayed value
`as of` each decision time (no lookahead). The existing 60s training loop then
learns how predictive that sentiment is from realized PnL — no LLM weights are
trained, and the footprint is a tiny SQLite table plus a few floats per query.
"""

from __future__ import annotations

import math
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path


DEFAULT_NEWS_SENTIMENT_STORE_PATH = "data/store/news_sentiment.sqlite3"


def _default_halflife_sec() -> float:
    try:
        return max(1.0, float(os.getenv("NEWS_SENTIMENT_HALFLIFE_SEC", "1800")))
    except (TypeError, ValueError):
        return 1800.0


def _default_ttl_sec() -> float:
    try:
        return max(60.0, float(os.getenv("NEWS_SENTIMENT_TTL_SEC", "21600")))
    except (TypeError, ValueError):
        return 21600.0


class NewsSentimentStore:
    def __init__(self, path: str | Path = DEFAULT_NEWS_SENTIMENT_STORE_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute(
                "create table if not exists news_sentiment (ticker text, observed_at text, score real)"
            )
            conn.execute(
                "create index if not exists idx_news_sentiment_ticker_time on news_sentiment(ticker, observed_at)"
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.execute("pragma busy_timeout = 5000")
        return conn

    def record(self, ticker: str, score: float, observed_at: datetime | None = None) -> None:
        """Record a sentiment observation. score is clamped to [-1, 1]. Best-effort."""
        ticker = str(ticker or "").strip()
        if not ticker:
            return
        try:
            clamped = max(-1.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            return
        if not math.isfinite(clamped):
            return
        when = observed_at or datetime.now(timezone.utc)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        try:
            with closing(self._connect()) as conn:
                conn.execute(
                    "insert into news_sentiment(ticker, observed_at, score) values (?, ?, ?)",
                    (ticker, when.astimezone(timezone.utc).isoformat(), clamped),
                )
                conn.commit()
        except sqlite3.Error:
            return

    def score_as_of(
        self,
        ticker: str,
        at: datetime,
        *,
        halflife_sec: float | None = None,
        ttl_sec: float | None = None,
    ) -> float:
        """Recency-decayed sentiment in [-1, 1] using only rows observed <= `at`.

        No lookahead: rows strictly after `at` are excluded. Rows older than
        `ttl_sec` are ignored. Recent observations dominate via exp(-dt/halflife).
        Returns 0.0 (neutral) when there is no fresh news.
        """
        ticker = str(ticker or "").strip()
        if not ticker:
            return 0.0
        if at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)
        at = at.astimezone(timezone.utc)
        halflife = halflife_sec if halflife_sec is not None else _default_halflife_sec()
        ttl = ttl_sec if ttl_sec is not None else _default_ttl_sec()
        floor = at - timedelta(seconds=ttl)
        try:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    "select observed_at, score from news_sentiment "
                    "where ticker = ? and observed_at <= ? and observed_at >= ? "
                    "order by observed_at desc limit 32",
                    (ticker, at.isoformat(), floor.isoformat()),
                ).fetchall()
        except sqlite3.Error:
            return 0.0
        numerator = 0.0
        denominator = 0.0
        for observed_at_str, score in rows:
            try:
                observed_time = datetime.fromisoformat(str(observed_at_str))
            except ValueError:
                continue
            if observed_time.tzinfo is None:
                observed_time = observed_time.replace(tzinfo=timezone.utc)
            dt_seconds = max(0.0, (at - observed_time).total_seconds())
            weight = math.exp(-dt_seconds / max(1.0, halflife))
            numerator += weight * float(score)
            denominator += weight
        if denominator <= 0.0:
            return 0.0
        return max(-1.0, min(1.0, numerator / denominator))

    def prune(self, older_than_days: float = 7.0) -> None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
        try:
            with closing(self._connect()) as conn:
                conn.execute("delete from news_sentiment where observed_at < ?", (cutoff,))
                conn.commit()
        except sqlite3.Error:
            return

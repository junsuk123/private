"""Shared refresh routine for the daily investor-flow store.

Deliberately the ONE implementation used by both the manual backfill script and
the server's background refresher. An earlier version of this codebase kept the
same numbers in two places (exit geometry in the session and in the training
labels); they drifted, and a model ended up scoring trades executed under
different rules. Collection logic gets one home for the same reason.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.data.investor_flow_store import InvestorFlowStore

DEFAULT_BAR_DATABASE = "data/store/realtime_market_data.sqlite3"


@dataclass
class InvestorFlowRefreshResult:
    symbols_attempted: int = 0
    rows_written: int = 0
    failures: list[str] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    skipped_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbols_attempted": self.symbols_attempted,
            "rows_written": self.rows_written,
            "failed_symbols": len(self.failures),
            # Truncated, but the COUNT above is always exact: a partial sweep must
            # never be able to look like a complete one.
            "failures": self.failures[:20],
            "coverage": self.coverage,
            "skipped_reason": self.skipped_reason,
        }


def krx_symbols_with_bars(
    database: str | Path = DEFAULT_BAR_DATABASE,
    minimum_bars: int = 100,
) -> tuple[str, ...]:
    """KRX symbols with enough minute bars for the labeller to use them.

    Collecting flow for a symbol the labeller discards is wasted API budget, so
    the bar store decides the work list rather than a hardcoded universe.
    """
    path = Path(database)
    if not path.exists():
        return ()
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            """
            SELECT symbol, COUNT(*) AS bars
            FROM realtime_minute_bars
            GROUP BY symbol
            HAVING bars >= ?
            ORDER BY bars DESC
            """,
            (int(minimum_bars),),
        ).fetchall()
    except sqlite3.Error:
        return ()
    finally:
        connection.close()
    return tuple(
        str(symbol)
        for symbol, _bars in rows
        if str(symbol).isdigit() and len(str(symbol)) == 6
    )


def refresh_investor_flow(
    *,
    client: Any | None = None,
    store: InvestorFlowStore | None = None,
    database: str | Path = DEFAULT_BAR_DATABASE,
    minimum_bars: int = 100,
    limit: int = 0,
    delay_seconds: float = 0.3,
) -> InvestorFlowRefreshResult:
    """Fetch and upsert each eligible KRX symbol's daily investor flow.

    One KIS call returns ~30 business days, so this both backfills a fresh store
    and keeps an existing one current — including correcting the current day,
    whose figures are still moving while the session runs.

    Read-only against the broker (``inquire-investor``); it never touches an order
    endpoint. Failures are collected rather than raised: one unreadable symbol must
    not abort the sweep, and the count is reported so a partial run is visible.
    """
    result = InvestorFlowRefreshResult()
    symbols = krx_symbols_with_bars(database, minimum_bars)
    if limit > 0:
        symbols = symbols[:limit]
    if not symbols:
        result.skipped_reason = "NO_KRX_SYMBOLS_WITH_ENOUGH_BARS"
        return result

    if client is None:
        from app.execution.kis_real import KisDevelopersApiClient

        client = KisDevelopersApiClient(paper=False, enabled=True)
    flow_store = store if store is not None else InvestorFlowStore()

    result.symbols_attempted = len(symbols)
    for index, symbol in enumerate(symbols):
        try:
            rows = client.get_domestic_investor_flow(symbol)
        except Exception as exc:  # noqa: BLE001 - one symbol must not stop the sweep
            result.failures.append(f"{symbol}:{type(exc).__name__}")
            continue
        result.rows_written += flow_store.upsert_many(rows)
        # No sleep after the final symbol; it only delays the caller.
        if delay_seconds > 0 and index < len(symbols) - 1:
            time.sleep(delay_seconds)

    result.coverage = flow_store.coverage()
    return result

"""Does this symbol print often enough to feed a sub-second trigger?

Why discovery needs this, separately from feasibility
----------------------------------------------------
``strategy_feasibility`` asks whether a strategy's barrier is *reachable* on a
symbol's chart. This module asks something strictly upstream of that: whether the
symbol prints at all often enough for a trigger to be evaluated. Every mechanical
entry rule in ``app.technical.strategy_algorithms`` gates on ``_tick_ready``, whose
``tick_data_ready`` only opens once the symbol printed at least twice within ten
seconds in two distinct seconds. A name that clears turnover ranking and clears
feasibility can still return ``TICK_WINDOW_NOT_READY`` on every single evaluation.

This is not hypothetical, and it is not new. ``web._recent_affordable_us_watchlist``
already fixed exactly this on the US side, where nine of fifteen candidate slots were
held by names producing nothing — AVS at 0.35 prints/min kept a realtime subscription
slot while every one of its evaluations returned ``TICK_WINDOW_NOT_READY``. The KRX
path never got the same treatment: it ranks by KIS turnover, then reorders by
feasibility, and neither input is measured print density.

Measured on the live store during the 2026-08-18 KRX session, 97 minutes in, over the
38 symbols then holding a collector subscription:

    DENSE (>=10 prints/market-minute)   20 symbols   005930 at 1,791/min down to 089970 at 12.6
    sparse                              18 symbols   nine of them with ZERO ticks all session,
                                                     plus 269620 (2 ticks), 277410 (1), 226400 (4)

Just under half the subscription budget was spent on names that could not have fired a
trigger no matter what the tape did.

Two properties that keep the measure honest
-------------------------------------------
**A fixed denominator, not the symbol's own span.** Dividing a symbol's print count by
the time between its own first and last print flatters exactly the names this is meant
to catch: 260660 printed 12 times inside one minute and nothing for the rest of the
session, which scores 12/min on its own span and 0.12/min against the session. The US
docstring makes the same point about observed range — a sparse sample inflates a
per-sample statistic rather than penalising it.

**Market minutes, not wall-clock minutes.** The denominator counts distinct minutes in
which *any* symbol in the peer group printed. Overnight and weekend gaps therefore
contribute nothing, so a lookback may span several sessions without diluting every
symbol toward zero — which is what lets the measure exist at 09:00, before the current
session has produced anything. Normalising against the peer group rather than against
all symbols keeps a KR name from being measured on a denominator inflated by US
session minutes.

The threshold
-------------
10 prints per market-minute, inherited from the US analysis: on a Poisson tape that
rate carries two or more prints in a given ten seconds about 59% of the time, against
about 3% at the 0.35/min the discarded names were running. On the KRX measurement above
it lands in a natural gap — the lowest DENSE name scored 11.2 and the highest sparse
name 7.4.

Why this reorders rather than filters
-------------------------------------
A hard density filter deadlocks: an unsubscribed symbol has no prints, so the set that
already has data would be the only set that can ever get data. Slots are therefore
split the same way the US path splits them — most go to measured density, and a small
reserve rotates through unmeasured names so each gets a real subscription window to
prove itself on. Both buckets backfill each other, so the returned list is never
shorter than what came in.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

__all__ = [
    "DEFAULT_MINIMUM_PRINTS_PER_MINUTE",
    "KR_SYMBOL_GLOB",
    "blend_with_exploration",
    "measure_print_density",
    "partition_by_density",
    "rank_by_print_density",
]

#: Six consecutive digits is a KRX ticker. Interpolated into a SQLite GLOB, so it is a
#: module constant rather than a caller-supplied string.
KR_SYMBOL_GLOB = "[0-9][0-9][0-9][0-9][0-9][0-9]"

DEFAULT_MINIMUM_PRINTS_PER_MINUTE = 10.0


def measure_print_density(
    symbols: Sequence[str],
    *,
    database: str | Path,
    lookback_hours: float = 30.0,
    now: datetime | None = None,
    symbol_glob: str = KR_SYMBOL_GLOB,
) -> dict[str, float]:
    """Prints per market-minute for each symbol, over a fixed lookback.

    Returns an empty mapping when the store cannot be read or the window contains no
    market minutes at all. That is deliberately indistinguishable from "no opinion":
    callers treat an empty measurement as a reason to leave their ordering alone, never
    as evidence that every symbol is sparse. A symbol that is present in ``symbols``
    but printed nothing in the window is reported as ``0.0`` — that IS an opinion, and
    the strongest one available.
    """
    wanted = tuple(dict.fromkeys(str(symbol or "").strip() for symbol in symbols if symbol))
    if not wanted:
        return {}
    moment = now or datetime.now(timezone.utc)
    since = (moment - timedelta(hours=max(0.5, float(lookback_hours)))).isoformat()
    path = Path(database)
    if not path.exists():
        return {}
    try:
        with closing(sqlite3.connect(str(path), timeout=5.0)) as connection:
            # The denominator is computed over the whole peer group, not over the
            # requested symbols: a caller asking about three names must not get a
            # three-name clock that makes all of them look dense.
            market_minutes = connection.execute(
                f"""
                select count(distinct substr(received_at, 1, 16))
                from realtime_ticks
                where received_at >= ? and symbol glob '{symbol_glob}'
                """,
                (since,),
            ).fetchone()
            minutes = float((market_minutes or (0,))[0] or 0.0)
            if minutes <= 0.0:
                return {}
            counted = dict(
                connection.execute(
                    f"""
                    select symbol, count(*)
                    from realtime_ticks
                    where received_at >= ? and symbol glob '{symbol_glob}'
                    group by symbol
                    """,
                    (since,),
                ).fetchall()
            )
    except sqlite3.Error:
        return {}
    return {
        symbol: float(counted.get(symbol, 0) or 0) / minutes for symbol in wanted
    }


def partition_by_density(
    symbols: Sequence[str],
    density: Mapping[str, float],
    *,
    minimum_prints_per_minute: float = DEFAULT_MINIMUM_PRINTS_PER_MINUTE,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split into (dense, sparse), preserving the caller's ordering within each.

    The incoming order already carries turnover and feasibility. Density decides which
    bucket a symbol lands in; it does not re-sort inside a bucket, so the ranking work
    upstream survives.

    A symbol with no entry in ``density`` is UNMEASURED and goes to ``sparse`` — not
    because it is known to be quiet, but because it has not shown it can feed a trigger.
    The exploration reserve in :func:`blend_with_exploration` is what keeps that from
    being a life sentence.
    """
    floor = max(0.0, float(minimum_prints_per_minute))
    dense: list[str] = []
    sparse: list[str] = []
    for symbol in dict.fromkeys(str(item or "").strip() for item in symbols if item):
        measured = density.get(symbol)
        bucket = dense if measured is not None and float(measured) >= floor else sparse
        bucket.append(symbol)
    return tuple(dense), tuple(sparse)


def blend_with_exploration(
    dense: Sequence[str],
    sparse: Sequence[str],
    *,
    limit: int,
    reserve: int = 2,
    cursor: int = 0,
) -> tuple[tuple[str, ...], int]:
    """Most slots to measured density, a rotating few to unproven names.

    Mirrors ``web._blend_watchlist_exploration``, but pure: the rotation cursor is
    passed in and the next one is returned, so no module-level cache is involved and
    the behaviour is reproducible from its arguments alone.

    Reserving nothing locks the universe to whatever already has data; reserving too
    much spends the live feed on names that have never shown they can feed a trigger.
    The reserve is therefore capped at a third of the budget, and each call advances the
    cursor so a different unproven candidate takes the slot next time.
    """
    budget = max(0, int(limit))
    if budget <= 0:
        return (), int(cursor)
    reserved = min(max(0, int(reserve)), budget // 3)
    exploit = list(dense[: max(0, budget - reserved)])
    explored: list[str] = []
    next_cursor = int(cursor)
    if reserved and sparse:
        start = int(cursor) % len(sparse)
        explored = [
            sparse[(start + offset) % len(sparse)]
            for offset in range(min(reserved, len(sparse)))
        ]
        next_cursor = (start + reserved) % len(sparse)
    selected = list(dict.fromkeys((*exploit, *explored)))
    # Backfill so the caller never ends up with fewer symbols than it had.
    for symbol in (*dense, *sparse):
        if len(selected) >= budget:
            break
        if symbol not in selected:
            selected.append(symbol)
    return tuple(selected[:budget]), next_cursor


def rank_by_print_density(
    symbols: Sequence[str],
    *,
    database: str | Path,
    limit: int,
    cursor: int = 0,
    minimum_prints_per_minute: float | None = None,
    lookback_hours: float | None = None,
    reserve: int | None = None,
    now: datetime | None = None,
    symbol_glob: str = KR_SYMBOL_GLOB,
) -> tuple[tuple[str, ...], int, dict[str, object]]:
    """Reorder a discovery ranking so subscription slots go to symbols that print.

    Returns ``(ordered, next_cursor, stats)``. ``stats`` is for telemetry and carries
    the bucket sizes and the threshold actually applied.

    FAILS OPEN. Any unreadable store, any empty measurement, or an ordering that would
    contain nothing returns the input unchanged with the cursor untouched. Discovery
    losing its ranking is a worse outcome than discovery keeping a few quiet names, so
    every uncertain path here declines to act rather than guessing.
    """
    incoming = tuple(dict.fromkeys(str(item or "").strip() for item in symbols if item))
    stats: dict[str, object] = {
        "applied": False,
        "incoming": len(incoming),
        "dense": 0,
        "sparse": 0,
    }
    if not incoming:
        return incoming, int(cursor), stats
    if not _flag("REALTIME_KRX_DENSITY_FILTER_ENABLED", True):
        stats["reason"] = "DISABLED"
        return incoming, int(cursor), stats

    floor = (
        float(minimum_prints_per_minute)
        if minimum_prints_per_minute is not None
        else _number("REALTIME_KRX_DENSITY_MIN_PRINTS_PER_MINUTE", DEFAULT_MINIMUM_PRINTS_PER_MINUTE)
    )
    window = (
        float(lookback_hours)
        if lookback_hours is not None
        else _number("REALTIME_KRX_DENSITY_LOOKBACK_HOURS", 30.0)
    )
    reserved = (
        int(reserve)
        if reserve is not None
        else int(_number("REALTIME_KRX_DENSITY_EXPLORATION_SLOTS", 2.0))
    )

    try:
        density = measure_print_density(
            incoming,
            database=database,
            lookback_hours=window,
            now=now,
            symbol_glob=symbol_glob,
        )
    except Exception:  # noqa: BLE001 - discovery must never break on telemetry.
        stats["reason"] = "MEASUREMENT_FAILED"
        return incoming, int(cursor), stats
    if not density:
        # No market minutes in the window: a cold store, or a call before any session
        # has printed. Nothing has been observed, so nothing has been ruled out.
        stats["reason"] = "NO_MEASUREMENT"
        return incoming, int(cursor), stats

    dense, sparse = partition_by_density(
        incoming, density, minimum_prints_per_minute=floor
    )
    stats.update({"dense": len(dense), "sparse": len(sparse), "threshold": floor})
    if not dense:
        # Every candidate is quiet. That is a statement about the tape or about a
        # subscription set that has not warmed up, not a reason to reshuffle blindly.
        stats["reason"] = "NO_DENSE_CANDIDATE"
        return incoming, int(cursor), stats

    ordered, next_cursor = blend_with_exploration(
        dense, sparse, limit=max(int(limit), 0) or len(incoming), reserve=reserved, cursor=cursor
    )
    if not ordered:
        stats["reason"] = "EMPTY_SELECTION"
        return incoming, int(cursor), stats
    stats["applied"] = True
    stats["selected"] = len(ordered)
    stats["dropped"] = [symbol for symbol in incoming if symbol not in set(ordered)]
    return ordered, next_cursor, stats


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _number(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default

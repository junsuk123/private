from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from app.trading.print_density import (
    blend_with_exploration,
    measure_print_density,
    partition_by_density,
    rank_by_print_density,
)


AT = datetime(2026, 8, 18, 1, 30, tzinfo=timezone.utc)


def _store(tmp_path, prints: dict[str, int], *, spread_minutes: int = 100):
    """A realtime store where each symbol printed ``prints[symbol]`` times.

    Prints are spread across distinct minutes so the market-minute denominator is
    the session length rather than one symbol's burst.
    """
    path = tmp_path / "realtime_market_data.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "create table realtime_ticks (symbol text, received_at text, price real)"
        )
        rows = []
        for symbol, count in prints.items():
            for index in range(count):
                moment = AT - timedelta(
                    minutes=index % max(1, spread_minutes),
                    seconds=index % 60,
                )
                rows.append((symbol, moment.isoformat(), 100.0))
        connection.executemany(
            "insert into realtime_ticks values (?, ?, ?)", rows
        )
    return path


def test_density_uses_a_fixed_denominator_not_the_symbols_own_span(tmp_path):
    """A burst inside one minute must not score like a symbol printing all session.

    This is the failure the KRX subscription set actually exhibited: 260660 printed
    12 times inside a single minute and nothing else all session. Measured against
    its own span that is 12 prints/min and looks healthy; measured against the
    session it is 0.12 and cannot feed a ten-second window.
    """
    database = _store(
        tmp_path,
        {"005930": 2000, "260660": 12},
        spread_minutes=100,
    )
    density = measure_print_density(
        ["005930", "260660"], database=database, lookback_hours=30.0, now=AT
    )
    assert density["005930"] > 10.0
    assert density["260660"] < 1.0


def test_unmeasured_and_silent_symbols_are_both_sparse(tmp_path):
    database = _store(tmp_path, {"005930": 2000, "269620": 2})
    density = measure_print_density(
        ["005930", "269620", "088350"], database=database, lookback_hours=30.0, now=AT
    )
    # A symbol that printed nothing is measured as 0.0 rather than omitted: that is
    # the strongest evidence available, not missing data.
    assert density["088350"] == 0.0
    dense, sparse = partition_by_density(["005930", "269620", "088350"], density)
    assert dense == ("005930",)
    assert sparse == ("269620", "088350")


def test_partition_preserves_upstream_order_within_each_bucket():
    """Density gates; turnover and feasibility still rank inside the gate."""
    density = {"a": 50.0, "b": 0.0, "c": 40.0, "d": 1.0}
    dense, sparse = partition_by_density(("a", "b", "c", "d"), density)
    assert dense == ("a", "c")
    assert sparse == ("b", "d")


def test_exploration_reserve_rotates_and_is_capped_at_a_third():
    dense = [f"D{i}" for i in range(20)]
    sparse = [f"S{i}" for i in range(6)]

    first, cursor = blend_with_exploration(dense, sparse, limit=12, reserve=2, cursor=0)
    second, cursor = blend_with_exploration(dense, sparse, limit=12, reserve=2, cursor=cursor)
    assert [s for s in first if s.startswith("S")] == ["S0", "S1"]
    assert [s for s in second if s.startswith("S")] == ["S2", "S3"]

    # Never more than a third of the budget, however much is asked for.
    greedy, _ = blend_with_exploration(dense, sparse, limit=6, reserve=99, cursor=0)
    assert len([s for s in greedy if s.startswith("S")]) <= 2


def test_blend_backfills_so_the_pool_never_shrinks():
    dense = ["D0"]
    sparse = ["S0", "S1", "S2"]
    selected, _ = blend_with_exploration(dense, sparse, limit=4, reserve=1, cursor=0)
    assert len(selected) == 4
    assert set(selected) == {"D0", "S0", "S1", "S2"}


def test_ranking_puts_exploration_inside_the_truncation_budget(tmp_path):
    """A reserved slot behind the cut line is not a reserved slot.

    ``resolve_universe`` truncates to ``size``. If exploration names landed after
    that boundary an unmeasured symbol could never earn a subscription window, and
    the density measurement would become self-fulfilling.
    """
    # Six-digit tickers, because the market-minute denominator is computed over the
    # KRX peer group via GLOB. Synthetic names like "D0" fall outside it, the window
    # then contains no market minutes, and the call correctly fails open instead --
    # which silently turns this into a test of nothing.
    loud = [f"1000{index:02d}" for index in range(20)]
    quiet = [f"2000{index:02d}" for index in range(8)]
    prints = {symbol: 2000 for symbol in loud}
    prints.update({symbol: 1 for symbol in quiet})
    database = _store(tmp_path, prints)

    selected, _cursor, stats = rank_by_print_density(
        [*loud, *quiet], database=database, limit=12, cursor=0
    )
    assert stats["applied"] is True
    assert stats["dense"] == 20
    assert len(selected) == 12
    assert any(symbol in quiet for symbol in selected)


@pytest.mark.parametrize(
    "database, candidates",
    [
        ("does/not/exist.sqlite3", ["005930"]),
        (None, []),
    ],
)
def test_ranking_fails_open(tmp_path, database, candidates):
    """Losing the ranking is worse than keeping a few quiet names."""
    path = database if database is not None else _store(tmp_path, {})
    selected, cursor, stats = rank_by_print_density(
        candidates, database=path, limit=10, cursor=7
    )
    assert stats["applied"] is False
    assert list(selected) == candidates
    assert cursor == 7


def test_all_quiet_leaves_the_ranking_alone(tmp_path):
    """No dense candidate is a statement about the tape, not a reason to reshuffle.

    Distinct from the no-measurement path below it: here the window HAS market
    minutes and every candidate was measured against them, and still nothing clears
    the floor.
    """
    database = _store(tmp_path, {"000110": 3, "000220": 2, "000330": 1})
    selected, cursor, stats = rank_by_print_density(
        ["000110", "000220", "000330"], database=database, limit=3, cursor=4
    )
    assert stats["applied"] is False
    assert stats["reason"] == "NO_DENSE_CANDIDATE"
    assert selected == ("000110", "000220", "000330")
    assert cursor == 4


def test_a_window_with_no_krx_market_minutes_is_not_an_opinion(tmp_path):
    """An empty peer group must fail open, not mark every candidate sparse.

    The denominator is computed over KRX tickers only. A store holding nothing but
    US prints has no KRX market minutes, and reading that as "every KRX name is
    quiet" would reshuffle the ranking on the strength of no evidence at all.
    """
    database = _store(tmp_path, {"AAPL": 5000, "INTC": 4000})
    selected, cursor, stats = rank_by_print_density(
        ["005930", "000660"], database=database, limit=2, cursor=9
    )
    assert stats["applied"] is False
    assert stats["reason"] == "NO_MEASUREMENT"
    assert selected == ("005930", "000660")
    assert cursor == 9


def test_filter_can_be_switched_off(tmp_path, monkeypatch):
    database = _store(tmp_path, {"005930": 2000, "088350": 0})
    monkeypatch.setenv("REALTIME_KRX_DENSITY_FILTER_ENABLED", "0")
    selected, _cursor, stats = rank_by_print_density(
        ["088350", "005930"], database=database, limit=2, cursor=0
    )
    assert stats["applied"] is False
    assert stats["reason"] == "DISABLED"
    assert selected == ("088350", "005930")

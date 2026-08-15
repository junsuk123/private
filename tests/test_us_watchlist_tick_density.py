from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import web


@pytest.fixture(autouse=True)
def _reset_exploration_cursor() -> None:
    web._us_learning_watchlist_cache.pop("exploration_index", None)


def _account(cash_usd: float = 500.0) -> SimpleNamespace:
    return SimpleNamespace(
        cash=0.0,
        base_currency="KRW",
        cash_by_currency={"USD": cash_usd},
        orderable_cash_by_currency={"USD": cash_usd},
        holdings=(),
    )


def _build_market_database(path: Path, tapes: dict[str, int], *, price: float = 20.0) -> None:
    """Write ``tapes`` prints per symbol, evenly spread over the last 10 minutes."""
    now = datetime.now(timezone.utc)
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            """
            create table realtime_ticks (
                symbol text, price real, volume integer, received_at text
            );
            create table realtime_orderbook (
                symbol text, spread_bps real, received_at text
            );
            """
        )
        for symbol, count in tapes.items():
            for index in range(count):
                stamp = (now - timedelta(seconds=600.0 * index / max(1, count))).isoformat()
                connection.execute(
                    "insert into realtime_ticks values (?, ?, 10, ?)",
                    (symbol, price + (index % 3) * 0.01, stamp),
                )
            connection.execute(
                "insert into realtime_orderbook values (?, 5.0, ?)",
                (symbol, now.isoformat()),
            )
        connection.commit()


def test_sparse_tape_does_not_take_a_slot_from_a_dense_one(tmp_path: Path) -> None:
    """A name printing under a print a minute cannot feed the sub-second triggers.

    Measured on the live store: AVS at 0.35 prints/min held a realtime slot and
    returned TICK_WINDOW_NOT_READY on every evaluation, because the old ordering
    led on observed price range, which a sparse sample inflates.
    """
    database = tmp_path / "market.sqlite3"
    _build_market_database(
        database,
        {"DENSE": 400, "ALSODENSE": 300, "THIRD": 250, "SPARSE": 4},
    )

    selected = web._recent_affordable_us_watchlist(_account(), limit=3, database=database)

    assert "SPARSE" not in selected[:2]
    assert selected[:2] == ("DENSE", "ALSODENSE")


def test_one_slot_is_reserved_for_an_untested_name(tmp_path: Path) -> None:
    """Density alone would deadlock: no subscription, no prints, no promotion."""
    database = tmp_path / "market.sqlite3"
    _build_market_database(
        database,
        {"DENSE": 400, "ALSODENSE": 300, "THIRD": 250, "SPARSE": 4},
    )

    selected = web._recent_affordable_us_watchlist(_account(), limit=3, database=database)

    assert "SPARSE" in selected
    assert selected[:2] == ("DENSE", "ALSODENSE")


def test_exploration_slot_rotates_across_refreshes(tmp_path: Path) -> None:
    database = tmp_path / "market.sqlite3"
    _build_market_database(
        database,
        {"DENSE": 400, "SPARSEA": 4, "SPARSEB": 4, "SPARSEC": 4},
    )

    explored = {
        web._recent_affordable_us_watchlist(_account(), limit=3, database=database)[1]
        for _ in range(3)
    }

    assert len(explored) == 3


def test_a_two_slot_budget_spends_nothing_on_exploration(tmp_path: Path) -> None:
    """Half the live feed is too much to hand an unproven name."""
    database = tmp_path / "market.sqlite3"
    _build_market_database(database, {"DENSE": 400, "ALSODENSE": 300, "SPARSE": 4})

    selected = web._recent_affordable_us_watchlist(_account(), limit=2, database=database)

    assert selected == ("DENSE", "ALSODENSE")


def test_pool_is_never_smaller_than_before_when_nothing_is_dense(tmp_path: Path) -> None:
    """With no dense name available the sparse ones still fill the pool."""
    database = tmp_path / "market.sqlite3"
    _build_market_database(database, {"SPARSEA": 4, "SPARSEB": 4, "SPARSEC": 4})

    selected = web._recent_affordable_us_watchlist(_account(), limit=3, database=database)

    assert sorted(selected) == ["SPARSEA", "SPARSEB", "SPARSEC"]


def test_unaffordable_symbols_are_still_excluded(tmp_path: Path) -> None:
    database = tmp_path / "market.sqlite3"
    _build_market_database(database, {"PRICEY": 400}, price=900.0)
    _build_market_database(tmp_path / "cheap.sqlite3", {"CHEAP": 400}, price=5.0)

    assert web._recent_affordable_us_watchlist(_account(50.0), limit=2, database=database) == ()


def test_density_floor_is_configurable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = tmp_path / "market.sqlite3"
    _build_market_database(database, {"DENSE": 400, "MEDIUM": 60})

    monkeypatch.setenv("REALTIME_US_WATCHLIST_MIN_TICKS_PER_MINUTE", "1")
    monkeypatch.setenv("REALTIME_US_WATCHLIST_EXPLORATION_SLOTS", "0")

    # 60 prints over 10 minutes is 6/min: above a floor of 1, below the default 10.
    assert web._recent_affordable_us_watchlist(_account(), limit=2, database=database) == (
        "DENSE",
        "MEDIUM",
    )


def _stub_pool(monkeypatch: pytest.MonkeyPatch, pool: tuple[str, ...]) -> None:
    account = _account()
    monkeypatch.setattr(web, "_last_live_account_basis", lambda: {"stub": True})
    monkeypatch.setattr(web, "_account_snapshot_from_live_basis", lambda _basis: account)
    monkeypatch.setattr(
        web,
        "_recent_affordable_us_watchlist",
        lambda _account, *, limit, database=None, maximum_price_usd=None: pool[:limit],
    )
    # A short pool otherwise pulls in the seed list and the discovery scan, which
    # would make the assertions here about those and not about the rotation.
    monkeypatch.setattr(
        web, "_liquid_affordable_us_seed_symbols", lambda _account, *, limit: ()
    )
    monkeypatch.setattr(web, "_live_affordable_buy_candidate_symbols", lambda *, limit: ())
    monkeypatch.setenv("REALTIME_US_ROTATION_POOL_MULTIPLIER", "3")
    monkeypatch.setenv("REALTIME_US_SESSION_ANCHOR_SLOTS", "3")
    web._us_learning_watchlist_cache.update(
        {
            "at": 0.0,
            "cash_usd": None,
            "price_cap_usd": None,
            "symbols": (),
            "pool": (),
            "rotation_index": 0,
        }
    )


def test_rotation_keeps_the_dense_head_subscribed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rotating the whole pool walked every dense name out of the feed.

    ``_recent_affordable_us_watchlist`` returns measured density first, so with a
    pool multiplier of 3 two of every three refreshes subscribed a slice made
    entirely of sparse tail names — and the session-boxed strategies, which need
    the same symbol at the opening range and again six hours later in the last
    continuous half hour, could never see one.
    """
    pool = ("D1", "D2", "D3", "S1", "S2", "S3", "S4", "S5", "S6")
    _stub_pool(monkeypatch, pool)

    rounds = []
    for _ in range(3):
        rounds.append(web._sticky_us_learning_symbols(6))
        with web._live_lock:
            web._us_learning_watchlist_cache["at"] = 0.0

    for selected in rounds:
        assert selected[:3] == ("D1", "D2", "D3")
        assert len(selected) == 6
    # The remaining slots still scan: three refreshes must not repeat one slice.
    assert len({tuple(sorted(selected[3:])) for selected in rounds}) > 1


def test_a_pool_no_larger_than_the_budget_is_taken_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = ("D1", "D2", "D3")
    _stub_pool(monkeypatch, pool)

    assert web._sticky_us_learning_symbols(6) == ("D1", "D2", "D3")

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from app.storage.local_store import LocalResearchStore


def _build_realtime_source(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            """
            create table realtime_minute_bars (
                stream_id text not null default '',
                symbol text not null,
                minute_start text not null,
                open real not null,
                high real not null,
                low real not null,
                close real not null,
                volume integer not null,
                primary key (stream_id, symbol, minute_start)
            );
            create table realtime_ticks (
                record_id text primary key,
                symbol text not null,
                exchange_timestamp text not null,
                price real not null,
                volume integer not null,
                source text not null
            );
            """
        )
        connection.commit()


def _append_bar(path: Path, symbol: str, minute: str) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "insert or replace into realtime_minute_bars values ('s', ?, ?, 1, 2, 0.5, 1.5, 10)",
            (symbol, minute),
        )
        connection.commit()


def _append_tick(path: Path, record_id: str, symbol: str, stamp: str) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "insert into realtime_ticks values (?, ?, ?, 10.0, 3, 'feed')",
            (record_id, symbol, stamp),
        )
        connection.commit()


def _typed_rows(store: LocalResearchStore, table: str) -> list[tuple]:
    with closing(sqlite3.connect(store.db_path)) as connection:
        return connection.execute(f"select ticker, observed_at from {table} order by observed_at").fetchall()


def test_second_sync_copies_only_rows_appended_since_the_first(tmp_path: Path) -> None:
    """The projection must not re-copy what it already wrote.

    Re-reading the newest 20,000 rows every refresh cycle is what made a cycle
    take longer than its own interval; the row count it reported was the batch
    size, not the number of genuinely new rows.
    """
    source = tmp_path / "realtime.sqlite3"
    _build_realtime_source(source)
    for index in range(5):
        _append_bar(source, "AAA", f"2026-08-10T00:0{index}:00+00:00")
        _append_tick(source, f"t{index}", "AAA", f"2026-08-10T00:0{index}:00+00:00")

    store = LocalResearchStore(root=tmp_path / "store")
    assert store.sync_realtime_ohlcv(source) == 5
    assert store.sync_realtime_quotes(source) == 5

    assert store.sync_realtime_ohlcv(source) == 0
    assert store.sync_realtime_quotes(source) == 0

    _append_bar(source, "AAA", "2026-08-10T00:09:00+00:00")
    _append_tick(source, "t9", "AAA", "2026-08-10T00:09:00+00:00")

    assert store.sync_realtime_ohlcv(source) == 1
    assert store.sync_realtime_quotes(source) == 1
    assert len(_typed_rows(store, "typed_ohlcv_bars")) == 6
    assert len(_typed_rows(store, "typed_realtime_quotes")) == 6


def test_cold_start_seeds_from_the_tail_instead_of_replaying_history(tmp_path: Path) -> None:
    source = tmp_path / "realtime.sqlite3"
    _build_realtime_source(source)
    for index in range(20):
        _append_tick(source, f"t{index:02d}", "AAA", f"2026-08-10T00:{index:02d}:00+00:00")

    store = LocalResearchStore(root=tmp_path / "store")
    assert store.sync_realtime_quotes(source, limit=5) == 5
    observed = [row[1] for row in _typed_rows(store, "typed_realtime_quotes")]
    assert observed == [f"2026-08-10T00:{index:02d}:00+00:00" for index in range(15, 20)]


def test_backlog_larger_than_the_batch_is_drained_across_calls(tmp_path: Path) -> None:
    source = tmp_path / "realtime.sqlite3"
    _build_realtime_source(source)
    store = LocalResearchStore(root=tmp_path / "store")
    _append_tick(source, "t00", "AAA", "2026-08-10T00:00:00+00:00")
    assert store.sync_realtime_quotes(source, limit=2) == 1

    for index in range(1, 8):
        _append_tick(source, f"t{index:02d}", "AAA", f"2026-08-10T00:{index:02d}:00+00:00")

    assert store.sync_realtime_quotes(source, limit=2) == 2
    assert store.sync_realtime_quotes(source, limit=2) == 2
    assert store.sync_realtime_quotes(source, limit=2) == 2
    assert store.sync_realtime_quotes(source, limit=2) == 1
    assert store.sync_realtime_quotes(source, limit=2) == 0
    assert len(_typed_rows(store, "typed_realtime_quotes")) == 8


def test_rebuilt_source_database_reseeds_instead_of_stalling(tmp_path: Path) -> None:
    """A watermark above the source's highest rowid means the source was replaced."""
    source = tmp_path / "realtime.sqlite3"
    _build_realtime_source(source)
    for index in range(6):
        _append_tick(source, f"t{index}", "AAA", f"2026-08-10T00:0{index}:00+00:00")
    store = LocalResearchStore(root=tmp_path / "store")
    assert store.sync_realtime_quotes(source) == 6

    source.unlink()
    _build_realtime_source(source)
    _append_tick(source, "n0", "BBB", "2026-08-11T00:00:00+00:00")

    assert store.sync_realtime_quotes(source) == 1


def test_updated_minute_bar_is_reprojected(tmp_path: Path) -> None:
    """``insert or replace`` in the source re-inserts at a higher rowid."""
    source = tmp_path / "realtime.sqlite3"
    _build_realtime_source(source)
    _append_bar(source, "AAA", "2026-08-10T00:00:00+00:00")
    store = LocalResearchStore(root=tmp_path / "store")
    assert store.sync_realtime_ohlcv(source) == 1

    with closing(sqlite3.connect(source)) as connection:
        connection.execute(
            """
            insert or replace into realtime_minute_bars
            values ('s', 'AAA', '2026-08-10T00:00:00+00:00', 1, 9, 0.5, 8.5, 99)
            """
        )
        connection.commit()

    assert store.sync_realtime_ohlcv(source) == 1
    with closing(sqlite3.connect(store.db_path)) as connection:
        close, volume = connection.execute(
            "select close, volume from typed_ohlcv_bars where ticker = 'AAA'"
        ).fetchone()
    assert (close, volume) == (8.5, 99)


def test_missing_source_database_is_a_no_op(tmp_path: Path) -> None:
    store = LocalResearchStore(root=tmp_path / "store")
    assert store.sync_realtime_ohlcv(tmp_path / "absent.sqlite3") == 0
    assert store.sync_realtime_quotes(tmp_path / "absent.sqlite3") == 0

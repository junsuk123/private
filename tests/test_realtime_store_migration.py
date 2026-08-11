"""v1 → v2 실시간 저장소 마이그레이션 검증.

핵심 계약:

* 기존 SQLite 파일을 **삭제하거나 재생성하지 않는다.** 행 수와 내용이 보존된다.
* 기존 행은 어느 venue/session 에서 왔는지 알 수 없으므로 ``metadata_inferred=1``.
* ``realtime_minute_bars`` identity 에 ``stream_id`` 가 포함되어 KRX·NXT·통합 피드가
  같은 행을 다투지 않는다.
* 최신 시세 조회가 마이그레이션 후에도 같은 값을 돌려준다.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone

import pytest

from app.data.market_capabilities import FeedScope, MarketGroup, SessionId, Venue
from app.data.realtime_store import (
    SCHEMA_VERSION,
    RealtimeMarketDataStore,
)
from app.data.realtime_types import (
    KIS_REALTIME_SOURCE,
    FeedMetadata,
    OrderbookLevel,
    RealtimeOrderbookSnapshot,
    RealtimeTradeTick,
)

BASE = datetime(2026, 8, 5, 0, 30, tzinfo=timezone.utc)


V1_SCHEMA = """
create table realtime_ticks (
    record_id text primary key,
    symbol text not null,
    exchange_timestamp text not null,
    received_at text not null,
    source text not null,
    price real not null,
    volume integer not null,
    trade_direction text,
    sequence_key text,
    raw_checksum text,
    latency_ms real not null
);
create table realtime_orderbook (
    record_id text primary key,
    symbol text not null,
    exchange_timestamp text not null,
    received_at text not null,
    source text not null,
    best_bid real not null,
    best_ask real not null,
    spread_bps real not null,
    total_bid_volume integer not null,
    total_ask_volume integer not null,
    imbalance real not null,
    levels_json text not null,
    sequence_key text,
    raw_checksum text,
    latency_ms real not null
);
create table realtime_minute_bars (
    symbol text not null,
    minute_start text not null,
    open real not null,
    high real not null,
    low real not null,
    close real not null,
    volume integer not null,
    vwap real not null,
    trade_count integer not null,
    spread_bps real not null,
    orderbook_imbalance real not null,
    liquidity_score real not null,
    volatility real not null,
    last_update_age_ms real not null,
    source_record_ids_json text not null,
    primary key(symbol, minute_start)
);
create table market_data_health (
    symbol text not null,
    checked_at text not null,
    quote_count integer not null,
    orderbook_count integer not null,
    latest_tick_at text,
    latest_orderbook_at text,
    max_quote_age_ms integer not null,
    max_orderbook_age_ms integer not null,
    source text not null,
    source_quality_score real not null,
    ok_for_live_buy integer not null,
    reason_codes_json text not null,
    primary key(symbol, checked_at)
);
create table data_source_events (
    event_id text primary key,
    event_type text not null,
    symbol text,
    observed_at text not null,
    source text not null,
    payload_json text not null
);
"""


def _build_v1_database(path, *, tick_count: int = 25) -> dict[str, int]:
    """metadata 컬럼이 전혀 없는 v1 파일을 실제 데이터와 함께 만든다."""
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(V1_SCHEMA)
        for index in range(tick_count):
            moment = (BASE + timedelta(seconds=index)).isoformat()
            conn.execute(
                "insert into realtime_ticks values (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    f"legacy-tick-{index}",
                    "005930",
                    moment,
                    moment,
                    KIS_REALTIME_SOURCE,
                    70000.0 + index,
                    10 + index,
                    "BUY",
                    None,
                    None,
                    1.0,
                ),
            )
        conn.execute(
            "insert into realtime_orderbook values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "legacy-book-1",
                "005930",
                BASE.isoformat(),
                BASE.isoformat(),
                KIS_REALTIME_SOURCE,
                69900.0,
                70100.0,
                28.6,
                500,
                400,
                0.111,
                json.dumps(
                    [{"bid_price": 69900.0, "bid_size": 500, "ask_price": 70100.0, "ask_size": 400}]
                ),
                None,
                None,
                2.0,
            ),
        )
        conn.execute(
            "insert into realtime_minute_bars values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "005930",
                BASE.replace(second=0).isoformat(),
                70000.0,
                70024.0,
                70000.0,
                70024.0,
                775,
                70012.0,
                25,
                28.6,
                0.111,
                0.5,
                7.2,
                12.0,
                json.dumps([f"legacy-tick-{i}" for i in range(tick_count)]),
            ),
        )
        conn.commit()
        counts = {
            table: conn.execute(f"select count(*) from {table}").fetchone()[0]
            for table in ("realtime_ticks", "realtime_orderbook", "realtime_minute_bars")
        }
    return counts


@pytest.fixture()
def v1_database(tmp_path):
    path = tmp_path / "realtime_market_data.sqlite3"
    counts = _build_v1_database(path)
    return path, counts


def test_migration_preserves_every_row(v1_database):
    path, before = v1_database
    store = RealtimeMarketDataStore(path)
    with closing(sqlite3.connect(path)) as conn:
        after = {
            table: conn.execute(f"select count(*) from {table}").fetchone()[0]
            for table in before
        }
    assert after == before
    assert store.schema_version() == SCHEMA_VERSION


def test_migration_records_version_and_history(v1_database):
    path, before = v1_database
    store = RealtimeMarketDataStore(path)
    history = store.migration_history()
    assert [item["version"] for item in history] == [2]
    entry = history[0]
    assert entry["rows_before"] == sum(before.values())
    assert entry["rows_after"] == sum(before.values())
    assert entry["applied_at"]
    assert "metadata" in entry["description"]


def test_migration_is_idempotent(v1_database):
    path, before = v1_database
    RealtimeMarketDataStore(path)
    store = RealtimeMarketDataStore(path)
    assert store.schema_version() == SCHEMA_VERSION
    assert len(store.migration_history()) == 1
    with closing(sqlite3.connect(path)) as conn:
        after = {
            table: conn.execute(f"select count(*) from {table}").fetchone()[0]
            for table in before
        }
    assert after == before


def test_legacy_rows_are_marked_metadata_inferred(v1_database):
    path, _ = v1_database
    RealtimeMarketDataStore(path)
    with closing(sqlite3.connect(path)) as conn:
        for table in ("realtime_ticks", "realtime_orderbook", "realtime_minute_bars"):
            rows = conn.execute(
                f"select count(*) from {table} where metadata_inferred = 1"
            ).fetchone()[0]
            total = conn.execute(f"select count(*) from {table}").fetchone()[0]
            assert rows == total, table


def test_legacy_rows_are_not_live_buy_eligible(v1_database):
    path, _ = v1_database
    store = RealtimeMarketDataStore(path)
    tick = store.latest_tick("005930")
    assert tick is not None
    ok, reasons = tick.meta.is_live_buy_eligible()
    assert ok is False
    assert reasons


def test_latest_quote_survives_migration(v1_database):
    """마이그레이션 전후 최신 시세 조회 결과가 같아야 한다."""
    path, _ = v1_database
    with closing(sqlite3.connect(path)) as conn:
        expected = conn.execute(
            "select price, volume from realtime_ticks"
            " order by received_at desc, exchange_timestamp desc limit 1"
        ).fetchone()
    store = RealtimeMarketDataStore(path)
    tick = store.latest_tick("005930")
    assert tick is not None
    assert (tick.price, tick.volume) == (float(expected[0]), int(expected[1]))

    book = store.latest_orderbook("005930")
    assert book is not None
    assert book.best_bid == 69900.0
    assert book.best_ask == 70100.0


def test_minute_bar_stream_id_has_a_default(v1_database):
    """새 NOT NULL 컬럼에는 반드시 DEFAULT 가 있어야 한다.

    운영에서 실제로 터진 사고: v2 가 ``stream_id text not null`` 을 DEFAULT 없이 만들었고,
    마이그레이션 시점에 이미 실행 중이던 구버전 서버 프로세스는 그 컬럼을 모르는 INSERT 를
    계속 보냈다. 결과는

        IntegrityError: NOT NULL constraint failed: realtime_minute_bars.stream_id

    가 수집 주기마다 반복되는 것이었다. additive 마이그레이션이 하위호환되려면 컬럼을
    모르는 writer 의 INSERT 가 그대로 성공해야 한다.
    """
    path, _ = v1_database
    RealtimeMarketDataStore(path)
    with closing(sqlite3.connect(path)) as conn:
        defaults = {
            str(row[1]): row[4]
            for row in conn.execute("pragma table_info(realtime_minute_bars)").fetchall()
        }
    assert defaults["stream_id"] is not None, "stream_id 에 DEFAULT 가 없다"


def test_writer_unaware_of_stream_id_still_inserts(v1_database):
    """구버전 코드가 보내는 형태의 INSERT (stream_id 미포함) 가 성공해야 한다."""
    path, _ = v1_database
    RealtimeMarketDataStore(path)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            """
            insert or replace into realtime_minute_bars
            (symbol, minute_start, open, high, low, close, volume, vwap, trade_count,
             spread_bps, orderbook_imbalance, liquidity_score, volatility,
             last_update_age_ms, source_record_ids_json)
            values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            ("AAPL", BASE.isoformat(), 190.0, 190.5, 189.5, 190.2, 100, 190.1, 5,
             12.0, 0.1, 0.4, 0.3, 10.0, "[]"),
        )
        conn.commit()
        stored = conn.execute(
            "select stream_id from realtime_minute_bars where symbol = 'AAPL'"
        ).fetchone()
    assert stored is not None
    assert stored[0] == ""


def test_v3_is_skipped_when_v2_already_produced_the_default(v1_database):
    """v1 → 최신으로 한 번에 올라오는 경로에서는 v3 이 할 일이 없다.

    v2 DDL 이 이미 ``stream_id text not null default ''`` 를 만들기 때문이다. v3 은
    **DEFAULT 없이 만들어진 기존 v2 파일** 을 고치기 위한 마이그레이션이므로 여기서는
    no-op 이고, 그래도 최종 스키마 버전은 최신이어야 한다.
    """
    path, _ = v1_database
    store = RealtimeMarketDataStore(path)
    assert store.schema_version() == SCHEMA_VERSION
    assert [item["version"] for item in store.migration_history()] == [2]


def test_v3_repairs_a_v2_database_without_a_default(tmp_path):
    """DEFAULT 없이 만들어진 v2 파일을 v3 이 실제로 고치는지."""
    path = tmp_path / "v2_without_default.sqlite3"
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(V1_SCHEMA)
        conn.execute("drop table realtime_minute_bars")
        # v3 이전의 v2 가 만들던 형태 — stream_id 에 DEFAULT 가 없다.
        conn.execute(
            """
            create table realtime_minute_bars (
                stream_id text not null,
                symbol text not null,
                minute_start text not null,
                open real not null, high real not null, low real not null,
                close real not null, volume integer not null, vwap real not null,
                trade_count integer not null, spread_bps real not null,
                orderbook_imbalance real not null, liquidity_score real not null,
                volatility real not null, last_update_age_ms real not null,
                source_record_ids_json text not null,
                metadata_inferred integer not null default 1,
                primary key(stream_id, symbol, minute_start)
            )
            """
        )
        conn.execute(
            "insert into realtime_minute_bars (stream_id, symbol, minute_start, open,"
            " high, low, close, volume, vwap, trade_count, spread_bps,"
            " orderbook_imbalance, liquidity_score, volatility, last_update_age_ms,"
            " source_record_ids_json) values"
            " ('KR:KRX:VENUE_SPECIFIC:H0STCNT0','A',?,1,1,1,1,1,1,1,0,0,0,0,0,'[]')",
            (BASE.isoformat(),),
        )
        conn.execute(
            "create table schema_version (id integer primary key check (id = 1),"
            " version integer not null)"
        )
        conn.execute("insert into schema_version (id, version) values (1, 2)")
        conn.execute(
            "create table schema_migrations (version integer primary key,"
            " applied_at text not null, description text not null,"
            " rows_before integer not null, rows_after integer not null)"
        )
        conn.commit()

    store = RealtimeMarketDataStore(path)
    assert store.schema_version() == SCHEMA_VERSION
    assert 3 in [item["version"] for item in store.migration_history()]
    with closing(sqlite3.connect(path)) as conn:
        defaults = {
            str(row[1]): row[4]
            for row in conn.execute("pragma table_info(realtime_minute_bars)").fetchall()
        }
        assert defaults["stream_id"] is not None
        # 기존 행이 보존되고 stream_id 도 그대로다.
        row = conn.execute(
            "select stream_id from realtime_minute_bars where symbol = 'A'"
        ).fetchone()
        assert row[0] == "KR:KRX:VENUE_SPECIFIC:H0STCNT0"


def test_v3_preserves_minute_bars(v1_database):
    path, before = v1_database
    RealtimeMarketDataStore(path)
    with closing(sqlite3.connect(path)) as conn:
        after = conn.execute("select count(*) from realtime_minute_bars").fetchone()[0]
    assert after == before["realtime_minute_bars"]


def test_minute_bar_primary_key_includes_stream(v1_database):
    path, _ = v1_database
    RealtimeMarketDataStore(path)
    with closing(sqlite3.connect(path)) as conn:
        index_info = conn.execute(
            "select name from pragma_index_list('realtime_minute_bars') where origin = 'pk'"
        ).fetchone()
        assert index_info is not None
        key_columns = [
            row[2]
            for row in conn.execute(f"pragma index_info({index_info[0]})").fetchall()
        ]
    assert "stream_id" in key_columns
    assert "symbol" in key_columns
    assert "minute_start" in key_columns


# --------------------------------------------------------------------------- #
# 신규 파일 + venue 분리
# --------------------------------------------------------------------------- #
def _meta(venue: Venue, session: SessionId, tr_id: str, scope: FeedScope) -> FeedMetadata:
    return FeedMetadata(
        market_group=MarketGroup.KR,
        exchange="KRX" if venue is not Venue.NXT else "NXT",
        venue=venue,
        session=session,
        currency="KRW",
        feed_scope=scope,
        tr_id=tr_id,
        subscription_key="005930",
        is_consolidated=scope is FeedScope.UNIFIED,
        is_tradeable=scope is not FeedScope.UNIFIED,
    )


KRX_META = _meta(Venue.KRX, SessionId.KRX_REGULAR, "H0STCNT0", FeedScope.VENUE_SPECIFIC)
NXT_META = _meta(Venue.NXT, SessionId.NXT_REGULAR, "H0NXCNT0", FeedScope.VENUE_SPECIFIC)
UNIFIED_META = _meta(
    Venue.KRX_NXT_UNIFIED, SessionId.KRX_REGULAR, "H0UNCNT0", FeedScope.UNIFIED
)


def _tick(meta: FeedMetadata, *, price: float, volume: int, offset: int = 0):
    moment = BASE + timedelta(seconds=offset)
    return RealtimeTradeTick(
        symbol="005930",
        exchange_timestamp=moment,
        received_at=moment,
        source=KIS_REALTIME_SOURCE,
        price=price,
        volume=volume,
        meta=meta,
    )


def test_fresh_database_does_not_mark_rows_inferred(tmp_path):
    store = RealtimeMarketDataStore(tmp_path / "fresh.sqlite3")
    assert store.schema_version() == SCHEMA_VERSION
    store.save_ticks((_tick(KRX_META, price=70000.0, volume=10),))
    tick = store.latest_tick("005930")
    assert tick is not None
    assert tick.meta.metadata_inferred is False
    ok, reasons = tick.meta.is_live_buy_eligible()
    assert ok is True, reasons


def test_same_symbol_krx_and_nxt_ticks_are_distinct_rows(tmp_path):
    store = RealtimeMarketDataStore(tmp_path / "venues.sqlite3")
    store.save_ticks(
        (
            _tick(KRX_META, price=70000.0, volume=10),
            _tick(NXT_META, price=70000.0, volume=10),
        )
    )
    with closing(sqlite3.connect(tmp_path / "venues.sqlite3")) as conn:
        venues = {
            row[0]
            for row in conn.execute("select venue from realtime_ticks").fetchall()
        }
        assert venues == {"KRX", "NXT"}
        assert conn.execute("select count(*) from realtime_ticks").fetchone()[0] == 2


def test_tick_and_orderbook_tables_do_not_duplicate_rows_into_source_events(tmp_path):
    path = tmp_path / "no_duplicate_events.sqlite3"
    store = RealtimeMarketDataStore(path)
    store.save_ticks((_tick(KRX_META, price=70_000.0, volume=10),))
    store.save_orderbooks(
        (
            RealtimeOrderbookSnapshot(
                symbol="005930",
                exchange_timestamp=BASE,
                received_at=BASE,
                source=KIS_REALTIME_SOURCE,
                levels=(OrderbookLevel(69_900.0, 100, 70_100.0, 100),),
                meta=KRX_META,
            ),
        )
    )

    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("select count(*) from data_source_events").fetchone()[0] == 0


def test_raw_microstructure_retention_prunes_in_bounded_batches(tmp_path):
    path = tmp_path / "retention.sqlite3"
    store = RealtimeMarketDataStore(path)
    store.save_ticks((_tick(KRX_META, price=70_000.0, volume=10),))

    deleted = store.prune_market_history(retention_hours=24, batch_size=1_000)

    assert deleted == 1
    assert store.latest_tick("005930") is None


def test_unified_and_venue_feeds_do_not_double_count_volume(tmp_path):
    """같은 체결이 통합·venue 피드로 둘 다 오더라도 bar 거래량은 두 배가 되지 않는다.

    bar 는 ``stream_id`` 별로 만들어지므로 애초에 합산되지 않는다.
    """
    path = tmp_path / "dedup.sqlite3"
    store = RealtimeMarketDataStore(path)
    store.save_ticks(
        (
            _tick(KRX_META, price=70000.0, volume=100),
            _tick(UNIFIED_META, price=70000.0, volume=100),
        )
    )
    bar = store.build_latest_minute_bar("005930", now=BASE + timedelta(seconds=5))
    assert bar is not None
    assert bar.volume == 100, "스트림별로 분리되어야 하며 200 이 되면 이중 계산이다"

    # 저장소에는 두 스트림의 bar 가 각각 존재한다 (venue 별 분석이 가능해야 한다).
    with closing(sqlite3.connect(path)) as conn:
        stored = {
            str(row[0]): int(row[1])
            for row in conn.execute(
                "select stream_id, volume from realtime_minute_bars where symbol = '005930'"
            ).fetchall()
        }
    assert set(stored) == {KRX_META.stream_id, UNIFIED_META.stream_id}
    assert all(volume == 100 for volume in stored.values())


def test_recent_minute_bars_returns_a_single_coherent_stream(tmp_path):
    """조회는 **한 스트림만** 돌려준다.

    여러 스트림을 섞어 반환하면 같은 ``minute_start`` 가 중복 등장해 "N개 관측 = N분"
    가정이 깨지고, 하류의 rolling return / 변동성이 서로 다른 피드에서 계산된 값으로
    조용히 오염된다 (실제로 macro frame 이 그 때문에 index_trend=None 이 됐다).
    """
    store = RealtimeMarketDataStore(tmp_path / "single_stream.sqlite3")
    store.save_ticks(
        (
            _tick(KRX_META, price=70000.0, volume=100),
            _tick(UNIFIED_META, price=70000.0, volume=100),
        )
    )
    store.build_latest_minute_bar("005930", now=BASE + timedelta(seconds=5))

    bars = store.recent_minute_bars("005930", BASE - timedelta(minutes=1))
    assert len({item.stream_id for item in bars}) == 1
    minutes = [item.minute_start for item in bars]
    assert len(minutes) == len(set(minutes)), "같은 분이 중복되면 안 된다"

    # 특정 스트림을 명시적으로 요구할 수도 있어야 한다.
    unified = store.recent_minute_bars(
        "005930", BASE - timedelta(minutes=1), stream_id=UNIFIED_META.stream_id
    )
    assert unified and all(item.stream_id == UNIFIED_META.stream_id for item in unified)
    krx = store.recent_minute_bars(
        "005930", BASE - timedelta(minutes=1), stream_id=KRX_META.stream_id
    )
    assert krx and all(item.stream_id == KRX_META.stream_id for item in krx)


def test_preferred_stream_prefers_coverage_then_tradeability(tmp_path):
    """커버리지가 넓은 스트림이 우선이고, 동수면 tradeable 한 쪽이다."""
    store = RealtimeMarketDataStore(tmp_path / "prefer.sqlite3")
    # KRX(tradeable) 는 1분, UNIFIED(비 tradeable) 는 3분 커버리지.
    store.save_ticks((_tick(KRX_META, price=70000.0, volume=10),))
    store.build_latest_minute_bar("005930", now=BASE + timedelta(seconds=5))
    for offset in (0, 60, 120):
        store.save_ticks(
            (_tick(UNIFIED_META, price=70000.0 + offset, volume=10, offset=offset),)
        )
        store.build_latest_minute_bar(
            "005930", now=BASE + timedelta(seconds=offset + 5)
        )
    since = BASE - timedelta(minutes=1)
    assert store.preferred_minute_bar_stream("005930", since) == UNIFIED_META.stream_id

    # 커버리지가 같으면 tradeable 한 스트림을 고른다.
    other = RealtimeMarketDataStore(tmp_path / "tie.sqlite3")
    other.save_ticks(
        (
            _tick(KRX_META, price=70000.0, volume=10),
            _tick(UNIFIED_META, price=70000.0, volume=10),
        )
    )
    other.build_latest_minute_bar("005930", now=BASE + timedelta(seconds=5))
    assert other.preferred_minute_bar_stream("005930", since) == KRX_META.stream_id


def test_preferred_stream_is_none_without_bars(tmp_path):
    store = RealtimeMarketDataStore(tmp_path / "empty.sqlite3")
    since = BASE - timedelta(minutes=1)
    assert store.preferred_minute_bar_stream("005930", since) is None
    assert store.recent_minute_bars("005930", since) == ()


def test_cross_stream_duplicates_are_observable(tmp_path):
    store = RealtimeMarketDataStore(tmp_path / "observe.sqlite3")
    store.save_ticks(
        (
            _tick(KRX_META, price=70000.0, volume=100),
            _tick(UNIFIED_META, price=70000.0, volume=100),
            _tick(KRX_META, price=70100.0, volume=5, offset=1),
        )
    )
    assert store.cross_stream_duplicate_count(BASE - timedelta(minutes=1)) == 1


def test_stream_inventory_reports_per_stream_counts(tmp_path):
    store = RealtimeMarketDataStore(tmp_path / "inventory.sqlite3")
    store.save_ticks(
        (
            _tick(KRX_META, price=70000.0, volume=10),
            _tick(KRX_META, price=70010.0, volume=10, offset=1),
            _tick(NXT_META, price=70020.0, volume=10, offset=2),
        )
    )
    inventory = {item["stream_id"]: item for item in store.stream_inventory(BASE - timedelta(minutes=1))}
    assert len(inventory) == 2
    krx = inventory["KR:KRX:VENUE_SPECIFIC:H0STCNT0"]
    assert krx["ticks"] == 2
    assert krx["venue"] == "KRX"
    assert krx["session"] == "KRX_REGULAR"
    assert krx["inferred_rows"] == 0
    nxt = inventory["KR:NXT:VENUE_SPECIFIC:H0NXCNT0"]
    assert nxt["ticks"] == 1


def test_minute_bar_uses_orderbook_from_the_same_stream(tmp_path):
    """다른 거래소 호가로 spread 를 채우지 않는다."""
    store = RealtimeMarketDataStore(tmp_path / "book.sqlite3")
    store.save_ticks((_tick(KRX_META, price=70000.0, volume=10),))
    store.save_orderbooks(
        (
            RealtimeOrderbookSnapshot(
                symbol="005930",
                exchange_timestamp=BASE,
                received_at=BASE,
                source=KIS_REALTIME_SOURCE,
                levels=(OrderbookLevel(69900.0, 100, 70100.0, 100),),
                meta=NXT_META,
            ),
        )
    )
    bar = store.build_latest_minute_bar(
        "005930", now=BASE + timedelta(seconds=5), stream_id=KRX_META.stream_id
    )
    assert bar is not None
    # KRX 스트림에는 호가가 없으므로 NXT 호가를 빌려오지 않는다.
    assert bar.spread_bps == 0.0


def test_metadata_roundtrips_through_storage(tmp_path):
    store = RealtimeMarketDataStore(tmp_path / "roundtrip.sqlite3")
    store.save_orderbooks(
        (
            RealtimeOrderbookSnapshot(
                symbol="AAPL",
                exchange_timestamp=BASE,
                received_at=BASE,
                source=KIS_REALTIME_SOURCE,
                levels=(OrderbookLevel(190.0, 10, 190.05, 12),),
                meta=FeedMetadata(
                    market_group=MarketGroup.US,
                    exchange="NASD",
                    venue=Venue.NASDAQ,
                    session=SessionId.US_REGULAR,
                    currency="USD",
                    feed_scope=FeedScope.FREE_REALTIME,
                    tr_id="HDFSASP0",
                    subscription_key="DNASAAPL",
                    is_consolidated=False,
                    is_tradeable=True,
                ),
            ),
        )
    )
    book = store.latest_orderbook("AAPL")
    assert book is not None
    assert book.meta.market_group is MarketGroup.US
    assert book.meta.venue is Venue.NASDAQ
    assert book.meta.session is SessionId.US_REGULAR
    assert book.meta.subscription_key == "DNASAAPL"
    assert book.meta.feed_scope is FeedScope.FREE_REALTIME
    assert book.meta.is_consolidated is False
    assert book.depth_level_count == 1

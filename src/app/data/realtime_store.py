from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.data.realtime_types import (
    MarketDataHealth,
    RealtimeMinuteBar,
    RealtimeOrderbookSnapshot,
    RealtimeTradeTick,
    to_jsonable,
)


class RealtimeMarketDataStore:
    def __init__(self, db_path: str | Path = "data/store/realtime_market_data.sqlite3") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def init_db(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute("pragma journal_mode=wal")
            conn.executescript(
                """
                create table if not exists realtime_ticks (
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
                create index if not exists idx_realtime_ticks_symbol_time
                    on realtime_ticks(symbol, exchange_timestamp);

                create table if not exists realtime_orderbook (
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
                create index if not exists idx_realtime_orderbook_symbol_time
                    on realtime_orderbook(symbol, exchange_timestamp);

                create table if not exists realtime_minute_bars (
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

                create table if not exists market_data_health (
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

                create table if not exists data_source_events (
                    event_id text primary key,
                    event_type text not null,
                    symbol text,
                    observed_at text not null,
                    source text not null,
                    payload_json text not null
                );
                """
            )
            conn.commit()

    def save_ticks(self, ticks: tuple[RealtimeTradeTick, ...]) -> int:
        rows = [
            (
                tick.record_id,
                tick.symbol,
                tick.exchange_timestamp.isoformat(),
                tick.received_at.isoformat(),
                tick.source,
                tick.price,
                tick.volume,
                tick.trade_direction,
                tick.sequence_key,
                tick.raw_checksum,
                tick.latency_ms,
            )
            for tick in ticks
        ]
        return self._insert_many(
            """
            insert or ignore into realtime_ticks
            (record_id, symbol, exchange_timestamp, received_at, source, price, volume,
             trade_direction, sequence_key, raw_checksum, latency_ms)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def save_orderbooks(self, snapshots: tuple[RealtimeOrderbookSnapshot, ...]) -> int:
        rows = [
            (
                snapshot.record_id,
                snapshot.symbol,
                snapshot.exchange_timestamp.isoformat(),
                snapshot.received_at.isoformat(),
                snapshot.source,
                snapshot.best_bid,
                snapshot.best_ask,
                snapshot.spread_bps,
                snapshot.total_bid_volume,
                snapshot.total_ask_volume,
                snapshot.imbalance,
                json.dumps(to_jsonable(snapshot.levels), ensure_ascii=True, sort_keys=True),
                snapshot.sequence_key,
                snapshot.raw_checksum,
                snapshot.latency_ms,
            )
            for snapshot in snapshots
        ]
        return self._insert_many(
            """
            insert or ignore into realtime_orderbook
            (record_id, symbol, exchange_timestamp, received_at, source, best_bid, best_ask,
             spread_bps, total_bid_volume, total_ask_volume, imbalance, levels_json,
             sequence_key, raw_checksum, latency_ms)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def save_minute_bars(self, bars: tuple[RealtimeMinuteBar, ...]) -> int:
        rows = [
            (
                bar.symbol,
                bar.minute_start.isoformat(),
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
                bar.vwap,
                bar.trade_count,
                bar.spread_bps,
                bar.orderbook_imbalance,
                bar.liquidity_score,
                bar.volatility,
                bar.last_update_age_ms,
                json.dumps(list(bar.source_record_ids), ensure_ascii=True),
            )
            for bar in bars
        ]
        return self._insert_many(
            """
            insert or replace into realtime_minute_bars
            (symbol, minute_start, open, high, low, close, volume, vwap, trade_count,
             spread_bps, orderbook_imbalance, liquidity_score, volatility, last_update_age_ms,
             source_record_ids_json)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def save_health(self, health: MarketDataHealth) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                insert or replace into market_data_health
                (symbol, checked_at, quote_count, orderbook_count, latest_tick_at, latest_orderbook_at,
                 max_quote_age_ms, max_orderbook_age_ms, source, source_quality_score,
                 ok_for_live_buy, reason_codes_json)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    health.symbol,
                    health.checked_at.isoformat(),
                    health.quote_count,
                    health.orderbook_count,
                    health.latest_tick_at.isoformat() if health.latest_tick_at else None,
                    health.latest_orderbook_at.isoformat() if health.latest_orderbook_at else None,
                    health.max_quote_age_ms,
                    health.max_orderbook_age_ms,
                    health.source,
                    health.source_quality_score,
                    1 if health.ok_for_live_buy else 0,
                    json.dumps(list(health.reason_codes), ensure_ascii=True),
                ),
            )
            conn.commit()

    def latest_tick(self, symbol: str) -> RealtimeTradeTick | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                select symbol, exchange_timestamp, received_at, source, price, volume,
                       trade_direction, sequence_key, raw_checksum, latency_ms
                from realtime_ticks
                where symbol = ?
                order by received_at desc, exchange_timestamp desc
                limit 1
                """,
                (symbol,),
            ).fetchone()
        if row is None:
            return None
        return RealtimeTradeTick(
            symbol=row[0],
            exchange_timestamp=_parse_dt(row[1]),
            received_at=_parse_dt(row[2]),
            source=row[3],
            price=float(row[4]),
            volume=int(row[5]),
            trade_direction=row[6],
            sequence_key=row[7],
            raw_checksum=row[8],
            latency_ms=float(row[9]),
        )

    def latest_orderbook(self, symbol: str) -> RealtimeOrderbookSnapshot | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                select symbol, exchange_timestamp, received_at, source, levels_json,
                       sequence_key, raw_checksum, latency_ms
                from realtime_orderbook
                where symbol = ?
                order by received_at desc, exchange_timestamp desc
                limit 1
                """,
                (symbol,),
            ).fetchone()
        if row is None:
            return None
        from app.data.realtime_types import OrderbookLevel

        levels = tuple(OrderbookLevel(**item) for item in json.loads(row[4]))
        return RealtimeOrderbookSnapshot(
            symbol=row[0],
            exchange_timestamp=_parse_dt(row[1]),
            received_at=_parse_dt(row[2]),
            source=row[3],
            levels=levels,
            sequence_key=row[5],
            raw_checksum=row[6],
            latency_ms=float(row[7]),
        )

    def recent_ticks(self, symbol: str, since: datetime) -> tuple[RealtimeTradeTick, ...]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select symbol, exchange_timestamp, received_at, source, price, volume,
                       trade_direction, sequence_key, raw_checksum, latency_ms
                from realtime_ticks
                where symbol = ? and exchange_timestamp >= ?
                order by exchange_timestamp asc
                """,
                (symbol, since.isoformat()),
            ).fetchall()
        return tuple(
            RealtimeTradeTick(
                symbol=row[0],
                exchange_timestamp=_parse_dt(row[1]),
                received_at=_parse_dt(row[2]),
                source=row[3],
                price=float(row[4]),
                volume=int(row[5]),
                trade_direction=row[6],
                sequence_key=row[7],
                raw_checksum=row[8],
                latency_ms=float(row[9]),
            )
            for row in rows
        )

    def recent_orderbooks(self, symbol: str, since: datetime) -> tuple[RealtimeOrderbookSnapshot, ...]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select symbol, exchange_timestamp, received_at, source, levels_json,
                       sequence_key, raw_checksum, latency_ms
                from realtime_orderbook
                where symbol = ? and exchange_timestamp >= ?
                order by exchange_timestamp asc
                """,
                (symbol, since.isoformat()),
            ).fetchall()
        from app.data.realtime_types import OrderbookLevel

        return tuple(
            RealtimeOrderbookSnapshot(
                symbol=row[0],
                exchange_timestamp=_parse_dt(row[1]),
                received_at=_parse_dt(row[2]),
                source=row[3],
                levels=tuple(OrderbookLevel(**item) for item in json.loads(row[4])),
                sequence_key=row[5],
                raw_checksum=row[6],
                latency_ms=float(row[7]),
            )
            for row in rows
        )

    def counts_since(self, symbol: str, since: datetime) -> tuple[int, int]:
        with closing(self._connect()) as conn:
            tick_count = conn.execute(
                "select count(*) from realtime_ticks where symbol = ? and exchange_timestamp >= ?",
                (symbol, since.isoformat()),
            ).fetchone()[0]
            orderbook_count = conn.execute(
                "select count(*) from realtime_orderbook where symbol = ? and exchange_timestamp >= ?",
                (symbol, since.isoformat()),
            ).fetchone()[0]
        return int(tick_count), int(orderbook_count)

    def build_latest_minute_bar(self, symbol: str, *, now: datetime | None = None) -> RealtimeMinuteBar | None:
        now = now or datetime.now(timezone.utc)
        minute_start = now.replace(second=0, microsecond=0)
        ticks = self.recent_ticks(symbol, minute_start)
        if not ticks:
            return None
        orderbook = self.latest_orderbook(symbol)
        prices = [tick.price for tick in ticks]
        volumes = [max(0, tick.volume) for tick in ticks]
        total_volume = sum(volumes)
        vwap = (
            sum(price * volume for price, volume in zip(prices, volumes, strict=True)) / total_volume
            if total_volume > 0
            else prices[-1]
        )
        mean = sum(prices) / len(prices)
        variance = sum((price - mean) ** 2 for price in prices) / max(1, len(prices))
        last_update_age_ms = max(0.0, (now - ticks[-1].received_at).total_seconds() * 1000)
        liquidity_score = min(1.0, total_volume / 100_000.0)
        bar = RealtimeMinuteBar(
            symbol=symbol,
            minute_start=minute_start,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=total_volume,
            vwap=vwap,
            trade_count=len(ticks),
            spread_bps=orderbook.spread_bps if orderbook else 0.0,
            orderbook_imbalance=orderbook.imbalance if orderbook else 0.0,
            liquidity_score=liquidity_score,
            volatility=variance**0.5,
            last_update_age_ms=last_update_age_ms,
            source_record_ids=tuple(tick.record_id for tick in ticks),
        )
        self.save_minute_bars((bar,))
        return bar

    def _insert_many(self, sql: str, rows: list[tuple[Any, ...]]) -> int:
        if not rows:
            return 0
        with closing(self._connect()) as conn:
            before = conn.total_changes
            conn.executemany(sql, rows)
            conn.commit()
            return int(conn.total_changes - before)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30)


def _parse_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def cutoff_ms(now: datetime, max_age_ms: int) -> datetime:
    return now - timedelta(milliseconds=max_age_ms)

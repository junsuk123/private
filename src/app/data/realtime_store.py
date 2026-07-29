from __future__ import annotations

import json
import os
import sqlite3
import time
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
    _last_prune_monotonic = 0.0

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
                create index if not exists idx_data_source_events_observed
                    on data_source_events(observed_at);
                """
            )
            conn.commit()
        now = time.monotonic()
        if now - self.__class__._last_prune_monotonic >= 3600:
            self.prune_operational_history()
            self.__class__._last_prune_monotonic = now

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
        inserted = self._insert_many(
            """
            insert or ignore into realtime_ticks
            (record_id, symbol, exchange_timestamp, received_at, source, price, volume,
             trade_direction, sequence_key, raw_checksum, latency_ms)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._save_source_events(
            [
                (
                    f"trade:{tick.record_id}",
                    "trade",
                    tick.symbol,
                    tick.exchange_timestamp.isoformat(),
                    tick.source,
                    json.dumps(
                        {"price": tick.price, "volume": tick.volume, "record_id": tick.record_id},
                        ensure_ascii=True,
                    ),
                )
                for tick in ticks
            ]
        )
        return inserted

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
        inserted = self._insert_many(
            """
            insert or ignore into realtime_orderbook
            (record_id, symbol, exchange_timestamp, received_at, source, best_bid, best_ask,
             spread_bps, total_bid_volume, total_ask_volume, imbalance, levels_json,
             sequence_key, raw_checksum, latency_ms)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._save_source_events(
            [
                (
                    f"orderbook:{snapshot.record_id}",
                    "orderbook",
                    snapshot.symbol,
                    snapshot.exchange_timestamp.isoformat(),
                    snapshot.source,
                    json.dumps(
                        {
                            "best_bid": snapshot.best_bid,
                            "best_ask": snapshot.best_ask,
                            "spread_bps": snapshot.spread_bps,
                            "record_id": snapshot.record_id,
                        },
                        ensure_ascii=True,
                    ),
                )
                for snapshot in snapshots
            ]
        )
        return inserted

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
        inserted = self._insert_many(
            """
            insert or replace into realtime_minute_bars
            (symbol, minute_start, open, high, low, close, volume, vwap, trade_count,
             spread_bps, orderbook_imbalance, liquidity_score, volatility, last_update_age_ms,
             source_record_ids_json)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._save_source_events(
            [
                (
                    f"minute_bar:{bar.symbol}:{bar.minute_start.isoformat()}",
                    "minute_bar",
                    bar.symbol,
                    bar.minute_start.isoformat(),
                    "realtime_aggregation",
                    json.dumps(
                        {"close": bar.close, "volume": bar.volume, "trade_count": bar.trade_count},
                        ensure_ascii=True,
                    ),
                )
                for bar in bars
            ]
        )
        return inserted

    def save_health(self, health: MarketDataHealth) -> None:
        # Health is a state sample, not a tick feed. A minute bucket prevents a
        # tight polling loop from producing millions of duplicate rows per day.
        checked_at = health.checked_at.replace(second=0, microsecond=0)
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
                    checked_at.isoformat(),
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

    def prune_operational_history(self, *, retention_hours: int | None = None) -> int:
        hours = (
            max(1, int(retention_hours))
            if retention_hours is not None
            else max(1, int(os.getenv("REALTIME_OPERATIONAL_RETENTION_HOURS", "24")))
        )
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with closing(self._connect()) as conn:
            before = conn.total_changes
            conn.execute("delete from market_data_health where checked_at < ?", (cutoff,))
            conn.execute("delete from data_source_events where observed_at < ?", (cutoff,))
            conn.commit()
            return int(conn.total_changes - before)

    def backfill_source_events(self, *, retention_hours: int | None = None) -> int:
        hours = max(
            1,
            int(
                retention_hours
                if retention_hours is not None
                else os.getenv("REALTIME_OPERATIONAL_RETENTION_HOURS", "24")
            ),
        )
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with closing(self._connect()) as conn:
            before = conn.total_changes
            conn.execute(
                """
                insert or ignore into data_source_events
                  (event_id, event_type, symbol, observed_at, source, payload_json)
                select 'trade:' || record_id, 'trade', symbol, exchange_timestamp, source,
                       json_object('price', price, 'volume', volume, 'record_id', record_id)
                from realtime_ticks where exchange_timestamp >= ?
                """,
                (cutoff,),
            )
            conn.execute(
                """
                insert or ignore into data_source_events
                  (event_id, event_type, symbol, observed_at, source, payload_json)
                select 'orderbook:' || record_id, 'orderbook', symbol, exchange_timestamp, source,
                       json_object('best_bid', best_bid, 'best_ask', best_ask,
                                   'spread_bps', spread_bps, 'record_id', record_id)
                from realtime_orderbook where exchange_timestamp >= ?
                """,
                (cutoff,),
            )
            conn.commit()
            return int(conn.total_changes - before)

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

    def active_symbols(self, since: datetime, *, limit: int = 200) -> tuple[str, ...]:
        """Symbols with at least one tick since the cutoff, most-recent first."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select symbol, max(received_at) as last_at
                from realtime_ticks
                where received_at >= ?
                group by symbol
                order by last_at desc
                limit ?
                """,
                (since.isoformat(), int(limit)),
            ).fetchall()
        return tuple(str(row[0]) for row in rows if row and row[0])

    def recent_ticks(
        self,
        symbol: str,
        since: datetime,
        *,
        until: datetime | None = None,
    ) -> tuple[RealtimeTradeTick, ...]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select symbol, exchange_timestamp, received_at, source, price, volume,
                       trade_direction, sequence_key, raw_checksum, latency_ms
                from realtime_ticks
                where symbol = ? and exchange_timestamp >= ?
                  and (? is null or exchange_timestamp <= ?)
                  and (? is null or received_at <= ?)
                order by exchange_timestamp asc, received_at asc
                """,
                (
                    symbol,
                    since.isoformat(),
                    until.isoformat() if until is not None else None,
                    until.isoformat() if until is not None else None,
                    until.isoformat() if until is not None else None,
                    until.isoformat() if until is not None else None,
                ),
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

    def recent_orderbooks(
        self,
        symbol: str,
        since: datetime,
        *,
        until: datetime | None = None,
    ) -> tuple[RealtimeOrderbookSnapshot, ...]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select symbol, exchange_timestamp, received_at, source, levels_json,
                       sequence_key, raw_checksum, latency_ms
                from realtime_orderbook
                where symbol = ? and exchange_timestamp >= ?
                  and (? is null or exchange_timestamp <= ?)
                  and (? is null or received_at <= ?)
                order by exchange_timestamp asc, received_at asc
                """,
                (
                    symbol,
                    since.isoformat(),
                    until.isoformat() if until is not None else None,
                    until.isoformat() if until is not None else None,
                    until.isoformat() if until is not None else None,
                    until.isoformat() if until is not None else None,
                ),
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

    def recent_minute_bars(
        self,
        symbol: str,
        since: datetime,
        *,
        limit: int = 120,
    ) -> tuple[RealtimeMinuteBar, ...]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select symbol, minute_start, open, high, low, close, volume, vwap,
                       trade_count, spread_bps, orderbook_imbalance, liquidity_score,
                       volatility, last_update_age_ms, source_record_ids_json
                from realtime_minute_bars
                where symbol = ? and minute_start >= ?
                order by minute_start desc
                limit ?
                """,
                (symbol, since.isoformat(), int(limit)),
            ).fetchall()
        bars = tuple(
            RealtimeMinuteBar(
                symbol=row[0],
                minute_start=_parse_dt(row[1]),
                open=float(row[2]),
                high=float(row[3]),
                low=float(row[4]),
                close=float(row[5]),
                volume=int(row[6]),
                vwap=float(row[7]),
                trade_count=int(row[8]),
                spread_bps=float(row[9]),
                orderbook_imbalance=float(row[10]),
                liquidity_score=float(row[11]),
                volatility=float(row[12]),
                last_update_age_ms=float(row[13]),
                source_record_ids=tuple(json.loads(row[14] or "[]")),
            )
            for row in rows
        )
        return tuple(reversed(bars))

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

    def _save_source_events(self, rows: list[tuple[Any, ...]]) -> int:
        if not rows:
            return 0
        return self._insert_many(
            """
            insert or ignore into data_source_events
              (event_id, event_type, symbol, observed_at, source, payload_json)
            values (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30)


def _parse_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def cutoff_ms(now: datetime, max_age_ms: int) -> datetime:
    return now - timedelta(milliseconds=max_age_ms)

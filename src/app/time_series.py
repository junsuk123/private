from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone
from typing import Iterable

from app.schemas.domain import (
    ClassifiedEvent,
    MacroMetricRecord,
    MarketSnapshot,
    RawSourceRecord,
    RealtimeExecution,
    RealtimeQuote,
    TimeSynchronizedTickerFrame,
)


def build_time_synchronized_frames(
    *,
    markets: tuple[MarketSnapshot, ...],
    events: tuple[ClassifiedEvent, ...] = (),
    raw_records: tuple[RawSourceRecord, ...] = (),
    macro_metrics: tuple[MacroMetricRecord, ...] = (),
    realtime_quotes: tuple[RealtimeQuote, ...] = (),
    realtime_executions: tuple[RealtimeExecution, ...] = (),
    bucket_minutes: int | None = None,
    event_lookback_minutes: int | None = None,
) -> tuple[TimeSynchronizedTickerFrame, ...]:
    """Fuse market, realtime, event, raw, and macro records on a shared time axis."""

    bucket_minutes = bucket_minutes or _env_int("TIME_SYNC_BUCKET_MINUTES", 15, minimum=1)
    event_lookback_minutes = event_lookback_minutes or _env_int(
        "TIME_SYNC_EVENT_LOOKBACK_MINUTES", 240, minimum=1
    )
    bucket_span = timedelta(minutes=bucket_minutes)
    event_lookback = timedelta(minutes=event_lookback_minutes)

    tickers = _ordered_tickers(markets, events, realtime_quotes, realtime_executions)
    if not tickers:
        return ()

    markets_by_ticker = _group_by_ticker(markets, lambda item: item.ticker)
    events_by_ticker = _group_events_by_ticker(events)
    quotes_by_ticker = _group_by_ticker(realtime_quotes, lambda item: item.ticker)
    executions_by_ticker = _group_by_ticker(realtime_executions, lambda item: item.ticker)
    raw_by_source_id = _raw_records_by_source_id(raw_records)
    macro_by_bucket = _group_by_bucket(macro_metrics, lambda item: item.observed_at, bucket_span)

    frames: list[TimeSynchronizedTickerFrame] = []
    for ticker in tickers:
        bucket_starts = _ticker_bucket_starts(
            markets_by_ticker.get(ticker, ()),
            events_by_ticker.get(ticker, ()),
            quotes_by_ticker.get(ticker, ()),
            executions_by_ticker.get(ticker, ()),
            bucket_span,
        )
        for bucket_start in bucket_starts:
            bucket_end = bucket_start + bucket_span
            market_snapshot = _latest_market_snapshot(markets_by_ticker.get(ticker, ()), bucket_start)
            frame_events = _events_in_window(
                events_by_ticker.get(ticker, ()),
                bucket_start - event_lookback,
                bucket_end,
            )
            frame_quotes = _items_in_bucket(
                quotes_by_ticker.get(ticker, ()),
                bucket_start,
                bucket_end,
                lambda item: item.observed_at,
            )
            frame_executions = _items_in_bucket(
                executions_by_ticker.get(ticker, ()),
                bucket_start,
                bucket_end,
                lambda item: item.executed_at,
            )
            frame_raw_records = _raw_records_for_events(frame_events, raw_by_source_id)
            frame_macros = macro_by_bucket.get(bucket_start, ())
            if not any((market_snapshot, frame_events, frame_quotes, frame_executions, frame_raw_records, frame_macros)):
                continue
            market = (
                market_snapshot.market
                if market_snapshot is not None
                else (frame_quotes[0].market if frame_quotes else frame_executions[0].market if frame_executions else "")
            )
            frames.append(
                TimeSynchronizedTickerFrame(
                    ticker=ticker,
                    market=market,
                    bucket_start=bucket_start,
                    bucket_end=bucket_end,
                    market_snapshot=market_snapshot,
                    realtime_quotes=frame_quotes,
                    realtime_executions=frame_executions,
                    events=frame_events,
                    raw_records=frame_raw_records,
                    macro_metrics=frame_macros,
                    impact_score=_impact_score(frame_events, frame_quotes, frame_executions, frame_macros),
                    data_source_ids=_source_ids(
                        market_snapshot,
                        frame_events,
                        frame_raw_records,
                        frame_macros,
                        frame_quotes,
                        frame_executions,
                    ),
                )
            )

    frames.sort(key=lambda frame: (frame.bucket_start, frame.ticker))
    return tuple(frames)


def add_time_frames_to_graph(graph: object, frames: tuple[TimeSynchronizedTickerFrame, ...]) -> None:
    for frame in frames:
        bucket_node = f"TimeBucket:{frame.bucket_start.isoformat()}"
        frame_node = frame.frame_id
        graph.add(bucket_node, "containsFrame", frame_node, "time-sync")
        graph.add(frame.ticker, "hasTimeFrame", frame_node, "time-sync")
        graph.add(frame_node, "observesTicker", frame.ticker, "time-sync")
        graph.add(frame_node, "hasImpactScore", f"ImpactScore:{frame.ticker}:{frame.impact_score:.3f}", "time-sync")
        if frame.market_snapshot is not None:
            snapshot_node = f"MarketSnapshot:{frame.ticker}:{frame.market_snapshot.source.retrieved_at.isoformat()}"
            graph.add(frame_node, "usesMarketSnapshot", snapshot_node, frame.market_snapshot.source.source_id)
        for quote in frame.realtime_quotes[-3:]:
            quote_node = f"RealtimeQuote:{quote.ticker}:{quote.observed_at.isoformat()}"
            graph.add(frame_node, "containsQuote", quote_node, quote.source.source_id if quote.source else "time-sync")
        for execution in frame.realtime_executions[-3:]:
            execution_node = f"RealtimeExecution:{execution.ticker}:{execution.executed_at.isoformat()}"
            graph.add(frame_node, "containsExecution", execution_node, execution.source.source_id if execution.source else "time-sync")
        for event in frame.events[:8]:
            event_node = f"{event.event_type}:{event.event_id}"
            graph.add(frame_node, "containsEvent", event_node, event.source.source_id)
            graph.add(event_node, "occursInTimeBucket", bucket_node, event.source.source_id)
        for raw in frame.raw_records[:4]:
            raw_node = f"RawSource:{raw.source.source_id or raw.source.raw_url or raw.source.retrieved_at.isoformat()}"
            graph.add(frame_node, "usesRawSource", raw_node, raw.source.source_id)
        for metric in frame.macro_metrics[:4]:
            metric_node = f"MacroMetric:{metric.name}:{metric.observed_at.isoformat()}"
            graph.add(frame_node, "hasMacroContext", metric_node, metric.source.source_id)


def _ordered_tickers(
    markets: tuple[MarketSnapshot, ...],
    events: tuple[ClassifiedEvent, ...],
    quotes: tuple[RealtimeQuote, ...],
    executions: tuple[RealtimeExecution, ...],
) -> tuple[str, ...]:
    tickers: list[str] = []
    tickers.extend(market.ticker for market in markets)
    tickers.extend(ticker for event in events for ticker in event.tickers)
    tickers.extend(quote.ticker for quote in quotes)
    tickers.extend(execution.ticker for execution in executions)
    return tuple(dict.fromkeys(ticker for ticker in tickers if ticker))


def _group_by_ticker(items: Iterable[object], key_fn: object) -> dict[str, tuple[object, ...]]:
    grouped: dict[str, list[object]] = {}
    for item in items:
        grouped.setdefault(str(key_fn(item)), []).append(item)
    return {key: tuple(value) for key, value in grouped.items()}


def _group_events_by_ticker(events: tuple[ClassifiedEvent, ...]) -> dict[str, tuple[ClassifiedEvent, ...]]:
    grouped: dict[str, list[ClassifiedEvent]] = {}
    for event in events:
        for ticker in event.tickers:
            grouped.setdefault(ticker, []).append(event)
    return {key: tuple(sorted(value, key=lambda item: item.event_date)) for key, value in grouped.items()}


def _group_by_bucket(
    items: tuple[object, ...],
    time_fn: object,
    bucket_span: timedelta,
) -> dict[datetime, tuple[object, ...]]:
    grouped: dict[datetime, list[object]] = {}
    for item in items:
        grouped.setdefault(_bucket_start(time_fn(item), bucket_span), []).append(item)
    return {key: tuple(value) for key, value in grouped.items()}


def _ticker_bucket_starts(
    markets: tuple[MarketSnapshot, ...],
    events: tuple[ClassifiedEvent, ...],
    quotes: tuple[RealtimeQuote, ...],
    executions: tuple[RealtimeExecution, ...],
    bucket_span: timedelta,
) -> tuple[datetime, ...]:
    times: list[datetime] = []
    times.extend(market.source.retrieved_at for market in markets)
    times.extend(event.event_date for event in events)
    times.extend(quote.observed_at for quote in quotes)
    times.extend(execution.executed_at for execution in executions)
    return tuple(sorted({_bucket_start(time, bucket_span) for time in times}))


def _bucket_start(value: datetime, bucket_span: timedelta) -> datetime:
    aware = _aware(value)
    seconds = int(aware.timestamp())
    bucket_seconds = int(bucket_span.total_seconds())
    return datetime.fromtimestamp(seconds - seconds % bucket_seconds, tz=timezone.utc)


def _latest_market_snapshot(
    markets: tuple[MarketSnapshot, ...],
    bucket_end: datetime,
) -> MarketSnapshot | None:
    candidates = [market for market in markets if _aware(market.source.retrieved_at) <= bucket_end]
    if not candidates:
        return None
    return max(candidates, key=lambda market: _aware(market.source.retrieved_at))


def _events_in_window(
    events: tuple[ClassifiedEvent, ...],
    start: datetime,
    end: datetime,
) -> tuple[ClassifiedEvent, ...]:
    return tuple(event for event in events if start <= _aware(event.event_date) < end)


def _items_in_bucket(
    items: tuple[object, ...],
    start: datetime,
    end: datetime,
    time_fn: object,
) -> tuple[object, ...]:
    return tuple(item for item in items if start <= _aware(time_fn(item)) < end)


def _raw_records_by_source_id(raw_records: tuple[RawSourceRecord, ...]) -> dict[str, tuple[RawSourceRecord, ...]]:
    grouped: dict[str, list[RawSourceRecord]] = {}
    for raw in raw_records:
        keys = [raw.source.source_id, raw.source.raw_url]
        for key in keys:
            if key:
                grouped.setdefault(str(key), []).append(raw)
    return {key: tuple(value) for key, value in grouped.items()}


def _raw_records_for_events(
    events: tuple[ClassifiedEvent, ...],
    raw_by_source_id: dict[str, tuple[RawSourceRecord, ...]],
) -> tuple[RawSourceRecord, ...]:
    selected: dict[str, RawSourceRecord] = {}
    for event in events:
        for key in (event.source.source_id, event.source.raw_url):
            if not key:
                continue
            for raw in raw_by_source_id.get(str(key), ()):
                selected[f"{raw.source.source_id}:{raw.source.retrieved_at.isoformat()}"] = raw
    return tuple(selected.values())


def _impact_score(
    events: tuple[ClassifiedEvent, ...],
    quotes: tuple[RealtimeQuote, ...],
    executions: tuple[RealtimeExecution, ...],
    macro_metrics: tuple[MacroMetricRecord, ...],
) -> float:
    event_score = sum(_event_direction(event) * max(0.25, event.classification_confidence) for event in events)
    quote_score = sum(float(quote.change_rate or 0.0) for quote in quotes[-5:])
    execution_score = math.log1p(sum(max(0, execution.quantity) for execution in executions)) * 0.05
    macro_score = sum(abs(metric.value) for metric in macro_metrics[:4]) * 0.005
    return round(event_score + quote_score + execution_score - macro_score, 6)


def _event_direction(event: ClassifiedEvent) -> float:
    sentiment = str(event.sentiment)
    if sentiment == "POSITIVE":
        return 1.0
    if sentiment == "NEGATIVE":
        return -1.0
    return 0.0


def _source_ids(
    market: MarketSnapshot | None,
    events: tuple[ClassifiedEvent, ...],
    raw_records: tuple[RawSourceRecord, ...],
    macro_metrics: tuple[MacroMetricRecord, ...],
    quotes: tuple[RealtimeQuote, ...],
    executions: tuple[RealtimeExecution, ...],
) -> tuple[str, ...]:
    values: list[str] = []
    if market and market.source.source_id:
        values.append(market.source.source_id)
    values.extend(str(event.source.source_id) for event in events if event.source.source_id)
    values.extend(str(raw.source.source_id) for raw in raw_records if raw.source.source_id)
    values.extend(str(metric.source.source_id) for metric in macro_metrics if metric.source.source_id)
    values.extend(str(quote.source.source_id) for quote in quotes if quote.source and quote.source.source_id)
    values.extend(str(execution.source.source_id) for execution in executions if execution.source and execution.source.source_id)
    return tuple(dict.fromkeys(values))


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _env_int(name: str, default: int, *, minimum: int) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default

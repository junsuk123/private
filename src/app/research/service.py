from __future__ import annotations

import json
import os
import hashlib
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from app.backtesting.accelerated_demo import load_krx_listed_universe, load_us_listed_universe
from app.data.classifier import classify_text_event
from app.data.llm_classifier import JsonEventLLMClassifier, build_event_llm_classifier_from_env
from app.data import (
    DynamicPageCollector,
    EcosMacroCollector,
    FredMacroCollector,
    HtmlResearchCollector,
    AlphaVantageDailyMarketDataCollector,
    OpenDartDisclosureCollector,
    RawArchive,
    RssNewsCollector,
    StooqMarketDataCollector,
    YahooChartMarketDataCollector,
    extract_focus_sections,
)
from app.schemas.domain import ClassifiedEvent, EventType, MacroMetricRecord, MarketSnapshot, RawSourceRecord, SourceMetadata


@dataclass(frozen=True)
class ResearchRunResult:
    events: tuple[ClassifiedEvent, ...]
    raw_records: tuple[RawSourceRecord, ...]
    market_snapshots: tuple[MarketSnapshot, ...]
    macro_metrics: tuple[MacroMetricRecord, ...]
    skipped_sources: tuple[str, ...]
    archived_paths: tuple[str, ...]
    diagnostics: dict[str, Any]


@dataclass
class _RetryJob:
    source_key: str
    action: Any
    attempts: int = 0


ProgressCallback = Callable[[str, int, int], None]


class ResearchService:
    def __init__(
        self,
        archive: RawArchive | None = None,
        progress_callback: ProgressCallback | None = None,
        llm_classifier: JsonEventLLMClassifier | None = None,
    ) -> None:
        self.archive = archive
        self.progress_callback = progress_callback
        self.llm_classifier = llm_classifier

    def run_from_config(self, path: Path) -> ResearchRunResult:
        config = json.loads(path.read_text(encoding="utf-8"))
        return self.run(config, base_dir=path.resolve().parent)

    def run(self, config: dict[str, Any], base_dir: Path | None = None) -> ResearchRunResult:
        known_tickers = dict(config.get("known_tickers", {}))
        universe_context = _load_listed_universe_context(config)
        if universe_context["include_in_known_tickers"]:
            known_tickers.update(universe_context["known_tickers"])
        llm_classifier = self.llm_classifier or build_event_llm_classifier_from_env()
        llm_items_per_source = _positive_int(os.getenv("LLM_EVENT_MAX_ITEMS_PER_SOURCE"), default=2)
        llm_items_remaining = _nonnegative_int(os.getenv("LLM_EVENT_MAX_ITEMS_PER_RUN"), default=3)
        retry_enabled = bool(config.get("retry_failed_sources", True))
        retry_attempts = max(1, int(config.get("retry_attempts", 3)))
        retry_backoff_ms = max(200, int(config.get("retry_backoff_ms", 700)))
        rss_fetch_articles = bool(config.get("rss_fetch_articles", False))
        rss_article_limit = _nonnegative_int(config.get("rss_article_fetch_limit_per_feed"), default=0)
        rss_article_limit_value = rss_article_limit if rss_article_limit > 0 else None
        events: list[ClassifiedEvent] = []
        raw_records: list[RawSourceRecord] = []
        market_snapshots: list[MarketSnapshot] = []
        macro_metrics: list[MacroMetricRecord] = []
        skipped: list[str] = []
        archived_paths: list[str] = []
        retry_queue: deque[_RetryJob] = deque()
        total_sources = _configured_source_count(config)
        completed_sources = 0

        def _mark(source_key: str) -> None:
            nonlocal completed_sources
            completed_sources += 1
            if self.progress_callback is not None:
                self.progress_callback(source_key, completed_sources, total_sources)

        if universe_context["enabled"]:
            source_key = "listed_universe:catalog"
            raw_records.append(_build_universe_record(universe_context))
            market_snapshots.extend(_build_universe_reference_snapshots(universe_context))
            _mark(source_key)

        rss_collector = RssNewsCollector()
        for feed_url in config.get("rss_feeds", []):
            feed_url = _resolve_source(feed_url, base_dir)
            source_key = f"rss:{feed_url}"

            def _action(url: str = str(feed_url)) -> None:
                nonlocal llm_items_remaining
                max_llm_items = min(llm_items_per_source, llm_items_remaining)
                rss_result = rss_collector.collect_with_articles(
                        str(feed_url),
                        known_tickers,
                        llm_classifier=llm_classifier if max_llm_items > 0 else None,
                        max_llm_items=max_llm_items,
                        fetch_articles=rss_fetch_articles,
                        article_limit=rss_article_limit_value,
                    )
                events.extend(rss_result.events)
                raw_records.extend(rss_result.raw_records)
                llm_items_remaining = max(0, llm_items_remaining - max_llm_items)

            if not _run_or_queue(source_key, _action, retry_enabled, retry_queue, skipped):
                _mark(source_key)
                continue
            _mark(source_key)

        html_collector = HtmlResearchCollector()
        for html_source in config.get("html_pages", []):
            page_url, title, event_type = _parse_html_source(html_source, base_dir)

            source_key = f"html:{page_url}"

            def _action(
                url: str = str(page_url),
                page_title: str | None = title,
                source_event_type: EventType = event_type,
            ) -> None:
                record = html_collector.collect(str(page_url))
                raw_records.append(record)
                events.append(
                    classify_text_event(
                        title=page_title or _title_from_payload(record.payload),
                        body=record.payload,
                        source=record.source,
                        event_type=source_event_type,
                        known_tickers=known_tickers,
                        llm_classifier=llm_classifier,
                    )
                )
                if self.archive is not None:
                    archived_paths.append(str(self.archive.write(record)))

            if not _run_or_queue(source_key, _action, retry_enabled, retry_queue, skipped):
                _mark(source_key)
                continue
            _mark(source_key)

        dynamic_collector = DynamicPageCollector()
        for dynamic_source in config.get("dynamic_pages", []):
            page_url, title, event_type, scroll_steps, wait_ms, timeout_ms = _parse_dynamic_source(dynamic_source, base_dir)

            source_key = f"dynamic:{page_url}"

            def _action(
                url: str = str(page_url),
                page_title: str | None = title,
                source_event_type: EventType = event_type,
                steps: int = scroll_steps,
                wait: int = wait_ms,
                timeout: int = timeout_ms,
            ) -> None:
                record = dynamic_collector.collect(
                    url,
                    scroll_steps=steps,
                    wait_ms=wait,
                    timeout_ms=timeout,
                )
                raw_records.append(record)
                sections = extract_focus_sections(record.payload)
                event_title = page_title or sections["headline"] or _title_from_payload(record.payload)
                event_body = sections["summary"]
                if sections["numeric_highlights"]:
                    event_body = f"{event_body} {' '.join(sections['numeric_highlights'])}".strip()
                events.append(
                    classify_text_event(
                        title=event_title,
                        body=event_body,
                        source=record.source,
                        event_type=source_event_type,
                        known_tickers=known_tickers,
                        llm_classifier=llm_classifier,
                    )
                )
                if self.archive is not None:
                    archived_paths.append(str(self.archive.write(record)))

            if not _run_or_queue(source_key, _action, retry_enabled, retry_queue, skipped):
                _mark(source_key)
                continue
            _mark(source_key)

        stooq = StooqMarketDataCollector()
        for item in config.get("stooq_symbols", []):
            source_key = f"stooq:{item.get('symbol')}"

            def _action(target: dict[str, Any] = item) -> None:
                market_snapshots.append(
                    stooq.collect_latest(
                        symbol=str(target["symbol"]),
                        ticker=str(target["ticker"]),
                        market=str(target["market"]),
                        company_name=str(target["company_name"]),
                        sector=str(target["sector"]),
                    )
                )

            if not _run_or_queue(source_key, _action, retry_enabled, retry_queue, skipped):
                _mark(source_key)
                continue
            _mark(source_key)

        yahoo_chart = YahooChartMarketDataCollector()
        for item in config.get("yahoo_chart_symbols", []):
            source_key = f"yahoo_chart:{item.get('symbol')}"

            def _action(target: dict[str, Any] = item) -> None:
                market_snapshots.append(
                    yahoo_chart.collect_latest(
                        symbol=str(target["symbol"]),
                        ticker=str(target["ticker"]),
                        market=str(target["market"]),
                        company_name=str(target["company_name"]),
                        sector=str(target["sector"]),
                    )
                )

            if not _run_or_queue(source_key, _action, retry_enabled, retry_queue, skipped):
                _mark(source_key)
                continue
            _mark(source_key)

        alpha_vantage = AlphaVantageDailyMarketDataCollector()
        for item in config.get("alpha_vantage_symbols", []):
            source_key = f"alpha_vantage:{item.get('symbol')}"

            def _action(target: dict[str, Any] = item) -> None:
                market_snapshots.append(
                    alpha_vantage.collect_latest(
                        symbol=str(target["symbol"]),
                        ticker=str(target["ticker"]),
                        market=str(target["market"]),
                        company_name=str(target["company_name"]),
                        sector=str(target["sector"]),
                    )
                )

            if not _run_or_queue(source_key, _action, retry_enabled, retry_queue, skipped):
                _mark(source_key)
                continue
            _mark(source_key)

        fred = FredMacroCollector()
        for item in config.get("fred_series", []):
            source_key = f"fred:{item.get('series_id')}"

            def _action(target: dict[str, Any] = item) -> None:
                metric = fred.collect_latest(str(target["series_id"]), str(target["name"]))
                if metric is None:
                    raise RuntimeError("missing_api_key_or_data")
                else:
                    macro_metrics.append(metric)

            if not _run_or_queue(source_key, _action, retry_enabled, retry_queue, skipped):
                _mark(source_key)
                continue
            _mark(source_key)

        ecos = EcosMacroCollector()
        for item in config.get("ecos_series", []):
            source_key = f"ecos:{item.get('statistic_code')}"

            def _action(target: dict[str, Any] = item) -> None:
                metric = ecos.collect_latest(
                    statistic_code=str(target["statistic_code"]),
                    item_code=str(target["item_code"]),
                    period=str(target["period"]),
                    start_date=str(target["start_date"]),
                    end_date=str(target["end_date"]),
                    name=str(target["name"]),
                )
                if metric is None:
                    raise RuntimeError("missing_api_key_or_data")
                else:
                    macro_metrics.append(metric)

            if not _run_or_queue(source_key, _action, retry_enabled, retry_queue, skipped):
                _mark(source_key)
                continue
            _mark(source_key)

        dart = OpenDartDisclosureCollector()
        for item in config.get("opendart_disclosures", []):
            source_key = f"opendart:{item.get('corp_code')}"

            def _action(target: dict[str, Any] = item) -> None:
                rows = dart.collect_disclosures(
                    corp_code=str(target["corp_code"]),
                    begin_date=str(target["begin_date"]),
                    end_date=str(target["end_date"]),
                    known_tickers=known_tickers,
                    llm_classifier=llm_classifier,
                )
                if not rows:
                    raise RuntimeError("missing_api_key_or_data")
                events.extend(rows)

            if not _run_or_queue(source_key, _action, retry_enabled, retry_queue, skipped):
                _mark(source_key)
                continue
            _mark(source_key)

        skipped.extend(
            _drain_retry_queue(
                retry_queue,
                max_attempts=retry_attempts,
                backoff_ms=retry_backoff_ms,
                progress_callback=self.progress_callback,
            )
        )

        deduped_events = _dedupe_events(tuple(events))
        deduped_raw_records = _dedupe_raw_records(tuple(raw_records))
        diagnostics = _build_diagnostics(
            deduped_events,
            deduped_raw_records,
            tuple(market_snapshots),
            tuple(macro_metrics),
            tuple(skipped),
        )
        diagnostics.update(universe_context["diagnostics"])
        diagnostics.update(_build_config_diagnostics(config))

        return ResearchRunResult(
            events=deduped_events,
            raw_records=deduped_raw_records,
            market_snapshots=tuple(market_snapshots),
            macro_metrics=tuple(macro_metrics),
            skipped_sources=tuple(skipped),
            archived_paths=tuple(archived_paths),
            diagnostics=diagnostics,
        )


def _resolve_source(value: Any, base_dir: Path | None) -> str:
    text = str(value)
    if text.startswith(("http://", "https://", "file://")):
        return text
    if base_dir is None:
        return text
    return (base_dir / text).resolve().as_uri()


def _configured_source_count(config: dict[str, Any]) -> int:
    source_count = sum(
        len(config.get(key, []))
        for key in (
            "rss_feeds",
            "html_pages",
            "dynamic_pages",
            "stooq_symbols",
            "yahoo_chart_symbols",
            "alpha_vantage_symbols",
            "fred_series",
            "ecos_series",
            "opendart_disclosures",
        )
    )
    if _listed_universe_enabled(config):
        source_count += 1
    return max(
        1,
        source_count,
    )


def _build_config_diagnostics(config: dict[str, Any]) -> dict[str, Any]:
    stooq_count = len(config.get("stooq_symbols", []))
    yahoo_chart_count = len(config.get("yahoo_chart_symbols", []))
    alpha_vantage_count = len(config.get("alpha_vantage_symbols", []))
    warnings: list[str] = []
    if stooq_count + yahoo_chart_count + alpha_vantage_count == 0:
        warnings.append(
            "No external stock chart source is configured; market snapshots will be limited to listed-universe reference records."
        )
    if yahoo_chart_count > 0:
        warnings.append(
            "Yahoo chart endpoints may be blocked by robots.txt in the built-in HTTP client."
        )
    return {
        "configured_source_counts": {
            "rss_feeds": len(config.get("rss_feeds", [])),
            "rss_fetch_articles": int(bool(config.get("rss_fetch_articles", False))),
            "rss_article_fetch_limit_per_feed": _nonnegative_int(
                config.get("rss_article_fetch_limit_per_feed"),
                default=0,
            ),
            "html_pages": len(config.get("html_pages", [])),
            "dynamic_pages": len(config.get("dynamic_pages", [])),
            "stooq_symbols": stooq_count,
            "yahoo_chart_symbols": yahoo_chart_count,
            "alpha_vantage_symbols": alpha_vantage_count,
            "fred_series": len(config.get("fred_series", [])),
            "ecos_series": len(config.get("ecos_series", [])),
            "opendart_disclosures": len(config.get("opendart_disclosures", [])),
        },
        "external_chart_sources_configured": stooq_count + yahoo_chart_count + alpha_vantage_count,
        "collection_warnings": warnings,
    }


def _listed_universe_enabled(config: dict[str, Any]) -> bool:
    value = config.get("listed_universe", {})
    if isinstance(value, dict):
        return bool(value.get("enabled", False))
    return bool(value)


def _load_listed_universe_context(config: dict[str, Any]) -> dict[str, Any]:
    options = config.get("listed_universe", {})
    if not isinstance(options, dict):
        options = {"enabled": bool(options)}
    enabled = bool(options.get("enabled", False))
    include_in_known_tickers = bool(options.get("include_in_known_tickers", True))
    markets = tuple(str(item).strip().upper() for item in options.get("markets", ["US", "KR"]) if str(item).strip())
    batch_size = _positive_int(
        os.getenv("RESEARCH_UNIVERSE_BATCH_SIZE") or options.get("batch_size"),
        default=500,
    )
    cursor_path = Path(str(options.get("cursor_path") or "data/research_universe_cursor.json"))

    groups: list[tuple[str, tuple[str, ...]]] = []
    if enabled and any(market in {"US", "OVERSEAS", "GLOBAL"} for market in markets):
        groups.append(("US", load_us_listed_universe(limit=None)))
    if enabled and any(market in {"KR", "KOREA", "DOMESTIC"} for market in markets):
        groups.append(("KR", load_krx_listed_universe(limit=None)))

    symbols = _interleave_universe_groups(tuple(group for _, group in groups))
    batch, start_index, next_index = _rotating_universe_batch(symbols, batch_size, cursor_path)
    known_tickers = _universe_known_tickers(symbols)

    return {
        "enabled": enabled,
        "include_in_known_tickers": include_in_known_tickers,
        "markets": markets,
        "symbols": symbols,
        "batch_symbols": batch,
        "known_tickers": known_tickers,
        "diagnostics": {
            "listed_universe_enabled": enabled,
            "listed_universe_markets": list(markets),
            "listed_universe_total": len(symbols),
            "listed_universe_known_tickers_count": len(known_tickers),
            "listed_universe_batch_size": len(batch),
            "listed_universe_batch_start": start_index,
            "listed_universe_batch_next": next_index,
        },
    }


def _build_universe_record(context: dict[str, Any]) -> RawSourceRecord:
    now = datetime.now(timezone.utc)
    payload = {
        "markets": context["markets"],
        "total_symbols": len(context["symbols"]),
        "batch_symbols": context["batch_symbols"],
        "batch_size": len(context["batch_symbols"]),
        "generated_at": now.isoformat(),
        "note": "Full listed stock universe is tracked; detailed collection should run as rotating batches.",
    }
    return RawSourceRecord(
        source=SourceMetadata(
            source_name="listed_universe_catalog",
            source_id=f"listed_universe:{now.date().isoformat()}",
            raw_url="local://listed-universe/catalog",
            retrieved_at=now,
        ),
        content_type="application/json",
        payload=json.dumps(payload, ensure_ascii=False, sort_keys=True),
    )


def _build_universe_reference_snapshots(context: dict[str, Any]) -> tuple[MarketSnapshot, ...]:
    now = datetime.now(timezone.utc)
    snapshots: list[MarketSnapshot] = []
    for symbol in context["batch_symbols"]:
        market = _market_for_universe_symbol(symbol)
        ticker = _display_ticker_for_universe_symbol(symbol)
        source_id = f"listed-universe-reference:{symbol}"
        snapshots.append(
            MarketSnapshot(
                ticker=ticker,
                market=market,
                company_name=symbol,
                sector=_sector_for_universe_symbol(symbol),
                last_price=_deterministic_price(symbol),
                average_daily_trading_value=_deterministic_trading_value(symbol),
                volatility_20d=_deterministic_volatility(symbol),
                source=SourceMetadata(
                    source_name="listed_universe_reference",
                    source_id=source_id,
                    raw_url=f"local://listed-universe/reference/{symbol}",
                    retrieved_at=now,
                ),
            )
        )
    return tuple(snapshots)


def _display_ticker_for_universe_symbol(symbol: str) -> str:
    if symbol.endswith((".KS", ".KQ")):
        return symbol.split(".", 1)[0]
    return symbol


def _market_for_universe_symbol(symbol: str) -> str:
    if symbol.endswith(".KS"):
        return "KOSPI"
    if symbol.endswith(".KQ"):
        return "KOSDAQ"
    return "US-LISTED"


def _sector_for_universe_symbol(symbol: str) -> str:
    buckets = ("Technology", "Healthcare", "Consumer", "Finance", "Industrial", "Energy", "Materials")
    return buckets[_hash_int(symbol, "sector") % len(buckets)]


def _deterministic_price(symbol: str) -> float:
    market = _market_for_universe_symbol(symbol)
    if market in {"KOSPI", "KOSDAQ"}:
        return float(1_000 + (_hash_int(symbol, "price") % 500_000))
    return round(5.0 + (_hash_int(symbol, "price") % 80_000) / 100.0, 2)


def _deterministic_trading_value(symbol: str) -> float:
    return float(5_000_000 + (_hash_int(symbol, "value") % 2_000_000_000))


def _deterministic_volatility(symbol: str) -> float:
    return round(0.015 + (_hash_int(symbol, "vol") % 9000) / 100_000.0, 4)


def _hash_int(symbol: str, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}:{symbol}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _interleave_universe_groups(groups: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    if not groups:
        return ()
    symbols: list[str] = []
    max_len = max((len(group) for group in groups), default=0)
    for index in range(max_len):
        for group in groups:
            if index < len(group):
                symbol = group[index].strip().upper()
                if symbol:
                    symbols.append(symbol)
    return tuple(dict.fromkeys(symbols))


def _universe_known_tickers(symbols: tuple[str, ...]) -> dict[str, str]:
    known: dict[str, str] = {}
    for symbol in symbols:
        known[symbol] = symbol
        if symbol.endswith((".KS", ".KQ")):
            known.setdefault(symbol.split(".", 1)[0], symbol)
    return known


def _rotating_universe_batch(symbols: tuple[str, ...], batch_size: int, cursor_path: Path) -> tuple[tuple[str, ...], int, int]:
    if not symbols:
        return (), 0, 0
    cursor_path.parent.mkdir(parents=True, exist_ok=True)
    start_index = _read_universe_cursor(cursor_path) % len(symbols)
    size = min(max(1, batch_size), len(symbols))
    end_index = start_index + size
    if end_index <= len(symbols):
        batch = symbols[start_index:end_index]
    else:
        batch = symbols[start_index:] + symbols[: end_index % len(symbols)]
    next_index = end_index % len(symbols)
    cursor_path.write_text(json.dumps({"next_index": next_index}, ensure_ascii=False), encoding="utf-8")
    return tuple(batch), start_index, next_index


def _read_universe_cursor(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    try:
        return max(0, int(data.get("next_index", 0)))
    except (TypeError, ValueError):
        return 0


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _nonnegative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _dedupe_events(events: tuple[ClassifiedEvent, ...]) -> tuple[ClassifiedEvent, ...]:
    seen: set[str] = set()
    deduped: list[ClassifiedEvent] = []
    for event in events:
        if event.event_id in seen:
            continue
        seen.add(event.event_id)
        deduped.append(event)
    return tuple(deduped)


def _dedupe_raw_records(records: tuple[RawSourceRecord, ...]) -> tuple[RawSourceRecord, ...]:
    seen: set[str] = set()
    deduped: list[RawSourceRecord] = []
    for record in records:
        key = record.source.raw_url or record.source.source_id or record.payload[:120]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return tuple(deduped)


def _parse_html_source(value: Any, base_dir: Path | None) -> tuple[str, str | None, EventType]:
    if isinstance(value, dict):
        return _resolve_source(value["url"], base_dir), value.get("title"), _parse_event_type(value.get("event_type"))
    return _resolve_source(value, base_dir), None, EventType.NEWS


def _parse_dynamic_source(value: Any, base_dir: Path | None) -> tuple[str, str | None, EventType, int, int, int]:
    if isinstance(value, dict):
        return (
            _resolve_source(value["url"], base_dir),
            value.get("title"),
            _parse_event_type(value.get("event_type")),
            int(value.get("scroll_steps", 6)),
            int(value.get("wait_ms", 700)),
            int(value.get("timeout_ms", 20_000)),
        )
    return _resolve_source(value, base_dir), None, EventType.NEWS, 6, 700, 20_000


def _parse_event_type(value: Any) -> EventType:
    if value is None:
        return EventType.NEWS
    try:
        return EventType(str(value).strip().upper())
    except ValueError:
        return EventType.NEWS


def _run_or_queue(
    source_key: str,
    action: Any,
    retry_enabled: bool,
    retry_queue: deque[_RetryJob],
    skipped: list[str],
) -> bool:
    try:
        action()
        return True
    except Exception as exc:
        if retry_enabled:
            retry_queue.append(_RetryJob(source_key=source_key, action=action, attempts=1))
            return False
        skipped.append(f"{source_key}:{exc}")
        return False


def _drain_retry_queue(
    retry_queue: deque[_RetryJob],
    max_attempts: int,
    backoff_ms: int,
    progress_callback: ProgressCallback | None = None,
) -> tuple[str, ...]:
    skipped: list[str] = []
    while retry_queue:
        job = retry_queue.popleft()
        if progress_callback is not None:
            progress_callback(f"retry:{job.source_key}:attempt {job.attempts}/{max_attempts}", 1, 1)
        try:
            delay_seconds = (backoff_ms / 1000.0) * max(1, job.attempts - 1)
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            job.action()
        except Exception as exc:
            if job.attempts < max_attempts:
                retry_queue.append(
                    _RetryJob(source_key=job.source_key, action=job.action, attempts=job.attempts + 1)
                )
            else:
                skipped.append(f"{job.source_key}:{exc}")
    return tuple(skipped)


def _title_from_payload(payload: str) -> str:
    compact = " ".join(payload.split())
    return compact[:90] or "Untitled research page"


def _build_diagnostics(
    events: tuple[ClassifiedEvent, ...],
    raw_records: tuple[RawSourceRecord, ...],
    market_snapshots: tuple[MarketSnapshot, ...],
    macro_metrics: tuple[MacroMetricRecord, ...],
    skipped: tuple[str, ...],
) -> dict[str, Any]:
    event_urls = tuple(event.source.raw_url or "" for event in events)
    raw_urls = tuple(record.source.raw_url or "" for record in raw_records)
    market_urls = tuple(snapshot.source.raw_url or "" for snapshot in market_snapshots)
    macro_urls = tuple(metric.source.raw_url or "" for metric in macro_metrics)
    all_urls = event_urls + raw_urls + market_urls + macro_urls
    live_urls = tuple(url for url in all_urls if _is_live_url(url))
    local_urls = tuple(url for url in all_urls if url.startswith(("file://", "local://")))

    event_dates = tuple(event.event_date for event in events)
    market_dates = tuple(snapshot.source.retrieved_at for snapshot in market_snapshots)
    macro_dates = tuple(metric.observed_at for metric in macro_metrics)
    latest_seen = _latest(event_dates + market_dates + macro_dates)

    return {
        "events_count": len(events),
        "raw_records_count": len(raw_records),
        "market_snapshots_count": len(market_snapshots),
        "macro_metrics_count": len(macro_metrics),
        "skipped_count": len(skipped),
        "live_source_count": len(live_urls),
        "local_source_count": len(local_urls),
        "live_data_present": bool(live_urls),
        "latest_observed_at": latest_seen.isoformat() if latest_seen else None,
        "source_names": sorted(
            {
                *(event.source.source_name for event in events),
                *(record.source.source_name for record in raw_records),
                *(snapshot.source.source_name for snapshot in market_snapshots),
                *(metric.source.source_name for metric in macro_metrics),
            }
        ),
        "per_ticker": _per_ticker_summary(events),
    }


def _is_live_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and parsed.netloc not in {"example.test"}


def _latest(values: tuple[datetime, ...]) -> datetime | None:
    return max(values) if values else None


def _per_ticker_summary(events: tuple[ClassifiedEvent, ...]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for event in events:
        for ticker in event.tickers:
            bucket = summary.setdefault(
                ticker,
                {
                    "events": 0,
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0,
                    "latest_event_at": None,
                    "live_source_urls": 0,
                },
            )
            bucket["events"] += 1
            bucket[event.sentiment.value.lower()] += 1
            if _is_live_url(event.source.raw_url or ""):
                bucket["live_source_urls"] += 1
            current_latest = bucket["latest_event_at"]
            if current_latest is None or event.event_date.isoformat() > current_latest:
                bucket["latest_event_at"] = event.event_date.isoformat()
    return summary

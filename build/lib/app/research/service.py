from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from app.data.classifier import classify_text_event
from app.data import (
    DynamicPageCollector,
    EcosMacroCollector,
    FredMacroCollector,
    HtmlResearchCollector,
    OpenDartDisclosureCollector,
    RawArchive,
    RssNewsCollector,
    StooqMarketDataCollector,
    YahooChartMarketDataCollector,
    extract_focus_sections,
)
from app.schemas.domain import ClassifiedEvent, MacroMetricRecord, MarketSnapshot, RawSourceRecord


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
    ) -> None:
        self.archive = archive
        self.progress_callback = progress_callback

    def run_from_config(self, path: Path) -> ResearchRunResult:
        config = json.loads(path.read_text(encoding="utf-8"))
        return self.run(config, base_dir=path.resolve().parent)

    def run(self, config: dict[str, Any], base_dir: Path | None = None) -> ResearchRunResult:
        known_tickers = dict(config.get("known_tickers", {}))
        retry_enabled = bool(config.get("retry_failed_sources", True))
        retry_attempts = max(1, int(config.get("retry_attempts", 3)))
        retry_backoff_ms = max(200, int(config.get("retry_backoff_ms", 700)))
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

        rss_collector = RssNewsCollector()
        for feed_url in config.get("rss_feeds", []):
            feed_url = _resolve_source(feed_url, base_dir)
            source_key = f"rss:{feed_url}"

            def _action(url: str = str(feed_url)) -> None:
                events.extend(rss_collector.collect(str(feed_url), known_tickers))

            if not _run_or_queue(source_key, _action, retry_enabled, retry_queue, skipped):
                _mark(source_key)
                continue
            _mark(source_key)

        html_collector = HtmlResearchCollector()
        for html_source in config.get("html_pages", []):
            page_url, title = _parse_html_source(html_source, base_dir)

            source_key = f"html:{page_url}"

            def _action(url: str = str(page_url), page_title: str | None = title) -> None:
                record = html_collector.collect(str(page_url))
                raw_records.append(record)
                events.append(
                    classify_text_event(
                        title=page_title or _title_from_payload(record.payload),
                        body=record.payload,
                        source=record.source,
                        known_tickers=known_tickers,
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
            page_url, title, scroll_steps, wait_ms, timeout_ms = _parse_dynamic_source(dynamic_source, base_dir)

            source_key = f"dynamic:{page_url}"

            def _action(
                url: str = str(page_url),
                page_title: str | None = title,
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
                        known_tickers=known_tickers,
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

        return ResearchRunResult(
            events=tuple(events),
            raw_records=tuple(raw_records),
            market_snapshots=tuple(market_snapshots),
            macro_metrics=tuple(macro_metrics),
            skipped_sources=tuple(skipped),
            archived_paths=tuple(archived_paths),
            diagnostics=_build_diagnostics(
                tuple(events),
                tuple(raw_records),
                tuple(market_snapshots),
                tuple(macro_metrics),
                tuple(skipped),
            ),
        )


def _resolve_source(value: Any, base_dir: Path | None) -> str:
    text = str(value)
    if text.startswith(("http://", "https://", "file://")):
        return text
    if base_dir is None:
        return text
    return (base_dir / text).resolve().as_uri()


def _configured_source_count(config: dict[str, Any]) -> int:
    return max(
        1,
        sum(
            len(config.get(key, []))
            for key in (
                "rss_feeds",
                "html_pages",
                "dynamic_pages",
                "stooq_symbols",
                "yahoo_chart_symbols",
                "fred_series",
                "ecos_series",
                "opendart_disclosures",
            )
        ),
    )


def _parse_html_source(value: Any, base_dir: Path | None) -> tuple[str, str | None]:
    if isinstance(value, dict):
        return _resolve_source(value["url"], base_dir), value.get("title")
    return _resolve_source(value, base_dir), None


def _parse_dynamic_source(value: Any, base_dir: Path | None) -> tuple[str, str | None, int, int, int]:
    if isinstance(value, dict):
        return (
            _resolve_source(value["url"], base_dir),
            value.get("title"),
            int(value.get("scroll_steps", 6)),
            int(value.get("wait_ms", 700)),
            int(value.get("timeout_ms", 20_000)),
        )
    return _resolve_source(value, base_dir), None, 6, 700, 20_000


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

from __future__ import annotations

import os
import hashlib
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Callable

from app.data.classifier import classify_text_event, source_now
from app.data.http_client import DataCollectionError, HttpClient
from app.schemas.domain import (
    ClassifiedEvent,
    EventType,
    MacroMetricRecord,
    MarketSnapshot,
    RawSourceRecord,
    SourceMetadata,
)


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            text = data.strip()
            if text:
                self._chunks.append(text)

    def text(self) -> str:
        return " ".join(self._chunks)


class HtmlResearchCollector:
    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient()

    def collect(self, url: str) -> RawSourceRecord:
        response = self.client.get_text(url)
        parser = TextExtractor()
        parser.feed(response.text)
        source = source_now("html", response.url, f"html:{_stable_id(response.url)}")
        return RawSourceRecord(source=source, content_type="text/html", payload=parser.text())

class DynamicPageCollector:
    def __init__(
        self,
        client: HttpClient | None = None,
        renderer: Callable[[str, int, int, int], tuple[str, str]] | None = None,
    ) -> None:
        self.client = client or HttpClient()
        self.renderer = renderer or _render_dynamic_page

    def collect(
        self,
        url: str,
        scroll_steps: int = 6,
        wait_ms: int = 700,
        timeout_ms: int = 20_000,
    ) -> RawSourceRecord:
        if url.startswith("file://"):
            response = self.client.get_text(url)
            html_text = response.text
            final_url = response.url
        else:
            final_url, html_text = self.renderer(url, scroll_steps, wait_ms, timeout_ms)

        parser = TextExtractor()
        parser.feed(html_text)
        source = source_now("dynamic_html", final_url, f"dynamic:{_stable_id(final_url)}")
        return RawSourceRecord(source=source, content_type="text/html", payload=parser.text())


def extract_focus_sections(text: str) -> dict[str, Any]:
    compact = " ".join(text.split())
    if not compact:
        return {
            "headline": "",
            "summary": "",
            "numeric_highlights": (),
        }

    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", compact) if item.strip()]
    headline = sentences[0] if sentences else compact[:160]
    summary = " ".join(sentences[:3]) if sentences else compact[:300]
    numeric = [sentence for sentence in sentences if re.search(r"\d|%|\$|USD|KRW|bps", sentence, re.IGNORECASE)]

    return {
        "headline": headline[:180],
        "summary": summary[:700],
        "numeric_highlights": tuple(numeric[:5]),
    }


class RssNewsCollector:
    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient()

    def collect(
        self,
        feed_url: str,
        known_tickers: dict[str, str] | None = None,
    ) -> tuple[ClassifiedEvent, ...]:
        response = self.client.get_text(feed_url)
        root = ET.fromstring(response.text)
        events: list[ClassifiedEvent] = []

        for item in root.findall(".//item"):
            title = _xml_text(item, "title")
            description = _xml_text(item, "description")
            link = _xml_text(item, "link") or response.url
            source = source_now("rss", link, f"rss:{_stable_id(link + title)}")
            events.append(
                classify_text_event(
                    title=title,
                    body=description,
                    source=source,
                    event_type=EventType.NEWS,
                    known_tickers=known_tickers,
                    event_date=_parse_rss_date(_xml_text(item, "pubDate")),
                )
            )

        return tuple(events)


class StooqMarketDataCollector:
    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient()

    def collect_latest(
        self,
        symbol: str,
        ticker: str,
        market: str,
        company_name: str,
        sector: str,
    ) -> MarketSnapshot:
        rows = self.client.get_csv_rows("https://stooq.com/q/d/l/", {"s": symbol, "i": "d"})
        if not rows:
            raise DataCollectionError(f"no market data for {symbol}")

        recent = rows[-20:]
        closes = [float(row["Close"]) for row in recent if row.get("Close")]
        volumes = [float(row["Volume"]) for row in recent if row.get("Volume")]
        if not closes:
            raise DataCollectionError(f"missing close prices for {symbol}")
        returns = [
            (closes[index] - closes[index - 1]) / closes[index - 1]
            for index in range(1, len(closes))
            if closes[index - 1]
        ]
        volatility = _stddev(returns)
        average_value = sum(close * volume for close, volume in zip(closes, volumes)) / max(1, len(closes))
        source = source_now("stooq", f"https://stooq.com/q/d/l/?s={symbol}&i=d", f"stooq:{symbol}")
        return MarketSnapshot(
            ticker=ticker,
            market=market,
            company_name=company_name,
            sector=sector,
            last_price=closes[-1],
            average_daily_trading_value=average_value,
            volatility_20d=volatility,
            source=source,
        )


class YahooChartMarketDataCollector:
    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient()

    def collect_latest(
        self,
        symbol: str,
        ticker: str,
        market: str,
        company_name: str,
        sector: str,
    ) -> MarketSnapshot:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        payload = self.client.get_json(url, {"range": "1mo", "interval": "1d"})
        result = payload.get("chart", {}).get("result", [])
        if not result:
            raise DataCollectionError(f"no yahoo chart data for {symbol}")

        chart = result[0]
        meta = chart.get("meta", {})
        quote = chart.get("indicators", {}).get("quote", [{}])[0]
        closes = _numbers(quote.get("close", []))
        volumes = _numbers(quote.get("volume", []))
        if not closes:
            price = meta.get("regularMarketPrice")
            if price is None:
                raise DataCollectionError(f"missing yahoo close prices for {symbol}")
            closes = [float(price)]
        returns = [
            (closes[index] - closes[index - 1]) / closes[index - 1]
            for index in range(1, len(closes))
            if closes[index - 1]
        ]
        average_value = sum(close * volume for close, volume in zip(closes, volumes)) / max(1, len(closes))
        if average_value <= 0 and meta.get("regularMarketVolume"):
            average_value = closes[-1] * float(meta["regularMarketVolume"])

        source = source_now("yahoo_chart", url, f"yahoo-chart:{_stable_id(symbol)}")
        return MarketSnapshot(
            ticker=ticker,
            market=market,
            company_name=company_name,
            sector=sector,
            last_price=closes[-1],
            average_daily_trading_value=average_value,
            volatility_20d=_stddev(returns),
            source=source,
        )


class FredMacroCollector:
    def __init__(self, api_key: str | None = None, client: HttpClient | None = None) -> None:
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        self.client = client or HttpClient()

    def collect_latest(self, series_id: str, name: str) -> MacroMetricRecord | None:
        if not self.api_key:
            return None
        payload = self.client.get_json(
            "https://api.stlouisfed.org/fred/series/observations",
            {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 1,
            },
        )
        observations = payload.get("observations", [])
        if not observations:
            return None
        item = observations[0]
        source = source_now("fred", "https://api.stlouisfed.org/fred/series/observations", f"fred:{series_id}")
        return MacroMetricRecord(
            name=name,
            value=float(item["value"]),
            observed_at=datetime.fromisoformat(item["date"]).replace(tzinfo=timezone.utc),
            source=source,
        )


class EcosMacroCollector:
    def __init__(self, api_key: str | None = None, client: HttpClient | None = None) -> None:
        self.api_key = api_key or os.getenv("ECOS_API_KEY")
        self.client = client or HttpClient()

    def collect_latest(
        self,
        statistic_code: str,
        item_code: str,
        period: str,
        start_date: str,
        end_date: str,
        name: str,
    ) -> MacroMetricRecord | None:
        if not self.api_key:
            return None
        url = (
            "https://ecos.bok.or.kr/api/StatisticSearch/"
            f"{self.api_key}/json/kr/1/1/{statistic_code}/{period}/{start_date}/{end_date}/{item_code}"
        )
        payload = self.client.get_json(url)
        rows = payload.get("StatisticSearch", {}).get("row", [])
        if not rows:
            return None
        item = rows[-1]
        source = source_now("ecos", url, f"ecos:{statistic_code}:{item_code}")
        return MacroMetricRecord(
            name=name,
            value=float(item["DATA_VALUE"]),
            observed_at=_parse_ecos_date(item["TIME"]),
            source=source,
        )


class OpenDartDisclosureCollector:
    def __init__(self, api_key: str | None = None, client: HttpClient | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENDART_API_KEY")
        self.client = client or HttpClient()

    def collect_disclosures(
        self,
        corp_code: str,
        begin_date: str,
        end_date: str,
        known_tickers: dict[str, str] | None = None,
    ) -> tuple[ClassifiedEvent, ...]:
        if not self.api_key:
            return ()
        payload = self.client.get_json(
            "https://opendart.fss.or.kr/api/list.json",
            {
                "crtfc_key": self.api_key,
                "corp_code": corp_code,
                "bgn_de": begin_date,
                "end_de": end_date,
                "page_count": 100,
            },
        )
        rows: list[dict[str, Any]] = payload.get("list", [])
        events: list[ClassifiedEvent] = []
        for row in rows:
            report = row.get("report_nm", "")
            company = row.get("corp_name", "")
            receipt = row.get("rcept_no", "")
            source = SourceMetadata(
                source_name="opendart",
                retrieved_at=datetime.now(timezone.utc),
                raw_url=f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt}",
                source_id=f"opendart:{receipt}",
            )
            events.append(
                classify_text_event(
                    title=report,
                    body=f"{company} {report}",
                    source=source,
                    event_type=EventType.DISCLOSURE,
                    known_tickers=known_tickers,
                    event_date=_parse_yyyymmdd(row.get("rcept_dt", "")),
                )
            )
        return tuple(events)


def _xml_text(item: ET.Element, tag: str) -> str:
    node = item.find(tag)
    return (node.text or "").strip() if node is not None else ""


def _parse_rss_date(value: str) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        from email.utils import parsedate_to_datetime

        return parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


def _parse_yyyymmdd(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def _parse_ecos_date(value: str) -> datetime:
    for fmt in ("%Y%m%d", "%Y%m", "%Y"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.now(timezone.utc)


def _render_dynamic_page(url: str, scroll_steps: int, wait_ms: int, timeout_ms: int) -> tuple[str, str]:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - optional dependency at runtime
        raise DataCollectionError(
            "playwright is required for dynamic_pages. Install with: pip install playwright and run playwright install"
        ) from exc

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_timeout(max(0, wait_ms))
            for _ in range(max(0, scroll_steps)):
                page.mouse.wheel(0, 5000)
                page.wait_for_timeout(max(0, wait_ms))
            html_text = page.content()
            final_url = page.url
            browser.close()
            return final_url, html_text
    except PlaywrightTimeoutError as exc:
        raise DataCollectionError(f"dynamic page timeout for {url}: {exc}") from exc


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return variance**0.5


def _stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _numbers(values: list[Any]) -> list[float]:
    return [float(value) for value in values if value is not None]

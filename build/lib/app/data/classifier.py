from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

from app.schemas.domain import ClassifiedEvent, EventType, SentimentDirection, SourceMetadata

POSITIVE_KEYWORDS = (
    "beat",
    "growth",
    "surge",
    "record",
    "profit",
    "upgrade",
    "buyback",
    "dividend",
    "contract",
    "수주",
    "성장",
    "흑자",
    "최대",
)
NEGATIVE_KEYWORDS = (
    "miss",
    "decline",
    "loss",
    "downgrade",
    "risk",
    "pressure",
    "selloff",
    "crash",
    "lawsuit",
    "recall",
    "default",
    "cut",
    "적자",
    "하락",
    "소송",
    "리콜",
)
SECTOR_KEYWORDS = {
    "Semiconductor": ("semiconductor", "chip", "memory", "hbm", "반도체", "메모리"),
    "Battery": ("battery", "ev", "lithium", "배터리", "전기차"),
    "Finance": ("bank", "insurance", "brokerage", "은행", "증권", "보험"),
}


def classify_text_event(
    title: str,
    body: str,
    source: SourceMetadata,
    event_type: EventType = EventType.NEWS,
    known_tickers: dict[str, str] | None = None,
    event_date: datetime | None = None,
) -> ClassifiedEvent:
    known_tickers = known_tickers or {}
    text = f"{title}\n{body}"
    lower = text.lower()
    sentiment = _sentiment(lower)
    aliases = _ticker_aliases(known_tickers)
    extracted = _extract_tickers(text)
    tickers_from_symbols = {
        aliases[token]
        for token in extracted
        if token in aliases
    }
    tickers_from_known_keys = {
        ticker
        for ticker in known_tickers
        if _contains_symbol(text, ticker)
    }
    tickers_from_company_names = {
        ticker
        for ticker, company_name in known_tickers.items()
        if company_name.lower() in lower
    }
    tickers = tuple(
        sorted(
            tickers_from_symbols
            | tickers_from_known_keys
            | tickers_from_company_names
        )
    )
    companies = tuple(dict.fromkeys(known_tickers[ticker] for ticker in tickers))
    sectors = tuple(
        sector
        for sector, keywords in SECTOR_KEYWORDS.items()
        if any(keyword in lower for keyword in keywords)
    )

    event_id = hashlib.sha256(f"{source.source_id}:{title}:{body}".encode("utf-8")).hexdigest()[:16]
    return ClassifiedEvent(
        event_id=event_id,
        event_type=event_type,
        title=title.strip(),
        summary=_summarize(body),
        companies=companies,
        tickers=tickers,
        sectors=sectors,
        sentiment=sentiment,
        event_date=event_date or source.retrieved_at,
        source=source,
    )


def _extract_tickers(text: str) -> set[str]:
    pattern = r"\b\d{6}(?:\.[A-Z]{1,4})?\b|\b[A-Z][A-Z0-9]{0,6}(?:[.-][A-Z0-9]{1,4})?\b"
    return {match.upper() for match in re.findall(pattern, text)}


def _ticker_aliases(known_tickers: dict[str, str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for ticker in known_tickers:
        upper = ticker.upper()
        aliases.setdefault(upper, ticker)
        aliases.setdefault(re.sub(r"[^A-Z0-9]", "", upper), ticker)
    return aliases


def _contains_symbol(text: str, symbol: str) -> bool:
    escaped = re.escape(symbol)
    pattern = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _sentiment(lower_text: str) -> SentimentDirection:
    positive = sum(1 for keyword in POSITIVE_KEYWORDS if keyword in lower_text)
    negative = sum(1 for keyword in NEGATIVE_KEYWORDS if keyword in lower_text)
    if positive > negative:
        return SentimentDirection.POSITIVE
    if negative > positive:
        return SentimentDirection.NEGATIVE
    return SentimentDirection.NEUTRAL


def _summarize(body: str) -> str:
    compact = " ".join(body.split())
    return compact[:280]


def source_now(source_name: str, raw_url: str | None = None, source_id: str | None = None) -> SourceMetadata:
    return SourceMetadata(
        source_name=source_name,
        retrieved_at=datetime.now(timezone.utc),
        raw_url=raw_url,
        source_id=source_id,
    )

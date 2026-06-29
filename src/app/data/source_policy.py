from __future__ import annotations

from urllib.parse import urlparse

from app.schemas.domain import SourceMetadata


SOURCE_TRUST_LEVELS: dict[str, int] = {
    "broker_api": 5,
    "official_exchange_api": 5,
    "official_disclosure_api": 5,
    "official_macro_api": 4,
    "licensed_market_api": 3,
    "public_api": 3,
    "rss_news": 2,
    "static_public_html": 2,
    "dynamic_page": 1,
    "unofficial_chart_endpoint": 1,
    "synthetic": 0,
    "sample": 0,
    "unknown": 0,
}


def infer_source_type(source_name: str, raw_url: str | None = None) -> str:
    lowered = source_name.lower()
    host = urlparse(raw_url or "").netloc.lower()
    haystack = f"{lowered} {host}"
    if any(token in haystack for token in ("synthetic", "simulation", "simulated")):
        return "synthetic"
    if "sample" in haystack or "demo" in haystack:
        return "sample"
    if any(token in haystack for token in ("kis", "korea investment", "broker")):
        return "broker_api"
    if any(token in haystack for token in ("kind.krx", "krx", "nasdaq", "nyse")):
        return "official_exchange_api"
    if any(token in haystack for token in ("dart", "sec.gov", "disclosure")):
        return "official_disclosure_api"
    if any(token in haystack for token in ("fred", "bok", "ecos", "macro")):
        return "official_macro_api"
    if any(token in haystack for token in ("alpha_vantage", "finnhub", "polygon", "licensed")):
        return "licensed_market_api"
    if "rss" in haystack:
        return "rss_news"
    if raw_url:
        return "public_api" if "api" in haystack else "static_public_html"
    return "unknown"


def default_trust_level(source_type: str) -> int:
    return SOURCE_TRUST_LEVELS.get(source_type, 0)


def compute_quality_score(metadata: SourceMetadata, missing_ratio: float = 0.0) -> float:
    missing = max(0.0, min(1.0, missing_ratio))
    trust = metadata.trust_level if metadata.trust_level > 0 else default_trust_level(metadata.source_type)
    score = trust / 5.0
    if metadata.is_synthetic:
        score *= 0.0
    if metadata.is_backfilled:
        score *= 0.85
    if metadata.is_delayed:
        score *= 0.90
    if metadata.latency_sec is not None and metadata.latency_sec > 60:
        score *= max(0.35, 1.0 - min(metadata.latency_sec, 3600.0) / 7200.0)
    score *= 1.0 - missing * 0.75
    return round(max(0.0, min(1.0, score)), 4)


def is_allowed_for_live_decision(
    metadata: SourceMetadata,
    min_trust: int,
    min_quality: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    source_type = metadata.source_type or infer_source_type(metadata.source_name, metadata.raw_url)
    trust = metadata.trust_level if metadata.trust_level > 0 else default_trust_level(source_type)
    quality = metadata.quality_score if metadata.quality_score > 0 else compute_quality_score(metadata)
    if metadata.is_synthetic or source_type in {"synthetic", "sample"}:
        reasons.append("synthetic_data_blocked")
    if source_type == "unknown":
        reasons.append("unknown_source_check")
    if trust < min_trust:
        reasons.append("source_trust_check")
    if quality < min_quality:
        reasons.append("data_quality_check")
    return not reasons, reasons


def is_allowed_for_live_buy_market_data(
    metadata: SourceMetadata,
    *,
    max_age_seconds: float,
    min_quality: float,
    now: object | None = None,
) -> tuple[bool, list[str]]:
    from datetime import datetime, timezone

    current = now if isinstance(now, datetime) else datetime.now(timezone.utc)
    reasons: list[str] = []
    allowed, base_reasons = is_allowed_for_live_decision(metadata, 5, min_quality)
    reasons.extend(reason.upper() for reason in base_reasons)
    observed_at = metadata.observed_at or metadata.retrieved_at
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    age = max(0.0, (current - observed_at).total_seconds())
    if age > max_age_seconds:
        reasons.append("MARKET_DATA_STALE")
    if not metadata.is_realtime:
        reasons.append("MARKET_DATA_NOT_REALTIME")
    if metadata.is_delayed:
        reasons.append("DELAYED_MARKET_DATA_BLOCKED")
    if metadata.is_backfilled:
        reasons.append("BACKFILLED_MARKET_DATA_BLOCKED")
    return allowed and not reasons, list(dict.fromkeys(reasons))

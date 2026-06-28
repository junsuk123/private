from __future__ import annotations

import hashlib

from app.schemas.domain import IndicatorSnapshot, MarketSnapshot


SAMPLE_INDICATOR_SOURCE_PREFIXES = ("reference:", "sample-indicator:", "demo-indicator:")


def build_sample_indicators(markets: tuple[MarketSnapshot, ...]) -> dict[str, IndicatorSnapshot]:
    """Demo/offline indicator fixture.

    This intentionally produces deterministic reference values for tests and
    local demos. Production realtime/live-readiness paths must use
    `build_trusted_indicators_from_markets` or measured indicator records
    instead.
    """
    values = {
        "005930": IndicatorSnapshot(
            ticker="005930",
            revenue_growth=0.12,
            operating_income_growth=0.28,
            operating_margin=0.19,
            roe=0.11,
            debt_ratio=0.31,
            per=16.8,
            pbr=1.35,
            rsi_14d=57.0,
            volume_ratio=1.18,
            macro_risk_score=0.38,
            source_ids=("sample-indicator:financial-005930", "sample-indicator:market-005930", "sample-indicator:macro-kr"),
        ),
        "000660": IndicatorSnapshot(
            ticker="000660",
            revenue_growth=0.31,
            operating_income_growth=0.47,
            operating_margin=0.22,
            roe=0.16,
            debt_ratio=0.44,
            per=22.4,
            pbr=2.1,
            rsi_14d=64.0,
            volume_ratio=1.42,
            macro_risk_score=0.43,
            source_ids=("sample-indicator:financial-000660", "sample-indicator:market-000660", "sample-indicator:macro-kr"),
        ),
    }
    indicators: dict[str, IndicatorSnapshot] = {}
    for market in markets:
        if market.ticker in values:
            indicators[market.ticker] = values[market.ticker]
        else:
            indicators[market.ticker] = _reference_indicator(market)
    return indicators


def build_trusted_indicators_from_markets(markets: tuple[MarketSnapshot, ...]) -> dict[str, IndicatorSnapshot]:
    indicators: dict[str, IndicatorSnapshot] = {}
    for market in markets:
        if market.source.is_synthetic or market.source.quality_score <= 0:
            continue
        source_id = market.source.source_id or f"market:{market.ticker}"
        indicators[market.ticker] = IndicatorSnapshot(
            ticker=market.ticker,
            revenue_growth=None,
            operating_income_growth=None,
            operating_margin=None,
            roe=None,
            debt_ratio=None,
            per=None,
            pbr=None,
            rsi_14d=None,
            volume_ratio=None,
            macro_risk_score=min(0.92, max(0.0, float(market.volatility_20d))),
            source_ids=(source_id,),
        )
    return indicators


def is_sample_or_hash_indicator(indicator: IndicatorSnapshot) -> bool:
    return any(
        str(source_id).startswith(SAMPLE_INDICATOR_SOURCE_PREFIXES)
        for source_id in indicator.source_ids
    )


def filter_trusted_indicators(indicators: dict[str, IndicatorSnapshot]) -> dict[str, IndicatorSnapshot]:
    return {
        ticker: indicator
        for ticker, indicator in indicators.items()
        if not is_sample_or_hash_indicator(indicator)
    }


def _reference_indicator(market: MarketSnapshot) -> IndicatorSnapshot:
    seed = _hash_int(market.ticker)
    revenue_growth = -0.04 + ((seed >> 3) % 3200) / 10_000.0
    operating_income_growth = -0.08 + ((seed >> 7) % 4200) / 10_000.0
    operating_margin = 0.03 + ((seed >> 11) % 2800) / 10_000.0
    per = 7.0 + ((seed >> 17) % 3600) / 100.0
    pbr = 0.5 + ((seed >> 23) % 500) / 100.0
    rsi = 25.0 + ((seed >> 29) % 5600) / 100.0
    volume_ratio = 0.45 + ((seed >> 31) % 180) / 100.0
    macro_risk = 0.18 + ((seed >> 37) % 50) / 100.0
    return IndicatorSnapshot(
        ticker=market.ticker,
        revenue_growth=round(revenue_growth, 4),
        operating_income_growth=round(operating_income_growth, 4),
        operating_margin=round(operating_margin, 4),
        roe=round(0.03 + ((seed >> 13) % 1800) / 10_000.0, 4),
        debt_ratio=round(0.15 + ((seed >> 19) % 6500) / 10_000.0, 4),
        per=round(per, 2),
        pbr=round(pbr, 2),
        rsi_14d=round(rsi, 2),
        volume_ratio=round(volume_ratio, 2),
        macro_risk_score=round(min(0.92, macro_risk), 4),
        source_ids=(f"reference:{market.ticker}",),
    )


def _hash_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)

from __future__ import annotations

import hashlib

from app.schemas.domain import IndicatorSnapshot, MarketSnapshot


def build_sample_indicators(markets: tuple[MarketSnapshot, ...]) -> dict[str, IndicatorSnapshot]:
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
            source_ids=("financial-005930", "market-005930", "macro-kr"),
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
            source_ids=("financial-000660", "market-000660", "macro-kr"),
        ),
    }
    indicators: dict[str, IndicatorSnapshot] = {}
    for market in markets:
        if market.ticker in values:
            indicators[market.ticker] = values[market.ticker]
        else:
            indicators[market.ticker] = _reference_indicator(market)
    return indicators


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
        source_ids=(market.source.source_id or f"reference:{market.ticker}",),
    )


def _hash_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)

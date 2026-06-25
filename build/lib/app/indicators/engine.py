from __future__ import annotations

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
    return {market.ticker: values[market.ticker] for market in markets if market.ticker in values}

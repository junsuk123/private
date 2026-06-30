from __future__ import annotations

from datetime import datetime, timezone

from app.schemas.domain import AccountSnapshot, MarketSnapshot, SourceMetadata
from app.market_affordability import (
    filter_markets_affordable_for_account,
    market_currency,
)


def test_filters_domestic_and_overseas_by_currency_cash() -> None:
    now = datetime.now(timezone.utc)
    source = SourceMetadata(
        source_name="KIS broker quote",
        retrieved_at=now,
        source_type="broker_api",
        trust_level=5,
        observed_at=now,
        is_realtime=True,
        quality_score=1.0,
    )
    markets = (
        MarketSnapshot("000001", "KOSPI", "Affordable KR", "Technology", 4_000.0, 10_000_000, 0.02, source),
        MarketSnapshot("005930", "KOSPI", "Expensive KR", "Technology", 70_000.0, 10_000_000, 0.02, source),
        MarketSnapshot("PENNY", "NASDAQ", "Affordable US", "Technology", 2.5, 10_000_000, 0.02, source),
        MarketSnapshot("MSFT", "NASDAQ", "Microsoft", "Technology", 367.6, 10_000_000, 0.02, source),
    )
    account = AccountSnapshot(
        cash=5_000.0,
        holdings=(),
        cash_by_currency={"KRW": 5_000.0, "USD": 3.22},
        cash_equivalent_krw=9_963.0,
    )

    filtered, diagnostics = filter_markets_affordable_for_account(markets, account)

    assert tuple(market.ticker for market in filtered) == ("000001", "PENNY")
    assert {item.ticker: item.reason for item in diagnostics if not item.affordable} == {
        "005930": "INSUFFICIENT_CASH_FOR_ONE_SHARE",
        "MSFT": "INSUFFICIENT_CASH_FOR_ONE_SHARE",
    }
    assert market_currency(markets[0]) == "KRW"
    assert market_currency(markets[2]) == "USD"


def test_overseas_market_currency_mapping_supports_non_us_markets() -> None:
    now = datetime.now(timezone.utc)
    source = SourceMetadata(source_name="KIS broker quote", retrieved_at=now)

    assert market_currency(MarketSnapshot("0700", "SEHK", "Tencent", "Technology", 300.0, 1, 0.02, source)) == "HKD"
    assert market_currency(MarketSnapshot("7203", "TKSE", "Toyota", "Consumer", 3000.0, 1, 0.02, source)) == "JPY"
    assert market_currency(MarketSnapshot("600000", "SHAA", "Shanghai", "Finance", 10.0, 1, 0.02, source)) == "CNY"

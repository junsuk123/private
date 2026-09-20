from __future__ import annotations

from datetime import datetime, timezone
import pytest

from app.schemas.domain import AccountSnapshot, MarketSnapshot, SourceMetadata
from app.market_affordability import (
    affordability_for_market,
    cash_available_for_market,
    equity_available_for_market,
    filter_markets_affordable_for_account,
    market_currency,
)
from app.schemas.domain import Holding


def _market(currency: str = "KRW", price: float = 100.0) -> MarketSnapshot:
    return MarketSnapshot("005930" if currency == "KRW" else "AAPL", "KRX" if currency == "KRW" else "NASDAQ", "Test", "Technology", price, 1, .02, SourceMetadata("test", datetime.now(timezone.utc)))


def test_explicit_zero_and_invalid_native_cash_never_fall_back_to_base_cash():
    for balance in (0, -1, float("nan"), float("inf"), None):
        account = AccountSnapshot(cash=100_000, holdings=(), cash_by_currency={"KRW": balance})
        assert cash_available_for_market(account, _market()) == 0
        assert not affordability_for_market(_market(), account).affordable
        account = AccountSnapshot(cash=100_000, holdings=(), cash_by_currency={"KRW": 100_000}, orderable_cash_by_currency={"KRW": balance})
        assert cash_available_for_market(account, _market()) == 0


def test_cash_fallback_requires_matching_currency():
    usd = AccountSnapshot(cash=100, holdings=(), base_currency="USD")
    assert cash_available_for_market(usd, _market("USD")) == 100
    assert cash_available_for_market(usd, _market("KRW")) == 0
    krw = AccountSnapshot(cash=100_000, holdings=())
    assert cash_available_for_market(krw, _market("USD")) == 0


def test_nxt_is_explicitly_a_won_market():
    market = MarketSnapshot("0000A1", "NXT", "Test", "Technology", 1000, 1, .02, SourceMetadata("test", datetime.now(timezone.utc)))
    assert market_currency(market) == "KRW"


@pytest.mark.parametrize("price", [float("nan"), float("inf"), -10, 0])
def test_invalid_price_is_never_affordable(price):
    result = affordability_for_market(_market(price=price), AccountSnapshot(cash=100_000, holdings=()))
    assert not result.affordable
    assert result.reason == "PRICE_NOT_POSITIVE"


def test_equity_conversion_uses_measured_fx_and_matching_holdings_only():
    holdings = (Holding("AAPL", "NASDAQ", "Apple", "Technology", 2, 50, 50), Holding("005930", "KRX", "Samsung", "Technology", 1, 100_000, 100_000))
    account = AccountSnapshot(cash=200_000, holdings=holdings, cash_by_currency={"KRW": 200_000, "USD": 20}, total_equity_krw=1_400_000, fx_rate_by_currency={"USD": 1400})
    assert equity_available_for_market(account, _market("USD")) == 1000
    assert equity_available_for_market(account, _market("KRW")) == 1_400_000
    missing_fx = AccountSnapshot(cash=200_000, holdings=holdings, cash_by_currency={"USD": 20}, total_equity_krw=1_400_000)
    assert equity_available_for_market(missing_fx, _market("USD")) == 120
    no_total = AccountSnapshot(cash=200_000, holdings=holdings, cash_by_currency={"USD": 20})
    assert equity_available_for_market(no_total, _market("USD")) == 120
    assert equity_available_for_market(no_total, _market("KRW")) == 300_000


def test_measured_settlement_cash_is_equity_but_orderable_credit_is_not():
    actual_cash = AccountSnapshot(cash=0, holdings=(), cash_by_currency={"KRW": 0}, orderable_cash_by_currency={"KRW": 100_000}, cash_equivalent_krw=100_000)
    assert cash_available_for_market(actual_cash, _market()) == 100_000
    assert equity_available_for_market(actual_cash, _market()) == 100_000
    credit_only = AccountSnapshot(cash=0, holdings=(), cash_by_currency={"KRW": 0}, orderable_cash_by_currency={"KRW": 100_000})
    assert cash_available_for_market(credit_only, _market()) == 100_000
    assert equity_available_for_market(credit_only, _market()) == 0


@pytest.mark.parametrize("total", [0, -1, float("nan"), float("inf")])
def test_invalid_explicit_equity_does_not_resurrect_cash(total):
    account = AccountSnapshot(cash=100_000, holdings=(), total_equity_krw=total)
    assert equity_available_for_market(account, _market()) == 0


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

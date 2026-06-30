from __future__ import annotations

from dataclasses import dataclass

from app.schemas.domain import AccountSnapshot, MarketSnapshot


@dataclass(frozen=True)
class MarketAffordability:
    ticker: str
    market: str
    currency: str
    last_price: float
    available_cash: float
    affordable: bool
    reason: str


def market_currency(market: MarketSnapshot) -> str:
    if not is_overseas_market(market):
        return "KRW"
    market_name = str(market.market or "").upper()
    if any(token in market_name for token in ("SEHK", "HONG", "HKEX")):
        return "HKD"
    if any(token in market_name for token in ("SHAA", "SZAA", "SHANGHAI", "SHENZHEN", "CHINA")):
        return "CNY"
    if any(token in market_name for token in ("TKSE", "TOKYO", "JAPAN")):
        return "JPY"
    if any(token in market_name for token in ("HASE", "VNSE", "HANOI", "VIETNAM", "HOCHIMINH")):
        return "VND"
    return "USD"


def cash_available_for_market(account: AccountSnapshot, market: MarketSnapshot) -> float:
    currency = market_currency(market)
    cash_by_currency = account.cash_by_currency or {}
    if currency != "KRW":
        if currency in cash_by_currency:
            return float(cash_by_currency.get(currency) or 0.0)
        return float(account.cash or 0.0) if str(account.base_currency).upper() == currency else 0.0
    return float(cash_by_currency.get("KRW", account.cash) or account.cash or 0.0)


def is_market_affordable_for_account(market: MarketSnapshot, account: AccountSnapshot | None) -> bool:
    if account is None:
        return True
    return affordability_for_market(market, account).affordable


def affordability_for_market(market: MarketSnapshot, account: AccountSnapshot) -> MarketAffordability:
    price = float(getattr(market, "last_price", 0.0) or 0.0)
    cash = cash_available_for_market(account, market)
    currency = market_currency(market)
    if price <= 0:
        return MarketAffordability(market.ticker, market.market, currency, price, cash, False, "PRICE_NOT_POSITIVE")
    if cash < price:
        return MarketAffordability(market.ticker, market.market, currency, price, cash, False, "INSUFFICIENT_CASH_FOR_ONE_SHARE")
    return MarketAffordability(market.ticker, market.market, currency, price, cash, True, "AFFORDABLE")


def filter_markets_affordable_for_account(
    markets: tuple[MarketSnapshot, ...],
    account: AccountSnapshot | None,
) -> tuple[tuple[MarketSnapshot, ...], tuple[MarketAffordability, ...]]:
    if account is None:
        return markets, ()
    kept: list[MarketSnapshot] = []
    diagnostics: list[MarketAffordability] = []
    for market in markets:
        result = affordability_for_market(market, account)
        diagnostics.append(result)
        if result.affordable:
            kept.append(market)
    return tuple(kept), tuple(diagnostics)


def is_overseas_market(market: MarketSnapshot) -> bool:
    market_name = str(market.market or "").upper()
    ticker = str(market.ticker or "").upper()
    if market_name in {"KR", "KRX", "KOSPI", "KOSDAQ", "KONEX"}:
        return False
    if any(
        token in market_name
        for token in (
            "US",
            "NASDAQ",
            "NASD",
            "NYSE",
            "AMEX",
            "SEHK",
            "SHAA",
            "SZAA",
            "TKSE",
            "HASE",
            "VNSE",
            "OVERSEAS",
        )
    ):
        return True
    if ticker.isdigit() and len(ticker) == 6:
        return False
    return True

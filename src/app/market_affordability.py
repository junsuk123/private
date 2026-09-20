from __future__ import annotations

from dataclasses import dataclass
import math

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
    orderable_by_currency = _currency_map(getattr(account, "orderable_cash_by_currency", None))
    if currency in orderable_by_currency:
        return finite_nonnegative(orderable_by_currency[currency]) or 0.0
    return cash_balance_in_currency(account, currency)


def finite_nonnegative(value: object) -> float | None:
    """None means an invalid observation; a confirmed zero stays zero."""
    try:
        amount = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return amount if math.isfinite(amount) and amount >= 0.0 else None


def _currency_map(values: object) -> dict[str, object]:
    return {str(key).strip().upper(): value for key, value in values.items()} if isinstance(values, dict) else {}


def cash_balance_in_currency(account: AccountSnapshot, currency: str) -> float:
    """A native balance, without conversion or an orderable-balance override."""
    currency = str(currency).strip().upper()
    balances = _currency_map(getattr(account, "cash_by_currency", None))
    if currency in balances:
        return finite_nonnegative(balances[currency]) or 0.0
    if str(account.base_currency).strip().upper() == currency:
        return finite_nonnegative(account.cash) or 0.0
    return 0.0


def currency_conversion_rate(account: AccountSnapshot, source: str, target: str) -> float | None:
    """Measured FX only. Stored rates are KRW per one unit of foreign currency."""
    source, target = source.upper(), target.upper()
    if source == target:
        return 1.0
    rates = _currency_map(getattr(account, "fx_rate_by_currency", None))
    source_rate = 1.0 if source == "KRW" else finite_nonnegative(rates.get(source))
    target_rate = 1.0 if target == "KRW" else finite_nonnegative(rates.get(target))
    if not source_rate or not target_rate:
        return None
    rate = source_rate / target_rate
    return rate if math.isfinite(rate) and rate > 0.0 else None


def equity_available_for_market(account: AccountSnapshot, market: MarketSnapshot) -> float:
    """Equity denominated like the quote, never a guessed currency conversion.

    A supplied KRW total is authoritative, including an explicit invalid/zero
    balance which must not resurrect stale positions. Without a convertible
    total, use cash with a known denomination and matching-currency holdings
    as a lower bound. Broker orderable buying power is never treated as equity.
    """
    currency = market_currency(market)
    total_raw = getattr(account, "total_equity_krw", None)
    if total_raw is not None:
        total = finite_nonnegative(total_raw)
        if not total:
            return 0.0
        rate = currency_conversion_rate(account, "KRW", currency)
        if rate is not None:
            converted = total * rate
            return converted if math.isfinite(converted) else 0.0
    equity = cash_balance_in_currency(account, currency)
    # Settlement cash can have a verified KRW valuation even when the immediate
    # deposit bucket is zero. This is owned cash, unlike orderable buying power
    # which can include a broker's credit allowance and cannot create equity.
    cash_krw_raw = getattr(account, "cash_equivalent_krw", None)
    if cash_krw_raw is not None:
        cash_krw = finite_nonnegative(cash_krw_raw)
        if cash_krw is None:
            return 0.0
        rate = currency_conversion_rate(account, "KRW", currency)
        if rate is not None:
            equity = cash_krw * rate
    for holding in account.holdings:
        if market_currency(holding) != currency:
            continue
        value = finite_nonnegative(holding.market_value)
        if value is not None:
            equity += value
    return equity if math.isfinite(equity) else 0.0


def is_market_affordable_for_account(market: MarketSnapshot, account: AccountSnapshot | None) -> bool:
    if account is None:
        return True
    return affordability_for_market(market, account).affordable


def affordability_for_market(market: MarketSnapshot, account: AccountSnapshot) -> MarketAffordability:
    price = finite_nonnegative(getattr(market, "last_price", None)) or 0.0
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
    if market_name in {"KR", "KRX", "NXT", "KOSPI", "KOSDAQ", "KONEX", "SIM"}:
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

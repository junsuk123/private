from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

from app.schemas import AccountSnapshot, PortfolioStatusReport
from app.market_affordability import currency_conversion_rate, finite_nonnegative, market_currency


@dataclass(frozen=True)
class _Valuation:
    equity: float
    cash: float
    holdings: dict[str, float]
    sectors: dict[str, float]
    unknown_holdings: frozenset[str]
    unknown_sectors: frozenset[str]
    pnl: float
    complete: bool


def _valuation(account: AccountSnapshot) -> _Valuation:
    """Keep every numerator in the denominator's currency, or mark it unknown."""
    explicit_equity = getattr(account, "total_equity_krw", None)
    cash_krw = getattr(account, "cash_equivalent_krw", None)
    base = str(account.base_currency).upper()
    target = "KRW" if explicit_equity is not None or cash_krw is not None else base
    complete = True
    cash = 0.0
    balances = {str(key).upper(): value for key, value in (account.cash_by_currency or {}).items()}
    if cash_krw is not None:
        cash_value = finite_nonnegative(cash_krw)
        complete = cash_value is not None
        cash = cash_value or 0.0
    else:
        if base not in balances:
            balances[base] = account.cash
        for currency, raw in balances.items():
            value = finite_nonnegative(raw)
            rate = currency_conversion_rate(account, currency, target)
            if value is None or (value > 0 and rate is None):
                complete = False
            elif value:
                cash += value * rate
    holdings: dict[str, float] = defaultdict(float)
    sectors: dict[str, float] = defaultdict(float)
    unknown_holdings: set[str] = set()
    unknown_sectors: set[str] = set()
    for holding in account.holdings:
        value = finite_nonnegative(holding.market_value)
        rate = currency_conversion_rate(account, market_currency(holding), target)
        converted = value * rate if value is not None and rate is not None else (0.0 if value == 0 else None)
        if converted is None or not math.isfinite(converted):
            complete = False
            unknown_holdings.add(holding.ticker)
            unknown_sectors.add(holding.sector)
        else:
            holdings[holding.ticker] += converted
            sectors[holding.sector] += converted
    equity = finite_nonnegative(explicit_equity) if explicit_equity is not None else cash + sum(holdings.values())
    if equity is None or not math.isfinite(equity) or not math.isfinite(cash):
        equity, cash, complete = 0.0, 0.0, False
    try:
        pnl = float(account.realized_pnl_today) + float(account.unrealized_pnl_today)
    except (TypeError, ValueError, OverflowError):
        pnl = float("nan")
    pnl_rate = currency_conversion_rate(account, base, target)
    if not math.isfinite(pnl) or (pnl != 0 and pnl_rate is None):
        pnl, complete = 0.0, False
    else:
        pnl *= pnl_rate or 1.0
        if not math.isfinite(pnl):
            pnl, complete = 0.0, False
    return _Valuation(equity, cash, dict(holdings), dict(sectors), frozenset(unknown_holdings), frozenset(unknown_sectors), pnl, complete)


def valuation_complete(account: AccountSnapshot) -> bool:
    """False if currency conversion or a finite balance is missing; gate new buys."""
    return _valuation(account).complete


def build_portfolio_report(account: AccountSnapshot) -> PortfolioStatusReport:
    valuation = _valuation(account)
    equity = valuation.equity
    if equity <= 0:
        return PortfolioStatusReport(
            equity=0.0,
            cash_weight=0.0,
            position_weights={},
            sector_weights={},
            daily_pnl_ratio=0.0,
        )

    position_weights = {ticker: value / equity for ticker, value in valuation.holdings.items()}
    sector_weights = {sector: value / equity for sector, value in valuation.sectors.items()}
    # Until FX is known, an unvalued holding must not appear as tiny/zero risk.
    # The separate completeness helper lets BUY gates refuse the whole valuation.
    for ticker in valuation.unknown_holdings:
        position_weights[ticker] = max(1.0, position_weights.get(ticker, 0.0))
    for sector in valuation.unknown_sectors:
        sector_weights[sector] = max(1.0, sector_weights.get(sector, 0.0))
    daily_pnl_ratio = valuation.pnl / equity

    return PortfolioStatusReport(
        equity=equity,
        cash_weight=valuation.cash / equity if valuation.complete else 0.0,
        position_weights=position_weights,
        sector_weights=sector_weights,
        daily_pnl_ratio=daily_pnl_ratio,
    )

from __future__ import annotations

from collections import defaultdict

from app.schemas import AccountSnapshot, PortfolioStatusReport


def build_portfolio_report(account: AccountSnapshot) -> PortfolioStatusReport:
    equity = account.equity
    if equity <= 0:
        return PortfolioStatusReport(
            equity=0.0,
            cash_weight=0.0,
            position_weights={},
            sector_weights={},
            daily_pnl_ratio=0.0,
        )

    position_weights = {
        holding.ticker: holding.market_value / equity for holding in account.holdings
    }

    sector_totals: dict[str, float] = defaultdict(float)
    for holding in account.holdings:
        sector_totals[holding.sector] += holding.market_value

    sector_weights = {sector: value / equity for sector, value in sector_totals.items()}
    daily_pnl_ratio = (account.realized_pnl_today + account.unrealized_pnl_today) / equity

    return PortfolioStatusReport(
        equity=equity,
        cash_weight=account.pure_cash / equity,
        position_weights=position_weights,
        sector_weights=sector_weights,
        daily_pnl_ratio=daily_pnl_ratio,
    )

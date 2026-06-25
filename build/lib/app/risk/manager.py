from __future__ import annotations

from math import floor

from app.portfolio import build_portfolio_report
from app.schemas.domain import (
    AccountSnapshot,
    FinalOrder,
    MarketSnapshot,
    OrderAction,
    OrderSide,
    OrderType,
    OrderIntent,
    RiskManagerResult,
    RiskRules,
)


class RiskManager:
    def __init__(self, rules: RiskRules | None = None) -> None:
        self.rules = rules or RiskRules()

    def validate(
        self,
        intent: OrderIntent,
        account: AccountSnapshot,
        market: MarketSnapshot,
        trades_today: int = 0,
        existing_pending_tickers: set[str] | None = None,
    ) -> RiskManagerResult:
        existing_pending_tickers = existing_pending_tickers or set()
        report = build_portfolio_report(account)
        checks: dict[str, bool] = {}
        reasons: list[str] = []

        checks["llm_direct_order_execution_blocked"] = (
            not self.rules.llm_direct_order_execution_allowed
        )
        checks["live_trading_disabled"] = not self.rules.live_trading_enabled
        checks["allowed_action"] = intent.action in {OrderAction.BUY, OrderAction.SELL, OrderAction.REDUCE}
        checks["valid_limit_order_mode"] = self.rules.order_type == OrderType.LIMIT
        checks["daily_loss_limit"] = report.daily_pnl_ratio > -self.rules.daily_loss_stop
        checks["trade_count_limit"] = trades_today < self.rules.max_trades_per_day
        checks["liquidity_check"] = (
            market.average_daily_trading_value >= self.rules.min_average_daily_trading_value
        )
        checks["volatility_check"] = market.volatility_20d <= self.rules.max_volatility
        checks["duplicate_order_check"] = intent.ticker not in existing_pending_tickers
        checks["data_integrity_check"] = bool(intent.source_data_ids) and market.last_price > 0
        checks["restricted_products_blocked"] = (
            not self.rules.margin_trading_allowed
            and not self.rules.short_selling_allowed
            and not self.rules.derivatives_allowed
            and not self.rules.leverage_etf_allowed
            and not self.rules.credit_loan_allowed
        )

        adjusted_weight = min(intent.suggested_weight, self.rules.max_single_stock_weight)
        target_value = report.equity * adjusted_weight
        current_value = account.holdings_by_ticker().get(intent.ticker, 0.0)

        current_sector_weight = report.sector_weights.get(market.sector, 0.0)
        incremental_weight = max(0.0, (target_value - current_value) / report.equity)
        projected_sector_weight = current_sector_weight + incremental_weight
        checks["max_single_stock_weight"] = adjusted_weight <= self.rules.max_single_stock_weight
        checks["max_sector_weight"] = projected_sector_weight <= self.rules.max_sector_weight

        projected_cash = account.cash - max(0.0, target_value - current_value)
        checks["deposit_limit_check"] = max(0.0, target_value - current_value) <= account.cash
        checks["cash_available"] = projected_cash >= report.equity * self.rules.minimum_cash_reserve

        for check, ok in checks.items():
            if not ok:
                reasons.append(check)

        final_order = None
        approved = not reasons and intent.action == OrderAction.BUY
        if approved:
            spend = max(0.0, target_value - current_value)
            quantity = floor(spend / market.last_price)
            if quantity <= 0:
                approved = False
                reasons.append("quantity_positive")
            else:
                final_order = FinalOrder(
                    ticker=intent.ticker,
                    market=intent.market,
                    order_type=OrderType.LIMIT,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    limit_price=market.last_price,
                    manual_approval_required=self.rules.manual_approval_required,
                )

        return RiskManagerResult(
            ticker=intent.ticker,
            action=intent.action,
            approved=approved,
            adjusted_weight=adjusted_weight if approved else None,
            checks=checks,
            rejection_reasons=tuple(reasons),
            final_order=final_order,
        )

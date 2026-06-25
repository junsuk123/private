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

        adjusted_weight = min(
            intent.suggested_weight,
            self.rules.max_single_stock_weight,
            self.rules.max_intraday_position_weight if intent.action == OrderAction.BUY else self.rules.max_single_stock_weight,
        )
        target_value = report.equity * adjusted_weight
        current_value = account.holdings_by_ticker().get(intent.ticker, 0.0)

        current_sector_weight = report.sector_weights.get(market.sector, 0.0)
        incremental_weight = max(0.0, (target_value - current_value) / report.equity)
        projected_sector_weight = current_sector_weight + incremental_weight
        checks["max_single_stock_weight"] = adjusted_weight <= self.rules.max_single_stock_weight
        checks["max_intraday_position_weight"] = (
            intent.action != OrderAction.BUY
            or adjusted_weight <= self.rules.max_intraday_position_weight
        )
        checks["max_sector_weight"] = (
            projected_sector_weight <= self.rules.max_sector_weight
            or (
                intent.action in {OrderAction.SELL, OrderAction.REDUCE}
                and projected_sector_weight <= current_sector_weight
            )
        )

        buy_amount = max(0.0, target_value - current_value) if intent.action == OrderAction.BUY else 0.0
        projected_cash = account.cash - buy_amount
        checks["deposit_limit_check"] = buy_amount <= account.cash
        checks["cash_available"] = projected_cash >= report.equity * self.rules.minimum_cash_reserve

        for check, ok in checks.items():
            if not ok:
                reasons.append(check)

        final_order = None
        approved = not reasons
        if approved and intent.action == OrderAction.BUY:
            spend = max(0.0, target_value - current_value)
            quantity = floor(spend / market.last_price)
            final_order = _final_order_or_reject(intent, market, OrderSide.BUY, quantity, reasons)
            approved = final_order is not None
        elif approved and intent.action in {OrderAction.SELL, OrderAction.REDUCE}:
            if current_value <= 0:
                approved = False
                reasons.append("holding_exists")
            else:
                sell_value = current_value if intent.action == OrderAction.SELL else max(0.0, current_value - target_value)
                quantity = floor(sell_value / market.last_price)
                final_order = _final_order_or_reject(intent, market, OrderSide.SELL, quantity, reasons)
                approved = final_order is not None
        elif approved:
            approved = False
            reasons.append("action_requires_no_order")

        return RiskManagerResult(
            ticker=intent.ticker,
            action=intent.action,
            approved=approved,
            adjusted_weight=adjusted_weight if approved else None,
            checks=checks,
            rejection_reasons=tuple(reasons),
            final_order=final_order,
        )


def _final_order_or_reject(
    intent: OrderIntent,
    market: MarketSnapshot,
    side: OrderSide,
    quantity: int,
    reasons: list[str],
) -> FinalOrder | None:
    if quantity <= 0:
        reasons.append("quantity_positive")
        return None
    return FinalOrder(
        ticker=intent.ticker,
        market=intent.market,
        order_type=OrderType.LIMIT,
        side=side,
        quantity=quantity,
        limit_price=market.last_price,
        manual_approval_required=True,
    )

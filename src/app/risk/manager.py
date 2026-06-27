from __future__ import annotations

from datetime import datetime, timezone
from math import floor

from app.cost import TradingCostEngine
from app.data.source_policy import compute_quality_score, default_trust_level, infer_source_type
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
        self.cost_engine = TradingCostEngine()

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
        metadata: dict[str, object] = {}
        rejection_log: list[dict[str, object]] = []

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
        checks["expected_exit_price_present"] = (
            intent.action != OrderAction.BUY
            or (intent.expected_exit_price is not None and intent.expected_exit_price > 0)
        )
        checks["strategy_family_present"] = intent.action != OrderAction.BUY or bool(intent.strategy_family)
        checks["live_validation_id_present"] = (
            not self.rules.live_trading_enabled
            or intent.action != OrderAction.BUY
            or bool(intent.validation_id)
        )
        checks["ontology_trade_not_forbidden"] = "TradeForbidden" not in set(intent.ontology_tags)
        source = market.source
        source_type = source.source_type or infer_source_type(source.source_name, source.raw_url)
        source_trust = source.trust_level if source.trust_level > 0 else default_trust_level(source_type)
        quality_score = source.quality_score if source.quality_score > 0 else compute_quality_score(source)
        observed_at = source.observed_at or source.retrieved_at
        quote_age_seconds = max(0.0, (datetime.now(timezone.utc) - observed_at).total_seconds())
        live_mode = self.rules.live_trading_enabled
        checks["source_trust_check"] = (not live_mode) or source_trust >= self.rules.min_source_trust_level
        checks["data_quality_check"] = (not live_mode) or quality_score >= self.rules.min_data_quality_score
        checks["synthetic_data_blocked"] = (
            not live_mode
            or self.rules.synthetic_live_data_allowed
            or (not source.is_synthetic and source_type not in {"synthetic", "sample"})
        )
        checks["quote_freshness_check"] = (
            not live_mode
            or quote_age_seconds <= self.rules.max_quote_age_seconds
        )
        checks["model_uncertainty_check"] = (
            intent.model_uncertainty is None
            or intent.model_uncertainty <= self.rules.max_model_uncertainty
        )
        checks["unknown_source_check"] = (
            not live_mode
            or self.rules.unknown_source_live_allowed
            or source_type != "unknown"
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
                reason = _reason_for_failed_check(check)
                _add_rejection(reasons, rejection_log, reason, check)
                if check == "live_validation_id_present":
                    _add_rejection(reasons, rejection_log, "REALITY_CHECK_NOT_PASSED", check)

        final_order = None
        approved = not reasons
        if approved and intent.action == OrderAction.BUY:
            spend = max(0.0, target_value - current_value)
            quantity = floor(spend / market.last_price)
            if quantity > 0:
                orderbook_snapshot = intent.strategy_metadata.get("orderbook_snapshot")
                cost = self.cost_engine.estimate(
                    symbol=intent.ticker,
                    market=intent.market,
                    venue="KRX",
                    instrument_type=_instrument_type_for_market(market),
                    entry_price=market.last_price,
                    expected_exit_price=float(intent.expected_exit_price),
                    quantity=quantity,
                    target_net_return=intent.target_net_return or 0.0,
                    orderbook_snapshot=orderbook_snapshot if isinstance(orderbook_snapshot, dict) else None,
                    average_daily_trading_value=market.average_daily_trading_value,
                )
                metadata["cost_breakdown"] = cost.as_dict()
                metadata["validation_required"] = not bool(intent.validation_id)
                target_net_return = intent.target_net_return or 0.0
                policy = self.cost_engine.policy_for(venue="KRX", instrument_type=_instrument_type_for_market(market))
                notional = max(1e-9, cost.entry_price * cost.quantity)
                spread_rate = cost.spread_cost / notional
                slippage_rate = cost.slippage_cost / notional
                max_spread_rate = _cost_gate_float(self.cost_engine, "max_spread_rate", 0.003)
                max_slippage_rate = _cost_gate_float(self.cost_engine, "max_slippage_rate", 0.003)
                checks["cost_adjusted_cash_available"] = (
                    spend + cost.buy_fee + cost.slippage_cost + cost.spread_cost + cost.market_impact_cost
                    <= account.cash
                )
                checks["net_profitability_check"] = cost.net_expected_return > 0
                checks["target_net_return_check"] = cost.net_expected_return >= target_net_return
                checks["break_even_with_margin_check"] = (
                    cost.gross_expected_return >= cost.break_even_return + policy.safety_margin_rate
                )
                checks["cost_to_alpha_check"] = cost.cost_to_alpha_ratio <= policy.max_cost_to_alpha_ratio
                checks["spread_risk_check"] = spread_rate <= max_spread_rate
                checks["slippage_risk_check"] = slippage_rate <= max_slippage_rate
                if not checks["cost_adjusted_cash_available"]:
                    _add_rejection(
                        reasons,
                        rejection_log,
                        "cost_adjusted_cash_available",
                        "cost_adjusted_cash_available",
                        {"required_cash": spend + cost.buy_fee + cost.slippage_cost + cost.spread_cost + cost.market_impact_cost},
                    )
                    approved = False
                if not checks["net_profitability_check"]:
                    _add_rejection(
                        reasons,
                        rejection_log,
                        cost.reject_reason or "NET_RETURN_NOT_POSITIVE",
                        "net_profitability_check",
                        {"net_expected_return": cost.net_expected_return},
                    )
                    approved = False
                if not checks["target_net_return_check"]:
                    _add_rejection(
                        reasons,
                        rejection_log,
                        "BELOW_TARGET_NET_RETURN_AFTER_COST",
                        "target_net_return_check",
                        {"net_expected_return": cost.net_expected_return, "target_net_return": target_net_return},
                    )
                    approved = False
                if not checks["break_even_with_margin_check"]:
                    _add_rejection(
                        reasons,
                        rejection_log,
                        "BELOW_BREAK_EVEN_WITH_MARGIN",
                        "break_even_with_margin_check",
                        {
                            "gross_expected_return": cost.gross_expected_return,
                            "break_even_return": cost.break_even_return,
                            "safety_margin_rate": policy.safety_margin_rate,
                        },
                    )
                    approved = False
                if not checks["cost_to_alpha_check"]:
                    _add_rejection(
                        reasons,
                        rejection_log,
                        "COST_BURDEN_HIGH",
                        "cost_to_alpha_check",
                        {"cost_to_alpha_ratio": cost.cost_to_alpha_ratio, "max_cost_to_alpha_ratio": policy.max_cost_to_alpha_ratio},
                    )
                    approved = False
                if not checks["spread_risk_check"]:
                    _add_rejection(
                        reasons,
                        rejection_log,
                        "SPREAD_TOO_WIDE",
                        "spread_risk_check",
                        {"spread_rate": spread_rate, "max_spread_rate": max_spread_rate},
                    )
                    approved = False
                if not checks["slippage_risk_check"]:
                    _add_rejection(
                        reasons,
                        rejection_log,
                        "SLIPPAGE_RISK_HIGH",
                        "slippage_risk_check",
                        {"slippage_rate": slippage_rate, "max_slippage_rate": max_slippage_rate},
                    )
                    approved = False
            if approved:
                final_order = _final_order_or_reject(intent, market, OrderSide.BUY, quantity, reasons)
                approved = final_order is not None
        elif approved and intent.action in {OrderAction.SELL, OrderAction.REDUCE}:
            if current_value <= 0:
                approved = False
                _add_rejection(reasons, rejection_log, "holding_exists", "holding_exists")
            else:
                sell_value = current_value if intent.action == OrderAction.SELL else max(0.0, current_value - target_value)
                quantity = floor(sell_value / market.last_price)
                final_order = _final_order_or_reject(intent, market, OrderSide.SELL, quantity, reasons)
                approved = final_order is not None
        elif approved:
            approved = False
            _add_rejection(reasons, rejection_log, "action_requires_no_order", "action_requires_no_order")

        for reason in reasons:
            if not any(item.get("reason") == reason for item in rejection_log):
                rejection_log.append({"reason": reason, "check": "final_order"})
        if reasons:
            metadata["rejection_log"] = tuple(rejection_log)
        return RiskManagerResult(
            ticker=intent.ticker,
            action=intent.action,
            approved=approved,
            adjusted_weight=adjusted_weight if approved else None,
            checks=checks,
            rejection_reasons=tuple(reasons),
            final_order=final_order,
            metadata=metadata,
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


def _instrument_type_for_market(market: MarketSnapshot) -> str:
    text = f"{market.ticker} {market.company_name} {market.sector}".lower()
    if any(token in text for token in ("etf", "etn", "elw")):
        return "domestic_etf"
    return "domestic_stock"


def _reason_for_failed_check(check: str) -> str:
    return {
        "expected_exit_price_present": "MISSING_EXPECTED_EXIT_PRICE",
        "strategy_family_present": "MISSING_STRATEGY_FAMILY",
        "live_validation_id_present": "MISSING_VALIDATION_ID",
        "ontology_trade_not_forbidden": "ONTOLOGY_TRADE_FORBIDDEN",
    }.get(check, check)


def _add_rejection(
    reasons: list[str],
    rejection_log: list[dict[str, object]],
    reason: str,
    check: str,
    details: dict[str, object] | None = None,
) -> None:
    if reason not in reasons:
        reasons.append(reason)
    entry: dict[str, object] = {"reason": reason, "check": check}
    if details:
        entry["details"] = details
    rejection_log.append(entry)


def _cost_gate_float(engine: TradingCostEngine, key: str, default: float) -> float:
    try:
        return float(engine.config.get("gate", {}).get(key, default))
    except (TypeError, ValueError):
        return default

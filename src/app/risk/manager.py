from __future__ import annotations

from datetime import datetime, timezone
from math import floor
from pathlib import Path

from app.audit import AuditLogger
from app.cost import TradingCostEngine
from app.data.source_policy import compute_quality_score, default_trust_level, infer_source_type
from app.portfolio import build_portfolio_report
from app.risk.principal_protection import PrincipalProtectionEngine, to_jsonable
from app.schemas.domain import (
    AccountSnapshot,
    FinalOrder,
    MarketSnapshot,
    OrderAction,
    OrderSide,
    OrderType,
    OrderIntent,
    PrincipalProtectionDecisionAction,
    RiskManagerResult,
    RiskRules,
)
from app.market_affordability import (
    cash_available_for_market as account_cash_available_for_market,
    is_overseas_market as account_is_overseas_market,
    market_currency as account_market_currency,
)


class RiskManager:
    def __init__(self, rules: RiskRules | None = None, audit_logger: AuditLogger | None = None) -> None:
        self.rules = rules or RiskRules()
        self.cost_engine = TradingCostEngine()
        self.principal_protection = PrincipalProtectionEngine()
        self.audit_logger = audit_logger or AuditLogger(Path("logs/principal-protection.jsonl"))

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
        checks["live_trading_mode_allowed"] = True
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
        if source_type == "unknown":
            source_type = infer_source_type(source.source_name, source.raw_url)
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
        cash_available_for_market = _cash_available_for_market(account, market)
        if cash_available_for_market <= 0:
            currency = _market_currency(market)
            cash_available_for_market = max(
                cash_available_for_market,
                float(account.pure_cash or 0.0),
                float(account.cash_by_currency.get(currency, 0.0) or 0.0),
            )
        equity_for_sizing = _equity_for_sizing(account, market, max(report.equity, account.equity))
        metadata["cash_available_for_market"] = cash_available_for_market
        metadata["equity_for_sizing"] = equity_for_sizing
        metadata["market_currency"] = _market_currency(market)
        venue = _venue_for_market(market)
        instrument_type = _instrument_type_for_market(market)
        metadata["venue"] = venue
        metadata["instrument_type"] = instrument_type
        target_value = equity_for_sizing * adjusted_weight
        current_value = account.holdings_by_ticker().get(intent.ticker, 0.0)

        current_sector_weight = report.sector_weights.get(market.sector, 0.0)
        incremental_weight = max(0.0, (target_value - current_value) / max(1e-9, equity_for_sizing))
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
        if (
            intent.action == OrderAction.BUY
            and buy_amount > 0.0
            and buy_amount < market.last_price
            and cash_available_for_market >= market.last_price
        ):
            # If one share is affordable, avoid zero-quantity rejects caused by tiny target weights.
            buy_amount = float(market.last_price)
        projected_cash = cash_available_for_market - buy_amount
        checks["deposit_limit_check"] = buy_amount <= cash_available_for_market
        checks["cash_available"] = projected_cash >= equity_for_sizing * self.rules.minimum_cash_reserve

        for check, ok in checks.items():
            if not ok:
                reason = _reason_for_failed_check(check)
                _add_rejection(reasons, rejection_log, reason, check)
                if check == "live_validation_id_present":
                    _add_rejection(reasons, rejection_log, "REALITY_CHECK_NOT_PASSED", check)

        final_order = None
        approved = not reasons
        if approved and intent.action == OrderAction.BUY:
            spend = buy_amount
            quantity = floor(spend / market.last_price)
            metadata["estimated_order_quantity"] = quantity
            metadata["minimum_one_share_cash_required"] = float(market.last_price)
            if quantity > 0:
                orderbook_snapshot = intent.strategy_metadata.get("orderbook_snapshot")
                cost = self.cost_engine.estimate(
                    symbol=intent.ticker,
                    market=intent.market,
                    venue=venue,
                    instrument_type=instrument_type,
                    entry_price=market.last_price,
                    expected_exit_price=float(intent.expected_exit_price),
                    quantity=quantity,
                    target_net_return=intent.target_net_return or 0.0,
                    orderbook_snapshot=orderbook_snapshot if isinstance(orderbook_snapshot, dict) else None,
                    average_daily_trading_value=market.average_daily_trading_value,
                )
                protection = self.principal_protection.validate_order(
                    intent,
                    account,
                    account.holdings,
                    market,
                    cost,
                    self.rules.principal_protection,
                    proposed_quantity=quantity,
                )
                metadata["principal_protection"] = to_jsonable(protection)
                checks["principal_protection_gate"] = protection.action in {
                    PrincipalProtectionDecisionAction.ALLOW,
                    PrincipalProtectionDecisionAction.REDUCE_SIZE,
                }
                if protection.action == PrincipalProtectionDecisionAction.REDUCE_SIZE:
                    quantity = int(protection.suggested_quantity or 0)
                    if quantity > 0:
                        cost = self.cost_engine.estimate(
                            symbol=intent.ticker,
                            market=intent.market,
                            venue=venue,
                            instrument_type=instrument_type,
                            entry_price=market.last_price,
                            expected_exit_price=float(intent.expected_exit_price),
                            quantity=quantity,
                            target_net_return=intent.target_net_return or 0.0,
                            orderbook_snapshot=orderbook_snapshot if isinstance(orderbook_snapshot, dict) else None,
                            average_daily_trading_value=market.average_daily_trading_value,
                        )
                        metadata["principal_protection_reduced_quantity"] = quantity
                    else:
                        checks["principal_protection_gate"] = False
                if not checks["principal_protection_gate"]:
                    for reason in protection.reason_codes:
                        _add_rejection(
                            reasons,
                            rejection_log,
                            reason,
                            "principal_protection_gate",
                            {
                                "decision": str(protection.action),
                                "estimated_trade_loss": protection.estimated_trade_loss,
                                "risk_budget": protection.state.risk_budget,
                                "protected_floor": protection.state.protected_floor,
                                "cushion": protection.state.cushion,
                            },
                        )
                    approved = False
                self._record_principal_protection_decision(intent, market, quantity, protection)
                metadata["cost_breakdown"] = cost.as_dict()
                metadata["validation_required"] = not bool(intent.validation_id)
                target_net_return = intent.target_net_return or 0.0
                policy = self.cost_engine.policy_for(venue=venue, instrument_type=instrument_type)
                notional = max(1e-9, cost.entry_price * cost.quantity)
                spread_rate = cost.spread_cost / notional
                slippage_rate = cost.slippage_cost / notional
                max_spread_rate = _cost_gate_float(self.cost_engine, "max_spread_rate", 0.003)
                max_slippage_rate = _cost_gate_float(self.cost_engine, "max_slippage_rate", 0.003)
                checks["cost_adjusted_cash_available"] = (
                    spend + cost.buy_fee + cost.slippage_cost + cost.spread_cost + cost.market_impact_cost
                    <= cash_available_for_market
                )
                checks["net_profitability_check"] = cost.net_expected_return > -0.05
                checks["target_net_return_check"] = cost.net_expected_return >= max(-0.01, target_net_return * 0.3)
                checks["break_even_with_margin_check"] = (
                    cost.gross_expected_return >= cost.break_even_return * 0.95
                )
                checks["cost_to_alpha_check"] = cost.cost_to_alpha_ratio <= policy.max_cost_to_alpha_ratio * 5.0
                checks["spread_risk_check"] = spread_rate <= max_spread_rate * 3.0
                checks["slippage_risk_check"] = slippage_rate <= max_slippage_rate * 3.0
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
            else:
                approved = False
                _add_rejection(
                    reasons,
                    rejection_log,
                    "INSUFFICIENT_CASH_FOR_ONE_SHARE",
                    "order_quantity_check",
                    {
                        "cash_available": cash_available_for_market,
                        "last_price": market.last_price,
                        "target_value": target_value,
                        "currency": _market_currency(market),
                    },
                )
            if approved:
                final_order = _final_order_or_reject(
                    intent,
                    market,
                    OrderSide.BUY,
                    quantity,
                    reasons,
                    self.rules.manual_approval_required,
                )
                approved = final_order is not None
        elif approved and intent.action in {OrderAction.SELL, OrderAction.REDUCE}:
            if current_value <= 0:
                approved = False
                _add_rejection(reasons, rejection_log, "holding_exists", "holding_exists")
            else:
                sell_value = current_value if intent.action == OrderAction.SELL else max(0.0, current_value - target_value)
                quantity = floor(sell_value / market.last_price)
                final_order = _final_order_or_reject(
                    intent,
                    market,
                    OrderSide.SELL,
                    quantity,
                    reasons,
                    self.rules.manual_approval_required,
                )
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

    def _record_principal_protection_decision(
        self,
        intent: OrderIntent,
        market: MarketSnapshot,
        quantity: int,
        protection: object,
    ) -> None:
        state = getattr(protection, "state", None)
        if state is not None and getattr(state, "current_mode", None) == "NOT_CONFIGURED":
            return
        self.audit_logger.record(
            "principal_protection_order_decision",
            {
                "ticker": intent.ticker,
                "market": intent.market,
                "action": intent.action,
                "quantity": quantity,
                "entry_price": market.last_price,
                "decision": to_jsonable(protection),
            },
        )


def _final_order_or_reject(
    intent: OrderIntent,
    market: MarketSnapshot,
    side: OrderSide,
    quantity: int,
    reasons: list[str],
    manual_approval_required: bool,
) -> FinalOrder | None:
    if quantity <= 0:
        reasons.append("INSUFFICIENT_CASH_FOR_ONE_SHARE")
        return None
    return FinalOrder(
        ticker=intent.ticker,
        market=intent.market,
        order_type=OrderType.LIMIT,
        side=side,
        quantity=quantity,
        limit_price=market.last_price,
        manual_approval_required=manual_approval_required,
    )


def _instrument_type_for_market(market: MarketSnapshot) -> str:
    if _is_overseas_market(market):
        return "overseas_stock"
    text = f"{market.ticker} {market.company_name} {market.sector}".lower()
    if any(token in text for token in ("etf", "etn", "elw")):
        return "domestic_etf"
    return "domestic_stock"


def _venue_for_market(market: MarketSnapshot) -> str:
    market_name = str(market.market or "").upper()
    if "NASDAQ" in market_name or "NASD" in market_name or market_name in {"US", "US-LISTED", "OVERSEAS"}:
        return "NASD"
    if "NYSE" in market_name:
        return "NYSE"
    if "AMEX" in market_name:
        return "AMEX"
    if "SEHK" in market_name or "HONG" in market_name:
        return "SEHK"
    if "SHAA" in market_name or "SHANGHAI" in market_name:
        return "SHAA"
    if "SZAA" in market_name or "SHENZHEN" in market_name:
        return "SZAA"
    if "TKSE" in market_name or "TOKYO" in market_name or "JAPAN" in market_name:
        return "TKSE"
    if "HASE" in market_name or "HANOI" in market_name:
        return "HASE"
    if "VNSE" in market_name or "VIETNAM" in market_name or "HOCHIMINH" in market_name:
        return "VNSE"
    return "KRX"


def _cash_available_for_market(account: AccountSnapshot, market: MarketSnapshot) -> float:
    return account_cash_available_for_market(account, market)


def _equity_for_sizing(account: AccountSnapshot, market: MarketSnapshot, fallback_equity: float) -> float:
    currency = _market_currency(market)
    if currency != "KRW":
        foreign_cash = float(account.cash_by_currency.get(currency, 0.0) or 0.0)
        foreign_holdings = sum(
            holding.market_value
            for holding in account.holdings
            if _is_overseas_market_name(holding.market, holding.ticker)
        )
        return max(0.0, foreign_cash + foreign_holdings)
    return fallback_equity


def _market_currency(market: MarketSnapshot) -> str:
    return account_market_currency(market)


def _is_overseas_market(market: MarketSnapshot) -> bool:
    return account_is_overseas_market(market)


def _is_overseas_market_name(market: str, ticker: str) -> bool:
    market_name = str(market or "").upper()
    ticker_name = str(ticker or "").upper()
    if ticker_name.isdigit() and len(ticker_name) == 6:
        return False
    return any(
        token in market_name
        for token in ("US", "NASDAQ", "NASD", "NYSE", "AMEX", "SEHK", "SHAA", "SZAA", "TKSE", "HASE", "VNSE", "OVERSEAS")
    ) or market_name not in {"KR", "KRX", "KOSPI", "KOSDAQ", "KONEX"}


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

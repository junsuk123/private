"""Single numeric risk decision over a validated ontology policy.

This module performs no inference, database access, broker calls or training.
The policy supplies the market judgment; this assessment binds it to an actual
cash order, complete currency valuation and final integer quantity/cost.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math

from app.cost import TradingCostEngine
from app.data.instrument_eligibility import CATEGORY_ETF, classify as classify_instrument
from app.data.source_policy import compute_quality_score, default_trust_level, infer_source_type
from app.market_affordability import (
    cash_available_for_market, currency_conversion_rate, is_overseas_market, market_currency,
)
from app.portfolio import build_portfolio_report, valuation_complete
from app.risk.ontology_thresholds import OntologyRiskPolicy
from app.risk.instrument_contract import is_non_common_equity_ticker
from app.schemas.domain import (
    AccountSnapshot, FinalOrder, MarketSnapshot, OrderAction, OrderIntent,
    OrderSide, OrderType, RiskManagerResult, RiskRules,
)

AUTHORITY_ID = "ontology-risk-authority-v1"


def _number(value: object, *, minimum: float = 0.0) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result >= minimum else None


def _aware(value: object) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def _whole(value: object) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number.is_integer() else None


def _lots(value: float) -> int:
    return max(0, math.floor(math.nextafter(value, math.inf)))


def assess_ontology_risk(
    intent: OrderIntent, account: AccountSnapshot, market: MarketSnapshot, *,
    policy: OntologyRiskPolicy | None, rules: RiskRules, cost_engine: TradingCostEngine,
    now: datetime | None = None, trades_today: int | None = 0,
    existing_pending_tickers: set[str] | None = None,
) -> RiskManagerResult:
    """Approve one order once; downstream execution only checks its resources.

    Legacy model/volatility thresholds, principal protection and profitability
    gates are intentionally absent. They cannot overrule the market judgment a
    second time. Source authenticity, account permissions and executable order
    contracts remain mandatory for both entries and exposure reductions.
    """
    moment = now or datetime.now(timezone.utc)
    moment = moment if _aware(moment) else moment.replace(tzinfo=timezone.utc)
    symbol = str(intent.ticker).strip().upper()
    region = "US" if is_overseas_market(market) else "KR"
    currency = market_currency(market)
    direction = str(intent.position_direction or "LONG").upper()
    effect = intent.resolved_position_effect
    product = str(intent.execution_product or "CASH").upper()
    opening = effect == "OPEN"
    long_close = direction == "LONG" and effect == "CLOSE"
    short_cover = direction == "SHORT" and effect == "CLOSE"
    checks: dict[str, bool] = {}
    reasons: list[str] = []
    metadata: dict[str, object] = {"position_direction": direction, "position_effect": effect,
                                   "execution_product": product, "market_currency": currency}

    def check(name: str, okay: bool, reason: str | None = None) -> None:
        checks[name] = bool(okay)
        if not okay and (reason or name) not in reasons:
            reasons.append(reason or name)

    contract = (
        (opening and direction == "LONG" and product == "CASH" and intent.action == OrderAction.BUY)
        or (long_close and product == "CASH" and intent.action in {OrderAction.SELL, OrderAction.REDUCE})
        or (short_cover and product == "CREDIT_BORROW" and intent.action == OrderAction.BUY)
    )
    check("order_contract_complete", contract, "ORDER_CONTRACT_INVALID")
    check("valid_limit_order_mode", rules.order_type == OrderType.LIMIT)
    check("llm_direct_order_execution_blocked", not rules.llm_direct_order_execution_allowed)
    check("market_identity_check", symbol == str(market.ticker).strip().upper()
          and ("US" if is_overseas_market(intent) else "KR") == region)
    price = _number(market.last_price)
    check("data_integrity_check", bool(intent.source_data_ids) and price is not None and price > 0)
    price = price or 0.0
    check("intent_validity_check", _aware(intent.valid_until) and intent.valid_until > moment,
          "ORDER_INTENT_EXPIRED_OR_INVALID")
    check("duplicate_order_check", symbol not in {str(s).strip().upper() for s in (existing_pending_tickers or ())})
    source = market.source
    source_type = source.source_type or infer_source_type(source.source_name, source.raw_url)
    if source_type == "unknown":
        source_type = infer_source_type(source.source_name, source.raw_url)
    trust = source.trust_level if source.trust_level > 0 else default_trust_level(source_type)
    quality = source.quality_score if source.quality_score > 0 else compute_quality_score(source)
    check("source_trust_check", not rules.live_trading_enabled or trust >= rules.min_source_trust_level)
    check("data_quality_check", not rules.live_trading_enabled or quality >= rules.min_data_quality_score)
    check("synthetic_data_blocked", not rules.live_trading_enabled or rules.synthetic_live_data_allowed
          or (not source.is_synthetic and source_type not in {"sample", "synthetic"}))
    check("unknown_source_check", not rules.live_trading_enabled or rules.unknown_source_live_allowed or source_type != "unknown")
    timestamps = (source.retrieved_at, source.observed_at or source.retrieved_at)
    chronology = all(_aware(stamp) and stamp <= moment for stamp in timestamps)
    quote_age = max((moment - stamp).total_seconds() for stamp in timestamps) if chronology else math.inf
    max_quote_age = _number(policy.max_quote_age_seconds if isinstance(policy, OntologyRiskPolicy) else rules.max_quote_age_seconds) or 0.0
    check("quote_freshness_check", chronology and quote_age <= max_quote_age)
    metadata["quote_age_seconds"] = quote_age if math.isfinite(quote_age) else None

    holdings = tuple(h for h in account.holdings if str(h.ticker).strip().upper() == symbol
                     and market_currency(h) == currency and h.is_short == short_cover)
    held_quantity = 0
    sellable_quantity = 0
    quantities_valid = True
    for holding in holdings:
        held = _whole(holding.quantity)
        sellable = held if holding.sellable_quantity is None else _whole(holding.sellable_quantity)
        if held is None or sellable is None:
            quantities_valid = False
        else:
            held_quantity += held
            sellable_quantity += min(held, sellable)
    check("holding_quantity_valid", quantities_valid)
    current_value = held_quantity * price
    cash = cash_available_for_market(account, market)
    report = build_portfolio_report(account)
    report_currency = "KRW" if account.total_equity_krw is not None or account.cash_equivalent_krw is not None else account.base_currency
    fx = currency_conversion_rate(account, report_currency, currency)
    equity = report.equity * fx if fx is not None else 0.0
    metadata.update(cash_available_for_market=cash, equity_for_sizing=equity, current_position_value=current_value)
    quantity = 0
    effective_cost = 0.0
    required_gross = 0.0
    actual_cost = None
    policy_current = False
    expires_at = intent.valid_until if _aware(intent.valid_until) else moment

    if opening:
        if isinstance(policy, OntologyRiskPolicy):
            metadata["ontology_risk_policy"] = policy.as_dict()
            try:
                policy_current = policy.symbol.upper() == symbol and policy.market == region and policy.valid_for_entry and policy.is_current(moment) and moment < policy.expires_at
            except (TypeError, ValueError):
                policy_current = False
        check("ontology_policy_current", policy_current, "ONTOLOGY_POLICY_UNAVAILABLE")
        check("market_supported", currency in {"KRW", "USD"})
        check("currency_valuation_complete", valuation_complete(account) and fx is not None and equity > 0)
        instrument = classify_instrument(symbol, market.company_name or None, market=market.market)
        check("instrument_permitted", (instrument.category == CATEGORY_ETF and rules.etf_trading_allowed)
              or (instrument.category != CATEGORY_ETF and instrument.tradable), "INSTRUMENT_NOT_PERMITTED")
        check("tradable_instrument_type", rules.warrant_unit_buys_allowed or not is_non_common_equity_ticker(symbol),
              "NON_COMMON_INSTRUMENT_BUY_BLOCKED")
        check("live_validation_id_present", not rules.live_trading_enabled or bool(intent.validation_id), "MISSING_VALIDATION_ID")
        weight = _number(intent.suggested_weight)
        check("requested_weight_valid", weight is not None and weight <= 1)
        expected_exit = _number(intent.expected_exit_price)
        check("expected_exit_price_present", expected_exit is not None and expected_exit > 0, "MISSING_EXPECTED_EXIT_PRICE")
        if policy_current:
            expires_at = min(expires_at, policy.expires_at)
            trade_count = _whole(trades_today)
            trade_limit = _whole(policy.max_trades_per_day)
            check("trade_count_known", trade_count is not None, "ONTOLOGY_TRADE_COUNT_UNAVAILABLE")
            metadata["observed_trade_count"] = trade_count
            operating = (policy.position_cap, policy.sector_cap, policy.trade_loss_budget_rate,
                         policy.daily_loss_budget_rate, policy.minimum_cash_reserve, policy.hard_stop_rate,
                         policy.soft_stop_rate, policy.net_profit_floor_rate, policy.minimum_reward_risk,
                         policy.all_in_cost_rate, policy.max_quote_age_seconds)
            check("ontology_policy_numeric_valid", all(_number(value) is not None for value in operating)
                  and 0 < policy.position_cap <= 1 and 0 < policy.sector_cap <= 1
                  and 0 <= policy.minimum_cash_reserve <= 1 and trade_limit is not None and trade_limit > 0)
            check("daily_loss_limit", report.daily_pnl_ratio > -policy.daily_loss_budget_rate)
            if trade_count is not None and trade_limit is not None:
                check("trade_count_limit", trade_count < trade_limit)
        if not reasons:
            # Integer sizing never rounds a strategy's requested total exposure
            # up to an unauthorised share. All bounds share the quote currency.
            target = equity * min(weight, policy.position_cap)
            prior_value = sum(float(h.market_value) for h in holdings)
            sector_weight = report.sector_weights.get(market.sector, 0.0) + (current_value - prior_value) / equity
            total_cash = report.cash_weight * equity
            spendable = max(0.0, min(cash, total_cash - equity * policy.minimum_cash_reserve))
            position_quantity = _lots(max(0.0, target - current_value) / price)
            sector_quantity = _lots(max(0.0, policy.sector_cap - sector_weight) * equity / price)
            effective_cost = policy.all_in_cost_rate
            risk_quantity = max(0, _lots(equity * policy.trade_loss_budget_rate / (price * max(policy.hard_stop_rate + effective_cost, 1e-12))) - held_quantity)
            quantity = min(position_quantity, sector_quantity, risk_quantity, _lots(spendable / price))
            metadata["ontology_quantity_bounds"] = {"position": position_quantity, "sector": sector_quantity, "loss_budget": risk_quantity}
            if quantity == 0:
                check("order_quantity_check", False, "ONTOLOGY_POSITION_BUDGET_EXCEEDED" if position_quantity == 0 else "ONTOLOGY_NO_EXECUTABLE_QUANTITY")
            venue = "KRX" if region == "KR" else ("NYSE" if "NYSE" in market.market.upper() else "AMEX" if "AMEX" in market.market.upper() else "NASD")
            for _ in range(3):
                if quantity <= 0:
                    break
                actual_cost = cost_engine.estimate(
                    symbol=symbol, market=market.market, venue=venue,
                    instrument_type="domestic_etf" if instrument.category == CATEGORY_ETF and region == "KR" else "domestic_stock" if region == "KR" else "overseas_stock",
                    entry_price=price, expected_exit_price=expected_exit, quantity=quantity,
                    target_net_return=policy.net_profit_floor_rate,
                    orderbook_snapshot=intent.strategy_metadata.get("orderbook_snapshot") if isinstance(intent.strategy_metadata.get("orderbook_snapshot"), dict) else None,
                    average_daily_trading_value=market.average_daily_trading_value,
                )
                rate = _number(actual_cost.total_cost_rate)
                entry_cost = _number(actual_cost.buy_fee + actual_cost.slippage_cost + actual_cost.spread_cost + actual_cost.market_impact_cost)
                if rate is None or entry_cost is None:
                    check("execution_cost_valid", False)
                    break
                effective_cost = max(policy.all_in_cost_rate, rate)
                cost_per_share = entry_cost / quantity
                available_quantity = min(_lots(spendable / (price + cost_per_share)),
                    max(0, _lots(equity * policy.trade_loss_budget_rate / (price * max(policy.hard_stop_rate + effective_cost, 1e-12))) - held_quantity))
                if quantity <= available_quantity:
                    break
                quantity = available_quantity
            check("order_quantity_check", quantity > 0, "ONTOLOGY_NO_EXECUTABLE_QUANTITY")
            if quantity > 0 and actual_cost is not None and actual_cost.quantity == quantity:
                notional = quantity * price
                entry_cost = actual_cost.buy_fee + actual_cost.slippage_cost + actual_cost.spread_cost + actual_cost.market_impact_cost
                required_gross = effective_cost + max(policy.net_profit_floor_rate, policy.soft_stop_rate * policy.minimum_reward_risk)
                check("ontology_expected_return", (expected_exit / price - 1) + 1e-12 >= required_gross, "POLICY_NET_REWARD_INSUFFICIENT")
                check("ontology_position_budget", current_value + notional <= equity * policy.position_cap + 1e-8, "ONTOLOGY_POSITION_BUDGET_EXCEEDED")
                check("max_sector_weight", sector_weight + notional / equity <= policy.sector_cap + 1e-12)
                check("ontology_trade_loss_budget", (current_value + notional) * (policy.hard_stop_rate + effective_cost) <= equity * policy.trade_loss_budget_rate + 1e-8)
                check("deposit_limit_check", notional + entry_cost <= cash + 1e-8)
                check("cash_available", total_cash - notional - entry_cost >= equity * policy.minimum_cash_reserve - 1e-8)
                metadata["cost_breakdown"] = actual_cost.as_dict()
            elif quantity > 0:
                check("execution_cost_valid", False)
    else:
        # Only executable resource/contract checks apply to an owned-position
        # exit. A missing entry policy, loss, sector excess or model uncertainty
        # cannot prevent reducing exposure.
        check("holding_exists", held_quantity > 0)
        if long_close and price > 0:
            if intent.action == OrderAction.SELL:
                quantity = sellable_quantity
            else:
                target_weight = _number(intent.suggested_weight)
                check("requested_weight_valid", target_weight is not None and target_weight <= 1)
                check("currency_valuation_complete", valuation_complete(account) and fx is not None and equity > 0)
                if target_weight is not None:
                    # Retaining floor(target / price) shares is the nearest
                    # executable exposure at or below the requested target.
                    quantity = min(sellable_quantity, max(0, held_quantity - _lots(equity * target_weight / price)))
        elif short_cover:
            meta = intent.strategy_metadata
            quantity = _whole(meta.get("cover_quantity", meta.get("quantity"))) or 0
            loan_date = str(meta.get("loan_date") or "").strip()
            lot_quantity = sum(_whole(h.quantity) or 0 for h in holdings if h.loan_date == loan_date)
            check("short_close_contract_valid", bool(loan_date) and 0 < quantity <= lot_quantity, "SHORT_COVER_CONTRACT_INVALID")
            check("deposit_limit_check", quantity * price <= cash)
        check("order_quantity_check", quantity > 0, "NO_SELLABLE_OR_COVER_QUANTITY")

    approved = not reasons and quantity > 0
    final_order = FinalOrder(
        ticker=intent.ticker, market=intent.market, order_type=OrderType.LIMIT,
        side=OrderSide.BUY if opening or short_cover else OrderSide.SELL,
        quantity=quantity, limit_price=price, manual_approval_required=rules.manual_approval_required,
        position_direction=direction, position_effect=effect, execution_product=product,
        credit_type="26" if short_cover else None,
        loan_date=str(intent.strategy_metadata.get("loan_date")) if short_cover else None,
    ) if approved else None
    receipt = {"authority_id": AUTHORITY_ID, "phase": "entry_assessment" if opening else "exposure_reduction",
               "approved": approved, "policy_id": policy.policy_id if isinstance(policy, OntologyRiskPolicy) else None,
               "evaluated_at": moment.isoformat(), "expires_at": expires_at.isoformat(),
               "symbol": symbol, "market": region, "side": "BUY" if opening or short_cover else "SELL",
               "position_direction": direction, "position_effect": effect, "execution_product": product,
               "quantity": quantity if approved else 0, "actual_quantity": quantity if approved else 0,
               "authorized_price": price, "actual_notional": quantity * price if approved else 0.0,
               "all_in_cost_rate": effective_cost, "required_gross_return": required_gross,
               "checks": dict(checks), "reason_codes": list(reasons)}
    receipt["evaluation_id"] = "ora-" + hashlib.sha256(json.dumps(receipt, sort_keys=True, allow_nan=False).encode()).hexdigest()[:24]
    metadata["ontology_risk_authority"] = receipt
    if reasons:
        metadata["rejection_log"] = tuple({"reason": reason, "check": "ontology_risk_authority"} for reason in reasons)
    adjusted_weight = (current_value + quantity * price) / equity if approved and opening and equity > 0 else intent.suggested_weight if approved else None
    return RiskManagerResult(intent.ticker, intent.action, approved, adjusted_weight, checks, tuple(reasons), final_order, metadata)

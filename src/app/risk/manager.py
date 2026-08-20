from __future__ import annotations

import os
from datetime import datetime, timezone
from math import floor
from pathlib import Path

import math

from app.audit import AuditLogger, log_path
from app.cost import ProfitabilityGate, ProfitabilityInput, TradingCostEngine
from app.data.instrument_eligibility import CATEGORY_DERIVATIVE as INSTRUMENT_DERIVATIVE
from app.data.instrument_eligibility import CATEGORY_ETN as INSTRUMENT_ETN
from app.data.instrument_eligibility import CATEGORY_LEVERAGED_ETP as INSTRUMENT_LEVERAGED_ETP
from app.data.instrument_eligibility import classify as classify_instrument
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
        self._profitability_gate = ProfitabilityGate(cost_engine=self.cost_engine)
        self.principal_protection = PrincipalProtectionEngine()
        # A RiskManager built without an explicit logger used to write straight into
        # the operator's principal-protection journal, which every test that
        # constructs one then did too. See app.audit.logger.log_path.
        self.audit_logger = audit_logger or AuditLogger(log_path("principal-protection.jsonl"))

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
        # --- Product / direction policy ----------------------------------------
        # The old single ``restricted_products_blocked`` check demanded that ALL of
        # margin, short selling, derivatives, leverage ETFs and credit be disallowed,
        # for every order. That made the flags un-usable as policy: enabling short
        # selling would have simultaneously permitted derivatives and leveraged ETFs,
        # because one boolean covered five unrelated products. Decomposed so each
        # product is gated independently and only when the order actually uses it.
        direction = _parse_direction(intent)
        effect = _parse_effect(intent)
        product = str(getattr(intent, "execution_product", "CASH") or "CASH").upper()
        is_short_entry = direction == "SHORT" and effect == "OPEN"
        is_short_exit = direction == "SHORT" and effect == "CLOSE"
        metadata["position_direction"] = direction
        metadata["position_effect"] = effect
        metadata["execution_product"] = product
        # Products this system has no execution path for at all. Still a global
        # assertion (as the original combined check was), because there is no
        # instrument classification on the intent to make it per-order — enabling one
        # of these flags would authorise orders the execution layer cannot construct.
        # Short selling and credit borrow are removed from this set precisely because
        # they now DO have an execution path and their own per-order gates below.
        checks["unsupported_products_blocked"] = (
            not self.rules.margin_trading_allowed
            and not self.rules.derivatives_allowed
            and not self.rules.leverage_etf_allowed
        )
        # The per-order half the comment above used to say was impossible. It is now
        # possible because ``app.data.instrument_eligibility`` can classify a listing
        # from its code and name, so the two product flags finally bind on what the
        # order actually buys instead of on a global assertion.
        #
        # This is a backstop, not the filter: discovery already removes these
        # instruments in ``fetch_domestic_ranking_symbols`` / ``resolve_universe``.
        # It exists because a candidate can also arrive from a held position, the
        # static collection list, or a manual intent, and none of those pass through
        # discovery. A leveraged ETP reaching this point is not a worse trade — it is
        # an order the account is not authorised to place at all.
        # The name lives on the market snapshot, not the intent. When it is absent the
        # classifier reports UNKNOWN and permits — see its module docstring on why
        # refusing every unnamed instrument is the wrong failure mode here.
        instrument = classify_instrument(
            str(getattr(intent, "ticker", "") or ""),
            str(getattr(market, "company_name", "") or "") or None,
            market=str(getattr(intent, "market", "") or getattr(market, "market", "") or "") or None,
        )
        metadata["instrument_category"] = instrument.category
        if instrument.tradable:
            checks["instrument_permitted"] = True
        elif instrument.category == INSTRUMENT_LEVERAGED_ETP:
            checks["instrument_permitted"] = bool(self.rules.leverage_etf_allowed)
        elif instrument.category in {INSTRUMENT_DERIVATIVE, INSTRUMENT_ETN}:
            checks["instrument_permitted"] = bool(self.rules.derivatives_allowed)
        else:
            # An unsupported code shape is not a permission question: the execution
            # layer cannot construct the order whatever the account may hold.
            checks["instrument_permitted"] = False
        if not checks["instrument_permitted"]:
            metadata["instrument_reason_codes"] = list(instrument.reason_codes)
        # These two are the real gates, and they only bind on an order that needs them.
        # A SHORT EXIT is deliberately NOT gated: refusing to cover a short because the
        # account-level short flag was turned off would trap an open, unbounded-loss
        # position. De-risking is never blocked here, exactly as SELL/REDUCE of a long
        # never was.
        checks["short_selling_policy_check"] = (
            not is_short_entry or self.rules.short_selling_allowed
        )
        checks["credit_borrow_policy_check"] = (
            product != "CREDIT_BORROW" or not is_short_entry or self.rules.credit_loan_allowed
        )
        # Every field of the order contract must be present and mutually consistent.
        # An order that cannot say what it is must not be sent.
        checks["order_contract_complete"] = _contract_consistent(intent, direction, effect, product)
        if is_short_entry or is_short_exit:
            short_checks, short_metadata = self._short_checks(
                intent, account, market, direction, effect, product
            )
            checks.update(short_checks)
            metadata.update(short_metadata)
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
        # Block BUYs of warrants / units / rights (non-common equity). These are
        # typically illiquid SPAC/IPO instruments that cannot be reliably exited
        # (e.g. a warrant whose order book is empty), so buying them traps capital.
        # SELL/REDUCE of an already-held position is never blocked here.
        checks["tradable_instrument_type"] = (
            self.rules.warrant_unit_buys_allowed
            or intent.action != OrderAction.BUY
            or not _is_non_common_equity_ticker(intent.ticker)
        )
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
        # --- SHORT ENTRY -------------------------------------------------------- #
        # Routed before the long branches because its broker side is SELL, which the
        # existing SELL/REDUCE branch would treat as "liquidate a holding" and reject
        # for having no position. Sizing is the minimum of four independent bounds, so
        # the tightest always wins and no single miscalculation can widen the position.
        if approved and is_short_entry:
            short_quantity, short_metadata = self._short_quantity(
                intent, account, market, adjusted_weight
            )
            metadata.update(short_metadata)
            if short_quantity <= 0:
                approved = False
                _add_rejection(
                    reasons,
                    rejection_log,
                    "SHORT_SIZING_YIELDED_ZERO_QUANTITY",
                    "short_position_sizing",
                    short_metadata,
                )
            else:
                decision = self._profitability_gate.evaluate(
                    ProfitabilityInput(
                        symbol=intent.ticker,
                        action="SELL",
                        market=intent.market,
                        venue=venue,
                        instrument_type=instrument_type,
                        entry_price=market.last_price,
                        expected_exit_price=float(intent.expected_exit_price or 0.0),
                        quantity=short_quantity,
                        liquidity_score=_liquidity_score_from_market(market),
                        orderbook_snapshot=(
                            intent.strategy_metadata.get("orderbook_snapshot")
                            if isinstance(
                                intent.strategy_metadata.get("orderbook_snapshot"), dict
                            )
                            else None
                        ),
                        average_daily_trading_value=market.average_daily_trading_value,
                        account_equity_krw=float(getattr(account, "equity", 0.0) or 0.0),
                        target_net_return=intent.target_net_return,
                        position_direction="SHORT",
                        position_effect="OPEN",
                        execution_product="CREDIT_BORROW",
                        borrow_fee_bps_annualised=_optional_metadata_float(
                            intent, "borrow_fee_bps_annualised"
                        ),
                        expected_holding_seconds=_optional_metadata_float(
                            intent, "expected_holding_seconds"
                        ),
                        borrow_uncertainty_buffer_bps=_optional_metadata_float(
                            intent, "borrow_uncertainty_buffer_bps"
                        ),
                        recall_risk_buffer_bps=_optional_metadata_float(
                            intent, "recall_risk_buffer_bps"
                        ),
                        minimum_cost_coverage_ratio_override=_optional_metadata_float(
                            intent, "minimum_cost_coverage_ratio"
                        ),
                    )
                )
                metadata["profitability_decision"] = decision.as_dict()
                checks["profitability_gate"] = decision.allowed
                if not decision.allowed:
                    for reason in decision.rejection_reasons:
                        _add_rejection(reasons, rejection_log, reason, "profitability_gate")
                    approved = False
            if approved:
                final_order = _final_order_or_reject(
                    intent,
                    market,
                    OrderSide.SELL,
                    short_quantity,
                    reasons,
                    self.rules.manual_approval_required,
                    position_direction="SHORT",
                    position_effect="OPEN",
                    execution_product="CREDIT_BORROW",
                    credit_type="22",
                )
                approved = final_order is not None
        elif approved and is_short_exit:
            # Buy-to-cover. Quantity comes from the borrow lot, and ``loan_date`` is
            # mandatory: without it the execution layer cannot say which lot is being
            # repaid and refuses the order rather than letting the broker pick.
            cover_quantity = max(
                0, int(_metadata_value(intent, "cover_quantity") or intent_quantity_hint(intent))
            )
            loan_date = str(_metadata_value(intent, "loan_date") or "").strip() or None
            if not loan_date:
                approved = False
                _add_rejection(
                    reasons, rejection_log, "SHORT_LOAN_DATE_MISSING", "credit_order_contract"
                )
            elif cover_quantity <= 0:
                approved = False
                _add_rejection(
                    reasons, rejection_log, "SHORT_COVER_QUANTITY_UNKNOWN", "credit_order_contract"
                )
            else:
                final_order = _final_order_or_reject(
                    intent,
                    market,
                    OrderSide.BUY,
                    cover_quantity,
                    reasons,
                    self.rules.manual_approval_required,
                    position_direction="SHORT",
                    position_effect="CLOSE",
                    execution_product="CREDIT_BORROW",
                    credit_type="26",
                    loan_date=loan_date,
                )
                approved = final_order is not None
        elif approved and intent.action == OrderAction.BUY:
            spend = buy_amount
            quantity = floor(spend / market.last_price)
            metadata["estimated_order_quantity"] = quantity
            metadata["minimum_one_share_cash_required"] = float(market.last_price)
            small_account_enabled = _env_bool("REALTIME_SMALL_ACCOUNT_MODE", False)
            small_account_equity = _env_float("REALTIME_SMALL_ACCOUNT_EQUITY_KRW", 300000.0)
            max_position_weight = _env_float("REALTIME_SMALL_ACCOUNT_MAX_POSITION_WEIGHT", 0.10)
            account_equity = max(0.0, float(getattr(account, "equity", 0.0) or 0.0))
            if (
                small_account_enabled
                and account_equity > 0.0
                and account_equity <= small_account_equity
                and max_position_weight > 0.0
                and market.last_price > account_equity * max_position_weight
            ):
                _add_rejection(
                    reasons,
                    rejection_log,
                    "SMALL_ACCOUNT_ONE_SHARE_TOO_EXPENSIVE",
                    "small_account_position_sizing",
                    {
                        "equity": account_equity,
                        "last_price": market.last_price,
                        "max_position_weight": max_position_weight,
                        "max_one_share_price": account_equity * max_position_weight,
                    },
                )
                approved = False
            if approved and quantity > 0:
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
                checks["cost_adjusted_cash_available"] = (
                    spend + cost.buy_fee + cost.slippage_cost + cost.spread_cost + cost.market_impact_cost
                    <= cash_available_for_market
                )
                if not checks["cost_adjusted_cash_available"]:
                    _add_rejection(
                        reasons,
                        rejection_log,
                        "cost_adjusted_cash_available",
                        "cost_adjusted_cash_available",
                        {"required_cash": spend + cost.buy_fee + cost.slippage_cost + cost.spread_cost + cost.market_impact_cost},
                    )
                    approved = False
                # Unified net-profitability authority. One ProfitabilityGate replaces the
                # previously fragmented net/target/break-even/cost-to-alpha/spread/slippage
                # checks so candidate generation, the RiskManager, the realtime engine, and
                # the GUI all judge a BUY by exactly the same cost-aware net-edge rule.
                # Realized volatility is left at 0.0 here (the realtime engine supplies the
                # short-horizon volatility buffer); this backstop therefore never rejects a
                # trade the engine's own stricter gate already approved.
                decision = self._profitability_gate.evaluate(
                    ProfitabilityInput(
                        symbol=intent.ticker,
                        action="BUY",
                        market=intent.market,
                        venue=venue,
                        instrument_type=instrument_type,
                        entry_price=market.last_price,
                        expected_exit_price=float(intent.expected_exit_price),
                        quantity=quantity,
                        spread_rate=None,
                        liquidity_score=_liquidity_score_from_market(market),
                        realized_volatility=0.0,
                        orderbook_snapshot=orderbook_snapshot if isinstance(orderbook_snapshot, dict) else None,
                        average_daily_trading_value=market.average_daily_trading_value,
                        account_equity_krw=float(getattr(account, "equity", 0.0) or 0.0),
                        target_net_return=intent.target_net_return,
                    )
                )
                metadata["profitability_decision"] = decision.as_dict()
                checks["profitability_gate"] = decision.allowed
                if not decision.allowed:
                    for reason in decision.rejection_reasons:
                        _add_rejection(
                            reasons,
                            rejection_log,
                            reason,
                            "profitability_gate",
                            {
                                "net_expected_return": decision.net_expected_return,
                                "required_min_net_return": decision.required_min_net_return,
                                "gross_expected_return": decision.gross_expected_return,
                                "break_even_exit_price": decision.break_even_exit_price,
                                "spread_rate": decision.spread_rate,
                                "cost_to_alpha_ratio": decision.cost_to_alpha_ratio,
                            },
                        )
                    approved = False
            elif quantity <= 0:
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

    def _short_checks(
        self,
        intent: OrderIntent,
        account: AccountSnapshot,
        market: MarketSnapshot,
        direction: str,
        effect: str,
        product: str,
    ) -> tuple[dict[str, bool], dict[str, object]]:
        """Risk checks that only exist for a short position.

        Every one of these has no long-side analogue, and each guards a failure mode
        that is specific to owing shares rather than owning them:

        * borrow availability / quantity / cost — a short with no locate is not a
          worse trade, it is an order the broker rejects or force-closes;
        * loan-date integrity — a cover cannot be routed without 대출일;
        * position / exposure limits — a short's loss is unbounded ABOVE, so the
          concentration question is different from a long's;
        * recall deadline — the lender, not the strategy, may decide the exit time;
        * stop-order capability — the one exit that must always be placeable.

        A CLOSE (buy-to-cover) is checked only for contract integrity. Blocking a
        cover on a risk limit would strand an open unbounded-loss position, which is
        strictly worse than any limit it might breach.
        """
        checks: dict[str, bool] = {}
        metadata: dict[str, object] = {}
        is_entry = effect == "OPEN"
        metadata_prefix = "short"

        if not is_entry:
            # Covering. The only thing that can stop it is an incomplete contract,
            # and that is a bug to surface rather than a risk to accept.
            checks["short_close_contract_valid"] = bool(
                str(getattr(intent, "strategy_metadata", {}).get("loan_date", "") or "").strip()
                or str(getattr(intent, "loan_date", "") or "").strip()
            )
            return checks, metadata

        rules = self.rules
        meta = getattr(intent, "strategy_metadata", {}) or {}
        symbol = str(intent.ticker or "").upper()
        now = datetime.now(timezone.utc)

        # Borrow locate, read from the point-in-time journal through the SAME
        # ``evaluate_borrow`` rule the shadow simulator and the KIS preflight use. Three
        # implementations of one rule would drift, and the drift would show up as
        # shadow results that do not reproduce live.
        snapshot = None
        verdict_reasons: tuple[str, ...] = ()
        try:
            from app.trading.borrow import (
                DEFAULT_MAX_BORROW_FEE_BPS_ANNUALISED,
                default_borrow_store,
                evaluate_borrow,
            )
            from app.trading.directional import ShortReasonCodes

            requested_quantity = max(1, int(meta.get("intended_quantity") or 1))
            store = default_borrow_store()
            snapshot = store.latest(symbol, as_of=now)
            # Inventory can disappear between periodic polling and submission. Do a
            # demand-driven JIT refresh here, at the final risk gate. If it fails, the
            # old snapshot is evaluated against the same short TTL and therefore
            # fails closed rather than becoming a silent fallback.
            jit_max_age = max(
                1.0,
                float(os.getenv("KIS_BORROW_JIT_MAX_AGE_SECONDS", "5.0")),
            )
            if snapshot is None or not snapshot.is_fresh(now, jit_max_age):
                from app.trading.borrow_source import default_borrow_source

                refreshed = default_borrow_source().snapshot(symbol, now=now)
                if refreshed is not None:
                    store.record(refreshed)
                    snapshot = refreshed
            verdict = evaluate_borrow(
                snapshot,
                quantity=requested_quantity,
                now=now,
                max_fee_bps_annualised=float(
                    meta.get("max_borrow_fee_bps_annualised")
                    or DEFAULT_MAX_BORROW_FEE_BPS_ANNUALISED
                ),
                max_age_seconds=min(
                    rules.min_borrow_snapshot_freshness_seconds,
                    jit_max_age,
                ),
                min_hours_before_deadline=rules.min_hours_before_recall_deadline,
            )
            verdict_reasons = verdict.reason_codes
            borrow_ok = verdict.allowed
        except Exception:  # noqa: BLE001 - an unavailable borrow store is a no-locate.
            from app.trading.directional import ShortReasonCodes

            borrow_ok = False
            verdict_reasons = (ShortReasonCodes.BORROW_LOOKUP_FAILED,)
        metadata[f"{metadata_prefix}_borrow_verdict"] = list(verdict_reasons)
        if snapshot is not None:
            metadata[f"{metadata_prefix}_borrow_snapshot"] = snapshot.as_dict()

        # Decomposed so a rejection names the actual cause instead of one opaque
        # "borrow failed". Each is derived from the SHARED verdict's reason codes, so a
        # sub-check can never disagree with the composite below.
        #
        # ``borrow_preflight_check`` is the composite and it is what actually binds: if
        # a future edit adds a reason code to ``evaluate_borrow`` without adding a row
        # here, the composite still fails closed rather than the new cause being
        # silently dropped.
        R = ShortReasonCodes
        for check_name, code in (
            ("borrow_availability_check", R.BORROW_UNAVAILABLE),
            ("borrow_quantity_check", R.BORROW_QUANTITY_INSUFFICIENT),
            ("borrow_cost_check", R.BORROW_COST_TOO_HIGH),
            ("borrow_snapshot_freshness_check", R.BORROW_SNAPSHOT_STALE),
            ("borrow_lookup_check", R.BORROW_LOOKUP_FAILED),
            ("recall_deadline_check", R.RECALL_DEADLINE_NEAR),
        ):
            checks[check_name] = code not in verdict_reasons
        checks["borrow_preflight_check"] = bool(borrow_ok)

        # Exposure. Computed from SIGNED exposure so a long and a short in the same
        # name net against each other for net exposure while both count toward gross —
        # which is the whole reason the two limits are separate.
        holdings = tuple(getattr(account, "holdings", ()) or ())
        equity = max(1e-9, float(getattr(account, "equity", 0.0) or 0.0))
        open_shorts = [h for h in holdings if getattr(h, "is_short", False)]
        gross_exposure = sum(abs(float(getattr(h, "market_value", 0.0) or 0.0)) for h in holdings)
        short_exposure = sum(
            abs(float(getattr(h, "market_value", 0.0) or 0.0)) for h in open_shorts
        )
        prospective_weight = min(
            float(intent.suggested_weight or 0.0), rules.max_single_short_weight
        )
        metadata[f"{metadata_prefix}_open_position_count"] = len(open_shorts)
        metadata[f"{metadata_prefix}_total_weight"] = short_exposure / equity
        metadata[f"{metadata_prefix}_gross_exposure_ratio"] = gross_exposure / equity
        checks["short_position_limit_check"] = len(open_shorts) < rules.max_open_short_positions
        checks["short_single_weight_check"] = (
            prospective_weight <= rules.max_single_short_weight + 1e-9
        )
        checks["short_total_weight_check"] = (
            (short_exposure / equity) + prospective_weight
            <= rules.max_total_short_weight + 1e-9
        )
        checks["gross_exposure_check"] = (
            (gross_exposure / equity) + prospective_weight <= rules.max_gross_exposure + 1e-9
        )
        checks["net_exposure_check"] = (
            (short_exposure / equity) + prospective_weight
            <= rules.max_net_short_exposure + 1e-9
        )
        # Crowding. A squeeze is where a short's unbounded loss actually materialises,
        # so an explicitly-flagged squeeze risk is a refusal rather than a penalty.
        squeeze = meta.get("short_squeeze_risk")
        checks["short_squeeze_risk_check"] = not bool(squeeze)
        # Daily loss budget, measured against the SHORT-specific stop, which is half
        # the account-wide one.
        report = build_portfolio_report(account)
        checks["short_daily_loss_check"] = (
            report.daily_pnl_ratio > -rules.short_daily_loss_stop
        )
        # A short we cannot place a protective stop for is a short we cannot bound.
        # ``None`` (nobody said) fails when the capability is required, because the
        # capability has to be demonstrated rather than assumed.
        stop_capable = meta.get("stop_order_capable")
        checks["short_stop_order_capability_check"] = (
            not rules.require_short_stop_order_capability or stop_capable is True
        )
        # Overnight carry. Every one of these strategies is intraday by construction
        # and the geometry table gives none of them an overnight horizon, so an
        # order that would be held past the close is a policy violation.
        checks["overnight_short_check"] = bool(
            rules.overnight_short_allowed
            or int(meta.get("expected_holding_minutes") or 0) <= rules.max_short_holding_minutes
        )
        return checks, metadata

    def _short_quantity(
        self,
        intent: OrderIntent,
        account: AccountSnapshot,
        market: MarketSnapshot,
        adjusted_weight: float,
    ) -> tuple[int, dict[str, object]]:
        """Short size = min of four independent bounds.

            min(position_sizer, borrow_available, state_limit, risk_budget)

        A minimum rather than a blend, so the tightest bound always wins and no single
        miscalculation can widen the position. Each bound answers a different question
        and none subsumes the others:

        * **position sizer** — how much this weight implies at this price;
        * **borrow available** — how much stock actually exists to borrow;
        * **state limit** — how much this deployment rung permits (LIVE_PROBE: 1 share);
        * **risk budget** — how much loss the account tolerates on this trade, at HALF
          a long's per-trade budget because the downside is unbounded and accelerates.

        Loss-streak scaling halves the size from two consecutive losses and stops new
        entries at four. That is stricter than the long path's, for the same reason.
        """
        meta = getattr(intent, "strategy_metadata", {}) or {}
        rules = self.rules
        price = max(1e-9, float(market.last_price or 0.0))
        equity = max(0.0, float(getattr(account, "equity", 0.0) or 0.0))
        bounds: dict[str, int] = {}

        weight = min(float(adjusted_weight or 0.0), rules.max_single_short_weight)
        bounds["position_sizer"] = int(max(0.0, equity * weight) // price)
        bounds["borrow_available"] = max(0, int(meta.get("borrow_available_quantity") or 0))
        bounds["state_limit"] = max(0, int(meta.get("state_max_quantity") or 0))
        # Risk budget: (equity * per-trade budget * short fraction) / stop distance.
        stop_rate = max(1e-6, float(meta.get("stop_loss_rate") or 0.0))
        per_trade = float(rules.principal_protection.per_trade_risk_budget_ratio or 0.0025)
        risk_capital = equity * per_trade * rules.short_risk_budget_ratio_of_long
        bounds["risk_budget"] = int(max(0.0, risk_capital / (price * stop_rate)))

        quantity = min(bounds.values()) if bounds else 0
        # Loss-streak scaling. Read from the DIRECTIONAL store so a long streak cannot
        # shrink a short (or vice versa) — they are different arms with different
        # evidence.
        streak = 0
        try:
            from app.trading.strategy_performance_store import default_store

            streak = default_store().loss_streak(
                str(intent.strategy_family or intent.signal_name or ""),
                market=intent.market,
                direction="SHORT",
            )
        except Exception:  # noqa: BLE001 - an unavailable store must not widen size.
            streak = 0
        if streak >= 4:
            quantity = 0
        elif streak >= 2:
            quantity = quantity // 2
        return max(0, quantity), {
            "short_size_bounds": bounds,
            "short_loss_streak": streak,
            "short_quantity": max(0, quantity),
        }

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
    *,
    position_direction: str = "LONG",
    position_effect: str = "OPEN",
    execution_product: str = "CASH",
    credit_type: str | None = None,
    loan_date: str | None = None,
) -> FinalOrder | None:
    if quantity <= 0:
        reasons.append("INSUFFICIENT_CASH_FOR_ONE_SHARE")
        return None
    # NOTE: market.last_price is only a *reference* price for sizing and gating — the
    # last trade print is NOT an executable price (a BUY posted at it may sit behind the
    # ask and never fill; a stop SELL posted at it may fail to exit). Live execution MUST
    # re-price this FinalOrder from the current order book, side, and exit urgency before
    # submission — see app.execution.order_pricing_policy.ExecutionPricingPolicy, applied
    # in RealtimeTradingEngine._prepare_order_for_execution.
    return FinalOrder(
        ticker=intent.ticker,
        market=intent.market,
        order_type=OrderType.LIMIT,
        side=side,
        quantity=quantity,
        limit_price=market.last_price,
        manual_approval_required=manual_approval_required,
        position_direction=position_direction,
        position_effect=position_effect,
        execution_product=execution_product,
        credit_type=credit_type,
        loan_date=loan_date,
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


def _liquidity_score_from_market(market: MarketSnapshot) -> float:
    """Normalize average-daily-trading-value into a 0..1 liquidity score.

    Same log-scaled mapping the realtime engine uses so the ProfitabilityGate sees
    a consistent liquidity input regardless of which caller invokes it.
    """
    adtv = max(0.0, float(getattr(market, "average_daily_trading_value", 0.0) or 0.0))
    return min(1.0, math.log1p(adtv) / math.log1p(10_000_000_000))


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


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def _metadata_value(intent: OrderIntent, key: str) -> object | None:
    """Read a key from ``strategy_metadata``, tolerating a missing mapping."""
    meta = getattr(intent, "strategy_metadata", None)
    if not isinstance(meta, dict):
        return None
    return meta.get(key)


def _optional_metadata_float(intent: OrderIntent, key: str) -> float | None:
    """``None``-preserving float from strategy metadata.

    Preserving ``None`` is the point: a borrow fee that is absent must reach the
    ProfitabilityGate as absent so the gate can reject it, rather than arriving as
    0.0 and being priced as a free borrow.
    """
    value = _metadata_value(intent, key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def intent_quantity_hint(intent: OrderIntent) -> int:
    """Quantity a cover should repay, from metadata. Zero when unstated.

    Zero, not a guess: covering the wrong quantity leaves either a residual short or
    an accidental long.
    """
    value = _metadata_value(intent, "quantity")
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _parse_direction(intent: OrderIntent) -> str:
    """Direction of the intent, defaulting to LONG.

    LONG is the safe default: it is what every caller that predates shorts means, and
    defaulting to SHORT would reclassify the entire existing order path.
    """
    value = str(getattr(intent, "position_direction", "") or "").strip().upper()
    return "SHORT" if value == "SHORT" else "LONG"


def _parse_effect(intent: OrderIntent) -> str:
    """OPEN / CLOSE for this intent. Delegates to the domain contract.

    Kept as a module function so the RiskManager reads the same resolution the rest of
    the system does, rather than re-deriving it and risking a disagreement.
    """
    resolver = getattr(intent, "resolved_position_effect", None)
    if isinstance(resolver, str):
        return resolver
    # Duck-typed intents (tests, adapters) that lack the property.
    value = str(getattr(intent, "position_effect", "") or "").strip().upper()
    if value in {"OPEN", "CLOSE"}:
        return value
    direction = _parse_direction(intent)
    opening_action = OrderAction.SELL if direction == "SHORT" else OrderAction.BUY
    return "OPEN" if getattr(intent, "action", None) == opening_action else "CLOSE"


def _contract_consistent(
    intent: OrderIntent, direction: str, effect: str, product: str
) -> bool:
    """Is the (action, direction, effect, product) tuple internally consistent?

    Catches the two contradictions that would place the wrong order:

    * an action that does not match the (direction, effect) pair — e.g. a SHORT/OPEN
      carrying BUY, which would open a LONG;
    * a SHORT settled as CASH, which has no borrow and therefore no way to deliver.
    """
    from app.trading.directional import (
        PositionDirection,
        PositionEffect,
        broker_side,
    )

    if intent.action not in {OrderAction.BUY, OrderAction.SELL}:
        # HOLD / WATCH / REBALANCE carry no side, so there is nothing to contradict.
        return True
    expected = broker_side(PositionDirection(direction), PositionEffect(effect))
    if str(intent.action) != expected:
        return False
    if direction == "SHORT" and product != "CREDIT_BORROW":
        return False
    if direction == "LONG" and product not in {"CASH", ""}:
        return False
    return True


def _reason_for_failed_check(check: str) -> str:
    return {
        "expected_exit_price_present": "MISSING_EXPECTED_EXIT_PRICE",
        "strategy_family_present": "MISSING_STRATEGY_FAMILY",
        "live_validation_id_present": "MISSING_VALIDATION_ID",
        "ontology_trade_not_forbidden": "ONTOLOGY_TRADE_FORBIDDEN",
        "tradable_instrument_type": "NON_COMMON_INSTRUMENT_BUY_BLOCKED",
        # --- short-side ---------------------------------------------------------
        "short_selling_policy_check": "SHORT_SELLING_NOT_PERMITTED_BY_POLICY",
        "credit_borrow_policy_check": "CREDIT_BORROW_NOT_PERMITTED_BY_POLICY",
        "order_contract_complete": "SHORT_ORDER_CONTRACT_INCOMPLETE",
        "short_close_contract_valid": "SHORT_LOAN_DATE_MISSING",
        "borrow_availability_check": "SHORT_BORROW_UNAVAILABLE",
        "borrow_quantity_check": "SHORT_BORROW_QUANTITY_INSUFFICIENT",
        "borrow_cost_check": "SHORT_BORROW_COST_TOO_HIGH",
        "borrow_snapshot_freshness_check": "SHORT_BORROW_SNAPSHOT_STALE",
        "borrow_preflight_check": "SHORT_BORROW_UNAVAILABLE",
        "borrow_lookup_check": "SHORT_BORROW_LOOKUP_FAILED",
        "recall_deadline_check": "SHORT_RECALL_DEADLINE_NEAR",
        "short_position_limit_check": "SHORT_MAX_OPEN_POSITIONS_REACHED",
        "short_single_weight_check": "SHORT_SINGLE_POSITION_WEIGHT_EXCEEDED",
        "short_total_weight_check": "SHORT_TOTAL_WEIGHT_EXCEEDED",
        "gross_exposure_check": "GROSS_EXPOSURE_LIMIT_EXCEEDED",
        "net_exposure_check": "NET_SHORT_EXPOSURE_LIMIT_EXCEEDED",
        "short_squeeze_risk_check": "SHORT_SQUEEZE_RISK_TOO_HIGH",
        "short_daily_loss_check": "SHORT_DAILY_LOSS_LIMIT_EXCEEDED",
        "short_stop_order_capability_check": "SHORT_STOP_ORDER_CAPABILITY_MISSING",
        "overnight_short_check": "OVERNIGHT_SHORT_NOT_PERMITTED",
        "unsupported_products_blocked": "restricted_products_blocked",
    }.get(check, check)


def _is_non_common_equity_ticker(ticker: str) -> bool:
    """Heuristic: does the ticker denote a warrant / unit / right (non-common equity)?

    Targets US/overseas SPAC-style securities that are typically illiquid and hard
    to exit (e.g. Locafy warrant ``LCFYW``, a unit ``LAFAU``). Uses the NASDAQ
    5th-letter convention (5 alphabetic chars ending in W=warrant, U=unit, R=right)
    plus explicit suffix forms (``.WS``, ``-WT``, ``-UN``, ``.RT`` …). Korean numeric
    codes and ordinary <=4-letter tickers are not affected.
    """
    symbol = str(ticker or "").upper().strip()
    if not symbol:
        return False
    suffix_markers = (
        ".WS", "-WT", ".WT", "/WS", "-WS", "+",       # warrants
        ".U", "-UN", ".UN", "-U", "/U",               # units
        ".RT", "-RT", ".RTS", "-RTS", "/R",           # rights
    )
    for marker in suffix_markers:
        if symbol.endswith(marker):
            return True
    # NASDAQ 5th-letter suffix convention (only for clean 5-letter alpha symbols).
    if symbol.isalpha() and len(symbol) == 5 and symbol[-1] in {"W", "U", "R"}:
        return True
    return False


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

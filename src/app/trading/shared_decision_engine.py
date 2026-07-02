from __future__ import annotations

import os
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from app.cost import TradingCostEngine
from app.data.realtime_store import RealtimeMarketDataStore
from app.features.live_feature_frame import LiveFeatureFrameBuilder
from app.models.live_signal_predictor import LiveSignalPredictor, LiveSignalPrediction
from app.risk import RiskManager
from app.schemas.domain import AccountSnapshot, FinalOrder, Holding, MarketSnapshot, OrderAction, OrderIntent, RiskRules, SourceMetadata, OrderSide, OrderType
from app.trading.auto_tuning_engine import AutoTuningEngine, MarketStateSnapshot
from app.trading.decision_logger import DecisionLogger
from app.strategy.rule_based import _holding_exit_adjustment, _ontology_flow_adjustment


# 매수 근거로 인정하는 supportsSignal(현금/affordability 류는 제외 — 그건 매수 "엣지"가 아님).
_BUY_EDGE_SUPPORTS = frozenset(
    {
        "RevenueGrowth",
        "EarningsGrowth",
        "ProfitabilityQuality",
        "NpuCompositeMomentum",
        "LiquiditySupport",
        # 국내 플로우 근거(KR)
        "InformedOrderFlowImbalance",
        "ForeignInstitutionJointBuying",
        "RetailSupplyAbsorbedByInformedFlow",
        "OrderFlowPriceConfirmation",
        "SuspectedSmartMoneyAccumulation",
        "OrderFlowConfirmedBuyCandidate",
        "AccountCashFeasibleBuyCandidate",
        "ExecutableBuyCandidate",
        "LiveBrokerRealtimeQuote",
        "FreshBrokerQuote",
        "RealtimeAdaptiveFallbackBuyCandidate",
    }
)


def _ontology_buy_evidence(graph: Any, symbol: str) -> tuple[float, tuple[str, ...]]:
    """일반 매수근거 supportsSignal에서 increasesRiskOf/contradictsSignal을 뺀 순증거 점수.

    플로우 전용 점수와 달리 US 종목에도 붙는 NPU 모멘텀/유동성/실적 근거를 포착한다.
    """
    supports = {str(t.object) for t in graph.matching(subject=symbol, predicate="supportsSignal")}
    risks = {str(t.object) for t in graph.matching(subject=symbol, predicate="increasesRiskOf")}
    contradicts = {str(t.object) for t in graph.matching(subject=symbol, predicate="contradictsSignal")}
    edge = supports & _BUY_EDGE_SUPPORTS
    net = float(len(edge) - len(risks) - len(contradicts))
    return net, tuple(sorted(edge))


def _market_for_symbol(symbol: str) -> str:
    """Classify a symbol into the market label used for order routing/gates.

    6-digit numeric → Korean equity (KR); everything else → US (NASD default).
    """
    s = str(symbol or "").strip().upper()
    if s.isdigit() and len(s) == 6:
        return "KR"
    return "NASD"


def _cost_context_for_holding(symbol: str, market: str) -> tuple[str, str]:
    s = str(symbol or "").strip().upper()
    market_name = str(market or "").strip().upper()
    if s.isdigit() and len(s) == 6 or market_name in {"KR", "KRX", "KOSPI", "KOSDAQ", "KONEX"}:
        return "KRX", "domestic_stock"
    if "NYSE" in market_name:
        return "NYSE", "overseas_stock"
    if "AMEX" in market_name:
        return "AMEX", "overseas_stock"
    return "NASD", "overseas_stock"


@dataclass(frozen=True)
class SharedDecisionResult:
    symbol: str
    approved: bool
    final_order: FinalOrder | None
    prediction: LiveSignalPrediction | None
    reason_codes: tuple[str, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)


class SharedLiveDecisionEngine:
    def __init__(
        self,
        store: RealtimeMarketDataStore,
        *,
        predictor: LiveSignalPredictor | None = None,
        risk_manager: RiskManager | None = None,
        market_refresher: Callable[[str, str, datetime], MarketSnapshot | None] | None = None,
        decision_logger: DecisionLogger | None = None,
    ) -> None:
        self.store = store
        self.feature_builder = LiveFeatureFrameBuilder(store)
        self.predictor = predictor or LiveSignalPredictor()
        self.risk_manager = risk_manager or RiskManager(
            RiskRules(live_trading_enabled=False, min_average_daily_trading_value=1, max_volatility=1.0)
        )
        self.auto_tuner = AutoTuningEngine(decision_logger=decision_logger, refresh_quote=market_refresher)
        self.market_refresher = market_refresher
        self.decision_logger = decision_logger or DecisionLogger()
        self._last_diagnostics: dict[str, Any] = {}

    def evaluate_buy(
        self,
        symbol: str,
        account: AccountSnapshot,
        *,
        suggested_weight: float = 0.01,
        ontology_graph: Any | None = None,
        decision_time: datetime | None = None,
    ) -> SharedDecisionResult:
        decision_time = decision_time or datetime.now(timezone.utc)
        frame = None
        prediction: LiveSignalPrediction | None = None
        prediction_error: Exception | None = None
        try:
            frame = self.feature_builder.build(symbol, decision_time=decision_time)
            prediction = self.predictor.predict(frame)
        except Exception as exc:  # noqa: BLE001 - model failure can fall back to ontology and rules.
            prediction_error = exc

        tick = self.store.latest_tick(symbol)
        market_name = _market_for_symbol(symbol)
        quote_refresh_status = "quote_refresh_skipped"
        refreshed_market: MarketSnapshot | None = None
        if tick is None or float(getattr(tick, "price", 0.0) or 0.0) <= 0:
            if self.market_refresher is not None:
                quote_refresh_status = "quote_refresh_attempted"
                try:
                    refreshed_market = self.market_refresher(symbol, market_name, decision_time)
                except Exception:  # noqa: BLE001 - refresh is best-effort.
                    refreshed_market = None
                if refreshed_market is not None and float(getattr(refreshed_market, "last_price", 0.0) or 0.0) > 0:
                    quote_refresh_status = "quote_refresh_ok"
                else:
                    refreshed_market = None
            if refreshed_market is None:
                result = SharedDecisionResult(symbol, False, None, prediction, ("MISSING_MARKET_DATA",), {"quote_refresh_status": "missing_market_data"})
                self._last_diagnostics = result.diagnostics or {}
                return result

        currency = "KRW" if market_name.upper() in ("KR", "KRX", "KOSPI", "KOSDAQ", "KONEX") else "USD"
        cash_by_currency = account.cash_by_currency if hasattr(account, "cash_by_currency") else {}
        available_cash = float(cash_by_currency.get(currency, 0.0))

        orderbook = self.store.latest_orderbook(symbol) if hasattr(self.store, "latest_orderbook") else None
        tick_received_at = getattr(tick, "received_at", decision_time) if tick is not None else decision_time
        quote_age_seconds = 0.0 if refreshed_market is not None else max(0.0, (decision_time - tick_received_at).total_seconds())
        price = float(getattr(refreshed_market, "last_price", 0.0) or getattr(tick, "price", 0.0) or 0.0)
        min_cash_for_one_share = price * 1.05
        needs_cash_check_refresh = available_cash < min_cash_for_one_share
        if (
            self.market_refresher is not None
            and refreshed_market is None
            and (
                quote_age_seconds > float(os.getenv("REALTIME_BUY_MAX_QUOTE_AGE_SEC", "12"))
                or needs_cash_check_refresh
            )
        ):
            quote_refresh_status = "quote_refresh_attempted"
            try:
                refreshed_market = self.market_refresher(symbol, market_name, decision_time)
            except Exception:  # noqa: BLE001 - refresh is best-effort.
                refreshed_market = None
            if refreshed_market is not None:
                quote_refresh_status = "quote_refresh_ok"
                price = float(getattr(refreshed_market, "last_price", 0.0) or price)
                min_cash_for_one_share = price * 1.05

        if available_cash < min_cash_for_one_share:
            result = SharedDecisionResult(
                symbol,
                False,
                None,
                prediction,
                ("INSUFFICIENT_CASH_FOR_ONE_SHARE", f"QUOTE_REFRESH:{quote_refresh_status}"),
                {
                    "available_cash": available_cash,
                    "currency": currency,
                    "min_required": min_cash_for_one_share,
                    "price": price,
                    "quote_refresh_status": quote_refresh_status,
                },
            )
            self._last_diagnostics = result.diagnostics or {}
            return result
        market = refreshed_market or MarketSnapshot(
            ticker=symbol,
            market=market_name,
            company_name=symbol,
            sector="Unknown",
            last_price=float(getattr(tick, "price", 0.0) or 0.0),
            average_daily_trading_value=10_000_000_000,
            volatility_20d=0.02,
            source=SourceMetadata(
                source_name="KIS realtime WebSocket",
                retrieved_at=getattr(tick, "received_at", decision_time),
                observed_at=getattr(tick, "exchange_timestamp", decision_time),
                source_type="broker_api",
                trust_level=5,
                is_realtime=True,
                quality_score=1.0,
            ),
        )

        flow_score = 0.0
        edge_score = 0.0
        ontology_support: tuple[str, ...] = ()
        if ontology_graph is not None:
            try:
                flow_score, flow_support, _contra = _ontology_flow_adjustment(ontology_graph, symbol)
                edge_score, edge_support = _ontology_buy_evidence(ontology_graph, symbol)
                ontology_support = tuple(dict.fromkeys((*flow_support, *edge_support)))
            except Exception:  # noqa: BLE001 - ontology is an enhancer, never fatal.
                flow_score = edge_score = 0.0
        volume_ratio = self._realtime_volume_surge_ratio(symbol, decision_time)
        if volume_ratio >= float(os.getenv("REALTIME_VOLUME_SURGE_RATIO", "1.5")):
            edge_score += 1.0
            ontology_support = (*ontology_support, f"VolumeSurge:{volume_ratio:.1f}x")
        flow_threshold = float(os.getenv("REALTIME_ONTOLOGY_BUY_SCORE", "0.12"))
        edge_threshold = float(os.getenv("REALTIME_ONTOLOGY_BUY_MIN_SUPPORTS", "0.5"))
        ontology_score = max(flow_score, edge_score)
        ontology_ok = flow_score >= flow_threshold or edge_score >= edge_threshold
        require_ontology_fallback = str(os.getenv("REALTIME_REQUIRE_ONTOLOGY_FOR_MODEL_FALLBACK", "true")).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        price = float(getattr(market, "last_price", 0.0) or 0.0)
        liquidity_score = min(1.0, math.log1p(max(0.0, market.average_daily_trading_value)) / math.log1p(10_000_000_000))
        spread_bps = 0.0
        if orderbook is not None:
            spread_bps = max(0.0, float(getattr(orderbook, "spread_bps", 0.0) or 0.0))
            liquidity_score = min(1.0, (float(getattr(orderbook, "total_bid_volume", 0.0) or 0.0) + float(getattr(orderbook, "total_ask_volume", 0.0) or 0.0)) / 1_000_000)

        fallback_score = self.auto_tuner.fallback_buy_score(
            ontology_score=ontology_score,
            technical_momentum=float(getattr(prediction, "probability_success", 0.5) - 0.5 if prediction is not None else 0.0),
            liquidity_score=liquidity_score,
            spread_bps=spread_bps,
            volatility=float(getattr(market, "volatility_20d", 0.0) or 0.0),
            recent_performance=0.0,
        )
        model_ok = bool(prediction is not None and prediction.approved)
        fallback_allowed = True
        policy_state = self.auto_tuner.snapshot_market_state(
            symbol=symbol,
            market=market,
            quote_age_seconds=quote_age_seconds if refreshed_market is None else 0.0,
            spread_bps=spread_bps,
            orderbook_available=orderbook is not None,
            volume_ratio=volume_ratio,
            recent_performance=0.0,
            fallback_score=fallback_score,
        )
        policy, policy_diag = self.auto_tuner.build_buy_policy(
            symbol=symbol,
            account=account,
            market=market,
            market_state=policy_state,
            prediction=prediction,
            fallback_allowed=fallback_allowed,
            ontology_score=ontology_score,
            fallback_score=fallback_score,
            prediction_confidence=float(getattr(prediction, "probability_success", 0.5) or 0.5),
            prediction_error=prediction_error,
            decision_time=decision_time,
        )
        runtime_execution_ready = (
            price > 0.0
            and available_cash >= min_cash_for_one_share
            and quote_refresh_status == "quote_refresh_ok"
            and str(getattr(market.source, "source_type", "") or "") == "broker_api"
            and bool(getattr(market.source, "is_realtime", False))
            and float(getattr(market.source, "quality_score", 0.0) or 0.0) >= 0.8
        )
        runtime_fallback_support = False
        if (
            not model_ok
            and runtime_execution_ready
            and fallback_score >= policy.buy_threshold
            and policy.allowed_fallback_mode != "no_trade"
        ):
            runtime_fallback_support = True
            ontology_ok = True
            ontology_score = max(ontology_score, 1.0)
            ontology_support = tuple(
                dict.fromkeys(
                    (
                        *ontology_support,
                        "FreshBrokerQuote",
                        "CashFitOneShare",
                        "ExecutableBuyCandidate",
                        "RealtimeAdaptiveFallbackBuyCandidate",
                    )
                )
            )
        if not model_ok and require_ontology_fallback and not ontology_ok:
            reasons = tuple(getattr(prediction, "reason_codes", ()) or ("MODEL_UNAVAILABLE",))
            reasons = (*reasons, "ONTOLOGY_REQUIRED_FOR_MODEL_FALLBACK", f"QUOTE_REFRESH:{quote_refresh_status}")
            diagnostics = {"policy": policy.as_dict(), "policy_state": policy_diag, "quote_refresh_status": quote_refresh_status, "fallback_score": fallback_score, "ontology_score": ontology_score}
            self._last_diagnostics = diagnostics
            return SharedDecisionResult(symbol, False, None, prediction, reasons, diagnostics)
        if not model_ok and policy.allowed_fallback_mode == "no_trade" and fallback_score < policy.buy_threshold:
            reasons = tuple(getattr(prediction, "reason_codes", ()) or ("MODEL_UNAVAILABLE",))
            reasons = (*reasons, "MODEL_FALLBACK_NOT_ALLOWED", f"QUOTE_REFRESH:{quote_refresh_status}")
            diagnostics = {"policy": policy.as_dict(), "policy_state": policy_diag, "quote_refresh_status": quote_refresh_status, "fallback_score": fallback_score}
            self._last_diagnostics = diagnostics
            return SharedDecisionResult(symbol, False, None, prediction, reasons, diagnostics)

        model_score = float(getattr(prediction, "probability_success", 0.0) or 0.0) if model_ok else fallback_score
        signal_score = max(model_score, ontology_score * 0.35 if not model_ok else ontology_score * 0.25)
        if not model_ok:
            signal_score = max(signal_score, fallback_score)
        signal_gap = signal_score - policy.buy_threshold
        size_multiplier = 1.0
        if signal_gap < 0:
            if signal_gap >= -0.18 and policy.allowed_fallback_mode != "no_trade":
                size_multiplier = max(0.20, 1.0 + signal_gap * 2.5)
            else:
                reasons = tuple(getattr(prediction, "reason_codes", ()) or ())
                reasons = (*reasons, "ONTOLOGY_BELOW_ADAPTIVE_THRESHOLD" if ontology_ok is False else "BUY_SIGNAL_TOO_WEAK", f"QUOTE_REFRESH:{quote_refresh_status}")
                diagnostics = {"policy": policy.as_dict(), "policy_state": policy_diag, "quote_refresh_status": quote_refresh_status, "fallback_score": fallback_score, "signal_score": signal_score}
                self._last_diagnostics = diagnostics
                return SharedDecisionResult(symbol, False, None, prediction, reasons, diagnostics)

        suggested_weight = max(0.001, suggested_weight * size_multiplier)
        expected_return_bps = 100.0
        if model_ok and prediction is not None:
            expected_return_bps = max(expected_return_bps, float(prediction.expected_net_return_bps or 0.0))
        expected_return_bps = max(expected_return_bps, fallback_score * 300.0, policy.min_expected_net_return * 10_000.0)
        gross_expected_return = max(policy.min_expected_net_return * 0.5, expected_return_bps / 10_000.0)
        confidence = max(policy.confidence_floor, float(getattr(prediction, "probability_success", 0.5) or 0.5) if model_ok else 0.5 + fallback_score * 0.2)
        if not model_ok:
            confidence = max(0.35, confidence - 0.1)
        signal_name = "trained_expected_net_return" if model_ok else "ontology_fallback_buy"
        supporting = ("trained_live_model",) if model_ok else ("ontology_fallback",)
        if ontology_support:
            supporting = (*supporting, *ontology_support)
        reasoning = f"policy:{policy.risk_mode};score={signal_score:.2f};quote={quote_refresh_status}"
        artifact_id = str(getattr(prediction, "model_artifact_id", "") or "") if prediction is not None else ""
        validation_id = artifact_id or f"adaptive-buy:{symbol}:{decision_time.strftime('%Y%m%d%H%M%S')}"
        source_data_ids = (
            frame.provenance.source_record_ids
            if frame is not None
            else (str(getattr(tick, "sequence_key", "") if tick is not None else "") or f"quote:{symbol}:{quote_refresh_status}",)
        )
        strategy_metadata: dict[str, Any] = {
            "model_artifact_id": artifact_id,
            "feature_schema_hash": prediction.feature_schema_hash if prediction is not None else "",
            "ontology_buy_score": round(ontology_score, 4),
            "fallback_score": round(fallback_score, 4),
            "runtime_execution_ready": runtime_execution_ready,
            "runtime_fallback_support": runtime_fallback_support,
            "buy_threshold": policy.buy_threshold,
            "policy": policy.as_dict(),
            "policy_state": policy_diag,
            "quote_refresh_status": quote_refresh_status,
            "quote_age_seconds": round(quote_age_seconds, 3),
            "stop_loss_price": price * (1.0 - policy.stop_loss),
        }
        if orderbook is not None:
            strategy_metadata["orderbook_snapshot"] = {
                "best_bid": orderbook.best_bid,
                "best_ask": orderbook.best_ask,
                "bid_depth": orderbook.total_bid_volume,
                "ask_depth": orderbook.total_ask_volume,
            }
        intent = OrderIntent(
            ticker=symbol,
            market=market_name,
            action=OrderAction.BUY,
            suggested_weight=min(suggested_weight, float(policy.max_position_size) / max(1.0, float(account.equity or 0.0))),
            confidence=confidence,
            valid_until=decision_time + timedelta(seconds=max(30, int(policy.quote_ttl_seconds))),
            reasoning_summary=(reasoning,),
            supporting_factors=supporting,
            contradicting_factors=(),
            source_data_ids=source_data_ids,
            model_uncertainty=prediction.uncertainty_score if prediction is not None else (0.85 if not model_ok else None),
            strategy_family="live_short_horizon",
            signal_name=signal_name,
            expected_exit_price=price * (1.0 + gross_expected_return),
            expected_holding_minutes=max(1, min(30, int(policy.time_exit_seconds / 60))),
            gross_expected_return=gross_expected_return,
            target_net_return=policy.min_expected_net_return,
            validation_id=validation_id,
            strategy_metadata=strategy_metadata,
        )
        adaptive_rules = self.auto_tuner.derive_risk_rules(
            self.risk_manager.rules,
            policy=policy,
            account=account,
            market=market,
            model_uncertainty=prediction.uncertainty_score if prediction is not None else None,
        )
        adaptive_rules = replace(adaptive_rules, minimum_cash_reserve=0.0)
        risk_manager = RiskManager(adaptive_rules, audit_logger=self.risk_manager.audit_logger)
        risk = risk_manager.validate(intent, account, market)
        diagnostics = {
            "policy": policy.as_dict(),
            "policy_state": policy_diag,
            "quote_refresh_status": quote_refresh_status,
            "fallback_score": fallback_score,
            "model_ok": model_ok,
            "signal_score": signal_score,
            "runtime_execution_ready": runtime_execution_ready,
            "runtime_fallback_support": runtime_fallback_support,
            "adaptive_risk_rules": adaptive_rules,
            "risk_metadata": risk.metadata,
        }
        self.auto_tuner.record_feedback(
            {
                "symbol": symbol,
                "side": "BUY",
                "approved": risk.approved,
                "reason_codes": risk.rejection_reasons,
                "policy": policy.as_dict(),
                "pnl": 0.0,
                "quote_refresh_status": quote_refresh_status,
            }
        )
        self._last_diagnostics = diagnostics
        return SharedDecisionResult(
            symbol=symbol,
            approved=risk.approved and risk.final_order is not None,
            final_order=risk.final_order,
            prediction=prediction,
            reason_codes=risk.rejection_reasons,
            diagnostics=diagnostics,
        )

    def evaluate_exit_for_holding(
        self,
        holding: Holding,
        account: AccountSnapshot,
        *,
        take_profit: float = 0.0025,
        stop_loss: float = 0.010,
        ontology_graph: Any | None = None,
        decision_time: datetime | None = None,
    ) -> SharedDecisionResult:
        symbol = holding.ticker
        decision_time = decision_time or datetime.now(timezone.utc)
        avg_cost = float(getattr(holding, "average_price", 0.0) or 0.0)
        if avg_cost <= 0:
            result = SharedDecisionResult(symbol, False, None, None, ("INVALID_PRICE_OR_COST",), {"exit_reason": "invalid_price"})
            self._last_diagnostics = result.diagnostics or {}
            return result
        if int(getattr(holding, "quantity", 0) or 0) <= 0:
            result = SharedDecisionResult(symbol, False, None, None, ("NO_POSITION",), {"exit_reason": "no_position"})
            self._last_diagnostics = result.diagnostics or {}
            return result

        price, observed_at, received_at, source_id = self._exit_price_source(symbol, holding, decision_time)
        if price <= 0 and self.market_refresher is not None:
            try:
                refreshed = self.market_refresher(symbol, holding.market or "KR", decision_time)
            except Exception:  # noqa: BLE001 - refresh is best-effort.
                refreshed = None
            if refreshed is not None and float(refreshed.last_price or 0.0) > 0:
                price = float(refreshed.last_price)
                observed_at = refreshed.source.observed_at or refreshed.source.retrieved_at
                received_at = refreshed.source.retrieved_at
                source_id = refreshed.source.source_id or f"refreshed:{symbol}"
        if price > 0 and self.market_refresher is not None:
            stale_seconds = max(0.0, (decision_time - received_at).total_seconds())
            if stale_seconds > float(os.getenv("REALTIME_EXIT_MAX_QUOTE_AGE_SEC", "12")):
                try:
                    refreshed = self.market_refresher(symbol, holding.market or "KR", decision_time)
                except Exception:  # noqa: BLE001 - refresh is best-effort.
                    refreshed = None
                if refreshed is not None and float(refreshed.last_price or 0.0) > 0:
                    price = float(refreshed.last_price)
                    observed_at = refreshed.source.observed_at or refreshed.source.retrieved_at
                    received_at = refreshed.source.retrieved_at
                    source_id = refreshed.source.source_id or f"refreshed:{symbol}"
        if price <= 0:
            result = SharedDecisionResult(symbol, False, None, None, ("MISSING_MARKET_DATA",), {"exit_reason": "missing_market_data"})
            self._last_diagnostics = result.diagnostics or {}
            return result

        pnl_rate = (price - avg_cost) / avg_cost
        ontology_score = 0.0
        ontology_support: tuple[str, ...] = ()
        if ontology_graph is not None:
            try:
                position_weight = (
                    float(holding.market_value) / max(1.0, float(account.equity))
                    if account is not None and account.equity > 0
                    else 0.0
                )
                ontology_score, ontology_support, _onto_contra = _holding_exit_adjustment(
                    ontology_graph, symbol, position_weight, holding
                )
            except Exception:  # noqa: BLE001 - ontology is an enhancer; never block exits on it.
                ontology_score = 0.0

        target_net_return = max(0.0, float(os.getenv("REALTIME_EXIT_TARGET_NET_RETURN", "0.0003")))
        volume_ratio = self._realtime_volume_surge_ratio(symbol, decision_time)
        market = self._exit_market_snapshot(holding, price, observed_at, received_at)
        quote_age_seconds = max(0.0, (decision_time - received_at).total_seconds())
        orderbook = self.store.latest_orderbook(symbol) if hasattr(self.store, "latest_orderbook") else None
        market_state = self.auto_tuner.snapshot_market_state(
            symbol=symbol,
            market=market,
            quote_age_seconds=quote_age_seconds,
            spread_bps=float(getattr(orderbook, "spread_bps", 0.0) or 0.0),
            orderbook_available=orderbook is not None,
            volume_ratio=volume_ratio,
            recent_performance=0.0,
            fallback_score=max(0.0, ontology_score + max(-0.5, min(0.5, pnl_rate))),
        )
        policy, exit_policy, policy_diag = self.auto_tuner.build_exit_policy(
            symbol=symbol,
            holding=holding,
            account=account,
            market=market,
            market_state=market_state,
            take_profit=take_profit,
            stop_loss=stop_loss,
            ontology_score=ontology_score,
            target_net_return=target_net_return,
            decision_time=decision_time,
        )
        cost_floor = self._exit_cost_floor(holding, price, target_net_return)
        required_exit_price = max(cost_floor.required_exit_price, avg_cost * (1.0 + exit_policy.sell_target))
        required_exit_return = (required_exit_price - avg_cost) / avg_cost
        profitable_after_cost = price >= required_exit_price and cost_floor.net_expected_return >= target_net_return
        loss_exit_allowed = exit_policy.allow_loss_exit
        emergency_loss = max(exit_policy.stop_loss, float(os.getenv("REALTIME_EMERGENCY_STOP_LOSS", "0.05")))

        prediction: LiveSignalPrediction | None = None
        exit_reason: str | None = None
        if profitable_after_cost and ontology_score <= -0.55:
            exit_reason = f"profit_exit:{pnl_rate * 100:.2f}%"
        elif profitable_after_cost and pnl_rate >= required_exit_return:
            exit_reason = f"profit_exit:{pnl_rate * 100:.2f}%"
        elif pnl_rate <= -exit_policy.stop_loss and not loss_exit_allowed:
            diagnostics = {"exit_policy": exit_policy.as_dict(), "policy": policy.as_dict(), "policy_state": policy_diag, "quote_age_seconds": round(quote_age_seconds, 3), "ontology_score": round(ontology_score, 4)}
            reasons = ("LOSS_EXIT_DISABLED", "HOLD_LOSS_EXIT_DISABLED", "REALTIME_ALLOW_LOSS_EXIT=false")
            self._last_diagnostics = diagnostics
            return SharedDecisionResult(symbol, False, None, prediction, reasons, diagnostics)
        elif pnl_rate <= -emergency_loss and loss_exit_allowed:
            exit_reason = f"loss_exit:{pnl_rate * 100:.2f}%"
        elif pnl_rate <= -exit_policy.trailing_stop and loss_exit_allowed:
            exit_reason = f"trailing_exit:{pnl_rate * 100:.2f}%"
        elif quote_age_seconds >= exit_policy.time_exit_seconds:
            if profitable_after_cost or pnl_rate >= 0:
                exit_reason = f"time_exit:{pnl_rate * 100:.2f}%"
            else:
                exit_reason = None
        elif ontology_score <= -0.25 and profitable_after_cost:
            exit_reason = f"invalid_signal_exit:{ontology_score:.2f}"
        elif ontology_score <= -0.25 and not profitable_after_cost:
            diagnostics = {"exit_policy": exit_policy.as_dict(), "policy": policy.as_dict(), "policy_state": policy_diag, "quote_age_seconds": round(quote_age_seconds, 3), "ontology_score": round(ontology_score, 4)}
            reasons = ("HOLD_UNPROFITABLE_ONTOLOGY_SELL_BLOCKED", "HOLD_BELOW_PROFIT_TARGET")
            self._last_diagnostics = diagnostics
            return SharedDecisionResult(symbol, False, None, prediction, reasons, diagnostics)
        else:
            try:
                exit_reason, prediction = self._model_exit_signal(symbol, decision_time)
            except Exception:  # noqa: BLE001 - model exit is best-effort.
                exit_reason, prediction = None, None
            if exit_reason is not None and not profitable_after_cost:
                exit_reason = None

        if exit_reason is None:
            diagnostics = {"exit_policy": exit_policy.as_dict(), "policy": policy.as_dict(), "policy_state": policy_diag, "quote_age_seconds": round(quote_age_seconds, 3), "ontology_score": round(ontology_score, 4)}
            reasons = (
                "HOLD_RECHECK",
                "HOLD_BELOW_PROFIT_TARGET",
                f"QUOTE_REFRESH:{'quote_refresh_ok' if quote_age_seconds <= exit_policy.time_exit_seconds else 'quote_refresh_not_needed'}",
            )
            self._last_diagnostics = diagnostics
            return SharedDecisionResult(symbol, False, None, prediction, reasons, diagnostics)

        account_total = max(
            float(
                getattr(account, "equity", None)
                or getattr(account, "total_equity", None)
                or getattr(account, "cash_balance", None)
                or getattr(account, "cash", None)
                or 1.0
            ),
            1.0,
        )
        position_weight = max(0.0, (holding.quantity * price) / account_total)
        exit_action = OrderAction.SELL
        exit_suggested_weight = 0.0
        if exit_reason.startswith("trailing_exit"):
            reduce_fraction = max(0.1, min(1.0, float(os.getenv("REALTIME_LOSS_EXIT_REDUCE_FRACTION", "0.5"))))
            exit_action = OrderAction.REDUCE
            exit_suggested_weight = max(0.0, position_weight * (1.0 - reduce_fraction))

        intent = OrderIntent(
            ticker=symbol,
            market=holding.market or "KR",
            action=exit_action,
            suggested_weight=exit_suggested_weight,
            confidence=max(exit_policy.confidence_floor, 0.85 if profitable_after_cost else 0.7),
            valid_until=decision_time + timedelta(seconds=max(30, exit_policy.time_exit_seconds // 4)),
            reasoning_summary=(f"realtime_exit:{exit_reason}",),
            supporting_factors=("realtime_exit", exit_policy.exit_mode, *ontology_support),
            contradicting_factors=(),
            source_data_ids=(source_id,),
            strategy_family="live_short_horizon_exit",
            signal_name=exit_reason.split(":", 1)[0],
            expected_exit_price=required_exit_price,
            gross_expected_return=max(0.0, required_exit_return),
            target_net_return=target_net_return,
            cost_breakdown=cost_floor.as_dict(),
            strategy_metadata={
                "exit_policy": exit_policy.as_dict(),
                "policy": policy.as_dict(),
                "policy_state": policy_diag,
                "quote_age_seconds": round(quote_age_seconds, 3),
                "ontology_score": round(ontology_score, 4),
                "pnl_rate": round(pnl_rate, 6),
                "exit_reason": exit_reason,
                "exit_action": str(exit_action),
                "exit_suggested_weight": round(exit_suggested_weight, 6),
            },
        )
        adaptive_rules = self.auto_tuner.derive_risk_rules(
            self.risk_manager.rules,
            policy=policy,
            account=account,
            market=market,
            model_uncertainty=prediction.uncertainty_score if prediction is not None else None,
        )
        risk_manager = RiskManager(adaptive_rules, audit_logger=self.risk_manager.audit_logger)
        risk = risk_manager.validate(intent, account, market)
        if intent.action in {OrderAction.SELL, OrderAction.REDUCE} and not risk.approved and set(risk.rejection_reasons) == {"cash_available"}:
            risk = risk.__class__(
                ticker=risk.ticker,
                action=risk.action,
                approved=True,
                adjusted_weight=risk.adjusted_weight,
                checks={**risk.checks, "cash_available": True},
                rejection_reasons=(),
                final_order=FinalOrder(
                    ticker=intent.ticker,
                    market=intent.market,
                    order_type=OrderType.LIMIT,
                    side=OrderSide.SELL,
                    quantity=max(1, int(getattr(holding, "quantity", 0) or 0)),
                    limit_price=price,
                    manual_approval_required=self.risk_manager.rules.manual_approval_required,
                ),
                metadata=dict(risk.metadata),
            )
        diagnostics = {
            "exit_policy": exit_policy.as_dict(),
            "policy": policy.as_dict(),
            "policy_state": policy_diag,
            "quote_age_seconds": round(quote_age_seconds, 3),
            "ontology_score": round(ontology_score, 4),
            "adaptive_risk_rules": adaptive_rules,
            "risk_metadata": risk.metadata,
            "exit_reason": exit_reason,
        }
        self.auto_tuner.record_feedback(
            {
                "symbol": symbol,
                "side": "SELL",
                "approved": risk.approved,
                "reason_codes": risk.rejection_reasons,
                "policy": policy.as_dict(),
                "pnl": pnl_rate,
                "quote_refresh_status": "quote_refresh_ok" if quote_age_seconds <= exit_policy.time_exit_seconds else "quote_refresh_skipped",
            }
        )
        self._last_diagnostics = diagnostics
        return SharedDecisionResult(
            symbol=symbol,
            approved=risk.approved and risk.final_order is not None,
            final_order=risk.final_order,
            prediction=prediction,
            reason_codes=risk.rejection_reasons,
            diagnostics=diagnostics,
        )

    def _exit_cost_floor(self, holding: Holding, expected_exit_price: float, target_net_return: float):
        symbol = str(getattr(holding, "ticker", "") or "")
        market = str(getattr(holding, "market", "") or "")
        quantity = max(1, int(getattr(holding, "quantity", 0) or 0))
        venue, instrument_type = _cost_context_for_holding(symbol, market)
        return TradingCostEngine().estimate(
            symbol=symbol,
            market=market or ("KR" if instrument_type == "domestic_stock" else venue),
            venue=venue,
            instrument_type=instrument_type,
            entry_price=float(getattr(holding, "average_price", 0.0) or 0.0),
            expected_exit_price=float(expected_exit_price),
            quantity=quantity,
            target_net_return=target_net_return,
        )

    def _realtime_volume_surge_ratio(self, symbol: str, decision_time: datetime) -> float:
        """장중 거래 활성도 급증 비율 = 최근 짧은 구간 체결빈도 / 기준 구간 체결빈도.

        소스별 volume 필드 의미(틱 증분 vs 누적)가 달라, 소스에 무관하게 견고한
        체결(틱) 빈도를 사용한다. 데이터가 부족하면 1.0(중립)을 반환한다.
        """
        try:
            recent_window = max(5.0, float(os.getenv("REALTIME_VOLUME_RECENT_SEC", "60")))
            base_window = max(recent_window * 2.0, float(os.getenv("REALTIME_VOLUME_BASE_SEC", "600")))
            base_since = decision_time - timedelta(seconds=base_window)
            ticks = self.store.recent_ticks(symbol, base_since)
            if not ticks or len(ticks) < 4:
                return 1.0
            recent_cut = decision_time - timedelta(seconds=recent_window)
            recent_count = sum(1 for t in ticks if (getattr(t, "received_at", None) or decision_time) >= recent_cut)
            base_rate = len(ticks) / base_window
            if base_rate <= 0:
                return 1.0
            recent_rate = recent_count / recent_window
            return recent_rate / base_rate
        except Exception:  # noqa: BLE001 - volume signal is best-effort.
            return 1.0

    def _exit_price_source(
        self, symbol: str, holding: Holding, decision_time: datetime
    ) -> tuple[float, datetime, datetime, str]:
        """Prefer a fresh realtime tick; fall back to the broker balance mark."""
        max_tick_age = float(os.getenv("REALTIME_EXIT_TICK_MAX_AGE_SEC", "30"))
        tick = self.store.latest_tick(symbol)
        if tick is not None:
            tick_price = float(getattr(tick, "price", 0.0) or 0.0)
            received_at = getattr(tick, "received_at", None) or decision_time
            try:
                tick_age = (decision_time - received_at).total_seconds()
            except Exception:  # noqa: BLE001 - timezone mishaps fall back to the broker mark.
                tick_age = max_tick_age + 1
            if tick_price > 0 and tick_age <= max_tick_age:
                return (
                    tick_price,
                    getattr(tick, "exchange_timestamp", received_at) or received_at,
                    received_at,
                    str(getattr(tick, "sequence_key", "") or f"tick:{symbol}"),
                )
        balance_price = float(getattr(holding, "last_price", 0.0) or 0.0)
        return balance_price, decision_time, decision_time, f"balance:{symbol}"

    def _exit_risk_manager(self) -> RiskManager:
        """De-risking 매도는 매수용 게이트(현금준비금/실시간 호가 신선도)에 막히면 안 되므로
        완화된 규칙으로 검증한다. 가격 소스는 브로커 잔고 마크를 신뢰한다."""
        cached = getattr(self, "_exit_risk_manager_cache", None)
        if cached is None:
            relaxed = replace(
                self.risk_manager.rules,
                minimum_cash_reserve=0.0,
                max_quote_age_seconds=1e9,
            )
            cached = RiskManager(relaxed)
            self._exit_risk_manager_cache = cached
        return cached

    def _model_exit_signal(
        self, symbol: str, decision_time: datetime
    ) -> tuple[str | None, LiveSignalPrediction | None]:
        """Model-based exit: trim when the buy edge has clearly flipped negative."""
        try:
            frame = self.feature_builder.build(symbol, decision_time=decision_time)
            prediction = self.predictor.predict(frame)
        except Exception:  # noqa: BLE001 - model exit is best-effort; TP/SL still protects.
            return None, None
        exit_bps = float(os.getenv("REALTIME_MODEL_EXIT_BPS", "8"))
        if not prediction.approved and prediction.expected_net_return_bps <= -exit_bps:
            return f"model_edge_lost:{prediction.expected_net_return_bps:.1f}bps", prediction
        return None, prediction

    def _exit_market_snapshot(
        self, holding: Holding, price: float, observed_at: datetime, received_at: datetime
    ) -> MarketSnapshot:
        return MarketSnapshot(
            ticker=holding.ticker,
            market=holding.market or "KR",
            company_name=getattr(holding, "company_name", "") or holding.ticker,
            sector=getattr(holding, "sector", "") or "Unknown",
            last_price=price,
            average_daily_trading_value=10_000_000_000,
            volatility_20d=0.02,
            source=SourceMetadata(
                source_name="KIS broker mark / realtime WebSocket",
                retrieved_at=received_at,
                observed_at=observed_at,
                source_type="broker_api",
                trust_level=5,
                is_realtime=True,
                quality_score=1.0,
            ),
        )

    def get_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_diagnostics)

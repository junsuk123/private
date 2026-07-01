from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from app.cost import TradingCostEngine
from app.data.realtime_store import RealtimeMarketDataStore
from app.features.live_feature_frame import LiveFeatureFrameBuilder
from app.models.live_signal_predictor import LiveSignalPredictor, LiveSignalPrediction
from app.risk import RiskManager
from app.schemas.domain import AccountSnapshot, FinalOrder, Holding, MarketSnapshot, OrderAction, OrderIntent, RiskRules, SourceMetadata
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


class SharedLiveDecisionEngine:
    def __init__(
        self,
        store: RealtimeMarketDataStore,
        *,
        predictor: LiveSignalPredictor | None = None,
        risk_manager: RiskManager | None = None,
    ) -> None:
        self.store = store
        self.feature_builder = LiveFeatureFrameBuilder(store)
        self.predictor = predictor or LiveSignalPredictor()
        self.risk_manager = risk_manager or RiskManager(
            RiskRules(live_trading_enabled=False, min_average_daily_trading_value=1, max_volatility=1.0)
        )

    def evaluate_buy(
        self,
        symbol: str,
        account: AccountSnapshot,
        *,
        suggested_weight: float = 0.01,
        ontology_graph: Any | None = None,
        decision_time: datetime | None = None,
    ) -> SharedDecisionResult:
        """Buy when the trained model approves OR ontology reasoning supports a buy.

        Mirrors the sell side: the decision is influenced by ontology, so a buy can
        fire on ontology flow evidence even when the live model is unavailable or
        (currently) degenerate. RiskManager still enforces cash/limits/freshness.
        """
        decision_time = decision_time or datetime.now(timezone.utc)
        frame = None
        prediction: LiveSignalPrediction | None = None
        try:
            frame = self.feature_builder.build(symbol, decision_time=decision_time)
            prediction = self.predictor.predict(frame)
        except Exception:  # noqa: BLE001 - model failure falls back to the ontology buy path.
            prediction = None

        model_ok = bool(prediction is not None and prediction.approved)

        # 온톨로지 매수 신호 두 갈래:
        #  - 플로우 점수(_ontology_flow_adjustment): 국내(KR) 투자자/주문 플로우 근거(weighted).
        #  - 일반 buy-edge 근거(_ontology_buy_evidence): NPU 모멘텀/유동성/실적 등 supportsSignal 순증거(US 포함).
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
        # 실시간 장중 거래량 급증(틱 스토어 기반) — 단타의 핵심 신호. 그래프 유무와 무관하게 매수 근거에 가산.
        volume_ratio = self._realtime_volume_surge_ratio(symbol, decision_time)
        if volume_ratio >= float(os.getenv("REALTIME_VOLUME_SURGE_RATIO", "1.5")):
            edge_score += 1.0
            ontology_support = (*ontology_support, f"VolumeSurge:{volume_ratio:.1f}x")

        flow_threshold = float(os.getenv("REALTIME_ONTOLOGY_BUY_SCORE", "0.20"))
        edge_threshold = float(os.getenv("REALTIME_ONTOLOGY_BUY_MIN_SUPPORTS", "1.0"))
        ontology_ok = flow_score >= flow_threshold or edge_score >= edge_threshold
        ontology_score = edge_score if edge_score >= edge_threshold else flow_score

        if not model_ok and not ontology_ok:
            reasons = tuple(prediction.reason_codes) if prediction is not None else ("MODEL_UNAVAILABLE",)
            reasons = (*reasons, f"ONTOLOGY_BELOW_BUY_THRESHOLD:flow={flow_score:.2f},edge={edge_score:.0f},vol={volume_ratio:.1f}x")
            return SharedDecisionResult(symbol, False, None, prediction, reasons)

        tick = self.store.latest_tick(symbol)
        if tick is None or float(getattr(tick, "price", 0.0) or 0.0) <= 0:
            return SharedDecisionResult(symbol, False, None, prediction, ("MISSING_MARKET_DATA",))
        orderbook = self.store.latest_orderbook(symbol)

        if model_ok:
            gross_expected_return = prediction.expected_net_return_bps / 10_000.0
            confidence = prediction.probability_success
            signal_name = "trained_expected_net_return"
            supporting = ("trained_live_model",)
            reasoning = f"model:{prediction.model_artifact_id}"
        else:
            # 온톨로지 주도 매수: 모델 대신 온톨로지 플로우 근거로 진입(모델은 보조).
            gross_expected_return = float(os.getenv("REALTIME_ONTOLOGY_BUY_TARGET", "0.012"))
            confidence = max(0.5, min(0.8, 0.5 + ontology_score * 0.3))
            signal_name = "ontology_flow_buy"
            supporting = ("ontology_flow", *ontology_support)
            reasoning = f"ontology_flow:{ontology_score:.2f}"

        expected_exit_price = tick.price * (1.0 + gross_expected_return)
        market_name = _market_for_symbol(symbol)
        artifact_id = prediction.model_artifact_id if prediction is not None else ""
        validation_id = artifact_id or f"ontology-buy:{symbol}"
        source_data_ids = (
            frame.provenance.source_record_ids
            if frame is not None
            else (str(getattr(tick, "sequence_key", "") or f"tick:{symbol}"),)
        )
        strategy_metadata: dict[str, Any] = {
            "model_artifact_id": artifact_id,
            "feature_schema_hash": prediction.feature_schema_hash if prediction is not None else "",
            "ontology_buy_score": round(ontology_score, 4),
            "stop_loss_price": tick.price * 0.99,
        }
        if orderbook is not None:
            strategy_metadata["orderbook_snapshot"] = {
                "best_bid": orderbook.best_bid,
                "best_ask": orderbook.best_ask,
                "bid_depth": orderbook.total_bid_volume,
                "ask_depth": orderbook.total_ask_volume,
            }
        market = MarketSnapshot(
            ticker=symbol,
            market=market_name,
            company_name=symbol,
            sector="Unknown",
            last_price=tick.price,
            average_daily_trading_value=10_000_000_000,
            volatility_20d=0.02,
            source=SourceMetadata(
                source_name="KIS realtime WebSocket",
                retrieved_at=tick.received_at,
                observed_at=tick.exchange_timestamp,
                source_type="broker_api",
                trust_level=5,
                is_realtime=True,
                quality_score=1.0,
            ),
        )
        intent = OrderIntent(
            ticker=symbol,
            market=market_name,
            action=OrderAction.BUY,
            suggested_weight=suggested_weight,
            confidence=confidence,
            valid_until=decision_time + timedelta(minutes=1),
            reasoning_summary=(reasoning,),
            supporting_factors=supporting,
            contradicting_factors=(),
            source_data_ids=source_data_ids,
            model_uncertainty=prediction.uncertainty_score if prediction is not None else None,
            strategy_family="live_short_horizon",
            signal_name=signal_name,
            expected_exit_price=expected_exit_price,
            expected_holding_minutes=5,
            gross_expected_return=gross_expected_return,
            target_net_return=0.0,
            validation_id=validation_id,
            strategy_metadata=strategy_metadata,
        )
        risk = self.risk_manager.validate(intent, account, market)
        return SharedDecisionResult(
            symbol=symbol,
            approved=risk.approved and risk.final_order is not None,
            final_order=risk.final_order,
            prediction=prediction,
            reason_codes=risk.rejection_reasons,
        )

    def evaluate_exit_for_holding(
        self,
        holding: Holding,
        account: AccountSnapshot,
        *,
        take_profit: float = 0.006,
        stop_loss: float = 0.010,
        ontology_graph: Any | None = None,
        decision_time: datetime | None = None,
    ) -> SharedDecisionResult:
        """Decide whether to exit a currently held position.

        Day-trading exit combining three influences:
        - fast take-profit / stop-loss against the current price,
        - ontology reasoning on the held position (reuses _holding_exit_adjustment),
        - a model-based exit when price is inside the (ontology-modulated) bands.

        Price source: a fresh realtime tick when available, otherwise the broker
        balance mark (holding.last_price) — thinly traded holdings rarely have a
        fresh websocket tick, so the broker mark keeps exits working.
        Returns a SELL FinalOrder (validated by RiskManager) when an exit is warranted.
        """
        symbol = holding.ticker
        decision_time = decision_time or datetime.now(timezone.utc)
        avg_cost = float(getattr(holding, "average_price", 0.0) or 0.0)
        if avg_cost <= 0:
            return SharedDecisionResult(symbol, False, None, None, ("INVALID_PRICE_OR_COST",))
        if int(getattr(holding, "quantity", 0) or 0) <= 0:
            return SharedDecisionResult(symbol, False, None, None, ("NO_POSITION",))

        price, observed_at, received_at, source_id = self._exit_price_source(symbol, holding, decision_time)
        if price <= 0:
            return SharedDecisionResult(symbol, False, None, None, ("MISSING_MARKET_DATA",))

        pnl_rate = (price - avg_cost) / avg_cost

        # 온톨로지 추론 반영: 보유 포지션 exit_score를 산출해 매도 판단을 좌우한다.
        ontology_score = 0.0
        if ontology_graph is not None:
            try:
                position_weight = (
                    float(holding.market_value) / max(1.0, float(account.equity))
                    if account is not None and account.equity > 0
                    else 0.0
                )
                ontology_score, _onto_support, _onto_contra = _holding_exit_adjustment(
                    ontology_graph, symbol, position_weight, holding
                )
            except Exception:  # noqa: BLE001 - ontology is an enhancer; never block exits on it.
                ontology_score = 0.0

        onto_sell_score = float(os.getenv("REALTIME_ONTOLOGY_SELL_SCORE", "-0.65"))
        onto_reduce_score = float(os.getenv("REALTIME_ONTOLOGY_REDUCE_SCORE", "-0.10"))
        # 온톨로지가 위험을 가리키면 익절/손절 밴드를 좁혀 더 빨리 빠져나오고,
        # 우호적이면 밴드를 넓혀 더 오래 보유한다.
        take_profit_eff, stop_loss_eff = take_profit, stop_loss
        if ontology_score <= onto_reduce_score:
            take_profit_eff = max(0.001, take_profit * 0.75)
        elif ontology_score >= 0.10:
            take_profit_eff = take_profit * 1.5

        target_net_return = max(0.0, float(os.getenv("REALTIME_EXIT_TARGET_NET_RETURN", "0.0015")))
        cost_floor = self._exit_cost_floor(holding, price, target_net_return)
        required_exit_price = max(cost_floor.required_exit_price, avg_cost * (1.0 + take_profit_eff))
        required_exit_return = (required_exit_price - avg_cost) / avg_cost
        profitable_after_cost = price >= required_exit_price and cost_floor.net_expected_return >= target_net_return
        loss_exit_allowed = os.getenv("REALTIME_ALLOW_LOSS_EXIT", "false").strip().lower() in {"1", "true", "yes", "on"}
        emergency_loss = max(stop_loss_eff, float(os.getenv("REALTIME_EMERGENCY_STOP_LOSS", "0.05")))

        prediction: LiveSignalPrediction | None = None
        reason: str | None = None
        blocked_reason: str | None = None
        if profitable_after_cost and ontology_score <= onto_sell_score:
            reason = (
                f"profit_protected_ontology_sell:{pnl_rate * 100:.2f}%"
                f"(net={cost_floor.net_expected_return * 100:.2f}%,required={required_exit_return * 100:.2f}%,onto={ontology_score:.2f})"
            )
        elif profitable_after_cost and pnl_rate >= required_exit_return:
            reason = (
                f"profit_protected_take_profit:{pnl_rate * 100:.2f}%"
                f"(net={cost_floor.net_expected_return * 100:.2f}%,required={required_exit_return * 100:.2f}%,onto={ontology_score:.2f})"
            )
        elif pnl_rate <= -emergency_loss and loss_exit_allowed:
            reason = f"emergency_stop_loss:{pnl_rate * 100:.2f}%(sl={emergency_loss * 100:.2f}%,onto={ontology_score:.2f})"
        elif pnl_rate <= -stop_loss_eff:
            blocked_reason = (
                f"HOLD_LOSS_EXIT_DISABLED:pnl={pnl_rate * 100:.2f}%,"
                f"enable_REALTIME_ALLOW_LOSS_EXIT=true_for_stop_loss"
            )
        elif ontology_score <= onto_sell_score:
            blocked_reason = (
                f"HOLD_UNPROFITABLE_ONTOLOGY_SELL_BLOCKED:pnl={pnl_rate * 100:.2f}%,"
                f"required={required_exit_return * 100:.2f}%,net={cost_floor.net_expected_return * 100:.2f}%"
            )
        else:
            reason, prediction = self._model_exit_signal(symbol, decision_time)
            if reason is not None and not profitable_after_cost:
                blocked_reason = (
                    f"HOLD_UNPROFITABLE_MODEL_EXIT_BLOCKED:pnl={pnl_rate * 100:.2f}%,"
                    f"required={required_exit_return * 100:.2f}%,net={cost_floor.net_expected_return * 100:.2f}%"
                )
                reason = None
        if reason is None:
            return SharedDecisionResult(symbol, False, None, prediction, (blocked_reason or "HOLD_BELOW_PROFIT_TARGET",))

        market = self._exit_market_snapshot(holding, price, observed_at, received_at)
        intent = OrderIntent(
            ticker=symbol,
            market=holding.market or "KR",
            action=OrderAction.SELL,
            suggested_weight=0.0,
            confidence=0.9,
            valid_until=decision_time + timedelta(minutes=1),
            reasoning_summary=(f"realtime_exit:{reason}",),
            supporting_factors=("realtime_exit", "ProfitProtectedExit"),
            contradicting_factors=(),
            source_data_ids=(source_id,),
            strategy_family="live_short_horizon_exit",
            signal_name=reason.split(":", 1)[0],
            expected_exit_price=required_exit_price,
            gross_expected_return=max(0.0, required_exit_return),
            target_net_return=target_net_return,
            cost_breakdown=cost_floor.as_dict(),
        )
        risk = self._exit_risk_manager().validate(intent, account, market)
        return SharedDecisionResult(
            symbol=symbol,
            approved=risk.approved and risk.final_order is not None,
            final_order=risk.final_order,
            prediction=prediction,
            reason_codes=risk.rejection_reasons,
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

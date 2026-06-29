from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.data.realtime_store import RealtimeMarketDataStore
from app.features.live_feature_frame import LiveFeatureFrameBuilder
from app.models.live_signal_predictor import LiveSignalPredictor, LiveSignalPrediction
from app.risk import RiskManager
from app.schemas.domain import AccountSnapshot, FinalOrder, MarketSnapshot, OrderAction, OrderIntent, RiskRules, SourceMetadata


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
        decision_time: datetime | None = None,
    ) -> SharedDecisionResult:
        decision_time = decision_time or datetime.now(timezone.utc)
        try:
            frame = self.feature_builder.build(symbol, decision_time=decision_time)
            prediction = self.predictor.predict(frame)
        except Exception as exc:  # noqa: BLE001 - decision engine returns gate reason.
            return SharedDecisionResult(symbol, False, None, None, (exc.__class__.__name__, str(exc)))
        if not prediction.approved:
            return SharedDecisionResult(symbol, False, None, prediction, prediction.reason_codes)
        tick = self.store.latest_tick(symbol)
        orderbook = self.store.latest_orderbook(symbol)
        if tick is None or orderbook is None:
            return SharedDecisionResult(symbol, False, None, prediction, ("MISSING_MARKET_DATA",))
        expected_exit_price = tick.price * (1.0 + prediction.expected_net_return_bps / 10_000.0)
        market = MarketSnapshot(
            ticker=symbol,
            market="KR",
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
            market="KR",
            action=OrderAction.BUY,
            suggested_weight=suggested_weight,
            confidence=prediction.probability_success,
            valid_until=decision_time + timedelta(minutes=1),
            reasoning_summary=(f"model:{prediction.model_artifact_id}",),
            supporting_factors=("trained_live_model",),
            contradicting_factors=(),
            source_data_ids=frame.provenance.source_record_ids,
            model_uncertainty=prediction.uncertainty_score,
            strategy_family="live_short_horizon",
            signal_name="trained_expected_net_return",
            expected_exit_price=expected_exit_price,
            expected_holding_minutes=5,
            gross_expected_return=prediction.expected_net_return_bps / 10_000.0,
            target_net_return=0.0,
            validation_id=prediction.model_artifact_id,
            strategy_metadata={
                "feature_schema_hash": prediction.feature_schema_hash,
                "model_artifact_id": prediction.model_artifact_id,
                "orderbook_snapshot": {
                    "best_bid": orderbook.best_bid,
                    "best_ask": orderbook.best_ask,
                    "bid_depth": orderbook.total_bid_volume,
                    "ask_depth": orderbook.total_ask_volume,
                },
                "stop_loss_price": tick.price * 0.99,
            },
        )
        risk = self.risk_manager.validate(intent, account, market)
        return SharedDecisionResult(
            symbol=symbol,
            approved=risk.approved and risk.final_order is not None,
            final_order=risk.final_order,
            prediction=prediction,
            reason_codes=risk.rejection_reasons,
        )

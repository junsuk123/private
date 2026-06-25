from __future__ import annotations

from dataclasses import dataclass

from app.schemas.domain import MarketSnapshot, OrderAction, OrderIntent


@dataclass(frozen=True)
class ShortHorizonSignal:
    ticker: str
    horizon_seconds: int
    expected_return: float
    downside_risk: float
    confidence: float
    action: OrderAction
    reason: str


@dataclass(frozen=True)
class ShortHorizonRiskPolicy:
    max_horizon_seconds: int = 3600
    min_confidence_for_buy: float = 0.58
    max_downside_risk_for_buy: float = 0.012
    max_expected_loss_before_reduce: float = 0.006
    max_position_weight_intraday: float = 0.025
    emergency_exit_loss: float = 0.018

    def classify(self, ticker: str, horizon_seconds: int, expected_return: float, downside_risk: float, confidence: float) -> ShortHorizonSignal:
        if horizon_seconds > self.max_horizon_seconds:
            return ShortHorizonSignal(ticker, horizon_seconds, expected_return, downside_risk, confidence, OrderAction.HOLD, "horizon_too_long")
        if downside_risk >= self.emergency_exit_loss or expected_return <= -self.max_expected_loss_before_reduce:
            return ShortHorizonSignal(ticker, horizon_seconds, expected_return, downside_risk, confidence, OrderAction.REDUCE, "short_horizon_drawdown_guard")
        if confidence >= self.min_confidence_for_buy and expected_return > downside_risk and downside_risk <= self.max_downside_risk_for_buy:
            return ShortHorizonSignal(ticker, horizon_seconds, expected_return, downside_risk, confidence, OrderAction.BUY, "short_horizon_positive_edge")
        return ShortHorizonSignal(ticker, horizon_seconds, expected_return, downside_risk, confidence, OrderAction.HOLD, "edge_not_strong_enough")

    def cap_intent(self, intent: OrderIntent) -> OrderIntent:
        if intent.action == OrderAction.BUY and intent.suggested_weight > self.max_position_weight_intraday:
            return OrderIntent(
                ticker=intent.ticker,
                market=intent.market,
                action=intent.action,
                suggested_weight=self.max_position_weight_intraday,
                confidence=intent.confidence,
                valid_until=intent.valid_until,
                reasoning_summary=(*intent.reasoning_summary, "Intraday position capped by short-horizon risk policy."),
                supporting_factors=intent.supporting_factors,
                contradicting_factors=intent.contradicting_factors,
                source_data_ids=intent.source_data_ids,
            )
        return intent

    def market_is_allowed(self, market: MarketSnapshot) -> bool:
        return market.volatility_20d <= 0.12 and market.average_daily_trading_value > 0

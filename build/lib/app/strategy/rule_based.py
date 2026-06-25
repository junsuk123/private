from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.graph import KnowledgeGraph
from app.schemas.domain import IndicatorSnapshot, MarketSnapshot, OrderAction, OrderIntent, StrategySignal


def generate_strategy_signals(
    markets: tuple[MarketSnapshot, ...],
    indicators: dict[str, IndicatorSnapshot],
    graph: KnowledgeGraph,
) -> tuple[StrategySignal, ...]:
    signals: list[StrategySignal] = []

    for market in markets:
        indicator = indicators.get(market.ticker)
        if indicator is None:
            signals.append(
                StrategySignal(
                    ticker=market.ticker,
                    action=OrderAction.HOLD,
                    confidence=0.0,
                    score=0.0,
                    supporting_factors=(),
                    contradicting_factors=("MissingIndicators",),
                    reasoning_path_ids=(),
                )
            )
            continue

        score = 0.0
        supporting: list[str] = []
        contradicting: list[str] = []

        if (indicator.revenue_growth or 0) > 0.08:
            score += 1.0
            supporting.append("RevenueGrowth")
        if (indicator.operating_income_growth or 0) > 0.15:
            score += 1.0
            supporting.append("EarningsGrowth")
        if (indicator.operating_margin or 0) > 0.15:
            score += 1.0
            supporting.append("ProfitabilityQuality")
        if indicator.per is not None and indicator.per > 20:
            score -= 0.8
            contradicting.append("ValuationSlightlyHigh")
        if indicator.macro_risk_score > 0.40:
            score -= 0.6
            contradicting.append("MacroRateRisk")
        if market.volatility_20d > 0.06:
            score -= 1.0
            contradicting.append("VolatilityRisk")

        action = OrderAction.BUY if score >= 1.8 else OrderAction.HOLD
        confidence = max(0.0, min(0.85, 0.45 + score * 0.1))

        signals.append(
            StrategySignal(
                ticker=market.ticker,
                action=action,
                confidence=confidence,
                score=score,
                supporting_factors=tuple(supporting),
                contradicting_factors=tuple(contradicting),
                reasoning_path_ids=graph.reasoning_path_ids(market.ticker),
            )
        )

    return tuple(signals)


def generate_order_intents(
    markets: tuple[MarketSnapshot, ...],
    indicators: dict[str, IndicatorSnapshot],
    signals: tuple[StrategySignal, ...],
) -> tuple[OrderIntent, ...]:
    market_by_ticker = {market.ticker: market for market in markets}
    intents: list[OrderIntent] = []

    for signal in signals:
        if signal.action is not OrderAction.BUY:
            continue

        market = market_by_ticker[signal.ticker]
        indicator = indicators[signal.ticker]
        suggested_weight = min(0.05, max(0.01, signal.confidence * 0.05))

        intents.append(
            OrderIntent(
                ticker=signal.ticker,
                market=market.market,
                action=signal.action,
                suggested_weight=suggested_weight,
                confidence=signal.confidence,
                valid_until=datetime.now(timezone.utc) + timedelta(hours=6),
                reasoning_summary=(
                    "Positive growth and profitability indicators support a buy candidate.",
                    "Contradicting factors are retained for deterministic risk review.",
                ),
                supporting_factors=signal.supporting_factors,
                contradicting_factors=signal.contradicting_factors,
                source_data_ids=indicator.source_ids,
            )
        )

    return tuple(intents)

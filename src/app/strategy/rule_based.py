from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.graph import KnowledgeGraph
from app.schemas.domain import IndicatorSnapshot, MarketSnapshot, OrderAction, OrderIntent, StrategySignal


ONTOLOGY_FLOW_SUPPORT_WEIGHTS = {
    "InformedOrderFlowImbalance": 0.35,
    "ForeignInstitutionJointBuying": 0.25,
    "RetailSupplyAbsorbedByInformedFlow": 0.20,
    "OrderFlowPriceConfirmation": 0.18,
    "SuspectedSmartMoneyAccumulation": 0.14,
    "OrderFlowConfirmedBuyCandidate": 0.20,
}
ONTOLOGY_FLOW_CONTRA_WEIGHTS = {
    "InformedOrderFlowDistribution": 0.75,
    "ForeignInstitutionJointSelling": 0.55,
    "RetailDemandMeetsInformedSelling": 0.36,
    "OrderFlowPriceDivergence": 0.24,
    "SuspectedSmartMoneyDistribution": 0.30,
    "BuyCandidate": 0.25,
}


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

        flow_score, flow_support, flow_contra = _ontology_flow_adjustment(graph, market.ticker)
        score += flow_score
        supporting.extend(flow_support)
        contradicting.extend(flow_contra)

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
        if "InformedOrderFlowImbalance" in signal.supporting_factors:
            suggested_weight = min(0.05, suggested_weight * 1.08)

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
                    "Domestic investor-flow evidence is supplied by ontology triples when available.",
                ),
                supporting_factors=signal.supporting_factors,
                contradicting_factors=signal.contradicting_factors,
                source_data_ids=indicator.source_ids,
            )
        )

    return tuple(intents)


def _ontology_flow_adjustment(graph: KnowledgeGraph, ticker: str) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
    support_objects = tuple(triple.object for triple in graph.matching(subject=ticker, predicate="supportsSignal"))
    contra_objects = tuple(triple.object for triple in graph.matching(subject=ticker, predicate="contradictsSignal"))
    support = tuple(item for item in support_objects if item in ONTOLOGY_FLOW_SUPPORT_WEIGHTS)
    contra = tuple(item for item in contra_objects if item in ONTOLOGY_FLOW_CONTRA_WEIGHTS)
    support_score = sum(ONTOLOGY_FLOW_SUPPORT_WEIGHTS[item] for item in support)
    contra_score = sum(ONTOLOGY_FLOW_CONTRA_WEIGHTS[item] for item in contra)
    return round(support_score - contra_score, 4), support, contra

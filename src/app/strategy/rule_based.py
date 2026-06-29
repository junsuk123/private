from __future__ import annotations

import hashlib
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
MARKET_CONTEXT_BUY_THRESHOLD = 1.0


def generate_strategy_signals(
    markets: tuple[MarketSnapshot, ...],
    indicators: dict[str, IndicatorSnapshot],
    graph: KnowledgeGraph,
) -> tuple[StrategySignal, ...]:
    signals: list[StrategySignal] = []

    for market in markets:
        indicator = indicators.get(market.ticker)
        if indicator is None:
            score = 0.0
            supporting: list[str] = []
            contradicting: list[str] = ["MissingFundamentalIndicators"]
            market_score, market_support, market_contra = _market_context_adjustment(market)
            flow_score, flow_support, flow_contra = _ontology_flow_adjustment(graph, market.ticker)
            score += market_score + flow_score
            supporting.extend(market_support)
            supporting.extend(flow_support)
            contradicting.extend(market_contra)
            contradicting.extend(flow_contra)
            action = OrderAction.BUY if score >= MARKET_CONTEXT_BUY_THRESHOLD else OrderAction.HOLD
            confidence = max(0.0, min(0.72, 0.38 + score * 0.12))
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
        indicator = indicators.get(signal.ticker)
        suggested_weight = min(0.05, max(0.01, signal.confidence * 0.05))
        if "InformedOrderFlowImbalance" in signal.supporting_factors:
            suggested_weight = min(0.05, suggested_weight * 1.08)
        gross_expected_return = max(0.012, min(0.08, signal.score * 0.012))

        intents.append(
            OrderIntent(
                ticker=signal.ticker,
                market=market.market,
                action=signal.action,
                suggested_weight=suggested_weight,
                confidence=signal.confidence,
                valid_until=datetime.now(timezone.utc) + timedelta(hours=6),
                reasoning_summary=_intent_reasoning_summary(indicator is not None),
                supporting_factors=signal.supporting_factors,
                contradicting_factors=signal.contradicting_factors,
                strategy_family="rule_based",
                signal_name="fundamental_ontology_buy" if indicator is not None else "market_context_ontology_buy",
                expected_exit_price=market.last_price * (1 + gross_expected_return),
                expected_holding_minutes=360,
                gross_expected_return=gross_expected_return,
                target_net_return=0.0,
                ontology_tags=tuple(signal.supporting_factors),
                source_data_ids=(
                    indicator.source_ids
                    if indicator is not None
                    else (market.source.source_id or f"market:{market.ticker}",)
                ),
                validation_id=_live_market_validation_id(market, signal),
                strategy_metadata={"score": signal.score, "indicator_available": indicator is not None},
            )
        )

    return tuple(intents)


def _market_context_adjustment(market: MarketSnapshot) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
    score = 0.0
    supporting: list[str] = []
    contradicting: list[str] = []
    if market.average_daily_trading_value >= 1_000_000_000:
        score += 0.45
        supporting.append("HighLiquidity")
    else:
        contradicting.append("LowLiquidity")
    if 0 < market.volatility_20d <= 0.06:
        score += 0.35
        supporting.append("ControlledVolatility")
    elif market.volatility_20d > 0.08:
        score -= 0.45
        contradicting.append("HighVolatility")
    if _is_overseas_market(market) and market.average_daily_trading_value >= 1_000_000_000:
        score += 0.25
        supporting.append("OverseasLiquidVenue")
    flow = market.investor_flow
    if flow is not None:
        if flow.price_change_rate >= 0.012:
            score += 0.45
            supporting.append("PositivePriceMomentum")
        elif flow.price_change_rate <= -0.012:
            score -= 0.35
            contradicting.append("NegativePriceMomentum")
        if flow.volume_change_rate >= 0.5:
            score += 0.25
            supporting.append("VolumeExpansion")
    return score, tuple(supporting), tuple(contradicting)


def _intent_reasoning_summary(has_indicator: bool) -> tuple[str, ...]:
    if has_indicator:
        return (
            "Positive growth and profitability indicators support a buy candidate.",
            "Contradicting factors are retained for deterministic risk review.",
            "Domestic investor-flow evidence is supplied by ontology triples when available.",
        )
    return (
        "Trusted fundamental indicators are unavailable, so the candidate uses market context only.",
        "Liquidity, volatility, venue, and ontology flow evidence are retained for deterministic risk review.",
        "Live execution still requires RiskManager, reality-check validation, and runtime arming gates.",
    )


def _is_overseas_market(market: MarketSnapshot) -> bool:
    market_name = str(market.market or "").upper()
    ticker = str(market.ticker or "").upper()
    domestic = market_name in {"KRX", "KOSPI", "KOSDAQ", "KONEX"} or ticker.endswith(".KS") or ticker.isdigit()
    return not domestic


def _live_market_validation_id(market: MarketSnapshot, signal: StrategySignal) -> str | None:
    source = market.source
    if source.source_type != "broker_api":
        return None
    if source.trust_level < 5 or source.quality_score < 0.8 or not source.is_realtime:
        return None
    observed_at = source.observed_at or source.retrieved_at
    payload = "|".join(
        (
            market.ticker,
            market.market,
            f"{market.last_price:.8f}",
            f"{signal.score:.6f}",
            observed_at.isoformat(),
            source.source_id or "",
        )
    )
    return "broker-reality-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _ontology_flow_adjustment(graph: KnowledgeGraph, ticker: str) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
    support_objects = tuple(triple.object for triple in graph.matching(subject=ticker, predicate="supportsSignal"))
    contra_objects = tuple(triple.object for triple in graph.matching(subject=ticker, predicate="contradictsSignal"))
    support = tuple(item for item in support_objects if item in ONTOLOGY_FLOW_SUPPORT_WEIGHTS)
    contra = tuple(item for item in contra_objects if item in ONTOLOGY_FLOW_CONTRA_WEIGHTS)
    support_score = sum(ONTOLOGY_FLOW_SUPPORT_WEIGHTS[item] for item in support)
    contra_score = sum(ONTOLOGY_FLOW_CONTRA_WEIGHTS[item] for item in contra)
    return round(support_score - contra_score, 4), support, contra

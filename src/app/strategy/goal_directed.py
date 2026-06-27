from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.goals import NegotiatedGoal
from app.graph import KnowledgeGraph
from app.schemas.domain import (
    AccountSnapshot,
    IndicatorSnapshot,
    MarketSnapshot,
    OrderAction,
    OrderIntent,
    StrategySignal,
)
from app.strategy.rule_based import _ontology_flow_adjustment


@dataclass(frozen=True)
class GoalExecutionPlan:
    goal: NegotiatedGoal
    required_period_return: float
    annualized_required_return: float
    signals: tuple[StrategySignal, ...]
    intents: tuple[OrderIntent, ...]
    notes: tuple[str, ...]


def build_goal_execution_plan(
    goal: NegotiatedGoal,
    account: AccountSnapshot,
    markets: tuple[MarketSnapshot, ...],
    indicators: dict[str, IndicatorSnapshot],
    graph: KnowledgeGraph,
) -> GoalExecutionPlan:
    annualized = (1 + goal.target_return_rate) ** (365 / goal.period_days) - 1
    signals = tuple(
        _score_market(goal, annualized, market, indicators.get(market.ticker), graph)
        for market in markets
    )
    intents = _build_goal_intents(goal, account, markets, indicators, signals)
    notes = (
        f"Target return {goal.target_return_rate * 100:.2f}% over {goal.period_days} days.",
        f"Annualized required return is {annualized * 100:.2f}%.",
        "Signals combine ontology support/risk with RSI, volume, valuation, liquidity, and volatility rules.",
        "Orders are generated as paper-trading intents and still pass deterministic risk checks.",
    )
    return GoalExecutionPlan(
        goal=goal,
        required_period_return=goal.target_return_rate,
        annualized_required_return=annualized,
        signals=signals,
        intents=intents,
        notes=notes,
    )


def _score_market(
    goal: NegotiatedGoal,
    annualized_required_return: float,
    market: MarketSnapshot,
    indicator: IndicatorSnapshot | None,
    graph: KnowledgeGraph,
) -> StrategySignal:
    if indicator is None:
        return StrategySignal(
            ticker=market.ticker,
            action=OrderAction.HOLD,
            confidence=0.05,
            score=-2.0,
            supporting_factors=(),
            contradicting_factors=("MissingIndicators",),
            reasoning_path_ids=graph.reasoning_path_ids(market.ticker),
        )

    score = 0.0
    support: list[str] = []
    contradiction: list[str] = []

    ontology_support = graph.matching(subject=market.ticker, predicate="supportsSignal")
    ontology_risk = graph.matching(subject=market.ticker, predicate="increasesRiskOf")
    ontology_contra = graph.matching(subject=market.ticker, predicate="contradictsSignal")
    if ontology_support:
        score += min(1.6, len(ontology_support) * 0.35)
        support.append("OntologySupport")
    if ontology_risk:
        score -= min(1.4, len(ontology_risk) * 0.45)
        contradiction.append("OntologyRisk")
    if ontology_contra:
        score -= min(1.2, len(ontology_contra) * 0.40)
        contradiction.append("OntologyContradiction")

    if indicator.rsi_14d is not None:
        if 45 <= indicator.rsi_14d <= 68:
            score += 0.9
            support.append("RSIHealthyTrend")
        elif indicator.rsi_14d < 32:
            score += 0.4
            support.append("RSIOversoldRebound")
        elif indicator.rsi_14d > 74:
            score -= 1.2
            contradiction.append("RSIOverbought")

    if indicator.volume_ratio is not None:
        if indicator.volume_ratio >= 1.15:
            score += 0.55
            support.append("VolumeConfirmation")
        elif indicator.volume_ratio < 0.70:
            score -= 0.45
            contradiction.append("WeakVolume")

    if (indicator.operating_income_growth or 0) > 0.15:
        score += 0.75
        support.append("EarningsGrowth")
    if (indicator.operating_margin or 0) > 0.15:
        score += 0.55
        support.append("ProfitabilityQuality")
    if indicator.per is not None and indicator.per > 25:
        score -= 0.65
        contradiction.append("ValuationHigh")
    elif indicator.per is not None and indicator.per < 18:
        score += 0.35
        support.append("ValuationReasonable")

    if market.volatility_20d > 0.06:
        score -= 1.2
        contradiction.append("HighVolatility")
    elif market.volatility_20d < 0.035:
        score += 0.35
        support.append("ControlledVolatility")

    if indicator.macro_risk_score > 0.55:
        score -= 0.9
        contradiction.append("MacroRiskHigh")

    flow_score, flow_support, flow_contra = _ontology_flow_adjustment(graph, market.ticker)
    score += flow_score
    support.extend(flow_support)
    contradiction.extend(flow_contra)

    compounding_mode = "Principal-preserving" in goal.label
    if compounding_mode:
        score += 0.85
        score -= min(0.35, max(0.0, annualized_required_return - 0.20) * 0.25)
    else:
        score -= min(1.3, max(0.0, annualized_required_return - 0.20) * 1.2)
    if goal.feasibility_percent < 35:
        score -= 0.9
        contradiction.append("LowGoalFeasibility")
    score += min(0.5, max(0, goal.feasibility_percent - 50) / 100)

    action = OrderAction.REDUCE if goal.feasibility_percent < 35 and score < 0.5 else _action_from_score(score, compounding_mode)
    confidence = max(0.05, min(0.92, 0.48 + score * 0.10))
    return StrategySignal(
        ticker=market.ticker,
        action=action,
        confidence=confidence,
        score=round(score, 4),
        supporting_factors=tuple(support),
        contradicting_factors=tuple(contradiction),
        reasoning_path_ids=graph.reasoning_path_ids(market.ticker),
    )


def _action_from_score(score: float, compounding_mode: bool = False) -> OrderAction:
    buy_threshold = 1.25 if compounding_mode else 2.2
    reduce_threshold = -0.8 if compounding_mode else -0.35
    if score >= buy_threshold:
        return OrderAction.BUY
    if score <= -1.1:
        return OrderAction.SELL
    if score <= reduce_threshold:
        return OrderAction.REDUCE
    return OrderAction.HOLD


def _build_goal_intents(
    goal: NegotiatedGoal,
    account: AccountSnapshot,
    markets: tuple[MarketSnapshot, ...],
    indicators: dict[str, IndicatorSnapshot],
    signals: tuple[StrategySignal, ...],
) -> tuple[OrderIntent, ...]:
    market_by_ticker = {market.ticker: market for market in markets}
    holding_values = account.holdings_by_ticker()
    intents: list[OrderIntent] = []
    buy_signals = [signal for signal in signals if signal.action == OrderAction.BUY]
    compounding_mode = "Principal-preserving" in goal.label
    if compounding_mode:
        max_goal_weight = min(0.50, max(0.18, 0.24 + min(goal.target_return_rate, 0.25) * 0.80))
    else:
        max_goal_weight = min(0.06, max(0.015, 0.025 + goal.target_return_rate))

    for signal in signals:
        if signal.action == OrderAction.HOLD:
            continue
        market = market_by_ticker[signal.ticker]
        indicator = indicators.get(signal.ticker)
        current_weight = holding_values.get(signal.ticker, 0.0) / max(1.0, account.equity)

        if signal.action == OrderAction.BUY:
            rank_bonus = (buy_signals.index(signal) + 1) / max(1, len(buy_signals))
            if compounding_mode:
                suggested_weight = min(max_goal_weight, max(0.12, signal.confidence * 0.34 + rank_bonus * 0.04))
            else:
                suggested_weight = min(max_goal_weight, max(0.01, signal.confidence * 0.04 + rank_bonus * 0.004))
            if "InformedOrderFlowImbalance" in signal.supporting_factors:
                suggested_weight = min(max_goal_weight, suggested_weight * 1.08)
            gross_expected_return = max(0.01, min(0.12, goal.target_return_rate + max(0.0, signal.score) * 0.004))
            expected_exit_price = market.last_price * (1 + gross_expected_return)
        elif signal.action == OrderAction.REDUCE:
            if current_weight <= 0:
                continue
            reduction_ratio = 0.70 if any("OrderFlow" in item or "Selling" in item for item in signal.contradicting_factors) else 0.50
            suggested_weight = max(0.0, current_weight * reduction_ratio)
            gross_expected_return = None
            expected_exit_price = None
        else:
            if current_weight <= 0:
                continue
            suggested_weight = 0.0
            gross_expected_return = None
            expected_exit_price = None

        intents.append(
            OrderIntent(
                ticker=signal.ticker,
                market=market.market,
                action=signal.action,
                suggested_weight=suggested_weight,
                confidence=signal.confidence,
                valid_until=datetime.now(timezone.utc) + timedelta(hours=3),
                reasoning_summary=(
                    f"Goal-directed {signal.action.value} based on ontology and chart score {signal.score:.2f}.",
                    f"Target feasibility is {goal.feasibility_percent}% for {goal.period_days} days.",
                    "Domestic investor-flow formulas are represented as ontology evidence when available.",
                ),
                supporting_factors=signal.supporting_factors,
                contradicting_factors=signal.contradicting_factors,
                source_data_ids=indicator.source_ids if indicator is not None else (market.source.source_id or market.ticker,),
                strategy_family="goal_directed",
                signal_name=f"goal_{signal.action.value.lower()}",
                expected_exit_price=expected_exit_price,
                expected_holding_minutes=max(1, goal.period_days * 390),
                gross_expected_return=gross_expected_return,
                target_net_return=0.0 if signal.action == OrderAction.BUY else None,
                ontology_tags=tuple(signal.supporting_factors),
                strategy_metadata={
                    "score": signal.score,
                    "goal_target_return_rate": goal.target_return_rate,
                    "goal_period_days": goal.period_days,
                },
            )
        )

    return tuple(intents)

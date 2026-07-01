from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import expm1, isfinite, log1p

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


MAX_ANNUALIZED_REQUIRED_RETURN = 1_000_000.0


@dataclass(frozen=True)
class GoalExecutionPlan:
    goal: NegotiatedGoal
    required_period_return: float
    annualized_required_return: float
    signals: tuple[StrategySignal, ...]
    intents: tuple[OrderIntent, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceVote:
    name: str
    action: OrderAction
    confidence: float
    reliability: float
    direction: float


def build_goal_execution_plan(
    goal: NegotiatedGoal,
    account: AccountSnapshot,
    markets: tuple[MarketSnapshot, ...],
    indicators: dict[str, IndicatorSnapshot],
    graph: KnowledgeGraph,
) -> GoalExecutionPlan:
    annualized = _safe_annualized_return(goal.target_return_rate, goal.period_days)
    signals = tuple(
        _score_market(goal, annualized, market, indicators.get(market.ticker), graph)
        for market in markets
    )
    intents = _build_goal_intents(goal, account, markets, indicators, signals)
    notes = (
        f"Target return {goal.target_return_rate * 100:.2f}% over {goal.period_days} days.",
        f"Annualized required return is {annualized * 100:.2f}%.",
        "Signals are selected by reliability-ranked evidence instead of a flat additive score.",
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

    support: list[str] = []
    contradiction: list[str] = []
    votes: list[EvidenceVote] = []

    ontology_support = graph.matching(subject=market.ticker, predicate="supportsSignal")
    ontology_risk = graph.matching(subject=market.ticker, predicate="increasesRiskOf")
    ontology_contra = graph.matching(subject=market.ticker, predicate="contradictsSignal")
    if ontology_support:
        support.append("OntologySupport")
        votes.append(_vote("OntologySupport", OrderAction.BUY, min(0.90, 0.55 + len(ontology_support) * 0.08), 0.78, 1.0))
    if ontology_risk:
        contradiction.append("OntologyRisk")
        votes.append(_vote("OntologyRisk", OrderAction.REDUCE, min(0.92, 0.58 + len(ontology_risk) * 0.10), 0.82, -1.0))
    if ontology_contra:
        contradiction.append("OntologyContradiction")
        votes.append(_vote("OntologyContradiction", OrderAction.REDUCE, min(0.88, 0.56 + len(ontology_contra) * 0.10), 0.80, -1.0))

    if indicator.rsi_14d is not None:
        if 45 <= indicator.rsi_14d <= 68:
            support.append("RSIHealthyTrend")
            votes.append(_vote("RSIHealthyTrend", OrderAction.BUY, 0.72, 0.62, 1.0))
        elif indicator.rsi_14d < 32:
            support.append("RSIOversoldRebound")
            votes.append(_vote("RSIOversoldRebound", OrderAction.BUY, 0.58, 0.48, 0.7))
        elif indicator.rsi_14d > 74:
            contradiction.append("RSIOverbought")
            votes.append(_vote("RSIOverbought", OrderAction.REDUCE, 0.76, 0.66, -1.0))

    if indicator.volume_ratio is not None:
        if indicator.volume_ratio >= 1.15:
            support.append("VolumeConfirmation")
            votes.append(_vote("VolumeConfirmation", OrderAction.BUY, min(0.82, 0.54 + indicator.volume_ratio * 0.10), 0.58, 0.8))
        elif indicator.volume_ratio < 0.70:
            contradiction.append("WeakVolume")
            votes.append(_vote("WeakVolume", OrderAction.HOLD, 0.60, 0.50, -0.5))

    if (indicator.operating_income_growth or 0) > 0.15:
        support.append("EarningsGrowth")
        votes.append(_vote("EarningsGrowth", OrderAction.BUY, 0.68, 0.55, 0.8))
    if (indicator.operating_margin or 0) > 0.15:
        support.append("ProfitabilityQuality")
        votes.append(_vote("ProfitabilityQuality", OrderAction.BUY, 0.64, 0.52, 0.7))
    if indicator.per is not None and indicator.per > 25:
        contradiction.append("ValuationHigh")
        votes.append(_vote("ValuationHigh", OrderAction.HOLD, 0.62, 0.46, -0.5))
    elif indicator.per is not None and indicator.per < 18:
        support.append("ValuationReasonable")
        votes.append(_vote("ValuationReasonable", OrderAction.BUY, 0.55, 0.40, 0.45))

    if market.volatility_20d > 0.06:
        contradiction.append("HighVolatility")
        votes.append(_vote("HighVolatility", OrderAction.REDUCE, min(0.88, 0.58 + market.volatility_20d * 3.0), 0.72, -1.0))
    elif market.volatility_20d < 0.035:
        support.append("ControlledVolatility")
        votes.append(_vote("ControlledVolatility", OrderAction.BUY, 0.56, 0.45, 0.45))

    if indicator.macro_risk_score > 0.55:
        contradiction.append("MacroRiskHigh")
        votes.append(_vote("MacroRiskHigh", OrderAction.REDUCE, min(0.88, 0.56 + indicator.macro_risk_score * 0.25), 0.74, -0.9))

    flow_score, flow_support, flow_contra = _ontology_flow_adjustment(graph, market.ticker)
    support.extend(flow_support)
    contradiction.extend(flow_contra)
    if flow_support:
        votes.append(_vote("OrderFlowSupport", OrderAction.BUY, min(0.88, 0.58 + max(0.0, flow_score) * 0.25), 0.76, 1.0))
    if flow_contra:
        votes.append(_vote("OrderFlowContradiction", OrderAction.REDUCE, min(0.90, 0.58 + abs(min(0.0, flow_score)) * 0.25), 0.80, -1.0))

    compounding_mode = "Principal-preserving" in goal.label
    goal_drag = min(0.95, max(0.0, annualized_required_return - 0.20) * (0.25 if compounding_mode else 0.90))
    if compounding_mode:
        votes.append(_vote("PrincipalPreservingMode", OrderAction.BUY, 0.66, 0.50, 0.6))
    if goal_drag > 0:
        contradiction.append("GoalDifficultyDrag")
        votes.append(_vote("GoalDifficultyDrag", OrderAction.HOLD, 0.50 + goal_drag * 0.35, 0.64, -goal_drag))
    if goal.feasibility_percent < 35:
        contradiction.append("LowGoalFeasibility")
        votes.append(_vote("LowGoalFeasibility", OrderAction.REDUCE, 0.74, 0.70, -0.9))
    elif goal.feasibility_percent > 50:
        votes.append(_vote("GoalFeasibilitySupport", OrderAction.BUY, min(0.72, 0.50 + (goal.feasibility_percent - 50) / 100), 0.44, 0.45))

    action, confidence, score = _select_signal_from_votes(votes, compounding_mode=compounding_mode)
    return StrategySignal(
        ticker=market.ticker,
        action=action,
        confidence=confidence,
        score=round(score, 4),
        supporting_factors=tuple(support),
        contradicting_factors=tuple(contradiction),
        reasoning_path_ids=graph.reasoning_path_ids(market.ticker),
    )


def _vote(name: str, action: OrderAction, confidence: float, reliability: float, direction: float) -> EvidenceVote:
    return EvidenceVote(
        name=name,
        action=action,
        confidence=max(0.0, min(1.0, confidence)),
        reliability=max(0.0, min(1.0, reliability)),
        direction=max(-1.0, min(1.0, direction)),
    )


def _select_signal_from_votes(votes: list[EvidenceVote], *, compounding_mode: bool) -> tuple[OrderAction, float, float]:
    if not votes:
        return OrderAction.HOLD, 0.05, -2.0

    ranked = sorted(votes, key=lambda item: (item.reliability * item.confidence, item.reliability), reverse=True)
    lead = ranked[0]
    lead_quality = lead.reliability * lead.confidence
    same_quality = sum(item.reliability * item.confidence for item in ranked if item.action == lead.action)
    opposite_quality = sum(
        item.reliability * item.confidence
        for item in ranked
        if _is_opposing_action(item.action, lead.action)
    )
    buy_quality = sum(item.reliability * item.confidence for item in ranked if item.action == OrderAction.BUY)
    risk_quality = sum(item.reliability * item.confidence for item in ranked if item.action == OrderAction.REDUCE)
    total_quality = max(lead_quality, same_quality + opposite_quality)
    agreement = same_quality / max(0.01, total_quality)
    contradiction = opposite_quality / max(0.01, total_quality)

    score = round((buy_quality - risk_quality) + lead.direction * lead_quality - contradiction * 0.5, 4)
    if risk_quality > buy_quality and risk_quality >= 0.75:
        action = OrderAction.SELL if risk_quality >= buy_quality + 0.65 else OrderAction.REDUCE
    elif lead.action == OrderAction.REDUCE and lead_quality >= 0.42:
        action = OrderAction.SELL if risk_quality >= buy_quality + 0.65 else OrderAction.REDUCE
    elif buy_quality > risk_quality and score > 0.20:
        action = OrderAction.BUY
    elif lead.action == OrderAction.BUY or buy_quality >= (0.50 if compounding_mode else 0.45):
        buy_threshold = 0.42 if compounding_mode else 0.52
        veto_threshold = 0.56 if compounding_mode else 0.42
        action = OrderAction.BUY if max(lead_quality, buy_quality) >= buy_threshold and contradiction <= veto_threshold else OrderAction.HOLD
    else:
        action = OrderAction.HOLD

    confidence = max(0.05, min(0.92, 0.36 + lead_quality * 0.42 + agreement * 0.18 - contradiction * 0.16))
    return action, confidence, score


def _is_opposing_action(action: OrderAction, lead_action: OrderAction) -> bool:
    if lead_action == OrderAction.BUY:
        return action in {OrderAction.REDUCE, OrderAction.SELL}
    if lead_action in {OrderAction.REDUCE, OrderAction.SELL}:
        return action == OrderAction.BUY
    return False


def _safe_annualized_return(return_rate: float, period_days: int) -> float:
    if period_days <= 0:
        raise ValueError("period_days must be positive")
    rate = float(return_rate)
    if not isfinite(rate) or rate <= 0:
        raise ValueError("target return must be a positive finite number")
    annualized_log_return = log1p(rate) * (365.0 / max(1, period_days))
    cap_log_return = log1p(MAX_ANNUALIZED_REQUIRED_RETURN)
    if annualized_log_return >= cap_log_return:
        return MAX_ANNUALIZED_REQUIRED_RETURN
    return expm1(annualized_log_return)


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

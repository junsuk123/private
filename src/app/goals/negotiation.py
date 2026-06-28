from __future__ import annotations

from dataclasses import dataclass
from math import exp

from app.graph import KnowledgeGraph
from app.portfolio import build_portfolio_report
from app.schemas.domain import AccountSnapshot, IndicatorSnapshot, MarketSnapshot, StrategySignal


@dataclass(frozen=True)
class GoalRequest:
    target_return_rate: float | None
    target_profit_amount: float | None
    period_days: int
    period_minutes: int | None = None


@dataclass(frozen=True)
class FeasibilityAssessment:
    requested_return_rate: float
    requested_profit_amount: float
    period_days: int
    annualized_required_return: float
    feasibility_percent: int
    market_support_percent: int
    risk_pressure_percent: int
    annualized_drag_percent: int
    period_minutes: int | None
    deployable_cash: float
    reasoning: tuple[str, ...]
    ontology_relations: tuple[str, ...]


@dataclass(frozen=True)
class NegotiatedGoal:
    target_return_rate: float
    target_profit_amount: float
    period_days: int
    feasibility_percent: int
    label: str
    period_minutes: int | None = None


def assess_goal(
    request: GoalRequest,
    account: AccountSnapshot,
    markets: tuple[MarketSnapshot, ...],
    indicators: dict[str, IndicatorSnapshot],
    signals: tuple[StrategySignal, ...],
    graph: KnowledgeGraph,
) -> FeasibilityAssessment:
    if request.period_days <= 0:
        raise ValueError("period_days must be positive")

    report = build_portfolio_report(account)
    if report.equity <= 0:
        raise ValueError("account equity must be positive")

    requested_return_rate = _resolve_requested_return_rate(request, report.equity)
    requested_profit_amount = requested_return_rate * report.equity
    annualized = (1 + requested_return_rate) ** (365 / request.period_days) - 1

    market_support = _market_support(markets, indicators, signals)
    risk_pressure = _risk_pressure(markets, indicators, report.cash_weight)
    goal_drag = _goal_difficulty_drag(requested_return_rate, request.period_days, request.period_minutes)
    feasibility = _combine_feasibility(market_support, risk_pressure, goal_drag)

    reasoning = (
        f"목표 기간 수익률은 {requested_return_rate * 100:.2f}%입니다.",
        f"연환산 요구 수익률은 {annualized * 100:.2f}%이며, 높을수록 가능성 점수에서 차감됩니다.",
        f"전략 신호와 온톨로지 요인에 따른 시장 지지 점수는 {market_support:.0f}%입니다.",
        f"변동성, 매크로 리스크, 현금 제약에 따른 리스크 압력은 {risk_pressure:.0f}%입니다.",
        "사용자가 타협 목표를 확정하기 전까지 프로그램 시작은 차단됩니다.",
    )

    return FeasibilityAssessment(
        requested_return_rate=requested_return_rate,
        requested_profit_amount=requested_profit_amount,
        period_days=request.period_days,
        annualized_required_return=annualized,
        feasibility_percent=feasibility,
        market_support_percent=round(market_support),
        risk_pressure_percent=round(risk_pressure),
        annualized_drag_percent=round(goal_drag),
        period_minutes=request.period_minutes,
        deployable_cash=max(0.0, account.cash - report.equity * 0.30),
        reasoning=reasoning,
        ontology_relations=_summarize_relations(graph),
    )


def build_compromise_goals(assessment: FeasibilityAssessment) -> tuple[NegotiatedGoal, ...]:
    current = NegotiatedGoal(
        target_return_rate=assessment.requested_return_rate,
        target_profit_amount=assessment.requested_profit_amount,
        period_days=assessment.period_days,
        feasibility_percent=assessment.feasibility_percent,
        label="Requested target",
        period_minutes=assessment.period_minutes,
    )

    lower_return = _estimate_feasibility(
        assessment.market_support_percent,
        assessment.risk_pressure_percent,
        assessment.requested_return_rate * 0.60,
        assessment.period_days,
        assessment.period_minutes,
    )
    return_adjusted = NegotiatedGoal(
        target_return_rate=assessment.requested_return_rate * 0.60,
        target_profit_amount=assessment.requested_profit_amount * 0.60,
        period_days=assessment.period_days,
        feasibility_percent=lower_return,
        label="Lower return",
        period_minutes=assessment.period_minutes,
    )

    longer_period = max(assessment.period_days + 30, round(assessment.period_days * 1.75))
    longer_minutes = (
        max(int(assessment.period_minutes * 1.75), int(assessment.period_minutes + 390))
        if assessment.period_minutes is not None
        else None
    )
    period_adjusted = NegotiatedGoal(
        target_return_rate=assessment.requested_return_rate,
        target_profit_amount=assessment.requested_profit_amount,
        period_days=longer_period,
        feasibility_percent=_estimate_feasibility(
            assessment.market_support_percent,
            assessment.risk_pressure_percent,
            assessment.requested_return_rate,
            longer_period,
            longer_minutes,
        ),
        label="Longer period",
        period_minutes=longer_minutes,
    )

    balanced_rate = assessment.requested_return_rate * 0.75
    balanced_period = max(assessment.period_days + 14, round(assessment.period_days * 1.40))
    balanced_minutes = (
        max(int(assessment.period_minutes * 1.40), int(assessment.period_minutes + 195))
        if assessment.period_minutes is not None
        else None
    )
    balanced = NegotiatedGoal(
        target_return_rate=balanced_rate,
        target_profit_amount=assessment.requested_profit_amount * 0.75,
        period_days=balanced_period,
        feasibility_percent=_estimate_feasibility(
            assessment.market_support_percent,
            assessment.risk_pressure_percent,
            balanced_rate,
            balanced_period,
            balanced_minutes,
        ),
        label="Balanced compromise",
        period_minutes=balanced_minutes,
    )

    return tuple(sorted((current, return_adjusted, period_adjusted, balanced), key=lambda g: -g.feasibility_percent))


def _resolve_requested_return_rate(request: GoalRequest, equity: float) -> float:
    if request.target_return_rate is not None and request.target_return_rate > 0:
        return request.target_return_rate
    if request.target_profit_amount is not None and request.target_profit_amount > 0:
        return request.target_profit_amount / equity
    raise ValueError("target_return_rate or target_profit_amount is required")


def _market_support(
    markets: tuple[MarketSnapshot, ...],
    indicators: dict[str, IndicatorSnapshot],
    signals: tuple[StrategySignal, ...],
) -> float:
    if not markets:
        return 20.0

    buy_confidence = [signal.confidence for signal in signals if signal.action == "BUY"]
    signal_score = (sum(buy_confidence) / len(buy_confidence) * 35) if buy_confidence else 8
    growth_score = 0.0
    for indicator in indicators.values():
        growth_score += max(0.0, min(1.0, (indicator.revenue_growth or 0) / 0.20)) * 7
        growth_score += max(0.0, min(1.0, (indicator.operating_income_growth or 0) / 0.35)) * 8
        growth_score += max(0.0, min(1.0, (indicator.operating_margin or 0) / 0.25)) * 5

    return min(78.0, 24.0 + signal_score + growth_score / max(1, len(indicators)))


def _risk_pressure(
    markets: tuple[MarketSnapshot, ...],
    indicators: dict[str, IndicatorSnapshot],
    cash_weight: float,
) -> float:
    if not markets:
        return 60.0

    avg_volatility = sum(market.volatility_20d for market in markets) / len(markets)
    avg_macro = sum(indicator.macro_risk_score for indicator in indicators.values()) / max(1, len(indicators))
    volatility_pressure = min(28.0, avg_volatility / 0.08 * 28)
    macro_pressure = min(22.0, avg_macro * 22)
    cash_pressure = 10.0 if cash_weight < 0.30 else 0.0
    return volatility_pressure + macro_pressure + cash_pressure


def _annualized_drag(annualized_required_return: float) -> float:
    return 45.0 / (1.0 + exp(-5.0 * (annualized_required_return - 0.18)))


def _goal_difficulty_drag(return_rate: float, period_days: int, period_minutes: int | None = None) -> float:
    if period_minutes is not None and period_minutes > 0:
        trading_day_minutes = 390.0
        trading_days = max(period_minutes / trading_day_minutes, 1.0 / trading_day_minutes)
        required_daily_return = return_rate / trading_days
        return 46.0 / (1.0 + exp(-110.0 * (required_daily_return - 0.018)))
    annualized = (1 + return_rate) ** (365 / period_days) - 1
    return _annualized_drag(annualized)


def _combine_feasibility(market_support_percent: float, risk_pressure_percent: float, goal_drag_percent: float) -> int:
    score = 50.0
    score += (market_support_percent - 40.0) * 0.70
    score -= risk_pressure_percent * 0.55
    score -= goal_drag_percent
    return round(max(3, min(96, score)))


def _estimate_feasibility(
    market_support_percent: float,
    risk_pressure_percent: float,
    return_rate: float,
    period_days: int,
    period_minutes: int | None = None,
) -> int:
    goal_drag = _goal_difficulty_drag(return_rate, period_days, period_minutes)
    return _combine_feasibility(market_support_percent, risk_pressure_percent, goal_drag)


def _summarize_relations(graph: KnowledgeGraph) -> tuple[str, ...]:
    relations: list[str] = []
    for triple in graph.triples():
        if triple.predicate not in {"supportsSignal", "contradictsSignal", "increasesRiskOf"}:
            continue
        relations.append(f"{triple.subject} --{triple.predicate}--> {triple.object}")
        if len(relations) >= 40:
            break
    return tuple(relations)

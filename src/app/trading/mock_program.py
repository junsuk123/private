from __future__ import annotations

from dataclasses import dataclass

from app.execution.broker import BrokerClient
from app.execution.kis_mock import MockKisDevelopersApi, MockKisExecution, MockKisOrderReceipt, MockKisPortfolio
from app.goals import NegotiatedGoal
from app.graph import KnowledgeGraph
from app.risk import RiskManager
from app.schemas.domain import AccountSnapshot, IndicatorSnapshot, MarketSnapshot, OrderIntent, RiskManagerResult
from app.strategy import GoalExecutionPlan, build_goal_execution_plan


@dataclass(frozen=True)
class MockLlmJudgment:
    decision: str
    confidence: float
    selected_tickers: tuple[str, ...]
    rationale: tuple[str, ...]


@dataclass(frozen=True)
class MockTradingRun:
    llm_judgment: MockLlmJudgment
    ontology_evidence: tuple[str, ...]
    goal_plan: GoalExecutionPlan
    order_intents: tuple[OrderIntent, ...]
    risk_results: tuple[RiskManagerResult, ...]
    kis_order_receipts: tuple[MockKisOrderReceipt, ...]
    kis_executions: tuple[MockKisExecution, ...]
    portfolio: MockKisPortfolio


def run_mock_trading_cycle(
    goal: NegotiatedGoal,
    account: AccountSnapshot,
    markets: tuple[MarketSnapshot, ...],
    indicators: dict[str, IndicatorSnapshot],
    graph: KnowledgeGraph,
    broker: BrokerClient | None = None,
) -> MockTradingRun:
    llm_judgment = _mock_llm_judgment(goal, markets, indicators, graph)
    ontology_evidence = _ontology_evidence(graph, llm_judgment.selected_tickers)
    goal_plan = build_goal_execution_plan(goal, account, markets, indicators, graph)
    market_by_ticker = {market.ticker: market for market in markets}
    risk_results = tuple(
        RiskManager().validate(intent, account, market_by_ticker[intent.ticker])
        for intent in goal_plan.intents
    )
    broker_client = broker or MockKisDevelopersApi(
        account=account,
        market_prices={market.ticker: market.last_price for market in markets},
        sectors={market.ticker: market.sector for market in markets},
        company_names={market.ticker: market.company_name for market in markets},
    )
    receipts = []
    executions = []
    for result in risk_results:
        if not result.approved or result.final_order is None:
            continue
        receipt = broker_client.place_limit_order(result.final_order)
        receipts.append(receipt)
        executions.append(broker_client.get_order_status(receipt.order_id))

    return MockTradingRun(
        llm_judgment=llm_judgment,
        ontology_evidence=ontology_evidence,
        goal_plan=goal_plan,
        order_intents=goal_plan.intents,
        risk_results=risk_results,
        kis_order_receipts=tuple(receipts),
        kis_executions=tuple(executions),
        portfolio=broker_client.get_portfolio(),
    )


def _mock_llm_judgment(
    goal: NegotiatedGoal,
    markets: tuple[MarketSnapshot, ...],
    indicators: dict[str, IndicatorSnapshot],
    graph: KnowledgeGraph,
) -> MockLlmJudgment:
    scored = []
    for market in markets:
        indicator = indicators.get(market.ticker)
        ontology_support = len(graph.matching(subject=market.ticker, predicate="supportsSignal"))
        ontology_risk = len(graph.matching(subject=market.ticker, predicate="increasesRiskOf"))
        indicator_score = 0.0
        if indicator is not None:
            indicator_score += (indicator.operating_income_growth or 0.0) * 2.0
            indicator_score += (indicator.operating_margin or 0.0)
            indicator_score += 0.2 if indicator.rsi_14d and 40 <= indicator.rsi_14d <= 70 else -0.1
            indicator_score -= indicator.macro_risk_score * 0.25
        scored.append((indicator_score + ontology_support * 0.25 - ontology_risk * 0.3, market.ticker))
    selected = tuple(ticker for _, ticker in sorted(scored, reverse=True)[:5])
    return MockLlmJudgment(
        decision="PROPOSE_LIMIT_ORDERS",
        confidence=max(0.05, min(0.92, goal.feasibility_percent / 100)),
        selected_tickers=selected,
        rationale=(
            "Goal and period were translated into required return pressure.",
            "Candidate tickers were ranked from ontology support, chart health, and macro risk.",
            "The LLM layer proposes structured intents only; execution is delegated to risk and mock KIS layers.",
        ),
    )


def _ontology_evidence(graph: KnowledgeGraph, tickers: tuple[str, ...]) -> tuple[str, ...]:
    rows = []
    for ticker in tickers:
        for triple in graph.for_subject(ticker):
            if triple.predicate in {"supportsSignal", "contradictsSignal", "increasesRiskOf"}:
                rows.append(f"{triple.subject} --{triple.predicate}--> {triple.object}")
    return tuple(rows[:20])

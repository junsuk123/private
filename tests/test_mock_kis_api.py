from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.data.sample_collectors import collect_sample_market
from app.execution import MockKisDevelopersApi
from app.goals import NegotiatedGoal
from app.graph import OntologyReasoner
from app.graph.builders import build_market_graph
from app.indicators import build_sample_indicators
from app.risk import RiskManager
from app.schemas.domain import AccountSnapshot, OrderSide
from app.strategy import build_goal_execution_plan
from app.trading import run_mock_trading_cycle


class MockKisApiTest(unittest.TestCase):
    def test_mock_kis_limit_order_fills_and_updates_portfolio(self) -> None:
        markets = collect_sample_market()
        market = markets[0]
        account = AccountSnapshot(cash=10_000_000, holdings=())
        indicators = build_sample_indicators(markets)
        graph = build_market_graph(markets, indicators)
        plan = build_goal_execution_plan(
            NegotiatedGoal(0.03, 300_000, 60, 70, "unit"),
            account,
            markets,
            indicators,
            graph,
        )
        intent = next(item for item in plan.intents if item.ticker == market.ticker)
        risk_result = RiskManager().validate(intent, account, market)
        broker = MockKisDevelopersApi(
            account,
            {market.ticker: market.last_price},
            {market.ticker: market.sector},
            {market.ticker: market.company_name},
        )

        receipt = broker.place_limit_order(risk_result.final_order)
        execution = broker.get_order_status(receipt.order_id)
        portfolio = broker.get_portfolio()

        self.assertTrue(receipt.accepted)
        self.assertEqual(execution.status, "FILLED")
        self.assertEqual(execution.side, OrderSide.BUY)
        self.assertLess(portfolio.account.cash, account.cash)
        self.assertEqual(portfolio.account.holdings[0].ticker, market.ticker)

    def test_mock_trading_cycle_exposes_replaceable_broker_flow(self) -> None:
        markets = collect_sample_market()
        account = AccountSnapshot(cash=10_000_000, holdings=())
        indicators = build_sample_indicators(markets)
        graph = build_market_graph(markets, indicators)
        OntologyReasoner(graph).infer()

        run = run_mock_trading_cycle(
            NegotiatedGoal(0.03, 300_000, 60, 70, "unit"),
            account,
            markets,
            indicators,
            graph,
        )

        self.assertEqual(run.llm_judgment.decision, "PROPOSE_LIMIT_ORDERS")
        self.assertGreater(len(run.ontology_evidence), 0)
        self.assertGreater(len(run.order_intents), 0)
        self.assertGreater(len(run.kis_order_receipts), 0)
        self.assertGreater(len(run.kis_executions), 0)
        self.assertGreater(len(run.portfolio.account.holdings), 0)


if __name__ == "__main__":
    unittest.main()

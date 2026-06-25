from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.data.sample_collectors import collect_sample_market
from app.goals import NegotiatedGoal
from app.graph.builders import build_market_graph
from app.indicators import build_sample_indicators
from app.risk import RiskManager
from app.schemas.domain import AccountSnapshot, Holding, IndicatorSnapshot, OrderAction, OrderSide
from app.strategy import build_goal_execution_plan


class GoalDirectedStrategyTest(unittest.TestCase):
    def test_goal_plan_generates_chart_and_ontology_based_buy_intents(self) -> None:
        account = AccountSnapshot(cash=10_000_000, holdings=())
        markets = collect_sample_market()
        indicators = build_sample_indicators(markets)
        graph = build_market_graph(markets, indicators)
        goal = NegotiatedGoal(
            target_return_rate=0.04,
            target_profit_amount=400_000,
            period_days=90,
            feasibility_percent=72,
            label="unit",
        )

        plan = build_goal_execution_plan(goal, account, markets, indicators, graph)

        self.assertGreater(len(plan.signals), 0)
        self.assertTrue(any(signal.action == OrderAction.BUY for signal in plan.signals))
        self.assertTrue(any(intent.action == OrderAction.BUY for intent in plan.intents))
        self.assertTrue(
            any("RSI" in factor or "Ontology" in factor for signal in plan.signals for factor in signal.supporting_factors)
        )

    def test_risk_manager_creates_sell_order_for_goal_reduce_signal(self) -> None:
        market = collect_sample_market()[0]
        account = AccountSnapshot(
            cash=1_000_000,
            holdings=(
                Holding(
                    ticker=market.ticker,
                    market=market.market,
                    company_name=market.company_name,
                    sector=market.sector,
                    quantity=10,
                    average_price=market.last_price * 1.1,
                    last_price=market.last_price,
                ),
            ),
        )
        markets = (market,)
        indicators = {
            market.ticker: IndicatorSnapshot(
                ticker=market.ticker,
                revenue_growth=-0.08,
                operating_income_growth=-0.12,
                operating_margin=0.06,
                roe=0.02,
                debt_ratio=0.72,
                per=31.0,
                pbr=2.8,
                rsi_14d=82.0,
                volume_ratio=0.55,
                macro_risk_score=0.72,
                source_ids=("unit-bad-chart",),
            )
        }
        graph = build_market_graph(markets, indicators)
        graph.add(market.ticker, "increasesRiskOf", "NegativeEventRisk", "unit-risk")
        goal = NegotiatedGoal(
            target_return_rate=0.25,
            target_profit_amount=250_000,
            period_days=20,
            feasibility_percent=25,
            label="aggressive",
        )

        plan = build_goal_execution_plan(goal, account, markets, indicators, graph)
        sell_intents = tuple(
            intent for intent in plan.intents if intent.action in {OrderAction.SELL, OrderAction.REDUCE}
        )
        result = RiskManager().validate(sell_intents[0], account, market)

        self.assertGreater(len(sell_intents), 0)
        self.assertTrue(result.approved)
        self.assertIsNotNone(result.final_order)
        self.assertEqual(result.final_order.side, OrderSide.SELL)


if __name__ == "__main__":
    unittest.main()

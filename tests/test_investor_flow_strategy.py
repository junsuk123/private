from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.graph import OntologyReasoner
from app.graph.builders import build_market_graph
from app.schemas.domain import (
    IndicatorSnapshot,
    InvestorFlowSnapshot,
    InvestorGroup,
    MarketSnapshot,
    OrderAction,
    SourceMetadata,
)
from app.strategy.investor_flow import assess_domestic_investor_flow, compute_investor_flow_metrics
from app.strategy.rule_based import generate_order_intents, generate_strategy_signals


class InvestorFlowStrategyTest(unittest.TestCase):
    def test_foreign_and_institution_buying_supports_domestic_buy_signal(self) -> None:
        market = _market(
            foreign_net_buy=4_000_000_000,
            institution_net_buy=3_000_000_000,
            retail_net_buy=-5_000_000_000,
            price_change_rate=0.025,
        )
        indicators = {market.ticker: _indicator(market.ticker)}
        graph = build_market_graph((market,), indicators)

        assessment = assess_domestic_investor_flow(market)
        signals = generate_strategy_signals((market,), indicators, graph)
        metrics = compute_investor_flow_metrics(market)

        self.assertEqual(assessment.action_bias, OrderAction.BUY)
        self.assertIn("ForeignInstitutionJointBuying", assessment.supporting_factors)
        self.assertGreater(metrics.informed_imbalance, 0)
        self.assertTrue(graph.matching(subject=market.ticker, predicate="hasFlowMetric"))
        self.assertTrue(graph.matching(subject=market.ticker, predicate="usesFlowModel"))
        self.assertEqual(signals[0].action, OrderAction.BUY)
        self.assertIn("InformedOrderFlowImbalance", signals[0].supporting_factors)

    def test_foreign_and_institution_selling_reduces_domestic_buy_conviction(self) -> None:
        market = _market(
            foreign_net_buy=-5_000_000_000,
            institution_net_buy=-4_000_000_000,
            retail_net_buy=6_000_000_000,
            price_change_rate=-0.018,
        )
        indicators = {market.ticker: _indicator(market.ticker)}
        graph = build_market_graph((market,), indicators)

        assessment = assess_domestic_investor_flow(market)
        signals = generate_strategy_signals((market,), indicators, graph)

        self.assertIn(assessment.action_bias, {OrderAction.SELL, OrderAction.REDUCE, OrderAction.HOLD})
        self.assertIn("ForeignInstitutionJointSelling", assessment.contradicting_factors)
        self.assertNotEqual(signals[0].action, OrderAction.BUY)
        self.assertIn("RetailDemandMeetsInformedSelling", signals[0].contradicting_factors)

    def test_unusual_turnover_with_hidden_flow_marks_suspected_smart_money(self) -> None:
        market = _market(
            foreign_net_buy=200_000_000,
            institution_net_buy=-100_000_000,
            retail_net_buy=-50_000_000,
            program_net_buy=1_500_000_000,
            volume_change_rate=1.8,
            price_change_rate=0.031,
        )

        assessment = assess_domestic_investor_flow(market)

        self.assertEqual(assessment.dominant_group, InvestorGroup.SUSPECTED_SMART_MONEY)
        self.assertIn("SuspectedSmartMoneyAccumulation", assessment.supporting_factors)

    def test_reasoner_uses_ontology_order_flow_evidence(self) -> None:
        market = _market(
            foreign_net_buy=4_500_000_000,
            institution_net_buy=4_000_000_000,
            retail_net_buy=-6_000_000_000,
            volume_change_rate=1.3,
            price_change_rate=0.032,
        )
        indicators = {market.ticker: _indicator(market.ticker)}
        graph = build_market_graph((market,), indicators)
        reasoner = OntologyReasoner(graph)

        reasoner.infer()
        paths = reasoner.build_reasoning_paths((market.ticker,))

        self.assertTrue(graph.matching(subject=market.ticker, predicate="supportsSignal", object_="OrderFlowConfirmedBuyCandidate"))
        self.assertEqual(paths[0].conclusion, "BuyCandidate")
        self.assertTrue(any("OrderFlow" in item for item in paths[0].supporting_triples))

    def test_market_context_can_emit_signal_without_fundamental_indicators(self) -> None:
        market = _market(
            foreign_net_buy=4_000_000_000,
            institution_net_buy=3_000_000_000,
            retail_net_buy=-5_000_000_000,
            volume_change_rate=1.1,
            price_change_rate=0.024,
        )
        graph = build_market_graph((market,), {})

        signals = generate_strategy_signals((market,), {}, graph)
        intents = generate_order_intents((market,), {}, signals)

        self.assertEqual(signals[0].action, OrderAction.BUY)
        self.assertIn("MissingFundamentalIndicators", signals[0].contradicting_factors)
        self.assertIn("HighLiquidity", signals[0].supporting_factors)
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].source_data_ids, ("unit-market",))


def _market(
    *,
    foreign_net_buy: float,
    institution_net_buy: float,
    retail_net_buy: float,
    program_net_buy: float = 0.0,
    volume_change_rate: float = 0.5,
    price_change_rate: float,
) -> MarketSnapshot:
    now = datetime.now(timezone.utc)
    source = SourceMetadata("unit", now, source_id="unit-market")
    return MarketSnapshot(
        ticker="005930",
        market="KOSPI",
        company_name="Samsung Electronics",
        sector="Semiconductor",
        last_price=75_000,
        average_daily_trading_value=100_000_000_000,
        volatility_20d=0.028,
        source=source,
        investor_flow=InvestorFlowSnapshot(
            ticker="005930",
            market="KOSPI",
            foreign_net_buy=foreign_net_buy,
            institution_net_buy=institution_net_buy,
            retail_net_buy=retail_net_buy,
            program_net_buy=program_net_buy,
            volume_change_rate=volume_change_rate,
            price_change_rate=price_change_rate,
            trading_value=100_000_000_000,
            observed_at=now,
            source=SourceMetadata("unit-flow", now, source_id="unit-flow"),
        ),
    )


def _indicator(ticker: str) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        ticker=ticker,
        revenue_growth=0.10,
        operating_income_growth=0.17,
        operating_margin=0.16,
        roe=0.12,
        debt_ratio=0.35,
        per=16.0,
        pbr=1.4,
        rsi_14d=58,
        volume_ratio=1.2,
        macro_risk_score=0.20,
        source_ids=("unit-indicator",),
    )


if __name__ == "__main__":
    unittest.main()

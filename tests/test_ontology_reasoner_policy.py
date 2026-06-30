from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.graph import KnowledgeGraph, OntologyReasoner, OntologyReasoningPolicy
from app.graph.builders import build_market_graph
from app.schemas.domain import AccountSnapshot, IndicatorSnapshot, MarketSnapshot, SourceMetadata
from datetime import datetime, timezone


class OntologyReasonerPolicyTest(unittest.TestCase):
    def test_reasoner_policy_override_changes_buy_threshold(self) -> None:
        graph = KnowledgeGraph()
        graph.add("TEST", "supportsSignal", "EarningsGrowth", "unit")
        graph.add("TEST", "supportsSignal", "ProfitabilityQuality", "unit")

        default_path = OntologyReasoner(graph).build_reasoning_paths(("TEST",))[0]
        strict_path = OntologyReasoner(
            graph,
            policy=OntologyReasoningPolicy(buy_threshold=0.95),
        ).build_reasoning_paths(("TEST",))[0]

        self.assertEqual(default_path.conclusion, "BuyCandidate")
        self.assertEqual(strict_path.conclusion, "HoldOrWatch")

    def test_risk_adjusted_sizing_is_not_bullish_support(self) -> None:
        graph = KnowledgeGraph()
        graph.add("TEST", "increasesRiskOf", "VolatilityRisk", "unit")

        reasoner = OntologyReasoner(graph)
        reasoner.infer()
        path = reasoner.build_reasoning_paths(("TEST",))[0]

        self.assertFalse(graph.matching(subject="TEST", predicate="supportsSignal", object_="RiskAdjustedSizing"))
        self.assertTrue(graph.matching(subject="TEST", predicate="requiresSizingAdjustment", object_="RiskAdjustedSizing"))
        self.assertEqual(path.conclusion, "HoldOrWatch")
        self.assertIn("sizing_adjustments=1", path.explanation)

    def test_account_cash_affordability_is_graph_evidence(self) -> None:
        now = datetime.now(timezone.utc)
        source = SourceMetadata(source_name="KIS broker quote", retrieved_at=now, source_id="quote:unit")
        markets = (
            MarketSnapshot("CHEAP", "NASDAQ", "Cheap", "Technology", 2.0, 10_000_000, 0.02, source),
            MarketSnapshot("EXP", "NASDAQ", "Expensive", "Technology", 200.0, 10_000_000, 0.02, source),
        )
        indicators = {
            market.ticker: IndicatorSnapshot(market.ticker, 0.2, 0.2, 0.2, None, None, 10, None, 60, 1.2, 0.0)
            for market in markets
        }
        account = AccountSnapshot(cash=0.0, holdings=(), cash_by_currency={"USD": 5.0}, base_currency="KRW")

        graph = build_market_graph(markets, indicators, account=account)
        OntologyReasoner(graph).infer()
        paths = {path.ticker: path for path in OntologyReasoner(graph).build_reasoning_paths(("CHEAP", "EXP"))}

        self.assertTrue(graph.matching(subject="CHEAP", predicate="supportsSignal", object_="CashFitOneShare"))
        self.assertTrue(graph.matching(subject="EXP", predicate="contradictsSignal", object_="CashBelowOneSharePrice"))
        self.assertGreater(paths["CHEAP"].confidence, paths["EXP"].confidence)


if __name__ == "__main__":
    unittest.main()

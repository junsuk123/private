from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.graph import KnowledgeGraph, OntologyReasoner, OntologyReasoningPolicy


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


if __name__ == "__main__":
    unittest.main()

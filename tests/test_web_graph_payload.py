from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.graph import KnowledgeGraph
from app.schemas.domain import ReasoningPath


class WebGraphPayloadTest(unittest.TestCase):
    def tearDown(self) -> None:
        try:
            from app.graph import macro_micro_feed

            macro_micro_feed.clear()
        except Exception:
            pass

    def test_semantic_relation_nodes_keep_visible_kinds(self) -> None:
        try:
            from app.web import _graph_payload
        except TypeError as exc:
            self.skipTest(f"web app import is unavailable in this dependency set: {exc}")

        graph = KnowledgeGraph()
        graph.add("TEST", "supportsSignal", "BuyCandidate", "test:support")
        graph.add("TEST", "increasesRiskOf", "OrderFlowDistributionRisk", "test:risk")
        graph.add("TEST", "contradictsSignal", "AggressiveBuy", "test:contradiction")
        graph.add("semantic:risk-feature", "increasesRiskOf", "ReduceRiskCandidate", "test:semantic-risk")

        context = SimpleNamespace(
            graph=graph,
            events=(),
            markets=(),
            reasoning_paths=(
                ReasoningPath(
                    path_id="test-path",
                    ticker="TEST",
                    conclusion="HoldOrWatch",
                    confidence=0.42,
                    supporting_triples=("TEST --supportsSignal--> BuyCandidate",),
                    contradicting_triples=("TEST --contradictsSignal--> AggressiveBuy",),
                    risk_triples=("TEST --increasesRiskOf--> OrderFlowDistributionRisk",),
                    explanation="test",
                ),
            ),
            ontology_runtime=SimpleNamespace(as_dict=lambda: {}),
            candidate_selection=None,
            parameter_tuning=(),
            temporal_frames=(),
        )

        payload = _graph_payload(context)
        kinds = {node["id"]: node["kind"] for node in payload["nodes"]}

        self.assertEqual(kinds["BuyCandidate"], "support")
        self.assertEqual(kinds["OrderFlowDistributionRisk"], "risk")
        self.assertEqual(kinds["AggressiveBuy"], "contradiction")
        self.assertEqual(kinds["semantic:risk-feature"], "risk")

    def test_macro_micro_overlay_uses_visible_dashboard_kinds(self) -> None:
        try:
            from app.graph import macro_micro_feed
            from app.web import _graph_payload
        except TypeError as exc:
            self.skipTest(f"web app import is unavailable in this dependency set: {exc}")

        macro_micro_feed.record_bundle(
            {
                "macro_result": {
                    "market_regime": "TREND_UP",
                    "candidate_symbols": ["005930"],
                    "blocked_micro_strategies": ["mean_reversion"],
                },
                "micro_results": [
                    {"symbol": "005930", "micro_regime": "BREAKOUT_CANDIDATE"},
                ],
            }
        )

        context = SimpleNamespace(
            graph=KnowledgeGraph(),
            events=(),
            markets=(),
            reasoning_paths=(),
            ontology_runtime=SimpleNamespace(as_dict=lambda: {}),
            candidate_selection=None,
            parameter_tuning=(),
            temporal_frames=(),
        )

        payload = _graph_payload(context)
        kinds = {node["id"]: node["kind"] for node in payload["nodes"]}
        visible_kinds = {
            "ticker",
            "event",
            "temporal",
            "sector",
            "support",
            "risk",
            "contradiction",
            "pipeline",
            "tuning",
            "parameter",
            "metric",
            "entity",
        }

        self.assertEqual(kinds["MacroMarket"], "pipeline")
        self.assertEqual(kinds["005930"], "ticker")
        self.assertEqual(kinds["MarketRegime:TREND_UP"], "support")
        self.assertEqual(kinds["MicroRegime:BREAKOUT_CANDIDATE"], "support")
        self.assertTrue(set(kinds.values()).issubset(visible_kinds))


if __name__ == "__main__":
    unittest.main()

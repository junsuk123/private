from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.graph import KnowledgeGraph
from app.schemas.domain import ReasoningPath


class WebGraphPayloadTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

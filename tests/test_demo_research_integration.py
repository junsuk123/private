from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.pipeline import build_analysis_context
from app.research import ResearchService


class DemoResearchIntegrationTest(unittest.TestCase):
    def test_demo_config_feeds_research_events_into_reasoning(self) -> None:
        research = ResearchService().run_from_config(Path("config/research_sources.demo.json"))
        context = build_analysis_context(research)

        self.assertGreaterEqual(len(research.events), 3)
        self.assertGreaterEqual(len(research.raw_records), 1)
        self.assertGreaterEqual(len(context.events), len(research.events))
        self.assertTrue(
            any(
                triple.predicate == "hasRecentNews" and triple.subject == "005930"
                for triple in context.graph.triples()
            )
        )
        self.assertTrue(any(path.ticker == "005930" for path in context.reasoning_paths))


if __name__ == "__main__":
    unittest.main()

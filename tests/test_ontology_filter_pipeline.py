from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.trading_pipeline import (
    build_lightweight_market_snapshots,
    ontology_filter_1,
    universe_from_tickers,
)


class OntologyFilterPipelineTest(unittest.TestCase):
    def test_filter_1_reduces_universe_before_chart_scope(self) -> None:
        tickers = tuple(f"SIM{i:04d}" for i in range(500))
        snapshots = build_lightweight_market_snapshots(universe_from_tickers(tickers), seed=11)

        result = ontology_filter_1(snapshots, target_count=80)

        self.assertEqual(result.full_universe_count, 500)
        self.assertGreaterEqual(len(result.candidate_stocks), 20)
        self.assertLessEqual(len(result.candidate_stocks), 80)
        self.assertEqual(result.chart_fetch_scope, result.candidate_stocks)
        self.assertEqual(result.api_call_count, 0)
        self.assertTrue(all(trace.stage == "ontology_filter_1" for trace in result.traces))


if __name__ == "__main__":
    unittest.main()

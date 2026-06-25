from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.cli import run_demo


class PipelineTest(unittest.TestCase):
    def test_demo_pipeline_produces_auditable_outputs(self) -> None:
        result = run_demo()

        self.assertGreater(result["portfolio_report"].equity, 0)
        self.assertEqual(result["portfolio_report"].equity, 1_000_000)
        self.assertGreater(len(result["graph_triples"]), 0)
        self.assertGreater(len(result["strategy_signals"]), 0)
        self.assertGreater(len(result["order_intents"]), 0)
        self.assertGreater(len(result["risk_results"]), 0)
        self.assertEqual(result["audit_log"], "logs/audit.jsonl")


if __name__ == "__main__":
    unittest.main()

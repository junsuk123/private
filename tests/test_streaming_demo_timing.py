from __future__ import annotations

import sys
import unittest
import os
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.backtesting import StreamingAcceleratedDemo, TimeMode, TimeScalerConfig
from app.web import _streaming_demos, app


TEST_TICKERS = ("AAPL", "MSFT", "NVDA", "005930.KS", "000660.KS")


class StreamingDemoTimingTest(unittest.TestCase):
    def test_twenty_minute_demo_has_twenty_visible_steps(self) -> None:
        demo = StreamingAcceleratedDemo(
            config=TimeScalerConfig(mode=TimeMode.REALTIME, acceleration_factor=1.0),
            period_minutes=20,
            initial_cash=10_000_000,
            seed=42,
            tickers=TEST_TICKERS,
        )

        results = demo.run_all_steps()

        self.assertEqual(len(results), 20)
        self.assertEqual(results[0].step_index, 15)
        self.assertEqual(results[-1].step_index, 34)
        self.assertEqual(demo.get_progress(), 100.0)
        self.assertTrue(demo.is_complete())
        final = demo.get_final_results()
        self.assertIsNotNone(final)
        self.assertEqual(final["final_positions"], {})
        self.assertEqual(results[-1].holdings, {})

    def test_final_step_liquidates_existing_holdings(self) -> None:
        demo = StreamingAcceleratedDemo(
            config=TimeScalerConfig(mode=TimeMode.REALTIME, acceleration_factor=1.0),
            period_minutes=1,
            initial_cash=10_000_000,
            seed=42,
            tickers=("AAPL",),
        )
        demo.initialize()
        demo._holdings["AAPL"] = 3

        result = demo.run_step()

        self.assertIsNotNone(result)
        self.assertEqual(result.holdings, {})
        self.assertTrue(
            any(
                trade.side == "SELL" and trade.reason == "mandatory final liquidation"
                for trade in result.trades_in_step
            )
        )

    def test_step_api_waits_until_next_simulated_minute_is_due(self) -> None:
        previous_limit = os.environ.get("SIM_STREAMING_UNIVERSE_LIMIT")
        previous_target = os.environ.get("ONTOLOGY_FILTER1_TARGET_COUNT")
        os.environ["SIM_STREAMING_UNIVERSE_LIMIT"] = str(len(TEST_TICKERS))
        os.environ["ONTOLOGY_FILTER1_TARGET_COUNT"] = str(len(TEST_TICKERS))
        client = TestClient(app)
        try:
            start_response = client.post(
                "/api/streaming-demo/start",
                json={
                    "target_return_rate": 0.02,
                    "period_minutes": 20,
                    "initial_cash": 10_000_000,
                    "acceleration_factor": 1,
                },
            )
            self.assertEqual(start_response.status_code, 200)
            demo_id = start_response.json()["demo_id"]

            early_step = client.post("/api/streaming-demo/step", json={"demo_id": demo_id}).json()

            self.assertEqual(early_step["status"], "waiting")
            self.assertEqual(early_step["progress"], 0.0)
            self.assertGreater(early_step["retry_after_seconds"], 50)

            _streaming_demos[demo_id]._started_at_monotonic -= 60
            due_step = client.post("/api/streaming-demo/step", json={"demo_id": demo_id}).json()

            self.assertEqual(due_step["status"], "running")
            self.assertEqual(due_step["step"], 1)
            self.assertEqual(due_step["raw_step"], 15)
            self.assertEqual(due_step["progress"], 5.0)
            self.assertIn("ontology_filter_1", due_step)
            self.assertLessEqual(due_step["ontology_filter_1"]["chart_fetch_count"], len(TEST_TICKERS))
            self.assertEqual(due_step["ontology_filter_1"]["chart_fetch_count"], due_step["universe_count"])
        finally:
            if previous_limit is None:
                os.environ.pop("SIM_STREAMING_UNIVERSE_LIMIT", None)
            else:
                os.environ["SIM_STREAMING_UNIVERSE_LIMIT"] = previous_limit
            if previous_target is None:
                os.environ.pop("ONTOLOGY_FILTER1_TARGET_COUNT", None)
            else:
                os.environ["ONTOLOGY_FILTER1_TARGET_COUNT"] = previous_target


if __name__ == "__main__":
    unittest.main()

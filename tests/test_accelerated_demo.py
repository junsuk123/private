from __future__ import annotations

import sys
import csv
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.backtesting import run_accelerated_demo


class AcceleratedDemoTest(unittest.TestCase):
    def test_accelerated_demo_generates_50_symbol_minute_based_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_accelerated_demo(
                target_return_rate=0.01,
                period_minutes=390,
                initial_cash=10_000_000,
                output_dir=Path(tmp),
                seed=7,
            )

            self.assertEqual(result.ticker_count, 50)
            self.assertEqual(result.simulated_minutes, 390)
            self.assertEqual(result.bars_per_ticker, 390)
            self.assertGreater(result.trade_count, 0)
            self.assertEqual(result.final_positions, {})
            with Path(result.trades_path).open(encoding="utf-8") as file:
                trades = list(csv.DictReader(file))
            self.assertTrue(
                any(row["side"] == "SELL" and row["reason"] == "mandatory final liquidation" for row in trades)
            )
            self.assertTrue(Path(result.report_path).exists())
            self.assertTrue(Path(result.trades_path).exists())
            self.assertTrue(Path(result.charts_path).exists())


if __name__ == "__main__":
    unittest.main()

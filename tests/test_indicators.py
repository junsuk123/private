from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.features import IndicatorEngine, OHLCVBar
from app.features.indicator_engine import bollinger_bands, obv, period_return, rsi, sma, stochastic_k


class IndicatorEngineTest(unittest.TestCase):
    def test_core_indicator_formulas(self) -> None:
        closes = [float(value) for value in range(1, 31)]

        self.assertAlmostEqual(period_return(closes, 5), 30 / 25 - 1)
        self.assertEqual(sma(closes, 5), 28.0)
        self.assertEqual(rsi(closes, 14), 100.0)
        self.assertEqual(stochastic_k([v + 1 for v in closes], [v - 1 for v in closes], closes, 14), 100 * (30 - 16) / (31 - 16))
        self.assertEqual(obv([10, 11, 10, 10, 12], [100, 200, 300, 400, 500]), 400)

        middle, upper, lower, width, percent_b = bollinger_bands(closes, 20, 2)
        self.assertEqual(middle, 20.5)
        self.assertIsNotNone(upper)
        self.assertIsNotNone(lower)
        self.assertIsNotNone(width)
        self.assertIsNotNone(percent_b)

    def test_engine_generates_records_with_as_of_metadata(self) -> None:
        bars = _bars(35)
        records = IndicatorEngine().calculate(bars)
        by_name = {record.indicator_name: record for record in records}

        self.assertIn("return_5d", by_name)
        self.assertIn("macd_histogram", by_name)
        self.assertIn("rsi_14", by_name)
        self.assertIn("bollinger_percent_b_20", by_name)
        self.assertEqual(by_name["return_5d"].ticker, "TEST")
        self.assertEqual(by_name["return_5d"].as_of, bars[-1].as_of)
        self.assertEqual(by_name["return_5d"].calculation_version, "semantic-indicators-v1")


def _bars(count: int) -> tuple[OHLCVBar, ...]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return tuple(
        OHLCVBar(
            ticker="TEST",
            as_of=start + timedelta(days=i),
            open=100 + i,
            high=102 + i,
            low=99 + i,
            close=101 + i,
            volume=1_000 + i * 10,
        )
        for i in range(count)
    )


if __name__ == "__main__":
    unittest.main()

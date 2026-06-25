from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.features import IndicatorEngine, OHLCVBar
from app.models import DatasetBuilder, LabelConfig, future_return, triple_barrier_label


class NoLookaheadTest(unittest.TestCase):
    def test_future_bar_does_not_change_as_of_features(self) -> None:
        bars = _bars(40)
        decision_time = bars[30].as_of
        future_shock = OHLCVBar(
            ticker="TEST",
            as_of=bars[-1].as_of + timedelta(days=1),
            open=1_000,
            high=1_200,
            low=900,
            close=1_150,
            volume=99_000_000,
        )

        engine = IndicatorEngine()
        baseline = engine.calculate(bars, as_of=decision_time)
        with_future = engine.calculate(bars + (future_shock,), as_of=decision_time)

        baseline_values = {record.indicator_name: record.value for record in baseline}
        future_values = {record.indicator_name: record.value for record in with_future}
        self.assertEqual(baseline_values, future_values)

    def test_labels_are_generated_separately_from_features(self) -> None:
        bars = _bars(45)
        decision_time = bars[20].as_of
        label = future_return(bars, decision_time, 5)
        triple = triple_barrier_label(bars, decision_time, LabelConfig(horizon_bars=5, profit_taking=0.01, stop_loss=-0.05))
        rows = DatasetBuilder(label_config=LabelConfig(horizon_bars=5)).build_rows(bars, (decision_time,))

        self.assertIsNotNone(label)
        self.assertIsNotNone(triple)
        self.assertEqual(len(rows), 1)
        self.assertIn("future_return_5d", rows[0].labels)
        self.assertNotIn("future_return_5d", rows[0].features)
        self.assertTrue(rows[0].metadata["no_lookahead"])


def _bars(count: int) -> tuple[OHLCVBar, ...]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = []
    price = 100.0
    for i in range(count):
        price *= 1.004
        bars.append(
            OHLCVBar(
                ticker="TEST",
                as_of=start + timedelta(days=i),
                open=price * 0.99,
                high=price * 1.02,
                low=price * 0.98,
                close=price,
                volume=1_000_000 + i * 1_000,
            )
        )
    return tuple(bars)


if __name__ == "__main__":
    unittest.main()

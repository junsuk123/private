from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.features import HybridSemanticFeaturePipeline, OHLCVBar, RegimeFormulaParameterTuner
from app.models import DatasetBuilder


class ParameterTuningTest(unittest.TestCase):
    def test_parameter_tuner_recommends_and_pipeline_uses_formula_parameters(self) -> None:
        training_bars = _bars(80, alternating=True)
        builder = DatasetBuilder()
        examples = builder.build_parameter_tuning_examples(
            training_bars,
            tuple(bar.as_of for bar in training_bars[30:70:5]),
        )
        tuner = RegimeFormulaParameterTuner()
        tuner.fit(examples)

        live_bars = _bars(80, alternating=True)
        pipeline = HybridSemanticFeaturePipeline(parameter_tuner=tuner)
        snapshot = pipeline.build_snapshot(live_bars)

        recommendations = {item.parameter_name: item.value for item in snapshot.parameter_recommendations}
        by_name = {record.indicator_name: record for record in snapshot.raw_indicators}

        self.assertIn("rsi_period", recommendations)
        self.assertIn("bollinger_stddev", recommendations)
        self.assertEqual(by_name["rsi_14"].metadata["parameters"]["period"], recommendations["rsi_period"])
        self.assertEqual(
            by_name["bollinger_band_width_20"].metadata["parameters"]["stddevs"],
            recommendations["bollinger_stddev"],
        )
        self.assertEqual(by_name["rsi_14"].source, "ohlcv+ai-parameters")
        self.assertTrue(all(item.model_version == tuner.model_version for item in snapshot.parameter_recommendations))

    def test_parameter_tuning_examples_do_not_use_future_values_as_context(self) -> None:
        bars = _bars(60, alternating=False)
        examples = DatasetBuilder().build_parameter_tuning_examples(
            bars,
            (bars[30].as_of,),
        )
        shocked_future = bars + (
            OHLCVBar("TEST", bars[-1].as_of + timedelta(days=1), 500, 600, 400, 550, 99_000_000),
        )
        shocked_examples = DatasetBuilder().build_parameter_tuning_examples(
            shocked_future,
            (bars[30].as_of,),
        )

        self.assertEqual(examples[0].context_features, shocked_examples[0].context_features)


def _bars(count: int, alternating: bool) -> tuple[OHLCVBar, ...]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    price = 100.0
    rows = []
    for index in range(count):
        if alternating:
            shock = 0.025 if index % 2 == 0 else -0.018
        else:
            shock = 0.003
        price *= 1 + shock
        rows.append(
            OHLCVBar(
                ticker="TEST",
                as_of=start + timedelta(days=index),
                open=price * 0.99,
                high=price * 1.04,
                low=price * 0.96,
                close=price,
                volume=1_000_000 + index * 5_000,
            )
        )
    return tuple(rows)


if __name__ == "__main__":
    unittest.main()

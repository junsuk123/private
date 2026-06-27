from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from app.features.schemas import OHLCVBar
from app.features.short_horizon_features import ShortHorizonFeatures
from app.strategy.short_horizon import TechnicalRuleEngine


class TechnicalRuleEngineTest(unittest.TestCase):
    def test_ma_crossover_candidate_created(self) -> None:
        bars = _ma_crossover_bars()
        features = _features(timestamp=bars[-1].as_of)

        candidate = TechnicalRuleEngine().generate_candidate(features, bars, entry_price=bars[-1].close)

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.strategy_family, "technical_rule")
        self.assertIn("MovingAverageBreakout", candidate.ontology_tags)
        self.assertIn("VolumeConfirmedBreakout", candidate.ontology_tags)
        self.assertIn("TechnicalBreakoutBuy", candidate.ontology_tags)
        self.assertEqual(candidate.features["ma_crossover"], 1.0)

    def test_range_breakout_candidate_created(self) -> None:
        bars = _range_breakout_bars()
        features = _features(timestamp=bars[-1].as_of)

        candidate = TechnicalRuleEngine().generate_candidate(features, bars, entry_price=bars[-1].close)

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertIn("TradingRangeBreakout", candidate.ontology_tags)
        self.assertIn("BreakoutWatch", candidate.ontology_tags)
        self.assertEqual(candidate.features["range_breakout"], 1.0)
        self.assertGreater(candidate.features["breakout_width"], 0)

    def test_rejects_false_breakout_without_volume(self) -> None:
        bars = _range_breakout_bars(volume_confirmed=False)
        features = _features(timestamp=bars[-1].as_of)

        candidate = TechnicalRuleEngine().generate_candidate(features, bars, entry_price=bars[-1].close)

        self.assertIsNone(candidate)

    def test_candidate_has_expected_exit_price(self) -> None:
        bars = _range_breakout_bars()
        features = _features(timestamp=bars[-1].as_of)

        candidate = TechnicalRuleEngine().generate_candidate(features, bars, entry_price=10_000)

        self.assertIsNotNone(candidate)
        assert candidate is not None
        expected_return = min(0.006, candidate.features["breakout_width"] * 0.4)
        self.assertAlmostEqual(candidate.expected_exit_price, 10_000 * (1 + expected_return))
        self.assertAlmostEqual(candidate.gross_expected_return, expected_return)

    def test_rejects_wide_spread_or_low_liquidity(self) -> None:
        bars = _range_breakout_bars()
        wide_spread = _features(timestamp=bars[-1].as_of, spread_rate=0.002)
        low_liquidity = _features(timestamp=bars[-1].as_of, liquidity_score=0.2)

        self.assertIsNone(TechnicalRuleEngine().generate_candidate(wide_spread, bars, entry_price=10_000))
        self.assertIsNone(TechnicalRuleEngine().generate_candidate(low_liquidity, bars, entry_price=10_000))

    def test_future_bar_does_not_create_as_of_breakout(self) -> None:
        bars = _range_breakout_bars()
        as_of = bars[-2].as_of
        features = _features(timestamp=as_of)

        candidate = TechnicalRuleEngine().generate_candidate(features, bars, entry_price=bars[-2].close)

        self.assertIsNone(candidate)

    def test_paper_only_by_default(self) -> None:
        bars = _range_breakout_bars()
        features = _features(timestamp=bars[-1].as_of)

        candidate = TechnicalRuleEngine().generate_candidate(
            features,
            bars,
            entry_price=bars[-1].close,
            trading_mode="live",
        )

        self.assertIsNone(candidate)


def _features(
    *,
    timestamp: datetime,
    spread_rate: float = 0.0007,
    liquidity_score: float = 0.8,
) -> ShortHorizonFeatures:
    return ShortHorizonFeatures(
        ticker="005930",
        timestamp=timestamp,
        returns_by_window={
            "ret_1m": 0.002,
            "ret_3m": 0.004,
            "ret_5m": 0.006,
            "ret_15m": 0.009,
            "ret_30m": 0.012,
            "ret_1d": 0.018,
            "ret_open_10m": 0.006,
            "ret_open_30m": 0.011,
            "ret_preclose_30m": None,
        },
        realized_volatility={
            "realized_volatility_5m": 0.002,
            "realized_volatility_30m": 0.004,
        },
        volume_zscore=2.0,
        spread_rate=spread_rate,
        orderbook_depth_score=0.8,
        liquidity_score=liquidity_score,
        market_alignment_score=0.8,
        time_of_day_weight=1.0,
        is_valid=True,
        missing_fields=(),
    )


def _range_breakout_bars(*, volume_confirmed: bool = True) -> tuple[OHLCVBar, ...]:
    start = datetime(2026, 1, 2, 9, 0, tzinfo=timezone.utc)
    bars = [
        OHLCVBar("005930", start + timedelta(minutes=i), 100, 101, 99, 100 + (i % 3) * 0.1, 1_000)
        for i in range(20)
    ]
    bars.append(
        OHLCVBar(
            "005930",
            start + timedelta(minutes=20),
            101,
            103,
            100,
            102.0,
            2_000 if volume_confirmed else 1_100,
        )
    )
    return tuple(bars)


def _ma_crossover_bars() -> tuple[OHLCVBar, ...]:
    start = datetime(2026, 1, 2, 9, 0, tzinfo=timezone.utc)
    closes = [100.0] * 16 + [96.0, 96.0, 96.0, 96.0, 118.0]
    return tuple(
        OHLCVBar(
            "005930",
            start + timedelta(minutes=i),
            close,
            close + 1,
            close - 1,
            close,
            2_000 if i == len(closes) - 1 else 1_000,
        )
        for i, close in enumerate(closes)
    )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from datetime import datetime, time, timezone

from app.features.short_horizon_features import ShortHorizonFeatures
from app.strategy.short_horizon import IntradayMomentumConfig, IntradayMomentumEngine


class IntradayMomentumEngineTest(unittest.TestCase):
    def test_creates_candidate_on_strong_opening_return(self) -> None:
        features = _features(ret_open_30m=0.012, volume_zscore=1.2, market_alignment_score=0.8)

        candidate = IntradayMomentumEngine().generate_candidate(features, entry_price=50_000)

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.strategy_family, "intraday_momentum")
        self.assertEqual(candidate.signal_name, "gao_2018_opening_return_momentum")
        self.assertIn("IntradayMomentum", candidate.ontology_tags)
        self.assertIn("OpeningReturnStrength", candidate.ontology_tags)
        self.assertIn("VolumeConfirmedMomentum", candidate.ontology_tags)
        self.assertIn("MarketDirectionAligned", candidate.ontology_tags)
        self.assertIn("LateDayContinuationCandidate", candidate.ontology_tags)

    def test_rejects_without_volume_confirmation(self) -> None:
        features = _features(ret_open_30m=0.012, volume_zscore=0.1, market_alignment_score=0.8)

        candidate = IntradayMomentumEngine().generate_candidate(features, entry_price=50_000)

        self.assertIsNone(candidate)

    def test_candidate_has_expected_exit_price(self) -> None:
        features = _features(ret_open_30m=0.012, volume_zscore=1.2, market_alignment_score=0.8)

        candidate = IntradayMomentumEngine().generate_candidate(features, entry_price=50_000)

        self.assertIsNotNone(candidate)
        assert candidate is not None
        expected_late_return = 0.25 * 0.012
        self.assertAlmostEqual(candidate.expected_exit_price, 50_000 * (1 + expected_late_return))
        self.assertAlmostEqual(candidate.gross_expected_return, expected_late_return)
        self.assertEqual(candidate.features["opening_return_feature"], "ret_open_30m")
        self.assertEqual(candidate.features["beta_r_open_to_late"], 0.25)

    def test_can_use_opening_10m_window(self) -> None:
        config = IntradayMomentumConfig(opening_window_minutes=10)
        features = _features(ret_open_10m=0.009, ret_open_30m=None, volume_zscore=1.2, market_alignment_score=0.8)

        candidate = IntradayMomentumEngine(config).generate_candidate(features, entry_price=50_000)

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.features["opening_return_feature"], "ret_open_10m")
        self.assertAlmostEqual(candidate.gross_expected_return, 0.25 * 0.009)

    def test_rejects_without_market_alignment_when_required(self) -> None:
        features = _features(ret_open_30m=0.012, volume_zscore=1.2, market_alignment_score=None)

        candidate = IntradayMomentumEngine().generate_candidate(features, entry_price=50_000)

        self.assertIsNone(candidate)

    def test_paper_only_by_default(self) -> None:
        features = _features(ret_open_30m=0.012, volume_zscore=1.2, market_alignment_score=0.8)

        candidate = IntradayMomentumEngine().generate_candidate(
            features,
            entry_price=50_000,
            trading_mode="live",
        )

        self.assertIsNone(candidate)

    def test_korean_market_time_config_is_adjustable(self) -> None:
        config = IntradayMomentumConfig(session_open=time(8, 30), session_close=time(16, 0))
        features = _features(ret_open_30m=0.012, volume_zscore=1.2, market_alignment_score=0.8)

        candidate = IntradayMomentumEngine(config).generate_candidate(features, entry_price=50_000)

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.features["session_open_hour"], 8.0)
        self.assertEqual(candidate.features["session_close_hour"], 16.0)


def _features(
    *,
    ret_open_30m: float | None,
    volume_zscore: float | None,
    market_alignment_score: float | None,
    ret_open_10m: float | None = 0.006,
    is_valid: bool = True,
) -> ShortHorizonFeatures:
    return ShortHorizonFeatures(
        ticker="005930",
        timestamp=datetime(2026, 1, 2, 9, 35, tzinfo=timezone.utc),
        returns_by_window={
            "ret_1m": 0.001,
            "ret_3m": 0.002,
            "ret_5m": 0.003,
            "ret_15m": 0.006,
            "ret_30m": 0.011,
            "ret_1d": 0.015,
            "ret_open_10m": ret_open_10m,
            "ret_open_30m": ret_open_30m,
            "ret_preclose_30m": None,
        },
        realized_volatility={
            "realized_volatility_5m": 0.002,
            "realized_volatility_30m": 0.004,
        },
        volume_zscore=volume_zscore,
        spread_rate=0.0007,
        orderbook_depth_score=0.8,
        liquidity_score=0.76,
        market_alignment_score=market_alignment_score,
        time_of_day_weight=1.0,
        is_valid=is_valid,
        missing_fields=(),
    )


if __name__ == "__main__":
    unittest.main()

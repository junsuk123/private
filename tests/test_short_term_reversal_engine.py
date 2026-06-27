from __future__ import annotations

import unittest
from datetime import datetime, timezone

from app.features.short_horizon_features import ShortHorizonFeatures
from app.strategy.short_horizon import ShortTermReversalEngine


class ShortTermReversalEngineTest(unittest.TestCase):
    def test_creates_candidate_on_large_negative_return(self) -> None:
        features = _features(ret_5m=-0.008, vol_5m=0.003, spread_rate=0.0008, liquidity_score=0.72)

        candidate = ShortTermReversalEngine().generate_candidate(features, entry_price=10_000)

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.strategy_family, "short_term_reversal")
        self.assertEqual(candidate.signal_name, "jegadeesh_1990_short_term_reversal")
        self.assertIn("ShortTermReversalCandidate", candidate.ontology_tags)
        self.assertIn("LiquiditySupportedReversal", candidate.ontology_tags)
        self.assertIn("BidAskBounceRisk", candidate.ontology_tags)
        self.assertAlmostEqual(candidate.features["shock_score"], abs(-0.008) / 0.003)

    def test_rejects_wide_spread(self) -> None:
        features = _features(ret_5m=-0.008, vol_5m=0.003, spread_rate=0.002, liquidity_score=0.80)

        candidate = ShortTermReversalEngine().generate_candidate(features, entry_price=10_000)

        self.assertIsNone(candidate)

    def test_candidate_has_expected_exit_price(self) -> None:
        features = _features(ret_5m=-0.01, vol_5m=0.003, spread_rate=0.0007, liquidity_score=0.82)

        candidate = ShortTermReversalEngine().generate_candidate(features, entry_price=20_000)

        self.assertIsNotNone(candidate)
        assert candidate is not None
        expected_rebound = min(0.006, abs(-0.01) * 0.35)
        self.assertAlmostEqual(candidate.expected_exit_price, 20_000 * (1 + expected_rebound))
        self.assertAlmostEqual(candidate.gross_expected_return, expected_rebound)
        self.assertGreater(candidate.expected_exit_price, candidate.entry_price)
        self.assertIn("target_net_return", candidate.features)

    def test_rejects_invalid_or_insufficient_data(self) -> None:
        features = _features(ret_5m=-0.01, vol_5m=0.003, spread_rate=0.0007, liquidity_score=0.82, is_valid=False)

        candidate = ShortTermReversalEngine().generate_candidate(features, entry_price=20_000)

        self.assertIsNone(candidate)

    def test_paper_only_by_default(self) -> None:
        features = _features(ret_5m=-0.01, vol_5m=0.003, spread_rate=0.0007, liquidity_score=0.82)

        candidate = ShortTermReversalEngine().generate_candidate(
            features,
            entry_price=20_000,
            trading_mode="live",
        )

        self.assertIsNone(candidate)

    def test_no_candidate_for_positive_return(self) -> None:
        features = _features(ret_5m=0.01, vol_5m=0.003, spread_rate=0.0007, liquidity_score=0.82)

        candidate = ShortTermReversalEngine().generate_candidate(features, entry_price=20_000)

        self.assertIsNone(candidate)


def _features(
    *,
    ret_5m: float,
    vol_5m: float,
    spread_rate: float,
    liquidity_score: float,
    is_valid: bool = True,
) -> ShortHorizonFeatures:
    return ShortHorizonFeatures(
        ticker="005930",
        timestamp=datetime(2026, 1, 2, 9, 35, tzinfo=timezone.utc),
        returns_by_window={
            "ret_1m": -0.002,
            "ret_3m": -0.004,
            "ret_5m": ret_5m,
            "ret_15m": -0.006,
            "ret_30m": -0.004,
            "ret_1d": -0.012,
            "ret_open_10m": -0.006,
            "ret_open_30m": -0.009,
            "ret_preclose_30m": None,
        },
        realized_volatility={
            "realized_volatility_5m": vol_5m,
            "realized_volatility_30m": 0.004,
        },
        volume_zscore=1.4,
        spread_rate=spread_rate,
        orderbook_depth_score=0.77,
        liquidity_score=liquidity_score,
        market_alignment_score=0.0,
        time_of_day_weight=1.0,
        is_valid=is_valid,
        missing_fields=(),
    )


if __name__ == "__main__":
    unittest.main()

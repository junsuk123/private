from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from app.features.schemas import OHLCVBar
from app.features.short_horizon_features import ShortHorizonFeatures
from app.strategy.pairs_relative_value import (
    PairAssetProfile,
    PairRelativeValueConfig,
    PairRelativeValueEngine,
    PairUniverseBuilder,
)


class PairRelativeValueTest(unittest.TestCase):
    def test_pair_universe_created(self) -> None:
        histories = {
            "AAA": _bars("AAA", [100 + i for i in range(60)]),
            "BBB": _bars("BBB", [101 + i * 1.01 for i in range(60)]),
        }
        profiles = {
            "AAA": PairAssetProfile("AAA", sector="Tech", theme="Memory", market_beta=1.05),
            "BBB": PairAssetProfile("BBB", sector="Tech", theme="Memory", market_beta=1.10),
        }

        pairs = PairUniverseBuilder().build(histories, profiles=profiles)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].ticker_a, "AAA")
        self.assertEqual(pairs[0].ticker_b, "BBB")
        self.assertLess(pairs[0].pair_distance, 0.15)

    def test_mean_reversion_candidate_created(self) -> None:
        histories = _diverged_histories()
        profiles = {
            "AAA": PairAssetProfile("AAA", sector="Tech", theme="Memory", market_beta=1.05),
            "BBB": PairAssetProfile("BBB", sector="Tech", theme="Memory", market_beta=1.07),
        }
        pair = PairUniverseBuilder().build(histories, profiles=profiles)[0]
        features = {"AAA": _features("AAA")}

        candidate = PairRelativeValueEngine().generate_candidate(pair, histories, features)

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.ticker, "AAA")
        self.assertEqual(candidate.strategy_family, "pair_relative_value")
        self.assertGreater(candidate.expected_exit_price, candidate.entry_price)
        self.assertIn("CloseSubstitutePair", candidate.ontology_tags)
        self.assertIn("MeanReversionCandidate", candidate.ontology_tags)
        self.assertIn("RelativeUndervaluation", candidate.ontology_tags)
        self.assertLess(candidate.features["spread_z"], -2.0)
        self.assertGreaterEqual(candidate.features["net_expected_return_after_cost"], 0.004)

    def test_rejects_high_distance_pair(self) -> None:
        histories = {
            "AAA": _bars("AAA", [100 + i for i in range(60)]),
            "BBB": _bars("BBB", [100 + i * 5 for i in range(60)]),
        }
        profiles = {
            "AAA": PairAssetProfile("AAA", sector="Tech", theme="Memory", market_beta=1.05),
            "BBB": PairAssetProfile("BBB", sector="Tech", theme="Memory", market_beta=1.07),
        }

        pairs = PairUniverseBuilder().build(histories, profiles=profiles)

        self.assertEqual(pairs, ())

    def test_rejects_low_liquidity_or_wide_spread(self) -> None:
        histories = _diverged_histories()
        profiles = {
            "AAA": PairAssetProfile("AAA", sector="Tech", theme="Memory", market_beta=1.05),
            "BBB": PairAssetProfile("BBB", sector="Tech", theme="Memory", market_beta=1.07),
        }
        pair = PairUniverseBuilder().build(histories, profiles=profiles)[0]

        self.assertIsNone(
            PairRelativeValueEngine().generate_candidate(pair, histories, {"AAA": _features("AAA", liquidity_score=0.2)})
        )
        self.assertIsNone(
            PairRelativeValueEngine().generate_candidate(pair, histories, {"AAA": _features("AAA", spread_rate=0.002)})
        )

    def test_paper_only_by_default(self) -> None:
        histories = _diverged_histories()
        profiles = {
            "AAA": PairAssetProfile("AAA", sector="Tech", theme="Memory", market_beta=1.05),
            "BBB": PairAssetProfile("BBB", sector="Tech", theme="Memory", market_beta=1.07),
        }
        pair = PairUniverseBuilder().build(histories, profiles=profiles)[0]

        candidate = PairRelativeValueEngine().generate_candidate(
            pair,
            histories,
            {"AAA": _features("AAA")},
            trading_mode="live",
        )

        self.assertIsNone(candidate)

    def test_cost_gate_blocks_small_convergence(self) -> None:
        histories = _diverged_histories()
        profiles = {
            "AAA": PairAssetProfile("AAA", sector="Tech", theme="Memory", market_beta=1.05),
            "BBB": PairAssetProfile("BBB", sector="Tech", theme="Memory", market_beta=1.07),
        }
        pair = PairUniverseBuilder().build(histories, profiles=profiles)[0]
        config = PairRelativeValueConfig(convergence_ratio=0.02)

        candidate = PairRelativeValueEngine(config).generate_candidate(pair, histories, {"AAA": _features("AAA")})

        self.assertIsNone(candidate)


def _diverged_histories() -> dict[str, tuple[OHLCVBar, ...]]:
    prices_a = [100 + i * 0.2 for i in range(55)] + [95, 94, 93, 92, 91]
    prices_b = [101 + i * 0.2 for i in range(60)]
    return {
        "AAA": _bars("AAA", prices_a),
        "BBB": _bars("BBB", prices_b),
    }


def _bars(ticker: str, closes: list[float]) -> tuple[OHLCVBar, ...]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return tuple(
        OHLCVBar(
            ticker=ticker,
            as_of=start + timedelta(days=index),
            open=close,
            high=close * 1.01,
            low=close * 0.99,
            close=close,
            volume=1_000_000,
        )
        for index, close in enumerate(closes)
    )


def _features(
    ticker: str,
    *,
    liquidity_score: float = 0.8,
    spread_rate: float = 0.0007,
) -> ShortHorizonFeatures:
    return ShortHorizonFeatures(
        ticker=ticker,
        timestamp=datetime(2026, 3, 1, tzinfo=timezone.utc),
        returns_by_window={
            "ret_1m": -0.001,
            "ret_3m": -0.002,
            "ret_5m": -0.003,
            "ret_15m": -0.004,
            "ret_30m": -0.005,
            "ret_1d": -0.02,
            "ret_open_10m": None,
            "ret_open_30m": None,
            "ret_preclose_30m": None,
        },
        realized_volatility={
            "realized_volatility_5m": 0.002,
            "realized_volatility_30m": 0.004,
        },
        volume_zscore=1.0,
        spread_rate=spread_rate,
        orderbook_depth_score=0.8,
        liquidity_score=liquidity_score,
        market_alignment_score=0.5,
        time_of_day_weight=0.8,
        is_valid=True,
        missing_fields=(),
    )


if __name__ == "__main__":
    unittest.main()

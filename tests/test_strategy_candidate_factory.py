from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from app.features.schemas import OHLCVBar
from app.features.short_horizon_features import ShortHorizonFeatures
from app.strategy import (
    PairAssetProfile,
    StrategyCandidateFactory,
    StrategyCandidateFactoryInput,
    StrategyFactoryConfig,
)


class StrategyCandidateFactoryTest(unittest.TestCase):
    def test_factory_integrates_enabled_engines_and_attaches_costs(self) -> None:
        bars = _bars("005930", [10_000 + i * 10 for i in range(25)] + [10_500])
        features = _momentum_features("005930", bars[-1].as_of)
        factory = StrategyCandidateFactory(
            StrategyFactoryConfig(
                enable_short_term_reversal=False,
                enable_pair_relative_value=False,
                target_net_return=0.0,
            )
        )

        result = factory.build(
            StrategyCandidateFactoryInput(
                features_by_ticker={"005930": features},
                price_history_by_ticker={"005930": bars},
                entry_prices={"005930": bars[-1].close},
            )
        )

        self.assertGreaterEqual(len(result.candidates), 1)
        self.assertTrue(all(item.cost_breakdown.as_dict() for item in result.candidates))
        self.assertTrue(all(item.ranking_score > 0 for item in result.candidates))
        self.assertIn(result.candidates[0].candidate.strategy_family, {"intraday_momentum", "technical_rule"})

    def test_factory_filters_candidates_below_net_return(self) -> None:
        bars = _bars("005930", [10_000 + i * 10 for i in range(25)] + [10_500])
        features = _momentum_features("005930", bars[-1].as_of)
        factory = StrategyCandidateFactory(
            StrategyFactoryConfig(
                enable_short_term_reversal=False,
                enable_pair_relative_value=False,
                target_net_return=0.02,
            )
        )

        result = factory.build(
            StrategyCandidateFactoryInput(
                features_by_ticker={"005930": features},
                price_history_by_ticker={"005930": bars},
                entry_prices={"005930": bars[-1].close},
            )
        )

        self.assertEqual(result.candidates, ())
        self.assertTrue(result.filtered_candidates)
        self.assertTrue(any(item.reason == "BELOW_TARGET_NET_RETURN_AFTER_COST" for item in result.filtered_candidates))

    def test_factory_result_converts_to_order_intent_with_cost_breakdown(self) -> None:
        bars = _bars("005930", [10_000 + i * 10 for i in range(25)] + [10_500])
        features = _momentum_features("005930", bars[-1].as_of)
        result = StrategyCandidateFactory(
            StrategyFactoryConfig(
                enable_short_term_reversal=False,
                enable_pair_relative_value=False,
                target_net_return=0.0,
            )
        ).build(
            StrategyCandidateFactoryInput(
                features_by_ticker={"005930": features},
                price_history_by_ticker={"005930": bars},
                entry_prices={"005930": bars[-1].close},
            )
        )

        intent = result.candidates[0].to_order_intent(
            market="KR",
            suggested_weight=0.01,
            valid_until=datetime.now(timezone.utc) + timedelta(minutes=30),
            source_data_ids=("factory-test",),
        )

        self.assertIsNotNone(intent.cost_breakdown)
        self.assertIsNotNone(intent.expected_exit_price)
        self.assertIn("ranking_score", intent.strategy_metadata)
        self.assertGreater(intent.strategy_metadata["ranking_score"], 0)

    def test_factory_can_build_pair_relative_value_candidate(self) -> None:
        histories = _diverged_histories()
        profiles = {
            "AAA": PairAssetProfile("AAA", sector="Tech", theme="Memory", market_beta=1.05),
            "BBB": PairAssetProfile("BBB", sector="Tech", theme="Memory", market_beta=1.07),
        }
        factory = StrategyCandidateFactory(
            StrategyFactoryConfig(
                enable_short_term_reversal=False,
                enable_intraday_momentum=False,
                enable_technical_rule=False,
                target_net_return=0.0,
            )
        )

        result = factory.build(
            StrategyCandidateFactoryInput(
                features_by_ticker={"AAA": _pair_features("AAA")},
                price_history_by_ticker=histories,
                pair_profiles=profiles,
            )
        )

        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].candidate.strategy_family, "pair_relative_value")
        self.assertIsNotNone(result.candidates[0].cost_breakdown)

    def test_factory_paper_only_by_default(self) -> None:
        bars = _bars("005930", [10_000 + i * 10 for i in range(25)] + [10_500])
        result = StrategyCandidateFactory().build(
            StrategyCandidateFactoryInput(
                features_by_ticker={"005930": _momentum_features("005930", bars[-1].as_of)},
                price_history_by_ticker={"005930": bars},
            ),
            trading_mode="live",
        )

        self.assertEqual(result.candidates, ())
        self.assertEqual(result.filtered_candidates, ())


def _momentum_features(ticker: str, timestamp: datetime) -> ShortHorizonFeatures:
    return ShortHorizonFeatures(
        ticker=ticker,
        timestamp=timestamp,
        returns_by_window={
            "ret_1m": 0.003,
            "ret_3m": 0.006,
            "ret_5m": 0.009,
            "ret_15m": 0.014,
            "ret_30m": 0.018,
            "ret_1d": 0.025,
            "ret_open_10m": 0.008,
            "ret_open_30m": 0.02,
            "ret_preclose_30m": None,
        },
        realized_volatility={
            "realized_volatility_5m": 0.002,
            "realized_volatility_30m": 0.004,
        },
        volume_zscore=2.5,
        spread_rate=0.0005,
        orderbook_depth_score=0.8,
        liquidity_score=0.85,
        market_alignment_score=0.9,
        time_of_day_weight=1.0,
        is_valid=True,
        missing_fields=(),
    )


def _pair_features(ticker: str) -> ShortHorizonFeatures:
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
        spread_rate=0.0007,
        orderbook_depth_score=0.8,
        liquidity_score=0.8,
        market_alignment_score=0.5,
        time_of_day_weight=0.8,
        is_valid=True,
        missing_fields=(),
    )


def _bars(ticker: str, closes: list[float]) -> tuple[OHLCVBar, ...]:
    start = datetime(2026, 1, 2, 9, 0, tzinfo=timezone.utc)
    return tuple(
        OHLCVBar(
            ticker=ticker,
            as_of=start + timedelta(minutes=index),
            open=close,
            high=close * 1.01,
            low=close * 0.99,
            close=close,
            volume=2_000 if index == len(closes) - 1 else 1_000,
        )
        for index, close in enumerate(closes)
    )


def _diverged_histories() -> dict[str, tuple[OHLCVBar, ...]]:
    prices_a = [100 + i * 0.2 for i in range(55)] + [95, 94, 93, 92, 91]
    prices_b = [101 + i * 0.2 for i in range(60)]
    return {
        "AAA": _daily_bars("AAA", prices_a),
        "BBB": _daily_bars("BBB", prices_b),
    }


def _daily_bars(ticker: str, closes: list[float]) -> tuple[OHLCVBar, ...]:
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


if __name__ == "__main__":
    unittest.main()

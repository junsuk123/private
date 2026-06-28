from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.features import OHLCVBar, ShortHorizonFeatureBuilder, TickerRollingFeatureState


class ShortHorizonFeatureBuilderTest(unittest.TestCase):
    def test_builds_windowed_returns_volume_volatility_and_orderbook_features(self) -> None:
        bars = _minute_bars(40)
        index_bars = _minute_bars(40, ticker="KOSPI", start_price=2500.0, step=2.0)
        daily = (
            OHLCVBar("TEST", bars[0].as_of - timedelta(days=1), 95, 96, 94, 95, 10_000),
        )

        features = ShortHorizonFeatureBuilder().build(
            bars,
            daily_bars=daily,
            market_index_bars=index_bars,
            orderbook={"best_bid": 138.8, "best_ask": 139.2, "bid_depth": 400_000, "ask_depth": 350_000},
        )

        self.assertTrue(features.is_valid)
        self.assertAlmostEqual(features.returns_by_window["ret_1m"], 139 / 138 - 1)
        self.assertAlmostEqual(features.returns_by_window["ret_3m"], 139 / 136 - 1)
        self.assertAlmostEqual(features.returns_by_window["ret_5m"], 139 / 134 - 1)
        self.assertAlmostEqual(features.returns_by_window["ret_15m"], 139 / 124 - 1)
        self.assertAlmostEqual(features.returns_by_window["ret_30m"], 139 / 109 - 1)
        self.assertAlmostEqual(features.returns_by_window["ret_1d"], 139 / 95 - 1)
        self.assertAlmostEqual(features.returns_by_window["ret_open_10m"], 110 / 100 - 1)
        self.assertAlmostEqual(features.returns_by_window["ret_open_30m"], 130 / 100 - 1)
        self.assertIsNotNone(features.realized_volatility["realized_volatility_5m"])
        self.assertIsNotNone(features.realized_volatility["realized_volatility_30m"])
        self.assertIsNotNone(features.volume_zscore)
        self.assertIsNotNone(features.spread_rate)
        self.assertIsNotNone(features.orderbook_depth_score)
        self.assertIsNotNone(features.liquidity_score)
        self.assertEqual(features.market_alignment_score, 1.0)
        self.assertEqual(features.time_of_day_weight, 0.6)

    def test_missing_data_is_reported_and_invalid(self) -> None:
        features = ShortHorizonFeatureBuilder().build(_minute_bars(3), as_of=_minute_bars(3)[-1].as_of)

        self.assertFalse(features.is_valid)
        self.assertIsNone(features.returns_by_window["ret_5m"])
        self.assertIn("ret_5m", features.missing_fields)
        self.assertIn("volume_zscore", features.missing_fields)
        self.assertIn("spread_rate", features.missing_fields)
        self.assertIn("market_alignment_score", features.missing_fields)

    def test_future_bars_do_not_change_as_of_features(self) -> None:
        bars = _minute_bars(20)
        as_of = bars[10].as_of
        future_shock = OHLCVBar("TEST", as_of + timedelta(minutes=1), 1_000, 1_010, 990, 1_000, 999_999)

        builder = ShortHorizonFeatureBuilder()
        baseline = builder.build(bars, as_of=as_of)
        with_future = builder.build(bars + (future_shock,), as_of=as_of)

        self.assertEqual(baseline.timestamp, with_future.timestamp)
        self.assertEqual(baseline.returns_by_window, with_future.returns_by_window)
        self.assertEqual(baseline.realized_volatility, with_future.realized_volatility)
        self.assertEqual(baseline.volume_zscore, with_future.volume_zscore)

    def test_preclose_return_uses_only_visible_last_thirty_minutes(self) -> None:
        start = datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)
        bars = tuple(
            OHLCVBar("TEST", start + timedelta(minutes=i), 100 + i, 101 + i, 99 + i, 100 + i, 1_000 + i)
            for i in range(31)
        )

        features = ShortHorizonFeatureBuilder().build(bars)

        self.assertAlmostEqual(features.returns_by_window["ret_preclose_30m"], 130 / 100 - 1)

    def test_rolling_state_matches_batch_builder(self) -> None:
        bars = _minute_bars(45)
        index_bars = _minute_bars(45, ticker="KOSPI", start_price=2500.0, step=2.0)
        orderbook = {"best_bid": 143.8, "best_ask": 144.2, "bid_depth": 400_000, "ask_depth": 350_000}
        daily = (
            OHLCVBar("TEST", bars[0].as_of - timedelta(days=1), 95, 96, 94, 95, 10_000),
        )

        state = TickerRollingFeatureState("TEST")
        for bar in bars:
            state.add_bar(bar)
        index_state = TickerRollingFeatureState("KOSPI")
        for bar in index_bars:
            index_state.add_bar(bar)

        rolling = state.build(daily_bars=daily, market_index_state=index_state, orderbook=orderbook)
        batch = ShortHorizonFeatureBuilder().build(
            bars,
            daily_bars=daily,
            market_index_bars=index_bars,
            orderbook=orderbook,
        )

        self.assertEqual(rolling.returns_by_window, batch.returns_by_window)
        self.assertEqual(rolling.realized_volatility, batch.realized_volatility)
        self.assertEqual(rolling.volume_zscore, batch.volume_zscore)
        self.assertEqual(rolling.market_alignment_score, batch.market_alignment_score)
        self.assertEqual(rolling.missing_fields, batch.missing_fields)

    def test_rolling_state_preserves_no_lookahead(self) -> None:
        bars = _minute_bars(30)
        as_of = bars[15].as_of
        state = TickerRollingFeatureState("TEST")
        for bar in bars:
            state.add_bar(bar)

        rolling = state.build(as_of=as_of)
        batch = ShortHorizonFeatureBuilder().build(bars, as_of=as_of)

        self.assertEqual(rolling.timestamp, batch.timestamp)
        self.assertEqual(rolling.returns_by_window, batch.returns_by_window)
        self.assertEqual(rolling.realized_volatility, batch.realized_volatility)
        self.assertEqual(rolling.volume_zscore, batch.volume_zscore)


def _minute_bars(
    count: int,
    *,
    ticker: str = "TEST",
    start_price: float = 100.0,
    step: float = 1.0,
) -> tuple[OHLCVBar, ...]:
    start = datetime(2026, 1, 2, 9, 0, tzinfo=timezone.utc)
    return tuple(
        OHLCVBar(
            ticker=ticker,
            as_of=start + timedelta(minutes=i),
            open=start_price + i * step,
            high=start_price + i * step + 1,
            low=start_price + i * step - 1,
            close=start_price + i * step,
            volume=1_000 + i * i,
        )
        for i in range(count)
    )


if __name__ == "__main__":
    unittest.main()

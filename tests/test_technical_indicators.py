from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from app.features.schemas import OHLCVBar
from app.technical import indicators as ti


def _bars(prices, *, highs=None, lows=None, vols=None, start=None):
    start = start or datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    out = []
    for i, close in enumerate(prices):
        high = highs[i] if highs else close
        low = lows[i] if lows else close
        vol = vols[i] if vols else 1000.0
        out.append(
            OHLCVBar(
                ticker="TEST",
                as_of=start + timedelta(minutes=i),
                open=close,
                high=high,
                low=low,
                close=close,
                volume=vol,
            )
        )
    return out


class TestMovingAverages:
    def test_sma_basic(self):
        assert ti.sma([1, 2, 3, 4, 5], 5) == 3.0
        assert ti.sma([1, 2, 3, 4, 5], 2) == 4.5

    def test_sma_insufficient(self):
        assert ti.sma([1, 2], 5) is None
        assert ti.sma([], 3) is None

    def test_ema_equals_flat_series(self):
        # A constant series has EMA equal to the constant.
        assert ti.ema([5.0] * 30, 10) == 5.0

    def test_ema_seed_is_sma(self):
        # With exactly `period` points the EMA equals the SMA seed.
        assert ti.ema([1, 2, 3, 4], 4) == 2.5

    def test_ema_reacts_faster_to_recent_move(self):
        # After a recent step up, EMA weights the newest values more than SMA.
        step_up = [10.0] * 20 + [20.0, 20.0, 20.0]
        assert ti.ema(step_up, 10) > ti.sma(step_up, 10)
        step_down = [20.0] * 20 + [10.0, 10.0, 10.0]
        assert ti.ema(step_down, 10) < ti.sma(step_down, 10)

    def test_ema_insufficient(self):
        assert ti.ema([1, 2], 10) is None


class TestMacd:
    def test_macd_flat_is_zero(self):
        result = ti.macd([10.0] * 60)
        assert result.ok
        assert abs(result.macd) < 1e-9
        assert abs(result.histogram) < 1e-9

    def test_macd_positive_in_uptrend(self):
        result = ti.macd([float(x) for x in range(1, 80)])
        assert result.ok
        assert result.macd > 0  # fast EMA above slow EMA when rising

    def test_macd_insufficient(self):
        result = ti.macd([1.0, 2.0, 3.0])
        assert not result.ok
        assert result.reason == "insufficient_data"
        assert result.macd is None

    def test_macd_invalid_periods(self):
        result = ti.macd([float(x) for x in range(100)], fast=26, slow=12)
        assert not result.ok
        assert result.reason == "invalid_periods"


class TestRsi:
    def test_rsi_all_gains_is_100(self):
        assert ti.rsi(list(range(1, 30)), 14) == 100.0

    def test_rsi_all_losses_is_0(self):
        assert ti.rsi(list(range(30, 1, -1)), 14) == 0.0

    def test_rsi_flat_is_canonical_100(self):
        # Canonical indicator_engine.rsi returns 100.0 when avg_loss == 0
        # (flat/only-gains window). Documented quirk; the signal layer treats a
        # near-zero price range as neutral rather than trusting this extreme.
        assert ti.rsi([10.0] * 30, 14) == 100.0

    def test_rsi_midrange_bounds(self):
        prices = [10, 11, 10.5, 11.5, 11, 12, 11.5, 12.5, 12, 13, 12.5, 13.5, 13, 14, 13.5, 14.5]
        value = ti.rsi(prices, 14)
        assert value is not None
        assert 0.0 <= value <= 100.0

    def test_rsi_insufficient(self):
        assert ti.rsi([1, 2, 3], 14) is None


class TestBollinger:
    def test_bollinger_flat(self):
        result = ti.bollinger([10.0] * 25, 20, 2.0)
        assert result.ok
        assert result.mid == 10.0
        assert result.upper == 10.0 and result.lower == 10.0
        assert result.percent_b == 0.5  # band width zero -> neutral

    def test_bollinger_percent_b_range(self):
        prices = [10 + math.sin(i / 3.0) for i in range(40)]
        result = ti.bollinger(prices, 20, 2.0)
        assert result.ok
        assert result.upper > result.mid > result.lower
        assert result.bandwidth is not None and result.bandwidth >= 0

    def test_bollinger_insufficient(self):
        result = ti.bollinger([1, 2, 3], 20)
        assert not result.ok
        assert result.reason == "insufficient_data"


class TestDonchian:
    def test_donchian_from_bars(self):
        bars = _bars([5, 6, 7, 8, 9], highs=[5, 7, 7, 10, 9], lows=[4, 5, 6, 7, 8])
        result = ti.donchian(bars, 5)
        assert result.ok
        assert result.high == 10 and result.low == 4
        assert result.mid == 7.0

    def test_donchian_from_high_low(self):
        result = ti.donchian([1, 2, 3, 9, 4], 3, [1, 1, 2, 3, 2])
        assert result.ok
        assert result.high == 9 and result.low == 2

    def test_donchian_insufficient(self):
        result = ti.donchian(_bars([1, 2]), 5)
        assert not result.ok


class TestAtrVwap:
    def test_atr_constant_range(self):
        bars = _bars([10] * 20, highs=[11] * 20, lows=[9] * 20)
        value = ti.atr(bars, 14)
        assert value is not None
        assert abs(value - 2.0) < 1e-9  # high-low = 2 every bar

    def test_atr_insufficient(self):
        assert ti.atr(_bars([1, 2, 3]), 14) is None

    def test_vwap_matches_manual(self):
        bars = _bars([10, 20], vols=[100, 300])
        # typical = close here; vwap = (10*100 + 20*300)/400 = 17.5
        assert ti.vwap(bars) == 17.5

    def test_vwap_window(self):
        bars = _bars([10, 20, 30], vols=[100, 100, 100])
        assert ti.vwap(bars, window=2) == 25.0

    def test_vwap_zero_volume_is_none(self):
        bars = _bars([10, 20], vols=[0, 0])
        assert ti.vwap(bars) is None


class TestRollingHelpers:
    def test_rolling_return(self):
        assert ti.rolling_return([100, 105, 110], 2) == (110 / 100 - 1)

    def test_rolling_return_zero_base(self):
        assert ti.rolling_return([0, 1, 2], 2) is None

    def test_rolling_zscore(self):
        assert ti.rolling_zscore([1, 1, 1, 5], 3) is None  # zero deviation baseline
        z = ti.rolling_zscore([1, 2, 3, 100], 3)
        assert z is not None and z > 0

    def test_volume_spike_ratio(self):
        assert ti.volume_spike_ratio([100, 100, 100, 300], 3) == 3.0

    def test_volume_spike_zero_baseline(self):
        assert ti.volume_spike_ratio([0, 0, 0, 5], 3) is None


class TestMicrostructure:
    def test_spread_bps(self):
        # bid 99.9, ask 100.1 -> mid 100, spread 0.2 -> 20 bps
        assert abs(ti.spread_bps(99.9, 100.1) - 20.0) < 1e-6

    def test_spread_bps_invalid(self):
        assert ti.spread_bps(0, 100) is None
        assert ti.spread_bps(101, 100) is None  # crossed book
        assert ti.spread_bps("x", 100) is None

    def test_orderbook_imbalance(self):
        assert ti.orderbook_imbalance(300, 100) == 0.5
        assert ti.orderbook_imbalance(100, 100) == 0.0
        assert ti.orderbook_imbalance(0, 0) is None


class TestNaNSafety:
    def test_nonfinite_inputs_return_none(self):
        assert ti.sma([1, 2, float("nan")], 3) is None
        assert ti.ema([1, float("inf"), 3], 2) is None
        assert not ti.macd([float("nan")] * 60).ok
        assert ti.rsi([1, 2, float("nan"), 4] * 5, 14) is None

    def test_empty_inputs(self):
        assert ti.vwap([]) is None
        assert ti.atr([]) is None
        assert not ti.donchian([], 5).ok

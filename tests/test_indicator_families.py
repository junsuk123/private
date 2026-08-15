"""Unit and causality tests for the indicator families added to the canonical engine.

Two properties matter more than exact numbers:
  * insufficient / degenerate input returns None instead of a fabricated neutral
    value, because a model cannot tell a real neutral from a missing one; and
  * no value changes when FUTURE bars are appended, which is what makes the
    indicator usable at decision_time.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.features import indicator_engine as ie
from app.features.schemas import OHLCVBar
from app.technical import indicators as ind

START = datetime(2026, 8, 6, 0, 0, tzinfo=timezone.utc)


def _bars(closes, *, volume=1000.0, spread=1.0):
    """Well-formed bars around each close so OHLC validation passes."""
    out = []
    for index, close in enumerate(closes):
        out.append(
            OHLCVBar(
                ticker="T",
                as_of=START + timedelta(minutes=index),
                open=close,
                high=close + spread,
                low=max(0.01, close - spread),
                close=close,
                volume=volume,
            )
        )
    return tuple(out)


def _ramp(count=120, start=100.0, step=0.5):
    return [start + step * index for index in range(count)]


def _flat(count=120, value=100.0):
    return [value] * count


def _zigzag(count=120, base=100.0, amplitude=2.0):
    return [base + amplitude * ((-1) ** index) for index in range(count)]


# --------------------------------------------------------------------------- #
# Sign / range behaviour on known series                                       #
# --------------------------------------------------------------------------- #
class TestKnownSeries:
    def test_uptrend_signs(self):
        up = _ramp()
        assert ie.ma_slope_bps(up, 20) > 0
        assert ie.ma_alignment_score(up, (5, 20, 60)) == 1.0
        assert ie.price_disparity(up, 20) > 0
        assert ie.roc_bps(up, 10) > 0
        assert ie.momentum(up, 10) > 0

    def test_downtrend_signs(self):
        down = _ramp(step=-0.5)
        assert ie.ma_slope_bps(down, 20) < 0
        assert ie.ma_alignment_score(down, (5, 20, 60)) == -1.0
        assert ie.price_disparity(down, 20) < 0
        assert ie.roc_bps(down, 10) < 0

    def test_adx_is_higher_in_a_trend_than_in_chop(self):
        bars_up = _bars(_ramp())
        bars_chop = _bars(_zigzag())
        _, _, adx_trend = ie.dmi_adx(
            ind.highs(bars_up), ind.lows(bars_up), ind.closes(bars_up)
        )
        _, _, adx_chop = ie.dmi_adx(
            ind.highs(bars_chop), ind.lows(bars_chop), ind.closes(bars_chop)
        )
        assert adx_trend is not None and adx_chop is not None
        assert adx_trend > adx_chop

    def test_dmi_direction_matches_trend(self):
        up = _bars(_ramp())
        plus_di, minus_di, _ = ie.dmi_adx(ind.highs(up), ind.lows(up), ind.closes(up))
        assert plus_di > minus_di

    def test_supertrend_and_keltner_follow_a_clean_uptrend(self):
        bars = _bars(_ramp(count=160))
        supertrend = ind.supertrend(bars)
        keltner = ind.keltner_channels(bars)

        assert supertrend.ok and supertrend.direction == 1
        assert supertrend.line < bars[-1].close
        assert keltner.ok and keltner.mid < bars[-1].close
        assert keltner.upper > keltner.mid > keltner.lower

    def test_choppiness_is_higher_for_sideways_zigzag_than_clean_trend(self):
        trending = ind.choppiness_index(_bars(_ramp(count=160)))
        sideways = ind.choppiness_index(_bars(_zigzag(count=160)))

        assert trending is not None and sideways is not None
        assert sideways > trending

    def test_williams_r_is_in_range_and_near_zero_at_highs(self):
        up = _bars(_ramp())
        value = ie.williams_r(ind.highs(up), ind.lows(up), ind.closes(up))
        assert -100.0 <= value <= 0.0
        assert value > -30.0  # closing near the window high

    def test_cci_positive_in_uptrend_negative_in_downtrend(self):
        up = _bars(_ramp())
        down = _bars(_ramp(step=-0.5))
        assert ie.cci(ind.highs(up), ind.lows(up), ind.closes(up)) > 0
        assert ie.cci(ind.highs(down), ind.lows(down), ind.closes(down)) < 0

    def test_trix_positive_in_uptrend(self):
        line, signal, hist = ie.trix(_ramp(count=200))
        assert line is not None and line > 0
        assert signal is not None and hist is not None

    def test_envelope_position_and_distances(self):
        values = _flat()
        mid, upper, lower, position, to_upper, to_lower = ie.envelope(values, 20, 0.02)
        assert mid == 100.0 and upper > mid > lower
        assert abs(position - 0.5) < 1e-9  # price sits on the middle band
        assert to_upper > 0 and to_lower > 0

    def test_envelope_position_is_not_clamped_outside_the_band(self):
        # A genuine excursion must stay visible rather than saturating at 0/1.
        values = _flat(119) + [130.0]
        _, _, _, position, _, _ = ie.envelope(values, 20, 0.02)
        assert position > 1.0

    def test_ichimoku_above_cloud_in_uptrend(self):
        up = _bars(_ramp(count=200))
        values = ie.ichimoku(ind.highs(up), ind.lows(up), ind.closes(up))
        assert values["price_vs_cloud"] == 1.0
        assert values["tenkan_kijun_gap_bps"] > 0
        assert values["cloud_thickness_bps"] >= 0

    def test_trendline_slope_and_fit_on_a_clean_ramp(self):
        values = ie.trendline(_ramp(count=120))
        assert values["slope_bps_per_bar"] > 0
        assert values["r_squared"] > 0.99
        assert values["support_residual_quantile"] <= values["resistance_residual_quantile"]

    def test_atr_percent_and_expansion(self):
        calm = _bars(_flat(), spread=0.5)
        assert ie.atr_percent(ind.highs(calm), ind.lows(calm), ind.closes(calm)) > 0
        wide = list(_flat(100, 100.0))
        bars = _bars(wide[:60], spread=0.2) + _bars(wide[60:], spread=5.0)
        expansion = ie.atr_expansion(
            ind.highs(bars), ind.lows(bars), ind.closes(bars), period=14
        )
        assert expansion is not None and expansion > 1.0

    def test_obv_and_volume_flow(self):
        up = _ramp()
        volume = [1000.0] * len(up)
        assert ie.obv_series(up, volume)[-1] > 0
        assert ie.obv_slope(up, volume, 20) > 0
        # Constant volume has zero dispersion -> z-score is undefined, not 0.
        assert ie.volume_zscore(volume, 20) is None
        spiky = volume[:-1] + [50_000.0]
        assert ie.volume_zscore(spiky, 20) > 0
        assert ie.relative_volume(spiky, 20) > 1.0

    def test_stochastic_diff_sign(self):
        up = _bars(_ramp())
        value = ie.stochastic_diff(ind.highs(up), ind.lows(up), ind.closes(up))
        assert value is not None


# --------------------------------------------------------------------------- #
# Degenerate input: None, never a fabricated number                            #
# --------------------------------------------------------------------------- #
class TestDegenerateInput:
    def test_insufficient_warmup_returns_none(self):
        short = _ramp(count=3)
        bars = _bars(short)
        assert ie.ma_slope_bps(short, 20) is None
        assert ie.ma_alignment_score(short) is None
        assert ie.trix(short) == (None, None, None)
        assert ie.dmi_adx(ind.highs(bars), ind.lows(bars), ind.closes(bars)) == (
            None, None, None,
        )
        assert ie.cci(ind.highs(bars), ind.lows(bars), ind.closes(bars)) is None
        assert ie.williams_r(ind.highs(bars), ind.lows(bars), ind.closes(bars)) is None
        assert ie.ichimoku(ind.highs(bars), ind.lows(bars), ind.closes(bars))["kijun"] is None
        assert ie.trendline(short)["slope_bps_per_bar"] is None
        assert ie.momentum(short, 10) is None
        assert ie.roc_bps(short, 10) is None
        assert ind.supertrend(bars).ok is False
        assert ind.keltner_channels(bars).ok is False
        assert ind.choppiness_index(bars) is None

    def test_zero_range_window_returns_none_not_a_midpoint(self):
        flat = _bars(_flat(), spread=0.0)
        assert ie.williams_r(ind.highs(flat), ind.lows(flat), ind.closes(flat)) is None
        assert ie.cci(ind.highs(flat), ind.lows(flat), ind.closes(flat)) is None

    def test_zero_volume_returns_none(self):
        closes_ = _ramp()
        zero = [0.0] * len(closes_)
        assert ie.obv_series(closes_, zero) is None
        assert ie.obv_slope(closes_, zero, 20) is None
        assert ie.relative_volume(zero, 20) is None

    def test_non_finite_input_is_treated_as_missing(self):
        values = _ramp()
        polluted = [*values[:-1], float("nan")]
        # The NaN is dropped, so the result stays finite rather than propagating.
        slope = ie.ma_slope_bps(polluted, 20)
        assert slope is None or math.isfinite(slope)
        assert ie.price_disparity([float("inf")] * 40, 20) is None

    def test_malformed_ohlc_is_rejected(self):
        # high < low: a negative true range must not become an ATR/ADX number.
        broken = (
            OHLCVBar(
                ticker="T", as_of=START, open=100.0, high=90.0, low=110.0,
                close=100.0, volume=10.0,
            ),
        ) * 60
        assert ie.dmi_adx(ind.highs(broken), ind.lows(broken), ind.closes(broken)) == (
            None, None, None,
        )
        assert ie.cci(ind.highs(broken), ind.lows(broken), ind.closes(broken)) is None
        assert ie.ichimoku(ind.highs(broken), ind.lows(broken), ind.closes(broken))["kijun"] is None

    def test_invalid_periods_return_none(self):
        values = _ramp()
        assert ie.ma_slope_bps(values, 20, lookback=0) is None
        assert ie.momentum(values, 0) is None
        assert ie.roc_bps(values, 0) is None
        assert ie.envelope(values, 20, percentage=0.0)[0] is None
        assert ie.volume_zscore([1.0] * 50, 1) is None

    def test_zero_base_price_returns_none(self):
        assert ie.roc_bps([0.0, 1.0, 2.0], 2) is None


# --------------------------------------------------------------------------- #
# Causality: appending future bars must not change a past value                #
# --------------------------------------------------------------------------- #
class TestCausality:
    def _snapshot(self, closes_):
        bars = _bars(closes_)
        high, low, close = ind.highs(bars), ind.lows(bars), ind.closes(bars)
        volume = ind.volumes(bars)
        return {
            "ma_slope": ie.ma_slope_bps(closes_, 20),
            "alignment": ie.ma_alignment_score(closes_),
            "disparity": ie.price_disparity(closes_, 20),
            "envelope": ie.envelope(closes_, 20, 0.02),
            "ichimoku": ie.ichimoku(high, low, close),
            "trendline": ie.trendline(closes_, 60),
            "dmi": ie.dmi_adx(high, low, close),
            "trix": ie.trix(closes_),
            "cci": ie.cci(high, low, close),
            "williams": ie.williams_r(high, low, close),
            "roc": ie.roc_bps(closes_, 10),
            "atr_pct": ie.atr_percent(high, low, close),
            "obv_slope": ie.obv_slope(close, volume, 20),
        }

    def test_future_bars_do_not_change_past_values(self):
        history = _ramp(count=200)
        before = self._snapshot(history)
        # Violent future move: if anything peeked ahead, these values would move.
        future = history + [history[-1] * 1.5 for _ in range(40)]
        after = self._snapshot(future[: len(history)])
        assert before == after

    def test_ichimoku_does_not_use_the_forward_shifted_cloud(self):
        """Senkou A/B must be computable from history alone.

        The charting convention plots the cloud 26 bars ahead. If this engine used
        that shifted series, senkou values would depend on bars after
        decision_time; recomputing on the truncated history would then differ.
        """
        history = _ramp(count=200)
        high, low, close = (
            ind.highs(_bars(history)),
            ind.lows(_bars(history)),
            ind.closes(_bars(history)),
        )
        baseline = ie.ichimoku(high, low, close)

        extended = history + [500.0] * 30
        bars_ext = _bars(extended)
        truncated = ie.ichimoku(
            ind.highs(bars_ext)[: len(history)],
            ind.lows(bars_ext)[: len(history)],
            ind.closes(bars_ext)[: len(history)],
        )
        assert baseline == truncated

    def test_trendline_uses_rolling_regression_not_future_pivots(self):
        history = _ramp(count=120)
        baseline = ie.trendline(history, 60)
        extended = history + [1_000.0] * 20
        assert ie.trendline(extended[: len(history)], 60) == baseline

    def test_wrappers_agree_with_the_canonical_engine(self):
        # The single-source rule: wrappers must not re-derive anything.
        closes_ = _ramp(count=200)
        bars = _bars(closes_)
        high, low, close = ind.highs(bars), ind.lows(bars), ind.closes(bars)
        assert ind.cci(bars) == ie.cci(high, low, close)
        assert ind.williams_r(bars) == ie.williams_r(high, low, close)
        assert ind.atr_percent(bars) == ie.atr_percent(high, low, close)
        assert ind.roc_bps(bars, 10) == ie.roc_bps(closes_, 10)
        assert ind.ma_alignment_score(bars) == ie.ma_alignment_score(closes_)
        assert ind.dmi_adx(bars).adx == ie.dmi_adx(high, low, close)[2]
        assert ind.trix(bars).trix == ie.trix(closes_)[0]
        assert ind.ichimoku(bars).kijun == ie.ichimoku(high, low, close)["kijun"]
        assert ind.trendline(bars).r_squared == ie.trendline(closes_, 60)["r_squared"]


class TestHotPathComplexity:
    """The live sweep recomputes these per symbol; O(n^2) here stalls the loop.

    A first cut recomputed the EMA from scratch for every prefix, nested three
    deep inside TRIX. That measured 160ms per symbol at 600 bars and wedged the
    server twice under a live KR session before the cause was found.
    """

    def test_ema_series_matches_recomputing_each_prefix(self):
        values = _ramp(count=200)
        series = ie.ema_series(values, 12)
        for index, end in enumerate(range(12, len(values) + 1)):
            assert abs(series[index] - ie.ema(values[:end], 12)) < 1e-9

    def test_ema_series_edge_cases(self):
        assert ie.ema_series([1.0, 2.0], 5) == []
        assert ie.ema_series([1.0, 2.0], 0) == []

    def test_cost_grows_sub_quadratically(self):
        import time

        def elapsed(count: int) -> float:
            bars = _bars(_ramp(count=count))
            start = time.perf_counter()
            for _ in range(3):
                ie.trix(ind.closes(bars))
                ie.macd(ind.closes(bars))
            return (time.perf_counter() - start) / 3

        small = elapsed(150)
        large = elapsed(600)
        # 4x the data must not cost ~16x the time. Generous bound so the test
        # measures complexity, not machine speed.
        assert large < small * 10, f"small={small:.4f}s large={large:.4f}s"


class TestWrapperContracts:
    def test_results_report_ok_false_instead_of_raising(self):
        short = _bars(_ramp(count=4))
        assert ind.ichimoku(short).ok is False
        assert ind.envelope(short).ok is False
        assert ind.trendline(short).ok is False
        assert ind.dmi_adx(short).ok is False
        assert ind.trix(short).ok is False

    def test_dmi_spread_sign_matches_direction(self):
        up = _bars(_ramp(count=120))
        result = ind.dmi_adx(up)
        assert result.ok
        assert result.dmi_spread > 0

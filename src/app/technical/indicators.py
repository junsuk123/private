"""Technical indicator surface for the short-term prediction layer.

Single-source-of-truth policy: the classic indicators (SMA, EMA, MACD, RSI,
Bollinger Bands, ATR, volume-spike, period return) already live in
``app.features.indicator_engine`` and are used across the system. This module
DELEGATES to them rather than reimplementing the math, exposing:

    * thin re-exports (``sma``, ``ema``, ``rsi``, ``volume_spike_ratio``,
      ``period_return``) so the technical layer has one import surface;
    * typed result wrappers (``macd`` -> :class:`MacdResult`, ``bollinger`` ->
      :class:`BollingerResult`, ``atr`` bar-aware) so downstream signal code
      gets a clean, self-describing API instead of positional tuples;
    * the genuinely-missing indicators this layer needs: ``vwap``,
      ``donchian``, ``rolling_zscore``, ``spread_bps``, ``orderbook_imbalance``.

Conventions (match the rest of the codebase):
    * Inputs are ``Sequence[float]`` (close lists) or ``Sequence[OHLCVBar]`` —
      the repo's typed frame. No pandas (keeps the realtime hot path light and
      identical on the Raspberry Pi CPU-only runtime).
    * No broker calls, no global mutable state, no wall-clock reads.
    * Insufficient/degenerate data returns a NaN-safe ``None`` (scalars) or a
      result object with ``ok=False`` and a ``reason``. Never fabricated values.
    * Callers pass only ``as_of``-visible bars in order; nothing here looks past
      the end of the sequence it is given.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Sequence

from app.features.schemas import OHLCVBar
from app.features.indicator_engine import (
    atr as _atr_hlc,
    atr_expansion as _atr_expansion,
    atr_percent as _atr_percent,
    bollinger_bands as _bollinger_tuple,
    cci as _cci,
    dmi_adx as _dmi_adx,
    ema as _ema,
    envelope as _envelope,
    ichimoku as _ichimoku,
    ma_alignment_score as _ma_alignment_score,
    ma_slope_bps as _ma_slope_bps,
    macd as _macd_tuple,
    momentum as _momentum,
    obv_slope as _obv_slope,
    obv_zscore as _obv_zscore,
    period_return as _period_return,
    price_disparity as _price_disparity,
    relative_volume as _relative_volume,
    roc_bps as _roc_bps,
    rsi as _rsi,
    sma as _sma,
    stochastic_diff as _stochastic_diff,
    trendline as _trendline,
    trix as _trix,
    volume_spike_ratio as _volume_spike_ratio,
    volume_zscore as _volume_zscore,
    williams_r as _williams_r,
)

__all__ = [
    "MacdResult",
    "BollingerResult",
    "DonchianResult",
    "RvgiResult",
    "BoxGeometryResult",
    "SupertrendResult",
    "KeltnerResult",
    "closes",
    "highs",
    "lows",
    "volumes",
    "sma",
    "ema",
    "macd",
    "rsi",
    "bollinger",
    "donchian",
    "atr",
    "vwap",
    "rolling_return",
    "rolling_zscore",
    "volume_spike_ratio",
    "spread_bps",
    "orderbook_imbalance",
    "rvgi",
    "causal_box_geometry",
    "supertrend",
    "keltner_channels",
    "choppiness_index",
]


# --------------------------------------------------------------------------- #
# Result objects for multi-value indicators                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MacdResult:
    macd: float | None
    signal: float | None
    histogram: float | None
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class BollingerResult:
    mid: float | None
    upper: float | None
    lower: float | None
    percent_b: float | None
    bandwidth: float | None
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class DonchianResult:
    high: float | None
    low: float | None
    mid: float | None
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class RvgiResult:
    main: float | None
    signal: float | None
    previous_main: float | None
    previous_signal: float | None
    slope: float | None
    bullish_cross: bool | None
    bearish_cross: bool | None
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class BoxGeometryResult:
    high: float | None
    low: float | None
    mid: float | None
    width: float | None
    width_pct: float | None
    position: float | None
    source_timestamp: object | None
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class SupertrendResult:
    line: float | None
    direction: int | None
    upper_band: float | None
    lower_band: float | None
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class KeltnerResult:
    mid: float | None
    upper: float | None
    lower: float | None
    position: float | None
    bandwidth: float | None
    ok: bool
    reason: str = ""


# --------------------------------------------------------------------------- #
# Bar-field extractors                                                         #
# --------------------------------------------------------------------------- #
def closes(bars: Sequence[OHLCVBar]) -> list[float]:
    return [float(bar.close) for bar in bars]


def highs(bars: Sequence[OHLCVBar]) -> list[float]:
    return [float(bar.high) for bar in bars]


def lows(bars: Sequence[OHLCVBar]) -> list[float]:
    return [float(bar.low) for bar in bars]


def volumes(bars: Sequence[OHLCVBar]) -> list[float]:
    return [float(bar.volume) for bar in bars]


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #
def _finite(values: Sequence[float]) -> list[float]:
    """Return the values as floats, or ``[]`` if any is non-finite/uncastable."""
    out: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(number):
            return []
        out.append(number)
    return out


def _is_pos(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number <= 0 or not math.isfinite(number):
        return None
    return number


# --------------------------------------------------------------------------- #
# Delegated primitives (single source of truth = indicator_engine)             #
# --------------------------------------------------------------------------- #
def sma(values: Sequence[float], period: int) -> float | None:
    data = _finite(values)
    return _sma(data, period) if data else None


def ema(values: Sequence[float], period: int) -> float | None:
    data = _finite(values)
    return _ema(data, period) if data else None


def rsi(values: Sequence[float], period: int = 14) -> float | None:
    """Wilder's RSI (0-100), delegated to ``indicator_engine.rsi``.

    Note: the canonical implementation returns 100.0 for a strictly flat/only-
    gains window (avg_loss == 0). The signal layer treats a near-zero price
    range as neutral rather than trusting an extreme RSI, so we keep the
    canonical numeric behavior here as the single source of truth.
    """
    data = _finite(values)
    return _rsi(data, period) if data else None


def rolling_return(values: Sequence[float], window: int) -> float | None:
    """Return over the last ``window`` steps (delegated to period_return)."""
    if window <= 0:
        return None
    data = _finite(values)
    return _period_return(data, window) if data else None


def volume_spike_ratio(volume_values: Sequence[float], window: int) -> float | None:
    data = _finite(volume_values)
    return _volume_spike_ratio(data, window) if data else None


def macd(
    values: Sequence[float],
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> MacdResult:
    """Typed wrapper over ``indicator_engine.macd``."""
    if not (0 < fast < slow) or signal <= 0:
        return MacdResult(None, None, None, ok=False, reason="invalid_periods")
    data = _finite(values)
    if not data:
        return MacdResult(None, None, None, ok=False, reason="non_finite_data")
    line, signal_value, hist = _macd_tuple(data, fast, slow, signal)
    if line is None or signal_value is None:
        return MacdResult(None, None, None, ok=False, reason="insufficient_data")
    return MacdResult(macd=line, signal=signal_value, histogram=hist, ok=True)


def bollinger(
    values: Sequence[float],
    period: int = 20,
    num_std: float = 2.0,
) -> BollingerResult:
    """Typed wrapper over ``indicator_engine.bollinger_bands``.

    ``percent_b`` is clamped to a neutral 0.5 when the band width is zero (flat
    window) so downstream scoring never divides by / reasons about a zero band.
    """
    if period <= 0 or num_std < 0:
        return BollingerResult(None, None, None, None, None, ok=False, reason="invalid_params")
    data = _finite(values)
    if not data:
        return BollingerResult(None, None, None, None, None, ok=False, reason="non_finite_data")
    mid, upper, lower, width, percent_b = _bollinger_tuple(data, period, num_std)
    if mid is None:
        return BollingerResult(None, None, None, None, None, ok=False, reason="insufficient_data")
    if percent_b is None:  # canonical returns None when upper == lower (flat)
        percent_b = 0.5
    return BollingerResult(
        mid=mid,
        upper=upper,
        lower=lower,
        percent_b=percent_b,
        bandwidth=width,
        ok=True,
    )


def atr(bars: Sequence[OHLCVBar], period: int = 14) -> float | None:
    """Bar-aware ATR wrapper over ``indicator_engine.atr(highs, lows, closes)``."""
    if period <= 0 or len(bars) <= period:
        return None
    high_vals = _finite(highs(bars))
    low_vals = _finite(lows(bars))
    close_vals = _finite(closes(bars))
    if not (high_vals and low_vals and close_vals):
        return None
    return _atr_hlc(high_vals, low_vals, close_vals, period)


# --------------------------------------------------------------------------- #
# New indicators (not present in indicator_engine)                             #
# --------------------------------------------------------------------------- #
def donchian(
    bars_or_highs: Sequence[OHLCVBar] | Sequence[float],
    period: int,
    lows_seq: Sequence[float] | None = None,
) -> DonchianResult:
    """Donchian channel over ``period``.

    Call with a sequence of ``OHLCVBar`` (``donchian(bars, period)``) or with
    explicit highs and lows (``donchian(highs, period, lows)``).
    """
    if period <= 0:
        return DonchianResult(None, None, None, ok=False, reason="invalid_period")
    if lows_seq is None:
        bars = list(bars_or_highs)  # type: ignore[arg-type]
        if not bars or not isinstance(bars[0], OHLCVBar):
            return DonchianResult(None, None, None, ok=False, reason="insufficient_data")
        high_vals = _finite(highs(bars))  # type: ignore[arg-type]
        low_vals = _finite(lows(bars))  # type: ignore[arg-type]
    else:
        high_vals = _finite(bars_or_highs)  # type: ignore[arg-type]
        low_vals = _finite(lows_seq)
    if len(high_vals) < period or len(low_vals) < period:
        return DonchianResult(None, None, None, ok=False, reason="insufficient_data")
    high = max(high_vals[-period:])
    low = min(low_vals[-period:])
    return DonchianResult(high=high, low=low, mid=(high + low) / 2.0, ok=True)


def vwap(bars: Sequence[OHLCVBar], window: int | None = None) -> float | None:
    """Volume-weighted average price over the last ``window`` bars.

    ``window=None`` uses all supplied bars (typically the session's visible
    bars). Typical price = (high + low + close) / 3. Returns ``None`` when total
    volume over the window is zero or any field is missing/negative.
    """
    if not bars:
        return None
    selected = list(bars[-window:]) if window else list(bars)
    if not selected:
        return None
    total_pv = 0.0
    total_volume = 0.0
    for bar in selected:
        high, low, close, volume = (
            float(bar.high),
            float(bar.low),
            float(bar.close),
            float(bar.volume),
        )
        if not all(math.isfinite(v) for v in (high, low, close, volume)) or volume < 0:
            return None
        typical = (high + low + close) / 3.0
        total_pv += typical * volume
        total_volume += volume
    if total_volume <= 0:
        return None
    return total_pv / total_volume


def rolling_zscore(values: Sequence[float], window: int) -> float | None:
    """Z-score of the latest value against the prior ``window`` values."""
    if window <= 1:
        return None
    data = _finite(values)
    if len(data) < window + 1:
        return None
    baseline = data[-window - 1 : -1]
    deviation = pstdev(baseline)
    if deviation == 0:
        return None
    return (data[-1] - fmean(baseline)) / deviation


def spread_bps(best_bid: object, best_ask: object) -> float | None:
    """Bid/ask spread in basis points of the midpoint. ``None`` if invalid."""
    bid = _is_pos(best_bid)
    ask = _is_pos(best_ask)
    if bid is None or ask is None or ask < bid:
        return None
    midpoint = (ask + bid) / 2.0
    if midpoint <= 0:
        return None
    return (ask - bid) / midpoint * 10_000.0


def orderbook_imbalance(bid_size: object, ask_size: object) -> float | None:
    """Order-flow imbalance in [-1, 1]: (bid - ask) / (bid + ask).

    Positive = more resting bid depth (buy pressure). ``None`` if both sides
    are missing/zero/negative.
    """
    try:
        bid = float(bid_size)
        ask = float(ask_size)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(bid) or not math.isfinite(ask) or bid < 0 or ask < 0:
        return None
    total = bid + ask
    if total <= 0:
        return None
    return (bid - ask) / total


def _true_ranges(bars: Sequence[OHLCVBar]) -> list[float]:
    ordered = tuple(bars)
    if not ordered:
        return []
    result: list[float] = []
    previous_close: float | None = None
    for bar in ordered:
        high, low, close = float(bar.high), float(bar.low), float(bar.close)
        if (
            not all(math.isfinite(value) for value in (high, low, close))
            or high < low
        ):
            return []
        true_range = high - low
        if previous_close is not None:
            true_range = max(
                true_range,
                abs(high - previous_close),
                abs(low - previous_close),
            )
        result.append(true_range)
        previous_close = close
    return result


def _wilder_rma_series(values: Sequence[float], period: int) -> list[float | None]:
    if period <= 0 or len(values) < period:
        return [None] * len(values)
    out: list[float | None] = [None] * len(values)
    current = fmean(values[:period])
    out[period - 1] = current
    for index in range(period, len(values)):
        current = (current * (period - 1) + float(values[index])) / period
        out[index] = current
    return out


def supertrend(
    bars: Sequence[OHLCVBar], period: int = 10, multiplier: float = 3.0
) -> SupertrendResult:
    """TradingView-compatible ATR Supertrend over completed bars."""
    ordered = tuple(bars)
    if period <= 0 or multiplier <= 0 or len(ordered) < period + 1:
        return SupertrendResult(None, None, None, None, False, "insufficient_data")
    true_ranges = _true_ranges(ordered)
    if not true_ranges:
        return SupertrendResult(None, None, None, None, False, "invalid_data")
    atr_values = _wilder_rma_series(true_ranges, period)
    final_upper: float | None = None
    final_lower: float | None = None
    line: float | None = None
    direction = -1
    previous_close: float | None = None
    for index, bar in enumerate(ordered):
        atr_value = atr_values[index]
        close = float(bar.close)
        if atr_value is None:
            previous_close = close
            continue
        midpoint = (float(bar.high) + float(bar.low)) / 2.0
        basic_upper = midpoint + multiplier * atr_value
        basic_lower = midpoint - multiplier * atr_value
        prior_upper, prior_lower, prior_line = final_upper, final_lower, line
        final_upper = (
            basic_upper
            if prior_upper is None
            or basic_upper < prior_upper
            or (previous_close is not None and previous_close > prior_upper)
            else prior_upper
        )
        final_lower = (
            basic_lower
            if prior_lower is None
            or basic_lower > prior_lower
            or (previous_close is not None and previous_close < prior_lower)
            else prior_lower
        )
        if prior_line is None or prior_upper is None:
            direction = -1
        elif abs(prior_line - prior_upper) <= 1e-12:
            direction = 1 if close > final_upper else -1
        else:
            direction = -1 if close < final_lower else 1
        line = final_lower if direction > 0 else final_upper
        previous_close = close
    if line is None or final_upper is None or final_lower is None:
        return SupertrendResult(None, None, None, None, False, "insufficient_data")
    return SupertrendResult(line, direction, final_upper, final_lower, True)


def keltner_channels(
    bars: Sequence[OHLCVBar],
    period: int = 20,
    atr_period: int = 14,
    multiplier: float = 2.0,
) -> KeltnerResult:
    """EMA centre with Wilder-ATR envelopes, using completed bars only."""
    ordered = tuple(bars)
    if period <= 0 or atr_period <= 0 or multiplier <= 0:
        return KeltnerResult(None, None, None, None, None, False, "invalid_params")
    mid = ema(closes(ordered), period)
    atr_value = atr(ordered, atr_period)
    if mid is None or atr_value is None or not ordered:
        return KeltnerResult(None, None, None, None, None, False, "insufficient_data")
    upper = mid + multiplier * atr_value
    lower = mid - multiplier * atr_value
    width = upper - lower
    if width <= 0 or mid <= 0:
        return KeltnerResult(mid, upper, lower, None, None, False, "invalid_geometry")
    position = (float(ordered[-1].close) - lower) / width
    return KeltnerResult(mid, upper, lower, position, width / mid, True)


def choppiness_index(bars: Sequence[OHLCVBar], period: int = 14) -> float | None:
    """E.W. Dreiss CHOP in [0, 100]; direction-neutral regime descriptor."""
    ordered = tuple(bars)
    if period <= 1 or len(ordered) < period + 1:
        return None
    true_ranges = _true_ranges(ordered)
    if not true_ranges:
        return None
    window = ordered[-period:]
    price_range = max(float(bar.high) for bar in window) - min(
        float(bar.low) for bar in window
    )
    total_range = sum(true_ranges[-period:])
    if price_range <= 0 or total_range <= 0:
        return None
    value = 100.0 * math.log10(total_range / price_range) / math.log10(period)
    return max(0.0, min(100.0, value))


def rvgi(
    bars: Sequence[OHLCVBar],
    period: int = 10,
    *,
    epsilon: float = 1e-12,
) -> RvgiResult:
    """Causal Relative Vigor Index over completed bars only.

    The caller owns completion filtering.  A signal needs four consecutive RVGI
    values, while each RVGI value needs ``period`` four-bar weighted values.
    """
    if period <= 0 or epsilon <= 0:
        return RvgiResult(None, None, None, None, None, None, None, False, "invalid_params")
    ordered = tuple(bars)
    required = period + 6
    if len(ordered) < required:
        return RvgiResult(None, None, None, None, None, None, None, False, "insufficient_data")
    bodies: list[float] = []
    ranges: list[float] = []
    for bar in ordered:
        values = (bar.open, bar.high, bar.low, bar.close)
        if not all(math.isfinite(float(value)) for value in values):
            return RvgiResult(None, None, None, None, None, None, None, False, "non_finite_data")
        bodies.append(float(bar.close) - float(bar.open))
        ranges.append(max(0.0, float(bar.high) - float(bar.low)))
    weighted_body: list[float] = []
    weighted_range: list[float] = []
    for index in range(3, len(ordered)):
        weighted_body.append(
            (bodies[index] + 2 * bodies[index - 1] + 2 * bodies[index - 2] + bodies[index - 3])
            / 6.0
        )
        weighted_range.append(
            (ranges[index] + 2 * ranges[index - 1] + 2 * ranges[index - 2] + ranges[index - 3])
            / 6.0
        )
    rvgi_values: list[float] = []
    for index in range(period - 1, len(weighted_body)):
        numerator = fmean(weighted_body[index - period + 1 : index + 1])
        denominator = fmean(weighted_range[index - period + 1 : index + 1])
        # A zero-range history has no directional information.  Dividing by
        # epsilon is safe numerically, but it must not manufacture a signal.
        rvgi_values.append(0.0 if denominator <= epsilon else numerator / max(denominator, epsilon))
    if len(rvgi_values) < 4:
        return RvgiResult(None, None, None, None, None, None, None, False, "insufficient_data")
    signals = [
        (rvgi_values[index] + 2 * rvgi_values[index - 1] + 2 * rvgi_values[index - 2] + rvgi_values[index - 3])
        / 6.0
        for index in range(3, len(rvgi_values))
    ]
    main = rvgi_values[-1]
    previous_main = rvgi_values[-2]
    signal = signals[-1]
    previous_signal = signals[-2] if len(signals) >= 2 else None
    bullish = (
        previous_signal is not None
        and main > signal
        and previous_main <= previous_signal
    )
    bearish = (
        previous_signal is not None
        and main < signal
        and previous_main >= previous_signal
    )
    return RvgiResult(
        main,
        signal,
        previous_main,
        previous_signal,
        main - previous_main,
        bullish,
        bearish,
        True,
    )


def causal_box_geometry(
    bars: Sequence[OHLCVBar],
    lookback: int = 20,
    *,
    epsilon: float = 1e-12,
) -> BoxGeometryResult:
    """Box ending immediately before the final (signal) completed bar."""
    ordered = tuple(bars)
    if lookback <= 0 or len(ordered) < lookback + 1:
        return BoxGeometryResult(None, None, None, None, None, None, None, False, "insufficient_data")
    history = ordered[-lookback - 1 : -1]
    high_values = _finite([bar.high for bar in history])
    low_values = _finite([bar.low for bar in history])
    if len(high_values) != lookback or len(low_values) != lookback:
        return BoxGeometryResult(None, None, None, None, None, None, None, False, "non_finite_data")
    high = max(high_values)
    low = min(low_values)
    width = high - low
    mid = (high + low) / 2.0
    if mid <= 0 or width <= epsilon:
        return BoxGeometryResult(high, low, mid, width, None, None, history[-1].as_of, False, "invalid_geometry")
    close = float(ordered[-1].close)
    position = max(0.0, min(1.0, (close - low) / max(width, epsilon)))
    return BoxGeometryResult(
        high,
        low,
        mid,
        width,
        width / mid,
        position,
        history[-1].as_of,
        True,
    )


# --------------------------------------------------------------------------- #
# Bar-oriented wrappers for the remaining indicator families.                  #
#                                                                              #
# Formulas live in ``indicator_engine`` and are NOT duplicated here; these only #
# unpack OHLCV bars and keep the ``None == unavailable`` contract.              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IchimokuResult:
    tenkan: float | None
    kijun: float | None
    senkou_a: float | None
    senkou_b: float | None
    tenkan_kijun_gap_bps: float | None
    price_vs_cloud: float | None
    cloud_thickness_bps: float | None
    cloud_direction: float | None
    ok: bool = False
    reason: str = ""


@dataclass(frozen=True)
class EnvelopeResult:
    mid: float | None
    upper: float | None
    lower: float | None
    position: float | None
    distance_to_upper_bps: float | None
    distance_to_lower_bps: float | None
    ok: bool = False
    reason: str = ""


@dataclass(frozen=True)
class TrendlineResult:
    slope_bps_per_bar: float | None
    r_squared: float | None
    residual_zscore: float | None
    support_residual_quantile: float | None
    resistance_residual_quantile: float | None
    ok: bool = False
    reason: str = ""


@dataclass(frozen=True)
class DmiResult:
    plus_di: float | None
    minus_di: float | None
    adx: float | None
    #: plus_di - minus_di. Sign is direction, magnitude is conviction.
    dmi_spread: float | None
    ok: bool = False
    reason: str = ""


@dataclass(frozen=True)
class TrixResult:
    trix: float | None
    signal: float | None
    histogram: float | None
    ok: bool = False
    reason: str = ""


def ichimoku(bars: Sequence[OHLCVBar], **kwargs: int) -> IchimokuResult:
    values = _ichimoku(highs(bars), lows(bars), closes(bars), **kwargs)
    ok = values.get("kijun") is not None
    return IchimokuResult(
        **values, ok=ok, reason="" if ok else "insufficient_data"
    )


def envelope(
    bars: Sequence[OHLCVBar],
    period: int = 20,
    percentage: float = 0.02,
    use_ema: bool = False,
) -> EnvelopeResult:
    mid, upper, lower, position, to_upper, to_lower = _envelope(
        closes(bars), period, percentage, use_ema
    )
    ok = mid is not None
    return EnvelopeResult(
        mid, upper, lower, position, to_upper, to_lower,
        ok=ok, reason="" if ok else "insufficient_data",
    )


def trendline(bars: Sequence[OHLCVBar], window: int = 60) -> TrendlineResult:
    values = _trendline(closes(bars), window)
    ok = values.get("slope_bps_per_bar") is not None
    return TrendlineResult(
        **values, ok=ok, reason="" if ok else "insufficient_data"
    )


def dmi_adx(bars: Sequence[OHLCVBar], period: int = 14) -> DmiResult:
    plus_di, minus_di, adx = _dmi_adx(highs(bars), lows(bars), closes(bars), period)
    spread = (
        plus_di - minus_di if plus_di is not None and minus_di is not None else None
    )
    ok = adx is not None
    return DmiResult(
        plus_di, minus_di, adx, spread, ok=ok, reason="" if ok else "insufficient_data"
    )


def trix(bars: Sequence[OHLCVBar], period: int = 15, signal_period: int = 9) -> TrixResult:
    line, signal_value, histogram = _trix(closes(bars), period, signal_period)
    ok = line is not None
    return TrixResult(
        line, signal_value, histogram, ok=ok, reason="" if ok else "insufficient_data"
    )


def cci(bars: Sequence[OHLCVBar], period: int = 20) -> float | None:
    return _cci(highs(bars), lows(bars), closes(bars), period)


def williams_r(bars: Sequence[OHLCVBar], period: int = 14) -> float | None:
    return _williams_r(highs(bars), lows(bars), closes(bars), period)


def stochastic_diff(
    bars: Sequence[OHLCVBar], period: int = 14, signal: int = 3
) -> float | None:
    return _stochastic_diff(highs(bars), lows(bars), closes(bars), period, signal)


def atr_percent(bars: Sequence[OHLCVBar], period: int = 14) -> float | None:
    return _atr_percent(highs(bars), lows(bars), closes(bars), period)


def atr_expansion(bars: Sequence[OHLCVBar], period: int = 14) -> float | None:
    return _atr_expansion(highs(bars), lows(bars), closes(bars), period)


def momentum(bars: Sequence[OHLCVBar], period: int = 10) -> float | None:
    return _momentum(closes(bars), period)


def roc_bps(bars: Sequence[OHLCVBar], period: int = 10) -> float | None:
    return _roc_bps(closes(bars), period)


def ma_slope_bps(bars: Sequence[OHLCVBar], period: int, lookback: int = 1) -> float | None:
    return _ma_slope_bps(closes(bars), period, lookback)


def ma_alignment_score(
    bars: Sequence[OHLCVBar], windows: tuple[int, ...] = (5, 20, 60)
) -> float | None:
    return _ma_alignment_score(closes(bars), windows)


def price_disparity(bars: Sequence[OHLCVBar], period: int = 20) -> float | None:
    return _price_disparity(closes(bars), period)


def obv_slope(bars: Sequence[OHLCVBar], window: int = 20) -> float | None:
    return _obv_slope(closes(bars), volumes(bars), window)


def obv_zscore(bars: Sequence[OHLCVBar], window: int = 20) -> float | None:
    return _obv_zscore(closes(bars), volumes(bars), window)


def volume_zscore(bars: Sequence[OHLCVBar], window: int = 20) -> float | None:
    return _volume_zscore(volumes(bars), window)


def relative_volume(bars: Sequence[OHLCVBar], window: int = 20) -> float | None:
    return _relative_volume(volumes(bars), window)

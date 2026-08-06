from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, pstdev

from app.features.schemas import OHLCVBar, RawIndicatorRecord

CALCULATION_VERSION = "semantic-indicators-v1"


@dataclass(frozen=True)
class IndicatorEngineConfig:
    return_windows: tuple[int, ...] = (1, 5, 20, 60, 120)
    sma_windows: tuple[int, ...] = (20, 60)
    ema_windows: tuple[int, ...] = (12, 20, 26)
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_stddev: float = 2.0
    atr_period: int = 14
    stochastic_period: int = 14
    stochastic_signal: int = 3
    mfi_period: int = 14
    volume_window: int = 20
    source: str = "ohlcv"


class IndicatorEngine:
    def __init__(self, config: IndicatorEngineConfig | None = None) -> None:
        self.config = config or IndicatorEngineConfig()

    def calculate(self, bars: tuple[OHLCVBar, ...], as_of: datetime | None = None) -> tuple[RawIndicatorRecord, ...]:
        ordered = tuple(sorted(bars, key=lambda bar: bar.as_of))
        if as_of is not None:
            ordered = tuple(bar for bar in ordered if bar.as_of <= as_of)
        if not ordered:
            return ()

        ticker = ordered[-1].ticker
        snapshot_time = ordered[-1].as_of
        closes = [bar.close for bar in ordered]
        highs = [bar.high for bar in ordered]
        lows = [bar.low for bar in ordered]
        opens = [bar.open for bar in ordered]
        volumes = [bar.volume for bar in ordered]
        records: list[RawIndicatorRecord] = []

        for window in self.config.return_windows:
            value = period_return(closes, window)
            records.append(self._record(ticker, snapshot_time, f"return_{window}d", value, "ratio", f"{window} bars", {"window": window}))

        records.extend(
            [
                self._record(ticker, snapshot_time, "close_location_value", close_location_value(highs, lows, closes), "ratio", "1 bar", {"window": 1}),
                self._record(ticker, snapshot_time, "gap_up_ratio", gap_ratio(opens, closes, direction="up"), "ratio", "1 bar", {"direction": "up"}),
                self._record(ticker, snapshot_time, "gap_down_ratio", gap_ratio(opens, closes, direction="down"), "ratio", "1 bar", {"direction": "down"}),
                self._record(ticker, snapshot_time, "distance_from_52w_high", distance_from_extreme(closes, 252, "high"), "ratio", "252 bars", {"window": 252, "extreme": "high"}),
                self._record(ticker, snapshot_time, "distance_from_52w_low", distance_from_extreme(closes, 252, "low"), "ratio", "252 bars", {"window": 252, "extreme": "low"}),
                self._record(ticker, snapshot_time, "rolling_drawdown_20d", rolling_drawdown(closes, 20), "ratio", "20 bars", {"window": 20}),
                self._record(ticker, snapshot_time, "intraday_range_ratio", intraday_range_ratio(highs, lows, closes), "ratio", "1 bar", {"window": 1}),
                self._record(ticker, snapshot_time, "candle_body_ratio", candle_body_ratio(opens, highs, lows, closes), "ratio", "1 bar", {"window": 1}),
                self._record(ticker, snapshot_time, "upper_shadow_ratio", shadow_ratio(opens, highs, lows, closes, "upper"), "ratio", "1 bar", {"side": "upper"}),
                self._record(ticker, snapshot_time, "lower_shadow_ratio", shadow_ratio(opens, highs, lows, closes, "lower"), "ratio", "1 bar", {"side": "lower"}),
            ]
        )

        for window in self.config.sma_windows:
            records.append(self._record(ticker, snapshot_time, f"sma_{window}", sma(closes, window), "price", f"{window} bars", {"window": window}))
        for window in self.config.ema_windows:
            records.append(self._record(ticker, snapshot_time, f"ema_{window}", ema(closes, window), "price", f"{window} bars", {"window": window}))

        macd_line, signal_line, histogram = macd(
            closes, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal
        )
        records.extend(
            [
                self._record(ticker, snapshot_time, "macd_line", macd_line, "price", f"{self.config.macd_fast}/{self.config.macd_slow} EMA", {"fast": self.config.macd_fast, "slow": self.config.macd_slow, "signal": self.config.macd_signal}),
                self._record(ticker, snapshot_time, "macd_signal", signal_line, "price", f"{self.config.macd_signal} EMA of MACD", {"fast": self.config.macd_fast, "slow": self.config.macd_slow, "signal": self.config.macd_signal}),
                self._record(ticker, snapshot_time, "macd_histogram", histogram, "price", "MACD-signal", {"fast": self.config.macd_fast, "slow": self.config.macd_slow, "signal": self.config.macd_signal}),
                self._record(ticker, snapshot_time, "rsi_14", rsi(closes, self.config.rsi_period), "index", f"{self.config.rsi_period} bars", {"period": self.config.rsi_period, "canonical_name": "rsi_14"}),
            ]
        )

        middle, upper, lower, width, percent_b = bollinger_bands(
            closes, self.config.bollinger_period, self.config.bollinger_stddev
        )
        records.extend(
            [
                self._record(ticker, snapshot_time, "bollinger_middle_20", middle, "price", f"{self.config.bollinger_period} bars", {"period": self.config.bollinger_period, "stddevs": self.config.bollinger_stddev, "canonical_name": "bollinger_middle_20"}),
                self._record(ticker, snapshot_time, "bollinger_upper_20", upper, "price", f"{self.config.bollinger_period} bars", {"period": self.config.bollinger_period, "stddevs": self.config.bollinger_stddev, "canonical_name": "bollinger_upper_20"}),
                self._record(ticker, snapshot_time, "bollinger_lower_20", lower, "price", f"{self.config.bollinger_period} bars", {"period": self.config.bollinger_period, "stddevs": self.config.bollinger_stddev, "canonical_name": "bollinger_lower_20"}),
                self._record(ticker, snapshot_time, "bollinger_band_width_20", width, "ratio", f"{self.config.bollinger_period} bars", {"period": self.config.bollinger_period, "stddevs": self.config.bollinger_stddev, "canonical_name": "bollinger_band_width_20"}),
                self._record(ticker, snapshot_time, "bollinger_percent_b_20", percent_b, "ratio", f"{self.config.bollinger_period} bars", {"period": self.config.bollinger_period, "stddevs": self.config.bollinger_stddev, "canonical_name": "bollinger_percent_b_20"}),
            ]
        )

        stoch_k = stochastic_k(highs, lows, closes, self.config.stochastic_period)
        stoch_d = stochastic_d(highs, lows, closes, self.config.stochastic_period, self.config.stochastic_signal)
        records.extend(
            [
                self._record(ticker, snapshot_time, "atr_14", atr(highs, lows, closes, self.config.atr_period), "price", f"{self.config.atr_period} bars", {"period": self.config.atr_period, "canonical_name": "atr_14"}),
                self._record(ticker, snapshot_time, "historical_volatility_20d", historical_volatility(closes, 20), "annualized_ratio", "20 bars", {"window": 20, "annualization": 252}),
                self._record(ticker, snapshot_time, "obv", obv(closes, volumes), "volume", None, {"method": "cumulative"}),
                self._record(ticker, snapshot_time, "volume_ma_20", sma(volumes, self.config.volume_window), "volume", f"{self.config.volume_window} bars", {"window": self.config.volume_window, "canonical_name": "volume_ma_20"}),
                self._record(ticker, snapshot_time, "volume_spike_ratio", volume_spike_ratio(volumes, self.config.volume_window), "ratio", f"{self.config.volume_window} bars", {"window": self.config.volume_window}),
                self._record(ticker, snapshot_time, "stochastic_k_14", stoch_k, "index", f"{self.config.stochastic_period} bars", {"period": self.config.stochastic_period, "canonical_name": "stochastic_k_14"}),
                self._record(ticker, snapshot_time, "stochastic_d_3", stoch_d, "index", f"{self.config.stochastic_signal} bars", {"period": self.config.stochastic_period, "signal": self.config.stochastic_signal, "canonical_name": "stochastic_d_3"}),
                self._record(ticker, snapshot_time, "mfi_14", mfi(highs, lows, closes, volumes, self.config.mfi_period), "index", f"{self.config.mfi_period} bars", {"period": self.config.mfi_period, "canonical_name": "mfi_14"}),
            ]
        )
        return tuple(records)

    def _record(
        self,
        ticker: str,
        as_of: datetime,
        name: str,
        value: float | None,
        unit: str,
        lookback: str | None,
        parameters: dict[str, float | int | str] | None = None,
    ) -> RawIndicatorRecord:
        return RawIndicatorRecord(
            ticker=ticker,
            as_of=as_of,
            indicator_name=name,
            value=None if value is None or not math.isfinite(value) else round(value, 10),
            unit=unit,
            lookback_window=lookback,
            source=self.config.source,
            calculation_version=CALCULATION_VERSION,
            calculation_method="formula",
            metadata={"parameters": parameters or {}},
        )


def period_return(values: list[float], window: int) -> float | None:
    if len(values) <= window or values[-window - 1] == 0:
        return None
    return values[-1] / values[-window - 1] - 1


def sma(values: list[float], window: int) -> float | None:
    if len(values) < window or window <= 0:
        return None
    return mean(values[-window:])


def ema(values: list[float], window: int) -> float | None:
    if len(values) < window or window <= 0:
        return None
    alpha = 2 / (window + 1)
    current = mean(values[:window])
    for value in values[window:]:
        current = alpha * value + (1 - alpha) * current
    return current


def ema_series(values: list[float], window: int) -> list[float]:
    """EMA at every point, in ONE pass.

    Exactly equivalent to calling :func:`ema` on each growing prefix, because the
    EMA is a recurrence: the seed is ``mean(values[:window])`` and every later
    value rolls it forward once. Recomputing each prefix from scratch is O(n^2),
    which is invisible on 60 bars and quadratic pain on 600 -- TRIX nests it three
    deep, and on the live path that measured 160ms per symbol per sweep and
    saturated the trading loop.
    """
    if window <= 0 or len(values) < window:
        return []
    alpha = 2 / (window + 1)
    current = mean(values[:window])
    out = [current]
    for value in values[window:]:
        current = alpha * value + (1 - alpha) * current
        out.append(current)
    return out


def macd(values: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float | None, float | None, float | None]:
    if len(values) < slow + signal - 1:
        return None, None, None
    fast_series = ema_series(values, fast)
    slow_series = ema_series(values, slow)
    if not fast_series or not slow_series:
        return None, None, None
    # Align on the slower series: index i of slow_series is prefix end slow+i,
    # which is fast_series index (slow - fast) + i.
    offset = slow - fast
    macd_series = [
        fast_series[offset + index] - slow_value
        for index, slow_value in enumerate(slow_series)
        if 0 <= offset + index < len(fast_series)
    ]
    signal_value = ema(macd_series, signal)
    line = macd_series[-1] if macd_series else None
    hist = line - signal_value if line is not None and signal_value is not None else None
    return line, signal_value, hist


def rsi(values: list[float], period: int = 14) -> float | None:
    if len(values) <= period:
        return None
    deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    gains = [max(0.0, delta) for delta in deltas]
    losses = [max(0.0, -delta) for delta in deltas]
    avg_gain = mean(gains[:period])
    avg_loss = mean(losses[:period])
    for gain, loss in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def bollinger_bands(values: list[float], period: int = 20, stddevs: float = 2.0) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    middle = sma(values, period)
    if middle is None:
        return None, None, None, None, None
    deviation = pstdev(values[-period:])
    upper = middle + stddevs * deviation
    lower = middle - stddevs * deviation
    width = (upper - lower) / middle if middle else None
    percent_b = (values[-1] - lower) / (upper - lower) if upper != lower else None
    return middle, upper, lower, width, percent_b


def atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float | None:
    if len(closes) <= period:
        return None
    true_ranges = []
    for i in range(1, len(closes)):
        true_ranges.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
    current = mean(true_ranges[:period])
    for value in true_ranges[period:]:
        current = (current * (period - 1) + value) / period
    return current


def historical_volatility(values: list[float], window: int = 20, annualization: int = 252) -> float | None:
    if len(values) <= window:
        return None
    returns = [values[i] / values[i - 1] - 1 for i in range(len(values) - window, len(values)) if values[i - 1]]
    if len(returns) < 2:
        return None
    return pstdev(returns) * math.sqrt(annualization)


def obv(closes: list[float], volumes: list[float]) -> float | None:
    if not closes or len(closes) != len(volumes):
        return None
    total = 0.0
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            total += volumes[i]
        elif closes[i] < closes[i - 1]:
            total -= volumes[i]
    return total


def stochastic_k(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period:
        return None
    high = max(highs[-period:])
    low = min(lows[-period:])
    if high == low:
        return 50.0
    return ((closes[-1] - low) / (high - low)) * 100


def stochastic_d(highs: list[float], lows: list[float], closes: list[float], period: int = 14, signal: int = 3) -> float | None:
    if len(closes) < period + signal - 1:
        return None
    values = [stochastic_k(highs[:end], lows[:end], closes[:end], period) for end in range(len(closes) - signal + 1, len(closes) + 1)]
    numeric = [value for value in values if value is not None]
    return mean(numeric) if len(numeric) == signal else None


def mfi(highs: list[float], lows: list[float], closes: list[float], volumes: list[float], period: int = 14) -> float | None:
    if len(closes) <= period:
        return None
    typical = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(closes))]
    positive = 0.0
    negative = 0.0
    for i in range(len(closes) - period, len(closes)):
        flow = typical[i] * volumes[i]
        if typical[i] > typical[i - 1]:
            positive += flow
        elif typical[i] < typical[i - 1]:
            negative += flow
    if negative == 0:
        return 100.0
    return 100 - (100 / (1 + positive / negative))


def close_location_value(highs: list[float], lows: list[float], closes: list[float]) -> float | None:
    if not closes or highs[-1] == lows[-1]:
        return None
    return ((closes[-1] - lows[-1]) - (highs[-1] - closes[-1])) / (highs[-1] - lows[-1])


def gap_ratio(opens: list[float], closes: list[float], direction: str) -> float | None:
    if len(opens) < 2 or closes[-2] == 0:
        return None
    gap = opens[-1] / closes[-2] - 1
    return max(0.0, gap) if direction == "up" else max(0.0, -gap)


def distance_from_extreme(values: list[float], window: int, kind: str) -> float | None:
    if not values:
        return None
    sample = values[-min(window, len(values)) :]
    extreme = max(sample) if kind == "high" else min(sample)
    if extreme == 0:
        return None
    return values[-1] / extreme - 1


def rolling_drawdown(values: list[float], window: int) -> float | None:
    if not values:
        return None
    sample = values[-min(window, len(values)) :]
    peak = max(sample)
    return values[-1] / peak - 1 if peak else None


def intraday_range_ratio(highs: list[float], lows: list[float], closes: list[float]) -> float | None:
    if not closes or closes[-1] == 0:
        return None
    return (highs[-1] - lows[-1]) / closes[-1]


def candle_body_ratio(opens: list[float], highs: list[float], lows: list[float], closes: list[float]) -> float | None:
    spread = highs[-1] - lows[-1] if highs else 0
    return abs(closes[-1] - opens[-1]) / spread if spread else None


def shadow_ratio(opens: list[float], highs: list[float], lows: list[float], closes: list[float], side: str) -> float | None:
    spread = highs[-1] - lows[-1] if highs else 0
    if not spread:
        return None
    upper = highs[-1] - max(opens[-1], closes[-1])
    lower = min(opens[-1], closes[-1]) - lows[-1]
    return (upper if side == "upper" else lower) / spread


def volume_spike_ratio(volumes: list[float], window: int = 20) -> float | None:
    baseline = sma(volumes[:-1], window) if len(volumes) > window else None
    if baseline is None or baseline == 0:
        return None
    return volumes[-1] / baseline


# ---------------------------------------------------------------------------
# Canonical formulas for the remaining indicator families.
#
# This module is the SINGLE SOURCE for standard technical formulas.
# ``app.technical.indicators`` provides typed, bar-oriented wrappers over these
# and must not re-implement any of them.
#
# Contract for everything below:
#   * insufficient data -> None. Never a fabricated neutral value: a neutral
#     number and "not computable" are different facts, and only the caller can
#     decide which one matters.
#   * non-finite input is treated as missing rather than propagated.
#   * flat series, zero volume, zero range and malformed OHLC return None instead
#     of dividing by zero.
#   * nothing here reads past the last element, so every value is computable at
#     decision_time from data at or before it.
# ---------------------------------------------------------------------------


def _clean(values: list[float] | tuple[float, ...]) -> list[float]:
    """Finite floats only. A NaN inside a window is missing data, not zero."""
    out: list[float] = []
    for value in values or ():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def _ohlc_ok(highs: list[float], lows: list[float], closes: list[float]) -> bool:
    """Reject malformed bars instead of producing a negative true range."""
    if not (len(highs) == len(lows) == len(closes)) or not closes:
        return False
    return all(
        high >= low and high >= close >= low
        for high, low, close in zip(highs, lows, closes)
    )


def _bps(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return (numerator / denominator) * 10_000.0


# --- Moving-average structure ------------------------------------------------

def ma_slope_bps(values: list[float], window: int, lookback: int = 1) -> float | None:
    """Slope of an SMA in bps per bar, measured over ``lookback`` bars."""
    series = _clean(values)
    if lookback <= 0 or window <= 0 or len(series) < window + lookback:
        return None
    now = sma(series, window)
    then = sma(series[: len(series) - lookback], window)
    if now is None or then is None or then == 0:
        return None
    return _bps(now - then, then * lookback)


def ma_alignment_score(
    values: list[float], windows: tuple[int, ...] = (5, 20, 60)
) -> float | None:
    """+1 when every faster MA sits above every slower one, -1 when fully inverted.

    Fraction of ordered MA pairs in bullish order, mapped to [-1, 1]. Partial
    alignment scores partially instead of being forced into a binary.
    """
    series = _clean(values)
    ordered = sorted({int(w) for w in windows if int(w) > 0})
    if len(ordered) < 2 or len(series) < max(ordered):
        return None
    averages: list[float] = []
    for window in ordered:
        value = sma(series, window)
        if value is None:
            return None
        averages.append(value)
    pairs = bullish = 0
    for faster in range(len(averages)):
        for slower in range(faster + 1, len(averages)):
            pairs += 1
            if averages[faster] > averages[slower]:
                bullish += 1
    if not pairs:
        return None
    return (2.0 * bullish / pairs) - 1.0


def price_disparity(values: list[float], window: int = 20) -> float | None:
    """이격도: price relative to its own moving average, in bps."""
    series = _clean(values)
    average = sma(series, window)
    if average is None or average == 0:
        return None
    return _bps(series[-1] - average, average)


# --- Envelope ----------------------------------------------------------------

def envelope(
    values: list[float],
    window: int = 20,
    percentage: float = 0.02,
    use_ema: bool = False,
) -> tuple[
    float | None, float | None, float | None, float | None, float | None, float | None
]:
    """(middle, upper, lower, position, distance_to_upper_bps, distance_to_lower_bps).

    ``position`` is 0 at the lower band and 1 at the upper band and is NOT clamped,
    so a price outside the envelope reports <0 or >1 rather than saturating and
    hiding the excursion.
    """
    series = _clean(values)
    if percentage <= 0 or not series:
        return (None,) * 6
    middle = ema(series, window) if use_ema else sma(series, window)
    if middle is None or middle <= 0:
        return (None,) * 6
    upper = middle * (1.0 + percentage)
    lower = middle * (1.0 - percentage)
    price = series[-1]
    span = upper - lower
    return (
        middle,
        upper,
        lower,
        ((price - lower) / span) if span else None,
        _bps(upper - price, price),
        _bps(price - lower, price),
    )


# --- Ichimoku (causal only) --------------------------------------------------

def ichimoku(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
) -> dict[str, float | None]:
    """Causal Ichimoku. The chart's forward-shifted cloud is deliberately absent.

    Senkou A/B are normally PLOTTED 26 bars into the future. Feeding that shifted
    series to a model at decision_time would be look-ahead. Here they are the
    values computed from data up to now (the cloud that will later be plotted
    ahead), and ``price_vs_cloud`` compares price against those current values.
    """
    high = _clean(highs)
    low = _clean(lows)
    close = _clean(closes)
    empty: dict[str, float | None] = {
        "tenkan": None,
        "kijun": None,
        "senkou_a": None,
        "senkou_b": None,
        "tenkan_kijun_gap_bps": None,
        "price_vs_cloud": None,
        "cloud_thickness_bps": None,
        "cloud_direction": None,
    }
    if not _ohlc_ok(high, low, close) or len(close) < senkou_b_period:
        return empty

    def midpoint(period: int) -> float | None:
        if len(high) < period or len(low) < period:
            return None
        return (max(high[-period:]) + min(low[-period:])) / 2.0

    tenkan = midpoint(tenkan_period)
    kijun = midpoint(kijun_period)
    senkou_b = midpoint(senkou_b_period)
    senkou_a = (
        (tenkan + kijun) / 2.0 if tenkan is not None and kijun is not None else None
    )
    price = close[-1]
    result = dict(empty)
    result["tenkan"] = tenkan
    result["kijun"] = kijun
    result["senkou_a"] = senkou_a
    result["senkou_b"] = senkou_b
    if tenkan is not None and kijun is not None and kijun != 0:
        result["tenkan_kijun_gap_bps"] = _bps(tenkan - kijun, kijun)
    if senkou_a is not None and senkou_b is not None and price > 0:
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        if price > cloud_top:
            result["price_vs_cloud"] = 1.0
        elif price < cloud_bottom:
            result["price_vs_cloud"] = -1.0
        else:
            result["price_vs_cloud"] = 0.0
        result["cloud_thickness_bps"] = _bps(cloud_top - cloud_bottom, price)
        if senkou_a > senkou_b:
            result["cloud_direction"] = 1.0
        elif senkou_a < senkou_b:
            result["cloud_direction"] = -1.0
        else:
            result["cloud_direction"] = 0.0
    return result


# --- Trendline (rolling regression, no future pivots) ------------------------

def trendline(values: list[float], window: int = 60) -> dict[str, float | None]:
    """Rolling OLS on log price: slope bps/bar, R^2, residual z, residual quantiles.

    Regression rather than swing pivots on purpose: a pivot is only confirmed by
    bars to its RIGHT, so pivot-based support/resistance computed online silently
    peeks at the future.
    """
    series = [value for value in _clean(values) if value > 0]
    empty: dict[str, float | None] = {
        "slope_bps_per_bar": None,
        "r_squared": None,
        "residual_zscore": None,
        "support_residual_quantile": None,
        "resistance_residual_quantile": None,
    }
    if window <= 1 or len(series) < max(8, window // 4):
        return empty
    sample = series[-window:] if len(series) >= window else series
    logs = [math.log(value) for value in sample]
    count = len(logs)
    xs = list(range(count))
    mean_x = (count - 1) / 2.0
    mean_y = mean(logs)
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    if variance_x == 0:
        return empty
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, logs))
    slope = covariance / variance_x
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, logs)]
    total_ss = sum((y - mean_y) ** 2 for y in logs)
    residual_ss = sum(value**2 for value in residuals)
    spread = pstdev(residuals) if count > 1 else 0.0
    ordered = sorted(residuals)

    def quantile(fraction: float) -> float:
        index = min(
            len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1))))
        )
        return ordered[index]

    return {
        # A log-price slope per bar is already a fractional rate; scale to bps.
        "slope_bps_per_bar": slope * 10_000.0,
        "r_squared": (1.0 - residual_ss / total_ss) if total_ss > 0 else None,
        "residual_zscore": (residuals[-1] / spread) if spread > 0 else None,
        "support_residual_quantile": quantile(0.10),
        "resistance_residual_quantile": quantile(0.90),
    }


# --- Directional movement / ADX ---------------------------------------------

def dmi_adx(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> tuple[float | None, float | None, float | None]:
    """(plus_di, minus_di, adx) with Wilder smoothing."""
    high = _clean(highs)
    low = _clean(lows)
    close = _clean(closes)
    if not _ohlc_ok(high, low, close) or len(close) < 2 * period + 1:
        return None, None, None
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    true_range: list[float] = []
    for index in range(1, len(close)):
        up_move = high[index] - high[index - 1]
        down_move = low[index - 1] - low[index]
        plus_dm.append(up_move if (up_move > down_move and up_move > 0) else 0.0)
        minus_dm.append(down_move if (down_move > up_move and down_move > 0) else 0.0)
        true_range.append(
            max(
                high[index] - low[index],
                abs(high[index] - close[index - 1]),
                abs(low[index] - close[index - 1]),
            )
        )

    def wilder(series: list[float]) -> list[float]:
        smoothed = [sum(series[:period])]
        for value in series[period:]:
            smoothed.append(smoothed[-1] - smoothed[-1] / period + value)
        return smoothed

    smoothed_tr = wilder(true_range)
    smoothed_plus = wilder(plus_dm)
    smoothed_minus = wilder(minus_dm)
    dx_series: list[float] = []
    plus_di = minus_di = None
    for tr_value, plus_value, minus_value in zip(
        smoothed_tr, smoothed_plus, smoothed_minus
    ):
        if tr_value == 0:
            continue
        plus_di = 100.0 * plus_value / tr_value
        minus_di = 100.0 * minus_value / tr_value
        total = plus_di + minus_di
        if total > 0:
            dx_series.append(100.0 * abs(plus_di - minus_di) / total)
    if len(dx_series) < period:
        return plus_di, minus_di, None
    adx = mean(dx_series[:period])
    for value in dx_series[period:]:
        adx = (adx * (period - 1) + value) / period
    return plus_di, minus_di, adx


# --- TRIX -------------------------------------------------------------------

def trix(
    values: list[float], period: int = 15, signal_period: int = 9
) -> tuple[float | None, float | None, float | None]:
    """(trix, signal, histogram). Triple-smoothed EMA rate of change, in bps."""
    series = [value for value in _clean(values) if value > 0]
    if period <= 0 or len(series) < 3 * period + signal_period:
        return None, None, None

    # Single-pass EMA at each level: the triple smoothing is O(n), not O(n^3).
    third = ema_series(ema_series(ema_series(series, period), period), period)
    if len(third) < 2:
        return None, None, None
    trix_series: list[float] = []
    for index in range(1, len(third)):
        change = _bps(third[index] - third[index - 1], third[index - 1])
        if change is not None:
            trix_series.append(change)
    if not trix_series:
        return None, None, None
    line = trix_series[-1]
    signal_value = (
        ema(trix_series, signal_period) if len(trix_series) >= signal_period else None
    )
    histogram = (line - signal_value) if signal_value is not None else None
    return line, signal_value, histogram


# --- CCI / Williams %R ------------------------------------------------------

def cci(
    highs: list[float], lows: list[float], closes: list[float], period: int = 20
) -> float | None:
    high = _clean(highs)
    low = _clean(lows)
    close = _clean(closes)
    if not _ohlc_ok(high, low, close) or len(close) < period:
        return None
    typical = [
        (h + l + c) / 3.0
        for h, l, c in zip(high[-period:], low[-period:], close[-period:])
    ]
    average = mean(typical)
    deviation = mean(abs(value - average) for value in typical)
    if deviation == 0:
        # Perfectly flat window: the index is undefined, not zero.
        return None
    return (typical[-1] - average) / (0.015 * deviation)


def williams_r(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> float | None:
    """[-100, 0]. None on a zero-range window rather than a fabricated midpoint."""
    high = _clean(highs)
    low = _clean(lows)
    close = _clean(closes)
    if not _ohlc_ok(high, low, close) or len(close) < period:
        return None
    window_high = max(high[-period:])
    window_low = min(low[-period:])
    span = window_high - window_low
    if span == 0:
        return None
    return -100.0 * (window_high - close[-1]) / span


# --- Momentum / rate of change ----------------------------------------------

def momentum(values: list[float], period: int = 10) -> float | None:
    """Absolute price change over ``period`` bars."""
    series = _clean(values)
    if period <= 0 or len(series) <= period:
        return None
    return series[-1] - series[-period - 1]


def roc_bps(values: list[float], period: int = 10) -> float | None:
    """Rate of change in bps."""
    series = _clean(values)
    if period <= 0 or len(series) <= period:
        return None
    base = series[-period - 1]
    if base == 0:
        return None
    return _bps(series[-1] - base, base)


def stochastic_diff(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
    signal: int = 3,
) -> float | None:
    """K - D. Positive means %K is crossing up through its own signal."""
    k = stochastic_k(highs, lows, closes, period)
    d = stochastic_d(highs, lows, closes, period, signal)
    if k is None or d is None:
        return None
    return k - d


# --- Volatility -------------------------------------------------------------

def atr_percent(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> float | None:
    """ATR as a fraction of price, so it is comparable across instruments."""
    value = atr(highs, lows, closes, period)
    close = _clean(closes)
    if value is None or not close or close[-1] <= 0:
        return None
    return value / close[-1]


def atr_expansion(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
    baseline_multiple: int = 3,
) -> float | None:
    """Recent ATR over a longer-window ATR. >1 means volatility is expanding."""
    recent = atr(highs, lows, closes, period)
    baseline = atr(highs, lows, closes, period * max(2, baseline_multiple))
    if recent is None or baseline is None or baseline == 0:
        return None
    return recent / baseline


# --- Volume and flow --------------------------------------------------------

def obv_series(closes: list[float], volumes: list[float]) -> list[float] | None:
    """Running OBV, returned as a series so slope/z-score need one pass."""
    close = _clean(closes)
    volume = _clean(volumes)
    if len(close) != len(volume) or len(close) < 2:
        return None
    if not any(value > 0 for value in volume):
        return None
    running = 0.0
    out = [0.0]
    for index in range(1, len(close)):
        if close[index] > close[index - 1]:
            running += volume[index]
        elif close[index] < close[index - 1]:
            running -= volume[index]
        out.append(running)
    return out


def obv_slope(
    closes: list[float], volumes: list[float], window: int = 20
) -> float | None:
    """OBV change per bar over ``window``, normalised by mean volume.

    Normalising matters: a raw OBV slope is in shares, which makes it an
    instrument-identity feature -- exactly the class of column that pinned the
    model's top-k to a single ticker.
    """
    series = obv_series(closes, volumes)
    volume = _clean(volumes)
    if series is None or window <= 0 or len(series) < window + 1:
        return None
    baseline = mean(volume[-window:]) if volume else 0.0
    if baseline <= 0:
        return None
    return (series[-1] - series[-window - 1]) / (window * baseline)


def obv_zscore(
    closes: list[float], volumes: list[float], window: int = 20
) -> float | None:
    series = obv_series(closes, volumes)
    if series is None or len(series) < window or window <= 1:
        return None
    recent = series[-window:]
    spread = pstdev(recent)
    if spread == 0:
        return None
    return (recent[-1] - mean(recent)) / spread


def volume_zscore(volumes: list[float], window: int = 20) -> float | None:
    volume = _clean(volumes)
    if window <= 1 or len(volume) < window:
        return None
    recent = volume[-window:]
    spread = pstdev(recent)
    if spread == 0:
        return None
    return (recent[-1] - mean(recent)) / spread


def relative_volume(volumes: list[float], window: int = 20) -> float | None:
    """Latest volume over its own trailing mean, excluding itself."""
    volume = _clean(volumes)
    if window <= 0 or len(volume) <= window:
        return None
    baseline = mean(volume[-window - 1 : -1])
    if baseline <= 0:
        return None
    return volume[-1] / baseline

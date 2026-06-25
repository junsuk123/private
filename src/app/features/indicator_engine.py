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


def macd(values: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float | None, float | None, float | None]:
    if len(values) < slow + signal - 1:
        return None, None, None
    macd_series = []
    for end in range(slow, len(values) + 1):
        fast_ema = ema(values[:end], fast)
        slow_ema = ema(values[:end], slow)
        if fast_ema is not None and slow_ema is not None:
            macd_series.append(fast_ema - slow_ema)
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

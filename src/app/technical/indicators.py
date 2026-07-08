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
    bollinger_bands as _bollinger_tuple,
    ema as _ema,
    macd as _macd_tuple,
    period_return as _period_return,
    rsi as _rsi,
    sma as _sma,
    volume_spike_ratio as _volume_spike_ratio,
)

__all__ = [
    "MacdResult",
    "BollingerResult",
    "DonchianResult",
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

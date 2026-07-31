"""Bridge: OHLCV bars (+ orderbook) -> :class:`TechnicalFeatureSet`.

Pure and deterministic. Reused by the label builder, the replay harness, the
shared decision engine, and (for the small live column subset) the live feature
frame. All computation delegates to ``app.technical.indicators``; missing/short
data yields ``None`` fields (NaN-safe) rather than fabricated numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import fmean, pstdev
from typing import Mapping, Sequence

from app.features.schemas import OHLCVBar
from app.technical import indicators as ti
from app.technical.signals import TechnicalFeatureSet


@dataclass(frozen=True)
class FeatureBuilderConfig:
    ema_fast: int = 12
    ema_slow: int = 26
    rsi_period: int = 14
    bb_period: int = 20
    donchian_period: int = 20
    atr_period: int = 14
    volume_window: int = 20
    short_return_window: int = 3
    persistence_window: int = 10
    vol_baseline_window: int = 20
    vol_recent_window: int = 5
    rvgi_period: int = 10
    box_lookback: int = 20


def _orderbook_field(orderbook: Mapping[str, float] | object, name: str) -> float | None:
    if orderbook is None:
        return None
    if isinstance(orderbook, Mapping):
        value = orderbook.get(name)
    else:
        value = getattr(orderbook, name, None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _realized_vol(closes: list[float], window: int) -> float | None:
    if len(closes) < window + 1:
        return None
    tail = closes[-window - 1 :]
    rets = [tail[i] / tail[i - 1] - 1.0 for i in range(1, len(tail)) if tail[i - 1]]
    return pstdev(rets) if len(rets) >= 2 else None


def build_technical_feature_set(
    bars: Sequence[OHLCVBar],
    *,
    symbol: str = "",
    orderbook: object = None,
    price: float | None = None,
    liquidity_score: float | None = None,
    config: FeatureBuilderConfig | None = None,
) -> TechnicalFeatureSet:
    cfg = config or FeatureBuilderConfig()
    bars = tuple(bars)
    close_vals = ti.closes(bars) if bars else []
    volume_vals = ti.volumes(bars) if bars else []
    last_price = price if price is not None else (close_vals[-1] if close_vals else None)

    ema_fast = ti.ema(close_vals, cfg.ema_fast)
    ema_slow = ti.ema(close_vals, cfg.ema_slow)
    macd_res = ti.macd(close_vals)
    boll = ti.bollinger(close_vals, cfg.bb_period)
    rsi_val = ti.rsi(close_vals, cfg.rsi_period)
    donch = ti.donchian(bars, cfg.donchian_period) if bars else None
    atr_val = ti.atr(bars, cfg.atr_period) if bars else None

    vwap_now = ti.vwap(bars) if bars else None
    vwap_distance_bps = None
    if vwap_now and last_price:
        vwap_distance_bps = (last_price / vwap_now - 1.0) * 10_000.0
    # VWAP slope: compare current vwap vs vwap excluding the last few bars.
    vwap_slope = None
    if len(bars) > cfg.vol_recent_window:
        vwap_prev = ti.vwap(bars[: -cfg.vol_recent_window])
        if vwap_now and vwap_prev:
            vwap_slope = (vwap_now / vwap_prev - 1.0) * 10_000.0

    breakout_strength = None
    donchian_low_distance = None
    donchian_high = donch.high if donch and donch.ok else None
    donchian_low = donch.low if donch and donch.ok else None
    if donchian_high and last_price:
        breakout_strength = (last_price - donchian_high) / last_price
    if donchian_low and last_price:
        donchian_low_distance = (last_price - donchian_low) / last_price

    volume_spike = ti.volume_spike_ratio(volume_vals, cfg.volume_window)
    relative_volume = None
    if len(volume_vals) > cfg.volume_window:
        baseline = fmean(volume_vals[-cfg.volume_window - 1 : -1])
        if baseline > 0:
            relative_volume = volume_vals[-1] / baseline

    atr_pct = atr_val / last_price if (atr_val and last_price) else None
    realized_vol = _realized_vol(close_vals, cfg.vol_baseline_window)
    recent_vol = _realized_vol(close_vals, cfg.vol_recent_window)
    volatility_expansion = None
    if realized_vol and recent_vol and realized_vol > 0:
        volatility_expansion = recent_vol / realized_vol

    short_return = ti.rolling_return(close_vals, cfg.short_return_window)
    momentum_persistence = None
    if len(close_vals) > cfg.persistence_window:
        tail = close_vals[-cfg.persistence_window - 1 :]
        ups = sum(1 for i in range(1, len(tail)) if tail[i] > tail[i - 1])
        momentum_persistence = ups / (len(tail) - 1)

    # False-breakout risk: price at the range high but weak volume / long upper
    # shadow -> higher risk. Conservative heuristic in [0, 1].
    false_breakout_risk = None
    if breakout_strength is not None and breakout_strength >= 0:
        risk = 0.0
        if volume_spike is None or volume_spike < 1.5:
            risk += 0.5
        if vwap_distance_bps is not None and vwap_distance_bps < 0:
            risk += 0.3
        false_breakout_risk = min(1.0, risk)

    rvgi_result = ti.rvgi(bars, cfg.rvgi_period) if bars else None
    box = ti.causal_box_geometry(bars, cfg.box_lookback) if bars else None
    breakout_distance_bps = None
    if box and box.ok and box.high and last_price:
        breakout_distance_bps = (last_price / box.high - 1.0) * 10_000.0

    best_bid = _orderbook_field(orderbook, "best_bid")
    best_ask = _orderbook_field(orderbook, "best_ask")
    spread_bps = _orderbook_field(orderbook, "spread_bps")
    if spread_bps is None and best_bid is not None and best_ask is not None:
        spread_bps = ti.spread_bps(best_bid, best_ask)
    imbalance = _orderbook_field(orderbook, "imbalance")
    if imbalance is None:
        imbalance = _orderbook_field(orderbook, "orderbook_imbalance")

    return TechnicalFeatureSet(
        symbol=symbol,
        price=last_price,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        macd=macd_res.macd if macd_res.ok else None,
        macd_signal=macd_res.signal if macd_res.ok else None,
        macd_histogram=macd_res.histogram if macd_res.ok else None,
        short_return=short_return,
        momentum_persistence=momentum_persistence,
        rsi=rsi_val,
        bb_percent_b=boll.percent_b if boll.ok else None,
        bb_bandwidth=boll.bandwidth if boll.ok else None,
        vwap=vwap_now,
        vwap_distance_bps=vwap_distance_bps,
        vwap_slope=vwap_slope,
        relative_volume=relative_volume,
        volume_spike_ratio=volume_spike,
        donchian_high=donchian_high,
        donchian_low=donchian_low,
        breakout_strength=breakout_strength,
        donchian_low_distance=donchian_low_distance,
        false_breakout_risk=false_breakout_risk,
        rvgi=rvgi_result.main if rvgi_result and rvgi_result.ok else None,
        rvgi_signal=rvgi_result.signal if rvgi_result and rvgi_result.ok else None,
        rvgi_diff=(
            rvgi_result.main - rvgi_result.signal
            if rvgi_result and rvgi_result.ok and rvgi_result.main is not None and rvgi_result.signal is not None
            else None
        ),
        rvgi_slope=rvgi_result.slope if rvgi_result and rvgi_result.ok else None,
        rvgi_bullish_cross=rvgi_result.bullish_cross if rvgi_result and rvgi_result.ok else None,
        rvgi_bearish_cross=rvgi_result.bearish_cross if rvgi_result and rvgi_result.ok else None,
        box_high=box.high if box and box.ok else None,
        box_low=box.low if box and box.ok else None,
        box_mid=box.mid if box and box.ok else None,
        box_width_pct=box.width_pct if box and box.ok else None,
        box_position=box.position if box and box.ok else None,
        breakout_distance_bps=breakout_distance_bps,
        box_context_timestamp=(
            box.source_timestamp.isoformat()
            if box and box.ok and hasattr(box.source_timestamp, "isoformat")
            else None
        ),
        box_previous_close=float(bars[-2].close) if len(bars) >= 2 else None,
        atr_pct=atr_pct,
        realized_volatility=realized_vol,
        volatility_expansion=volatility_expansion,
        liquidity_score=liquidity_score,
        spread_bps=spread_bps,
        orderbook_imbalance=imbalance,
        expected_slippage_bps=None,
    )


def technical_feature_set_from_live_frame(frame, symbol: str = "") -> TechnicalFeatureSet:
    """Map an already-built ``LiveFeatureFrame`` to a :class:`TechnicalFeatureSet`.

    Reuses the technical columns the live frame now emits (rsi_14,
    macd_histogram, bollinger_percent_b, ema_gap_bps, donchian_breakout,
    volume_spike_ratio) plus the existing microstructure/vol columns — so the
    shared decision engine needs no extra store reads. Missing keys degrade to
    ``None`` (NaN-safe).
    """
    d = frame.as_feature_dict()
    price = float(getattr(frame, "mark_price", 0.0) or 0.0) or None

    def g(name):
        v = d.get(name)
        return float(v) if v is not None else None

    ema_gap_bps = g("ema_gap_bps")
    # Reconstruct ema_fast/ema_slow so the regime gap calc matches ema_gap_bps.
    ema_slow = price
    ema_fast = price * (1.0 + ema_gap_bps / 10_000.0) if (price and ema_gap_bps is not None) else None
    dist = g("distance_from_vwap")  # fraction (price/vwap - 1)
    vwap = price / (1.0 + dist) if (price and dist is not None and dist != -1.0) else None
    return TechnicalFeatureSet(
        symbol=symbol or getattr(frame, "symbol", ""),
        price=price,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        macd_histogram=g("macd_histogram"),
        short_return=g("return_1m"),
        rsi=g("rsi_14"),
        bb_percent_b=g("bollinger_percent_b"),
        vwap=vwap,
        vwap_distance_bps=(dist * 10_000.0) if dist is not None else None,
        relative_volume=g("volume_spike_ratio"),
        volume_spike_ratio=g("volume_spike_ratio"),
        breakout_strength=g("donchian_breakout"),
        rvgi=g("rvgi") if (g("rvgi_available") or 0.0) >= 1.0 else None,
        rvgi_signal=g("rvgi_signal") if (g("rvgi_available") or 0.0) >= 1.0 else None,
        rvgi_diff=g("rvgi_diff") if (g("rvgi_available") or 0.0) >= 1.0 else None,
        rvgi_slope=g("rvgi_slope") if (g("rvgi_available") or 0.0) >= 1.0 else None,
        rvgi_bullish_cross=bool(g("rvgi_bullish_cross")) if (g("rvgi_available") or 0.0) >= 1.0 else None,
        box_high=g("box_high") if (g("box_available") or 0.0) >= 1.0 else None,
        box_low=g("box_low") if (g("box_available") or 0.0) >= 1.0 else None,
        box_mid=g("box_mid") if (g("box_available") or 0.0) >= 1.0 else None,
        box_width_pct=g("box_width_pct") if (g("box_available") or 0.0) >= 1.0 else None,
        box_position=g("box_position") if (g("box_available") or 0.0) >= 1.0 else None,
        breakout_distance_bps=g("breakout_distance_bps") if (g("box_available") or 0.0) >= 1.0 else None,
        box_previous_close=g("box_previous_close") if (g("box_available") or 0.0) >= 1.0 else None,
        box_context_timestamp=(
            datetime.fromtimestamp(
                g("box_context_timestamp_epoch") or 0.0,
                tz=timezone.utc,
            ).isoformat()
            if (g("box_available") or 0.0) >= 1.0
            and (g("box_context_timestamp_epoch") or 0.0) > 0
            else None
        ),
        realized_volatility=g("realized_volatility_3m"),
        liquidity_score=g("liquidity_score"),
        spread_bps=g("spread_bps"),
        orderbook_imbalance=g("orderbook_imbalance"),
        bid_depth=g("bid_depth"),
        ask_depth=g("ask_depth"),
        depth_ratio=g("depth_ratio"),
        # Sub-second window. The live frame has always computed these; without
        # this mapping they never reached a strategy, so every "algorithm" was
        # effectively minute-bar only.
        return_1s=g("return_1s"),
        return_5s=g("return_5s"),
        return_10s=g("return_10s"),
        tick_count_1s=g("tick_count_1s"),
        tick_count_5s=g("tick_count_5s"),
        volume_1s_log=g("volume_1s_log"),
        volume_5s_log=g("volume_5s_log"),
        aggressor_imbalance_5s=g("aggressor_imbalance_5s"),
        realized_volatility_10s=g("realized_volatility_10s"),
        spread_change_5s=g("spread_change_5s"),
        orderbook_imbalance_change_5s=g("orderbook_imbalance_change_5s"),
        second_data_ready=g("second_data_ready"),
    )

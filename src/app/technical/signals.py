"""Methodology-specific technical signal providers + composite engine.

Each provider maps a computed :class:`TechnicalFeatureSet` to a
:class:`TechnicalSignal` (signed score in [-1, 1], confidence in [0, 1], a
conservative expected edge in bps, an expected horizon, and supporting /
contradicting features + reason codes). The :class:`CompositeTechnicalSignalEngine`
runs them, applies regime gating and the mandatory VWAP/volume confirmation
layer, weights the enabled methodologies, and returns one composite advisory
signal plus a full per-methodology breakdown.

Hard rules encoded here (mirroring the task spec):
    * Advisory only — nothing here builds or submits an order.
    * No single indicator triggers BUY alone; BUY evidence requires methodology
      agreement AND VWAP/volume confirmation AND a non-blocking regime.
    * Expected edge is a conservative function of measured volatility and the
      horizon — there is NO fabricated minimum-alpha floor. Net profitability is
      decided later by the ProfitabilityGate, not here.
    * Mean reversion is disabled in a downtrend; breakout requires volume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from app.technical import reason_codes as rc
from app.technical.regime import (
    BUY_BLOCKING_REGIMES,
    MarketRegime,
    RegimeConfig,
    RegimeDiagnostics,
    RegimeInput,
    TechnicalRegimeClassifier,
)


class SignalDirection(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    REDUCE = "REDUCE"
    HOLD = "HOLD"


@dataclass(frozen=True)
class TechnicalFeatureSet:
    """Everything the regime classifier and signal providers consume.

    Built from ``app.technical.indicators`` and the live feature frame. Every
    field is NaN-safe (``None`` == unavailable).
    """

    symbol: str = ""
    price: float | None = None
    # Trend
    ema_fast: float | None = None
    ema_slow: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    short_return: float | None = None
    momentum_persistence: float | None = None  # fraction of recent bars up, [0,1]
    adx: float | None = None
    plus_di: float | None = None
    minus_di: float | None = None
    dmi_spread: float | None = None
    supertrend: float | None = None
    supertrend_direction: float | None = None
    supertrend_distance_bps: float | None = None
    # Oscillator / bands
    rsi: float | None = None
    bb_percent_b: float | None = None
    bb_bandwidth: float | None = None
    keltner_mid: float | None = None
    keltner_upper: float | None = None
    keltner_lower: float | None = None
    keltner_position: float | None = None
    keltner_bandwidth: float | None = None
    prior_keltner_squeeze_ratio: float | None = None
    choppiness: float | None = None
    # VWAP / volume
    vwap: float | None = None
    vwap_distance_bps: float | None = None
    vwap_slope: float | None = None
    relative_volume: float | None = None
    volume_spike_ratio: float | None = None
    # Breakout
    donchian_high: float | None = None
    donchian_low: float | None = None
    breakout_strength: float | None = None       # (price - donchian_high)/price, signed
    donchian_low_distance: float | None = None    # (price - donchian_low)/price, signed
    false_breakout_risk: float | None = None      # [0,1], higher == riskier
    rvgi: float | None = None
    rvgi_signal: float | None = None
    rvgi_diff: float | None = None
    rvgi_slope: float | None = None
    rvgi_bullish_cross: bool | None = None
    rvgi_bearish_cross: bool | None = None
    box_high: float | None = None
    box_low: float | None = None
    box_mid: float | None = None
    box_width_pct: float | None = None
    box_position: float | None = None
    breakout_distance_bps: float | None = None
    box_context_timestamp: str | None = None
    box_previous_close: float | None = None
    # Volatility / risk
    atr_pct: float | None = None
    realized_volatility: float | None = None      # per-bar return stdev (fraction)
    volatility_expansion: float | None = None     # recent/ baseline vol ratio
    # Microstructure
    liquidity_score: float | None = None
    spread_bps: float | None = None
    orderbook_imbalance: float | None = None
    expected_slippage_bps: float | None = None
    # Book depth. Present in the live feature frame all along but never mapped
    # through, so no strategy could see whether the bid side was being replenished
    # or the ask side depleted.
    bid_depth: float | None = None
    ask_depth: float | None = None
    depth_ratio: float | None = None
    # Tick / sub-second window (live feed only; ``None`` for bar-only callers).
    # These are what the mechanical entry triggers in
    # ``app.technical.strategy_algorithms`` actually fire on — the bar columns
    # above only supply slower context.
    return_1s: float | None = None
    return_5s: float | None = None
    return_10s: float | None = None
    return_30s: float | None = None
    tick_count_1s: float | None = None
    tick_count_5s: float | None = None
    volume_1s_log: float | None = None
    volume_5s_log: float | None = None
    aggressor_imbalance_5s: float | None = None
    realized_volatility_10s: float | None = None
    spread_change_5s: float | None = None
    orderbook_imbalance_change_5s: float | None = None
    second_data_ready: float | None = None

    @property
    def tick_data_ready(self) -> bool:
        """True when the sub-second window is populated enough to trigger on."""
        return bool(self.second_data_ready and self.second_data_ready >= 1.0)

    @property
    def microprice_edge_bps(self) -> float | None:
        """Depth-weighted microprice minus mid, in bps. ``None`` when unknown.

        For a two-sided book the size-weighted microprice satisfies

            microprice - mid = (spread / 2) * (bid_size - ask_size)/(bid_size + ask_size)

        and ``orderbook_imbalance`` is exactly that depth ratio, so this is an
        identity rather than an approximation — except that the imbalance is
        computed over ALL book levels, making this a depth-weighted microprice
        tilt rather than the strict best-bid/best-ask microprice. Positive means
        the book leans bid, i.e. the fair price sits above the mid.
        """
        if self.spread_bps is None or self.orderbook_imbalance is None:
            return None
        if self.spread_bps < 0:
            return None
        return 0.5 * float(self.spread_bps) * float(self.orderbook_imbalance)

    @property
    def residual_volatility_bps(self) -> float | None:
        """Best available per-observation volatility in bps, for normalisation."""
        for candidate in (self.realized_volatility, self.realized_volatility_10s):
            if candidate is not None and candidate > 0:
                return float(candidate) * 10_000.0
        return None

    @property
    def vwap_zscore(self) -> float | None:
        """VWAP displacement normalised by intraday volatility.

        A fixed 25bps VWAP band is noise in a tape moving 4-10% a day and a wall
        in a quiet one. Dividing the displacement by realised volatility makes the
        same threshold mean the same thing in both.
        """
        volatility_bps = self.residual_volatility_bps
        if self.vwap_distance_bps is None or not volatility_bps:
            return None
        return float(self.vwap_distance_bps) / volatility_bps

    def to_regime_input(self) -> RegimeInput:
        return RegimeInput(
            symbol=self.symbol,
            price=self.price,
            ema_fast=self.ema_fast,
            ema_slow=self.ema_slow,
            macd_histogram=self.macd_histogram,
            rsi=self.rsi,
            bb_percent_b=self.bb_percent_b,
            bb_bandwidth=self.bb_bandwidth,
            vwap_distance_bps=self.vwap_distance_bps,
            breakout_strength=self.breakout_strength,
            donchian_low_distance=self.donchian_low_distance,
            volume_spike_ratio=self.volume_spike_ratio,
            atr_pct=self.atr_pct,
            realized_volatility=self.realized_volatility,
            short_return=self.short_return,
            liquidity_score=self.liquidity_score,
            spread_bps=self.spread_bps,
            orderbook_imbalance=self.orderbook_imbalance,
        )


@dataclass(frozen=True)
class TechnicalSignal:
    methodology: str
    direction: SignalDirection
    score: float                 # signed, [-1, 1]
    confidence: float            # [0, 1]
    expected_edge_bps: float     # conservative gross edge; >= 0
    expected_horizon_seconds: int
    supporting_features: tuple[str, ...] = ()
    contradicting_features: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    available: bool = True

    def as_dict(self) -> dict:
        return {
            "methodology": self.methodology,
            "direction": self.direction.value,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "expected_edge_bps": round(self.expected_edge_bps, 3),
            "expected_horizon_seconds": self.expected_horizon_seconds,
            "supporting_features": list(self.supporting_features),
            "contradicting_features": list(self.contradicting_features),
            "reason_codes": list(self.reason_codes),
            "available": self.available,
        }


@dataclass(frozen=True)
class CompositeTechnicalSignal:
    symbol: str
    direction: SignalDirection
    score: float
    confidence: float
    expected_edge_bps: float
    expected_horizon_seconds: int
    regime: MarketRegime
    regime_confidence: float
    selected_methodology: str
    contributing: tuple[TechnicalSignal, ...]
    reason_codes: tuple[str, ...]
    blocks_buy: bool
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "expected_edge_bps": round(self.expected_edge_bps, 3),
            "expected_horizon_seconds": self.expected_horizon_seconds,
            "regime": self.regime.value,
            "regime_confidence": round(self.regime_confidence, 4),
            "selected_methodology": self.selected_methodology,
            "contributing": [s.as_dict() for s in self.contributing],
            "reason_codes": list(self.reason_codes),
            "blocks_buy": self.blocks_buy,
            "diagnostics": dict(self.diagnostics),
        }


# --------------------------------------------------------------------------- #
# Conservative expected-move model                                             #
# --------------------------------------------------------------------------- #
def _expected_move_bps(
    volatility_proxy: float | None,
    horizon_seconds: int,
    *,
    bar_seconds: int = 60,
    capture_fraction: float = 0.5,
) -> float:
    """Conservative expected favorable move over the horizon, in bps.

    Uses the per-bar volatility proxy scaled by sqrt(time) and a <1 capture
    fraction (we never expect to capture the full move). Returns 0.0 when no
    volatility proxy is available — deliberately NO fabricated floor.
    """
    if volatility_proxy is None or volatility_proxy <= 0 or horizon_seconds <= 0:
        return 0.0
    scaled = volatility_proxy * math.sqrt(horizon_seconds / bar_seconds)
    return max(0.0, capture_fraction * scaled * 10_000.0)


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _unavailable(methodology: str, horizon: int) -> TechnicalSignal:
    return TechnicalSignal(
        methodology=methodology,
        direction=SignalDirection.HOLD,
        score=0.0,
        confidence=0.0,
        expected_edge_bps=0.0,
        expected_horizon_seconds=horizon,
        reason_codes=(rc.TECHNICAL_SIGNAL_UNAVAILABLE,),
        available=False,
    )


# --------------------------------------------------------------------------- #
# Providers                                                                    #
# --------------------------------------------------------------------------- #
class MomentumTrendSignalProvider:
    methodology = "momentum_trend_following"
    horizon_seconds = 60

    def evaluate(self, f: TechnicalFeatureSet) -> TechnicalSignal:
        if f.price is None or f.ema_fast is None or f.ema_slow is None:
            return _unavailable(self.methodology, self.horizon_seconds)
        support: list[str] = []
        contra: list[str] = []
        codes: list[str] = []
        score = 0.0

        ema_gap_bps = (f.ema_fast - f.ema_slow) / f.price * 10_000.0 if f.price else 0.0
        gap_component = _clamp(ema_gap_bps / 20.0)  # 20bps gap ~ full strength
        score += 0.4 * gap_component
        (support if gap_component >= 0 else contra).append("ema_fast_minus_slow")

        if f.macd_histogram is not None:
            macd_component = _clamp(math.copysign(min(1.0, abs(f.macd_histogram) / (abs(f.macd or 0.0) + 1e-9)), f.macd_histogram))
            score += 0.3 * macd_component
            (support if f.macd_histogram >= 0 else contra).append("macd_histogram")

        if f.short_return is not None:
            ret_component = _clamp(f.short_return / 0.01)  # 1% move ~ full
            score += 0.2 * ret_component
            (support if f.short_return >= 0 else contra).append("short_return")

        if f.momentum_persistence is not None:
            persist_component = _clamp((f.momentum_persistence - 0.5) * 2.0)
            score += 0.1 * persist_component
            (support if persist_component >= 0 else contra).append("momentum_persistence")

        score = _clamp(score)
        direction = SignalDirection.BUY if score > 0 else (SignalDirection.SELL if score < 0 else SignalDirection.HOLD)
        confidence = min(1.0, abs(score) * (0.6 + 0.4 * (len(support) / 4.0)))
        vol_proxy = f.realized_volatility if f.realized_volatility is not None else f.atr_pct
        edge = _expected_move_bps(vol_proxy, self.horizon_seconds) * abs(score)
        if score > 0:
            codes.append(rc.MOMENTUM_CONFIRMED)
        elif score < 0:
            codes.append(rc.MOMENTUM_WEAKENED)
        return TechnicalSignal(
            methodology=self.methodology,
            direction=direction,
            score=score,
            confidence=confidence,
            expected_edge_bps=edge,
            expected_horizon_seconds=self.horizon_seconds,
            supporting_features=tuple(support),
            contradicting_features=tuple(contra),
            reason_codes=tuple(codes),
        )


class BreakoutSignalProvider:
    methodology = "breakout_trading_range_break"
    horizon_seconds = 60

    def evaluate(self, f: TechnicalFeatureSet) -> TechnicalSignal:
        if f.price is None or f.breakout_strength is None:
            return _unavailable(self.methodology, self.horizon_seconds)
        support: list[str] = []
        contra: list[str] = []
        codes: list[str] = []

        # breakout_strength >= 0 means price at/above the recent range high.
        at_high = f.breakout_strength >= 0
        volume_ok = f.volume_spike_ratio is not None and f.volume_spike_ratio >= 1.5
        vwap_ok = f.vwap_distance_bps is not None and f.vwap_distance_bps > 0
        false_risk = f.false_breakout_risk if f.false_breakout_risk is not None else 0.0

        score = 0.0
        if at_high:
            score += 0.5
            support.append("donchian_high")
        else:
            score += _clamp(f.breakout_strength / 0.01) * 0.2  # approaching
        if volume_ok:
            score += 0.3
            support.append("breakout_volume_confirmation")
        else:
            codes.append(rc.VOLUME_CONFIRMATION_MISSING)
            contra.append("breakout_volume_confirmation")
        if vwap_ok:
            score += 0.2
            support.append("breakout_vwap_confirmation")
        else:
            contra.append("breakout_vwap_confirmation")
        # Penalize by false-breakout risk.
        score -= 0.4 * _clamp(false_risk, 0.0, 1.0)
        if false_risk >= 0.6:
            codes.append(rc.FALSE_BREAKOUT_RISK_HIGH)
            contra.append("false_breakout_risk")

        score = _clamp(score)
        confirmed = at_high and volume_ok and vwap_ok and false_risk < 0.6
        if confirmed:
            codes.append(rc.BREAKOUT_CONFIRMED)
        direction = SignalDirection.BUY if score > 0 else SignalDirection.HOLD
        confidence = min(1.0, max(0.0, score) * (0.7 if confirmed else 0.4))
        vol_proxy = f.realized_volatility if f.realized_volatility is not None else f.atr_pct
        edge = _expected_move_bps(vol_proxy, self.horizon_seconds) * max(0.0, score)
        return TechnicalSignal(
            methodology=self.methodology,
            direction=direction,
            score=score,
            confidence=confidence,
            expected_edge_bps=edge,
            expected_horizon_seconds=self.horizon_seconds,
            supporting_features=tuple(support),
            contradicting_features=tuple(contra),
            reason_codes=tuple(codes),
        )


class MeanReversionSignalProvider:
    methodology = "mean_reversion"
    horizon_seconds = 120

    def evaluate(self, f: TechnicalFeatureSet) -> TechnicalSignal:
        if f.rsi is None and f.bb_percent_b is None:
            return _unavailable(self.methodology, self.horizon_seconds)
        support: list[str] = []
        contra: list[str] = []
        codes: list[str] = []
        score = 0.0

        if f.rsi is not None:
            if f.rsi <= 30:
                score += _clamp((30 - f.rsi) / 30.0)  # deeper oversold -> stronger
                support.append("rsi_7")
            elif f.rsi >= 70:
                score -= _clamp((f.rsi - 70) / 30.0)
                contra.append("rsi_7")
        if f.bb_percent_b is not None:
            if f.bb_percent_b <= 0.1:
                score += (0.1 - f.bb_percent_b) * 5.0  # 0 -> +0.5
                support.append("bollinger_percent_b")
            elif f.bb_percent_b >= 0.9:
                score -= (f.bb_percent_b - 0.9) * 5.0
                contra.append("bollinger_percent_b")
        if f.donchian_low_distance is not None and f.donchian_low_distance <= 0:
            support.append("oversold_recovery_score")

        score = _clamp(score)
        direction = SignalDirection.BUY if score > 0 else (SignalDirection.SELL if score < 0 else SignalDirection.HOLD)
        confidence = min(1.0, abs(score) * 0.6)
        vol_proxy = f.realized_volatility if f.realized_volatility is not None else f.atr_pct
        edge = _expected_move_bps(vol_proxy, self.horizon_seconds) * abs(score)
        if score > 0:
            codes.append(rc.MEAN_REVERSION_CANDIDATE)
        return TechnicalSignal(
            methodology=self.methodology,
            direction=direction,
            score=score,
            confidence=confidence,
            expected_edge_bps=edge,
            expected_horizon_seconds=self.horizon_seconds,
            supporting_features=tuple(support),
            contradicting_features=tuple(contra),
            reason_codes=tuple(codes),
        )


class VwapVolumeSignalProvider:
    """Mandatory execution-quality confirmation layer for momentum/breakout.

    Its score is a confirmation strength in [-1, 1]: positive when price is
    above VWAP with supportive volume/order-flow, negative on a VWAP breakdown.
    """

    methodology = "vwap_volume_liquidity"
    horizon_seconds = 60

    def evaluate(self, f: TechnicalFeatureSet) -> TechnicalSignal:
        if f.vwap_distance_bps is None:
            return _unavailable(self.methodology, self.horizon_seconds)
        support: list[str] = []
        contra: list[str] = []
        codes: list[str] = []
        score = 0.0

        vwap_component = _clamp(f.vwap_distance_bps / 15.0)
        score += 0.5 * vwap_component
        if f.vwap_distance_bps >= 0:
            support.append("price_to_vwap_bps")
            codes.append(rc.VWAP_CONFIRMATION_OK)
        else:
            contra.append("price_to_vwap_bps")
            codes.append(rc.VWAP_BREAKDOWN)

        if f.vwap_slope is not None:
            score += 0.2 * _clamp(f.vwap_slope / 5.0)
            (support if f.vwap_slope >= 0 else contra).append("vwap_slope")
        if f.relative_volume is not None:
            score += 0.2 * _clamp((f.relative_volume - 1.0))
            (support if f.relative_volume >= 1.0 else contra).append("relative_volume")
        if f.orderbook_imbalance is not None:
            score += 0.1 * _clamp(f.orderbook_imbalance)
            (support if f.orderbook_imbalance >= 0 else contra).append("orderbook_imbalance")

        score = _clamp(score)
        direction = SignalDirection.BUY if score > 0 else (SignalDirection.SELL if score < 0 else SignalDirection.HOLD)
        confidence = min(1.0, abs(score) * 0.7)
        return TechnicalSignal(
            methodology=self.methodology,
            direction=direction,
            score=score,
            confidence=confidence,
            expected_edge_bps=0.0,  # confirmation layer contributes no standalone edge
            expected_horizon_seconds=self.horizon_seconds,
            supporting_features=tuple(support),
            contradicting_features=tuple(contra),
            reason_codes=tuple(codes),
        )


class VolatilityBandSignalProvider:
    """Risk-oriented provider feeding regime, exit policy, and expected-move.

    Produces a NEGATIVE score (REDUCE/HOLD bias) as volatility expands or the
    band regime becomes hostile; near-neutral in calm regimes.
    """

    methodology = "volatility_band_regime"
    horizon_seconds = 300

    def evaluate(self, f: TechnicalFeatureSet) -> TechnicalSignal:
        if f.realized_volatility is None and f.atr_pct is None and f.bb_bandwidth is None:
            return _unavailable(self.methodology, self.horizon_seconds)
        support: list[str] = []
        contra: list[str] = []
        codes: list[str] = []
        risk = 0.0

        if f.volatility_expansion is not None and f.volatility_expansion > 1.0:
            risk += _clamp((f.volatility_expansion - 1.0))
            contra.append("volatility_expansion_score")
        if f.atr_pct is not None and f.atr_pct > 0.02:
            risk += _clamp((f.atr_pct - 0.02) / 0.02)
            contra.append("atr_5m")
        if f.bb_bandwidth is not None and f.bb_bandwidth > 0.06:
            risk += _clamp((f.bb_bandwidth - 0.06) / 0.06)
            contra.append("bollinger_bandwidth")

        risk = _clamp(risk, 0.0, 1.0)
        score = -risk  # higher risk -> more negative
        codes.append(rc.HIGH_VOLATILITY_TECHNICAL_BLOCK) if risk >= 0.6 else None
        direction = SignalDirection.REDUCE if risk >= 0.6 else SignalDirection.HOLD
        confidence = min(1.0, risk)
        return TechnicalSignal(
            methodology=self.methodology,
            direction=direction,
            score=score,
            confidence=confidence,
            expected_edge_bps=0.0,
            expected_horizon_seconds=self.horizon_seconds,
            supporting_features=tuple(support),
            contradicting_features=tuple(contra),
            reason_codes=tuple(c for c in codes if c),
        )


# --------------------------------------------------------------------------- #
# Composite engine                                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SignalEngineConfig:
    methodology_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "momentum_trend_following": 1.0,
            "breakout_trading_range_break": 1.0,
            "mean_reversion": 0.8,
            "vwap_volume_liquidity": 1.2,
            "volatility_band_regime": 0.7,
        }
    )
    minimum_confidence: Mapping[str, float] = field(
        default_factory=lambda: {
            "momentum_trend_following": 0.55,
            "breakout_trading_range_break": 0.6,
            "mean_reversion": 0.62,
            "vwap_volume_liquidity": 0.5,
        }
    )
    require_vwap_confirmation: bool = True
    block_mean_reversion_in_downtrend: bool = True
    block_breakout_without_volume: bool = True
    spread_alpha_max_ratio: float = 0.5  # spread must not exceed this share of edge


# Which methodologies are preferred per regime (regime-first policy).
_REGIME_PREFERRED: dict[MarketRegime, frozenset[str]] = {
    MarketRegime.TREND_UP: frozenset(
        {"momentum_trend_following", "breakout_trading_range_break", "vwap_volume_liquidity"}
    ),
    MarketRegime.BREAKOUT_CANDIDATE: frozenset(
        {"breakout_trading_range_break", "momentum_trend_following", "vwap_volume_liquidity"}
    ),
    MarketRegime.RANGE_BOUND: frozenset(
        {"mean_reversion", "vwap_volume_liquidity", "volatility_band_regime"}
    ),
    MarketRegime.MEAN_REVERSION_CANDIDATE: frozenset(
        {"mean_reversion", "vwap_volume_liquidity"}
    ),
}

_OWNED_STRATEGY_METHODOLOGY: dict[str, str] = {
    "intraday_momentum": "momentum_trend_following",
    "breakout_volume": "breakout_trading_range_break",
    "vwap_mean_reversion": "mean_reversion",
    "liquidity_shock_reversal": "mean_reversion",
    "event_momentum": "momentum_trend_following",
    "cross_sectional_relative_strength": "momentum_trend_following",
    "gap_context": "momentum_trend_following",
    "rvgi_box_breakout": "breakout_trading_range_break",
}


class CompositeTechnicalSignalEngine:
    def __init__(
        self,
        config: SignalEngineConfig | None = None,
        *,
        regime_classifier: TechnicalRegimeClassifier | None = None,
    ) -> None:
        self.config = config or SignalEngineConfig()
        self.regime_classifier = regime_classifier or TechnicalRegimeClassifier(RegimeConfig())
        self.providers = {
            p.methodology: p
            for p in (
                MomentumTrendSignalProvider(),
                BreakoutSignalProvider(),
                MeanReversionSignalProvider(),
                VwapVolumeSignalProvider(),
                VolatilityBandSignalProvider(),
            )
        }
        # Per-strategy mechanical algorithms, built once. Imported lazily to
        # keep this module importable from the bar-only offline harnesses.
        from app.technical.strategy_algorithms import AlgorithmConfig, build_algorithm_registry

        self._algorithms = build_algorithm_registry()
        self._algorithms_by_market = {
            market: build_algorithm_registry(AlgorithmConfig(market=market))
            for market in ("KR", "US")
        }

    def evaluate(self, features: TechnicalFeatureSet) -> CompositeTechnicalSignal:
        regime_diag: RegimeDiagnostics = self.regime_classifier.classify(features.to_regime_input())
        signals = {name: p.evaluate(features) for name, p in self.providers.items()}
        reason_codes: list[str] = []

        # ---- Hard BUY blocks from regime ---- #
        if regime_diag.blocks_buy:
            if regime_diag.regime == MarketRegime.LOW_LIQUIDITY_RISK:
                reason_codes.append(rc.LOW_LIQUIDITY_TECHNICAL_BLOCK)
            elif regime_diag.regime == MarketRegime.HIGH_VOLATILITY_RISK:
                reason_codes.append(rc.HIGH_VOLATILITY_TECHNICAL_BLOCK)
            return self._blocked(features, regime_diag, signals, reason_codes)

        preferred = _REGIME_PREFERRED.get(regime_diag.regime, frozenset())
        vwap_signal = signals["vwap_volume_liquidity"]
        vwap_confirms = vwap_signal.available and vwap_signal.score > 0

        # ---- Assemble enabled BUY-direction contributors ---- #
        contributors: list[tuple[TechnicalSignal, float]] = []
        for name, sig in signals.items():
            if name in ("vwap_volume_liquidity", "volatility_band_regime"):
                continue  # confirmation / risk layers, handled separately
            if not sig.available or sig.direction != SignalDirection.BUY or sig.score <= 0:
                continue
            # Regime gating.
            if name == "mean_reversion" and self.config.block_mean_reversion_in_downtrend:
                if regime_diag.regime == MarketRegime.TREND_DOWN:
                    reason_codes.append(rc.MEAN_REVERSION_BLOCKED_BY_DOWNTREND)
                    continue
            if name == "breakout_trading_range_break" and self.config.block_breakout_without_volume:
                if features.volume_spike_ratio is None or features.volume_spike_ratio < 1.5:
                    reason_codes.append(rc.VOLUME_CONFIRMATION_MISSING)
                    continue
            # Minimum confidence.
            if sig.confidence < self.config.minimum_confidence.get(name, 0.0):
                reason_codes.append(rc.TECHNICAL_CONFIDENCE_TOO_LOW)
                continue
            # Prefer regime-appropriate methods: out-of-regime methods are down-weighted.
            regime_factor = 1.0 if (not preferred or name in preferred) else 0.4
            weight = self.config.methodology_weights.get(name, 1.0) * regime_factor
            contributors.append((sig, weight))

        # ---- Mandatory VWAP/volume confirmation for BUY ---- #
        if contributors and self.config.require_vwap_confirmation and not vwap_confirms:
            reason_codes.append(rc.VWAP_BREAKDOWN)
            return self._hold(features, regime_diag, signals, reason_codes, vwap_signal)

        if not contributors:
            # No methodology emitted a BUY is an inactive setup, not evidence
            # that an otherwise valid strategy has negative expectancy.  The old
            # code mixed those states and made the GUI's "non-positive edge"
            # count look like an algorithm-performance verdict.
            reason_codes.append(rc.TECHNICAL_SETUP_NOT_ACTIVE)
            return self._hold(features, regime_diag, signals, reason_codes, vwap_signal)

        # ---- Weighted aggregation ---- #
        total_w = sum(w for _, w in contributors)
        agg_score = sum(s.score * w for s, w in contributors) / total_w
        agg_conf = sum(s.confidence * w for s, w in contributors) / total_w
        # Fold in the VWAP confirmation strength as a confidence multiplier.
        agg_conf = min(1.0, agg_conf * (0.7 + 0.3 * min(1.0, vwap_signal.score)))
        agg_edge = sum(s.expected_edge_bps * w for s, w in contributors) / total_w
        # Volatility-risk penalty reduces edge/confidence.
        vol_sig = signals["volatility_band_regime"]
        if vol_sig.available and vol_sig.score < 0:
            agg_edge *= max(0.3, 1.0 + vol_sig.score)  # score in [-1,0]
            agg_conf *= max(0.4, 1.0 + 0.5 * vol_sig.score)

        # Spread-consumes-alpha check.
        if features.spread_bps is not None and agg_edge > 0:
            if features.spread_bps > self.config.spread_alpha_max_ratio * agg_edge:
                reason_codes.append(rc.SPREAD_CONSUMES_TECHNICAL_ALPHA)

        selected = max(contributors, key=lambda cw: cw[0].score * cw[1])[0]
        horizon = selected.expected_horizon_seconds
        reason_codes.extend(selected.reason_codes)
        if vwap_confirms:
            reason_codes.append(rc.VWAP_CONFIRMATION_OK)

        return CompositeTechnicalSignal(
            symbol=features.symbol,
            direction=SignalDirection.BUY,
            score=_clamp(agg_score),
            confidence=max(0.0, min(1.0, agg_conf)),
            expected_edge_bps=max(0.0, agg_edge),
            expected_horizon_seconds=horizon,
            regime=regime_diag.regime,
            regime_confidence=regime_diag.confidence,
            selected_methodology=selected.methodology,
            contributing=tuple(s for s, _ in contributors),
            reason_codes=tuple(dict.fromkeys(reason_codes)),
            blocks_buy=False,
            diagnostics={
                "regime": regime_diag.as_dict(),
                "vwap_confirms": vwap_confirms,
                "signals": {n: s.as_dict() for n, s in signals.items()},
            },
        )

    def evaluate_owned_strategy(
        self,
        features: TechnicalFeatureSet,
        strategy_id: str,
        context: object | None = None,
    ) -> CompositeTechnicalSignal:
        """Run the elected strategy's own mechanical entry trigger.

        Ontology/GNN elects ``strategy_id`` and supplies ``context`` (an
        :class:`~app.technical.strategy_algorithms.ElectionContext`). Each
        strategy id resolves to its OWN algorithm — previously four ids shared
        one momentum provider, so electing different strategies ran identical
        code.

        Regime, liquidity, volatility and spread admissibility are deliberately
        NOT evaluated here. They are resolved once at election time and then
        watched by ``app.trading.strategy_supervisor``, which can halt this
        algorithm. The regime is still classified, but only as diagnostics.
        """
        from app.technical.strategy_algorithms import ElectionContext, get_algorithm

        strategy = str(strategy_id or "").strip().lower()
        regime_diag = self.regime_classifier.classify(features.to_regime_input())
        signals = {name: provider.evaluate(features) for name, provider in self.providers.items()}
        market = "KR" if features.symbol.isdigit() and len(features.symbol) == 6 else "US"
        algorithm = get_algorithm(
            strategy,
            registry=self._algorithms_by_market.get(market, self._algorithms),
        )
        if algorithm is None:
            return self._blocked(
                features,
                regime_diag,
                signals,
                ("STRATEGY_IMPLEMENTATION_MISSING",),
            )

        election = context if isinstance(context, ElectionContext) else ElectionContext(strategy_id=strategy)
        decision = algorithm.entry(features, election)
        methodology = _OWNED_STRATEGY_METHODOLOGY.get(strategy, strategy)
        reference = signals.get(methodology)
        diagnostics = {
            "owned_strategy_id": strategy,
            "strategy_locked": True,
            "algorithm": decision.as_dict(),
            "election_context": election.as_dict(),
            # Diagnostics only — the supervisor, not this method, acts on these.
            "regime": regime_diag.as_dict(),
            "reference_methodology_signal": reference.as_dict() if reference else None,
        }
        if not decision.triggered:
            return CompositeTechnicalSignal(
                symbol=features.symbol,
                direction=SignalDirection.HOLD,
                score=0.0,
                confidence=0.0,
                expected_edge_bps=0.0,
                expected_horizon_seconds=0,
                regime=regime_diag.regime,
                regime_confidence=regime_diag.confidence,
                selected_methodology="",
                contributing=(),
                reason_codes=decision.reason_codes,
                blocks_buy=False,
                diagnostics=diagnostics,
            )
        return CompositeTechnicalSignal(
            symbol=features.symbol,
            direction=SignalDirection.BUY,
            score=_clamp(decision.score),
            confidence=max(0.0, min(1.0, decision.confidence)),
            expected_edge_bps=max(0.0, decision.expected_edge_bps),
            expected_horizon_seconds=decision.horizon_seconds,
            regime=regime_diag.regime,
            regime_confidence=regime_diag.confidence,
            selected_methodology=strategy,
            contributing=(reference,) if reference else (),
            reason_codes=decision.reason_codes,
            blocks_buy=False,
            diagnostics=diagnostics,
        )

    def evaluate_exit_deterioration(self, features: TechnicalFeatureSet) -> tuple[str, ...]:
        """Advisory SELL/REDUCE deterioration reason codes for a held position.

        Consumed by the dynamic-exit integration (Phase 8). Never forces an
        exit by itself — hard/emergency stops remain the circuit breakers.
        """
        codes: list[str] = []
        if features.vwap_distance_bps is not None and features.vwap_distance_bps < 0:
            codes.append(rc.VWAP_BREAKDOWN)
        if features.macd_histogram is not None and features.macd_histogram < 0:
            codes.append(rc.MOMENTUM_WEAKENED)
        if features.volatility_expansion is not None and features.volatility_expansion > 1.5:
            codes.append(rc.HIGH_VOLATILITY_TECHNICAL_BLOCK)
        if features.false_breakout_risk is not None and features.false_breakout_risk >= 0.6:
            codes.append(rc.FALSE_BREAKOUT_RISK_HIGH)
        if features.liquidity_score is not None and features.liquidity_score < 0.35:
            codes.append(rc.LOW_LIQUIDITY_TECHNICAL_BLOCK)
        if codes:
            codes.append(rc.TECHNICAL_EXIT_DETERIORATION)
        return tuple(dict.fromkeys(codes))

    # ------------------------------------------------------------------ #
    def _blocked(self, features, regime_diag, signals, reason_codes) -> CompositeTechnicalSignal:
        return CompositeTechnicalSignal(
            symbol=features.symbol,
            direction=SignalDirection.HOLD,
            score=0.0,
            confidence=0.0,
            expected_edge_bps=0.0,
            expected_horizon_seconds=0,
            regime=regime_diag.regime,
            regime_confidence=regime_diag.confidence,
            selected_methodology="",
            contributing=(),
            reason_codes=tuple(dict.fromkeys(reason_codes)),
            blocks_buy=True,
            diagnostics={"regime": regime_diag.as_dict()},
        )

    def _hold(self, features, regime_diag, signals, reason_codes, vwap_signal) -> CompositeTechnicalSignal:
        return CompositeTechnicalSignal(
            symbol=features.symbol,
            direction=SignalDirection.HOLD,
            score=0.0,
            confidence=0.0,
            expected_edge_bps=0.0,
            expected_horizon_seconds=0,
            regime=regime_diag.regime,
            regime_confidence=regime_diag.confidence,
            selected_methodology="",
            contributing=(),
            reason_codes=tuple(dict.fromkeys(reason_codes)),
            blocks_buy=False,
            diagnostics={
                "regime": regime_diag.as_dict(),
                "vwap_confirms": bool(vwap_signal.available and vwap_signal.score > 0),
            },
        )

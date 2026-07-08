"""Rule-based short-term market regime classifier.

Transparent, deterministic scoring — no ML (that comes only after feature
correctness is proven). Every result carries the winning regime, a confidence
in [0, 1], human-readable reasons, and per-feature contributions so a human can
audit exactly why a regime was chosen. Advisory only: a regime never authorizes
a trade; it steers which methodologies are *preferred* and whether BUY is
*blocked* (risk regimes), while RiskManager/ProfitabilityGate stay authoritative.

The classifier consumes an already-computed :class:`RegimeInput` (built from
``app.technical.indicators`` or the live feature frame) rather than raw bars, so
it is decoupled from any particular data source and trivially testable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class MarketRegime(str, Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE_BOUND = "RANGE_BOUND"
    BREAKOUT_CANDIDATE = "BREAKOUT_CANDIDATE"
    MEAN_REVERSION_CANDIDATE = "MEAN_REVERSION_CANDIDATE"
    HIGH_VOLATILITY_RISK = "HIGH_VOLATILITY_RISK"
    LOW_LIQUIDITY_RISK = "LOW_LIQUIDITY_RISK"
    NO_TRADE = "NO_TRADE"


# Regimes in which a BUY must be blocked outright (capital-protection first).
BUY_BLOCKING_REGIMES = frozenset(
    {MarketRegime.HIGH_VOLATILITY_RISK, MarketRegime.LOW_LIQUIDITY_RISK, MarketRegime.NO_TRADE}
)


@dataclass(frozen=True)
class RegimeInput:
    """Computed features for a single symbol at decision time.

    Every field is optional and NaN-safe (``None`` == unavailable). The
    classifier degrades gracefully and reports missing inputs rather than
    fabricating values.
    """

    symbol: str = ""
    price: float | None = None
    ema_fast: float | None = None
    ema_slow: float | None = None
    macd_histogram: float | None = None
    rsi: float | None = None
    bb_percent_b: float | None = None
    bb_bandwidth: float | None = None
    vwap_distance_bps: float | None = None
    breakout_strength: float | None = None  # (price - donchian_high) / price, signed
    donchian_low_distance: float | None = None  # (price - donchian_low) / price, signed
    volume_spike_ratio: float | None = None
    atr_pct: float | None = None  # ATR / price
    realized_volatility: float | None = None  # per-bar stdev of returns
    short_return: float | None = None  # recent rolling return (e.g. 1m)
    liquidity_score: float | None = None  # [0, 1]
    spread_bps: float | None = None
    orderbook_imbalance: float | None = None  # [-1, 1]


@dataclass(frozen=True)
class RegimeConfig:
    # Risk gates (checked first).
    min_liquidity_score: float = 0.35
    max_spread_bps: float = 50.0
    high_realized_volatility: float = 0.015  # per-bar return stdev
    high_atr_pct: float = 0.02
    high_bandwidth: float = 0.06
    # Trend / range thresholds.
    trend_ema_gap_bps: float = 8.0  # |ema_fast - ema_slow| / price in bps
    range_ema_gap_bps: float = 4.0
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    rsi_neutral_band: float = 8.0  # within +/- of 50 == neutral
    # Breakout thresholds.
    breakout_min_strength: float = 0.0  # price at/above donchian high
    breakout_min_volume_spike: float = 1.5
    # Data sufficiency: minimum count of core features present.
    min_core_features: int = 4

    @classmethod
    def from_env(cls) -> "RegimeConfig":
        def _f(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None or raw.strip() == "":
                return default
            try:
                return float(raw)
            except ValueError:
                return default

        return cls(
            min_liquidity_score=_f("TECHNICAL_REGIME_MIN_LIQUIDITY", cls.min_liquidity_score),
            max_spread_bps=_f("TECHNICAL_REGIME_MAX_SPREAD_BPS", cls.max_spread_bps),
            high_realized_volatility=_f("TECHNICAL_REGIME_HIGH_RV", cls.high_realized_volatility),
            high_atr_pct=_f("TECHNICAL_REGIME_HIGH_ATR_PCT", cls.high_atr_pct),
            high_bandwidth=_f("TECHNICAL_REGIME_HIGH_BANDWIDTH", cls.high_bandwidth),
        )


@dataclass(frozen=True)
class RegimeDiagnostics:
    symbol: str
    regime: MarketRegime
    confidence: float
    reasons: tuple[str, ...]
    feature_contributions: Mapping[str, float]
    scores: Mapping[str, float]
    missing_features: tuple[str, ...]
    blocks_buy: bool

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "regime": self.regime.value,
            "confidence": round(self.confidence, 4),
            "reasons": list(self.reasons),
            "feature_contributions": {k: round(v, 4) for k, v in self.feature_contributions.items()},
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "missing_features": list(self.missing_features),
            "blocks_buy": self.blocks_buy,
        }


_CORE_FEATURES = (
    "price",
    "ema_fast",
    "ema_slow",
    "macd_histogram",
    "rsi",
    "bb_percent_b",
    "vwap_distance_bps",
    "short_return",
)


class TechnicalRegimeClassifier:
    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()

    def classify(self, features: RegimeInput) -> RegimeDiagnostics:
        cfg = self.config
        missing = tuple(name for name in _CORE_FEATURES if getattr(features, name) is None)
        present_core = len(_CORE_FEATURES) - len(missing)

        # ---- Risk gates first (capital protection precedes opportunity) ---- #
        if features.liquidity_score is not None and features.liquidity_score < cfg.min_liquidity_score:
            return self._risk(
                features,
                MarketRegime.LOW_LIQUIDITY_RISK,
                f"liquidity_score {features.liquidity_score:.2f} < {cfg.min_liquidity_score:.2f}",
                {"liquidity_score": features.liquidity_score},
                missing,
            )
        if features.spread_bps is not None and features.spread_bps > cfg.max_spread_bps:
            return self._risk(
                features,
                MarketRegime.LOW_LIQUIDITY_RISK,
                f"spread {features.spread_bps:.1f}bps > {cfg.max_spread_bps:.1f}bps",
                {"spread_bps": -features.spread_bps},
                missing,
            )
        vol_reasons: list[str] = []
        if features.realized_volatility is not None and features.realized_volatility > cfg.high_realized_volatility:
            vol_reasons.append(f"realized_vol {features.realized_volatility:.4f} > {cfg.high_realized_volatility:.4f}")
        if features.atr_pct is not None and features.atr_pct > cfg.high_atr_pct:
            vol_reasons.append(f"atr_pct {features.atr_pct:.4f} > {cfg.high_atr_pct:.4f}")
        if features.bb_bandwidth is not None and features.bb_bandwidth > cfg.high_bandwidth:
            vol_reasons.append(f"bb_bandwidth {features.bb_bandwidth:.4f} > {cfg.high_bandwidth:.4f}")
        if vol_reasons:
            return self._risk(
                features,
                MarketRegime.HIGH_VOLATILITY_RISK,
                "; ".join(vol_reasons),
                {"volatility": -1.0},
                missing,
            )

        # ---- Data sufficiency ---- #
        if present_core < cfg.min_core_features:
            return RegimeDiagnostics(
                symbol=features.symbol,
                regime=MarketRegime.NO_TRADE,
                confidence=0.0,
                reasons=(f"insufficient features: {present_core}/{len(_CORE_FEATURES)} core present",),
                feature_contributions={},
                scores={},
                missing_features=missing,
                blocks_buy=True,
            )

        # ---- Directional / structural scoring ---- #
        scores, contributions, reasons = self._score(features)
        # Pick the winner.
        best_regime = max(scores, key=lambda r: scores[r])
        best_score = scores[best_regime]
        if best_score <= 0.0:
            regime = MarketRegime.NO_TRADE
            confidence = 0.0
            reasons = reasons + ("no positive directional/structural evidence",)
            blocks_buy = True
        else:
            regime = best_regime
            # Confidence = winner's share of total positive score, tempered by margin.
            positive_total = sum(max(0.0, s) for s in scores.values()) or 1.0
            share = best_score / positive_total
            ordered = sorted((max(0.0, s) for s in scores.values()), reverse=True)
            margin = (ordered[0] - ordered[1]) / ordered[0] if len(ordered) > 1 and ordered[0] > 0 else 1.0
            confidence = max(0.0, min(1.0, 0.5 * share + 0.5 * margin))
            blocks_buy = regime in BUY_BLOCKING_REGIMES

        return RegimeDiagnostics(
            symbol=features.symbol,
            regime=regime,
            confidence=confidence,
            reasons=tuple(reasons),
            feature_contributions=contributions,
            scores={r.value: s for r, s in scores.items()},
            missing_features=missing,
            blocks_buy=blocks_buy,
        )

    # ------------------------------------------------------------------ #
    def _score(
        self, f: RegimeInput
    ) -> tuple[dict[MarketRegime, float], dict[str, float], tuple[str, ...]]:
        cfg = self.config
        scores: dict[MarketRegime, float] = {
            MarketRegime.TREND_UP: 0.0,
            MarketRegime.TREND_DOWN: 0.0,
            MarketRegime.RANGE_BOUND: 0.0,
            MarketRegime.BREAKOUT_CANDIDATE: 0.0,
            MarketRegime.MEAN_REVERSION_CANDIDATE: 0.0,
        }
        contrib: dict[str, float] = {}
        reasons: list[str] = []

        # EMA gap (trend vs range).
        ema_gap_bps = None
        if f.ema_fast is not None and f.ema_slow is not None and f.price:
            ema_gap_bps = (f.ema_fast - f.ema_slow) / f.price * 10_000.0
            contrib["ema_gap_bps"] = ema_gap_bps
            if ema_gap_bps >= cfg.trend_ema_gap_bps:
                scores[MarketRegime.TREND_UP] += 1.0
                reasons.append(f"ema_fast above ema_slow by {ema_gap_bps:.1f}bps")
            elif ema_gap_bps <= -cfg.trend_ema_gap_bps:
                scores[MarketRegime.TREND_DOWN] += 1.0
                reasons.append(f"ema_fast below ema_slow by {abs(ema_gap_bps):.1f}bps")
            elif abs(ema_gap_bps) <= cfg.range_ema_gap_bps:
                scores[MarketRegime.RANGE_BOUND] += 1.0
                reasons.append("ema gap flat (range)")

        # MACD histogram.
        if f.macd_histogram is not None:
            contrib["macd_histogram"] = f.macd_histogram
            if f.macd_histogram > 0:
                scores[MarketRegime.TREND_UP] += 0.6
            elif f.macd_histogram < 0:
                scores[MarketRegime.TREND_DOWN] += 0.6

        # VWAP position.
        if f.vwap_distance_bps is not None:
            contrib["vwap_distance_bps"] = f.vwap_distance_bps
            if f.vwap_distance_bps > 0:
                scores[MarketRegime.TREND_UP] += 0.5
                scores[MarketRegime.BREAKOUT_CANDIDATE] += 0.3
            else:
                scores[MarketRegime.TREND_DOWN] += 0.5

        # Short-horizon momentum.
        if f.short_return is not None:
            contrib["short_return"] = f.short_return
            if f.short_return > 0:
                scores[MarketRegime.TREND_UP] += 0.4
            elif f.short_return < 0:
                scores[MarketRegime.TREND_DOWN] += 0.4

        # RSI: extremes -> reversion candidate; midband -> range.
        if f.rsi is not None:
            contrib["rsi"] = f.rsi
            if f.rsi <= cfg.rsi_oversold:
                scores[MarketRegime.MEAN_REVERSION_CANDIDATE] += 0.9
                reasons.append(f"rsi {f.rsi:.0f} oversold")
            elif f.rsi >= cfg.rsi_overbought:
                scores[MarketRegime.TREND_UP] += 0.2  # strong momentum, but reversion risk
                reasons.append(f"rsi {f.rsi:.0f} overbought")
            elif abs(f.rsi - 50.0) <= cfg.rsi_neutral_band:
                scores[MarketRegime.RANGE_BOUND] += 0.5

        # Bollinger %b extremes support reversion.
        if f.bb_percent_b is not None:
            contrib["bb_percent_b"] = f.bb_percent_b
            if f.bb_percent_b <= 0.05:
                scores[MarketRegime.MEAN_REVERSION_CANDIDATE] += 0.6
            elif f.bb_percent_b >= 0.95:
                scores[MarketRegime.BREAKOUT_CANDIDATE] += 0.4
            elif 0.4 <= f.bb_percent_b <= 0.6:
                scores[MarketRegime.RANGE_BOUND] += 0.3

        # Breakout: price at/above recent range high with volume confirmation.
        if f.breakout_strength is not None:
            contrib["breakout_strength"] = f.breakout_strength
            if f.breakout_strength >= cfg.breakout_min_strength:
                spike_ok = (
                    f.volume_spike_ratio is not None
                    and f.volume_spike_ratio >= cfg.breakout_min_volume_spike
                )
                if spike_ok:
                    # A confirmed range break with volume is the defining, highest-
                    # conviction setup — weight it to dominate the overlapping trend
                    # evidence so the regime resolves to BREAKOUT_CANDIDATE.
                    scores[MarketRegime.BREAKOUT_CANDIDATE] += 2.0
                    reasons.append(
                        f"price at range high with volume x{f.volume_spike_ratio:.1f}"
                    )
                else:
                    scores[MarketRegime.BREAKOUT_CANDIDATE] += 0.3
                    reasons.append("price at range high but volume unconfirmed")

        return scores, contrib, tuple(reasons)

    def _risk(
        self,
        f: RegimeInput,
        regime: MarketRegime,
        reason: str,
        contributions: dict[str, float],
        missing: tuple[str, ...],
    ) -> RegimeDiagnostics:
        return RegimeDiagnostics(
            symbol=f.symbol,
            regime=regime,
            confidence=0.9,
            reasons=(reason,),
            feature_contributions=contributions,
            scores={regime.value: 1.0},
            missing_features=missing,
            blocks_buy=True,
        )

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


# Every name here must describe MARKET STATE, not WHICH INSTRUMENT this is.
#
# A feature whose value is essentially constant per symbol is an identity label
# wearing a number's clothes. Because top-k is selected by absolute probability
# across a pooled cross-section, one such feature is enough to pin the top of the
# ranking to a single instrument forever, and precision@k then measures that one
# instrument's luck in the holdout window instead of model skill.
#
# Measured on 2026-08-05 (between-symbol variance / total variance; 1.0 = pure
# identity). ``bid_depth`` 0.986 and ``ask_depth`` 0.984 are raw share counts, so
# an ETF quoted by a liquidity provider sat at z=+4.03 and its MINIMUM depth still
# outranked almost every stock's maximum: 44 of the top 53 predictions were that
# one ticker, top-k net return was -51bps, and precision@k was 0.148 against a
# 0.395 base rate — while deciles 2-10 ranked perfectly monotonically (+26bps down
# to -43bps). The model was fine; the tip of its ranking was one instrument.
#
# Dropping the eight identity features on the same holdout split: precision@k
# 0.148 -> 0.444, top-k net -50.96 -> +18.32bps, AUC 0.727 -> 0.736, top-k symbol
# concentration 44/54 -> 13/54. No information is lost because the scale-free
# counterpart of each was ALREADY here: depth_ratio (0.208) for the depths,
# box_position (0.211) / box_width_pct / breakout_distance_bps for the box levels,
# orderbook_imbalance (0.436) alongside the depth ratio.
#
# Before adding a feature, check this ratio. Raw prices, raw sizes, and raw
# per-symbol levels belong in the strategy layer, which compares a symbol against
# ITSELF; they do not belong in a pooled cross-sectional regressor.
LIVE_FEATURE_NAMES: tuple[str, ...] = (
    # Tick/microstructure features. These are deliberately separate from the
    # 30s+ features below so the model can react to entry-timing changes without
    # pretending that a minute aggregate is a real-time signal.
    "return_1s",
    "return_5s",
    "return_10s",
    "tick_count_1s",
    "tick_count_5s",
    "volume_1s_log",
    "volume_5s_log",
    "aggressor_imbalance_5s",
    "realized_volatility_10s",
    "spread_change_5s",
    "orderbook_imbalance_change_5s",
    "second_data_ready",
    "return_30s",
    "return_1m",
    "return_3m",
    "distance_from_vwap",
    "spread_bps",
    "orderbook_imbalance",
    # bid_depth / ask_depth removed: raw share counts, identity ratio 0.986/0.984.
    # depth_ratio below is the scale-free form and stays.
    "depth_ratio",
    # liquidity_score (0.891) and realized_volatility_3m (0.779) removed for the
    # same reason. Both remain COMPUTED and are still consumed by the profitability
    # gate, the strategy supervisor, and the bandit, which compare a symbol against
    # its own history rather than pooling across instruments.
    "max_drop_3m",
    "cost_to_volatility_ratio",
    "principal_cushion_ratio",
    # Recency-decayed local-LLM news sentiment in [-1, 1] as of the decision time
    # (0.0 = no fresh news). Lets the real-time numeric learner weigh news by
    # realized outcomes. Adding this bumps schema_hash, so existing artifacts are
    # retired and a fresh one retrains — expected and safe (model is advisory).
    "news_sentiment",
    # Evidence-based technical indicators (Phase 4 of the technical prediction
    # layer). Computed from the same realtime tick series already used above and
    # emitted with NEUTRAL finite defaults on short data (rsi 50, %b 0.5, ratios
    # 0/1), so they never make a live frame fail validation. Adding these bumps
    # schema_hash: prior artifacts retire and a fresh model retrains — the
    # designed, advisory-safe flow (see live_short_horizon_v3_seconds below).
    "rsi_14",
    "macd_histogram",
    "bollinger_percent_b",
    "ema_gap_bps",
    "donchian_breakout",
    "volume_spike_ratio",
    # Completed one-minute-bar RVGI/box context. Availability flags prevent
    # numeric model tensors from disguising missing slow history as a signal.
    "rvgi_available",
    "rvgi",
    "rvgi_signal",
    "rvgi_diff",
    "rvgi_slope",
    "rvgi_bullish_cross",
    "box_available",
    # box_high / box_low / box_mid / box_previous_close removed: raw price levels,
    # identity ratio 0.943. The box is still built and still drives the box-breakout
    # strategies; only its ABSOLUTE levels leave the model vector. What the model
    # needs from a box is where price sits inside it and how wide it is, which the
    # three scale-free columns below already carry.
    "box_width_pct",
    "box_position",
    "breakout_distance_bps",
    "box_context_timestamp_epoch",
)


@dataclass(frozen=True)
class FeatureSchema:
    version: str
    feature_names: tuple[str, ...]
    dtypes: tuple[str, ...]
    missing_policy: str = "reject"
    source_requirements: tuple[str, ...] = ("kis_realtime_websocket",)

    @property
    def schema_hash(self) -> str:
        payload = {
            "version": self.version,
            "feature_names": self.feature_names,
            "dtypes": self.dtypes,
            "missing_policy": self.missing_policy,
            "source_requirements": self.source_requirements,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


LIVE_SHORT_HORIZON_SCHEMA = FeatureSchema(
    # v5 removes the eight instrument-identity columns documented above. The version
    # string is part of the schema_hash, so every v4 artifact retires and a fresh
    # model retrains — required here, because a v4 artifact scored with v5 inputs
    # would silently read the wrong column for every weight.
    version="live_short_horizon_v5_state_only",
    feature_names=LIVE_FEATURE_NAMES,
    dtypes=tuple("float64" for _ in LIVE_FEATURE_NAMES),
)

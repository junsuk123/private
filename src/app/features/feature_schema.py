from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


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
    "bid_depth",
    "ask_depth",
    "depth_ratio",
    "liquidity_score",
    "realized_volatility_3m",
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
    # v3 adds true 1/5/10-second microstructure columns. The version string is
    # part of the schema_hash, so prior artifacts cannot be mistaken for a model
    # trained on these new inputs.
    version="live_short_horizon_v3_seconds",
    feature_names=LIVE_FEATURE_NAMES,
    dtypes=tuple("float64" for _ in LIVE_FEATURE_NAMES),
)

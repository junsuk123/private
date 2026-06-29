from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


LIVE_FEATURE_NAMES: tuple[str, ...] = (
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
    version="live_short_horizon_v1",
    feature_names=LIVE_FEATURE_NAMES,
    dtypes=tuple("float64" for _ in LIVE_FEATURE_NAMES),
)

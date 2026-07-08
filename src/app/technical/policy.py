"""Loader for config/technical_prediction_policy.yaml.

Builds the regime / signal-engine / prediction configs from YAML, applying
environment-variable overrides (the individual config dataclasses read their
own env vars via ``from_env``) and logging the effective values. Missing file
or missing keys fall back to the dataclass defaults — the layer is fully
functional with no config present.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from app.technical.prediction import PredictionConfig
from app.technical.regime import RegimeConfig
from app.technical.signals import SignalEngineConfig

logger = logging.getLogger(__name__)

DEFAULT_POLICY_PATH = "config/technical_prediction_policy.yaml"


@dataclass(frozen=True)
class TechnicalPolicy:
    enabled: bool = True
    default_horizons_seconds: tuple[int, ...] = (5, 15, 30, 60, 300)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    signal_engine: SignalEngineConfig = field(default_factory=SignalEngineConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    log_each_decision: bool = True
    persist_signal_snapshots: bool = True
    expose_gui_payload: bool = True


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def load_technical_policy(path: str | Path | None = None) -> TechnicalPolicy:
    resolved = Path(path or os.getenv("TECHNICAL_PREDICTION_POLICY_PATH", DEFAULT_POLICY_PATH))
    raw: Mapping[str, Any] = {}
    if resolved.exists():
        try:
            loaded = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, Mapping):
                raw = loaded
        except (OSError, yaml.YAMLError) as exc:  # pragma: no cover - defensive
            logger.warning("technical policy load failed (%s); using defaults", exc)

    # Env overrides take precedence inside each from_env(); YAML fills the rest.
    regime_env = RegimeConfig.from_env()
    regime_yaml = raw.get("regime") or {}
    regime = RegimeConfig(
        min_liquidity_score=float(regime_yaml.get("min_liquidity_score", regime_env.min_liquidity_score)),
        max_spread_bps=float(regime_yaml.get("max_spread_bps", regime_env.max_spread_bps)),
        high_realized_volatility=float(regime_yaml.get("high_realized_volatility", regime_env.high_realized_volatility)),
        high_atr_pct=float(regime_yaml.get("high_atr_pct", regime_env.high_atr_pct)),
        high_bandwidth=float(regime_yaml.get("high_bandwidth", regime_env.high_bandwidth)),
    )
    # Env still wins where set.
    if os.getenv("TECHNICAL_REGIME_MIN_LIQUIDITY"):
        regime = RegimeConfig(**{**regime.__dict__, "min_liquidity_score": regime_env.min_liquidity_score})

    weights = dict(SignalEngineConfig().methodology_weights)
    weights.update({str(k): float(v) for k, v in (raw.get("methodology_weights") or {}).items()})
    min_conf = dict(SignalEngineConfig().minimum_confidence)
    min_conf.update({str(k): float(v) for k, v in (raw.get("minimum_confidence") or {}).items()})
    risk_blocks = raw.get("risk_blocks") or {}
    pred_yaml = raw.get("prediction") or {}
    signal_engine = SignalEngineConfig(
        methodology_weights=weights,
        minimum_confidence=min_conf,
        require_vwap_confirmation=True,
        block_mean_reversion_in_downtrend=_as_bool(
            risk_blocks.get("block_mean_reversion_in_strong_downtrend"), True
        ),
        block_breakout_without_volume=_as_bool(
            risk_blocks.get("block_breakout_without_volume_confirmation"), True
        ),
        spread_alpha_max_ratio=float(pred_yaml.get("spread_alpha_max_ratio", SignalEngineConfig().spread_alpha_max_ratio)),
    )

    prediction_env = PredictionConfig.from_env()
    horizon_buffers = {
        int(k): float(v) for k, v in (pred_yaml.get("horizon_edge_buffer_bps") or {}).items()
    }
    prediction = PredictionConfig(
        min_confidence=float(pred_yaml.get("min_confidence", prediction_env.min_confidence)),
        min_net_return_bps=float(pred_yaml.get("min_net_return_bps", prediction_env.min_net_return_bps)),
        horizon_edge_buffer_bps=horizon_buffers,
    )

    diagnostics = raw.get("diagnostics") or {}
    horizons = raw.get("default_horizons_seconds") or [5, 15, 30, 60, 300]

    policy = TechnicalPolicy(
        enabled=_as_bool(raw.get("enabled"), True),
        default_horizons_seconds=tuple(int(h) for h in horizons),
        regime=regime,
        signal_engine=signal_engine,
        prediction=prediction,
        log_each_decision=_as_bool(diagnostics.get("log_each_decision"), True),
        persist_signal_snapshots=_as_bool(diagnostics.get("persist_signal_snapshots"), True),
        expose_gui_payload=_as_bool(diagnostics.get("expose_gui_payload"), True),
    )
    logger.info(
        "technical policy loaded (enabled=%s, min_liquidity=%.2f, min_confidence=%.2f, weights=%s)",
        policy.enabled,
        policy.regime.min_liquidity_score,
        policy.prediction.min_confidence,
        dict(policy.signal_engine.methodology_weights),
    )
    return policy


def build_prediction_engine(policy: TechnicalPolicy | None = None):
    """Construct a :class:`TechnicalPredictionEngine` wired to the loaded policy."""
    from app.technical.prediction import TechnicalPredictionEngine
    from app.technical.regime import TechnicalRegimeClassifier
    from app.technical.signals import CompositeTechnicalSignalEngine

    policy = policy or load_technical_policy()
    signal_engine = CompositeTechnicalSignalEngine(
        policy.signal_engine,
        regime_classifier=TechnicalRegimeClassifier(policy.regime),
    )
    return TechnicalPredictionEngine(signal_engine=signal_engine, config=policy.prediction)

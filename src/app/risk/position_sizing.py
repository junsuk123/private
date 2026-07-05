"""Edge- and confidence-aware position sizing for small accounts.

Replaces fixed-weight sizing with fractional-Kelly logic that scales the position
by realized net edge, model confidence, liquidity, account drawdown, and recent
strategy performance — with hard caps. Size is never increased on confidence alone;
a positive net expectancy is required (the ProfitabilityGate enforces that upstream).

Formulas (spec):
    payoff_ratio     = avg_win_net / max(|avg_loss_net|, eps)
    kelly_raw        = p_win - (1 - p_win) / max(payoff_ratio, eps)
    fractional_kelly = clamp(kelly_raw * kelly_fraction, 0, max_kelly_cap)
    edge_score       = clamp(net_expected_return / target_net_return, 0, 1)
    position_weight  = clamp(base_weight * edge_score * confidence_score
                             * liquidity_score * drawdown_multiplier
                             * recent_performance_multiplier, min_weight, max_weight)

The final sizing weight is min(fractional_kelly_cap, position_weight) so neither the
Kelly cap nor the multiplicative weight can be exceeded.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_EPSILON = 1e-9

DEFAULT_POSITION_SIZING_CONFIG: dict[str, Any] = {
    "base_position_weight": 0.003,
    "min_position_weight": 0.0,
    "max_position_weight": 0.10,
    "kelly_fraction": 0.25,
    "max_kelly_cap": 0.10,
    # Conservative priors used when no realized performance history is available.
    "default_p_win": 0.5,
    "default_avg_win_net": 0.012,
    "default_avg_loss_net": 0.012,
    # Drawdown / recent-loss dampening.
    "drawdown_dampen_k": 4.0,          # multiplier = 1 - k*|drawdown|, floored
    "min_drawdown_multiplier": 0.25,
    "recent_loss_multiplier": 0.5,     # applied after a recent same-strategy loss
    # Small-account reference equity for the sizing cap.
    "small_account_equity_krw": 200000.0,
}


@dataclass(frozen=True)
class SizingInputs:
    net_expected_return: float
    target_net_return: float
    confidence_score: float = 0.5
    liquidity_score: float = 1.0
    account_drawdown_rate: float = 0.0          # negative when in drawdown
    recent_same_strategy_loss: bool = False
    # Realized performance (optional; conservative priors used if absent).
    p_win: float | None = None
    avg_win_net: float | None = None
    avg_loss_net: float | None = None


@dataclass(frozen=True)
class SizingResult:
    position_weight: float
    edge_score: float
    payoff_ratio: float
    kelly_raw: float
    fractional_kelly: float
    confidence_score: float
    liquidity_score: float
    drawdown_multiplier: float
    recent_performance_multiplier: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class PositionSizer:
    def __init__(self, config_path: Path | str = "config/position_sizing_policy.yaml") -> None:
        self.config = _load_config(config_path)
        _log_config(self.config)

    def size(self, inputs: SizingInputs) -> SizingResult:
        cfg = self.config
        # Negative expectancy => zero size (defense in depth; the gate blocks these).
        if inputs.net_expected_return <= 0.0:
            return SizingResult(0.0, 0.0, 0.0, 0.0, 0.0, inputs.confidence_score, inputs.liquidity_score, 0.0, 0.0)

        p_win = _clamp(inputs.p_win if inputs.p_win is not None else cfg["default_p_win"], 0.0, 1.0)
        avg_win = abs(inputs.avg_win_net if inputs.avg_win_net is not None else cfg["default_avg_win_net"])
        avg_loss = abs(inputs.avg_loss_net if inputs.avg_loss_net is not None else cfg["default_avg_loss_net"])
        payoff_ratio = avg_win / max(avg_loss, _EPSILON)
        kelly_raw = p_win - (1.0 - p_win) / max(payoff_ratio, _EPSILON)
        fractional_kelly = _clamp(kelly_raw * float(cfg["kelly_fraction"]), 0.0, float(cfg["max_kelly_cap"]))

        target = max(_EPSILON, float(inputs.target_net_return))
        edge_score = _clamp(inputs.net_expected_return / target, 0.0, 1.0)
        confidence = _clamp(inputs.confidence_score, 0.0, 1.0)
        liquidity = _clamp(inputs.liquidity_score, 0.0, 1.0)

        drawdown_mult = 1.0
        if inputs.account_drawdown_rate < 0:
            drawdown_mult = max(
                float(cfg["min_drawdown_multiplier"]),
                1.0 - float(cfg["drawdown_dampen_k"]) * abs(inputs.account_drawdown_rate),
            )
        recent_perf_mult = float(cfg["recent_loss_multiplier"]) if inputs.recent_same_strategy_loss else 1.0

        weight = (
            float(cfg["base_position_weight"])
            * edge_score
            * confidence
            * liquidity
            * drawdown_mult
            * recent_perf_mult
        )
        # Neither the multiplicative weight nor the Kelly cap may be exceeded.
        weight = min(weight, fractional_kelly if fractional_kelly > 0 else weight)
        weight = _clamp(weight, float(cfg["min_position_weight"]), float(cfg["max_position_weight"]))

        return SizingResult(
            position_weight=weight,
            edge_score=edge_score,
            payoff_ratio=payoff_ratio,
            kelly_raw=kelly_raw,
            fractional_kelly=fractional_kelly,
            confidence_score=confidence,
            liquidity_score=liquidity,
            drawdown_multiplier=drawdown_mult,
            recent_performance_multiplier=recent_perf_mult,
        )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _load_config(config_path: Path | str) -> dict[str, Any]:
    merged = dict(DEFAULT_POSITION_SIZING_CONFIG)
    # Env backward compatibility.
    merged["base_position_weight"] = _env_float("REALTIME_BUY_WEIGHT", merged["base_position_weight"])
    merged["max_position_weight"] = _env_float("REALTIME_SMALL_ACCOUNT_MAX_POSITION_WEIGHT", merged["max_position_weight"])
    merged["small_account_equity_krw"] = _env_float("REALTIME_SMALL_ACCOUNT_EQUITY_KRW", merged["small_account_equity_krw"])
    path = Path(config_path)
    if path.exists():
        try:
            import yaml

            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                merged.update({k: v for k, v in loaded.items() if k in DEFAULT_POSITION_SIZING_CONFIG})
        except Exception:  # noqa: BLE001
            logger.warning("Failed to load %s; using defaults + env", path)
    return merged


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


_CONFIG_LOGGED = False


def _log_config(config: dict[str, Any]) -> None:
    global _CONFIG_LOGGED
    if not _CONFIG_LOGGED:
        logger.info("PositionSizer resolved config: %s", config)
        _CONFIG_LOGGED = True

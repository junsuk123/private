"""Unified dynamic exit-policy resolver.

The audit found exit thresholds scattered across ~12 sources (RealtimeTradingConfig,
adaptive_exit_policy, ~15 REALTIME_* env vars read inline in the exit path, plus
ExecutionPolicy defaults) with several conflicts. This module is the single
authoritative resolver: it produces one :class:`ResolvedExitLevels` object per
holding, applying the spec's dynamic formulas, and LOGS the resolved values so the
effective policy is always auditable.

Resolution precedence (highest wins), logged once per process:
    1. Explicit environment variable (backward compatible with the existing runtime)
    2. Dynamic formula default (cost + volatility + liquidity + spread aware)
    3. Built-in constant

Exit levels are dynamic — take-profit, profit-lock, trailing-giveback and the
soft-stop depend on the break-even cost (from the cost engine), realized short-horizon
volatility, spread, liquidity, model downside risk, and account drawdown. Hard-stop and
emergency-stop are capital circuit-breakers and stay near their configured constants.

Loss-exit permission is NOT "always block" or "always allow": it is granted only when
deterioration evidence is strong (hard/emergency-stop breach, strongly negative net
forecast, SELL/REDUCE ontology dominance, sharp liquidity/spread deterioration, or the
daily loss budget nearing breach), and blocked for noise-level losses.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_DYNAMIC_EXIT_CONFIG: dict[str, Any] = {
    # Take-profit floor and dynamic coefficients.
    "min_take_profit_rate": 0.008,       # REALTIME_QUICK_TAKE_PROFIT_NET
    "min_net_profit_buffer": 0.004,      # REALTIME_MIN_NET_PROFIT_EXIT
    "k_vol_take_profit": 1.0,            # take_profit += k_vol * realized_volatility
    "liquidity_take_profit_buffer": 0.002,
    "spread_take_profit_buffer_k": 1.0,  # take_profit += k * spread_rate
    # Profit lock / trailing.
    "profit_lock_arm_net": 0.010,        # REALTIME_PROFIT_LOCK_ARM_NET
    "min_trailing_giveback": 0.35,       # REALTIME_PROFIT_LOCK_GIVEBACK (old inline default)
    "k_trail_volatility": 5.0,           # trailing_giveback += k_trail * volatility
    "profit_time_exit_sec": 300.0,       # REALTIME_PROFIT_TIME_EXIT_SEC
    # Stops. Defaults match the previous inline defaults for exact backward parity when
    # the env var is unset; run.ps1 pins the production values (hard 0.03 / emergency 0.035).
    "stop_loss_net": 0.0,                # REALTIME_STOP_LOSS_NET (0 = off by default)
    "min_soft_stop_rate": 0.006,
    "k_downside_soft_stop": 1.0,         # soft_stop += k_downside * predicted_downside_risk
    "hard_stop_loss_rate": 0.08,         # REALTIME_HARD_STOP_LOSS (old inline default)
    "emergency_stop_loss_rate": 0.05,    # REALTIME_EMERGENCY_STOP_LOSS (old inline default)
    # Loss-exit governance.
    "allow_loss_exit": False,            # REALTIME_ALLOW_LOSS_EXIT
    "block_sell_below_breakeven": False, # REALTIME_BLOCK_SELL_BELOW_BREAKEVEN (old inline default)
    # Deterioration thresholds that PERMIT a controlled loss exit.
    "ontology_sell_dominance": -0.55,    # ontology_score <= this permits loss exit
    "strong_negative_forecast_bps": 8.0, # model expected_net_return_bps <= -this permits exit
    "noise_band_loss_rate": 0.004,       # losses within this band are noise (blocked)
}


@dataclass(frozen=True)
class ResolvedExitLevels:
    """Fully resolved, per-holding exit levels. All rates are fractions (0.01 == 1%)."""

    take_profit_rate: float
    quick_take_profit_net: float
    min_net_profit_exit: float
    profit_lock_arm_net: float
    trailing_giveback_rate: float
    profit_time_exit_sec: float
    stop_loss_net: float
    soft_stop_rate: float
    hard_stop_rate: float
    emergency_stop_rate: float
    allow_loss_exit: bool
    block_sell_below_breakeven: bool
    ontology_sell_dominance: float
    strong_negative_forecast_bps: float
    noise_band_loss_rate: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LossExitEvidence:
    """Signals used to decide whether a loss-realizing exit is permitted."""

    pnl_rate: float
    net_pnl_rate: float
    ontology_score: float = 0.0
    predicted_net_return_bps: float = 0.0
    spread_rate: float = 0.0
    liquidity_deteriorating: bool = False
    market_regime_high_risk: bool = False
    daily_loss_budget_near_breach: bool = False
    position_age_seconds: float | None = None


class DynamicExitPolicy:
    """Resolves the single authoritative exit policy for a holding."""

    def __init__(self, config_path: Path | str = "config/dynamic_exit_policy.yaml") -> None:
        self.config = _load_config(config_path)

    def resolve(
        self,
        *,
        all_in_cost_rate: float,
        realized_volatility: float = 0.0,
        spread_rate: float = 0.0,
        liquidity_score: float = 1.0,
        predicted_downside_risk: float = 0.0,
        account_drawdown_rate: float = 0.0,
    ) -> ResolvedExitLevels:
        cfg = self.config
        vol = max(0.0, float(realized_volatility))
        spread = max(0.0, float(spread_rate))
        liq_buffer = float(cfg["liquidity_take_profit_buffer"]) * (1.0 - max(0.0, min(1.0, liquidity_score)))

        # --- Dynamic take-profit (spec formula) --------------------------------
        min_tp = _resolve("REALTIME_QUICK_TAKE_PROFIT_NET", cfg["min_take_profit_rate"])
        min_net_profit = _resolve("REALTIME_MIN_NET_PROFIT_EXIT", cfg["min_net_profit_buffer"])
        dynamic_take_profit = max(
            min_tp,
            all_in_cost_rate
            + min_net_profit
            + float(cfg["k_vol_take_profit"]) * vol
            + liq_buffer
            + float(cfg["spread_take_profit_buffer_k"]) * spread,
        )

        # --- Profit lock / trailing giveback -----------------------------------
        profit_lock_arm = _resolve("REALTIME_PROFIT_LOCK_ARM_NET", cfg["profit_lock_arm_net"])
        trailing_giveback = max(
            _resolve("REALTIME_PROFIT_LOCK_GIVEBACK", cfg["min_trailing_giveback"]),
            float(cfg["k_trail_volatility"]) * vol,
        )
        trailing_giveback = min(0.95, trailing_giveback)

        # --- Stops -------------------------------------------------------------
        stop_loss_net = _resolve("REALTIME_STOP_LOSS_NET", cfg["stop_loss_net"])
        soft_stop = max(
            float(cfg["min_soft_stop_rate"]),
            float(cfg["k_downside_soft_stop"]) * max(0.0, float(predicted_downside_risk)) + vol,
        )
        hard_stop = _resolve("REALTIME_HARD_STOP_LOSS", cfg["hard_stop_loss_rate"])
        emergency_stop = _resolve("REALTIME_EMERGENCY_STOP_LOSS", cfg["emergency_stop_loss_rate"])
        # Account drawdown tightens the stops (defensive posture in a losing book).
        if account_drawdown_rate < 0:
            tighten = min(0.5, abs(account_drawdown_rate) * 5.0)
            soft_stop = max(float(cfg["min_soft_stop_rate"]), soft_stop * (1.0 - tighten))

        resolved = ResolvedExitLevels(
            take_profit_rate=dynamic_take_profit,
            quick_take_profit_net=min_tp,
            min_net_profit_exit=min_net_profit,
            profit_lock_arm_net=profit_lock_arm,
            trailing_giveback_rate=trailing_giveback,
            profit_time_exit_sec=_resolve("REALTIME_PROFIT_TIME_EXIT_SEC", cfg["profit_time_exit_sec"]),
            stop_loss_net=stop_loss_net,
            soft_stop_rate=soft_stop,
            hard_stop_rate=hard_stop,
            emergency_stop_rate=emergency_stop,
            allow_loss_exit=_resolve_bool("REALTIME_ALLOW_LOSS_EXIT", cfg["allow_loss_exit"]),
            block_sell_below_breakeven=_resolve_bool("REALTIME_BLOCK_SELL_BELOW_BREAKEVEN", cfg["block_sell_below_breakeven"]),
            ontology_sell_dominance=float(cfg["ontology_sell_dominance"]),
            strong_negative_forecast_bps=float(cfg["strong_negative_forecast_bps"]),
            noise_band_loss_rate=float(cfg["noise_band_loss_rate"]),
        )
        _log_resolved(resolved)
        return resolved

    def loss_exit_decision(
        self, levels: ResolvedExitLevels, evidence: LossExitEvidence
    ) -> tuple[bool, str]:
        """Decide whether a loss-realizing exit is permitted.

        Returns (allowed, reason_code). Hard/emergency breaches always allow (capital
        circuit-breaker). Otherwise a loss exit is allowed only on strong deterioration
        evidence, and blocked for noise-level losses.
        """
        pnl = evidence.pnl_rate
        # 1. Capital circuit-breakers always fire, regardless of allow_loss_exit.
        if levels.hard_stop_rate > 0.0 and pnl <= -levels.hard_stop_rate:
            return True, "hard_stop_loss"
        if levels.emergency_stop_rate > 0.0 and pnl <= -levels.emergency_stop_rate:
            return True, "emergency_stop_loss"
        # 2. Net tight stop (opt-in) fires regardless of allow_loss_exit.
        if levels.stop_loss_net > 0.0 and evidence.net_pnl_rate <= -levels.stop_loss_net:
            return True, "stop_loss_net"
        # 3. Everything below is a discretionary loss exit — requires allow_loss_exit.
        if not levels.allow_loss_exit:
            return False, "LOSS_EXIT_DISABLED"
        # 4. Noise-band losses are never realized on discretion.
        if abs(pnl) <= levels.noise_band_loss_rate:
            return False, "LOSS_WITHIN_NOISE_BAND"
        # 5. Strong deterioration evidence permits a controlled loss exit.
        if evidence.ontology_score <= levels.ontology_sell_dominance:
            return True, "ontology_sell_dominance"
        if evidence.predicted_net_return_bps <= -levels.strong_negative_forecast_bps:
            return True, "strong_negative_forecast"
        if evidence.liquidity_deteriorating:
            return True, "liquidity_deterioration"
        if evidence.market_regime_high_risk:
            return True, "market_regime_high_risk"
        if evidence.daily_loss_budget_near_breach:
            return True, "daily_loss_budget_near_breach"
        if pnl <= -levels.soft_stop_rate:
            return True, "soft_stop_loss"
        return False, "HOLD_INSUFFICIENT_DETERIORATION_EVIDENCE"


def _resolve(env_name: str, default: float) -> float:
    raw = os.getenv(env_name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _resolve_bool(env_name: str, default: bool) -> bool:
    raw = os.getenv(env_name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_RESOLVED_LOGGED = False


def _log_resolved(levels: ResolvedExitLevels) -> None:
    global _RESOLVED_LOGGED
    if not _RESOLVED_LOGGED:
        logger.info("DynamicExitPolicy resolved base levels: %s", levels.as_dict())
        _RESOLVED_LOGGED = True


def _load_config(config_path: Path | str) -> dict[str, Any]:
    merged = dict(DEFAULT_DYNAMIC_EXIT_CONFIG)
    path = Path(config_path)
    if not path.exists():
        return merged
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            merged.update({k: v for k, v in loaded.items() if k in DEFAULT_DYNAMIC_EXIT_CONFIG})
    except Exception:  # noqa: BLE001 - malformed config falls back to defaults + env.
        logger.warning("Failed to load %s; using defaults + env", path)
    return merged

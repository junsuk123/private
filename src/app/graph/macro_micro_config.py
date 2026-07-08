"""Loader for config/macro_micro_ontology.yaml.

Builds the macro / micro / coordinator configs from YAML, applying env-var
overrides and logging the effective values. Invalid values fall back to
conservative defaults and are recorded in ``diagnostics`` (never fail open).
Missing file -> full defaults (the layer is functional with no config present).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from app.graph.macro_reasoner import DEFAULT_STRATEGY_PERMISSIONS, MacroReasonerConfig
from app.graph.micro_reasoner import MicroReasonerConfig
from app.graph.ontology_coordinator import CoordinatorConfig

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config/macro_micro_ontology.yaml"


@dataclass(frozen=True)
class MacroMicroPolicy:
    enabled: bool = True
    macro_enabled: bool = True
    micro_enabled: bool = True
    macro_loop_interval_seconds: int = 60
    micro_loop_interval_seconds: int = 5
    macro_config: MacroReasonerConfig = field(default_factory=MacroReasonerConfig)
    micro_config: MicroReasonerConfig = field(default_factory=MicroReasonerConfig)
    coordinator_config: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    persist_snapshots: bool = True
    expose_dashboard_payload: bool = True
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _clamp_int(value: Any, default: int, *, lo: int, hi: int, fallbacks: list[str], name: str) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        fallbacks.append(f"{name}=invalid->default({default})")
        return default
    if v < lo or v > hi:
        fallbacks.append(f"{name}={v}->clamped[{lo},{hi}]")
        return max(lo, min(hi, v))
    return v


def _float(value: Any, default: float, *, fallbacks: list[str], name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        fallbacks.append(f"{name}=invalid->default({default})")
        return default


def load_macro_micro_policy(path: str | Path | None = None) -> MacroMicroPolicy:
    resolved = Path(path or os.getenv("MACRO_MICRO_CONFIG_PATH", DEFAULT_CONFIG_PATH))
    raw: Mapping[str, Any] = {}
    fallbacks: list[str] = []
    if resolved.exists():
        try:
            loaded = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, Mapping):
                raw = loaded
            else:
                fallbacks.append("config_not_mapping->defaults")
        except (OSError, yaml.YAMLError) as exc:  # pragma: no cover - defensive
            fallbacks.append(f"load_error:{exc}->defaults")
    else:
        fallbacks.append("config_missing->defaults")

    macro_raw = raw.get("macro") or {}
    micro_raw = raw.get("micro") or {}

    # Strategy permissions: YAML overrides defaults per regime.
    perms = {k: {"allow": tuple(v.get("allow") or ()), "block": tuple(v.get("block") or ())}
             for k, v in dict(DEFAULT_STRATEGY_PERMISSIONS).items()}
    for regime, spec in (raw.get("strategy_permissions") or {}).items():
        if isinstance(spec, Mapping):
            perms[str(regime)] = {"allow": tuple(spec.get("allow") or ()), "block": tuple(spec.get("block") or ())}

    macro_env = MacroReasonerConfig.from_env()
    macro_config = MacroReasonerConfig(
        candidate_limit=_clamp_int(macro_raw.get("candidate_limit", macro_env.candidate_limit), macro_env.candidate_limit, lo=1, hi=500, fallbacks=fallbacks, name="candidate_limit"),
        minimum_macro_confidence=_float(macro_raw.get("minimum_macro_confidence", macro_env.minimum_macro_confidence), macro_env.minimum_macro_confidence, fallbacks=fallbacks, name="minimum_macro_confidence"),
        block_buy_on_high_volatility=_as_bool(macro_raw.get("block_buy_on_high_volatility"), True),
        block_buy_on_news_shock=_as_bool(macro_raw.get("block_buy_on_news_shock"), True),
        block_buy_on_low_liquidity_market=_as_bool(macro_raw.get("block_buy_on_low_liquidity_market"), True),
        strategy_permissions=perms,
    )

    micro_env = MicroReasonerConfig.from_env()
    micro_config = MicroReasonerConfig(
        minimum_micro_confidence=_float(micro_raw.get("minimum_micro_confidence", micro_env.minimum_micro_confidence), micro_env.minimum_micro_confidence, fallbacks=fallbacks, name="minimum_micro_confidence"),
        minimum_expected_net_return_bps=_float(micro_raw.get("minimum_expected_net_return_bps", micro_env.minimum_expected_net_return_bps), micro_env.minimum_expected_net_return_bps, fallbacks=fallbacks, name="minimum_expected_net_return_bps"),
        block_if_spread_consumes_alpha=_as_bool(micro_raw.get("block_if_spread_consumes_alpha"), True),
        block_if_stale_quote=_as_bool(micro_raw.get("block_if_stale_quote"), True),
        block_if_low_liquidity=_as_bool(micro_raw.get("block_if_low_liquidity"), True),
    )

    coordinator_config = CoordinatorConfig(
        max_parallel_symbols=_clamp_int(micro_raw.get("max_parallel_symbols", 20), 20, lo=1, hi=200, fallbacks=fallbacks, name="max_parallel_symbols"),
        worker_timeout_seconds=_float(micro_raw.get("worker_timeout_seconds", 3.0), 3.0, fallbacks=fallbacks, name="worker_timeout_seconds"),
    )

    diagnostics_raw = raw.get("diagnostics") or {}
    policy = MacroMicroPolicy(
        enabled=_as_bool(raw.get("enabled"), True),
        macro_enabled=_as_bool(macro_raw.get("enabled"), True),
        micro_enabled=_as_bool(micro_raw.get("enabled"), True),
        macro_loop_interval_seconds=_clamp_int(macro_raw.get("loop_interval_seconds", 60), 60, lo=30, hi=3600, fallbacks=fallbacks, name="macro_loop_interval_seconds"),
        micro_loop_interval_seconds=_clamp_int(micro_raw.get("loop_interval_seconds", 5), 5, lo=1, hi=600, fallbacks=fallbacks, name="micro_loop_interval_seconds"),
        macro_config=macro_config,
        micro_config=micro_config,
        coordinator_config=coordinator_config,
        persist_snapshots=_as_bool(diagnostics_raw.get("persist_macro_micro_snapshots"), True),
        expose_dashboard_payload=_as_bool(diagnostics_raw.get("expose_dashboard_payload"), True),
        diagnostics={"config_fallbacks": fallbacks, "config_path": str(resolved)},
    )
    logger.info(
        "macro/micro policy loaded (enabled=%s, macro_interval=%ss, micro_interval=%ss, max_parallel=%s, fallbacks=%s)",
        policy.enabled, policy.macro_loop_interval_seconds, policy.micro_loop_interval_seconds,
        policy.coordinator_config.max_parallel_symbols, fallbacks,
    )
    return policy

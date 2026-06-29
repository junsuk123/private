from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class LiveConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class LiveTradingSafetyConfig:
    max_quote_age_ms: int
    max_orderbook_age_ms: int
    minimum_source_quality_score: float
    minimum_model_confidence: float
    minimum_probability_success: float
    minimum_expected_net_return_bps: float
    maximum_spread_bps: float
    maximum_volatility_5m_bps: float
    maximum_single_order_pct_of_cash: float
    maximum_position_pct_of_equity: float
    maximum_orders_per_day: int
    maximum_orders_per_symbol_per_day: int
    order_cooldown_seconds: int
    market_orders_allowed: bool
    require_trained_model: bool
    allow_heuristic_fallback_in_live: bool
    require_principal_protection: bool
    require_recent_readiness_report: bool
    readiness_report_max_age_seconds: int
    require_manual_arming: bool
    arming_ttl_seconds: int

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "LiveTradingSafetyConfig":
        cfg = cls(
            max_quote_age_ms=_int(data, "max_quote_age_ms"),
            max_orderbook_age_ms=_int(data, "max_orderbook_age_ms"),
            minimum_source_quality_score=_float(data, "minimum_source_quality_score"),
            minimum_model_confidence=_float(data, "minimum_model_confidence"),
            minimum_probability_success=_float(data, "minimum_probability_success"),
            minimum_expected_net_return_bps=_float(data, "minimum_expected_net_return_bps"),
            maximum_spread_bps=_float(data, "maximum_spread_bps"),
            maximum_volatility_5m_bps=_float(data, "maximum_volatility_5m_bps"),
            maximum_single_order_pct_of_cash=_float(data, "maximum_single_order_pct_of_cash"),
            maximum_position_pct_of_equity=_float(data, "maximum_position_pct_of_equity"),
            maximum_orders_per_day=_int(data, "maximum_orders_per_day"),
            maximum_orders_per_symbol_per_day=_int(data, "maximum_orders_per_symbol_per_day"),
            order_cooldown_seconds=_int(data, "order_cooldown_seconds"),
            market_orders_allowed=_bool(data, "market_orders_allowed"),
            require_trained_model=_bool(data, "require_trained_model"),
            allow_heuristic_fallback_in_live=_bool(data, "allow_heuristic_fallback_in_live"),
            require_principal_protection=_bool(data, "require_principal_protection"),
            require_recent_readiness_report=_bool(data, "require_recent_readiness_report"),
            readiness_report_max_age_seconds=_int(data, "readiness_report_max_age_seconds"),
            require_manual_arming=_bool(data, "require_manual_arming"),
            arming_ttl_seconds=_int(data, "arming_ttl_seconds"),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.max_quote_age_ms <= 0 or self.max_orderbook_age_ms <= 0:
            raise LiveConfigError("market-data age limits must be positive")
        if not 0 <= self.minimum_source_quality_score <= 1:
            raise LiveConfigError("minimum_source_quality_score must be between 0 and 1")
        if not 0 <= self.minimum_model_confidence <= 1:
            raise LiveConfigError("minimum_model_confidence must be between 0 and 1")
        if not 0 <= self.minimum_probability_success <= 1:
            raise LiveConfigError("minimum_probability_success must be between 0 and 1")
        if self.maximum_single_order_pct_of_cash <= 0 or self.maximum_single_order_pct_of_cash > 1:
            raise LiveConfigError("maximum_single_order_pct_of_cash must be in (0, 1]")
        if self.maximum_position_pct_of_equity <= 0 or self.maximum_position_pct_of_equity > 1:
            raise LiveConfigError("maximum_position_pct_of_equity must be in (0, 1]")
        if self.market_orders_allowed:
            raise LiveConfigError("live config must default to limit-only order execution")
        if self.allow_heuristic_fallback_in_live:
            raise LiveConfigError("heuristic fallback is not allowed in live config")


@dataclass(frozen=True)
class OrderExecutionConfig:
    order_type: str
    allow_market_orders: bool
    max_unfilled_order_age_seconds: int
    poll_order_status_interval_seconds: int
    max_order_status_poll_seconds: int
    cancel_stale_unfilled_orders: bool
    idempotency_ttl_seconds: int

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "OrderExecutionConfig":
        cfg = cls(
            order_type=str(data.get("order_type", "")).strip(),
            allow_market_orders=_bool(data, "allow_market_orders"),
            max_unfilled_order_age_seconds=_int(data, "max_unfilled_order_age_seconds"),
            poll_order_status_interval_seconds=_int(data, "poll_order_status_interval_seconds"),
            max_order_status_poll_seconds=_int(data, "max_order_status_poll_seconds"),
            cancel_stale_unfilled_orders=_bool(data, "cancel_stale_unfilled_orders"),
            idempotency_ttl_seconds=_int(data, "idempotency_ttl_seconds"),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.order_type != "LIMIT_ONLY":
            raise LiveConfigError("order_type must be LIMIT_ONLY")
        if self.allow_market_orders:
            raise LiveConfigError("allow_market_orders must be false")
        if min(
            self.max_unfilled_order_age_seconds,
            self.poll_order_status_interval_seconds,
            self.max_order_status_poll_seconds,
            self.idempotency_ttl_seconds,
        ) <= 0:
            raise LiveConfigError("order execution timing values must be positive")


def load_live_trading_safety_config(
    path: str | Path = "config/live_trading_safety.json",
    *,
    allow_example: bool = False,
) -> LiveTradingSafetyConfig:
    return LiveTradingSafetyConfig.from_mapping(_load_json(path, allow_example=allow_example))


def load_order_execution_config(
    path: str | Path = "config/order_execution.json",
    *,
    allow_example: bool = False,
) -> OrderExecutionConfig:
    return OrderExecutionConfig.from_mapping(_load_json(path, allow_example=allow_example))


def _load_json(path: str | Path, *, allow_example: bool) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists() and allow_example:
        example_path = config_path.with_name(f"{config_path.stem}.example{config_path.suffix}")
        config_path = example_path
    if not config_path.exists():
        raise LiveConfigError(f"missing required live config: {config_path}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LiveConfigError(f"invalid JSON in live config: {config_path}") from exc
    if not isinstance(payload, dict):
        raise LiveConfigError(f"live config must be a JSON object: {config_path}")
    return payload


def _int(data: dict[str, Any], key: str) -> int:
    if key not in data:
        raise LiveConfigError(f"missing live config key: {key}")
    try:
        return int(data[key])
    except (TypeError, ValueError) as exc:
        raise LiveConfigError(f"live config key must be integer: {key}") from exc


def _float(data: dict[str, Any], key: str) -> float:
    if key not in data:
        raise LiveConfigError(f"missing live config key: {key}")
    try:
        return float(data[key])
    except (TypeError, ValueError) as exc:
        raise LiveConfigError(f"live config key must be numeric: {key}") from exc


def _bool(data: dict[str, Any], key: str) -> bool:
    if key not in data:
        raise LiveConfigError(f"missing live config key: {key}")
    value = data[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    raise LiveConfigError(f"live config key must be boolean: {key}")

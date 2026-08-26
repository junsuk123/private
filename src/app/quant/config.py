from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_PATH = Path("config/quant_reference.yaml")


class QuantConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class QuantConfig:
    activation_mode: str = "auto"
    minimum_python_major: int = 3
    minimum_python_minor: int = 10
    auto_disable_consecutive_errors: int = 3
    auto_retry_cooldown_seconds: int = 300
    price_window: int = 20
    return_window: int = 20
    rsi_window: int = 14
    ema_fast: int = 12
    ema_slow: int = 26
    macd_signal: int = 9
    bollinger_stddev: float = 2.0
    annualization: int = 252
    stale_after_ms: int = 120_000
    cache_size: int = 10_000
    schema_version: str = "quant-evidence-v1"

    def __post_init__(self) -> None:
        integers = (
            self.minimum_python_major, self.minimum_python_minor,
            self.auto_disable_consecutive_errors, self.auto_retry_cooldown_seconds,
            self.price_window, self.return_window, self.rsi_window, self.ema_fast,
            self.ema_slow, self.macd_signal, self.annualization, self.stale_after_ms,
            self.cache_size,
        )
        if any(value <= 0 for value in integers):
            raise QuantConfigError("quant windows and limits must be positive")
        if self.ema_fast >= self.ema_slow:
            raise QuantConfigError("ema_fast must be less than ema_slow")
        if self.bollinger_stddev <= 0:
            raise QuantConfigError("bollinger_stddev must be positive")
        if self.activation_mode not in {"auto", "off", "on"}:
            raise QuantConfigError("activation_mode must be auto, off, or on")


def load_quant_config(path: str | Path = DEFAULT_CONFIG_PATH) -> QuantConfig:
    target = Path(path)
    if not target.exists():
        return QuantConfig()
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise QuantConfigError(f"cannot parse {target}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise QuantConfigError("quant config must be a mapping")
    known = {field for field in QuantConfig.__dataclass_fields__}
    unknown = set(raw) - known
    if unknown:
        raise QuantConfigError(f"unknown quant config keys: {sorted(unknown)}")
    try:
        return QuantConfig(**{key: raw[key] for key in known if key in raw})
    except (TypeError, ValueError) as exc:
        raise QuantConfigError(str(exc)) from exc


@lru_cache(maxsize=1)
def default_quant_config() -> QuantConfig:
    return load_quant_config()

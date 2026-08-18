"""Typed access to ``config/temporal_context.yaml``.

Loaded once and cached. Every value the temporal layer needs that is a *choice* rather
than an exchange fact lives here, so no phase boundary, expiry rule or shrinkage constant
is written as a literal in the code that uses it.

A missing file is not an error: the dataclass defaults are the same numbers the YAML
ships with, so a deployment that has not copied the config still behaves identically
rather than failing to start. A file that IS present but malformed raises, because a
half-read policy is worse than no policy.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "CalendarAdjacencyConfig",
    "ExpiryRule",
    "SeasonalityConfig",
    "SessionPhaseConfig",
    "TemporalConfig",
    "TemporalConfigError",
    "default_temporal_config",
    "load_temporal_config",
    "reset_temporal_config_cache",
]

DEFAULT_CONFIG_PATH = Path("config/temporal_context.yaml")


class TemporalConfigError(RuntimeError):
    """The temporal config exists but cannot be read as the declared schema."""


@dataclass(frozen=True)
class SessionPhaseConfig:
    open_transition_minutes: float = 2.0
    opening_minutes: float = 30.0
    morning_end_fraction: float = 0.33
    midday_end_fraction: float = 0.66
    closing_minutes: float = 30.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.open_transition_minutes <= self.opening_minutes:
            raise TemporalConfigError(
                "open_transition_minutes must be within [0, opening_minutes]"
            )
        if not 0.0 < self.morning_end_fraction < self.midday_end_fraction < 1.0:
            raise TemporalConfigError(
                "require 0 < morning_end_fraction < midday_end_fraction < 1"
            )
        if self.closing_minutes < 0.0:
            raise TemporalConfigError("closing_minutes must not be negative")


@dataclass(frozen=True)
class ExpiryRule:
    """Nth-weekday-of-month derivatives expiry for one market group."""

    weekday: int
    ordinal: int
    quarterly_months: tuple[int, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.weekday <= 6:
            raise TemporalConfigError("expiry weekday must be 0..6 (Mon=0)")
        if self.ordinal < 1:
            raise TemporalConfigError("expiry ordinal must be >= 1")


@dataclass(frozen=True)
class ExpiryConfig:
    rules: Mapping[str, ExpiryRule] = field(default_factory=dict)
    adjacent_trading_days: int = 1

    def rule_for(self, group: str) -> ExpiryRule | None:
        return self.rules.get(str(group or "").upper())


@dataclass(frozen=True)
class CalendarAdjacencyConfig:
    holiday_gap_days: int = 1
    max_lookback_days: int = 21


@dataclass(frozen=True)
class SeasonalityConfig:
    epsilon: float = 1e-9
    shrinkage_k: float = 30.0
    baseline_window: int = 500
    minimum_samples: int = 5
    staleness_days: int = 45

    def __post_init__(self) -> None:
        if self.epsilon <= 0.0:
            raise TemporalConfigError("seasonality epsilon must be positive")
        if self.shrinkage_k < 0.0:
            raise TemporalConfigError("seasonality shrinkage_k must not be negative")
        if self.baseline_window < 2:
            raise TemporalConfigError("seasonality baseline_window must be >= 2")


_DEFAULT_EXPIRY = {
    "KR": ExpiryRule(weekday=3, ordinal=2, quarterly_months=(3, 6, 9, 12)),
    "US": ExpiryRule(weekday=4, ordinal=3, quarterly_months=(3, 6, 9, 12)),
}


@dataclass(frozen=True)
class TemporalConfig:
    session_phase: SessionPhaseConfig = field(default_factory=SessionPhaseConfig)
    expiry: ExpiryConfig = field(
        default_factory=lambda: ExpiryConfig(rules=dict(_DEFAULT_EXPIRY))
    )
    calendar: CalendarAdjacencyConfig = field(default_factory=CalendarAdjacencyConfig)
    seasonality: SeasonalityConfig = field(default_factory=SeasonalityConfig)
    source_path: str | None = None


def _mapping(raw: Any, name: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TemporalConfigError(f"{name} must be a mapping")
    return raw


def _number(raw: Mapping[str, Any], key: str, default: float) -> float:
    if key not in raw:
        return default
    try:
        return float(raw[key])
    except (TypeError, ValueError) as exc:
        raise TemporalConfigError(f"{key} must be a number") from exc


def _integer(raw: Mapping[str, Any], key: str, default: int) -> int:
    if key not in raw:
        return default
    try:
        return int(raw[key])
    except (TypeError, ValueError) as exc:
        raise TemporalConfigError(f"{key} must be an integer") from exc


def _expiry_config(raw: Mapping[str, Any]) -> ExpiryConfig:
    rules: dict[str, ExpiryRule] = dict(_DEFAULT_EXPIRY)
    for key, value in raw.items():
        name = str(key).upper()
        if name == "ADJACENT_TRADING_DAYS":
            continue
        entry = _mapping(value, f"expiry.{key}")
        default = _DEFAULT_EXPIRY.get(name)
        months = entry.get("quarterly_months")
        if months is None:
            quarterly = default.quarterly_months if default else ()
        else:
            if not isinstance(months, (list, tuple)):
                raise TemporalConfigError("expiry quarterly_months must be a list")
            quarterly = tuple(int(month) for month in months)
        rules[name] = ExpiryRule(
            weekday=_integer(entry, "weekday", default.weekday if default else 4),
            ordinal=_integer(entry, "ordinal", default.ordinal if default else 3),
            quarterly_months=quarterly,
        )
    return ExpiryConfig(
        rules=rules,
        adjacent_trading_days=_integer(raw, "adjacent_trading_days", 1),
    )


def load_temporal_config(path: str | Path = DEFAULT_CONFIG_PATH) -> TemporalConfig:
    """Read the config, or return defaults when the file is absent."""
    target = Path(path)
    if not target.exists():
        return TemporalConfig()
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is a declared dependency.
        raise TemporalConfigError("PyYAML is required to read the temporal config") from exc
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 - surfaced as a config error, never swallowed.
        raise TemporalConfigError(f"cannot parse {target}: {exc}") from exc
    document = _mapping(raw, str(target))

    phase_raw = _mapping(document.get("session_phase"), "session_phase")
    phase = SessionPhaseConfig(
        open_transition_minutes=_number(phase_raw, "open_transition_minutes", 2.0),
        opening_minutes=_number(phase_raw, "opening_minutes", 30.0),
        morning_end_fraction=_number(phase_raw, "morning_end_fraction", 0.33),
        midday_end_fraction=_number(phase_raw, "midday_end_fraction", 0.66),
        closing_minutes=_number(phase_raw, "closing_minutes", 30.0),
    )
    calendar_raw = _mapping(document.get("calendar"), "calendar")
    calendar = CalendarAdjacencyConfig(
        holiday_gap_days=_integer(calendar_raw, "holiday_gap_days", 1),
        max_lookback_days=_integer(calendar_raw, "max_lookback_days", 21),
    )
    seasonality_raw = _mapping(document.get("seasonality"), "seasonality")
    seasonality = SeasonalityConfig(
        epsilon=_number(seasonality_raw, "epsilon", 1e-9),
        shrinkage_k=_number(seasonality_raw, "shrinkage_k", 30.0),
        baseline_window=_integer(seasonality_raw, "baseline_window", 500),
        minimum_samples=_integer(seasonality_raw, "minimum_samples", 5),
        staleness_days=_integer(seasonality_raw, "staleness_days", 45),
    )
    return TemporalConfig(
        session_phase=phase,
        expiry=_expiry_config(_mapping(document.get("expiry"), "expiry")),
        calendar=calendar,
        seasonality=seasonality,
        source_path=str(target),
    )


_cache: TemporalConfig | None = None
_cache_lock = threading.Lock()


def default_temporal_config() -> TemporalConfig:
    """Process-wide cached config. The trading loop reads this, never the file."""
    global _cache
    with _cache_lock:
        if _cache is None:
            _cache = load_temporal_config()
        return _cache


def reset_temporal_config_cache() -> None:
    """Test hook. Never called from the trading path."""
    global _cache
    with _cache_lock:
        _cache = None

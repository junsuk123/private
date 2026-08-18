"""Global / cross-asset context — the layer above the domestic market.

``Calendar/Session -> GLOBAL -> Domestic -> Sector -> Stock``.

What this layer is allowed to conclude
--------------------------------------
That risk appetite is rising or falling, how much of the world agrees, and how much
pressure the rates and FX complex is putting on domestic equities. That is all. It has no
opinion about any Korean stock, and by construction it cannot form one: nothing here
takes a ticker.

The rule that makes this safe is enforced downstream — a weak S&P never becomes a
domestic SELL on its own; it becomes a *conflicting factor* that the domestic layer's
relative strength, breadth and flow must overcome or confirm. See
:mod:`app.context.domestic_context`.

Construction
------------
Pure. The builder is handed :class:`IndicatorObservation` records and a
:class:`GlobalIndicatorConfig`, and returns the same context for the same inputs. Group
membership, orientation, normalisation scale and freshness allowance all live in
``config/global_indicators.yaml`` so no market fact is written as a literal here.

Absence is never a zero. A group with no usable observation scores ``None`` and lowers
``confidence``; a context whose observed group weight falls below
``minimum_group_coverage`` reports ``direction=None`` rather than a direction derived
from one surviving series.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "GlobalContext",
    "GlobalContextBuilder",
    "GlobalGroupScore",
    "GlobalIndicatorConfig",
    "IndicatorGroupConfig",
    "IndicatorObservation",
    "default_global_indicator_config",
    "load_global_indicator_config",
    "reset_global_indicator_config_cache",
]

DEFAULT_CONFIG_PATH = Path("config/global_indicators.yaml")

GLOBAL_NO_COVERAGE = "GLOBAL_NO_COVERAGE"
GLOBAL_GROUP_MISSING = "GLOBAL_GROUP_MISSING"
GLOBAL_OBSERVATIONS_STALE = "GLOBAL_OBSERVATIONS_STALE"
GLOBAL_UNKNOWN_INDICATOR = "GLOBAL_UNKNOWN_INDICATOR"


class GlobalIndicatorConfigError(RuntimeError):
    """The global indicator config exists but does not parse as the declared schema."""


@dataclass(frozen=True)
class IndicatorGroupConfig:
    name: str
    members: tuple[str, ...]
    weight: float = 1.0
    scale: float = 0.01
    risk_on_sign: int = 1
    max_age_seconds: float = 86400.0
    reference_level: float | None = None

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise GlobalIndicatorConfigError(f"group {self.name}: scale must be positive")
        if self.risk_on_sign not in (-1, 1):
            raise GlobalIndicatorConfigError(f"group {self.name}: risk_on_sign must be +-1")


@dataclass(frozen=True)
class CrossMarketConfig:
    beta_window: int = 60
    beta_minimum_samples: int = 20
    beta_bounds: tuple[float, float] = (0.0, 3.0)
    max_lag_minutes: int = 120


@dataclass(frozen=True)
class GlobalIndicatorConfig:
    groups: Mapping[str, IndicatorGroupConfig]
    direction_groups: tuple[str, ...]
    minimum_group_coverage: float = 0.35
    cross_market: CrossMarketConfig = field(default_factory=CrossMarketConfig)
    source_path: str | None = None

    def group_for(self, indicator: str) -> IndicatorGroupConfig | None:
        name = str(indicator or "").strip().upper()
        for group in self.groups.values():
            if name in group.members:
                return group
        return None

    @property
    def total_weight(self) -> float:
        return sum(group.weight for group in self.groups.values()) or 1.0


@dataclass(frozen=True)
class IndicatorObservation:
    """One reading of one global series.

    ``change_ratio`` is the fractional move over the short reference window (0.012 = up
    1.2%); ``change_ratio_long`` the same over the longer window used for momentum. Both
    may be ``None`` — a level with no move attached still contributes to the volatility
    output but not to direction.
    """

    name: str
    value: float
    observed_at: datetime
    source: str = ""
    change_ratio: float | None = None
    change_ratio_long: float | None = None
    #: Group override. Normally resolved from config membership.
    group: str | None = None

    def age_seconds(self, now: datetime) -> float:
        return max(0.0, (now - _aware(self.observed_at)).total_seconds())


@dataclass(frozen=True)
class GlobalGroupScore:
    """One group's contribution, with everything needed to audit it."""

    group: str
    #: Risk-on adjusted score in [-1, 1]. ``None`` when the group had no usable move.
    score: float | None
    #: Raw (unadjusted) score in [-1, 1] — the direction the series itself moved.
    raw_score: float | None
    momentum: float | None
    level: float | None
    observed_members: tuple[str, ...]
    stale_members: tuple[str, ...]
    freshness: float
    weight: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "score": self.score,
            "raw_score": self.raw_score,
            "momentum": self.momentum,
            "level": self.level,
            "observed_members": list(self.observed_members),
            "stale_members": list(self.stale_members),
            "freshness": self.freshness,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class GlobalContext:
    """Cross-asset state at one instant. No ticker, no order, no direction to trade."""

    captured_at: datetime
    context_id: str
    direction: float | None = None
    momentum: float | None = None
    risk_sentiment: float | None = None
    volatility: float | None = None
    rates_pressure: float | None = None
    fx_pressure: float | None = None
    global_alignment: float | None = None
    confidence: float = 0.0
    groups: Mapping[str, GlobalGroupScore] = field(default_factory=dict)
    coverage: float = 0.0
    reason_codes: tuple[str, ...] = ()

    def numeric_features(self) -> dict[str, float]:
        values: dict[str, float] = {"global_confidence": self.confidence}
        for name, value in (
            ("global_direction", self.direction),
            ("global_momentum", self.momentum),
            ("global_risk_sentiment", self.risk_sentiment),
            ("global_volatility", self.volatility),
            ("global_rates_pressure", self.rates_pressure),
            ("global_fx_pressure", self.fx_pressure),
            ("global_alignment", self.global_alignment),
        ):
            if value is not None:
                values[name] = float(value)
        return values

    def as_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "captured_at": _aware(self.captured_at).isoformat(),
            "direction": self.direction,
            "momentum": self.momentum,
            "risk_sentiment": self.risk_sentiment,
            "volatility": self.volatility,
            "rates_pressure": self.rates_pressure,
            "fx_pressure": self.fx_pressure,
            "global_alignment": self.global_alignment,
            "confidence": self.confidence,
            "coverage": self.coverage,
            "groups": {name: score.as_dict() for name, score in self.groups.items()},
            "reason_codes": list(self.reason_codes),
        }


def _aware(moment: datetime) -> datetime:
    return (
        moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    ).astimezone(timezone.utc)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _squash(change: float, scale: float) -> float:
    """Bound a fractional move to [-1, 1] without a hard clip.

    ``tanh`` rather than a clamp so an ordinary move stays proportional while a shock
    saturates: a clamp would make a -3% and a -8% session score identically, which is
    exactly the distinction ``risk_sentiment`` exists to carry.
    """
    return math.tanh(change / scale)


def _freshness(age_seconds: float, max_age_seconds: float) -> float:
    """1.0 while fresh, decaying linearly to 0 at ``max_age_seconds``."""
    if max_age_seconds <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - age_seconds / max_age_seconds))


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _mapping(raw: Any, name: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise GlobalIndicatorConfigError(f"{name} must be a mapping")
    return raw


_DEFAULT_GROUPS: dict[str, IndicatorGroupConfig] = {
    "equity": IndicatorGroupConfig(
        "equity", ("SP500", "NASDAQ", "DOW", "RUSSELL2000"), 1.0, 0.010, 1, 86400.0
    ),
    "semiconductor": IndicatorGroupConfig(
        "semiconductor", ("SOX", "NVDA", "AMD", "AVGO", "MU", "TSM", "INTC"),
        1.0, 0.018, 1, 86400.0,
    ),
    "risk": IndicatorGroupConfig("risk", ("VIX",), 1.0, 0.080, -1, 86400.0, 20.0),
    "rates": IndicatorGroupConfig("rates", ("US2Y", "US10Y"), 0.8, 0.030, -1, 172800.0),
    "fx": IndicatorGroupConfig("fx", ("USDKRW", "DXY"), 0.8, 0.006, -1, 86400.0),
    "commodity": IndicatorGroupConfig(
        "commodity", ("WTI", "GOLD", "COPPER"), 0.5, 0.020, 1, 86400.0
    ),
    "asia": IndicatorGroupConfig(
        "asia", ("NIKKEI", "HANGSENG", "CSI300"), 0.8, 0.010, 1, 86400.0
    ),
    "futures": IndicatorGroupConfig(
        "futures", ("ES", "NQ", "YM", "US_INDEX_FUTURES"), 1.0, 0.008, 1, 3600.0
    ),
}


def load_global_indicator_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> GlobalIndicatorConfig:
    target = Path(path)
    if not target.exists():
        return GlobalIndicatorConfig(
            groups=dict(_DEFAULT_GROUPS),
            direction_groups=("equity", "semiconductor", "asia", "futures"),
        )
    try:
        import yaml

        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed.
        raise GlobalIndicatorConfigError(f"cannot parse {target}: {exc}") from exc
    document = _mapping(raw, str(target))
    groups: dict[str, IndicatorGroupConfig] = {}
    for name, value in _mapping(document.get("groups"), "groups").items():
        entry = _mapping(value, f"groups.{name}")
        members = entry.get("members") or ()
        if not isinstance(members, (list, tuple)):
            raise GlobalIndicatorConfigError(f"groups.{name}.members must be a list")
        reference = entry.get("reference_level")
        groups[str(name)] = IndicatorGroupConfig(
            name=str(name),
            members=tuple(str(item).strip().upper() for item in members),
            weight=float(entry.get("weight", 1.0)),
            scale=float(entry.get("scale", 0.01)),
            risk_on_sign=int(entry.get("risk_on_sign", 1)),
            max_age_seconds=float(entry.get("max_age_seconds", 86400.0)),
            reference_level=float(reference) if reference is not None else None,
        )
    if not groups:
        groups = dict(_DEFAULT_GROUPS)
    direction = document.get("direction_groups") or ("equity", "semiconductor", "asia", "futures")
    cross_raw = _mapping(document.get("cross_market"), "cross_market")
    bounds = cross_raw.get("beta_bounds") or (0.0, 3.0)
    cross = CrossMarketConfig(
        beta_window=int(cross_raw.get("beta_window", 60)),
        beta_minimum_samples=int(cross_raw.get("beta_minimum_samples", 20)),
        beta_bounds=(float(bounds[0]), float(bounds[1])),
        max_lag_minutes=int(cross_raw.get("max_lag_minutes", 120)),
    )
    return GlobalIndicatorConfig(
        groups=groups,
        direction_groups=tuple(str(item) for item in direction),
        minimum_group_coverage=float(document.get("minimum_group_coverage", 0.35)),
        cross_market=cross,
        source_path=str(target),
    )


_config_cache: GlobalIndicatorConfig | None = None
_config_lock = threading.Lock()


def default_global_indicator_config() -> GlobalIndicatorConfig:
    global _config_cache
    with _config_lock:
        if _config_cache is None:
            _config_cache = load_global_indicator_config()
        return _config_cache


def reset_global_indicator_config_cache() -> None:
    """Test hook. Never called from the trading path."""
    global _config_cache
    with _config_lock:
        _config_cache = None


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
class GlobalContextBuilder:
    """Turns global indicator observations into one :class:`GlobalContext`."""

    def __init__(self, config: GlobalIndicatorConfig | None = None) -> None:
        self._config = config or default_global_indicator_config()

    @property
    def config(self) -> GlobalIndicatorConfig:
        return self._config

    def build(
        self,
        observations: Iterable[IndicatorObservation],
        *,
        captured_at: datetime,
        context_id: str | None = None,
    ) -> GlobalContext:
        now = _aware(captured_at)
        reasons: list[str] = []
        by_group: dict[str, list[tuple[IndicatorObservation, IndicatorGroupConfig]]] = {}
        for observation in observations:
            group = (
                self._config.groups.get(str(observation.group))
                if observation.group
                else self._config.group_for(observation.name)
            )
            if group is None:
                reasons.append(GLOBAL_UNKNOWN_INDICATOR)
                continue
            by_group.setdefault(group.name, []).append((observation, group))

        scores: dict[str, GlobalGroupScore] = {}
        for name, group in self._config.groups.items():
            entries = by_group.get(name, [])
            if not entries:
                reasons.append(GLOBAL_GROUP_MISSING)
                continue
            score = self._score_group(group, entries, now)
            scores[name] = score
            if score.stale_members:
                reasons.append(GLOBAL_OBSERVATIONS_STALE)

        observed_weight = sum(
            score.weight * score.freshness for score in scores.values()
        )
        coverage = observed_weight / self._config.total_weight
        if coverage < self._config.minimum_group_coverage:
            reasons.append(GLOBAL_NO_COVERAGE)
            return GlobalContext(
                captured_at=now,
                context_id=context_id or _context_id(now),
                confidence=round(max(0.0, coverage), 6),
                coverage=round(max(0.0, coverage), 6),
                groups=scores,
                reason_codes=tuple(dict.fromkeys(reasons)),
            )

        direction = self._weighted(
            scores, self._config.direction_groups, attribute="score"
        )
        momentum = self._weighted(
            scores, self._config.direction_groups, attribute="momentum"
        )
        risk_group = scores.get("risk")
        risk_sentiment = _mean_of(
            [value for value in (direction, risk_group.score if risk_group else None)
             if value is not None]
        )
        volatility = self._volatility(scores)
        rates_pressure = self._raw(scores, ("rates",))
        fx_pressure = self._raw(scores, ("fx",))
        alignment = self._alignment(scores)
        confidence = round(
            min(1.0, coverage) * _mean_of([score.freshness for score in scores.values()] or [0.0]),
            6,
        )

        return GlobalContext(
            captured_at=now,
            context_id=context_id or _context_id(now),
            direction=direction,
            momentum=momentum,
            risk_sentiment=risk_sentiment,
            volatility=volatility,
            rates_pressure=rates_pressure,
            fx_pressure=fx_pressure,
            global_alignment=alignment,
            confidence=confidence,
            groups=scores,
            coverage=round(coverage, 6),
            reason_codes=tuple(dict.fromkeys(reasons)),
        )

    # -- internals ---------------------------------------------------------- #
    def _score_group(
        self,
        group: IndicatorGroupConfig,
        entries: Sequence[tuple[IndicatorObservation, IndicatorGroupConfig]],
        now: datetime,
    ) -> GlobalGroupScore:
        raw_terms: list[tuple[float, float]] = []
        momentum_terms: list[tuple[float, float]] = []
        levels: list[float] = []
        observed: list[str] = []
        stale: list[str] = []
        freshness_terms: list[float] = []

        for observation, _ in entries:
            age = observation.age_seconds(now)
            fresh = _freshness(age, group.max_age_seconds)
            freshness_terms.append(fresh)
            name = str(observation.name).strip().upper()
            if fresh <= 0.0:
                stale.append(name)
                continue
            observed.append(name)
            level = _finite(observation.value)
            if level is not None:
                levels.append(level)
            change = _finite(observation.change_ratio)
            if change is not None:
                raw_terms.append((_squash(change, group.scale), fresh))
            long_change = _finite(observation.change_ratio_long)
            if long_change is not None:
                momentum_terms.append((_squash(long_change, group.scale), fresh))

        raw_score = _weighted_mean(raw_terms)
        momentum = _weighted_mean(momentum_terms)
        return GlobalGroupScore(
            group=group.name,
            score=None if raw_score is None else raw_score * group.risk_on_sign,
            raw_score=raw_score,
            momentum=None if momentum is None else momentum * group.risk_on_sign,
            level=_mean_of(levels),
            observed_members=tuple(observed),
            stale_members=tuple(stale),
            freshness=_mean_of(freshness_terms) or 0.0,
            weight=group.weight,
        )

    def _weighted(
        self,
        scores: Mapping[str, GlobalGroupScore],
        names: Sequence[str],
        *,
        attribute: str,
    ) -> float | None:
        terms: list[tuple[float, float]] = []
        for name in names:
            score = scores.get(name)
            if score is None:
                continue
            value = getattr(score, attribute)
            if value is None:
                continue
            terms.append((float(value), score.weight * score.freshness))
        return _weighted_mean(terms)

    def _raw(
        self, scores: Mapping[str, GlobalGroupScore], names: Sequence[str]
    ) -> float | None:
        terms: list[tuple[float, float]] = []
        for name in names:
            score = scores.get(name)
            if score is None or score.raw_score is None:
                continue
            terms.append((score.raw_score, score.weight * score.freshness))
        return _weighted_mean(terms)

    def _volatility(self, scores: Mapping[str, GlobalGroupScore]) -> float | None:
        """Level-based, not move-based: 'how volatile is the world', in [0, 1]."""
        risk = scores.get("risk")
        reference = self._config.groups.get("risk")
        if risk is None or risk.level is None or reference is None:
            return None
        if not reference.reference_level:
            return None
        return round(min(2.0, max(0.0, risk.level / reference.reference_level)), 6)

    def _alignment(self, scores: Mapping[str, GlobalGroupScore]) -> float | None:
        """Signed agreement across groups, in [-1, 1].

        ``mean / mean(|.|)`` is +1 when every group leans the same way, -1 when they all
        lean the other way and 0 when they cancel. A plain mean cannot distinguish "all
        mildly positive" from "half strongly positive, half strongly negative", and those
        are the two states a cross-market gate most needs to tell apart.
        """
        values = [
            score.score
            for score in scores.values()
            if score.score is not None and score.freshness > 0.0
        ]
        if len(values) < 2:
            return None
        magnitude = _mean_of([abs(value) for value in values]) or 0.0
        if magnitude <= 0.0:
            return 0.0
        return round(max(-1.0, min(1.0, (_mean_of(values) or 0.0) / magnitude)), 6)


def _weighted_mean(terms: Sequence[tuple[float, float]]) -> float | None:
    usable = [(value, weight) for value, weight in terms if weight > 0.0]
    if not usable:
        return None
    total = sum(weight for _, weight in usable)
    if total <= 0.0:
        return None
    return round(sum(value * weight for value, weight in usable) / total, 6)


def _mean_of(values: Sequence[float]) -> float | None:
    usable = [float(value) for value in values if _finite(value) is not None]
    if not usable:
        return None
    return round(sum(usable) / len(usable), 6)


def _context_id(moment: datetime) -> str:
    from uuid import uuid4

    return f"gctx-{moment.strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"

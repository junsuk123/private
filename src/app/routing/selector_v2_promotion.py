"""Evidence-gated automatic authority for StrategySelectorV2.

The feature flag says whether automatic promotion is allowed.  This controller says
whether the selector has *earned* authority.  It persists one row per resolved market
context, advances by one rung at a time, and demotes faster than it promotes.

SHADOW -> LIVE_PROBE is the first transition that can affect an order.  It therefore
requires a positive lower confidence bound after an additional cost shock, breadth
across days and chronological windows, and bounded selector regret.  LIVE_PROBE -> LIVE
requires real selected outcomes; simulated outcomes cannot grant full authority.
"""

from __future__ import annotations

import json
import math
import statistics
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.evaluation.selector_regret import compute_context_regret

__all__ = [
    "SelectorAuthorityState",
    "SelectorPromotionConfig",
    "SelectorPromotionController",
]


class SelectorAuthorityState(StrEnum):
    SHADOW = "SHADOW"
    LIVE_PROBE = "LIVE_PROBE"
    LIVE = "LIVE"
    SUSPENDED = "SUSPENDED"

    @property
    def live_authority(self) -> bool:
        return self in {self.LIVE_PROBE, self.LIVE}

    @property
    def order_size_fraction(self) -> float:
        return 0.10 if self is self.LIVE_PROBE else 1.0 if self is self.LIVE else 0.0


@dataclass(frozen=True)
class SelectorPromotionConfig:
    minimum_contexts: int = 120
    minimum_traded_contexts: int = 40
    minimum_distinct_days: int = 10
    minimum_chronological_windows: int = 4
    minimum_positive_windows: int = 3
    minimum_lower_bound_net_bps: float = 3.0
    additional_cost_stress_bps: float = 5.0
    maximum_mean_regret_bps: float = 15.0
    maximum_wrong_regime_trade_rate: float = 0.15
    minimum_top1_hit_rate: float = 0.40
    required_consecutive_passes: int = 3
    minimum_live_probe_contexts: int = 30
    minimum_live_probe_days: int = 10
    live_minimum_lower_bound_net_bps: float = 0.0
    demotion_lower_bound_bps: float = -5.0
    demotion_wrong_regime_trade_rate: float = 0.25
    state_path: str = "data/store/selector-v2-promotion.json"
    maximum_evidence_rows: int = 4000

    @classmethod
    def load(
        cls, path: str | Path = "config/selector_v2_promotion.yaml"
    ) -> "SelectorPromotionConfig":
        source = Path(path)
        if not source.exists():
            return cls()
        try:
            import yaml

            raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError, TypeError):
            return cls()
        if not isinstance(raw, Mapping):
            return cls()
        values = {
            key: value
            for key, value in raw.items()
            if key in cls.__dataclass_fields__ and value is not None
        }
        try:
            return cls(**values)
        except (TypeError, ValueError):
            return cls()


@dataclass(frozen=True)
class SelectorPromotionDecision:
    from_state: SelectorAuthorityState
    to_state: SelectorAuthorityState
    changed: bool
    reason_codes: tuple[str, ...]
    metrics: Mapping[str, Any] = field(default_factory=dict)
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def as_dict(self) -> dict[str, Any]:
        return {
            "from_state": str(self.from_state),
            "to_state": str(self.to_state),
            "changed": self.changed,
            "reason_codes": list(self.reason_codes),
            "metrics": dict(self.metrics),
            "evaluated_at": _aware(self.evaluated_at).isoformat(),
        }


class SelectorPromotionController:
    """Persisted, fail-closed selector authority state machine."""

    def __init__(self, config: SelectorPromotionConfig | None = None) -> None:
        self.config = config or SelectorPromotionConfig()
        self._path = Path(self.config.state_path)
        self._lock = threading.RLock()
        self._state = SelectorAuthorityState.SHADOW
        self._passes = 0
        self._rows: dict[str, dict[str, Any]] = {}
        self._history: list[dict[str, Any]] = []
        self._last_metrics: dict[str, Any] = {}
        self._load()

    @property
    def state(self) -> SelectorAuthorityState:
        with self._lock:
            return self._state

    @property
    def live_authority(self) -> bool:
        return self.state.live_authority

    @property
    def order_size_fraction(self) -> float:
        return self.state.order_size_fraction

    def evaluate(
        self, groups: Sequence[Any], *, now: datetime | None = None
    ) -> SelectorPromotionDecision:
        moment = _aware(now or datetime.now(timezone.utc))
        with self._lock:
            self._ingest(groups)
            metrics = self._metrics()
            current = self._state
            target = current
            reasons = self._promotion_failures(metrics, current)

            # Demotion requires much less evidence than promotion.  A live selector
            # with a clearly negative conservative edge or excessive wrong-regime
            # trading loses authority on the first evaluation.
            if current.live_authority and int(metrics.get("context_count") or 0) >= 20:
                lower_bound = metrics.get("lower_bound_net_bps")
                if lower_bound is not None and float(lower_bound) <= self.config.demotion_lower_bound_bps:
                    target = SelectorAuthorityState.SHADOW
                    reasons = ("SELECTOR_AUTO_DEMOTION_NEGATIVE_EDGE",)
                elif float(metrics.get("wrong_regime_trade_rate") or 0.0) > self.config.demotion_wrong_regime_trade_rate:
                    target = SelectorAuthorityState.SHADOW
                    reasons = ("SELECTOR_AUTO_DEMOTION_WRONG_REGIME",)

            if target is current and not reasons:
                self._passes += 1
                if self._passes >= self.config.required_consecutive_passes:
                    target = (
                        SelectorAuthorityState.LIVE_PROBE
                        if current is SelectorAuthorityState.SHADOW
                        else SelectorAuthorityState.LIVE
                        if current is SelectorAuthorityState.LIVE_PROBE
                        else current
                    )
                    self._passes = 0
                    reasons = ("SELECTOR_AUTOMATIC_PROMOTION",)
                else:
                    reasons = (
                        "SELECTOR_PROMOTION_DEBOUNCE",
                        f"PASSES:{self._passes}/{self.config.required_consecutive_passes}",
                    )
            elif target is current:
                self._passes = 0
            else:
                self._passes = 0

            changed = target is not current
            self._state = target
            self._last_metrics = metrics
            decision = SelectorPromotionDecision(
                from_state=current,
                to_state=target,
                changed=changed,
                reason_codes=tuple(reasons),
                metrics=metrics,
                evaluated_at=moment,
            )
            if changed:
                self._history.append(decision.as_dict())
                self._history = self._history[-200:]
            persisted = self._flush()
            if changed and target.live_authority and not persisted:
                # No durable audit record, no authority.  Revert in memory as well;
                # granting permission until the next restart would be the least
                # observable failure mode.
                self._state = current
                self._passes = 0
                decision = SelectorPromotionDecision(
                    from_state=current,
                    to_state=current,
                    changed=False,
                    reason_codes=("SELECTOR_PROMOTION_PERSISTENCE_FAILED",),
                    metrics=metrics,
                    evaluated_at=moment,
                )
            return decision

    def suspend(self, reason: str, *, now: datetime | None = None) -> SelectorPromotionDecision:
        moment = _aware(now or datetime.now(timezone.utc))
        with self._lock:
            current = self._state
            self._state = SelectorAuthorityState.SUSPENDED
            self._passes = 0
            decision = SelectorPromotionDecision(
                from_state=current,
                to_state=self._state,
                changed=current is not self._state,
                reason_codes=("SELECTOR_AUTOMATIC_SUSPENSION", str(reason or "UNKNOWN")),
                metrics=self._last_metrics,
                evaluated_at=moment,
            )
            self._history.append(decision.as_dict())
            self._history = self._history[-200:]
            self._flush()
            return decision

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": str(self._state),
                "live_authority": self._state.live_authority,
                "order_size_fraction": self._state.order_size_fraction,
                "consecutive_passes": self._passes,
                "metrics": dict(self._last_metrics),
                "evidence_rows": len(self._rows),
                "recent_transitions": list(self._history[-10:]),
            }

    def _ingest(self, groups: Sequence[Any]) -> None:
        for group in groups or ():
            regret = compute_context_regret(group)
            if regret is None:
                continue
            context_id = str(regret.context_id or "")
            if not context_id:
                continue
            opened = getattr(group, "opened_at", None)
            opened_at = _aware(opened) if isinstance(opened, datetime) else datetime.now(timezone.utc)
            self._rows[context_id] = {
                "context_id": context_id,
                "opened_at": opened_at.isoformat(),
                "selected_strategy": regret.selected_strategy,
                "selected_outcome_bps": float(regret.selected_outcome_bps),
                "best_outcome_bps": float(regret.best_outcome_bps),
                "regret_bps": float(regret.regret_bps),
                "top1_hit": bool(regret.top1_hit),
                "selected_from_live": bool(regret.selected_from_live),
                "alternative_count": int(regret.alternative_count),
            }
        if len(self._rows) > self.config.maximum_evidence_rows:
            ordered = sorted(self._rows.values(), key=lambda item: item.get("opened_at", ""))
            keep = ordered[-self.config.maximum_evidence_rows :]
            self._rows = {str(item["context_id"]): item for item in keep}

    def _metrics(self) -> dict[str, Any]:
        rows = sorted(self._rows.values(), key=lambda item: item.get("opened_at", ""))
        values = [float(item["selected_outcome_bps"]) for item in rows]
        regrets = [float(item["regret_bps"]) for item in rows]
        traded = [item for item in rows if item.get("selected_strategy")]
        wrong = [
            item
            for item in traded
            if float(item["selected_outcome_bps"]) < 0.0
            and float(item["best_outcome_bps"]) <= 0.0
        ]
        live = [item for item in traded if item.get("selected_from_live")]
        days = {str(item.get("opened_at", ""))[:10] for item in rows if item.get("opened_at")}
        live_days = {str(item.get("opened_at", ""))[:10] for item in live if item.get("opened_at")}
        windows = _window_means(values, self.config.minimum_chronological_windows)
        return {
            "context_count": len(rows),
            "traded_context_count": len(traded),
            "distinct_days": len(days),
            "mean_net_bps": _mean(values),
            "lower_bound_net_bps": _lower_bound(values),
            "cost_stressed_lower_bound_bps": _subtract_if_measured(
                _lower_bound(values), self.config.additional_cost_stress_bps
            ),
            "mean_regret_bps": _mean(regrets),
            "top1_hit_rate": _mean([1.0 if item.get("top1_hit") else 0.0 for item in rows]),
            "wrong_regime_trade_rate": (len(wrong) / len(traded) if traded else None),
            "chronological_window_means_bps": windows,
            "positive_window_count": sum(1 for value in windows if value > 0.0),
            "live_context_count": len(live),
            "live_distinct_days": len(live_days),
            "live_lower_bound_net_bps": _lower_bound(
                [float(item["selected_outcome_bps"]) for item in live]
            ),
        }

    def _promotion_failures(
        self, metrics: Mapping[str, Any], state: SelectorAuthorityState
    ) -> tuple[str, ...]:
        if state in {SelectorAuthorityState.LIVE, SelectorAuthorityState.SUSPENDED}:
            return ("SELECTOR_AUTHORITY_STATE_TERMINAL",)
        failures: list[str] = []
        if state is SelectorAuthorityState.SHADOW:
            if int(metrics.get("context_count") or 0) < self.config.minimum_contexts:
                failures.append(f"SELECTOR_CONTEXT_SAMPLE_BELOW_{self.config.minimum_contexts}")
            if int(metrics.get("traded_context_count") or 0) < self.config.minimum_traded_contexts:
                failures.append(f"SELECTOR_TRADED_SAMPLE_BELOW_{self.config.minimum_traded_contexts}")
            if int(metrics.get("distinct_days") or 0) < self.config.minimum_distinct_days:
                failures.append(f"SELECTOR_DAY_BREADTH_BELOW_{self.config.minimum_distinct_days}")
            if _lt(metrics.get("lower_bound_net_bps"), self.config.minimum_lower_bound_net_bps):
                failures.append("SELECTOR_NET_LOWER_BOUND_NOT_POSITIVE")
            if _lt(metrics.get("cost_stressed_lower_bound_bps"), 0.0):
                failures.append("SELECTOR_COST_STRESS_NOT_SURVIVED")
            if _gt(metrics.get("mean_regret_bps"), self.config.maximum_mean_regret_bps):
                failures.append("SELECTOR_REGRET_TOO_HIGH")
            if _gt(metrics.get("wrong_regime_trade_rate"), self.config.maximum_wrong_regime_trade_rate):
                failures.append("SELECTOR_WRONG_REGIME_RATE_TOO_HIGH")
            if _lt(metrics.get("top1_hit_rate"), self.config.minimum_top1_hit_rate):
                failures.append("SELECTOR_TOP1_HIT_RATE_TOO_LOW")
            if int(metrics.get("positive_window_count") or 0) < self.config.minimum_positive_windows:
                failures.append("SELECTOR_CHRONOLOGICAL_STABILITY_FAILED")
        else:
            if int(metrics.get("live_context_count") or 0) < self.config.minimum_live_probe_contexts:
                failures.append(f"SELECTOR_LIVE_SAMPLE_BELOW_{self.config.minimum_live_probe_contexts}")
            if int(metrics.get("live_distinct_days") or 0) < self.config.minimum_live_probe_days:
                failures.append(f"SELECTOR_LIVE_DAY_BREADTH_BELOW_{self.config.minimum_live_probe_days}")
            if _lt(metrics.get("live_lower_bound_net_bps"), self.config.live_minimum_lower_bound_net_bps):
                failures.append("SELECTOR_LIVE_NET_LOWER_BOUND_NOT_POSITIVE")
        return tuple(failures)

    def _flush(self) -> bool:
        payload = {
            "version": 1,
            "state": str(self._state),
            "consecutive_passes": self._passes,
            "rows": self._rows,
            "last_metrics": self._last_metrics,
            "history": self._history,
            "config": asdict(self.config),
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temporary.replace(self._path)
            return True
        except OSError:
            # Persistence failure cannot grant authority.  Existing in-memory state is
            # retained for this process; the next process starts from SHADOW.
            return False

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            state = SelectorAuthorityState(str(payload.get("state") or "SHADOW"))
            rows = payload.get("rows") or {}
            if not isinstance(rows, Mapping):
                return
        except (OSError, ValueError, TypeError):
            return
        # SUSPENDED is a fail-closed runtime fault, not an authorisation rung.
        # A clean process may resume collecting evidence, but only from SHADOW.
        self._state = (
            SelectorAuthorityState.SHADOW
            if state is SelectorAuthorityState.SUSPENDED
            else state
        )
        self._passes = max(0, int(payload.get("consecutive_passes") or 0))
        self._rows = {
            str(key): dict(value)
            for key, value in rows.items()
            if isinstance(value, Mapping)
        }
        self._last_metrics = dict(payload.get("last_metrics") or {})
        self._history = [dict(item) for item in payload.get("history") or () if isinstance(item, Mapping)][-200:]


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _lower_bound(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    return mean - 1.96 * standard_error


def _window_means(values: Sequence[float], windows: int) -> list[float]:
    count = max(1, int(windows))
    if len(values) < count:
        return []
    result: list[float] = []
    for index in range(count):
        start = index * len(values) // count
        end = (index + 1) * len(values) // count
        bucket = values[start:end]
        if bucket:
            result.append(statistics.fmean(bucket))
    return result


def _subtract_if_measured(value: float | None, amount: float) -> float | None:
    return value - amount if value is not None else None


def _lt(value: Any, threshold: float) -> bool:
    try:
        return not math.isfinite(float(value)) or float(value) < threshold
    except (TypeError, ValueError):
        return True


def _gt(value: Any, threshold: float) -> bool:
    try:
        return not math.isfinite(float(value)) or float(value) > threshold
    except (TypeError, ValueError):
        return True

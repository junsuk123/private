"""Thread-safe source freshness tracking used by the final order gates.

The runtime data directory is excluded from source-only archives, but this module is
code and must travel with the application.  Keep the implementation independent of
the collectors so it remains usable during startup and in diagnostics.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import yaml


DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[3] / "config" / "data_freshness.yaml"


class FreshnessState(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    STALE = "STALE"


_STATE_RANK = {
    FreshnessState.HEALTHY: 0,
    FreshnessState.DEGRADED: 1,
    FreshnessState.STALE: 2,
}


@dataclass(frozen=True)
class FreshnessPolicy:
    source: str
    data_type: str
    healthy_max_age_seconds: float
    degraded_max_age_seconds: float
    max_receive_lag_seconds: float = 10.0
    max_process_lag_seconds: float = 10.0
    critical: bool = False

    def __post_init__(self) -> None:
        if self.healthy_max_age_seconds < 0:
            raise ValueError("healthy_max_age_seconds must be non-negative")
        if self.degraded_max_age_seconds < self.healthy_max_age_seconds:
            raise ValueError("degraded_max_age_seconds must not be below the healthy limit")
        if self.max_receive_lag_seconds < 0 or self.max_process_lag_seconds < 0:
            raise ValueError("lag limits must be non-negative")


@dataclass(frozen=True)
class FreshnessReading:
    source: str
    data_type: str
    scope_key: str
    state: FreshnessState
    critical: bool
    event_time: datetime | None
    received_time: datetime | None
    processed_time: datetime | None
    age_seconds: float | None
    receive_lag_seconds: float | None
    process_lag_seconds: float | None
    reason_codes: tuple[str, ...] = ()

    @property
    def blocks_new_entry(self) -> bool:
        return self.critical and self.state is FreshnessState.STALE

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        for key in ("event_time", "received_time", "processed_time"):
            value = payload[key]
            payload[key] = value.isoformat() if value is not None else None
        payload["blocks_new_entry"] = self.blocks_new_entry
        return payload


@dataclass(frozen=True)
class _Observation:
    event_time: datetime | None
    received_time: datetime | None
    processed_time: datetime | None


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _policy_from_mapping(
    values: dict[str, Any],
    *,
    defaults: dict[str, Any],
    source: str,
    data_type: str,
) -> FreshnessPolicy:
    merged = {**defaults, **values}
    return FreshnessPolicy(
        source=source,
        data_type=data_type,
        healthy_max_age_seconds=float(merged["healthy_max_age_seconds"]),
        degraded_max_age_seconds=float(merged["degraded_max_age_seconds"]),
        max_receive_lag_seconds=float(merged.get("max_receive_lag_seconds", 10.0)),
        max_process_lag_seconds=float(merged.get("max_process_lag_seconds", 10.0)),
        critical=bool(merged.get("critical", False)),
    )


def load_freshness_policies(
    path: str | Path = DEFAULT_POLICY_PATH,
) -> tuple[dict[tuple[str, str], FreshnessPolicy], FreshnessPolicy]:
    policy_path = Path(path)
    payload = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
    defaults = dict(payload.get("defaults") or {})
    default = _policy_from_mapping(
        {}, defaults=defaults, source="*", data_type="*"
    )
    policies: dict[tuple[str, str], FreshnessPolicy] = {}
    for raw in payload.get("policies") or ():
        values = dict(raw or {})
        source = str(values.pop("source", "")).strip()
        data_type = str(values.pop("data_type", "")).strip()
        if not source or not data_type:
            raise ValueError(f"invalid freshness policy in {policy_path}")
        policies[(source, data_type)] = _policy_from_mapping(
            values,
            defaults=defaults,
            source=source,
            data_type=data_type,
        )
    return policies, default


class DataFreshnessRegistry:
    def __init__(
        self,
        *,
        policies: dict[tuple[str, str], FreshnessPolicy] | None = None,
        default_policy: FreshnessPolicy | None = None,
        policy_path: str | Path = DEFAULT_POLICY_PATH,
    ) -> None:
        if policies is None or default_policy is None:
            loaded, loaded_default = load_freshness_policies(policy_path)
            policies = loaded if policies is None else policies
            default_policy = loaded_default if default_policy is None else default_policy
        self._policies = dict(policies)
        self._default = default_policy
        self._observations: dict[tuple[str, str, str], _Observation] = {}
        self._lock = threading.RLock()

    def policy_for(self, source: str, data_type: str) -> FreshnessPolicy:
        return self._policies.get((str(source), str(data_type))) or FreshnessPolicy(
            source=str(source),
            data_type=str(data_type),
            healthy_max_age_seconds=self._default.healthy_max_age_seconds,
            degraded_max_age_seconds=self._default.degraded_max_age_seconds,
            max_receive_lag_seconds=self._default.max_receive_lag_seconds,
            max_process_lag_seconds=self._default.max_process_lag_seconds,
            critical=self._default.critical,
        )

    def expect(self, source: str, data_type: str, *, scope_key: str = "") -> None:
        key = (str(source), str(data_type), str(scope_key or "").strip())
        with self._lock:
            self._observations.setdefault(key, _Observation(None, None, None))

    def expect_all(self) -> None:
        for source, data_type in self._policies:
            self.expect(source, data_type)

    def record_event(
        self,
        source: str,
        data_type: str,
        event_time: datetime,
        *,
        scope_key: str = "",
        received_time: datetime | None = None,
        processed_time: datetime | None = None,
        now: datetime | None = None,
    ) -> FreshnessReading:
        event = _aware(event_time)
        received = _aware(received_time) or event
        processed = _aware(processed_time) or received
        key = (str(source), str(data_type), str(scope_key or "").strip())
        with self._lock:
            self._observations[key] = _Observation(event, received, processed)
        return self.reading(*key[:2], scope_key=key[2], now=now)

    def reading(
        self,
        source: str,
        data_type: str,
        *,
        scope_key: str = "",
        now: datetime | None = None,
    ) -> FreshnessReading:
        key = (str(source), str(data_type), str(scope_key or "").strip())
        with self._lock:
            observation = self._observations.get(key)
        policy = self.policy_for(key[0], key[1])
        if observation is None or observation.event_time is None:
            return FreshnessReading(
                *key,
                state=FreshnessState.STALE,
                critical=policy.critical,
                event_time=None,
                received_time=None,
                processed_time=None,
                age_seconds=None,
                receive_lag_seconds=None,
                process_lag_seconds=None,
                reason_codes=("FRESHNESS_NO_OBSERVATION",),
            )

        moment = _aware(now) or datetime.now(timezone.utc)
        event = observation.event_time
        received = observation.received_time or event
        processed = observation.processed_time or received
        age = (moment - event).total_seconds()
        receive_lag = (received - event).total_seconds()
        process_lag = (processed - received).total_seconds()
        reasons: list[str] = []

        if age < -1.0 or receive_lag < -1.0 or process_lag < -1.0:
            state = FreshnessState.STALE
            reasons.append("FRESHNESS_CLOCK_SKEW")
        elif age > policy.degraded_max_age_seconds:
            state = FreshnessState.STALE
            reasons.append("FRESHNESS_AGE_STALE")
        elif age > policy.healthy_max_age_seconds:
            state = FreshnessState.DEGRADED
            reasons.append("FRESHNESS_AGE_DEGRADED")
        else:
            state = FreshnessState.HEALTHY

        if receive_lag > policy.max_receive_lag_seconds:
            reasons.append("FRESHNESS_RECEIVE_LAG")
            if state is FreshnessState.HEALTHY:
                state = FreshnessState.DEGRADED
        if process_lag > policy.max_process_lag_seconds:
            reasons.append("FRESHNESS_PROCESS_LAG")
            if state is FreshnessState.HEALTHY:
                state = FreshnessState.DEGRADED

        return FreshnessReading(
            *key,
            state=state,
            critical=policy.critical,
            event_time=event,
            received_time=received,
            processed_time=processed,
            age_seconds=max(0.0, age),
            receive_lag_seconds=max(0.0, receive_lag),
            process_lag_seconds=max(0.0, process_lag),
            reason_codes=tuple(dict.fromkeys(reasons)),
        )

    def readings(
        self,
        *,
        now: datetime | None = None,
        scope_keys: Iterable[str] | None = None,
    ) -> tuple[FreshnessReading, ...]:
        with self._lock:
            keys = tuple(sorted(self._observations))
        if scope_keys is not None:
            wanted = {str(value or "").strip().upper() for value in scope_keys}
            # Unscoped infrastructure/account observations apply to every order.
            keys = tuple(
                key for key in keys
                if not key[2] or str(key[2]).strip().upper() in wanted
            )
        return tuple(
            self.reading(source, data_type, scope_key=scope, now=now)
            for source, data_type, scope in keys
        )

    def blocking_reasons(
        self,
        *,
        now: datetime | None = None,
        scope_keys: Iterable[str] | None = None,
    ) -> tuple[str, ...]:
        reasons = []
        for item in self.readings(now=now, scope_keys=scope_keys):
            if not item.blocks_new_entry:
                continue
            suffix = f":{item.scope_key}" if item.scope_key else ""
            reasons.append(f"STALE_DATA:{item.source}/{item.data_type}{suffix}")
        return tuple(reasons)

    def worst_state(
        self,
        *,
        now: datetime | None = None,
        scope_keys: Iterable[str] | None = None,
    ) -> FreshnessState:
        items = self.readings(now=now, scope_keys=scope_keys)
        if not items:
            return FreshnessState.HEALTHY
        return max((item.state for item in items), key=_STATE_RANK.__getitem__)

    def report(
        self,
        *,
        now: datetime | None = None,
        scope_keys: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        items = self.readings(now=now, scope_keys=scope_keys)
        counts = {state.value: 0 for state in FreshnessState}
        for item in items:
            counts[item.state.value] += 1
        return {
            "worst_state": self.worst_state(now=now, scope_keys=scope_keys).value,
            "counts": counts,
            "blocking_reasons": list(
                self.blocking_reasons(now=now, scope_keys=scope_keys)
            ),
            "scope_filter": (
                sorted({str(value or "").strip().upper() for value in scope_keys})
                if scope_keys is not None else None
            ),
            "streams": [item.as_dict() for item in items],
        }


_DEFAULT_REGISTRY: DataFreshnessRegistry | None = None
_DEFAULT_LOCK = threading.Lock()


def default_freshness_registry() -> DataFreshnessRegistry:
    global _DEFAULT_REGISTRY
    with _DEFAULT_LOCK:
        if _DEFAULT_REGISTRY is None:
            _DEFAULT_REGISTRY = DataFreshnessRegistry()
        return _DEFAULT_REGISTRY

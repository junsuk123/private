"""Data freshness: three timestamps, three states, one fail-closed answer.

Every observation carries three times, and they answer different questions:

``event_time``
    When the exchange says it happened. ``age = now - event_time`` is what a strategy
    cares about.
``received_time``
    When this process took delivery. ``received_time - event_time`` is *feed* lag.
``processed_time``
    When this process finished turning it into a feature. ``processed_time -
    received_time`` is *our own* lag.

Age alone cannot separate "the venue is quiet" from "the websocket stalled" from "our
pipeline is behind", and those three call for completely different responses. Keeping all
three means the reason code says which one it is.

States and what they permit
---------------------------
``HEALTHY`` / ``DEGRADED`` / ``STALE``, per ``(source, data_type)`` from
``config/data_freshness.yaml``. A **critical** input that goes ``STALE`` blocks new orders
— that is the hard ``STALE_DATA`` gate. It does **not** block exits: being unable to close
a position is a worse failure than being unable to open one, and a stale feed is exactly
when a position most needs closing.

An input with **no observation at all** is ``STALE``, not absent. "We never heard from the
order book" and "the order book is 400ms old" must not resolve the same way, and the
fail-closed reading of the first is the unsafe one.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "DataFreshnessRegistry",
    "FreshnessObservation",
    "FreshnessPolicy",
    "FreshnessReading",
    "FreshnessState",
    "default_freshness_registry",
    "load_freshness_policies",
    "reset_default_freshness_registry",
]

DEFAULT_CONFIG_PATH = Path("config/data_freshness.yaml")

FRESHNESS_NO_OBSERVATION = "FRESHNESS_NO_OBSERVATION"
FRESHNESS_AGE_DEGRADED = "FRESHNESS_AGE_DEGRADED"
FRESHNESS_AGE_STALE = "FRESHNESS_AGE_STALE"
FRESHNESS_RECEIVE_LAG = "FRESHNESS_RECEIVE_LAG"
FRESHNESS_PROCESS_LAG = "FRESHNESS_PROCESS_LAG"
FRESHNESS_CLOCK_SKEW = "FRESHNESS_CLOCK_SKEW"

#: An event_time this far in the future is a clock problem, not a fresh observation.
#: Treating it as fresh would let a mis-set exchange clock defeat every staleness check.
CLOCK_SKEW_TOLERANCE_SECONDS = 5.0


class FreshnessState(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    STALE = "STALE"


@dataclass(frozen=True)
class FreshnessPolicy:
    source: str
    data_type: str
    healthy_max_age_seconds: float = 60.0
    degraded_max_age_seconds: float = 300.0
    max_receive_lag_seconds: float | None = 10.0
    max_process_lag_seconds: float | None = 10.0
    critical: bool = False

    @property
    def key(self) -> tuple[str, str]:
        return (self.source, self.data_type)

    def __post_init__(self) -> None:
        if self.degraded_max_age_seconds < self.healthy_max_age_seconds:
            raise ValueError(
                f"{self.source}/{self.data_type}: degraded_max_age_seconds must not be "
                "below healthy_max_age_seconds"
            )


@dataclass(frozen=True)
class FreshnessObservation:
    """One reading of one stream."""

    source: str
    data_type: str
    event_time: datetime
    received_time: datetime | None = None
    processed_time: datetime | None = None
    scope_key: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FreshnessReading:
    """The state of one ``(source, data_type, scope)`` at an instant."""

    source: str
    data_type: str
    scope_key: str
    state: FreshnessState
    observed_at: datetime
    age_seconds: float | None
    receive_lag_seconds: float | None
    process_lag_seconds: float | None
    critical: bool
    event_time: datetime | None = None
    received_time: datetime | None = None
    processed_time: datetime | None = None
    reason_codes: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source, self.data_type, self.scope_key)

    @property
    def blocks_new_entry(self) -> bool:
        return self.critical and self.state is FreshnessState.STALE

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "data_type": self.data_type,
            "scope_key": self.scope_key,
            "state": self.state.value,
            "observed_at": self.observed_at.isoformat(),
            "age_seconds": self.age_seconds,
            "receive_lag_seconds": self.receive_lag_seconds,
            "process_lag_seconds": self.process_lag_seconds,
            "critical": self.critical,
            "event_time": self.event_time.isoformat() if self.event_time else None,
            "received_time": (
                self.received_time.isoformat() if self.received_time else None
            ),
            "processed_time": (
                self.processed_time.isoformat() if self.processed_time else None
            ),
            "reason_codes": list(self.reason_codes),
            "blocks_new_entry": self.blocks_new_entry,
        }


def _aware(moment: datetime) -> datetime:
    return (
        moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    ).astimezone(timezone.utc)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def load_freshness_policies(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[dict[tuple[str, str], FreshnessPolicy], FreshnessPolicy]:
    """Policies keyed by ``(source, data_type)``, plus the default for unlisted streams."""
    target = Path(path)
    fallback = FreshnessPolicy(source="*", data_type="*")
    if not target.exists():
        return {}, fallback
    try:
        import yaml

        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 - a malformed freshness policy is fatal.
        raise RuntimeError(f"cannot parse {target}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"{target} must be a mapping")

    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, Mapping):
        raise RuntimeError(f"{target}: defaults must be a mapping")
    fallback = FreshnessPolicy(
        source="*",
        data_type="*",
        healthy_max_age_seconds=float(defaults.get("healthy_max_age_seconds", 60.0)),
        degraded_max_age_seconds=float(defaults.get("degraded_max_age_seconds", 300.0)),
        max_receive_lag_seconds=_optional_float(defaults.get("max_receive_lag_seconds")),
        max_process_lag_seconds=_optional_float(defaults.get("max_process_lag_seconds")),
        critical=bool(defaults.get("critical", False)),
    )

    policies: dict[tuple[str, str], FreshnessPolicy] = {}
    for entry in raw.get("policies") or ():
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"{target}: every policy must be a mapping")
        policy = FreshnessPolicy(
            source=str(entry["source"]),
            data_type=str(entry["data_type"]),
            healthy_max_age_seconds=float(
                entry.get("healthy_max_age_seconds", fallback.healthy_max_age_seconds)
            ),
            degraded_max_age_seconds=float(
                entry.get("degraded_max_age_seconds", fallback.degraded_max_age_seconds)
            ),
            max_receive_lag_seconds=_optional_float(
                entry.get("max_receive_lag_seconds", fallback.max_receive_lag_seconds)
            ),
            max_process_lag_seconds=_optional_float(
                entry.get("max_process_lag_seconds", fallback.max_process_lag_seconds)
            ),
            critical=bool(entry.get("critical", fallback.critical)),
        )
        policies[policy.key] = policy
    return policies, fallback


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


class DataFreshnessRegistry:
    """Records observations and answers "is this input usable right now".

    Thread-safe. Writers call :meth:`record` from the feed threads; the trading loop calls
    :meth:`readings` and :meth:`blocking_reasons` once per cycle.
    """

    def __init__(
        self,
        policies: Mapping[tuple[str, str], FreshnessPolicy] | None = None,
        *,
        default_policy: FreshnessPolicy | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        if policies is None:
            loaded, fallback = load_freshness_policies(config_path)
            self._policies = dict(loaded)
            self._default = default_policy or fallback
        else:
            self._policies = dict(policies)
            self._default = default_policy or FreshnessPolicy(source="*", data_type="*")
        self._lock = threading.RLock()
        self._observations: dict[tuple[str, str, str], FreshnessObservation] = {}
        #: Streams that are expected to report. Registered explicitly so a feed that
        #: never starts is STALE rather than invisible.
        self._expected: set[tuple[str, str, str]] = set()

    # ------------------------------------------------------------------ #
    @property
    def policies(self) -> Mapping[tuple[str, str], FreshnessPolicy]:
        return dict(self._policies)

    def policy_for(self, source: str, data_type: str) -> FreshnessPolicy:
        return self._policies.get((str(source), str(data_type)), self._default)

    def expect(self, source: str, data_type: str, scope_key: str = "") -> None:
        """Declare that a stream must report. Until it does, it reads STALE."""
        with self._lock:
            self._expected.add((str(source), str(data_type), str(scope_key)))

    def expect_all(self, streams: Iterable[tuple[str, str]] | None = None) -> None:
        targets = streams if streams is not None else list(self._policies)
        for source, data_type in targets:
            self.expect(source, data_type)

    def record(
        self, observation: FreshnessObservation, *, now: datetime | None = None
    ) -> FreshnessReading:
        """Store an observation and return its reading.

        ``now`` is the instant the reading is evaluated against; it defaults to the wall
        clock. Callers replaying history must pass it, or every replayed observation
        reads as stale against the present.
        """
        key = (
            str(observation.source),
            str(observation.data_type),
            str(observation.scope_key),
        )
        with self._lock:
            self._observations[key] = observation
            self._expected.add(key)
        return self.reading(*key, now=now)

    def record_event(
        self,
        source: str,
        data_type: str,
        event_time: datetime,
        *,
        scope_key: str = "",
        received_time: datetime | None = None,
        processed_time: datetime | None = None,
        detail: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> FreshnessReading:
        return self.record(
            FreshnessObservation(
                source=source,
                data_type=data_type,
                event_time=event_time,
                received_time=received_time,
                processed_time=processed_time,
                scope_key=scope_key,
                detail=detail or {},
            ),
            now=now,
        )

    # ------------------------------------------------------------------ #
    def reading(
        self,
        source: str,
        data_type: str,
        scope_key: str = "",
        *,
        now: datetime | None = None,
    ) -> FreshnessReading:
        moment = _aware(now or _utcnow())
        policy = self.policy_for(source, data_type)
        key = (str(source), str(data_type), str(scope_key))
        with self._lock:
            observation = self._observations.get(key)
        if observation is None:
            return FreshnessReading(
                source=key[0],
                data_type=key[1],
                scope_key=key[2],
                state=FreshnessState.STALE,
                observed_at=moment,
                age_seconds=None,
                receive_lag_seconds=None,
                process_lag_seconds=None,
                critical=policy.critical,
                reason_codes=(FRESHNESS_NO_OBSERVATION,),
            )
        return self._evaluate(observation, policy, moment)

    def readings(self, *, now: datetime | None = None) -> tuple[FreshnessReading, ...]:
        moment = _aware(now or _utcnow())
        with self._lock:
            keys = set(self._expected) | set(self._observations)
        return tuple(
            self.reading(source, data_type, scope_key, now=moment)
            for source, data_type, scope_key in sorted(keys)
        )

    def blocking_reasons(self, *, now: datetime | None = None) -> tuple[str, ...]:
        """Reason codes for critical inputs that are stale. Empty means clear.

        Each is rendered ``STALE_DATA:<source>/<data_type>[:<scope>]`` so the gate's
        rejection names the stream that caused it rather than a generic code an operator
        then has to hunt down.
        """
        codes: list[str] = []
        for reading in self.readings(now=now):
            if not reading.blocks_new_entry:
                continue
            suffix = f":{reading.scope_key}" if reading.scope_key else ""
            codes.append(f"STALE_DATA:{reading.source}/{reading.data_type}{suffix}")
        return tuple(codes)

    def worst_state(self, *, now: datetime | None = None) -> FreshnessState:
        states = [reading.state for reading in self.readings(now=now)]
        if any(state is FreshnessState.STALE for state in states):
            return FreshnessState.STALE
        if any(state is FreshnessState.DEGRADED for state in states):
            return FreshnessState.DEGRADED
        return FreshnessState.HEALTHY

    def report(self, *, now: datetime | None = None) -> dict[str, Any]:
        moment = _aware(now or _utcnow())
        readings = self.readings(now=moment)
        by_state: dict[str, int] = {state.value: 0 for state in FreshnessState}
        for reading in readings:
            by_state[reading.state.value] += 1
        return {
            "as_of": moment.isoformat(),
            "worst_state": self.worst_state(now=moment).value,
            "counts": by_state,
            "blocking_reasons": list(self.blocking_reasons(now=moment)),
            "streams": [reading.as_dict() for reading in readings],
        }

    # ------------------------------------------------------------------ #
    def _evaluate(
        self,
        observation: FreshnessObservation,
        policy: FreshnessPolicy,
        now: datetime,
    ) -> FreshnessReading:
        event_time = _aware(observation.event_time)
        received = _aware(observation.received_time) if observation.received_time else None
        processed = (
            _aware(observation.processed_time) if observation.processed_time else None
        )
        age = (now - event_time).total_seconds()
        reasons: list[str] = []
        state = FreshnessState.HEALTHY

        if age < -CLOCK_SKEW_TOLERANCE_SECONDS:
            # An event stamped in the future cannot be trusted to be fresh; the safe
            # reading is the one that blocks rather than the one that permits.
            reasons.append(FRESHNESS_CLOCK_SKEW)
            state = FreshnessState.STALE
        elif age > policy.degraded_max_age_seconds:
            reasons.append(FRESHNESS_AGE_STALE)
            state = FreshnessState.STALE
        elif age > policy.healthy_max_age_seconds:
            reasons.append(FRESHNESS_AGE_DEGRADED)
            state = FreshnessState.DEGRADED

        receive_lag = (received - event_time).total_seconds() if received else None
        if (
            receive_lag is not None
            and policy.max_receive_lag_seconds is not None
            and receive_lag > policy.max_receive_lag_seconds
        ):
            reasons.append(FRESHNESS_RECEIVE_LAG)
            state = _worse(state, FreshnessState.DEGRADED)

        process_lag = (
            (processed - received).total_seconds() if processed and received else None
        )
        if (
            process_lag is not None
            and policy.max_process_lag_seconds is not None
            and process_lag > policy.max_process_lag_seconds
        ):
            reasons.append(FRESHNESS_PROCESS_LAG)
            state = _worse(state, FreshnessState.DEGRADED)

        return FreshnessReading(
            source=observation.source,
            data_type=observation.data_type,
            scope_key=observation.scope_key,
            state=state,
            observed_at=now,
            age_seconds=round(age, 6),
            receive_lag_seconds=None if receive_lag is None else round(receive_lag, 6),
            process_lag_seconds=None if process_lag is None else round(process_lag, 6),
            critical=policy.critical,
            event_time=event_time,
            received_time=received,
            processed_time=processed,
            reason_codes=tuple(dict.fromkeys(reasons)),
        )


_ORDER = {FreshnessState.HEALTHY: 0, FreshnessState.DEGRADED: 1, FreshnessState.STALE: 2}


def _worse(left: FreshnessState, right: FreshnessState) -> FreshnessState:
    return left if _ORDER[left] >= _ORDER[right] else right


_default_registry: DataFreshnessRegistry | None = None
_registry_lock = threading.Lock()


def default_freshness_registry() -> DataFreshnessRegistry:
    global _default_registry
    with _registry_lock:
        if _default_registry is None:
            _default_registry = DataFreshnessRegistry()
        return _default_registry


def reset_default_freshness_registry() -> None:
    """Test hook. Never called from the trading path."""
    global _default_registry
    with _registry_lock:
        _default_registry = None

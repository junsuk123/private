"""The lifecycle ledger: what each strategy has earned, and who changed it.

Distinct from ``app.strategy.registry`` on purpose. That module reports the strategy's
DECLARED identity and requirements, derived from code and config. This one records
VALIDATION EVIDENCE and lifecycle transitions, which are facts about the past and must be
persisted rather than derived — a lifecycle state that could be recomputed from code would be
silently rewritten by every code change.

Transitions are a whitelist, not a rank comparison
--------------------------------------------------
Copied in spirit from ``app.trading.directional.ALLOWED_TRANSITIONS`` and for the same
reason: a rank comparison would permit ``RESEARCH -> LIVE`` as "an increase of 4", which is
the single transition this whole subsystem exists to forbid. A promotion also requires
evidence attached to it (``StrategyValidationRecord``); a promotion with no record is
rejected, so "we think it's fine" cannot become an authorisation.

Demotions are deliberately easier than promotions: any downward transition in the ladder is
allowed and needs no evidence, because refusing to demote without a completed study is how a
broken strategy keeps trading.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from app.strategy.spec import StrategyLifecycleState

__all__ = [
    "ALLOWED_PROMOTIONS",
    "LifecycleTransition",
    "StrategyValidationRecord",
    "StrategyValidationRegistry",
]

#: Promotions that may ever happen, one rung at a time.
ALLOWED_PROMOTIONS: frozenset[tuple[StrategyLifecycleState, StrategyLifecycleState]] = (
    frozenset(
        {
            (StrategyLifecycleState.RESEARCH, StrategyLifecycleState.VALIDATED),
            (StrategyLifecycleState.VALIDATED, StrategyLifecycleState.SHADOW),
            (StrategyLifecycleState.SHADOW, StrategyLifecycleState.LIVE_PROBE),
            (StrategyLifecycleState.LIVE_PROBE, StrategyLifecycleState.LIVE),
            # Recovery from a drift demotion re-enters the ladder at SHADOW, never above.
            (StrategyLifecycleState.DEGRADED, StrategyLifecycleState.SHADOW),
        }
    )
)

#: Demotions. ``RETIRED`` is reachable from anywhere and is terminal.
_LADDER_ORDER: tuple[StrategyLifecycleState, ...] = (
    StrategyLifecycleState.RESEARCH,
    StrategyLifecycleState.VALIDATED,
    StrategyLifecycleState.SHADOW,
    StrategyLifecycleState.LIVE_PROBE,
    StrategyLifecycleState.LIVE,
)


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class StrategyValidationRecord:
    """Evidence attached to one strategy at one point in time.

    Every numeric field is ``None``-able and ``None`` means UNMEASURED. That is not a
    formality: most of this catalogue is genuinely unmeasured, and the task is explicit that
    missing data must read as insufficient rather than as a number.
    """

    strategy_id: str
    validated_at: datetime
    validation_version: str
    algorithm_version: str
    sample_count: int = 0
    effective_sample_count: float = 0.0
    net_ev_bps: float | None = None
    gross_ev_bps: float | None = None
    lower_confidence_bound_bps: float | None = None
    profit_factor: float | None = None
    cost_to_edge_ratio: float | None = None
    break_even_cost_multiple: float | None = None
    out_of_sample_stability: float | None = None
    parameter_stability: bool | None = None
    approved_markets: tuple[str, ...] = ()
    approved_regimes: tuple[str, ...] = ()
    evidence_mix: Mapping[str, int] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()

    @property
    def has_live_evidence(self) -> bool:
        return any(
            str(source).upper() in {"LIVE", "LIVE_PROBE"} and count > 0
            for source, count in dict(self.evidence_mix).items()
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "validated_at": _aware(self.validated_at).isoformat(),
            "validation_version": self.validation_version,
            "algorithm_version": self.algorithm_version,
            "sample_count": self.sample_count,
            "effective_sample_count": round(self.effective_sample_count, 3),
            "net_ev_bps": self.net_ev_bps,
            "gross_ev_bps": self.gross_ev_bps,
            "lower_confidence_bound_bps": self.lower_confidence_bound_bps,
            "profit_factor": self.profit_factor,
            "cost_to_edge_ratio": self.cost_to_edge_ratio,
            "break_even_cost_multiple": self.break_even_cost_multiple,
            "out_of_sample_stability": self.out_of_sample_stability,
            "parameter_stability": self.parameter_stability,
            "approved_markets": list(self.approved_markets),
            "approved_regimes": list(self.approved_regimes),
            "evidence_mix": dict(self.evidence_mix),
            "has_live_evidence": self.has_live_evidence,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StrategyValidationRecord":
        return cls(
            strategy_id=str(payload.get("strategy_id") or ""),
            validated_at=_parse(payload.get("validated_at")) or datetime.now(timezone.utc),
            validation_version=str(payload.get("validation_version") or "unvalidated"),
            algorithm_version=str(payload.get("algorithm_version") or "0.0.0"),
            sample_count=int(payload.get("sample_count") or 0),
            effective_sample_count=float(payload.get("effective_sample_count") or 0.0),
            net_ev_bps=_optional(payload.get("net_ev_bps")),
            gross_ev_bps=_optional(payload.get("gross_ev_bps")),
            lower_confidence_bound_bps=_optional(payload.get("lower_confidence_bound_bps")),
            profit_factor=_optional(payload.get("profit_factor")),
            cost_to_edge_ratio=_optional(payload.get("cost_to_edge_ratio")),
            break_even_cost_multiple=_optional(payload.get("break_even_cost_multiple")),
            out_of_sample_stability=_optional(payload.get("out_of_sample_stability")),
            parameter_stability=(
                bool(payload["parameter_stability"])
                if payload.get("parameter_stability") is not None
                else None
            ),
            approved_markets=tuple(payload.get("approved_markets") or ()),
            approved_regimes=tuple(payload.get("approved_regimes") or ()),
            evidence_mix=dict(payload.get("evidence_mix") or {}),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )


@dataclass(frozen=True)
class LifecycleTransition:
    strategy_id: str
    from_state: StrategyLifecycleState
    to_state: StrategyLifecycleState
    at: datetime
    actor: str
    reason_codes: tuple[str, ...]
    record: StrategyValidationRecord | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "from_state": str(self.from_state),
            "to_state": str(self.to_state),
            "at": _aware(self.at).isoformat(),
            "actor": self.actor,
            "reason_codes": list(self.reason_codes),
            "record": self.record.as_dict() if self.record else None,
        }


@dataclass(frozen=True)
class PromotionGates:
    """What a promotion into a trading state requires.

    Only the two transitions that put real money at risk are gated. Everything below
    LIVE_PROBE is a bookkeeping move between non-trading states, and gating those buys
    nothing but friction.
    """

    minimum_samples: int = 30
    minimum_lower_bound_bps: float = 0.0
    minimum_break_even_cost_multiple: float = 1.25
    minimum_out_of_sample_stability: float = 0.6
    require_parameter_stability: bool = True
    require_live_evidence_for_live: bool = True


class StrategyValidationRegistry:
    """Persisted lifecycle state + evidence, with a whitelisted transition graph."""

    def __init__(
        self,
        *,
        state_path: str | Path | None = "data/store/strategy-validation.json",
        gates: PromotionGates | None = None,
    ) -> None:
        self._path = Path(state_path) if state_path else None
        self._gates = gates or PromotionGates()
        self._lock = threading.RLock()
        self._states: dict[str, StrategyLifecycleState] = {}
        self._records: dict[str, StrategyValidationRecord] = {}
        self._history: list[LifecycleTransition] = []
        self._load()

    # -- reads -------------------------------------------------------------- #
    def state(self, strategy_id: str) -> StrategyLifecycleState | None:
        with self._lock:
            return self._states.get(_norm(strategy_id))

    def states(self) -> dict[str, StrategyLifecycleState]:
        with self._lock:
            return dict(self._states)

    def record(self, strategy_id: str) -> StrategyValidationRecord | None:
        with self._lock:
            return self._records.get(_norm(strategy_id))

    def history(self, *, limit: int = 200) -> tuple[LifecycleTransition, ...]:
        with self._lock:
            return tuple(self._history[-max(0, int(limit)) :])

    def table(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "strategy_id": strategy_id,
                    "lifecycle_state": str(state),
                    **(
                        self._records[strategy_id].as_dict()
                        if strategy_id in self._records
                        else {}
                    ),
                }
                for strategy_id, state in sorted(self._states.items())
            ]

    # -- writes ------------------------------------------------------------- #
    def upsert_record(self, record: StrategyValidationRecord) -> StrategyValidationRecord:
        """Store evidence. Storing evidence NEVER changes lifecycle state."""
        with self._lock:
            self._records[_norm(record.strategy_id)] = record
        return record

    def transition(
        self,
        strategy_id: str,
        *,
        to_state: StrategyLifecycleState,
        actor: str,
        reason_codes: Sequence[str] = (),
        record: StrategyValidationRecord | None = None,
        now: datetime | None = None,
    ) -> tuple[bool, tuple[str, ...]]:
        """Attempt a transition. Returns ``(applied, reason_codes)``.

        A refusal is a normal outcome and carries its reasons, so a caller can report why a
        promotion did not happen without inspecting the registry's internals.
        """
        key = _norm(strategy_id)
        moment = _aware(now or datetime.now(timezone.utc))
        with self._lock:
            current = self._states.get(key, StrategyLifecycleState.RESEARCH)
            if current is to_state:
                return False, ("LIFECYCLE_NO_CHANGE",)
            if current is StrategyLifecycleState.RETIRED:
                return False, ("LIFECYCLE_RETIRED_IS_TERMINAL",)

            promoting = _is_promotion(current, to_state)
            if promoting:
                if (current, to_state) not in ALLOWED_PROMOTIONS:
                    return False, (
                        f"LIFECYCLE_TRANSITION_NOT_ALLOWED:{current}->{to_state}",
                    )
                evidence = record or self._records.get(key)
                if to_state in {
                    StrategyLifecycleState.LIVE_PROBE,
                    StrategyLifecycleState.LIVE,
                }:
                    failures = self._gate_failures(evidence, to_state)
                    if failures:
                        return False, failures

            applied_record = record or self._records.get(key)
            if record is not None:
                self._records[key] = record
            self._states[key] = to_state
            self._history.append(
                LifecycleTransition(
                    strategy_id=key,
                    from_state=current,
                    to_state=to_state,
                    at=moment,
                    actor=str(actor or "unknown"),
                    reason_codes=tuple(dict.fromkeys(str(code) for code in reason_codes)),
                    record=applied_record,
                )
            )
            self._history = self._history[-1000:]
        self.flush()
        return True, ("LIFECYCLE_TRANSITION_APPLIED",)

    def apply_demotion_proposals(
        self, proposals: Iterable[Any], *, actor: str = "drift_monitor"
    ) -> tuple[LifecycleTransition, ...]:
        """Apply :class:`~app.monitoring.strategy_drift.DemotionProposal` objects.

        Demotions need no evidence gate — see the module docstring — but they are still
        recorded in the history with the monitor as the actor, so an operator can see that a
        state change was automatic rather than human.
        """
        applied: list[LifecycleTransition] = []
        for proposal in proposals or ():
            strategy_id = str(getattr(proposal, "strategy_id", "") or "")
            to_state = getattr(proposal, "to_state", None)
            if not strategy_id or to_state is None:
                continue
            ok, _ = self.transition(
                strategy_id,
                to_state=to_state,
                actor=actor,
                reason_codes=tuple(getattr(proposal, "reason_codes", ()) or ()),
            )
            if ok:
                latest = self.history(limit=1)
                if latest:
                    applied.append(latest[0])
        return tuple(applied)

    def seed_from(self, states: Mapping[str, StrategyLifecycleState]) -> None:
        """Initialise unknown strategies from the declared registry.

        Only fills gaps. An already-recorded state is authoritative, because it was reached
        through a transition and the declared one is merely what the code currently says.
        """
        with self._lock:
            for strategy_id, state in states.items():
                self._states.setdefault(_norm(strategy_id), state)
        self.flush()

    # -- gates -------------------------------------------------------------- #
    def _gate_failures(
        self, record: StrategyValidationRecord | None, to_state: StrategyLifecycleState
    ) -> tuple[str, ...]:
        gates = self._gates
        if record is None:
            return ("LIFECYCLE_NO_VALIDATION_RECORD",)
        failures: list[str] = []
        if record.sample_count < gates.minimum_samples:
            failures.append(f"LIFECYCLE_SAMPLE_BELOW_{gates.minimum_samples}")
        if (
            record.lower_confidence_bound_bps is None
            or record.lower_confidence_bound_bps <= gates.minimum_lower_bound_bps
        ):
            # The primary criterion: a positive LOWER bound after cost, not a positive mean.
            failures.append("LIFECYCLE_LOWER_BOUND_NOT_POSITIVE")
        if (
            record.break_even_cost_multiple is None
            or record.break_even_cost_multiple < gates.minimum_break_even_cost_multiple
        ):
            failures.append("LIFECYCLE_COST_STRESS_NOT_SURVIVED")
        if (
            record.out_of_sample_stability is None
            or record.out_of_sample_stability < gates.minimum_out_of_sample_stability
        ):
            failures.append("LIFECYCLE_OUT_OF_SAMPLE_UNSTABLE")
        if gates.require_parameter_stability and record.parameter_stability is not True:
            failures.append("LIFECYCLE_PARAMETERS_UNSTABLE")
        if (
            to_state is StrategyLifecycleState.LIVE
            and gates.require_live_evidence_for_live
            and not record.has_live_evidence
        ):
            # Shadow evidence cannot promote to LIVE. A strategy cannot be promoted on
            # trades it never took.
            failures.append("LIFECYCLE_NO_LIVE_EVIDENCE")
        return tuple(failures)

    # -- persistence -------------------------------------------------------- #
    def flush(self) -> bool:
        if self._path is None:
            return False
        with self._lock:
            payload = {
                "version": 1,
                "written_at": datetime.now(timezone.utc).isoformat(),
                "states": {key: str(value) for key, value in self._states.items()},
                "records": {key: value.as_dict() for key, value in self._records.items()},
                "history": [item.as_dict() for item in self._history[-500:]],
            }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temporary.replace(self._path)
            return True
        except OSError:
            return False

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for key, value in (payload.get("states") or {}).items():
            try:
                self._states[_norm(key)] = StrategyLifecycleState(str(value))
            except ValueError:
                continue
        for key, value in (payload.get("records") or {}).items():
            if isinstance(value, Mapping):
                self._records[_norm(key)] = StrategyValidationRecord.from_dict(value)


def _is_promotion(
    current: StrategyLifecycleState, target: StrategyLifecycleState
) -> bool:
    """Is ``target`` further up the trading ladder than ``current``?

    ``DEGRADED`` is not on the ladder, so moving INTO it is always a demotion and moving OUT
    of it is always a promotion — which is why it is handled by membership rather than by an
    index comparison.
    """
    if target is StrategyLifecycleState.RETIRED:
        return False
    if target is StrategyLifecycleState.DEGRADED:
        return False
    if current is StrategyLifecycleState.DEGRADED:
        return True
    try:
        return _LADDER_ORDER.index(target) > _LADDER_ORDER.index(current)
    except ValueError:
        return False


def _norm(strategy_id: str) -> str:
    return str(strategy_id or "").strip().lower()


def _optional(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return _aware(parsed)

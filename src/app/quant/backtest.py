from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


class ActionType(str, Enum):
    PROPOSE_ENTRY = "ProposeEntry"
    PROPOSE_EXIT = "ProposeExit"
    SCALE_POSITION = "ScalePosition"
    REBALANCE = "Rebalance"
    HEDGE_WHEN_SUPPORTED = "HedgeWhenSupported"


@dataclass(frozen=True)
class QuantTradeIntent:
    """An order-free proposal; authoritative gates must convert it downstream."""

    symbol: str
    market: str
    action: ActionType
    generated_at: datetime
    strategy_id: str
    requested_weight: float
    evidence_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    requires_risk_manager: bool = True
    requires_final_trade_gate: bool = True

    def __post_init__(self) -> None:
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("intent timestamp must be timezone-aware")
        if not 0.0 <= self.requested_weight <= 1.0:
            raise ValueError("requested_weight must be in [0, 1]")


class Trigger(Protocol):
    def fires(self, context: Mapping[str, Any], *, as_of: datetime) -> bool: ...


@dataclass(frozen=True)
class TimeTrigger:
    instants: tuple[datetime, ...]

    def fires(self, context: Mapping[str, Any], *, as_of: datetime) -> bool:
        return any(instant == as_of for instant in self.instants)


@dataclass(frozen=True)
class MarketTrigger:
    field: str
    operator: str
    threshold: float

    def fires(self, context: Mapping[str, Any], *, as_of: datetime) -> bool:
        value = context.get(self.field)
        if value is None:
            return False
        number = float(value)
        return {">": number > self.threshold, ">=": number >= self.threshold, "<": number < self.threshold, "<=": number <= self.threshold}.get(self.operator, False)


@dataclass(frozen=True)
class AggregateTrigger:
    triggers: tuple[Trigger, ...]
    mode: str = "all"

    def fires(self, context: Mapping[str, Any], *, as_of: datetime) -> bool:
        decisions = tuple(trigger.fires(context, as_of=as_of) for trigger in self.triggers)
        return all(decisions) if self.mode == "all" else any(decisions)


@dataclass(frozen=True)
class IntentAction:
    action: ActionType
    requested_weight: float

    def propose(self, *, symbol: str, market: str, strategy_id: str, as_of: datetime, evidence_ids: Sequence[str] = ()) -> QuantTradeIntent:
        return QuantTradeIntent(symbol, market, self.action, as_of, strategy_id, self.requested_weight, tuple(evidence_ids))


@dataclass(frozen=True)
class QuantStrategy:
    strategy_id: str
    trigger: Trigger
    action: IntentAction
    initial_state: Mapping[str, Any] = field(default_factory=dict)
    risk_constraints: Mapping[str, Any] = field(default_factory=dict)

    def evaluate(self, context: Mapping[str, Any], *, symbol: str, market: str, as_of: datetime, evidence_ids: Sequence[str] = ()) -> QuantTradeIntent | None:
        if not self.trigger.fires(context, as_of=as_of):
            return None
        return self.action.propose(symbol=symbol, market=market, strategy_id=self.strategy_id, as_of=as_of, evidence_ids=evidence_ids)

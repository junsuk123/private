from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import StrEnum
from math import isclose
from typing import Any


SCHEMA_VERSION = "1.0.0"


def _require_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _require_non_empty(value: str, field_name: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{field_name} must not be empty")


@dataclass(frozen=True)
class EventMetadata:
    event_id: str
    source: str
    symbol: str
    venue: str
    event_time: datetime
    receive_time: datetime
    correlation_id: str
    sequence: int | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("event_id", "source", "symbol", "venue", "correlation_id", "schema_version"):
            _require_non_empty(str(getattr(self, name)), name)
        _require_aware(self.event_time, "event_time")
        _require_aware(self.receive_time, "receive_time")
        if self.sequence is not None and self.sequence < 0:
            raise ValueError("sequence must be non-negative")


@dataclass(frozen=True)
class OrderBookLevel:
    bid_price: float
    bid_quantity: float
    ask_price: float
    ask_quantity: float

    def __post_init__(self) -> None:
        if min(self.bid_price, self.ask_price) <= 0:
            raise ValueError("book prices must be positive")
        if min(self.bid_quantity, self.ask_quantity) < 0:
            raise ValueError("book quantities must be non-negative")
        if self.ask_price < self.bid_price:
            raise ValueError("crossed order book is invalid")


@dataclass(frozen=True)
class TradeTick:
    metadata: EventMetadata
    price: float
    quantity: float
    aggressor_side: str | None = None
    classification_confidence: float | None = None

    def __post_init__(self) -> None:
        if self.price <= 0 or self.quantity < 0:
            raise ValueError("trade price must be positive and quantity non-negative")
        if self.classification_confidence is not None and not 0 <= self.classification_confidence <= 1:
            raise ValueError("classification_confidence must be in [0, 1]")


@dataclass(frozen=True)
class OrderBookSnapshotOrDelta:
    metadata: EventMetadata
    levels: tuple[OrderBookLevel, ...]
    is_snapshot: bool = True

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("order book requires at least one level")


@dataclass(frozen=True)
class SessionStatus:
    metadata: EventMetadata
    phase: str
    tradable: bool


@dataclass(frozen=True)
class Bar:
    symbol: str
    venue: str
    interval: str
    start_time: datetime
    end_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_aware(self.start_time, "start_time")
        _require_aware(self.end_time, "end_time")
        if self.end_time <= self.start_time:
            raise ValueError("bar end_time must be after start_time")
        if min(self.open, self.high, self.low, self.close) <= 0 or self.volume < 0:
            raise ValueError("bar prices must be positive and volume non-negative")
        if self.high < max(self.open, self.low, self.close) or self.low > min(
            self.open, self.high, self.close
        ):
            raise ValueError("bar OHLC values are inconsistent")


@dataclass(frozen=True)
class FeatureSnapshot:
    snapshot_id: str
    as_of: datetime
    symbol: str
    venue: str
    values: dict[str, float | int | bool | None]
    feature_schema_version: str
    normalization_version: str
    source_event_ids: tuple[str, ...]
    fresh: bool

    def __post_init__(self) -> None:
        _require_aware(self.as_of, "as_of")


@dataclass(frozen=True)
class OntologyDecision:
    snapshot_id: str
    as_of: datetime
    symbol: str
    allowed_strategy_ids: tuple[str, ...]
    blocked_strategy_reasons: dict[str, tuple[str, ...]]
    compatibility_scores: dict[str, float]
    explanation_paths: dict[str, tuple[str, ...]]
    valid_until: datetime

    def __post_init__(self) -> None:
        _require_aware(self.as_of, "as_of")
        _require_aware(self.valid_until, "valid_until")
        if self.valid_until < self.as_of:
            raise ValueError("ontology decision cannot expire before as_of")
        if any(not 0 <= value <= 1 for value in self.compatibility_scores.values()):
            raise ValueError("compatibility scores must be in [0, 1]")


@dataclass(frozen=True)
class StrategyUtilityEvidence:
    evidence_id: str
    as_of: datetime
    symbol: str
    strategy_id: str
    ontology_allowed: bool
    hard_block_reasons: tuple[str, ...]
    compatibility_score: float
    probability_success: float
    expected_gross_return_bps: float
    expected_cost_bps: float
    expected_net_return_bps: float
    expected_adverse_excursion_bps: float
    expected_favorable_excursion_bps: float
    fill_probability: float
    expected_holding_seconds: float
    aleatoric_uncertainty: float
    epistemic_uncertainty_or_proxy: float
    utility: float
    model_version: str
    feature_snapshot_id: str
    ontology_snapshot_id: str
    explanation_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_aware(self.as_of, "as_of")
        for name in ("compatibility_score", "probability_success", "fill_probability"):
            if not 0 <= float(getattr(self, name)) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        expected = self.expected_gross_return_bps - self.expected_cost_bps
        if not isclose(self.expected_net_return_bps, expected, abs_tol=1e-6):
            raise ValueError("expected_net_return_bps must equal gross return minus costs")
        if not self.ontology_allowed and not self.hard_block_reasons:
            raise ValueError("ontology-blocked evidence requires a hard-block reason")


@dataclass(frozen=True)
class TradePlan:
    strategy_id: str
    strategy_instance_id: str
    symbol: str
    side: str
    thesis: str
    entry_trigger: dict[str, Any]
    entry_price_policy: dict[str, Any]
    proposed_quantity: int
    initial_stop: dict[str, Any]
    profit_policy: dict[str, Any]
    trailing_policy: dict[str, Any]
    max_holding_seconds: int
    invalidation_conditions: tuple[str, ...]
    max_entry_slippage_bps: float
    expires_at: datetime
    feature_snapshot_id: str
    utility_evidence_id: str
    # ``side`` is retained as the broker-facing BUY/SELL, but it is no longer the
    # plan's meaning: direction and effect are. Kept rather than replaced so
    # existing readers of ``side`` keep working, and validated below so the two
    # can never disagree — a plan whose ``side`` contradicts its
    # (direction, effect) pair is the ambiguity this split exists to remove.
    position_direction: str = "LONG"
    position_effect: str = "OPEN"
    execution_product: str = "CASH"
    # Deployment state of the owning arm at the moment the plan was formed. A plan
    # carrying SHADOW must never reach an order submitter; the submitter re-reads
    # the authoritative state anyway, so this is an audit record of what the
    # planner believed, not the permission itself.
    deployment_state: str = "SHADOW"

    def __post_init__(self) -> None:
        _require_aware(self.expires_at, "expires_at")
        if self.proposed_quantity <= 0 or self.max_holding_seconds <= 0:
            raise ValueError("trade plan quantity and holding period must be positive")
        from app.trading.directional import broker_side, parse_direction, parse_effect

        expected = broker_side(
            parse_direction(self.position_direction), parse_effect(self.position_effect)
        )
        actual = str(self.side or "").strip().upper()
        if actual and actual != expected:
            raise ValueError(
                f"trade plan side {actual} contradicts "
                f"{self.position_direction}/{self.position_effect} (expected {expected})"
            )


class IntentAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    AMEND = "AMEND"
    CANCEL = "CANCEL"


@dataclass(frozen=True)
class OrderIntent:
    intent_id: str
    idempotency_key: str
    strategy_instance_id: str
    position_id_if_any: str | None
    symbol: str
    action: IntentAction
    quantity: int
    limit_or_price_policy: dict[str, Any]
    urgency: str
    reason_code: str
    created_at: datetime
    expires_at: datetime
    parent_intent_id_if_replacement: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "intent_id",
            "idempotency_key",
            "strategy_instance_id",
            "symbol",
            "urgency",
            "reason_code",
        ):
            _require_non_empty(str(getattr(self, name)), name)
        _require_aware(self.created_at, "created_at")
        _require_aware(self.expires_at, "expires_at")
        if self.expires_at <= self.created_at:
            raise ValueError("intent must expire after creation")
        if self.quantity <= 0:
            raise ValueError("intent quantity must be positive")


class RiskVerdictAction(StrEnum):
    APPROVE = "APPROVE"
    RESIZE = "RESIZE"
    REJECT = "REJECT"
    EMERGENCY_EXIT = "EMERGENCY_EXIT"


@dataclass(frozen=True)
class RiskVerdict:
    verdict_id: str
    intent_id: str
    action: RiskVerdictAction
    approved_quantity: int
    limits_evaluated: dict[str, bool | float | int | str]
    reason_codes: tuple[str, ...]
    account_snapshot_id: str
    timestamp: datetime

    def __post_init__(self) -> None:
        _require_aware(self.timestamp, "timestamp")
        if self.approved_quantity < 0:
            raise ValueError("approved quantity must be non-negative")
        if self.action == RiskVerdictAction.REJECT and self.approved_quantity != 0:
            raise ValueError("rejected intent must approve zero quantity")


@dataclass(frozen=True)
class BrokerOrder:
    broker_order_id: str
    intent_id: str
    idempotency_key: str
    symbol: str
    side: str
    quantity: int
    submitted_at: datetime
    status: str
    position_direction: str = "LONG"
    position_effect: str = "OPEN"
    execution_product: str = "CASH"
    # What the BROKER said the credit classification was. Reconciliation compares
    # this against the internal contract; a mismatch means the order we believe we
    # placed is not the order that exists, which suspends the strategy.
    broker_credit_type: str | None = None
    loan_date: str | None = None


@dataclass(frozen=True)
class OrderUpdate:
    broker_order_id: str
    status: str
    filled_quantity: int
    remaining_quantity: int
    event_time: datetime
    broker_sequence: str | None = None


@dataclass(frozen=True)
class Fill:
    fill_id: str
    broker_order_id: str
    intent_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    event_time: datetime
    fees: float = 0.0
    taxes: float = 0.0
    position_direction: str = "LONG"
    position_effect: str = "OPEN"
    execution_product: str = "CASH"
    loan_date: str | None = None
    # Borrow fee accrued on this leg, in currency. Separate from ``fees`` because
    # it is time-dependent rather than per-execution, and the promotion evaluator
    # has to attribute it to the holding period.
    borrow_fee: float = 0.0


@dataclass(frozen=True)
class Position:
    position_id: str
    symbol: str
    quantity: int
    average_price: float
    origin_strategy_id: str
    strategy_instance_id: str
    opened_at: datetime
    realized_pnl: float = 0.0
    direction: str = "LONG"
    execution_product: str = "CASH"
    # One lot per (symbol, loan_date). The broker treats borrows opened on
    # different dates as distinct positions with distinct repayment obligations, so
    # netting them internally would produce a buy-to-cover the broker rejects.
    loan_date: str | None = None
    borrow_fee_rate: float | None = None
    return_deadline: datetime | None = None

    @property
    def is_short(self) -> bool:
        return str(self.direction or "LONG").upper() == "SHORT"

    @property
    def lot_key(self) -> tuple[str, str, str]:
        """Identity used for reconciliation against broker state."""
        return (self.symbol, str(self.direction or "LONG").upper(), self.loan_date or "")


class StrategyLifecycleStatus(StrEnum):
    PLANNED = "PLANNED"
    ARMED = "ARMED"
    ENTRY_PENDING = "ENTRY_PENDING"
    OPEN = "OPEN"
    EXIT_PENDING = "EXIT_PENDING"
    CLOSED = "CLOSED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class StrategyInstanceState:
    strategy_instance_id: str
    strategy_id: str
    symbol: str
    status: StrategyLifecycleStatus
    created_at: datetime
    updated_at: datetime
    position_id: str | None = None
    state_version: int = 1


@dataclass(frozen=True)
class AccountSnapshot:
    snapshot_id: str
    captured_at: datetime
    cash_by_currency: dict[str, float]
    positions: tuple[Position, ...]
    broker_authoritative: bool = True


_LATENCY_STAGES = (
    "feed_receive_monotonic_ns",
    "normalized_monotonic_ns",
    "feature_ready_monotonic_ns",
    "ontology_snapshot_monotonic_ns",
    "model_start_monotonic_ns",
    "model_end_monotonic_ns",
    "strategy_decision_monotonic_ns",
    "risk_verdict_monotonic_ns",
    "order_submit_monotonic_ns",
    "broker_ack_monotonic_ns",
    "first_fill_monotonic_ns",
    "final_fill_monotonic_ns",
)


@dataclass(frozen=True)
class LatencyTrace:
    trace_id: str
    correlation_id: str
    symbol: str
    exchange_timestamp: datetime
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    feed_receive_monotonic_ns: int | None = None
    normalized_monotonic_ns: int | None = None
    feature_ready_monotonic_ns: int | None = None
    ontology_snapshot_monotonic_ns: int | None = None
    model_start_monotonic_ns: int | None = None
    model_end_monotonic_ns: int | None = None
    strategy_decision_monotonic_ns: int | None = None
    risk_verdict_monotonic_ns: int | None = None
    order_submit_monotonic_ns: int | None = None
    broker_ack_monotonic_ns: int | None = None
    first_fill_monotonic_ns: int | None = None
    final_fill_monotonic_ns: int | None = None

    def __post_init__(self) -> None:
        _require_aware(self.exchange_timestamp, "exchange_timestamp")
        _require_aware(self.created_at, "created_at")
        observed = [getattr(self, stage) for stage in _LATENCY_STAGES]
        present = [value for value in observed if value is not None]
        if any(value < 0 for value in present):
            raise ValueError("monotonic timestamps must be non-negative")
        if present != sorted(present):
            raise ValueError("latency stages must be monotonic")

    def mark(self, stage: str, monotonic_ns: int) -> "LatencyTrace":
        if stage not in _LATENCY_STAGES:
            raise ValueError(f"unknown latency stage: {stage}")
        if getattr(self, stage) is not None:
            raise ValueError(f"latency stage already recorded: {stage}")
        return replace(self, **{stage: monotonic_ns})

    def duration_ns(self, start_stage: str, end_stage: str) -> int | None:
        start = getattr(self, start_stage)
        end = getattr(self, end_stage)
        if start is None or end is None:
            return None
        return end - start

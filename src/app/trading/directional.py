"""Direction as a first-class dimension of a strategy, a position, and an order.

Why this module exists
----------------------
Before it, "direction" was implicit everywhere: a ``BUY`` meant *open a long*, a
``SELL`` meant *close a long*, and every profit/stop/trailing calculation assumed
price going up was good. Adding short selling by reusing ``SELL`` would have made
"close the long I hold" and "open a new short" the same instruction — the single
most dangerous ambiguity available in an order path, because the two differ in
whether the account ends up flat or ends up owing shares.

So the four axes are separated and named:

* :class:`PositionDirection` — which way the exposure points (LONG / SHORT).
* :class:`PositionEffect` — whether this order OPENs or CLOSEs that exposure.
* :class:`ExecutionProduct` — how the broker settles it (CASH / CREDIT_BORROW).
* :class:`StrategyDeploymentState` — how much real money this *specific*
  (strategy, direction, market, product) tuple is allowed to touch.

Quantities are always positive magnitudes. Direction is never encoded as a sign
on quantity: a negative quantity survives one refactor and then silently becomes
a buy somewhere.

Deployment state is deliberately NOT a boolean and NOT global. The system already
has a model-level ``live_authorized``; that answers "is the model calibrated",
which is a different question from "has THIS short strategy demonstrated a
positive net edge in forward, out-of-sample, borrow-aware evaluation". A model
can be well calibrated and a strategy still unprofitable after borrow fees.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable


# --------------------------------------------------------------------------- #
# Axes                                                                         #
# --------------------------------------------------------------------------- #
class PositionDirection(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        """+1 for LONG, -1 for SHORT. The only place this sign is defined."""
        return 1 if self is PositionDirection.LONG else -1

    @property
    def opposite(self) -> "PositionDirection":
        return (
            PositionDirection.SHORT
            if self is PositionDirection.LONG
            else PositionDirection.LONG
        )


class PositionEffect(StrEnum):
    OPEN = "OPEN"
    CLOSE = "CLOSE"


class ExecutionProduct(StrEnum):
    CASH = "CASH"
    CREDIT_BORROW = "CREDIT_BORROW"


class StrategyDeploymentState(StrEnum):
    """How much real money a directional strategy may touch, in ascending order.

    ``SUSPENDED`` is deliberately outside the ordering: it is a *fault* state, not
    a rung on the ladder, and it is only ever left via ``SHADOW``.
    """

    DISABLED = "DISABLED"
    SHADOW = "SHADOW"
    LIVE_PROBE = "LIVE_PROBE"
    LIVE_LIMITED = "LIVE_LIMITED"
    LIVE_FULL = "LIVE_FULL"
    SUSPENDED = "SUSPENDED"

    @property
    def submits_orders(self) -> bool:
        return self in _ORDER_SUBMITTING_STATES

    @property
    def rank(self) -> int:
        """Ladder position. ``SUSPENDED`` sorts below ``DISABLED`` (worse)."""
        return _STATE_RANK[self]


_ORDER_SUBMITTING_STATES: frozenset[StrategyDeploymentState] = frozenset(
    {
        StrategyDeploymentState.LIVE_PROBE,
        StrategyDeploymentState.LIVE_LIMITED,
        StrategyDeploymentState.LIVE_FULL,
    }
)

_STATE_RANK: dict[StrategyDeploymentState, int] = {
    StrategyDeploymentState.SUSPENDED: -1,
    StrategyDeploymentState.DISABLED: 0,
    StrategyDeploymentState.SHADOW: 1,
    StrategyDeploymentState.LIVE_PROBE: 2,
    StrategyDeploymentState.LIVE_LIMITED: 3,
    StrategyDeploymentState.LIVE_FULL: 4,
}


# The transition graph is a whitelist, not a rank comparison. A rank comparison
# would silently permit SHADOW -> LIVE_FULL as "an increase of 3", which is the
# single transition this whole subsystem exists to forbid: it would put an
# unvalidated short strategy straight onto full size.
ALLOWED_TRANSITIONS: frozenset[tuple[StrategyDeploymentState, StrategyDeploymentState]] = (
    frozenset(
        {
            (StrategyDeploymentState.DISABLED, StrategyDeploymentState.SHADOW),
            (StrategyDeploymentState.SHADOW, StrategyDeploymentState.LIVE_PROBE),
            (StrategyDeploymentState.LIVE_PROBE, StrategyDeploymentState.LIVE_LIMITED),
            (StrategyDeploymentState.LIVE_LIMITED, StrategyDeploymentState.LIVE_FULL),
            # Demotions.
            (StrategyDeploymentState.LIVE_FULL, StrategyDeploymentState.LIVE_LIMITED),
            (StrategyDeploymentState.LIVE_LIMITED, StrategyDeploymentState.LIVE_PROBE),
            (StrategyDeploymentState.LIVE_PROBE, StrategyDeploymentState.SHADOW),
            (StrategyDeploymentState.SHADOW, StrategyDeploymentState.DISABLED),
            # Recovery is only ever back to SHADOW; never directly to a live state.
            (StrategyDeploymentState.SUSPENDED, StrategyDeploymentState.SHADOW),
        }
        # ANY -> SUSPENDED.
        | {
            (state, StrategyDeploymentState.SUSPENDED)
            for state in StrategyDeploymentState
            if state is not StrategyDeploymentState.SUSPENDED
        }
    )
)


def transition_allowed(
    current: StrategyDeploymentState, target: StrategyDeploymentState
) -> bool:
    """Is ``current -> target`` a permitted deployment-state move?

    A no-op (``current == target``) is allowed so callers can re-assert a state
    idempotently without special-casing.
    """
    if current is target:
        return True
    return (current, target) in ALLOWED_TRANSITIONS


def next_promotion_state(
    current: StrategyDeploymentState,
) -> StrategyDeploymentState | None:
    """The single state a successful promotion may move to, or ``None``."""
    return {
        StrategyDeploymentState.DISABLED: StrategyDeploymentState.SHADOW,
        StrategyDeploymentState.SHADOW: StrategyDeploymentState.LIVE_PROBE,
        StrategyDeploymentState.LIVE_PROBE: StrategyDeploymentState.LIVE_LIMITED,
        StrategyDeploymentState.LIVE_LIMITED: StrategyDeploymentState.LIVE_FULL,
    }.get(current)


def previous_safer_state(
    current: StrategyDeploymentState,
) -> StrategyDeploymentState | None:
    """The single state a demotion may move to, or ``None`` at the bottom."""
    return {
        StrategyDeploymentState.LIVE_FULL: StrategyDeploymentState.LIVE_LIMITED,
        StrategyDeploymentState.LIVE_LIMITED: StrategyDeploymentState.LIVE_PROBE,
        StrategyDeploymentState.LIVE_PROBE: StrategyDeploymentState.SHADOW,
    }.get(current)


def parse_direction(value: Any, default: PositionDirection = PositionDirection.LONG) -> PositionDirection:
    try:
        return PositionDirection(str(value or "").strip().upper())
    except ValueError:
        return default


def parse_effect(value: Any, default: PositionEffect = PositionEffect.OPEN) -> PositionEffect:
    try:
        return PositionEffect(str(value or "").strip().upper())
    except ValueError:
        return default


def parse_product(
    value: Any, default: ExecutionProduct = ExecutionProduct.CASH
) -> ExecutionProduct:
    try:
        return ExecutionProduct(str(value or "").strip().upper())
    except ValueError:
        return default


def parse_state(
    value: Any, default: StrategyDeploymentState = StrategyDeploymentState.SHADOW
) -> StrategyDeploymentState:
    """Unparseable state resolves to the supplied default (``SHADOW``).

    Never to a live state: an unreadable persisted value must not authorise
    orders.
    """
    try:
        return StrategyDeploymentState(str(value or "").strip().upper())
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
# Broker-side order semantics                                                  #
# --------------------------------------------------------------------------- #
def broker_side(direction: PositionDirection, effect: PositionEffect) -> str:
    """The BUY/SELL the broker sees for a (direction, effect) pair.

    The four rows of the semantics table, in one place:

    ==============  ==========  ======  ===============
    meaning         direction   effect  broker side
    ==============  ==========  ======  ===============
    open long       LONG        OPEN    BUY
    close long      LONG        CLOSE   SELL
    open short      SHORT       OPEN    SELL
    close short     SHORT       CLOSE   BUY
    ==============  ==========  ======  ===============
    """
    if direction is PositionDirection.LONG:
        return "BUY" if effect is PositionEffect.OPEN else "SELL"
    return "SELL" if effect is PositionEffect.OPEN else "BUY"


def default_product(direction: PositionDirection) -> ExecutionProduct:
    """A short needs borrowed stock; a long is settled in cash."""
    return (
        ExecutionProduct.CREDIT_BORROW
        if direction is PositionDirection.SHORT
        else ExecutionProduct.CASH
    )


# --------------------------------------------------------------------------- #
# Directional exit geometry                                                    #
# --------------------------------------------------------------------------- #
def target_price(
    entry_price: float, target_rate: float, direction: PositionDirection
) -> float:
    """Profit target. A short profits as price FALLS, so the sign flips."""
    return entry_price * (1.0 + direction.sign * abs(target_rate))


def stop_price(
    entry_price: float, stop_rate: float, direction: PositionDirection
) -> float:
    """Stop loss. A short loses as price RISES."""
    return entry_price * (1.0 - direction.sign * abs(stop_rate))


def trailing_price(
    watermark: float, trailing_rate: float, direction: PositionDirection
) -> float:
    """Trailing stop off the favourable extreme.

    ``watermark`` is the HIGH watermark for a long and the LOW watermark for a
    short — in both cases the best price the position has seen.
    """
    return watermark * (1.0 - direction.sign * abs(trailing_rate))


def favourable_watermark(
    previous: float | None, price: float, direction: PositionDirection
) -> float:
    """Update the favourable extreme: max for LONG, min for SHORT."""
    if previous is None or previous <= 0:
        return price
    return (
        max(previous, price)
        if direction is PositionDirection.LONG
        else min(previous, price)
    )


def gross_return_bps(
    entry_price: float, exit_price: float, direction: PositionDirection
) -> float:
    """``direction_sign * (exit/entry - 1) * 10000``.

    The single definition of directional PnL. A short that entered at 100 and
    covered at 95 made +500bps gross; the sign is applied here and nowhere else.
    """
    if entry_price <= 0 or exit_price <= 0:
        return 0.0
    return direction.sign * (exit_price / entry_price - 1.0) * 10_000.0


def target_reached(
    last_price: float, target: float | None, direction: PositionDirection
) -> bool:
    if not target or target <= 0 or last_price <= 0:
        return False
    return (
        last_price >= target
        if direction is PositionDirection.LONG
        else last_price <= target
    )


def stop_breached(
    last_price: float, stop: float | None, direction: PositionDirection
) -> bool:
    if not stop or stop <= 0 or last_price <= 0:
        return False
    return (
        last_price <= stop
        if direction is PositionDirection.LONG
        else last_price >= stop
    )


def trailing_breached(
    last_price: float,
    trailing: float,
    entry_price: float,
    direction: PositionDirection,
) -> bool:
    """A trailing stop only binds once it has moved past break-even.

    Mirrors the long-side rule (``trailing_price > average_price``) rather than
    re-deriving it, so a short's trailing stop likewise cannot fire while it
    still sits on the losing side of entry.
    """
    if trailing <= 0 or last_price <= 0 or entry_price <= 0:
        return False
    if direction is PositionDirection.LONG:
        return trailing > entry_price and last_price <= trailing
    return trailing < entry_price and last_price >= trailing


# --------------------------------------------------------------------------- #
# The arm / storage key                                                        #
# --------------------------------------------------------------------------- #
_KEY_SEPARATOR = ":"


@dataclass(frozen=True)
class DirectionalStrategyKey:
    """Identity of one tradable arm: strategy x direction x market x product.

    This replaces the bare ``strategy_id`` as the key for posteriors, realized
    outcomes and deployment state. Sharing a key between LONG and SHORT would
    pool their realized returns into one posterior, and a strategy that makes
    60bps long and loses 60bps short would read as break-even and therefore
    permanently untradable in both directions — while a genuinely one-sided edge
    would be diluted into invisibility.
    """

    strategy_id: str
    direction: PositionDirection = PositionDirection.LONG
    market: str = "KR"
    execution_product: ExecutionProduct = ExecutionProduct.CASH

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy_id", str(self.strategy_id or "").strip().lower())
        object.__setattr__(self, "market", str(self.market or "").strip().upper() or "UNKNOWN")

    @property
    def is_short(self) -> bool:
        return self.direction is PositionDirection.SHORT

    def as_text(self) -> str:
        """``strategy_id:DIRECTION:MARKET:PRODUCT`` — the persisted form."""
        return _KEY_SEPARATOR.join(
            (
                self.strategy_id,
                str(self.direction),
                self.market,
                str(self.execution_product),
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_key": self.as_text(),
            "strategy_id": self.strategy_id,
            "direction": str(self.direction),
            "market": self.market,
            "execution_product": str(self.execution_product),
        }

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.as_text()

    @classmethod
    def parse(cls, text: str) -> "DirectionalStrategyKey":
        """Parse the persisted form. Raises on a malformed key.

        Deliberately strict: a key that cannot be parsed must not silently become
        ``(that_text, LONG, KR, CASH)``, because that would attach a short
        strategy's realized history to a long arm.
        """
        parts = str(text or "").split(_KEY_SEPARATOR)
        if len(parts) != 4:
            raise ValueError(f"malformed directional strategy key: {text!r}")
        strategy_id, direction, market, product = parts
        if not strategy_id.strip():
            raise ValueError(f"directional strategy key has no strategy id: {text!r}")
        return cls(
            strategy_id=strategy_id,
            direction=PositionDirection(direction.strip().upper()),
            market=market,
            execution_product=ExecutionProduct(product.strip().upper()),
        )

    @classmethod
    def for_long(cls, strategy_id: str, market: str = "KR") -> "DirectionalStrategyKey":
        return cls(
            strategy_id=strategy_id,
            direction=PositionDirection.LONG,
            market=market,
            execution_product=ExecutionProduct.CASH,
        )

    @classmethod
    def for_short(cls, strategy_id: str, market: str = "KR") -> "DirectionalStrategyKey":
        return cls(
            strategy_id=strategy_id,
            direction=PositionDirection.SHORT,
            market=market,
            execution_product=ExecutionProduct.CREDIT_BORROW,
        )


def parse_key_or_none(text: str) -> DirectionalStrategyKey | None:
    try:
        return DirectionalStrategyKey.parse(text)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Reason codes                                                                 #
# --------------------------------------------------------------------------- #
#: An arm was elected but its deployment state does not submit orders. Direction
#: agnostic: the ladder gates LONG arms too, and reporting a long demotion as
#: ``SHORT_STRATEGY_SHADOW_ONLY`` sent operators looking for a borrow problem that
#: was not there.
DEPLOYMENT_SHADOW_ONLY = "STRATEGY_DEPLOYMENT_SHADOW_ONLY"


class ShortReasonCodes:
    """Every reason a short signal can fail to become an order, or lose rank.

    Grouped by the blockade stage they belong to so the dashboard can show the
    FIRST stage that stopped an entry rather than the last complaint emitted.
    """

    # deployment authorization
    SHADOW_ONLY = "SHORT_STRATEGY_SHADOW_ONLY"
    DEPLOYMENT_SUSPENDED = "SHORT_DEPLOYMENT_SUSPENDED"
    DEPLOYMENT_DISABLED = "SHORT_DEPLOYMENT_DISABLED"
    DEPLOYMENT_STATE_UNREADABLE = "SHORT_DEPLOYMENT_STATE_UNREADABLE"
    LIVE_PROBE_LIMIT_REACHED = "SHORT_LIVE_PROBE_LIMIT_REACHED"
    # shadow validation / promotion
    PROMOTION_SAMPLE_INSUFFICIENT = "SHORT_PROMOTION_SAMPLE_INSUFFICIENT"
    CONFIDENCE_BELOW_THRESHOLD = "SHORT_CONFIDENCE_BELOW_THRESHOLD"
    CONSERVATIVE_EDGE_NON_POSITIVE = "SHORT_CONSERVATIVE_EDGE_NON_POSITIVE"
    COST_COVERAGE_INSUFFICIENT = "SHORT_COST_COVERAGE_INSUFFICIENT"
    HOLDOUT_NOT_PASSED = "SHORT_PROMOTION_HOLDOUT_NOT_PASSED"
    CONSECUTIVE_CYCLES_PENDING = "SHORT_PROMOTION_CONSECUTIVE_CYCLES_PENDING"
    DRAWDOWN_EXCEEDED = "SHORT_DRAWDOWN_EXCEEDED"
    LOSS_STREAK_EXCEEDED = "SHORT_LOSS_STREAK_EXCEEDED"
    CALIBRATION_ERROR_HIGH = "SHORT_CALIBRATION_ERROR_HIGH"
    SLIPPAGE_ERROR_HIGH = "SHORT_SLIPPAGE_ERROR_HIGH"
    RESCUE_RATE_INSUFFICIENT = "SHORT_RESCUE_RATE_INSUFFICIENT"
    BROKER_REJECTION_RATE_HIGH = "SHORT_BROKER_REJECTION_RATE_HIGH"
    # borrow
    BORROW_UNAVAILABLE = "SHORT_BORROW_UNAVAILABLE"
    BORROW_QUANTITY_INSUFFICIENT = "SHORT_BORROW_QUANTITY_INSUFFICIENT"
    BORROW_COST_TOO_HIGH = "SHORT_BORROW_COST_TOO_HIGH"
    BORROW_SNAPSHOT_STALE = "SHORT_BORROW_SNAPSHOT_STALE"
    BORROW_LOOKUP_FAILED = "SHORT_BORROW_LOOKUP_FAILED"
    BORROW_AVAILABILITY_RATE_LOW = "SHORT_BORROW_AVAILABILITY_RATE_LOW"
    RECALL_DEADLINE_NEAR = "SHORT_RECALL_DEADLINE_NEAR"
    SHORT_SALE_NOT_PERMITTED = "SHORT_SALE_NOT_PERMITTED"
    # order contract integrity
    LOAN_DATE_MISSING = "SHORT_LOAN_DATE_MISSING"
    ORDER_CONTRACT_INCOMPLETE = "SHORT_ORDER_CONTRACT_INCOMPLETE"
    POSITION_DIRECTION_MISMATCH = "SHORT_POSITION_DIRECTION_MISMATCH"
    BROKER_STATE_UNRESTORED = "SHORT_BROKER_STATE_UNRESTORED"
    STOP_ORDER_CAPABILITY_MISSING = "SHORT_STOP_ORDER_CAPABILITY_MISSING"
    # model / data / regime
    MODEL_NOT_CALIBRATED = "SHORT_MODEL_NOT_CALIBRATED"
    DATA_QUALITY_FAILED = "SHORT_DATA_QUALITY_FAILED"
    REGIME_UNSTABLE = "SHORT_REGIME_UNSTABLE"
    HIGH_VOL_DISLOCATED = "SHORT_HIGH_VOL_DISLOCATED"
    DAILY_LOSS_LIMIT = "SHORT_DAILY_LOSS_LIMIT_EXCEEDED"
    # selection outcome
    BOTH_DIRECTIONS_NEGATIVE = "BOTH_DIRECTIONS_NEGATIVE"


# The blockade stages, in the order an entry must clear them. The dashboard
# reports the FIRST failing stage, which is the actionable one.
ENTRY_BLOCKADE_STAGES: tuple[str, ...] = (
    "directional_candidates",
    "short_signal",
    "shadow_validation",
    "deployment_authorization",
    "borrow_preflight",
    "profitability",
    "short_risk",
    "credit_order_contract",
    "broker_execution",
)


def first_blocking_stage(failed: Iterable[str]) -> str | None:
    """The earliest blockade stage present in ``failed``, or ``None``."""
    failures = {str(item) for item in failed}
    return next((stage for stage in ENTRY_BLOCKADE_STAGES if stage in failures), None)

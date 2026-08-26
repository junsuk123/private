"""A virtual position. It cannot reach a broker, by construction.

Why a separate object rather than reusing ``ShadowTradePlan``
------------------------------------------------------------
``app.trading.directional_shadow.ShadowTradePlan`` exists for the short-promotion ladder
and is persisted per plan. What the counterfactual engine needs is different in one
important way: it opens a position for EVERY eligible, entry-ready strategy in the same
context, so the group has to be identifiable as a group (``context_id``) and resolved
against the same quote stream, and the *unselected* ones must be distinguishable from the
selected one. A plan store keyed by strategy cannot express "these nine are alternatives
to that one".

Three properties are load-bearing and enforced here rather than documented:

* **Frozen at signal time.** Entry reference, barriers and horizon are captured when the
  proposal was made. Nothing is re-derived from a later quote — that re-derivation is the
  leak that would make a shadow result unachievable live.
* **No broker path.** This module imports nothing from ``app.execution``. A shadow outcome
  is arithmetic over quotes.
* **Marked as simulated.** Every outcome carries ``evidence_source`` and
  ``fill_assumed=True``. A shadow fill is an assumption about liquidity, and promotion
  logic must be able to weight it below a real one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping
from uuid import uuid4

from app.cost.cost_coverage import minimum_trailing_net_bps

__all__ = [
    "EVIDENCE_BACKTEST",
    "EVIDENCE_LIVE",
    "EVIDENCE_LIVE_PROBE",
    "EVIDENCE_SHADOW",
    "ShadowExitReason",
    "ShadowOutcome",
    "ShadowPosition",
]

EVIDENCE_LIVE = "LIVE"
EVIDENCE_LIVE_PROBE = "LIVE_PROBE"
EVIDENCE_SHADOW = "SHADOW"
EVIDENCE_BACKTEST = "BACKTEST"


class ShadowExitReason:
    TARGET = "SHADOW_TARGET_HIT"
    STOP = "SHADOW_STOP_HIT"
    TRAILING = "SHADOW_TRAILING_STOP"
    TIME = "SHADOW_TIME_STOP"
    UNRESOLVED = "SHADOW_UNRESOLVED"


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class ShadowOutcome:
    """A resolved virtual trade, in the shape the performance store consumes."""

    position_id: str
    context_id: str
    strategy_id: str
    symbol: str
    market: str
    direction: str
    entry_price: float
    exit_price: float
    opened_at: datetime
    closed_at: datetime
    gross_return_bps: float
    net_return_bps: float
    cost_bps: float
    max_adverse_excursion_bps: float
    max_favorable_excursion_bps: float
    holding_seconds: float
    exit_reason: str
    evidence_source: str = EVIDENCE_SHADOW
    #: Always ``True`` for a shadow outcome: the fill is assumed, not observed.
    fill_assumed: bool = True
    quotes_observed: int = 0
    regime: str = "UNKNOWN"

    @property
    def is_win(self) -> bool:
        return self.net_return_bps > 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "position_id": self.position_id,
            "context_id": self.context_id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "market": self.market,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat(),
            "gross_return_bps": round(self.gross_return_bps, 3),
            "net_return_bps": round(self.net_return_bps, 3),
            "cost_bps": round(self.cost_bps, 3),
            "max_adverse_excursion_bps": round(self.max_adverse_excursion_bps, 3),
            "max_favorable_excursion_bps": round(self.max_favorable_excursion_bps, 3),
            "holding_seconds": round(self.holding_seconds, 1),
            "exit_reason": self.exit_reason,
            "evidence_source": self.evidence_source,
            "fill_assumed": self.fill_assumed,
            "quotes_observed": self.quotes_observed,
            "regime": self.regime,
        }


@dataclass
class ShadowPosition:
    """One virtual position, walked forward one quote at a time.

    ``update`` is intentionally incremental rather than given a price series: the live loop
    hands it whichever quote just arrived, so the walk uses only information available at
    that instant. A batch method over a stored series would make it trivially easy to
    resolve a position from a quote that had not happened yet.
    """

    position_id: str
    context_id: str
    strategy_id: str
    symbol: str
    market: str
    direction: str
    entry_price: float
    target_price: float | None
    stop_price: float | None
    trailing_bps: float | None
    max_holding_seconds: int
    cost_bps: float
    opened_at: datetime
    regime: str = "UNKNOWN"
    #: True for the strategy the live selector chose. Its outcome comes from the REAL
    #: fill, so the engine must not also score it here — that would enter the same trade
    #: twice into the same arm's posterior.
    was_selected: bool = False
    proposal_id: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    # -- running state ------------------------------------------------------ #
    _best_price: float = field(init=False, default=0.0)
    _worst_price: float = field(init=False, default=0.0)
    _last_price: float = field(init=False, default=0.0)
    _last_at: datetime | None = field(init=False, default=None)
    _quotes: int = field(init=False, default=0)
    _outcome: ShadowOutcome | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.opened_at = _aware(self.opened_at)
        self.direction = str(self.direction or "LONG").strip().upper()
        self._best_price = self.entry_price
        self._worst_price = self.entry_price
        self._last_price = self.entry_price

    # -- properties --------------------------------------------------------- #
    @property
    def is_short(self) -> bool:
        return self.direction == "SHORT"

    @property
    def resolved(self) -> bool:
        return self._outcome is not None

    @property
    def outcome(self) -> ShadowOutcome | None:
        return self._outcome

    @property
    def deadline(self) -> datetime:
        return self.opened_at + timedelta(seconds=max(1, int(self.max_holding_seconds)))

    # -- walk --------------------------------------------------------------- #
    def update(self, price: float, at: datetime) -> ShadowOutcome | None:
        """Feed one quote. Returns the outcome on the quote that resolves it.

        Barrier precedence is STOP before TARGET when a single quote is beyond both. That
        is the conservative reading and it matches what the live executor would suffer: a
        quote that gapped through both barriers most likely traded through the near one
        first, and assuming otherwise would systematically flatter every shadow result.
        """
        if self._outcome is not None:
            return None
        at = _aware(at)
        if at < self.opened_at:
            # A quote from before the signal cannot resolve it. Silently ignoring this is
            # what keeps a mis-ordered replay from manufacturing outcomes.
            return None
        price = float(price)
        if not math.isfinite(price) or price <= 0:
            return None

        self._quotes += 1
        self._last_price = price
        self._last_at = at
        favourable = price if not self.is_short else -price
        best = self._best_price if not self.is_short else -self._best_price
        worst = self._worst_price if not self.is_short else -self._worst_price
        if favourable > best:
            self._best_price = price
        if favourable < worst:
            self._worst_price = price

        if self._breached_stop(price):
            return self._close(price, at, ShadowExitReason.STOP)
        trailing = self._trailing_trigger()
        target_gross_bps = (
            self._return_bps(self.target_price)
            if self.target_price is not None
            else 0.0
        )
        trailing_required_gross_bps = self.cost_bps + minimum_trailing_net_bps(
            target_gross_bps,
            self.cost_bps,
        )
        trailing_locks_required_reward = (
            trailing is not None
            and self._return_bps(trailing) >= trailing_required_gross_bps
        )
        if (
            trailing is not None
            and trailing_locks_required_reward
            and self._breached_price(price, trailing, adverse=True)
        ):
            return self._close(trailing, at, ShadowExitReason.TRAILING)
        if self._breached_target(price):
            return self._close(self.target_price or price, at, ShadowExitReason.TARGET)
        if at >= self.deadline:
            return self._close(price, at, ShadowExitReason.TIME)
        return None

    def expire(self, at: datetime) -> ShadowOutcome | None:
        """Force a time-stop close at the last observed price.

        Used when the quote stream ends (market closed, symbol unsubscribed) so a position
        does not sit open forever. Closes at the LAST OBSERVED price, never at a fresh
        read — the position stopped being observable, which is itself information.
        """
        if self._outcome is not None:
            return None
        if self._quotes <= 0:
            # Never observed after the signal: there is no outcome to report, and
            # resolving at the entry price would enter a fabricated break-even trade.
            return None
        return self._close(self._last_price, _aware(at), ShadowExitReason.TIME)

    def mark_unobserved(self, at: datetime) -> ShadowOutcome:
        """Retire a position that never received a quote after its signal.

        Returns a marker outcome with ``quotes_observed=0`` and a zero return, and stores
        it so the position stops being walked. The caller must NOT feed this to a
        posterior: a zero-return row would be a fabricated break-even trade. It exists so
        "we could not observe this" is distinguishable from "this broke even", which are
        different facts about the strategy.
        """
        if self._outcome is not None:
            return self._outcome
        moment = _aware(at)
        self._outcome = ShadowOutcome(
            position_id=self.position_id,
            context_id=self.context_id,
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            market=self.market,
            direction=self.direction,
            entry_price=self.entry_price,
            exit_price=self.entry_price,
            opened_at=self.opened_at,
            closed_at=moment,
            gross_return_bps=0.0,
            net_return_bps=0.0,
            cost_bps=self.cost_bps,
            max_adverse_excursion_bps=0.0,
            max_favorable_excursion_bps=0.0,
            holding_seconds=max(0.0, (moment - self.opened_at).total_seconds()),
            exit_reason=ShadowExitReason.UNRESOLVED,
            evidence_source=EVIDENCE_SHADOW,
            fill_assumed=True,
            quotes_observed=0,
            regime=self.regime,
        )
        return self._outcome

    # -- internals ---------------------------------------------------------- #
    def _breached_stop(self, price: float) -> bool:
        if self.stop_price is None:
            return False
        return price <= self.stop_price if not self.is_short else price >= self.stop_price

    def _breached_target(self, price: float) -> bool:
        if self.target_price is None:
            return False
        return price >= self.target_price if not self.is_short else price <= self.target_price

    def _breached_price(self, price: float, level: float, *, adverse: bool) -> bool:
        if not adverse:
            return price >= level if not self.is_short else price <= level
        return price <= level if not self.is_short else price >= level

    def _trailing_trigger(self) -> float | None:
        """Trailing level from the favourable extreme, or ``None``.

        Only armed once the position has actually moved in its favour. A trailing stop that
        can fire from the entry price is just a second, tighter stop.
        """
        if self.trailing_bps is None or self.trailing_bps <= 0:
            return None
        extreme = self._best_price
        moved = (extreme - self.entry_price) if not self.is_short else (self.entry_price - extreme)
        if moved <= 0:
            return None
        offset = extreme * (self.trailing_bps / 10_000.0)
        return extreme - offset if not self.is_short else extreme + offset

    def _close(self, price: float, at: datetime, reason: str) -> ShadowOutcome:
        gross = self._return_bps(price)
        adverse = abs(min(0.0, self._return_bps(self._worst_price)))
        favourable = max(0.0, self._return_bps(self._best_price))
        self._outcome = ShadowOutcome(
            position_id=self.position_id,
            context_id=self.context_id,
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            market=self.market,
            direction=self.direction,
            entry_price=self.entry_price,
            exit_price=price,
            opened_at=self.opened_at,
            closed_at=at,
            gross_return_bps=gross,
            net_return_bps=gross - self.cost_bps,
            cost_bps=self.cost_bps,
            max_adverse_excursion_bps=adverse,
            max_favorable_excursion_bps=favourable,
            holding_seconds=max(0.0, (at - self.opened_at).total_seconds()),
            exit_reason=reason,
            evidence_source=EVIDENCE_SHADOW,
            fill_assumed=True,
            quotes_observed=self._quotes,
            regime=self.regime,
        )
        return self._outcome

    def _return_bps(self, price: float) -> float:
        """Direction-signed return. The sign is applied ONCE, here."""
        if self.entry_price <= 0:
            return 0.0
        raw = (price / self.entry_price - 1.0) * 10_000.0
        return raw if not self.is_short else -raw

    @staticmethod
    def new_id() -> str:
        return f"shadow-{uuid4().hex}"

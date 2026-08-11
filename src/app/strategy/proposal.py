"""What a strategy is allowed to say, and nothing more.

A :class:`StrategyProposal` is a *thesis statement*: "under context X my entry condition
is (or is not) met, here is my reference price, my target, my stop and my horizon". It is
not an order, not a size, and not an approval.

The hard rules from the contract are enforced structurally rather than by review:

* There is no field for a quantity, an order side, a broker, a venue or a price policy.
  A proposal therefore cannot *be* an order, whatever a caller does with it.
* ``target_price`` / ``stop_price`` are proposals. The executable exit is resolved later
  by ``app.strategy.exit_geometry`` / ``DynamicExitPolicy`` against the real fill, which
  is the only price that exists.
* ``eligible`` and ``entry_ready`` are separate. Eligibility is the ontology's answer
  (may this thesis apply at all); entry readiness is the algorithm's (has my trigger
  fired). Collapsing them is what made "no strategy applies here" and "every strategy
  said no" the same reason code.

Every proposal carries ``context_id``, so a selection, a shadow outcome and a training
row can all be tied back to one market snapshot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

__all__ = ["StrategyProposal", "new_proposal_id"]


def new_proposal_id() -> str:
    return f"prop-{uuid4().hex}"


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class StrategyProposal:
    """One strategy's verdict on one context."""

    proposal_id: str
    context_id: str
    strategy_id: str
    symbol: str
    eligible: bool
    entry_ready: bool
    raw_signal_strength: float = 0.0
    #: The algorithm's own confidence in ``[0, 1]``. Distinct from signal strength: a
    #: weak-but-certain signal and a strong-but-doubtful one must not score alike.
    confidence: float = 0.0
    expected_horizon_seconds: int = 0
    reference_entry_price: float | None = None
    target_price: float | None = None
    stop_price: float | None = None
    #: The algorithm's own gross expected move, in bps. GROSS: costs are subtracted by
    #: ``TradingCostEngine`` downstream, never here, so a fee change does not require
    #: touching a strategy.
    expected_gross_edge_bps: float | None = None
    strategy_reason_codes: tuple[str, ...] = ()
    #: Point-in-time features the verdict was produced from, carried by value so a later
    #: read cannot change what this proposal was based on.
    feature_snapshot: Mapping[str, Any] = field(default_factory=dict)
    direction: str = "LONG"
    proposed_at: datetime | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy_id", str(self.strategy_id or "").strip().lower())
        object.__setattr__(self, "symbol", str(self.symbol or "").strip().upper())
        object.__setattr__(self, "direction", str(self.direction or "LONG").strip().upper())
        object.__setattr__(
            self, "expected_horizon_seconds", max(0, int(self.expected_horizon_seconds))
        )
        object.__setattr__(
            self, "raw_signal_strength", float(_finite(self.raw_signal_strength) or 0.0)
        )
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(_finite(self.confidence) or 0.0))))
        object.__setattr__(self, "reference_entry_price", _positive(self.reference_entry_price))
        object.__setattr__(self, "target_price", _positive(self.target_price))
        object.__setattr__(self, "stop_price", _positive(self.stop_price))
        object.__setattr__(self, "expected_gross_edge_bps", _finite(self.expected_gross_edge_bps))
        object.__setattr__(
            self,
            "strategy_reason_codes",
            tuple(dict.fromkeys(str(code) for code in self.strategy_reason_codes if str(code))),
        )
        if self.proposed_at is not None and self.proposed_at.tzinfo is None:
            object.__setattr__(self, "proposed_at", self.proposed_at.replace(tzinfo=timezone.utc))

    @property
    def is_short(self) -> bool:
        return self.direction == "SHORT"

    @property
    def selectable(self) -> bool:
        """May this proposal enter the utility ranking at all?

        Both conditions, because the two answer different questions and a candidate
        needs a yes from each: the ontology must permit the thesis here, and the thesis
        itself must have fired. Ranking an ``entry_ready=False`` proposal would let the
        selector arm a strategy whose own trigger said no.
        """
        return bool(self.eligible and self.entry_ready)

    @property
    def target_move_bps(self) -> float | None:
        """Signed favourable move to the proposed target, in bps.

        Direction-aware: a short's target sits BELOW its entry, so an unconditional
        ``target/entry - 1`` would report a short's target as a loss. Returns ``None``
        rather than 0.0 when either price is missing — an unpriceable proposal must not
        look like a zero-edge one.
        """
        entry = self.reference_entry_price
        target = self.target_price
        if entry is None or target is None:
            return None
        move = (target / entry - 1.0) * 10_000.0
        return move if not self.is_short else -move

    @property
    def stop_move_bps(self) -> float | None:
        """Adverse move to the proposed stop, as a POSITIVE magnitude."""
        entry = self.reference_entry_price
        stop = self.stop_price
        if entry is None or stop is None:
            return None
        return abs((stop / entry - 1.0) * 10_000.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "context_id": self.context_id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "eligible": self.eligible,
            "entry_ready": self.entry_ready,
            "selectable": self.selectable,
            "raw_signal_strength": round(self.raw_signal_strength, 4),
            "confidence": round(self.confidence, 4),
            "expected_horizon_seconds": self.expected_horizon_seconds,
            "reference_entry_price": self.reference_entry_price,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "expected_gross_edge_bps": (
                round(self.expected_gross_edge_bps, 3)
                if self.expected_gross_edge_bps is not None
                else None
            ),
            "target_move_bps": (
                round(value, 3) if (value := self.target_move_bps) is not None else None
            ),
            "stop_move_bps": (
                round(value, 3) if (value := self.stop_move_bps) is not None else None
            ),
            "strategy_reason_codes": list(self.strategy_reason_codes),
            "proposed_at": self.proposed_at.isoformat() if self.proposed_at else None,
            "diagnostics": dict(self.diagnostics),
        }


def _positive(value: Any) -> float | None:
    number = _finite(value)
    return number if number is not None and number > 0.0 else None

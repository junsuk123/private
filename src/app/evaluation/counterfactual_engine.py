"""Opens a virtual position for every strategy that WASN'T chosen.

The missing-data problem this addresses
--------------------------------------
``counterfactual_data_missing``: only the selected strategy's outcome is observable, so
there is nothing to learn "the other one would have been better" from. Without it, a
selector cannot be measured — the best it can report is whether its own picks made money,
which conflates strategy quality with selection quality.

So for each context the engine opens one :class:`ShadowPosition` per eligible, entry-ready
strategy. The live-selected strategy is recorded as a member of the group but is NOT
walked here: its outcome comes from the real fill, and scoring it twice would enter the
same trade into the same arm's posterior twice — the same defect the existing
``_journal_shadow_proposals(counterfactual=True)`` guard exists to prevent.

Correlation caveat, stated because it changes how the data may be used
---------------------------------------------------------------------
Shadow outcomes from one context are heavily correlated: they are the same price path cut
by different barriers. Treating them as independent samples would inflate every sample
count by roughly the number of strategies in the group. Every outcome therefore carries
its ``context_id``, and the group is retrievable, so downstream statistics can cluster by
context instead of counting rows. ``app.evaluation.selector_regret`` does exactly that.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from app.evaluation.shadow_position import ShadowOutcome, ShadowPosition

__all__ = [
    "CounterfactualEngine",
    "CounterfactualGroup",
    "CounterfactualStats",
]


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


@dataclass
class CounterfactualGroup:
    """The alternatives considered in one context, plus what the selector did."""

    context_id: str
    symbol: str
    market: str
    opened_at: datetime
    selected_strategy: str | None
    decision: str
    positions: dict[str, ShadowPosition] = field(default_factory=dict)
    outcomes: dict[str, ShadowOutcome] = field(default_factory=dict)
    #: Utility the selector predicted per strategy, kept so predicted-vs-realised
    #: calibration is answerable without re-running the selector.
    predicted_utility_bps: dict[str, float] = field(default_factory=dict)
    predicted_net_bps: dict[str, float] = field(default_factory=dict)
    #: Set once the live fill for ``selected_strategy`` is known.
    live_outcome_net_bps: float | None = None
    live_outcome_source: str | None = None

    @property
    def open_positions(self) -> tuple[ShadowPosition, ...]:
        return tuple(item for item in self.positions.values() if not item.resolved)

    @property
    def resolved(self) -> bool:
        return not self.open_positions

    def as_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "symbol": self.symbol,
            "market": self.market,
            "opened_at": self.opened_at.isoformat(),
            "selected_strategy": self.selected_strategy,
            "decision": self.decision,
            "position_count": len(self.positions),
            "resolved_count": len(self.outcomes),
            "predicted_utility_bps": {
                key: round(value, 3) for key, value in self.predicted_utility_bps.items()
            },
            "predicted_net_bps": {
                key: round(value, 3) for key, value in self.predicted_net_bps.items()
            },
            "live_outcome_net_bps": self.live_outcome_net_bps,
            "live_outcome_source": self.live_outcome_source,
            "outcomes": {key: value.as_dict() for key, value in self.outcomes.items()},
        }


@dataclass(frozen=True)
class CounterfactualStats:
    groups_open: int
    groups_resolved: int
    positions_open: int
    outcomes_recorded: int
    positions_rejected: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "groups_open": self.groups_open,
            "groups_resolved": self.groups_resolved,
            "positions_open": self.positions_open,
            "outcomes_recorded": self.outcomes_recorded,
            "positions_rejected": self.positions_rejected,
        }


class CounterfactualEngine:
    """Tracks alternative-strategy shadow positions per context.

    Bounded: ``max_groups`` caps how many contexts are tracked at once and the oldest is
    evicted first, so a long-running process cannot accumulate positions for a symbol whose
    quotes stopped arriving. Eviction force-expires the group so its partial evidence is
    still recorded rather than discarded.
    """

    def __init__(
        self,
        *,
        max_groups: int = 256,
        outcome_sink: Any | None = None,
    ) -> None:
        self._max_groups = max(1, int(max_groups))
        # Injected so the engine can write outcomes without importing the performance
        # store. Nothing here may reach a broker, and a sink is the narrowest possible
        # seam: it receives a resolved ShadowOutcome and returns nothing.
        self._sink = outcome_sink
        self._lock = threading.RLock()
        self._groups: "OrderedDict[str, CounterfactualGroup]" = OrderedDict()
        self._resolved_groups: "OrderedDict[str, CounterfactualGroup]" = OrderedDict()
        self._outcomes_recorded = 0
        self._positions_rejected = 0

    # -- opening ------------------------------------------------------------ #
    def open_from_selection(
        self,
        *,
        context: Any,
        selection: Any,
        proposals: Sequence[Any],
        costs: Mapping[str, Any] | None = None,
        trailing_bps_by_strategy: Mapping[str, float] | None = None,
    ) -> CounterfactualGroup | None:
        """Open one group for a selection result.

        Returns ``None`` when no proposal was both eligible and entry-ready — there is
        nothing counterfactual about a context in which nothing fired.
        """
        candidates = {
            str(getattr(item, "strategy_id", "")): item
            for item in tuple(getattr(selection, "ranked_candidates", ()) or ())
        }
        eligible = [
            proposal
            for proposal in proposals
            if bool(getattr(proposal, "eligible", False))
            and bool(getattr(proposal, "entry_ready", False))
        ]
        if not eligible:
            return None

        context_id = str(getattr(context, "context_id", "") or "")
        symbol = str(getattr(context, "symbol_id", "") or "")
        market = str(getattr(context, "market", "") or "")
        opened_at = _aware(getattr(context, "captured_at", datetime.now(timezone.utc)))
        regime = str(getattr(getattr(context, "macro", None), "market_regime", "") or "UNKNOWN")
        selected = getattr(selection, "selected_strategy", None)
        trailing = dict(trailing_bps_by_strategy or {})

        group = CounterfactualGroup(
            context_id=context_id,
            symbol=symbol,
            market=market,
            opened_at=opened_at,
            selected_strategy=str(selected) if selected else None,
            decision=str(getattr(selection, "decision", "") or ""),
        )
        for proposal in eligible:
            strategy_id = str(getattr(proposal, "strategy_id", ""))
            entry = getattr(proposal, "reference_entry_price", None)
            if not entry or float(entry) <= 0:
                # Without a point-in-time entry reference there is nothing to measure a
                # return against, and taking one from a later quote is the leak this whole
                # module is built to avoid.
                self._positions_rejected += 1
                continue
            candidate = candidates.get(strategy_id)
            cost = costs.get(strategy_id) if costs else None
            cost_bps = float(
                getattr(candidate, "expected_cost_bps", None)
                if candidate is not None
                else getattr(cost, "expected_cost_bps", 0.0) or 0.0
            )
            group.positions[strategy_id] = ShadowPosition(
                position_id=ShadowPosition.new_id(),
                context_id=context_id,
                strategy_id=strategy_id,
                symbol=symbol,
                market=market,
                direction=str(getattr(proposal, "direction", "LONG")),
                entry_price=float(entry),
                target_price=getattr(proposal, "target_price", None),
                stop_price=getattr(proposal, "stop_price", None),
                trailing_bps=trailing.get(strategy_id),
                max_holding_seconds=int(getattr(proposal, "expected_horizon_seconds", 0) or 600),
                cost_bps=cost_bps,
                opened_at=opened_at,
                regime=regime,
                was_selected=bool(selected) and strategy_id == str(selected),
                proposal_id=str(getattr(proposal, "proposal_id", "") or ""),
            )
            if candidate is not None:
                group.predicted_utility_bps[strategy_id] = float(
                    getattr(candidate, "final_utility_bps", 0.0)
                )
                group.predicted_net_bps[strategy_id] = float(
                    getattr(candidate, "expected_net_return_bps", 0.0)
                )
        if not group.positions:
            return None

        with self._lock:
            self._groups[context_id] = group
            self._groups.move_to_end(context_id)
            self._evict_locked()
        return group

    # -- walking ------------------------------------------------------------ #
    def observe_quote(
        self, symbol: str, price: float, at: datetime
    ) -> tuple[ShadowOutcome, ...]:
        """Feed one quote to every open position on ``symbol``.

        Positions belonging to the LIVE-selected strategy are walked too but their outcome
        is tagged and NOT emitted to the sink — see ``_emit``.
        """
        wanted = str(symbol or "").strip().upper()
        resolved: list[ShadowOutcome] = []
        with self._lock:
            groups = [item for item in self._groups.values() if item.symbol == wanted]
        for group in groups:
            for position in group.open_positions:
                outcome = position.update(price, at)
                if outcome is None:
                    continue
                group.outcomes[position.strategy_id] = outcome
                resolved.append(outcome)
                self._emit(position, outcome)
            self._retire_if_done(group)
        return tuple(resolved)

    def expire_stale(self, now: datetime) -> tuple[ShadowOutcome, ...]:
        """Close positions whose horizon has passed with no further quotes."""
        moment = _aware(now)
        resolved: list[ShadowOutcome] = []
        with self._lock:
            groups = list(self._groups.values())
        for group in groups:
            for position in group.open_positions:
                if moment < position.deadline:
                    continue
                outcome = position.expire(moment)
                if outcome is None:
                    # Never observed after the signal. Retire it with a marker so it stops
                    # being walked, and do NOT emit it: a zero-return row in a posterior
                    # would be a fabricated break-even trade.
                    position.mark_unobserved(moment)
                    continue
                group.outcomes[position.strategy_id] = outcome
                resolved.append(outcome)
                self._emit(position, outcome)
            self._retire_if_done(group)
        return tuple(resolved)

    # -- live outcome linkage ----------------------------------------------- #
    def record_live_outcome(
        self,
        *,
        context_id: str,
        strategy_id: str,
        net_return_bps: float,
        evidence_source: str,
    ) -> CounterfactualGroup | None:
        """Attach the REAL outcome of the selected strategy to its group.

        This is what makes regret computable: the selected strategy's number comes from a
        broker fill, the alternatives' from simulation, and the two are never mixed up
        because they arrive through different methods and carry different
        ``evidence_source`` values.
        """
        with self._lock:
            group = self._groups.get(str(context_id)) or self._resolved_groups.get(
                str(context_id)
            )
        if group is None:
            return None
        group.live_outcome_net_bps = float(net_return_bps)
        group.live_outcome_source = str(evidence_source)
        if group.selected_strategy is None:
            group.selected_strategy = str(strategy_id)
        return group

    # -- reads -------------------------------------------------------------- #
    def group(self, context_id: str) -> CounterfactualGroup | None:
        with self._lock:
            return self._groups.get(str(context_id)) or self._resolved_groups.get(
                str(context_id)
            )

    def resolved_groups(self, *, limit: int = 200) -> tuple[CounterfactualGroup, ...]:
        with self._lock:
            values = list(self._resolved_groups.values())
        return tuple(values[-max(0, int(limit)) :])

    @property
    def open_symbols(self) -> tuple[str, ...]:
        """Symbols that must keep receiving quotes until their groups resolve."""
        with self._lock:
            return tuple(dict.fromkeys(group.symbol for group in self._groups.values()))

    def stats(self) -> CounterfactualStats:
        with self._lock:
            return CounterfactualStats(
                groups_open=len(self._groups),
                groups_resolved=len(self._resolved_groups),
                positions_open=sum(
                    len(item.open_positions) for item in self._groups.values()
                ),
                outcomes_recorded=self._outcomes_recorded,
                positions_rejected=self._positions_rejected,
            )

    # -- internals ---------------------------------------------------------- #
    def _emit(self, position: ShadowPosition, outcome: ShadowOutcome) -> None:
        self._outcomes_recorded += 1
        if position.was_selected:
            # The selected strategy's evidence is its real fill. Emitting the simulated
            # version too would double-count it against its own posterior.
            return
        sink = self._sink
        if sink is None:
            return
        try:
            sink(outcome)
        except Exception:  # noqa: BLE001 - a sink failure costs a sample, never a cycle.
            pass

    def _retire_if_done(self, group: CounterfactualGroup) -> None:
        if not group.resolved:
            return
        with self._lock:
            self._groups.pop(group.context_id, None)
            self._resolved_groups[group.context_id] = group
            self._resolved_groups.move_to_end(group.context_id)
            while len(self._resolved_groups) > self._max_groups:
                self._resolved_groups.popitem(last=False)

    def _evict_locked(self) -> None:
        while len(self._groups) > self._max_groups:
            _, victim = self._groups.popitem(last=False)
            # Force-expire so partial evidence survives eviction. Discarding it silently
            # would bias the record toward contexts whose quotes kept flowing.
            for position in victim.open_positions:
                outcome = position.expire(datetime.now(timezone.utc))
                if outcome is not None:
                    victim.outcomes[position.strategy_id] = outcome
                    self._emit(position, outcome)
            self._resolved_groups[victim.context_id] = victim
            while len(self._resolved_groups) > self._max_groups:
                self._resolved_groups.popitem(last=False)



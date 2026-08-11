"""Does the edge survive costs being worse than measured?

The measured pathology this targets
-----------------------------------
The project has already recorded strategies with a positive gross edge and a deeply negative
net one (``gap_context``: +12.1bps gross, -15.8bps net), and US round-trip cost coming in at
a median 63.2bps against the 28bps a KRX-sized table assumed. A validation that uses the
mean measured cost therefore says nothing about whether the strategy can be traded: cost is
the term with the most variance across venue, spread and time of day.

So the stress is multiplicative on the cost actually recorded per trade, and the reported
number is the ``break_even_cost_multiple`` — how much worse costs can get before the edge is
gone. A strategy that breaks even at 1.05x is not tradable, whatever its mean says.

Spread and slippage are stressed separately from fees because they behave differently: fees
are a known rate, spread is a market state that widens exactly when a strategy most wants to
exit. ``spread_shock_bps`` is therefore additive rather than multiplicative — a shock adds
basis points to the round trip, it does not scale a rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from app.strategy_validation.metrics import TradeObservation

__all__ = [
    "CostStressResult",
    "CostStressScenario",
    "DEFAULT_SCENARIOS",
    "cost_stress",
]


@dataclass(frozen=True)
class CostStressScenario:
    name: str
    cost_multiple: float = 1.0
    #: Additive basis points on top of the scaled cost, for a spread/slippage shock.
    spread_shock_bps: float = 0.0

    def applied_cost_bps(self, measured_cost_bps: float) -> float:
        return max(0.0, measured_cost_bps * self.cost_multiple + self.spread_shock_bps)


#: The multiples are not arbitrary. 1.0 is what was measured; 1.5 is roughly the KRX-to-US
#: ratio the stored tape showed (28 -> 63bps median); 2.0 covers the US p90 of 125bps against
#: its own median. The 20bps spread shock is one typical KRX top-of-book spread, which is the
#: unit ``exit_geometry`` sizes its stops against.
DEFAULT_SCENARIOS: tuple[CostStressScenario, ...] = (
    CostStressScenario("measured", 1.0, 0.0),
    CostStressScenario("cost_x1.25", 1.25, 0.0),
    CostStressScenario("cost_x1.5", 1.5, 0.0),
    CostStressScenario("cost_x2", 2.0, 0.0),
    CostStressScenario("spread_shock_20bps", 1.0, 20.0),
    CostStressScenario("cost_x1.5_spread_shock_20bps", 1.5, 20.0),
)


@dataclass(frozen=True)
class ScenarioOutcome:
    scenario: str
    net_ev_bps: float | None
    hit_rate: float | None
    survives: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "net_ev_bps": _round(self.net_ev_bps),
            "hit_rate": _round(self.hit_rate, 4),
            "survives": self.survives,
        }


@dataclass(frozen=True)
class CostStressResult:
    strategy_id: str
    trade_count: int
    gross_ev_bps: float | None
    measured_cost_bps: float | None
    scenarios: tuple[ScenarioOutcome, ...]
    break_even_cost_multiple: float | None
    reason_codes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def survives_all(self) -> bool:
        return bool(self.scenarios) and all(item.survives for item in self.scenarios)

    @property
    def survives_measured_only(self) -> bool:
        """Positive at the measured cost and negative at any stress. A fragile edge."""
        if not self.scenarios:
            return False
        measured = next((item for item in self.scenarios if item.scenario == "measured"), None)
        return bool(measured and measured.survives and not self.survives_all)

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "trade_count": self.trade_count,
            "gross_ev_bps": _round(self.gross_ev_bps),
            "measured_cost_bps": _round(self.measured_cost_bps),
            "break_even_cost_multiple": _round(self.break_even_cost_multiple, 3),
            "survives_all": self.survives_all,
            "survives_measured_only": self.survives_measured_only,
            "scenarios": [item.as_dict() for item in self.scenarios],
            "reason_codes": list(self.reason_codes),
        }


def cost_stress(
    strategy_id: str,
    trades: Sequence[TradeObservation],
    *,
    scenarios: Iterable[CostStressScenario] = DEFAULT_SCENARIOS,
    minimum_net_bps: float = 0.0,
) -> CostStressResult:
    """Re-price every trade under each scenario and report whether the edge survives."""
    rows = tuple(trades)
    if not rows:
        return CostStressResult(
            strategy_id=strategy_id,
            trade_count=0,
            gross_ev_bps=None,
            measured_cost_bps=None,
            scenarios=(),
            break_even_cost_multiple=None,
            reason_codes=("COST_STRESS_NO_TRADES",),
        )

    gross_values = [trade.gross_return_bps for trade in rows]
    cost_values = [trade.cost_bps for trade in rows]
    gross_ev = sum(gross_values) / len(gross_values)
    measured_cost = sum(cost_values) / len(cost_values)

    outcomes: list[ScenarioOutcome] = []
    for scenario in scenarios:
        nets = [
            trade.gross_return_bps - scenario.applied_cost_bps(trade.cost_bps)
            for trade in rows
        ]
        net_ev = sum(nets) / len(nets)
        outcomes.append(
            ScenarioOutcome(
                scenario=scenario.name,
                net_ev_bps=net_ev,
                hit_rate=sum(1 for value in nets if value > 0) / len(nets),
                survives=net_ev > minimum_net_bps,
            )
        )

    reasons: list[str] = []
    if measured_cost <= 0:
        reasons.append("COST_STRESS_NO_MEASURED_COST")
    return CostStressResult(
        strategy_id=strategy_id,
        trade_count=len(rows),
        gross_ev_bps=gross_ev,
        measured_cost_bps=measured_cost,
        scenarios=tuple(outcomes),
        break_even_cost_multiple=_break_even_multiple(rows, minimum_net_bps),
        reason_codes=tuple(reasons),
    )


def _break_even_multiple(
    trades: Sequence[TradeObservation], minimum_net_bps: float
) -> float | None:
    """Cost multiple at which mean net EV falls to ``minimum_net_bps``.

    Closed form rather than a search, because mean net is linear in the multiple:

        mean(gross) - m * mean(cost) = minimum  ->  m = (mean(gross) - minimum) / mean(cost)

    ``None`` when the mean measured cost is zero (nothing to scale) and a value below 1.0
    means the edge is already gone at the cost that was actually paid.
    """
    gross = sum(trade.gross_return_bps for trade in trades) / len(trades)
    cost = sum(trade.cost_bps for trade in trades) / len(trades)
    if cost <= 0:
        return None
    multiple = (gross - minimum_net_bps) / cost
    return multiple if math.isfinite(multiple) else None


def _round(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None

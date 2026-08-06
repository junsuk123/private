from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.routing.actions import StrategyRoutingAction
from app.trading.contracts import OntologyDecision, Position, StrategyUtilityEvidence


@dataclass(frozen=True)
class RoutingDecision:
    as_of: datetime
    symbol: str
    action: str
    selected: StrategyUtilityEvidence | None
    reason_codes: tuple[str, ...]
    ranked_evidence_ids: tuple[str, ...]
    weighted_utility: float | None = None

    @property
    def is_no_trade(self) -> bool:
        return self.action == StrategyRoutingAction.NO_TRADE.value


def _net_surplus_bps(item: StrategyUtilityEvidence) -> float:
    """How far past its OWN requirement this candidate sits.

    ``required_net_return_bps`` is optional on the evidence contract; when a
    producer has not supplied it the surplus degrades to the raw net, which is the
    previous behaviour. Missing data must not silently promote a candidate, so the
    fallback is the more conservative of the two readings.
    """
    net = float(getattr(item, "expected_net_return_bps", 0.0) or 0.0)
    required = getattr(item, "required_net_return_bps", None)
    try:
        required_value = float(required) if required is not None else 0.0
    except (TypeError, ValueError):
        required_value = 0.0
    return net - max(0.0, required_value)


class StrategyRouter:
    def __init__(
        self,
        *,
        minimum_utility: float = 0.0,
        maximum_total_uncertainty: float = 1.0,
        minimum_net_edge_bps: float = 0.0,
    ) -> None:
        self.minimum_utility = minimum_utility
        self.maximum_total_uncertainty = maximum_total_uncertainty
        self.minimum_net_edge_bps = max(0.0, float(minimum_net_edge_bps))

    def route(
        self,
        *,
        as_of: datetime,
        symbol: str,
        ontology: OntologyDecision,
        evidence: tuple[StrategyUtilityEvidence, ...],
        open_positions: tuple[Position, ...] = (),
    ) -> RoutingDecision:
        if any(position.symbol == symbol and position.quantity != 0 for position in open_positions):
            return self._no_trade(as_of, symbol, ("POSITION_ALREADY_OWNED",), evidence)
        if not (ontology.as_of <= as_of <= ontology.valid_until):
            return self._no_trade(as_of, symbol, ("ONTOLOGY_SNAPSHOT_STALE",), evidence)

        candidates: list[tuple[StrategyUtilityEvidence, float]] = []
        blocked_reasons: list[str] = []
        allowed = set(ontology.allowed_strategy_ids)
        for item in evidence:
            if item.symbol != symbol or item.as_of > as_of:
                continue
            if item.strategy_id not in allowed or not item.ontology_allowed:
                blocked_reasons.append(f"ONTOLOGY_BLOCKED:{item.strategy_id}")
                continue
            if item.hard_block_reasons:
                blocked_reasons.append(f"HARD_BLOCK:{item.strategy_id}")
                continue
            if item.expected_net_return_bps <= 0.0:
                blocked_reasons.append(f"NON_POSITIVE_NET_EDGE:{item.strategy_id}")
                continue
            if item.expected_net_return_bps < self.minimum_net_edge_bps:
                blocked_reasons.append(
                    f"NET_EDGE_BELOW_EXECUTION_FLOOR:{item.strategy_id}"
                )
                continue
            uncertainty = item.aleatoric_uncertainty + item.epistemic_uncertainty_or_proxy
            if uncertainty > self.maximum_total_uncertainty:
                blocked_reasons.append(f"UNCERTAINTY_TOO_HIGH:{item.strategy_id}")
                continue
            weighted_utility = item.utility * item.compatibility_score
            if weighted_utility <= self.minimum_utility:
                blocked_reasons.append(f"UTILITY_BELOW_THRESHOLD:{item.strategy_id}")
                continue
            candidates.append((item, weighted_utility))
        # Rank by SURPLUS over the bar each candidate had to clear, not by raw net.
        #
        # Raw net favours whichever strategy happens to carry the highest cost, and
        # an arbitrary per-strategy target made the comparison worse still: two
        # candidates with different targets were being ordered on numbers that did
        # not mean the same thing. Surplus (net minus its own requirement) is
        # comparable across strategies, markets and horizons by construction.
        ranked = sorted(
            candidates,
            key=lambda pair: (
                _net_surplus_bps(pair[0]),
                pair[1],
                pair[0].fill_probability,
                pair[0].strategy_id,
            ),
            reverse=True,
        )
        if not ranked:
            reasons = tuple(dict.fromkeys(blocked_reasons)) or ("NO_ADMISSIBLE_EVIDENCE",)
            return self._no_trade(as_of, symbol, reasons, evidence)
        return RoutingDecision(
            as_of=as_of,
            symbol=symbol,
            action=StrategyRoutingAction.ACTIVATE_STRATEGY.value,
            selected=ranked[0][0],
            reason_codes=("MAX_ONTOLOGY_WEIGHTED_NET_UTILITY",),
            ranked_evidence_ids=tuple(item.evidence_id for item, _ in ranked),
            weighted_utility=ranked[0][1],
        )

    @staticmethod
    def _no_trade(
        as_of: datetime,
        symbol: str,
        reasons: tuple[str, ...],
        evidence: tuple[StrategyUtilityEvidence, ...],
    ) -> RoutingDecision:
        return RoutingDecision(
            as_of=as_of,
            symbol=symbol,
            action=StrategyRoutingAction.NO_TRADE.value,
            selected=None,
            reason_codes=reasons,
            ranked_evidence_ids=tuple(item.evidence_id for item in evidence),
            weighted_utility=None,
        )

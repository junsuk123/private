from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.trading.contracts import OntologyDecision, Position, StrategyUtilityEvidence


@dataclass(frozen=True)
class RoutingDecision:
    as_of: datetime
    symbol: str
    action: str
    selected: StrategyUtilityEvidence | None
    reason_codes: tuple[str, ...]
    ranked_evidence_ids: tuple[str, ...]

    @property
    def is_no_trade(self) -> bool:
        return self.action == "NO_TRADE"


class StrategyRouter:
    def __init__(
        self,
        *,
        minimum_utility: float = 0.0,
        maximum_total_uncertainty: float = 1.0,
    ) -> None:
        self.minimum_utility = minimum_utility
        self.maximum_total_uncertainty = maximum_total_uncertainty

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

        candidates: list[StrategyUtilityEvidence] = []
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
            if item.expected_net_return_bps <= 0:
                blocked_reasons.append(f"NON_POSITIVE_NET_EDGE:{item.strategy_id}")
                continue
            uncertainty = item.aleatoric_uncertainty + item.epistemic_uncertainty_or_proxy
            if uncertainty > self.maximum_total_uncertainty:
                blocked_reasons.append(f"UNCERTAINTY_TOO_HIGH:{item.strategy_id}")
                continue
            if item.utility < self.minimum_utility:
                blocked_reasons.append(f"UTILITY_BELOW_THRESHOLD:{item.strategy_id}")
                continue
            candidates.append(item)
        ranked = sorted(
            candidates,
            key=lambda item: (
                item.utility,
                item.expected_net_return_bps,
                item.fill_probability,
                item.strategy_id,
            ),
            reverse=True,
        )
        if not ranked:
            reasons = tuple(dict.fromkeys(blocked_reasons)) or ("NO_ADMISSIBLE_EVIDENCE",)
            return self._no_trade(as_of, symbol, reasons, evidence)
        return RoutingDecision(
            as_of=as_of,
            symbol=symbol,
            action="ACTIVATE_STRATEGY",
            selected=ranked[0],
            reason_codes=("MAX_NET_UTILITY",),
            ranked_evidence_ids=tuple(item.evidence_id for item in ranked),
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
            action="NO_TRADE",
            selected=None,
            reason_codes=reasons,
            ranked_evidence_ids=tuple(item.evidence_id for item in evidence),
        )

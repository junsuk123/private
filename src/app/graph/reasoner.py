from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from app.graph.knowledge_graph import KnowledgeGraph, Triple
from app.graph.runtime import OntologyRuntime, get_ontology_runtime
from app.schemas.domain import ReasoningPath

SUPPORT_WEIGHTS = {
    "EarningsGrowth": 0.18,
    "ProfitabilityQuality": 0.14,
    "PositiveEventImpact": 0.10,
    "PositiveInvestorFlow": 0.10,
    "InformedOrderFlowImbalance": 0.13,
    "ForeignInstitutionJointBuying": 0.12,
    "RetailSupplyAbsorbedByInformedFlow": 0.10,
    "OrderFlowPriceConfirmation": 0.09,
    "SuspectedSmartMoneyAccumulation": 0.08,
    "OrderFlowConfirmedBuyCandidate": 0.10,
    "CashFitOneShare": 0.07,
    "AffordableByAccountCash": 0.08,
    "SectorMomentum": 0.08,
}
CONTRADICTION_WEIGHTS = {
    "ValuationDiscipline": 0.12,
    "ValuationSlightlyHigh": 0.10,
    "InformedOrderFlowDistribution": 0.15,
    "ForeignInstitutionJointSelling": 0.13,
    "RetailDemandMeetsInformedSelling": 0.11,
    "OrderFlowPriceDivergence": 0.10,
    "SuspectedSmartMoneyDistribution": 0.12,
    "CashBelowOneSharePrice": 0.18,
}
RISK_WEIGHTS = {
    "MacroRateRisk": 0.14,
    "VolatilityRisk": 0.18,
    "NegativeEventRisk": 0.18,
    "LiquidityRisk": 0.20,
    "OrderFlowDistributionRisk": 0.16,
    "ThinLiquidityPriceImpactRisk": 0.14,
    "InsufficientAccountCashRisk": 0.20,
}


@dataclass(frozen=True)
class OntologyReasoningPolicy:
    base_confidence: float = 0.40
    buy_threshold: float = 0.58
    support_weights: dict[str, float] = field(default_factory=lambda: dict(SUPPORT_WEIGHTS))
    contradiction_weights: dict[str, float] = field(default_factory=lambda: dict(CONTRADICTION_WEIGHTS))
    risk_weights: dict[str, float] = field(default_factory=lambda: dict(RISK_WEIGHTS))
    default_support_weight: float = 0.05
    default_contradiction_weight: float = 0.05
    default_risk_weight: float = 0.06
    min_confidence: float = 0.05
    max_confidence: float = 0.95


class OntologyReasoner:
    def __init__(
        self,
        graph: KnowledgeGraph,
        runtime: OntologyRuntime | None = None,
        policy: OntologyReasoningPolicy | None = None,
    ) -> None:
        self.graph = graph
        self.runtime = runtime or get_ontology_runtime()
        self.policy = policy or OntologyReasoningPolicy()

    def infer(self) -> KnowledgeGraph:
        self._infer_buy_candidates()
        self._infer_risk_adjustments()
        return self.graph

    def build_reasoning_paths(self, tickers: tuple[str, ...]) -> tuple[ReasoningPath, ...]:
        paths = []
        for ticker in tickers:
            support = self.graph.matching(subject=ticker, predicate="supportsSignal")
            contradiction = self.graph.matching(subject=ticker, predicate="contradictsSignal")
            risk = self.graph.matching(subject=ticker, predicate="increasesRiskOf")
            sizing = self.graph.matching(subject=ticker, predicate="requiresSizingAdjustment")
            support_score = sum(
                self.policy.support_weights.get(triple.object, self.policy.default_support_weight)
                for triple in support
                if triple.object != "RiskAdjustedSizing"
            )
            contradiction_score = sum(
                self.policy.contradiction_weights.get(triple.object, self.policy.default_contradiction_weight)
                for triple in contradiction
            )
            risk_score = sum(
                self.policy.risk_weights.get(triple.object, self.policy.default_risk_weight)
                for triple in risk
            )
            confidence = max(
                self.policy.min_confidence,
                min(self.policy.max_confidence, self.policy.base_confidence + support_score - contradiction_score - risk_score),
            )
            conclusion = "BuyCandidate" if confidence >= self.policy.buy_threshold else "HoldOrWatch"
            path_id = hashlib.sha256(
                f"{ticker}:{support}:{contradiction}:{risk}:{conclusion}".encode("utf-8")
            ).hexdigest()[:16]
            paths.append(
                ReasoningPath(
                    path_id=path_id,
                    ticker=ticker,
                    conclusion=conclusion,
                    confidence=confidence,
                    supporting_triples=_format_triples(support),
                    contradicting_triples=_format_triples(contradiction),
                    risk_triples=_format_triples(risk),
                    explanation=_explain(ticker, conclusion, confidence, support, contradiction, risk, sizing),
                )
            )
        return tuple(paths)

    def _infer_buy_candidates(self) -> None:
        subjects = {triple.subject for triple in self.graph.triples()}
        for subject in subjects:
            support_objects = set(self.graph.objects(subject, "supportsSignal"))
            contra_objects = set(self.graph.objects(subject, "contradictsSignal"))
            risk_objects = set(self.graph.objects(subject, "increasesRiskOf"))
            if {"EarningsGrowth", "ProfitabilityQuality"}.issubset(support_objects):
                self.graph.add(subject, "supportsSignal", "BuyCandidate", "reasoner:growth-quality")
            if "BuyCandidate" in support_objects and "MacroRateRisk" in risk_objects:
                self.graph.add(subject, "contradictsSignal", "AggressiveBuy", "reasoner:macro-risk")
            if {"InformedOrderFlowImbalance", "OrderFlowPriceConfirmation"}.issubset(support_objects):
                self.graph.add(subject, "supportsSignal", "OrderFlowConfirmedBuyCandidate", "reasoner:ofi-price-impact")
            if "InformedOrderFlowDistribution" in contra_objects:
                self.graph.add(subject, "contradictsSignal", "BuyCandidate", "reasoner:ofi-distribution")
            if {"CashFitOneShare", "AffordableByAccountCash"}.issubset(support_objects):
                self.graph.add(subject, "supportsSignal", "AccountCashFeasibleBuyCandidate", "reasoner:account-cash")
            if "CashBelowOneSharePrice" in contra_objects:
                self.graph.add(subject, "contradictsSignal", "BuyCandidate", "reasoner:account-cash")

    def _infer_risk_adjustments(self) -> None:
        for triple in self.graph.matching(predicate="increasesRiskOf"):
            if triple.object in {"MacroRateRisk", "VolatilityRisk", "NegativeEventRisk"}:
                self.graph.add(triple.subject, "requiresSizingAdjustment", "RiskAdjustedSizing", "reasoner:risk-sizing")
            if triple.object in {"OrderFlowDistributionRisk", "ThinLiquidityPriceImpactRisk", "InsufficientAccountCashRisk"}:
                self.graph.add(triple.subject, "requiresSizingAdjustment", "RiskAdjustedSizing", "reasoner:flow-risk-sizing")


def _format_triples(triples: tuple[Triple, ...]) -> tuple[str, ...]:
    return tuple(f"{triple.subject} --{triple.predicate}--> {triple.object}" for triple in triples)


def _explain(
    ticker: str,
    conclusion: str,
    confidence: float,
    support: tuple[Triple, ...],
    contradiction: tuple[Triple, ...],
    risk: tuple[Triple, ...],
    sizing: tuple[Triple, ...] = (),
) -> str:
    return (
        f"{ticker} conclusion={conclusion}, confidence={confidence:.2f}. "
        f"Support={len(support)}, contradiction={len(contradiction)}, risk={len(risk)}, sizing_adjustments={len(sizing)}."
    )

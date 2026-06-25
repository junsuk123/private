from __future__ import annotations

import hashlib

from app.graph.knowledge_graph import KnowledgeGraph, Triple
from app.graph.runtime import OntologyRuntime, get_ontology_runtime
from app.schemas.domain import ReasoningPath

SUPPORT_WEIGHTS = {
    "EarningsGrowth": 0.18,
    "ProfitabilityQuality": 0.14,
    "PositiveEventImpact": 0.10,
    "PositiveInvestorFlow": 0.10,
    "SectorMomentum": 0.08,
}
CONTRADICTION_WEIGHTS = {
    "ValuationDiscipline": 0.12,
    "ValuationSlightlyHigh": 0.10,
}
RISK_WEIGHTS = {
    "MacroRateRisk": 0.14,
    "VolatilityRisk": 0.18,
    "NegativeEventRisk": 0.18,
    "LiquidityRisk": 0.20,
}


class OntologyReasoner:
    def __init__(self, graph: KnowledgeGraph, runtime: OntologyRuntime | None = None) -> None:
        self.graph = graph
        self.runtime = runtime or get_ontology_runtime()

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
            support_score = sum(SUPPORT_WEIGHTS.get(triple.object, 0.05) for triple in support)
            contradiction_score = sum(CONTRADICTION_WEIGHTS.get(triple.object, 0.05) for triple in contradiction)
            risk_score = sum(RISK_WEIGHTS.get(triple.object, 0.06) for triple in risk)
            confidence = max(0.05, min(0.95, 0.40 + support_score - contradiction_score - risk_score))
            conclusion = "BuyCandidate" if confidence >= 0.58 else "HoldOrWatch"
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
                    explanation=_explain(ticker, conclusion, confidence, support, contradiction, risk),
                )
            )
        return tuple(paths)

    def _infer_buy_candidates(self) -> None:
        subjects = {triple.subject for triple in self.graph.triples()}
        for subject in subjects:
            support_objects = set(self.graph.objects(subject, "supportsSignal"))
            risk_objects = set(self.graph.objects(subject, "increasesRiskOf"))
            if {"EarningsGrowth", "ProfitabilityQuality"}.issubset(support_objects):
                self.graph.add(subject, "supportsSignal", "BuyCandidate", "reasoner:growth-quality")
            if "BuyCandidate" in support_objects and "MacroRateRisk" in risk_objects:
                self.graph.add(subject, "contradictsSignal", "AggressiveBuy", "reasoner:macro-risk")

    def _infer_risk_adjustments(self) -> None:
        for triple in self.graph.matching(predicate="increasesRiskOf"):
            if triple.object in {"MacroRateRisk", "VolatilityRisk", "NegativeEventRisk"}:
                self.graph.add(triple.subject, "supportsSignal", "RiskAdjustedSizing", "reasoner:risk-sizing")


def _format_triples(triples: tuple[Triple, ...]) -> tuple[str, ...]:
    return tuple(f"{triple.subject} --{triple.predicate}--> {triple.object}" for triple in triples)


def _explain(
    ticker: str,
    conclusion: str,
    confidence: float,
    support: tuple[Triple, ...],
    contradiction: tuple[Triple, ...],
    risk: tuple[Triple, ...],
) -> str:
    return (
        f"{ticker} conclusion={conclusion}, confidence={confidence:.2f}. "
        f"Support={len(support)}, contradiction={len(contradiction)}, risk={len(risk)}."
    )

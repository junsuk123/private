"""Trading-domain ontology (theory-driven, deterministic, explainable).

This package sits alongside the existing semantic-graph / RDF-OWL layer in
``app.graph`` and the hierarchical macro–micro reasoners. It adds a compact,
deterministic *decision ontology* that encodes the semantic relationships the
system must respect before a live order — market regime, signal validity,
execution feasibility, cost-adjusted edge, risk state, and validation evidence —
and returns an explainable :class:`OntologyReasoningResult`.

It is ADVISORY by design: it can annotate, strengthen, weaken, or BLOCK a signal,
but it can never *authorize* a trade. RiskManager / ProfitabilityGate / FinalTradeGate
remain the sole execution gates (see ``app.graph`` README).
"""

from app.ontology.trading_domain_ontology import (
    DataTier,
    IntentType,
    OntologyReasoningResult,
    OrderPolicyRecommendation,
    ValidationState,
)
from app.ontology.trading_fact_builder import TradingFacts, build_trading_facts
from app.ontology.trading_reasoner import TradingDomainReasoner

__all__ = [
    "DataTier",
    "IntentType",
    "OntologyReasoningResult",
    "OrderPolicyRecommendation",
    "ValidationState",
    "TradingFacts",
    "build_trading_facts",
    "TradingDomainReasoner",
]

from __future__ import annotations

CLASSES = (
    "Company",
    "Stock",
    "Sector",
    "IndustryTheme",
    "FinancialMetric",
    "MarketMetric",
    "TechnicalIndicator",
    "MacroFactor",
    "DisclosureEvent",
    "NewsEvent",
    "SentimentSignal",
    "SemanticFeature",
    "MarketRegime",
    "RiskFactor",
    "PortfolioState",
    "Position",
    "StrategySignal",
    "OrderIntent",
    "RiskManagerDecision",
    "FinalOrder",
    "ExecutedOrder",
    "ReasoningPath",
)

RELATIONSHIPS = (
    "belongsToSector",
    "hasTicker",
    "hasFinancialMetric",
    "hasMarketMetric",
    "hasTechnicalIndicator",
    "affectedByMacroFactor",
    "hasRecentDisclosure",
    "hasRecentNews",
    "generatesSemanticFeature",
    "supportsSignal",
    "contradictsSignal",
    "increasesRiskOf",
    "decreasesRiskOf",
    "hasExposureTo",
    "isIncludedInPortfolio",
    "generatesOrderIntent",
    "isRejectedByRiskRule",
    "isApprovedByRiskManager",
    "isExecutedAs",
)


def validate_triples(subject_predicates: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(predicate for predicate in subject_predicates if predicate not in RELATIONSHIPS)


def build_base_dictionary():
    """Pre-intern the base vocabulary so ids are stable and deterministic.

    Interning ``CLASSES`` as terms and ``RELATIONSHIPS`` as predicates first means
    the base vocabulary always gets the same low ids regardless of what dynamic
    facts are added afterwards. This gives ``FactDictionary.signature()`` a stable
    prefix for cache versioning. Imported lazily to avoid a hard dependency for
    callers that only need the vocabulary tuples.
    """
    from .fact_dictionary import FactDictionary

    dictionary = FactDictionary()
    for class_name in CLASSES:
        dictionary.intern_term(class_name)
    for relationship in RELATIONSHIPS:
        dictionary.intern_predicate(relationship)
    return dictionary

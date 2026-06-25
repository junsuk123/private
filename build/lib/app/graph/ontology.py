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
    "RiskFactor",
    "PortfolioState",
    "Position",
    "StrategySignal",
    "OrderIntent",
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

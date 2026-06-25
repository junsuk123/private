from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

SemanticRelation = Literal["supportsSignal", "contradictsSignal", "increasesRiskOf", "decreasesRiskOf"]


@dataclass(frozen=True)
class OHLCVBar:
    ticker: str
    as_of: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class RawIndicatorRecord:
    ticker: str
    as_of: datetime
    indicator_name: str
    value: float | str | None
    unit: str
    lookback_window: str | None
    source: str
    calculation_version: str
    calculation_method: str = "formula"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FormulaParameterRecommendation:
    ticker: str
    as_of: datetime
    parameter_name: str
    value: float | int | str
    recommended_by: str
    model_version: str
    confidence: float
    reason: str
    source_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticFeatureRecord:
    ticker: str
    as_of: datetime
    feature_name: str
    feature_category: str
    state: str
    confidence: float
    supporting_indicators: tuple[str, ...]
    semantic_relation: SemanticRelation
    target_signal: str | None
    ontology_node_id: str | None
    generation_method: str = "formula_rule"
    model_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReasoningPathRecord:
    ticker: str
    as_of: datetime
    strategy_signal: str
    positive_features: tuple[str, ...]
    negative_features: tuple[str, ...]
    risk_features: tuple[str, ...]
    contradiction_score: float
    final_confidence: float
    explanation: str


@dataclass(frozen=True)
class FeatureSnapshot:
    ticker: str
    as_of: datetime
    raw_indicators: tuple[RawIndicatorRecord, ...]
    semantic_features: tuple[SemanticFeatureRecord, ...]
    reasoning_paths: tuple[ReasoningPathRecord, ...] = ()
    parameter_recommendations: tuple[FormulaParameterRecommendation, ...] = ()

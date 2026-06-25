from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from app.features.schemas import RawIndicatorRecord, SemanticFeatureRecord

AI_LAYER_VERSION = "ai-semantic-layer-v1"


@dataclass(frozen=True)
class AISemanticTarget:
    feature_name: str
    feature_category: str
    semantic_relation: str
    target_signal: str | None
    input_indicators: tuple[str, ...]
    threshold: float = 0.55


@dataclass(frozen=True)
class AISemanticTrainingExample:
    ticker: str
    as_of: datetime
    inputs: dict[str, float]
    labels: dict[str, int]


@dataclass(frozen=True)
class AISemanticModelState:
    model_version: str
    targets: tuple[AISemanticTarget, ...]
    class_centroids: dict[str, dict[int, dict[str, float]]] = field(default_factory=dict)
    feature_means: dict[str, float] = field(default_factory=dict)
    feature_stdevs: dict[str, float] = field(default_factory=dict)


class AISemanticModel(Protocol):
    model_version: str

    def fit(self, examples: tuple[AISemanticTrainingExample, ...]) -> AISemanticModelState:
        ...

    def predict(
        self,
        indicators: tuple[RawIndicatorRecord, ...],
        as_of: datetime | None = None,
    ) -> tuple[SemanticFeatureRecord, ...]:
        ...


class CentroidAISemanticModel:
    """Small dependency-free classifier for tunable semantic states.

    It learns class centroids per target from as-of training examples. This is
    intentionally modest: production can replace it with LightGBM, a neural time
    series model, or an LLM/text model while keeping the same interface.
    """

    def __init__(
        self,
        targets: tuple[AISemanticTarget, ...],
        state: AISemanticModelState | None = None,
        model_version: str = AI_LAYER_VERSION,
    ) -> None:
        self.targets = targets
        self.model_version = model_version
        self.state = state or AISemanticModelState(model_version=model_version, targets=targets)

    def fit(self, examples: tuple[AISemanticTrainingExample, ...]) -> AISemanticModelState:
        all_inputs = sorted({key for example in examples for key in example.inputs})
        means = {
            key: _mean([example.inputs[key] for example in examples if key in example.inputs])
            for key in all_inputs
        }
        stdevs = {
            key: _stdev([example.inputs[key] for example in examples if key in example.inputs])
            for key in all_inputs
        }
        centroids: dict[str, dict[int, dict[str, float]]] = {}
        for target in self.targets:
            target_centroids: dict[int, dict[str, float]] = {}
            for label in (0, 1):
                matching = [example for example in examples if example.labels.get(target.feature_name) == label]
                if not matching:
                    continue
                target_centroids[label] = {
                    key: _mean([_standardize(example.inputs.get(key), means.get(key), stdevs.get(key)) for example in matching])
                    for key in target.input_indicators
                }
            centroids[target.feature_name] = target_centroids
        self.state = AISemanticModelState(
            model_version=self.model_version,
            targets=self.targets,
            class_centroids=centroids,
            feature_means=means,
            feature_stdevs=stdevs,
        )
        return self.state

    def predict(
        self,
        indicators: tuple[RawIndicatorRecord, ...],
        as_of: datetime | None = None,
    ) -> tuple[SemanticFeatureRecord, ...]:
        if not indicators:
            return ()
        values = _indicator_values(indicators)
        ticker = indicators[-1].ticker
        snapshot_time = as_of or indicators[-1].as_of
        features: list[SemanticFeatureRecord] = []
        for target in self.targets:
            probability = self._predict_probability(target, values)
            if probability < target.threshold:
                continue
            node_id = hashlib.sha256(
                f"{ticker}:{snapshot_time.isoformat()}:{target.feature_name}:{self.model_version}".encode("utf-8")
            ).hexdigest()[:16]
            features.append(
                SemanticFeatureRecord(
                    ticker=ticker,
                    as_of=snapshot_time,
                    feature_name=target.feature_name,
                    feature_category=target.feature_category,
                    state="active",
                    confidence=round(probability, 6),
                    supporting_indicators=target.input_indicators,
                    semantic_relation=target.semantic_relation,  # type: ignore[arg-type]
                    target_signal=target.target_signal,
                    ontology_node_id=f"semantic-ai:{node_id}",
                    generation_method="ai_model",
                    model_version=self.model_version,
                    metadata={
                        "model_type": "centroid_classifier",
                        "threshold": target.threshold,
                        "input_indicators": target.input_indicators,
                    },
                )
            )
        return tuple(features)

    def _predict_probability(self, target: AISemanticTarget, values: dict[str, float]) -> float:
        centroids = self.state.class_centroids.get(target.feature_name, {})
        positive = centroids.get(1)
        negative = centroids.get(0)
        if not positive and not negative:
            return 0.0
        point = {
            key: _standardize(
                values.get(key),
                self.state.feature_means.get(key),
                self.state.feature_stdevs.get(key),
            )
            for key in target.input_indicators
        }
        positive_distance = _distance(point, positive) if positive else 0.0
        negative_distance = _distance(point, negative) if negative else positive_distance + 1.0
        return 1 / (1 + math.exp(positive_distance - negative_distance))


class TextHeuristicSemanticModel:
    """Deterministic placeholder for unstructured text until an LLM/text model is connected."""

    model_version = "text-heuristic-semantic-v1"

    def predict_text(
        self,
        ticker: str,
        as_of: datetime,
        documents: tuple[str, ...],
    ) -> tuple[SemanticFeatureRecord, ...]:
        text = " ".join(documents).lower()
        outputs: list[SemanticFeatureRecord] = []
        rules = (
            ("MajorSupplyContract", ("contract", "supply", "order backlog"), "supportsSignal", "BuyCandidate"),
            ("AnalystUpgrade", ("upgrade", "raised target", "outperform"), "supportsSignal", "BuyCandidate"),
            ("RegulatoryPenaltyNegative", ("penalty", "fine", "sanction"), "increasesRiskOf", "ReduceRiskCandidate"),
            ("LitigationRiskHigh", ("lawsuit", "litigation", "class action"), "increasesRiskOf", "ReduceRiskCandidate"),
        )
        for feature_name, keywords, relation, signal in rules:
            hits = sum(1 for keyword in keywords if keyword in text)
            if hits == 0:
                continue
            node_id = hashlib.sha256(f"{ticker}:{as_of.isoformat()}:{feature_name}:text".encode("utf-8")).hexdigest()[:16]
            outputs.append(
                SemanticFeatureRecord(
                    ticker=ticker,
                    as_of=as_of,
                    feature_name=feature_name,
                    feature_category="unstructured_event",
                    state="active",
                    confidence=min(1.0, 0.45 + hits * 0.2),
                    supporting_indicators=tuple(keywords),
                    semantic_relation=relation,  # type: ignore[arg-type]
                    target_signal=signal,
                    ontology_node_id=f"semantic-text:{node_id}",
                    generation_method="ai_text_proxy",
                    model_version=self.model_version,
                    metadata={"document_count": len(documents), "matched_keywords": hits},
                )
            )
        return tuple(outputs)


def default_ai_semantic_targets() -> tuple[AISemanticTarget, ...]:
    return (
        AISemanticTarget(
            feature_name="AdaptiveBreakoutCandidate",
            feature_category="learned_pattern",
            semantic_relation="supportsSignal",
            target_signal="BuyCandidate",
            input_indicators=("return_5d", "volume_spike_ratio", "bollinger_band_width_20", "macd_histogram"),
            threshold=0.58,
        ),
        AISemanticTarget(
            feature_name="AdaptiveRiskOffCandidate",
            feature_category="learned_risk",
            semantic_relation="increasesRiskOf",
            target_signal="ReduceRiskCandidate",
            input_indicators=("historical_volatility_20d", "rolling_drawdown_20d", "return_5d", "macd_histogram"),
            threshold=0.58,
        ),
    )


def _indicator_values(indicators: tuple[RawIndicatorRecord, ...]) -> dict[str, float]:
    values = {}
    for indicator in indicators:
        if isinstance(indicator.value, (int, float)):
            values[indicator.indicator_name] = float(indicator.value)
    return values


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    avg = _mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) or 1.0


def _standardize(value: float | None, mean: float | None, stdev: float | None) -> float:
    if value is None:
        return 0.0
    return (value - (mean or 0.0)) / (stdev or 1.0)


def _distance(point: dict[str, float], centroid: dict[str, float] | None) -> float:
    if not centroid:
        return 0.0
    keys = set(point) | set(centroid)
    return math.sqrt(sum((point.get(key, 0.0) - centroid.get(key, 0.0)) ** 2 for key in keys))

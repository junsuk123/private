from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.features.ai_semantic_layer import AISemanticModel, TextHeuristicSemanticModel
from app.features.indicator_engine import IndicatorEngine
from app.features.parameter_tuning import FormulaParameterTuner
from app.features.schemas import FeatureSnapshot, OHLCVBar, SemanticFeatureRecord
from app.features.semantic_feature_engine import SemanticFeatureEngine
from app.graph.reasoning_rules import build_semantic_reasoning_paths


@dataclass(frozen=True)
class HybridSemanticPipelineConfig:
    enable_ai_numeric_layer: bool = True
    enable_text_layer: bool = True
    enable_ai_parameter_tuning: bool = True


class HybridSemanticFeaturePipeline:
    """Routes formula-calculable features to formulas and adaptive features to AI models."""

    def __init__(
        self,
        indicator_engine: IndicatorEngine | None = None,
        formula_semantic_engine: SemanticFeatureEngine | None = None,
        ai_numeric_model: AISemanticModel | None = None,
        parameter_tuner: FormulaParameterTuner | None = None,
        text_model: TextHeuristicSemanticModel | None = None,
        config: HybridSemanticPipelineConfig | None = None,
    ) -> None:
        self.indicator_engine = indicator_engine or IndicatorEngine()
        self.formula_semantic_engine = formula_semantic_engine or SemanticFeatureEngine()
        self.ai_numeric_model = ai_numeric_model
        self.parameter_tuner = parameter_tuner
        self.text_model = text_model or TextHeuristicSemanticModel()
        self.config = config or HybridSemanticPipelineConfig()

    def build_snapshot(
        self,
        bars: tuple[OHLCVBar, ...],
        as_of: datetime | None = None,
        documents: tuple[str, ...] = (),
    ) -> FeatureSnapshot:
        parameter_recommendations = ()
        indicator_engine = self.indicator_engine
        if self.config.enable_ai_parameter_tuning and self.parameter_tuner is not None:
            tuned_config, parameter_recommendations = self.parameter_tuner.recommend(
                bars,
                self.indicator_engine.config,
                as_of=as_of,
            )
            indicator_engine = IndicatorEngine(tuned_config)
        raw_indicators = indicator_engine.calculate(bars, as_of=as_of)
        if not raw_indicators:
            raise ValueError("Cannot build a feature snapshot without OHLCV bars.")
        formula_features = self.formula_semantic_engine.generate(raw_indicators)
        ai_features: tuple[SemanticFeatureRecord, ...] = ()
        if self.config.enable_ai_numeric_layer and self.ai_numeric_model is not None:
            ai_features = self.ai_numeric_model.predict(raw_indicators, as_of=raw_indicators[-1].as_of)
        text_features: tuple[SemanticFeatureRecord, ...] = ()
        if self.config.enable_text_layer and documents:
            text_features = self.text_model.predict_text(raw_indicators[-1].ticker, raw_indicators[-1].as_of, documents)
        semantic_features = _deduplicate_features(formula_features + ai_features + text_features)
        reasoning_paths = build_semantic_reasoning_paths(semantic_features)
        return FeatureSnapshot(
            ticker=raw_indicators[-1].ticker,
            as_of=raw_indicators[-1].as_of,
            raw_indicators=raw_indicators,
            semantic_features=semantic_features,
            reasoning_paths=reasoning_paths,
            parameter_recommendations=parameter_recommendations,
        )


def _deduplicate_features(features: tuple[SemanticFeatureRecord, ...]) -> tuple[SemanticFeatureRecord, ...]:
    by_key: dict[tuple[str, str, str], SemanticFeatureRecord] = {}
    for feature in features:
        key = (feature.ticker, feature.feature_name, feature.target_signal or "")
        existing = by_key.get(key)
        if existing is None or feature.confidence > existing.confidence:
            by_key[key] = feature
    return tuple(by_key.values())

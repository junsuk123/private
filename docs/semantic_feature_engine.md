# Semantic Feature Engine Developer Guide

This guide covers the semantic feature subsystem under `src/app/features`. It is an extensible layer for richer indicators, semantic states, graph triples, and model rows. The current web dashboard still uses the simpler `IndicatorSnapshot` decision path as its primary production path, while these modules support incremental expansion.

## Add a New Raw Indicator

1. Document the formula in `research_notes/`.
2. Add the machine-readable entry to `src/app/features/formula_catalog.json`.
3. Implement an independently testable function in `src/app/features/indicator_engine.py`.
4. Add a `RawIndicatorRecord` in `IndicatorEngine.calculate`.
5. Add unit tests in `tests/test_indicators.py`.

## Add a New Semantic Feature

1. Add the rule to `src/app/features/feature_registry.yaml`.
2. Add mapping logic to `SemanticFeatureEngine`.
3. Include confidence, supporting indicators, semantic relation, and target signal.
4. Add tests in `tests/test_semantic_features.py`.

## Formula vs AI Routing

Use deterministic formulas when the feature has a stable mathematical definition:

- Returns, SMA/EMA, MACD, RSI, Bollinger Bands, ATR, OBV, MFI, stochastic oscillator
- Accounting ratios such as margins, ROE, debt ratio, PER/PBR when source data is structured
- Portfolio weights, cash reserve, drawdown, VaR/CVaR after the risk method is specified

Use the AI semantic layer when the mapping requires tuning, context, or unstructured interpretation:

- Adaptive breakout/risk-off candidates where thresholds should be learned by regime
- Chart patterns such as cup-and-handle, flags, head-and-shoulders, and support/resistance quality
- News/disclosure event extraction, source credibility, sentiment clusters, rumor risk
- Macro/sector regime classification and contradiction scoring
- Graph features whose useful thresholds depend on historical outcomes

`HybridSemanticFeaturePipeline` combines both paths:

- `IndicatorEngine`: formula-only raw indicators
- `SemanticFeatureEngine`: transparent formula/rule semantic states
- `CentroidAISemanticModel` or a replacement model: tunable numeric semantic states
- `TextHeuristicSemanticModel` or a replacement LLM/text model: unstructured text semantic states

Every semantic feature records `generation_method` and `model_version`, so downstream ontology and AI training can distinguish formula truth from learned inference.

## LLM Event Classification

News, disclosures, and scraped text can be classified by a small LLM before they enter the ontology. The LLM is asked to return strict JSON:

- `sentiment`: `POSITIVE`, `NEGATIVE`, or `NEUTRAL`
- `summary`: concise factual summary
- `key_facts`: important extracted facts
- `event_labels`: labels such as `MajorSupplyContract`, `AnalystUpgrade`, `GuidanceLowered`, `RegulatoryPenaltyNegative`
- `companies`, `tickers`, `sectors`
- `confidence`

Runtime behavior:

- `JsonEventLLMClassifier` parses the model JSON.
- `classify_text_event` uses LLM output when a classifier is provided.
- If the LLM call fails or is not configured, keyword classification remains the fallback.
- `ClassifiedEvent` stores `key_facts`, `event_labels`, `classification_confidence`, and `classification_model`.
- `event_mapper` maps LLM labels and key facts into graph triples.

Local LLM is recommended for this simple classification task.

In-process local model, no local server required:

```text
pip install ".[local-llm]"
LLM_EVENT_CLASSIFIER_ENABLED=true
LLM_EVENT_PROVIDER=embedded
LLM_EVENT_MODEL=models/local-llm/event-classifier
LLM_EVENT_LOCAL_FILES_ONLY=true
```

For multimodal Hugging Face checkpoints that expose `AutoProcessor` and `AutoModelForMultimodalLM`, use:

```text
pip install ".[local-llm]"
LLM_EVENT_CLASSIFIER_ENABLED=true
LLM_EVENT_PROVIDER=multimodal
LLM_EVENT_MODEL=google/diffusiongemma-26B-A4B-it
LLM_EVENT_LOCAL_FILES_ONLY=true
```

Use `LLM_EVENT_MODEL` as either a local Hugging Face model directory or a model id. For live use, prefer a local directory and `LLM_EVENT_LOCAL_FILES_ONLY=true` so startup does not unexpectedly download model files. For Intel OpenVINO/NPU inference:

```text
pip install ".[openvino-llm]"
LLM_EVENT_CLASSIFIER_ENABLED=true
LLM_EVENT_PROVIDER=openvino-llm
LLM_EVENT_MODEL=models/local-llm/event-classifier
LLM_EVENT_DEVICE=NPU
```

Local server mode is also supported. For example, with Ollama:

```text
ollama pull qwen2.5:1.5b-instruct
ollama serve
```

Environment toggle:

```text
LLM_EVENT_CLASSIFIER_ENABLED=true
LLM_EVENT_PROVIDER=local
LLM_EVENT_MODEL=qwen2.5:1.5b-instruct
LLM_EVENT_LOCAL_ENDPOINT=http://127.0.0.1:11434/v1/chat/completions
```

The endpoint is OpenAI-compatible, but the code is provider-agnostic through the `LLMTextClient` protocol. Remote API mode is still available by setting `LLM_EVENT_PROVIDER=remote`, `LLM_EVENT_API_KEY`, and `LLM_EVENT_ENDPOINT`. Keep the prompt factual; ambiguous text should become `NEUTRAL` with low confidence.

## AI-Tuned Formula Parameters

Formula indicators keep their mathematical definitions, but their parameters can be selected by an AI tuner before calculation. The current implementation uses `RegimeFormulaParameterTuner`.

Examples:

- RSI formula remains Wilder RSI, but `rsi_period` can be tuned.
- Bollinger Bands remain `SMA +/- k * stdev`, but `bollinger_period` and `bollinger_stddev` can be tuned.
- ATR remains Wilder ATR, but `atr_period` can be tuned.
- Volume spike remains `current volume / moving average volume`, but `volume_window` can be tuned.

The pipeline order is:

1. Build as-of parameter context from historical bars only.
2. AI tuner recommends formula parameters.
3. Indicator engine calculates deterministic formulas using those parameters.
4. Raw indicator metadata stores the actual parameters.
5. Snapshot stores `parameter_recommendations` with tuner version, confidence, and reason.

This keeps the equation auditable while allowing parameters to adapt to volatility, trend, volume, and later learned market-regime models.

## No-Lookahead Rule

Feature generation must only use bars with `bar.as_of <= decision_time`. Future-return and triple-barrier labels live in `src/app/models/labeling.py` and must not be used by feature engines.

## Ontology Output

Use `build_semantic_feature_graph(raw_indicators, semantic_features)` to convert feature records into graph triples:

- `ticker generatesSemanticFeature semantic_node`
- `semantic_node supportsSignal target_signal`
- ticker-level support/risk/contradiction triples for existing reasoners

## Trading Safety

These modules produce analysis, graph, and dataset records only. They do not generate executable orders and do not enable live trading.

Any future semantic-feature-derived trading intent must still flow through:

```text
semantic feature
  -> ontology evidence
  -> StrategySignal
  -> OrderIntent
  -> RiskManager
  -> mock/paper FinalOrder only
```

Live automated execution remains disabled.

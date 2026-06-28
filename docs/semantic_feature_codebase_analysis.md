# Codebase Analysis: Semantic Feature Integration

The semantic feature layer is present as an extensible analysis/modeling layer. The current main web decision path still uses `IndicatorSnapshot` from `src/app/indicators/engine.py` plus lightweight ontology candidate filtering, ontology reasoning, goal feasibility, deterministic strategies, and risk validation.

The repository-level diagram `ontology base trading system diagram.png` summarizes where this layer fits: feature engineering feeds candidate filtering, evidence scoring, ontology reasoning, strategy construction, and finally the deterministic risk gate.

## Existing Integration Points

- `src/app/data/`: read-only public/sample collectors and raw archive helpers.
- `src/app/research/`: public research orchestration, retries, diagnostics, and optional LLM event classification.
- `src/app/storage/`: realtime-only local SQLite research store and model artifact store.
- `src/app/indicators/`: existing lightweight indicator support for the current pipeline.
- `src/app/graph/`: ontology constants, in-memory knowledge graph, event mapper, graph builder, and rule reasoner.
- `src/app/trading_pipeline.py`: first-pass ontology candidate screening over the broad universe before heavier analysis.
- `src/app/goals/`: target feasibility scoring and compromise goal generation.
- `src/app/strategy/`: rule-based and goal-directed signal generation from indicators and graph relations.
- `src/app/risk/`: deterministic risk manager that gates every `OrderIntent` before any final order.
- `src/app/execution/`: paper/mock broker boundary; live KIS client remains disabled by default.
- `src/app/trading/`: mock program cycle using LLM-style judgment, ontology evidence, risk validation, and mock execution.
- `src/app/backtesting/streaming_demo.py`: in-memory stepwise simulation with target return and target minutes.
- `src/app/realtime/learning.py`: realtime supervised example creation and hypothetical testing with zero submitted orders.
- `src/app/web.py`: web workflow for operation modes, goal negotiation, paper trading, streaming simulation, diagnostics, and ontology graph output.

## Added Backward-Compatible Layer

- `src/app/features/indicator_engine.py`: independently testable OHLCV indicator calculations.
- `src/app/features/semantic_feature_engine.py`: transparent raw-indicator to semantic-state mapping.
- `src/app/features/schemas.py`: raw indicator, semantic feature, reasoning path, and snapshot records.
- `src/app/graph/semantic_builder.py`: maps semantic feature records into ontology-ready graph triples.
- `src/app/graph/reasoning_rules.py`: produces contradiction-aware reasoning path records.
- `src/app/models/dataset_builder.py`: builds as-of feature rows and keeps future labels separate.
- `src/app/models/labeling.py`: future-return and triple-barrier labels.

## Safety Notes

Live trading remains disabled by default. The new modules generate analysis records, graph triples, reasoning paths, and model rows only. They do not call brokerage APIs or create executable orders.

Realtime testing remains hypothetical and reports `orders_submitted = 0`. Streaming simulation remains bounded to in-memory mock state and updates simulated cash, holdings, trades, progress, and return rate only.

## Current Implementation Scope

Implemented first feature group:

- Price and return features
- Trend features
- Momentum features
- Volatility features
- Volume and flow features
- Leakage-safe dataset and labeling scaffolding

Fundamental, disclosure, macro, sentiment, sector/theme, order-book, and advanced graph features should be added incrementally using the same registry/catalog pattern.

## Current Web Path Relationship

The web app currently combines both older and newer layers as follows:

1. Research sources and the realtime local store create the research base.
2. `ontology_filter_1` screens the broad universe and limits heavier downstream work to candidates plus priority tickers.
3. `build_analysis_context` creates `IndicatorSnapshot` values for the primary dashboard.
4. The ontology graph and reasoner convert indicators/events/temporal frames/candidate-selection metadata into explainable support, contradiction, and risk relationships.
5. Goal feasibility uses target return and period to score achievability.
6. Goal-directed strategy uses the selected target to produce target-aware intents.
7. RiskManager gates every intent.
8. Mock KIS, realtime hypothetical testing, and streaming simulation use only approved or hypothetical non-live execution paths.

The semantic feature modules are ready to enrich steps 2 and 3, but they are not yet the only source of dashboard decisions.

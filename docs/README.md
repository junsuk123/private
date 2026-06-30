# Documentation Index

![End-to-end ontology trading system flow](ontology%20base%20trading%20system%20diagram.png)

The diagram above is the canonical repository-level overview: trusted market and broker data enter validation/storage, become quantitative and semantic features, pass candidate filtering and evidence scoring, then flow through ontology reasoning, strategy construction, deterministic risk validation, controlled paper/live-readiness execution, and post-trade feedback.

## Core Architecture

- `architecture.md`: runtime modules, API surfaces, operation modes, and deterministic safety boundaries.
- `system_algorithm_analysis.md`: algorithm-by-algorithm implementation map under `src/app`.
- `data_environment_separation.md`: realtime-only data layout and synthetic-data rejection rules.
- `realtime_short_horizon_policy.md`: low-latency realtime learning, paper-trading, and readiness behavior.

## Strategy And Feature Design

- `short_term_trading_strategy_design.md`: short-horizon strategy research mode and gatekeeping.
- `semantic_feature_engine.md`: semantic feature subsystem, indicator routing, LLM classification, and no-lookahead rules.
- `semantic_feature_codebase_analysis.md`: codebase-level analysis of semantic feature integration.
- `current_short_term_trading_audit.md`: historical audit notes from the short-term strategy review.
- `live_short_horizon_model_decision.md`: decision record (2026-07-01) — buys run via the ontology path; the live ML model stays advisory (AUC≈0.29, not predictive) and is not force-promoted.

## Acceleration And Native Hot Paths

- `npu_runtime_architecture.md`: CPU/NPU split, environment controls, and fallback behavior.
- `npu_optimization_audit.md`: vectorized screening, Rust/PyO3 native core, rolling cache, and trusted-indicator policy.
- `npu_benchmark_results.md`, `npu_benchmark_results_npu.md`, `npu_realtime_benchmark_results.md`, `npu_realtime_benchmark_results_npu.md`: benchmark result snapshots.

## Runtime Defaults

The web server now starts realtime collection/learning automatically and launches a read-only KIS live-readiness account check on startup. The UI keeps only the user-facing goal and trading controls; account basis, simulation initial cash, and profit-gain scaling are computed automatically from the live account/readiness state and target settings.

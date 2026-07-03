# Documentation Index

## Current Runtime Contract

As of the current `run.ps1` entry point, the system is a guarded KIS live-capable realtime runtime. KIS realtime collection, read-only account probing, periodic live short-horizon training, and the independent realtime trading loop can start automatically. Numeric ontology/candidate evidence scoring requests OpenVINO `NPU` and falls back to CPU when unavailable; final action selection, graph explanations, risk checks, order gating, idempotency, and broker submission remain deterministic CPU-controlled paths. NPU output is evidence, not trade authorization.

![End-to-end ontology trading system flow](ontology%20base%20trading%20system%20diagram.png)

The diagram above is the canonical repository-level overview: trusted market and broker data enter validation/storage, become quantitative and semantic features, pass candidate filtering and evidence scoring, then flow through ontology reasoning, strategy construction, deterministic risk validation, controlled paper/live execution, and post-trade feedback.

## Current Runtime Summary

The current `run.ps1` runtime is a KIS live-capable realtime trading system. It opens `/account`, starts KIS realtime collection, starts periodic short-horizon model retraining, starts the independent realtime trading loop, and keeps deterministic execution gates mandatory.

The current acceleration boundary is explicit: OpenVINO/NPU is used for compatible numeric ontology/candidate evidence scoring when available, and CPU fallback is automatic. On the verified local environment, OpenVINO exposes `CPU`, `GPU`, and `NPU`, and the ontology runtime selects `NPU` under `run.ps1`. Trading decisions, risk checks, final order creation, and broker submission remain CPU-controlled deterministic paths.

The live loop is conservative:

- SELL/REDUCE is evaluated before BUY.
- Existing open SELL orders are kept unless an amend is materially useful.
- BUY is skipped when `REALTIME_BUY_ENABLED=false`.
- The live ML model is advisory only when `REALTIME_MODEL_AUXILIARY_ONLY=true`.
- BUY requires cash, quote freshness, acceptable spread/liquidity, ontology/runtime support, and final risk approval.
- The `/account` termination button disables BUY, submits profit-seeking liquidation SELL orders when live gates pass, then schedules server shutdown.

## Core Architecture

- `architecture.md`: runtime modules, API surfaces, operation modes, and deterministic safety boundaries.
- `live_trading_runbook.md`: current live operating procedure, status checks, termination behavior, and stall diagnosis.
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

The web server now starts KIS realtime collection, account refresh, live training, and the realtime trading engine automatically. The account dashboard is the primary control surface for holdings, cash, asset history, decision flow, rejection reasons, and program termination.

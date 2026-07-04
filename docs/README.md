# Documentation Index

## Current Runtime Contract

As of the current `run.ps1` entry point, the system is a guarded KIS live-capable realtime runtime. KIS realtime collection, read-only account probing, periodic live short-horizon training, and the independent realtime trading loop can start automatically. Numeric ontology/candidate evidence scoring requests OpenVINO `NPU` and falls back to CPU when unavailable; final action selection, graph explanations, risk checks, order gating, idempotency, and broker submission remain deterministic CPU-controlled paths. NPU output is evidence, not trade authorization.

![End-to-end ontology trading system flow](diagrams/system_overview.svg)

The diagram above is the canonical repository-level overview and is an editable SVG (`diagrams/system_overview.svg`): trusted market and broker data enter validation/storage, become quantitative and semantic features, pass candidate filtering and evidence scoring, then flow through ontology reasoning, strategy construction (SELL before BUY), deterministic risk validation, guarded paper/live execution, and post-trade feedback.

## Current Runtime Summary

The `run.ps1` runtime is a KIS live-capable realtime trading system. It opens `/account`, starts KIS realtime collection, starts periodic short-horizon model retraining, starts the independent realtime trading loop, and keeps deterministic execution gates mandatory. The live loop is conservative:

- SELL/REDUCE is evaluated before BUY.
- Existing open SELL orders are kept unless an amend is materially useful (`open_sell_kept`).
- BUY is skipped when `REALTIME_BUY_ENABLED=false`.
- The live ML model is advisory only when `REALTIME_MODEL_AUXILIARY_ONLY=true`.
- BUY requires cash, quote freshness, acceptable spread/liquidity, ontology/runtime support, and final risk approval.
- Small-account mode is on by default (blocks 1-share loss churn and below-break-even non-emergency SELLs, caps per-position weight).
- The `/account` termination button disables BUY, submits profit-seeking liquidation SELL orders when live gates pass, then schedules server shutdown.

OpenVINO/NPU is used only for compatible numeric ontology/candidate evidence scoring; CPU fallback is automatic. On the verified local environment OpenVINO exposes `CPU`, `GPU`, and `NPU` and the ontology runtime selects `NPU` under `run.ps1`. Decisions, risk checks, final order creation, and broker submission stay on CPU-controlled deterministic paths.

## Core Architecture

- `architecture.md`: runtime modules, API surfaces, operation modes, and deterministic safety boundaries.
- `system_algorithm_analysis.md`: algorithm-by-algorithm implementation map under `src/app`.
- `data_environment_separation.md`: realtime-only data layout and synthetic-data rejection rules.
- `realtime_short_horizon_policy.md`: low-latency realtime learning, paper-trading, and readiness behavior.

## Live Trading

- `live_trading_setup.md`: one-time install, local secrets/config, readiness dry-runs, and arming.
- `live_trading_runbook.md`: day-to-day operating procedure, dashboards, stall diagnosis, termination, and emergency stop.
- `live_trading_safety_gates.md`: the mandatory submission, BUY, and SELL/REDUCE gates and their rejection codes.
- `small_account_loss_sell_fix_report.md`: small-account loss-churn guards and dashboard cash decomposition.
- `live_short_horizon_model_decision.md`: decision record (2026-07-02) — the live ML model stays advisory and cannot approve a model-only BUY.

## Strategy And Feature Design

- `short_term_trading_strategy_design.md`: short-horizon strategy research mode and gatekeeping.
- `semantic_feature_engine.md`: semantic feature subsystem, indicator routing, LLM classification, and no-lookahead rules.
- `semantic_feature_codebase_analysis.md`: codebase-level analysis of semantic feature integration.
- `theory_aware_ontology_voting.md`: converting ontology triples into theory votes before final action selection.

## Standards-Based Ontology (RDF/RDFS/OWL + SHACL)

- `ontology_standardization_report.md`: design of the additive RDF/OWL/SHACL layer and its guardrails.
- `ontology_migration_audit.md`: old custom triple store → standards-based ontology mapping.
- Diagrams: `diagrams/ontology_framework.svg`, `diagrams/ontology_layered_architecture.svg`, `diagrams/ontology_reasoning_boundary.svg`, `diagrams/ontology_standardization_components.svg`, `diagrams/ontology_migration_beforeafter.svg`.

## Acceleration And Native Hot Paths

- `npu_runtime_architecture.md`: CPU/NPU split, environment controls, and fallback behavior.
- `npu_optimization_audit.md`: vectorized screening, Rust/PyO3 native core, rolling cache, and trusted-indicator policy.
- `npu_benchmark_results.md`: CPU vs NPU scoring and realtime-pipeline benchmark snapshots.

## Deployment

- `raspberry_pi_deployment.md`: one-command CPU-only (NPU-free) Raspberry Pi build and headless run.

# Ontology-Gated Strategy-Owned Refactor

Status: P0-P8 implementation and shadow paths are present. The production legacy path is unchanged and all new execution/routing flags remain disabled by default pending historical-data acceptance.

## Current architecture from executable code

The runtime entrypoint is `python run.py`, which delegates to `app.run:main` and starts `app.web:app`. The FastAPI module also owns background research, training, data refresh, dashboard, account reconciliation, and live-cycle orchestration. `scripts/start_live_trading_loop.py` is intentionally fail-closed and does not start order execution.

Current live decision flow:

```text
KIS domestic WebSocket ─┐
KIS overseas REST poll ─┴─> RealtimeMarketDataStore (SQLite)
                                 │
stored windows + news/macro ─> feature/technical builders
                                 │
ontology + theory votes + model prediction
                                 │
legacy OrderIntent ─> RiskManager ─> FinalOrder
                                 │
LiveExecutionCoordinator ─> KIS REST order API
                                 │
status polling + balance reads ─> web/engine reconciliation
```

There are two guarded order call sites: `RealtimeTradingEngine` and the web live execution cycle. Both use `LiveExecutionCoordinator`, which performs runtime and KIS health gates and provides post-acknowledgement idempotency. This is a useful broker boundary, but `FinalOrder` does not carry a durable strategy-instance owner or mandatory causal risk verdict.

Domestic trades and order books have a KIS WebSocket collector. The collector parses frames correctly but persists to SQLite on the message-consumption path. U.S. held symbols are refreshed by a daemon REST poll every 12 seconds by default, and therefore do not meet the target fast-path invariant.

The existing ontology stack includes RDF/OWL/SHACL, macro/micro reasoners, theory registries, and voting. Its output still participates in directional aggregation. It is not yet a distinct time-valid closed-world strategy admissibility snapshot.

The current NPU code accelerates fixed feature scoring and provides CPU fallback telemetry. It is not a temporal relational GNN and current benchmarks do not establish stock-strategy ranking or NoTrade parity.

## Confirmed modes and safety boundaries

- `OperationModeManager` exposes learning, live readiness/test, and live trading. Legacy paper mode names normalize to live trading internally.
- Actual KIS order submission additionally requires `LIVE_TRADING_ENABLED`, `KIS_LIVE_ENABLED`, `LIVE_ORDER_SUBMIT_ENABLED`, configured live safety policy, KIS health, and optional manual arming.
- `.env.example` is fail-closed.
- Unit/integration tests use mocks; no live-order test was run during this audit.
- The refactor flags in `app.config.refactor_flags` preserve `legacy_vote_path=true` and disable all new routing/execution behavior.

## Target two-speed flow mapped to this repository

```text
FAST PATH
KIS WS provider
  -> normalized versioned event
  -> in-memory market state + incremental bar/features
  -> active StrategyInstance only
  -> OrderIntent
  -> RiskVerdict
  -> execution policy
  -> KIS gateway
  -> order/fill event
  -> owning Position/StrategyInstance

SLOW INTELLIGENCE PATH
point-in-time FeatureSnapshot
  -> validated operational OntologyDecision
  -> fixed-shape graph/tensor projection
  -> CPU temporal R-GCN shadow (NPU only after promotion evidence)
  -> StrategyUtilityEvidence
  -> NoTrade/strategy router
  -> new TradePlan only for an unowned symbol

BACKGROUND
WebSocket bootstrap/replay, counterfactual simulation, training/calibration,
model registry, audit compaction, and dashboard read models
```

Existing modules will be adapted rather than moved gratuitously. Versioned P1 contracts are in `app.trading.contracts`; durable causal records are in `app.execution.causal_journal`; ownership enforcement is in `app.strategy.ownership`.

## Highest-risk gaps and bottlenecks

1. Strategy ownership is not durable. A later signal can influence an existing holding without proving it is the origin strategy instance.
2. U.S. data uses REST polling in a latency-sensitive loop. Domestic WebSocket callbacks also perform synchronous persistence.
3. Idempotency is recorded after broker acknowledgement. A process/network failure between submission and recording is ambiguous.
4. Ontology open-world knowledge, validation, voting, and operational trading gates are not separated into a versioned closed-world snapshot.
5. `app.web` combines UI, orchestration, data collection, account access, and live execution, making latency isolation and restart recovery difficult.
6. Features are often rebuilt from stored windows; there is no single event-time-ordered incremental market state with explicit gap uncertainty.
7. Current ML/NPU outputs are signal/vote scorers, not calibrated net utility for each stock-strategy pair.
8. JSONL decision and feature logs are hundreds of MB and do not have a documented retention/compaction policy.

## Phased edit plan using real paths

- P1: integrate `app.trading.contracts`, `app.execution.causal_journal`, and `app.strategy.ownership` behind refactor flags; require intent and risk records before any new-path side effect.
- P2: adapt `app.data.kis_realtime` to publish normalized events into bounded queues; add in-memory state and incremental bars before persistence. Keep `app.trading.us_realtime_bridge` as fallback only.
- P3: wrap engines in `app.strategy.short_horizon` as lifecycle experts; route exits through the durable owner and adapt `app.risk.manager` to `RiskVerdict`.
- P4: materialize the operational snapshot from the existing `app.ontology` and `app.graph` assets with TTL, required-field validation, deterministic gates, and explanations.
- P5: extend `app.technical.replay`, `app.backtesting`, and `app.evaluation.walk_forward` into a causal event simulator with counterfactual strategy labels and realistic costs/fills.
- P6/P7: add fixed-shape temporal relational utility inference alongside `app.models.inference_backend`; promote CPU shadow first and NPU only after golden parity and end-to-end benchmarks.
- P8: compare legacy, ontology-only, CPU-GNN, and NPU-GNN decisions under explicit shadow/paper/canary flags before deprecating legacy voting.

No blocker prevents P1/P2 engineering. Live/canary promotion will later require explicit account-mode authority and acceptance evidence.

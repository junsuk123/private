# Completion Audit

## Proven implementation

| Requirement | Evidence | Status |
|---|---|---|
| Code-based current-state audit | `current_system_inventory.json`, `architecture.md` | complete |
| Typed immutable contracts | `app.trading.contracts`, contract tests | complete |
| Persist intent before side effect and at-most-once key | fsynced `CausalOrderJournal`, atomic idempotency reservation, mock E2E | complete for new path |
| WebSocket fast path without DB/model blocking | event sink, bounded bus, in-memory state, separate persistence/slow workers | complete behind flag |
| Incremental bars and microstructure | `app.data.event_pipeline` and formula/replay tests | complete |
| Sequence/freshness/reconnect handling | duplicate/out-of-order/gap/reconnect tests | complete |
| Seven independent strategy experts | `app.strategy.experts` | complete |
| Durable strategy ownership and restart | lifecycle migrations, orchestrator restart test | complete |
| Closed-world ontology gate with TTL | `app.ontology.operational_gate` golden tests | complete |
| NoTrade and utility router | `app.routing.strategy_router` | complete |
| Cost/fill counterfactual simulator | `app.backtesting.event_simulator` | complete |
| Stored-data causal labels and reproducibility hashes | `app.evaluation.stored_counterfactual`, JSON report | complete; promotion rejected |
| Purging and embargo | `app.evaluation.purged_walk_forward` | complete |
| Fixed-shape relational temporal utility model | `app.models.strategy_utility` | complete as shadow architecture |
| CPU/OpenVINO parity | golden CPU test | complete |
| Actual NPU compile and benchmark | `npu_benchmark.md`, JSON report | complete; promotion rejected |
| Legacy/ontology/CPU/NPU comparison | shadow service and recorder | complete behind flag |
| Research/replay/paper/shadow/canary/live boundaries | validated refactor profile | complete |
| No live-order testing | all broker E2E tests use mocks | complete |

## Deliberately not promoted

- The temporal relational model has deterministic validation weights, not a trained production checkpoint.
- The available store produced 34,860 causal labels, but covers only 11 UTC
  dates, is concentrated in US equities, and lacks point-in-time event, sector,
  session-calendar, and legacy-decision data. Net performance superiority,
  calibration, PBO, and Deflated Sharpe therefore remain unproven.
- NPU inference compiled on the real NPU but was slower than CPU and exceeded the strict utility parity tolerance.
- `REFACTOR_LIVE_ENABLED`, strategy-owned execution, ontology routing, GNN reranking, and NPU inference remain disabled in the example profile.
- The legacy production path is retained, as required, until shadow/paper acceptance evidence exists. The new path itself satisfies the causal chain; legacy retirement is not claimed.

These are promotion outcomes, not hidden implementation claims. Enabling canary/live without new historical and paper evidence violates the runbook.

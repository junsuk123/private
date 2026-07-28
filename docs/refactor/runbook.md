# Refactor Runbook

## Safe verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_refactor_contracts.py tests\test_causal_order_journal.py tests\test_refactor_flags.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\benchmark_strategy_utility_openvino.py --iterations 30
```

No command in this phase submits a live order. Do not set any live flag for contract testing.

## Feature flags

The defaults preserve current behavior:

```text
REFACTOR_LEGACY_VOTE_PATH=true
REFACTOR_WEBSOCKET_MARKET_DATA=false
REFACTOR_LOCAL_CHART_ENGINE=false
REFACTOR_ONTOLOGY_ROUTER=false
REFACTOR_GNN_SHADOW=false
REFACTOR_GNN_RERANK=false
REFACTOR_NPU_INFERENCE=false
REFACTOR_STRATEGY_OWNED_EXECUTION=false
REFACTOR_LIVE_ENABLED=false
```

Invalid combinations fail during `RefactorFeatureFlags` validation. In particular, refactor live mode requires strategy-owned execution; GNN reranking requires the ontology router; and NPU inference requires a GNN shadow or rerank mode.

## Rollback

The P1 code is additive. Leave every `REFACTOR_*` flag at its default and the existing legacy path remains active. The new causal journal defaults to `data/store/causal-order-journal.jsonl` and is not created unless the new journal is instantiated.

## Promotion state

- Research/replay: contracts, event simulator, and purged walk-forward available.
- Shadow: implementation available but disabled by default.
- Paper: disabled.
- Canary: disabled.
- Live: disabled.

The NPU promotion gate failed because CPU inference was faster. Historical-data model/strategy promotion also remains blocked by missing representative point-in-time replay data. Do not enable `REFACTOR_GNN_RERANK`, `REFACTOR_NPU_INFERENCE`, or `REFACTOR_LIVE_ENABLED`.

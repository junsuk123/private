# NPU Optimization Audit

## Current NPU Usage

- `src/app/graph/npu_classifier.py` owns OpenVINO/NPU candidate evidence scoring with deterministic CPU NumPy fallback.
- `src/app/trading_pipeline.py` performs CPU hard filtering first, then optional NPU top-k scoring controlled by `ONTOLOGY_NPU_ENABLED`.
- `src/app/graph/builders.py` consumes NPU scores only as ontology evidence triples.
- `src/app/realtime/short_horizon_npu_predictor.py` adds an opt-in short-horizon evidence predictor with a CPU linear baseline.
- `src/app/data/event_classifier*.py` adds a lightweight event classifier interface with keyword fallback.

## CPU-Only Decision Boundaries

- `src/app/risk/manager.py` remains the mandatory final gate for every `OrderIntent`.
- `src/app/strategy/rule_based.py` still creates intents from deterministic policy and ontology evidence.
- `src/app/execution/*` remains separate from NPU scoring.
- Manual approval stays on `FinalOrder.manual_approval_required=True`.

## NPU-Suitable Modules

- Numeric candidate ranking over `[N, F]` float32 feature matrices.
- Lightweight event score inference when a local OpenVINO model exists.
- Short-horizon expected-return evidence when explicitly enabled.

## Bottleneck Candidates

- Full-universe score dictionaries are replaced by top-k materialization in the default scorer path.
- Graph materialization is scoped with `ONTOLOGY_GRAPH_SCOPE`, `ONTOLOGY_GRAPH_MAX_TICKERS`, `ONTOLOGY_GRAPH_MAX_EVENTS_PER_TICKER`, and `ONTOLOGY_GRAPH_EVENT_TTL_HOURS`.
- Realtime candidate metrics now record before/after counts and scorer timing.

## Safety Check

NPU output is treated as evidence only. It does not submit orders, bypass `RiskManager`, disable manual approval, or replace deterministic risk checks.

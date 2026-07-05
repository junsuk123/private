# NPU Runtime Architecture

## Current Runtime Contract

As of the current `run.ps1` entry point, the system is a guarded KIS live-capable realtime runtime. KIS realtime collection, read-only account probing, periodic live short-horizon training, and the independent realtime trading loop can start automatically. Numeric ontology/candidate evidence scoring requests OpenVINO `NPU` and falls back to CPU when unavailable; final action selection, graph explanations, risk checks, order gating, idempotency, and broker submission remain deterministic CPU-controlled paths. NPU output is evidence, not trade authorization.

The repository-level flow diagram is `docs/diagrams/system_overview.png`. In that diagram, this document mainly covers stages 4 and 5: candidate universe filtering and evidence scoring. Risk validation and execution remain CPU/deterministic even when scoring is accelerated.

Current local verification shows the active Python/OpenVINO environment can see `CPU`, `GPU`, and `NPU`, and the ontology runtime reports `active_backend=NPU` with `uses_npu=true` when `OPENVINO_DEVICE=NPU` and `ONTOLOGY_ACCELERATOR=NPU` are set by `run.ps1`.

## CPU/NPU Split

| Stage | Device | Notes |
|---|---|---|
| Data collection | CPU | Broker, news, disclosure, macro, and storage paths. |
| Hard filter | CPU | Trading halt, management stock, liquidity, invalid data, and deterministic rejects. |
| Candidate scoring | NPU with CPU fallback | `OntologyNpuLinearScorer` builds a small OpenVINO matmul graph and compiles it to `OPENVINO_DEVICE`, normally `NPU`; top-k ranking output remains evidence. |
| Realtime candidate ranking | NPU with CPU fallback | `trading_pipeline._rank_accepted_with_npu` scores accepted lightweight snapshots before heavier reasoning. |
| Event classification | CPU/keyword by default; optional OpenVINO status path | The current OpenVINO event-classifier wrapper still falls back to keyword semantics for labels; do not treat its `provider=openvino` status as a guaranteed LLM-on-NPU path. |
| Short-horizon prediction | CPU/live model by default; optional NPU helper modules | The live model remains auxiliary. OpenVINO/NPU helper classes exist for exported models or batch linear scorers, but final decisions stay CPU-gated. |
| Ontology graph reasoning | CPU | Explanation and reasoning trace construction. |
| Strategy decision | CPU | Converts evidence into candidate intents. |
| RiskManager | CPU | Mandatory final validation for all trade intents. |
| Execution | CPU | Only approved `FinalOrder` objects submitted through `LiveExecutionCoordinator` reach broker adapters. |

## Environment Controls

- `OPENVINO_DEVICE`: requested OpenVINO device.
- `ONTOLOGY_ACCELERATOR`: requested ontology runtime, usually `NPU` in `run.ps1`.
- `ONTOLOGY_NPU_ENABLED`: enables candidate scoring evidence path, default `true`.
- `ONTOLOGY_NPU_BATCH_SIZE`: `auto` or `512/1024/2048/4096`.
- `ONTOLOGY_NPU_TOP_K`: max candidate count passed to graph reasoning, default `50`.
- `NPU_DEVICE_PREFERENCE`: requested device for shared `NpuRuntimeManager` modules, default `NPU`.
- `NPU_MIN_BATCH_FOR_NPU`: minimum batch size before shared NPU modules request NPU instead of CPU NumPy.
- `EVENT_CLASSIFIER_PROVIDER`: `keyword`, `openvino`, or `llm`; default `keyword`.
- `EVENT_CLASSIFIER_DEVICE`: `AUTO`, `NPU`, or `CPU`.
- `SHORT_HORIZON_PREDICTOR_ENABLED`: enables the optional short-horizon evidence provider. The current live runtime can also use the live-trained model path when the latest artifact is eligible.
- `SHORT_HORIZON_PREDICTOR_DEVICE`: `AUTO`, `NPU`, or `CPU`.
- `ONTOLOGY_GRAPH_SCOPE`: `candidate_only`, `candidate_and_holdings`, or `full_debug`.

## Fallback Behavior

If OpenVINO or an NPU is unavailable, ontology scoring falls back to NumPy or OpenVINO CPU scoring with the same output schema. Missing event and short-horizon model files fall back to deterministic keyword and linear baselines.

The runtime reports fallback explicitly:

- `/api/ontology/runtime`: requested backend, active backend, available OpenVINO devices, and fallback reason.
- `/api/realtime/runtime`: acceleration summary plus ontology NPU status and live-model status.
- `/api/npu/runtime`: shared NPU module status for candidate, theory-vote, conflict, short-horizon, and execution-edge scorers.

Windows performance counters may not expose a separate NPU utilization engine for this workload. Prefer OpenVINO device discovery and the application runtime status over `GPU Engine(*)` counters when deciding whether the app selected NPU.

## Benchmarks

Run:

```powershell
python scripts/benchmark_npu_scoring.py --device CPU
python scripts/benchmark_realtime_pipeline.py --device CPU
```

Use `--device NPU` on machines with OpenVINO NPU support.

## Expanded Theory-Aware NPU Modules

The NPU boundary is intentionally numerical. OpenVINO/NPU may accelerate batch
matrix work, but CPU remains authoritative for symbolic graph reasoning, final
action selection, risk validation, and broker execution.

| Stage | Device | Notes |
| --- | --- | --- |
| Candidate evidence scoring | NPU with CPU fallback | Existing ontology candidate scorer. |
| Evidence cluster compression | NPU with CPU fallback | Compresses correlated indicators before voting. |
| Theory vote scoring | NPU with CPU fallback | Produces BUY/SELL/HOLD/REDUCE/WATCH vote vectors. |
| Conflict penalty scoring | NPU with CPU fallback | Dense numeric penalties; labels stay on CPU. |
| Short-horizon prediction | NPU with CPU fallback | Predicts short returns, net-positive probability, and uncertainty. |
| Execution edge scoring | NPU with CPU fallback | Estimates fill/slippage/adverse-selection edge in batch. |
| Graph traversal and explanations | CPU | Branch-heavy and explainability-critical. |
| Final action decision | CPU | Applies margin, position rules, and non-order HOLD/WATCH handling. |
| Broker execution | CPU | Deterministic safety-critical control. |

Runtime status is available at `/api/npu/runtime`.

Benchmark commands:

```bash
python scripts/benchmark_npu_theory_voting.py --device CPU
python scripts/benchmark_npu_theory_voting.py --device NPU
python scripts/benchmark_npu_full_decision_pipeline.py --device CPU
python scripts/benchmark_npu_full_decision_pipeline.py --device NPU
```

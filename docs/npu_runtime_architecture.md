# NPU Runtime Architecture

The repository-level flow diagram is `ontology base trading system diagram.png`. In that diagram, this document mainly covers stages 4 and 5: candidate universe filtering and evidence scoring. Risk validation and execution remain CPU/deterministic even when scoring is accelerated.

## CPU/NPU Split

| Stage | Device | Notes |
|---|---|---|
| Data collection | CPU | Broker, news, disclosure, macro, and storage paths. |
| Hard filter | CPU | Trading halt, management stock, liquidity, invalid data, and deterministic rejects. |
| Candidate scoring | NPU with CPU fallback | Vectorized float32 evidence scoring and top-k ranking. |
| Event classification | NPU/CPU fallback | Lightweight labels and sentiment evidence; keyword fallback by default. |
| Short-horizon prediction | NPU/CPU fallback | Optional evidence provider, disabled by default. |
| Ontology graph reasoning | CPU | Explanation and reasoning trace construction. |
| Strategy decision | CPU | Converts evidence into candidate intents. |
| RiskManager | CPU | Mandatory final validation for all trade intents. |
| Execution | CPU | Only approved/manual orders reach broker adapters. |

## Environment Controls

- `OPENVINO_DEVICE`: requested OpenVINO device.
- `ONTOLOGY_NPU_ENABLED`: enables candidate scoring evidence path, default `true`.
- `ONTOLOGY_NPU_BATCH_SIZE`: `auto` or `512/1024/2048/4096`.
- `ONTOLOGY_NPU_TOP_K`: max candidate count passed to graph reasoning, default `50`.
- `EVENT_CLASSIFIER_PROVIDER`: `keyword`, `openvino`, or `llm`; default `keyword`.
- `EVENT_CLASSIFIER_DEVICE`: `AUTO`, `NPU`, or `CPU`.
- `SHORT_HORIZON_PREDICTOR_ENABLED`: default `false`.
- `SHORT_HORIZON_PREDICTOR_DEVICE`: `AUTO`, `NPU`, or `CPU`.
- `ONTOLOGY_GRAPH_SCOPE`: `candidate_only`, `candidate_and_holdings`, or `full_debug`.

## Fallback Behavior

If OpenVINO or an NPU is unavailable, ontology scoring falls back to NumPy CPU scoring with the same output schema. Missing event and short-horizon model files fall back to deterministic keyword and linear baselines.

## Benchmarks

Run:

```powershell
python scripts/benchmark_npu_scoring.py --device CPU
python scripts/benchmark_realtime_pipeline.py --device CPU
```

Use `--device NPU` on machines with OpenVINO NPU support.

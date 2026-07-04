# NPU / CPU Benchmark Results

## Current Runtime Contract

As of the current `run.ps1` entry point, the system is a guarded KIS live-capable realtime runtime. KIS realtime collection, read-only account probing, periodic live short-horizon training, and the independent realtime trading loop can start automatically. Numeric ontology/candidate evidence scoring requests OpenVINO `NPU` and falls back to CPU when unavailable; final action selection, graph explanations, risk checks, order gating, idempotency, and broker submission remain deterministic CPU-controlled paths. NPU output is evidence, not trade authorization.

These are point-in-time snapshots. The benchmark scripts write fresh raw tables under `data/reports/` (override with `--output`); paste updated numbers here when refreshing this snapshot:

```powershell
python scripts/benchmark_npu_scoring.py --device CPU
python scripts/benchmark_npu_scoring.py --device NPU
python scripts/benchmark_realtime_pipeline.py --device CPU
python scripts/benchmark_realtime_pipeline.py --device NPU
```

## Ontology Candidate Scoring (`benchmark_npu_scoring.py`)

Requested device `CPU`:

| scenario | device | batch | top_k | preprocess_ms | inference_ms | postprocess_ms | total_ms | memory_mb |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | CPU | 512 | 50 | 0.145 | 1.807 | 0.454 | 751.608 | 8.155 |
| 1024 | CPU | 1024 | 50 | 0.202 | 0.576 | 0.289 | 28.189 | 8.183 |
| 4096 | CPU | 4096 | 50 | 0.443 | 0.587 | 0.437 | 28.125 | 8.373 |
| 10000 | CPU | 4096 | 50 | 0.882 | 1.389 | 0.927 | 33.25 | 8.556 |

Requested device `NPU`:

| scenario | device | batch | top_k | preprocess_ms | inference_ms | postprocess_ms | total_ms | memory_mb |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | NPU | 512 | 50 | 0.186 | 5.61 | 0.327 | 690.583 | 8.144 |
| 1024 | NPU | 1024 | 50 | 0.194 | 5.184 | 0.333 | 26.221 | 8.189 |
| 4096 | NPU | 4096 | 50 | 0.341 | 4.528 | 0.427 | 25.036 | 8.378 |
| 10000 | NPU | 4096 | 50 | 1.063 | 6.577 | 0.381 | 16.289 | 8.568 |

## Realtime Pipeline (`benchmark_realtime_pipeline.py`)

Requested device `CPU`:

| scenario | input | hard_filter | topk | device | scoring_ms | total_pipeline_ms | peak_memory_mb |
|---|---:|---:|---:|---|---:|---:|---:|
| small_universe | 128 | 80 | 50 | CPU | 611.272 | 619.743 | 5.324 |
| medium_universe | 1024 | 644 | 50 | CPU | 28.529 | 105.521 | 6.734 |
| large_universe | 4096 | 2647 | 50 | CPU | 33.606 | 285.633 | 12.538 |
| extra_large_universe | 10000 | 6460 | 50 | CPU | 31.122 | 612.053 | 24.676 |

Requested device `NPU`:

| scenario | input | hard_filter | topk | device | scoring_ms | total_pipeline_ms | peak_memory_mb |
|---|---:|---:|---:|---|---:|---:|---:|
| small_universe | 128 | 80 | 50 | NPU | 1067.647 | 1077.938 | 5.309 |
| medium_universe | 1024 | 644 | 50 | NPU | 406.346 | 468.341 | 6.732 |
| large_universe | 4096 | 2647 | 50 | NPU | 244.137 | 521.194 | 12.536 |
| extra_large_universe | 10000 | 6460 | 50 | NPU | 32.273 | 623.123 | 24.673 |

## Notes

- The first small-batch scenario carries one-time OpenVINO compile/warm-up cost, so its `total_ms` is not representative of steady-state latency.
- NPU wins at large batch sizes; at small batches the fixed dispatch overhead makes CPU competitive, which is why `NPU_MIN_BATCH_FOR_NPU` routes small batches to CPU NumPy.
- Regardless of device, scoring output is evidence only; execution authority stays on the deterministic CPU path.

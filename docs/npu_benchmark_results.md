# NPU Benchmark Results

## Current Runtime Contract

As of the current `run.ps1` entry point, the system is a guarded KIS live-capable realtime runtime. KIS realtime collection, read-only account probing, periodic live short-horizon training, and the independent realtime trading loop can start automatically. Numeric ontology/candidate evidence scoring requests OpenVINO `NPU` and falls back to CPU when unavailable; final action selection, graph explanations, risk checks, order gating, idempotency, and broker submission remain deterministic CPU-controlled paths. NPU output is evidence, not trade authorization.


Requested device: `CPU`

| scenario | device | batch | top_k | preprocess_ms | inference_ms | postprocess_ms | total_ms | memory_mb |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | CPU | 512 | 50 | 0.145 | 1.807 | 0.454 | 751.608 | 8.155 |
| 1024 | CPU | 1024 | 50 | 0.202 | 0.576 | 0.289 | 28.189 | 8.183 |
| 4096 | CPU | 4096 | 50 | 0.443 | 0.587 | 0.437 | 28.125 | 8.373 |
| 10000 | CPU | 4096 | 50 | 0.882 | 1.389 | 0.927 | 33.25 | 8.556 |

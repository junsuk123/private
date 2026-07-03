# NPU Benchmark Results

## Current Runtime Contract

As of the current `run.ps1` entry point, the system is a guarded KIS live-capable realtime runtime. KIS realtime collection, read-only account probing, periodic live short-horizon training, and the independent realtime trading loop can start automatically. Numeric ontology/candidate evidence scoring requests OpenVINO `NPU` and falls back to CPU when unavailable; final action selection, graph explanations, risk checks, order gating, idempotency, and broker submission remain deterministic CPU-controlled paths. NPU output is evidence, not trade authorization.


Requested device: `NPU`

| scenario | device | batch | top_k | preprocess_ms | inference_ms | postprocess_ms | total_ms | memory_mb |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | NPU | 512 | 50 | 0.186 | 5.61 | 0.327 | 690.583 | 8.144 |
| 1024 | NPU | 1024 | 50 | 0.194 | 5.184 | 0.333 | 26.221 | 8.189 |
| 4096 | NPU | 4096 | 50 | 0.341 | 4.528 | 0.427 | 25.036 | 8.378 |
| 10000 | NPU | 4096 | 50 | 1.063 | 6.577 | 0.381 | 16.289 | 8.568 |

# OpenVINO CPU/NPU Benchmark

Status: NPU promotion rejected.

Command:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_strategy_utility_openvino.py --iterations 30
```

Fixed shape: `B=1, T=4, N=16, F=12, R=4, S=7`, FP32.

| device | actual compiled device | compile ms | p50 ms | p95 ms | p99 ms | throughput/s |
|---|---|---:|---:|---:|---:|---:|
| CPU | CPU | 89.91 | 0.329 | 0.660 | 0.667 | 2789 |
| NPU | NPU | 43.79 | 1.137 | 1.398 | 2.263 | 837 |

CPU/NPU top-1 strategy agreement was 100%. Maximum utility absolute error was `0.030865`; NoTrade probability maximum error was `0.0000766`.

The NPU was genuinely compiled—there was no CPU fallback—but end-to-end inference was slower than CPU and utility error exceeded the configured `0.001` golden tolerance. `promotion_eligible=false`; CPU remains the verified runtime. The JSON evidence is stored in `data/reports/strategy_utility_openvino.json`.

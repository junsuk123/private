# Realtime Pipeline Benchmark Results

Requested device: `CPU`

| scenario | input | hard_filter | topk | device | scoring_ms | total_pipeline_ms | peak_memory_mb |
|---|---:|---:|---:|---|---:|---:|---:|
| small_universe | 128 | 80 | 50 | CPU | 585.23 | 597.921 | 6.285 |
| medium_universe | 1024 | 644 | 50 | CPU | 22.514 | 60.553 | 7.12 |
| large_universe | 4096 | 2647 | 50 | CPU | 29.01 | 155.238 | 10.171 |
| extra_large_universe | 10000 | 6460 | 50 | CPU | 28.322 | 318.851 | 15.943 |

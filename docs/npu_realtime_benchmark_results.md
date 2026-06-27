# Realtime Pipeline Benchmark Results

Requested device: `CPU`

| scenario | input | hard_filter | topk | device | scoring_ms | total_pipeline_ms | peak_memory_mb |
|---|---:|---:|---:|---|---:|---:|---:|
| small_universe | 128 | 80 | 50 | CPU | 611.272 | 619.743 | 5.324 |
| medium_universe | 1024 | 644 | 50 | CPU | 28.529 | 105.521 | 6.734 |
| large_universe | 4096 | 2647 | 50 | CPU | 33.606 | 285.633 | 12.538 |
| extra_large_universe | 10000 | 6460 | 50 | CPU | 31.122 | 612.053 | 24.676 |

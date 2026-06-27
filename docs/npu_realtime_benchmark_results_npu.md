# Realtime Pipeline Benchmark Results

Requested device: `NPU`

| scenario | input | hard_filter | topk | device | scoring_ms | total_pipeline_ms | peak_memory_mb |
|---|---:|---:|---:|---|---:|---:|---:|
| small_universe | 128 | 80 | 50 | NPU | 1067.647 | 1077.938 | 5.309 |
| medium_universe | 1024 | 644 | 50 | NPU | 406.346 | 468.341 | 6.732 |
| large_universe | 4096 | 2647 | 50 | NPU | 244.137 | 521.194 | 12.536 |
| extra_large_universe | 10000 | 6460 | 50 | NPU | 32.273 | 623.123 | 24.673 |

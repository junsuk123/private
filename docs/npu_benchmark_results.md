# NPU Benchmark Results

Requested device: `CPU`

| scenario | device | batch | top_k | preprocess_ms | inference_ms | postprocess_ms | total_ms | memory_mb |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | CPU | 512 | 50 | 0.145 | 1.807 | 0.454 | 751.608 | 8.155 |
| 1024 | CPU | 1024 | 50 | 0.202 | 0.576 | 0.289 | 28.189 | 8.183 |
| 4096 | CPU | 4096 | 50 | 0.443 | 0.587 | 0.437 | 28.125 | 8.373 |
| 10000 | CPU | 4096 | 50 | 0.882 | 1.389 | 0.927 | 33.25 | 8.556 |

# P0 Baseline

Captured from commit `b0913d5` on 2026-07-27 before functional integration of the new path.

## Repository and test surface

- Branch: `main`.
- Worktree was clean at audit start.
- Python requirement: 3.11 or newer.
- Test collection: 822 pre-existing tests; 12 P1 tests were then added.
- Installed runtime includes FastAPI, HTTPX, NumPy, RDF/OWL/SHACL, WebSockets, and optional OpenVINO support.
- Ruff configuration exists in `pyproject.toml`, but Ruff is not installed in the project virtual environment.

## Static latency and I/O baseline

- Domestic feed: KIS WebSocket trade/order-book parsing exists.
- U.S. fast refresh: REST poll, default 12 seconds, one quote/order-book request group per held symbol.
- Web live execution: no more than one automatic order per cycle by default.
- Domestic message handling can synchronously write ticks/order books and build feature frames.
- Order status uses REST polling after submission.
- `app.web` contains synchronous broker/account, storage, graph, model, and UI aggregation work around the live cycle.

## Existing observability data

The local ignored logs show that decision, feature, and order telemetry is active. At audit time the largest files were approximately 229 MB for decisions, 250 MB for feature frames, and 7.8 MB for live orders. This is evidence of missing retention/compaction, not a performance benchmark. No credentials or log payloads were copied into refactor artifacts.

## Baseline limitations

The existing realtime benchmark measures batch candidate filtering on synthetic universes and labels requested device; it does not measure WebSocket-to-order p50/p95/p99, CPU utilization, OpenVINO operator coverage, or CPU/NPU decision-ranking parity. Those values must remain “not measured” until dedicated instrumentation and a representative replay exist.

## CPU batch candidate benchmark

Command:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_realtime_pipeline.py --device CPU --output data\reports\refactor_p0_cpu_benchmark.md
```

| synthetic input | accepted after hard filter | top K | total pipeline ms | peak Python-traced MB |
|---:|---:|---:|---:|---:|
| 128 | 80 | 50 | 597.921 | 6.285 |
| 1,024 | 644 | 50 | 60.553 | 7.120 |
| 4,096 | 2,647 | 50 | 155.238 | 10.171 |
| 10,000 | 6,460 | 50 | 318.851 | 15.943 |

The first scenario includes model/runtime warm-up and is not comparable to the warm larger batches. Although legacy telemetry emits `npu_enabled=1`, the requested and reported device was `CPU`; this result must not be described as NPU acceleration.

## Verification result

- New P1 tests: 12 passed.
- Existing realtime/execution characterization subset: 95 passed.
- Realtime exit and ownership-adjacent suite after isolation fix: 63 passed.
- P0/P1 checkpoint: 834 passed, 47 deprecation warnings, in 136.72 seconds.
- P1-P8 implementation checkpoint: 870 passed before the final integration additions.
- Final integrated repository: 879 passed, 47 deprecation warnings, in 255.93 seconds.

The warnings are existing FastAPI `on_event` and RDFLib `default_context` deprecations. No live-order test or broker submission was executed.

# Realtime Short-Horizon Policy

The realtime layer is optimized for responsive UI actions, short-horizon diagnostics, hypothetical testing, and safe local learning. It uses one realtime data environment and never submits automatic live broker orders.

Typical analysis horizons:

- 5 seconds
- 15 seconds
- 30 seconds
- 60 seconds
- 5 minutes
- 1 hour

## Acceleration and NPU Policy

`RealtimeAccelerationPolicy` applies low-latency OpenVINO/NPU process hints and reports the active backend. If an NPU runtime is unavailable, deterministic CPU fallback remains enabled.

Common environment:

```text
ONTOLOGY_ACCELERATOR=NPU
REALTIME_LATENCY_PROFILE=low_latency
OPENVINO_DEVICE=NPU
OPENVINO_HINT_PERFORMANCE_MODE=LATENCY
OPENVINO_ENABLE_CPU_PINNING=YES
OPENVINO_CACHE_DIR=data/runtime/openvino_cache
```

This acceleration policy is for inference/runtime preference. It is not a user-facing simulation speed control.

## GUI Modes

The current `OperationModeManager` supports:

- `learning`: realtime data collection and supervised example/model-artifact updates.
- `testing`: realtime hypothetical trade test using current analysis frames and strategy signals.
- `live_trading`: realtime trading gate; automatic execution remains guarded/blocked.

All modes use:

```text
data/store
data/models
```

Synthetic and simulation rows are not valid inputs for learning, testing, or live trading.

## Learning Behavior

When `POST /api/operation-mode/start` receives `mode = learning`, the app starts the live worker. The worker:

1. Refreshes configured public research.
2. Stores new records in `data/store`.
3. Builds a fresh analysis context.
4. Builds time-synchronized ticker frames.
5. Creates supervised examples from adjacent realtime frames and strategy signals.
6. Writes model artifacts under `data/models/realtime_supervised`.

The learning loop is stopped with:

```text
POST /api/operation-mode/stop-learning
```

## Testing Behavior

When `POST /api/operation-mode/start` receives `mode = testing`, the app:

1. Forces a live refresh.
2. Builds the current analysis context.
3. Runs `run_hypothetical_realtime_test`.
4. Writes a hypothetical testing artifact under `data/models/hypothetical_testing`.
5. Reports `orders_submitted = 0`.

Testing uses inferred entry/exit prices from adjacent time frames and does not call a broker.

## Streaming Simulation Behavior

Streaming simulation is separate from operation-mode testing. It starts through:

```text
POST /api/streaming-demo/start
```

with:

- `target_return_rate`
- `period_minutes`
- `initial_cash`

The UI then calls `/api/streaming-demo/step` on a timer. In realtime simulation mode, one visible synthetic minute is due every wall-clock minute.

Each step:

1. Uses synthetic charts generated in memory.
2. Screens the universe through ontology/NPU candidate selection.
3. Builds market snapshots and indicators for selected candidates.
4. Runs ontology reasoning.
5. Builds a target-aware execution plan.
6. Validates candidate orders through `RiskManager`.
7. Applies approved mock trades only to simulated cash and holdings.
8. Updates progress, account value, return rate, positions, and execution tables.

If a step is requested before the next synthetic minute is due, the API returns `status = waiting`. If the session expired, it returns `status = expired` with HTTP 200.

## Avoiding Trapped Positions

Current guardrails:

- Intraday BUY position cap through `max_intraday_position_weight`.
- BUY weights are capped by both strategy sizing and `RiskManager`.
- SELL/REDUCE intents are ranked before BUY intents in streaming simulation.
- High volatility, insufficient liquidity, duplicate orders, cash reserve, and sector exposure can block orders.
- Final streaming step liquidates remaining simulated holdings.
- Live automatic execution remains disabled.

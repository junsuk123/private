# Realtime Short-Horizon Policy

The realtime layer is optimized for fast prediction horizons and responsive UI actions. It supports low-latency diagnostics, live snapshot refreshes, and stepwise simulation testing.

Typical analysis horizons:

- 5 seconds
- 15 seconds
- 30 seconds
- 60 seconds
- 5 minutes
- 1 hour

## NPU Preference

`RealtimeAccelerationPolicy` sets low-latency OpenVINO/NPU hints and reports the active backend. If NPU is not available, deterministic CPU fallback remains enabled.

This runtime acceleration policy is not a user-facing simulation speed control. The visible simulation test no longer exposes a speed or acceleration multiplier.

Environment:

```text
ONTOLOGY_ACCELERATOR=NPU
REALTIME_LATENCY_PROFILE=low_latency
OPENVINO_DEVICE=NPU
OPENVINO_HINT_PERFORMANCE_MODE=LATENCY
OPENVINO_ENABLE_CPU_PINNING=YES
```

## GUI Modes

- Simulation training: synthetic/offline data, stored under `data/sim`.
- Simulation testing: stepwise streaming simulation using the form's target return and target minutes.
- Live training: current market data only, training allowed, orders prohibited.
- Live trading: risk and manual approval gate. Automatic order execution remains blocked.

## Simulation Testing Behavior

Simulation testing starts from `POST /api/operation-mode/start` with:

- `mode = simulation_testing`
- `target_return_rate`
- `period_minutes`

The server returns a `demo_id`, initial cash, target return, and period minutes. The UI then calls `/api/streaming-demo/step` on a timer. Each step:

1. Advances one visible simulation minute.
2. Rebuilds market snapshots and indicators from synthetic prices.
3. Runs ontology reasoning.
4. Builds a target-aware execution plan.
5. Validates orders through `RiskManager`.
6. Applies approved mock trades to simulated cash and holdings.
7. Updates progress, account value, return rate, positions, and execution tables.

The top mock-return card is updated from the streaming account state, so the displayed mock return follows the simulation in real time.

If a simulation session expires or the server restarts, `/api/streaming-demo/step` returns `status="expired"` with HTTP 200. The UI stops the loop and asks the user to restart simulation testing instead of showing a raw 404.

## UI Responsiveness

`/api/live-snapshot` runs its heavier snapshot work in a threadpool. This keeps periodic dashboard refreshes from blocking operation-mode button clicks such as simulation learning or simulation testing.

## Avoiding Trapped Positions

Short-term losses can happen, but the system should avoid becoming stuck in large adverse positions.

Current guardrails:

- Intraday buy position cap: 2.5% by default.
- Buy requires confidence and expected edge over downside risk.
- Reduce signal is preferred when expected loss or downside risk crosses the guard.
- RiskManager caps BUY weights further through `max_intraday_position_weight`.
- Live automatic execution remains disabled unless a later explicit approval workflow enables it.

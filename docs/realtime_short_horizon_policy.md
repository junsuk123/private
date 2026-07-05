# Realtime Short-Horizon Policy

## Current Runtime Contract

As of the current `run.ps1` entry point, the system is a guarded KIS live-capable realtime runtime. KIS realtime collection, read-only account probing, periodic live short-horizon training, and the independent realtime trading loop can start automatically. Numeric ontology/candidate evidence scoring requests OpenVINO `NPU` and falls back to CPU when unavailable; final action selection, graph explanations, risk checks, order gating, idempotency, and broker submission remain deterministic CPU-controlled paths. NPU output is evidence, not trade authorization.


The realtime layer is optimized for responsive UI actions, short-horizon diagnostics, safe local learning, and guarded KIS live auto-trading. It uses one realtime data environment and submits live broker orders only through deterministic gates.

See `docs/diagrams/system_overview.png` for the repository-level flow from trusted data inputs through candidate scoring, ontology reasoning, risk validation, paper/live execution, and feedback.

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
- `testing`: backward-compatible legacy paper-trading replay.
- `paper_trading` / `paper_trading_test`: KIS paper-trading API check plus local paper-trading flow.
- `live_readiness` / `live_trading_test`: KIS live-readiness/authentication check without broker orders.
- `live_trading`: realtime KIS live auto-trading loop; execution remains guarded by runtime, KIS, source, cost, risk, idempotency, and kill-switch gates.

All modes use:

```text
data/store
data/models
```

Synthetic and simulation rows are not valid inputs for learning, testing, or live trading.

Default `run.ps1` startup behavior:

- KIS realtime tick/orderbook collection starts automatically when `AUTO_START_KIS_REALTIME_COLLECTOR=true`.
- Periodic live short-horizon model training starts automatically when `AUTO_START_LIVE_TRAINING=true`.
- The independent realtime trading engine starts automatically when `AUTO_START_REALTIME_TRADING=true`.
- A read-only KIS live-readiness account check starts automatically when `AUTO_START_LIVE_READINESS=true`.
- The `/account` dashboard becomes the primary control surface for account state, asset history, realtime decision flow, rejection reasons, and termination.

## Learning Behavior

On default server startup, and also when `POST /api/operation-mode/start` receives `mode = learning`, the app starts the live worker. The worker:

1. Refreshes configured public research.
2. Stores new records in `data/store`.
3. Builds a fresh analysis context.
4. Builds time-synchronized ticker frames.
5. Creates supervised examples from adjacent realtime frames and strategy signals.
6. Writes model artifacts under `data/models/realtime_supervised`.

The learning loop can still be stopped through the API for diagnostics:

```text
POST /api/operation-mode/stop-learning
```

## Live Trading Behavior

The independent realtime trading engine is implemented in `src/app/trading/realtime_trading_engine.py`.

Each cycle:

1. Reads a KIS account snapshot.
2. Evaluates SELL/REDUCE for holdings first.
3. Keeps existing open SELL orders when the replacement price is effectively unchanged.
4. Evaluates BUY candidates only if `REALTIME_BUY_ENABLED=true`.
5. Uses `SharedLiveDecisionEngine` for model/ontology/runtime evaluation.
6. Submits approved `FinalOrder` objects through `LiveExecutionCoordinator`.

BUY is intentionally rejected when:

- the model feature frame is unavailable or stale
- broker quote cash is insufficient for one share plus buffer
- spread exceeds adaptive max spread
- liquidity is thin
- fallback score is below the adaptive threshold
- ontology/runtime confirmation is missing
- model-only support is present while `REALTIME_MODEL_AUXILIARY_ONLY=true`

SELL/REDUCE can be triggered by profit target, trailing/loss exit, domestic drawdown, emergency exit, concentration reduction, and time/quote policies.

### News/event sentiment (soft confirmation)

When a local LLM is configured (shared `config/local_llm.env`; see the main `README.md`), RSS/disclosure text is classified into `POSITIVE`/`NEGATIVE`/`NEUTRAL` and mapped into the ontology graph. Its effect on the live BUY path is intentionally limited:

- **Negative** news (`increasesRiskOf NegativeEventRisk`) already subtracts from buy evidence in `_ontology_buy_evidence`.
- **Positive** news is a **soft confirmation only**: it adds `REALTIME_NEWS_CONFIRM_BONUS` (default `0.15`) to the ontology score of a candidate that already has other support, appends a `PositiveNewsConfirm` tag, and never sets `ontology_ok` on its own. News alone cannot create a BUY, and if negative news is also present the bonus is withheld.
- Set `REALTIME_NEWS_SENTIMENT_ENABLED=false` to turn the buy-side reflection off. All spread/liquidity/cash/`RiskManager` gates still apply regardless.

## Paper-Trading and Readiness Behavior

When `POST /api/operation-mode/start` receives `mode = testing`, `paper_trading`, or `paper_trading_test`, the app:

1. Forces a live refresh.
2. Builds the current analysis context.
3. Runs `run_hypothetical_realtime_test`.
4. Writes a hypothetical testing artifact under `data/models/hypothetical_testing`.
5. Reports `orders_submitted = 0`.
6. For KIS paper modes, performs the KIS paper API readiness path and keeps live orders disabled.

Legacy testing uses inferred entry/exit prices from adjacent time frames and does not call a broker. KIS paper-trading modes may use the virtual broker domain only.

When `mode = live_readiness` or `live_trading_test`, the app checks KIS live-readiness/authentication boundaries and does not submit broker orders.

In the default UI, this same read-only live-readiness path runs automatically at server startup and stores the most recent account basis for later paper-trading sizing.

## Paper-Trading Simulation Behavior

The paper-trading simulation is separate from operation-mode readiness checks. It starts through:

```text
POST /api/paper-trading/start
```

with:

- `target_return_rate`
- `period_minutes`
- `initial_cash_source = auto` by default

The UI then calls `/api/paper-trading/step` on a timer. In realtime simulation mode, one visible synthetic minute is due every wall-clock minute.

`initial_cash` is computed automatically from the latest read-only KIS live account basis when available. If no cached basis exists, `initial_cash_source = auto` triggers a read-only KIS live account refresh before falling back to the default. The profit-gain multiplier is also automatic and is derived from target return, target horizon, account size, and cash weight.

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
- Live automatic execution is available only in the guarded `live_trading` runtime and remains subject to the same risk/freshness/cost controls.

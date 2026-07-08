# Live Trading Runbook

## Current Runtime Contract

As of the current `run.ps1` entry point, the system is a guarded KIS live-capable realtime runtime. KIS realtime collection, read-only account probing, periodic live short-horizon training, and the independent realtime trading loop can start automatically. Numeric ontology/candidate evidence scoring requests OpenVINO `NPU` and falls back to CPU when unavailable; final action selection, graph explanations, risk checks, order gating, idempotency, and broker submission remain deterministic CPU-controlled paths. NPU output is evidence, not trade authorization.


This runbook describes the current local `run.ps1` runtime. It is KIS live-capable and starts the realtime trading engine automatically, while keeping deterministic gates mandatory.

## Start

```powershell
.\run.ps1
```

The launcher:

- stops any existing local server on port `8010`
- sets live process flags such as `TRADING_MODE=live_trading`, `LIVE_TRADING_ENABLED=true`, `KIS_LIVE_ENABLED=true`, `KIS_PAPER_TRADING=false`, and `LIVE_ORDER_SUBMIT_ENABLED=true`
- sets `AUTO_START_REALTIME_TRADING=true`
- sets `AUTO_START_LIVE_TRAINING=true`
- sets `REALTIME_BUY_ENABLED=true`
- opens `http://127.0.0.1:8010/account`
- stops the server when the managed browser window closes

## Main Dashboards

- Account dashboard: `http://127.0.0.1:8010/account`
- Realtime trading status: `http://127.0.0.1:8010/api/realtime-trading/status`
- Account payload: `http://127.0.0.1:8010/api/account/dashboard`
- AI/model validation: `http://127.0.0.1:8010/api/ai/validation`
- Live training status: `http://127.0.0.1:8010/api/live-training/status`

## Normal Live Flow

Each realtime engine cycle:

1. Reads the latest KIS live account snapshot.
2. Evaluates SELL/REDUCE exits for current holdings.
3. Keeps an existing open SELL order if the proposed replacement price is effectively unchanged.
4. Evaluates BUY candidates only when `REALTIME_BUY_ENABLED=true`.
5. Requires broker quote freshness, one-share cash, acceptable spread/liquidity, ontology/runtime support, and risk approval.
6. Submits only `FinalOrder` limit orders through `LiveExecutionCoordinator`.

The live ML model is auxiliary. With `REALTIME_MODEL_AUXILIARY_ONLY=true`, model-only BUY approval is rejected.

## Understanding Stalls

A quiet cycle is not necessarily a fault. Use `/api/realtime-trading/status`.

Common explanations:

- `open_sell_kept`: a SELL order is already open at the same effective price, so no duplicate order is submitted.
- `HOLD_BELOW_PROFIT_TARGET`: SELL was evaluated, but current price is below the net-profit target after costs.
- `MODEL_FEATURE_UNAVAILABLE:...QUOTE_STALE,ORDERBOOK_STALE`: the model cannot score because KIS realtime tick/orderbook inputs are stale or missing.
- `WIDE_SPREAD:x>ybps`: current spread is too wide for the adaptive policy.
- `LOW_LIQUIDITY`: candidate liquidity is thin.
- `FALLBACK_SCORE_BELOW_THRESHOLD`: rule/ontology fallback score is below the adaptive threshold.
- `INSUFFICIENT_CASH_FOR_ONE_SHARE`: available cash in the relevant currency is below one-share price plus buffer.

If `blocked=0` and `errors=0`, the engine is usually functioning and intentionally rejecting low-quality trades.

## Termination Button

The `/account` dashboard has a termination button. It:

1. Sets `REALTIME_BUY_ENABLED=false`.
2. Calls the engine's BUY-disable control.
3. Stops the realtime trading loop.
4. Submits profit-seeking limit SELL orders for current KIS holdings when live gates pass.
5. Schedules server shutdown.
6. Lets `run.ps1` close the managed browser and stop local processes.

Profit-seeking termination prices default to `average_price * 1.0025` or better. If a holding is beyond the hard loss exit threshold, the current broker mark is used to avoid deeper exposure.

## Emergency Stop

Run:

```powershell
python scripts/disarm_live_trading.py
```

Then set:

```powershell
$env:KILL_SWITCH_ENABLED="true"
```

This blocks new live submissions through `LiveExecutionCoordinator`.

You can also disable new buys without stopping sells:

```powershell
$env:REALTIME_BUY_ENABLED="false"
```

## Inspect Logs

- Server stdout: `logs/run-server.out.log`
- Server stderr: `logs/run-server.err.log`
- Live order journal: `logs/live-orders.jsonl`
- Account dashboard store: `data/store/account_dashboard.sqlite3`
- Realtime market store: `data/store/realtime_market_data.sqlite3`
- Feature journal: `logs/live-feature-frames.jsonl`
- Live model artifacts: `data/models/live_short_horizon/`
- Readiness reports: `data/reports/live_readiness_*.json`
- Dry-run reports: `data/reports/live_order_dry_run_*.json`

Logs are JSONL where applicable and redacted through the audit logger.

## Reconcile Broker State

Before restarting after unknown network or broker errors:

1. Read KIS live account balance.
2. Check order status for submitted broker order ids.
3. Confirm whether open SELL orders are still pending, filled, canceled, or amendable.
4. Do not retry an order after an unknown result until broker state is reconciled.

## Current Risk Posture

The engine is intentionally conservative. Wide spreads, thin liquidity, stale realtime inputs, insufficient cash, missing ontology/runtime support, and model-only approvals should lead to no trade.

## Profitability-first decision flow (refactor)

Every BUY is now judged by ONE authoritative net-profitability rule and every exit by ONE
resolved exit policy:

1. **ProfitabilityGate** (`src/app/cost/profitability_gate.py`, `config/profitability_policy.yaml`):
   a BUY is allowed only if expected NET return after all costs clears a dynamic minimum
   edge (KR 0.8% / US 1.2% by default), the exit price clears break-even+buffer, and
   spread/liquidity/cost-to-alpha are within bounds. The live buy path derives a **real**
   predicted exit price (no fabricated 100 bps floor). See `docs/profitability_gate.md`.
2. **DynamicExitPolicy** (`src/app/trading/dynamic_exit_policy.py`,
   `config/dynamic_exit_policy.yaml`): unifies all exit thresholds (logged once). Loss exits
   are permitted only on strong deterioration evidence and blocked for noise-level losses.
   See `docs/dynamic_exit_policy.md`.
3. **PositionSizer** (`src/app/risk/position_sizing.py`): edge/confidence/liquidity/drawdown
   fractional-Kelly sizing; never sizes a negative-expectancy trade.
4. **ExecutionQuality** (`src/app/execution/execution_quality.py`): rejects buys whose alpha
   is consumed by spread/slippage; no-chase guard; realized-slippage store.

### Arming is NOT ARM/Raspberry Pi

`scripts/arm_live_trading.py` / `disarm_live_trading.py` are the **live-order-submission
safety switch** (they write/clear `config/secrets/live_trading_armed.json`). "Arm" means
*arm live submission* — it is unrelated to the ARM CPU architecture or Raspberry Pi. Before
arming, `submit` raises `LiveExecutionBlocked` and the engine records `blocked` (no real
order). Runtime profiles and the optional Raspberry Pi monitor node are documented in
`docs/runtime_profiles.md`.

### Measuring profitability

`PYTHONPATH=src python scripts/profitability_replay_report.py` reports order-flow outcomes,
rejection-reason distribution, and cost-aware realized metrics (net PnL, win rate, payoff,
expectancy) from `logs/live-orders.jsonl`. Run it before and after to compare — the success
criterion is improved NET expectancy and fewer net-negative trades, not more trades.

## Technical prediction diagnostics

The evidence-based technical prediction layer (`src/app/technical/`, see
`docs/technical_prediction_layer.md`) is advisory — it never bypasses the
ProfitabilityGate or RiskManager.

- **GUI**: `/account` → "기술적 예측 (자문 전용)" panel shows, per symbol, the
  regime, selected methodology, expected edge/horizon, predicted exit price,
  downside, VWAP distance, confidence, and the gate result (net vs required),
  with rejection reason cards (below net edge / spread consumes alpha / low
  liquidity / high volatility / model feature unavailable / no ontology support).
  API: `GET /api/account/technical`.
- **Per-decision diagnostics**: `SharedDecisionResult.diagnostics` carries
  `technical_prediction`, `technical_methodology`, `technical_regime`, and (on
  exits) `technical_exit_deterioration`.
- **Config**: `config/technical_prediction_policy.yaml` (env overrides logged at
  load). Set `enabled: false` to disable the layer entirely (buys fall back to
  the prior model/ontology behavior).
- **Validation**: run `python scripts/replay_technical_prediction.py` and read
  `data/models/technical_replay_reports/`; judge on realized **net-after-cost**,
  not gross hit rate (`docs/technical_prediction_validation.md`).
- **Schema/model**: the live feature schema is now `live_short_horizon_v2` (6
  technical columns added). Prior model artifacts are retired; retrain before
  the trained model contributes again — buys stay safe (advisory fallback) until
  then.

## Macro–micro ontology diagnostics

The hierarchical macro–micro reasoning layer (`src/app/graph/`, see
`docs/macro_micro_ontology_architecture.md`) is advisory — it selects candidates
and strategy permissions and ranks intents, but never bypasses the
ProfitabilityGate or RiskManager.

- **GUI**: `/account` → "거시–미시 온톨로지 (자문 전용)" panel shows the market
  regime, macro risk level, sector ranking, candidate symbols, allowed/blocked
  strategies, per-symbol micro regime / entry-exit / expected net return /
  execution quality, and the SELL/REDUCE-first ranked intents.
  API: `GET /api/account/macro-micro`.
- **Config**: `config/macro_micro_ontology.yaml` (env overrides logged at load;
  invalid values clamp to conservative defaults, recorded in
  diagnostics.config_fallbacks). Set `enabled: false` to disable the layer.
- **Loop cadence**: macro reasons slower (default 60s) than micro (default 5s).
  Macro `BLOCK_BUY` (high volatility / news shock / low liquidity) stops new BUY
  micro reasoning but held-symbol SELL/REDUCE checks continue.
- **Validation**: `python scripts/replay_macro_micro_ontology.py --from-bars <file>`
  → `data/models/macro_micro_replay_reports/`; judge on realized net-after-cost
  and `avg_edge_error_bps`, not gross signals.
- **Safety check**: confirm no macro/micro node ever asserts a FinalOrder — every
  BUY still shows TradingCostEngine/ProfitabilityGate/RiskManager reason codes,
  and only `LiveExecutionCoordinator` submits (limit orders).

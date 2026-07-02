# Live Trading Runbook

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

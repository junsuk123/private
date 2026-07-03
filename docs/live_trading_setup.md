# Live Trading Setup

## Current Runtime Contract

As of the current `run.ps1` entry point, the system is a guarded KIS live-capable realtime runtime. KIS realtime collection, read-only account probing, periodic live short-horizon training, and the independent realtime trading loop can start automatically. Numeric ontology/candidate evidence scoring requests OpenVINO `NPU` and falls back to CPU when unavailable; final action selection, graph explanations, risk checks, order gating, idempotency, and broker submission remain deterministic CPU-controlled paths. NPU output is evidence, not trade authorization.


This repository is fail-closed at the order boundary. The current `run.ps1` runtime enables live process flags and starts the realtime trading engine automatically, but real KIS orders are still blocked unless all live flags, KIS health checks, runtime guards, idempotency, source freshness, cost/risk checks, and backend approval gates pass.

For day-to-day operation, prefer:

```powershell
.\run.ps1
```

Then use `http://127.0.0.1:8010/account` as the primary dashboard.

## Setup

1. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

2. Create local ignored config files:

```powershell
copy config\secrets\kis_api_keys.env.example config\secrets\kis_api_keys.env
copy config\principal_protection.example.json config\principal_protection.json
copy config\trading_costs.example.json config\trading_costs.json
copy config\live_trading_safety.example.json config\live_trading_safety.json
copy config\order_execution.example.json config\order_execution.json
```

3. Fill local-only values:

- KIS app key and app secret.
- KIS account number and product code.
- KIS HTS ID and customer type.
- Initial principal and protected floor settings.
- User-verified current KIS fee rates and applicable Korean stock transaction tax.

Do not commit these files.

## Checks

Readiness dry-run:

```powershell
python scripts/live_readiness_check.py --dry-run
```

KIS read-only checks after credentials are filled:

```powershell
python scripts/live_readiness_check.py --check kis-auth,kis-account --no-orders
```

No-submit order path report:

```powershell
python scripts/live_order_dry_run.py --symbols 005930,000660 --no-submit
```

Realtime fixture check:

```powershell
python scripts/check_realtime_market_data.py --symbols 005930 --fixture path\to\kis_fixture.txt
```

Train from a real JSONL dataset:

```powershell
python scripts/train_live_short_horizon_models.py --dataset data\training\live_short_horizon.jsonl
```

The `--demo-fixture` option is for code-path validation only and is always marked
not live-eligible.

## Arming

Arming creates a short-lived local file. It does not bypass readiness checks.

```powershell
python scripts/arm_live_trading.py
python scripts/disarm_live_trading.py
```

Required live environment flags:

```powershell
$env:LIVE_TRADING_ENABLED="true"
$env:KIS_LIVE_ENABLED="true"
$env:KIS_PAPER_TRADING="false"
$env:LIVE_ORDER_SUBMIT_ENABLED="true"
$env:KILL_SWITCH_ENABLED="false"
```

## Current Status

Guarded live execution is available in the local `run.ps1` runtime. The realtime loop:

- reads KIS account state
- evaluates exits before entries
- uses KIS realtime ticks/orderbooks and broker quote refresh
- trains and validates live short-horizon artifacts in the background
- treats the live model as auxiliary
- submits only approved limit `FinalOrder` objects through `LiveExecutionCoordinator`

Execution should still be treated as experimental and conservative. No code in this repository guarantees profit or capital protection. The controls are engineering gates only.

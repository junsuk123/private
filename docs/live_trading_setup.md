# Live Trading Setup

This repository is fail-closed. Real KIS orders are blocked unless all live flags,
manual arming, KIS health checks, and backend approval gates pass.

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

Dry-run and guarded execution primitives are available. The true live loop remains
blocked until live KIS WebSocket network connection/reconnect handling, real
historical/realtime outcome datasets, tradable-universe/session gates, and a recent
full readiness report are available.

No code in this repository guarantees profit or capital protection. The controls are
engineering gates only.

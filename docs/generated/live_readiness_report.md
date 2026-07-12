# Live Readiness Report

- **Date:** 2026-07-13
- **Checker:** `python scripts/live_readiness_check.py` (fail-closed; exit 0 = ready, 1 = blocked)
- **Policy version (with corrected run.ps1 env):** `tp_61682c968224`

## Gate results (dry-run, no KIS network)
`python scripts/live_readiness_check.py --dry-run` with the run.ps1 live profile env:

| Gate | Result | Note |
|------|--------|------|
| `live_trading_safety_config` | PASS | config loads |
| `order_execution_config` | PASS | config loads |
| `live_eligible_model` | see note | requires a live-eligible artifact whose feature schema matches `LIVE_SHORT_HORIZON_SCHEMA` |
| `kis_secret_file` | PASS (dry-run) | real check needs `config/secrets` KIS keys |
| `trading_policy` | **PASS** | `policy_version=tp_61682c968224`; no FAIL conflicts |
| `live_flags` | diagnostic | reported, not enforced in dry-run |
| `kis_health` | skipped (dry-run) | real auth/account/websocket check |

**WARNING (non-blocking):** `POSITION_WEIGHT_ABOVE_ONE` — `maximum_position_weight=1.25`
(intentional for small-account single-lot affordability; RiskManager still clamps per-name
exposure).

## Enforcement proof
With the pre-fix config (`REALTIME_ALLOW_LOSS_EXIT=false`, `REALTIME_STOP_LOSS_NET=0.0`) the
`trading_policy` gate correctly **FAILS** with `STOP_LOSS_DISABLED` and the checker exits 1. The
corrected run.ps1 config clears it.

## What "live-ready" means here
Passing readiness = system integrity, data quality, order-safety and policy consistency — **not**
a profit guarantee. Small-account short-horizon day trading remains structurally
negative-expectancy after round-trip cost; the discipline minimises loss, it does not promise gain.

## Manual arming (fail-closed by default)
Live order submission additionally requires a valid, unexpired arming file:
```
python scripts/arm_live_trading.py       # creates a 15-min TTL arming file
python scripts/disarm_live_trading.py    # revokes it
```
`run.ps1` does NOT arm. The server therefore runs, collects data, analyses and shadow-evaluates
but **submits no real orders** until an operator arms explicitly and all live flags + secrets are valid.

## Run order for a real readiness check (secrets present)
```
python -m compileall src
python -m pytest
python scripts/live_readiness_check.py            # full, with KIS network
python scripts/check_kis_connection.py --account  # when secrets exist
```
Any required-gate FAIL must keep `LIVE_ORDER_SUBMIT_ENABLED` effectively off (do not arm).

# Runtime Profiles & the Arming-vs-ARM Distinction

![Profitability architecture](diagrams/profitability_architecture.svg)

## ⚠️ `arm_live_trading.py` is LIVE-TRADING ARMING, not ARM/Raspberry Pi support

`scripts/arm_live_trading.py` / `scripts/disarm_live_trading.py` are a **safety switch**:
they write/clear the manual-arming file (`config/secrets/live_trading_armed.json`, TTL
~900 s) that `LiveExecutionCoordinator` requires before any real KIS order is submitted.
"Arm" here means *arm live order submission* — it has **nothing** to do with the ARM CPU
architecture or Raspberry Pi. Before arming, `submit` raises `LiveExecutionBlocked` and
the engine records `blocked` (no real order).

## Profiles

Trading-policy config is shared across profiles (`config/profitability_policy.yaml`,
`config/dynamic_exit_policy.yaml`, `config/position_sizing_policy.yaml`). Profiles only
change runtime posture (accelerators, universe size, refresh cadence, safety flags).

| Profile | File | Role |
|---|---|---|
| Windows / Intel NPU (primary) | `run.ps1` (env defaults) | Full node: OpenVINO NPU, ontology, training, GUI kiosk |
| Small account | `config/runtime_profiles/small_account.env` | Conservative net-edge floors + sizing for a small KRW account |
| Raspberry Pi (optional) | `config/runtime_profiles/raspberrypi.env` + `scripts/run_raspberrypi.sh` | Low-power **monitor / execution-guard** node |

### Windows / Intel NPU (primary)
`run.ps1` is Windows-only (`Get-NetTCPConnection`, `Get-CimInstance`) and assumes an Intel
NPU + OpenVINO (`OPENVINO_DEVICE=NPU`, `ONTOLOGY_ACCELERATOR=NPU`). This is the primary
trading/analysis/training node.

### Small account
Source `config/runtime_profiles/small_account.env` before launch to pin the
profitability/exit/sizing env vars for a ~200k KRW account (net-edge floors 0.8%/1.2%,
tight stops, small position weight).

### Raspberry Pi (optional, disabled by default)
The Pi is a **low-power monitor / kill-switch / health node**, NOT a primary
NPU/ontology/training node. `config/runtime_profiles/raspberrypi.env` disables the Intel
NPU / OpenVINO / local LLM / large-universe scanning / high-frequency refresh, and keeps
`LIVE_ORDER_SUBMIT_ENABLED=false` + `REQUIRE_MANUAL_ARMING=true` so the Pi never submits
unless deliberately armed.

```
# On the Pi:
set -a; . config/runtime_profiles/raspberrypi.env; set +a
./scripts/run_raspberrypi.sh --port 8010
```

Recommended Pi roles: KIS account/order-status monitor, kill-switch/disarm watchdog,
lightweight dashboard, REST/WebSocket health monitor. **Not recommended**: large-universe
scanning, heavy ontology reasoning, OpenVINO NPU inference, model training, large charts.
(A separate `packaging/raspberrypi/run.sh` also exists for image packaging.)

## Arming / disarming live trading (any profile)

1. Set the live flags (run.ps1 does this on the primary node): `LIVE_TRADING_ENABLED=true`,
   `KIS_LIVE_ENABLED=true`, `KIS_PAPER_TRADING=false`, `LIVE_ORDER_SUBMIT_ENABLED=true`,
   `KILL_SWITCH_ENABLED=false`.
2. Arm: `python scripts/arm_live_trading.py` → writes the arming file (TTL-bounded).
3. Disarm / kill switch: `python scripts/disarm_live_trading.py`, or set
   `KILL_SWITCH_ENABLED=true`, or use the GUI terminate button.

Note: `config/live_trading_safety.json` sets `require_manual_arming` — confirm it is `true`
if you want arming strictly enforced under the primary `run.ps1` runtime.

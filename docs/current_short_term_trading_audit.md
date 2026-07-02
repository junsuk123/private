# Current Short-Term Trading Audit

Status: historical audit snapshot.

This file previously contained a 2026-06-27 audit of an older branch that did not yet use the current guarded live runtime. That is no longer the current operating baseline.

For the current system flow, use:

- `README.md`
- `docs/README.md`
- `docs/architecture.md`
- `docs/realtime_short_horizon_policy.md`
- `docs/live_trading_runbook.md`
- `docs/live_trading_safety_gates.md`
- `docs/live_short_horizon_model_decision.md`

Current baseline as of 2026-07-02:

- `run.ps1` starts the local server at `/account`.
- KIS realtime collection starts automatically.
- Periodic live short-horizon model training starts automatically.
- The independent realtime trading engine starts automatically.
- SELL/REDUCE is evaluated before BUY.
- Existing open SELL orders are kept unless a useful amend is needed.
- BUY is disabled when `REALTIME_BUY_ENABLED=false`.
- The live model is auxiliary and cannot approve model-only BUY.
- Live orders are limit `FinalOrder` objects submitted only through `LiveExecutionCoordinator` after runtime, KIS, source, cost, risk, idempotency, and kill-switch gates pass.

This document remains only as an index pointer for older audit references.

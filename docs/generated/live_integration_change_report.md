# Live Integration Change Report

- **Date:** 2026-07-13
- **Base commit:** `e9e9920`
- **Test baseline before changes:** `765 passed`
- **Scope of this session:** the highest-impact, individually-verified correctness/safety
  fixes plus the versioning + readiness-enforcement scaffold. Larger architectural items
  (order state machine, fill ledger, DB migrations, web.py split, ontology vocabulary merge,
  incremental training) are **explicitly deferred** with rationale in §Remaining risk — they
  are not claimed as done.

## 1. Fail-closed BUY execution preparation (P1)
`src/app/trading/realtime_trading_engine.py` — `_prepare_order_for_execution`:
- **No order-book source:** was `return final_order, True, "EXEC_NO_BOOK_SOURCE"` (submitted at
  last_price). Now urgency-aware: BUY and non-urgent SELL **blocked**
  (`None, False, "EXEC_NO_BOOK_SOURCE"`); only urgent stop/hard-stop/emergency SELL still exits
  (`EXEC_NO_BOOK_SOURCE_EMERGENCY_SELL`).
- **Exception during prep:** was `return final_order, True, "EXEC_PREPARE_SKIPPED"` (submitted at
  last_price). Now BUY / non-urgent SELL **blocked** (`EXEC_PREPARE_FAILED`); urgent SELL still
  exits (`EXEC_PREPARE_SKIPPED_EMERGENCY_SELL`).
- Tests: `tests/test_execution_risk_fixes.py::PrepareOrderFailClosedTest` (6 new).

## 2. Model threshold-merge safety direction (P3)
`src/app/models/live_signal_predictor.py::_prediction_thresholds`:
- Probability / expected-net-return **floors** now take the stricter `max()` of (artifact,
  safety); uncertainty **ceiling** takes the stricter `min()`. The safety config can now only
  tighten the gate, never loosen it.
- Tests: `tests/test_live_signal_predictor.py::ThresholdMergeSafetyTest` (2 new).

## 3. Ontology fact builder uses the real model confidence (P4)
`src/app/trading/shared_decision_engine.py`: `getattr(prediction, "confidence", 0.0)` (always
0.0 — attribute does not exist) → `getattr(prediction, "probability_success", 0.0)`.

## 4. GlobalTradeArbiter honours liquidity & spread (P4)
`src/app/graph/global_trade_arbiter.py`:
- `_buy_score` now applies `liquidity_weight` (bonus by execution quality) and
  `spread_penalty_weight` (penalty by execution quality) — previously computed then discarded.
- BUYs whose `execution_quality` is `WEAK`/`BLOCKED` are excluded from the BUY ranking even with
  a positive edge (routed to `blocked_candidates`).
- Tests: `tests/test_global_trade_arbiter.py` (2 new); existing 5 unchanged.

## 5. TradingPolicySnapshot + policy_version (P2)
New `src/app/trading/trading_policy.py`:
- Immutable snapshot resolved from the SAME env vars the exit policy / gate / sizer read;
  deterministic `policy_version` (sha256 → `tp_<12hex>`).
- `conflicts()` detects contradictory settings: `STOP_LOSS_DISABLED` (FAIL),
  `HARD_STOP_NOT_WIDER_THAN_STOP` (FAIL), `TAKE_PROFIT_DISABLED` (FAIL),
  `EMERGENCY_STOP_NOT_WIDER_THAN_HARD` (WARNING), `POSITION_WEIGHT_ABOVE_ONE` (WARNING).
- Stamped into `ProfitabilityGate` (every `ProfitabilityDecision.as_dict()` now carries
  `policy_version`) and exposed on `DynamicExitPolicy` — both read the same snapshot so their
  versions match.
- `scripts/live_readiness_check.py` gained a `trading_policy` gate: reports `policy_version`,
  **FAILS** readiness on any FAIL-severity conflict, surfaces WARNING conflicts.
- Tests: `tests/test_trading_policy_snapshot.py` (5 new, incl. gate/exit parity + decision stamp).

## 6. run.ps1 stop-loss config aligned with documented discipline (P2)
`run.ps1` (backed up before edit): the values contradicted the file's own day-trading comment.
- `REALTIME_ALLOW_LOSS_EXIT` `false → true`
- `REALTIME_STOP_LOSS_NET` `0.0 → 0.008`
- `REALTIME_HARD_STOP_LOSS` `0.03 → 0.02`
- Ordering now net 0.8% < hard 2% < emergency 5%; verified by `TradingPolicySnapshot.conflicts()`
  (no `STOP_LOSS_DISABLED`). `config/dynamic_exit_policy.yaml` stale comment corrected.

## 7. Repository hygiene (P5)
- Untracked + removed: `build/` (40 files), 7 `*.bak*` source backups (incl. two ~7 MB
  `web.py.bak_*`), 3 root `patch_*.py`, `server.log`.
- Untracked (kept on disk) 21 `data/runtime/*.log`.
- `.gitignore`: added `server.log`, `*.log`, `data/runtime/`, `build/`, `*.bak`, `*.bak_*`,
  `patch_*.py`.

## Safety posture / how "live" is gated
Live order submission requires ALL live flags true AND a valid, unexpired **manual arming file**
(`create_arming_file`, 15-min TTL). `run.ps1` does NOT create it, so the server runs and
collects/analyses/shadow-evaluates but **submits no real orders** until an operator explicitly
runs `scripts/arm_live_trading.py`. Fail-closed by default.

## Remaining risk (explicitly NOT done this session)
- **Order state machine + FillRecord ledger + startup broker reconciliation (P0):** engine
  win/loss & re-buy cooldown still use submitted limit price, not actual fills. Reported P&L is
  broker-sourced (correct), but internal performance accounting is not fill-based. Requires live
  KIS integration testing to do safely — deferred.
- **Single reconciliation worker** replacing per-order polling threads (P0) — deferred.
- **Ontology vocabulary unification** across `app.graph`/`app.ontology` (P4) — large refactor, deferred.
- **Incremental/rolling training dataset + single training lock (P3)** — deferred.
- **web.py decomposition (P5)** — 12k-line monolith, deferred (high regression risk).
- **max_position_weight > 1.0** kept intentionally (small-account single-lot affordability);
  surfaced as a readiness WARNING, not clamped.

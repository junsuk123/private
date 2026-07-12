# Latest System Audit — Personal Investment Agent

- **Base commit audited:** `e9e9920` (feature/hierarchical_macro_micro_ontology_refactor)
- **Audit date:** 2026-07-13
- **Method:** parallel read-only subsystem audits (execution, model, ontology, risk/policy,
  runtime) with file:line evidence; full `pytest` baseline (765 passed) established before
  any change.

This document records ground truth as found. The fixes applied in the same session are in
[`live_integration_change_report.md`](live_integration_change_report.md).

## 현재 커밋 해시 / Runtime entry points
- `run.ps1` → `python .\run.py --skip-startup-checks --port 8010 --strict-port` → `app.run.main()`
  → `uvicorn.run("app.web:app", app_dir="src")`. Effective bind `127.0.0.1:8010`, kiosk at `/account`.
- Workers start in the FastAPI `startup` event (not on import), gated by `AUTO_START_*` env
  flags, each a `daemon` thread with an `is_alive()` duplicate guard (per-process only).

## 실시간 데이터 흐름
- KIS realtime collector → `RealtimeMarketDataStore` (`latest_orderbook`, ticks) → feature
  frames → decision engine. Closed-market REST snapshot fallback exists. A non-realtime
  (REST) book is age-penalised (+1e6 s) so it is never treated as a fresh tradeable book for BUY.

## 실제 주문 흐름
- Decision (`SharedLiveDecisionEngine`) → `OrderIntent` → `RiskManager.validate` →
  `ProfitabilityGate` → `_prepare_order_for_execution` (exchange resolve + re-price from book +
  execution-quality gate) → `LiveExecutionCoordinator.submit_final_order` (SHA-256 idempotency).
- **Order state:** NO unified state machine; states are scattered string literals
  (`OrderStatusTracker.terminal_statuses`, raw KIS strings). Idempotency IS enforced.
- **Fill ledger:** NO strategy-side FillRecord ledger. Reported realized P&L comes from KIS
  settled-P&L endpoints (authoritative), BUT the engine's internal win/loss & re-buy cooldown
  accounting uses **submitted limit prices**, not fills (`realtime_trading_engine.py:814-832,872`).
- **Startup reconciliation:** NONE with the broker. Loss-cooldowns are seeded from the local
  journal only; pre-existing broker open orders are unknown at startup.
- **Status polling:** per-order daemon threads (`_poll_submitted_order_status`), one blocking
  `while True: sleep` loop per order — no shared worker/pool/cap.

## 모델 학습 및 추론 흐름
- Runtime: `SharedLiveDecisionEngine.evaluate_buy` calls the predictor inside try/except
  (model failure is non-fatal → heuristic fallback). No mode enum; controlled by scattered
  booleans (`LIVE_SIGNAL_MODEL_INFERENCE_ENABLED`, `REALTIME_MODEL_AUXILIARY_ONLY` default
  true = de-facto advisory, `REALTIME_REQUIRE_ONTOLOGY_FOR_MODEL_FALLBACK`).
- **Threshold merge (FIXED):** `_prediction_thresholds` combined artifact & safety thresholds
  with `min()` for the probability/return floors and `max()` for the uncertainty ceiling — the
  LOOSER direction, weakening the safety floor.
- **Label ≠ live exit:** live labels use `LIVE_LABEL_*` (TP 25 bps / SL 100 bps / 600 s) while
  the exit policy uses `DynamicExitPolicy` defaults (TP ≈80 bps floor, hard SL 800 bps). Not aligned.
- Full JSONL re-read + full re-fit every training interval; no shared training lock (only a
  thread-duplication guard). Live-eligibility = AUC ≥0.55 AND precision@k ≥0.35 AND top-k net
  expectancy > 0 (not AUC-only); no calibration/Brier gate.

## 온톨로지 추론 흐름
- Two independent packages, **no shared vocabulary**: `app.graph.*` (macro/micro +
  `GlobalTradeArbiter`) vs `app.ontology.*` (`TradingDomainReasoner`). Duplicate `IntentType`
  (3 vs 6 members), `MACRO_BLOCK_BUY` string mismatch, `MarketRegime` defined 3×.
- Ontology runs AFTER `OrderIntent` + RiskManager as an advisory annotation on BUY only; it
  can veto only when `TRADING_ONTOLOGY_ENFORCE=true` (default off).
- **Phantom attribute (FIXED):** fact builder read `prediction.confidence` (nonexistent) →
  always 0.0. Real attribute is `probability_success`.
- Macro reasoning uses a single hardcoded `market="KR"` (US not separated; the reasoner never
  reads `market`).
- **GlobalTradeArbiter (FIXED):** `liquidity_weight`/`spread_penalty_weight` were computed then
  discarded in `_buy_score`. SELL/REDUCE are correctly ranked before BUY.
- Emergency SELL correctly bypasses ontology blocking (SELL is never `BLOCK`ed).

## 리스크 및 비용 흐름
- `ProfitabilityGate` = single net-edge authority for BUY; `DynamicExitPolicy` = single exit
  authority. `PrincipalProtectionEngine` correctly ALWAYS allows SELL/REDUCE (cannot trap a stop).
- **No `policy_version` anywhere (FIXED):** TP/SL/horizon were defined in ≥8 places with
  conflicting values.
- **Stop-loss effectively disabled (FIXED in run.ps1):** `REALTIME_ALLOW_LOSS_EXIT=false` +
  `REALTIME_STOP_LOSS_NET=0.0` held losers from 0 down to the 3% hard stop — contradicting the
  file's own day-trading-discipline comment.
- Position sizing takes `min(weight, fractional_Kelly)` and clamps to `max_position_weight`,
  but does not itself cap `max_position_weight ≤ 1.0` (run.ps1 set 1.25 for small-account
  single-lot affordability; RiskManager clamps per-name exposure separately).

## 환경변수 충돌 목록
- `REALTIME_ALLOW_LOSS_EXIT` + `REALTIME_STOP_LOSS_NET` → routine stop disabled (now detected
  by `TradingPolicySnapshot.conflicts()` as a FAIL and blocked at readiness).
- run.ps1 comment vs values (loss exit, hard stop) — corrected.
- `config/dynamic_exit_policy.yaml` comment claimed run.ps1 pins 0.03/0.035 — corrected to 0.02/0.05.

## 중복 구현 목록
- Exchange resolution: strict `ExchangeResolver` (execution) vs hardcoded `NASD` default in
  `shared_decision_engine.py` (decision-time stamp; re-resolved strictly at execution).
- `IntentType`, `MarketRegime`, `MACRO_BLOCK_BUY` across `app.graph` / `app.ontology` / `technical`.
- `build/lib/app/*` stale full source copy (removed).

## fail-open 경로 목록
1. `_prepare_order_for_execution` no-book-source → submit at last_price (**FIXED → fail-closed**).
2. `_prepare_order_for_execution` exception handler → submit at last_price (**FIXED → fail-closed**).
- (By design) urgent SELL no-book fallback — preserved.

## 실제 체결 미반영 경로 목록
- Engine win/loss & re-buy cooldown use submitted limit price, not fills
  (`realtime_trading_engine.py:814-832,872`). **REMAINING** — see change report §remaining risk.
- No FillRecord ledger; no startup broker reconciliation. **REMAINING.**

## 수정 전 위험도
- **HIGH:** two fail-open BUY escape hatches (submit at stale last_price); routine stop-loss
  disabled (uncontrolled loss to 3%); safety threshold merge weakened the model gate.
- **MEDIUM:** arbiter ignored liquidity/spread; phantom model-confidence always 0.0; no policy
  version to detect config conflicts.
- **LOWER (architectural, unchanged this session):** no order state machine / fill ledger /
  startup reconciliation; per-order polling threads; ontology vocabulary split; web.py monolith.

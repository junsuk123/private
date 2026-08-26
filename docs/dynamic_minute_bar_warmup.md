# Dynamic minute-bar warmup

## Before the refactor

The live candidate path called `_candidate_has_strategy_feature_history()` with one
process-wide `REALTIME_STRATEGY_MINUTE_HISTORY_BARS` value (default 20). A symbol with
fewer bars was rejected and no historical request was made. It became ready only after
enough live prints happened to form the fixed number of bars.

The effective critical path was:

`dynamic universe -> live subscription -> wait N wall-clock minutes -> fixed history gate -> feature frame -> strategy -> GNN -> risk -> order`

Consequences observed on 2026-08-25:

- KRX after-session status reported `0/20` after a restart even though the persistent
  store contained valid older history.
- One cold symbol was indistinguishable from global system warmup.
- The strategy registry already declared 20/30-bar dependencies, while the feature/GNN
  frame required 64 bars and preferred session context; the fixed admission value did
  not represent either contract.
- There was no KIS historical minute-bar request in the candidate path, so cache reuse,
  gap-only backfill, request coalescing and provider concurrency did not exist there.
- Reconnect did not delete the SQLite cache, but no explicit gap reconciliation state or
  reconnect warmup metric existed.

## Current flow

`dynamic universe -> cheap live screening -> applicable regime strategies -> requirement resolver -> reconciled persistent cache -> optional missing-tail backfill -> per-symbol DATA_READY -> existing feature/strategy/GNN/risk/order path`

`RequirementResolver` consumes `StrategySpec.minimum_history_bars` and dependency
metadata exposed by `LiveFeatureFrameBuilder`. The current hard requirement is the
maximum active dependency; preferred observations are retained separately and never
made into a universal gate.

`HistoricalDataCoordinator` is process-wide and lazy. It provides:

- per-symbol readiness rather than global warmup;
- a priority queue (held position, armed symbol, candidate);
- one in-flight request per market/symbol/timeframe;
- bounded adaptive concurrency with AIMD response to throttling, errors, latency, CPU
  load and memory pressure;
- per-symbol failure isolation and cancellation when a candidate becomes irrelevant;
- progress-aware retry cooldowns: a provider response that does not advance the
  newest persisted timestamp backs off exponentially per symbol/timeframe, while a
  genuinely newer observation clears the cooldown immediately;
- structured audit events and metrics.

`PersistentBarRepository` uses the existing realtime SQLite database. Historical KIS
rows retain `HISTORICAL` feed metadata and can supply rolling history, while the live
row wins at an overlapping minute. Different venues are never combined. A historical
row remains non-tradeable and cannot satisfy the downstream fresh live quote/book gate.

KIS omits minutes with no trade. Therefore, a fresh cache with the required number of
observations is complete even if wall-clock minutes are absent. A stale or short cache
requests one bounded missing-tail envelope, avoiding repeated calls for no-trade minutes.

WebSocket reconnects preserve all cached rows, mark only affected symbols `STALE`, and
let the next relevant request reconcile the tail. Full rewarm is not used.

## Observability

`GET /api/realtime-trading/warmup-status` reports global operation independently of
per-symbol states, queue depth, adaptive concurrency, cache reuse, downloads, backfills,
coalesced requests, throttling, reconnects and latency.

Structured `minute_bar_warmup` audit events report requests, dependency sources, exact
missing envelopes, coalescing, failures and readiness transitions.

Sequential retry suppression is reported separately as
`warmup_requests_suppressed`, and active cooldowns expose their remaining delay and
no-progress streak. This prevents low-print extended-hours symbols from issuing a new
KIS request on every one-second engine cycle.

## Measured baseline and result

Measured against `data/store/realtime_market_data.sqlite3` on 2026-08-25:

| Measurement | Before | After |
|---|---:|---:|
| candidate with persistent valid history | wait for fixed live-bar gate after restart | immediate cache readiness |
| two cached US candidates (INTC, SOFI) | fixed gate, no preparation API | 0.0548 s wall time, both ready |
| same simultaneous request | independent callers could repeat work | same Future; one request coalesced |
| 450-row store read, 10 high-volume symbols | 19.669 ms mean | 20.420 ms mean |
| full-universe warmup dependency | implicit candidate-wide fixed rule | none |

The KIS adapter was also probed with actual configuration for both 005930 and INTC; each
returned the exact requested two-minute interval after KST/ET to UTC normalization.

## Safety invariants

This subsystem does not alter RiskManager, FinalTradeGate, principal protection,
account reconciliation, source freshness, market-session eligibility, live flags or
order submission. Historical rows prepare features only. Actual entry still requires
fresh tradeable market data and every existing downstream gate.

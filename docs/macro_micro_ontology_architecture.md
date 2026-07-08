# Hierarchical Macro–Micro Ontology Reasoning Architecture

The ontology reasoning layer is split from one mixed layer into a **Common
Trading Ontology** plus a **Macro Market Ontology** and a **Micro Symbol
Ontology**, coordinated by an **OntologyCoordinator** and a **GlobalTrade
Arbiter**. The macro layer judges market-wide flow and picks candidate symbols;
the micro layer reasons per candidate about entry/exit price, timing, risk, and
expected net return — in parallel.

> The ontology does **not predict prices directly**. It **structures**
> prediction signals and risk signals and **selects strategies**; the numeric
> prediction/scoring lives in Python (`app.technical`, `SemanticPolicyScorer`),
> and everything the layer emits is **advisory evidence**. No ontology, LLM, ML,
> or NPU output has final order authority.

## Runtime flow

```
Market/Macro data + realtime broker data
        ↓
Common Trading Ontology (trading_core.ttl — shared vocabulary)
        ↓
Macro Market Ontology  ──►  MacroMarketReasoner
        ↓  candidate symbols + allowed/blocked strategies + macro risk
┌───────────────┬───────────────┬───────────────┐
│ Micro Symbol A │ Micro Symbol B │ Micro Symbol C │  ... bounded parallel workers
└───────────────┴───────────────┴───────────────┘   (MicroSymbolReasoner each)
        ↓
OntologyCoordinator  (macro-first; BLOCK_BUY skips new BUY; held symbols always)
        ↓
GlobalTradeArbiter   (SELL/REDUCE ranked before BUY; advisory RankedTradeIntent)
        ↓
SharedLiveDecisionEngine.consume_bundle  (adapter; unchanged gate flow)
        ↓
TradingCostEngine + ProfitabilityGate + PrincipalProtectionEngine + RiskManager
        ↓
LiveExecutionCoordinator  (KIS limit orders only)
```

## Modules

| Module | Role |
|--------|------|
| `graph/macro_micro_common.py` | Shared enums (MarketRegime, MacroRiskLevel, MicroRegime, SelectedStrategy, Entry/Exit signals, ExecutionQuality, IntentType) + reason codes. |
| `graph/macro_reasoner.py` | `MacroReasoningInput/Result`, `MacroMarketReasoner` — rule-based regime/risk/sector/candidate/strategy-permission. |
| `graph/micro_reasoner.py` | `MicroReasoningInput/Result`, `MicroSymbolReasoner` — reuses the technical composite + prediction engines; macro strategy-permission gate; expected-net-return required for BUY. |
| `graph/ontology_coordinator.py` | `OntologyCoordinator`, `ParallelMicroReasoningPool`, `MacroMicroReasoningBundle` — macro-first, bounded-parallel micro, failure/timeout isolation. |
| `graph/global_trade_arbiter.py` | `GlobalTradeArbiter`, `RankedTradeIntent` (advisory-only) — SELL/REDUCE-first ranking. |
| `graph/macro_micro_config.py` | Loads `config/macro_micro_ontology.yaml` (env override, conservative fallback). |
| `graph/rdf_adapter.py` | `attach_macro_result_rdf` / `attach_micro_result_rdf` — RDF evidence projection (SHACL-validated). |
| `graph/macro_micro_replay.py` | No-look-ahead replay/validation. |
| `graph/macro_micro_feed.py` | Latest-bundle holder for the GUI. |
| `ontology/{macro_market,micro_symbol}_ontology.ttl` | Macro/micro vocabulary; import the common `trading_core.ttl`. |

## MacroMarketReasoner

Reads index trend, breadth, volatility, total trading value, macro news
severity, sector snapshots, and the candidate universe. Risk gates fire first
(news shock / high volatility / low liquidity → BLOCK_BUY). Otherwise it
classifies TREND_UP / TREND_DOWN / RANGE_BOUND, ranks sectors, selects candidate
symbols (preferring strong sectors, capped by `candidate_limit`, each with a
reason code), and sets allowed/blocked micro strategies from the per-regime
permission map. Insufficient data → `NO_TRADE_MARKET` + `BLOCK_BUY`. **It never
creates a buy order.**

## MicroSymbolReasoner

For one macro-selected symbol: evaluates held-position exit deterioration first
(SELL/REDUCE evidence), then a freshness gate (stale → BLOCKED), then the
technical composite signal + conservative `TechnicalPredictionEngine`. It maps
the selected methodology to a micro regime, applies the **macro strategy
permission gate** (a strategy in `blocked_micro_strategies` → BLOCKED), computes
execution quality (spread/liquidity vs edge), and **requires a positive expected
net return** for a `BUY_CANDIDATE` — otherwise HOLD_OR_WATCH / BLOCKED. **It may
emit OrderIntent-like evidence but never submits an order.**

## OntologyCoordinator — parallel processing

Runs the macro reasoner once. Held symbols are **always** micro-evaluated (for
SELL/REDUCE) even under `BLOCK_BUY`; new BUY candidates are dispatched only when
macro does not block. Micro workers run in a bounded `ThreadPoolExecutor`
(`max_parallel_symbols`) with a per-worker timeout; a worker exception, timeout,
or input-builder failure is isolated to that symbol (added to `failed_symbols`)
and never crashes the loop. Results are aggregated into a
`MacroMicroReasoningBundle`. CPU-only safe; NPU/OpenVINO is used only where the
reused technical/model layer already supports it.

## GlobalTradeArbiter — ranking

Splits micro results into exits (SELL/REDUCE), BUY candidates, and blocked.
**SELL/REDUCE are ranked before any BUY** (capital protection first), ordered by
downside; BUYs are then ranked by expected net return, confidence, and a
downside penalty. Output is a list of advisory `RankedTradeIntent`s — no broker
submission authority.

## How the authoritative gates are preserved

`SharedLiveDecisionEngine.consume_bundle` is a backward-compatible adapter: it
iterates the bundle's ranked intents (already SELL/REDUCE-first), routing
SELL/REDUCE through the unchanged `evaluate_exit_for_holding` and BUY through the
unchanged `evaluate_buy`. Therefore TradingCostEngine, ProfitabilityGate,
PrincipalProtectionEngine, and RiskManager remain authoritative, and
`LiveExecutionCoordinator` remains the sole (limit-only) submission path. Macro
`BLOCK_BUY` short-circuits a BUY to a rejected result **before** any gate/broker
call — it can only prevent, never authorize. The macro/micro context rides along
as advisory diagnostics only. The legacy candidate/`run_once` path still works
unchanged; the bundle path is additive.

## Configuration & safety

`config/macro_micro_ontology.yaml`: enable flags, macro/micro loop intervals
(macro slower, e.g. 60s; micro faster, e.g. 5s), candidate limit, confidence
thresholds, parallelism/timeout, per-regime strategy permissions, diagnostics.
Env overrides win; invalid values clamp to conservative defaults and are logged.

Preserved invariants: no margin/leverage/derivatives/short/credit/leveraged-ETF;
no synthetic/sample/hash data in live/paper decisions; no single indicator
triggers BUY alone; trading cost/spread/slippage/tax/quote-freshness always
considered; credentials never logged.

## Validation

See `docs/macro_micro_ontology_validation.md` and
`scripts/replay_macro_micro_ontology.py` — a no-look-ahead replay that compares
predicted vs realized cost-adjusted net edge and confirms SELL/REDUCE-first.

# Execution-risk layer & order pricing

Live short-horizon profitability is destroyed by bad fills. The decision layer
(`app.risk.manager`) stamps a **reference** price onto a `FinalOrder` (historically
`market.last_price`), but the last trade print is **not an executable price**: a BUY
posted at it may sit behind the ask and never fill; a stop SELL posted at it may fail
to exit while the book runs away. Live execution therefore **re-prices every order
from the current book, side, and exit urgency before submission**, blocks a BUY with
no usable book, and never silently routes an unknown US order to the wrong venue.

Added 2026-07-10 (commit `cb668ae`). Tests: `tests/test_execution_risk_fixes.py`,
`tests/test_trading_domain_ontology.py`.

## Modules

### `app.execution.order_pricing_policy` — executable limit price (single authority)
`ExecutionPricingPolicy.price(PricingContext) -> PricingDecision`.

| Side / reason | Price policy | Notes |
|---|---|---|
| BUY (ENTRY) | `best_ask`, capped at `EXEC_BUY_MAX_CHASE_BPS` above the reference | rounds to a marketable tick; **declines to price when no book** (caller blocks) |
| SELL TAKE_PROFIT / MODEL_EXIT / REDUCE / TIME_STOP | `best_bid` | marketable to buyers; warns if TP net < min |
| SELL STOP_LOSS / HARD_STOP / EMERGENCY | `best_bid − offset ticks` | marketable so the exit actually fills |
| urgent SELL, **no book** | `reference × (1 − EXEC_SELL_EMERGENCY_FALLBACK_OFFSET_RATE)` | still exits; warns `NO_ORDERBOOK_EMERGENCY_SELL_ALLOWED` |

`classify_action_reason(side, exit_reason, reason_codes)` maps the decision engine's
free-form `diagnostics["exit_reason"]` (e.g. `stop_loss:-1.2%`, `quick_take_profit:0.8%`)
to the normalized reason. Prices are snapped to venue ticks (KRX price bands / US `0.01`
above \$1, `0.0001` below) so the broker never rejects an off-tick limit.

### `app.execution.exchange_resolver` — routing exchange resolution & validation
`ExchangeResolver.resolve(symbol, side, account, live) -> ExchangeResolution`.
Priority: domestic 6-digit → `KR`; else holdings → env `KIS_US_EXCHANGE_MAP` →
`data/universe/us_exchange_map.csv` → quote → configured default. In **live strict
mode** an unknown US **BUY** is blocked (`US_EXCHANGE_UNKNOWN`) instead of defaulting
to NASD; a SELL is never blocked (exiting must not be prevented by routing ambiguity).

### `app.execution.execution_quality` (extended, side-aware)
- A BUY with a **missing or stale** book is blocked (`EXEC_NO_ORDERBOOK_BLOCKED`); the
  unknown-book spread is a penalty rate, never `0.0`.
- An **urgent** SELL exit is allowed without a book; a **non-urgent** no-book SELL is
  blocked (`EXEC_NO_ORDERBOOK_SELL_BLOCKED`).
- Entry-edge rejections (spread/slippage/fill-probability) apply to BUY only — they
  never veto an approved SELL exit.

### `RealtimeTradingEngine._prepare_order_for_execution`
Replaces the old BUY-only `_execution_quality_gate`. For **both** buys and sells it
resolves the exchange, fetches the order book once, re-prices via
`ExecutionPricingPolicy`, runs the quality gate, and returns a re-priced `FinalOrder`.
It **degrades to the legacy submit path when no order-book source is wired** (unit
harnesses); production always wires `RealtimeMarketDataStore`. Pricing/exchange
diagnostics are attached to the submission event under `event["execution"]`.

## Environment variables (strict live defaults are set in `run.ps1`)

| Var | Default | Meaning |
|---|---|---|
| `EXEC_REQUIRE_ORDERBOOK_FOR_BUY` | `true` | block a BUY with no usable book |
| `EXEC_REQUIRE_FRESH_ORDERBOOK_FOR_BUY` | `true` | treat a stale book as no book for a BUY |
| `EXEC_MAX_ORDERBOOK_AGE_SEC` | `3.0` | freshness window |
| `EXEC_UNKNOWN_SPREAD_PENALTY_RATE` | `0.006` | assumed spread when the book is unknown |
| `EXEC_BUY_MAX_CHASE_BPS` | `20` | max chase above reference for a BUY |
| `EXEC_ALLOW_NO_ORDERBOOK_EMERGENCY_SELL` | `true` | let urgent stops exit without a book |
| `EXEC_SELL_EMERGENCY_OFFSET_TICKS` / `EXEC_SELL_STOP_OFFSET_TICKS` | `1` | marketable stop offset |
| `EXEC_SELL_EMERGENCY_FALLBACK_OFFSET_RATE` | `0.003` | no-book urgent-sell discount |
| `KIS_US_EXCHANGE_STRICT` | `true` | block unknown US BUY in live |
| `KIS_ALLOW_DEFAULT_US_EXCHANGE_IN_LIVE` | `false` | never default unknown US BUY to NASD in live |

## Trading-domain ontology (advisory)

`app.ontology` (`trading_domain_ontology`, `trading_fact_builder`, `trading_rules`,
`trading_reasoner`) is a deterministic, explainable decision ontology. `build_trading_facts(...)`
→ `TradingDomainReasoner.reason(facts)` → `OntologyReasoningResult` with `intent`,
`candidate_intent`, `confidence`, `theory_support`, `theory_conflict`,
`required_conditions`, `blocked_by`, `reason_codes`, `recommended_order_policy`,
`validation_state`. Rules span execution feasibility, cost-adjusted edge, risk,
validation evidence, and SELL-reason distinction, configured by
`config/ontology/{execution,cost,risk,micro,macro,validation}_ontology_rules.yaml`.

It is **advisory**: it can annotate, weaken, or block — never authorize. It is wired
into `SharedLiveDecisionEngine.evaluate_buy` as `diagnostics["ontology_reasoning"]`;
blocking is opt-in via `TRADING_ONTOLOGY_ENFORCE` (default **off**). RiskManager,
ProfitabilityGate, and FinalTradeGate remain the sole execution gates. This complements
the existing `app.graph` semantic layer and the macro–micro reasoners; it does not
replace them.

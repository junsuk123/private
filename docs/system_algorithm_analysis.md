# System Algorithm Analysis

This document summarizes the current algorithmic design of the personal investment analysis system as implemented in `src/app`.

The system is a safe, read-only-first investment research and paper-trading simulator. It combines public research collection, local storage, indicator snapshots, ontology reasoning, goal feasibility scoring, deterministic strategy generation, deterministic risk validation, and mock/streaming simulation. Live automated trading is intentionally disabled.

## 1. Top-Level Flow

The main runtime path is:

```text
Research sources
  -> event/market/macro normalization
  -> local SQLite store
  -> analysis context
  -> indicators
  -> ontology graph
  -> ontology reasoning paths
  -> strategy signals
  -> order intents
  -> risk validation
  -> mock KIS / streaming simulation / UI output
```

Primary entry points:

- `run.py` inserts `src` into `sys.path` and calls `app.run.main`.
- `src/app/run.py` runs startup checks, selects an available web port, then starts `uvicorn app.web:app`.
- `src/app/web.py` exposes the FastAPI web UI and API endpoints.
- `src/app/cli.py` exposes demo, research, accelerated-demo, and synthetic-data commands.
- Current web runtime uses FastAPI/Uvicorn and SQLite-backed local stores.

## 2. Data Collection Algorithm

Implemented mainly in `src/app/research/service.py`.

`ResearchService.run_from_config(path)` loads a JSON config and calls `run(config, base_dir)`.

Supported sources:

- full listed-stock universe catalogs for US/overseas and domestic KRX markets
- RSS feeds via `RssNewsCollector`
- static HTML pages via `HtmlResearchCollector`
- dynamic pages via `DynamicPageCollector`
- Stooq market data
- Yahoo chart market data, subject to robots.txt restrictions
- FRED macro data
- ECOS macro data
- OpenDART disclosures

For each configured source:

1. Load the configured listed universe when `listed_universe.enabled=true`.
2. Add every listed symbol to the internal ticker-recognition map.
3. Store a `listed_universe_catalog` raw record containing the full universe count and the current rotating batch.
4. Build a source key such as `rss:<url>` or `yahoo_chart:<symbol>`.
5. Run the collector action.
6. Normalize output into one of:
   - `ClassifiedEvent`
   - `RawSourceRecord`
   - `MarketSnapshot`
   - `MacroMetricRecord`
7. If a source fails and retry is enabled, enqueue a `_RetryJob`.
8. Drain the retry queue with configurable attempts and backoff.
9. Return `ResearchRunResult` with diagnostics.

The full listed universe is intentionally separated from expensive per-symbol network crawling. The system tracks all symbols and creates a rotating `listed_universe_reference` market snapshot batch for graph/analysis visibility. Detailed external fetching remains bounded so the UI and API do not hang while trying to fetch thousands of symbols in a single request.

Event classification uses:

- `JsonEventLLMClassifier` when configured through environment variables.
- Local/OpenAI-compatible/embedded model clients when available.
- Keyword fallback in `app.data.classifier`.
- A capped ticker hint list for LLM prompts, while the complete universe remains available to deterministic ticker matching.
- The default local LLM path uses Ollama with `qwen2.5:1.5b-instruct` when available. LLM calls are capped per refresh to keep collection responsive.

Diagnostics include event counts, skipped sources, live/local source counts, latest observed timestamp, per-ticker sentiment counts, listed-universe total count, known ticker count, rotating batch size, and rotating cursor position.

## 3. Storage Algorithm

Implemented in `src/app/storage/local_store.py`.

The storage layer uses SQLite at:

- live mode: `data/live/store/research.sqlite3`
- simulation mode: `data/sim/store/research.sqlite3`

The table schema is generic:

```text
records(kind, record_key, observed_at, inserted_at, payload)
primary key: (kind, record_key)
```

Save algorithm:

1. Prune records older than `RESEARCH_RETENTION_DAYS`.
2. Convert dataclass records to JSON-compatible dictionaries.
3. Compute a stable `record_key` per kind:
   - events: `event_id`
   - raw records: source id/url plus retrieval time
   - market snapshots: ticker/source/retrieval time
   - graph triples: subject/predicate/object/evidence
   - reasoning paths: `path_id`
4. Insert with `insert or ignore`.
5. In live mode, reject simulated rows using `_is_simulated_row`.

This gives deduplication, retention pruning, and live/simulation separation.

## 4. Analysis Context Construction

Implemented in `src/app/pipeline.py`.

`build_analysis_context(research_result, stored_research)` creates a complete decision snapshot:

1. Load sample account.
2. Load sample markets.
3. Merge stored and live market snapshots by ticker.
4. Build indicators with `build_sample_indicators`.
5. Merge stored, live, and sample research events by event id.
6. Build ontology graph.
7. Run ontology inference.
8. Build reasoning paths for tickers.
9. Build portfolio report.
10. Generate rule-based strategy signals.
11. Generate order intents.
12. Run deterministic risk validation.

The result is an `AnalysisContext` containing account, markets, indicators, events, graph, reasoning paths, report, signals, intents, risk results, and ontology runtime status.

## 5. Indicator Algorithm

The current production context uses `build_sample_indicators` in `src/app/indicators/engine.py`. It creates `IndicatorSnapshot` values from market snapshots and sample assumptions.

The broader feature system under `src/app/features` includes:

- semantic feature generation
- parameter tuning context features
- AI semantic layer prototypes
- OHLCV label/dataset builders

Those modules support future model training and semantic indicator expansion, but the main web decision path currently uses the simpler `IndicatorSnapshot` pipeline.

## 6. Ontology Graph Algorithm

Implemented in:

- `src/app/graph/builders.py`
- `src/app/graph/event_mapper.py`
- `src/app/graph/knowledge_graph.py`
- `src/app/graph/reasoner.py`

The graph is a set of unique triples:

```text
(subject, predicate, object, evidence_id)
```

Market graph construction:

For each market:

- company `hasTicker` ticker
- company `belongsToSector` sector
- ticker `isListedOn` market

Indicator-derived relations:

- operating income growth > 15% -> `supportsSignal EarningsGrowth`
- operating margin > 15% -> `supportsSignal ProfitabilityQuality`
- PER > 20 -> `contradictsSignal ValuationDiscipline`
- macro risk score > 0.40 -> `increasesRiskOf MacroRateRisk`
- 20-day volatility > 0.06 -> `increasesRiskOf VolatilityRisk`

Event-derived relations are added by `add_events_to_graph`, linking classified news/disclosures to tickers, sectors, positive impact, or risk concepts.

Ontology inference:

- If a subject has both `EarningsGrowth` and `ProfitabilityQuality`, add `supportsSignal BuyCandidate`.
- If a subject has `BuyCandidate` and `MacroRateRisk`, add `contradictsSignal AggressiveBuy`.
- If a subject has macro, volatility, or negative event risk, add `supportsSignal RiskAdjustedSizing`.

Reasoning path score:

```text
confidence = clamp(0.05, 0.95,
  0.40
  + weighted_support
  - weighted_contradiction
  - weighted_risk
)
```

Conclusion:

- confidence >= 0.58 -> `BuyCandidate`
- otherwise -> `HoldOrWatch`

## 7. Strategy Signal Algorithm

There are two strategy layers.

### 7.1 Rule-Based Strategy

Implemented in `src/app/strategy/rule_based.py`.

For each ticker:

```text
score starts at 0
+1.0 if revenue_growth > 8%
+1.0 if operating_income_growth > 15%
+1.0 if operating_margin > 15%
-0.8 if PER > 20
-0.6 if macro_risk_score > 0.40
-1.0 if volatility_20d > 0.06
```

Decision:

- score >= 1.8 -> `BUY`
- otherwise -> `HOLD`

Confidence:

```text
confidence = clamp(0.0, 0.85, 0.45 + score * 0.1)
```

BUY signals are converted to `OrderIntent` with:

```text
suggested_weight = clamp(0.01, 0.05, confidence * 0.05)
```

### 7.2 Goal-Directed Strategy

Implemented in `src/app/strategy/goal_directed.py`.

This layer is used after the user selects a target return and period.

Annualized required return:

```text
annualized_required_return =
  (1 + target_return_rate) ** (365 / period_days) - 1
```

Per-market score includes:

- ontology support/risk/contradiction
- RSI
- volume ratio
- earnings growth
- operating margin
- valuation
- volatility
- macro risk
- annualized target difficulty
- feasibility penalty/bonus

Important scoring examples:

```text
ontology support: + min(1.6, support_count * 0.35)
ontology risk:    - min(1.4, risk_count * 0.45)
ontology contra:  - min(1.2, contra_count * 0.40)
RSI 45..68:       +0.9
RSI < 32:         +0.4
RSI > 74:         -1.2
volume >= 1.15:   +0.55
volume < 0.70:    -0.45
PER > 25:         -0.65
PER < 18:         +0.35
volatility > .06: -1.2
volatility < .035:+0.35
macro risk > .55: -0.9
```

Target difficulty penalty:

```text
score -= min(1.3, max(0, annualized_required_return - 0.20) * 1.2)
```

Action thresholds:

- score >= 2.2 -> `BUY`
- score <= -1.1 -> `SELL`
- score <= -0.35 -> `REDUCE`
- otherwise -> `HOLD`

If feasibility is below 35 and score is low, action is forced toward `REDUCE`.

Goal-directed intent sizing:

```text
max_goal_weight = min(0.06, max(0.015, 0.025 + target_return_rate))
BUY suggested weight =
  min(max_goal_weight, max(0.01, confidence * 0.04 + rank_bonus * 0.004))
REDUCE suggested weight =
  current_weight * 0.50
SELL suggested weight =
  0
```

## 8. Goal Feasibility Algorithm

Implemented in `src/app/goals/negotiation.py`.

Input:

- target return rate or target profit amount
- period days
- account snapshot
- market snapshots
- indicators
- strategy signals
- ontology graph

Requested return:

```text
target_return_rate if provided
else target_profit_amount / account_equity
```

Annualized required return:

```text
(1 + requested_return_rate) ** (365 / period_days) - 1
```

Market support:

```text
base 24
+ average BUY signal confidence contribution
+ growth / operating income / margin contribution
cap at 78
```

Risk pressure:

```text
volatility pressure, capped at 28
macro pressure, capped at 22
cash pressure: +10 if cash_weight < 30%
```

Annualized drag:

```text
55 / (1 + exp(-5 * (annualized_required_return - 0.18)))
```

Feasibility:

```text
feasibility = clamp(3, 96,
  market_support - risk_pressure - annualized_drag
)
```

Compromise goals:

- requested target
- lower return target at 60%
- longer period target
- balanced target at 75% return and longer period

The UI sorts compromise goals by feasibility descending.

## 9. Risk Manager Algorithm

Implemented in `src/app/risk/manager.py`.

The risk manager is deterministic and is the final gate before a `FinalOrder`.

Checks:

- LLM direct execution is blocked.
- live trading is disabled.
- action is BUY/SELL/REDUCE.
- only limit orders are allowed.
- daily loss limit is not breached.
- trade count limit is not breached.
- liquidity is above minimum average daily trading value.
- volatility is below maximum.
- duplicate pending order is blocked.
- source data is present and market price is valid.
- restricted products are blocked.
- single-stock weight cap.
- intraday position weight cap.
- sector weight cap.
- deposit/cash reserve checks.

Weight adjustment:

```text
adjusted_weight = min(
  intent.suggested_weight,
  max_single_stock_weight,
  max_intraday_position_weight for BUY
)
```

For BUY:

```text
target_value = equity * adjusted_weight
buy_amount = max(0, target_value - current_value)
quantity = floor(buy_amount / last_price)
```

For SELL/REDUCE:

```text
sell_value = current_value for SELL
sell_value = max(0, current_value - target_value) for REDUCE
quantity = floor(sell_value / last_price)
```

Orders with quantity <= 0 are rejected.

Approved orders are always `LIMIT` and `manual_approval_required=True`.

## 10. Mock KIS Trading Algorithm

Implemented in:

- `src/app/trading/mock_program.py`
- `src/app/execution/kis_mock.py`

Mock trading cycle:

```text
goal
  -> mock LLM judgment
  -> ontology evidence
  -> goal execution plan
  -> risk manager validation
  -> MockKisDevelopersApi place_limit_order
  -> mock fill check
  -> portfolio update
```

Mock LLM judgment is deterministic:

```text
score =
  indicator_score
  + ontology_support_count * 0.25
  - ontology_risk_count * 0.30
```

Top 5 tickers are selected.

Mock KIS fill rule:

- BUY fills if limit price >= mock market price.
- SELL fills if limit price <= mock market price.
- BUY also requires sufficient mock cash.
- Filled BUY updates cash, quantity, average price, and holdings.
- Filled SELL updates cash and removes/reduces holdings.

No real brokerage API is called.

## 11. Streaming Simulation Algorithm

Implemented in `src/app/backtesting/streaming_demo.py` and exposed through `src/app/web.py`.

The simulation test path is:

```text
POST /api/operation-mode/start
  mode = simulation_testing
  target_return_rate
  period_minutes
  -> create StreamingAcceleratedDemo in memory
  -> return demo_id

UI loop:
  POST /api/streaming-demo/step
  -> run one visible simulation step
  -> update portfolio, holdings, trades, progress, return rate
```

Current UI behavior:

- The user-facing acceleration option has been removed.
- The simulation advances by UI step calls, but the server returns `status="waiting"` until the next real-time one-minute bar is due.
- Missing/stale `demo_id` is treated as `status="expired"` with HTTP 200, not a 404 error.
- The UI stops the loop when it receives `expired`.
- The top mock-return card is updated from the streaming account state on every successful step.
- URL parameters `target_return_rate` and `period_minutes` are copied into the goal form on page load.

Initialization:

1. Load the configured US/overseas and domestic KRX listed-stock universe.
2. Generate synthetic one-minute charts for the simulation universe.
3. Add warmup bars.
4. Set initial cash.
5. Clear holdings and trade history.
6. Use realtime mode by default: one visible simulation bar requires one wall-clock minute.

Each step:

1. Select a bounded active ticker batch from the full universe, plus current holdings.
2. Get prices at the current synthetic bar for the active batch.
2. Build account snapshot from cash and holdings.
3. Build market snapshots and indicators at the step.
4. Build ontology graph and infer.
5. Build a goal-directed execution plan using the target return and period.
6. Rank intents:
   - SELL/REDUCE first
   - then higher confidence
7. Validate each intent through `RiskManager` with simulation-specific rules.
8. Apply approved orders directly to simulated cash/holdings.
9. Record `SimulatedTrade`.
10. Recompute account value and return rate.
11. Return visible step, raw chart bar, active/universe ticker counts, prices, progress, account, holdings, trades, and final results if complete.

Return calculation:

```text
account_value = cash + sum(quantity * current_price)
return_rate = (account_value - initial_cash) / initial_cash
progress = visible_steps_completed / period_minutes
```

The web UI updates:

- left simulation status panel
- cash / invested / profit / return rate
- top mock-return card
- recent executions table
- positions table
- return sparkline and system-flow progress cards

## 12. Web API and UI Algorithm

Implemented in `src/app/web.py`.

Startup:

- starts a background live worker
- live worker refreshes research, stores records, rebuilds analysis context, and updates progress state

Important endpoints:

- `GET /api/status`: portfolio/account status
- `GET /api/research`: events, graph triples, reasoning paths
- `GET /api/research/diagnostics`: source/store/runtime diagnostics
- `GET /api/ontology/graph`: graph payload for 3D visualization
- `GET /api/realtime/runtime`: NPU/CPU runtime status and operation mode
- `POST /api/live-snapshot`: threadpool snapshot response with optional goal assessment
- `POST /api/assess-goal`: goal feasibility and compromise goals
- `POST /api/start`: start mock KIS paper-trading run after accepted goal
- `POST /api/operation-mode/start`: live/simulation training/testing mode
- `POST /api/streaming-demo/step`: one streaming simulation step

Concurrency note:

`/api/live-snapshot` uses `run_in_threadpool` so periodic UI refreshes do not block the event loop and delay user actions such as learning/test mode starts.

Port behavior:

`app.run` selects the next free port if the requested port is occupied. If port `8000` is already in use, it may print and use `8001`.

Use the printed Web UI URL. Opening a neighboring port that was not printed will produce a connection error because no server is listening there.

## 13. Runtime Modes

Implemented in `src/app/realtime/mode_manager.py` and `src/app/runtime/environment.py`.

Modes:

- `simulation_training`
- `simulation_testing`
- `live_training`
- `live_trading`

Simulation modes:

- use `data/sim`
- allow synthetic data
- block live orders
- store models under `data/sim/models`

Live modes:

- use `data/live`
- reject synthetic rows in live store
- keep live orders blocked
- live trading remains a guarded/manual-approval boundary

## 14. Safety Model

The system is designed so that:

- LLM-like judgment is advisory only.
- LLM or mock LLM never directly submits real orders.
- `OrderIntent` must pass `RiskManager`.
- `FinalOrder` requires limit order mode.
- `manual_approval_required=True` is retained.
- live trading is disabled by default.
- restricted products are blocked.
- simulation and live storage are separated.

## 15. Known Algorithmic Limitations

Current limitations visible in the implementation:

- Main production indicators are still sample/snapshot-based, not full historical production indicators.
- Some live market chart sources can be skipped due to robots.txt or missing Playwright.
- Full listed-stock universe catalogs are tracked, but expensive per-symbol external crawling is intentionally rotated in bounded batches.
- Streaming simulation sessions are in memory; they expire on server restart.
- The mock LLM judgment is deterministic scoring, not a real independent model decision.
- Risk rules are deterministic and conservative, but not a full brokerage compliance engine.
- The system is research infrastructure and not financial advice.

## 16. Key Files

- `src/app/run.py`: startup and web server launch
- `src/app/web.py`: FastAPI UI/API orchestration
- `src/app/research/service.py`: source collection and diagnostics
- `src/app/storage/local_store.py`: SQLite storage and live/sim separation
- `src/app/pipeline.py`: analysis context pipeline
- `src/app/graph/builders.py`: market graph construction
- `src/app/graph/reasoner.py`: ontology inference and reasoning paths
- `src/app/goals/negotiation.py`: feasibility and compromise scoring
- `src/app/strategy/rule_based.py`: baseline strategy signals
- `src/app/strategy/goal_directed.py`: target-aware strategy
- `src/app/risk/manager.py`: deterministic risk validation
- `src/app/trading/mock_program.py`: mock trading cycle
- `src/app/execution/kis_mock.py`: mock KIS order/fill/portfolio behavior
- `src/app/backtesting/streaming_demo.py`: stepwise simulation

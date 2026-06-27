# System Algorithm Analysis

This document summarizes the current algorithmic design implemented under `src/app`.

The system is a safe realtime-only investment research, learning, hypothetical-testing, paper-trading, and readiness-check framework. It combines public research collection, local SQLite storage, indicator snapshots, ontology screening, ontology reasoning, goal feasibility scoring, deterministic strategy generation, deterministic risk validation, mock KIS paper trading, KIS paper/live-readiness boundaries, realtime hypothetical testing, and in-memory paper-trading simulation. Live automated trading is intentionally disabled.

## 1. Top-Level Flow

```text
Research sources
  -> event/market/macro normalization
  -> data/store SQLite persistence
  -> analysis context
  -> ontology_filter_1 candidate screening
  -> indicators + time-synchronized frames
  -> ontology graph
  -> reasoning paths
  -> strategy signals
  -> order intents
  -> deterministic risk validation
  -> mock KIS / KIS paper-readiness / realtime hypothetical test / paper-trading simulation / UI output
```

Primary entry points:

- `run.py`: inserts `src` into `sys.path` and calls `app.run.main`.
- `src/app/run.py`: performs startup checks, resolves port behavior, and starts `uvicorn app.web:app`.
- `run.ps1`: starts a strict local app server on port `8010`, applies safe realtime/NPU defaults, and opens a managed browser.
- `src/app/web.py`: FastAPI web UI and API orchestration.
- `src/app/cli.py`: demo, research, accelerated-demo, and synthetic-data commands.

## 2. Data Collection Algorithm

Implemented mainly in `src/app/research/service.py`.

Supported sources:

- US/overseas and KRX listed-stock universe catalogs
- RSS feeds
- static HTML pages
- dynamic pages through Playwright when installed
- Stooq daily/latest market data
- Yahoo chart market data, subject to robots/runtime restrictions
- Alpha Vantage daily market data
- FRED macro data
- ECOS macro data
- OpenDART disclosures

For each configured run:

1. Load the configured listed universe when `listed_universe.enabled = true`.
2. Add listed symbols to deterministic ticker recognition.
3. Store a `listed_universe_catalog` raw record.
4. Create deterministic `listed_universe_reference` market snapshots for the current rotating batch.
5. Build source keys such as `rss:<url>`, `html:<url>`, `yahoo_chart:<symbol>`.
6. Run each collector action.
7. Normalize output into `ClassifiedEvent`, `RawSourceRecord`, `MarketSnapshot`, or `MacroMetricRecord`.
8. Optionally classify event text through a JSON-returning event LLM.
9. Fall back to keyword classification when no LLM is available or the LLM call fails.
10. Queue eligible failed sources for retry.
11. Drain retries with configured attempts and backoff.
12. Deduplicate events/raw records and return `ResearchRunResult` with diagnostics.

The full universe is tracked separately from expensive per-symbol collection. `RESEARCH_UNIVERSE_BATCH_SIZE` controls how many reference symbols are surfaced per refresh, and `data/research_universe_cursor.json` stores the rotating cursor.

LLM event classification supports:

- remote OpenAI-compatible chat completions
- local OpenAI-compatible servers such as Ollama
- embedded Hugging Face causal/chat models
- embedded multimodal Transformers models
- embedded OpenVINO/Optimum Intel models for NPU-oriented local inference

Prompt ticker hints are capped by `LLM_EVENT_KNOWN_TICKER_PROMPT_LIMIT`, while deterministic matching can still use the complete known ticker map.

Source trust and quality are normalized through `app.data.source_policy`. Broker, exchange, disclosure, and official macro sources receive higher default trust; unofficial chart endpoints, dynamic pages, sample, synthetic, and unknown sources are downgraded or blocked for live decisions.

## 3. Storage Algorithm

Implemented in `src/app/storage/local_store.py`.

The active web runtime writes to:

```text
data/store/research.sqlite3
```

Generic table:

```text
records(kind, record_key, observed_at, inserted_at, payload)
primary key: (kind, record_key)
```

Save algorithm:

1. Prune rows older than `RESEARCH_RETENTION_DAYS`.
2. Convert dataclasses to JSON-compatible dictionaries.
3. Reject simulated/synthetic rows.
4. Compute stable keys:
   - events: `event_id`
   - raw records: source id/url/payload prefix plus retrieved time
   - market snapshots: ticker/source/retrieved time
   - macro metrics: name/observed time
   - realtime quotes: ticker/market/source/observed time
   - realtime executions: ticker/market/source/trade id or execution tuple
   - graph triples: subject/predicate/object/evidence
   - reasoning paths: `path_id`
5. Insert with `insert or ignore`.

`ModelArtifactStore` writes JSON artifacts under:

```text
data/models/<model_family>/<model>.<timestamp>.json
data/models/<model_family>/<model>.latest.json
```

It refuses artifacts marked `simulated=True`.

## 4. Analysis Context Construction

Implemented in `src/app/pipeline.py`.

`build_analysis_context(research_result, stored_research)` builds a complete decision snapshot:

1. Load sample account and sample fallback market data.
2. Merge stored markets and fresh research markets by ticker.
3. Run `ontology_filter_1` over lightweight market snapshots.
4. Keep selected candidates plus priority tickers such as `005930`, `000660`, `AAPL`, `MSFT`, `NVDA`, `SPY`, and `QQQ`.
5. Build lightweight `IndicatorSnapshot` values.
6. Merge stored, live, and sample research events.
7. Limit events by relevance, directionality, classification confidence, labels, facts, and recency.
8. Merge raw records, macro metrics, realtime quotes, and realtime executions into time-synchronized frames.
9. Build ontology graph from markets, indicators, events, temporal frames, candidate-selection metadata, and tuning metadata.
10. Add ontology tuning relationships for risk-adaptive sizing, momentum breakout thresholding, and event-risk weighting.
11. Run ontology inference and build reasoning paths.
12. Build portfolio report.
13. Generate baseline rule-based strategy signals.
14. Generate order intents.
15. Validate intents through `RiskManager`.

`AnalysisContext` includes account, markets, indicators, events, graph, reasoning paths, report, signals, intents, risk results, ontology runtime status, candidate selection, parameter tuning, and temporal frames.

## 5. Candidate Screening Algorithm

Implemented in `src/app/trading_pipeline.py`.

`ontology_filter_1` screens a large universe before heavier chart/strategy analysis.

Inputs are `LightweightMarketSnapshot` records:

- current price
- price change rate
- trading value
- trading volume
- volume change rate
- market cap
- foreign/institution/retail net buy values
- upper-limit proximity
- new 52-week-high flag
- halt status
- management-stock status
- liquidity score

Reject immediately when:

- trading is halted
- management-stock status is active
- trading value is below `min_trading_value`
- liquidity score is below `min_liquidity_score`

Candidate score:

```text
score =
  liquidity_score * 0.35
  + max(0, volume_change_rate) * 0.18
  + max(0, price_change_rate) * 3.0
  + 0.18 if high trading value and volume surge
  + 0.20 if foreign and institution net buying align with momentum
  + 0.12 if upper-limit-near or new 52-week high
```

The result records selected candidates, rejected stocks, per-stock reasoning traces, latency, API call count, full universe count, and chart-fetch scope. Results can be cached for a short TTL.

## 6. Indicator and Feature Algorithm

The primary web decision path uses `build_sample_indicators` from `src/app/indicators/engine.py`. It creates interpretable `IndicatorSnapshot` values for markets in the active analysis set.

The broader feature subsystem under `src/app/features` provides:

- formula-only OHLCV indicators
- semantic feature generation
- hybrid formula plus AI semantic states
- parameter tuning context
- no-lookahead dataset rows and labels

Those modules are ready to enrich the main pipeline, but the current dashboard decisions still rely primarily on lightweight `IndicatorSnapshot` plus ontology reasoning.

## 7. Ontology Graph Algorithm

Implemented in:

- `src/app/graph/builders.py`
- `src/app/graph/event_mapper.py`
- `src/app/graph/knowledge_graph.py`
- `src/app/graph/reasoner.py`
- `src/app/time_series.py`

Graph triples are unique:

```text
(subject, predicate, object, evidence_id)
```

Market and indicator relations include:

- company `hasTicker` ticker
- company `belongsToSector` sector
- ticker `isListedOn` market
- growth and margin support positive signals
- high PER can contradict valuation discipline
- macro risk and volatility increase risk nodes
- NPU scores can add support/risk/contradiction-style evidence where available

Event relations link classified news/disclosures to tickers, sectors, positive impact, negative risk, and event labels/key facts.

Temporal-frame relations link time buckets to tickers, events, quotes, executions, market snapshots, raw sources, macro context, and impact score.

Pipeline metadata adds relations for:

- `OntologyFilter1:LightweightScreening`
- `SelectiveChartFetching`
- `SemanticFeatureExtraction`
- `OntologyFilter2:EntryDecision`
- `AIPredictionSmallSet`
- `OntologyFilter3:FinalRiskApproval`
- ontology tuning modes and tuned parameter values

Reasoner inference:

- earnings growth plus profitability quality can support `BuyCandidate`
- buy candidate plus macro rate risk can contradict `AggressiveBuy`
- macro, volatility, or negative-event risk can support `RiskAdjustedSizing`

Reasoning confidence:

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

## 8. Strategy Algorithms

### 8.1 Rule-Based Strategy

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

BUY signals become `OrderIntent` records with:

```text
suggested_weight = clamp(0.01, 0.05, confidence * 0.05)
```

### 8.2 Goal-Directed Strategy

Implemented in `src/app/strategy/goal_directed.py`.

Annualized required return:

```text
(1 + target_return_rate) ** (365 / period_days) - 1
```

Score inputs include:

- ontology support/risk/contradiction
- RSI
- volume ratio
- earnings growth
- operating margin
- valuation
- volatility
- macro risk
- annualized target difficulty
- target feasibility

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

If feasibility is below 35 and score is weak, action is forced toward `REDUCE`.

BUY sizing:

```text
max_goal_weight = min(0.06, max(0.015, 0.025 + target_return_rate))
suggested_weight =
  min(max_goal_weight, max(0.01, confidence * 0.04 + rank_bonus * 0.004))
```

REDUCE sizing is half of current weight. SELL target weight is zero.

## 9. Goal Feasibility Algorithm

Implemented in `src/app/goals/negotiation.py`.

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
+ BUY-signal confidence contribution
+ growth / operating income / margin contribution
cap at 78
```

Risk pressure includes volatility, macro risk, and low cash pressure.

Annualized drag:

```text
55 / (1 + exp(-5 * (annualized_required_return - 0.18)))
```

Feasibility:

```text
clamp(3, 96, market_support - risk_pressure - annualized_drag)
```

Compromise goals include the requested target, a lower return target, a longer period target, and a balanced lower-return/longer-period target.

## 10. Risk Manager Algorithm

Implemented in `src/app/risk/manager.py`.

Checks:

- LLM direct order execution blocked
- live trading disabled
- action is BUY, SELL, or REDUCE
- only limit order mode
- daily loss limit
- trade count limit
- minimum average daily trading value
- maximum volatility
- duplicate pending ticker
- source data and valid last price
- restricted products blocked
- max single-stock weight
- max intraday BUY weight
- max sector weight
- deposit and cash reserve

Adjusted weight:

```text
adjusted_weight = min(
  intent.suggested_weight,
  max_single_stock_weight,
  max_intraday_position_weight for BUY
)
```

BUY:

```text
target_value = equity * adjusted_weight
buy_amount = max(0, target_value - current_value)
quantity = floor(buy_amount / last_price)
```

SELL/REDUCE:

```text
sell_value = current_value for SELL
sell_value = max(0, current_value - target_value) for REDUCE
quantity = floor(sell_value / last_price)
```

Quantity `<= 0` is rejected. Approved orders are `LIMIT` orders with `manual_approval_required=True`.

## 11. Mock KIS Trading Algorithm

Implemented in:

- `src/app/trading/mock_program.py`
- `src/app/execution/kis_mock.py`

Mock trading cycle:

```text
goal
  -> deterministic mock LLM-style judgment
  -> ontology evidence
  -> goal execution plan
  -> RiskManager validation
  -> MockKisDevelopersApi place_limit_order
  -> fill check
  -> mock portfolio update
```

Mock LLM-style score:

```text
score =
  indicator_score
  + ontology_support_count * 0.25
  - ontology_risk_count * 0.30
```

No real brokerage API is called.

## 11.1 KIS Developers Adapter

Implemented in `src/app/execution/kis_real.py`.

The adapter implements the KIS Developers domestic cash-stock REST contract behind the same broker boundary as mock execution:

- environment/secrets loading from `config/secrets/kis_api_keys.env`
- paper base URL `https://openapivts.koreainvestment.com:29443`
- live base URL `https://openapi.koreainvestment.com:9443`
- `/oauth2/tokenP` access-token issuance
- `/uapi/hashkey` for cash-order POST bodies
- domestic cash-order TR IDs `VTTC0012U` / `VTTC0011U` for paper buy/sell
- domestic cash-order TR IDs `TTTC0012U` / `TTTC0011U` for live buy/sell
- order-status polling with `VTTC8001R` or `TTTC8001R`
- balance lookup with `VTTC8434R` or `TTTC8434R`

`KIS_PAPER_TRADING=true` selects the virtual domain. `KIS_LIVE_ENABLED=false` blocks order, status, and account calls by default. `scripts/check_kis_connection.py` can issue a token-only check or a read-only balance check with `--account`.

## 12. Realtime Learning and Testing Algorithms

Implemented in `src/app/realtime/learning.py`.

Learning examples are built from adjacent `TimeSynchronizedTickerFrame` records and current strategy signals:

```text
realized_return = (current_price - previous_price) / previous_price * action_direction
realized_pnl = (current_price - previous_price) * action_direction
label = realized_pnl > 0
```

Feature snapshots include:

- impact score
- event count
- quote count
- execution count
- macro count
- signal confidence
- signal score

Hypothetical testing creates one-share hypothetical trades from adjacent frames when a signal action is BUY, SELL, or REDUCE. It reports trade count, winning trades, win rate, realized PnL, and `orders_submitted = 0`.

Artifacts are saved under:

```text
data/models/realtime_supervised/
data/models/hypothetical_testing/
```

The model support layer also includes no-lookahead training-plan summaries, ranked-signal evaluation summaries, a CPU NumPy inference backend, an OpenVINO/NPU backend with CPU fallback, and an OpenVINO export hook. The export hook does not convert an arbitrary model by itself; a concrete trained-model adapter must supply the conversion.

## 13. Paper-Trading Simulation Algorithm

Implemented in `src/app/backtesting/streaming_demo.py` and exposed through `src/app/web.py`.

Start:

```text
POST /api/paper-trading/start
target_return_rate
period_minutes
initial_cash
```

Step:

```text
POST /api/paper-trading/step
demo_id
```

Initialization:

1. Load the global listed universe.
2. Optionally cap it with `SIM_STREAMING_UNIVERSE_LIMIT`.
3. Build lightweight snapshots.
4. Run `ontology_filter_1`.
5. Generate synthetic one-minute charts for selected candidate tickers.
6. Add warmup bars.
7. Initialize cash, holdings, trade history, and realtime step timing.

Each due step:

1. Get synthetic prices at the current bar.
2. Build account from cash and holdings.
3. Build market snapshots and indicators.
4. Run NPU/CPU ontology classifier scores over the universe.
5. Select candidate tickers by ontology/NPU score, keeping current holdings.
6. Build ontology graph and infer.
7. Build goal-directed execution plan.
8. Rank SELL/REDUCE before BUY, then by confidence.
9. Validate up to 10 intents through `RiskManager`.
10. Apply approved orders to simulated cash/holdings.
11. Record `SimulatedTrade`.
12. Liquidate remaining holdings on the final step.
13. Return account value, return rate, progress, prices, holdings, trades, and NPU status.

Step timing:

```text
seconds_until_next_step = 60 / scale_factor - elapsed_since_next_due
```

With the current web start path, scale factor is `1.0`, so one visible synthetic minute is due per wall-clock minute.

Return:

```text
account_value = cash + sum(quantity * current_price)
return_rate = (account_value - initial_cash) / initial_cash
progress = visible_steps_completed / period_minutes
```

If a step is too early, the API returns `status = waiting`. If the session is missing or stale, it returns `status = expired` with HTTP 200.

## 14. Web API and UI Algorithm

Implemented in `src/app/web.py`.

Startup:

- applies realtime acceleration hints
- starts the live worker only when `AUTO_START_LIVE_WORKER=true`
- otherwise refreshes on demand through API/UI calls

Concurrency:

- `/api/live-snapshot` uses `run_in_threadpool`.
- operation-mode starts use a lock and busy status to avoid overlapping starts.
- paper-trading simulation steps use per-demo locks.

Important endpoints:

- `GET /api/status`
- `GET /api/research`
- `POST /api/research/refresh`
- `GET /api/research/diagnostics`
- `GET /api/research/volume`
- `GET /api/ontology/graph`
- `GET /api/ontology/runtime`
- `GET /api/realtime/runtime`
- `POST /api/live-snapshot`
- `POST /api/assess-goal`
- `POST /api/start`
- `POST /api/operation-mode/start`
- `GET /api/operation-mode/status`
- `POST /api/operation-mode/stop-learning`
- `POST /api/paper-trading/start`
- `POST /api/paper-trading/step`
- `GET /api/paper-trading/status/{demo_id}`
- `POST /api/paper-trading/pause/{demo_id}`
- `POST /api/paper-trading/resume/{demo_id}`
- `POST /api/paper-trading/cleanup/{demo_id}`
- `POST /api/mock-kis/orders`
- `GET /api/mock-kis/orders/{order_id}`
- `GET /api/mock-kis/portfolio`
- `POST /api/mock-trading/run`
- `GET /api/mock-trading/performance`

## 15. Runtime Modes

Implemented in `src/app/realtime/mode_manager.py` and `src/app/runtime/environment.py`.

Current operation modes:

- `learning`
- `testing`
- `paper_trading`
- `paper_trading_test`
- `live_readiness`
- `live_trading_test`
- `live_trading`

All modes use:

```text
data/store
data/models
```

`DataEnvironment.live()` and `DataEnvironment.simulation()` currently resolve to the same realtime environment. Synthetic data is not accepted by the active realtime store or model store.

## 16. Safety Model

The system is designed so that:

- LLM-like judgment is advisory only.
- Event LLM output is strict JSON and falls back to keyword rules.
- `OrderIntent` must pass `RiskManager`.
- `FinalOrder` is always a limit order.
- `manual_approval_required=True` is retained.
- live trading is disabled by default.
- restricted products are blocked.
- hypothetical testing reports zero broker orders.
- paper-trading simulation changes only in-memory simulated state.
- KIS live-readiness checks do not submit orders.

## 17. Known Algorithmic Limitations

- Main production indicators are still lightweight snapshots, not a full historical production indicator engine.
- Some live market/chart sources can be skipped due to robots.txt, missing API keys, source outages, or missing Playwright.
- Full listed-stock universe catalogs are tracked, but expensive external fetching is intentionally bounded.
- Paper-trading simulation sessions are in memory and expire on server restart.
- Mock LLM judgment is deterministic scoring, not an independent model decision.
- Risk rules are conservative hard gates, not a full brokerage compliance engine.
- `live_trading` is a guarded boundary, not automatic execution.
- This is personal research infrastructure, not financial advice.

## 18. Key Files

- `src/app/run.py`: startup and web server launch
- `src/app/web.py`: FastAPI UI/API orchestration
- `src/app/research/service.py`: source collection and diagnostics
- `src/app/storage/local_store.py`: SQLite storage
- `src/app/storage/model_store.py`: JSON model artifact store
- `src/app/pipeline.py`: analysis context pipeline
- `src/app/trading_pipeline.py`: lightweight ontology candidate filter
- `src/app/graph/builders.py`: market graph construction
- `src/app/graph/reasoner.py`: ontology inference and reasoning paths
- `src/app/goals/negotiation.py`: feasibility and compromise scoring
- `src/app/strategy/rule_based.py`: baseline strategy signals
- `src/app/strategy/goal_directed.py`: target-aware strategy
- `src/app/risk/manager.py`: deterministic risk validation
- `src/app/realtime/learning.py`: realtime supervised examples and hypothetical tests
- `src/app/trading/mock_program.py`: mock trading cycle
- `src/app/execution/kis_mock.py`: mock KIS behavior
- `src/app/execution/kis_real.py`: KIS Developers REST adapter
- `src/app/backtesting/streaming_demo.py`: stepwise in-memory simulation

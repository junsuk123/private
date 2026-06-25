# Architecture

## Principle

The system separates probabilistic reasoning from deterministic control. Classifiers, semantic layers, ontology reasoning, and mock LLM-style scoring can explain, classify, and propose. They cannot directly execute live trades. Every proposed order must pass the deterministic risk manager before it can become a `FinalOrder`, and live automated execution remains disabled.

## Runtime Flow

1. Read-only collectors load the listed-stock universe and fetch market, financial, macro, disclosure, and news data.
2. Normalizers convert raw data into typed records with source metadata.
3. `LocalResearchStore` persists normalized records in mode-specific SQLite stores.
4. `build_analysis_context` merges stored data, live snapshots, sample markets, indicators, events, graph triples, reasoning paths, strategy signals, order intents, and risk results.
5. Indicator engines calculate interpretable metrics.
6. The ontology layer links companies, sectors, indicators, events, risks, and signals.
7. The ontology reasoner infers buy candidates, risk adjustments, and reasoning paths.
8. Deterministic strategy modules produce `StrategySignal` and `OrderIntent` records.
9. `RiskManager` validates each `OrderIntent` against hard rules.
10. Approved intents become mock/paper orders or streaming-simulation trades only. Live trading remains blocked.
11. Audit logging records inputs, mode changes, decisions, rejections, and outputs.

## Implemented Public Data Layer

- `HtmlResearchCollector` fetches allowed HTML pages, extracts text, and stores source metadata.
- `RssNewsCollector` reads RSS feeds and classifies items into structured events.
- `ResearchService` loads the configured US/overseas and domestic KRX listed universe, stores a `listed_universe_catalog` record, and exposes universe diagnostics.
- `StooqMarketDataCollector` can fetch public daily CSV market data.
- `YahooChartCollector` is supported where robots.txt and runtime conditions allow it.
- `OpenDartDisclosureCollector`, `EcosMacroCollector`, and `FredMacroCollector` are implemented with optional API keys from environment variables.
- `RawArchive` stores raw source records as JSON for auditability.
- `ResearchService` retries eligible failed sources with configurable attempts and backoff.
- Expensive per-symbol collection is bounded by a rotating universe batch cursor so the app does not block while trying to process every listed symbol in one request.

Brokerage account collection and live order placement remain intentionally excluded.

## Web Runtime

`src/app/run.py` performs startup checks and then starts `uvicorn app.web:app`.

Port behavior:

- The default requested port is `8000`.
- If that port is occupied, the runner selects the next available port and prints the actual Web UI URL.
- Browser URLs must use the printed port. For example, if the runner prints `http://127.0.0.1:8000`, opening `http://127.0.0.1:8001` will fail unless a server is also running there.

The FastAPI app starts a background live worker that refreshes research, stores records, rebuilds the analysis context, and updates UI progress state.

Important UI/API paths:

- `GET /`: single-page web UI
- `GET /api/status`: account, report, signals, orders, and mock run status
- `GET /api/research`: events, graph triples, reasoning paths, and diagnostics
- `GET /api/ontology/graph`: graph payload for visualization
- `GET /api/realtime/runtime`: runtime backend, operation mode, and acceleration-policy diagnostics
- `POST /api/live-snapshot`: goal-aware live snapshot, executed in a threadpool to avoid blocking UI actions
- `POST /api/assess-goal`: target feasibility and compromise goals
- `POST /api/start`: accepted-goal mock KIS paper trading
- `POST /api/operation-mode/start`: live/simulation training or testing mode selection
- `POST /api/streaming-demo/step`: one visible streaming-simulation step

## Agent Boundaries

### Portfolio and Capital Management Agent

Reads portfolio state and produces allocation suggestions, exposure summaries, and rebalancing candidates. It does not access secrets and does not call brokerage order APIs.

### Data Crawling and Classification Agent

Classifies official API, RSS, disclosure, and news data into structured event records. It must not fabricate missing values and must preserve source metadata.

### Ontology-Based Strategy and Execution Planning Agent

Uses typed indicators and graph relationships to generate explainable signals and order intents. It must separate facts, assumptions, inferred relationships, and conclusions.

### Goal-Directed Simulation Planner

Uses the selected target return and target time to create a goal execution plan for mock/streaming simulation. It may rank BUY, SELL, REDUCE, and HOLD intents, but it still has to pass `RiskManager`.

## Deterministic Modules

### Risk Manager

Rejects orders that violate cash reserve, position size, sector exposure, daily loss, liquidity, volatility, duplicate-order, data-integrity, or trading-mode rules.

### Order Executor

Accepts only `FinalOrder` objects. The current implementation supports mock/paper interfaces only. Live trading raises an explicit error or is blocked at the risk/mode boundary.

### Streaming Simulation

`StreamingAcceleratedDemo` maintains an in-memory simulation session. It builds synthetic charts from the configured US/overseas and domestic KRX listed universe, then evaluates a bounded active ticker batch on each visible step while preserving full-universe coverage over time. The current web UI does not expose a user-facing acceleration multiplier. Simulation testing uses the target return and target minutes from the form and advances one visible step per `/api/streaming-demo/step` call.

If a stale or missing `demo_id` is sent to `/api/streaming-demo/step`, the API returns HTTP 200 with `status="expired"` so the UI can stop cleanly instead of surfacing a 404 error.

### Audit and Monitoring

Writes append-only JSONL records with timestamps and structured payloads.

## Current Implementation Choice

The current workspace uses FastAPI/Uvicorn for the web runtime and SQLite for local persistence. The graph is in-memory and record-oriented. PostgreSQL, TimescaleDB, Neo4j/RDF4J, pgvector, APScheduler, and Prometheus can still be added phase-by-phase once the core contracts are stable.

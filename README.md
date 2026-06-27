# Personal Multi-Agent Ontology-Based Automated Stock Investment System

Personal-use research system for safe, auditable, explainable stock-investment analysis and realtime-only paper-trading and live-readiness workflows.

The current implementation is intentionally conservative: it collects public/current-market research, builds indicators and ontology evidence, negotiates target feasibility, creates strategy intents, and validates every possible order through deterministic risk rules. Live automated brokerage execution remains blocked.

## Safety Model

- LLM or LLM-like components never execute trades.
- AI/semantic/ontology layers may classify events, rank candidates, tune analysis parameters, and produce `OrderIntent` objects.
- `RiskManager` is the deterministic final gate before an intent can become a `FinalOrder`.
- Source trust, data quality, freshness, synthetic status, and model uncertainty are checked before live approval can proceed.
- Approved orders are limit orders with `manual_approval_required=True`.
- Live automated execution is disabled by default and remains blocked in the current app flow.
- Synthetic, sample, pseudo, or hash-derived features are allowed only in clearly labeled offline fixtures and must not be used as trusted paper-trading or live-trading evidence.
- Margin, leverage, derivatives, short selling, credit loans, and leveraged ETFs are rejected.
- Paper trading uses the KIS virtual domain or the local paper engine; live-readiness checks do not submit broker orders.

## Current Scope

Implemented capabilities include:

- FastAPI web UI and API runtime
- Public research collectors for RSS, HTML, dynamic pages, Stooq, Yahoo chart, Alpha Vantage, OpenDART, ECOS, and FRED
- Listed-universe ingestion for US/overseas and KRX symbols
- Rotating universe batches so large universes do not block UI refreshes
- Local/OpenAI-compatible/embedded/OpenVINO event LLM classification with keyword fallback
- SQLite-backed local research store with stable record keys and retention pruning
- Lightweight indicator snapshots for the main decision path
- Ontology graph, event mapping, NPU/CPU ontology candidate screening, and rule-based reasoning paths
- Source trust policy and lightweight feature provenance labels for measured, estimated, and synthetic fields
- Goal feasibility negotiation and compromise target generation
- Rule-based and goal-directed strategy signal generation
- Deterministic risk validation
- KIS paper-trading boundary and local paper-trading loop
- Realtime learning and paper-trading evaluation artifacts under `data/models`
- Audit logging and diagnostics endpoints
- Recursive audit redaction for credentials, tokens, account numbers, and broker secrets
- No-lookahead dataset scaffolding, ranked-signal evaluation summaries, and CPU/OpenVINO inference backend hooks

## Data Trust And Provenance

`SourceMetadata` records source type, trust level, observed/retrieved time, latency, realtime/delayed flags, synthetic/backfilled flags, license policy, and quality score. Legacy metadata defaults to low trust. `app.data.source_policy` centralizes source-type inference, default trust levels, quality scoring, and live-decision validation.

Hash-derived or pseudo quote-like fields in the lightweight ontology filter are marked as synthetic or estimated. Offline fixture paths may use these labeled fields; paper-trading and live/realtime decision paths should reject `is_synthetic=true`, synthetic fields, unknown sources, stale quotes, or low quality scores. `RiskManager` remains the final deterministic authority.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python .\run.py
```

One-command Windows launch:

```powershell
.\run.ps1
```

`run.ps1` starts a managed local app server on `http://127.0.0.1:8010` by default, opens Chrome or Edge when available, and stops the server when the managed browser window closes. It sets safe defaults including:

- `DATA_ENV=realtime`
- `DATA_ROOT=data`
- `REALTIME_STORE_ROOT=data/store`
- `TRADING_MODE=learning`
- `LIVE_TRADING_ENABLED=false`
- `ONTOLOGY_ACCELERATOR=NPU`
- OpenVINO/NPU low-latency hints with CPU fallback
- small LLM event-classification limits for responsive refreshes

If `models/local-llm/event-classifier` exists, `run.ps1` enables the embedded local model. Otherwise it checks for an Ollama-compatible local server at `127.0.0.1:11434`; if unavailable, event classification falls back to deterministic keyword rules.

## KIS Developers Broker Adapter

`src/app/execution/kis_real.py` implements the Korea Investment & Securities Open API REST contract for domestic cash-stock limit orders, order-status polling, and balance lookup. It uses the same broker interface as the in-memory mock broker, so the same paper-trading flow can be run with an injected fake KIS transport and later switched to the real transport.

Safe defaults:

- `KIS_PAPER_TRADING=true` uses `https://openapivts.koreainvestment.com:29443`.
- `KIS_PAPER_TRADING=false` uses `https://openapi.koreainvestment.com:9443`.
- `KIS_LIVE_ENABLED=false` blocks all KIS order and account calls.
- `KIS_ACCOUNT_NO` may be `12345678-01` or paired with `KIS_ACCOUNT_PRODUCT_CODE=01`.

The adapter follows the current KIS guide pattern: `/oauth2/tokenP` for access tokens, `/uapi/hashkey` before cash-order POSTs, and the new domestic cash-order TR IDs `TTTC0011U`/`TTTC0012U` for live and `VTTC0011U`/`VTTC0012U` for paper. Keep token issuance and order calls on the same base URL; mixing paper and live domains will fail at the KIS gateway.

Secrets are loaded automatically from the ignored local file `config/secrets/kis_api_keys.env` when the KIS client starts. Copy `config/secrets/kis_api_keys.env.example` and keep real values out of Git. Use `python scripts/check_kis_connection.py` for a token-only check, or add `--account` for the read-only balance endpoint.

Optional in-process local LLM setup:

```powershell
pip install ".[local-llm]"
mkdir models\local-llm
# Put a local Hugging Face causal/chat model in:
# models\local-llm\event-classifier
.\run.ps1
```

Optional Intel OpenVINO/NPU event classification:

```powershell
pip install ".[openvino-llm]"
$env:LLM_EVENT_PROVIDER="openvino-llm"
$env:LLM_EVENT_MODEL="models\local-llm\event-classifier"
$env:LLM_EVENT_DEVICE="NPU"
.\run.ps1
```

For dynamic web pages that require browser rendering:

```powershell
pip install playwright
playwright install
```

Useful individual commands:

```powershell
$env:PYTHONPATH="src"
python -m app.cli demo  # legacy local sample pipeline
python -m app.cli research --config config/research_sources.live.json
python -m app.cli research --config config/research_sources.demo.json
python -m unittest discover -s tests
uvicorn app.web:app --app-dir src --reload
```

## Data Layout

The current runtime uses one realtime-only layout:

```text
data/store/research.sqlite3
data/raw/
data/models/<model_family>/
data/reports/
data/synthetic_disabled/
```

Synthetic/offline fixture rows are rejected from the realtime research/model stores. Historical `data/live`, `data/sim`, and `data/legacy` files may exist from older phases, but the active web runtime uses `data/store` and `data/models`.

Model artifacts are versioned and also written to `<model>.latest.json` inside each model-family folder.

## Runtime Modes

The web operation-mode manager exposes:

```text
learning        realtime collection and supervised PnL-label artifact updates
testing         legacy paper-trading replay alias; no live broker orders
paper_trading   KIS paper-trading API check + local paper buy/sell loop
paper_trading_test alias for paper_trading
live_readiness  KIS live-readiness check; no broker orders submitted
live_trading_test alias for live_readiness
live_trading    realtime trading gate; brokerage execution remains guarded/blocked
```

The paper-trading loop starts through:

```text
POST /api/paper-trading/start
POST /api/paper-trading/step
```

It creates labeled local one-minute paper-trading bars in memory, uses the listed universe plus ontology CPU/NPU heuristic candidate screening, runs goal-directed strategy and `RiskManager`, then applies approved orders only to virtual cash and holdings.

## Core Algorithm

```text
Public/current research sources
  -> normalization and optional LLM event classification
  -> data/store SQLite persistence
  -> analysis context
  -> lightweight ontology candidate filter
  -> indicator snapshots and time-synchronized frames
  -> ontology graph + reasoning paths
  -> goal feasibility and strategy scoring
  -> OrderIntent records
  -> deterministic RiskManager validation
  -> KIS paper-trading / local paper-trading output
```

The first ontology filter screens the full available universe with low-cost quote-like features such as liquidity, volume change, price momentum, foreign/institution flow, halt status, and management-stock status. In offline fixture mode these quote-like values can be synthetic or estimated and are labeled accordingly. Only selected candidates continue to heavier graph/strategy stages, with priority tickers retained for visibility.

The current ontology NPU path is a heuristic fixed linear scorer accelerated by OpenVINO when available. It is not a trained AI model unless a separately trained/exported model is plugged into the inference backend. CPU fallback remains enabled.

## Resource Profile

Local resource probe measured on 2026-06-25 23:53 KST with `C:\Python311\python.exe`.

Workload:

```text
30 iterations
4096 synthetic market snapshots
30 OpenVINO ontology classifier calls per iteration
3,686,400 total NPU score rows
600 time-synchronized frames per iteration
300 realtime supervised learning examples per iteration
200 paper-trading evaluation trades per iteration
model artifacts written on the first and last iteration
```

Observed result:

```text
total elapsed time:              11.871 s
average iteration time:          377.81 ms
iteration time range:            339.20-470.60 ms
OpenVINO backend:                NPU
NPU active according to runtime: true
NPU batch size:                  4096
NPU batches per 4096-row call:   1
NPU latency per 4096-row call:   avg 9.996 ms, min 7.911 ms, max 13.202 ms
last measured NPU throughput:    425,258 rows/s
process CPU usage:               avg 3.98%, max 4.40% of total logical CPU
process working set memory:      avg 130.5 MB, max 135.3 MB
process private memory:          avg 757.49 MB, max 760.7 MB
system memory used:              avg 82.40%, max 82.69%
GPU/NPU compute counter:         avg 0.00%, max 0.00%
GPU 3D counter for process:      avg 6.61%, max 13.22%
adapter shared memory:           avg 1606.25 MB, max 1617.2 MB
```

Measurement notes:

- The application runtime reported `backend=NPU`, `uses_npu=True`, and no fallback reason.
- The ontology NPU classifier uses a reusable input buffer and defaults to `ONTOLOGY_NPU_BATCH_SIZE=4096` to reduce per-call allocation churn and increase work per NPU dispatch.
- Windows `GPU Engine(*)\Utilization Percentage` did not expose a separate measurable NPU compute engine for this OpenVINO workload on this machine, so the OS counter stayed at `0.00%` even while OpenVINO reported NPU execution.
- CPU and process memory are sampled from the Python process; adapter memory is the aggregate Windows GPU adapter memory counter, not a per-model allocation.

## Important API Paths

```text
GET  /api/status
GET  /api/research
POST /api/research/refresh
GET  /api/research/diagnostics
GET  /api/research/volume
GET  /api/ontology/graph
GET  /api/ontology/runtime
GET  /api/realtime/runtime
POST /api/live-snapshot
POST /api/assess-goal
POST /api/start
POST /api/operation-mode/start
GET  /api/operation-mode/status
POST /api/operation-mode/stop-learning
POST /api/paper-trading/start
POST /api/paper-trading/step
POST /api/mock-kis/orders
GET  /api/mock-kis/portfolio
```

## Repository Layout

```text
src/app/
  agents/          LLM-facing interfaces and contracts
  audit/           Append-only audit logger
  backtesting/     Local paper-trading and accelerated replay tools
  data/            Public collectors, classifiers, HTTP helpers
  execution/       Mock broker plus KIS Developers REST adapter
  features/        Formula indicators, semantic features, model-row scaffolding
  goals/           Target feasibility and compromise goal negotiation
  graph/           Ontology graph, event mapper, NPU classifier, reasoner
  indicators/      Main lightweight IndicatorSnapshot engine
  models/          Dataset and no-lookahead labeling helpers
  realtime/        Operation modes, acceleration policy, learning/paper-trading helpers
  research/        Source orchestration, retries, diagnostics
  risk/            Deterministic hard-rule risk manager
  storage/         SQLite research store and model artifact store
  strategy/        Rule-based and goal-directed strategy generation
docs/
  architecture.md
  system_algorithm_analysis.md
  realtime_short_horizon_policy.md
  data_environment_separation.md
  semantic_feature_engine.md
  semantic_feature_codebase_analysis.md
research_notes/
  technical_indicator_formulas.md
```

## Development Phases

1. Realtime-only public data collection and normalized local storage
2. Indicator and semantic feature expansion
3. Ontology graph, candidate filtering, and reasoning
4. Local/remote LLM classification with strict JSON output
5. Goal-directed strategy and deterministic risk manager
6. Paper-trading evaluation and local accelerated replay
7. KIS paper/mock brokerage workflows
8. Brokerage read-only integration
9. Manual-approval trading gate
10. Limited automation only after proven stability and explicit controls

## Disclaimer

This is engineering infrastructure for personal research. It is not financial advice, an investment advisory service, or third-party fund management software.

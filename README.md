# Personal Multi-Agent Ontology-Based Automated Stock Investment System

Personal-use research system for safe, auditable, explainable stock-investment analysis and realtime-only paper/hypothetical trading experiments.

The current implementation is intentionally conservative: it collects public/current-market research, builds indicators and ontology evidence, negotiates target feasibility, creates strategy intents, and validates every possible order through deterministic risk rules. Live automated brokerage execution remains blocked.

## Safety Model

- LLM or LLM-like components never execute trades.
- AI/semantic/ontology layers may classify events, rank candidates, tune analysis parameters, and produce `OrderIntent` objects.
- `RiskManager` is the deterministic final gate before an intent can become a `FinalOrder`.
- Approved orders are limit orders with `manual_approval_required=True`.
- Live automated execution is disabled by default and remains blocked in the current app flow.
- Margin, leverage, derivatives, short selling, credit loans, and leveraged ETFs are rejected.
- Testing and streaming simulation submit no real broker orders.

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
- Goal feasibility negotiation and compromise target generation
- Rule-based and goal-directed strategy signal generation
- Deterministic risk validation
- Mock KIS paper-trading boundary and streaming simulation
- Realtime learning and hypothetical testing artifacts under `data/models`
- Audit logging and diagnostics endpoints

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
python -m app.cli demo
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

Simulation and synthetic rows are rejected from the realtime research/model stores. Historical `data/live`, `data/sim`, and `data/legacy` files may exist from older phases, but the active web runtime uses `data/store` and `data/models`.

Model artifacts are versioned and also written to `<model>.latest.json` inside each model-family folder.

## Runtime Modes

The web operation-mode manager exposes:

```text
learning      realtime collection and supervised PnL-label artifact updates
testing       realtime hypothetical trade test; orders_submitted = 0
live_trading  realtime trading gate; brokerage execution remains guarded/blocked
```

Streaming simulation is separate from operation-mode testing and starts through:

```text
POST /api/streaming-demo/start
POST /api/streaming-demo/step
```

It creates synthetic one-minute charts in memory, uses the listed universe plus ontology/NPU candidate screening, runs goal-directed strategy and `RiskManager`, then applies approved orders only to simulated cash and holdings.

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
  -> mock KIS / hypothetical test / streaming simulation output
```

The first ontology filter screens the full available universe with low-cost quote-like features such as liquidity, volume change, price momentum, foreign/institution flow, halt status, and management-stock status. Only selected candidates continue to heavier graph/strategy stages, with priority tickers retained for visibility.

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
200 hypothetical trades per iteration
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
POST /api/streaming-demo/start
POST /api/streaming-demo/step
POST /api/mock-kis/orders
GET  /api/mock-kis/portfolio
```

## Repository Layout

```text
src/app/
  agents/          LLM-facing interfaces and contracts
  audit/           Append-only audit logger
  backtesting/     Streaming and accelerated simulation tools
  data/            Public collectors, classifiers, HTTP helpers
  execution/       Mock/paper broker boundary; real KIS client disabled
  features/        Formula indicators, semantic features, model-row scaffolding
  goals/           Target feasibility and compromise goal negotiation
  graph/           Ontology graph, event mapper, NPU classifier, reasoner
  indicators/      Main lightweight IndicatorSnapshot engine
  models/          Dataset and no-lookahead labeling helpers
  realtime/        Operation modes, acceleration policy, learning/test helpers
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
6. Hypothetical testing and streaming simulation
7. Paper/mock brokerage workflows
8. Brokerage read-only integration
9. Manual-approval trading gate
10. Limited automation only after proven stability and explicit controls

## Disclaimer

This is engineering infrastructure for personal research. It is not financial advice, an investment advisory service, or third-party fund management software.

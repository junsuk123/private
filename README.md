# Personal Multi-Agent Ontology-Based Automated Stock Investment System

Personal-use research system for safe, auditable, explainable stock-investment analysis.

This repository starts at the safe phase: read-only data collection, indicator calculation, ontology-style reasoning, structured strategy signals, and deterministic risk validation. Live trading is disabled by design.

## Safety Model

- LLM agents never execute trades.
- LLM agents may only produce structured analysis, signals, and `OrderIntent` objects.
- The deterministic `RiskManager` validates every intent before it can become a `FinalOrder`.
- Live order execution is not implemented in this scaffold.
- Margin, leverage, derivatives, short selling, credit loans, and leveraged ETFs are rejected.
- Manual approval is required by default.

## Current Scope

This initial workspace includes:

- Architecture documentation
- Typed domain schemas
- Sample read-only data collectors
- Public research collectors for RSS, HTML, Stooq, OpenDART, ECOS, and FRED
- Indicator calculation engine
- Lightweight ontology/knowledge graph layer
- Ontology event mapping and rule-based reasoning paths
- Deterministic strategy signal generation
- Deterministic risk manager
- Audit logging
- CLI demo pipeline
- Unit tests for core risk and pipeline behavior
- Web UI for target-return negotiation before program start

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python .\run.py
```

Run these as separate commands in PowerShell, or use `.\run.ps1` to start the app with the same launch flow.

One-command start on Windows PowerShell:

```powershell
.\run.ps1
```

`run.ps1` sets safe defaults automatically, including `DATA_ENV=realtime`,
learning mode, NPU/low-latency preferences, and live refresh interval.
If `models/local-llm/event-classifier` exists, event classification loads that
model directly inside this Python process. Otherwise, if a local
Ollama-compatible server is running on `127.0.0.1:11434`, event classification
uses that local small LLM. If neither local model path is available, it falls
back to keyword classification.

Optional in-process local LLM setup:

```powershell
pip install ".[local-llm]"
mkdir models\local-llm
# Put a local Hugging Face causal/chat model in:
# models\local-llm\event-classifier
.\run.ps1
```

For multimodal Hugging Face checkpoints that use `AutoProcessor` plus `AutoModelForMultimodalLM`, set:

```powershell
pip install ".[local-llm]"
$env:LLM_EVENT_CLASSIFIER_ENABLED="true"
$env:LLM_EVENT_PROVIDER="multimodal"
$env:LLM_EVENT_MODEL="google/diffusiongemma-26B-A4B-it"
$env:LLM_EVENT_DEVICE="auto"
.\run.ps1
```

For Intel OpenVINO/NPU local inference, install:

```powershell
pip install ".[openvino-llm]"
$env:LLM_EVENT_PROVIDER="openvino-llm"
$env:LLM_EVENT_MODEL="models\local-llm\event-classifier"
$env:LLM_EVENT_DEVICE="NPU"
.\run.ps1
```

This sequence runs startup checks with live Yahoo Finance RSS/chart data, writes audit logs, and starts the web UI.

For dynamic web pages that require scrolling/rendering, enable `dynamic_pages` in config and install Playwright:

```powershell
pip install playwright
playwright install
```

Optional individual commands:

```powershell
$env:PYTHONPATH="src"
python -m app.cli demo
python -m app.cli research --config config/research_sources.live.json
python -m app.cli research --config config/research_sources.demo.json
python -m unittest discover -s tests
uvicorn app.web:app --app-dir src --reload
```

Ontology NPU mode:

```powershell
pip install ".[npu]"
$env:ONTOLOGY_ACCELERATOR="NPU"
python .\run.py
```

The ontology runtime prefers OpenVINO NPU when an `NPU` device is available. If no NPU
runtime is detected, the API reports a CPU fallback in `/api/ontology/graph` and
`/api/research/diagnostics`.

Live data diagnostics:

```text
http://127.0.0.1:8000/api/research/diagnostics
```

The live worker continuously refreshes data in the background, rebuilds indicators,
updates the ontology graph, and stores both raw research and reasoning outputs in
SQLite at `data/store/research.sqlite3`. Duplicate rows are ignored by stable record
keys, and stale rows are pruned automatically.

```powershell
$env:LIVE_REFRESH_SECONDS="15"
$env:RESEARCH_RETENTION_DAYS="30"
.\run.ps1
```

When a user selects a target return and period, the start flow builds a
goal-directed paper-trading plan. The plan combines ontology relations with common
chart/market rules such as RSI, volume confirmation, valuation, liquidity, volatility,
and macro risk. It can produce BUY, HOLD, REDUCE, and SELL intents, then passes them
through the deterministic risk manager before recording paper orders. Live brokerage
execution remains disabled.

Realtime operating modes:

```powershell
학습: 실시간 수집 데이터로 추론하고, 예측 매수/매도와 이후 실시간 손익을 지도학습 라벨로 저장합니다.
테스트: 실제 주문 없이 체결했다고 가정한 거래의 실현 손익을 계산합니다.
실전: 실시간 데이터와 리스크 게이트를 통과한 주문만 브로커 실행 경계로 이동할 수 있습니다.
```

Data is stored in one realtime layout:

- `data/store`
- `data/raw`
- `data/models/<model_family>`
- `data/reports`

Synthetic/simulation records are rejected from the realtime store and model store.
Model artifacts are versioned and also written to `<model>.latest.json` inside
each model-family folder.

Mock KIS Developers flow:

```text
LLM judgment -> ontology evidence -> OrderIntent -> RiskManager
-> Mock KIS limit order -> fill check -> mock portfolio update
```

Simulation endpoints:

```text
POST /api/mock-trading/run
POST /api/mock-kis/orders
GET  /api/mock-kis/orders/{order_id}
GET  /api/mock-kis/portfolio
```

The broker boundary is isolated behind `BrokerClient` in
`src/app/execution/broker.py`. The current implementation uses
`MockKisDevelopersApi`; a disabled `KisDevelopersApiClient` skeleton is available
for later real KIS Developers integration without changing the strategy, ontology,
risk, or trading-cycle code.

No external service credentials are required for the demo.

## Repository Layout

```text
src/app/
  agents/          LLM-facing interfaces and placeholder agents
  audit/           Append-only audit logger
  backtesting/     Placeholder package for later phases
  data/            Read-only sample collectors and source metadata
  execution/       Order executor interface; live trading disabled
  graph/           Ontology schema and in-memory knowledge graph
  indicators/      Financial, market, technical, and macro indicators
  risk/            Deterministic hard-rule risk manager
  strategy/        Signal and order-intent generation
  schemas/         Shared domain models
docs/
  architecture.md  System architecture and phase plan
config/
  risk_rules.example.json
tests/
  Unit tests
```

## Development Phases

1. Read-only data and normalized storage
2. Indicator engine
3. Ontology and knowledge graph
4. LLM agent layer with strict JSON outputs
5. Strategy and deterministic risk manager
6. Backtesting
7. Paper trading
8. Brokerage read-only integration
9. Manual-approval trading
10. Limited automation only after proven stability

## Disclaimer

This is engineering infrastructure for personal research. It is not financial advice, an investment advisory service, or third-party fund management software.

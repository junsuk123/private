# Realtime Data Environment Policy

The current system uses a realtime-only local data environment. Older live/simulation split directories may still exist in the repository history or under `data/legacy`, but they are not the active web runtime layout.

The end-to-end data, inference, risk, and execution boundary is summarized in `ontology base trading system diagram.png`.

## Active Layout

- `data/store`: realtime research SQLite store
- `data/raw`: archived raw realtime documents when archiving is enabled
- `data/models`: realtime learning and hypothetical-test model artifacts
- `data/reports`: optional reports and demo outputs
- `data/synthetic_disabled`: placeholder directory; synthetic data is not accepted by the realtime store

`default_environment()` reads:

- `DATA_ROOT`, default `data`
- `DATA_ENV`, default `realtime`

`DataEnvironment.live()` and `DataEnvironment.simulation()` both resolve to the same realtime environment. This is intentional in the current implementation.

## Store Rules

`LocalResearchStore` writes to `data/store/research.sqlite3` unless an explicit root is passed.

The generic SQLite table is:

```text
records(kind, record_key, observed_at, inserted_at, payload)
primary key: (kind, record_key)
```

Supported record kinds include:

- `events`
- `raw_records`
- `market_snapshots`
- `macro_metrics`
- `realtime_quotes`
- `realtime_executions`
- `graph_triples`
- `reasoning_paths`

Before saving, the store prunes rows older than `RESEARCH_RETENTION_DAYS` and inserts records with `insert or ignore` stable keys.

## Synthetic Data Rejection

Realtime storage rejects simulated or synthetic records. A row is treated as simulated when it contains signals such as:

- market `SIM`
- source names starting with `sim`, `synthetic`, or `accelerated_demo`
- raw URLs starting with `local://sim`, `local://synthetic`, or `local://accelerated-demo`
- source IDs starting with `sim:`, `synthetic:`, or `demo-chart:`
- graph evidence IDs starting with simulation/synthetic prefixes

`ModelArtifactStore` also refuses `simulated=True` artifacts in the realtime-only model store.

## Operation Mode Mapping

The web UI exposes these operation modes through `OperationModeManager`:

- `learning`: realtime collection and supervised PnL-label artifact updates under `data/models`.
- `testing`: backward-compatible legacy paper-trading replay; no live broker orders are submitted.
- `paper_trading`: KIS paper-trading API check plus local paper-trading flow.
- `paper_trading_test`: alias for KIS paper-trading checks.
- `live_readiness`: KIS live-readiness/authentication check; no orders are submitted.
- `live_trading_test`: alias for live-readiness checks.
- `live_trading`: realtime trading gate; live brokerage execution remains guarded and blocked by default app flow.

All modes use the unified realtime store:

```text
data/store
```

The runtime policy exposed in `/api/research/diagnostics` states:

```text
Learning, paper trading, live-readiness checks, and live trading all use the unified realtime data store only.
```

In the default web runtime, realtime collection/learning and the read-only KIS live-readiness account probe start automatically with the server. The UI no longer exposes separate manual learning, refresh, or live-readiness buttons.

## Paper-Trading Simulation

Paper-trading simulation is a separate in-memory workflow. It starts through:

```text
POST /api/paper-trading/start
POST /api/paper-trading/step
```

The simulation generates synthetic one-minute charts in memory from the listed universe, screens candidates with ontology/NPU logic, and applies approved mock orders only to simulated cash and holdings. The session is not persisted; restarting the server expires the `demo_id`.

Simulation cash is not persisted as a separate data environment. It is initialized from the latest read-only KIS live account basis when available, or from the configured default if no account basis can be read.

If the UI sends a stale `demo_id`, `/api/paper-trading/step` returns:

```text
HTTP 200
status = expired
```

## Separation Rules

- Treat `data/store` as the only active research store.
- Do not copy synthetic rows into `data/store`.
- Do not save simulated model artifacts into `data/models`.
- Treat paper-trading simulation state as temporary in-memory state.
- Treat `live_trading` as a guarded/manual approval boundary, not automatic execution.

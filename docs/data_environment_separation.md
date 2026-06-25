# Live vs Simulation Data Separation

The system keeps live and simulation assets in separate directory trees.

## Live

- `data/live/store`: live research SQLite store
- `data/live/raw`: archived raw live documents
- `data/live/models`: live-approved model artifacts
- `data/live/reports`: optional live reports

Live stores reject records marked as synthetic or simulated. Live trading is still blocked; live mode is used for current-market research, learning, diagnostics, and manually gated analysis only.

## Simulation

- `data/sim/store`: simulated/offline research store
- `data/sim/raw`: simulated raw documents
- `data/sim/models`: models trained on synthetic or offline test data
- `data/sim/synthetic`: generated OHLCV/news bundles
- `data/sim/reports`: batch and streaming simulation reports

Simulation mode is the only place where synthetic data and model artifacts trained on synthetic/offline data should be written.

## Synthetic Data Policy

Synthetic data is allowed only under `data/sim`. It is explicitly tagged with:

- `synthetic: true`
- `market: SIM` where applicable
- `source_name` such as `synthetic_news`
- `source_id` beginning with `synthetic:`
- `raw_url` beginning with `local://synthetic`

`LocalResearchStore(mode="live")` rejects simulated market/research records.
`ModelArtifactStore(mode="live")` rejects simulated model artifacts.

## Runtime Mode Mapping

The web UI exposes four operation modes:

- `simulation_training`: synthetic/offline learning, writes under `data/sim`.
- `simulation_testing`: in-memory streaming simulation using target return and target minutes.
- `live_training`: current market data learning, writes under `data/live`, orders prohibited.
- `live_trading`: guarded/manual approval boundary, automatic execution remains blocked.

The visible simulation test flow starts through:

```text
POST /api/operation-mode/start
mode = simulation_testing
target_return_rate = form value
period_minutes = form value
```

The server creates an in-memory `StreamingAcceleratedDemo` and returns a `demo_id`. The UI then calls `/api/streaming-demo/step` repeatedly. The demo session is intentionally not persisted; restarting the server expires it.

If the UI sends an old `demo_id`, `/api/streaming-demo/step` returns:

```text
HTTP 200
status = expired
```

This avoids a user-facing 404 during normal session expiration and lets the UI display a clean restart message.

## Market Session Simulation

`MarketCalendar` defines regular sessions for simulation:

- US: 09:30-16:00 America/New_York
- KRX: 09:00-15:30 Asia/Seoul

The utility currently excludes weekends. Exchange holidays should be integrated
later through an official exchange calendar or a maintained local holiday file.

Generate test data:

```powershell
$env:PYTHONPATH="src"
python -m app.cli generate-sim-data --exchange US --trading-days 15 --interval-minutes 5
```

Generate a larger randomized corpus for repeated model training and validation:

```powershell
$env:PYTHONPATH="src"
python -m app.cli generate-sim-data `
  --exchange US `
  --scenarios 5 `
  --ticker-count 30 `
  --trading-days 20 `
  --interval-minutes 5 `
  --randomness-scale 1.7
```

Randomized scenario controls include:

- `--scenarios`: number of independent market scenarios
- `--ticker-count`: number of synthetic tickers per scenario
- `--randomness-scale`: volatility and wick scale multiplier
- `--shock-probability`: intraday jump/crash probability for a single bundle
- `--volume-spike-probability`: volume spike probability for a single bundle
- `--news-events-per-ticker`: synthetic news frequency for a single bundle

## Separation Rules

- Do not copy synthetic records from `data/sim` into `data/live`.
- Do not train live-approved models from simulation-only rows unless they are explicitly reviewed and promoted.
- Do not reuse an in-memory streaming `demo_id` across server restarts.
- Treat `data/live` as read-only-first research state, not as an automatic execution source.

# GS Quant reference layer

## Decision

GS Quant `2.1.3` is an optional, offline mathematical reference. It is not an
order engine, data source, pricing service, model authority, or live dependency.
The production environment is intentionally unchanged: dependency resolution
would replace NumPy `2.5.2` with `2.3.5` and add 28 packages, so parity was run in
an isolated Python 3.12 environment.

The local implementation is original project code. Evidence provenance says
`local_quant_engine:incremental-v1`; `method_reference` identifies the GS Quant
function used for independent comparison and never claims that Goldman Sachs
provided the data or live result.

## Architecture

Before:

```text
KIS -> ticks/orderbook -> causal bars -> local features/ontology/GNN
    -> strategy -> RiskManager -> FinalTradeGate -> execution guards -> KIS
```

After:

```text
KIS -> ticks/orderbook -> causal completed bars
    -> IncrementalQuantEngine -> bounded QuantEvidenceCache
       |-> ontology evidence adapter (advisory)
       |-> fixed-version GNN frame (raw + normalized + freshness + quality + validation + mask)
       |-> strategy/risk sizing (factor clamped to [0, 1])
    -> RiskManager -> FinalTradeGate -> execution guards -> KIS

historical bars -> optional GSQuantReferenceAdapter -> parity results
                                               (offline/reference environment only)
```

No tick ingestion, websocket, broker, or execution module imports GS Quant or
pandas. Correlation/beta/portfolio operations consume explicitly aligned real
series and are intended for bar-close/offline workers.

The live hook defaults to `activation_mode: auto`. It activates only when Python
meets the configured minimum, configuration parsing succeeds, a deterministic
local math self-test passes, and the evidence SQLite store initializes beside
the active market-data store. GS Quant installation is intentionally not a
condition. `QUANT_REFERENCE_ACTIVATION_MODE=off` always disables it; the legacy
`QUANT_REFERENCE_LIVE_ENABLED=false` also remains an explicit override.

Once active it receives only bars closed by the next minute's trade and runs in
the existing background persistence worker. Periodically persisted forming-bar
snapshots are excluded. Three consecutive calculation/persistence errors open a
five-minute circuit breaker; after cooldown the self-test must pass before the
layer automatically resumes. This circuit affects evidence only, never KIS
market-data collection, RiskManager, FinalTradeGate, or exits.

## Implemented and reused

- Price statistics: simple/log return, rolling sample mean/std, z-score,
  realized volatility, maximum drawdown.
- Technicals: moving average, EMA, smoothed moving average, Bollinger
  lower/upper/width/position, MACD line/signal, Wilder RSI, trend.
- Portfolio math: return, volatility, concentration, covariance, correlation,
  benchmark beta, drawdown. A missing benchmark or risk-free series is not
  synthesized. Sharpe is deliberately not emitted without observed risk-free
  data.
- Existing `KnowledgeGraph` relations are reused. No new ontology class can
  authorize an order.
- Existing GNN design is followed: ordering/version is fixed; absent evidence
  has a separate mask and cannot be mistaken for an observed zero.
- Existing `RiskManager`, `FinalTradeGate`, duplicate-order protection,
  exposure ceilings, principal protection, profitability gate, and execution
  guards remain authoritative.
- GS-inspired Strategy/Trigger/Action objects produce only `QuantTradeIntent`;
  both authoritative gates are mandatory downstream.

## GS Quant classification

Used at reference-test runtime only:

- `timeseries.statistics.mean`, `std`, `zscores`
- `timeseries.technicals.moving_average`,
  `exponential_moving_average`, `smoothed_moving_average`,
  `relative_strength_index`

Architecture referenced, not imported into live runtime:

- strategy/trigger/action separation
- intent and result objects
- local rebalancing/basket concepts

Explicitly excluded:

- `gs_quant.api.gs.*`, `gs_quant.target.*`
- Marquee datasets and all GS credentials
- `GsDataApi`, `GsAssetApi`, `GsPortfolioApi`
- `PricingContext`, remote/generic pricing paths, proprietary risk models
- any broker or KIS execution integration

## Validation record (2026-08-24)

Deterministic 80-point price series, GS Quant `2.1.3`, Python 3.12,
`numpy.isclose(atol=1e-10, rtol=1e-8)`:

| Metric | Local | GS Quant | Result |
|---|---:|---:|---|
| rolling mean | 117.375 | 117.375 | pass |
| rolling std | 1.5661299939324123 | 1.566129993932266 | pass |
| z-score | 1.261070286407673 | 1.2610702864076733 | pass |
| moving average | 117.375 | 117.375 | pass |
| EMA | 118.23542150242032 | 118.23542150242032 | pass |
| smoothed moving average | 115.07217329785291 | 115.07217329785291 | pass |
| RSI | 62.78153887693825 | 62.78153887693825 | pass |

The focused safety, causal replay, fill simulation, risk and gate suite passed
154 tests. This is a regression check of the existing event-driven simulator;
it uses its existing cost/slippage and conservative same-bar barrier rules. No
real orders or fabricated market history were used.

The repository-wide run completed with **3,346 passed, 1 skipped, 5 failed**.
The five failures are pre-existing/unrelated dirty-worktree expectations in
short/long promotion sample accounting (four tests) and mismatched CSS/JS asset
cache-bust versions (one test). The quant/risk/event-pipeline focused rerun after
the final live hook passed **143/143**. The repository-root `PYTHONPATH` is needed
because several existing tests import helpers as `tests.*`.

## Latency record

Five thousand completed-bar updates and 20,000 cache reads on the development
host:

| Path | Median | p95 |
|---|---:|---:|
| completed-bar incremental calculation | 0.043821 ms | 0.081483 ms |
| hot-path cache lookup | 0.001570 ms | 0.001708 ms |

Before integration the tick path had no quant-layer work. After integration it
still has no GS/pandas/full-window calculation; consumers may do the measured
cache lookup. Computation happens only on completed bars.

## Operations and limitations

- `QuantEvidenceStore` creates additive `quant_evidence`,
  `quant_provider_health`, and `quant_validation_result` tables in an isolated
  SQLite database with the requested indexes. Retention is not guessed; an
  operator must align it with actual database growth policy.
- Provider health exposes version, latest evidence time, invalid count, mean
  calculation latency, cache counts/hit rate, and the no-hot-path-import fact.
- GS parity is unavailable—not silently replaced—when the optional package is
  absent or not exactly `2.1.3`.
- Portfolio scenario and Sharpe results remain unavailable until real aligned
  scenario/risk-free inputs are supplied.
- No server was restarted and no production flag or order setting was changed.

Live-readiness: **ready as an automatically enabled additive evidence/reference
layer when its safety conditions pass**. It does not independently make the whole trading system production-ready;
real trading remains governed by the repository's existing deployment,
validation, approval, and execution-authority policies.

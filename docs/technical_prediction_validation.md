# Technical Prediction — Replay & Walk-Forward Validation

The technical prediction layer is **not** assumed profitable. It must be judged
on **realized, cost-adjusted** outcomes via reproducible replay before any
weight is placed on it. This is the harness for that.

## What it does

`src/app/technical/replay.py` (`TechnicalReplayEvaluator`) replays the layer
over a time-ordered bar series with a hard **no-look-ahead** split:

- Features at decision index `i` are built from **past bars only** (`bars[:i+1]`).
- Labels at `i` are built from **future bars only** (`bars[i+1:]`), deducting
  realistic round-trip cost via `TradingCostEngine`.

For each decision it records: predicted tradability, methodology, regime,
predicted net edge (bps), realized net-after-cost (bps), the net-profitable
label, and MFE/MAE. It then aggregates:

- **overall** — n, tradable rate, hit rate (realized net-profitable among
  tradable BUYs), precision, avg predicted vs realized net bps, **expected-vs-
  realized edge error**, avg MFE/MAE, turnover.
- **by_methodology** and **by_regime** breakdowns.
- **walk_forward** segments (time-split) so regime drift and non-stationarity
  are visible; no segment is evaluated on data used to tune thresholds (the
  layer is rule-based, so there is no fitting — the split is a drift check).

## Running it

Offline / CI (no broker or store needed):

```
python scripts/replay_technical_prediction.py --from-bars path/to/bars.json --stamp 2026-07-09
```

`bars.json` is a list of `{ticker,as_of,open,high,low,close,volume}` rows.

From the live realtime store (aggregates ticks into 1-minute bars):

```
python scripts/replay_technical_prediction.py --symbol 005930 --since 2026-07-01
```

Reports are written to `data/models/technical_replay_reports/technical_replay.<stamp>.json`.

## How to read the results

- **Gross hit rate is not enough.** Look at `avg_realized_net_bps` and
  `precision_net_profitable` — after cost. A high hit rate with negative average
  net means the winners are smaller than the cost of the losers.
- **Edge error** (`avg_edge_error_bps` = predicted − realized) calibrates the
  expected-move model. Persistent positive error means the layer is optimistic
  and the capture fraction / horizon buffers should be tightened.
- **MAE vs MFE** informs the DynamicExitPolicy stop/target geometry.
- **Rejected-trade analysis**: rows with `tradable=false` still carry realized
  MFE/MAE, so you can confirm the layer is correctly declining low-edge setups
  (they should show poor realized net had they been taken).

## No-look-ahead guarantees (tested)

`tests/test_technical_replay_no_lookahead.py` asserts:

- A prediction at index `i` is **identical** whether or not future bars exist
  beyond it (feature isolation).
- Injecting a future spike changes **labels/MFE** but never the **prediction**
  at earlier indices.
- Labels never invent horizons whose data does not exist (they return `None`).

## Caveat

Small-account, short-horizon KR day-trading is structurally close to
negative-expectancy once round-trip cost is deducted. Replay routinely shows
`tradable=0` on low-volatility series — that is the layer correctly declining,
not a bug. Only deploy weight when replay shows a **positive realized net**
edge on out-of-sample segments, and re-validate after any config change.

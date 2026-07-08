# Macro–Micro Ontology — Replay & Validation

The hierarchical macro–micro reasoning layer is advisory and must be judged on
**realized, cost-adjusted** outcomes before any weight is placed on it. This is
the reproducible, no-look-ahead harness for that.

## What it does

`src/app/graph/macro_micro_replay.py` (`MacroMicroReplayEvaluator`) replays the
real `MacroMarketReasoner` + `MicroSymbolReasoner` (via the `OntologyCoordinator`
and `GlobalTradeArbiter`) over a time-ordered, multi-symbol bar series with a
hard **no-look-ahead** split:

- The macro context and each symbol's micro features are built from **past bars
  only** (`bars[:i+1]`).
- The realized outcome is computed from **future bars only** (`bars[i+1:]`),
  deducting realistic round-trip cost via `TradingCostEngine` (through the
  technical `LabelBuilder`).

For each decision it records the macro regime, whether macro blocked buys, the
per-symbol micro entry signal, the predicted net edge (bps), the realized
cost-adjusted net (bps), and the arbiter's ranked side. It aggregates: macro
regime distribution, BUY-candidate count, **expected-vs-realized edge error**,
and average predicted/realized net.

## Running it

Offline / CI (multi-symbol bar JSON, no store needed):

```
python scripts/replay_macro_micro_ontology.py --from-bars bars.json --stamp 2026-07-09
```

From the live store (aggregates ticks into 1-minute bars per symbol):

```
python scripts/replay_macro_micro_ontology.py --symbol 005930 --symbol 000660 --since 2026-07-01
```

Reports are written to
`data/models/macro_micro_replay_reports/macro_micro_replay.<stamp>.json`.

## How to read the results

- **`avg_edge_error_bps`** (predicted − realized) calibrates the micro
  expected-net-return model. Persistent positive error ⇒ optimistic; tighten the
  capture fraction / horizon buffers in the technical layer.
- **`regime_distribution`** shows how often each macro regime fired — a sanity
  check that the macro classifier is not stuck.
- **`buy_candidates`** vs realized net confirms whether the macro→micro funnel
  is selecting genuinely profitable setups after cost. `buy_candidates == 0` on
  low-volatility series is the layer correctly declining, not a bug.

## No-look-ahead guarantees (tested)

`tests/test_macro_micro_replay_no_lookahead.py` asserts:

- A macro/micro decision at step `i` is **identical** whether or not future bars
  exist beyond it (macro context + micro features are past-only).
- Injecting a future spike changes **realized** values but never the
  **prediction** at earlier steps.
- The report structure is deterministic.

## Safety recap

The replay exercises the advisory layer only. In live trading every BUY still
passes TradingCostEngine → ProfitabilityGate → PrincipalProtectionEngine →
RiskManager, SELL/REDUCE is evaluated before BUY, and `LiveExecutionCoordinator`
is the sole (limit-only) submission path. Replay results inform tuning; they
never widen a gate.

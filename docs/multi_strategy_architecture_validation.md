# Multi-strategy architecture and validation record

Updated: 2026-09-03

## Runtime architecture

The executable authority path is:

1. market data and completed-bar feature construction;
2. independent multi-label market-state estimation;
3. hysteresis projection to `TREND_LOW_VOL`, `TREND_HIGH_VOL`,
   `RANGE_LOW_VOL`, `RANGE_HIGH_VOL`, or `RISK_OFF`;
4. explainable cross-sectional candidate ranking (**WHAT**);
5. per-symbol context and ontology/GNN auxiliary evidence;
6. independent symbol × strategy evaluation (**WHEN**);
7. after-cost ranking and `NO_TRADE` alternative;
8. account/risk/session/pre-submit gates;
9. one atomic KIS order commit and reconciliation.

Only steps 8-9 are serial. Candidate preparation, scoring, context reasoning and
symbol × strategy algorithms are map/fan-out stages. An unready symbol is not a
barrier for a ready symbol. GNN evidence is auxiliary while its required heads or
live checkpoint validation are incomplete.

All configuration and model/data paths in this change resolve from the project
root or the module location. No workstation-specific absolute path is embedded.
Per-machine writable runtime storage continues to be selected by `app.paths`, so
the Synology-synchronised source tree can run unchanged on CPU, GPU, or NPU hosts.

## Derived concepts and local adaptation

- Regime routing adopts the useful structural idea of trend/range crossed with
  low/high volatility, plus a separate risk-off state. It does not copy upstream
  implementation code.
- `bar_trend_continuation` retains its legacy identifier for schema compatibility
  but is now versioned as `bar-trend-pullback-v5`. It uses completed-bar
  MA20/50/200 structure, MA50/200 slopes, MACD, RSI, short retracement, ATR-normalised
  MA20 distance and volume contraction. It no longer buys high-volume extension.
- Candidate ranking uses a timeframe-normalised 200-minus-20-bar momentum input,
  volatility adjustment, cross-sectional relative strength, long-trend state,
  liquidity, regime/ontology fit and optional GNN suitability. Missing GNN values
  are neutral rather than a veto.
- Long-term trend is a candidate/risk filter, not an independent entry signal.
- Candidate strategy generation and changed algorithms begin in SHADOW. Stored
  outcomes from the previous continuation thesis cannot promote the new version.

Conceptual references:

- <https://github.com/quantsarahz/btcusdt-regime-multistrategy-trading>
- <https://github.com/s9213712/BTC_trade>
- <https://github.com/miguelariveracabezas/sp500-momentum-strategy>
- <https://github.com/dayfine/trading>
- <https://github.com/ceyhanmolla/freqtrade-strategies>

## Defects corrected

The context refresher captured one timestamp before a slow source scan, then used
that old timestamp to process quotes received during the scan. Fresh quotes were
therefore labelled as future/late and all candidates displayed `HARD:STALE_DATA`.
Production cycles now evaluate at collection completion, and source processing
uses the actual processing timestamp. Explicit replay clocks remain deterministic.

Freshness reports are scoped to the active decision candidates plus unscoped
infrastructure streams. Retained observations from old rotating subscriptions
remain available to the pre-submit audit but no longer make the dashboard's whole
market look stale. Candidate discovery now uses the same 15-second age contract as
the critical realtime policy instead of a ten-minute window.

The training UI distinguishes the bounded materialised window (`100,000` rows)
from the monotonic cumulative ingestion counter. The former may prune by design;
the latter is the accumulation measure.

The entry DAG now carries an explicit `cycle_id` and `advancing_symbols` lineage.
Ranking, context, symbol-by-strategy and selection rows are intersected with that
same-cycle set before they are rendered. A retained diagnostic from a prior US
cycle can therefore no longer appear downstream of current Korean candidates.

Accepted live orders are now reconciled repeatedly on later engine cycles instead
of only once immediately after submission. Polling is concurrency-safe and
throttled, terminal orders release the global duplicate-entry lock, and a stale
unfilled BUY follows the configured 120-second cancel policy. The domestic KIS
status parser also distinguishes OPEN, PARTIALLY_FILLED, FILLED, CANCELED,
REJECTED and the zero-fill/zero-remaining EXPIRED state.

## Verification evidence

- Focused regime, ranking, feature, strategy, context, API, dashboard and freshness
  tests: 203 passed.
- Independent live-feature, ontology eligibility, strategy-session parallel map,
  proposal, blockade and pre-submit tests: 141 passed.
- Combined change-scope result: 344 passed, 0 failed.
- Final post-reconciliation/dashboard-lineage regression: 272 passed, 0 failed.
- Stored-bar replay: 25 symbols, 4,392 causal decision windows, 11 pullback
  triggers. Net-of-configured-cost mean was -88.08 bp; chronological 30% OOS was
  4 trades at -113.27 bp mean and 0% win rate.

The negative OOS result is a failed promotion result, not a reason to hide the
strategy or force an order. `bar-trend-pullback-v5` therefore remains SHADOW-only.
It can now accumulate version-correct counterfactual outcomes, but it cannot submit
a live order until the existing after-cost, sample-size, stability and drawdown
promotion gates pass.

After the final restart, the bounded training window was 100,000 rows and
cumulative ingestion was at least 194,642 rows and increasing. The incumbent live-eligible
short-horizon model remained preserved, both KR/US serving paths reported healthy,
and temporal-GNN training was running. The realtime engine was in `live_trading`,
reliability score was 1.0, the cycle error count was zero, and there were no open
BUY orders. A transient stale quote still blocks only its own symbol at the final
freshness gate; it does not prevent other ready branches from continuing.

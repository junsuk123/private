# Event Backtest Status

Status: causal simulator and stored-data evaluation implemented; strategy performance promotion rejected.

Implemented evidence:

- event-time-ordered limit fill simulation;
- conservative same-bar barrier resolution;
- KRX fees, taxes, spread, slippage, and impact through the shared cost engine;
- fill, gross/net return, MAE/MFE, holding time, and exit reason;
- counterfactual result for every supplied strategy plus NoTrade;
- purged walk-forward splits with embargo;
- existing block-bootstrap reality check, fee-converted-loss ratio, cost-to-alpha ratio, Sharpe, and drawdown metrics.

## Stored-data result

`scripts/build_refactor_counterfactual_report.py` evaluated the local
`realtime_market_data.sqlite3` store and wrote the reproducible machine-readable
result to `data/reports/refactor_counterfactual_evaluation.json`.

- Input coverage: 27,306 minute bars, 255 symbols, 11 UTC dates; only 22 symbols
  contain at least 100 bars.
- Scope mismatch: the dense symbols and sessions are US equities, whereas the
  target system is KRX/NXT.
- Labels: 4,980 point-in-time snapshots and 34,860 stock-strategy outcomes.
- Leakage controls: every rolling quantile uses only the preceding 30 bars;
  fills begin after the feature cutoff; two purged/embargoed walk-forward splits
  were produced.
- Available triggered experts had negative mean net returns after the configured
  overseas commission, tax, and slippage model. The train-window-mean tabular
  policy therefore selected zero trades across 1,992 test observations and
  returned the correct first-class `NoTrade` action.
- Event momentum, sector-relative strength, and gap context cannot be validated
  because point-in-time event data, sector membership, and an authoritative
  session calendar are absent. Historical legacy decisions were not journaled,
  and the temporal R-GCN has no trained/calibrated checkpoint.

Consequently the mandatory legacy versus ontology-only versus tabular versus
temporal R-GCN comparison remains incomplete and no strategy/model is promoted.
The report includes label-data, configuration, and evaluation-code SHA-256
hashes. Generating favorable synthetic PnL would not satisfy the acceptance gate.

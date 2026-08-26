# Open-source strategy review

Reviewed 2026-08-25 for regime-coverage expansion. Public code is treated as a
research reference, never as evidence of profitability or live authorization.

## Screened and rejected

- Jesse `RSI2` was implemented provisionally with completed one-minute bars,
  causal indicators, exact target geometry, and local execution costs. On the
  available counterfactual corpus it produced 692 triggers / 613 fills, mean net
  return of -92.46 bps, and a 17.94% positive-outcome rate. It was removed from
  the executable catalogue.
- Jesse `Donchian` was implemented provisionally with a prior completed-bar
  channel to prevent look-ahead. It produced 317 triggers / 287 fills, mean net
  return of -87.20 bps, and a 21.25% positive-outcome rate. It was also removed.

These numbers are screening evidence, not a performance forecast: the corpus has
185,779 bars but only 30 distinct UTC dates, is concentrated in a few US sessions,
and cannot reconstruct queue position or intrabar barrier ordering from OHLC.

References:

- https://github.com/jesse-ai/example-strategies/tree/master/RSI2
- https://github.com/jesse-ai/example-strategies/tree/master/Donchian
- https://github.com/jesse-ai/example-strategies/blob/master/LICENSE

## Reviewed but not integrated

- Jesse Dual Thrust/Turtle and generic moving-average crossover: materially
  overlap existing opening-range, breakout, and trend-continuation families.
- Freqtrade strategy examples: GPL runtime and crypto-oriented examples; the
  project documentation explicitly says generated/sample strategies are not
  profitable out of the box.
- Backtrader: GPL framework architecture, not a source of validated equity alpha.
- VectorBT: useful offline research/backtest design, but its Commons Clause makes
  direct runtime adoption unnecessary and it does not supply validated alpha.
- QuantConnect LEAN: Apache-2.0 event-driven engine; useful architecture, but
  replacing the current KIS execution/risk stack would add risk without filling a
  strategy-family gap.

## Deployment decision

No reviewed open-source strategy was added to the executable catalogue. This
preserves the existing GNN output schema/checkpoint and prevents a negative-screened
idea from accumulating order authority. A future candidate may enter SHADOW only
after causal replay is non-negative after costs, and may progress only through
market-scoped, versioned forward outcomes and the existing promotion ladder.

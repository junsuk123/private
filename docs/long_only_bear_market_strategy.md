# Long-only bear-market strategy

`residual_relative_strength` now owns a defensive submode for `TREND_DOWN` and
`HIGH_VOL_TRENDING_DOWN`. It trades individual cash equities only. It does not
short, buy inverse/leveraged products, or treat cash as a synthetic position.

## Entry thesis

The broad market must be falling and breadth must confirm that condition. A stock
is eligible only when all of the following point-in-time facts agree:

- positive short- and long-horizon returns after removing market and sector beta;
- positive absolute stock return (being a smaller decliner is insufficient);
- market beta at or below the defensive ceiling;
- price above VWAP and completed-bar fast EMA above slow EMA;
- persistent momentum, top-three within-sector rank, relative volume, supportive
  microprice, acceptable spread, and (for KRX) investor-flow confirmation;
- stable regime and a cost-covering attainable target.

The bear submode uses a longer, three-hour maximum horizon so a genuine move has a
chance to clear KR/US round-trip cost. The exit contract is frozen at election,
uses the same volatility-derived target reported by entry, a tighter volatility
stop, and a trailing exit. Loss of residual rank, absolute strength, or book
support invalidates the thesis.

## Deployment

The strategy remains `live_authorized: false` and collects market-scoped SHADOW
outcomes. Promotion is additionally scoped to the exact macro regime, so an
uptrend result cannot authorize a downtrend order. Existing automatic promotion,
profitability, risk, session, instrument,
and final-order gates remain mandatory. Adding the bearish submode grants no order
authority and does not change the GNN output schema because it reuses the existing
`residual_relative_strength` head.

Research basis: defensive/low-beta evidence motivates the beta ceiling, while the
documented crash risk of momentum after market declines motivates the absolute
trend, breadth, change-point, and tighter stop requirements. These references are
hypothesis support, not evidence that this implementation is profitable.

## Stored-data validation (2026-08-25)

Run `scripts/validate_long_only_bear_strategy.py`. The run consumed 186,429
preferred-stream minute bars across 2,810 symbols and retained 121 instruments that
could be positively identified as individual cash equities. It produced no executable
signals under the full rule set, so neither KR nor US qualified for promotion. This is
an insufficient-coverage/trigger result, not evidence of profitability. The submode
therefore remains SHADOW while fresh, regime-matched outcomes accumulate.

The promotion floor is 30 after-cost outcomes with at least 10 winners and three
trading days, a positive one-sided conservative bound, at least 60% positive trading
days, and positive mean net edge after costs are stressed to 1.25x. Backtests cannot
advance beyond SHADOW; real LIVE_PROBE fills are independently required for higher
rungs.

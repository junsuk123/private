# Profitability Replay Report

![Before/after net-profitability gate](diagrams/profitability_before_after.svg)

Source journal: `logs\live-orders.jsonl`

> Realized PnL uses limit-price as a fill-price proxy and is APPROXIMATE.
> Net PnL/return apply the live TradingCostEngine. Compare BEFORE vs AFTER the
> refactor; the success criterion is improved NET expectancy and fewer
> negative-cost trades, not more trades.

## Order-flow outcomes

| event_type | count |
|---|---|
| live_order_amend_attempt | 3897 |
| live_order_amend_error | 3861 |
| live_order_cancel_attempt | 3846 |
| live_order_submission_attempt | 1343 |
| live_order_submission_error | 957 |
| live_order_status | 728 |
| live_order_submitted | 386 |
| live_order_amend_blocked | 171 |
| live_order_blocked | 72 |
| live_order_amended | 36 |
| live_order_canceled | 3 |

## Broker statuses

| status | count |
|---|---|
| OPEN | 627 |
| ACCEPTED | 422 |
| FILLED | 101 |
| CANCELED | 3 |

## Block / rejection reasons

| reason | count |
|---|---|
| LIVE_TRADING_ENABLED_NOT_TRUE | 237 |
| KIS_HEALTH_ACCOUNT_READ_FAILED | 5 |
| KIS_HEALTH_WEBSOCKET_APPROVAL_KEY_FAILED | 1 |

## Cost-aware realized metrics (matched round-trips)

| metric | value |
|---|---|
| round_trips | 181 |
| gross_pnl | 1342.45 |
| net_pnl | -1701.37 |
| win_rate | 0.442 |
| avg_win_net | 0.01689 |
| avg_loss_net | -0.01547 |
| payoff_ratio | 1.092 |
| expectancy_net | -0.00117 |
| negative_net_trades | 101 |
| max_drawdown_net_pnl | -7237.15 |

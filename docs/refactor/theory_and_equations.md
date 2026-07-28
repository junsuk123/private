# Implemented Theory and Equations

All online “unusual” conditions consume causal rolling quantiles in `[0,1]`; experts do not apply raw universal RSI/ADX thresholds.

## Incremental microstructure

- `mid=(ask+bid)/2`
- `spread_bps=10000*(ask-bid)/mid`
- `microprice=(ask*bid_qty + bid*ask_qty)/(bid_qty+ask_qty)`
- `OBI=(sum_bid_qty-sum_ask_qty)/(sum_bid_qty+sum_ask_qty)`
- OFI follows the best-level Cont-style price/quantity update implemented in `app.data.event_pipeline`.
- `trade_imbalance=(buy_volume-sell_volume)/(buy_volume+sell_volume)`
- `VWAP=sum(price*quantity)/sum(quantity)`
- `RV=sqrt(sum(log_return^2))`

Updates are event incremental. Duplicate or out-of-order events do not mutate state; numeric sequence gaps mark state uncertain. Reconnect resets OFI and requires a fresh book snapshot.

## Strategy experts

Seven independent experts are implemented in `app.strategy.experts`: intraday momentum, volume breakout, VWAP mean reversion, liquidity-shock reversal, event momentum, cross-sectional relative strength, and gap context. Each creates its own `TradePlan` and strategy instance. Stops, profit targets, time exits, invalidation, and stale-data exits are mechanical and remain owned by that instance.

## Cost and labels

`TradingCostEngine` deducts broker fees, sell tax, spread, slippage, and impact. The event simulator reports gross return, cost, net return, MAE, MFE, holding time, fill status, and a first-class zero-return `NO_TRADE` counterfactual. Unknown same-bar stop/target order is resolved conservatively in favor of the stop.

## Utility model

Dense fixed-shape relational propagation is:

`H_t = ReLU(X_t W_self + sum_r A_r,t X_t W_r)`

Causal temporal pooling uses fixed weights over observations at or before the decision snapshot. Heads estimate success probability, gross return, cost, MAE, MFE, fill probability, holding time, and aleatoric uncertainty.

`net_bps = gross_bps - cost_bps`

`utility = p_success*net_bps - (1-p_success)*MAE - uncertainty + 0.1*fill_probability*MFE`

Ontology and padding masks set utility to negative infinity. A separate NoTrade head is retained.

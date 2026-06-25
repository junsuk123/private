# Technical Indicator Formula Notes

This note documents the first implemented feature group. All calculations are as-of only and must use no bars after the decision timestamp.

## Returns

- Formula: `close_t / close_(t-n) - 1`.
- Inputs: adjusted or raw close series. Use one convention consistently.
- Defaults: 1, 5, 20, 60, 120 bars.
- Interpretation: positive values indicate price appreciation over the lookback.
- Limitations: ignores dividends, splits, taxes, costs, and benchmark context unless adjusted data is supplied.

## SMA and EMA

- SMA formula: arithmetic mean of the latest `n` values.
- EMA formula: recursive smoothing with `alpha = 2 / (n + 1)`.
- Defaults: SMA 20/60, EMA 12/20/26.
- Interpretation: price above moving averages and short MA above long MA are trend-supporting states.
- Limitations: lagging indicators and prone to whipsaws in sideways markets.

## MACD

- Formula: `EMA(12) - EMA(26)`, signal line `EMA(9)` of MACD, histogram `MACD - signal`.
- Inputs: close series.
- Interpretation: MACD above signal supports bullish momentum; below signal can contradict aggressive buying.
- Limitations: lagging and false crossovers in range-bound conditions.
- Sources: OANDA MACD overview and Investopedia MACD definition.

## RSI

- Formula: `RSI = 100 - 100 / (1 + RS)`, where `RS = average gain / average loss`.
- Inputs: close-to-close changes.
- Defaults: Wilder-style 14-period smoothing.
- Interpretation: commonly above 70 is overbought and below 30 is oversold. Trend context should affect interpretation.
- Limitations: can remain overbought in strong trends and oversold in persistent downtrends.
- Sources: OANDA RSI guide and Investopedia RSI definition.

## Bollinger Bands

- Formula: middle `SMA(20)`, upper `SMA + 2 * stddev`, lower `SMA - 2 * stddev`.
- Derived values: band width `(upper - lower) / middle`, percent B `(close - lower) / (upper - lower)`.
- Interpretation: narrow width can indicate compression; upper/lower touches can be overextension or breakout context.
- Limitations: lagging and not a standalone signal.
- Sources: Investopedia and Schwab Bollinger Bands descriptions.

## ATR

- True range: max of high-low, abs(high-previous close), abs(low-previous close).
- ATR: Wilder-style smoothed average of true range.
- Defaults: 14 bars.
- Interpretation: volatility and stop-distance input.
- Limitations: measures magnitude, not direction.
- Sources: Investopedia ATR and StockCharts ATR references.

## Historical Volatility

- Formula: standard deviation of recent returns times `sqrt(252)`.
- Defaults: 20 bars, 252 annualization factor.
- Interpretation: annualized recent realized volatility.
- Limitations: unstable for short windows and assumes the sampling period is comparable to daily bars.

## Stochastic Oscillator

- Formula: `%K = (close - lowest_low_n) / (highest_high_n - lowest_low_n) * 100`, `%D = SMA(3) of %K`.
- Defaults: 14-period %K and 3-period %D.
- Interpretation: above 80 is often overbought and below 20 oversold.
- Limitations: frequent false signals without trend and support/resistance context.
- Sources: StockCharts and OANDA stochastic oscillator references.

## OBV

- Formula: cumulative volume added on up closes, subtracted on down closes, unchanged on flat closes.
- Inputs: close and volume.
- Interpretation: rising OBV can support accumulation.
- Limitations: absolute value depends on start date; slope/divergence is more meaningful.
- Sources: StockCharts and Investopedia OBV references.

## MFI

- Typical price: `(high + low + close) / 3`.
- Raw money flow: `typical price * volume`.
- Money flow ratio: positive money flow / negative money flow.
- MFI: `100 - 100 / (1 + money_flow_ratio)`.
- Defaults: 14 periods.
- Interpretation: above 80 may indicate buying pressure/overbought; below 20 selling pressure/oversold.
- Limitations: volume quality and venue coverage matter.
- Sources: TradingView MFI calculation reference and Investopedia MFI description.

# Live Trading Safety Gates

Real order submission is possible in the current `run.ps1` runtime, but only through the guarded live path. The live model, LLM, ontology, and strategy layers cannot submit orders directly.

## Mandatory Submission Gates

`LiveExecutionCoordinator` and the KIS adapter require all of the following before a live order can be submitted:

- Input object is `FinalOrder`.
- Order type is `LIMIT`.
- Side is supported by KIS for the selected market.
- Quantity and limit price are positive.
- Domestic symbols are valid six-digit KRX-style symbols when routed domestically.
- `LIVE_TRADING_ENABLED=true`.
- `KIS_LIVE_ENABLED=true`.
- `KIS_PAPER_TRADING=false`.
- `LIVE_ORDER_SUBMIT_ENABLED=true`.
- `KILL_SWITCH_ENABLED=false`.
- Manual arming/runtime live gates pass when required by the live runtime guard.
- KIS credentials validate.
- KIS token can be issued or loaded.
- KIS account balance can be read.
- KIS WebSocket approval key can be issued when required.
- Idempotency key has not been used for a different order payload.

## Realtime Engine Gates

Before a BUY can become a `FinalOrder`, the realtime engine also requires:

- `REALTIME_BUY_ENABLED=true`.
- enough cash in the relevant currency for at least one share plus buffer
- fresh broker quote refresh or fresh KIS realtime tick/orderbook evidence
- acceptable spread versus the adaptive `max_spread_bps`
- adequate liquidity
- fallback/ontology/runtime score above the adaptive buy threshold
- model support only as an auxiliary input when `REALTIME_MODEL_AUXILIARY_ONLY=true`
- deterministic `RiskManager` approval

Common BUY rejection codes include:

- `MODEL_FEATURE_UNAVAILABLE:...`
- `WIDE_SPREAD:x>ybps`
- `LOW_LIQUIDITY`
- `FALLBACK_SCORE_BELOW_THRESHOLD:x<y`
- `ONTOLOGY_REQUIRED_FOR_MODEL_FALLBACK`
- `MODEL_AUXILIARY_ONLY_NEEDS_CONFIRMATION`
- `INSUFFICIENT_CASH_FOR_ONE_SHARE`

## SELL/REDUCE Gates

SELL/REDUCE is evaluated before BUY. Exits can be approved by:

- profit target after costs
- trailing/loss exit when `REALTIME_ALLOW_LOSS_EXIT=true`
- domestic drawdown reduction
- domestic emergency exit
- domestic concentration reduction
- time/quote-based exit policy

If an open SELL order already exists and the replacement price is effectively unchanged, the engine records `open_sell_kept` rather than submitting a duplicate order.

## Termination Flow

The `/account` termination button:

1. Sets `REALTIME_BUY_ENABLED=false`.
2. Stops the realtime trading loop.
3. Submits profit-seeking limit SELL orders for current holdings when all live gates pass.
4. Schedules local server shutdown.

This is not a market-order panic button. The default order type remains LIMIT.

## Hard Stops

- `KILL_SWITCH_ENABLED=true` blocks new live submissions.
- `REALTIME_BUY_ENABLED=false` blocks new BUY evaluations/submissions while allowing exit management.
- Missing KIS credentials, failed runtime gates, failed idempotency checks, or failed account/token checks block live submission.

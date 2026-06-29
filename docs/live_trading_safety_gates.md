# Live Trading Safety Gates

Real order submission requires all of the following:

- Input object is `FinalOrder`.
- Order type is `LIMIT`.
- KRX symbol is six digits.
- Quantity and limit price are positive.
- `LIVE_TRADING_ENABLED=true`.
- `KIS_LIVE_ENABLED=true`.
- `KIS_PAPER_TRADING=false`.
- `LIVE_ORDER_SUBMIT_ENABLED=true`.
- `KILL_SWITCH_ENABLED=false`.
- Manual arming file exists and has not expired.
- KIS credentials validate.
- KIS token can be issued or loaded.
- KIS account balance can be read.
- KIS WebSocket approval key can be issued.
- Idempotency key has not been used for a different order payload.

The live strategy loop is still blocked until realtime/model/provenance gates are
implemented.

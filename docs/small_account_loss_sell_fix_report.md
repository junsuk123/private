# Small Account Loss-Sell Guard Report

Updated: 2026-07-03

## Root Cause

The live account is small enough that a single share can be a large portfolio weight. Older SELL/REDUCE paths could interpret drawdown reduction, concentration reduction, or an opt-in tight stop as permission to sell a 1-share position at a small loss. In that case a REDUCE is effectively a full SELL, so repeated churn can push realized PnL negative.

The account dashboard also conflated broker cash-equivalent residuals with foreign cash. That made foreign cash look too large when KIS account totals contained settlement or account-total adjustments.

## Changes

- `AccountSnapshot.foreign_cash_krw` now carries actual foreign cash in KRW separately from `cash_equivalent_krw`.
- KIS live portfolio parsing now reads foreign cash summary fields and orderable foreign-currency balances, then converts them with broker FX.
- `/api/status` and `/api/account/dashboard` now keep total equity from the broker/account view while displaying foreign cash as actual USD cash times sane FX.
- `SharedLiveDecisionEngine` blocks non-emergency 1-share loss exits in small-account mode with `SMALL_ACCOUNT_ONE_SHARE_LOSS_BLOCK`.
- Non-emergency below-break-even exits can be blocked with `REALTIME_BLOCK_SELL_BELOW_BREAKEVEN=true`.
- `RiskManager` rejects small-account BUY candidates where one share exceeds `REALTIME_SMALL_ACCOUNT_MAX_POSITION_WEIGHT` of equity.
- `run.ps1` defaults now enable small-account mode, below-break-even sell blocking, one-share loss blocking, conservative buy sizing, and daily loss/cooldown settings.

## Break-Even Guard

Normal SELL/REDUCE exits must be net-profitable after estimated round-trip cost unless they are explicit hard emergency exits.

```text
required_exit_price = max(cost_floor.required_exit_price, avg_cost * (1 + sell_target))
profitable_after_cost = current_price >= required_exit_price
hard_emergency = pnl_rate <= -REALTIME_HARD_STOP_LOSS
```

If `REALTIME_BLOCK_SELL_BELOW_BREAKEVEN=true` and the exit is not a hard emergency, a below-break-even SELL is rejected with `SELL_BELOW_BREAK_EVEN_BLOCKED`.

## Small-Account Defaults

```text
REALTIME_SMALL_ACCOUNT_MODE=true
REALTIME_SMALL_ACCOUNT_EQUITY_KRW=300000
REALTIME_SMALL_ACCOUNT_MAX_POSITION_WEIGHT=0.10
REALTIME_ALLOW_LOSS_EXIT=false
REALTIME_HARD_STOP_LOSS=0.03
REALTIME_BLOCK_SELL_BELOW_BREAKEVEN=true
REALTIME_BLOCK_ONE_SHARE_LOSS_REDUCE=true
REALTIME_MAX_BUY_ORDERS_PER_CYCLE=1
REALTIME_BUY_WEIGHT=0.003
REALTIME_DAILY_REALIZED_LOSS_LIMIT_KRW=1500
REALTIME_LOSS_REENTRY_COOLDOWN_SEC=7200
```

## Live Validation

Current live account probe after the patch:

```text
total_asset_krw ~= 219,000
krw_cash = 5,855
foreign_cash_krw ~= 86,268
cash_by_currency.USD = 55.51
USD/KRW = 1,554.1
```

The dashboard now shows the remaining broker/account-total difference as `other` instead of foreign cash.

Realtime auto-trading starts when `AUTO_START_REALTIME_TRADING=true`. In the validation run, the engine started automatically and attempted a net-profitable LCFYW SELL, but KIS rejected the order because the overseas market was closed for the day.

## Tests

```text
.venv\Scripts\python.exe -m pytest tests/test_realtime_exit_decision.py tests/test_risk_manager.py tests/test_mock_kis_api.py tests/test_account_dashboard.py tests/test_realtime_modes.py tests/test_market_affordability.py
120 passed
```

## Remaining Risks

- This patch prevents avoidable small loss churn; it does not prove the strategy is profitable.
- KIS holiday/session rejection still blocks otherwise valid orders outside broker-supported trading windows.
- The dashboard may still show a large `other` allocation when KIS account totals include settlement or account-total fields that are not decomposed into cash/stock buckets.

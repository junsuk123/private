# Dynamic Exit Policy

The **single authoritative** resolver for exit thresholds. Replaces the ~12 scattered
sources the audit found (RealtimeTradingConfig, adaptive_exit_policy, ~15 inline
`REALTIME_*` reads, ExecutionPolicy defaults) with one object that is **logged once** so
the effective policy is always auditable.

- Code: `src/app/trading/dynamic_exit_policy.py`
- Config: `config/dynamic_exit_policy.yaml`
- Consumed by: `SharedLiveDecisionEngine.evaluate_exit_for_holding`.

![Dynamic exit policy and loss-exit governance](diagrams/profitability_dynamic_exit.svg)

## Resolution precedence

Highest wins, logged once: explicit `REALTIME_*` env var → `config/dynamic_exit_policy.yaml`
→ built-in default. **Defaults match the previous inline defaults exactly**, so behavior
is unchanged when an env var is unset — only the *source of truth* is unified. `run.ps1`
pins the production values.

## Dynamic levels (formulas)

```
take_profit_rate     = max(min_take_profit_rate,
                           all_in_cost_rate + min_net_profit_buffer
                           + k_vol_take_profit * realized_volatility
                           + liquidity_take_profit_buffer * (1 - liquidity_score)
                           + spread_take_profit_buffer_k * spread_rate)
trailing_giveback    = max(min_trailing_giveback, k_trail_volatility * realized_volatility)
soft_stop_rate       = max(min_soft_stop_rate, k_downside_soft_stop * predicted_downside_risk
                           + realized_volatility)          # tightened by account drawdown
hard_stop_rate       = hard_stop_loss_rate                 # capital circuit-breaker
emergency_stop_rate  = emergency_stop_loss_rate            # capital circuit-breaker
```

Take-profit, profit-lock, trailing-giveback and the soft-stop are dynamic (cost /
volatility / liquidity / spread aware). Hard-stop and emergency-stop stay near their
configured constants.

## Loss-exit governance (neither always-block nor always-allow)

`loss_exit_decision(levels, evidence)` returns `(allowed, reason)`:

- **Always allowed** (capital circuit-breakers, regardless of `allow_loss_exit`):
  hard-stop breach, emergency-stop breach, net tight-stop breach (`stop_loss_net > 0`).
- **Blocked** when `allow_loss_exit` is false, or the loss is within the noise band
  (`|loss| <= noise_band_loss_rate`).
- **Allowed on strong deterioration evidence** (when `allow_loss_exit` is true and loss is
  beyond noise): ontology SELL/REDUCE dominance (`ontology_score <= ontology_sell_dominance`),
  strongly negative model forecast (`predicted_net_return_bps <= -strong_negative_forecast_bps`),
  sharp liquidity/spread deterioration, high-risk market regime, daily loss budget near
  breach, or a soft-stop breach.

This directly addresses the audit's "trapped losing positions" risk: losing positions are
no longer simply always held (until the wide hard stop) or always dumped — they exit when
evidence is strong and hold through noise otherwise.

## Reason codes

`hard_stop_loss`, `emergency_stop_loss`, `stop_loss_net`, `LOSS_EXIT_DISABLED`,
`LOSS_WITHIN_NOISE_BAND`, `ontology_sell_dominance`, `strong_negative_forecast`,
`liquidity_deterioration`, `market_regime_high_risk`, `daily_loss_budget_near_breach`,
`soft_stop_loss`, `HOLD_INSUFFICIENT_DETERIORATION_EVIDENCE`.

The resolved levels are attached to each exit decision's `strategy_metadata`
(`resolved_exit_policy`) for the GUI and audit logs.

## Technical deterioration evidence (evidence-based technical layer)

`evaluate_exit_for_holding` consults the advisory technical layer for
**deterioration evidence** on the held position via
`CompositeTechnicalSignalEngine.evaluate_exit_deterioration`:

- Deterioration signals — VWAP breakdown, momentum loss (negative MACD
  histogram), volatility expansion, high false-breakout risk, liquidity
  deterioration — are surfaced as reason codes
  (`TECHNICAL_EXIT_DETERIORATION`, `VWAP_BREAKDOWN`, `MOMENTUM_WEAKENED`, …) and
  recorded in the exit diagnostics (`technical_exit_deterioration`).
- Strong deterioration applies a **bounded penalty** (≤ 0.5) to the effective
  ontology score, which can bring a **profitable** position into the existing
  `invalid_signal_exit` branch sooner (realize profit as the setup breaks down).
- It **never forces a loss exit**: the hard-stop and emergency-stop circuit
  breakers, and the `REALTIME_ALLOW_LOSS_EXIT` gate, are unchanged and remain the
  sole authorities for exiting below break-even.

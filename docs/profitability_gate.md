# Profitability Gate

The **single authoritative** net-profitability decision used by every BUY path:
strategy candidate generation, the RiskManager, the realtime trading engine, and the
GUI. A BUY is allowed only when its expected **net** return after **all** costs clears a
dynamic minimum edge. Directional signal strength alone is never sufficient.

- Code: `src/app/cost/profitability_gate.py`
- Config: `config/profitability_policy.yaml`
- Cost math: delegated to `src/app/cost/trading_cost_engine.py` (one cost model).

> **2026-07 small-account tuning:** the dynamic buffers were relaxed
> (`volatility_buffer_k` 0.5→0.2, `max_cost_to_alpha_ratio` 0.5→0.7,
> `small_account_extra_net` 0.002→0.0, KR/default floor →0.004) so a ~0.7% expected move
> clears instead of ~1.4%. Full trading-activation write-up:
> [`trading_activation_2026-07.md`](trading_activation_2026-07.md).

![BUY decision flow](diagrams/profitability_decision_flow.svg)

## Decision rule

```
allow_buy = (
    expected_exit_price   >= break_even_exit_price * (1 + min_net_profit_buffer_rate)
    and net_expected_return >= required_min_net_return
    and cost_to_alpha_ratio <= max_cost_to_alpha_ratio
    and spread_rate         <= max_spread_rate
    and spread_alpha_ratio  <= max_spread_alpha_ratio
    and liquidity_score     >= min_liquidity_score
    and expected_slippage_rate <= max_slippage_rate
)
```

## Formulas

```
mid                  = (bid + ask) / 2
spread_rate          = (ask - bid) / mid
gross_expected_return= (expected_exit_price - entry_price) / entry_price
all_in_cost_rate     = buy_fee + sell_fee + sell_tax + entry_slippage + exit_slippage
                       + market_impact + spread_cost + safety_margin        (per notional)
net_expected_return  = gross_expected_return - all_in_cost_rate             (cost-engine exact)
break_even_exit_price= entry_price * (1 + all_in_cost_rate-ish)             (see cost engine)
cost_to_alpha_ratio  = all_in_cost_rate / max(|gross_expected_return|, eps)
spread_alpha_ratio   = spread_rate / max(|gross_expected_return|, eps)
```

### Dynamic minimum net return

```
required_min_net_return = max(
    min_required_net_return[market],                # hard floor per market (KR/US)
    min_net_profit_buffer_rate
      + volatility_buffer_k * realized_volatility
      + liquidity_buffer_max * (1 - liquidity_score)
      + account_buffer(if small account),
)
```

An explicit per-theory `target_net_return` can only **tighten** the requirement, never
drop it below the market floor.

## Rejection reason codes

| Code | Meaning |
|---|---|
| `MISSING_EXPECTED_EXIT_PRICE` | No/invalid predicted exit price |
| `BELOW_BREAK_EVEN_WITH_MARGIN` | Exit below break-even + buffer |
| `BELOW_TARGET_NET_RETURN_AFTER_COST` | Net edge below the dynamic minimum |
| `COST_BURDEN_HIGH` | Cost dominates alpha (`cost_to_alpha_ratio` too high) |
| `SPREAD_TOO_WIDE` | Absolute spread over ceiling |
| `SPREAD_CONSUMES_ALPHA` | Spread too large relative to alpha |
| `LIQUIDITY_TOO_LOW` | Liquidity score below floor |
| `SLIPPAGE_RISK_HIGH` | Expected slippage over ceiling |
| `PROFITABILITY_GATE_REJECTED` | Umbrella code appended by the engine |

## Configuration & precedence

Precedence (highest wins), logged once at startup:
1. Environment variables (backward compatible): `REALTIME_MIN_BUY_NET_RETURN_KR/_US`,
   `REALTIME_MIN_NET_PROFIT_BUFFER_RATE`.
2. `config/profitability_policy.yaml`.
3. Built-in defaults in `profitability_gate.py`.

Initial defaults (tune from realized PnL, not assumed optimal): KR floor `0.008`,
US floor `0.012`, `max_spread_rate 0.003`, `max_cost_to_alpha_ratio 0.5`,
`min_liquidity_score 0.3`.

## Where it is enforced

- **RiskManager** (`risk/manager.py`): the BUY cost checks are one `ProfitabilityGate.evaluate`
  call; a failing decision sets `approved=False` and records `profitability_decision` in
  `metadata`.
- **Realtime engine** (`trading/shared_decision_engine.py`): `evaluate_buy` derives a
  **real** expected exit price from the model's predicted edge (or a conservative
  fallback estimate — no fabricated 100 bps floor) and calls the gate before building the
  order. Rejections short-circuit with the reason codes above.
- **Candidate factory** (`strategy/candidate_factory.py`): replaces its bespoke checks
  with the same gate.

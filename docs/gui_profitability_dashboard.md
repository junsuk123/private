# GUI Profitability Dashboard

The GUI is reorganized around **realized net profitability** and **decision
explainability**. Most of the data already exists server-side after the refactor; the GUI
surfaces it rather than recomputing.

![BUY decision flow](diagrams/profitability_decision_flow.svg)

## Where the data comes from

Every decision now carries a structured profitability object (see
`docs/profitability_gate.md`):

- **Buy decisions** — `diagnostics["profitability_decision"]` and the order
  `strategy_metadata["profitability_decision"]`: `entry_price`, `expected_exit_price`,
  `break_even_exit_price`, `gross_expected_return`, `all_in_cost_rate`,
  `net_expected_return`, `required_min_net_return`, `spread_rate`, `liquidity_score`,
  `cost_to_alpha_ratio`, `rejection_reasons`, `warnings`, `data_quality_flags`. Plus
  `position_sizing` (edge_score, fractional_kelly, position_weight) and
  `model_provider` / `model_is_fallback`.
- **Exit decisions** — `strategy_metadata["resolved_exit_policy"]` (take_profit_rate,
  profit_lock_arm_net, trailing_giveback_rate, soft_stop_rate, hard_stop_rate,
  emergency_stop_rate, allow_loss_exit, block_sell_below_breakeven) plus
  `round_trip_cost_rate`, `net_pnl_rate`, `peak_net_pnl`.
- **RiskManager** — `metadata["profitability_decision"]` and `metadata["cost_breakdown"]`.
- **Preview endpoint** — `/api/.../preview-order` already returns a full `cost.as_dict()`.

## Panels

**Top account panel** — total equity, KRW/USD/orderable cash, today realized gross PnL,
today **net PnL after estimated costs**, win rate, average win, average loss, payoff ratio,
expectancy, daily loss budget remaining, live-armed state. Values that cannot be computed
from available data show `n/a` (never fabricated).

**Candidate / decision cards** — show the **rejection reason first** (human-readable, via
the extended `_HOLD_REASON_TEXT` map incl. `BELOW_TARGET_NET_RETURN_AFTER_COST`,
`BELOW_BREAK_EVEN_WITH_MARGIN`, `COST_BURDEN_HIGH`, `SPREAD_TOO_WIDE`,
`SPREAD_CONSUMES_ALPHA`, `LIQUIDITY_TOO_LOW`, `SLIPPAGE_RISK_HIGH`,
`PROFITABILITY_GATE_REJECTED`), then expected exit price, break-even exit price, all-in
cost rate, net expected return, required minimum net return, spread rate, liquidity score,
and the ontology final action — before the raw strategy score.

**Holding cards** — average entry, current price, break-even price, unrealized gross PnL,
estimated net PnL (via `round_trip_cost_rate`), dynamic take-profit / soft stop / hard
stop (from `resolved_exit_policy`), profit-lock status, and ontology SELL/REDUCE/HOLD state.

**Model & data status** — model provider and fallback state (`trained_model` /
`heuristic_fallback` / `unavailable`), data freshness (stale badges), and
synthetic/estimated-data warnings.

**Logs / errors** — moved to a collapsible lower panel.

> Display-only: the GUI never changes trading/decision logic. If a value is unavailable it
> shows `n/a` rather than a fabricated number.

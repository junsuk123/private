# Short-Term Trading Strategy Design

This document connects the literature-based short-term strategy modules to the current guarded KIS live runtime.

The strategy engines do not call brokerage APIs directly. They produce `StrategyCandidate` records that must pass cost checks, ontology/runtime evidence, `RealityCheckValidator` where required, `RiskManager`, FinalTradeGate, and finally `LiveExecutionCoordinator` before a live order can be submitted.

## Current Runtime Position

`run.ps1` starts the local app on `/account` and enables the guarded realtime loop:

- KIS account, balance, holdings, quotes, and open orders are refreshed into the account dashboard.
- Live training runs continuously when `AUTO_START_LIVE_TRAINING=true`.
- The realtime engine runs when `AUTO_START_REALTIME_TRADING=true`.
- SELL/REDUCE is evaluated before BUY.
- BUY is skipped when `REALTIME_BUY_ENABLED=false`.
- Approved live orders are limit `FinalOrder` objects submitted only through `LiveExecutionCoordinator`.

The short-term strategy modules are evidence and candidate-generation layers inside this flow. They are not an execution shortcut.

## Literature Signals

The strategy layer is based on conservative interpretations of:

- Jegadeesh (1990): short-horizon return predictability and short-term reversal.
- Gao, Han, Li, Zhou (2018): first half-hour return and intraday momentum.
- Brock, Lakonishok, LeBaron (1992): moving-average and trading-range breakout rules.
- Gatev, Goetzmann, Rouwenhorst (2006): pairs and relative-value mean reversion.
- Sullivan, Timmermann, White (1999): data-snooping controls and out-of-sample validation.

These papers provide hypotheses, not guaranteed alpha. The live system still requires current market data, cost feasibility, liquidity, and risk approval.

## Implemented Engines

`src/app/strategy/short_horizon.py` contains:

- `ShortTermReversalEngine`: looks for conservative rebound candidates after recent short-horizon declines.
- `IntradayMomentumEngine`: uses early-session return features such as `ret_open_30m`.
- `TechnicalRuleEngine`: evaluates moving-average and breakout-style signals from available rolling features.

`src/app/strategy/pairs_relative_value.py` contains:

- `PairUniverseBuilder`: selects similar pairs by sector/theme/beta and normalized price path distance.
- `PairRelativeValueEngine`: long-only relative-value candidate generation for underperformers. It does not implement short selling, leverage, or derivatives.

## Feature Builder

`src/app/features/short_horizon_features.py` builds minute/day features such as:

- returns: `ret_1m`, `ret_3m`, `ret_5m`, `ret_15m`, `ret_30m`, `ret_1d`
- opening/pre-close returns: `ret_open_10m`, `ret_open_30m`, `ret_preclose_30m`
- risk/liquidity: `realized_volatility_5m`, `realized_volatility_30m`, `volume_zscore`, `spread_rate`, `orderbook_depth_score`, `liquidity_score`
- context: `market_alignment_score`, `time_of_day_weight`

Data after `as_of` is filtered to avoid look-ahead bias. Missing data is represented with `None`, `missing_fields`, and `is_valid=false`.

## Candidate Factory

`src/app/strategy/candidate_factory.py` merges strategy outputs and keeps only candidates that pass cost and feasibility checks:

- `expected_exit_price > 0`
- `net_expected_return > target_net_return`
- `gross_expected_return > break_even_return + safety_margin`
- `cost_to_alpha_ratio < max_cost_to_alpha_ratio`
- `spread_rate < max_spread_rate`
- `liquidity_score > min_liquidity_score`

Ranking favors net edge after costs:

```text
excess_return_after_cost = net_expected_return - target_net_return
ranking_score = excess_return_after_cost
                * confidence
                * ontology_score
                * liquidity_score
                * risk_adjustment
```

## Cost And Risk Gates

`TradingCostEngine` reflects buy/sell commission, sell tax, slippage, spread, market impact, and safety margin. Both the candidate stage and `RiskManager` use this cost model.

A candidate that passes local cost screening still does not become an order. Final executable intent must pass:

- account and position limits
- positive expected net return after cost
- break-even plus safety margin
- spread and liquidity limits
- ontology/runtime risk tags
- source freshness and quote/orderbook quality
- live mode verification and KIS readiness
- idempotency/open-order checks

## Ontology Connection

`src/app/graph/trading_strategy_semantics.py` converts ranked candidates and `ontology_tags` into semantic evidence records.

Positive evidence can include:

- `ShortTermReversalBuy`
- `IntradayMomentumBuy`
- `PairMeanReversionBuy`
- `TechnicalBreakoutBuy`
- `CostEfficientTrade`
- `RealityCheckPassed`

Risk evidence can include:

- `BidAskBounceRisk`
- `FalseBreakoutRisk`
- `SpreadTooWide`
- `SlippageRiskHigh`
- `CostBurdenHigh`
- `DataSnoopingRisk`
- `NoOutOfSampleValidation`

`TradeForbidden` can be mapped into `RiskManager` rejection through ontology tags.

## Reality Check

`src/app/evaluation/reality_check.py` provides `RealityCheckValidator`.

Validation reports include:

- gross/net total return
- gross/net win rate
- average cost per trade
- average net profit per trade
- break-even failure ratio
- fee-converted loss ratio
- cost-to-alpha ratio mean/median
- out-of-sample net return
- out-of-sample Sharpe
- max drawdown after cost
- block bootstrap p-value

Passing validation can attach `RealityCheckPassed`. Failing validation attaches risk evidence such as `NoOutOfSampleValidation` or `DataSnoopingRisk`.

## Configuration

The main strategy config is `config/short_horizon_strategies.yaml`.

Important sections:

- `short_term_reversal`
- `intraday_momentum`
- `technical_rule`
- `pair_relative_value`
- `strategy_candidate_factory`
- `reality_check`
- `parameter_reestimation`
- `execution`

Historical strategy defaults may still describe paper or dry-run execution for isolated strategy tests. The current operational entry point is `run.ps1`, which starts the guarded live-capable realtime engine and keeps the actual broker boundary in `LiveExecutionCoordinator`.

## Operational Reading

Use `/account` as the primary view. When there are no trades, inspect rejection reasons first:

- `MODEL_FEATURE_UNAVAILABLE`: required live model features such as fresh quote/orderbook data are missing.
- `WIDE_SPREAD`: spread cost is too high for the expected edge.
- `LOW_LIQUIDITY`: depth/liquidity is not enough for a safe fill.
- `FALLBACK_SCORE_BELOW_THRESHOLD`: ontology/runtime fallback score is not strong enough.
- `INSUFFICIENT_CASH_FOR_ONE_SHARE`: available cash cannot buy one share after sizing and price checks.
- `MODEL_AUXILIARY_ONLY_NEEDS_CONFIRMATION`: model evidence exists but deterministic confirmation is missing.

These are expected fail-closed behaviors. The goal is not to maximize order count; it is to let through only candidates with current data, feasible execution, and positive expected net return after cost.

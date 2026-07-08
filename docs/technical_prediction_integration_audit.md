# Technical Prediction Layer — Integration Audit (Phase 0)

Branch: `feature/evidence_based_technical_prediction_layer`
Date: 2026-07-09

This audit reconciles the task specification against the **actual** repository so
later phases build on real modules, reuse existing math, and never bypass a
safety gate. It also records the baseline test status so new failures are
distinguishable from pre-existing ones.

## 1. Baseline test status (pre-existing failures)

Full `python -m pytest` on the branch tip **before** any technical-layer code:
**465 passed, 5 failed** (≈4m21s). All 5 failures are pre-existing and unrelated
to this feature; they must not be attributed to the technical layer:

| Test | Area | Note |
|------|------|------|
| `test_profitability_gate.py::...test_small_account_adds_extra_required_net` | cost/ | Stems from the earlier YAML profitability relaxation, not touched here |
| `test_profitability_refactor_integration.py::...test_gross_positive_net_negative_buy_is_rejected` | cost/ | Same root cause (gate approves a case the test expects rejected) |
| `test_realtime_modes.py::...affordable_us_discovery_adds_symbols_for_small_usd_balance` | trading/ | USD affordable-discovery sizing; pre-existing |
| `test_realtime_modes.py::...affordable_us_discovery_excludes_held_and_recent_buy_tickers` | trading/ | Pre-existing |
| `test_realtime_modes.py::...buy_candidates_include_affordable_discovery_when_context_empty` | trading/ | Pre-existing |

**Acceptance-criterion mapping:** "existing safety tests continue to pass or any
pre-existing failures are documented separately" — done here.

## 2. Spec-path → real-path reconciliation

Most spec paths exist as written. The ones that differ:

| Spec reference | Reality | Action |
|----------------|---------|--------|
| `models/live_signal_predictor.py` | ✅ exists (`LiveSignalPredictor.predict(frame) -> LiveSignalPrediction`) | reuse |
| `cost/profitability_gate.py` `ProfitabilityGate` | ✅ exists; `evaluate(ProfitabilityInput) -> ProfitabilityDecision`; **already accepts `expected_exit_price`** | feed technical exit price in Phase 7 |
| `cost/trading_cost_engine.py` `TradingCostEngine` | ✅ exists; `estimate(...) -> CostBreakdown`; costs are **rates** (×10_000 for bps); config is `config/trading_costs.json` (JSON) | reuse for net labels/prediction |
| `trading/dynamic_exit_policy.py` `DynamicExitPolicy` | ✅ exists; `resolve(...) -> ResolvedExitLevels`, `loss_exit_decision(...)` | feed deterioration evidence in Phase 8 |
| `trading/shared_decision_engine.py` | ✅ exists; class is **`SharedLiveDecisionEngine`** (not `SharedDecisionEngine`); `evaluate_buy(...)`, `evaluate_exit_for_holding(...) -> SharedDecisionResult` | primary integration point (Phase 7) |
| `strategy/candidate_factory.py` + `rule_based.py` | ✅ both exist; `StrategyCandidateFactory.build(...)`, `generate_strategy_signals(...)` | integrate CompositeTechnicalSignalEngine here |
| `BuyCandidate/SellCandidate/ReduceRiskCandidate/HoldOrWatch` classes | ❌ **not Python classes** | Use `StrategyCandidate` + `OrderAction` enum (`BUY/SELL/REDUCE/HOLD/WATCH`, `schemas/domain.py`) and ontology candidates (`tr:BuyCandidate/SellCandidate/HoldCandidate`) |
| `models/npu_short_horizon_predictor.py` | ❌ wrong name | Real: `realtime/short_horizon_npu_predictor.py` (`ShortHorizonNpuPredictor`); backends in `models/inference_backend.py` (`CpuSignalModel`, `OpenVinoNpuSignalModel`) |
| `FinalTradeGate` (class) | ❌ **conceptual name only** | Enforced by `RiskManager.validate(...)` + `LiveExecutionCoordinator._validate_final_order` (requires `FinalOrder` + `OrderType.LIMIT`) |
| ontology `.ttl` files | ✅ all exist under `src/app/ontology/`; single prefix `tr: <https://example.com/ontology/trading#>` | extend in Phase 6 |

## 3. Indicators — REUSE, do not duplicate (critical)

`src/app/features/indicator_engine.py` is the **canonical TA library** and
already implements: `sma, ema, macd(12/26/9), rsi(14), bollinger_bands(20,2),
atr(14), obv, stochastic_k/d, mfi, historical_volatility, volume_spike_ratio,
period_return, distance_from_extreme, rolling_drawdown, candle/​shadow ratios`.

**Genuinely missing** (and therefore added by the technical layer): **VWAP**
(currently computed inline in `live_feature_frame.py`) and **Donchian
channels** (closest existing proxy is `distance_from_extreme`).

`src/app/technical/indicators.py` therefore **delegates** to `indicator_engine`
for all existing math and adds only VWAP, Donchian, rolling z-score, spread-bps,
and orderbook-imbalance, plus typed result wrappers (`MacdResult`,
`BollingerResult`, `DonchianResult`). `src/app/indicators/engine.py` is NOT a TA
library (it builds fundamental `IndicatorSnapshot` fixtures) — do not confuse.

Existing short-horizon features (`features/short_horizon_features.py`,
`ShortHorizonFeatureBuilder`, `TickerRollingFeatureState`) already produce
rolling returns, realized volatility, volume z-score, spread rate, orderbook
depth, and liquidity score — the technical layer complements these.

## 4. Data types to build on

- `OHLCVBar(ticker, as_of, open, high, low, close, volume)` — `features/schemas.py`.
- Realtime store (`data/realtime_store.py`, `RealtimeMarketDataStore`):
  `latest_tick`, `latest_orderbook`, `recent_ticks(symbol, since)`,
  `recent_orderbooks(symbol, since)`. Types in `data/realtime_types.py`:
  `RealtimeTradeTick`, `RealtimeOrderbookSnapshot` (+ derived `best_bid/best_ask/
  spread_bps/imbalance`), `RealtimeMinuteBar` (has `vwap`, `spread_bps`,
  `orderbook_imbalance`, `liquidity_score`, `volatility`), `MarketDataHealth`.
- Live feature vector schema `LIVE_SHORT_HORIZON_SCHEMA` (`features/feature_schema.py`):
  fixed ordered names `return_30s, return_1m, return_3m, distance_from_vwap,
  spread_bps, orderbook_imbalance, bid_depth, ask_depth, depth_ratio,
  liquidity_score, realized_volatility_3m, max_drop_3m,
  cost_to_volatility_ratio, principal_cushion_ratio, news_sentiment`. Adding
  columns requires a schema + `feature_schema_hash` bump and model retrain — to
  be handled carefully in Phase 4 (additive, behind availability diagnostics).

## 5. Ontology / graph integration points (Phase 6)

- `graph/rdf_adapter.py`: `attach_scoring_provenance` creates per-ticker
  `tr:EvidenceItem` nodes — where technical evidence projects (under the
  existing `tr:TechnicalIndicator` class), via `supportsSignal`/`contradictsSignal`.
- `graph/reasoner.py`: `SemanticPolicyScorer` (aka `OntologyReasoner`) is the
  numeric policy scorer — consumes evidence weights; **advisory only**.
- `graph/owl_reasoner.py` + `graph/semantic_materializer.py`: logical entailment
  only, never authorizes.
- SHACL in `ontology/trading_shapes.ttl` (closed-world validation).

## 6. Authoritative gates (must remain sole authorities)

`RiskManager.validate(intent, account, market, ...)` (instantiates
`TradingCostEngine`, `ProfitabilityGate`, `PrincipalProtectionEngine`) →
`LiveExecutionCoordinator.submit_final_order` (limit-only). The technical layer
produces **evidence and a conservative expected-exit-price only**; it never
constructs a `FinalOrder`.

## 7. Phase status

- [x] **Phase 0** — this audit + baseline recorded.
- [x] **Phase 1** — `src/app/technical/indicators.py` (delegates + VWAP/Donchian/
      z-score/spread/imbalance) with `tests/test_technical_indicators.py` (36 tests, green).
- [ ] Phases 2–11 — pending (regime, signals, labels, prediction, ontology,
      shared-engine integration, profitability/exit alignment, replay, GUI, docs).
      Phases that modify safety-critical modules (4, 6, 7, 8, 10) will be done
      additively, each behind availability diagnostics and verified in isolation.

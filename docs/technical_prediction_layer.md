# Evidence-Based Technical Prediction Layer

`src/app/technical/` adds a short-horizon predictive layer grounded in
evidence-backed technical trading methodologies. **Everything it produces is
advisory evidence.** It never creates or submits an order and never relaxes a
safety gate. `TradingCostEngine`, `ProfitabilityGate`, `PrincipalProtectionEngine`,
`RiskManager`, and `LiveExecutionCoordinator` (limit-only) remain the sole
authorities; the ontology stays advisory.

> This is **not** guaranteed-profit logic. Small-account short-horizon KR
> day-trading is structurally close to negative-expectancy after round-trip
> cost. Validate with realized PnL and the replay harness
> (`docs/technical_prediction_validation.md`) before trusting it.

## Module map

| Module | Role |
|--------|------|
| `technical/indicators.py` | Delegates SMA/EMA/MACD/RSI/Bollinger/ATR/volume-spike to `features/indicator_engine.py` (single source of truth); adds VWAP, Donchian, rolling z-score, spread-bps, orderbook imbalance. Pure, NaN-safe, no pandas. |
| `technical/regime.py` | Rule-based `TechnicalRegimeClassifier` → 8 regimes with confidence, reasons, feature contributions. Risk gates first. |
| `technical/signals.py` | Methodology providers + `CompositeTechnicalSignalEngine` (regime gating, mandatory VWAP/volume confirmation, no-single-indicator-BUY). |
| `technical/feature_builder.py` | OHLCV(+orderbook) → `TechnicalFeatureSet`; also maps a live feature frame. |
| `technical/labels.py` | No-look-ahead supervised labels + net-after-cost via `TradingCostEngine`; synthetic-source guard. |
| `technical/prediction.py` | `TechnicalPredictionEngine`: conservative expected exit price / net return / downside; returns `NO_TRADE` when weak. Never approves. |
| `technical/replay.py` | Walk-forward, no-look-ahead replay evaluation. |
| `technical/policy.py` | Loads `config/technical_prediction_policy.yaml` (env overrides, logged). |
| `technical/reason_codes.py` | Stable advisory reason-code constants. |
| `graph/technical_evidence.py` | Projects signals into `KnowledgeGraph` triples + RDF evidence. |
| `technical/decision_feed.py` | Bounded per-symbol feed for the GUI panel. |

## Methodologies — what, why, when enabled, when blocked

Each provider outputs a signed score `[-1,1]`, confidence `[0,1]`, a
**conservative** expected edge (bps; derived from measured volatility × horizon
with a <1 capture fraction — **no fabricated alpha floor**), horizon, supporting
/ contradicting features, and reason codes.

1. **Momentum / trend-following** (Jegadeesh–Titman; Brock–Lakonishok–LeBaron MA
   rules). EMA gap, MACD histogram, short return, persistence. *Enabled* in
   `TREND_UP` / `BREAKOUT_CANDIDATE`; contributes SELL evidence in downtrends.
2. **Breakout / trading-range break**. Donchian high + volume + VWAP + false-
   breakout risk. *Enabled* in `BREAKOUT_CANDIDATE` / `TREND_UP`; **blocked
   without volume confirmation**.
3. **Mean reversion** (short-term reversal). RSI / Bollinger %b extremes,
   distance from VWAP. *Enabled* in `RANGE_BOUND` / `MEAN_REVERSION_CANDIDATE`;
   **blocked in a strong downtrend** (`MEAN_REVERSION_BLOCKED_BY_DOWNTREND`).
4. **VWAP / volume / liquidity** — the **mandatory confirmation layer**. A BUY
   from momentum/breakout requires price above VWAP with supportive flow;
   otherwise `VWAP_BREAKDOWN` and no BUY evidence.
5. **Volatility band / regime** — risk-oriented. Feeds the regime classifier and
   the dynamic-exit deterioration signal; a `HIGH_VOLATILITY_RISK` regime blocks
   BUY.

### Regime-first policy

`HIGH_VOLATILITY_RISK` / `LOW_LIQUIDITY_RISK` / `NO_TRADE` → **BLOCK_BUY**.
Otherwise the regime selects which methodologies are preferred; out-of-regime
methods are down-weighted, not silently dropped.

## How BUY evidence flows (and cannot bypass a gate)

```
realtime frame ─► TechnicalFeatureSet ─► CompositeTechnicalSignalEngine
      │                                        │ (regime gate + VWAP confirm)
      ▼                                        ▼
 model prediction ───────► TechnicalPredictionEngine ──► expected_exit_price
                                                             │ (conservative;
                                                             │  min(model, tech))
                                                             ▼
                                    ProfitabilityGate (authoritative net check)
                                                             ▼
                                    RiskManager.validate ──► LiveExecutionCoordinator
                                                             (limit orders only)
```

- No single indicator triggers a BUY; methodology agreement **and** VWAP/volume
  confirmation **and** a non-blocking regime are required to form BUY evidence.
- The technical expected exit price is *preferred* for the gate but never
  inflated above the honest model estimate.
- A positive technical signal with negative net-after-cost is **rejected** by
  the ProfitabilityGate.

## SELL / REDUCE (exit) evidence

`evaluate_exit_for_holding` consults `evaluate_exit_deterioration` (VWAP
breakdown, momentum loss, volatility expansion, false-breakout, liquidity drop).
Strong deterioration applies a **bounded** penalty (≤0.5) to the effective
ontology score so a *profitable* position can exit sooner. It **never forces a
loss exit** — hard/emergency stops and the loss-exit gate are unchanged.

## Ontology evidence

Technical outputs project to `KnowledgeGraph` triples
(`supportsSignal`/`contradictsSignal`/`increasesRiskOf`) mapped to existing
research-theory objects, so `SemanticPolicyScorer` weights them like any other
evidence. Richer RDF (`tr:TechnicalSignal` + methodology subclass + data
properties + provenance `tr:EvidenceItem` + regime) is available for the graph
UI, and a live SHACL shape (`LiveTechnicalEvidenceShape`) requires symbol,
expected edge, confidence, horizon, and freshness. OWL/SHACL never materialize a
`FinalOrder`.

## NPU / CPU

Deterministic indicator/regime/signal computation runs on CPU. Only the trained
model inference uses OpenVINO/NPU where available, with an identical-schema CPU
fallback (`models/inference_backend.py`). The technical layer is reusable
unchanged on the Raspberry Pi CPU-only runtime.

## Configuration

`config/technical_prediction_policy.yaml`: `enabled`, methodology weights,
per-methodology minimum confidence, regime thresholds, prediction min-confidence
/ per-horizon net-edge buffers, risk blocks, diagnostics flags. Env vars of the
same name override; effective values are logged at load. Thresholds are never
relaxed automatically.

## Known limitations / next tuning

- Expected-move capture fraction (0.5) and downside multiple (1.5) are priors;
  calibrate against replay `avg_edge_error_bps`.
- Technical features are computed from the tick series in the live frame; true
  minute-bar OHLC would sharpen Donchian/ATR — a future feed enhancement.
- The trained model must be **retrained** on the `live_short_horizon_v2` schema
  (6 new technical columns) before it contributes; until then it falls back and
  the technical layer supplies the edge estimate.

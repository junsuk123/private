# Live Short-Horizon Model Decision Record

## Current Runtime Contract

As of the current `run.ps1` entry point, the system is a guarded KIS live-capable realtime runtime. KIS realtime collection, read-only account probing, periodic live short-horizon training, and the independent realtime trading loop can start automatically. Numeric ontology/candidate evidence scoring requests OpenVINO `NPU` and falls back to CPU when unavailable; final action selection, graph explanations, risk checks, order gating, idempotency, and broker submission remain deterministic CPU-controlled paths. NPU output is evidence, not trade authorization.


Date: 2026-07-02

## Decision

The live short-horizon model is enabled for inference when the latest artifact is live-eligible, but it is an auxiliary decision component. It cannot approve a live BUY by itself.

Current policy:

- `LiveSignalPredictor` may score a candidate when KIS realtime feature frames are fresh.
- `REALTIME_MODEL_AUXILIARY_ONLY=true` keeps the model from becoming a sole execution authority.
- Model-only BUY is rejected with `MODEL_AUXILIARY_ONLY_NEEDS_CONFIRMATION`.
- BUY still needs ontology support or runtime broker evidence, cash feasibility, spread/liquidity controls, and `RiskManager` approval.

## Current Validation Snapshot

The latest observed live-eligible artifact in this workspace reported:

- training rows: about `17,648`
- validation rows: about `5,295`
- AUC: about `0.563`
- precision@k: about `0.788`
- top-k average forward net return: about `51bps`
- live eligibility: true

Interpretation:

- The model is meaningful enough for candidate ranking and top-k assistance.
- The edge is not strong enough to override ontology, realtime market-quality checks, cash checks, or risk controls.
- If feature frames are stale or missing, the model is unavailable for that candidate and the engine must fall back to ontology/runtime rules.

## Feature Requirements

The model uses `LIVE_SHORT_HORIZON_SCHEMA` from `src/app/features/feature_schema.py` and feature frames from `LiveFeatureFrameBuilder`.

Live BUY model scoring requires:

- KIS realtime tick source
- KIS realtime orderbook source
- max quote age within `LIVE_FEATURE_MAX_QUOTE_AGE_MS`
- max orderbook age within `LIVE_FEATURE_MAX_ORDERBOOK_AGE_MS`
- finite feature values
- feature schema hash matching the deployed artifact

If these are not satisfied, the candidate can show:

```text
MODEL_FEATURE_UNAVAILABLE:MARKET_DATA_NOT_LIVE_BUY_ELIGIBLE:QUOTE_STALE,ORDERBOOK_STALE
```

## Training Loop

The default `run.ps1` runtime starts periodic training:

```text
AUTO_START_LIVE_TRAINING=true
LIVE_TRAINING_INTERVAL_SECONDS=60
```

The loop:

1. Builds live feature frames from `data/store/realtime_market_data.sqlite3`.
2. Journals feature frames to `logs/live-feature-frames.jsonl`.
3. Builds triple-barrier short-horizon labels after costs.
4. Trains the live short-horizon artifact.
5. Saves versioned artifacts under `data/models/live_short_horizon/`.
6. Promotes only live-eligible artifacts to `latest.json`.

## Label Methodology & Confounding Controls

Realized PnL is a confounded target: a stock's forward return is mostly the common market/sector move plus noise, so a naive "did it go up" label credits whatever features happen to be present (including news sentiment) for beta they did not cause. The pipeline mitigates this at three levels:

- **Market-adjusted (abnormal) label (R1).** `_market_adjust_rows` demeans each frame's forward net return against other symbols that entered in the same short time bucket (`LIVE_LABEL_MARKET_BUCKET_SECONDS`, default 300s). The binary label is the sign of the resulting *abnormal* return, so a name that merely rode a market rally is not labeled positive. When a bucket has fewer than 2 distinct symbols the market factor is unidentifiable and it degrades gracefully to the raw label. Disable with `LIVE_LABEL_MARKET_ADJUST=false`.
- **Multivariate conditioning (B).** `news_sentiment` is one feature in the logistic/linear fit alongside spread, momentum, volatility, flow, etc., so its coefficient reflects marginal predictive value *given* the co-features — not a univariate correlation.
- **Purged + embargoed holdout (R3).** Triple-barrier labels look up to `LIVE_LABEL_HORIZON_SECONDS` forward, so train rows near the split leak validation prices into their own labels. The holdout fit drops train rows whose label window (`horizon + LIVE_MODEL_EMBARGO_SECONDS`) reaches the validation start, so the eligibility AUC is leakage-free.

The outcome-calibrated news trust (`news_confirm_scale`) is computed on these *abnormal* returns and kept a mild, well-gated multiplier (clamped 0.5–1.5, `NEWS_TRUST_MIN_SAMPLES` default 30). Even with these controls this remains controlled correlation, not proven causation — the model stays advisory and `RiskManager` is the final gate.

## Live Decision Integration

The model participates in `SharedLiveDecisionEngine.evaluate_buy`.

The engine combines:

- model probability and expected net return when available
- ontology flow/evidence score
- KIS broker quote refresh
- realtime spread and liquidity
- account cash by currency
- adaptive buy threshold from `AutoTuningEngine`
- runtime probe/fallback support for very small buys
- deterministic `RiskManager`

The model is never allowed to bypass:

- `REALTIME_BUY_ENABLED`
- `REALTIME_MODEL_AUXILIARY_ONLY`
- spread/liquidity limits
- one-share cash check
- source freshness checks
- `RiskManager`
- `LiveExecutionCoordinator`

## Tunable Environment

- `LIVE_MODEL_MIN_AUC`
- `LIVE_MODEL_MIN_PRECISION_AT_K`
- `LIVE_MODEL_MIN_AVG_RETURN_BPS`
- `LIVE_MODEL_TOP_K_FRACTION`
- `LIVE_LABEL_MIN_NET_RETURN_BPS`
- `LIVE_LABEL_HORIZON_SECONDS`
- `LIVE_LABEL_TAKE_PROFIT_BPS`
- `LIVE_LABEL_STOP_LOSS_BPS`
- `REALTIME_MODEL_AUXILIARY_ONLY`
- `REALTIME_REQUIRE_ONTOLOGY_FOR_MODEL_FALLBACK`

## Operational Reading

When the GUI shows no BUY submissions, first distinguish:

- model unavailable because feature frames are stale
- model available but model-only support rejected
- candidate rejected by spread/liquidity/cash
- candidate rejected by final risk gates

The correct response to wide spreads or thin liquidity is usually no trade, not threshold relaxation.

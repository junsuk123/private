# Model / NPU Provider & Validation

![Profitability architecture](diagrams/profitability_architecture.svg)

## The live predictor vs the dead NPU predictor

- **Live path**: `app.models.live_signal_predictor.LiveSignalPredictor` produces
  `LiveSignalPrediction(probability_success, expected_net_return_bps, uncertainty_score,
  approved, reason_codes, model_artifact_id, feature_schema_hash, provider, is_fallback)`.
  It loads `data/models/live_short_horizon/latest.json` via `ModelArtifactRegistry`
  (a linear logistic + linear-regression model — **no OpenVINO/NPU on the live path**).
- **Not live**: `src/app/realtime/short_horizon_npu_predictor.py` is imported only by its
  own test. It is **not** wired into the live decision path. Do not assume its outputs are
  used.

## Provider / fallback visibility (Phase 5)

`LiveSignalPrediction` now carries `provider` and `is_fallback`. The engine resolves and
surfaces the effective provider into buy diagnostics and order `strategy_metadata`:

| provider | meaning |
|---|---|
| `trained_model` | Live-eligible trained artifact produced the prediction |
| `heuristic_fallback` | Model unavailable/mismatched → ontology/adaptive fallback drove the decision |
| `unavailable` | No usable prediction |

On failure the predictor **raises** (e.g. `NO_LIVE_ELIGIBLE_MODEL_ARTIFACT`,
`MODEL_FEATURE_SCHEMA_MISMATCH`); `evaluate_buy` catches it, leaves `prediction=None`,
and routes through the fallback with `model_ok=False`. **Fallback never fabricates a
positive edge** and never increases size: the ProfitabilityGate judges the (conservatively
estimated) edge honestly, and position sizing scales by confidence (which fallback lowers).

## Expected exit price is a real predicted edge

The old buy path fabricated `expected_return_bps = 100` (plus `fallback_score*300` and a
policy floor). That is removed. The expected exit price now comes from the model's
`expected_net_return_bps` when available, or a conservative
`fallback_score * REALTIME_FALLBACK_EDGE_BPS_PER_SCORE` estimate otherwise — and the
ProfitabilityGate rejects it if the net edge does not clear the dynamic minimum.

## Validation metrics that matter

See `data/models/README.md`. Labels must be computed from realized outcomes **after** fees,
tax, spread, and slippage. Track: precision on profitable trades, false-positive rate,
calibration error, and realized expectancy by decile. Do not increase order size based on
model output unless validation metrics clear the eligibility gate. Training/metrics:
`scripts/train_live_short_horizon_models.py` (+ `app.models.live_model_trainer`,
purged/embargoed holdout).

# Live model artifacts & provider tiers

This directory holds the fitted artifacts consumed by the **live** short-horizon
trading path. The authoritative predictor is
`app.models.live_signal_predictor.LiveSignalPredictor`, and the buy decision that
uses it is `app.trading.shared_decision_engine.SharedLiveDecisionEngine.evaluate_buy`.

## Layout

- `live_short_horizon/` — live-eligible artifacts. `latest.json` is the single file
  the predictor loads every prediction (atomic-replaced by the trainer so a partial
  write is never read). Only a **live-eligible** artifact overwrites `latest.json`;
  a re-train that fails eligibility preserves the previous eligible model.
- `live_short_horizon_demo/` — deterministic demo fixture (never live-eligible).
- `realtime_supervised/`, `hypothetical_testing/` — offline supervised timing models
  and hypothetical PnL reports; **not** on the live buy/sell path.
- `strategy_utility/rgcn_shadow.npz` + `.json` — despite the historical filename,
  this is the live trust-gated 8-strategy ontology/GNN checkpoint. The JSON model
  card records schema, coverage, checkpoint hash, and authorization scope.

## Strategy-utility ontology/GNN authority

The ontology and R-GCN form one pipeline:

1. the closed-world ontology gate creates the admissible strategy mask;
2. the R-GCN learns relation weights and conditional win/loss utility inside it;
3. `GnnRealtimeTrustEvaluator` validates model calibration on forward live data;
4. only strategies in `trusted_strategy_ids` can own a new live entry;
5. profitability, risk, execution quality, and KIS gates remain authoritative.

`live_authorized=true` in the model card means the checkpoint schema and training
coverage are valid for the runtime. It does **not** mean every strategy can trade.
Use `GET /api/gnn/realtime-trust` to distinguish `calibrated_strategy_ids` from
entry-authorized `trusted_strategy_ids`.

The current output expectation is:

```text
net_bps = P(win) * E[net win] - P(loss) * E[net loss]
gross_bps = net_bps + expected all-in costs
```

Forward validation uses the first strategy target/stop reached within the strategy
horizon, then subtracts the same all-in cost engine used by live execution.

## Provider tiers (what actually drives a decision)

The engine surfaces its provider/fallback state in the buy diagnostics and in the
order `strategy_metadata` as `model_provider` / `model_is_fallback`, so the GUI can
show which tier fired. `LiveSignalPrediction` also carries `provider` / `is_fallback`.

| `model_provider`      | `is_fallback` | Meaning |
|-----------------------|---------------|---------|
| `trained_model`       | `false`       | The fitted live-eligible artifact loaded, the feature-schema hash + feature order matched, and the prediction was **approved** (cleared probability / expected-net-return / uncertainty thresholds). The trained model drives the decision. |
| `heuristic_fallback`  | `true`        | The model produced a prediction but it was **not approved**; the ontology + adaptive fallback score drives the decision instead. |
| `unavailable`         | `true`        | The predictor produced no prediction at all — missing/ineligible `latest.json`, `MODEL_FEATURE_SCHEMA_MISMATCH`, `MODEL_FEATURE_ORDER_MISMATCH`, or `LIVE_SIGNAL_MODEL_INFERENCE_DISABLED`. The heuristic fallback path is used. |

### Fallback behavior (safety)

- The predictor **raises** on every failure (it never fabricates a model score);
  `evaluate_buy` catches it into `prediction_error`, leaves `prediction=None`, and
  falls back to the ontology / adaptive path.
- Fallback **never increases size**. Sizing is owned by `app.risk.position_sizing`
  off `confidence`; on any fallback the engine both **flags** the decision
  (`model_is_fallback=true`) and **lowers confidence** (`max(0.35, confidence-0.1)`).
  A fallback buy still requires ontology / runtime-execution support and must clear
  the unified `ProfitabilityGate` (net-return-after-cost), which rejects
  negative-expectancy trades regardless of tier.

## Validation metrics that matter

Produced by `app.models.live_model_trainer.train_live_short_horizon_model` and stored
in each artifact's `metrics` block. A **purged + embargoed** time-ordered holdout is
used so triple-barrier labels do not leak validation-period prices into training.

- **Precision on profitable trades** — `precision_at_k` (fraction of the top-scored
  candidates that were truly profitable after cost). This is the key precision signal.
- **False-positive rate** — the complement of precision@k on the traded slice; keep it
  low because a false positive is a real after-cost loss.
- **Calibration error** — how well `probability_success` matches realized win rate by
  score bucket; a mis-calibrated model over/under-sizes via `confidence`.
- **Realized expectancy by decile** — `avg_forward_net_return_bps_top_k` is the current
  decile-0 proxy; expectancy should rise monotonically with score decile.
- Supporting: `auc`, `holdout_evaluated`, `validation_example_count`,
  `holdout_train_count`, positive/negative label counts.

### Live-eligibility gate

An artifact only becomes live-eligible (and only then overwrites `latest.json`) when
**all** hold (env-overridable):

- `auc >= LIVE_MODEL_MIN_AUC` (default 0.55)
- `precision_at_k >= LIVE_MODEL_MIN_PRECISION_AT_K` (default 0.35)
- `avg_forward_net_return_bps_top_k > LIVE_MODEL_MIN_AVG_RETURN_BPS` (default 0.0)

Otherwise `reason_codes` includes `METRICS_BELOW_LIVE_THRESHOLDS` and the previous
eligible model is preserved.

## Producing / refreshing validation metrics

Labels MUST be realized outcomes measured **after** fees, tax, spread, and slippage
(`forward_net_return_bps`); `label=1` when that after-cost forward return clears the
configured minimum. Never label off gross/pre-cost returns.

```bash
# Real dataset (JSONL rows: {"features": {...}, "label": 0|1,
#   "forward_net_return_bps": <after-cost>, "as_of": "<iso8601>"})
PYTHONPATH=src python scripts/train_live_short_horizon_models.py \
    --dataset path/to/labeled_rows.jsonl \
    --model-dir data/models/live_short_horizon

# Deterministic dry-run fixture (writes to live_short_horizon_demo, never eligible)
PYTHONPATH=src python scripts/train_live_short_horizon_models.py --demo-fixture
```

The script prints `live_eligible`, `artifact_id`, and `reason_codes`; the full metric
block is written into the artifact JSON. Relevant env knobs: `LIVE_MODEL_HOLDOUT_FRACTION`,
`LIVE_LABEL_HORIZON_SECONDS`, `LIVE_MODEL_EMBARGO_SECONDS`, `LIVE_MODEL_TOP_K_*`,
and the eligibility thresholds above.

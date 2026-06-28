# NPU and Realtime Hot Path Optimization Audit

## Applied

- `ontology_filter_1` now uses a NumPy vectorized CPU rules path by default.
- The legacy loop path remains available with `ONTOLOGY_FILTER1_BACKEND=legacy`.
- Optional Rust/PyO3 native screening core is implemented under `native/screening_core`.
- Python automatically uses `screening_core` when installed and otherwise falls back to NumPy.
- Human-readable screening traces are materialized only for selected candidates and a compact reject sample.
- NPU candidate scoring keeps top-k before graph materialization.
- NPU scorer profile now separates:
  - `feature_build_ms`
  - `inference_ms`
  - `topk_ms`
  - `postprocess_ms`

## Preserved Safety Boundaries

- Screening and NPU outputs remain evidence only.
- They do not submit orders.
- Orders still require strategy construction, `RiskManager`, final gates, and manual approval.
- Live trading remains disabled by default.
- Production/realtime indicator paths do not promote `reference:`, `sample-indicator:`, or `demo-indicator:` source ids as trusted evidence.
- Demo/offline sample indicators are only enabled through explicit demo context wiring.

## Rolling Feature Cache

- `TickerRollingFeatureState` keeps per-ticker ordered ring buffers for short-horizon bars.
- It preserves no-lookahead behavior with `as_of` filtering.
- Tests compare rolling outputs to the batch builder for returns, volatility, volume z-score, market alignment, and missing fields.

## Remaining Native Work

- Build/install the optional native module on machines with Rust:
  - `cd native/screening_core`
  - `python -m pip install maturin`
  - `maturin develop --release`
- The Rust/PyO3 extension accepts contiguous arrays and returns `selected_indices`, `rejected_indices`, `accepted_indices`, `scores`, `reason_masks`, and `hard_reject_masks`.
- The Rust/PyO3 extension releases the GIL during score and top-k phases.

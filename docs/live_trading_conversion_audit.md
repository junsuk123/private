# Live Trading Conversion Audit

Generated for the backend-only KIS domestic cash-stock conversion task.

## Current Backend Inventory

Already present:

- KIS REST adapter: `src/app/execution/kis_real.py`
  - Domestic cash-stock order endpoint, balance read, order-status read, OAuth token issuance/cache, hashkey request, paper/live TR ID switching.
- Mock broker: `src/app/execution/kis_mock.py`
  - Deterministic local fills for tests and paper-like flows.
- Cost engine: `src/app/cost/trading_cost_engine.py`
  - Buy/sell fees, sell tax, slippage, spread, market-impact reserve, break-even, net return gate.
- Risk manager: `src/app/risk/manager.py`
  - Data quality, source trust, live validation ID, restricted products, portfolio, duplicate order, cost, and principal protection gates.
- Principal protection: `src/app/risk/principal_protection.py`
  - Protected floor, high-watermark, cushion, risk budget, buy lockdown, sell-only/reduce-size decisions.
- Ontology/reasoning: `src/app/graph/*`
  - Graph builders, reasoner, ontology rules, semantic trading features.
- Short-horizon/realtime research components: `src/app/realtime/*`, `src/app/features/short_horizon_features.py`, `src/app/models/*`
  - Research, benchmark, and NPU-oriented model utilities exist but are not yet a complete live-eligible artifact registry.
- Storage: `src/app/storage/*`
  - Local store/model store helpers exist, but live realtime tick/orderbook tables are not yet complete.

Added in this pass:

- KIS auth/mode/readiness helpers:
  - `src/app/execution/kis_auth.py`
  - `src/app/execution/kis_types.py`
  - `src/app/execution/kis_errors.py`
  - `src/app/execution/kis_rate_limit.py`
- Guarded live execution:
  - `src/app/execution/live_execution_coordinator.py`
  - `src/app/execution/idempotency_store.py`
  - `src/app/execution/order_status_tracker.py`
  - `src/app/execution/live_order_journal.py`
  - `src/app/trading/live_runtime_guard.py`
- Live configuration validation:
  - `src/app/config/live_config.py`
  - example JSON files under `config/*.example.json`
- Operator scripts:
  - `scripts/live_readiness_check.py`
  - `scripts/live_order_dry_run.py`
  - `scripts/arm_live_trading.py`
  - `scripts/disarm_live_trading.py`
  - `scripts/start_live_trading_loop.py`
  - `scripts/run_live_trading_test_suite.py`
- Realtime market-data foundation:
  - `src/app/data/realtime_types.py`
  - `src/app/data/kis_realtime.py`
  - `src/app/data/realtime_store.py`
  - `src/app/data/market_data_health.py`
  - `scripts/check_realtime_market_data.py`
- Live feature/model/decision foundation:
  - `src/app/features/feature_schema.py`
  - `src/app/features/feature_provenance.py`
  - `src/app/features/live_feature_frame.py`
  - `src/app/models/model_artifact_registry.py`
  - `src/app/models/live_model_trainer.py`
  - `src/app/models/live_signal_predictor.py`
  - `src/app/trading/shared_decision_engine.py`
  - `scripts/train_live_short_horizon_models.py`

## Exact Fail-Closed Points

- Real broker submission is blocked unless `LiveExecutionCoordinator.submit_final_order()` receives a `FinalOrder`.
- Live BUY evidence is blocked when realtime tick or orderbook data is missing, stale, or not sourced from `kis_realtime_websocket`.
- Live inference requires a live-eligible artifact with the exact feature schema hash and ordered feature list.
- Demo fixture training is explicitly marked not live-eligible by the CLI and cannot create the production latest pointer.
- Non-limit orders, invalid KRX symbols, zero quantity, and zero/negative price are rejected before any KIS order call.
- Live submission requires:
  - `LIVE_TRADING_ENABLED=true`
  - `KIS_LIVE_ENABLED=true`
  - `KIS_PAPER_TRADING=false`
  - `LIVE_ORDER_SUBMIT_ENABLED=true`
  - `KILL_SWITCH_ENABLED=false`
  - valid, unexpired manual arming file
  - KIS token, account read, and WebSocket approval-key health checks
- Idempotency keys prevent duplicate submissions. Reusing a key with a different order payload is blocked.
- `scripts/start_live_trading_loop.py` intentionally exits nonzero until the remaining realtime/model pipeline gates are completed.

## Remaining Blockers Before Real Live Trading

The system is not fully live-trading-ready yet. These gates remain incomplete:

- KIS WebSocket network transport and reconnect/resubscribe lifecycle. Deterministic parsing, persistence, minute bar building, and freshness/source health tables now exist.
- Production dataset extraction from historical realtime rows and realized outcomes. The fitted trainer, live-eligible registry, schema validation, and inference gate now exist.
- Tradable universe/session/tick-size validation for KRX/KOSDAQ cash stocks.
- Shared live strategy pipeline that joins realtime data, features, model output, ontology evidence, cost, principal protection, risk, and final trade gate.
- Full order-status recovery after unknown network outcomes.
- Complete live audit schema and session summary.

## Files Modified Or Created

- `config/secrets/kis_api_keys.env.example`
- `.gitignore`
- `config/live_trading_safety.example.json`
- `config/trading_costs.example.json`
- `config/order_execution.example.json`
- `config/tradable_universe.example.json`
- `config/model_training.example.json`
- `config/realtime_market_data.example.json`
- `src/app/config/__init__.py`
- `src/app/config/live_config.py`
- `src/app/execution/__init__.py`
- `src/app/execution/kis_auth.py`
- `src/app/execution/kis_errors.py`
- `src/app/execution/kis_rate_limit.py`
- `src/app/execution/kis_types.py`
- `src/app/execution/idempotency_store.py`
- `src/app/execution/live_execution_coordinator.py`
- `src/app/execution/live_order_journal.py`
- `src/app/execution/order_status_tracker.py`
- `src/app/trading/live_runtime_guard.py`
- `src/app/data/realtime_types.py`
- `src/app/data/kis_realtime.py`
- `src/app/data/realtime_store.py`
- `src/app/data/market_data_health.py`
- `scripts/check_realtime_market_data.py`
- `src/app/features/feature_schema.py`
- `src/app/features/feature_provenance.py`
- `src/app/features/live_feature_frame.py`
- `src/app/models/model_artifact_registry.py`
- `src/app/models/live_model_trainer.py`
- `src/app/models/live_signal_predictor.py`
- `src/app/models/model_validation.py`
- `src/app/trading/shared_decision_engine.py`
- `scripts/train_live_short_horizon_models.py`
- `scripts/live_readiness_check.py`
- `scripts/live_order_dry_run.py`
- `scripts/arm_live_trading.py`
- `scripts/disarm_live_trading.py`
- `scripts/start_live_trading_loop.py`
- `scripts/run_live_trading_test_suite.py`
- `tests/test_live_config_validation.py`
- `tests/test_kis_auth_and_mode.py`
- `tests/test_live_execution_coordinator.py`
- `tests/test_live_arming.py`

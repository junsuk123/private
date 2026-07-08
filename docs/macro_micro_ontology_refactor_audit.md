# Macro–Micro Ontology Refactor — Current-System Audit (Phase 0)

Branch: `feature/hierarchical_macro_micro_ontology_refactor` (cut from the
`feature/evidence_based_technical_prediction_layer` branch, so the technical
prediction layer + ontology work it depends on is present).
Date: 2026-07-09

This audit records the **actual** current structure so the refactor adapts to
reality (not the prompt's assumed paths) and preserves every safety gate.

## Spec-path → real-path deltas

| Prompt reference | Reality |
|---|---|
| `src/app/trading/realtime_engine.py` | **`src/app/trading/realtime_trading_engine.py`** (`RealtimeTradingEngine`) |
| `FinalTradeGate` (class) | **conceptual** — enforced by `RiskManager.validate` + `LiveExecutionCoordinator._validate_final_order` (requires `FinalOrder` + `OrderType.LIMIT`) |
| `src/app/technical/` "if it exists" | **exists** (indicators, regime, signals, prediction, feature_builder, labels, replay, policy, reason_codes, decision_feed) — integrate, don't recreate |
| `OntologyReasoner` | class is **`SemanticPolicyScorer`** in `graph/reasoner.py` (alias `OntologyReasoner` kept) |

## 1. Automated trading loop flow

`RealtimeTradingEngine.run_once(decision_time)` (`trading/realtime_trading_engine.py`):
1. Pulls `account = account_provider()`, `ontology_graph = ontology_graph_provider()`.
2. **SELL/REDUCE first**: iterates current holdings → `decision_engine.evaluate_exit_for_holding(holding, account, ontology_graph=…)`; approved exits go to `coordinator.submit_final_order`.
3. **BUY second**: iterates `candidate_symbols_provider()` → `decision_engine.evaluate_buy(symbol, account, ontology_graph=…)`; approved buys submitted.
4. `run_forever(stop_event)` loops at the realtime interval; `cycle_observer` records a summary consumed by the GUI.
- Wired in `web.py::_build_realtime_trading_engine`: `decision_engine=SharedLiveDecisionEngine(...)`, `candidate_symbols_provider=_realtime_buy_candidates`, `ontology_graph_provider=_latest_ontology_graph`, `coordinator=LiveExecutionCoordinator(broker_client)`.

## 2. Ontology graph creation flow

- `graph/builders.py::build_market_graph(...)` builds the in-memory `KnowledgeGraph` (string `Triple`s) from market/indicator/flow/news data.
- `web.py::_latest_ontology_graph` supplies the current graph to the engine.
- `graph/technical_evidence.py` (added in the prior phase) projects technical signals into the same `KnowledgeGraph` and into RDF.

## 3. RDF / OWL / SHACL locations

- `graph/rdf_graph.py`: `RdfTradingGraph` (rdflib Dataset, `TR`/`RES`/`EV` namespaces, `resource_iri`/`evidence_iri`).
- `graph/rdf_adapter.py`: `KnowledgeGraph` → RDF (`knowledge_graph_to_rdf`, `add_triple_to_rdf`, `attach_scoring_provenance`), RDF → UI payload (`rdf_to_ui_payload`).
- `graph/owl_reasoner.py`: RDFS/OWL-RL materialization (`load_schema_graph`, cached).
- `graph/shacl_validator.py`: `validate_graph(graph, mode)` against `ontology/trading_shapes.ttl` (live=blocking, paper=warning).
- Ontology assets: `ontology/trading_core.ttl`, `trading_rules.ttl`, `trading_shapes.ttl` (single prefix `tr: <https://example.com/ontology/trading#>`).

## 4. Python policy scoring / reasoning locations

- `graph/reasoner.py::SemanticPolicyScorer` — numeric support/contradiction/risk weighting over the `KnowledgeGraph`; emits reasoning paths + action decisions (advisory).
- `graph/theory_vote.py`, `action_aggregator.py`, `theory_registry.py` — theory-vote aggregation.
- `technical/` — advisory technical prediction layer (regime, composite signal, conservative expected exit price).

## 5. Candidate creation flow

- `web.py::_realtime_buy_candidates()` returns the BUY candidate universe (the `candidate_symbols_provider`). **This is the macro insertion point** — MacroMarketReasoner will refine/produce this universe.
- `strategy/candidate_factory.py::StrategyCandidateFactory` ranks strategy candidates (uses ProfitabilityGate).

## 6. SharedLiveDecisionEngine BUY/SELL/REDUCE order

- `trading/shared_decision_engine.py::SharedLiveDecisionEngine`. Order is enforced by the **caller** (`run_once`): `evaluate_exit_for_holding` (SELL/REDUCE) before `evaluate_buy` (BUY).
- `evaluate_buy(symbol, account, *, ontology_graph, decision_time) -> SharedDecisionResult`.
- `evaluate_exit_for_holding(holding, account, *, ontology_graph, decision_time) -> SharedDecisionResult`.
- `SharedDecisionResult(symbol, approved, final_order, prediction, reason_codes, diagnostics)`.

## 7. Authoritative gates — call sites (MUST NOT be bypassed)

- `TradingCostEngine.estimate(...)` — cost floor inside `evaluate_buy`/`evaluate_exit_for_holding` and `ProfitabilityGate`.
- `ProfitabilityGate.evaluate(ProfitabilityInput(...))` — inside `evaluate_buy`; rejection → not approved.
- `PrincipalProtectionEngine` — inside `RiskManager`.
- `RiskManager.validate(intent, account, market, ...)` — produces `.approved` + `.final_order`; authoritative.
- `LiveExecutionCoordinator.submit_final_order(FinalOrder)` — **only** broker submission path; `_validate_final_order` requires `FinalOrder` + `OrderType.LIMIT`.

## 8. GUI / API payload structure

- `account_dashboard.py::AccountDashboardService.build_dashboard()` → dict: `snapshot`, `holdings`, `cash`, `trades`, `holding_orders`, `profitability`, `technical` (added prior phase), `logs`. Providers: `status_provider`, `logs_provider`, `technical_provider`.
- `web_account_routes.py::create_account_router` → `/account` (HTML), `/api/account/{dashboard,holdings,cash,profit,trades,asset-history,logs,technical}`.
- Static: `static/account_dashboard.{js,css}`. **Macro/Micro panel insertion point**: add a `macro_micro` payload section + a collapsible panel below core panels.

## 9. Test structure

- `tests/` uses `unittest` + `pytest`. Relevant: `test_realtime_exit_decision.py`, `test_profitability_refactor_integration.py`, `test_realtime_modes.py`, `test_account_dashboard*.py`, `test_ontology_framework.py`, `test_technical_*` (prior phase), `test_shacl_*`.

## 10. Baseline test status

`python -m pytest` on this branch tip **before** refactor: **605 passed, 5 failed**
(identical to the technical-layer branch). The 5 are **pre-existing** and unrelated:
`test_profitability_gate::test_small_account_adds_extra_required_net`,
`test_profitability_refactor_integration::test_gross_positive_net_negative_buy_is_rejected`,
and 3 `test_realtime_modes::…affordable_us_discovery…`. New failures will be
tracked separately from these.

## Safety gates that must not break

1. Ontology/LLM/ML/NPU never gain final order authority.
2. Every BUY passes TradingCostEngine → ProfitabilityGate → PrincipalProtectionEngine → RiskManager → (limit-only) LiveExecutionCoordinator.
3. LiveExecutionCoordinator is the sole broker-submission path; KIS limit orders only.
4. SELL/REDUCE evaluated before BUY.
5. No margin/leverage/derivatives/short/credit/leveraged-ETF.
6. No synthetic/sample/hash-derived data in live/paper decisions.

## Compatibility plan (where the refactor plugs in)

- **Additive, backward-compatible.** The macro/micro layer is an **advisory
  producer** feeding the two existing hooks:
  - `MacroMarketReasoner` → macro-refined candidate universe → `candidate_symbols_provider` (`_realtime_buy_candidates`), plus macro evidence attached to the ontology graph.
  - `OntologyCoordinator` runs macro then dispatches `MicroSymbolReasoner` in bounded parallel for macro-selected symbols; `GlobalTradeArbiter` ranks SELL/REDUCE-first then BUY into advisory `RankedTradeIntent`s.
- `SharedLiveDecisionEngine` gains an **adapter** to consume a
  `MacroMicroReasoningBundle`/ranked intents (macro risk + micro edge as
  advisory inputs and diagnostics) while its authoritative gate calls and the
  `run_once` SELL-first/BUY-second order are **unchanged**.
- All new reasoning is advisory: no module constructs or submits a `FinalOrder`.
- CPU-only safe; NPU optional (reuse the technical layer's backend policy).

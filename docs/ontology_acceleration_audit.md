# Ontology Acceleration Audit

**Status:** Pre-implementation audit (deliverable required before any refactor).
**Scope:** Answers the audit questions for the "Ontology Reasoning Acceleration and Quantized
Evidence Scoring" task against the *actual* code on this branch, and separates spec assumptions
that are already satisfied from the real gaps worth implementing.
**Method:** Four independent read-only investigations of `src/app/graph/**`, `src/app/npu/**`,
`src/app/realtime/**`, and the live trading decision path (`trading/`, `cost/`, `risk/`,
`execution/`), cross-checked against `docs/ontology_migration_audit.md` and
`docs/npu_runtime_architecture.md`. No files were modified during the audit.

> **Headline finding:** Several of the task's stated problems do **not** exist in this codebase as
> written. The live tick loop already does **not** re-run full graph/OWL reasoning per tick; OWL is
> already restricted to OWL 2 RL (not DL); and there are **no trained numeric models to quantize**
> (every "NPU model" is a hardcoded FP32 linear layer). This audit records what is real so the
> implementation targets genuine bottlenecks instead of rebuilding things that already hold.

---

## 1. Two reasoning engines (essential context)

The `src/app/graph/` layer is **two separate subsystems** that share vocabulary but not machinery:

| | **Track A — custom triple engine** | **Track B — RDF/OWL/SHACL layer** |
|---|---|---|
| Files | `knowledge_graph.py`, `reasoner.py`, `reasoning_rules.py`, `ontology.py`, `builders.py`, `event_mapper.py`, `semantic_builder.py`, `trading_strategy_semantics.py`, `theory_vote.py`, `action_aggregator.py`, `conflict_resolver.py` | `rdf_graph.py`, `rdf_adapter.py`, `owl_reasoner.py`, `semantic_materializer.py`, `shacl_validator.py`, `ontology_layer.py`, `src/app/ontology/*.ttl` |
| Storage | `list[Triple]` of **bare Python strings** ([knowledge_graph.py:6-24](../src/app/graph/knowledge_graph.py#L6-L24)) | `rdflib.Dataset` of named graphs, real IRIs ([rdf_graph.py:81-152](../src/app/graph/rdf_graph.py#L81-L152)) |
| Reasoning | Numeric policy scoring + Datalog-style set joins in Python ([reasoner.py:226-260](../src/app/graph/reasoner.py#L226-L260)) | `owlrl` OWL 2 RL forward closure + `pyshacl` ([owl_reasoner.py:86-91](../src/app/graph/owl_reasoner.py#L86-L91)) |
| Influence on orders | **Yes** — feeds the live decision path | **No** — advisory/UI/validation only, feature-flagged `ONTOLOGY_RDF_LAYER`, wrapped in `try/except → None` ([pipeline.py:175-177](../src/app/pipeline.py#L175-L177)) |
| Profile | n/a (not logical) | OWL 2 RL / RDFS only — **not** OWL DL, no owlready2/Pellet/HermiT |

**Consequence:** the task's phrase "do not perform full OWL DL reasoning in the live tick loop" is
already true twice over — the live path uses Track A (no OWL at all), and even Track B is RL, not DL.
`SemanticPolicyScorer` accepts `inferred_classes` but **never reads them** in any scoring method
([reasoner.py:102,112](../src/app/graph/reasoner.py#L102)) — Track B is presently pure diagnostics.

---

## 2. Audit questions answered

### Q1 — Where is ontology reasoning triggered?

- **Amortized analysis/training loop, not the tick loop.** The reasoner runs once per analysis cycle
  inside `build_analysis_context` → [pipeline.py:120-123](../src/app/pipeline.py#L120-L123):
  ```python
  reasoner = OntologyReasoner(graph)   # alias of SemanticPolicyScorer
  reasoner.infer()
  reasoning_paths = reasoner.build_reasoning_paths(...)
  ```
  Called from `cli.py:29` and `web.py:4952/4964/6107`. The result (`context.graph`,
  `reasoning_paths`) is persisted into `_live_state["context"]` ([web.py:5006-5017](../src/app/web.py#L5006)).
- **Track B** entry: `_build_ontology_layer` ([pipeline.py:154-177](../src/app/pipeline.py#L154-L177)) →
  `knowledge_graph_to_rdf` → `materialize` (OWL closure) → `validate_graph` (SHACL). Fail-safe.
- Other callers: `features/hybrid_pipeline.py:65-67`, backtesting demos. `materialize()` directly only in tests.

### Q2 — Does the live tick path re-run full graph reasoning? **No.**

`RealtimeTradingEngine.run_once` (default 1000ms, [realtime_trading_engine.py:74,795-802](../src/app/trading/realtime_trading_engine.py#L74))
fetches the graph **once per cycle** from a provider that just returns the pre-computed snapshot:
```python
# web.py:4690-4695  _latest_ontology_graph
with _live_lock:
    context = _live_state.get("context")
return getattr(context, "graph", None)
```
No reasoning executes in the tick path. The graph is consumed only via cheap read-only triple
lookups (`graph.matching(...)`) inside `SharedLiveDecisionEngine`
([shared_decision_engine.py:390-391](../src/app/trading/shared_decision_engine.py#L390-L391),
`strategy/rule_based.py:323-374`).

> **The real per-tick cost is not reasoning — it is `KnowledgeGraph.matching()` doing an O(n) linear
> scan of the string-triple list** ([knowledge_graph.py:29-41](../src/app/graph/knowledge_graph.py#L29-L41))
> **once per consumed predicate per symbol.** That is the genuine acceleration target, and it is a
> lookup/indexing problem, not a reasoning problem.

### Q3 — How are facts stored? Strings, objects, dicts, or compact IDs?

- **Track A: bare Python strings** in a flat `list[Triple]`. Every lookup (`matching`, `objects`,
  `for_subject`, `reasoning_path_ids`) and even insert-dedup is a **linear O(n) scan** with string
  equality. **No indexing, no integer IDs, no timestamps, no validity windows, no confidence field.**
- **Track B: real rdflib `URIRef`/`Literal`** with deterministic IRIs.
- **Nowhere are facts stored as compact integer IDs.** This is the single largest structural gap and
  the strongest justification in the whole spec.

### Q4 — Which rules are hard-reject / risk / conflict / scoring / explanation?

Rules are **scattered across many modules**, not registered anywhere:

| Category | Where it lives today |
|---|---|
| **Hard reject** | `reasoning_rules.HARD_RISK_FEATURES` → `TradeForbidden` ([reasoning_rules.py:24-35](../src/app/graph/reasoning_rules.py#L24-L35)); `reasoner._infer_buy_candidates` MissingMarketData → `increasesRiskOf TradeForbidden` ([reasoner.py:251-253](../src/app/graph/reasoner.py#L251-L253)); SHACL live shapes (stale/synthetic) blocking only in `mode="live"` ([shacl_validator.py:135](../src/app/graph/shacl_validator.py#L135)) |
| **Risk / sizing** | `reasoner._infer_risk_adjustments` → `RiskAdjustedSizing` ([reasoner.py:255-260](../src/app/graph/reasoner.py#L255-L260)); `reasoning_rules` ReduceRiskCandidate |
| **Conflict** | `ConflictResolver` 5 types w/ fixed penalties ([conflict_resolver.py:62-77](../src/app/graph/conflict_resolver.py#L62-L77)) |
| **Scoring / candidate** | `SUPPORT/CONTRADICTION/RISK_WEIGHTS` tables ([reasoner.py:14-60](../src/app/graph/reasoner.py#L14-L60)); buy-candidate joins ([reasoner.py:226-250](../src/app/graph/reasoner.py#L226-L250)) |
| **Explanation-only** | Track B OWL classification rules ([trading_rules.ttl:50-105](../src/app/ontology/trading_rules.ttl)); `build_reasoning_paths` NL explanations ([reasoner.py:119-161](../src/app/graph/reasoner.py#L119-L161)) |

**There is no rule registry, no priority ordering, and no short-circuit.** Hard-reject and scoring
logic are interleaved in the same `_infer_*` methods.

### Q5 — Which ontology outputs are consumed by the gates?

Ontology output is consumed **only inside `SharedLiveDecisionEngine`**, then reduced to scalars.
None of the gates receive a graph or a structured ontology object.

| Consumer | What it actually gets | Evidence |
|---|---|---|
| **ProfitabilityGate** | Nothing directly. Only the derived `expected_exit_price`/`gross_expected_return`. | [shared_decision_engine.py:653-670](../src/app/trading/shared_decision_engine.py#L653-L670); gate has no ontology input ([profitability_gate.py:220-362](../src/app/cost/profitability_gate.py#L220-L362)) |
| **RiskManager** | `intent.supporting_factors` (support-tag strings); checks `intent.ontology_tags` for `"TradeForbidden"` → `ONTOLOGY_TRADE_FORBIDDEN` **but the live buy path never populates `ontology_tags`** | [manager.py:88,544](../src/app/risk/manager.py#L88); [shared_decision_engine.py:709-712](../src/app/trading/shared_decision_engine.py#L709-L712) |
| **DynamicExitPolicy** | Nothing directly (scalar market inputs only). Ontology affects exits via `ontology_score` thresholds applied *in* `evaluate_exit_for_holding`. `LossExitEvidence.ontology_score` exists but the live path uses its own branch logic instead of `loss_exit_decision`. | [dynamic_exit_policy.py:111-120,204](../src/app/trading/dynamic_exit_policy.py#L111-L120); [shared_decision_engine.py:1075-1119](../src/app/trading/shared_decision_engine.py#L1075-L1119) |
| **ExecutionPolicy** | Nothing — plain threshold dataclass. | [execution_policy.py:7-26](../src/app/trading/execution_policy.py#L7-L26) |

The scalars that survive: `ontology_score` (float), `ontology_ok` (bool), `ontology_support`
(tuple of strings). **Reasoning-path IDs and evidence IDs are computed and persisted for the GUI but
never attached to a `FinalOrder` or rejection.**

### Q6 — Symbolic vs numeric split

- **Symbolic:** Track A rule bodies are Datalog-style set joins, e.g.
  `{"EarningsGrowth","ProfitabilityQuality"}.issubset(support_objects)`
  ([reasoner.py:232](../src/app/graph/reasoner.py#L232)); `reasoning_rules._select_signal` membership
  tests; all of Track B (OWL RL `owl:hasValue` classification + SHACL).
- **Numeric:** all weights/thresholds/confidence math in Python
  ([reasoner.py:14-60,127-143](../src/app/graph/reasoner.py#L14-L60)); `TheoryVote.effective_weight`
  six-float product ([theory_vote.py:37-46](../src/app/graph/theory_vote.py#L37-L46)); conflict
  penalty attenuation; `ActionAggregator` cluster compression `raw/sqrt(n)`
  ([action_aggregator.py:64-98](../src/app/graph/action_aggregator.py#L64-L98)).

### Q7 — Which NPU/OpenVINO components are real, and which fall back silently?

**Only 1 of 7 scorers is on the live path. Every "model" is a hardcoded FP32 linear layer — there
are no trained artifacts and no quantization anywhere.**

| Scorer | On live path? | Reality |
|---|---|---|
| `OntologyNpuLinearScorer` ([npu_classifier.py:67](../src/app/graph/npu_classifier.py#L67)) | **Yes** ([builders.py:53-57](../src/app/graph/builders.py#L53-L57), [trading_pipeline.py:828-829](../src/app/trading_pipeline.py#L828)) | Real OpenVINO matmul, no min-batch gate; uses NPU if present, else `_NumpyLinearModel`. FP32 hardcoded weights. |
| `NpuTheoryVoteScorer`, `NpuConflictScorer`, `NpuEvidenceClusterCompressor`, `NpuExecutionEdgeScorer`, `NpuShortHorizonPredictor` | **No** | Instantiated only in tests/benchmarks. Route to numpy in practice: `min_batch_for_npu=128` ([runtime_manager.py:79](../src/app/npu/runtime_manager.py#L79)) but live batches are tiny. |
| `ShortHorizonNpuPredictor` ([short_horizon_npu_predictor.py:52](../src/app/realtime/short_horizon_npu_predictor.py#L52)) | **No** | Its `.xml` artifact `models/short_horizon/openvino_model.xml` **does not exist** → **always** `_linear_baseline`, `provider="linear_baseline"`. |

- **No INT8/FP16 anywhere** — every OpenVINO param is `ov.Type.f32`. Grep for
  `INT8|FP16|quantiz|NNCF` → zero hits in NPU source.
- **No drift/calibration validation** — only tensor-shape checks in `tensor_schemas.py`.
- **Provider reporting exists but is fragmented** across three dataclasses (`NpuModuleStatus`,
  `OntologyNpuStatus`, `ShortHorizonPrediction.provider`) with ad-hoc string values — **no single
  canonical enum** like `OPENVINO_NPU / OPENVINO_CPU / CPU_NUMPY / LINEAR_BASELINE`. Only the
  ontology status is surfaced in web endpoints.

### Q8 — Are full explanation paths stored per call / causing overhead?

- Track A `build_reasoning_paths` builds one `ReasoningPath` per ticker every analysis cycle
  (formatted triple strings + NL explanation + a SHA-256 `path_id`), via multiple full
  `graph.matching` scans per ticker → ~O(tickers × triples)
  ([reasoner.py:119-161,263-283](../src/app/graph/reasoner.py#L119-L161)).
- Track B builds a full UI node/edge payload with multiple full-graph passes per cycle
  ([rdf_adapter.py:318-386](../src/app/graph/rdf_adapter.py#L318-L386)) — timed (`build_ms`,
  `reason_ms`, `validate_ms`), the expected expensive path.
- **This overhead is in the amortized analysis loop, not the tick loop.** It is real but off the
  latency-critical path. There is no evidence explanation building is a per-tick cost.

---

## 3. Spec assumptions vs reality

| Spec premise / acceptance criterion | Reality on this branch | Verdict |
|---|---|---|
| "Live tick path re-runs full graph reasoning" | Tick loop reads a cached snapshot; reasoning is amortized in the analysis loop | **Already satisfied** |
| "Do not perform full OWL DL in the tick loop" | No OWL in tick path at all; Track B is OWL 2 RL, advisory-only | **Already satisfied** |
| "NPU scorers must support INT8/FP16 with CPU fallback + drift validation" | No trained models exist; all scorers are hardcoded FP32 linear layers; 6 of 7 unused | **Misguided as written** — quantizing a hand-coded linear layer to INT8 has no accuracy/latency payoff; there is nothing to calibrate against |
| "Facts stored as strings → replace with integer FactTable" | Exactly true; `list[Triple]` of strings, O(n) scans, no index | **Real, high-value gap** |
| "No compact OntologyDecisionContext; ontology output is loose" | True; degrades to `float/bool/tuple[str]`; `ontology_tags` hook exists but is never populated | **Real, high-value gap** |
| "Hard reject rules must run first with short-circuit" | Hard-reject logic exists but is interleaved with scoring; no registry/priority | **Real, moderate-value gap** |
| "Materialize stable facts offline; delta reasoning for live" | No inference cache; analysis loop recomputes closure each cycle | **Real gap for the analysis loop** (not the tick loop) |
| "Native Rust/C++ rule kernel" | No profiling evidence any kernel is Python-loop-bound; live cost is a list scan | **Premature** — solve with an index first |

---

## 4. Old flow vs proposed new flow

**Today**
```
Analysis loop (seconds):  build graph → OntologyReasoner.infer() → build_reasoning_paths
                          → [optional Track B: RDF→OWL RL→SHACL, advisory] → persist context.graph
Tick loop (1s):           read cached context.graph → per symbol: N× graph.matching() O(n) string scans
                          → scalar ontology_score/ok/support → gate cascade (profitability→risk→exec→live-guard)
```

**Proposed (grounded in the real bottleneck)**
```
Pre-market/offline:       validate rules → materialize stable taxonomy facts → build FactDictionary +
                          indexed integer FactTable snapshot → cache (schema/rule/universe hash)
Analysis loop:            delta-update FactTable for changed facts only → refresh cached scores
Tick loop (1s):           indexed FactTable lookups (O(1) by subject/predicate) → hard-reject short-circuit
                          → build typed OntologyDecisionContext (score, hard_reject_flags, risk_flags,
                          top-k evidence ids, explanation_hash, latency) → pass to gates unchanged
Post-trade:               store full explanation graph async keyed by explanation_hash
```

---

## 5. Recommended scope (honest prioritization)

Ordered by value-to-effort, given the findings above:

1. **Typed `OntologyDecisionContext` + populate `ontology_tags`/`TradeForbidden`** (Phases 4-slice, 5, 9).
   Highest value, lowest risk. Closes a real safety gap: `RiskManager` already checks
   `ontology_tags` for `TradeForbidden` but the live buy path never sets it. Gives every FinalOrder/
   rejection a stable ontology context id + explanation hash. **No behavior change beyond activating a
   dormant safety hook.**
2. **Indexed FactTable + FactDictionary with a compatibility adapter** (Phase 2). Replaces the O(n)
   `matching()` scans that are the true per-tick cost. Keep `KnowledgeGraph` API working via adapter.
3. **Rule registry + hard-reject short-circuit** (Phases 1, 5). Extract scattered rules into a
   registry with priority; run hard rejects first. Correctness-preserving refactor.
4. **Offline materialization cache** (Phase 3) for the analysis loop's OWL closure recompute.
5. **Unify provider-state reporting into one enum** (part of Phase 6/10) and surface it in GUI. Cheap,
   honest, useful.
6. **Benchmarks** (Phase 11) — measure `matching()` vs indexed lookup, closure recompute vs cache.

**Recommend against / defer:** INT8/FP16 quantization of the linear scorers (Phase 6 core) and the
native Rust/C++ kernel (Phase 8) — no trained models to quantize and no profiling evidence of a
Python-loop bottleneck. Revisit only if (a) real trained scorer artifacts are introduced, or (b)
benchmarks after step 2 still show a rule-matching hotspot.

**Non-negotiable safety invariants (unchanged):** ontology may **block** a trade (hard reject) but a
positive ontology score can **never** bypass `ProfitabilityGate`, `RiskManager`, the kill switch, the
arming file ([live_runtime_guard.py:10,52,80-95](../src/app/trading/live_runtime_guard.py#L10)), or
principal protection. On any ontology error, **fail closed for BUY**.

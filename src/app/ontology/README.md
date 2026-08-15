# Trading Ontology (`src/app/ontology/`)

Standards-based RDF/RDFS/OWL ontology and SHACL shapes for OBAITS, the Ontology
Based AI Trading System. This layer adds **semantic
representation, logical inference, consistency checking, and explainability** on top of the
existing trading pipeline. It does **not** replace numerical scoring or trading decisions.

## Files

| File | Role |
|---|---|
| `trading_core.ttl` | Core RDFS/OWL vocabulary: classes, class hierarchy, object/data properties, `rdfs:domain`/`rdfs:range`, property hierarchy (`rdfs:subPropertyOf`), and disjointness axioms. |
| `trading_rules.ttl` | OWL 2 RL classification axioms (`owl:hasValue` restriction classes) that let the reasoner infer semantic class memberships from asserted facts. Imports `trading_core.ttl`. |
| `trading_shapes.ttl` | SHACL shapes for closed-world operational validation (required fields, positive prices, stale/synthetic blocking, account/order structure, approved-vs-rejected conflict, final-order preconditions). |
| `macro_market_ontology.ttl`, `micro_symbol_ontology.ttl` | Macro/micro reasoning vocabularies for the hierarchical reasoners in `app.graph`. |
| `operational_gate.py` | `ClosedWorldOntologyGate` — point-in-time fact snapshot → allowed strategy ids + hard-block reasons + soft compatibility. |
| `short_rules.py` | Short-side closed-world facts and the per-regime directional allow mask. |
| `strategy_ontology.py` | Strategy relations keyed on **real strategy ids**: 8 hard relation types, 4 soft ones. |
| `strategy_eligibility.py` | `StrategyEligibilityEngine` — the boolean mask `M_s(x)` and the soft score `O_s(x)` consumed by `StrategySelectorV2`. |
| `trading_domain_ontology.py`, `trading_reasoner.py`, `trading_rules.py`, `trading_fact_builder.py` | Deterministic decision ontology: vocabulary, reasoner, YAML-backed rules, fact builder. |
| `README.md` | This file. |

## Namespaces

| Prefix | IRI | Use |
|---|---|---|
| `tr:` | `https://example.com/ontology/trading#` | Schema terms (classes, properties). |
| `res:` | `https://example.com/resource/` | Runtime instances (stocks, snapshots, candidates, decisions). |
| `ev:` | `https://example.com/evidence/` | Provenance-bearing `tr:EvidenceItem` individuals. |

Instance IRIs are **stable and deterministic** (derived by slugging tickers/ids), not random
blank nodes, so the same entity gets the same IRI across analysis cycles.

## Reasoning boundary (why hybrid)

![Reasoning boundary: OWL, SHACL, Python, RiskManager](../../../docs/diagrams/ontology_reasoning_boundary.svg)

```
OWL / RDFS  -> class & property hierarchy, domain/range typing, semantic
               categorization (BuyCandidate, TradeForbiddenAsset, SyntheticDataAsset...),
               consistency (disjoint classes).
SHACL       -> closed-world data-quality & live-readiness validation.
Python       -> support/contradiction/risk/confidence scoring, ranking, thresholding,
               short-horizon policy (SemanticPolicyScorer).
Engines      -> TradingCostEngine, PrincipalProtectionEngine, position sizing.
RiskManager  -> the SOLE final execution gate.
```

- **OWL is open-world**: a missing fact is *unknown*, not *false*. Therefore OWL never blocks a
  trade — absence of a risk assertion does not mean the asset is safe.
- **SHACL is closed-world**: it is used exactly where "missing = invalid" is the correct semantics
  (required fields, stale/synthetic data for live orders).
- **Python owns all numbers**: no score, weight, ranking, cost, tax, slippage, or
  principal-protection amount is encoded as an OWL axiom.
- **OWL/SHACL never grant permission to trade.** An inferred `tr:TradeEligibleAsset` or
  `tr:BuyCandidate` is a *semantic label*; the RiskManager still decides.

## Strategy eligibility (`strategy_eligibility.py`)

The ontology's ONLY output in the V2 selection pipeline. It does not pick a strategy, does not
rank, and cannot authorise anything — those were the overlapping responsibilities the
strategy-selection refactor separated.

Relations are keyed on **concrete `strategy_id`s**. Before the refactor the hard mask was keyed on
a generic METHODOLOGY name (`momentum` / `breakout` / `mean_reversion` / `vwap_reversion`) and an
alias table translated it into an executable id. That table's own comment records the problem —
`mean_reversion → vwap_mean_reversion` is "the loosest fit", because the generic thesis reverts to
a Bollinger midline while the catalogued strategy reverts to VWAP. An ontology verdict about one
hypothesis was authorising a different one, and a catalogue of 19 theses was addressable through 4
names. The methodology enum survives only as coarse macro permission tokens
(`MACRO_FAMILY_BY_STRATEGY`), which is the one job it can do correctly.

```
HARD (may zero the mask)                SOFT (evidence only, never blocks)
  requires              a MarketContext field    worksWellUnder
  requiresFeature       a TechnicalFeatureSet    prefers
  requiresLiquidity     floor / spread ceiling   supportedBy
  requiresSession       session phases           historicallyCompatibleWith
  requiresHistory       completed bars
  requiresDataQuality   tick window / book
  allowedMarket
  forbiddenUnder        state where invalid
```

Two independent numbers come out per strategy: `eligible` (the hard verdict, `0.0`/`1.0` as
`mask`) and `compatibility_score` (soft evidence in `[-1, 1]`, the weighted mean of the relations
that fired). A strategy with no matching soft relation scores `0.0` — **neutral, not penalised**,
because absence of evidence is not evidence.

Why the split is load-bearing: **a preference expressed as a block loses candidates that should
merely have ranked lower.** `app.technical.signals` hard-disables mean reversion in a downtrend;
here the same fact is a penalty, because the reversion thesis is *sometimes* right in a falling
tape and vetoing it removes the evidence needed to find out when.

Fail-closed with one deliberate exception: a missing *requirement* blocks, but a missing *market
state label* does not. `forbiddenUnder` and the no-entry set fire only on a label that is actually
present — an unresolved regime is an unanswered question, and the existing code already treats an
unanswerable permission check as "not a withdrawal"
(`strategy_algorithms.macro_strategy_permitted`). The data-quality relations are what catch a
context too empty to trust, and they block explicitly.

The soft relations are derived from each thesis and from gating that already exists in the code.
**None is fitted to realized performance** — scoring relations from past results would make the
ontology a backtest. See [docs/strategy_selection_v2.md](../../../docs/strategy_selection_v2.md)
and [docs/ontology_and_gnn.md](../../../docs/ontology_and_gnn.md) L5-E.

## How classification works (OWL RL)

`trading_rules.ttl` declares `owl:hasValue` restriction classes as subclasses of target
categories. Under OWL RL rule `cls-hv2`, an asserted `x p v` entails `x a <restriction>`, and via
`rdfs:subClassOf` (`cax-sco`) `x a <target>`. Example:

```turtle
[ a owl:Restriction ; owl:onProperty tr:increasesRiskOf ; owl:hasValue tr:Risk_TradeForbidden ]
    rdfs:subClassOf tr:TradeForbiddenAsset .
```

So asserting `res:SYMBOL tr:increasesRiskOf tr:Risk_TradeForbidden` makes `res:SYMBOL` a
`tr:TradeForbiddenAsset` after materialization. Class hierarchy (`tr:DomesticStock ⊑ tr:Stock ⊑
tr:MarketEntity`) and property hierarchy (`tr:supportsSignal ⊑ tr:hasSemanticEvidence`) are
inferred by standard RDFS/OWL RL rules.

## Provenance model

Each source- or model-derived fact is linked to an `ev:{evidence_id}` `tr:EvidenceItem` via
`tr:hasEvidence` / `tr:derivedFromEvidence`. Evidence items carry source name, source type,
timestamp, data-quality score, synthetic flag, stale flag, confidence, and analysis-cycle id.
Per-cycle assertions live in a named graph inside an `rdflib.Dataset`. No RDF reification or
RDF-star is used (keeps the model OWL RL-friendly and maintainable).

## Inspecting / serializing the ontology

```bash
# Parse-check every file
python -c "import rdflib; [rdflib.Graph().parse(f, format='turtle') for f in \
  ['src/app/ontology/trading_core.ttl','src/app/ontology/trading_rules.ttl','src/app/ontology/trading_shapes.ttl']]; print('ok')"
```

At runtime, `app.graph.rdf_graph.RdfTradingGraph.serialize(format="turtle" | "json-ld")` dumps the
current assertion graph, and `app.graph.semantic_materializer` returns the inferred-triple set
separately from the asserted set.

## Extending the ontology safely

1. Add new classes/properties to `trading_core.ttl` with `rdfs:subClassOf` / `rdfs:subPropertyOf`
   and `rdfs:domain`/`rdfs:range`. Prefer reusing existing super-properties
   (e.g. new evidence relations under `tr:hasSemanticEvidence`).
2. Add semantic categorization as `owl:hasValue` restriction subclass axioms in
   `trading_rules.ttl` (stay OWL 2 RL — avoid cardinality and complex DL).
3. Add closed-world/data-quality checks as SHACL shapes in `trading_shapes.ttl`, **not** as OWL.
4. Never encode numerical thresholds, scores, or trade permissions in OWL — those belong in the
   Python policy scorer and the deterministic engines.
5. Map any new emitted predicate/object string in `app.graph.rdf_adapter` and add a test in
   `tests/test_ontology_*.py`.
6. When adding a **strategy relation** to `strategy_ontology.py`, decide hard vs soft first and
   name a concrete `strategy_id` — never a methodology name. A relation that expresses a
   *preference* must be soft: a hard relation removes the strategy from the ranking entirely, and
   that is only correct when the thesis is genuinely undefined in that state. Record the rationale
   on the relation (`SoftRelation.rationale`); `tests/test_ontology_strategy_eligibility.py`
   asserts that only catalogued ids are addressable and that soft relations cannot block.

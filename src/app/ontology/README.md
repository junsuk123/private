# Trading Ontology (`src/app/ontology/`)

Standards-based RDF/RDFS/OWL ontology and SHACL shapes for the Personal Multi-Agent
Ontology-Based Automated Stock Investment System. This layer adds **semantic
representation, logical inference, consistency checking, and explainability** on top of the
existing trading pipeline. It does **not** replace numerical scoring or trading decisions.

## Files

| File | Role |
|---|---|
| `trading_core.ttl` | Core RDFS/OWL vocabulary: classes, class hierarchy, object/data properties, `rdfs:domain`/`rdfs:range`, property hierarchy (`rdfs:subPropertyOf`), and disjointness axioms. |
| `trading_rules.ttl` | OWL 2 RL classification axioms (`owl:hasValue` restriction classes) that let the reasoner infer semantic class memberships from asserted facts. Imports `trading_core.ttl`. |
| `trading_shapes.ttl` | SHACL shapes for closed-world operational validation (required fields, positive prices, stale/synthetic blocking, account/order structure, approved-vs-rejected conflict, final-order preconditions). |
| `README.md` | This file. |

## Namespaces

| Prefix | IRI | Use |
|---|---|---|
| `tr:` | `https://junsuk123.github.io/private/ontology/trading#` | Schema terms (classes, properties). |
| `res:` | `https://junsuk123.github.io/private/resource/` | Runtime instances (stocks, snapshots, candidates, decisions). |
| `ev:` | `https://junsuk123.github.io/private/evidence/` | Provenance-bearing `tr:EvidenceItem` individuals. |

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

## How classification works (OWL RL)

`trading_rules.ttl` declares `owl:hasValue` restriction classes as subclasses of target
categories. Under OWL RL rule `cls-hv2`, an asserted `x p v` entails `x a <restriction>`, and via
`rdfs:subClassOf` (`cax-sco`) `x a <target>`. Example:

```turtle
[ a owl:Restriction ; owl:onProperty tr:increasesRiskOf ; owl:hasValue tr:Risk_TradeForbidden ]
    rdfs:subClassOf tr:TradeForbiddenAsset .
```

So asserting `res:005930 tr:increasesRiskOf tr:Risk_TradeForbidden` makes `res:005930` a
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

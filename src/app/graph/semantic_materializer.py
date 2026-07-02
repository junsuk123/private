"""Semantic materialization service (T04 / T14).

Merges the cached ontology schema with a per-cycle RDF assertion graph, runs
scoped RDFS/OWL RL closure, and returns the enriched graph together with the
**separately identified inferred triples**, timing information, and any
diagnostic errors.

Separation of concerns:
* This service = OWL/RDFS logical entailment (semantic categorization).
* ``SemanticPolicyScorer`` (formerly ``OntologyReasoner``) = numerical scoring.
* ``RiskManager`` = final execution authority.

Fail-safe: any reasoning error is captured in ``MaterializationResult.error``
and the enriched graph falls back to the (schema + assertions) union without
inferences. Live trading must never proceed on a failed/incomplete inference;
callers treat a non-empty ``error`` as a hard signal in live mode.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rdflib import BNode, Graph
from rdflib.namespace import RDF

from app.graph.owl_reasoner import apply_closure, load_schema_graph, profile_from_env
from app.graph.rdf_graph import RdfTradingGraph, bind_namespaces


@dataclass(frozen=True)
class MaterializationResult:
    """Outcome of one materialization pass."""

    enriched_graph: Graph
    inferred_triples: tuple[tuple, ...]
    asserted_count: int
    inferred_count: int
    schema_count: int
    profile: str
    build_ms: float = 0.0
    reason_ms: float = 0.0
    error: str | None = None
    inferred_types: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


def _inferred_type_index(inferred: tuple[tuple, ...]) -> dict[str, list[str]]:
    """Map subject IRI -> list of inferred class local names (skip blank nodes)."""
    index: dict[str, list[str]] = {}
    for s, p, o in inferred:
        if p != RDF.type or isinstance(o, BNode) or isinstance(s, BNode):
            continue
        local = str(o).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        index.setdefault(str(s), []).append(local)
    return index


def materialize(
    rdf: RdfTradingGraph,
    *,
    profile: str | None = None,
    include_rules: bool = True,
) -> MaterializationResult:
    """Run scoped OWL/RDFS materialization over one cycle's assertion graph."""
    chosen_profile = profile or profile_from_env()

    t0 = time.perf_counter()
    assertions = rdf.merged_graph()
    asserted_count = len(assertions)

    schema_result = load_schema_graph(include_rules=include_rules)
    enriched = Graph()
    bind_namespaces(enriched)
    for triple in schema_result.graph:
        enriched.add(triple)
    for triple in assertions:
        enriched.add(triple)
    build_ms = (time.perf_counter() - t0) * 1000.0

    # Snapshot the pre-closure state so we can diff out the inferred triples.
    pre_closure = set(enriched)

    error = schema_result.error
    t1 = time.perf_counter()
    if error is None:
        error = apply_closure(enriched, profile=chosen_profile)
    reason_ms = (time.perf_counter() - t1) * 1000.0

    inferred = tuple(t for t in enriched if t not in pre_closure)

    return MaterializationResult(
        enriched_graph=enriched,
        inferred_triples=inferred,
        asserted_count=asserted_count,
        inferred_count=len(inferred),
        schema_count=schema_result.triple_count,
        profile=chosen_profile,
        build_ms=round(build_ms, 3),
        reason_ms=round(reason_ms, 3),
        error=error,
        inferred_types=_inferred_type_index(inferred),
    )


def inferred_classes_for(result: MaterializationResult, subject_iri: str) -> list[str]:
    """Convenience: inferred class local names for a subject IRI."""
    return result.inferred_types.get(subject_iri, [])

"""RDF-backed trading graph store (T03).

A thin, dependency-light wrapper over ``rdflib`` that provides:

* stable, deterministic IRI generation for domain entities (no random blank
  nodes for core entities);
* namespace helpers bound to the trading ontology;
* named-graph support (per analysis cycle / per evidence source) via an
  ``rdflib.Dataset``, plus a ``merged_graph`` view that OWL RL materialization
  and SHACL validation can consume as a single ``rdflib.Graph``;
* a small, ``KnowledgeGraph``-flavoured convenience API
  (``add``, ``triples``, ``matching``, ``objects``, ``subjects``,
  ``predicates``, ``serialize``).

This module does not import anything from the trading pipeline, so it is safe
to use from tests and adapters without circular-import risk.
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator

from rdflib import Dataset, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

# ---------------------------------------------------------------------------
# Namespaces (kept in sync with src/app/ontology/*.ttl)
# ---------------------------------------------------------------------------
TRADING_NS = "https://example.com/ontology/trading#"
RESOURCE_NS = "https://example.com/resource/"
EVIDENCE_NS = "https://example.com/evidence/"

TR = Namespace(TRADING_NS)
RES = Namespace(RESOURCE_NS)
EV = Namespace(EVIDENCE_NS)

_SLUG_RE = re.compile(r"[^0-9A-Za-z]+")


def slug(value: object) -> str:
    """Deterministically slug an arbitrary label into an IRI-safe local name."""
    cleaned = _SLUG_RE.sub("_", str(value).strip()).strip("_")
    return cleaned or "node"


def resource_iri(name: object) -> URIRef:
    """Stable instance IRI in the resource namespace."""
    return RES[slug(name)]


def evidence_iri(evidence_id: object) -> URIRef:
    """Stable IRI for an EvidenceItem in the evidence namespace."""
    return EV[slug(evidence_id)]


def term_iri(local_name: str) -> URIRef:
    """Schema-term IRI in the trading namespace."""
    return TR[local_name]


def bind_namespaces(graph: Graph) -> None:
    graph.bind("tr", TR)
    graph.bind("res", RES)
    graph.bind("ev", EV)
    graph.bind("owl", OWL)
    graph.bind("rdfs", RDFS)
    graph.bind("xsd", XSD)


class RdfTradingGraph:
    """A named-graph capable RDF assertion store for one analysis scope.

    Assertions are written into a per-cycle named graph inside an
    ``rdflib.Dataset`` so provenance/cycle scoping is preserved. Reasoning and
    validation consume :meth:`merged_graph`, a plain ``rdflib.Graph`` union of
    all contexts.
    """

    def __init__(self, cycle_id: str | None = None) -> None:
        self._dataset = Dataset(default_union=True)
        bind_namespaces(self._dataset)
        self.cycle_id = str(cycle_id) if cycle_id else "cycle"
        # Default context for this cycle's assertions.
        self._context = self._dataset.graph(RES[f"graph_{slug(self.cycle_id)}"])

    # -- context management -------------------------------------------------
    def context(self, name: str | None = None) -> Graph:
        """Return (creating if needed) a named graph for the given source/cycle."""
        if name is None:
            return self._context
        return self._dataset.graph(RES[f"graph_{slug(name)}"])

    # -- write API ----------------------------------------------------------
    def add(
        self,
        subject: URIRef,
        predicate: URIRef,
        object_: URIRef | Literal,
        *,
        graph_name: str | None = None,
    ) -> None:
        self.context(graph_name).add((subject, predicate, object_))

    def add_type(self, subject: URIRef, class_local: str, *, graph_name: str | None = None) -> None:
        self.add(subject, RDF.type, TR[class_local], graph_name=graph_name)

    # -- read API (KnowledgeGraph-flavoured) --------------------------------
    def merged_graph(self) -> Graph:
        """A single ``rdflib.Graph`` union of all contexts (for OWL/SHACL)."""
        merged = Graph()
        bind_namespaces(merged)
        for triple in self._dataset.triples((None, None, None)):
            merged.add(triple)
        return merged

    def triples(self) -> tuple[tuple, ...]:
        return tuple(self._dataset.triples((None, None, None)))

    def matching(
        self,
        subject: URIRef | None = None,
        predicate: URIRef | None = None,
        object_: URIRef | Literal | None = None,
    ) -> tuple[tuple, ...]:
        return tuple(self._dataset.triples((subject, predicate, object_)))

    def objects(self, subject: URIRef, predicate: URIRef) -> tuple:
        return tuple(o for _, _, o in self._dataset.triples((subject, predicate, None)))

    def subjects(self, predicate: URIRef, object_: URIRef | Literal) -> tuple:
        return tuple(s for s, _, _ in self._dataset.triples((None, predicate, object_)))

    def predicates(self, subject: URIRef, object_: URIRef | Literal) -> tuple:
        return tuple(p for _, p, _ in self._dataset.triples((subject, None, object_)))

    def __len__(self) -> int:
        return sum(1 for _ in self._dataset.triples((None, None, None)))

    def __iter__(self) -> Iterator[tuple]:
        return iter(self.triples())

    # -- serialization ------------------------------------------------------
    def serialize(self, format: str = "turtle") -> str:
        """Serialize the merged assertion graph (Turtle or JSON-LD)."""
        return self.merged_graph().serialize(format=format)

    def extend(self, triples: Iterable[tuple], *, graph_name: str | None = None) -> None:
        ctx = self.context(graph_name)
        for triple in triples:
            ctx.add(triple)

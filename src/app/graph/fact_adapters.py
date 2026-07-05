from __future__ import annotations

"""Bridge between the string-based ``KnowledgeGraph`` and the integer ``FactTable``.

This is the compatibility layer: existing code keeps producing/consuming string
``Triple``s via ``KnowledgeGraph``, while acceleration-aware code (delta
reasoning, hard-reject short-circuit) can operate on the integer ``FactTable``
and convert back to human-readable form for explanations / GUI.

See docs/ontology_acceleration_audit.md (Phase 2).
"""

from collections.abc import Iterable

from .fact_dictionary import FactDictionary
from .fact_table import FLAG_EXPLICIT, NO_TIME, FactRow, FactTable
from .knowledge_graph import KnowledgeGraph, Triple


def fact_table_from_graph(
    graph: KnowledgeGraph,
    dictionary: FactDictionary | None = None,
    *,
    confidence: float = 1.0,
    source_quality: float = 1.0,
    flags: int = FLAG_EXPLICIT,
    valid_from: int = 0,
    valid_until: int = NO_TIME,
) -> FactTable:
    """Build an integer ``FactTable`` from every triple in ``graph``.

    Asserted string triples carry no numeric confidence, so they default to
    fully-confident, trusted-quality explicit facts. Pass a shared ``dictionary``
    (e.g. from ``ontology.build_base_dictionary()``) to keep ids stable across
    graphs.
    """
    table = FactTable(dictionary)
    for triple in graph.triples():
        table.add_fact(
            triple.subject,
            triple.predicate,
            triple.object,
            confidence=confidence,
            source_quality=source_quality,
            flags=flags,
            valid_from=valid_from,
            valid_until=valid_until,
            evidence=triple.evidence_id,
        )
    return table


def rows_to_triples(table: FactTable, rows: Iterable[FactRow]) -> tuple[Triple, ...]:
    """Decode integer fact rows back into string ``Triple``s (order preserved)."""
    dictionary = table.dictionary
    return tuple(
        Triple(
            dictionary.term(row.subject_id),
            dictionary.predicate(row.predicate_id),
            dictionary.term(row.object_id),
            dictionary.evidence(row.evidence_id),
        )
        for row in rows
    )


def graph_from_fact_table(table: FactTable, as_of: int | None = None) -> KnowledgeGraph:
    """Project a ``FactTable`` back into a string ``KnowledgeGraph``.

    Round-trips ``fact_table_from_graph`` (modulo exact triple ordering, which is
    preserved for insertion order). ``as_of`` restricts to facts active at a
    timestamp.
    """
    graph = KnowledgeGraph()
    for row in table.query(as_of=as_of):
        graph.add(
            table.dictionary.term(row.subject_id),
            table.dictionary.predicate(row.predicate_id),
            table.dictionary.term(row.object_id),
            table.dictionary.evidence(row.evidence_id),
        )
    return graph

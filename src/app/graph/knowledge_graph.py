from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str
    evidence_id: str | None = None


class KnowledgeGraph:
    """In-memory triple store.

    Public behaviour is unchanged from the original list-backed implementation:
    ``triples()`` preserves insertion order, ``add`` de-duplicates exact triples,
    and every accessor returns ``tuple[Triple, ...]`` in insertion order.

    Internally it now maintains lightweight dictionary indices so ``matching`` /
    ``for_subject`` / ``objects`` / ``reasoning_path_ids`` are O(matches) instead
    of O(total triples). The live tick loop calls ``matching`` repeatedly against a
    cached snapshot graph, so this removes the per-tick linear-scan cost without
    changing any result. See docs/ontology_and_gnn.md.
    """

    def __init__(self) -> None:
        self._triples: list[Triple] = []
        self._seen: set[Triple] = set()
        self._by_subject: dict[str, list[int]] = {}
        self._by_predicate: dict[str, list[int]] = {}
        self._by_object: dict[str, list[int]] = {}
        self._by_subject_predicate: dict[tuple[str, str], list[int]] = {}

    def add(self, subject: str, predicate: str, object_: str, evidence_id: str | None = None) -> None:
        triple = Triple(subject, predicate, object_, evidence_id)
        if triple in self._seen:
            return
        index = len(self._triples)
        self._triples.append(triple)
        self._seen.add(triple)
        self._by_subject.setdefault(subject, []).append(index)
        self._by_predicate.setdefault(predicate, []).append(index)
        self._by_object.setdefault(object_, []).append(index)
        self._by_subject_predicate.setdefault((subject, predicate), []).append(index)

    def triples(self) -> tuple[Triple, ...]:
        return tuple(self._triples)

    def for_subject(self, subject: str) -> tuple[Triple, ...]:
        return tuple(self._triples[i] for i in self._by_subject.get(subject, ()))

    def _candidate_indices(
        self,
        subject: str | None,
        predicate: str | None,
        object_: str | None,
    ):
        """Pick the most selective available index; fall back to a full scan."""
        if subject is not None and predicate is not None:
            return self._by_subject_predicate.get((subject, predicate), ())
        if subject is not None:
            return self._by_subject.get(subject, ())
        if predicate is not None:
            return self._by_predicate.get(predicate, ())
        if object_ is not None:
            return self._by_object.get(object_, ())
        return range(len(self._triples))

    def matching(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object_: str | None = None,
    ) -> tuple[Triple, ...]:
        candidates = self._candidate_indices(subject, predicate, object_)
        result = []
        for i in candidates:
            triple = self._triples[i]
            if subject is not None and triple.subject != subject:
                continue
            if predicate is not None and triple.predicate != predicate:
                continue
            if object_ is not None and triple.object != object_:
                continue
            result.append(triple)
        return tuple(result)

    def objects(self, subject: str, predicate: str) -> tuple[str, ...]:
        return tuple(triple.object for triple in self.matching(subject=subject, predicate=predicate))

    def reasoning_path_ids(self, subject: str) -> tuple[str, ...]:
        return tuple(
            self._triples[i].evidence_id
            for i in self._by_subject.get(subject, ())
            if self._triples[i].evidence_id is not None
        )

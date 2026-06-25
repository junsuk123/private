from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str
    evidence_id: str | None = None


class KnowledgeGraph:
    def __init__(self) -> None:
        self._triples: list[Triple] = []

    def add(self, subject: str, predicate: str, object_: str, evidence_id: str | None = None) -> None:
        triple = Triple(subject, predicate, object_, evidence_id)
        if triple not in self._triples:
            self._triples.append(triple)

    def triples(self) -> tuple[Triple, ...]:
        return tuple(self._triples)

    def for_subject(self, subject: str) -> tuple[Triple, ...]:
        return tuple(triple for triple in self._triples if triple.subject == subject)

    def matching(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object_: str | None = None,
    ) -> tuple[Triple, ...]:
        return tuple(
            triple
            for triple in self._triples
            if (subject is None or triple.subject == subject)
            and (predicate is None or triple.predicate == predicate)
            and (object_ is None or triple.object == object_)
        )

    def objects(self, subject: str, predicate: str) -> tuple[str, ...]:
        return tuple(triple.object for triple in self.matching(subject=subject, predicate=predicate))

    def reasoning_path_ids(self, subject: str) -> tuple[str, ...]:
        return tuple(
            triple.evidence_id
            for triple in self._triples
            if triple.subject == subject and triple.evidence_id is not None
        )

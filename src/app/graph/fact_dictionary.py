from __future__ import annotations

"""Label <-> integer id interning for the integer FactTable.

Live ontology facts are stored as compact integer ids (see ``fact_table.py``);
human-readable labels are kept only here so that reasoning works on ids while
explanations / GUI can be reconstructed on demand. Ids are assigned densely and
monotonically per namespace, so a dictionary built deterministically (base
vocabulary first) yields stable ids that can be hashed for cache versioning.

See docs/ontology_and_gnn.md (working triple store).
"""

# Namespaces. Subjects and objects share the ``TERM`` space because an object of
# one triple is frequently the subject of another; a shared space lets id
# equality express node identity directly (and matches the int32/int32 schema).
NS_TERM = "term"
NS_PREDICATE = "predicate"
NS_EVIDENCE = "evidence"

# Sentinel id for "absent" (e.g. a triple with no evidence id).
NO_ID = -1


class FactDictionary:
    def __init__(self) -> None:
        self._label_to_id: dict[str, dict[str, int]] = {}
        self._id_to_label: dict[str, list[str]] = {}

    def _space(self, namespace: str) -> tuple[dict[str, int], list[str]]:
        forward = self._label_to_id.get(namespace)
        if forward is None:
            forward = {}
            self._label_to_id[namespace] = forward
            self._id_to_label[namespace] = []
        return forward, self._id_to_label[namespace]

    def intern(self, namespace: str, label: str) -> int:
        """Return the id for ``label`` in ``namespace``, assigning a new one if needed."""
        forward, reverse = self._space(namespace)
        existing = forward.get(label)
        if existing is not None:
            return existing
        new_id = len(reverse)
        forward[label] = new_id
        reverse.append(label)
        return new_id

    def get_id(self, namespace: str, label: str) -> int | None:
        forward = self._label_to_id.get(namespace)
        if forward is None:
            return None
        return forward.get(label)

    def label(self, namespace: str, id_: int) -> str:
        reverse = self._id_to_label.get(namespace)
        if reverse is None or id_ < 0 or id_ >= len(reverse):
            raise KeyError(f"unknown id {id_} in namespace {namespace!r}")
        return reverse[id_]

    def size(self, namespace: str) -> int:
        reverse = self._id_to_label.get(namespace)
        return 0 if reverse is None else len(reverse)

    def namespaces(self) -> tuple[str, ...]:
        return tuple(self._id_to_label.keys())

    # -- convenience wrappers for the FactTable's three core namespaces --------
    def intern_term(self, label: str) -> int:
        return self.intern(NS_TERM, label)

    def intern_predicate(self, label: str) -> int:
        return self.intern(NS_PREDICATE, label)

    def intern_evidence(self, label: str | None) -> int:
        if label is None:
            return NO_ID
        return self.intern(NS_EVIDENCE, label)

    def term(self, id_: int) -> str:
        return self.label(NS_TERM, id_)

    def predicate(self, id_: int) -> str:
        return self.label(NS_PREDICATE, id_)

    def evidence(self, id_: int) -> str | None:
        if id_ == NO_ID:
            return None
        return self.label(NS_EVIDENCE, id_)

    def signature(self) -> str:
        """Order-sensitive hash of every namespace's label sequence.

        Stable for a deterministically built dictionary; used later for
        materialization-cache versioning.
        """
        import hashlib

        hasher = hashlib.sha256()
        for namespace in sorted(self._id_to_label):
            hasher.update(namespace.encode("utf-8"))
            hasher.update(b"\x00")
            for label in self._id_to_label[namespace]:
                hasher.update(label.encode("utf-8"))
                hasher.update(b"\x00")
            hasher.update(b"\x01")
        return hasher.hexdigest()

from __future__ import annotations

"""Compact integer-id fact store.

Facts are held as parallel integer columns (subject/predicate/object ids,
validity window, quantized confidence & source quality, and bit flags) with
dictionary indices for O(1) lookup by subject, predicate, (subject, predicate),
and object. This is the integer substrate the acceleration plan calls for; the
string-based ``KnowledgeGraph`` remains the primary store and is bridged in via
``fact_adapters.py`` so existing callers are unaffected.

Confidence and source quality are quantized to 0..255 (uint8-range):
    confidence_q = round(confidence_float * 255)
    confidence_float = confidence_q / 255.0

See docs/ontology_acceleration_audit.md (Phase 2).
"""

from typing import NamedTuple

from .fact_dictionary import NO_ID, FactDictionary

# Open-ended validity window sentinel (uint-nullable valid_until).
NO_TIME = -1

# Bit flags (fit in uint16). "explicit" and "inferred" are mutually exclusive by
# convention but not enforced.
FLAG_EXPLICIT = 1 << 0
FLAG_INFERRED = 1 << 1
FLAG_MATERIALIZED = 1 << 2
FLAG_SYNTHETIC = 1 << 3
FLAG_ESTIMATED = 1 << 4
FLAG_STALE = 1 << 5
FLAG_LIVE = 1 << 6
FLAG_TRUSTED = 1 << 7

_Q_MAX = 255


def quantize_unit(value: float) -> int:
    """Quantize a [0.0, 1.0] value to an integer in [0, 255] (clamped)."""
    if value <= 0.0:
        return 0
    if value >= 1.0:
        return _Q_MAX
    return int(round(value * _Q_MAX))


def dequantize_unit(q: int) -> float:
    """Inverse of :func:`quantize_unit`."""
    if q <= 0:
        return 0.0
    if q >= _Q_MAX:
        return 1.0
    return q / float(_Q_MAX)


class FactRow(NamedTuple):
    fact_id: int
    subject_id: int
    predicate_id: int
    object_id: int
    valid_from: int
    valid_until: int
    confidence_q: int
    source_quality_q: int
    flags: int
    evidence_id: int

    def is_active_at(self, timestamp: int) -> bool:
        if timestamp < self.valid_from:
            return False
        if self.valid_until != NO_TIME and timestamp >= self.valid_until:
            return False
        return True

    def has_flag(self, flag: int) -> bool:
        return bool(self.flags & flag)


class FactTable:
    def __init__(self, dictionary: FactDictionary | None = None) -> None:
        self.dictionary = dictionary if dictionary is not None else FactDictionary()
        # Columns (row-aligned).
        self._subject: list[int] = []
        self._predicate: list[int] = []
        self._object: list[int] = []
        self._valid_from: list[int] = []
        self._valid_until: list[int] = []
        self._confidence_q: list[int] = []
        self._source_quality_q: list[int] = []
        self._flags: list[int] = []
        self._evidence: list[int] = []
        # Indices (values are row ids, appended in insertion order).
        self._by_subject: dict[int, list[int]] = {}
        self._by_predicate: dict[int, list[int]] = {}
        self._by_object: dict[int, list[int]] = {}
        self._by_sp: dict[tuple[int, int], list[int]] = {}
        # Unique (s, p, o) -> row id, for has_fact / dedup / expire / update.
        self._row_of: dict[tuple[int, int, int], int] = {}
        # Tombstoned row ids (removed but columns kept to preserve row ids).
        self._deleted: set[int] = set()

    # -- id resolution ---------------------------------------------------------
    def _term_id(self, value: int | str) -> int:
        return value if isinstance(value, int) else self.dictionary.intern_term(value)

    def _pred_id(self, value: int | str) -> int:
        return value if isinstance(value, int) else self.dictionary.intern_predicate(value)

    def _term_id_lookup(self, value: int | str) -> int | None:
        if isinstance(value, int):
            return value
        return self.dictionary.get_id("term", value)

    def _pred_id_lookup(self, value: int | str) -> int | None:
        if isinstance(value, int):
            return value
        return self.dictionary.get_id("predicate", value)

    # -- mutation --------------------------------------------------------------
    def add_fact(
        self,
        subject: int | str,
        predicate: int | str,
        object_: int | str,
        confidence: float = 1.0,
        source_quality: float = 1.0,
        flags: int = FLAG_EXPLICIT,
        valid_from: int = 0,
        valid_until: int = NO_TIME,
        evidence: int | str | None = None,
    ) -> int:
        """Insert a fact (or update it in place if the (s, p, o) already exists).

        Returns the row id. ``subject``/``predicate``/``object_`` may be labels
        (interned) or existing ids. ``confidence``/``source_quality`` are floats
        in [0, 1] and are stored quantized.
        """
        s = self._term_id(subject)
        p = self._pred_id(predicate)
        o = self._term_id(object_)
        ev = NO_ID if evidence is None else (
            evidence if isinstance(evidence, int) else self.dictionary.intern_evidence(evidence)
        )
        key = (s, p, o)
        existing = self._row_of.get(key)
        if existing is not None:
            self._valid_from[existing] = valid_from
            self._valid_until[existing] = valid_until
            self._confidence_q[existing] = quantize_unit(confidence)
            self._source_quality_q[existing] = quantize_unit(source_quality)
            self._flags[existing] = flags
            self._evidence[existing] = ev
            self._deleted.discard(existing)
            return existing

        row_id = len(self._subject)
        self._subject.append(s)
        self._predicate.append(p)
        self._object.append(o)
        self._valid_from.append(valid_from)
        self._valid_until.append(valid_until)
        self._confidence_q.append(quantize_unit(confidence))
        self._source_quality_q.append(quantize_unit(source_quality))
        self._flags.append(flags)
        self._evidence.append(ev)
        self._row_of[key] = row_id
        self._by_subject.setdefault(s, []).append(row_id)
        self._by_predicate.setdefault(p, []).append(row_id)
        self._by_object.setdefault(o, []).append(row_id)
        self._by_sp.setdefault((s, p), []).append(row_id)
        return row_id

    def update_live_fact(
        self,
        symbol_id: int | str,
        predicate_id: int | str,
        object_id: int | str,
        timestamp: int,
        confidence_q: int,
        flags: int = FLAG_LIVE,
    ) -> int:
        """Refresh (or create) a live fact with an already-quantized confidence.

        ``timestamp`` becomes the fact's ``valid_from``; the window stays open.
        """
        s = self._term_id(symbol_id)
        p = self._pred_id(predicate_id)
        o = self._term_id(object_id)
        key = (s, p, o)
        existing = self._row_of.get(key)
        if existing is not None:
            self._valid_from[existing] = timestamp
            self._valid_until[existing] = NO_TIME
            self._confidence_q[existing] = max(0, min(_Q_MAX, confidence_q))
            self._flags[existing] = flags
            self._deleted.discard(existing)
            return existing
        return self.add_fact(
            s,
            p,
            o,
            confidence=dequantize_unit(confidence_q),
            flags=flags,
            valid_from=timestamp,
            valid_until=NO_TIME,
        )

    def remove_or_expire_fact(
        self,
        subject: int | str,
        predicate: int | str,
        object_: int | str,
        valid_until: int | None = None,
    ) -> bool:
        """Remove a fact (``valid_until`` None) or expire it at ``valid_until``.

        Returns True if a matching fact existed. Removal is a tombstone: the row
        id is retired so existing ids stay valid, and the fact stops appearing in
        queries.
        """
        s = self._term_id_lookup(subject)
        p = self._pred_id_lookup(predicate)
        o = self._term_id_lookup(object_)
        if s is None or p is None or o is None:
            return False
        row_id = self._row_of.get((s, p, o))
        if row_id is None:
            return False
        if valid_until is None:
            self._deleted.add(row_id)
            del self._row_of[(s, p, o)]
        else:
            self._valid_until[row_id] = valid_until
        return True

    # -- access ----------------------------------------------------------------
    def _row(self, row_id: int) -> FactRow:
        return FactRow(
            row_id,
            self._subject[row_id],
            self._predicate[row_id],
            self._object[row_id],
            self._valid_from[row_id],
            self._valid_until[row_id],
            self._confidence_q[row_id],
            self._source_quality_q[row_id],
            self._flags[row_id],
            self._evidence[row_id],
        )

    def _collect(self, row_ids, as_of: int | None) -> tuple[FactRow, ...]:
        rows = []
        for row_id in row_ids:
            if row_id in self._deleted:
                continue
            row = self._row(row_id)
            if as_of is not None and not row.is_active_at(as_of):
                continue
            rows.append(row)
        return tuple(rows)

    def get_facts_by_subject(self, subject_id: int | str, as_of: int | None = None) -> tuple[FactRow, ...]:
        s = self._term_id_lookup(subject_id)
        if s is None:
            return ()
        return self._collect(self._by_subject.get(s, ()), as_of)

    def get_facts_by_predicate(self, predicate_id: int | str, as_of: int | None = None) -> tuple[FactRow, ...]:
        p = self._pred_id_lookup(predicate_id)
        if p is None:
            return ()
        return self._collect(self._by_predicate.get(p, ()), as_of)

    def has_fact(self, subject_id: int | str, predicate_id: int | str, object_id: int | str) -> bool:
        s = self._term_id_lookup(subject_id)
        p = self._pred_id_lookup(predicate_id)
        o = self._term_id_lookup(object_id)
        if s is None or p is None or o is None:
            return False
        row_id = self._row_of.get((s, p, o))
        return row_id is not None and row_id not in self._deleted

    def query(
        self,
        subject_id: int | str | None = None,
        predicate_id: int | str | None = None,
        object_id: int | str | None = None,
        as_of: int | None = None,
    ) -> tuple[FactRow, ...]:
        """Pattern query; None components are wildcards. Uses the most selective index."""
        s = None if subject_id is None else self._term_id_lookup(subject_id)
        p = None if predicate_id is None else self._pred_id_lookup(predicate_id)
        o = None if object_id is None else self._term_id_lookup(object_id)
        # An unknown (never-interned) label matches nothing.
        if (subject_id is not None and s is None) or (predicate_id is not None and p is None) or (object_id is not None and o is None):
            return ()

        if s is not None and p is not None:
            candidates = self._by_sp.get((s, p), ())
        elif s is not None:
            candidates = self._by_subject.get(s, ())
        elif p is not None:
            candidates = self._by_predicate.get(p, ())
        elif o is not None:
            candidates = self._by_object.get(o, ())
        else:
            candidates = range(len(self._subject))

        rows = []
        for row_id in candidates:
            if row_id in self._deleted:
                continue
            if s is not None and self._subject[row_id] != s:
                continue
            if p is not None and self._predicate[row_id] != p:
                continue
            if o is not None and self._object[row_id] != o:
                continue
            row = self._row(row_id)
            if as_of is not None and not row.is_active_at(as_of):
                continue
            rows.append(row)
        return tuple(rows)

    def __len__(self) -> int:
        return len(self._subject) - len(self._deleted)

    # -- decoding --------------------------------------------------------------
    def to_human_readable(self, rows: FactRow | int | "tuple[FactRow, ...] | list") -> tuple[dict, ...]:
        """Decode row(s) to dicts of labels + dequantized floats.

        Accepts a single ``FactRow``, a single row-id int, or an iterable of either.
        """
        if isinstance(rows, (FactRow, int)):
            rows = [rows]
        out = []
        for item in rows:
            row = self._row(item) if isinstance(item, int) else item
            out.append(
                {
                    "fact_id": row.fact_id,
                    "subject": self.dictionary.term(row.subject_id),
                    "predicate": self.dictionary.predicate(row.predicate_id),
                    "object": self.dictionary.term(row.object_id),
                    "confidence": dequantize_unit(row.confidence_q),
                    "source_quality": dequantize_unit(row.source_quality_q),
                    "evidence_id": self.dictionary.evidence(row.evidence_id),
                    "valid_from": row.valid_from,
                    "valid_until": None if row.valid_until == NO_TIME else row.valid_until,
                    "flags": row.flags,
                }
            )
        return tuple(out)

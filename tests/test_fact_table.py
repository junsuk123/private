from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.graph import (
    FactDictionary,
    FactTable,
    KnowledgeGraph,
    Triple,
    dequantize_unit,
    fact_table_from_graph,
    graph_from_fact_table,
    quantize_unit,
    rows_to_triples,
)
from app.graph.fact_dictionary import NO_ID, NS_PREDICATE, NS_TERM
from app.graph.fact_table import FLAG_LIVE, NO_TIME
from app.graph.ontology import CLASSES, RELATIONSHIPS, build_base_dictionary


class FactDictionaryTest(unittest.TestCase):
    def test_intern_is_stable_and_reversible(self) -> None:
        d = FactDictionary()
        a = d.intern(NS_TERM, "AAPL")
        b = d.intern(NS_TERM, "MSFT")
        self.assertNotEqual(a, b)
        self.assertEqual(d.intern(NS_TERM, "AAPL"), a)  # idempotent
        self.assertEqual(d.label(NS_TERM, a), "AAPL")
        self.assertEqual(d.get_id(NS_TERM, "MSFT"), b)
        self.assertIsNone(d.get_id(NS_TERM, "GOOG"))

    def test_namespaces_are_independent(self) -> None:
        d = FactDictionary()
        term_id = d.intern(NS_TERM, "supportsSignal")
        pred_id = d.intern(NS_PREDICATE, "supportsSignal")
        # Same label, different namespace: ids assigned independently (both 0 here).
        self.assertEqual(d.label(NS_TERM, term_id), "supportsSignal")
        self.assertEqual(d.label(NS_PREDICATE, pred_id), "supportsSignal")

    def test_evidence_none_maps_to_no_id(self) -> None:
        d = FactDictionary()
        self.assertEqual(d.intern_evidence(None), NO_ID)
        self.assertIsNone(d.evidence(NO_ID))
        ev = d.intern_evidence("ev-1")
        self.assertEqual(d.evidence(ev), "ev-1")

    def test_unknown_id_raises(self) -> None:
        d = FactDictionary()
        with self.assertRaises(KeyError):
            d.label(NS_TERM, 99)

    def test_signature_is_deterministic(self) -> None:
        self.assertEqual(build_base_dictionary().signature(), build_base_dictionary().signature())


class QuantizationTest(unittest.TestCase):
    def test_bounds(self) -> None:
        self.assertEqual(quantize_unit(0.0), 0)
        self.assertEqual(quantize_unit(1.0), 255)
        self.assertEqual(quantize_unit(-5.0), 0)
        self.assertEqual(quantize_unit(5.0), 255)

    def test_roundtrip_within_tolerance(self) -> None:
        for v in (0.0, 0.1, 0.25, 0.5, 0.73, 0.9, 1.0):
            err = abs(dequantize_unit(quantize_unit(v)) - v)
            self.assertLessEqual(err, 1.0 / 255.0)


class FactTableTest(unittest.TestCase):
    def _table(self) -> FactTable:
        t = FactTable()
        t.add_fact("AAPL", "supportsSignal", "EarningsGrowth", confidence=0.8)
        t.add_fact("AAPL", "increasesRiskOf", "VolatilityRisk", confidence=0.4)
        t.add_fact("MSFT", "supportsSignal", "EarningsGrowth", confidence=0.6)
        return t

    def test_add_and_has_fact(self) -> None:
        t = self._table()
        self.assertEqual(len(t), 3)
        self.assertTrue(t.has_fact("AAPL", "supportsSignal", "EarningsGrowth"))
        self.assertFalse(t.has_fact("AAPL", "supportsSignal", "VolatilityRisk"))

    def test_add_duplicate_updates_in_place(self) -> None:
        t = self._table()
        row_id = t.add_fact("AAPL", "supportsSignal", "EarningsGrowth", confidence=0.95)
        self.assertEqual(len(t), 3)  # no new row
        rows = t.query("AAPL", "supportsSignal", "EarningsGrowth")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].fact_id, row_id)
        self.assertEqual(rows[0].confidence_q, quantize_unit(0.95))

    def test_query_by_each_index(self) -> None:
        t = self._table()
        self.assertEqual(len(t.get_facts_by_subject("AAPL")), 2)
        self.assertEqual(len(t.get_facts_by_predicate("supportsSignal")), 2)
        self.assertEqual(len(t.query(object_id="EarningsGrowth")), 2)
        self.assertEqual(len(t.query(predicate_id="supportsSignal", object_id="EarningsGrowth")), 2)
        self.assertEqual(len(t.query()), 3)

    def test_query_unknown_label_matches_nothing(self) -> None:
        t = self._table()
        self.assertEqual(t.query("NVDA"), ())
        self.assertEqual(t.get_facts_by_subject("NVDA"), ())

    def test_remove_tombstones_fact(self) -> None:
        t = self._table()
        self.assertTrue(t.remove_or_expire_fact("AAPL", "increasesRiskOf", "VolatilityRisk"))
        self.assertFalse(t.has_fact("AAPL", "increasesRiskOf", "VolatilityRisk"))
        self.assertEqual(len(t), 2)
        self.assertEqual(len(t.get_facts_by_subject("AAPL")), 1)
        # Removing again reports no match.
        self.assertFalse(t.remove_or_expire_fact("AAPL", "increasesRiskOf", "VolatilityRisk"))

    def test_expire_sets_validity_window(self) -> None:
        t = FactTable()
        t.add_fact("AAPL", "hasTicker", "AAPL", valid_from=100)
        self.assertTrue(t.remove_or_expire_fact("AAPL", "hasTicker", "AAPL", valid_until=200))
        # Still present, but only active inside the window.
        self.assertTrue(t.has_fact("AAPL", "hasTicker", "AAPL"))
        self.assertEqual(len(t.query("AAPL", as_of=150)), 1)
        self.assertEqual(len(t.query("AAPL", as_of=250)), 0)
        self.assertEqual(len(t.query("AAPL", as_of=50)), 0)

    def test_update_live_fact_creates_and_refreshes(self) -> None:
        t = FactTable()
        rid = t.update_live_fact("AAPL", "hasQuote", "Fresh", timestamp=1000, confidence_q=200, flags=FLAG_LIVE)
        row = t.query("AAPL", "hasQuote", "Fresh")[0]
        self.assertEqual(row.fact_id, rid)
        self.assertEqual(row.valid_from, 1000)
        self.assertEqual(row.confidence_q, 200)
        self.assertTrue(row.has_flag(FLAG_LIVE))
        # Refresh same fact.
        t.update_live_fact("AAPL", "hasQuote", "Fresh", timestamp=2000, confidence_q=100)
        row2 = t.query("AAPL", "hasQuote", "Fresh")[0]
        self.assertEqual(row2.fact_id, rid)
        self.assertEqual(row2.valid_from, 2000)
        self.assertEqual(row2.confidence_q, 100)

    def test_to_human_readable_decodes(self) -> None:
        t = self._table()
        rows = t.query("AAPL", "supportsSignal", "EarningsGrowth")
        decoded = t.to_human_readable(rows)[0]
        self.assertEqual(decoded["subject"], "AAPL")
        self.assertEqual(decoded["predicate"], "supportsSignal")
        self.assertEqual(decoded["object"], "EarningsGrowth")
        self.assertAlmostEqual(decoded["confidence"], dequantize_unit(quantize_unit(0.8)))
        self.assertIsNone(decoded["valid_until"])


class KnowledgeGraphParityTest(unittest.TestCase):
    """The indexed KnowledgeGraph must match the original list-scan semantics."""

    def _graph(self) -> KnowledgeGraph:
        g = KnowledgeGraph()
        g.add("AAPL", "supportsSignal", "EarningsGrowth", "ev1")
        g.add("AAPL", "supportsSignal", "ProfitabilityQuality", "ev2")
        g.add("AAPL", "increasesRiskOf", "VolatilityRisk")
        g.add("MSFT", "supportsSignal", "EarningsGrowth", "ev3")
        return g

    def test_dedup_and_insertion_order(self) -> None:
        g = self._graph()
        g.add("AAPL", "supportsSignal", "EarningsGrowth", "ev1")  # duplicate
        triples = g.triples()
        self.assertEqual(len(triples), 4)
        self.assertEqual(triples[0], Triple("AAPL", "supportsSignal", "EarningsGrowth", "ev1"))
        self.assertEqual(triples[-1], Triple("MSFT", "supportsSignal", "EarningsGrowth", "ev3"))

    def test_matching_equivalence_to_reference_scan(self) -> None:
        g = self._graph()
        ref = g.triples()

        def reference(subject=None, predicate=None, object_=None):
            return tuple(
                t
                for t in ref
                if (subject is None or t.subject == subject)
                and (predicate is None or t.predicate == predicate)
                and (object_ is None or t.object == object_)
            )

        cases = [
            {},
            {"subject": "AAPL"},
            {"predicate": "supportsSignal"},
            {"object_": "EarningsGrowth"},
            {"subject": "AAPL", "predicate": "supportsSignal"},
            {"subject": "AAPL", "object_": "VolatilityRisk"},
            {"predicate": "supportsSignal", "object_": "EarningsGrowth"},
            {"subject": "NVDA"},  # absent
        ]
        for kw in cases:
            self.assertEqual(g.matching(**kw), reference(**kw), msg=str(kw))

    def test_for_subject_objects_and_reasoning_paths(self) -> None:
        g = self._graph()
        self.assertEqual(len(g.for_subject("AAPL")), 3)
        self.assertEqual(g.objects("AAPL", "supportsSignal"), ("EarningsGrowth", "ProfitabilityQuality"))
        self.assertEqual(g.reasoning_path_ids("AAPL"), ("ev1", "ev2"))  # None evidence skipped
        self.assertEqual(g.reasoning_path_ids("MSFT"), ("ev3",))


class AdapterRoundTripTest(unittest.TestCase):
    def test_graph_to_table_to_graph(self) -> None:
        g = KnowledgeGraph()
        g.add("AAPL", "supportsSignal", "EarningsGrowth", "ev1")
        g.add("AAPL", "increasesRiskOf", "VolatilityRisk")
        g.add("MSFT", "hasTicker", "MSFT")

        table = fact_table_from_graph(g)
        self.assertEqual(len(table), 3)
        self.assertTrue(table.has_fact("AAPL", "supportsSignal", "EarningsGrowth"))

        back = graph_from_fact_table(table)
        self.assertEqual(set(back.triples()), set(g.triples()))

    def test_rows_to_triples_preserves_evidence(self) -> None:
        g = KnowledgeGraph()
        g.add("AAPL", "supportsSignal", "EarningsGrowth", "ev1")
        table = fact_table_from_graph(g)
        rows = table.get_facts_by_subject("AAPL")
        triples = rows_to_triples(table, rows)
        self.assertEqual(triples[0], Triple("AAPL", "supportsSignal", "EarningsGrowth", "ev1"))

    def test_shared_dictionary_keeps_base_ids_low(self) -> None:
        d = build_base_dictionary()
        # Base vocabulary occupies the first ids; dynamic terms come after.
        self.assertEqual(d.size(NS_TERM), len(CLASSES))
        self.assertEqual(d.size(NS_PREDICATE), len(RELATIONSHIPS))
        g = KnowledgeGraph()
        g.add("AAPL", "supportsSignal", "EarningsGrowth")
        fact_table_from_graph(g, d)
        # A brand-new term ("AAPL") is interned after the base classes.
        self.assertEqual(d.get_id(NS_TERM, "AAPL"), len(CLASSES))
        # A base predicate keeps its original low id.
        self.assertEqual(d.get_id(NS_PREDICATE, "supportsSignal"), RELATIONSHIPS.index("supportsSignal"))


if __name__ == "__main__":
    unittest.main()

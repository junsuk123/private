# Ontology Acceleration Benchmark Report

Results for the integer-FactTable / indexed-KnowledgeGraph work (Phase 2). Numbers
are honest measurements from `scripts/benchmark_fact_table.py` on the development
machine — reproduce with:

```powershell
python scripts/benchmark_fact_table.py
```

## String-triple lookup: old linear scan vs new dict index

The live tick loop reads a cached snapshot graph and issues repeated point lookups
(`matching(subject=…, predicate=…)`), one per consumed predicate per symbol. The
pre-refactor `KnowledgeGraph` scanned the whole triple list on every call
(O(total triples)); the indexed version is O(matches).

The benchmark times the full per-symbol × per-predicate lookup sweep, scaled so
total work is comparable across sizes. Both implementations are asserted to return
identical rows before timing.

| symbols | triples | linear ms | indexed ms | speedup |
|--------:|--------:|----------:|-----------:|--------:|
| 1       | 6       | 5.21      | 2.84       | 1.8×    |
| 10      | 60      | 16.82     | 3.26       | 5.2×    |
| 50      | 300     | 62.89     | 3.22       | 19.5×   |
| 100     | 600     | 118.01    | 3.23       | 36.5×   |
| 500     | 3000    | 565.30    | 3.09       | 183.1×  |

**Reading:** linear-scan cost rises with total graph size (as expected for O(n));
indexed-lookup cost stays flat (~3 ms) because it depends only on the number of
matching triples, not the graph size. The speedup therefore grows with the graph —
183× at 500 symbols — which is exactly the per-tick regime the audit flagged.

## FactTable build + query

| operation | value |
|---|---|
| Build FactTable from a 3000-fact graph | 6.24 ms |
| 500 `get_facts_by_subject` queries | 1.80 ms |

FactTable construction is a one-time/amortized cost (offline or per analysis cycle);
per-subject queries are ~3.6 µs each.

## Correctness

`tests/test_fact_table.py` (33 assertions across 4 test classes) verifies:
- `FactDictionary` interning is stable, reversible, and namespace-isolated; `NO_ID`
  handling for absent evidence; deterministic `signature()`.
- Quantization round-trip error ≤ 1/255 across the range.
- `FactTable` add/dedup-update/query-by-every-index/has_fact/remove(tombstone)/
  expire(validity-window)/`update_live_fact`/`to_human_readable`.
- **KnowledgeGraph parity**: the indexed `matching`/`for_subject`/`objects`/
  `reasoning_path_ids` return byte-identical results to a reference linear scan
  across 8 query shapes, plus dedup and insertion-order preservation.
- Adapter round-trip `graph → FactTable → graph` and evidence preservation.

Existing graph regression suites pass unchanged: `test_graph_memory_limits`,
`test_ontology_framework`, `test_ontology_reasoner_policy`, `test_web_graph_payload`,
`test_investor_flow_strategy`, `test_trading_strategy_semantics`,
`test_semantic_features`, `test_realtime_exit_decision`, `test_risk_manager`,
`test_time_series_fusion`, `test_demo_research_integration` (113 tests total).

## What this does and does not change

- **Does:** removes the per-tick O(n) linear scan from `KnowledgeGraph` accessors;
  adds the integer `FactTable` substrate (quantized confidence/source-quality, bit
  flags, validity windows) with O(1) indexed lookup, behind a compatibility adapter.
- **Does not:** change any decision logic, gate ordering, or trade behavior. The
  indexed graph is behavior-identical to the old one; the FactTable is additive and
  not yet wired into the live decision path (that is delta-reasoning, a later phase).

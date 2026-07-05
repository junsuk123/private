"""Microbenchmark: indexed KnowledgeGraph / FactTable lookup vs the old linear scan.

Simulates the live tick loop's access pattern: repeated point lookups
(``matching(subject=..., predicate=...)``) against a static snapshot graph. The
old implementation scanned the full triple list per call (O(total)); the new one
uses dict indices (O(matches)).

Run:  python scripts/benchmark_fact_table.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.graph import KnowledgeGraph, fact_table_from_graph  # noqa: E402


class LinearGraph:
    """Reference implementation matching the pre-refactor list-scan semantics."""

    def __init__(self) -> None:
        self._triples = []

    def add(self, s, p, o, ev=None):
        t = (s, p, o, ev)
        if t not in self._triples:
            self._triples.append(t)

    def matching(self, subject=None, predicate=None):
        return tuple(
            t
            for t in self._triples
            if (subject is None or t[0] == subject) and (predicate is None or t[1] == predicate)
        )


PREDICATES = ("supportsSignal", "contradictsSignal", "increasesRiskOf", "hasTechnicalIndicator")
OBJECTS = ("EarningsGrowth", "VolatilityRisk", "LiquiditySupport", "MacroRateRisk", "OrderFlow")


def build(n_symbols: int, per_symbol: int):
    symbols = [f"SYM{i:05d}" for i in range(n_symbols)]
    linear = LinearGraph()
    indexed = KnowledgeGraph()
    for si, sym in enumerate(symbols):
        for j in range(per_symbol):
            p = PREDICATES[j % len(PREDICATES)]
            o = OBJECTS[(si + j) % len(OBJECTS)]
            linear.add(sym, p, o, f"ev{si}-{j}")
            indexed.add(sym, p, o, f"ev{si}-{j}")
    return symbols, linear, indexed


def time_lookups(graph, symbols, rounds: int) -> float:
    start = time.perf_counter()
    for _ in range(rounds):
        for sym in symbols:
            for p in PREDICATES:
                graph.matching(subject=sym, predicate=p)
    return (time.perf_counter() - start) * 1000.0


def main() -> None:
    print(f"{'symbols':>8} {'triples':>8} {'linear ms':>12} {'indexed ms':>12} {'speedup':>9}")
    for n_symbols in (1, 10, 50, 100, 500):
        per_symbol = 6
        rounds = max(1, 2000 // n_symbols)
        symbols, linear, indexed = build(n_symbols, per_symbol)
        total_triples = len(indexed.triples())
        # Correctness check: both return the same rows.
        for sym in symbols[: min(5, len(symbols))]:
            for p in PREDICATES:
                a = {(t[0], t[1], t[2]) for t in linear.matching(subject=sym, predicate=p)}
                b = {(t.subject, t.predicate, t.object) for t in indexed.matching(subject=sym, predicate=p)}
                assert a == b, f"mismatch for {sym}/{p}"
        lin_ms = time_lookups(linear, symbols, rounds)
        idx_ms = time_lookups(indexed, symbols, rounds)
        speedup = lin_ms / idx_ms if idx_ms else float("inf")
        print(f"{n_symbols:>8} {total_triples:>8} {lin_ms:>12.2f} {idx_ms:>12.2f} {speedup:>8.1f}x")

    # FactTable build + query sanity timing.
    _, _, indexed = build(500, 6)
    t0 = time.perf_counter()
    table = fact_table_from_graph(indexed)
    build_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    for sym in (f"SYM{i:05d}" for i in range(500)):
        table.get_facts_by_subject(sym)
    query_ms = (time.perf_counter() - t0) * 1000.0
    print(f"\nFactTable build ({len(table)} facts): {build_ms:.2f} ms; 500 subject queries: {query_ms:.2f} ms")


if __name__ == "__main__":
    main()

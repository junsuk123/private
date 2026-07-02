"""OWL/RDFS schema loading and cached materialization primitives (T04).

Loads the ontology Turtle files once and caches the parsed schema graph
(keyed by file mtime) so repeated analysis cycles do not re-parse. Provides a
single ``apply_closure`` entry point that runs RDFS or OWL RL materialization
via ``owlrl``.

This module performs **logical entailment only**. It never computes scores,
ranks candidates, or authorizes trades.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from rdflib import Graph

from app.graph.rdf_graph import bind_namespaces

_ONTOLOGY_DIR = Path(__file__).resolve().parent.parent / "ontology"
_CORE = _ONTOLOGY_DIR / "trading_core.ttl"
_RULES = _ONTOLOGY_DIR / "trading_rules.ttl"

# module-level cache: (paths signature) -> parsed schema Graph
_schema_cache: dict[tuple, Graph] = {}


@dataclass(frozen=True)
class SchemaLoadResult:
    graph: Graph
    triple_count: int
    files: tuple[str, ...]
    error: str | None = None


def _signature(paths: tuple[Path, ...]) -> tuple:
    parts = []
    for path in paths:
        try:
            parts.append((str(path), path.stat().st_mtime_ns))
        except OSError:
            parts.append((str(path), None))
    return tuple(parts)


def load_schema_graph(*, include_rules: bool = True) -> SchemaLoadResult:
    """Load (and cache) the core (+ rules) ontology as a single schema graph.

    Cached by file mtime; editing a TTL invalidates the cache automatically.
    On failure returns an empty graph and a populated ``error`` so callers can
    surface it in diagnostics and fall back safely (never fail open).
    """
    paths = (_CORE, _RULES) if include_rules else (_CORE,)
    signature = _signature(paths)
    cached = _schema_cache.get(signature)
    if cached is not None:
        return SchemaLoadResult(cached, len(cached), tuple(str(p) for p in paths))

    graph = Graph()
    bind_namespaces(graph)
    error: str | None = None
    try:
        for path in paths:
            graph.parse(str(path), format="turtle")
        _schema_cache[signature] = graph
    except Exception as exc:  # pragma: no cover - defensive
        error = f"ontology schema load failed: {exc}"
    return SchemaLoadResult(graph, len(graph), tuple(str(p) for p in paths), error)


def reset_schema_cache() -> None:
    _schema_cache.clear()


def apply_closure(graph: Graph, *, profile: str = "owlrl") -> str | None:
    """Materialize entailments in-place on ``graph``.

    ``profile``: ``"owlrl"`` (default) for OWL 2 RL semantics, ``"rdfs"`` for
    the cheaper RDFS-only closure. Returns ``None`` on success or an error
    string on failure (the graph is left as-is so the caller keeps the
    non-inferred assertions).
    """
    try:
        import owlrl

        if profile == "rdfs":
            owlrl.DeductiveClosure(owlrl.RDFS_Semantics).expand(graph)
        else:
            owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(graph)
        return None
    except Exception as exc:  # pragma: no cover - defensive
        return f"owl materialization failed: {exc}"


def profile_from_env(default: str = "owlrl") -> str:
    """Reasoning profile from ONTOLOGY_REASONING_PROFILE (owlrl|rdfs)."""
    value = os.getenv("ONTOLOGY_REASONING_PROFILE", default).strip().lower()
    return value if value in {"owlrl", "rdfs"} else default

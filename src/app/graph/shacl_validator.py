"""SHACL validation service (T05).

Runs the SHACL shapes in ``src/app/ontology/trading_shapes.ttl`` against a
per-cycle RDF graph and returns **structured** results.

Modes:
* ``mode="live"``  -> a violation BLOCKS the candidate (``blocking=True``).
* ``mode="paper"`` -> violations are reported as warnings (``blocking=False``);
  paper/mock mode never triggers live execution on them.

Fail-safe: if pyshacl or the shapes fail to load/run, ``ValidationReport.error``
is populated and ``conforms`` is ``False``. Callers must treat a failed
validation as non-conforming for live trading (never fail open).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from rdflib import Graph

_SHAPES_FILE = Path(__file__).resolve().parent.parent / "ontology" / "trading_shapes.ttl"

_SH = "http://www.w3.org/ns/shacl#"


@dataclass(frozen=True)
class ValidationViolation:
    focus_node: str
    message: str
    severity: str
    source_shape: str
    path: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    conforms: bool
    mode: str
    blocking: bool
    violations: tuple[ValidationViolation, ...] = ()
    validate_ms: float = 0.0
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "conforms": self.conforms,
            "mode": self.mode,
            "blocking": self.blocking,
            "error": self.error,
            "validate_ms": self.validate_ms,
            "violations": [
                {
                    "focus_node": v.focus_node,
                    "message": v.message,
                    "severity": v.severity,
                    "source_shape": v.source_shape,
                    "path": v.path,
                }
                for v in self.violations
            ],
        }


@lru_cache(maxsize=1)
def _load_shapes() -> Graph:
    graph = Graph()
    graph.parse(str(_SHAPES_FILE), format="turtle")
    return graph


def reset_shapes_cache() -> None:
    _load_shapes.cache_clear()


def _local(text: str) -> str:
    for sep in ("#", "/"):
        if sep in text:
            text = text.rsplit(sep, 1)[-1]
    return text


def _parse_results(report_graph: Graph) -> tuple[ValidationViolation, ...]:
    from rdflib import URIRef
    from rdflib.namespace import RDF

    sh = lambda name: URIRef(_SH + name)  # noqa: E731
    violations: list[ValidationViolation] = []
    for result in report_graph.subjects(RDF.type, sh("ValidationResult")):
        focus = next(report_graph.objects(result, sh("focusNode")), None)
        message = next(report_graph.objects(result, sh("resultMessage")), None)
        severity = next(report_graph.objects(result, sh("resultSeverity")), None)
        shape = next(report_graph.objects(result, sh("sourceShape")), None)
        path = next(report_graph.objects(result, sh("resultPath")), None)
        violations.append(
            ValidationViolation(
                focus_node=str(focus) if focus is not None else "",
                message=str(message) if message is not None else "",
                severity=_local(str(severity)) if severity is not None else "Violation",
                source_shape=_local(str(shape)) if shape is not None else "",
                path=_local(str(path)) if path is not None else None,
            )
        )
    return tuple(violations)


def validate_graph(graph: Graph, *, mode: str = "paper") -> ValidationReport:
    """Validate ``graph`` against the trading SHACL shapes.

    ``graph`` should already contain the data to validate (typically the
    materialized RDF graph). Ontology/inference is not re-run here.
    """
    import time

    normalized_mode = "live" if str(mode).lower() == "live" else "paper"
    t0 = time.perf_counter()
    try:
        import pyshacl

        shapes = _load_shapes()
        conforms, report_graph, _text = pyshacl.validate(
            graph,
            shacl_graph=shapes,
            inference="none",  # OWL closure is already done upstream
            advanced=True,
            meta_shacl=False,
        )
        violations = _parse_results(report_graph)
        validate_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return ValidationReport(
            conforms=bool(conforms),
            mode=normalized_mode,
            blocking=(normalized_mode == "live" and not conforms),
            violations=violations,
            validate_ms=validate_ms,
        )
    except Exception as exc:  # pragma: no cover - defensive
        validate_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        # Fail closed for live: an error means "not conforming".
        return ValidationReport(
            conforms=False,
            mode=normalized_mode,
            blocking=(normalized_mode == "live"),
            violations=(),
            validate_ms=validate_ms,
            error=f"shacl validation failed: {exc}",
        )

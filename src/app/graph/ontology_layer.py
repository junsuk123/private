"""Ontology layer orchestration (T09 / T14).

Ties the additive RDF/OWL/SHACL layer together into one call the pipeline can
make after it has built the custom ``KnowledgeGraph``:

    KnowledgeGraph (+ markets, account, NPU scores)
        -> RDF assertion graph            (rdf_adapter.knowledge_graph_to_rdf)
        -> merge schema + OWL RL closure  (semantic_materializer.materialize)
        -> SHACL validation               (shacl_validator.validate_graph)
        -> UI payload (asserted vs inferred), timings, diagnostics

This layer is *advisory* to the pipeline: it enriches explanation and provides
inferred semantic classes + validation results. It does NOT generate orders and
does NOT bypass the RiskManager. It is fully fail-safe — any error is captured
in ``errors`` and the pipeline continues with the unchanged trading path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

from app.graph.knowledge_graph import KnowledgeGraph
from app.graph.rdf_adapter import (
    attach_account_snapshot,
    attach_scoring_provenance,
    knowledge_graph_to_rdf,
    rdf_to_ui_payload,
)
from app.graph.rdf_graph import RdfTradingGraph
from app.graph.semantic_materializer import MaterializationResult, materialize
from app.graph.shacl_validator import ValidationReport, validate_graph


@dataclass(frozen=True)
class OntologyLayerResult:
    """Structured output of the additive ontology layer for one analysis cycle."""

    rdf: RdfTradingGraph
    materialization: MaterializationResult
    validation: ValidationReport
    ui_payload: dict = field(default_factory=dict)
    inferred_types: dict[str, list[str]] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    mode: str = "paper"

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        """GUI/API friendly payload separating asserted, inferred, validation."""
        return {
            "mode": self.mode,
            "ok": self.ok,
            "errors": list(self.errors),
            "counts": {
                "asserted": self.materialization.asserted_count,
                "inferred": self.materialization.inferred_count,
                "schema": self.materialization.schema_count,
            },
            "reasoning_profile": self.materialization.profile,
            "inferred_types": self.inferred_types,
            "validation": self.validation.as_dict(),
            "timings_ms": self.timings_ms,
            "graph": self.ui_payload,
        }


def rdf_layer_enabled() -> bool:
    """Feature flag: ONTOLOGY_RDF_LAYER (default on). Set to 0 to disable."""
    return os.getenv("ONTOLOGY_RDF_LAYER", "1").strip().lower() not in {"0", "false", "no", "off"}


def build_ontology_layer(
    graph: KnowledgeGraph,
    *,
    markets: Mapping[str, object] | None = None,
    account: object | None = None,
    live: bool = False,
    cycle_id: str | None = None,
    score_sink: Mapping[str, object] | None = None,
    reasoning_profile: str | None = None,
) -> OntologyLayerResult | None:
    """Run the additive RDF/OWL/SHACL layer. Returns None if disabled.

    Never raises: on any internal failure the error is captured in the result's
    ``errors`` and (for live mode) the validation is marked non-conforming so
    the caller never fails open.
    """
    if not rdf_layer_enabled():
        return None

    mode = "live" if live else "paper"
    errors: list[str] = []

    try:
        live_tickers = tuple(markets.keys()) if (markets and live) else ()
        rdf = knowledge_graph_to_rdf(
            graph,
            cycle_id=cycle_id,
            markets=markets,
            live_tickers=live_tickers,
        )
        # Represent NPU/CPU/heuristic scorer output as RDF evidence (T08).
        if score_sink:
            status = score_sink.get("npu_status")
            attach_scoring_provenance(
                rdf,
                backend=getattr(status, "backend", "unknown"),
                model_kind=getattr(status, "model_kind", "unknown"),
                npu_scores=score_sink.get("npu_scores"),  # type: ignore[arg-type]
            )
        if account is not None:
            attach_account_snapshot(rdf, account)

        result = materialize(rdf, profile=reasoning_profile)
        if result.error:
            errors.append(result.error)

        validation = validate_graph(result.enriched_graph, mode=mode)
        if validation.error:
            errors.append(validation.error)

        ui_payload = rdf_to_ui_payload(rdf, inferred_triples=result.inferred_triples)
        timings = {
            "rdf_build_ms": result.build_ms,
            "owl_materialize_ms": result.reason_ms,
            "shacl_validate_ms": validation.validate_ms,
        }
        return OntologyLayerResult(
            rdf=rdf,
            materialization=result,
            validation=validation,
            ui_payload=ui_payload,
            inferred_types=result.inferred_types,
            timings_ms=timings,
            errors=tuple(errors),
            mode=mode,
        )
    except Exception as exc:  # pragma: no cover - defensive, never fail open
        errors.append(f"ontology layer failed: {exc}")
        # Build a non-conforming validation report for live safety.
        safe_validation = ValidationReport(
            conforms=False,
            mode=mode,
            blocking=(mode == "live"),
            error=f"ontology layer failed: {exc}",
        )
        empty = RdfTradingGraph(cycle_id=cycle_id)
        empty_mat = MaterializationResult(
            enriched_graph=empty.merged_graph(),
            inferred_triples=(),
            asserted_count=0,
            inferred_count=0,
            schema_count=0,
            profile="none",
            error=f"ontology layer failed: {exc}",
        )
        return OntologyLayerResult(
            rdf=empty,
            materialization=empty_mat,
            validation=safe_validation,
            errors=tuple(errors),
            mode=mode,
        )

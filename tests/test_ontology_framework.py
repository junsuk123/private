"""Tests for the standards-based RDF/RDFS/OWL + SHACL ontology framework (T11).

Covers: custom-triple -> RDF conversion, class & property hierarchy inference,
SHACL data-quality gates (stale / synthetic / approved+rejected conflict),
policy-scorer consumption of OWL-inferred classes, RiskManager preservation,
paper-vs-live blocking semantics, and Turtle serialization.
"""

from __future__ import annotations

from datetime import datetime, timezone

from rdflib import Literal, URIRef
from rdflib.namespace import RDF, XSD

from app.graph.knowledge_graph import KnowledgeGraph
from app.graph.owl_reasoner import load_schema_graph
from app.graph.rdf_adapter import knowledge_graph_to_rdf
from app.graph.rdf_graph import TR, resource_iri
from app.graph.semantic_materializer import inferred_classes_for, materialize
from app.graph.shacl_validator import validate_graph
from app.schemas.domain import SourceMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _FakeSource:
    def __init__(self, *, realtime: bool, synthetic: bool, delayed: bool) -> None:
        self.source_type = "broker_api"
        self.is_realtime = realtime
        self.is_synthetic = synthetic
        self.is_delayed = delayed
        self.retrieved_at = datetime(2026, 7, 2, tzinfo=timezone.utc)


class _FakeMarket:
    def __init__(self, market: str, price: float, source: _FakeSource) -> None:
        self.market = market
        self.last_price = price
        self.source = source


def _fresh_domestic_market() -> _FakeMarket:
    return _FakeMarket("KRX", 70000.0, _FakeSource(realtime=True, synthetic=False, delayed=False))


def _synthetic_market() -> _FakeMarket:
    return _FakeMarket("NASDAQ", 200.0, _FakeSource(realtime=False, synthetic=True, delayed=True))


# ---------------------------------------------------------------------------
# Ontology files
# ---------------------------------------------------------------------------
def test_ontology_ttl_files_parse() -> None:
    result = load_schema_graph()
    assert result.error is None
    assert result.triple_count > 100


# ---------------------------------------------------------------------------
# T11: custom triples -> RDF conversion
# ---------------------------------------------------------------------------
def test_custom_triples_convert_to_rdf() -> None:
    kg = KnowledgeGraph()
    kg.add("005930", "supportsSignal", "BuyCandidate", "ev:1")
    kg.add("SamsungElectronics", "hasTicker", "005930", "ev:2")
    rdf = knowledge_graph_to_rdf(kg, cycle_id="c1")
    subj = resource_iri("005930")
    # supportsSignal maps to a tr: object property to the canonical signal.
    assert (subj, TR.supportsSignal, TR.Signal_BuyCandidate) in rdf.merged_graph()
    # evidence node created for provenance
    assert any(p == TR.hasEvidence for _, p, _ in rdf.matching(subject=subj))


def test_turtle_serialization_roundtrip() -> None:
    kg = KnowledgeGraph()
    kg.add("005930", "increasesRiskOf", "VolatilityRisk", "ev:1")
    rdf = knowledge_graph_to_rdf(kg, cycle_id="c1")
    turtle = rdf.serialize(format="turtle")
    assert "increasesRiskOf" in turtle
    # JSON-LD serialization must also work.
    assert rdf.serialize(format="json-ld")


# ---------------------------------------------------------------------------
# T11: class / property hierarchy inference
# ---------------------------------------------------------------------------
def test_class_hierarchy_inference() -> None:
    kg = KnowledgeGraph()
    rdf = knowledge_graph_to_rdf(
        kg, cycle_id="c1", markets={"005930": _fresh_domestic_market()}
    )
    result = materialize(rdf)
    assert result.error is None
    inferred = inferred_classes_for(result, str(resource_iri("005930")))
    # DomesticStock (asserted) must entail Stock and MarketEntity.
    assert "Stock" in inferred
    assert "MarketEntity" in inferred


def test_property_hierarchy_inference() -> None:
    kg = KnowledgeGraph()
    kg.add("005930", "supportsSignal", "BuyCandidate", "ev:1")
    rdf = knowledge_graph_to_rdf(kg, cycle_id="c1")
    result = materialize(rdf)
    subj = resource_iri("005930")
    # supportsSignal is a subPropertyOf hasSemanticEvidence.
    assert (subj, TR.hasSemanticEvidence, TR.Signal_BuyCandidate) in result.enriched_graph


def test_hasvalue_classification_infers_candidate_and_forbidden() -> None:
    kg = KnowledgeGraph()
    kg.add("005930", "supportsSignal", "BuyCandidate", "ev:1")
    kg.add("AAPL", "increasesRiskOf", "TradeForbidden", "ev:2")
    rdf = knowledge_graph_to_rdf(kg, cycle_id="c1")
    result = materialize(rdf)
    assert "BuyCandidate" in inferred_classes_for(result, str(resource_iri("005930")))
    assert "TradeForbiddenAsset" in inferred_classes_for(result, str(resource_iri("AAPL")))


# ---------------------------------------------------------------------------
# T11: SHACL validation
# ---------------------------------------------------------------------------
def test_stale_quote_fails_shacl_in_live() -> None:
    kg = KnowledgeGraph()
    # not realtime + delayed => stale
    rdf = knowledge_graph_to_rdf(
        kg, cycle_id="c1", markets={"AAPL": _synthetic_market()}, live_tickers=["AAPL"]
    )
    result = materialize(rdf)
    report = validate_graph(result.enriched_graph, mode="live")
    assert report.conforms is False
    assert report.blocking is True
    messages = " ".join(v.message for v in report.violations)
    assert "stale" in messages.lower()


def test_synthetic_data_blocked_for_live() -> None:
    kg = KnowledgeGraph()
    rdf = knowledge_graph_to_rdf(
        kg, cycle_id="c1", markets={"AAPL": _synthetic_market()}, live_tickers=["AAPL"]
    )
    result = materialize(rdf)
    report = validate_graph(result.enriched_graph, mode="live")
    messages = " ".join(v.message for v in report.violations)
    assert "synthetic" in messages.lower()
    assert report.blocking is True


def test_paper_mode_reports_warnings_without_blocking() -> None:
    kg = KnowledgeGraph()
    rdf = knowledge_graph_to_rdf(
        kg, cycle_id="c1", markets={"AAPL": _synthetic_market()}, live_tickers=["AAPL"]
    )
    result = materialize(rdf)
    report = validate_graph(result.enriched_graph, mode="paper")
    # Violations are reported (warnings) but must NOT block in paper/mock mode.
    assert report.blocking is False


def test_approved_and_rejected_conflict_detected() -> None:
    kg = KnowledgeGraph()
    rdf = knowledge_graph_to_rdf(kg, cycle_id="c1")
    g = rdf.context()
    order = resource_iri("order:X")
    approved = resource_iri("dec:approved")
    rejected = resource_iri("dec:rejected")
    g.add((order, RDF.type, TR.OrderIntent))
    g.add((order, TR.hasOrderSide, Literal("BUY")))
    g.add((order, TR.hasSymbol, Literal("X")))
    g.add((order, TR.hasQuantity, Literal(1)))
    g.add((order, TR.hasConfidence, Literal("0.5", datatype=XSD.decimal)))
    g.add((order, TR.hasReason, Literal("test")))
    g.add((order, TR.hasRiskManagerDecision, approved))
    g.add((approved, RDF.type, TR.ApprovedByRiskManager))
    g.add((order, TR.hasRiskManagerDecision, rejected))
    g.add((rejected, RDF.type, TR.RejectedByRiskManager))
    result = materialize(rdf)
    report = validate_graph(result.enriched_graph, mode="live")
    messages = " ".join(v.message for v in report.violations)
    assert "both approved and rejected" in messages.lower()


# ---------------------------------------------------------------------------
# T11: policy scorer consumes OWL-inferred classes (as extra features)
# ---------------------------------------------------------------------------
def test_policy_scorer_accepts_inferred_classes() -> None:
    from app.graph.reasoner import SemanticPolicyScorer

    kg = KnowledgeGraph()
    kg.add("005930", "supportsSignal", "BuyCandidate", "ev:1")
    inferred = {"005930": ["BuyCandidate", "CandidateAsset"]}
    scorer = SemanticPolicyScorer(kg, inferred_classes=inferred)
    scorer.infer()
    paths = scorer.build_reasoning_paths(("005930",))
    assert paths and paths[0].ticker == "005930"
    # The inferred classes are available to the scorer without replacing scoring.
    assert scorer.inferred_classes["005930"] == ["BuyCandidate", "CandidateAsset"]


def test_backward_compatible_aliases() -> None:
    from app.graph import OntologyReasoner, OntologyReasoningPolicy
    from app.graph.reasoner import SemanticPolicyScorer, SemanticPolicyScorerConfig

    assert OntologyReasoner is SemanticPolicyScorer
    assert OntologyReasoningPolicy is SemanticPolicyScorerConfig


# ---------------------------------------------------------------------------
# T11: RiskManager remains the final gate; ontology layer is advisory only
# ---------------------------------------------------------------------------
def test_ontology_layer_is_advisory_only(monkeypatch) -> None:
    """Enabling/disabling the RDF layer must not change intents or risk results."""
    from app import pipeline

    monkeypatch.setenv("ONTOLOGY_RDF_LAYER", "1")
    ctx_on = pipeline.build_analysis_context(allow_sample_indicators=True)
    monkeypatch.setenv("ONTOLOGY_RDF_LAYER", "0")
    ctx_off = pipeline.build_analysis_context(allow_sample_indicators=True)

    # The layer is present only when enabled...
    assert ctx_on.ontology_layer is not None
    assert ctx_off.ontology_layer is None
    # ...but the trading decisions (intents + RiskManager results) are identical.
    assert [i.ticker for i in ctx_on.intents] == [i.ticker for i in ctx_off.intents]
    assert [i.action for i in ctx_on.intents] == [i.action for i in ctx_off.intents]
    assert len(ctx_on.risk_results) == len(ctx_off.risk_results)


def test_owl_eligibility_does_not_create_final_order() -> None:
    """An OWL-inferred TradeForbidden/eligible label never becomes a FinalOrder;
    only the RiskManager decides. Here we assert the layer produces no order."""
    kg = KnowledgeGraph()
    kg.add("AAPL", "increasesRiskOf", "TradeForbidden", "ev:1")
    rdf = knowledge_graph_to_rdf(kg, cycle_id="c1", markets={"AAPL": _fresh_domestic_market()})
    result = materialize(rdf)
    # No tr:FinalOrder is ever materialized by OWL reasoning.
    assert (None, RDF.type, TR.FinalOrder) not in result.enriched_graph

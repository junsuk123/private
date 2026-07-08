from __future__ import annotations

from datetime import datetime, timezone

from rdflib import Graph, URIRef

from app.graph.knowledge_graph import KnowledgeGraph
from app.graph.rdf_graph import RdfTradingGraph, TR, resource_iri
from app.graph.reasoner import SemanticPolicyScorer
from app.graph.shacl_validator import reset_shapes_cache, validate_graph
from app.graph.technical_evidence import (
    add_technical_evidence_to_graph,
    attach_technical_evidence_rdf,
    composite_to_evidence,
    exit_deterioration_to_evidence,
)
from app.technical import reason_codes as rc
from app.technical.prediction import TechnicalPredictionEngine, PredictionConfig
from app.technical.signals import CompositeTechnicalSignalEngine
from tests.test_technical_signals import trend_up_features

CORE = "src/app/ontology/trading_core.ttl"
SHAPES = "src/app/ontology/trading_shapes.ttl"


class TestOntologyParses:
    def test_core_parses_with_technical_vocab(self):
        g = Graph()
        g.parse(CORE, format="turtle")
        assert (TR.TechnicalSignal, None, None) in [(s, None, None) for s, _, _ in g if s == TR.TechnicalSignal] or \
            any(s == TR.TechnicalSignal for s in g.subjects())
        # Key new terms exist as subjects.
        subjects = set(g.subjects())
        for term in ("TechnicalSignal", "MomentumSignal", "BreakoutSignal", "TrendRegime",
                     "expectedEdgeBps", "technicalConfidence", "generatedByMethodology"):
            assert TR[term] in subjects, f"missing {term}"

    def test_shapes_parse(self):
        g = Graph()
        g.parse(SHAPES, format="turtle")
        assert TR.LiveTechnicalEvidenceShape in set(g.subjects())


class TestEvidenceMapping:
    def _composite(self, **over):
        return CompositeTechnicalSignalEngine().evaluate(trend_up_features(**over))

    def test_buy_maps_to_support(self):
        ev = composite_to_evidence(self._composite())
        preds = {(p, o) for _, p, o, _ in ev}
        assert ("supportsSignal", "BuyCandidate") in preds
        # selected methodology object present
        assert any(p == "supportsSignal" and o in ("VolumeConfirmedMomentum", "TechnicalBreakoutBuy", "RSIOversold", "OrderFlowPriceConfirmation") for _, p, o, _ in ev)

    def test_blocked_maps_to_risk(self):
        ev = composite_to_evidence(self._composite(realized_volatility=0.05))
        preds = {(p, o) for _, p, o, _ in ev}
        assert ("increasesRiskOf", "VolatilityRisk") in preds
        assert ("contradictsSignal", "BuyCandidate") in preds

    def test_low_liquidity_maps_to_risk(self):
        ev = composite_to_evidence(self._composite(liquidity_score=0.05))
        preds = {(p, o) for _, p, o, _ in ev}
        assert ("increasesRiskOf", "LiquidityRisk") in preds

    def test_exit_deterioration_mapping(self):
        ev = exit_deterioration_to_evidence("005930", (rc.VWAP_BREAKDOWN, rc.TECHNICAL_EXIT_DETERIORATION))
        preds = {(p, o) for _, p, o, _ in ev}
        assert ("contradictsSignal", "OrderFlowPriceDivergence") in preds
        assert ("increasesRiskOf", "OrderFlowDistributionRisk") in preds

    def test_never_emits_final_order(self):
        for comp in (self._composite(), self._composite(realized_volatility=0.05)):
            for _, predicate, obj, _ in composite_to_evidence(comp):
                assert "FinalOrder" not in obj
                assert predicate in ("supportsSignal", "contradictsSignal", "increasesRiskOf")


class TestScorerIntegration:
    def test_technical_support_raises_confidence(self):
        composite = CompositeTechnicalSignalEngine().evaluate(trend_up_features())

        baseline_graph = KnowledgeGraph()
        baseline_graph.add("005930", "supportsSignal", "FreshBrokerQuote", "seed")
        base = SemanticPolicyScorer(baseline_graph).build_reasoning_paths(("005930",))[0]

        graph = KnowledgeGraph()
        graph.add("005930", "supportsSignal", "FreshBrokerQuote", "seed")
        composite = CompositeTechnicalSignalEngine().evaluate(trend_up_features(symbol="005930"))
        added = add_technical_evidence_to_graph(graph, composite)
        assert added >= 2
        boosted = SemanticPolicyScorer(graph).build_reasoning_paths(("005930",))[0]
        assert boosted.confidence > base.confidence

    def test_scorer_produces_no_order_object(self):
        graph = KnowledgeGraph()
        composite = CompositeTechnicalSignalEngine().evaluate(trend_up_features(symbol="005930"))
        add_technical_evidence_to_graph(graph, composite)
        scorer = SemanticPolicyScorer(graph)
        scorer.infer()
        # The scorer never asserts a FinalOrder / RiskManager approval.
        objs = {t.object for t in graph.triples()}
        assert "FinalOrder" not in objs
        assert "ApprovedByRiskManager" not in objs


class TestRdfAndShacl:
    def _rdf_with_prediction(self, *, live: bool, drop_edge: bool = False):
        features = trend_up_features(symbol="005930")
        engine = CompositeTechnicalSignalEngine()
        composite = engine.evaluate(features)
        prediction = TechnicalPredictionEngine(config=PredictionConfig(min_confidence=0.3)).predict(features)
        rdf = RdfTradingGraph(cycle_id="test")
        attach_technical_evidence_rdf(
            rdf, composite, prediction=prediction, live=live,
            decision_time=None if drop_edge else datetime(2026, 7, 9, tzinfo=timezone.utc),
        )
        return rdf, composite

    def test_rdf_projection_has_technical_signal(self):
        rdf, composite = self._rdf_with_prediction(live=False)
        merged = rdf.merged_graph()
        # A TechnicalSignal individual exists with methodology + edge.
        sig_nodes = list(merged.subjects(predicate=TR.hasMethodologyName))
        assert sig_nodes
        assert any(True for _ in merged.triples((None, TR.expectedEdgeBps, None)))
        assert any(True for _ in merged.triples((None, TR.hasRegime, None)))

    def test_shacl_live_candidate_conforms_when_complete(self):
        reset_shapes_cache()
        rdf, _ = self._rdf_with_prediction(live=True)
        report = validate_graph(rdf.merged_graph(), mode="live")
        assert report.conforms, report.as_dict()

    def test_shacl_live_candidate_blocks_when_missing_timestamp(self):
        reset_shapes_cache()
        rdf, _ = self._rdf_with_prediction(live=True, drop_edge=True)  # no hasAsOfTime
        report = validate_graph(rdf.merged_graph(), mode="live")
        assert not report.conforms
        assert report.blocking
        assert any("timestamp" in v.message.lower() or "hasAsOfTime" in (v.path or "") for v in report.violations)

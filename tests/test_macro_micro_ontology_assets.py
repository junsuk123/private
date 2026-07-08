from __future__ import annotations

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from app.graph.owl_reasoner import load_schema_graph
from app.graph.rdf_graph import TR, resource_iri
from app.graph.shacl_validator import reset_shapes_cache, validate_graph

MACRO = "src/app/ontology/macro_market_ontology.ttl"
MICRO = "src/app/ontology/micro_symbol_ontology.ttl"


class TestParsing:
    def test_macro_parses_with_classes(self):
        g = Graph()
        g.parse(MACRO, format="turtle")
        subjects = set(g.subjects())
        for term in ("MarketContext", "MarketRegime", "TrendUpMarket", "NoTradeMarket",
                     "SectorStrength", "MacroCandidateSymbol", "MacroRiskState",
                     "AllowedMicroStrategy", "BlockedMicroStrategy", "MacroReasoningResult",
                     "hasMarketRegime", "selectsCandidateSymbol", "allowsMicroStrategy"):
            assert TR[term] in subjects, f"macro missing {term}"

    def test_micro_parses_with_classes(self):
        g = Graph()
        g.parse(MICRO, format="turtle")
        subjects = set(g.subjects())
        for term in ("SymbolContext", "MicroRegime", "MomentumCandidate", "ExitDeterioration",
                     "NoTradeSymbol", "ExpectedEntryPrice", "ExpectedExitPrice", "ExpectedNetReturn",
                     "ExecutionQuality", "MicroReasoningResult", "hasMicroRegime",
                     "hasExpectedNetReturnBps", "isBlockedByMacroStrategy"):
            assert TR[term] in subjects, f"micro missing {term}"

    def test_loader_opt_in_macro_micro(self):
        base = load_schema_graph()
        assert base.error is None
        extended = load_schema_graph(include_macro_micro=True)
        assert extended.error is None
        # Opt-in adds triples (macro + micro vocabulary) without breaking load.
        assert extended.triple_count > base.triple_count

    def test_default_load_unchanged(self):
        # Default path must still load core+rules only, with no error.
        result = load_schema_graph()
        assert result.error is None
        assert result.triple_count > 0


class TestShaclMacroMicro:
    def _macro_graph(self, *, complete: bool):
        g = Graph()
        node = resource_iri("macro:test")
        g.add((node, RDF.type, TR.MacroReasoningResult))
        g.add((node, TR.hasMarketRegimeName, Literal("TREND_UP")))
        g.add((node, TR.hasMacroConfidence, Literal("0.7", datatype=XSD.decimal)))
        if complete:
            g.add((node, TR.hasMacroRiskLevelName, Literal("LOW")))
        return g

    def _micro_graph(self, *, buy_with_exit: bool):
        g = Graph()
        node = resource_iri("micro:test")
        g.add((node, RDF.type, TR.MicroReasoningResult))
        g.add((node, TR.hasMicroRegimeName, Literal("MOMENTUM_CANDIDATE")))
        g.add((node, TR.hasMicroConfidence, Literal("0.7", datatype=XSD.decimal)))
        g.add((node, TR.hasExecutionQualityName, Literal("GOOD")))
        g.add((node, TR.hasEntrySignalName, Literal("BUY_CANDIDATE")))
        if buy_with_exit:
            g.add((node, TR.hasExpectedExitPrice, Literal("101.0", datatype=XSD.decimal)))
        return g

    def test_macro_result_conforms_when_complete(self):
        reset_shapes_cache()
        report = validate_graph(self._macro_graph(complete=True), mode="live")
        assert report.conforms, report.as_dict()

    def test_macro_result_blocks_when_missing_risk_level(self):
        reset_shapes_cache()
        report = validate_graph(self._macro_graph(complete=False), mode="live")
        assert not report.conforms and report.blocking

    def test_micro_buy_requires_expected_exit(self):
        reset_shapes_cache()
        ok = validate_graph(self._micro_graph(buy_with_exit=True), mode="live")
        assert ok.conforms, ok.as_dict()
        bad = validate_graph(self._micro_graph(buy_with_exit=False), mode="live")
        assert not bad.conforms and bad.blocking

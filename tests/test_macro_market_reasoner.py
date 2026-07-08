from __future__ import annotations

from datetime import datetime, timezone

from app.graph.macro_micro_common import MacroRiskLevel, MarketRegime
from app.graph.macro_reasoner import MacroMarketReasoner, MacroReasonerConfig, MacroReasoningInput
from app.graph.rdf_adapter import attach_macro_result_rdf
from app.graph.rdf_graph import RdfTradingGraph, TR
from app.graph.shacl_validator import reset_shapes_cache, validate_graph


def _now():
    return datetime(2026, 7, 9, tzinfo=timezone.utc)


def _reasoner():
    return MacroMarketReasoner(MacroReasonerConfig())


def _inp(**over):
    base = dict(
        timestamp=_now(),
        index_snapshots={"KOSPI": {"trend": 0.004}},
        sector_snapshots={"tech": {"strength": 0.8, "volume_change": 0.3}, "energy": {"strength": -0.5, "volume_change": -0.2}},
        market_breadth=0.6,
        market_volatility=0.005,
        candidate_universe=("005930", "000660", "035720"),
    )
    base.update(over)
    return MacroReasoningInput(**base)


class TestRegimeClassification:
    def test_insufficient_data_no_trade(self):
        r = _reasoner().reason(MacroReasoningInput(timestamp=_now()))
        assert r.market_regime == MarketRegime.NO_TRADE_MARKET
        assert r.macro_risk_level == MacroRiskLevel.BLOCK_BUY
        assert r.blocks_buy
        assert r.candidate_symbols == ()

    def test_trend_up(self):
        r = _reasoner().reason(_inp())
        assert r.market_regime == MarketRegime.TREND_UP
        assert not r.blocks_buy
        assert "momentum" in r.allowed_micro_strategies
        assert "breakout" in r.allowed_micro_strategies

    def test_trend_down(self):
        r = _reasoner().reason(_inp(index_snapshots={"KOSPI": {"trend": -0.004}}, market_breadth=0.4))
        assert r.market_regime == MarketRegime.TREND_DOWN
        assert "sell" in r.allowed_micro_strategies
        assert "weak_breakout_buy" in r.blocked_micro_strategies

    def test_high_volatility_blocks_buy(self):
        r = _reasoner().reason(_inp(market_volatility=0.05))
        assert r.market_regime == MarketRegime.HIGH_VOLATILITY_RISK
        assert r.macro_risk_level == MacroRiskLevel.BLOCK_BUY
        assert r.candidate_symbols == ()  # blocked-buy selects no new candidates

    def test_news_shock_blocks_buy(self):
        r = _reasoner().reason(_inp(macro_news_evidence=({"severity": 0.9},)))
        assert r.market_regime == MarketRegime.NEWS_SHOCK
        assert r.blocks_buy

    def test_range_bound(self):
        r = _reasoner().reason(_inp(index_snapshots={"KOSPI": {"trend": 0.0}}))
        assert r.market_regime == MarketRegime.RANGE_BOUND
        assert "mean_reversion" in r.allowed_micro_strategies


class TestCandidateSelection:
    def test_candidates_selected_and_capped(self):
        universe = tuple(f"SYM{i}" for i in range(50))
        r = MacroMarketReasoner(MacroReasonerConfig(candidate_limit=10)).reason(_inp(candidate_universe=universe))
        assert 0 < len(r.candidate_symbols) <= 10

    def test_strong_sector_preference(self):
        r = _reasoner().reason(_inp(
            candidate_universe=("A", "B"),
            provenance={"sector_of": {"A": "energy", "B": "tech"}},
        ))
        # tech is the strong sector -> B preferred first.
        assert r.candidate_symbols[0] == "B"

    def test_always_has_reason_and_explanation(self):
        r = _reasoner().reason(_inp())
        assert r.reason_codes
        assert r.explanation_paths


class TestRdfProjectionAdvisory:
    def test_projection_conforms_to_shacl(self):
        reset_shapes_cache()
        rdf = RdfTradingGraph(cycle_id="macro-test")
        attach_macro_result_rdf(rdf, _reasoner().reason(_inp()))
        report = validate_graph(rdf.merged_graph(), mode="live")
        assert report.conforms, report.as_dict()

    def test_macro_never_asserts_final_order(self):
        rdf = RdfTradingGraph(cycle_id="macro-test")
        attach_macro_result_rdf(rdf, _reasoner().reason(_inp()))
        merged = rdf.merged_graph()
        assert (None, None, TR.FinalOrder) not in merged  # no FinalOrder typing
        result = _reasoner().reason(_inp())
        assert not hasattr(result, "final_order")

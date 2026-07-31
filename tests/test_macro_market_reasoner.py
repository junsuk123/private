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

    def test_unclassifiable_high_volatility_blocks_buy(self):
        # No index trend and no breadth: volatility is high but nothing says WHY,
        # so the conservative single-state behaviour is the honest answer.
        r = _reasoner().reason(
            _inp(market_volatility=0.05, index_snapshots={}, market_breadth=None)
        )
        assert r.market_regime == MarketRegime.HIGH_VOLATILITY_RISK
        assert r.macro_risk_level == MacroRiskLevel.BLOCK_BUY
        assert r.blocks_buy
        assert r.candidate_symbols == ()  # blocked-buy selects no new candidates

    def test_high_volatility_subregime_classification_is_not_a_blanket_ban(self):
        # High volatility is four different market states, not one. Only the
        # dislocated (and unclassifiable) case bans every new buy; the others
        # narrow the allow-list instead.
        dislocated = _reasoner().reason(
            _inp(
                market_volatility=0.05,
                index_snapshots={"KOSPI": {"trend": -0.02}},
                market_breadth=0.1,
                spread_percentile=0.95,
            )
        )
        assert dislocated.market_regime == MarketRegime.HIGH_VOL_DISLOCATED
        assert dislocated.blocks_buy
        assert dislocated.candidate_symbols == ()

        trending = _reasoner().reason(
            _inp(
                market_volatility=0.05,
                index_snapshots={"KOSPI": {"trend": -0.02}},
                market_breadth=0.2,
                spread_percentile=0.4,
                average_market_correlation=0.5,
            )
        )
        assert trending.market_regime == MarketRegime.HIGH_VOL_TRENDING
        assert not trending.blocks_buy
        # A weak tape can still support stock-specific strength, never a dip buy.
        assert "relative_strength" in trending.allowed_micro_strategies
        assert "mean_reversion" in trending.blocked_micro_strategies

        mean_reverting = _reasoner().reason(
            _inp(
                market_volatility=0.05,
                index_snapshots={"KOSPI": {"trend": 0.0005}},
                market_breadth=0.5,
                spread_percentile=0.4,
                average_market_correlation=0.5,
            )
        )
        assert mean_reverting.market_regime == MarketRegime.HIGH_VOL_MEAN_REVERTING
        assert "vwap_reversion" in mean_reverting.allowed_micro_strategies
        assert "momentum" in mean_reverting.blocked_micro_strategies

        recovery = _reasoner().reason(
            _inp(
                market_volatility=0.05,
                index_snapshots={"KOSPI": {"trend": 0.01}},
                market_breadth=0.7,
                breadth_momentum=0.2,
                spread_percentile=0.4,
                average_market_correlation=0.5,
            )
        )
        assert recovery.market_regime == MarketRegime.HIGH_VOL_RECOVERY
        assert not recovery.blocks_buy
        assert recovery.candidate_symbols  # exploratory long risk is permitted

    def test_detected_change_point_blocks_entry_in_high_volatility(self):
        r = _reasoner().reason(
            _inp(
                market_volatility=0.05,
                index_snapshots={"KOSPI": {"trend": 0.0005}},
                market_breadth=0.5,
                change_point_probability=0.9,
            )
        )
        assert r.market_regime == MarketRegime.HIGH_VOL_DISLOCATED
        assert r.blocks_buy
        assert "MACRO_CHANGE_POINT_BLOCKS_ENTRY" in r.reason_codes

    def test_high_volatility_single_state_mode_is_restorable(self):
        reasoner = MacroMarketReasoner(
            MacroReasonerConfig(classify_high_volatility_subregimes=False)
        )
        r = reasoner.reason(
            _inp(
                market_volatility=0.05,
                index_snapshots={"KOSPI": {"trend": 0.01}},
                market_breadth=0.7,
            )
        )
        assert r.market_regime == MarketRegime.HIGH_VOLATILITY_RISK
        assert r.blocks_buy

    def test_within_sector_rank_uses_residual_returns_not_global_rank(self):
        r = _reasoner().reason(
            _inp(
                provenance={
                    "sector_of": {"005930": "semi", "000660": "semi", "035720": "internet"}
                },
                symbol_residual_returns={"005930": 0.004, "000660": 0.001, "035720": 0.02},
                symbol_long_residual_returns={"005930": 0.006, "000660": 0.002},
            )
        )
        table = r.sector_rank_table
        # Ranked WITHIN the sector by residual strength, not by any global ordering.
        assert table.rank_for("005930") == (1, 2)
        assert table.rank_for("000660") == (2, 2)
        # A sector with a single tracked name cannot answer "strongest in sector",
        # so it returns None and the consuming algorithm fails closed.
        assert table.rank_for("035720") is None
        assert table.long_residual_for("005930") == 0.006

    def test_news_shock_blocks_buy(self):
        r = _reasoner().reason(
            _inp(macro_news_evidence=({"severity": 0.9}, {"severity": 0.85}))
        )
        assert r.market_regime == MarketRegime.NEWS_SHOCK
        assert r.blocks_buy

    def test_single_headline_does_not_declare_a_market_wide_shock(self):
        """Regression: one mislabelled headline blocked every buy for a full TTL.

        A market-wide shock must be corroborated by independent items.
        """
        r = _reasoner().reason(_inp(macro_news_evidence=({"severity": 0.95},)))
        assert r.market_regime != MarketRegime.NEWS_SHOCK
        assert not r.blocks_buy

    def test_many_weak_items_do_not_declare_a_shock(self):
        r = _reasoner().reason(
            _inp(macro_news_evidence=tuple({"severity": 0.3} for _ in range(6)))
        )
        assert r.market_regime != MarketRegime.NEWS_SHOCK
        assert not r.blocks_buy

    def test_corroboration_requirement_is_configurable(self):
        strict = MacroMarketReasoner(MacroReasonerConfig(news_shock_minimum_events=1))
        r = strict.reason(_inp(macro_news_evidence=({"severity": 0.95},)))
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

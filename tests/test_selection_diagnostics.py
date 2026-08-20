from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.technical.selection_diagnostics import (
    SelectionDiagnosticsCollector,
    SelectionStage,
    collector_from_algorithm_evaluations,
    merge_stage_counts,
)


def _collector() -> SelectionDiagnosticsCollector:
    return SelectionDiagnosticsCollector()


class TestFirstFailureWins:
    """A later layer must not rewrite history and blame itself."""

    def test_earliest_stop_is_kept(self):
        c = _collector()
        record = c.candidate("005930", "intraday_momentum", market="KR")
        record.mark(SelectionStage.GROSS_EDGE_NON_POSITIVE, rule_gross_bps=-3.0)
        record.mark(SelectionStage.ONTOLOGY_BLOCKED)
        assert record.stage is SelectionStage.GROSS_EDGE_NON_POSITIVE

    def test_numbers_still_accumulate_after_the_stop(self):
        # The stage freezes; the decomposition keeps filling in so the audit trail
        # is complete even for a candidate that died early.
        c = _collector()
        record = c.candidate("F", "breakout_volume", market="US")
        record.mark(SelectionStage.HORIZON_COST_UNVIABLE, all_in_cost_bps=70.0)
        record.mark(SelectionStage.FUSED_NET_NON_POSITIVE, fused_net_bps=-12.0)
        assert record.stage is SelectionStage.HORIZON_COST_UNVIABLE
        assert record.all_in_cost_bps == 70.0
        assert record.fused_net_bps == -12.0

    def test_selected_is_reachable_and_not_a_stop(self):
        c = _collector()
        record = c.candidate("005930", "vwap_mean_reversion", market="KR")
        record.mark(SelectionStage.SELECTED, fused_net_bps=30.0, required_net_bps=10.0)
        assert not record.stopped
        assert record.net_surplus_bps == 20.0


class TestFieldSafety:
    def test_unknown_field_is_rejected(self):
        c = _collector()
        record = c.candidate("005930", "s", market="KR")
        try:
            record.mark(SelectionStage.RAW_CANDIDATE, not_a_field=1.0)
        except AttributeError:
            return
        raise AssertionError("unknown diagnostic field must raise")

    def test_non_finite_and_none_are_ignored_not_stored_as_zero(self):
        c = _collector()
        record = c.candidate("005930", "s", market="KR")
        record.mark(
            SelectionStage.RAW_CANDIDATE,
            rule_gross_bps=float("nan"),
            all_in_cost_bps=None,
            model_net_bps=float("inf"),
        )
        assert record.rule_gross_bps is None
        assert record.all_in_cost_bps is None
        assert record.model_net_bps is None

    def test_existing_reason_codes_are_preserved_and_deduped(self):
        c = _collector()
        record = c.candidate("005930", "s", market="KR")
        record.mark(SelectionStage.RAW_CANDIDATE, reason_codes=["A", "B"])
        record.mark(SelectionStage.GROSS_EDGE_NON_POSITIVE, reason_codes=["B", "C"])
        assert record.reason_codes == ("A", "B", "C")

    def test_surplus_is_none_when_either_side_is_unmeasured(self):
        c = _collector()
        record = c.candidate("005930", "s", market="KR")
        record.mark(SelectionStage.FUSED_NET_NON_POSITIVE, fused_net_bps=-1.0)
        assert record.net_surplus_bps is None


class TestFunnel:
    def _mixed(self) -> SelectionDiagnosticsCollector:
        c = _collector()
        c.candidate("A", "s1", market="KR").mark(SelectionStage.STRATEGY_TRIGGER_FALSE)
        c.candidate("B", "s1", market="KR").mark(SelectionStage.GROSS_EDGE_NON_POSITIVE)
        c.candidate("C", "s2", market="US").mark(SelectionStage.HORIZON_COST_UNVIABLE)
        c.candidate("D", "s2", market="US").mark(SelectionStage.FUSED_NET_NON_POSITIVE)
        c.candidate("E", "s3", market="KR").mark(SelectionStage.SHADOW_ONLY)
        c.candidate("F", "s3", market="KR").mark(
            SelectionStage.SELECTED, fused_net_bps=25.0, required_net_bps=10.0
        )
        return c

    def test_funnel_is_monotonically_non_increasing(self):
        funnel = self._mixed().funnel()
        order = [
            "raw", "trigger", "gross_positive", "horizon_viable",
            "net_positive", "gate_passed", "live_authorized", "selected",
        ]
        values = [funnel[key] for key in order]
        assert values == sorted(values, reverse=True), values
        assert funnel["raw"] == 6
        assert funnel["selected"] == 1

    def test_shadow_only_is_not_counted_as_net_negative(self):
        # Deployment state and economics must stay separable.
        counts = self._mixed().stage_counts()
        assert counts["SHADOW_ONLY"] == 1
        assert counts["FUSED_NET_NON_POSITIVE"] == 1

    def test_grouping_by_market_and_strategy(self):
        grouped = self._mixed().by_market_and_strategy()
        assert grouped["US"]["s2"]["HORIZON_COST_UNVIABLE"] == 1
        assert grouped["KR"]["s3"]["SELECTED"] == 1

    def test_blocking_summary_reports_stages_not_raw_codes(self):
        summary = self._mixed().blocking_summary()
        assert "SELECTED" not in summary
        assert "STRATEGY_TRIGGER_FALSE" in summary

    def test_edge_decomposition_does_not_invent_zeros(self):
        c = _collector()
        c.candidate("A", "s1", market="KR").mark(
            SelectionStage.GROSS_EDGE_NON_POSITIVE, rule_gross_bps=-2.0
        )
        stats = c.edge_decomposition()["GROSS_EDGE_NON_POSITIVE"]
        assert stats["rule_gross_bps"] == -2.0
        # The model layer never ran for this candidate.
        assert stats["model_net_bps"] is None

    def test_as_dict_is_json_serialisable(self):
        import json

        json.dumps(self._mixed().as_dict())


class TestMerge:
    def test_merge_sums_and_ignores_malformed(self):
        merged = merge_stage_counts(
            [
                {"stage_counts": {"SELECTED": 1, "SHADOW_ONLY": 2}},
                {"stage_counts": {"SELECTED": 3}},
                {"stage_counts": {"SELECTED": "x"}},
                {"no_stage_counts": True},
                "not a mapping",
            ]
        )
        assert merged["SELECTED"] == 4
        assert merged["SHADOW_ONLY"] == 2
        assert merged["MACRO_BLOCKED"] == 0


class TestCycleSnapshotAdapter:
    def test_reconstructs_real_feature_and_trigger_failures(self):
        collector = collector_from_algorithm_evaluations(
            [
                {
                    "symbol": "INTC",
                    "strategy_id": "vwap_mean_reversion",
                    "triggered": False,
                    "expected_edge_bps": 0.0,
                    "horizon_seconds": 240,
                    "reason_codes": ["TICK_WINDOW_NOT_READY"],
                },
                {
                    "symbol": "SOFI",
                    "strategy_id": "bar_confirmed_vwap_recovery",
                    "triggered": False,
                    "expected_edge_bps": 0.0,
                    "reason_codes": ["BAR_VWAP_DISPLACEMENT_TOO_SMALL"],
                },
            ],
            session={"last_reason": "GNN_NOT_LIVE_AUTHORIZED"},
        )

        assert collector.records[0].stage is SelectionStage.FEATURE_UNAVAILABLE
        assert collector.records[1].stage is SelectionStage.STRATEGY_TRIGGER_FALSE
        assert collector.funnel()["raw"] == 2

    def test_positive_trigger_without_gnn_authority_is_reported_as_authorization(self):
        collector = collector_from_algorithm_evaluations(
            [
                {
                    "symbol": "INTC",
                    "strategy_id": "intraday_momentum",
                    "triggered": True,
                    "expected_edge_bps": 12.5,
                    "reason_codes": [],
                }
            ],
            session={"last_reason": "GNN_NOT_LIVE_AUTHORIZED"},
        )

        assert collector.records[0].stage is SelectionStage.LIVE_NOT_AUTHORIZED

    def test_negative_bandit_edge_is_profitability_not_macro_block(self):
        collector = collector_from_algorithm_evaluations(
            [{
                "symbol": "019010",
                "strategy_id": "bar_confirmed_vwap_recovery",
                "triggered": True,
                "expected_edge_bps": 138.0,
                "reason_codes": ["BAR_CONFIRMED_VWAP_RECOVERY"],
            }],
            session={
                "macro_regime": "RANGE_BOUND",
                "ontology_reason_codes": ["MACRO_RANGE_BOUND"],
                "last_reason": "BANDIT_NO_TRADE_NO_POSITIVE_CONSERVATIVE_EDGE",
                "bandit_evaluations": [{
                    "symbol": "019010",
                    "arm": "bar_confirmed_vwap_recovery",
                    "admissible": False,
                    "shadow_only": False,
                    "conservative_edge_bps": -121.193,
                    "posterior_mean_net_bps": -60.758,
                    "uncertainty_penalty_bps": 60.434,
                    "reason_codes": ["BANDIT_ARM_MEASURED_NEGATIVE_EDGE"],
                }],
            },
        )

        record = collector.records[0]
        assert record.stage is SelectionStage.PROFITABILITY_REJECTED
        assert record.market == "KR"
        assert record.regime == "RANGE_BOUND"
        assert record.fused_net_bps == -121.193
        assert "MACRO_RANGE_BOUND" in record.reason_codes

    def test_only_explicit_macro_block_is_classified_as_macro(self):
        collector = collector_from_algorithm_evaluations(
            [{
                "symbol": "INTC",
                "strategy_id": "intraday_momentum",
                "triggered": True,
                "expected_edge_bps": 12.5,
                "reason_codes": [],
            }],
            session={"ontology_reason_codes": ["MACRO_STRATEGY_BLOCKED"]},
        )

        assert collector.records[0].stage is SelectionStage.MACRO_BLOCKED

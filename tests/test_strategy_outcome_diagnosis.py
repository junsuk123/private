"""A strategy with no samples is either waiting for a market or for an engineer.

Eight of sixteen strategies could not be selected at all - the routing relation
map had no entry for them, so the ontology gate's ``compatible:{id}`` fact was
always false. Their counterfactual reports still said "no triggered samples",
which reads as weak performance and invites retiring a strategy that was never
evaluated. The two cases now carry different diagnoses.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.evaluation.stored_counterfactual import CounterfactualLabel
from app.models.strategy_utility.training import _label_outcome_summary
from app.routing.shadow_intelligence import COMPATIBILITY_UNAVAILABLE_REASONS


def _label(strategy_id: str, *, triggered: bool, filled: bool, net_bps: float = 0.0) -> CounterfactualLabel:
    as_of = datetime(2026, 8, 3, 5, 0, tzinfo=timezone.utc)
    return CounterfactualLabel(
        as_of=as_of,
        label_end=as_of + timedelta(minutes=10),
        features=tuple([0.0] * 28),
        symbol="005930",
        strategy_id=strategy_id,
        triggered=triggered,
        filled=filled,
        net_return_bps=net_bps,
        cost_bps=28.0,
        exit_reason="horizon",
    )


def test_unreachable_strategy_is_not_reported_as_weak_performance() -> None:
    unreachable = "residual_relative_strength"
    assert unreachable in COMPATIBILITY_UNAVAILABLE_REASONS

    summary = _label_outcome_summary((_label(unreachable, triggered=False, filled=False),))

    diagnosis = summary[unreachable]["performance_diagnosis"]
    assert str(diagnosis).startswith("STRUCTURALLY_UNREACHABLE:")
    # The reason travels with it, so the fix is identifiable without reading code.
    assert "PEER_RESIDUALS" in str(diagnosis)


def test_reachable_strategy_with_no_trigger_still_reports_no_samples() -> None:
    reachable = "intraday_momentum"
    assert reachable not in COMPATIBILITY_UNAVAILABLE_REASONS

    summary = _label_outcome_summary((_label(reachable, triggered=False, filled=False),))

    # This one really is waiting for its conditions to occur.
    assert summary[reachable]["performance_diagnosis"] == "NO_TRIGGERED_SAMPLES"


def test_cost_exceeding_edge_keeps_its_own_diagnosis() -> None:
    # 25 fills, positive gross (net + cost = +12bp), negative net: the
    # gap_context pathology, which must stay distinguishable from both
    # "no samples" and "no gross edge".
    rows = tuple(
        _label("gap_context", triggered=True, filled=True, net_bps=-16.0)
        for _ in range(25)
    )

    summary = _label_outcome_summary(rows)

    assert summary["gap_context"]["performance_diagnosis"] == "EXECUTION_COST_EXCEEDS_GROSS_EDGE"

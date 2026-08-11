"""Selector regret: how much better the best available strategy would have been.

    regret_t = best_available_strategy_outcome_t - selected_strategy_outcome_t

This is the metric that separates the two failures the problem statement keeps conflating.
A strategy with no edge makes both the selector and the strategy look bad; regret does not:

* every alternative loses and the selector picked the least-bad one -> regret ~0, the
  *strategies* are the problem;
* an alternative made money and the selector picked a loser -> regret large, the
  *selector* is the problem.

Two properties that make the number honest
------------------------------------------
**NO_TRADE is a competitor, not an absence.** When the selector declined, its realised
outcome is 0.0 (zero exposure, and cost is only paid on a trade). Regret against a group
where every alternative lost is then NEGATIVE — the selector did better than any trade —
and that has to be representable, otherwise declining can never be rewarded.

**Contexts, not rows.** Alternatives within one context are the same price path cut by
different barriers, so regret is computed per context and then averaged over contexts. A
row-level average would weight a context with nine alternatives nine times as heavily as
one with a single alternative.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["ContextRegret", "RegretSummary", "compute_context_regret", "summarize_regret"]

#: Outcome assigned to a NO_TRADE decision. Zero, not the negative of a cost: declining
#: means no round trip was paid.
NO_TRADE_OUTCOME_BPS = 0.0


@dataclass(frozen=True)
class ContextRegret:
    """Regret for one context, with the counterfactual that produced it."""

    context_id: str
    symbol: str
    decision: str
    selected_strategy: str | None
    selected_outcome_bps: float
    best_strategy: str | None
    best_outcome_bps: float
    #: How many alternatives had a resolvable outcome. Regret from a group of one is
    #: reported but is weak evidence, and the field is what lets a consumer say so.
    alternative_count: int
    #: ``True`` when the selected strategy's number came from a real fill rather than a
    #: simulation. Mixed groups are the normal case (live selected, shadow alternatives).
    selected_from_live: bool
    predicted_best_strategy: str | None = None
    reason_codes: tuple[str, ...] = ()

    @property
    def regret_bps(self) -> float:
        return self.best_outcome_bps - self.selected_outcome_bps

    @property
    def top1_hit(self) -> bool:
        """Did the selector pick the strategy that actually turned out best?

        A NO_TRADE decision counts as a hit when nothing beat doing nothing.
        """
        if self.selected_strategy is None:
            return self.best_outcome_bps <= NO_TRADE_OUTCOME_BPS
        return self.selected_strategy == self.best_strategy

    def as_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "symbol": self.symbol,
            "decision": self.decision,
            "selected_strategy": self.selected_strategy,
            "selected_outcome_bps": round(self.selected_outcome_bps, 3),
            "best_strategy": self.best_strategy,
            "best_outcome_bps": round(self.best_outcome_bps, 3),
            "regret_bps": round(self.regret_bps, 3),
            "top1_hit": self.top1_hit,
            "alternative_count": self.alternative_count,
            "selected_from_live": self.selected_from_live,
            "predicted_best_strategy": self.predicted_best_strategy,
            "reason_codes": list(self.reason_codes),
        }


def compute_context_regret(group: Any, *, top_k: int = 3) -> ContextRegret | None:
    """Regret for one :class:`~app.evaluation.counterfactual_engine.CounterfactualGroup`.

    Returns ``None`` when nothing in the group resolved — a context whose quotes never
    arrived carries no information about the selector.
    """
    outcomes: Mapping[str, Any] = dict(getattr(group, "outcomes", {}) or {})
    usable = {
        strategy_id: outcome
        for strategy_id, outcome in outcomes.items()
        if int(getattr(outcome, "quotes_observed", 1) or 0) > 0
    }
    if not usable:
        return None

    selected = getattr(group, "selected_strategy", None)
    selected_id = str(selected) if selected else None
    live_net = _finite(getattr(group, "live_outcome_net_bps", None))

    if selected_id is None:
        selected_outcome = NO_TRADE_OUTCOME_BPS
        selected_from_live = True  # declining is not a simulation
    elif live_net is not None:
        selected_outcome = live_net
        selected_from_live = True
    else:
        simulated = usable.get(selected_id)
        if simulated is None:
            # The selected strategy has neither a live fill nor a resolved simulation, so
            # regret is not computable for this context. Substituting zero would silently
            # credit the selector with a break-even trade it never made.
            return None
        selected_outcome = float(getattr(simulated, "net_return_bps", 0.0))
        selected_from_live = False

    # NO_TRADE competes: the best available is the best alternative OR doing nothing.
    ranked = sorted(
        (
            (strategy_id, float(getattr(outcome, "net_return_bps", 0.0)))
            for strategy_id, outcome in usable.items()
        ),
        key=lambda item: -item[1],
    )
    best_strategy, best_outcome = ranked[0]
    if best_outcome < NO_TRADE_OUTCOME_BPS:
        best_strategy, best_outcome = None, NO_TRADE_OUTCOME_BPS

    predicted = dict(getattr(group, "predicted_utility_bps", {}) or {})
    predicted_best = (
        max(predicted, key=lambda key: predicted[key]) if predicted else None
    )
    reasons: list[str] = []
    if not selected_from_live and selected_id is not None:
        reasons.append("REGRET_SELECTED_OUTCOME_SIMULATED")
    if len(usable) <= 1:
        reasons.append("REGRET_SINGLE_ALTERNATIVE")
    del top_k  # top-k hit rates are computed over a set of contexts, not one

    return ContextRegret(
        context_id=str(getattr(group, "context_id", "") or ""),
        symbol=str(getattr(group, "symbol", "") or ""),
        decision=str(getattr(group, "decision", "") or ""),
        selected_strategy=selected_id,
        selected_outcome_bps=selected_outcome,
        best_strategy=best_strategy,
        best_outcome_bps=best_outcome,
        alternative_count=len(usable),
        selected_from_live=selected_from_live,
        predicted_best_strategy=predicted_best,
        reason_codes=tuple(reasons),
    )


@dataclass(frozen=True)
class RegretSummary:
    """Aggregate selector-quality metrics over a set of contexts."""

    context_count: int
    mean_selector_regret_bps: float | None
    median_selector_regret_bps: float | None
    top1_selection_hit_rate: float | None
    top3_selection_hit_rate: float | None
    no_trade_precision: float | None
    no_trade_recall: float | None
    missed_opportunity_bps: float | None
    wrong_regime_trade_rate: float | None
    selected_strategy_net_ev_bps: float | None
    oracle_best_strategy_net_ev_bps: float | None
    live_selected_share: float
    reason_codes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "context_count": self.context_count,
            "mean_selector_regret_bps": _round(self.mean_selector_regret_bps),
            "median_selector_regret_bps": _round(self.median_selector_regret_bps),
            "top1_selection_hit_rate": _round(self.top1_selection_hit_rate, 4),
            "top3_selection_hit_rate": _round(self.top3_selection_hit_rate, 4),
            "no_trade_precision": _round(self.no_trade_precision, 4),
            "no_trade_recall": _round(self.no_trade_recall, 4),
            "missed_opportunity_bps": _round(self.missed_opportunity_bps),
            "wrong_regime_trade_rate": _round(self.wrong_regime_trade_rate, 4),
            "selected_strategy_net_ev_bps": _round(self.selected_strategy_net_ev_bps),
            "oracle_best_strategy_net_ev_bps": _round(
                self.oracle_best_strategy_net_ev_bps
            ),
            "live_selected_share": round(self.live_selected_share, 4),
            "reason_codes": list(self.reason_codes),
        }


def summarize_regret(
    regrets: Iterable[ContextRegret],
    *,
    groups: Sequence[Any] = (),
    minimum_contexts: int = 20,
) -> RegretSummary:
    """Aggregate per-context regrets.

    ``minimum_contexts`` does not filter anything — it adds a reason code when the sample
    is too small to interpret. Suppressing the numbers would be worse: the operator would
    see nothing and could not tell "no data" from "not computed".
    """
    rows = tuple(regrets)
    if not rows:
        return RegretSummary(
            context_count=0,
            mean_selector_regret_bps=None,
            median_selector_regret_bps=None,
            top1_selection_hit_rate=None,
            top3_selection_hit_rate=None,
            no_trade_precision=None,
            no_trade_recall=None,
            missed_opportunity_bps=None,
            wrong_regime_trade_rate=None,
            selected_strategy_net_ev_bps=None,
            oracle_best_strategy_net_ev_bps=None,
            live_selected_share=0.0,
            reason_codes=("REGRET_NO_CONTEXTS",),
        )

    regret_values = [row.regret_bps for row in rows]
    reasons: list[str] = []
    if len(rows) < max(1, int(minimum_contexts)):
        reasons.append(f"REGRET_SAMPLE_BELOW_{int(minimum_contexts)}")

    traded = [row for row in rows if row.selected_strategy is not None]
    declined = [row for row in rows if row.selected_strategy is None]

    # NO_TRADE precision: of the contexts where we declined, how often was declining
    # actually right (nothing beat zero)?
    precision = (
        sum(1 for row in declined if row.best_outcome_bps <= NO_TRADE_OUTCOME_BPS)
        / len(declined)
        if declined
        else None
    )
    # NO_TRADE recall: of the contexts where declining WOULD have been right, how often
    # did we decline?
    should_decline = [row for row in rows if row.best_outcome_bps <= NO_TRADE_OUTCOME_BPS]
    recall = (
        sum(1 for row in should_decline if row.selected_strategy is None)
        / len(should_decline)
        if should_decline
        else None
    )
    # Missed opportunity: value left on the table by declining when a trade would have
    # paid. Only over declined contexts — a bad pick is regret, not a missed opportunity.
    missed = [
        row.best_outcome_bps
        for row in declined
        if row.best_outcome_bps > NO_TRADE_OUTCOME_BPS
    ]
    missed_mean = sum(missed) / len(missed) if missed else None
    # Wrong-regime trade rate: traded, and every alternative including doing nothing beat
    # the pick. That is the signature of trading a state no strategy suited.
    wrong_regime = (
        sum(
            1
            for row in traded
            if row.selected_outcome_bps < NO_TRADE_OUTCOME_BPS
            and row.best_outcome_bps <= NO_TRADE_OUTCOME_BPS
        )
        / len(traded)
        if traded
        else None
    )

    top3 = _top_k_hit_rate(rows, groups, k=3)
    return RegretSummary(
        context_count=len(rows),
        mean_selector_regret_bps=sum(regret_values) / len(regret_values),
        median_selector_regret_bps=statistics.median(regret_values),
        top1_selection_hit_rate=sum(1 for row in rows if row.top1_hit) / len(rows),
        top3_selection_hit_rate=top3,
        no_trade_precision=precision,
        no_trade_recall=recall,
        missed_opportunity_bps=missed_mean,
        wrong_regime_trade_rate=wrong_regime,
        selected_strategy_net_ev_bps=sum(row.selected_outcome_bps for row in rows)
        / len(rows),
        oracle_best_strategy_net_ev_bps=sum(row.best_outcome_bps for row in rows)
        / len(rows),
        live_selected_share=sum(1 for row in rows if row.selected_from_live) / len(rows),
        reason_codes=tuple(reasons),
    )


def _top_k_hit_rate(
    rows: Sequence[ContextRegret], groups: Sequence[Any], k: int
) -> float | None:
    """Was the realised-best strategy inside the selector's predicted top ``k``?

    Needs the groups because the predicted ranking lives there. Without them the answer is
    ``None`` rather than a guess — reporting top-1 twice under two names would be worse
    than reporting one number.
    """
    if not groups:
        return None
    by_context = {
        str(getattr(group, "context_id", "")): dict(
            getattr(group, "predicted_utility_bps", {}) or {}
        )
        for group in groups
    }
    scored = 0
    hits = 0
    for row in rows:
        predicted = by_context.get(row.context_id)
        if not predicted:
            continue
        scored += 1
        ranking = sorted(predicted, key=lambda key: -predicted[key])[: max(1, int(k))]
        if row.best_strategy is None:
            # Doing nothing was best. That is a hit only if the selector declined.
            hits += 1 if row.selected_strategy is None else 0
            continue
        hits += 1 if row.best_strategy in ranking else 0
    return hits / scored if scored else None


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None

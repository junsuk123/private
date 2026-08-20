"""Why was no strategy selected? One record per candidate, one stage per record.

The engine had reason codes at every layer, but no single place said *which layer*
stopped a candidate and *what the numbers were when it did*. On 2026-08-05 the
answer to "why is nothing trading" required reading four subsystems by hand:
`micro_buy_intents` said `TECHNICAL_EDGE_NON_POSITIVE`, but not whether the edge
was non-positive because the rule found nothing, because cost exceeded any
plausible edge for that horizon, or because an unreliable model dragged a positive
rule edge down.

This module does not decide anything. It records. Ordering matters: stages are
declared in pipeline order so the FIRST failing stage is the cause, and callers
must not re-stamp a candidate that already failed earlier.

Existing reason codes are untouched — a stage is an ADDITIONAL, coarser label that
groups them.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


class SelectionStage(str, Enum):
    """Pipeline order. The first stage a candidate fails is its cause."""

    RAW_CANDIDATE = "RAW_CANDIDATE"
    FEATURE_UNAVAILABLE = "FEATURE_UNAVAILABLE"
    STRATEGY_TRIGGER_FALSE = "STRATEGY_TRIGGER_FALSE"
    GROSS_EDGE_NON_POSITIVE = "GROSS_EDGE_NON_POSITIVE"
    #: The best plausible gross edge for this strategy/horizon cannot clear cost.
    #: Distinct from GROSS_EDGE_NON_POSITIVE: the setup may be real and still be
    #: uneconomic, and the honest output is NO_TRADE rather than a negative number.
    HORIZON_COST_UNVIABLE = "HORIZON_COST_UNVIABLE"
    MODEL_NOT_RELIABLE = "MODEL_NOT_RELIABLE"
    MODEL_DISAGREEMENT = "MODEL_DISAGREEMENT"
    FUSED_NET_NON_POSITIVE = "FUSED_NET_NON_POSITIVE"
    COST_FLOOR_REJECTED = "COST_FLOOR_REJECTED"
    PROFITABILITY_REJECTED = "PROFITABILITY_REJECTED"
    MACRO_BLOCKED = "MACRO_BLOCKED"
    ONTOLOGY_BLOCKED = "ONTOLOGY_BLOCKED"
    #: Deployment state, NOT economics. Kept separate so "no edge" and "not
    #: allowed to trade yet" never collapse into one number.
    SHADOW_ONLY = "SHADOW_ONLY"
    LIVE_NOT_AUTHORIZED = "LIVE_NOT_AUTHORIZED"
    SELECTED = "SELECTED"


#: Stage order used for "first failing stage wins" and for funnel reporting.
STAGE_ORDER: tuple[SelectionStage, ...] = tuple(SelectionStage)

_STAGE_INDEX = {stage: index for index, stage in enumerate(STAGE_ORDER)}

#: Stages that mean "this candidate could still have been selected". Everything
#: else is a stop. SELECTED is terminal-positive.
_PASSTHROUGH = frozenset({SelectionStage.RAW_CANDIDATE})


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass
class StrategySelectionDiagnostic:
    """One candidate's journey. Numbers are None when that layer never ran."""

    symbol: str
    strategy_id: str
    stage: SelectionStage = SelectionStage.RAW_CANDIDATE
    market: str = ""
    regime: str = ""
    horizon_seconds: float | None = None
    # Rule side
    rule_gross_bps: float | None = None
    all_in_cost_bps: float | None = None
    rule_net_bps: float | None = None
    # Model side
    model_net_bps: float | None = None
    model_weight: float | None = None
    uncertainty_penalty_bps: float | None = None
    # Fusion and the bar it had to clear
    fused_net_bps: float | None = None
    required_net_bps: float | None = None
    cost_coverage_ratio: float | None = None
    #: Existing lower-level codes, preserved verbatim.
    reason_codes: tuple[str, ...] = ()
    detail: str = ""

    def mark(
        self,
        stage: SelectionStage,
        *,
        reason_codes: Iterable[str] = (),
        detail: str = "",
        **numbers: Any,
    ) -> "StrategySelectionDiagnostic":
        """Advance to ``stage``, keeping the EARLIEST stop.

        A later layer cannot overwrite an earlier failure: if a candidate died at
        GROSS_EDGE_NON_POSITIVE, a downstream ontology check reporting
        ONTOLOGY_BLOCKED would otherwise rewrite history and the funnel would
        blame the wrong layer.
        """
        for key, value in numbers.items():
            if not hasattr(self, key):
                raise AttributeError(f"unknown diagnostic field: {key}")
            number = _finite(value)
            if number is not None:
                setattr(self, key, number)
        if reason_codes:
            merged = [*self.reason_codes, *(str(code) for code in reason_codes)]
            self.reason_codes = tuple(dict.fromkeys(merged))
        already_stopped = self.stage not in _PASSTHROUGH
        if already_stopped and _STAGE_INDEX[stage] > _STAGE_INDEX[self.stage]:
            return self
        self.stage = stage
        if detail:
            self.detail = detail
        return self

    @property
    def stopped(self) -> bool:
        return self.stage not in _PASSTHROUGH and self.stage is not SelectionStage.SELECTED

    @property
    def net_surplus_bps(self) -> float | None:
        """fused net minus the bar it had to clear. The ranking quantity."""
        if self.fused_net_bps is None or self.required_net_bps is None:
            return None
        return self.fused_net_bps - self.required_net_bps

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stage"] = self.stage.value
        payload["reason_codes"] = list(self.reason_codes)
        payload["net_surplus_bps"] = self.net_surplus_bps
        return payload


@dataclass
class SelectionDiagnosticsCollector:
    """Accumulates one cycle's candidates and summarises the funnel."""

    records: list[StrategySelectionDiagnostic] = field(default_factory=list)

    def candidate(
        self,
        symbol: str,
        strategy_id: str,
        *,
        market: str = "",
        regime: str = "",
    ) -> StrategySelectionDiagnostic:
        record = StrategySelectionDiagnostic(
            symbol=str(symbol or ""),
            strategy_id=str(strategy_id or ""),
            market=str(market or ""),
            regime=str(regime or ""),
        )
        self.records.append(record)
        return record

    def stage_counts(self) -> dict[str, int]:
        counts = Counter(record.stage.value for record in self.records)
        return {stage.value: counts.get(stage.value, 0) for stage in STAGE_ORDER}

    def funnel(self) -> dict[str, int]:
        """Survivor counts at each checkpoint the acceptance criteria name."""
        total = len(self.records)

        def survived(stage: SelectionStage) -> int:
            index = _STAGE_INDEX[stage]
            return sum(
                1
                for record in self.records
                if record.stage is SelectionStage.SELECTED
                or not record.stopped
                or _STAGE_INDEX[record.stage] > index
            )

        return {
            "raw": total,
            "trigger": survived(SelectionStage.STRATEGY_TRIGGER_FALSE),
            "gross_positive": survived(SelectionStage.GROSS_EDGE_NON_POSITIVE),
            "horizon_viable": survived(SelectionStage.HORIZON_COST_UNVIABLE),
            "net_positive": survived(SelectionStage.FUSED_NET_NON_POSITIVE),
            "gate_passed": survived(SelectionStage.PROFITABILITY_REJECTED),
            "live_authorized": survived(SelectionStage.LIVE_NOT_AUTHORIZED),
            "selected": sum(
                1 for record in self.records if record.stage is SelectionStage.SELECTED
            ),
        }

    def by_market_and_strategy(self) -> dict[str, dict[str, dict[str, int]]]:
        grouped: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
        for record in self.records:
            grouped[record.market or "UNKNOWN"][record.strategy_id or "UNKNOWN"][
                record.stage.value
            ] += 1
        return {
            market: {strategy: dict(counts) for strategy, counts in strategies.items()}
            for market, strategies in grouped.items()
        }

    def edge_decomposition(self) -> dict[str, dict[str, float | None]]:
        """Mean rule/cost/model/fused decomposition per stage.

        Averaged only over records that actually carry the number, so a stage that
        never reached the model does not report a fabricated 0.0 model edge.
        """
        fields = (
            "rule_gross_bps",
            "all_in_cost_bps",
            "rule_net_bps",
            "model_net_bps",
            "model_weight",
            "uncertainty_penalty_bps",
            "fused_net_bps",
            "required_net_bps",
            "cost_coverage_ratio",
        )
        out: dict[str, dict[str, float | None]] = {}
        buckets: dict[str, list[StrategySelectionDiagnostic]] = defaultdict(list)
        for record in self.records:
            buckets[record.stage.value].append(record)
        for stage, group in buckets.items():
            stats: dict[str, float | None] = {"count": float(len(group))}
            for name in fields:
                values = [
                    value
                    for value in (getattr(record, name) for record in group)
                    if value is not None
                ]
                stats[name] = (sum(values) / len(values)) if values else None
            out[stage] = stats
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_counts": self.stage_counts(),
            "funnel": self.funnel(),
            "by_market_and_strategy": self.by_market_and_strategy(),
            "edge_decomposition": self.edge_decomposition(),
            "records": [record.as_dict() for record in self.records],
        }

    def blocking_summary(self) -> tuple[str, ...]:
        """Distinct first-failing stages, most common first — the headline."""
        counts = Counter(
            record.stage.value for record in self.records if record.stopped
        )
        return tuple(stage for stage, _ in counts.most_common())


def merge_stage_counts(payloads: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Sum ``stage_counts`` across cycles (for the offline report script)."""
    total: Counter = Counter()
    for payload in payloads:
        counts = payload.get("stage_counts") if isinstance(payload, Mapping) else None
        if not isinstance(counts, Mapping):
            continue
        for stage, value in counts.items():
            try:
                total[str(stage)] += int(value)
            except (TypeError, ValueError):
                continue
    return {stage.value: total.get(stage.value, 0) for stage in STAGE_ORDER}


def collector_from_algorithm_evaluations(
    evaluations: Iterable[Mapping[str, Any]],
    *,
    session: Mapping[str, Any] | None = None,
) -> SelectionDiagnosticsCollector:
    """Reconstruct a measured funnel from the engine's cycle snapshot.

    Older engine instances publish ``algorithm_evaluations`` but do not retain a
    collector object.  Returning NO_SELECTION_CYCLE_YET in that case discards
    evidence the engine already exposes.  This adapter is diagnostic-only: it
    maps the recorded trigger/reason/edge fields and never influences selection.
    """

    owner = session or {}
    collector = SelectionDiagnosticsCollector()
    selected_symbol = str(owner.get("selected_symbol") or "")
    selected_strategy = str(owner.get("selected_strategy") or "")
    session_reasons = tuple(
        dict.fromkeys(
            str(code)
            for group in (
                owner.get("gnn_reason_codes") or (),
                owner.get("ontology_reason_codes") or (),
                owner.get("bandit_reason_codes") or (),
            )
            for code in group
        )
    )
    last_reason = str(owner.get("last_reason") or "")
    if last_reason:
        session_reasons = tuple(dict.fromkeys((*session_reasons, last_reason)))

    bandit_by_pair: dict[tuple[str, str], Mapping[str, Any]] = {}
    for bandit in owner.get("bandit_evaluations") or ():
        if not isinstance(bandit, Mapping):
            continue
        arm = str(bandit.get("arm") or "").split(":", 1)[0]
        pair = (str(bandit.get("symbol") or ""), arm)
        if all(pair):
            bandit_by_pair[pair] = bandit

    regime = str(owner.get("macro_regime") or "")
    explicit_macro_blocks = {
        "MACRO_BLOCKED",
        "MACRO_STRATEGY_BLOCKED",
        "BANDIT_ARM_NOT_MACRO_PERMITTED",
        "MICRO_STRATEGY_BLOCKED_BY_MACRO",
    }
    for item in evaluations:
        symbol = str(item.get("symbol") or "")
        strategy_id = str(item.get("strategy_id") or "")
        market = "KR" if len(symbol) == 6 and symbol.isdigit() else "US"
        record = collector.candidate(
            symbol,
            strategy_id,
            market=market,
            regime=regime,
        )
        reasons = tuple(str(code) for code in item.get("reason_codes") or ())
        horizon = item.get("horizon_seconds")
        edge = _finite(item.get("expected_edge_bps"))
        record.mark(
            SelectionStage.RAW_CANDIDATE,
            horizon_seconds=horizon,
            rule_gross_bps=edge,
            reason_codes=reasons,
        )
        if not bool(item.get("triggered")):
            feature_missing = any(
                marker in code
                for code in reasons
                for marker in ("NOT_READY", "MISSING", "UNAVAILABLE", "INSUFFICIENT")
            )
            record.mark(
                SelectionStage.FEATURE_UNAVAILABLE
                if feature_missing
                else SelectionStage.STRATEGY_TRIGGER_FALSE,
                reason_codes=reasons,
                detail="recorded algorithm trigger did not fire",
            )
            continue
        if edge is None:
            record.mark(
                SelectionStage.FEATURE_UNAVAILABLE,
                reason_codes=reasons,
                detail="triggered evaluation has no finite expected edge",
            )
        elif edge <= 0:
            record.mark(
                SelectionStage.GROSS_EDGE_NON_POSITIVE,
                reason_codes=reasons,
                detail="recorded expected edge is non-positive",
            )
        elif symbol == selected_symbol and strategy_id == selected_strategy:
            record.mark(SelectionStage.SELECTED, reason_codes=session_reasons)
        elif (bandit := bandit_by_pair.get((symbol, strategy_id))) is not None and not bool(
            bandit.get("admissible")
        ):
            bandit_reasons = tuple(str(code) for code in bandit.get("reason_codes") or ())
            if bool(bandit.get("shadow_only")):
                record.mark(
                    SelectionStage.SHADOW_ONLY,
                    reason_codes=(*reasons, *bandit_reasons, *session_reasons),
                    fused_net_bps=bandit.get("conservative_edge_bps"),
                    uncertainty_penalty_bps=bandit.get("uncertainty_penalty_bps"),
                    required_net_bps=0.0,
                    detail="bandit arm was measured but remains shadow-only",
                )
            else:
                record.mark(
                    SelectionStage.PROFITABILITY_REJECTED,
                    reason_codes=(*reasons, *bandit_reasons, *session_reasons),
                    fused_net_bps=bandit.get("conservative_edge_bps"),
                    model_net_bps=bandit.get("posterior_mean_net_bps"),
                    uncertainty_penalty_bps=bandit.get("uncertainty_penalty_bps"),
                    required_net_bps=0.0,
                    detail="realized-history conservative edge did not beat NO_TRADE",
                )
        elif "GNN_NOT_LIVE_AUTHORIZED" in session_reasons:
            record.mark(
                SelectionStage.LIVE_NOT_AUTHORIZED,
                reason_codes=session_reasons,
                detail="cycle reached a positive trigger but live GNN authority was absent",
            )
        elif any(
            code in explicit_macro_blocks or code.endswith("_BLOCKED_BY_MACRO")
            for code in session_reasons
        ):
            record.mark(SelectionStage.MACRO_BLOCKED, reason_codes=session_reasons)
        elif any("ONTOLOGY" in code for code in session_reasons):
            record.mark(SelectionStage.ONTOLOGY_BLOCKED, reason_codes=session_reasons)
        else:
            record.mark(
                SelectionStage.STRATEGY_TRIGGER_FALSE,
                reason_codes=(*reasons, *session_reasons),
                detail="positive trigger was not selected in the recorded cycle",
            )
    return collector

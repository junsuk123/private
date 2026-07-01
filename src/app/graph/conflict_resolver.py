from __future__ import annotations

from dataclasses import replace
from itertools import combinations

from app.graph.theory_registry import TheoryRegistry, get_theory_registry
from app.graph.theory_vote import ConflictRecord, TheoryVote


STYLE_CONFLICTS = {
    frozenset(("contrarian", "continuation")),
    frozenset(("contrarian", "breakout")),
    frozenset(("mean_reversion", "breakout")),
    frozenset(("risk_reduction", "continuation")),
    frozenset(("risk_reduction", "breakout")),
}
ACTION_CONFLICTS = {
    frozenset(("BUY", "SELL")),
    frozenset(("BUY", "REDUCE")),
}
HORIZON_CONFLICTS = {
    frozenset(("scalp", "late_intraday")),
    frozenset(("scalp", "position")),
    frozenset(("short_intraday", "position")),
}


class ConflictResolver:
    def __init__(self, registry: TheoryRegistry | None = None) -> None:
        self.registry = registry or get_theory_registry()

    def resolve(self, votes: tuple[TheoryVote, ...]) -> tuple[tuple[TheoryVote, ...], tuple[ConflictRecord, ...]]:
        conflicts: list[ConflictRecord] = []
        penalty_by_key: dict[tuple[str, str, str], float] = {}
        for left, right in combinations(votes, 2):
            detected = self._detect(left, right)
            for conflict in detected:
                conflicts.append(conflict)
                penalty_by_key[(left.ticker, left.theory_id, left.evidence_cluster_id)] = max(
                    penalty_by_key.get((left.ticker, left.theory_id, left.evidence_cluster_id), 0.0),
                    conflict.penalty,
                )
                penalty_by_key[(right.ticker, right.theory_id, right.evidence_cluster_id)] = max(
                    penalty_by_key.get((right.ticker, right.theory_id, right.evidence_cluster_id), 0.0),
                    conflict.penalty,
                )

        adjusted: list[TheoryVote] = []
        for vote in votes:
            penalty = penalty_by_key.get((vote.ticker, vote.theory_id, vote.evidence_cluster_id), 0.0)
            conflict_labels = tuple(conflict.type for conflict in conflicts if vote.theory_id in {conflict.theory_a, conflict.theory_b})
            adjusted.append(
                replace(
                    vote,
                    action=vote.normalized_action,
                    raw_signal=max(0.0, vote.raw_signal * (1.0 - penalty)),
                    conflicts=tuple(dict.fromkeys((*vote.conflicts, *conflict_labels))),
                )
            )
        return tuple(adjusted), tuple(conflicts)

    def _detect(self, left: TheoryVote, right: TheoryVote) -> tuple[ConflictRecord, ...]:
        if left.ticker != right.ticker or left.theory_id == right.theory_id:
            return ()
        records: list[ConflictRecord] = []
        explicit = self._explicit_conflict(left, right)
        if explicit:
            records.append(_record("registry_conflict", left, right, 0.25, "Configured theory conflict reduced both votes."))
        if frozenset((left.style, right.style)) in STYLE_CONFLICTS:
            records.append(_record("style_conflict", left, right, 0.20, "Incompatible theory styles were not summed as independent support."))
        if frozenset((left.normalized_action, right.normalized_action)) in ACTION_CONFLICTS:
            records.append(_record("action_conflict", left, right, 0.25, "BUY and SELL/REDUCE evidence were separated by action."))
        if frozenset((left.horizon_bucket, right.horizon_bucket)) in HORIZON_CONFLICTS:
            records.append(_record("horizon_conflict", left, right, 0.15, "Different holding horizons reduced combined confidence."))
        if left.evidence_cluster_id == right.evidence_cluster_id and left.theory_id != right.theory_id:
            records.append(_record("duplicate_evidence", left, right, 0.10, "Correlated evidence cluster was compressed."))
        return tuple(records)

    def _explicit_conflict(self, left: TheoryVote, right: TheoryVote) -> bool:
        left_meta = self.registry.get(left.theory_id)
        right_meta = self.registry.get(right.theory_id)
        return bool(
            (left_meta and right.theory_id in left_meta.conflicts_with)
            or (right_meta and left.theory_id in right_meta.conflicts_with)
        )


def _record(kind: str, left: TheoryVote, right: TheoryVote, penalty: float, resolution: str) -> ConflictRecord:
    return ConflictRecord(
        type=kind,
        theory_a=left.theory_id,
        theory_b=right.theory_id,
        penalty=penalty,
        resolution=resolution,
    )

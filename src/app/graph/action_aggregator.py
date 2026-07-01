from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Mapping

from app.graph.conflict_resolver import ConflictResolver
from app.graph.theory_registry import TheoryRegistry, get_theory_registry
from app.graph.theory_vote import ACTION_NAMES, EvidenceClusterVote, FinalActionDecision, PositionContext, TheoryVote


class ActionAggregator:
    def __init__(self, registry: TheoryRegistry | None = None) -> None:
        self.registry = registry or get_theory_registry()
        self.conflict_resolver = ConflictResolver(self.registry)

    def decide(
        self,
        ticker: str,
        votes: tuple[TheoryVote, ...],
        *,
        position_context: PositionContext | None = None,
        npu_profile: Mapping[str, object] | None = None,
    ) -> FinalActionDecision:
        position = position_context or PositionContext()
        relevant = tuple(vote for vote in votes if vote.ticker == ticker)
        if self.registry.ontology_voting.enable_conflict_resolver:
            relevant, conflicts = self.conflict_resolver.resolve(relevant)
        else:
            conflicts = ()
        clusters = self._cluster(relevant)
        scores = {action: 0.0 for action in ACTION_NAMES}
        theory_by_action: dict[str, list[str]] = defaultdict(list)
        for cluster in clusters:
            scores[cluster.action] += cluster.compressed_score
            theory_by_action[cluster.action].extend(cluster.theory_ids)

        conflict_score = min(1.0, sum(conflict.penalty for conflict in conflicts))
        if conflict_score >= self.registry.ontology_voting.high_conflict_hold_threshold:
            scores["HOLD" if position.has_position else "WATCH"] += conflict_score
        if not relevant:
            scores["HOLD" if position.has_position else "WATCH"] = 1.0
        selected = self._select(scores, position, conflict_score)
        dominant = tuple(sorted(relevant, key=lambda vote: vote.effective_weight, reverse=True)[:5])
        explanation = (
            f"{ticker} selected_action={selected}; "
            f"scores BUY={scores['BUY']:.3f}, SELL={scores['SELL']:.3f}, HOLD={scores['HOLD']:.3f}, "
            f"REDUCE={scores['REDUCE']:.3f}, WATCH={scores['WATCH']:.3f}; conflicts={len(conflicts)}."
        )
        return FinalActionDecision(
            ticker=ticker,
            selected_action=selected,
            scores={key: round(value, 6) for key, value in scores.items()},
            decision_margin=self.registry.ontology_voting.decision_margin,
            dominant_theories=dominant,
            conflicts=conflicts,
            evidence_clusters=clusters,
            position_context=position,
            final_explanation=explanation,
            npu_accelerated=bool((npu_profile or {}).get("uses_npu")),
            npu_profile=dict(npu_profile or {}),
        )

    def _cluster(self, votes: tuple[TheoryVote, ...]) -> tuple[EvidenceClusterVote, ...]:
        grouped: dict[tuple[str, str], list[TheoryVote]] = defaultdict(list)
        for vote in votes:
            grouped[(vote.evidence_cluster_id, vote.normalized_action)].append(vote)
        clusters: list[EvidenceClusterVote] = []
        cap = self.registry.ontology_voting.max_same_cluster_contribution
        for (cluster_id, action), cluster_votes in grouped.items():
            raw = sum(vote.effective_weight for vote in cluster_votes)
            compressed = raw / sqrt(len(cluster_votes)) if len(cluster_votes) > 1 else raw
            clusters.append(
                EvidenceClusterVote(
                    ticker=cluster_votes[0].ticker,
                    cluster_id=cluster_id,
                    action=action,
                    raw_feature_count=len(cluster_votes),
                    compressed_score=min(cap, compressed),
                    theory_ids=tuple(dict.fromkeys(vote.theory_id for vote in cluster_votes)),
                )
            )
        return tuple(clusters)

    def _select(self, scores: dict[str, float], position: PositionContext, conflict_score: float) -> str:
        margin = self.registry.ontology_voting.decision_margin
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_action, top_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else 0.0
        if top_score - second_score < margin:
            return "HOLD" if position.has_position else "WATCH"
        if conflict_score >= self.registry.ontology_voting.high_conflict_hold_threshold:
            return "HOLD" if position.has_position else "WATCH"
        if not position.has_position and top_action in {"SELL", "REDUCE", "HOLD"}:
            return "WATCH"
        if position.has_position and top_action == "BUY":
            return "HOLD"
        return top_action

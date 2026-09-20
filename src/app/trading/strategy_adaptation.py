"""Bounded, cash-net strategy evidence used before *every* entry election.

Market, regime, direction and algorithm version remain separate. Simulation may
show that an arm is unsuitable, but cannot masquerade as a live fill or silently
restore full live authority. Correlated, overlapping outcomes of the same symbol
form one episode; confidence does not grow merely by journaling the same move.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from app.trading.strategy_performance_store import (
    PosteriorConfig,
    StrategyOutcome,
    normalize_direction,
    normalize_market,
    normalize_regime,
)

PERFORMANCE_SHADOW_ONLY = "STRATEGY_PERFORMANCE_SHADOW_ONLY"


@dataclass(frozen=True)
class EvidenceEstimate:
    evidence_source: str
    sample_count: int = 0
    independent_episode_count: int = 0
    effective_sample_count: float = 0.0
    realized_net_bps: float | None = None
    expected_net_bps: float | None = None
    posterior_net_bps: float | None = None
    lower_net_bps: float | None = None
    upper_net_bps: float | None = None
    prediction_bias_bps: float | None = None
    round_trip_cost_bps: float | None = None
    last_observed_at: str | None = None

    @property
    def demonstrated_loss(self) -> bool:
        return self.upper_net_bps is not None and self.upper_net_bps < 0.0

    @property
    def demonstrated_gain(self) -> bool:
        return self.lower_net_bps is not None and self.lower_net_bps > 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrategyAdaptationAssessment:
    strategy_id: str
    market: str
    regime: str
    direction: str
    execution_product: str
    observed_at: str
    performance_state: str
    live_entry_allowed: bool
    live: EvidenceEstimate
    shadow: EvidenceEstimate
    reason_codes: tuple[str, ...]
    evidence_refs: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class StrategyAdaptation:
    """Small replay queries with a zero-edge prior, not a new trained model.

    The store's uncertainty/prior settings govern the evidence bands. They are
    conservative diagnostics, not calibrated frequentist confidence guarantees.
    No positive history is manufactured for an unseen regime and no entry safety
    or strategy deployment authorization is granted by this component.
    """

    def __init__(self, *, store: Any | None = None) -> None:
        self._store = store

    def _resolve_store(self) -> Any:
        if self._store is None:
            from app.trading.strategy_performance_store import default_store

            self._store = default_store()
        return self._store

    def assess(
        self,
        strategy_id: str,
        *,
        market: str,
        regime: str | None,
        now: datetime,
        direction: str = "LONG",
        execution_product: str = "CASH",
        deployment_state: str = "LIVE_FULL",
        change_point_probability: float = 0.0,
        required_policy_family: str | None = None,
    ) -> StrategyAdaptationAssessment:
        moment = _utc(now)
        market = normalize_market(market)
        regime = normalize_regime(regime)
        direction = normalize_direction(direction)
        product = str(execution_product or "CASH").upper()
        reasons: list[str] = []
        rows: tuple[StrategyOutcome, ...] = ()
        try:
            store = self._resolve_store()
            cfg = getattr(store, "posterior_config", PosteriorConfig())
            rows = tuple(
                row
                for sources in (("live", "live_probe"), ("shadow",))
                for row in store.recent_outcomes(
                    strategy_id, market=market, regime=regime, direction=direction,
                    execution_product=product, evaluation_sources=sources,
                    limit=max(1, min(120, cfg.window)), as_of=moment,
                )
            )
        except Exception:  # A missing evidence channel cannot affirm performance.
            cfg = PosteriorConfig()
            reasons.append("STRATEGY_PERFORMANCE_UNAVAILABLE")

        # Defensive validation also protects injected stores and historical imports.
        age_days = max(0.0, float(cfg.max_age_days))
        cutoff = moment - timedelta(days=age_days) if age_days else None
        rows = tuple(
            row for row in rows
            if _utc(row.recorded_at) <= moment
            and (cutoff is None or _utc(row.recorded_at) >= cutoff)
            and math.isfinite(row.realized_net_bps)
            and row.signal_executable
            and normalize_market(row.market) == market
            and normalize_regime(row.regime) == regime
            and normalize_direction(row.direction) == direction
            and str(row.execution_product).upper() == product
        )
        rows = tuple(sorted(rows, key=lambda row: _utc(row.recorded_at), reverse=True))
        if required_policy_family:
            if any(getattr(row, "risk_policy_family", "legacy") != required_policy_family for row in rows):
                reasons.append("OLD_RISK_POLICY_LOSSES_RETAINED_GAINS_NOT_PROMOTABLE")
            rows = tuple(row for row in rows if (
                getattr(row, "risk_policy_family", "legacy") == required_policy_family
                or row.realized_net_bps <= 0.0
            ))
        live_rows = tuple(row for row in rows if row.is_live)
        shadow_rows = tuple(row for row in rows if row.evaluation_source == "shadow")
        # Use the same prior uncertainty as the existing store, with time relevance
        # and change-point discount applied to each independent episode.
        common = dict(now=moment, cfg=cfg, change_point_probability=change_point_probability)
        live = _estimate(live_rows, source="LIVE", **common)
        shadow = _estimate(shadow_rows, source="SHADOW", **common)
        # A structural-break discount expresses reduced predictive relevance; it
        # is not evidence that an already demonstrated losing arm recovered.
        # Preserve that quarantine until fresh matching shadow recovery exists.
        undiscounted = dict(common, change_point_probability=0.0)
        live_loss = live.demonstrated_loss or _estimate(live_rows, source="LIVE", **undiscounted).demonstrated_loss
        shadow_loss = shadow.demonstrated_loss or _estimate(shadow_rows, source="SHADOW", **undiscounted).demonstrated_loss
        if ((live_loss and not live.demonstrated_loss)
                or (shadow_loss and not shadow.demonstrated_loss)):
            reasons.append("PERFORMANCE_LOSS_RETAINED_ACROSS_CHANGE_POINT")
        state, allowed = "ACTIVE" if rows else "COLD", True
        if "STRATEGY_PERFORMANCE_UNAVAILABLE" in reasons:
            state, allowed = "UNAVAILABLE", False
            reasons.append(PERFORMANCE_SHADOW_ONLY)
        if not rows:
            reasons.append("STRATEGY_PERFORMANCE_NO_MATCHING_MATURE_EVIDENCE")
        if live_loss or shadow_loss:
            state, allowed = "SHADOW_ONLY", False
            reasons.append(PERFORMANCE_SHADOW_ONLY)
            reasons.extend(
                code for condition, code in (
                    (live_loss, "LIVE_NET_EVIDENCE_NEGATIVE"),
                    (shadow_loss, "SHADOW_NET_EVIDENCE_NEGATIVE"),
                ) if condition
            )
        if live_loss and live_rows:
            last_live = max(_utc(row.recorded_at) for row in live_rows)
            recovery = _estimate(
                tuple(row for row in shadow_rows if _utc(row.recorded_at) > last_live
                      and (not required_policy_family or getattr(row, "risk_policy_family", "legacy") == required_policy_family)),
                source="SHADOW", **common,
            )
            if recovery.demonstrated_gain:
                state = "RECOVERY_READY"
                allowed = str(deployment_state).upper() == "LIVE_PROBE"
                reasons.append("SHADOW_RECOVERY_REQUIRES_AUTHORIZED_LIVE_PROBE")
                if allowed:
                    reasons = [reason for reason in reasons if reason != PERFORMANCE_SHADOW_ONLY]
        if rows:
            reasons.append("PERFORMANCE_MARKET_REGIME_VERSION_CONDITIONED")
        return StrategyAdaptationAssessment(
            strategy_id=str(strategy_id), market=market, regime=regime,
            direction=direction, execution_product=product, observed_at=moment.isoformat(),
            performance_state=state, live_entry_allowed=allowed,
            live=live, shadow=shadow, reason_codes=tuple(dict.fromkeys(reasons)),
            evidence_refs=tuple(
                f"performance:{market}:{regime}:{strategy_id}:{row.evaluation_source}:{row.recorded_at.isoformat()}"
                for row in rows[:12]
            ),
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _episodes(rows: Sequence[StrategyOutcome]) -> list[tuple[datetime, list[StrategyOutcome]]]:
    groups: dict[str, list[StrategyOutcome]] = {}
    for row in rows:
        groups.setdefault(row.symbol.upper(), []).append(row)
    episodes: list[tuple[datetime, list[StrategyOutcome]]] = []
    for group in groups.values():
        intervals = []
        for row in group:
            end = _utc(row.recorded_at)
            duration = float(row.holding_seconds or 0.0)
            duration = duration if math.isfinite(duration) and duration > 0.0 else 0.0
            intervals.append((end - timedelta(seconds=duration), end, row))
        intervals.sort(key=lambda item: item[0])
        end: datetime | None = None
        members: list[StrategyOutcome] = []
        for start, finish, row in intervals:
            if end is not None and start > end:
                episodes.append((end, members))
                members = []
                end = None
            members.append(row)
            end = max(end, finish) if end is not None else finish
        if end is not None:
            episodes.append((end, members))
    return episodes


def _estimate(
    rows: Sequence[StrategyOutcome], *, source: str, now: datetime,
    cfg: Any, change_point_probability: float,
) -> EvidenceEstimate:
    if not rows:
        return EvidenceEstimate(source)
    episodes = _episodes(rows)
    prior_weight = max(1.0, float(cfg.prior_weight))
    prior_sd = max(1.0, float(cfg.prior_stdev_bps))
    # Half the evidence window is a relevance horizon, not a trade exit threshold.
    half_life = max(1.0, float(cfg.max_age_days) * 86400.0 / 2.0)
    cp = float(change_point_probability)
    cp = min(1.0, max(0.0, cp)) if math.isfinite(cp) else 1.0
    weights = [
        (0.5 ** (max(0.0, (now - end).total_seconds()) / half_life)
         if cfg.max_age_days > 0 else 1.0) * (1.0 - cp)
        for end, _ in episodes
    ]
    values = [sum(row.realized_net_bps for row in group) / len(group) for _, group in episodes]
    weight = sum(weights)
    # Sum of relevance weights never exceeds independent episode count.
    effective = weight
    observed = sum(w * value for w, value in zip(weights, values)) / weight if weight else None
    mean = (sum(w * value for w, value in zip(weights, values)) / (prior_weight + weight))
    variance = (
        sum(w * (value - observed) ** 2 for w, value in zip(weights, values)) / weight
        if weight and observed is not None else prior_sd ** 2
    )
    # Prior variance remains represented: repeated identical shadow outcomes must
    # not acquire zero uncertainty, especially after a change point.
    posterior_variance = (prior_weight * prior_sd ** 2 + weight * variance) / (prior_weight + weight)
    uncertainty = max(1.0, float(cfg.pessimism_z)) * math.sqrt(posterior_variance / max(1.0, effective))
    expected = [row.expected_net_bps for row in rows if row.expected_net_bps is not None and math.isfinite(row.expected_net_bps)]
    residual = [row.realized_net_bps - row.expected_net_bps for row in rows if row.expected_net_bps is not None and math.isfinite(row.expected_net_bps)]
    costs = [row.realized_gross_bps - row.realized_net_bps for row in rows if row.realized_gross_bps is not None and math.isfinite(row.realized_gross_bps)]
    return EvidenceEstimate(
        evidence_source=source, sample_count=len(rows), independent_episode_count=len(episodes),
        effective_sample_count=effective, realized_net_bps=observed,
        expected_net_bps=sum(expected) / len(expected) if expected else None,
        posterior_net_bps=mean, lower_net_bps=mean - uncertainty, upper_net_bps=mean + uncertainty,
        prediction_bias_bps=sum(residual) / len(residual) if residual else None,
        round_trip_cost_bps=sum(costs) / len(costs) if costs else None,
        last_observed_at=max(_utc(row.recorded_at) for row in rows).isoformat(),
    )

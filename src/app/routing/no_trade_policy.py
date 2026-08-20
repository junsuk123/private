"""NO_TRADE as a first-class action with its own utility.

The problem
-----------
``forced_selection``: when no strategy suits the current tape, the least-bad negative
strategy is armed. The conservative bandit already has a ``no_trade`` arm, but it is
internal to the bandit — there is no object that says *what doing nothing is worth here*,
no configurable minimum edge per market or horizon, and no way to evaluate the NO_TRADE
decision itself (precision, recall, missed opportunity).

The rule
--------
``NO_TRADE`` competes in the same ranking, with utility

    U_NO_TRADE = minimum_required_edge_bps(market, horizon)

so a strategy wins only by clearing that bar. ``minimum_required_edge_bps`` is not one
number: KR pays ~28bps a round trip and US 51-70, and a 3-minute horizon cannot reach what
a session-long one can. Using a single threshold for both is what let US candidates be
compared against a KRX bar.

The primary criterion is a positive LOWER bound, not a positive mean
-------------------------------------------------------------------
``lower_confidence_bound_bps = net - uncertainty`` must clear the bar. Comparing means to
cost is how a strategy with a wide error band and a slightly positive mean keeps trading:
half its distribution is below cost, and cost is paid every time.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "NO_TRADE_REASONS",
    "NoTradePolicy",
    "NoTradePolicyConfig",
    "NoTradeVerdict",
]


class NO_TRADE_REASONS:
    COVERAGE_GAP = "NO_TRADE_STRATEGY_COVERAGE_GAP"
    ALL_HARD_BLOCKED = "NO_TRADE_ALL_STRATEGIES_HARD_BLOCKED"
    NO_ENTRY_READY = "NO_TRADE_NO_ENTRY_READY_STRATEGY"
    DATA_QUALITY = "NO_TRADE_DATA_QUALITY_INSUFFICIENT"
    BELOW_MINIMUM_EDGE = "NO_TRADE_BELOW_MINIMUM_EDGE"
    NEGATIVE_LOWER_BOUND = "NO_TRADE_NEGATIVE_LOWER_CONFIDENCE_BOUND"
    NO_CANDIDATES = "NO_TRADE_NO_CANDIDATES"
    NO_PROPOSALS = "NO_TRADE_NO_PROPOSALS"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


@dataclass(frozen=True)
class NoTradePolicyConfig:
    """Per-market minimum edge, plus the horizon attainability rule.

    ``base_minimum_edge_bps`` is a V2-only utility preference. It is not the live
    deterministic algorithm's entry floor: that path charges the complete round-trip
    cost once and requires strictly positive net edge without this additional buffer.
    """

    base_minimum_edge_bps: Mapping[str, float] = field(
        default_factory=lambda: {"KR": 10.0, "US": 20.0}
    )
    default_minimum_edge_bps: float = 15.0
    #: Extra bar charged when the utility estimate is not a measured/trusted one.
    unmeasured_penalty_bps: float = 5.0
    #: Feature completeness below which no trade is admissible at all, regardless of edge.
    minimum_feature_completeness: float = 0.25
    #: Require ``net - uncertainty > bar`` rather than ``net > bar``.
    require_positive_lower_bound: bool = True
    #: Horizons shorter than this (seconds) have their bar raised, because a short horizon
    #: cannot attain a large move: the measured median absolute US move is 6.7bps over 3
    #: minutes against a 51.2bps round trip. Set to 0 to disable.
    short_horizon_seconds: float = 300.0
    short_horizon_extra_bps: float = 5.0

    @classmethod
    def from_env(cls) -> "NoTradePolicyConfig":
        return cls(
            base_minimum_edge_bps={
                "KR": max(0.0, _env_float("NO_TRADE_MIN_EDGE_BPS_KR", 10.0)),
                "US": max(0.0, _env_float("NO_TRADE_MIN_EDGE_BPS_US", 20.0)),
            },
            default_minimum_edge_bps=max(
                0.0, _env_float("NO_TRADE_MIN_EDGE_BPS_DEFAULT", 15.0)
            ),
            unmeasured_penalty_bps=max(
                0.0, _env_float("NO_TRADE_UNMEASURED_PENALTY_BPS", 5.0)
            ),
            minimum_feature_completeness=max(
                0.0, _env_float("NO_TRADE_MIN_FEATURE_COMPLETENESS", 0.25)
            ),
        )

    def minimum_edge_bps(
        self,
        *,
        market: str,
        horizon_seconds: float | None = None,
        measured: bool = True,
    ) -> float:
        bar = dict(self.base_minimum_edge_bps).get(
            str(market or "").strip().upper(), self.default_minimum_edge_bps
        )
        if not measured:
            bar += self.unmeasured_penalty_bps
        if (
            self.short_horizon_seconds > 0
            and horizon_seconds is not None
            and horizon_seconds > 0
            and horizon_seconds < self.short_horizon_seconds
        ):
            bar += self.short_horizon_extra_bps
        return max(0.0, bar)

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_minimum_edge_bps": dict(self.base_minimum_edge_bps),
            "default_minimum_edge_bps": self.default_minimum_edge_bps,
            "unmeasured_penalty_bps": self.unmeasured_penalty_bps,
            "minimum_feature_completeness": self.minimum_feature_completeness,
            "require_positive_lower_bound": self.require_positive_lower_bound,
            "short_horizon_seconds": self.short_horizon_seconds,
            "short_horizon_extra_bps": self.short_horizon_extra_bps,
        }


@dataclass(frozen=True)
class NoTradeVerdict:
    """Whether doing nothing wins, and by what margin."""

    no_trade: bool
    no_trade_utility_bps: float
    best_candidate_utility_bps: float | None
    best_candidate_lower_bound_bps: float | None
    best_strategy_id: str | None
    reason_codes: tuple[str, ...]

    @property
    def margin_bps(self) -> float | None:
        """How far the best candidate cleared (or missed) the bar."""
        if self.best_candidate_utility_bps is None:
            return None
        return self.best_candidate_utility_bps - self.no_trade_utility_bps

    def as_dict(self) -> dict[str, Any]:
        return {
            "no_trade": self.no_trade,
            "no_trade_utility_bps": round(self.no_trade_utility_bps, 3),
            "best_candidate_utility_bps": (
                round(self.best_candidate_utility_bps, 3)
                if self.best_candidate_utility_bps is not None
                else None
            ),
            "best_candidate_lower_bound_bps": (
                round(self.best_candidate_lower_bound_bps, 3)
                if self.best_candidate_lower_bound_bps is not None
                else None
            ),
            "best_strategy_id": self.best_strategy_id,
            "margin_bps": round(value, 3) if (value := self.margin_bps) is not None else None,
            "reason_codes": list(self.reason_codes),
        }


class NoTradePolicy:
    """Decides whether NO_TRADE beats the best ranked candidate."""

    def __init__(self, *, config: NoTradePolicyConfig | None = None) -> None:
        self._config = config or NoTradePolicyConfig.from_env()

    @property
    def config(self) -> NoTradePolicyConfig:
        return self._config

    def no_trade_utility_bps(
        self,
        *,
        market: str,
        horizon_seconds: float | None = None,
        measured: bool = True,
    ) -> float:
        return self._config.minimum_edge_bps(
            market=market, horizon_seconds=horizon_seconds, measured=measured
        )

    def evaluate(
        self,
        *,
        market: str,
        candidates: tuple[Any, ...],
        feature_completeness: float | None = None,
        eligible_count: int = 0,
        entry_ready_count: int = 0,
        coverage_gap: bool = False,
    ) -> NoTradeVerdict:
        """``candidates`` are ranked candidates carrying the utility decomposition.

        Each must expose ``strategy_id``, ``final_utility_bps``, ``expected_net_return_bps``,
        ``uncertainty_penalty_bps``, ``expected_holding_seconds`` and ``cost_measured``.
        Ordering is not assumed — the best is selected here.
        """
        reasons: list[str] = []

        completeness = feature_completeness if feature_completeness is not None else 0.0
        if completeness < self._config.minimum_feature_completeness:
            reasons.append(NO_TRADE_REASONS.DATA_QUALITY)
        if coverage_gap:
            reasons.append(NO_TRADE_REASONS.COVERAGE_GAP)
        if eligible_count <= 0:
            reasons.append(NO_TRADE_REASONS.ALL_HARD_BLOCKED)
        if entry_ready_count <= 0:
            reasons.append(NO_TRADE_REASONS.NO_ENTRY_READY)

        if not candidates:
            reasons.append(NO_TRADE_REASONS.NO_CANDIDATES)
            return NoTradeVerdict(
                no_trade=True,
                no_trade_utility_bps=self.no_trade_utility_bps(market=market),
                best_candidate_utility_bps=None,
                best_candidate_lower_bound_bps=None,
                best_strategy_id=None,
                reason_codes=tuple(dict.fromkeys(reasons)),
            )

        best = max(candidates, key=lambda item: float(getattr(item, "final_utility_bps", 0.0)))
        horizon = _optional(getattr(best, "expected_holding_seconds", None))
        measured = bool(getattr(best, "cost_measured", True))
        bar = self.no_trade_utility_bps(
            market=market, horizon_seconds=horizon, measured=measured
        )
        utility = float(getattr(best, "final_utility_bps", 0.0))
        net = _optional(getattr(best, "expected_net_return_bps", None))
        uncertainty = _optional(getattr(best, "uncertainty_penalty_bps", None)) or 0.0
        lower_bound = (net - uncertainty) if net is not None else None

        if utility <= bar:
            reasons.append(NO_TRADE_REASONS.BELOW_MINIMUM_EDGE)
        if (
            self._config.require_positive_lower_bound
            and lower_bound is not None
            and lower_bound <= bar
        ):
            reasons.append(NO_TRADE_REASONS.NEGATIVE_LOWER_BOUND)

        return NoTradeVerdict(
            # Any surviving reason means no trade. Reasons are collected rather than
            # short-circuited so the diagnostics show every condition that failed, not
            # just the first — "why did nothing trade" was previously answerable only one
            # reason at a time.
            no_trade=bool(reasons),
            no_trade_utility_bps=bar,
            best_candidate_utility_bps=utility,
            best_candidate_lower_bound_bps=lower_bound,
            best_strategy_id=str(getattr(best, "strategy_id", "") or "") or None,
            reason_codes=tuple(dict.fromkeys(reasons)),
        )


def _optional(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

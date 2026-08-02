"""Conservative contextual bandit strategy selector, with NO_TRADE as an arm.

The problem this replaces
------------------------
Election used to be "take the first admissible ranked BUY intent". That rule has
no notion of *how well this strategy has actually been doing lately*, and — worse
— it has no way to answer "none of these is worth taking". In a tape where every
measured strategy has a negative net expectancy, a selector that must always pick
something will pick the least-bad negative expectancy, over and over.

So the arm set here explicitly includes ``no_trade``, and it wins by default.

The selection rule
------------------
For every candidate arm::

    ConservativeEdge = posterior_expected_net_bps - uncertainty_penalty_bps

and the arm is admissible only if ``ConservativeEdge > 0``. This is the whole
point of the module: selection is on a pessimistic LOWER bound, not on a mean.
A strategy with three lucky samples has a huge uncertainty penalty and therefore
loses to ``no_trade``; a strategy with sixty samples and a real edge wins.

Three inputs shape the penalty:

* realized history (:mod:`app.trading.strategy_performance_store`) — sample count,
  dispersion, loss streak;
* ``change_point_probability`` (:mod:`app.graph.change_point`) — discounts the
  effective sample count, so a regime break widens uncertainty automatically
  instead of needing a manual history reset;
* the context (regime, volatility percentile, breadth, flow, spread, liquidity),
  which selects *which* history is relevant and applies explicit context
  penalties for a dislocated or unreadable market.

The forward-looking edge supplied by the caller (the model / ontology / GNN
estimate for this specific candidate) is blended in with an explicit, configurable
trust weight. It is never taken at face value: on a cold arm it is shrunk hard,
which is what stops an untested strategy from being armed on its own optimism.

Advisory: this module ranks and can veto. It never submits an order, and the
authoritative gates (ProfitabilityGate, RiskManager, FinalTradeGate) still run
afterwards.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from app.trading.directional import (
    DirectionalStrategyKey,
    ExecutionProduct,
    PositionDirection,
    ShortReasonCodes,
    StrategyDeploymentState,
)
from app.trading.strategy_performance_store import (
    NO_TRADE_ARM,
    StrategyPerformanceStore,
    StrategyPosterior,
    default_store,
    market_for_symbol,
    normalize_market,
    normalize_regime,
)

# Macro regimes in which a directional SHORT is fighting the tape. Kept as a small
# explicit set rather than a substring test so a new regime name is a deliberate
# addition instead of an accidental match.
_SHORT_UNFAVOURABLE_REGIMES: frozenset[str] = frozenset(
    {"TREND_UP", "STRONG_TREND_UP", "HIGH_VOL_TRENDING_UP", "RISK_ON", "BULL"}
)


def _regime_opposes_short(macro_regime: str | None) -> bool:
    return normalize_regime(macro_regime) in _SHORT_UNFAVOURABLE_REGIMES

# --- Reason codes ----------------------------------------------------------- #
BANDIT_NO_POSITIVE_CONSERVATIVE_EDGE = "BANDIT_NO_POSITIVE_CONSERVATIVE_EDGE"
BANDIT_ARM_SELECTED = "BANDIT_ARM_SELECTED"
BANDIT_NO_CANDIDATE_ARMS = "BANDIT_NO_CANDIDATE_ARMS"
BANDIT_CHANGE_POINT_STAND_DOWN = "BANDIT_CHANGE_POINT_STAND_DOWN"
BANDIT_CONTEXT_PENALTY_APPLIED = "BANDIT_CONTEXT_PENALTY_APPLIED"
BANDIT_ARM_COLD_START_SHADOW_ONLY = "BANDIT_ARM_COLD_START_SHADOW_ONLY"
BANDIT_ARM_LOSS_STREAK_SUSPENDED = "BANDIT_ARM_LOSS_STREAK_SUSPENDED"
BANDIT_ARM_NOT_MACRO_PERMITTED = "BANDIT_ARM_NOT_MACRO_PERMITTED"
BANDIT_ARM_COLD_START_EXPLORATION = "BANDIT_ARM_COLD_START_EXPLORATION"
BANDIT_EXPLORATION_ARM_SELECTED = "BANDIT_EXPLORATION_ARM_SELECTED"
BANDIT_ARM_MEASURED_NEGATIVE_EDGE = "BANDIT_ARM_MEASURED_NEGATIVE_EDGE"
BANDIT_ARM_PREDICTION_CAPPED_BY_REALIZED_LOSS = (
    "BANDIT_ARM_PREDICTION_CAPPED_BY_REALIZED_LOSS"
)
BANDIT_SHORT_DIRECTION_PENALTY_APPLIED = "BANDIT_SHORT_DIRECTION_PENALTY_APPLIED"
BANDIT_SHORT_AGAINST_REGIME_PENALTY = "BANDIT_SHORT_AGAINST_REGIME_PENALTY"
BANDIT_BOTH_DIRECTIONS_NEGATIVE = ShortReasonCodes.BOTH_DIRECTIONS_NEGATIVE


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.getenv(name)
        return float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.getenv(name)
        return int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class BanditContext:
    """The slow market context the arms are conditioned on."""

    market: str = "KR"
    macro_regime: str = "UNKNOWN"
    change_point_probability: float = 0.0
    regime_stability: float | None = None
    volatility_percentile: float | None = None
    market_breadth: float | None = None
    foreign_flow_zscore: float | None = None
    spread_percentile: float | None = None
    liquidity_score: float | None = None
    time_of_day_bucket: str = ""
    symbol_market_beta: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "macro_regime": self.macro_regime,
            "change_point_probability": round(float(self.change_point_probability), 6),
            "regime_stability": self.regime_stability,
            "volatility_percentile": self.volatility_percentile,
            "market_breadth": self.market_breadth,
            "foreign_flow_zscore": self.foreign_flow_zscore,
            "spread_percentile": self.spread_percentile,
            "liquidity_score": self.liquidity_score,
            "time_of_day_bucket": self.time_of_day_bucket,
            "symbol_market_beta": self.symbol_market_beta,
        }


@dataclass(frozen=True)
class ArmCandidate:
    """One strategy the electing layer considers admissible on other grounds.

    ``arm`` remains the plain strategy id for backward compatibility, but the
    posterior is looked up by :attr:`key` — the full
    :class:`~app.trading.directional.DirectionalStrategyKey`. LONG and SHORT arms of
    the same strategy therefore compete against each other, and against
    ``no_trade``, from SEPARATE realized histories.
    """

    arm: str
    symbol: str = ""
    # Forward-looking net edge for THIS candidate from the model / ontology / GNN.
    predicted_net_edge_bps: float | None = None
    predicted_gross_edge_bps: float | None = None
    expected_cost_bps: float | None = None
    confidence: float | None = None
    macro_permitted: bool | None = None
    live_authorized: bool = True
    reason_codes: tuple[str, ...] = ()
    payload: Mapping[str, Any] = field(default_factory=dict)
    # --- Direction ---------------------------------------------------------- #
    direction: PositionDirection = PositionDirection.LONG
    execution_product: ExecutionProduct = ExecutionProduct.CASH
    # Committed deployment state of this arm. SHADOW means the arm may be RANKED and
    # reported but never becomes an executable candidate — it appears in
    # ``shadow_arms``, not as the selection.
    #
    # ``None`` resolves per DIRECTION in ``__post_init__``, and that asymmetry is the
    # point. A flat SHADOW default would have silently made every existing LONG caller
    # (which does not pass this field) unselectable — the long path's gate has always
    # been ``live_authorized``, and it still is. A flat LIVE_FULL default would have
    # made an unevaluated SHORT tradable. So: LONG defaults open and keeps its
    # existing gate, SHORT defaults closed.
    deployment_state: StrategyDeploymentState | None = None
    # Borrow verdict for a SHORT candidate, resolved by the caller from a fresh
    # snapshot. ``None`` on a short arm means "not established", which is refused.
    borrow_available: bool | None = None
    borrow_fee_bps_annualised: float | None = None

    def __post_init__(self) -> None:
        if self.deployment_state is None:
            object.__setattr__(
                self,
                "deployment_state",
                StrategyDeploymentState.SHADOW
                if self.direction is PositionDirection.SHORT
                else StrategyDeploymentState.LIVE_FULL,
            )

    @property
    def is_short(self) -> bool:
        return self.direction is PositionDirection.SHORT

    def directional_key(self, market: str) -> DirectionalStrategyKey:
        return DirectionalStrategyKey(
            strategy_id=self.arm,
            direction=self.direction,
            market=market,
            execution_product=self.execution_product,
        )

    @property
    def display_arm(self) -> str:
        """``strategy_id:DIRECTION`` for a short, bare id for a long.

        Longs keep their historical label so existing dashboards, reason strings and
        stored session state are unaffected by the addition of shorts.
        """
        return self.arm if self.direction is PositionDirection.LONG else f"{self.arm}:SHORT"


@dataclass(frozen=True)
class ArmEvaluation:
    arm: str
    symbol: str
    conservative_edge_bps: float
    posterior_mean_net_bps: float
    uncertainty_penalty_bps: float
    context_penalty_bps: float
    predicted_net_edge_bps: float | None
    history_weight: float
    sample_count: int
    effective_sample_count: float
    loss_streak: int
    admissible: bool
    shadow_only: bool
    reason_codes: tuple[str, ...]
    exploration: bool = False
    direction: str = "LONG"
    execution_product: str = "CASH"
    deployment_state: str = "SHADOW"
    borrow_available: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "exploration": self.exploration,
            "symbol": self.symbol,
            "direction": self.direction,
            "execution_product": self.execution_product,
            "deployment_state": self.deployment_state,
            "borrow_available": self.borrow_available,
            "conservative_edge_bps": round(self.conservative_edge_bps, 3),
            "posterior_mean_net_bps": round(self.posterior_mean_net_bps, 3),
            "uncertainty_penalty_bps": round(self.uncertainty_penalty_bps, 3),
            "context_penalty_bps": round(self.context_penalty_bps, 3),
            "predicted_net_edge_bps": self.predicted_net_edge_bps,
            "history_weight": round(self.history_weight, 4),
            "sample_count": self.sample_count,
            "effective_sample_count": round(self.effective_sample_count, 3),
            "loss_streak": self.loss_streak,
            "admissible": self.admissible,
            "shadow_only": self.shadow_only,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class BanditSelection:
    selected_arm: str
    selected_symbol: str
    conservative_edge_bps: float
    is_no_trade: bool
    evaluations: tuple[ArmEvaluation, ...]
    shadow_arms: tuple[str, ...]
    context: BanditContext
    reason_codes: tuple[str, ...]
    timestamp: datetime
    # True when the winning arm was selected to LEARN, not because it has a
    # demonstrated edge. Sizing should treat it as a minimum-size probe.
    is_exploration: bool = False
    selected_direction: str = "LONG"

    # --- LONG vs SHORT vs NO_TRADE ------------------------------------------- #
    # Recorded on every selection, whether or not a short was selectable. Without
    # this the question "would a short have helped here?" is unanswerable after the
    # fact, and it is the question the whole short programme has to justify itself
    # against: adding arms that only ever fire alongside a better long buys nothing
    # but exposure. It also supplies ``short_rescue_rate`` to the promotion gates.
    @property
    def best_long_edge_bps(self) -> float | None:
        return _best_edge(self.evaluations, "LONG")

    @property
    def best_short_edge_bps(self) -> float | None:
        return _best_edge(self.evaluations, "SHORT")

    @property
    def short_rescued(self) -> bool:
        """No long had a positive edge, but a short did."""
        long_edge = self.best_long_edge_bps
        short_edge = self.best_short_edge_bps
        if short_edge is None or short_edge <= 0.0:
            return False
        return long_edge is None or long_edge <= 0.0

    @property
    def both_directions_negative(self) -> bool:
        long_edge = self.best_long_edge_bps
        short_edge = self.best_short_edge_bps
        return (long_edge is None or long_edge <= 0.0) and (
            short_edge is None or short_edge <= 0.0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "selected_arm": self.selected_arm,
            "selected_symbol": self.selected_symbol,
            "selected_direction": self.selected_direction,
            "conservative_edge_bps": round(self.conservative_edge_bps, 3),
            "is_no_trade": self.is_no_trade,
            "is_exploration": self.is_exploration,
            "shadow_arms": list(self.shadow_arms),
            "context": self.context.as_dict(),
            "reason_codes": list(self.reason_codes),
            "directional_comparison": {
                "best_long_conservative_edge_bps": self.best_long_edge_bps,
                "best_short_conservative_edge_bps": self.best_short_edge_bps,
                "short_rescued": self.short_rescued,
                "both_directions_negative": self.both_directions_negative,
            },
            "evaluations": [item.as_dict() for item in self.evaluations],
        }


def _best_edge(evaluations: Sequence[ArmEvaluation], direction: str) -> float | None:
    """Highest conservative edge among arms of one direction, or ``None``.

    Deliberately ignores admissibility. The question is "did this direction have an
    edge", not "was it tradable" — a short that was blocked only by its deployment
    state still shows what shorting WOULD have offered, which is exactly the
    evidence the promotion ladder needs in order to ever unblock it.
    """
    edges = [
        item.conservative_edge_bps for item in evaluations if item.direction == direction
    ]
    return max(edges) if edges else None


@dataclass(frozen=True)
class BanditConfig:
    """Pessimism knobs. Every default errs toward ``no_trade``."""

    # How much of the caller's forward-looking edge is trusted on a COLD arm.
    cold_start_prediction_trust: float = 0.25
    # ...and on a fully sampled arm, where realized history dominates instead.
    warm_prediction_trust: float = 0.5
    # Samples at which an arm counts as fully warm.
    warm_sample_count: int = 20
    # Consecutive losses that suspend an arm outright for this market/regime.
    loss_streak_suspension: int = 4
    # Change-point probability above which no new entry is elected at all.
    change_point_stand_down: float = 0.5
    # Context penalties (bps) applied on top of the statistical penalty.
    dislocation_spread_percentile: float = 0.9
    dislocation_penalty_bps: float = 25.0
    thin_liquidity_score: float = 0.35
    thin_liquidity_penalty_bps: float = 15.0
    unknown_context_penalty_bps: float = 10.0
    # Arms below this sample count may only run in shadow, never live.
    minimum_live_sample_count: int = 0
    require_positive_conservative_edge: bool = True
    # --- Cold-start exploration -------------------------------------------------
    # Pessimism alone is not a runnable policy: with no realized history EVERY arm
    # has a large uncertainty penalty, so nothing is ever selected, so no history
    # is ever accumulated. That is a deadlock, not caution.
    #
    # An arm with at most ``cold_start_max_sample_count`` samples and NO losing
    # streak may therefore be selected on its forward-looking edge alone, flagged
    # as exploration so sizing can keep it minimal. The important asymmetry: an arm
    # whose MEASURED history is negative is never explored — it has already
    # answered the question, and "keep trying it" is exactly the failure mode this
    # module exists to prevent.
    cold_start_exploration_enabled: bool = True
    cold_start_max_sample_count: int = 3
    cold_start_min_predicted_edge_bps: float = 5.0
    # --- Direction penalties (bps) ---------------------------------------------
    # Charged to SHORT arms on top of the statistical and context penalties, to
    # cover the asymmetry the realized posterior cannot see: an unbounded and
    # ACCELERATING downside (the position grows as it moves against you), and the
    # possibility of a forced cover on recall. Expressed as pessimism rather than as
    # cost so the cost engine cannot double-count it as a fee.
    short_direction_penalty_bps: float = 10.0
    # Additional charge for shorting into a rising broad market. A penalty and not a
    # veto: a beta-neutral thesis is legitimately valid in an up tape, and a hard
    # block would remove precisely the short worth keeping.
    short_against_regime_penalty_bps: float = 15.0

    @classmethod
    def from_env(cls) -> "BanditConfig":
        return cls(
            cold_start_prediction_trust=max(
                0.0,
                min(1.0, _env_float("BANDIT_COLD_START_PREDICTION_TRUST", cls.cold_start_prediction_trust)),
            ),
            warm_prediction_trust=max(
                0.0, min(1.0, _env_float("BANDIT_WARM_PREDICTION_TRUST", cls.warm_prediction_trust))
            ),
            warm_sample_count=max(1, _env_int("BANDIT_WARM_SAMPLE_COUNT", cls.warm_sample_count)),
            loss_streak_suspension=max(
                1, _env_int("BANDIT_LOSS_STREAK_SUSPENSION", cls.loss_streak_suspension)
            ),
            change_point_stand_down=_env_float(
                "BANDIT_CHANGE_POINT_STAND_DOWN", cls.change_point_stand_down
            ),
            dislocation_spread_percentile=_env_float(
                "BANDIT_DISLOCATION_SPREAD_PERCENTILE", cls.dislocation_spread_percentile
            ),
            dislocation_penalty_bps=max(
                0.0, _env_float("BANDIT_DISLOCATION_PENALTY_BPS", cls.dislocation_penalty_bps)
            ),
            thin_liquidity_score=_env_float("BANDIT_THIN_LIQUIDITY_SCORE", cls.thin_liquidity_score),
            thin_liquidity_penalty_bps=max(
                0.0, _env_float("BANDIT_THIN_LIQUIDITY_PENALTY_BPS", cls.thin_liquidity_penalty_bps)
            ),
            unknown_context_penalty_bps=max(
                0.0, _env_float("BANDIT_UNKNOWN_CONTEXT_PENALTY_BPS", cls.unknown_context_penalty_bps)
            ),
            minimum_live_sample_count=max(
                0, _env_int("BANDIT_MINIMUM_LIVE_SAMPLE_COUNT", cls.minimum_live_sample_count)
            ),
            require_positive_conservative_edge=_env_bool(
                "BANDIT_REQUIRE_POSITIVE_CONSERVATIVE_EDGE",
                cls.require_positive_conservative_edge,
            ),
            cold_start_exploration_enabled=_env_bool(
                "BANDIT_COLD_START_EXPLORATION_ENABLED", cls.cold_start_exploration_enabled
            ),
            cold_start_max_sample_count=max(
                0, _env_int("BANDIT_COLD_START_MAX_SAMPLE_COUNT", cls.cold_start_max_sample_count)
            ),
            cold_start_min_predicted_edge_bps=_env_float(
                "BANDIT_COLD_START_MIN_PREDICTED_EDGE_BPS",
                cls.cold_start_min_predicted_edge_bps,
            ),
            short_direction_penalty_bps=max(
                0.0,
                _env_float(
                    "BANDIT_SHORT_DIRECTION_PENALTY_BPS", cls.short_direction_penalty_bps
                ),
            ),
            short_against_regime_penalty_bps=max(
                0.0,
                _env_float(
                    "BANDIT_SHORT_AGAINST_REGIME_PENALTY_BPS",
                    cls.short_against_regime_penalty_bps,
                ),
            ),
        )


class ConservativeStrategyBandit:
    """Selects one strategy arm, or ``no_trade``, on a pessimistic edge bound."""

    def __init__(
        self,
        *,
        store: StrategyPerformanceStore | None = None,
        config: BanditConfig | None = None,
    ) -> None:
        self.store = store if store is not None else default_store()
        self.config = config or BanditConfig.from_env()

    def select(
        self,
        candidates: Sequence[ArmCandidate],
        context: BanditContext,
        *,
        now: datetime | None = None,
    ) -> BanditSelection:
        moment = now if now is not None else datetime.now(timezone.utc)
        moment = moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
        cfg = self.config
        reasons: list[str] = []

        if not candidates:
            return BanditSelection(
                selected_arm=NO_TRADE_ARM,
                selected_symbol="",
                conservative_edge_bps=0.0,
                is_no_trade=True,
                evaluations=(),
                shadow_arms=(),
                context=context,
                reason_codes=(BANDIT_NO_CANDIDATE_ARMS,),
                timestamp=moment,
            )

        change_point = max(0.0, min(1.0, float(context.change_point_probability or 0.0)))
        stand_down = change_point >= cfg.change_point_stand_down
        context_penalty = self._context_penalty(context)

        evaluations: list[ArmEvaluation] = []
        for candidate in candidates:
            evaluations.append(
                self._evaluate(
                    candidate,
                    context,
                    change_point=change_point,
                    context_penalty=context_penalty,
                    stand_down=stand_down,
                )
            )
        # Exploitation before exploration: an arm with a demonstrated positive lower
        # bound always outranks a cold arm being tried on its own optimism, however
        # attractive that optimism looks.
        evaluations.sort(
            key=lambda item: (not item.exploration, item.conservative_edge_bps),
            reverse=True,
        )

        if context_penalty > 0:
            reasons.append(BANDIT_CONTEXT_PENALTY_APPLIED)
        if stand_down:
            reasons.append(BANDIT_CHANGE_POINT_STAND_DOWN)

        best = next((item for item in evaluations if item.admissible), None)
        shadow = tuple(item.arm for item in evaluations if item.shadow_only)
        if best is None:
            reasons.append(BANDIT_NO_POSITIVE_CONSERVATIVE_EDGE)
            no_trade = BanditSelection(
                selected_arm=NO_TRADE_ARM,
                selected_symbol="",
                conservative_edge_bps=0.0,
                is_no_trade=True,
                evaluations=tuple(evaluations),
                shadow_arms=shadow,
                context=context,
                reason_codes=tuple(dict.fromkeys(reasons)),
                timestamp=moment,
            )
            if no_trade.both_directions_negative:
                # Distinguishes "we looked both ways and neither paid" from "we only
                # had one direction to look". The first is a finding; the second is a
                # coverage gap.
                reasons.append(BANDIT_BOTH_DIRECTIONS_NEGATIVE)
                return replace(no_trade, reason_codes=tuple(dict.fromkeys(reasons)))
            return no_trade
        reasons.append(BANDIT_ARM_SELECTED)
        if best.exploration:
            reasons.append(BANDIT_EXPLORATION_ARM_SELECTED)
        return BanditSelection(
            selected_arm=best.arm,
            selected_symbol=best.symbol,
            selected_direction=best.direction,
            conservative_edge_bps=best.conservative_edge_bps,
            is_no_trade=False,
            is_exploration=best.exploration,
            evaluations=tuple(evaluations),
            shadow_arms=shadow,
            context=context,
            reason_codes=tuple(dict.fromkeys((*reasons, *best.reason_codes))),
            timestamp=moment,
        )

    # -- internals ---------------------------------------------------------- #
    def _evaluate(
        self,
        candidate: ArmCandidate,
        context: BanditContext,
        *,
        change_point: float,
        context_penalty: float,
        stand_down: bool,
    ) -> ArmEvaluation:
        cfg = self.config
        reasons: list[str] = list(candidate.reason_codes)
        # Per-candidate market, not the context's: a mixed KR/US proposal list would
        # otherwise score every arm against one market's realized history, and the
        # two differ by 2-3x in round-trip cost.
        market = (
            market_for_symbol(candidate.symbol)
            if candidate.symbol
            else normalize_market(context.market)
        )
        regime = normalize_regime(context.macro_regime)
        # Posterior is looked up per DIRECTION. Pooling the two would mean a strategy
        # pair making 60bps long and losing 60bps short reads as break-even — and
        # therefore as permanently untradable in both directions — while a genuine
        # one-sided edge is diluted into invisibility.
        posterior: StrategyPosterior = self.store.posterior(
            candidate.arm,
            market=market,
            regime=regime,
            change_point_probability=change_point,
            direction=str(candidate.direction),
            execution_product=str(candidate.execution_product),
        )
        reasons.extend(posterior.reason_codes)

        # Blend realized history with the caller's forward-looking estimate. The
        # weight on the estimate falls as realized samples accumulate: early on we
        # have nothing else, later the realized series is simply better evidence.
        warm_fraction = min(1.0, posterior.effective_sample_count / max(1, cfg.warm_sample_count))
        prediction_trust = (
            cfg.cold_start_prediction_trust
            + (cfg.warm_prediction_trust - cfg.cold_start_prediction_trust) * warm_fraction
        )
        predicted = candidate.predicted_net_edge_bps
        if predicted is None or not math.isfinite(float(predicted)):
            predicted_component = None
            blended_mean = posterior.posterior_mean_net_bps
            history_weight = 1.0
        else:
            predicted_component = float(predicted)
            history_weight = 1.0 - prediction_trust
            blended_mean = (
                history_weight * posterior.posterior_mean_net_bps
                + prediction_trust * predicted_component
            )

        # A warm arm that has MEASURED a loss cannot be argued back to profitability
        # by the forward-looking estimate. ``warm_prediction_trust`` is 0.5, so
        # without this cap half the score of a fully-sampled arm still comes from a
        # claim about the future -- and that claim scales with the configured profit
        # target. Raising the exit geometry's take-profit from 100bps to 160bps was
        # by itself enough to turn an arm with 25 realized samples averaging -43bps
        # into a positive conservative edge, which is precisely the trade this
        # module exists to refuse. Optimism may not outvote realized losses.
        if (
            predicted_component is not None
            and posterior.sample_count > cfg.cold_start_max_sample_count
            and posterior.posterior_mean_net_bps <= 0.0
            and blended_mean > posterior.posterior_mean_net_bps
        ):
            blended_mean = posterior.posterior_mean_net_bps
            history_weight = 1.0
            reasons.append(BANDIT_ARM_PREDICTION_CAPPED_BY_REALIZED_LOSS)

        penalty = posterior.uncertainty_penalty_bps + context_penalty
        # A short pays an accruing borrow fee and a wider expected exit slippage than
        # a long, and neither is in the realized posterior of a SHADOW arm whose
        # simulated fills already deducted them once. The directional penalty covers
        # the residual asymmetry the posterior cannot see: an unbounded, accelerating
        # downside, and recall risk. Charged as pessimism rather than as cost so it
        # cannot be mistaken for a fee and double-counted by the cost engine.
        if candidate.is_short:
            penalty += cfg.short_direction_penalty_bps
            reasons.append(BANDIT_SHORT_DIRECTION_PENALTY_APPLIED)
            # Shorting into a rising, broad market is fighting the tape. Charged
            # here rather than vetoed, because a beta-neutral short thesis
            # (residual_relative_weakness) is legitimately valid in an up market and
            # a hard block would remove exactly the arm worth keeping.
            if _regime_opposes_short(context.macro_regime):
                penalty += cfg.short_against_regime_penalty_bps
                reasons.append(BANDIT_SHORT_AGAINST_REGIME_PENALTY)
        conservative = blended_mean - penalty

        # An arm is "cold" while its realized history is too short to have
        # answered anything. Cold arms may be explored on their forward-looking
        # edge; arms that have MEASURED a negative edge may not.
        cold = (
            cfg.cold_start_exploration_enabled
            and posterior.sample_count <= cfg.cold_start_max_sample_count
            and posterior.loss_streak == 0
        )
        exploration = bool(
            cold
            and predicted_component is not None
            and predicted_component >= cfg.cold_start_min_predicted_edge_bps
        )

        admissible = True
        shadow_only = False
        # --- Deployment authorisation ---------------------------------------- #
        # Checked BEFORE anything about edge, and it is not a penalty. An arm that is
        # not deployment-authorised is not a worse trade, it is not a candidate: the
        # whole point of the SHADOW rung is that the arm keeps being evaluated and
        # journaled while being structurally incapable of producing an order. Setting
        # ``exploration = False`` alongside matters — cold-start exploration exists to
        # let an unproven arm be tried at minimum size, and a SHADOW short must not
        # be reachable through that door either.
        if not candidate.deployment_state.submits_orders:
            admissible = False
            exploration_blocked = True
            shadow_only = True
            reasons.append(
                ShortReasonCodes.DEPLOYMENT_SUSPENDED
                if candidate.deployment_state is StrategyDeploymentState.SUSPENDED
                else ShortReasonCodes.SHADOW_ONLY
            )
        else:
            exploration_blocked = False
        # A short with no established locate cannot be executed regardless of edge.
        # ``None`` (nobody asked) is treated exactly like False, because at order time
        # the two are the same thing.
        if candidate.is_short and candidate.borrow_available is not True:
            admissible = False
            exploration_blocked = True
            shadow_only = True
            reasons.append(ShortReasonCodes.BORROW_UNAVAILABLE)
        if candidate.macro_permitted is False:
            admissible = False
            reasons.append(BANDIT_ARM_NOT_MACRO_PERMITTED)
        if not candidate.live_authorized:
            admissible = False
            shadow_only = True
        if posterior.loss_streak >= cfg.loss_streak_suspension:
            admissible = False
            exploration = False
            reasons.append(BANDIT_ARM_LOSS_STREAK_SUSPENDED)
        if posterior.sample_count < cfg.minimum_live_sample_count:
            admissible = False
            exploration = False
            shadow_only = True
            reasons.append(BANDIT_ARM_COLD_START_SHADOW_ONLY)
        if stand_down:
            # A structural break invalidates the forward-looking estimate too, so
            # exploration is suspended along with exploitation.
            admissible = False
            exploration = False
            shadow_only = True
        if cfg.require_positive_conservative_edge and conservative <= 0.0:
            if exploration:
                reasons.append(BANDIT_ARM_COLD_START_EXPLORATION)
            else:
                admissible = False
                if posterior.sample_count > cfg.cold_start_max_sample_count:
                    reasons.append(BANDIT_ARM_MEASURED_NEGATIVE_EDGE)
        else:
            # A positive lower bound is exploitation, not exploration.
            exploration = False

        return ArmEvaluation(
            arm=candidate.display_arm,
            symbol=candidate.symbol,
            conservative_edge_bps=conservative,
            direction=str(candidate.direction),
            execution_product=str(candidate.execution_product),
            deployment_state=str(candidate.deployment_state),
            borrow_available=candidate.borrow_available,
            posterior_mean_net_bps=blended_mean,
            uncertainty_penalty_bps=posterior.uncertainty_penalty_bps,
            context_penalty_bps=context_penalty,
            predicted_net_edge_bps=predicted_component,
            history_weight=history_weight,
            sample_count=posterior.sample_count,
            effective_sample_count=posterior.effective_sample_count,
            loss_streak=posterior.loss_streak,
            admissible=admissible,
            shadow_only=shadow_only and not admissible,
            reason_codes=tuple(dict.fromkeys(reasons)),
            exploration=exploration and admissible and not exploration_blocked,
        )

    def _context_penalty(self, context: BanditContext) -> float:
        """Extra bps of pessimism the current market context deserves."""
        cfg = self.config
        penalty = 0.0
        spread_percentile = context.spread_percentile
        if spread_percentile is not None and spread_percentile >= cfg.dislocation_spread_percentile:
            penalty += cfg.dislocation_penalty_bps
        liquidity = context.liquidity_score
        if liquidity is not None and liquidity <= cfg.thin_liquidity_score:
            penalty += cfg.thin_liquidity_penalty_bps
        # An unreadable market is not a calm one. Missing breadth AND missing
        # volatility percentile means we are electing blind, so charge for it.
        if context.market_breadth is None and context.volatility_percentile is None:
            penalty += cfg.unknown_context_penalty_bps
        return penalty

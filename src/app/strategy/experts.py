from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import os
from typing import Mapping
from uuid import uuid4

from app.trading.contracts import IntentAction, OrderIntent, TradePlan
from app.strategy.catalog import STRATEGY_IDS
from app.strategy.exit_geometry import exit_geometry
from app.trading.directional import (
    PositionDirection,
    PositionEffect,
    StrategyDeploymentState,
    broker_side,
    default_product,
    parse_direction,
    stop_breached,
    stop_price,
    target_price,
    target_reached,
)


def _intent_action(direction: PositionDirection, effect: PositionEffect) -> IntentAction:
    """Broker-facing intent action for a (direction, effect) pair."""
    return IntentAction(broker_side(direction, effect))


@dataclass(frozen=True)
class ExpertContext:
    symbol: str
    as_of: datetime
    price: float
    proposed_quantity: int
    feature_snapshot_id: str
    utility_evidence_id: str
    quantiles: Mapping[str, float]

    def q(self, name: str) -> float:
        return max(0.0, min(1.0, float(self.quantiles.get(name, 0.5))))


@dataclass(frozen=True)
class ExpertConfig:
    entry_quantile: float = 0.8
    confirmation_quantile: float = 0.65
    stop_bps: float = 20.0
    profit_bps: float = 35.0
    trailing_bps: float = 15.0
    max_holding_seconds: int = 300
    max_entry_slippage_bps: float = 5.0
    entry_ttl_seconds: int = 5
    minimum_trailing_net_bps: float = 5.0


# Thresholds are strategy parameters, not a universal knob. Stored replay found
# that VWAP reversion only cleared costs once both displacement and recovery were
# moderately tightened; applying the same 0.85/0.75 pair to momentum or relative
# strength made them materially worse, so their defaults remain unchanged.
_STRATEGY_THRESHOLD_DEFAULTS: dict[str, tuple[float, float]] = {
    "vwap_mean_reversion": (0.85, 0.75),
}
_STRATEGY_MIN_TRAILING_NET_BPS: dict[str, float] = {
    "intraday_momentum": 15.0,
    "cross_sectional_relative_strength": 30.0,
    "residual_relative_strength": 15.0,
    "opening_range_breakout": 15.0,
}


def _geometry_config(strategy_id: str, **overrides: float) -> ExpertConfig:
    """Expert config whose exits come from the ONE geometry table.

    Previously each expert restated its own stop/target/holding numbers, which
    then had to agree by hand with ``strategy_session`` and with the training
    labels. They stopped agreeing, and a model trained on one geometry was scoring
    trades executed under another. Now all three read
    :mod:`app.strategy.exit_geometry`.
    """
    geometry = exit_geometry(strategy_id)
    normalized = strategy_id.upper().replace("-", "_")
    default_entry, default_confirmation = _STRATEGY_THRESHOLD_DEFAULTS.get(
        strategy_id, (0.8, 0.65)
    )

    def threshold(field: str, default: float) -> float:
        raw = os.getenv(
            f"STRATEGY_{normalized}_{field.upper()}",
            os.getenv(f"STRATEGY_{field.upper()}", str(default)),
        )
        try:
            return max(0.5, min(0.99, float(raw)))
        except (TypeError, ValueError):
            return default

    return ExpertConfig(
        entry_quantile=threshold("entry_quantile", default_entry),
        confirmation_quantile=threshold(
            "confirmation_quantile", default_confirmation
        ),
        stop_bps=geometry.stop_loss_bps,
        profit_bps=geometry.take_profit_bps,
        trailing_bps=geometry.trailing_bps,
        max_holding_seconds=geometry.max_holding_seconds,
        minimum_trailing_net_bps=_STRATEGY_MIN_TRAILING_NET_BPS.get(
            strategy_id, 5.0
        ),
        **overrides,
    )


class StrategyExpert:
    strategy_id = "base"
    thesis = "base"
    # LONG unless a subclass says otherwise. Direction is a class fact for an
    # expert (each expresses one thesis in one direction) while the tradable ARM
    # identity carries it explicitly — see DirectionalStrategyKey.
    direction = PositionDirection.LONG
    default_config = ExpertConfig()

    def __init__(self, config: ExpertConfig | None = None) -> None:
        self.config = config or self.default_config

    def admissible(self, context: ExpertContext) -> bool:
        raise NotImplementedError

    def propose(self, context: ExpertContext) -> TradePlan | None:
        if context.price <= 0 or context.proposed_quantity <= 0 or not self.admissible(context):
            return None
        instance_id = f"{self.strategy_id}-{uuid4().hex}"
        product = default_product(self.direction)
        # Stop and target are placed by direction rather than by a hardcoded
        # (1 - stop) / (1 + profit): for a short the profit target sits BELOW the
        # entry and the stop ABOVE it. Getting this pair backwards would arm a
        # position whose "stop" is its target, i.e. one that exits on the winning
        # side and runs on the losing one.
        return TradePlan(
            strategy_id=self.strategy_id,
            strategy_instance_id=instance_id,
            symbol=context.symbol,
            side=broker_side(self.direction, PositionEffect.OPEN),
            thesis=self.thesis,
            entry_trigger={"kind": "robust_quantile", "as_of": context.as_of.isoformat()},
            entry_price_policy={"kind": "passive_limit", "reference": context.price},
            proposed_quantity=context.proposed_quantity,
            initial_stop={
                "price": stop_price(
                    context.price, self.config.stop_bps / 10_000, self.direction
                ),
                "bps": self.config.stop_bps,
            },
            profit_policy={
                "price": target_price(
                    context.price, self.config.profit_bps / 10_000, self.direction
                ),
                "bps": self.config.profit_bps,
            },
            trailing_policy={
                "bps": self.config.trailing_bps,
                "minimum_net_bps": self.config.minimum_trailing_net_bps,
            },
            max_holding_seconds=self.config.max_holding_seconds,
            invalidation_conditions=(
                ("DATA_STALE", "ONTOLOGY_BLOCKED", "THESIS_INVALIDATED")
                if self.direction is PositionDirection.LONG
                # A borrow that disappears or is recalled invalidates a short thesis
                # regardless of whether the price thesis still holds.
                else (
                    "DATA_STALE",
                    "ONTOLOGY_BLOCKED",
                    "THESIS_INVALIDATED",
                    "BORROW_UNAVAILABLE",
                    "BORROW_RECALLED",
                )
            ),
            max_entry_slippage_bps=self.config.max_entry_slippage_bps,
            expires_at=context.as_of + timedelta(seconds=self.config.entry_ttl_seconds),
            feature_snapshot_id=context.feature_snapshot_id,
            utility_evidence_id=context.utility_evidence_id,
            position_direction=str(self.direction),
            position_effect=str(PositionEffect.OPEN),
            execution_product=str(product),
            # A proposal is not an authorisation. Every plan leaves here marked
            # SHADOW; the submitter reads the authoritative deployment state from
            # the promotion controller and will refuse anything not cleared there.
            deployment_state=str(StrategyDeploymentState.SHADOW),
        )


class IntradayMomentumExpert(StrategyExpert):
    strategy_id = "intraday_momentum"
    thesis = "robust intraday return continuation with volume confirmation"
    default_config = _geometry_config("intraday_momentum")

    def admissible(self, c: ExpertContext) -> bool:
        return (
            c.q("return") >= self.config.entry_quantile
            and c.q("volume") >= self.config.confirmation_quantile
        )


class BreakoutVolumeExpert(StrategyExpert):
    strategy_id = "breakout_volume"
    thesis = "causal range breakout confirmed by unusual volume"
    default_config = _geometry_config("breakout_volume")

    def admissible(self, c: ExpertContext) -> bool:
        return (
            c.q("breakout") >= self.config.entry_quantile
            and c.q("volume") >= self.config.entry_quantile
            # A one-bar range break in a thin book was the dominant losing
            # pattern in the stored replay.  Require both directional follow-
            # through and an executable spread/depth state before calling it a
            # breakout rather than a transient print.
            and c.q("return") >= self.config.confirmation_quantile
            and c.q("liquidity") >= self.config.confirmation_quantile
        )


class VwapMeanReversionExpert(StrategyExpert):
    strategy_id = "vwap_mean_reversion"
    thesis = "liquid downside displacement from VWAP with reversion confirmation"
    default_config = _geometry_config("vwap_mean_reversion")

    def admissible(self, c: ExpertContext) -> bool:
        return (
            c.q("vwap_deviation") <= 1 - self.config.entry_quantile
            and c.q("reversion") >= self.config.confirmation_quantile
            # Mean reversion is only executable when the displacement occurs
            # in an actively traded, tight-spread state.  Without these two
            # confirmations the GNN repeatedly elected stale/thin symbols whose
            # apparent edge disappeared after realistic execution costs.
            and c.q("volume") >= self.config.confirmation_quantile
            and c.q("liquidity") >= self.config.confirmation_quantile
        )


class LiquidityShockReversalExpert(StrategyExpert):
    strategy_id = "liquidity_shock_reversal"
    thesis = "temporary liquidity shock reversal after spread/depth normalization"
    default_config = _geometry_config("liquidity_shock_reversal")

    def admissible(self, c: ExpertContext) -> bool:
        return (
            c.q("liquidity_shock") >= self.config.entry_quantile
            and c.q("price_drop") >= self.config.entry_quantile
            and c.q("recovery") >= self.config.confirmation_quantile
            # The thesis is specifically a *temporary* shock. Price bouncing
            # while the spread is still blown out is adverse selection, not
            # normalization, and its apparent gross edge did not survive costs.
            and c.q("liquidity_recovery") >= self.config.confirmation_quantile
            and c.q("liquidity") >= self.config.confirmation_quantile
        )


class EventMomentumExpert(StrategyExpert):
    strategy_id = "event_momentum"
    thesis = "fresh high-relevance event continuation with market confirmation"
    default_config = _geometry_config("event_momentum")

    def admissible(self, c: ExpertContext) -> bool:
        return c.q("event_relevance") >= self.config.entry_quantile and c.q(
            "event_direction"
        ) >= self.config.confirmation_quantile


class CrossSectionalRelativeStrengthExpert(StrategyExpert):
    strategy_id = "cross_sectional_relative_strength"
    thesis = "sector-neutral relative strength with sufficient liquidity"
    default_config = _geometry_config("cross_sectional_relative_strength")

    def admissible(self, c: ExpertContext) -> bool:
        return c.q("relative_strength") >= self.config.entry_quantile and c.q(
            "liquidity"
        ) >= self.config.confirmation_quantile


class GapContextExpert(StrategyExpert):
    strategy_id = "gap_context"
    thesis = "opening gap continuation only after price-discovery confirmation"
    default_config = _geometry_config("gap_context")

    def admissible(self, c: ExpertContext) -> bool:
        return (
            c.q("gap") >= self.config.entry_quantile
            and c.q("opening_confirmation") >= self.config.confirmation_quantile
            and c.q("gap_entry_window") >= self.config.entry_quantile
        )


class RvgiBoxBreakoutExpert(StrategyExpert):
    strategy_id = "rvgi_box_breakout"
    thesis = "RVGI-confirmed causal price-box breakout with volume acceptance"
    default_config = _geometry_config("rvgi_box_breakout")

    def admissible(self, c: ExpertContext) -> bool:
        return (
            c.q("rvgi_diff") >= self.config.confirmation_quantile
            # A persistent positive RVGI state produced more samples but a
            # negative gross edge in replay. Keep the causal cross as the timing
            # event and solve its fragility with execution quality, not looseness.
            and c.q("rvgi_cross") >= self.config.entry_quantile
            and c.q("box_position") >= self.config.entry_quantile
            and c.q("volume") >= self.config.confirmation_quantile
            and c.q("liquidity") >= self.config.confirmation_quantile
            and c.q("false_breakout_risk") <= 1 - self.config.confirmation_quantile
        )


class ResidualRelativeStrengthExpert(StrategyExpert):
    strategy_id = "residual_relative_strength"
    thesis = "market/sector-neutral residual strength confirmed by informed flow"
    default_config = _geometry_config("residual_relative_strength")

    def admissible(self, c: ExpertContext) -> bool:
        # Both residual horizons must be strong: one window alone cannot separate
        # persistent idiosyncratic strength from a single-window bounce. Investor
        # flow is required, not optional — residual strength with no informed flow
        # behind it is usually a squeeze.
        return (
            c.q("residual_strength_short") >= self.config.entry_quantile
            and c.q("residual_strength_long") >= self.config.confirmation_quantile
            and c.q("investor_flow") >= self.config.confirmation_quantile
            and c.q("liquidity") >= self.config.confirmation_quantile
        )


class AdaptiveAnchoredVwapReversionExpert(StrategyExpert):
    strategy_id = "adaptive_anchored_vwap_reversion"
    thesis = "volatility-normalised displacement below anchored VWAP with liquidity returning"
    default_config = _geometry_config("adaptive_anchored_vwap_reversion")

    def admissible(self, c: ExpertContext) -> bool:
        return (
            # Low quantile == deeply displaced below the anchor.
            c.q("vwap_zscore") <= 1 - self.config.entry_quantile
            and c.q("liquidity_recovery") >= self.config.confirmation_quantile
            and c.q("microprice_edge") >= self.config.confirmation_quantile
            and c.q("liquidity") >= self.config.confirmation_quantile
        )


class OfiMicropriceExhaustionReversalExpert(StrategyExpert):
    strategy_id = "ofi_microprice_exhaustion_reversal"
    thesis = "sell-side exhaustion confirmed by order flow, depth recovery and microprice"
    default_config = _geometry_config("ofi_microprice_exhaustion_reversal")

    def admissible(self, c: ExpertContext) -> bool:
        return (
            c.q("price_drop") >= self.config.entry_quantile
            and c.q("ofi_slope") >= self.config.entry_quantile
            and c.q("depth_recovery") >= self.config.confirmation_quantile
            and c.q("microprice_edge") >= self.config.confirmation_quantile
            # Flow toxicity is a risk statement: a toxic tape means the
            # counterparty is better informed than this thesis is.
            and c.q("flow_toxicity") <= 1 - self.config.confirmation_quantile
        )


class OpeningRangeBreakoutExpert(StrategyExpert):
    strategy_id = "opening_range_breakout"
    thesis = "opening-range breakout restricted to stocks in play by relative volume"
    default_config = _geometry_config("opening_range_breakout")

    def admissible(self, c: ExpertContext) -> bool:
        # The relative-volume gate is load-bearing, not a filter bolted on for
        # tidiness. In the published results the unrestricted opening-range
        # breakout does not pay, and restricting it to the highest relative-volume
        # names is what produces the edge; practitioner studies put the useful
        # threshold at roughly 1.5-2x average volume. So RVOL is required at the
        # same strength as the breakout itself.
        return (
            c.q("opening_range_breakout") >= self.config.entry_quantile
            and c.q("relative_volume") >= self.config.entry_quantile
            and c.q("volume") >= self.config.confirmation_quantile
            and c.q("liquidity") >= self.config.confirmation_quantile
        )


class MarketIntradayMomentumExpert(StrategyExpert):
    strategy_id = "market_intraday_momentum"
    thesis = "positive first half-hour return continues into the last half-hour"
    default_config = _geometry_config("market_intraday_momentum")

    def admissible(self, c: ExpertContext) -> bool:
        # Long-only, so only the POSITIVE leg of the published effect is tradable:
        # a negative first half-hour predicts a negative last half-hour, which this
        # account cannot express (see the short-side analysis — retail KRX shorting
        # needs 대주 and adds borrow cost to the one thing already binding).
        return (
            # >0.5 means the first half-hour return was positive.
            c.q("intraday_momentum_signal") >= self.config.entry_quantile
            # Only inside the last continuous half-hour; outside it there is nothing
            # to trade, and after 15:20 KRX is in a closing auction.
            and c.q("intraday_momentum_window") >= self.config.entry_quantile
            # The effect is strongest on volatile days, and — independently — only a
            # volatile day moves far enough to clear a ~33bps round trip. Both
            # reasons point the same way, so this is a hard precondition.
            and c.q("first_half_hour_volatility") >= self.config.confirmation_quantile
            and c.q("liquidity") >= self.config.confirmation_quantile
        )


# --------------------------------------------------------------------------- #
# SHORT-side experts                                                           #
# --------------------------------------------------------------------------- #
# Every one of these adds a borrow precondition that has no long-side analogue.
# ``borrow_available`` is not a score contribution and not a preference: a short
# with no locatable borrow is not a worse trade, it is not a trade. Treating it as
# a soft factor is how a signal becomes an order the broker rejects — or worse,
# accepts and then forcibly closes.
class ShortStrategyExpert(StrategyExpert):
    """Base for short theses: enforces the borrow precondition uniformly."""

    direction = PositionDirection.SHORT

    def borrow_locatable(self, c: ExpertContext) -> bool:
        # Deliberately a HIGH bar on a quantile that means "borrow comfortably
        # available", and absent quantiles default to 0.5 in ``ExpertContext.q``,
        # which is below the bar. So "we did not ask" reads as "not available"
        # rather than as "available" — fail-closed on the one input whose absence
        # is unrecoverable at execution time.
        return c.q("borrow_available") >= self.config.entry_quantile

    def admissible(self, c: ExpertContext) -> bool:
        return self.borrow_locatable(c) and self.thesis_admissible(c)

    def thesis_admissible(self, c: ExpertContext) -> bool:
        raise NotImplementedError


class MarketIntradayMomentumShortExpert(ShortStrategyExpert):
    strategy_id = "market_intraday_momentum_short"
    thesis = "negative first half-hour return continues into the last half-hour"
    default_config = _geometry_config("market_intraday_momentum_short")

    def thesis_admissible(self, c: ExpertContext) -> bool:
        # The NEGATIVE leg of the same published effect the long expert takes the
        # positive leg of (Gao/Han/Li/Zhou). Both legs are in the paper; only the
        # positive one was expressible before, because the negative one needs 대주.
        #
        # This is not "the long expert with the sign flipped": it is the other half
        # of a two-sided finding, and it is evaluated on its own forward outcomes
        # because the borrow cost falls only on this side.
        return (
            # Low quantile == the first half-hour return was strongly negative.
            c.q("intraday_momentum_signal") <= 1 - self.config.entry_quantile
            and c.q("intraday_momentum_window") >= self.config.entry_quantile
            # Same volatility precondition, and it binds harder here: the move must
            # clear a round trip PLUS borrow.
            and c.q("first_half_hour_volatility") >= self.config.confirmation_quantile
            and c.q("liquidity") >= self.config.confirmation_quantile
        )


class OpeningRangeBreakdownExpert(ShortStrategyExpert):
    strategy_id = "opening_range_breakdown"
    thesis = "opening-range breakdown on high relative volume with sell-side aggression"
    default_config = _geometry_config("opening_range_breakdown")

    def thesis_admissible(self, c: ExpertContext) -> bool:
        # The relative-volume "stocks in play" restriction is load-bearing on this
        # side too, for the same reason as the breakout: the unrestricted version
        # does not pay. Sell-side aggression is required because a breakdown into
        # buying is the mirror of the classic false break.
        return (
            c.q("opening_range_breakdown") >= self.config.entry_quantile
            and c.q("relative_volume") >= self.config.entry_quantile
            and c.q("aggressor_imbalance") <= 1 - self.config.confirmation_quantile
            and c.q("liquidity") >= self.config.confirmation_quantile
            # A breakdown in a name that has already been squeezed once is where
            # short losses come from, so squeeze risk is a hard exclusion.
            and c.q("squeeze_risk") <= 1 - self.config.confirmation_quantile
        )


class ResidualRelativeWeaknessExpert(ShortStrategyExpert):
    strategy_id = "residual_relative_weakness"
    thesis = "idiosyncratic weakness net of market and sector beta persists while flow confirms it"
    default_config = _geometry_config("residual_relative_weakness")

    def thesis_admissible(self, c: ExpertContext) -> bool:
        # Beta-neutral by construction, which is the point: it is the one short
        # thesis here that does not need the index to fall, so it stays valid in a
        # rising market where a directional short is fighting the tape.
        return (
            c.q("residual_strength_short") <= 1 - self.config.entry_quantile
            and c.q("residual_strength_long") <= 1 - self.config.confirmation_quantile
            and c.q("investor_flow") <= 1 - self.config.confirmation_quantile
            and c.q("liquidity") >= self.config.confirmation_quantile
            and c.q("squeeze_risk") <= 1 - self.config.confirmation_quantile
        )


class BarConfirmedVwapRecoveryExpert(StrategyExpert):
    strategy_id = "bar_confirmed_vwap_recovery"
    thesis = "deep VWAP displacement recovers after a completed one-minute trend turn"
    default_config = _geometry_config("bar_confirmed_vwap_recovery")

    def admissible(self, c: ExpertContext) -> bool:
        # The robust-quantile expert is the model-side admissibility contract.
        # Exact point-in-time thresholds (EMA reclaim, MACD turn, spread and
        # liquidity) are independently enforced by the owned algorithm before
        # any proposal can be armed.
        return (
            c.q("vwap_deviation") <= 1 - self.config.entry_quantile
            and c.q("reversion") >= self.config.confirmation_quantile
            and c.q("liquidity") >= self.config.confirmation_quantile
        )


class OvernightGapCarryExpert(StrategyExpert):
    strategy_id = "overnight_gap_carry"
    thesis = "a closing drive on a volatile day carries through the overnight gap"
    default_config = _geometry_config("overnight_gap_carry")

    def admissible(self, c: ExpertContext) -> bool:
        # Every quantile here already exists, deliberately: a new thesis must not
        # widen the model's input schema, or the checkpoint contract breaks for a
        # reason that has nothing to do with the thesis.
        return (
            # The carry is a decision about the close, so only near it. The window
            # is now the SYMBOL's exchange clock, which is what makes this thesis
            # expressible on US names at all.
            c.q("overnight_carry_window") >= self.config.entry_quantile
            # Closing at the top of its own recent range: the observable form of
            # "buyers ended the day in control". Deliberately NOT ``vwap_deviation``
            # -- that key is the reversion family's displacement, where strength is
            # a LOW value, and borrowing it would have this thesis read the sign
            # backwards. The serving-side algorithm states the same condition as a
            # positive VWAP premium plus buy-side aggressor flow.
            and c.q("breakout") >= self.config.entry_quantile
            # A quiet day's gap does not clear a 51bps round trip. This is the
            # cost condition and the thesis condition at once.
            and c.q("first_half_hour_volatility") >= self.config.confirmation_quantile
            and c.q("momentum_persistence_long") >= self.config.confirmation_quantile
            and c.q("liquidity") >= self.config.confirmation_quantile
        )


ALL_EXPERT_TYPES = (
    IntradayMomentumExpert,
    BreakoutVolumeExpert,
    VwapMeanReversionExpert,
    LiquidityShockReversalExpert,
    EventMomentumExpert,
    CrossSectionalRelativeStrengthExpert,
    GapContextExpert,
    RvgiBoxBreakoutExpert,
    ResidualRelativeStrengthExpert,
    AdaptiveAnchoredVwapReversionExpert,
    OfiMicropriceExhaustionReversalExpert,
    OpeningRangeBreakoutExpert,
    MarketIntradayMomentumExpert,
    MarketIntradayMomentumShortExpert,
    OpeningRangeBreakdownExpert,
    ResidualRelativeWeaknessExpert,
    BarConfirmedVwapRecoveryExpert,
    OvernightGapCarryExpert,
)

assert tuple(kind.strategy_id for kind in ALL_EXPERT_TYPES) == STRATEGY_IDS


class OwnedStrategyLifecycle:
    def __init__(self, plan: TradePlan) -> None:
        self.plan = plan
        self.direction = parse_direction(plan.position_direction)

    def entry_intent(self, created_at: datetime) -> OrderIntent:
        # OPEN, not "BUY". For a short thesis the opening order is a SELL, and
        # hardcoding BUY here would have opened a long against a short plan.
        return self._intent(
            action=_intent_action(self.direction, PositionEffect.OPEN),
            quantity=self.plan.proposed_quantity,
            created_at=created_at,
            reason="PLAN_ENTRY",
            position_id=None,
        )

    def exit_intent(
        self,
        *,
        position_id: str,
        quantity: int,
        price: float,
        opened_at: datetime,
        as_of: datetime,
        invalidated: bool = False,
        data_stale: bool = False,
    ) -> OrderIntent | None:
        stop = float(self.plan.initial_stop["price"])
        target = float(self.plan.profit_policy["price"])
        reason = None
        if data_stale:
            reason = "DATA_STALE_FAIL_SAFE"
        elif invalidated:
            reason = "THESIS_INVALIDATED"
        # Barrier comparisons are directional. ``price <= stop`` is a stop-out for a
        # long and a PROFIT for a short; using the long comparison on a short plan
        # would have exited every winner at its stop and held every loser.
        elif stop_breached(price, stop, self.direction):
            reason = "INITIAL_STOP"
        elif target_reached(price, target, self.direction):
            reason = "PROFIT_TARGET"
        elif (as_of - opened_at).total_seconds() >= self.plan.max_holding_seconds:
            reason = "MAX_HOLDING_TIME"
        if reason is None:
            return None
        return self._intent(
            action=_intent_action(self.direction, PositionEffect.CLOSE),
            quantity=quantity,
            created_at=as_of,
            reason=reason,
            position_id=position_id,
        )

    def _intent(
        self,
        *,
        action: IntentAction,
        quantity: int,
        created_at: datetime,
        reason: str,
        position_id: str | None,
    ) -> OrderIntent:
        intent_id = f"intent-{uuid4().hex}"
        return OrderIntent(
            intent_id=intent_id,
            idempotency_key=f"{self.plan.strategy_instance_id}:{intent_id}",
            strategy_instance_id=self.plan.strategy_instance_id,
            position_id_if_any=position_id,
            symbol=self.plan.symbol,
            action=action,
            quantity=quantity,
            limit_or_price_policy=self.plan.entry_price_policy,
            # Urgency belongs to the EFFECT, not to the broker side: exiting is
            # urgent, entering is not. Keyed off SELL, a short's buy-to-cover — the
            # one exit whose loss is unbounded if it is slow — would have been sent
            # at NORMAL urgency.
            urgency="HIGH" if reason != "PLAN_ENTRY" else "NORMAL",
            reason_code=reason,
            created_at=created_at,
            expires_at=created_at + timedelta(seconds=5),
        )

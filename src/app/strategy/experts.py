from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Mapping
from uuid import uuid4

from app.trading.contracts import IntentAction, OrderIntent, TradePlan
from app.strategy.catalog import STRATEGY_IDS


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


class StrategyExpert:
    strategy_id = "base"
    thesis = "base"
    default_config = ExpertConfig()

    def __init__(self, config: ExpertConfig | None = None) -> None:
        self.config = config or self.default_config

    def admissible(self, context: ExpertContext) -> bool:
        raise NotImplementedError

    def propose(self, context: ExpertContext) -> TradePlan | None:
        if context.price <= 0 or context.proposed_quantity <= 0 or not self.admissible(context):
            return None
        instance_id = f"{self.strategy_id}-{uuid4().hex}"
        return TradePlan(
            strategy_id=self.strategy_id,
            strategy_instance_id=instance_id,
            symbol=context.symbol,
            side="BUY",
            thesis=self.thesis,
            entry_trigger={"kind": "robust_quantile", "as_of": context.as_of.isoformat()},
            entry_price_policy={"kind": "passive_limit", "reference": context.price},
            proposed_quantity=context.proposed_quantity,
            initial_stop={
                "price": context.price * (1 - self.config.stop_bps / 10_000),
                "bps": self.config.stop_bps,
            },
            profit_policy={
                "price": context.price * (1 + self.config.profit_bps / 10_000),
                "bps": self.config.profit_bps,
            },
            trailing_policy={"bps": self.config.trailing_bps},
            max_holding_seconds=self.config.max_holding_seconds,
            invalidation_conditions=("DATA_STALE", "ONTOLOGY_BLOCKED", "THESIS_INVALIDATED"),
            max_entry_slippage_bps=self.config.max_entry_slippage_bps,
            expires_at=context.as_of + timedelta(seconds=self.config.entry_ttl_seconds),
            feature_snapshot_id=context.feature_snapshot_id,
            utility_evidence_id=context.utility_evidence_id,
        )


class IntradayMomentumExpert(StrategyExpert):
    strategy_id = "intraday_momentum"
    thesis = "robust intraday return continuation with volume confirmation"
    default_config = ExpertConfig(stop_bps=22, profit_bps=100, max_holding_seconds=1800)

    def admissible(self, c: ExpertContext) -> bool:
        return c.q("return") >= self.config.entry_quantile and c.q(
            "volume"
        ) >= self.config.confirmation_quantile


class BreakoutVolumeExpert(StrategyExpert):
    strategy_id = "breakout_volume"
    thesis = "causal range breakout confirmed by unusual volume"
    default_config = ExpertConfig(stop_bps=25, profit_bps=120, max_holding_seconds=2700)

    def admissible(self, c: ExpertContext) -> bool:
        return c.q("breakout") >= self.config.entry_quantile and c.q(
            "volume"
        ) >= self.config.entry_quantile


class VwapMeanReversionExpert(StrategyExpert):
    strategy_id = "vwap_mean_reversion"
    thesis = "liquid downside displacement from VWAP with reversion confirmation"
    default_config = ExpertConfig(stop_bps=18, profit_bps=100, max_holding_seconds=1800)

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
    default_config = ExpertConfig(stop_bps=30, profit_bps=100, max_holding_seconds=1200)

    def admissible(self, c: ExpertContext) -> bool:
        return (
            c.q("liquidity_shock") >= self.config.entry_quantile
            and c.q("price_drop") >= self.config.entry_quantile
            and c.q("recovery") >= self.config.confirmation_quantile
        )


class EventMomentumExpert(StrategyExpert):
    strategy_id = "event_momentum"
    thesis = "fresh high-relevance event continuation with market confirmation"
    default_config = ExpertConfig(stop_bps=35, profit_bps=140, max_holding_seconds=3600)

    def admissible(self, c: ExpertContext) -> bool:
        return c.q("event_relevance") >= self.config.entry_quantile and c.q(
            "event_direction"
        ) >= self.config.confirmation_quantile


class CrossSectionalRelativeStrengthExpert(StrategyExpert):
    strategy_id = "cross_sectional_relative_strength"
    thesis = "sector-neutral relative strength with sufficient liquidity"
    default_config = ExpertConfig(stop_bps=28, profit_bps=120, max_holding_seconds=3600)

    def admissible(self, c: ExpertContext) -> bool:
        return c.q("relative_strength") >= self.config.entry_quantile and c.q(
            "liquidity"
        ) >= self.config.confirmation_quantile


class GapContextExpert(StrategyExpert):
    strategy_id = "gap_context"
    thesis = "opening gap continuation only after price-discovery confirmation"
    default_config = ExpertConfig(stop_bps=32, profit_bps=130, max_holding_seconds=2700)

    def admissible(self, c: ExpertContext) -> bool:
        return c.q("gap") >= self.config.entry_quantile and c.q(
            "opening_confirmation"
        ) >= self.config.confirmation_quantile


class RvgiBoxBreakoutExpert(StrategyExpert):
    strategy_id = "rvgi_box_breakout"
    thesis = "RVGI-confirmed causal price-box breakout with volume acceptance"
    default_config = ExpertConfig(stop_bps=20, profit_bps=120, max_holding_seconds=1800)

    def admissible(self, c: ExpertContext) -> bool:
        return (
            c.q("rvgi_diff") >= self.config.confirmation_quantile
            and c.q("rvgi_cross") >= self.config.entry_quantile
            and c.q("box_position") >= self.config.entry_quantile
            and c.q("volume") >= self.config.confirmation_quantile
            and c.q("false_breakout_risk") <= 1 - self.config.confirmation_quantile
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
)

assert tuple(kind.strategy_id for kind in ALL_EXPERT_TYPES) == STRATEGY_IDS


class OwnedStrategyLifecycle:
    def __init__(self, plan: TradePlan) -> None:
        self.plan = plan

    def entry_intent(self, created_at: datetime) -> OrderIntent:
        return self._intent(
            action=IntentAction.BUY,
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
        elif price <= stop:
            reason = "INITIAL_STOP"
        elif price >= target:
            reason = "PROFIT_TARGET"
        elif (as_of - opened_at).total_seconds() >= self.plan.max_holding_seconds:
            reason = "MAX_HOLDING_TIME"
        if reason is None:
            return None
        return self._intent(
            action=IntentAction.SELL,
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
            urgency="HIGH" if action == IntentAction.SELL else "NORMAL",
            reason_code=reason,
            created_at=created_at,
            expires_at=created_at + timedelta(seconds=5),
        )

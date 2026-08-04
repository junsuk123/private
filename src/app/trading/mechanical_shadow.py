from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from app.cost.trading_cost_engine import TradingCostEngine
from app.data.realtime_store import RealtimeMarketDataStore
from app.features.live_feature_frame import LiveFeatureFrame
from app.strategy.exit_geometry import exit_geometry
from app.technical.feature_builder import technical_feature_set_from_live_frame
from app.technical.strategy_algorithms import ElectionContext, get_algorithm
from app.trading.directional import DirectionalStrategyKey
from app.trading.directional_shadow import ShadowTradePlan, ShadowPlanStore, default_shadow_store
from app.trading.shadow_evaluation_service import (
    ShadowEvaluationService,
    default_shadow_evaluation_service,
)


@dataclass(frozen=True)
class MechanicalShadowCollection:
    evaluated: int = 0
    triggered: int = 0
    recorded: int = 0
    adopted: int = 0


class MechanicalShadowCollector:
    """Collect causal forward samples without creating an executable order.

    This deliberately depends only on validated feature frames, the read-only
    market-data store and the shadow journal/simulator.  It has no broker or
    execution-coordinator dependency, so an eligible signal cannot escape into
    the live order path.  For now only the US ask-heavy absorption submode is
    collected; the ordinary liquidity-shock branch remains on its existing path.
    """

    strategy_id = "liquidity_shock_reversal"

    def __init__(
        self,
        *,
        market_store: RealtimeMarketDataStore | None = None,
        shadow_store: ShadowPlanStore | None = None,
        evaluation_service: ShadowEvaluationService | None = None,
        cost_engine: TradingCostEngine | None = None,
        cooldown_seconds: int | None = None,
        notional_usd: float | None = None,
    ) -> None:
        self.market_store = market_store or RealtimeMarketDataStore()
        self.shadow_store = shadow_store or default_shadow_store()
        self.evaluation_service = evaluation_service or default_shadow_evaluation_service()
        self.cost_engine = cost_engine or TradingCostEngine()
        self.cooldown_seconds = max(
            1,
            int(cooldown_seconds or os.getenv("MECHANICAL_SHADOW_COOLDOWN_SECONDS", "600")),
        )
        self.notional_usd = max(
            1.0,
            float(notional_usd or os.getenv("MECHANICAL_SHADOW_NOTIONAL_USD", "1000")),
        )
        self._lock = threading.RLock()
        self._last_signal_at: dict[tuple[str, str], datetime] = {}

    def collect(
        self,
        frames: Iterable[LiveFeatureFrame],
        *,
        observed_at: datetime | None = None,
    ) -> MechanicalShadowCollection:
        now = _aware(observed_at or datetime.now(timezone.utc))
        evaluated = triggered = recorded = adopted = 0
        algorithm = get_algorithm(self.strategy_id)
        if algorithm is None:
            return MechanicalShadowCollection()

        for frame in frames:
            evaluated += 1
            symbol = str(frame.symbol or "").strip().upper()
            if not symbol or (symbol.isdigit() and len(symbol) == 6):
                continue
            features = technical_feature_set_from_live_frame(frame, symbol)
            context = ElectionContext(
                strategy_id=self.strategy_id,
                elected_at=now,
                reference_price=frame.mark_price or None,
            )
            decision = algorithm.entry(features, context)
            # Shadow collection measures the raw thesis, including cases where
            # sparse REST ticks cannot produce a volatility-based expected edge.
            # The algorithm preserves the absorption reason when its edge floor
            # rejects live entry, so collect that fact without weakening the live
            # decision (which remains HOLD and live_authorized=0).
            if "ASK_HEAVY_ABSORPTION_CONFIRMED" not in decision.reason_codes:
                continue
            triggered += 1

            book = self.market_store.latest_orderbook(symbol)
            if book is None or book.best_bid <= 0 or book.best_ask <= 0 or book.best_ask < book.best_bid:
                continue
            # Never attach a quote that arrived after the feature decision. That
            # would leak future information into the frozen signal-time plan.
            if _aware(book.received_at) > _aware(frame.decision_time):
                continue

            key = (self.strategy_id, symbol)
            with self._lock:
                previous = self._last_signal_at.get(key)
                if previous is not None and now - previous < timedelta(seconds=self.cooldown_seconds):
                    continue

                geometry = exit_geometry(self.strategy_id)
                quantity = max(1, int(self.notional_usd / book.best_ask))
                target_rate = geometry.take_profit_bps / 10_000.0
                stop_rate = geometry.stop_loss_bps / 10_000.0
                # A raw absorption match can be rejected by the live expected-edge
                # floor. That rejection carries the base algorithm horizon (120s),
                # not the absorption thesis horizon. Shadow evaluation must still
                # measure the configured 600-second thesis rather than silently
                # truncating it to the unrelated shock horizon.
                holding_seconds = min(
                    geometry.max_holding_seconds,
                    max(1, int(algorithm.p("absorption_horizon_seconds"))),
                )
                cost = self.cost_engine.estimate(
                    symbol=symbol,
                    market="US",
                    venue="NASD",
                    instrument_type="overseas_stock",
                    entry_price=book.best_ask,
                    expected_exit_price=book.best_ask * (1.0 + target_rate),
                    quantity=quantity,
                    orderbook_snapshot={
                        "bid_price": book.best_bid,
                        "ask_price": book.best_ask,
                    },
                )
                cost_bps = cost.total_cost_rate * 10_000.0
                plan = ShadowTradePlan(
                    plan_id="",
                    key=DirectionalStrategyKey.for_long(self.strategy_id, "US"),
                    symbol=symbol,
                    signal_at=now,
                    entry_reference_price=book.best_ask,
                    target_rate=target_rate,
                    stop_rate=stop_rate,
                    max_holding_seconds=holding_seconds,
                    expected_trading_cost_bps=cost_bps,
                    predicted_gross_edge_bps=decision.expected_edge_bps,
                    predicted_net_edge_bps=decision.expected_edge_bps - cost_bps,
                    predicted_success_probability=decision.confidence,
                    regime="MECHANICAL_ABSORPTION_SHADOW",
                    signal_reason_codes=(
                        *decision.reason_codes,
                        "GNN_INDEPENDENT_MECHANICAL_SHADOW",
                        "ORDER_SUBMISSION_DISABLED",
                    ),
                    intended_quantity=quantity,
                    spread_bps_at_signal=book.spread_bps,
                    liquidity_score_at_signal=features.liquidity_score,
                    feature_snapshot_id=frame.provenance.orderbook_record_id or book.record_id,
                    model_version="mechanical-absorption-v1",
                    deployment_state="SHADOW",
                    diagnostics={
                        "order_submission_capable": False,
                        "algorithm_live_triggered": decision.triggered,
                        "source": "validated_live_feature_frame",
                        "return_30s_bps": (features.return_30s or 0.0) * 10_000.0,
                        "orderbook_imbalance": features.orderbook_imbalance,
                        "spread_change_5s": features.spread_change_5s,
                        "target_basis": "strategy_exit_geometry",
                        "stop_basis": "strategy_exit_geometry",
                    },
                )
                if not self.shadow_store.record_plan(plan):
                    continue
                recorded += 1
                # A full evaluator may leave this journaled plan unresolved, but
                # must not cause another durable plan every polling cycle.
                self._last_signal_at[key] = now
                adopted_count, _ = self.evaluation_service.adopt((plan,))
                adopted += adopted_count

        return MechanicalShadowCollection(evaluated, triggered, recorded, adopted)


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


_DEFAULT_COLLECTOR: MechanicalShadowCollector | None = None
_DEFAULT_COLLECTOR_LOCK = threading.Lock()


def default_mechanical_shadow_collector() -> MechanicalShadowCollector:
    global _DEFAULT_COLLECTOR
    if _DEFAULT_COLLECTOR is None:
        with _DEFAULT_COLLECTOR_LOCK:
            if _DEFAULT_COLLECTOR is None:
                _DEFAULT_COLLECTOR = MechanicalShadowCollector()
    return _DEFAULT_COLLECTOR

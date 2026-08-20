"""Live wiring: real data sources -> context hierarchy -> decision pipeline.

Everything above this module is pure and testable. This is where the impurity lives — the
stores are read, the clock is taken, and the result is cached for the API and the
dashboard.

Honest degradation
------------------
A source that cannot answer produces ``None`` and a reason code, never a substitute
number. The consequences travel with it: the context's ``confidence`` falls, the
freshness registry marks the stream, and the gate blocks on the critical ones. Nothing
here manufactures a value to keep a dashboard tidy — a readiness figure computed from
invented inputs is exactly the "100% ready with a core module OFFLINE" contradiction this
refactor exists to remove.

Threading
---------
:meth:`ContextRuntime.refresh` is called from the trading loop or a sampler thread and is
guarded by a lock; the API reads the cached result without blocking the writer.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

from app.context.domestic_context import (
    DomesticContextBuilder,
    DomesticContextInputs,
    VenueQuote,
)
from app.context.global_context import (
    GlobalContext,
    GlobalContextBuilder,
    IndicatorObservation,
)
from app.context.sector_context import (
    SectorContext,
    SectorContextBuilder,
    SectorMemberObservation,
)
from app.context.temporal_context import TemporalSnapshot, build_temporal_snapshot
from app.data.freshness import DataFreshnessRegistry, default_freshness_registry
from app.execution.order_state_machine import OrderStateMachine
from app.models.gnn_runtime import GnnRuntime
from app.models.graph_snapshot import FEATURE_DIM, GraphSnapshotBuilder
from app.models.temporal_hetero_gnn import TemporalHeteroGnnConfig
from app.storage.trading_state_store import (
    TradingStateStore,
    default_trading_state_store,
    iso_column,
)
from app.trading.context_decision_pipeline import (
    AccountState,
    CandidateInput,
    ContextDecisionPipeline,
    CycleResult,
)

__all__ = [
    "ContextRuntime",
    "GRAPH_MAX_NODES",
    "GRAPH_TIME_STEPS",
    "default_context_runtime",
    "reset_default_context_runtime",
]

#: Node budget for the market graph. The static ontology is ~46 nodes, leaving room for
#: roughly 40 sectors and candidates before truncation is reported.
GRAPH_MAX_NODES = 96

#: Window the TCN sees. Eight one-minute steps: long enough for the dilations (1, 2, 4)
#: to reach back a full session quarter-hour, short enough that a candidate set which
#: changes between cycles still shares most of its history.
GRAPH_TIME_STEPS = 8

#: How long a cached cycle is served before the API reports it as stale.
CYCLE_STALE_SECONDS = 300.0

#: FRED / ECOS series name -> global indicator. The research collectors store macro
#: metrics under these names (see ``config/research_sources.live.json``).
_MACRO_SERIES_TO_INDICATOR: dict[str, str] = {
    "us_vix_close": "VIX",
    "us_treasury_10y_yield": "US10Y",
    "us_broad_dollar_index": "DXY",
}


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class RuntimeStatus:
    """What the runtime itself is doing, separate from what it observed."""

    last_refresh_at: datetime | None
    last_cycle_id: str | None
    refresh_count: int
    error_count: int
    last_error: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "last_refresh_at": iso_column(self.last_refresh_at),
            "last_cycle_id": self.last_cycle_id,
            "refresh_count": self.refresh_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
        }


class ContextRuntime:
    """Owns the pipeline singletons and the most recent cycle."""

    def __init__(
        self,
        *,
        store: TradingStateStore | None = None,
        freshness: DataFreshnessRegistry | None = None,
        gnn_runtime: GnnRuntime | None = None,
        pipeline: ContextDecisionPipeline | None = None,
        state_machine: OrderStateMachine | None = None,
        require_checkpoint: bool = True,
        market_group: str = "KR",
        session_snapshot_provider: Any | None = None,
    ) -> None:
        self._store = store or default_trading_state_store()
        self._freshness = freshness or default_freshness_registry()
        self._states = state_machine or OrderStateMachine(self._store)
        self._gnn = gnn_runtime or GnnRuntime(
            config=TemporalHeteroGnnConfig(
                max_nodes=GRAPH_MAX_NODES,
                feature_dim=FEATURE_DIM,
                time_steps=GRAPH_TIME_STEPS,
            ),
            require_checkpoint=require_checkpoint,
        )
        self._pipeline = pipeline or ContextDecisionPipeline(
            store=self._store,
            gnn_runtime=self._gnn,
            snapshot_builder=GraphSnapshotBuilder(
                max_nodes=GRAPH_MAX_NODES, time_steps=GRAPH_TIME_STEPS
            ),
            state_machine=self._states,
            freshness=self._freshness,
        )
        self._market_group = str(market_group).upper()
        #: Live strategy-session snapshot, for the authority-path panel. Injected so the
        #: runtime does not reach into the trading engine's globals.
        self._session_snapshot_provider = session_snapshot_provider
        self._lock = threading.RLock()
        self._latest: CycleResult | None = None
        self._refresh_count = 0
        self._error_count = 0
        self._last_error: str | None = None
        self._last_refresh_at: datetime | None = None
        self._global_builder = GlobalContextBuilder()
        self._domestic_builder = DomesticContextBuilder()
        self._sector_builder = SectorContextBuilder()

    # ------------------------------------------------------------------ #
    # accessors
    # ------------------------------------------------------------------ #
    @property
    def freshness(self) -> DataFreshnessRegistry:
        return self._freshness

    @property
    def gnn(self) -> GnnRuntime:
        return self._gnn

    @property
    def state_machine(self) -> OrderStateMachine:
        return self._states

    @property
    def store(self) -> TradingStateStore:
        return self._store

    def latest(self) -> CycleResult | None:
        with self._lock:
            return self._latest

    def status(self) -> RuntimeStatus:
        with self._lock:
            return RuntimeStatus(
                last_refresh_at=self._last_refresh_at,
                last_cycle_id=self._latest.cycle_id if self._latest else None,
                refresh_count=self._refresh_count,
                error_count=self._error_count,
                last_error=self._last_error,
            )

    def is_stale(self, *, now: datetime | None = None) -> bool:
        with self._lock:
            if self._last_refresh_at is None:
                return True
            age = ((now or _utcnow()) - self._last_refresh_at).total_seconds()
        return age > CYCLE_STALE_SECONDS

    # ------------------------------------------------------------------ #
    # refresh
    # ------------------------------------------------------------------ #
    def refresh(
        self,
        *,
        now: datetime | None = None,
        candidates: Sequence[CandidateInput] | None = None,
        account: AccountState | None = None,
        websocket_connected: bool | None = None,
        trading_halted: bool | None = None,
        create_order_intents: bool = False,
    ) -> CycleResult | None:
        """Build the contexts from live sources and run one decision cycle.

        Returns ``None`` and records the error when a source fails hard. The trading loop
        treats ``None`` as "no fresh decision this cycle", which is a state the gate
        already refuses to trade on, rather than as a reason to stop.
        """
        moment = now or _utcnow()
        try:
            temporal = build_temporal_snapshot(self._market_group, moment)
            global_context = self.build_global_context(moment)
            resolved_candidates = list(
                candidates if candidates is not None else self.discover_candidates(moment)
            )
            domestic_context = self.build_domestic_context(
                moment, global_context=global_context, candidates=resolved_candidates
            )
            sector_contexts = self.build_sector_contexts(
                moment,
                candidates=resolved_candidates,
                global_context=global_context,
                domestic_context=domestic_context,
            )
            result = self._pipeline.run_cycle(
                captured_at=moment,
                temporal=temporal,
                candidates=resolved_candidates,
                global_context=global_context,
                domestic_context=domestic_context,
                sector_contexts=sector_contexts,
                account=account,
                websocket_connected=websocket_connected,
                trading_halted=trading_halted,
                create_order_intents=create_order_intents,
            )
        except Exception as exc:  # noqa: BLE001 - recorded and surfaced, never swallowed.
            with self._lock:
                self._error_count += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
            return None
        with self._lock:
            self._latest = result
            self._refresh_count += 1
            self._last_refresh_at = moment
            self._last_error = None
        return result

    # ------------------------------------------------------------------ #
    # source adapters
    # ------------------------------------------------------------------ #
    def build_global_context(self, moment: datetime) -> GlobalContext:
        """Global indicators from whatever the research collectors actually stored.

        Every observation is registered with the freshness registry, so a macro series
        that stopped updating shows up as DEGRADED on the data-health endpoint instead of
        quietly continuing to contribute its last value at full weight.
        """
        observations = list(self._macro_observations(moment))
        for observation in observations:
            self._freshness.record_event(
                "global_indicator",
                "index_level",
                observation.observed_at,
                scope_key=observation.name,
                received_time=moment,
                processed_time=moment,
            )
        return self._global_builder.build(observations, captured_at=moment)

    def _macro_observations(self, moment: datetime) -> Iterable[IndicatorObservation]:
        try:
            from app.storage import LocalResearchStore

            records = LocalResearchStore().load_analysis_inputs(prune=False).macro_metrics
        except Exception:  # noqa: BLE001 - no store, no global context.
            return ()
        by_indicator: dict[str, list[Any]] = {}
        for record in records or ():
            indicator = _MACRO_SERIES_TO_INDICATOR.get(str(getattr(record, "name", "")))
            if indicator is None:
                continue
            by_indicator.setdefault(indicator, []).append(record)

        observations: list[IndicatorObservation] = []
        for indicator, rows in by_indicator.items():
            ordered = sorted(rows, key=lambda item: _parse_datetime(item.observed_at) or moment)
            latest = ordered[-1]
            value = _finite(latest.value)
            if value is None:
                continue
            previous = _finite(ordered[-2].value) if len(ordered) > 1 else None
            # A level with no prior observation contributes to the volatility reading but
            # not to direction: one print is not a move.
            change = (
                (value - previous) / previous
                if previous is not None and previous != 0.0
                else None
            )
            observations.append(
                IndicatorObservation(
                    name=indicator,
                    value=value,
                    observed_at=_parse_datetime(latest.observed_at) or moment,
                    source=str(getattr(getattr(latest, "source", None), "source_name", "")),
                    change_ratio=change,
                )
            )
        return observations

    def build_domestic_context(
        self,
        moment: datetime,
        *,
        global_context: GlobalContext | None,
        candidates: Sequence[CandidateInput],
    ):
        """Domestic state from the realtime store, the flow store and the candidates.

        The index return is the cap-unweighted mean of the tracked universe rather than a
        KOSPI print, because this system has no index feed. That is a real limitation and
        it is stated rather than hidden: the value goes in as ``kospi_return`` with the
        universe's own breadth beside it, and a caller comparing the two can see that the
        first is derived from the second.
        """
        frame = self._macro_frame(moment, candidates)
        advancing = declining = 0
        for value in (frame.per_symbol_return or {}).values():
            number = _finite(value)
            if number is None:
                continue
            if number > 0:
                advancing += 1
            elif number < 0:
                declining += 1

        flows = self._investor_flows(candidates)
        venues = self._venue_quotes(moment, candidates)
        if frame.timestamp is not None:
            self._freshness.record_event(
                "internal",
                "domestic_context",
                frame.timestamp,
                received_time=moment,
                processed_time=moment,
            )
        return self._domestic_builder.build(
            DomesticContextInputs(
                kospi_return=_finite(frame.index_trend),
                advancing_count=advancing or None,
                declining_count=declining or None,
                breadth_momentum=_finite(frame.breadth_momentum),
                total_trading_value=_finite(frame.total_trading_value),
                realized_volatility=_finite(frame.market_volatility),
                foreign_flow=flows.get("foreign"),
                institution_flow=flows.get("institution"),
                retail_flow=flows.get("retail"),
                average_spread_bps=self._average_spread_bps(candidates),
                sector_dispersion=_finite(frame.cross_sectional_dispersion),
                sector_returns=dict(frame.sector_returns or {}),
                venues=venues,
                symbol_count=int(frame.symbol_count or 0),
            ),
            captured_at=moment,
            global_context=global_context,
        )

    def build_sector_contexts(
        self,
        moment: datetime,
        *,
        candidates: Sequence[CandidateInput],
        global_context: GlobalContext | None,
        domestic_context,
    ) -> tuple[SectorContext, ...]:
        by_sector: dict[str, list[CandidateInput]] = {}
        for candidate in candidates:
            if candidate.sector:
                by_sector.setdefault(candidate.sector, []).append(candidate)
        if not by_sector:
            return ()
        market_return = (
            domestic_context.components.get("index_return")
            if domestic_context is not None
            else None
        )
        contexts: list[SectorContext] = []
        for sector, members in by_sector.items():
            contexts.append(
                self._sector_builder.build(
                    sector,
                    [
                        SectorMemberObservation(
                            ticker=member.ticker,
                            session_return=member.session_return,
                            volume=member.volume_intensity,
                            realized_volatility=member.realized_volatility,
                            trading_value=None,
                        )
                        for member in members
                    ],
                    captured_at=moment,
                    market_return=_finite(market_return),
                    domestic_context=domestic_context,
                    global_context=global_context,
                    global_group=_global_group_for(sector),
                )
            )
            self._freshness.record_event(
                "internal",
                "sector_context",
                moment,
                scope_key=sector,
                received_time=moment,
                processed_time=moment,
            )
        return tuple(contexts)

    def discover_candidates(self, moment: datetime) -> tuple[CandidateInput, ...]:
        """Candidates from the realtime store's recently-active symbols.

        Deliberately thin: this is the fallback for a caller that has no candidate set of
        its own. The live engine passes its own elected candidates, which carry the
        election context this cannot reconstruct.
        """
        try:
            from app.data.realtime_store import RealtimeMarketDataStore

            store = RealtimeMarketDataStore()
            symbols = store.active_symbols(moment - timedelta(minutes=10), limit=40)
        except Exception:  # noqa: BLE001 - no feed, no candidates.
            return ()
        candidates: list[CandidateInput] = []
        for symbol in symbols:
            try:
                tick = store.latest_tick(symbol)
                book = store.latest_orderbook(symbol)
            except Exception:  # noqa: BLE001
                continue
            if tick is None:
                continue
            age = (moment - tick.received_at).total_seconds()
            self._freshness.record_event(
                "kis_realtime",
                "trade",
                tick.exchange_timestamp,
                scope_key=symbol,
                received_time=tick.received_at,
                processed_time=moment,
            )
            if book is not None:
                self._freshness.record_event(
                    "kis_realtime",
                    "orderbook",
                    book.exchange_timestamp,
                    scope_key=symbol,
                    received_time=book.received_at,
                    processed_time=moment,
                )
            candidates.append(
                CandidateInput(
                    ticker=symbol,
                    market_group=self._market_group,
                    spread_bps=_finite(getattr(book, "spread_bps", None)),
                    orderbook_imbalance=_finite(getattr(book, "imbalance", None)),
                    reference_price=_finite(tick.price),
                    data_age_seconds=age,
                )
            )
        return tuple(candidates)

    # ------------------------------------------------------------------ #
    def _macro_frame(self, moment: datetime, candidates: Sequence[CandidateInput]):
        from app.features.macro_feature_frame import (
            MacroFeatureFrame,
            macro_feature_frame_from_store,
        )

        symbols = [candidate.ticker for candidate in candidates]
        if not symbols:
            return MacroFeatureFrame(
                timestamp=moment,
                index_trend=None,
                market_breadth=None,
                market_volatility=None,
                total_trading_value=None,
                per_symbol_return={},
                sector_snapshots={},
                sector_of={},
                symbol_count=0,
            )
        try:
            from app.data.realtime_store import RealtimeMarketDataStore

            return macro_feature_frame_from_store(
                RealtimeMarketDataStore(),
                symbols,
                now=moment,
                sector_of={
                    candidate.ticker: candidate.sector
                    for candidate in candidates
                    if candidate.sector
                },
            )
        except Exception:  # noqa: BLE001 - an unreadable store yields an empty frame.
            return MacroFeatureFrame(
                timestamp=moment,
                index_trend=None,
                market_breadth=None,
                market_volatility=None,
                total_trading_value=None,
                per_symbol_return={},
                sector_snapshots={},
                sector_of={},
                symbol_count=0,
            )

    def _investor_flows(
        self, candidates: Sequence[CandidateInput]
    ) -> dict[str, float | None]:
        try:
            from app.data.investor_flow_store import InvestorFlowStore

            store = InvestorFlowStore()
        except Exception:  # noqa: BLE001
            return {}
        foreign = institution = retail = 0.0
        seen = 0
        for candidate in candidates:
            try:
                history = store.history(candidate.ticker)
            except Exception:  # noqa: BLE001
                continue
            if not history:
                continue
            latest = history[-1]
            foreign += _finite(getattr(latest, "foreign_net_buy_value", None)) or 0.0
            institution += (
                _finite(getattr(latest, "institution_net_buy_value", None)) or 0.0
            )
            retail += _finite(getattr(latest, "retail_net_buy_value", None)) or 0.0
            seen += 1
        if seen == 0:
            return {}
        self._freshness.record_event(
            "investor_flow", "flow_daily", _utcnow(), processed_time=_utcnow()
        )
        return {"foreign": foreign, "institution": institution, "retail": retail}

    def _venue_quotes(
        self, moment: datetime, candidates: Sequence[CandidateInput]
    ) -> tuple[VenueQuote, ...]:
        """KRX and NXT mids across the tracked universe, for the divergence reading.

        Averaged across symbols rather than compared per symbol: the question this feeds
        is "do the two books agree about the market", and one illiquid name disagreeing is
        not that.
        """
        try:
            from app.data.realtime_store import RealtimeMarketDataStore

            store = RealtimeMarketDataStore()
        except Exception:  # noqa: BLE001
            return ()
        totals: dict[str, list[float]] = {}
        for candidate in candidates:
            try:
                book = store.latest_orderbook(candidate.ticker)
            except Exception:  # noqa: BLE001
                continue
            if book is None:
                continue
            venue = str(getattr(getattr(book, "metadata", None), "venue", "") or "").upper()
            bid = _finite(getattr(book, "best_bid", None))
            ask = _finite(getattr(book, "best_ask", None))
            if venue not in {"KRX", "NXT"} or bid is None or ask is None:
                continue
            totals.setdefault(venue, []).append((bid + ask) / 2.0)
        return tuple(
            VenueQuote(venue=venue, mid=sum(values) / len(values), observed_at=moment)
            for venue, values in totals.items()
            if values
        )

    @staticmethod
    def _average_spread_bps(candidates: Sequence[CandidateInput]) -> float | None:
        values = [
            value
            for candidate in candidates
            if (value := _finite(candidate.spread_bps)) is not None
        ]
        return sum(values) / len(values) if values else None

    # ------------------------------------------------------------------ #
    # API views
    # ------------------------------------------------------------------ #
    def session_view(self, *, now: datetime | None = None) -> dict[str, Any]:
        moment = now or _utcnow()
        payload = {
            group: build_temporal_snapshot(group, moment).as_dict()
            for group in ("KR", "US")
        }
        return {"as_of": iso_column(moment), "groups": payload}

    def global_view(self) -> dict[str, Any]:
        latest = self.latest()
        if latest is None:
            return {"available": False, "reason": "NO_CYCLE_YET"}
        decision = latest.decisions[0] if latest.decisions else None
        payload = dict(decision.global_context) if decision else {}
        return {"available": bool(payload), "context": payload}

    def domestic_view(self) -> dict[str, Any]:
        latest = self.latest()
        if latest is None:
            return {"available": False, "reason": "NO_CYCLE_YET"}
        decision = latest.decisions[0] if latest.decisions else None
        payload = dict(decision.domestic_context) if decision else {}
        return {"available": bool(payload), "context": payload}

    def sector_view(self, sector: str) -> dict[str, Any]:
        latest = self.latest()
        if latest is None:
            return {"available": False, "reason": "NO_CYCLE_YET"}
        wanted = str(sector).strip().lower()
        for decision in latest.decisions:
            context = dict(decision.sector_context)
            if str(context.get("sector", "")).lower() == wanted:
                return {"available": True, "context": context}
        return {"available": False, "reason": "SECTOR_NOT_IN_LAST_CYCLE", "sector": sector}

    def stock_view(self, ticker: str) -> dict[str, Any]:
        decision = self._decision_for(ticker)
        if decision is None:
            return {"available": False, "reason": "TICKER_NOT_IN_LAST_CYCLE"}
        return {
            "available": True,
            "ticker": decision.ticker,
            "micro_context": dict(decision.micro_context),
            "sector_context": dict(decision.sector_context),
            "model_prediction": dict(decision.model_prediction),
        }

    def regime_view(self) -> dict[str, Any]:
        latest = self.latest()
        if latest is None:
            return {"available": False, "reason": "NO_CYCLE_YET"}
        return {"available": True, **latest.regime.as_dict()}

    def candidates_view(self, *, limit: int = 50) -> dict[str, Any]:
        latest = self.latest()
        if latest is None:
            return {"available": False, "reason": "NO_CYCLE_YET", "candidates": []}
        rows = []
        for decision in latest.decisions[: max(0, int(limit))]:
            sector = dict(decision.sector_context)
            micro = dict(decision.micro_context)
            rows.append(
                {
                    "ticker": decision.ticker,
                    "sector": sector.get("sector"),
                    "relative_strength": micro.get("relative_strength"),
                    "order_flow": micro.get("orderbook_imbalance"),
                    "global_alignment": sector.get("global_alignment"),
                    "strategy": decision.strategy_family,
                    "model_confidence": decision.model_confidence,
                    "gate": "PASS" if decision.gate_result else "BLOCK",
                    "gate_reasons": list(decision.gate_reasons),
                    "position_multiplier": decision.position_multiplier,
                    "decision_id": decision.decision_id,
                }
            )
        return {
            "available": True,
            "as_of": iso_column(latest.captured_at),
            "cycle_id": latest.cycle_id,
            "candidates": rows,
        }

    def decision_view(self, ticker: str) -> dict[str, Any]:
        decision = self._decision_for(ticker)
        if decision is None:
            return {"available": False, "reason": "TICKER_NOT_IN_LAST_CYCLE"}
        return {"available": True, **decision.as_dict()}

    def gate_view(self, ticker: str) -> dict[str, Any]:
        decision = self._decision_for(ticker)
        if decision is None:
            return {"available": False, "reason": "TICKER_NOT_IN_LAST_CYCLE"}
        return {
            "available": True,
            "ticker": decision.ticker,
            "gate_id": decision.gate_id,
            "approved": decision.gate_result,
            "reasons": list(decision.gate_reasons),
            "position_multiplier": decision.position_multiplier,
            "data_health": dict(decision.data_health),
            "model_health": dict(decision.model_health),
        }

    def model_health_view(self, *, now: datetime | None = None) -> dict[str, Any]:
        moment = now or _utcnow()
        health = self._gnn.health(now=moment)
        latest = self.latest()
        return {
            **health.as_dict(),
            "model_id": "temporal_hetero_gnn",
            "model_role": "context_regime_auxiliary",
            "cycle_stale": self.is_stale(now=moment),
            "last_cycle_at": iso_column(latest.captured_at) if latest else None,
            "runtime": self.status().as_dict(),
        }

    def data_health_view(self, *, now: datetime | None = None) -> dict[str, Any]:
        moment = now or _utcnow()
        report = self._freshness.report(now=moment)
        report["order_state"] = self._states.summary()
        report["cycle_stale"] = self.is_stale(now=moment)
        return report

    def authority_path_view(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Stage-by-stage description of who decides what, as the code actually runs.

        The stages are listed with the authority that owns each and, for the ones that
        moved, where they used to sit. An operator comparing this against the code should
        find no third opinion between the plan and the broker.
        """
        moment = now or _utcnow()
        session = self._strategy_session_snapshot()
        plan = (session or {}).get("trade_plan")
        return {
            "as_of": iso_column(moment),
            "authority": (session or {}).get("execution_authority")
            or ("TRADE_PLAN" if plan else "LEGACY_GATED_PATH"),
            "stages": [
                {
                    "stage": "pre_selection_context",
                    "authority": "ContextDecisionPipeline",
                    "decides": [
                        "calendar/session/weekday",
                        "global/domestic/sector/stock context",
                        "regime",
                    ],
                },
                {
                    "stage": "pre_selection_cost_size_risk",
                    "authority": "TradePlanBuilder",
                    "decides": [
                        "ProfitabilityGate: all-in cost and net edge",
                        "PositionSizer: position weight",
                        "RiskManager: exposure, concentration, eligibility, quantity",
                    ],
                    "moved_from": "SharedLiveDecisionEngine.evaluate_buy (post-election)",
                },
                {
                    "stage": "election",
                    "authority": "StrategySessionManager",
                    "decides": ["symbol + direction + strategy, or NO_TRADE"],
                    "output": "TradePlan (immutable)",
                },
                {
                    "stage": "fast_loop",
                    "authority": "StrategyFastExecutor",
                    "decides": ["entry trigger", "TP / SL / trailing / time / signal exit"],
                    "forbidden": [
                        "ontology rebuild",
                        "GNN inference",
                        "portfolio risk",
                        "position resizing",
                        "profitability re-evaluation",
                    ],
                },
                {
                    "stage": "execution_guard",
                    "authority": "ExecutionGuard",
                    "decides": [
                        "plan validity/expiry",
                        "price/quantity/instrument shape",
                        "session orderability",
                        "quote and book freshness",
                        "cash / sellable / borrow",
                        "duplicate, idempotency, kill switch, broker health",
                    ],
                    "forbidden": list(_forbidden_guard_checks()),
                },
                {
                    "stage": "broker",
                    "authority": "LiveExecutionCoordinator",
                    "decides": ["idempotent submission, amend/cancel, journal"],
                },
            ],
            "trade_plan": plan,
            "strategy_state": {
                key: (session or {}).get(key)
                for key in (
                    "phase",
                    "selected_symbol",
                    "selected_strategy",
                    "selected_direction",
                    "last_reason",
                )
            }
            if session
            else None,
            "removed_post_selection_vetoes": [
                "post-selection ProfitabilityGate",
                "post-selection PositionSizer",
                "post-selection portfolio RiskManager",
                "post-selection ontology re-approval",
                "post-selection selector size re-clip",
                "post-selection ExecutionQuality profitability judgement",
            ],
        }

    def latency_view(self, *, limit: int = 25) -> dict[str, Any]:
        from app.monitoring.execution_latency import default_latency_recorder

        recorder = default_latency_recorder()
        return {
            "available": True,
            "summary": recorder.summary(),
            "recent": list(recorder.recent(limit)),
        }

    def _strategy_session_snapshot(self) -> dict[str, Any] | None:
        provider = self._session_snapshot_provider
        if provider is None:
            return None
        try:
            snapshot = provider()
        except Exception:  # noqa: BLE001 - the panel degrades, it does not fail.
            return None
        return snapshot if isinstance(snapshot, dict) else None

    def dashboard_view(self, *, now: datetime | None = None) -> dict[str, Any]:
        """The top strip: clocks, sessions, regimes and health.

        ``readiness`` is computed from the same health objects the gate reads, so the
        dashboard cannot report ready while a core module is OFFLINE.
        """
        moment = now or _utcnow()
        kr = build_temporal_snapshot("KR", moment)
        us = build_temporal_snapshot("US", moment)
        latest = self.latest()
        model = self._gnn.health(now=moment)
        data = self._freshness.report(now=moment)
        blocking = list(data.get("blocking_reasons", []))
        gate_state = (
            "BLOCK"
            if blocking or not model.allows_new_entry or self.is_stale(now=moment)
            else "PASS"
        )
        return {
            # Rendered in the display timezone, NOT normalised to UTC: this field is the
            # wall clock an operator in Seoul reads off the top of the dashboard, and
            # `iso_column` would quietly hand them 13:40 for a 22:40 market.
            "KST": kr.display_time.isoformat(),
            "KRX_SESSION": kr.session_phase.value,
            "NXT_SESSION": _nxt_session(kr),
            "US_SESSION": us.session_phase.value,
            "GLOBAL_REGIME": _global_regime(latest),
            "KR_REGIME": latest.regime.dominant if latest else None,
            "VOLATILITY": _domestic_field(latest, "volatility"),
            "BREADTH": _domestic_field(latest, "breadth"),
            "LIQUIDITY": _domestic_field(latest, "liquidity"),
            "DATA_AGE": data.get("worst_state"),
            "GNN_HEALTH": model.state.value,
            "FINAL_GATE": gate_state,
            "readiness": {
                "data": data.get("worst_state"),
                "model": model.state.value,
                "cycle_stale": self.is_stale(now=moment),
                "blocking_reasons": blocking,
                "new_entry_permitted": gate_state == "PASS",
            },
        }

    def _decision_for(self, ticker: str):
        latest = self.latest()
        if latest is None:
            return None
        wanted = str(ticker).strip().upper()
        for decision in latest.decisions:
            if decision.ticker.upper() == wanted:
                return decision
        return None


def _forbidden_guard_checks() -> tuple[str, ...]:
    from app.execution.execution_guard import FORBIDDEN_INVESTMENT_CHECKS

    return FORBIDDEN_INVESTMENT_CHECKS


def _nxt_session(temporal: TemporalSnapshot) -> str:
    """NXT's own session, which is wider than KRX's and often differs from the phase."""
    for session in temporal.phase_state.active_sessions:
        if session.startswith("NXT"):
            return session
    return "CLOSED"


def _global_regime(latest: CycleResult | None) -> str | None:
    if latest is None or not latest.decisions:
        return None
    context = dict(latest.decisions[0].global_context)
    direction = context.get("direction")
    if direction is None:
        return None
    if direction > 0.25:
        return "RISK_ON"
    if direction < -0.25:
        return "RISK_OFF"
    return "NEUTRAL"


def _domestic_field(latest: CycleResult | None, name: str) -> Any:
    if latest is None or not latest.decisions:
        return None
    return dict(latest.decisions[0].domestic_context).get(name)


def _global_group_for(sector: str) -> str | None:
    from app.models.graph_snapshot import _global_group_for_sector

    return _global_group_for_sector(sector)


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


_default_runtime: ContextRuntime | None = None
_runtime_lock = threading.Lock()


def default_context_runtime(**kwargs: Any) -> ContextRuntime:
    global _default_runtime
    with _runtime_lock:
        if _default_runtime is None:
            _default_runtime = ContextRuntime(**kwargs)
        return _default_runtime


def reset_default_context_runtime() -> None:
    """Test hook. Never called from the trading path."""
    global _default_runtime
    with _runtime_lock:
        _default_runtime = None

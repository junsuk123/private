from __future__ import annotations

import asyncio
import math
import inspect
import json
import os
import re
import sqlite3
import threading
import time
import traceback
from contextlib import asynccontextmanager, closing
from dataclasses import asdict, fields, is_dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette import routing as starlette_routing

from app.web_account_routes import create_account_router
from app.web_short_strategy_routes import create_short_strategy_router
from app.account_dashboard import AccountDashboardService
from app.refactor_dashboard import (
    build_refactor_dashboard,
    build_strategy_market_stream,
    build_strategy_market_view,
)
from app.audit import AuditLogger
from app.backtesting import StreamingAcceleratedDemo, TimeScalerConfig, TimeMode
from app.data.kis_realtime import run_kis_realtime_websocket_collector
from app.data.llm_classifier import build_event_llm_classifier_from_env, configure_default_event_llm_env, event_llm_runtime_status
from app.data.realtime_store import RealtimeMarketDataStore
from app.execution import KisApiError, KisDevelopersApiClient, LiveExecutionCoordinator, MockKisDevelopersApi, PaperOrderExecutor
from app.execution.kis_auth import build_kis_client, run_kis_health_check, validate_live_secret_file
from app.execution.kis_errors import LiveExecutionBlocked
from app.graph import KnowledgeGraph, get_ontology_runtime
from app.goals import GoalRequest, NegotiatedGoal, assess_goal, build_compromise_goals
from app.config import LiveConfigError, load_live_trading_safety_config, load_order_execution_config
from app.config.refactor_flags import RefactorFeatureFlags
from app.features.feature_schema import LIVE_SHORT_HORIZON_SCHEMA
from app.models.model_artifact_registry import ModelArtifactRegistry
from app.models.live_training_pipeline import (
    backfill_live_feature_frames_from_realtime_store,
    collect_live_feature_frames_from_realtime_store,
    live_training_status,
    materialized_training_row_count,
    train_live_short_horizon_from_collected_features,
)
from app.models.live_signal_predictor import live_signal_model_inference_enabled
from app.pipeline import build_analysis_context
from app.research import ResearchRunResult, ResearchService
from app.realtime import OperationModeManager, RealtimeAccelerationPolicy, ShortHorizonRiskPolicy
from app.realtime.learning import (
    build_realtime_supervised_examples,
    run_hypothetical_realtime_test,
    update_realtime_model_artifacts,
)
from app.graph.npu_classifier import get_ontology_npu_classifier
from app.npu.runtime_manager import get_npu_runtime_manager
from app.risk import PrincipalProtectionEngine, RiskManager
from app.routing.gnn_realtime_trust import GnnRealtimeTrustEvaluator
from app.schemas.domain import (
    AccountSnapshot,
    FinalOrder,
    Holding,
    MarketSnapshot,
    OrderSide,
    OrderType,
    OrderAction,
    OrderIntent,
    PrincipalProtectionConfig,
    RealtimeExecution,
    RealtimeQuote,
    RiskRules,
    SourceMetadata,
)
from app.storage import LocalResearchStore, ModelArtifactStore, StoredResearch
from app.strategy import build_goal_execution_plan
from app.trading import run_mock_trading_cycle
from app.trading.live_runtime_guard import env_bool as live_env_bool, evaluate_live_runtime_gates
from app.trading.trading_policy import TradingPolicySnapshot
from app.trading.shared_decision_engine import SharedLiveDecisionEngine, _load_us_listed_exchange_map
from app.trading.realtime_trading_engine import RealtimeTradingEngine
from app.backtesting.accelerated_demo import load_krx_listed_universe, load_us_listed_universe
from app.market_affordability import (
    cash_available_for_market,
    is_market_affordable_for_account,
    is_overseas_market,
    market_currency,
)
from app.trading_pipeline import build_lightweight_market_snapshots_from_markets, load_short_horizon_strategy_config, ontology_filter_1


def _ensure_starlette_router_event_compatibility() -> None:
  """Support FastAPI 0.x when Starlette 1.x removed router event hooks."""
  signature = inspect.signature(starlette_routing.Router.__init__)
  if "on_startup" in signature.parameters:
    return
  if getattr(starlette_routing.Router, "_fastapi_event_compat", False):
    return

  original_init = starlette_routing.Router.__init__

  @asynccontextmanager
  async def event_lifespan(router: Any) -> Any:
    await router.startup()
    try:
      yield
    finally:
      await router.shutdown()

  def event_lifespan_context(_app: Any) -> Any:
    return event_lifespan(_app.router if hasattr(_app, "router") else _app)

  def compatible_init(
      self: Any,
      routes: Any = None,
      redirect_slashes: bool = True,
      default: Any = None,
      on_startup: Any = None,
      on_shutdown: Any = None,
      lifespan: Any = None,
      **kwargs: Any,
  ) -> None:
    original_init(
        self,
        routes=routes,
        redirect_slashes=redirect_slashes,
        default=default,
        lifespan=lifespan,
        **kwargs,
    )
    self.on_startup = list(on_startup or [])
    self.on_shutdown = list(on_shutdown or [])
    if lifespan is None:
      self.lifespan_context = event_lifespan_context

  def add_event_handler(self: Any, event_type: str, func: Any) -> None:
    if event_type == "startup":
      self.on_startup.append(func)
      return
    if event_type == "shutdown":
      self.on_shutdown.append(func)
      return
    raise ValueError(f"Unsupported event type: {event_type}")

  async def startup(self: Any) -> None:
    for handler in list(getattr(self, "on_startup", ())):
      result = handler()
      if inspect.isawaitable(result):
        await result

  async def shutdown(self: Any) -> None:
    for handler in list(getattr(self, "on_shutdown", ())):
      result = handler()
      if inspect.isawaitable(result):
        await result

  starlette_routing.Router.__init__ = compatible_init
  starlette_routing.Router.add_event_handler = add_event_handler
  starlette_routing.Router.startup = startup
  starlette_routing.Router.shutdown = shutdown
  starlette_routing.Router._fastapi_event_compat = True


_ensure_starlette_router_event_compatibility()

app = FastAPI(title="개인 투자 분석 시스템")
_gnn_realtime_trust_evaluator = GnnRealtimeTrustEvaluator(
    stale_while_refresh=True,
)
_live_shadow_lock = threading.RLock()
_live_shadow_service: Any | None = None
_live_shadow_state: dict[str, Any] = {
    "enabled": False,
    "last_attempt_at": None,
    "last_success_at": None,
    "last_symbol": None,
    "generated": 0,
    "errors": {},
}
app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")

_APP_ICON_PATH = Path(__file__).resolve().parent / "static" / "icon.png"


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    # Browsers (and the Pi Chromium kiosk) auto-request /favicon.ico; serve the
    # app icon so the tab/home-screen icon matches the in-page <link rel=icon>.
    return FileResponse(_APP_ICON_PATH, media_type="image/png")
audit = AuditLogger(Path("logs/web-audit.jsonl"))
sessions: dict[str, dict[str, Any]] = {}
DEFAULT_RESEARCH_CONFIG = Path(os.getenv("RESEARCH_CONFIG", "config/research_sources.live.json"))
LIVE_REFRESH_SECONDS = max(5, int(os.getenv("LIVE_REFRESH_SECONDS", "15")))
LIVE_STALE_SECONDS = max(LIVE_REFRESH_SECONDS * 2, int(os.getenv("LIVE_STALE_SECONDS", "45")))
ONTOLOGY_UI_NODE_LIMIT = max(40, int(os.getenv("ONTOLOGY_UI_NODE_LIMIT", "360")))
ONTOLOGY_UI_LINK_LIMIT = max(80, int(os.getenv("ONTOLOGY_UI_LINK_LIMIT", "900")))
ONTOLOGY_UI_REASONING_STEP_LIMIT = max(10, int(os.getenv("ONTOLOGY_UI_REASONING_STEP_LIMIT", "320")))
LEARNING_COLLECTION_INTERVAL_SECONDS = max(
    60,
    int(os.getenv("LEARNING_COLLECTION_INTERVAL_SECONDS", "3600")),
)
LIVE_RESEARCH_COLLECTION_INTERVAL_SECONDS = max(
    60,
    int(os.getenv("LIVE_RESEARCH_COLLECTION_INTERVAL_SECONDS", "300")),
)
AUTO_START_LIVE_WORKER = os.getenv("AUTO_START_LIVE_WORKER", "true").lower() not in {"0", "false", "no", "off"}
AUTO_START_LIVE_READINESS = os.getenv("AUTO_START_LIVE_READINESS", "true").lower() not in {"0", "false", "no", "off"}
AUTO_START_KIS_REALTIME_COLLECTOR = os.getenv("AUTO_START_KIS_REALTIME_COLLECTOR", "true").lower() not in {"0", "false", "no", "off"}
# 안전 기본값: 실시간 거래 엔진은 서버 기동 시 자동 시작하지 않는다(live_trading 모드 진입 시 시작).
AUTO_START_REALTIME_TRADING = os.getenv("AUTO_START_REALTIME_TRADING", "false").lower() in {"1", "true", "yes", "on"}
# 주기적 백그라운드 학습: 실시간 수집 데이터로 라이브 단기 모델을 주기적으로 재학습·배포한다.
# 데이터 수집은 실시간(KIS 수집기 + 트레이딩 평가 프레임 저널링), 학습은 이 워커가 주기적으로 수행.
AUTO_START_LIVE_TRAINING = os.getenv("AUTO_START_LIVE_TRAINING", "true").lower() not in {"0", "false", "no", "off"}
LIVE_TRAINING_INTERVAL_SECONDS = max(60, int(os.getenv("LIVE_TRAINING_INTERVAL_SECONDS", "300")))
_live_training_history_cache: dict[str, Any] = {
    "loaded_at": 0.0,
    "root": "",
    "limit": 0,
    "payload": None,
}
_system_diagnostics_cache_lock = threading.Lock()
_system_diagnostics_cache: dict[str, Any] = {
    "loaded_at": 0.0,
    "payload": None,
    "refreshing": False,
}
# US ticker→exchange map (NASD/NYSE/AMEX) auto-refresh: the NASDAQ Trader listings
# change over time, so a background worker rebuilds data/universe/us_exchange_map.csv
# whenever it is missing or older than US_EXCHANGE_MAP_MAX_AGE_DAYS.
AUTO_START_US_EXCHANGE_MAP_REFRESH = os.getenv("AUTO_START_US_EXCHANGE_MAP_REFRESH", "true").lower() not in {"0", "false", "no", "off"}
# Daily investor-flow (개인/외국인/기관 순매수) top-up. KIS reports this per business
# day and residual_relative_strength treats informed flow as mandatory, so without
# a scheduled refresh that strategy silently stops being evaluable as the stored
# 30-day window ages out.
AUTO_START_INVESTOR_FLOW_REFRESH = os.getenv("AUTO_START_INVESTOR_FLOW_REFRESH", "true").lower() not in {"0", "false", "no", "off"}
# Weekend research. Both venues are shut from the KRX Friday close to the KRX Monday
# open, which is the one window with spare compute and nothing to disturb. The loop
# builds a committed Monday-gap prior across Sat/Sun and grades it after the open.
AUTO_START_WEEKEND_BRIEF = os.getenv("AUTO_START_WEEKEND_BRIEF", "true").lower() not in {"0", "false", "no", "off"}
_auto_live_readiness_started = False
# Background total-asset sampler: periodically persist a dashboard snapshot so the
# asset-history curve accumulates continuously even when no browser is open (the Pi
# kiosk shows the trade display, not /account). Read-only w.r.t. trading.
AUTO_START_ASSET_HISTORY_SAMPLER = os.getenv("AUTO_START_ASSET_HISTORY_SAMPLER", "true").lower() not in {"0", "false", "no", "off"}
ASSET_HISTORY_SAMPLE_SECONDS = max(15, int(os.getenv("ASSET_HISTORY_SAMPLE_SECONDS", "60")))
AUTO_RELIABILITY_MODE_ENABLED = live_env_bool("AUTO_RELIABILITY_MODE_ENABLED", False)


def _account_dashboard_status_provider() -> dict[str, Any]:
  basis = _refresh_live_account_basis_for_auto() or _last_live_account_basis()
  if basis is None:
    snapshot = _live_snapshot()
    context = snapshot.get("context")
    account = getattr(context, "account", None)
    report = getattr(context, "report", None)
    return {
        "cash": float(getattr(account, "cash", 0.0) or 0.0),
        "krw_cash": float(getattr(account, "cash", 0.0) or 0.0),
        "cash_equivalent_krw": float(getattr(account, "pure_cash", 0.0) or 0.0),
        "equity": float(getattr(report, "equity", 0.0) or getattr(account, "equity", 0.0) or 0.0),
        "cash_weight": float(getattr(report, "cash_weight", 0.0) or 0.0),
        "base_currency": getattr(account, "base_currency", "KRW") if account is not None else "KRW",
        "cash_by_currency": dict(getattr(account, "cash_by_currency", {}) or {}),
        "positions": [],
        "basis_source": "analysis_context_fallback",
        "updated_at": _iso_or_none(snapshot.get("last_updated")),
        "last_error": snapshot.get("last_error"),
    }
  basis = _account_basis_with_realtime_holding_prices(basis)
  return {
      **basis,
      "basis_source": basis.get("source", "kis_live_account"),
      "account_checked": True,
      "updated_at": datetime.now(timezone.utc).isoformat(),
  }


def _account_basis_with_realtime_holding_prices(basis: dict[str, Any]) -> dict[str, Any]:
  positions = list(basis.get("positions") or ())
  if not positions:
    return basis
  try:
    max_age_seconds = max(1.0, float(os.getenv("ACCOUNT_DASHBOARD_REALTIME_PRICE_MAX_AGE_SEC", "600")))
  except (TypeError, ValueError):
    max_age_seconds = 600.0
  store = RealtimeMarketDataStore()
  now = datetime.now(timezone.utc)
  updated_positions: list[dict[str, Any]] = []
  changed = False
  for raw in positions:
    if not isinstance(raw, dict):
      updated_positions.append(raw)
      continue
    position = dict(raw)
    ticker = str(position.get("ticker") or "").upper().strip()
    quantity = _number_or_zero(position.get("quantity"))
    if not ticker or quantity <= 0:
      updated_positions.append(position)
      continue
    quote = _latest_dashboard_holding_quote(store, ticker, now, max_age_seconds)
    if quote is None:
      updated_positions.append(position)
      continue
    price, source, received_at = quote
    if price <= 0:
      updated_positions.append(position)
      continue
    old_price = _number_or_zero(position.get("last_price") or position.get("current_price"))
    position["last_price"] = price
    position["current_price"] = price
    position["last_price_source"] = source
    position["last_price_updated_at"] = received_at.isoformat()
    currency = str(position.get("currency") or "").upper()
    market = str(position.get("market") or "").upper()
    if currency == "KRW" or market in {"KR", "KRX", "KOSPI", "KOSDAQ", "KONEX"}:
      market_value = quantity * price
      average_price = _number_or_zero(position.get("average_price") or position.get("avg_price"))
      purchase = _number_or_zero(position.get("purchase_amount_krw")) or (quantity * average_price)
      position["market_value"] = market_value
      position["market_value_krw"] = market_value
      position["evaluation_amount_original"] = market_value
      position["unrealized_pnl_krw"] = market_value - purchase
      position["unrealized_pnl_original"] = market_value - purchase
    changed = changed or abs(price - old_price) > 1e-9
    updated_positions.append(position)
  if not changed:
    return basis
  merged = dict(basis)
  merged["positions"] = updated_positions
  domestic_value = sum(
      _number_or_zero(item.get("market_value_krw") or item.get("market_value"))
      for item in updated_positions
      if isinstance(item, dict)
      and (
          str(item.get("currency") or "").upper() == "KRW"
          or str(item.get("market") or "").upper() in {"KR", "KRX", "KOSPI", "KOSDAQ", "KONEX"}
      )
  )
  overseas_value = sum(
      _number_or_zero(item.get("market_value_krw") or item.get("market_value"))
      for item in updated_positions
      if isinstance(item, dict)
      and not (
          str(item.get("currency") or "").upper() == "KRW"
          or str(item.get("market") or "").upper() in {"KR", "KRX", "KOSPI", "KOSDAQ", "KONEX"}
      )
  )
  invested = domestic_value + overseas_value
  if invested > 0:
    merged["invested_value"] = invested
  return merged


def _latest_dashboard_holding_quote(
    store: RealtimeMarketDataStore,
    ticker: str,
    now: datetime,
    max_age_seconds: float,
) -> tuple[float, str, datetime] | None:
  candidates: list[tuple[datetime, float, str]] = []
  try:
    tick = store.latest_tick(ticker)
  except Exception:
    tick = None
  if tick is not None:
    received_at = getattr(tick, "received_at", None)
    price = _number_or_zero(getattr(tick, "price", 0.0))
    if received_at is not None and price > 0:
      candidates.append((received_at, price, str(getattr(tick, "source", "realtime_tick") or "realtime_tick")))
  try:
    orderbook = store.latest_orderbook(ticker)
  except Exception:
    orderbook = None
  if orderbook is not None:
    received_at = getattr(orderbook, "received_at", None)
    bid = _number_or_zero(getattr(orderbook, "best_bid", 0.0))
    ask = _number_or_zero(getattr(orderbook, "best_ask", 0.0))
    if received_at is not None and bid > 0 and ask > 0:
      candidates.append((received_at, (bid + ask) / 2.0, "realtime_orderbook_mid"))
  fresh = [
      item
      for item in candidates
      if max(0.0, (now - item[0]).total_seconds()) <= max_age_seconds
  ]
  if not fresh:
    return None
  received_at, price, source = max(fresh, key=lambda item: item[0])
  return price, source, received_at


def _account_dashboard_logs_provider() -> dict[str, Any]:
  snapshot = _live_snapshot()
  return {
      "collection_log": snapshot.get("collection_log") or [],
      "last_error": snapshot.get("last_error"),
      "live_execution_summary": snapshot.get("live_execution_summary"),
      "live_order_journal": _live_order_journal_snapshot(),
      "learning": snapshot.get("learning"),
  }


# Single shared service so the HTTP routes and the background asset-history
# sampler (see _asset_history_sampler_loop) persist to the same snapshot store.
def _account_dashboard_technical_provider() -> list[dict]:
    """Latest per-symbol technical decision context for the advisory GUI panel."""
    try:
        from app.technical.decision_feed import snapshot

        return snapshot()
    except Exception:  # noqa: BLE001 - advisory panel; never break the dashboard.
        return []


def _account_dashboard_macro_micro_provider() -> dict | None:
    """Latest macro–micro reasoning bundle for the advisory GUI panel."""
    try:
        from app.graph.macro_micro_feed import snapshot

        return snapshot()
    except Exception:  # noqa: BLE001 - advisory panel; never break the dashboard.
        return None


def _strategy_market_view_with_live_session(
    symbol: str | None,
    limit: int,
) -> dict[str, Any]:
    """Overlay the authoritative live owner on the read-only strategy terminal."""
    session: dict[str, Any] = {}
    engine_status: dict[str, Any] = {}
    engine_running = False
    try:
        with _realtime_trading_lock:
            engine = _realtime_trading_engine
            engine_running = bool(
                _realtime_trading_worker is not None
                and _realtime_trading_worker.is_alive()
            )
        if engine is not None:
            engine_status = dict(engine.get_status() or {})
            session = dict(engine_status.get("strategy_session") or {})
    except (NameError, AttributeError, TypeError):
        session = {}
        engine_status = {}
    owned_symbol = str(session.get("selected_symbol") or "").upper()
    active_owner = bool(
        owned_symbol
        and session.get("phase") in {"ARMED", "ENTERING", "OWNED", "EXITING", "COOLDOWN"}
    )
    candidate_symbols = tuple(
        str(item or "").strip().upper()
        for item in tuple((engine_status.get("last_summary") or {}).get("buy_candidate_sample") or ())
        if str(item or "").strip()
    )
    requested_symbol = str(symbol or "").strip().upper()
    # The browser keeps polling its previous symbol. Once the live session owns
    # a symbol, that client-side query must not override the authoritative
    # selection. During SCANNING, also replace an old/off-market selection with
    # the first candidate the engine is actually evaluating. This prevents a
    # a stale shadow row from masquerading as the current market decision.
    if active_owner:
        selected = owned_symbol
    elif candidate_symbols and requested_symbol not in candidate_symbols:
        selected = candidate_symbols[0]
    else:
        selected = requested_symbol or (candidate_symbols[0] if candidate_symbols else None)
    view = build_strategy_market_view(selected, limit=limit)
    view["strategy_session"] = session
    if session and active_owner:
        strategy_id = str(session.get("selected_strategy") or "")
        view["selection"] = {
            **dict(view.get("selection") or {}),
            "action": (
                "OWNED"
                if session.get("phase") in {"OWNED", "EXITING"}
                else session.get("phase")
            ),
            "strategy_id": strategy_id,
            "ontology_strategy_id": strategy_id,
            "ontology_allowed": bool(strategy_id and session.get("phase") != "SCANNING"),
            "path": session.get("selection_source"),
            "reason_codes": (
                [session.get("last_reason")] if session.get("last_reason") else []
            ),
            "session_id": session.get("session_id"),
        }
        if strategy_id:
            try:
                from app.refactor_dashboard import _algorithm

                view["algorithm"] = _algorithm(strategy_id) or view.get("algorithm")
            except Exception:
                pass
    journal = _live_order_journal_snapshot(limit=80)
    view["execution"] = _live_market_execution(
        str(view.get("symbol") or selected or ""),
        session=session,
        base_execution=dict(view.get("execution") or {}),
        journal=journal,
    )
    buy_enabled = bool(engine_status.get("buy_enabled"))
    live_armed_value = (engine_status.get("last_summary") or {}).get("live_armed")
    live_armed = buy_enabled if live_armed_value is None else bool(live_armed_value)
    view["live_trading"] = {
        "engine_running": engine_running,
        "buy_enabled": buy_enabled,
        "live_armed": live_armed,
        "phase": session.get("phase") or "OFFLINE",
        "adopted": active_owner,
        "symbol": owned_symbol if active_owner else None,
        "strategy_id": session.get("selected_strategy") if active_owner else None,
        "execution_authority": (
            session.get("selected_strategy") if active_owner else None
        ),
        "intelligence_role": "SELECTION_ONLY",
        "session_id": session.get("session_id") if active_owner else None,
        "selection_source": session.get("selection_source") if active_owner else None,
        "last_reason": session.get("last_reason"),
        "last_cycle_at": engine_status.get("last_cycle_at"),
        "buy_disabled_reason": engine_status.get("buy_disabled_reason"),
        "candidate_symbols": list(candidate_symbols),
    }
    with _live_shadow_lock:
        live_shadow = {
            **_live_shadow_state,
            "errors": dict(_live_shadow_state.get("errors") or {}),
        }
    last_shadow_success = _parse_iso_datetime(live_shadow.get("last_success_at"))
    live_shadow["age_seconds"] = (
        max(0.0, (datetime.now(timezone.utc) - last_shadow_success).total_seconds())
        if last_shadow_success is not None
        else None
    )
    live_shadow["healthy"] = bool(
        live_shadow.get("enabled")
        and last_shadow_success is not None
        and live_shadow["age_seconds"] <= max(
            15.0,
            _env_float_web("REALTIME_STRATEGY_SHADOW_HEALTH_MAX_AGE_SECONDS", 30.0),
        )
    )
    view["live_shadow"] = live_shadow
    if engine_running:
        view["mode"] = "live_trading"
    # The refactor dashboard's promotion flags describe shadow-model promotion,
    # not whether the production trading engine can submit an approved order.
    # The terminal must represent the latter.
    view["live_order_capable"] = bool(engine_running and buy_enabled and live_armed)
    return view


def _live_market_execution(
    symbol: str,
    *,
    session: dict[str, Any],
    base_execution: dict[str, Any],
    journal: dict[str, Any],
) -> dict[str, Any]:
    """Build an actual broker lifecycle for the symbol shown in the terminal."""
    normalized_symbol = str(symbol or "").upper()
    records = [
        dict(item)
        for item in journal.get("recent_orders", ())
        if isinstance(item, dict)
        and str(item.get("ticker") or "").upper() == normalized_symbol
    ]
    records = records[-18:]
    event_types = {str(item.get("event_type") or "") for item in records}
    statuses = {str(item.get("status") or "").upper() for item in records}
    blocked = any("blocked" in event_type for event_type in event_types)
    attempted = "live_order_submission_attempt" in event_types
    submitted = bool(
        event_types
        & {"live_order_submitted", "live_trading_order_submitted", "live_order_status"}
    )
    acknowledged = submitted and bool(statuses & {"ACCEPTED", "FILLED", "PARTIALLY_FILLED"})
    filled = bool(statuses & {"FILLED", "PARTIALLY_FILLED"})
    active_owner = bool(
        str(session.get("selected_symbol") or "").upper() == normalized_symbol
        and session.get("phase") in {"ARMED", "ENTERING", "OWNED", "EXITING", "COOLDOWN"}
    )

    def stage_status(done: bool, current: bool = False, failed: bool = False) -> str:
        if failed:
            return "blocked"
        if done:
            return "complete"
        return "current" if current else "waiting"

    phase = str(session.get("phase") or "SCANNING")
    strategy_id = str(session.get("selected_strategy") or "")
    latest = records[-1] if records else {}
    stages = [
        {
            "label": "전략·종목 채택",
            "detail": f"{normalized_symbol} · {strategy_id}" if active_owner else "온톨로지 후보 평가 중",
            "status": stage_status(active_owner, not active_owner),
        },
        {
            "label": "매수·매도 판단",
            "detail": phase if active_owner else str(session.get("last_reason") or "신호 대기"),
            "status": stage_status(attempted or submitted, active_owner and not attempted),
        },
        {
            "label": "주문 승인",
            "detail": "리스크·수익성 게이트 통과" if attempted else ("차단" if blocked else "승인 대기"),
            "status": stage_status(attempted, active_owner and not attempted, blocked),
        },
        {
            "label": "KIS 주문 전송",
            "detail": str(latest.get("broker_order_id") or "전송 대기"),
            "status": stage_status(submitted, attempted and not submitted, blocked),
        },
        {
            "label": "접수·상태 확인",
            "detail": str(latest.get("status") or "브로커 응답 대기"),
            "status": stage_status(acknowledged, submitted and not acknowledged),
        },
        {
            "label": "체결·잔고 반영",
            "detail": (
                f"{latest.get('side') or '-'} {latest.get('filled_quantity') or latest.get('quantity') or '-'}주"
                if filled
                else "체결 대기"
            ),
            "status": stage_status(filled, acknowledged and not filled),
        },
    ]
    events = [
        {
            "event_type": item.get("event_type"),
            "recorded_at": item.get("recorded_at"),
            "payload": {
                key: value
                for key, value in item.items()
                if key not in {"event_type", "recorded_at", "raw"}
            },
        }
        for item in reversed(records)
    ]
    if not events:
        events = list(base_execution.get("events") or ())
    return {
        **base_execution,
        "stages": stages,
        "events": events,
        "event_count": len(events),
        "source": "live_order_journal" if records else base_execution.get("source", "causal_order_journal"),
        "symbol": normalized_symbol,
    }


_account_service = AccountDashboardService(
    status_provider=_account_dashboard_status_provider,
    logs_provider=_account_dashboard_logs_provider,
    technical_provider=_account_dashboard_technical_provider,
    macro_micro_provider=_account_dashboard_macro_micro_provider,
)
app.include_router(
    create_account_router(
        service=_account_service,
        refactor_provider=build_refactor_dashboard,
        market_view_provider=_strategy_market_view_with_live_session,
        market_stream_provider=lambda symbol, limit: build_strategy_market_stream(
            symbol,
            limit=limit,
        ),
        market_stream_observer=lambda symbol: _observe_dashboard_market_stream(symbol),
    )
)
# Short-strategy deployment ladder. Its own router rather than another block of
# endpoints in this module, which is already 17k lines. Read-mostly: the single
# mutating route SUSPENDS an arm, and there is deliberately no promote route —
# promotion has to be earned from forward evidence, not requested over HTTP.
app.include_router(
    create_short_strategy_router(
        session_snapshot_provider=lambda: _short_strategy_session_snapshot(),
    )
)


def _short_strategy_session_snapshot() -> dict[str, Any]:
    """Live strategy-session state for the directional-comparison endpoint.

    Returns ``{}`` when the engine is not running, which the route renders as "no
    election yet" rather than as an error. A short strategy is invisible until
    promoted, so the dashboard is the only view into it; an exception here would
    remove the only way to see why an arm is not trading.
    """
    try:
        with _realtime_trading_lock:
            engine = _realtime_trading_engine
        if engine is None or not hasattr(engine, "get_status"):
            return {}
        status = engine.get_status() or {}
        return dict(status.get("strategy_session") or {})
    except Exception:  # noqa: BLE001 - diagnostics must never break the dashboard.
        return {}


def _env_flag(name: str, default: bool = False) -> bool:
  value = os.getenv(name)
  if value is None:
    return default
  return value.strip().lower() in {"1", "true", "yes", "on"}

_live_lock = threading.Lock()
_refresh_guard = threading.Lock()
_kis_realtime_collector_stop = threading.Event()
_kis_realtime_collector_resubscribe = threading.Event()
_kis_overseas_realtime_stop = threading.Event()
# Set to make the persistent overseas session re-diff its subscriptions in
# place — used when the symbol set changes and when the US daytime quote window
# opens or closes (the tr_key family changes with the session).
_kis_overseas_realtime_resubscribe = threading.Event()
_kis_overseas_realtime_worker: threading.Thread | None = None
_kis_overseas_observed_subscription_capacity: int | None = None
_kis_overseas_observed_capacity_at = 0.0
_kis_overseas_realtime_state: dict[str, Any] = {
    "running": False,
    "last_attempt_at": None,
    "last_success_at": None,
    "symbols": (),
    "counts": {},
    "last_error": None,
}
_pending_krx_buy_candidate_warmup: dict[str, float] = {}
_dashboard_krx_watch: dict[str, float] = {}
_dashboard_krx_stale_recovery_at: dict[str, float] = {}
_kis_realtime_collector_skipped_subscriptions: dict[tuple[str, str], float] = {}
_kis_realtime_observed_subscription_capacity: int | None = None
# When the capacity above was learned. A KIS OPSP0008 ("MAX SUBSCRIBE OVER") is
# usually transient — it happens when a previous process's WebSocket sessions
# still hold the approval key's slots. Without an expiry the learned capacity
# was a one-way ratchet: one transient limit hit pinned the collector to a
# single symbol for the whole process lifetime, which starves candidate
# discovery and stalls trading entirely.
_kis_realtime_observed_capacity_at: float = 0.0
_kis_realtime_complete_symbols: tuple[str, ...] = ()
_kis_realtime_last_resubscribe_request_at = 0.0
_mock_kis_lock = threading.Lock()
_mock_kis: MockKisDevelopersApi | None = None
_mock_trading_state: dict[str, Any] = {
    "active": False,
    "session_id": None,
    "goal": None,
    "started_at": None,
    "initial_equity": None,
    "last_run": None,
}
_streaming_demos: dict[str, StreamingAcceleratedDemo] = {}
_streaming_demos_lock = threading.Lock()
_streaming_demo_step_locks: dict[str, threading.Lock] = {}
_operation_mode_lock = threading.Lock()
_operation_mode_state: dict[str, Any] = {
    "active": None,
    "last_kis_connection": None,
    "last_kis_connection_checked_at": None,
    "request": {
        "busy": False,
        "stage": "idle",
        "message": "Waiting",
        "started_at": None,
        "updated_at": None,
        "last_error": None,
    },
      "live_trading_baseline_equity": None,
}
_auto_reliability_worker: threading.Thread | None = None
_auto_reliability_stop = threading.Event()
_auto_reliability_state: dict[str, Any] = {
    "enabled": AUTO_RELIABILITY_MODE_ENABLED,
    "mode": "learning",
    "score": 0.0,
    "ready": False,
    "ready_streak": 0,
    "unready_streak": 0,
    "reasons": ["NOT_EVALUATED"],
    "components": {},
    "active_markets": [],
    "evaluated_at": None,
    "last_transition_at": None,
    "last_transition_reason": None,
    "last_error": None,
    "last_learning_refresh_at": 0.0,
    "last_us_warm_at": 0.0,
}


def _kis_account_cache_seconds() -> float:
  try:
    return max(1.0, float(os.getenv("KIS_ACCOUNT_CACHE_SECONDS", "10")))
  except ValueError:
    return 10.0


def _get_store_root() -> Path:
  return Path(os.getenv("REALTIME_STORE_ROOT", "data/store"))


def _principal_config_path() -> Path:
  return Path(os.getenv("PRINCIPAL_PROTECTION_CONFIG", "config/principal_protection.json"))


def _principal_state_path() -> Path:
  return Path(os.getenv("PRINCIPAL_PROTECTION_STATE", "data/store/principal_protection_state.json"))


def _load_principal_protection_config() -> PrincipalProtectionConfig:
  path = _principal_config_path()
  if not path.exists():
    return PrincipalProtectionConfig()
  try:
    payload = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return PrincipalProtectionConfig()
  return _principal_config_from_payload(payload)


def _principal_config_with_live_account_basis(config: PrincipalProtectionConfig) -> PrincipalProtectionConfig:
  basis = _last_live_account_basis()
  if basis is None:
    return config
  principal_cash = max(0.0, float(basis.get("cash") or 0.0))
  if principal_cash <= 0:
    return config
  if abs(float(config.initial_principal or 0.0) - principal_cash) < 1.0:
    return config
  updated = _principal_config_from_payload({**asdict(config), "initial_principal": principal_cash})
  _save_principal_protection_config(updated)
  return updated


def _save_principal_protection_config(config: PrincipalProtectionConfig) -> None:
  path = _principal_config_path()
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(_to_jsonable(config), indent=2), encoding="utf-8")


def _ensure_initial_principal_configured(initial_principal: float) -> PrincipalProtectionConfig:
  config = _load_principal_protection_config()
  if config.initial_principal > 0:
    return config
  updated = _principal_config_from_payload({**asdict(config), "initial_principal": max(0.0, float(initial_principal))})
  _save_principal_protection_config(updated)
  return updated


def _principal_config_from_payload(payload: dict[str, Any]) -> PrincipalProtectionConfig:
  allowed = {item.name for item in fields(PrincipalProtectionConfig)}
  values = {key: value for key, value in payload.items() if key in allowed}
  return PrincipalProtectionConfig(**values)


def _load_principal_high_watermark(current_equity: float, config: PrincipalProtectionConfig) -> float | None:
  path = _principal_state_path()
  if not path.exists():
    return max(float(current_equity), float(config.initial_principal or 0.0))
  try:
    payload = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return max(float(current_equity), float(config.initial_principal or 0.0))
  try:
    return max(float(payload.get("high_watermark", 0.0)), float(current_equity), float(config.initial_principal or 0.0))
  except (TypeError, ValueError):
    return max(float(current_equity), float(config.initial_principal or 0.0))


def _save_principal_high_watermark(high_watermark: float) -> None:
  path = _principal_state_path()
  path.parent.mkdir(parents=True, exist_ok=True)
  payload = {"high_watermark": high_watermark, "updated_at": datetime.now(timezone.utc).isoformat()}
  path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _principal_capital_allocation(state: Any) -> dict[str, float]:
  protected = min(float(state.protected_floor), float(state.current_equity))
  exposed = max(0.0, float(state.active_risky_exposure))
  growth = max(0.0, float(state.available_growth_capital))
  locked = max(0.0, float(state.locked_profit))
  unavailable = max(0.0, float(state.current_equity) - protected - locked - growth)
  return {
      "protected_principal": round(protected, 4),
      "locked_profit": round(locked, 4),
      "growth_capital": round(growth, 4),
      "active_exposure": round(exposed, 4),
      "unavailable_capital": round(unavailable, 4),
  }


def _principal_protection_account_snapshot(config: PrincipalProtectionConfig) -> AccountSnapshot:
  live_basis = _last_live_account_basis()
  if live_basis is not None:
    cash_by_currency = dict(live_basis.get("cash_by_currency") or {})
    return AccountSnapshot(
        cash=max(0.0, float(live_basis.get("krw_cash") or 0.0)),
        holdings=(),
        base_currency=str(live_basis.get("base_currency") or "KRW"),
        cash_by_currency={str(key): float(value) for key, value in cash_by_currency.items()},
        orderable_cash_by_currency={
            str(key): float(value)
            for key, value in dict(live_basis.get("orderable_cash_by_currency") or {}).items()
        },
        cash_equivalent_krw=max(0.0, float(live_basis.get("cash_equivalent_krw") or live_basis.get("cash") or 0.0)),
        captured_at=datetime.now(timezone.utc),
    )
  with _live_lock:
    context = _live_state.get("context")
  account = getattr(context, "account", None)
  if isinstance(account, AccountSnapshot):
    return account
  return AccountSnapshot(cash=max(0.0, float(config.initial_principal or 0.0)), holdings=())


def _number_or_zero(value: Any) -> float:
  try:
    if value in (None, ""):
      return 0.0
    return float(str(value).replace(",", ""))
  except (TypeError, ValueError):
    return 0.0


def _cash_by_currency_payload(
    cash_by_currency: dict[str, Any] | None,
    fallback_cash: Any,
    base_currency: str = "KRW",
) -> dict[str, float]:
  result: dict[str, float] = {}
  if isinstance(cash_by_currency, dict):
    for currency, amount in cash_by_currency.items():
      code = str(currency or "").upper().strip()
      if code:
        result[code] = _number_or_zero(amount)
  code = str(base_currency or "KRW").upper().strip() or "KRW"
  if code not in result:
    result[code] = _number_or_zero(fallback_cash)
  return result


def _foreign_cash_by_currency(cash_by_currency: dict[str, Any] | None) -> dict[str, float]:
  if not isinstance(cash_by_currency, dict):
    return {}
  return {
      str(currency).upper(): _number_or_zero(amount)
      for currency, amount in cash_by_currency.items()
      if str(currency or "").upper() != "KRW"
  }


def _account_basis_from_kis_connection(connection: dict[str, Any] | None) -> dict[str, Any] | None:
  if not connection or not connection.get("account_checked"):
    return None
  krw_cash = _number_or_zero(connection.get("krw_cash") or connection.get("actual_deposit") or 0)
  cash = krw_cash if krw_cash > 0 else _number_or_zero(connection.get("cash") or 0)
  cash_by_currency = _cash_by_currency_payload(connection.get("cash_by_currency"), krw_cash)
  orderable_cash_by_currency = _cash_by_currency_payload(connection.get("orderable_cash_by_currency"), krw_cash)
  foreign_cash_krw = _number_or_zero(connection.get("foreign_cash_krw") or 0)
  cash_equivalent_krw = _number_or_zero(connection.get("cash_equivalent_krw") or 0)
  if cash <= 0:
    cash = krw_cash
  equity = _number_or_zero(
      connection.get("actual_equity")
      or connection.get("equity")
      or connection.get("account_value")
      or connection.get("total_evaluation_amount")
      or 0
  )
  invested = _number_or_zero(connection.get("invested_value") or 0)
  if cash_equivalent_krw <= 0:
    cash_equivalent_krw = cash + foreign_cash_krw
  if equity > 0:
    # The broker integrated total is authoritative. Component fields may include
    # integrated-margin buying power, so explanatory cash buckets must not exceed
    # total equity after subtracting positions.
    cash_component_cap = max(0.0, equity - invested)
    if cash_equivalent_krw - cash_component_cap > 0.005:
      cash_equivalent_krw = cash_component_cap
    cash = min(cash, cash_equivalent_krw)
    krw_cash = min(krw_cash, cash_equivalent_krw)
    foreign_cash_cap = max(0.0, cash_equivalent_krw - krw_cash)
    if foreign_cash_krw > foreign_cash_cap:
      foreign_cash_krw = foreign_cash_cap
  if equity <= 0 and (cash > 0 or invested > 0):
    equity = cash + invested
  if equity <= 0 and not connection.get("ok"):
    return None
  cash = max(0.0, cash)
  krw_cash_value = max(0.0, krw_cash)
  cash_by_currency["KRW"] = krw_cash_value
  return {
      "cash": cash,
      "krw_cash": krw_cash_value,
      "foreign_cash_krw": foreign_cash_krw,
      "cash_equivalent_krw": cash_equivalent_krw,
      "cash_by_currency": cash_by_currency,
      "orderable_cash_by_currency": orderable_cash_by_currency,
      "orderable_cash_reconciliation": dict(
          connection.get("orderable_cash_reconciliation") or {}
      ),
      "foreign_cash_by_currency": _foreign_cash_by_currency(cash_by_currency),
      "base_currency": "KRW",
      "equity": equity,
      "invested_value": invested,
      "cash_weight": max(0.0, min(1.0, cash / equity)) if equity > 0 else 0.0,
      "source": "kis_live_account",
      "account_suffix": connection.get("account_suffix") or "",
      "positions": list(connection.get("positions") or ()),
      # Broker-authoritative realized (settled) P&L for today, sourced from the
      # KIS period trade-profit inquiry in the account probe. Passed through so
      # the dashboard's realized-P&L field updates as sells settle.
      "realized_pnl_today_krw": _number_or_zero(connection.get("realized_pnl_today_krw")),
      "realized_pnl_period_krw": _number_or_zero(
          connection.get("realized_pnl_period_krw") or connection.get("realized_pnl_today_krw")
      ),
  }


def _holdings_from_live_positions(positions: Any) -> tuple[Holding, ...]:
  holdings: list[Holding] = []
  for position in tuple(positions or ()):
    if not isinstance(position, dict):
      continue
    if str(position.get("position_state") or "").lower() == "pending_balance":
      continue
    ticker = str(position.get("ticker") or "").upper().strip()
    quantity = int(_number_or_zero(position.get("quantity")))
    if not ticker or quantity <= 0:
      continue
    currency = str(position.get("currency") or "").upper()
    market = str(position.get("market") or ("KR" if currency == "KRW" else "NASDAQ"))
    holdings.append(
        Holding(
            ticker=ticker,
            market=market,
            company_name=ticker,
            sector="Unknown",
            quantity=quantity,
            average_price=_number_or_zero(position.get("average_price")),
            last_price=_number_or_zero(position.get("last_price")),
            sellable_quantity=(
                int(_number_or_zero(position.get("sellable_quantity")))
                if position.get("sellable_quantity") is not None
                else None
            ),
        )
    )
  return tuple(holdings)


def _last_live_account_basis() -> dict[str, Any] | None:
  with _live_lock:
    stable = _operation_mode_state.get("stable_account_basis")
    connection = _operation_mode_state.get("last_kis_connection")
  if isinstance(stable, dict):
    return dict(stable)
  return _account_basis_from_kis_connection(connection)


def _merge_live_account_basis_with_previous(
    basis: dict[str, Any] | None,
    previous: dict[str, Any] | None,
) -> dict[str, Any] | None:
  if basis is None:
    return previous
  if previous is None:
    return basis
  merged = dict(basis)
  current_positions = {
      str(item.get("ticker") or "").upper(): _number_or_zero(item.get("quantity"))
      for item in tuple(basis.get("positions") or ())
      if isinstance(item, dict)
  }
  previous_positions = {
      str(item.get("ticker") or "").upper(): _number_or_zero(item.get("quantity"))
      for item in tuple(previous.get("positions") or ())
      if isinstance(item, dict)
  }
  current_orderable = dict(basis.get("orderable_cash_by_currency") or {})
  previous_orderable = dict(previous.get("orderable_cash_by_currency") or {})
  current_krw_orderable = _number_or_zero(current_orderable.get("KRW"))
  previous_krw_orderable = _number_or_zero(previous_orderable.get("KRW"))
  current_equity = _number_or_zero(basis.get("equity"))
  previous_equity = _number_or_zero(previous.get("equity"))
  equity_consistent = (
      current_equity <= 0
      or previous_equity <= 0
      or abs(current_equity - previous_equity) <= max(1_000.0, previous_equity * 0.05)
  )
  # KIS's orderable-cash inquiry can transiently return zero while the balance
  # inquiries remain complete. Carry the previous value only when holdings and
  # total equity did not change, so a real fill/cash use is never hidden.
  if (
      current_krw_orderable <= 0
      and previous_krw_orderable > 0
      and current_positions == previous_positions
      and equity_consistent
  ):
    current_orderable["KRW"] = previous_krw_orderable
    merged["orderable_cash_by_currency"] = current_orderable
  basis = merged
  krw_cash = _number_or_zero(basis.get("krw_cash") or 0)
  previous_krw_cash = _number_or_zero(previous.get("krw_cash") or 0)
  foreign_cash_krw = _number_or_zero(basis.get("foreign_cash_krw") or 0)
  previous_foreign_cash_krw = _number_or_zero(previous.get("foreign_cash_krw") or 0)
  if krw_cash > 0 or previous_krw_cash <= 0 or foreign_cash_krw <= 0:
    return basis
  if previous_foreign_cash_krw > 0 and abs(previous_foreign_cash_krw - foreign_cash_krw) > max(100.0, previous_foreign_cash_krw * 0.05):
    return basis
  merged = dict(basis)
  cash_by_currency = dict(merged.get("cash_by_currency") or {})
  cash_by_currency["KRW"] = previous_krw_cash
  merged["cash_by_currency"] = cash_by_currency
  orderable_cash_by_currency = dict(merged.get("orderable_cash_by_currency") or {})
  if "KRW" not in orderable_cash_by_currency:
    orderable_cash_by_currency["KRW"] = previous_krw_cash
  merged["orderable_cash_by_currency"] = orderable_cash_by_currency
  merged["foreign_cash_by_currency"] = _foreign_cash_by_currency(cash_by_currency)
  merged["krw_cash"] = previous_krw_cash
  merged["cash"] = max(_number_or_zero(merged.get("cash")), previous_krw_cash)
  merged["cash_equivalent_krw"] = max(_number_or_zero(merged.get("cash_equivalent_krw")), previous_krw_cash + foreign_cash_krw)
  merged["equity"] = max(_number_or_zero(merged.get("equity")), previous_krw_cash + foreign_cash_krw)
  if _number_or_zero(merged.get("equity")) > 0:
    merged["cash_weight"] = max(0.0, min(1.0, _number_or_zero(merged.get("cash")) / _number_or_zero(merged.get("equity"))))
  return merged


def _connection_with_account_basis(connection: dict[str, Any], basis: dict[str, Any]) -> dict[str, Any]:
  merged = dict(connection)
  merged["cash"] = basis.get("cash", merged.get("cash"))
  merged["cash_equivalent_krw"] = basis.get("cash_equivalent_krw", merged.get("cash_equivalent_krw"))
  merged["actual_deposit"] = basis.get("krw_cash", merged.get("actual_deposit"))
  merged["krw_cash"] = basis.get("krw_cash", merged.get("krw_cash"))
  merged["foreign_cash_krw"] = basis.get("foreign_cash_krw", merged.get("foreign_cash_krw"))
  merged["cash_by_currency"] = basis.get("cash_by_currency", merged.get("cash_by_currency"))
  merged["orderable_cash_by_currency"] = basis.get("orderable_cash_by_currency", merged.get("orderable_cash_by_currency"))
  merged["orderable_cash_reconciliation"] = basis.get(
      "orderable_cash_reconciliation",
      merged.get("orderable_cash_reconciliation"),
  )
  merged["foreign_cash_by_currency"] = basis.get("foreign_cash_by_currency", merged.get("foreign_cash_by_currency"))
  merged["actual_equity"] = basis.get("equity", merged.get("actual_equity"))
  merged["invested_value"] = basis.get("invested_value", merged.get("invested_value"))
  merged["cash_weight"] = basis.get("cash_weight", merged.get("cash_weight"))
  basis_positions = list(basis.get("positions") or ())
  # A lightweight readiness probe/test may report only holdings_count without
  # position rows. Do not turn that authoritative count into zero merely because
  # the normalized basis has no details. Full portfolio responses always include
  # the positions key, including an explicit empty list.
  if "positions" in connection or basis_positions:
    merged["positions"] = basis_positions
    merged["holdings"] = len(basis_positions)
    merged["holdings_count"] = len(basis_positions)
  return merged


def _goal_account_snapshot(context: Any) -> AccountSnapshot:
  basis = _last_live_account_basis()
  if basis is None:
    return context.account
  return AccountSnapshot(
      cash=max(0.0, float(basis.get("krw_cash") or 0.0)),
      holdings=_holdings_from_live_positions(basis.get("positions")),
      cash_by_currency=dict(basis.get("cash_by_currency") or {}),
      orderable_cash_by_currency=dict(basis.get("orderable_cash_by_currency") or {}),
      cash_equivalent_krw=max(0.0, float(basis.get("cash_equivalent_krw") or basis["cash"])),
      captured_at=datetime.now(timezone.utc),
  )


def _live_account_snapshot_for_analysis() -> AccountSnapshot | None:
  basis = _refresh_live_account_basis_for_auto() or _last_live_account_basis()
  return _account_snapshot_from_live_basis(basis)


def _account_snapshot_from_live_basis(basis: dict[str, Any] | None) -> AccountSnapshot | None:
  """Build an account snapshot without making another broker request."""
  if basis is None:
    return None
  cash_by_currency = dict(basis.get("cash_by_currency") or {"KRW": basis.get("cash", 0.0)})
  return AccountSnapshot(
      cash=max(0.0, float(basis.get("krw_cash") or 0.0)),
      holdings=_holdings_from_live_positions(basis.get("positions")),
      base_currency=str(basis.get("base_currency") or "KRW"),
      cash_by_currency={str(key): float(value) for key, value in cash_by_currency.items()},
      orderable_cash_by_currency={
          str(key): float(value)
          for key, value in dict(basis.get("orderable_cash_by_currency") or {}).items()
      },
      cash_equivalent_krw=max(0.0, float(basis.get("cash_equivalent_krw") or basis.get("cash") or 0.0)),
      captured_at=datetime.now(timezone.utc),
  )


def _live_risk_rules_for_account(account: AccountSnapshot | None) -> RiskRules:
  try:
    safety = load_live_trading_safety_config()
  except LiveConfigError:
    safety = None

  if account is None:
    if safety is None:
      return RiskRules(live_trading_enabled=True)
    return RiskRules(
        live_trading_enabled=True,
        max_single_stock_weight=max(0.01, float(safety.maximum_position_pct_of_equity)),
        max_intraday_position_weight=max(0.01, float(safety.maximum_single_order_pct_of_cash)),
      max_sector_weight=1.0,
      minimum_cash_reserve=0.05,
        max_trades_per_day=max(1, int(safety.maximum_orders_per_day)),
        max_volatility=max(0.001, float(safety.maximum_volatility_5m_bps) / 10_000.0),
        min_data_quality_score=max(0.0, min(1.0, float(safety.minimum_source_quality_score))),
        max_quote_age_seconds=max(1.0, float(safety.max_quote_age_ms) / 1000.0),
        manual_approval_required=bool(safety.require_manual_arming),
    )

  equity = max(0.0, float(account.equity or 0.0))
  if safety is None:
    return RiskRules(
        live_trading_enabled=True,
        min_average_daily_trading_value=max(1_000.0, equity * 0.02),
    )
  return RiskRules(
      live_trading_enabled=True,
      max_single_stock_weight=max(0.01, float(safety.maximum_position_pct_of_equity)),
      max_intraday_position_weight=max(0.01, float(safety.maximum_single_order_pct_of_cash)),
      max_sector_weight=1.0,
      minimum_cash_reserve=0.05,
      max_trades_per_day=max(1, int(safety.maximum_orders_per_day)),
      max_volatility=max(0.001, float(safety.maximum_volatility_5m_bps) / 10_000.0),
      min_data_quality_score=max(0.0, min(1.0, float(safety.minimum_source_quality_score))),
      max_quote_age_seconds=max(1.0, float(safety.max_quote_age_ms) / 1000.0),
      manual_approval_required=bool(safety.require_manual_arming),
      min_average_daily_trading_value=max(1_000.0, equity * 0.02),
  )


def _stabilize_account_basis(basis: dict[str, Any] | None) -> dict[str, Any] | None:
  """Reject an obviously-degraded (partial) KIS balance fetch and keep the last complete
  one, so the displayed total asset does not wobble (e.g. 205k -> 124k -> 205k).

  KIS occasionally returns a partial balance (settlement deposit and/or a holding missing),
  which shows up as a sharp equity DROP with no explaining trade — a real sell ADDS cash
  rather than shrinking equity, and price moves are gradual. When we detect that, carry
  forward the last complete basis, bounded by ACCOUNT_DEGRADE_MAX_STALE_SEC so a genuine
  sustained change still gets through. State lives in _operation_mode_state.
  """
  if basis is None:
    return basis
  with _live_lock:
    stable = _operation_mode_state.get("stable_account_basis")
    degraded_since = _operation_mode_state.get("account_degraded_since")
  new_equity = _number_or_zero(basis.get("equity"))
  new_positions = len(tuple(basis.get("positions") or ()))
  new_cash_equiv = _number_or_zero(basis.get("cash_equivalent_krw"))
  new_components = _number_or_zero(basis.get("krw_cash")) + _number_or_zero(basis.get("foreign_cash_krw"))
  drop_ratio = float(os.getenv("ACCOUNT_DEGRADE_DROP_RATIO", "0.85"))
  max_stale = float(os.getenv("ACCOUNT_DEGRADE_MAX_STALE_SEC", "600"))
  now = time.time()
  degraded = False
  if stable is not None:
    stable_equity = _number_or_zero(stable.get("equity"))
    stable_positions = len(tuple(stable.get("positions") or ()))
    stable_cash_equiv = _number_or_zero(stable.get("cash_equivalent_krw"))
    stable_quantities = {
        str(item.get("ticker") or "").upper(): _number_or_zero(item.get("quantity"))
        for item in tuple(stable.get("positions") or ())
        if isinstance(item, dict) and str(item.get("position_state") or "").lower() != "pending_balance"
    }
    new_quantities = {
        str(item.get("ticker") or "").upper(): _number_or_zero(item.get("quantity"))
        for item in tuple(basis.get("positions") or ())
        if isinstance(item, dict) and str(item.get("position_state") or "").lower() != "pending_balance"
    }
    position_disappeared = any(
        quantity > new_quantities.get(symbol, 0.0)
        for symbol, quantity in stable_quantities.items()
    )
    cash_increase = new_cash_equiv - stable_cash_equiv
    if stable_equity > 0 and new_equity < stable_equity * drop_ratio and (
        new_positions < stable_positions or new_cash_equiv + 1.0 < new_components
    ):
      degraded = True
    # KIS can retain the total-equity summary while intermittently omitting an
    # overseas position or one cash component. A real sale increases cash; a
    # disappearing position with flat/falling cash is a partial response.
    if position_disappeared and cash_increase < 1_000.0:
      degraded = True
    if stable_cash_equiv > 0 and new_cash_equiv < stable_cash_equiv * drop_ratio:
      degraded = True
  if degraded:
    since = degraded_since or now
    with _live_lock:
      _operation_mode_state["account_degraded_since"] = since
    if now - since < max_stale:
      audit.record(
          "account_basis_degraded_carry_forward",
          {"new_equity": new_equity, "stable_equity": _number_or_zero((stable or {}).get("equity")),
           "new_positions": new_positions},
      )
      return stable
  # Accept: this is either complete or a sustained change. Refresh the stable snapshot.
  with _live_lock:
    _operation_mode_state["stable_account_basis"] = basis
    _operation_mode_state["account_degraded_since"] = None
  return basis


def _refresh_live_account_basis_for_auto() -> dict[str, Any] | None:
  cached = _cached_kis_connection_probe(paper=False, include_account=True)
  if cached.get("account_checked"):
    basis = _account_basis_from_kis_connection(cached)
    if basis is not None:
      return _stabilize_account_basis(basis)
  # _cached_kis_connection_probe already performs one live probe when its cache is
  # stale. Retrying immediately doubled the broker traffic, hit KIS per-second
  # limits, and kept the single web worker blocked for another full timeout.
  return None


def _cached_kis_connection_probe(
    paper: bool,
    include_account: bool = False,
    *,
    max_age_seconds: float | None = None,
) -> dict[str, Any]:
  if not include_account:
    return _kis_connection_probe(paper=False, include_account=include_account)
  now = time.time()
  ttl = _kis_account_cache_seconds() if max_age_seconds is None else max(0.0, float(max_age_seconds))
  with _live_lock:
    cached = _operation_mode_state.get("last_kis_connection")
    checked_at = _operation_mode_state.get("last_kis_connection_checked_at")
  if isinstance(cached, dict) and cached.get("account_checked") and isinstance(checked_at, (int, float)):
    if now - float(checked_at) <= ttl:
      connection = dict(cached)
      basis = _merge_live_account_basis_with_previous(
          _account_basis_from_kis_connection(connection),
          _last_live_account_basis(),
      )
      basis = _stabilize_account_basis(basis)
      if basis is not None:
        connection = _connection_with_account_basis(connection, basis)
        with _live_lock:
          _operation_mode_state["last_kis_connection"] = connection
      return connection
  connection = _kis_connection_probe(paper=False, include_account=True)
  previous = _last_live_account_basis()
  basis = _merge_live_account_basis_with_previous(_account_basis_from_kis_connection(connection), previous)
  basis = _stabilize_account_basis(basis)
  if connection.get("account_checked") and basis is not None:
    connection = _connection_with_account_basis(connection, basis)
  with _live_lock:
    _operation_mode_state["last_kis_connection"] = connection
    _operation_mode_state["last_kis_connection_checked_at"] = now
  return dict(connection)


def _start_auto_live_readiness_check() -> None:
  global _auto_live_readiness_started
  with _live_lock:
    if _auto_live_readiness_started:
      return
    _auto_live_readiness_started = True

  def worker() -> None:
    _set_operation_request(True, "starting", "Auto live readiness check", None)
    try:
      basis = None
      try:
        attempts = max(1, int(float(os.getenv("AUTO_LIVE_READINESS_RETRIES", "3"))))
      except (TypeError, ValueError):
        attempts = 3
      try:
        delay_seconds = max(0.5, float(os.getenv("AUTO_LIVE_READINESS_RETRY_DELAY_SECONDS", "5")))
      except (TypeError, ValueError):
        delay_seconds = 5.0
      for attempt in range(1, attempts + 1):
        basis = _refresh_live_account_basis_for_auto()
        if basis is not None:
          break
        audit.record("auto_live_readiness_retry", {"attempt": attempt, "attempts": attempts})
        if attempt < attempts:
          time.sleep(delay_seconds)
      if basis is not None:
        _set_operation_request(False, "checked", "Auto live readiness checked", None)
        audit.record("auto_live_readiness_checked", {"basis": basis})
      else:
        _set_operation_request(False, "error", "Auto live readiness did not return account basis", None)
        audit.record("auto_live_readiness_missing_basis", {"attempts": attempts})
    except Exception as exc:  # pragma: no cover - thread-level safety guard
      _set_operation_request(False, "error", f"Auto live readiness failed: {exc}", str(exc))
      audit.record("auto_live_readiness_failed", {"error": str(exc)})

  threading.Thread(target=worker, name="auto-live-readiness", daemon=True).start()


def _resolve_operating_initial_cash(payload: dict[str, Any], default: float = 10_000_000) -> float:
  source = str(payload.get("initial_cash_source") or "").lower()
  if source in {"auto", "live", "live_account", "kis_live_account"} or "initial_cash" not in payload:
    basis = _last_live_account_basis()
    if basis is not None:
      return max(1.0, float(basis["cash"]))
    if source in {"auto", "live", "live_account", "kis_live_account"}:
      _start_auto_live_readiness_check()
  return max(100_000.0, float(payload.get("initial_cash", default)))


def _resolved_initial_cash_source(payload: dict[str, Any]) -> str:
  source = str(payload.get("initial_cash_source") or "").lower()
  if source in {"auto", "live", "live_account", "kis_live_account"} or "initial_cash" not in payload:
    if _last_live_account_basis() is not None:
      return "kis_live_account"
    return "default_auto"
  return "manual_legacy"


def _normalised_target_return(value: Any, default: float = 0.02) -> float:
  try:
    target = float(value)
  except (TypeError, ValueError):
    target = default
  if target > 1:
    target /= 100.0
  return max(0.0, target)


def _resolve_auto_profit_gain(payload: dict[str, Any], initial_cash: float) -> float:
  """Derive the simulation gain from the goal, account scale, and live cash mix."""
  target_return_rate = _normalised_target_return(payload.get("target_return_rate", 0.02))
  try:
    period_minutes = max(1, int(payload.get("period_minutes", 390)))
  except (TypeError, ValueError):
    period_minutes = 390

  trading_day_minutes = 390.0
  goal_daily_return = target_return_rate / max(period_minutes / trading_day_minutes, 1.0 / trading_day_minutes)
  goal_pressure = max(0.35, min(2.75, goal_daily_return / 0.02))

  account_cash_weight = 1.0
  basis = _last_live_account_basis()
  if basis is not None:
    account_cash_weight = max(0.0, min(1.0, float(basis.get("cash_weight", 1.0))))
  liquidity_factor = 0.85 + (0.30 * account_cash_weight)

  account_size = max(1.0, float(initial_cash))
  if account_size < 100_000:
    account_factor = 1.15
  elif account_size < 1_000_000:
    account_factor = 1.08
  else:
    account_factor = 1.0

  risk_damper = 0.88 if target_return_rate >= 0.05 and period_minutes < trading_day_minutes else 1.0
  gain = (0.72 + goal_pressure * 0.36) * liquidity_factor * account_factor * risk_damper
  return max(0.25, min(4.0, gain))


def _pending_kis_connection_payload(paper: bool, include_account: bool) -> dict[str, Any]:
  return {
      "ok": None,
      "status": "checking",
      "mode": "live",
      "account_checked": False,
      "account_check_requested": bool(include_account),
      "message": "KIS connection check is running in the background.",
  }


def _start_kis_connection_probe_background(
  *,
  paper: bool,
  include_account: bool = False,
  update_live_basis: bool = False,
) -> None:
  def worker() -> None:
    connection = _kis_connection_probe(paper=False, include_account=include_account)
    if update_live_basis:
      basis = _merge_live_account_basis_with_previous(
          _account_basis_from_kis_connection(connection),
          _last_live_account_basis(),
      )
      basis = _stabilize_account_basis(basis)
      if connection.get("account_checked") and basis is not None:
        connection = _connection_with_account_basis(connection, basis)
      with _live_lock:
        _operation_mode_state["last_kis_connection"] = connection
        _operation_mode_state["last_kis_connection_checked_at"] = time.time()
    audit.record(
        "kis_connection_probe_background_finished",
        {
            "mode": "live",
            "ok": connection.get("ok"),
            "account_checked": connection.get("account_checked", False),
        },
    )

  threading.Thread(target=worker, name="kis-connection-probe", daemon=True).start()


def _active_operation_mode() -> str:
  mode = _operation_mode_state.get("active")
  if mode is None:
    return "learning"
  mode_value = getattr(mode, "mode", mode)
  return str(getattr(mode_value, "value", mode_value))


def _auto_reliability_int(name: str, default: int, minimum: int = 1) -> int:
  try:
    return max(minimum, int(float(os.getenv(name, str(default)))))
  except (TypeError, ValueError):
    return max(minimum, default)


def _latest_model_reliability(now: datetime) -> dict[str, Any]:
  root = Path("data/models/live_short_horizon")
  try:
    active_path = root / "latest.json"
    active = json.loads(active_path.read_text(encoding="utf-8"))
    candidates = [
        path for path in root.glob("live_short_horizon.*.json")
        if path.name != "latest.json"
    ]
    latest_challenger_path = max(candidates, key=lambda path: path.stat().st_mtime)
    challenger = json.loads(latest_challenger_path.read_text(encoding="utf-8"))
    active_modified_at = datetime.fromtimestamp(active_path.stat().st_mtime, timezone.utc)
    training_modified_at = datetime.fromtimestamp(
        latest_challenger_path.stat().st_mtime,
        timezone.utc,
    )
    canonical_staleness = ModelArtifactRegistry(root).staleness(now=now)
  except (OSError, ValueError, json.JSONDecodeError):
    return {"ok": False, "reason": "MODEL_ARTIFACT_MISSING_OR_INVALID"}
  active_age_seconds = (
      canonical_staleness.age_seconds
      if canonical_staleness.age_seconds is not None
      else max(0.0, (now - active_modified_at).total_seconds())
  )
  training_age_seconds = max(0.0, (now - training_modified_at).total_seconds())
  heartbeat_at = None
  heartbeat_ok = False
  training_in_progress = False
  with _live_lock:
    heartbeat = dict(_live_training_heartbeat)
  heartbeat_value = heartbeat.get("finished_at")
  if heartbeat_value:
    try:
      heartbeat_at = datetime.fromisoformat(str(heartbeat_value).replace("Z", "+00:00"))
      if heartbeat_at.tzinfo is None:
        heartbeat_at = heartbeat_at.replace(tzinfo=timezone.utc)
      heartbeat_ok = bool(heartbeat.get("ok"))
    except (TypeError, ValueError):
      heartbeat_at = None
  started_at = None
  if heartbeat.get("started_at"):
    try:
      started_at = datetime.fromisoformat(
          str(heartbeat.get("started_at")).replace("Z", "+00:00")
      )
      if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
      started_at = None
  latest_cycle_is_running = bool(
      started_at is not None
      and (heartbeat_at is None or started_at > heartbeat_at)
  )
  running_heartbeat_limit = max(
      60.0,
      _env_float_web(
          "LIVE_TRAINING_RUNNING_HEARTBEAT_MAX_SECONDS",
          max(900.0, LIVE_TRAINING_INTERVAL_SECONDS * 2.0),
      ),
  )
  if (
      latest_cycle_is_running
      and started_at is not None
      and not heartbeat.get("error")
      and max(0.0, (now - started_at).total_seconds()) <= running_heartbeat_limit
  ):
    # A cycle can spend minutes materializing labels or waiting for the shared
    # incremental-training lock. Report that recent running heartbeat as healthy
    # instead of presenting a false "training stopped" alarm until it finishes.
    heartbeat_at = started_at
    heartbeat_ok = True
    training_in_progress = True
  # An unchanged labelled dataset is a successful training cycle. Use that
  # worker heartbeat so a normal skip does not make a valid incumbent appear
  # stale merely because no new artifact needed to be written.
  if heartbeat_ok and heartbeat_at is not None:
    training_age_seconds = min(
        training_age_seconds,
        max(0.0, (now - heartbeat_at).total_seconds()),
    )
  maximum_training_age = float(
      _auto_reliability_int("AUTO_RELIABILITY_MODEL_MAX_AGE_SECONDS", 1800, 60)
  )
  maximum_active_age = float(
      canonical_staleness.diagnostics.get("max_age_seconds") or 21_600.0
  )
  live_eligible = bool(active.get("live_eligible"))
  schema_matches = active.get("feature_schema_hash") == LIVE_SHORT_HORIZON_SCHEMA.schema_hash
  reason_codes: list[str] = []
  if not live_eligible:
    reason_codes.append("ACTIVE_MODEL_NOT_LIVE_ELIGIBLE")
  if not schema_matches:
    reason_codes.append("ACTIVE_MODEL_SCHEMA_MISMATCH")
  if canonical_staleness.stale:
    reason_codes.extend(canonical_staleness.reason_codes)
  if training_age_seconds > maximum_training_age:
    reason_codes.append("MODEL_TRAINING_STALE")
  return {
      "ok": not reason_codes,
      "live_eligible": live_eligible,
      "schema_matches": schema_matches,
      "age_seconds": active_age_seconds,
      "maximum_age_seconds": maximum_active_age,
      "trust_level": canonical_staleness.trust_level.value,
      "training_age_seconds": training_age_seconds,
      "training_heartbeat_at": heartbeat_at.isoformat() if heartbeat_at else None,
      "training_heartbeat_ok": heartbeat_ok,
      "training_in_progress": training_in_progress,
      "maximum_training_age_seconds": maximum_training_age,
      "artifact_id": active.get("artifact_id"),
      "reason_codes": reason_codes,
      "metrics": {
          "auc": _number_or_zero((active.get("metrics") or {}).get("auc")),
          "precision_at_k": _number_or_zero((active.get("metrics") or {}).get("precision_at_k")),
          "live_eligible": live_eligible,
      },
      "latest_challenger": {
          "artifact_id": challenger.get("artifact_id"),
          "live_eligible": bool(challenger.get("live_eligible")),
          "reason_codes": list(challenger.get("reason_codes") or ()),
          "metrics": {
              "auc": _number_or_zero((challenger.get("metrics") or {}).get("auc")),
              "precision_at_k": _number_or_zero(
                  (challenger.get("metrics") or {}).get("precision_at_k")
              ),
          },
      },
  }


def _auto_market_health(now: datetime, groups: tuple[str, ...]) -> dict[str, Any]:
  minimum = _auto_reliability_int("AUTO_RELIABILITY_MIN_HEALTHY_SYMBOLS", 2)
  minimum_by_market = {
      "KRX": _auto_reliability_int("AUTO_RELIABILITY_KRX_MIN_HEALTHY_SYMBOLS", 1),
      "US": _auto_reliability_int("AUTO_RELIABILITY_US_MIN_HEALTHY_SYMBOLS", minimum),
  }
  kr_age = _auto_reliability_int("AUTO_RELIABILITY_KRX_MAX_AGE_SECONDS", 20)
  kr_trade_activity_age = _auto_reliability_int(
      "AUTO_RELIABILITY_KRX_TRADE_ACTIVITY_MAX_AGE_SECONDS",
      120,
  )
  us_age = _auto_reliability_int("AUTO_RELIABILITY_US_MAX_AGE_SECONDS", 90)
  database = Path(os.getenv("REALTIME_MARKET_DATA_DB", "data/store/realtime_market_data.sqlite3"))
  healthy: dict[str, list[str]] = {"KRX": [], "US": []}
  # No core entry session means there is no realtime feed to require. Market
  # session gates still reject new orders, while the reliability controller can
  # remain armed for the next open instead of misclassifying normal closure as
  # a data outage and demoting the whole process to learning mode.
  if not groups:
    return {
        "ok": True,
        "healthy": healthy,
        "minimum_per_market": minimum,
        "minimum_by_market": minimum_by_market,
        "ready_markets": [],
        "partial": False,
        "missing_markets": [],
        "max_age_seconds": {
            "KRX": kr_age,
            "KRX_TRADE_ACTIVITY": kr_trade_activity_age,
            "US": us_age,
        },
    }
  if not database.exists():
    return {"ok": False, "healthy": healthy, "minimum_per_market": minimum, "reason": "MARKET_DATA_STORE_MISSING"}
  try:
    with sqlite3.connect(database, timeout=5.0) as connection:
      for group in groups:
        age_seconds = us_age if group == "US" else kr_age
        book_cutoff = (now - timedelta(seconds=age_seconds)).isoformat()
        tick_cutoff = (
            now
            - timedelta(
                seconds=kr_trade_activity_age if group == "KRX" else age_seconds
            )
        ).isoformat()
        symbol_filter = (
            "symbol not glob '[0-9][0-9][0-9][0-9][0-9][0-9]'"
            if group == "US"
            else "symbol glob '[0-9][0-9][0-9][0-9][0-9][0-9]'"
        )
        ticks = {
            str(row[0]).upper()
            for row in connection.execute(
                f"""
                select symbol from realtime_ticks
                where received_at >= ? and source = 'kis_realtime_websocket'
                  and {symbol_filter}
                group by symbol
                limit 500
                """,
                (tick_cutoff,),
            )
        }
        books = {
            str(row[0]).upper()
            for row in connection.execute(
                f"""
                select symbol from realtime_orderbook
                where received_at >= ? and source = 'kis_realtime_websocket'
                  and {symbol_filter}
                group by symbol
                limit 500
                """,
                (book_cutoff,),
            )
        }
        # A KRX orderbook can update continuously while a particular stock has no
        # trade print for several seconds. A recent trade anywhere in the subscribed
        # KRX stream proves the trade channel is alive; fresh per-symbol books then
        # prove those symbols are quote-ready. Per-order symbol freshness remains a
        # stricter gate in SharedLiveDecisionEngine.
        healthy[group] = sorted(books if group == "KRX" and ticks else ticks & books)
  except sqlite3.Error as exc:
    return {
        "ok": False,
        "healthy": healthy,
        "minimum_per_market": minimum,
        "reason": f"MARKET_DATA_QUERY_FAILED:{type(exc).__name__}",
    }
  missing = [
      group
      for group in groups
      if len(healthy.get(group, ())) < minimum_by_market.get(group, minimum)
  ]
  ready_markets = [group for group in groups if group not in missing]
  return {
      # Market outages are isolated. Per-symbol quote/orderbook freshness remains
      # mandatory in SharedLiveDecisionEngine, so a healthy US market can continue
      # while KRX is degraded (and vice versa) without permitting stale orders.
      "ok": bool(ready_markets),
      "healthy": healthy,
      "minimum_per_market": minimum,
      "minimum_by_market": minimum_by_market,
      "ready_markets": ready_markets,
      "partial": bool(ready_markets and missing),
      "missing_markets": missing,
      "max_age_seconds": {
          "KRX": kr_age,
          "KRX_TRADE_ACTIVITY": kr_trade_activity_age,
          "US": us_age,
      },
  }


def _evaluate_auto_reliability(now: datetime | None = None) -> dict[str, Any]:
  now = now or datetime.now(timezone.utc)
  groups = _active_live_market_groups(now)
  # The domestic H0STCNT0/H0STASP0 stream used by this service is a regular-session
  # feed. KRX opening-auction/after-hours order routes may be available while those
  # TRs still emit no tradeable ticks. Requiring 20-second KRX ticks during that
  # interval creates a permanent false failure and masks a healthy US feed.
  market_data_groups = tuple(
      group
      for group in groups
      if group != "KRX" or _is_live_market_core_open("KRX", now)
  )
  owner = _kis_realtime_session_owner(now)
  if owner in {"KRX", "US"}:
    # A KIS AppKey permits one realtime WebSocket session. During overlapping
    # KRX/US-daytime windows only the elected owner can produce fresh WS books;
    # do not make the deliberately idle market a reliability hard gate.
    market_data_groups = tuple(
        group for group in market_data_groups if group == owner
    )
  connection = _cached_kis_connection_probe(paper=False, include_account=True)
  broker_ok = bool(
      connection.get("ok")
      and connection.get("account_checked")
      and _number_or_zero(connection.get("actual_equity") or connection.get("equity")) > 0
  )
  runtime = evaluate_live_runtime_gates(require_manual_arming=_manual_arming_required())
  strategy_config = load_short_horizon_strategy_config()
  execution_config = dict(strategy_config.get("execution") or {})
  config_ok = bool(
      execution_config.get("live_trading_enabled")
      and _env_flag("LIVE_TRADING_ENABLED", False)
      and _env_flag("KIS_LIVE_ENABLED", False)
  )
  policy_conflicts = [
      conflict.code
      for conflict in TradingPolicySnapshot.from_environment().conflicts()
      if conflict.severity == "FAIL"
  ]
  policy_ok = not policy_conflicts
  model = _latest_model_reliability(now)
  market = _auto_market_health(now, market_data_groups)
  market["required_markets"] = list(market_data_groups)
  market["extended_order_markets"] = list(groups)
  components = {
      "broker": {"ok": broker_ok, "weight": 0.20},
      "runtime": {"ok": bool(runtime.ok), "weight": 0.15, "failures": list(runtime.failures)},
      "config": {"ok": config_ok, "weight": 0.10},
      "risk_policy": {"ok": policy_ok, "weight": 0.15, "failures": policy_conflicts},
      "model": {**model, "weight": 0.20},
      "market_data": {**market, "weight": 0.20},
  }
  score = sum(float(item["weight"]) for item in components.values() if item.get("ok"))
  threshold = float(os.getenv("AUTO_RELIABILITY_PROMOTE_THRESHOLD", "0.90"))
  reasons = []
  if not groups:
    reasons.append("NO_OPEN_MARKET")
  for name, item in components.items():
    if not item.get("ok"):
      reasons.append(f"{name.upper()}_NOT_READY")
  return {
      "score": round(score, 4),
      "threshold": threshold,
      "ready": bool(groups) and score >= threshold and not reasons,
      "reasons": reasons,
      "components": components,
      "active_markets": list(groups),
      "evaluated_at": now.isoformat(),
  }


def _auto_reliability_transition_to_learning(reason: str) -> None:
  with _realtime_trading_lock:
    engine = _realtime_trading_engine
  if engine is not None and hasattr(engine, "disable_buys"):
    engine.disable_buys(f"AUTO_RELIABILITY_DEMOTION:{reason}")
  # Keep the engine alive in sell-only mode. Stopping the worker here stranded
  # existing positions without take-profit, stop-loss, or time-exit monitoring.
  state = OperationModeManager().start("learning")
  with _live_lock:
    _operation_mode_state["active"] = state
  _start_live_worker("learning")
  audit.record("auto_reliability_demoted_to_learning", {"reason": reason})


def _auto_reliability_enforce_sell_only(reason: str) -> None:
  """Fail closed when reliability is low even if learning mode was already active."""
  with _realtime_trading_lock:
    engine = _realtime_trading_engine
  if engine is None or not hasattr(engine, "disable_buys"):
    return
  try:
    status = engine.get_status() if hasattr(engine, "get_status") else {}
  except Exception:  # noqa: BLE001 - disabling buys remains the safe fallback.
    status = {}
  expected_reason = f"AUTO_RELIABILITY_DEMOTION:{reason}"
  if (
      status.get("buy_enabled") is not False
      or str(status.get("buy_disabled_reason") or "") != expected_reason
  ):
    # Reliability can recover one component while another remains blocked. Keep
    # the operator-visible reason synchronized even when the mode itself does
    # not transition (for example MARKET_DATA_NOT_READY -> MODEL_NOT_READY).
    engine.disable_buys(expected_reason)


def _auto_reliability_transition_to_live() -> dict[str, Any]:
  result = _operation_mode_start_response({"mode": "live_trading"})
  if result.get("ok") and result.get("live_trading_status") == "armed":
    with _realtime_trading_lock:
      engine = _realtime_trading_engine
    if engine is not None and hasattr(engine, "enable_buys"):
      engine.enable_buys("AUTO_RELIABILITY_PROMOTION")
  audit.record(
      "auto_reliability_live_transition",
      {"ok": result.get("ok"), "status": result.get("status"), "live_status": result.get("live_trading_status")},
  )
  return result


def _auto_reliability_enter_learning() -> None:
  state = OperationModeManager().start("learning")
  with _live_lock:
    _operation_mode_state["active"] = state
  _start_live_worker("learning")


def _auto_reliability_learning_maintenance(now: datetime, groups: tuple[str, ...]) -> None:
  refresh_interval = _auto_reliability_int("AUTO_RELIABILITY_LEARNING_REFRESH_SECONDS", 300, 30)
  now_monotonic = time.monotonic()
  with _live_lock:
    last_refresh = float(_auto_reliability_state.get("last_learning_refresh_at") or 0.0)
  if now_monotonic - last_refresh >= refresh_interval:
    _ensure_background_refresh()
    with _live_lock:
      _auto_reliability_state["last_learning_refresh_at"] = now_monotonic
  if "US" in groups:
    _ensure_us_fast_poll_started()


def _auto_reliability_step(now: datetime | None = None) -> dict[str, Any]:
  now = now or datetime.now(timezone.utc)
  snapshot = _evaluate_auto_reliability(now)
  current_mode = _active_operation_mode()
  with _live_lock:
    active_state = _operation_mode_state.get("active")
    ready_streak = int(_auto_reliability_state.get("ready_streak") or 0)
    unready_streak = int(_auto_reliability_state.get("unready_streak") or 0)
  if snapshot["ready"]:
    ready_streak += 1
    unready_streak = 0
  else:
    ready_streak = 0
    unready_streak += 1
  promote_after = _auto_reliability_int("AUTO_RELIABILITY_PROMOTE_CONSECUTIVE", 4)
  demote_after = _auto_reliability_int("AUTO_RELIABILITY_DEMOTE_CONSECUTIVE", 2)
  transition_reason = None
  if snapshot["ready"] and ready_streak >= promote_after and current_mode != "live_trading":
    result = _auto_reliability_transition_to_live()
    if result.get("ok") and result.get("live_trading_status") == "armed":
      current_mode = "live_trading"
      transition_reason = "RELIABILITY_SUSTAINED"
    else:
      snapshot["ready"] = False
      snapshot["reasons"] = [*snapshot["reasons"], "LIVE_TRANSITION_NOT_ARMED"]
      unready_streak = 1
  elif not snapshot["ready"] and current_mode == "live_trading":
    critical = any(
        reason in snapshot["reasons"]
        for reason in (
            "BROKER_NOT_READY",
            "RUNTIME_NOT_READY",
            "CONFIG_NOT_READY",
            "RISK_POLICY_NOT_READY",
            "MODEL_NOT_READY",
            "NO_OPEN_MARKET",
        )
    )
    startup_grace = _live_market_data_startup_grace_active(now)
    market_data_only = set(snapshot["reasons"]) == {"MARKET_DATA_NOT_READY"}
    if critical or (unready_streak >= demote_after and not (market_data_only and startup_grace)):
      reason = ",".join(snapshot["reasons"]) or "RELIABILITY_BELOW_THRESHOLD"
      _auto_reliability_transition_to_learning(reason)
      current_mode = "learning"
      transition_reason = reason
  elif not snapshot["ready"] and active_state is None:
    _auto_reliability_enter_learning()
    current_mode = "learning"
    transition_reason = ",".join(snapshot["reasons"]) or "INITIAL_LOW_RELIABILITY"
  if not snapshot["ready"] and current_mode != "live_trading":
    reason = ",".join(snapshot["reasons"]) or "RELIABILITY_BELOW_THRESHOLD"
    _auto_reliability_enforce_sell_only(reason)
  if current_mode != "live_trading":
    _auto_reliability_learning_maintenance(now, tuple(snapshot["active_markets"]))
  with _live_lock:
    _auto_reliability_state.update(
        {
            **snapshot,
            "enabled": True,
            "mode": current_mode,
            "ready_streak": ready_streak,
            "unready_streak": unready_streak,
            "last_error": None,
        }
    )
    if transition_reason:
      _auto_reliability_state["last_transition_at"] = now.isoformat()
      _auto_reliability_state["last_transition_reason"] = transition_reason
    return _to_jsonable(dict(_auto_reliability_state))


def _live_market_data_startup_grace_active(now: datetime) -> bool:
  """Allow the realtime socket to warm up before a data-only demotion.

  Per-order tick/book freshness remains fail-closed throughout this window, so
  the grace period cannot authorize an order using stale data.  It only avoids
  tearing live mode down before KIS has acknowledged subscriptions and emitted
  the first trade/book pair after a clean process restart.
  """
  with _live_lock:
    active = _operation_mode_state.get("active")
  payload = _to_jsonable(active) if active is not None else {}
  if not isinstance(payload, dict) or str(payload.get("mode") or "") != "live_trading":
    return False
  try:
    started_at = datetime.fromisoformat(
        str(payload.get("started_at") or "").replace("Z", "+00:00")
    )
    if started_at.tzinfo is None:
      started_at = started_at.replace(tzinfo=timezone.utc)
  except (TypeError, ValueError):
    return False
  grace_seconds = max(
      30,
      _auto_reliability_int("AUTO_RELIABILITY_MARKET_DATA_STARTUP_GRACE_SECONDS", 180, 30),
  )
  return max(0.0, (now - started_at).total_seconds()) < grace_seconds


def _auto_reliability_loop() -> None:
  interval = _auto_reliability_int("AUTO_RELIABILITY_CHECK_SECONDS", 15, 5)
  while not _auto_reliability_stop.is_set():
    try:
      _auto_reliability_step()
    except Exception as exc:  # noqa: BLE001 - controller fails closed and retries.
      error = f"{type(exc).__name__}: {exc}"
      with _live_lock:
        _auto_reliability_state["ready"] = False
        _auto_reliability_state["last_error"] = error
        _auto_reliability_state["evaluated_at"] = datetime.now(timezone.utc).isoformat()
      if _active_operation_mode() == "live_trading":
        _auto_reliability_transition_to_learning(error)
      audit.record("auto_reliability_controller_error", {"error": error})
    _auto_reliability_stop.wait(interval)


def _start_auto_reliability_controller() -> None:
  global _auto_reliability_worker
  if not AUTO_RELIABILITY_MODE_ENABLED:
    return
  if _auto_reliability_worker is not None and _auto_reliability_worker.is_alive():
    return
  _auto_reliability_stop.clear()
  _auto_reliability_worker = threading.Thread(
      target=_auto_reliability_loop,
      name="auto-reliability-mode-controller",
      daemon=True,
  )
  _auto_reliability_worker.start()


def _stop_auto_reliability_controller() -> None:
  _auto_reliability_stop.set()


def _is_simulation_mode(mode: Any | None = None) -> bool:
  return False


def _simulation_can_use_live_store() -> bool:
  return False


def _analysis_research_for_current_mode(current_store: LocalResearchStore) -> StoredResearch:
  # Both learning and live trading operate against the unified realtime store.
  # The durable corpus is intentionally large and remains available to the
  # training pipeline, but an online ontology refresh must never materialize the
  # entire history (currently >1M snapshots) merely because reliability control
  # temporarily switched from live_trading to learning.
  return current_store.load_live_analysis_inputs()


def _current_data_policy() -> dict[str, Any]:
  return {
      "mode": _active_operation_mode(),
      "primary_store": _get_store_root().as_posix(),
      "analysis_input_stores": [_get_store_root().as_posix()],
      "synthetic_data_allowed": False,
      "orders_in_paper_trading": False,
      "model_root": "data/models",
      "rule": "Learning and live trading use the unified realtime data store only; paper trading is removed.",
  }


def _merge_stored_research(base: StoredResearch, overlay: StoredResearch) -> StoredResearch:
  return StoredResearch(
      events=_unique_by_attr((*base.events, *overlay.events), "event_id"),
      raw_records=_unique_raw_records((*base.raw_records, *overlay.raw_records)),
      market_snapshots=_unique_market_snapshots((*base.market_snapshots, *overlay.market_snapshots)),
      macro_metrics=_unique_macro_metrics((*base.macro_metrics, *overlay.macro_metrics)),
      realtime_quotes=(*base.realtime_quotes, *overlay.realtime_quotes),
      realtime_executions=(*base.realtime_executions, *overlay.realtime_executions),
      graph_triples=_unique_graph_triples((*base.graph_triples, *overlay.graph_triples)),
      reasoning_paths=_unique_by_attr((*base.reasoning_paths, *overlay.reasoning_paths), "path_id"),
  )


def _unique_by_attr(items: tuple[Any, ...], attr_name: str) -> tuple[Any, ...]:
  by_key: dict[str, Any] = {}
  for item in items:
    by_key[str(getattr(item, attr_name))] = item
  return tuple(by_key.values())


def _unique_raw_records(records: tuple[Any, ...]) -> tuple[Any, ...]:
  by_key: dict[str, Any] = {}
  for record in records:
    source = record.source
    key = f"{source.source_id or source.raw_url or record.payload[:80]}:{source.retrieved_at.isoformat()}"
    by_key[key] = record
  return tuple(by_key.values())


def _unique_market_snapshots(records: tuple[Any, ...]) -> tuple[Any, ...]:
  by_key: dict[str, Any] = {}
  for record in records:
    source = record.source
    key = f"{record.ticker}:{source.source_id or source.raw_url}:{source.retrieved_at.isoformat()}"
    by_key[key] = record
  return tuple(by_key.values())


def _unique_macro_metrics(records: tuple[Any, ...]) -> tuple[Any, ...]:
  by_key: dict[str, Any] = {}
  for record in records:
    by_key[f"{record.name}:{record.observed_at.isoformat()}"] = record
  return tuple(by_key.values())


def _unique_graph_triples(records: tuple[Any, ...]) -> tuple[Any, ...]:
  by_key: dict[str, Any] = {}
  for record in records:
    by_key[f"{record.subject}|{record.predicate}|{record.object}|{record.evidence_id}"] = record
  return tuple(by_key.values())


def _start_streaming_demo(
  target_return_rate: float = 0.02,
  period_minutes: int = 390,
  initial_cash: float = 10_000_000,
  seed: int = 42,
  acceleration_factor: float = 1.0,
  profit_gain: float = 1.0,
  max_speed: bool = False,
) -> str:
  """Start a streaming accelerated demo and return demo_id."""
  demo_id = str(uuid4())
  
  if target_return_rate > 1:
    target_return_rate /= 100.0
  initial_cash = max(1.0, float(initial_cash))
  # 시간 팩터 제거: 배속/가상시간을 쓰지 않고 항상 실시간(1.0배) 기준으로 동작한다.
  # acceleration_factor 인자는 하위 호환을 위해 남겨두지만 더 이상 속도에 영향을 주지 않는다.
  acceleration_factor = 1.0
  profit_gain = max(0.25, min(4.0, float(profit_gain)))

  demo = StreamingAcceleratedDemo(
      config=TimeScalerConfig(
          mode=TimeMode.REALTIME,
          acceleration_factor=acceleration_factor,
      ),
      target_return_rate=target_return_rate,
      period_minutes=period_minutes,
      initial_cash=initial_cash,
      profit_gain_multiplier=profit_gain,
      seed=seed,
      max_speed=max_speed,
  )
  demo.initialize()
  
  with _streaming_demos_lock:
    _streaming_demos[demo_id] = demo
    _streaming_demo_step_locks[demo_id] = threading.Lock()
  
  audit.record("streaming_demo_started", {
      "demo_id": demo_id,
      "target_return_rate": target_return_rate,
      "period_minutes": period_minutes,
      "initial_cash": initial_cash,
      "acceleration_factor": acceleration_factor,
      "profit_gain": profit_gain,
  })
  
  return demo_id


_live_worker: threading.Thread | None = None
_refresh_worker: threading.Thread | None = None
_kis_realtime_collector_worker: threading.Thread | None = None
# 총자산 이력 샘플러: 수집·학습·거래와 독립된 읽기 전용 스레드.
_asset_history_sampler_worker: threading.Thread | None = None
_asset_history_sampler_stop = threading.Event()
# 실시간 거래 엔진: 학습 워커(_live_worker)와 완전히 독립된 스레드/상태.
_realtime_trading_worker: threading.Thread | None = None
_realtime_trading_stop = threading.Event()
_realtime_trading_engine: Any | None = None
_realtime_trading_lock = threading.Lock()
# 주기적 백그라운드 학습 워커: 수집·트레이딩과 독립된 스레드.
_live_training_worker: threading.Thread | None = None
_live_training_stop = threading.Event()
_live_training_heartbeat: dict[str, Any] = {
    "started_at": None,
    "finished_at": None,
    "ok": False,
    "skipped": False,
    "artifact_id": None,
    "error": None,
}
_live_state: dict[str, Any] = {
    "context": None,
    "research_result": None,
    "research_last_collected_at": None,
    "context_mode": None,
    "store_summary": {},
    "stored_new_records": {},
    "last_updated": None,
    "last_error": None,
    "is_refreshing": False,
    "progress": {
        "active": False,
        "percent": 0,
        "stage": "idle",
        "message": "Waiting",
        "started_at": None,
        "updated_at": None,
    },
    "stop": False,
    "learning_active": False,
    "learning_mode": None,
    "learning_started_at": None,
    "learning_stopped_at": None,
    "learning_next_collection_at": None,
    "collection_cycle": 0,
    "collection_log": [],
    "graph_payload": None,
    "graph_payload_context_id": None,
    "live_execution_summary": None,
    "refresh_requested_after_current": False,
}
_realtime_runtime_static_cache: dict[str, Any] = {}


def _clear_live_analysis_cache_unlocked() -> None:
  store_summary = dict(_live_state.get("store_summary") or {})
  if not store_summary:
    try:
      store_summary = LocalResearchStore(root=_get_store_root()).summary()
    except Exception:  # noqa: BLE001 - cache clearing must never block mode changes.
      store_summary = {}
  _live_state["context"] = None
  _live_state["research_result"] = None
  _live_state["context_mode"] = None
  _live_state["graph_payload"] = None
  _live_state["graph_payload_context_id"] = None
  _live_state["store_summary"] = store_summary
  _live_state["stored_new_records"] = {}
  _live_state["live_trading_baseline_equity"] = None


def _asset_history_sampler_loop() -> None:
    """Persist a total-asset snapshot every ASSET_HISTORY_SAMPLE_SECONDS.

    Runs independently of any open browser so the dashboard asset-history curve
    accumulates continuously (the Pi kiosk shows the trade display, not /account).
    Zero/empty snapshots are skipped so the chart is not polluted before the first
    live account read. Never raises — sampling must not affect trading.
    """
    while not _asset_history_sampler_stop.is_set():
        try:
            dashboard = _account_service.build_dashboard(persist=False)
            snapshot = dashboard.get("snapshot") or {}
            if float(snapshot.get("total_asset_krw") or 0.0) > 0.0:
                _account_service.store.save_dashboard(dashboard)
        except Exception:  # noqa: BLE001 - a sampling error must never break the server.
            pass
        _asset_history_sampler_stop.wait(ASSET_HISTORY_SAMPLE_SECONDS)


def _start_asset_history_sampler() -> None:
    global _asset_history_sampler_worker
    if _asset_history_sampler_worker is not None and _asset_history_sampler_worker.is_alive():
        return
    _asset_history_sampler_stop.clear()
    _asset_history_sampler_worker = threading.Thread(
        target=_asset_history_sampler_loop, name="asset-history-sampler", daemon=True
    )
    _asset_history_sampler_worker.start()


def _stop_asset_history_sampler() -> None:
    _asset_history_sampler_stop.set()


@app.on_event("startup")
def _startup_live_worker() -> None:
    RealtimeAccelerationPolicy().apply_process_hints()
    configure_default_event_llm_env()
    try:
      llm_status = event_llm_runtime_status()
      llm_status["probe_skipped"] = True
      llm_status["probed_at_startup"] = True
      _realtime_runtime_static_cache.update(
          {
              "acceleration": RealtimeAccelerationPolicy().status(),
              "event_llm": llm_status,
              "ontology_npu": get_ontology_npu_classifier().status(),
          }
      )
    except Exception as exc:  # noqa: BLE001 - diagnostics must not block startup.
      _realtime_runtime_static_cache["error"] = f"{exc.__class__.__name__}: {exc}"
    if AUTO_START_ASSET_HISTORY_SAMPLER:
      _start_asset_history_sampler()
    if AUTO_START_KIS_REALTIME_COLLECTOR:
      _start_kis_realtime_collector()
      _start_kis_overseas_realtime_collector()
    if AUTO_START_US_EXCHANGE_MAP_REFRESH:
      _start_us_exchange_map_refresher()
    if AUTO_START_INVESTOR_FLOW_REFRESH:
      _start_investor_flow_refresher()
    if AUTO_START_WEEKEND_BRIEF:
      _start_weekend_brief_worker()
    if AUTO_START_LIVE_WORKER:
      _start_live_worker("learning")
    if AUTO_START_LIVE_READINESS:
      _start_auto_live_readiness_check()
    if AUTO_START_REALTIME_TRADING:
      _start_realtime_trading_engine()
    if AUTO_START_LIVE_TRAINING:
      _start_live_training_worker()
    _ensure_us_fast_poll_started()
    _ensure_krx_feature_frame_started()
    _start_auto_reliability_controller()


def _graceful_teardown() -> list[str]:
    """Stop every background worker in dependency order. Returns what was stopped.

    The ONE teardown path. It used to exist only as a FastAPI shutdown handler,
    which meant it ran on a clean uvicorn stop and never on the two exits that
    actually happen in practice: ``run.ps1`` force-killing the process, and
    ``_schedule_app_process_shutdown`` calling ``os._exit(0)``. Both skipped it, so
    the realtime trading engine was terminated mid-cycle with in-flight state.

    Trading stops FIRST and market-data collectors after it, never the reverse: an
    engine still evaluating while its price feed disappears is exactly the
    stale-data condition every gate in this codebase exists to prevent.
    """
    stopped: list[str] = []
    for name, stop in (
        # Trading first — nothing else matters if orders can still be submitted.
        ("realtime_trading_engine", _stop_realtime_trading_engine),
        ("auto_reliability_controller", _stop_auto_reliability_controller),
        ("live_training_worker", _stop_live_training_worker),
        ("krx_feature_frame_worker", _stop_krx_feature_frame_worker),
        # Feeds last, so exits could still price correctly on the way down.
        ("kis_overseas_realtime_collector", _stop_kis_overseas_realtime_collector),
        ("kis_realtime_collector", _stop_kis_realtime_collector),
        ("asset_history_sampler", _stop_asset_history_sampler),
        ("live_worker", _stop_live_worker),
    ):
        try:
            stop()
            stopped.append(name)
        except Exception as exc:  # noqa: BLE001 - one stuck worker must not block the rest
            audit.record(
                "graceful_teardown_worker_failed",
                {"worker": name, "error": str(exc) or exc.__class__.__name__},
            )
    return stopped


@app.on_event("shutdown")
def _shutdown_live_worker() -> None:
    _graceful_teardown()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


DISPLAY_HTML = """<!doctype html>
<html lang="ko"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>온톨로지 디스플레이</title>
<link rel="icon" type="image/png" href="/static/icon.png">
<link rel="apple-touch-icon" href="/static/icon.png">
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  html,body{height:100%;width:100%;overflow:hidden;background:#0b0f16;font-family:system-ui,-apple-system,"Malgun Gothic",sans-serif;cursor:default}
  #c{position:fixed;inset:0;display:block;touch-action:none}
  #bar{position:fixed;top:0;left:0;right:0;display:flex;align-items:center;gap:8px;padding:5px 52px 5px 9px;font-size:11px;color:#8b98a9;background:linear-gradient(#0b0f16cc,transparent);z-index:3;pointer-events:none}
  #bar .t{color:#dce6f2;font-weight:600}
  #bar .dot{width:7px;height:7px;border-radius:50%;background:#38bdf8;box-shadow:0 0 8px #38bdf8}
  #cnt{margin-left:auto;font-variant-numeric:tabular-nums}
  #exit{position:fixed;top:6px;right:8px;z-index:5;width:36px;height:36px;border-radius:9px;border:1px solid #37424f;background:rgba(20,26,34,.78);color:#e6edf3;font-size:17px;line-height:1;cursor:pointer;display:flex;align-items:center;justify-content:center;pointer-events:auto;-webkit-tap-highlight-color:transparent}
  #exit:active{background:#ef4444;border-color:#ef4444}
  #lg{position:fixed;bottom:3px;left:0;right:0;display:flex;flex-wrap:wrap;justify-content:center;gap:3px 8px;padding:3px;font-size:9px;color:#aeb9c7;z-index:3;pointer-events:none}
  #lg span{display:inline-flex;align-items:center;gap:3px}
  #lg i{width:8px;height:8px;border-radius:2px;display:inline-block}
  #empty{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;color:#5b6675;font-size:13px;z-index:2}
</style></head>
<body>
<canvas id="c"></canvas>
<button id="exit" title="종료" aria-label="종료">✕</button>
<div id="bar"><span class="dot"></span><span class="t">온톨로지 지식 그래프</span><span id="cnt">연결 중…</span></div>
<div id="empty">온톨로지 그래프 생성 대기…</div>
<div id="lg">
  <span><i style="background:#38bdf8"></i>종목</span><span><i style="background:#f97316"></i>이벤트</span>
  <span><i style="background:#22c55e"></i>긍정</span><span><i style="background:#ef4444"></i>리스크</span>
  <span><i style="background:#d946ef"></i>상충</span><span><i style="background:#84cc16"></i>섹터</span>
  <span><i style="background:#a9b6c6"></i>개체</span>
</div>
<script>
(function(){
  "use strict";
  const KIND = {
    ticker:{ko:"종목",c:"#38bdf8"}, event:{ko:"이벤트",c:"#f97316"},
    temporal:{ko:"시간",c:"#22d3ee"}, support:{ko:"긍정",c:"#22c55e"},
    risk:{ko:"리스크",c:"#ef4444"}, contradiction:{ko:"상충",c:"#d946ef"},
    sector:{ko:"섹터",c:"#84cc16"}, pipeline:{ko:"파이프",c:"#3b82f6"},
    tuning:{ko:"튜닝",c:"#eab308"}, parameter:{ko:"파라미터",c:"#ec4899"},
    metric:{ko:"메트릭",c:"#94a3b8"}, entity:{ko:"개체",c:"#a9b6c6"}
  };
  const colorOf = (k)=> (KIND[k]||{c:"#94a3b8"}).c;

  const canvas = document.getElementById("c");
  const ctx = canvas.getContext("2d");
  const view = {scale:1, tx:0, ty:0};
  let W=0,H=0,DPR=1;
  function resize(){
    DPR = Math.min(window.devicePixelRatio||1, 2);
    W = window.innerWidth; H = window.innerHeight;
    canvas.width = Math.round(W*DPR); canvas.height = Math.round(H*DPR);
    ctx.setTransform(DPR,0,0,DPR,0,0);
  }
  window.addEventListener("resize", resize);

  let state = null;   // {nodes, links, nodeMap, adj, maxDeg}
  let alpha = 0, hover=null, dragNode=null, panning=false, lastX=0, lastY=0, moved=false, autoFit=true;
  const reheat=(a)=>{ alpha=Math.max(alpha,a); };
  const worldToScreen=(x,y)=>({x:x*view.scale+view.tx, y:y*view.scale+view.ty});
  const screenToWorld=(sx,sy)=>({x:(sx-view.tx)/view.scale, y:(sy-view.ty)/view.scale});

  function build(data){
    const prev = state ? state.nodeMap : null;   // reuse positions across polls -> stable/tappable
    const raw = (data.nodes||[]).map((n,i)=>{
      const p = prev ? prev.get(n.id) : null;
      const a=i*2.399, r=90+Math.sqrt(i)*26;
      return {id:n.id,label:n.label||n.id,kind:n.kind||"entity",imp:Number(n.importance_score||0),deg:0,
        x: p?p.x:Math.cos(a)*r, y: p?p.y:Math.sin(a)*r, vx:0, vy:0, fixed:false};
    });
    const map = new Map(raw.map(n=>[n.id,n]));
    const links = (data.links||[]).filter(l=>map.has(l.source)&&map.has(l.target)).map(l=>({s:map.get(l.source),t:map.get(l.target),p:l.predicate}));
    const adj = new Map(raw.map(n=>[n.id,new Set()]));
    links.forEach(l=>{ l.s.deg++; l.t.deg++; adj.get(l.s.id).add(l.t.id); adj.get(l.t.id).add(l.s.id); });
    let maxDeg=1; raw.forEach(n=>{ if(n.deg>maxDeg) maxDeg=n.deg; });
    state = {nodes:raw, links, nodeMap:map, adj, maxDeg};
    if(prev){ alpha = 0.045; }                              // poll update: barely move, keep view -> accurate touch
    else { alpha = raw.length>260 ? 0.5 : 1; autoFit=true; fit(); }  // first load: bloom + fit
  }
  const radiusOf=(n)=> 3.2 + Math.sqrt(n.deg)*2.6;
  const screenRadius=(n)=> Math.max(2, radiusOf(n)*Math.max(0.6, Math.min(view.scale,1.7)));

  function fit(){
    if(!state||!state.nodes.length){ view.scale=1; view.tx=W/2; view.ty=H/2; return; }
    let a=Infinity,b=Infinity,c=-Infinity,d=-Infinity;
    state.nodes.forEach(n=>{ if(n.x<a)a=n.x; if(n.x>c)c=n.x; if(n.y<b)b=n.y; if(n.y>d)d=n.y; });
    const w=Math.max(1,c-a), h=Math.max(1,d-b);
    const pad=22;
    view.scale=Math.max(0.15, Math.min(6, Math.min((W-pad)/w, (H-pad)/h)));
    view.tx=W/2-((a+c)/2)*view.scale; view.ty=H/2-((b+d)/2)*view.scale;
  }

  function stepSim(){
    if(!state || alpha<0.02) return;
    const ns=state.nodes;
    for(let i=0;i<ns.length;i++){ const p=ns[i];
      for(let j=i+1;j<ns.length;j++){ const q=ns[j];
        let dx=p.x-q.x, dy=p.y-q.y, d2=dx*dx+dy*dy; if(d2<0.01){d2=0.01;dx=Math.random()-0.5;dy=Math.random()-0.5;}
        const dd=Math.sqrt(d2), f=Math.min(6500/d2,42), fx=dx/dd*f, fy=dy/dd*f;
        p.vx+=fx;p.vy+=fy;q.vx-=fx;q.vy-=fy;
      }
    }
    state.links.forEach(l=>{ let dx=l.t.x-l.s.x, dy=l.t.y-l.s.y; const dd=Math.sqrt(dx*dx+dy*dy)||0.01, f=(dd-84)*0.045, fx=dx/dd*f, fy=dy/dd*f;
      l.s.vx+=fx;l.s.vy+=fy;l.t.vx-=fx;l.t.vy-=fy; });
    var aspect=(H>0?W/H:1), gx=0.0032, gy=gx*aspect*aspect; // anisotropic: blob matches screen aspect
    ns.forEach(n=>{ n.vx+=(-n.x)*gx; n.vy+=(-n.y)*gy; if(n.fixed){n.vx=0;n.vy=0;return;} n.vx*=0.86; n.vy*=0.86; n.x+=n.vx*alpha*1.4; n.y+=n.vy*alpha*1.4; });
    alpha*=0.992;
  }

  function draw(){
    stepSim();
    if(autoFit && state && alpha>0.05) fit();   // keep the graph filling the screen while it settles
    const g=ctx.createLinearGradient(0,0,0,H); g.addColorStop(0,"#0b0f16"); g.addColorStop(1,"#0a1622");
    ctx.setTransform(DPR,0,0,DPR,0,0); ctx.fillStyle=g; ctx.fillRect(0,0,W,H);
    if(!state){ requestAnimationFrame(draw); return; }
    const foc=hover, neigh=foc?state.adj.get(foc.id):null, DIM=0.13;
    ctx.setTransform(DPR,0,0,DPR,0,0);
    state.links.forEach(l=>{ const a=worldToScreen(l.s.x,l.s.y), b=worldToScreen(l.t.x,l.t.y);
      const near=!foc||foc===l.s||foc===l.t;
      ctx.strokeStyle= near&&foc? "#5aa9ff" : "rgba(150,162,178,.26)";
      ctx.globalAlpha= foc? (near?0.8:DIM) : 0.5;
      ctx.lineWidth= near&&foc?1.5:0.7;
      ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
    });
    ctx.globalAlpha=1;
    const showHubLabels = state.nodes.length<=140;
    state.nodes.forEach(n=>{ const p=worldToScreen(n.x,n.y), r=screenRadius(n), near=!foc||n===foc||(neigh&&neigh.has(n.id));
      ctx.globalAlpha= foc?(near?1:DIM):0.95;
      if(n===foc){ ctx.beginPath(); ctx.arc(p.x,p.y,r+6,0,7); ctx.fillStyle=colorOf(n.kind); ctx.globalAlpha=0.2; ctx.fill(); ctx.globalAlpha=near?1:DIM; }
      ctx.beginPath(); ctx.arc(p.x,p.y,r,0,Math.PI*2); ctx.fillStyle=colorOf(n.kind); ctx.fill();
      if(n===foc){ ctx.lineWidth=1.5; ctx.strokeStyle="rgba(255,255,255,.8)"; ctx.stroke(); }
      const wantLabel = (foc&&near) || view.scale>1.5 || (showHubLabels && n.deg>=Math.max(3, state.maxDeg*0.5));
      if(wantLabel){ const fs=Math.max(9,11/Math.max(1,view.scale*0.6)); ctx.font=fs+"px system-ui,sans-serif"; ctx.textAlign="center"; ctx.textBaseline="top";
        const t=n.label.length>18?n.label.slice(0,17)+"…":n.label; const w=ctx.measureText(t).width;
        ctx.globalAlpha= foc?(near?1:DIM):0.9; ctx.fillStyle="rgba(8,12,18,.78)"; ctx.fillRect(p.x-w/2-2,p.y+r+2,w+4,fs+2);
        ctx.fillStyle="#dce6f2"; ctx.fillText(t,p.x,p.y+r+3);
      }
    });
    ctx.globalAlpha=1;
    requestAnimationFrame(draw);
  }

  function pick(sx,sy){ if(!state) return null; let best=null,bd=Infinity;
    for(const n of state.nodes){ const p=worldToScreen(n.x,n.y), r=Math.max(screenRadius(n)+8,13), dx=p.x-sx, dy=p.y-sy, d=dx*dx+dy*dy; if(d<r*r&&d<bd){bd=d;best=n;} }
    return best; }
  canvas.addEventListener("pointerdown",e=>{ moved=false; lastX=e.clientX; lastY=e.clientY; const n=pick(e.clientX,e.clientY); if(n){dragNode=n;n.fixed=true;reheat(0.5);} else panning=true; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove",e=>{ if(dragNode){autoFit=false; const w=screenToWorld(e.clientX,e.clientY); dragNode.x=w.x; dragNode.y=w.y; dragNode.vx=0; dragNode.vy=0; moved=true; reheat(0.4); return;}
    if(panning){ autoFit=false; view.tx+=e.clientX-lastX; view.ty+=e.clientY-lastY; lastX=e.clientX; lastY=e.clientY; moved=true; return; }
    hover=pick(e.clientX,e.clientY); });
  canvas.addEventListener("pointerup",()=>{ if(dragNode)dragNode.fixed=false; dragNode=null; panning=false; });
  canvas.addEventListener("pointerleave",()=>{ hover=null; });
  canvas.addEventListener("wheel",e=>{ e.preventDefault(); autoFit=false; const f=e.deltaY>0?0.9:1.1, ns=Math.max(0.2,Math.min(4,view.scale*f));
    view.tx=e.clientX-(e.clientX-view.tx)*(ns/view.scale); view.ty=e.clientY-(e.clientY-view.ty)*(ns/view.scale); view.scale=ns; },{passive:false});
  window.addEventListener("dblclick",()=>{ autoFit=true; fit(); reheat(0.6); });

  let sig=null;
  async function poll(){
    try{
      const d=await (await fetch("/api/ontology/graph",{cache:"no-store"})).json();
      const total=(d.counts&&d.counts.nodes)||0, shown=(d.nodes||[]).length;
      document.getElementById("cnt").textContent = total? (shown+" / "+total+" 노드 · "+(d.links||[]).length+" 링크") : "대기 중";
      const s=total+":"+shown+":"+((d.reasoning_steps||[]).length);
      const empty=document.getElementById("empty");
      if(total>0 && shown>0){ if(s!==sig){ sig=s; build(d); } empty.style.display="none"; }
      else { empty.style.display="flex"; }
    }catch(err){ /* keep last graph */ }
  }
  var exitBtn=document.getElementById("exit");
  var embedded = location.search.indexOf("embed")>=0;
  if(embedded){ if(exitBtn) exitBtn.style.display="none"; canvas.style.pointerEvents="none"; }
  if(exitBtn && !embedded){
    exitBtn.addEventListener("click", async function(){
      exitBtn.textContent="…";
      try{ await fetch("/api/kiosk/exit",{method:"POST"}); }catch(e){}
      try{ window.close(); }catch(e){}
    });
  }
  resize(); requestAnimationFrame(draw); poll(); setInterval(poll,12000);
})();

</script>
</body></html>"""


TRADE_DISPLAY_HTML = """<!doctype html>
<html lang="ko"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>온톨로지 기반 투자 프로그램</title>
<link rel="icon" type="image/png" href="/static/icon.png">
<link rel="apple-touch-icon" href="/static/icon.png">
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  :root{--bg:#0b0f16;--card:#141b26;--line:#263243;--muted:#8b98a9;--txt:#e6edf3}
  html,body{height:100%;width:100%;background:var(--bg);color:var(--txt);font-family:system-ui,-apple-system,"Malgun Gothic",sans-serif;overflow:hidden}
  #app{display:flex;flex-direction:column;height:100vh}
  header{display:flex;align-items:center;gap:8px;padding:8px 48px 8px 12px;border-bottom:1px solid var(--line);flex:0 0 auto}
  header .dot{width:9px;height:9px;border-radius:50%;background:#22c55e;box-shadow:0 0 8px #22c55e;flex:0 0 auto}
  header .dot.off{background:#64748b;box-shadow:none}
  header h1{font-size:15px;font-weight:800;letter-spacing:-.2px;white-space:nowrap}
  header .clock{margin-left:auto;font-variant-numeric:tabular-nums;color:var(--muted);font-size:13px}
  #exit{position:fixed;top:6px;right:8px;z-index:5;width:34px;height:34px;border-radius:9px;border:1px solid var(--line);background:rgba(20,26,34,.8);color:var(--txt);font-size:16px;line-height:1;cursor:pointer;display:flex;align-items:center;justify-content:center;-webkit-tap-highlight-color:transparent}
  #exit:active{background:#ef4444;border-color:#ef4444}
  #overview{flex:0 0 auto;padding:8px;border-bottom:1px solid var(--line);display:grid;grid-template-columns:1.15fr 1fr 1fr;gap:7px;background:#0d131d}
  .ov{min-width:0;border:1px solid var(--line);border-radius:8px;background:#121a26;padding:7px 8px;position:relative;overflow:hidden}
  .ov::before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--accent,#64748b)}
  .ov b{display:block;font-size:14px;line-height:1.15;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .ov span{display:block;margin-top:3px;font-size:10px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .ov small{display:block;margin-top:3px;font-size:10px;color:#9fb0c4;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .ov-primary b{font-size:18px}
  .ov-open{--accent:#22c55e}.ov-pre{--accent:#f59e0b}.ov-after{--accent:#38bdf8}.ov-day{--accent:#a78bfa}.ov-busy{--accent:#f97316}.ov-idle{--accent:#64748b}.ov-closed{--accent:#475569}
  main{flex:1 1 auto;overflow-y:auto;padding:8px;display:flex;flex-direction:column;gap:8px}
  .card{position:relative;flex:0 0 auto;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 12px 10px 18px;overflow:hidden}
  .card::before{content:"";position:absolute;left:0;top:0;bottom:0;width:6px;background:var(--accent,#64748b)}
  .row1{display:flex;align-items:flex-start;gap:8px;margin-bottom:3px}
  .badge{font-size:12px;font-weight:800;padding:2px 9px;border-radius:999px;background:var(--accent,#64748b);color:#06121e;white-space:nowrap;flex:0 0 auto}
  .name{font-size:15px;font-weight:700}
  .tk{font-size:11px;color:var(--muted);align-self:center}
  .when{margin-left:auto;font-size:11px;color:var(--muted);text-align:right;line-height:1.35;white-space:nowrap}
  .headline{font-size:17px;font-weight:800;letter-spacing:-.3px;margin:3px 0 7px}
  .why{font-size:11px;color:var(--muted);margin-bottom:4px}
  .chips{display:flex;flex-wrap:wrap;gap:5px}
  .chip{font-size:12px;padding:3px 9px;border-radius:8px;background:#0f1622;border:1px solid var(--line);color:#cdd8e6}
  .tone-buy{--accent:#22c55e}.tone-profit{--accent:#22c55e}.tone-sell{--accent:#38bdf8}
  .tone-loss{--accent:#ef4444}.tone-warn{--accent:#f59e0b}.tone-hold{--accent:#64748b}
  #empty{margin:auto;text-align:center;color:#5b6675;font-size:14px;line-height:1.8;padding:16px}
  #status{padding:6px 12px;border-top:1px solid var(--line);font-size:11px;color:var(--muted);flex:0 0 auto;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  #asset{flex:0 0 auto;display:flex;align-items:center;gap:10px;padding:6px 12px;border-bottom:1px solid var(--line);background:#0d131d}
  #asset .col{flex:0 0 auto;min-width:0}
  #asset .lab{font-size:10px;color:var(--muted);line-height:1.2}
  #asset .val{font-size:18px;font-weight:800;font-variant-numeric:tabular-nums;line-height:1.15}
  #asset .delta{font-size:12px;font-weight:700;font-variant-numeric:tabular-nums;white-space:nowrap}
  #asset .delta.up{color:#22c55e}#asset .delta.down{color:#ef4444}
  #asset canvas{flex:1 1 auto;height:54px;min-width:0;display:block}
  #cash{flex:0 0 auto;display:flex;align-items:stretch;gap:8px;padding:6px 12px;border-bottom:1px solid var(--line);background:#0d131d}
  #cash .col{flex:1 1 0;min-width:0;border:1px solid var(--line);border-radius:8px;background:#121a26;padding:6px 9px;position:relative;overflow:hidden}
  #cash .col::before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--accent,#64748b)}
  #cash .col.krw{--accent:#a78bfa}#cash .col.fx{--accent:#38bdf8}
  #cash .lab{font-size:10px;color:var(--muted);line-height:1.2}
  #cash .val{margin-top:2px;font-size:17px;font-weight:800;font-variant-numeric:tabular-nums;line-height:1.15;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  #cash .sub{margin-top:1px;font-size:10px;color:#9fb0c4;line-height:1.2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  #cash.off{opacity:.55}
</style></head>
<body>
<div id="app">
  <header><span class="dot" id="dot"></span><h1>온톨로지 기반 투자 프로그램</h1><span class="clock" id="clock"></span></header>
  <section id="asset" aria-label="총자산 추이">
    <div class="col"><div class="lab">총자산</div><div class="val" id="asset-val">—</div></div>
    <div class="col delta" id="asset-delta"></div>
    <canvas id="asset-spark" height="54"></canvas>
  </section>
  <section id="cash" aria-label="주문 가능 잔액">
    <div class="col krw"><div class="lab">주문가능 원화</div><div class="val" id="cash-krw">—</div><div class="sub" id="cash-krw-sub">KRW</div></div>
    <div class="col fx"><div class="lab">주문가능 외화</div><div class="val" id="cash-fx">—</div><div class="sub" id="cash-fx-sub">원화환산</div></div>
  </section>
  <section id="overview" aria-label="현재 상태"></section>
  <main id="list"></main>
  <div id="status">연결 중…</div>
</div>
<button id="exit" title="종료" aria-label="종료">✕</button>
<script>
(function(){
  "use strict";
  var TONE_LABEL={buy:"매수",profit:"매도·이익",sell:"매도",loss:"매도·손실",warn:"차단",hold:"보류"};
  function h(tag,cls,txt){var e=document.createElement(tag);if(cls)e.className=cls;if(txt!=null)e.textContent=txt;return e;}
  function clockTick(){document.getElementById("clock").textContent=new Date().toLocaleTimeString("ko-KR",{hour12:false});}
  setInterval(clockTick,1000);clockTick();
  var exitBtn=document.getElementById("exit");
  exitBtn.addEventListener("click",async function(){exitBtn.textContent="…";try{await fetch("/api/kiosk/exit",{method:"POST"});}catch(e){}try{window.close();}catch(e){}});
  function toneClass(t){return "ov-"+(t||"idle");}
  function overviewCard(label,detail,tone,sub,primary){
    var box=h("div","ov "+toneClass(tone)+(primary?" ov-primary":""));
    box.appendChild(h("b","",label||"-"));
    box.appendChild(h("span","",detail||""));
    if(sub)box.appendChild(h("small","",sub));
    return box;
  }
  function renderOverview(o){
    var root=document.getElementById("overview");
    root.innerHTML="";
    o=o||{};
    var p=o.primary||{};
    var times=o.times||{};
    root.appendChild(overviewCard(p.label||"상태 확인 중",p.detail||"",p.tone||"idle","KST "+(times.kst||"--:--")+" · ET "+(times.et||"--:--"),true));
    (o.markets||[]).slice(0,2).forEach(function(m){root.appendChild(overviewCard(m.label,m.detail,m.tone,m.key));});
    var w=o.work||{};
    root.appendChild(overviewCard(w.label||"뉴스 분석 대기",w.detail||"",w.tone||"idle","뉴스/데이터"));
  }
  function wonShort(v){return "₩"+Math.round(Number(v||0)).toLocaleString("ko-KR");}
  function renderOrderableCash(oc){
    var sec=document.getElementById("cash");
    var krwEl=document.getElementById("cash-krw"),fxEl=document.getElementById("cash-fx"),fxSub=document.getElementById("cash-fx-sub");
    if(!oc||!oc.available){sec.className="off";krwEl.textContent="—";fxEl.textContent="—";fxSub.textContent="계좌 연결 대기";return;}
    sec.className="";
    krwEl.textContent=wonShort(oc.krw);
    var ccy=oc.foreign_currency||"USD";
    var amt=Number(oc.foreign_native||0);
    fxEl.textContent=(ccy==="USD"?"$"+amt.toLocaleString("en-US",{maximumFractionDigits:2}):amt.toLocaleString("en-US",{maximumFractionDigits:2})+" "+ccy);
    fxSub.textContent="≈ "+wonShort(oc.foreign_krw)+" 원화환산";
  }
  function render(d){
    var list=document.getElementById("list");
    renderOverview(d.overview);
    renderOrderableCash(d.orderable_cash);
    document.getElementById("dot").className="dot"+(d.running?"":" off");
    var a=d.activity||{};
    var activityText="cycle "+(a.cycle||0)+" | buy "+(a.buy_evaluated||0)+"/"+(a.buy_rejected||0)+" | sell "+(a.sell_evaluated||0)+"/"+(a.sell_rejected||0)+" | orders "+(a.submitted||0)+" | ignored "+(a.skipped_ignored||0);
    document.getElementById("status").textContent=(d.running?"자동매매 실행 중":"자동매매 정지")+(d.buy_enabled===false?" · 매수 비활성":"")+" · "+activityText+" · 갱신 "+new Date().toLocaleTimeString("ko-KR",{hour12:false});
    var cards=(d&&d.cards)||[];
    list.innerHTML="";
    if(!cards.length){
      var emp=h("div");emp.id="empty";
      emp.innerHTML="아직 체결된 매매가 없습니다.<br>조건이 충족되면 여기에 <b>왜 샀는지 / 왜 이 가격·시점에 팔았는지</b>가 표시됩니다.";
      list.appendChild(emp);return;
    }
    cards.forEach(function(c){
      var card=h("div","card tone-"+(c.tone||"hold"));
      var r1=h("div","row1");
      r1.appendChild(h("span","badge",TONE_LABEL[c.tone]||c.verb||""));
      r1.appendChild(h("span","name",c.name||c.symbol||""));
      if(c.symbol&&c.symbol!==c.name)r1.appendChild(h("span","tk",c.symbol));
      var when=h("div","when");when.innerHTML=(c.time_ago||"")+(c.time_hm?"<br>"+c.time_hm:"");
      r1.appendChild(when);
      card.appendChild(r1);
      card.appendChild(h("div","headline",c.headline||""));
      var rs=(c.reasons||[]).filter(function(x){return x;});
      if(rs.length){
        card.appendChild(h("div","why","왜:"));
        var chips=h("div","chips");
        rs.forEach(function(x){chips.appendChild(h("span","chip",x));});
        card.appendChild(chips);
      }
      list.appendChild(card);
    });
  }
  async function poll(){
    try{var r=await fetch("/api/trade-explanations",{cache:"no-store"});render(await r.json());}
    catch(e){document.getElementById("status").textContent="연결 오류 — 재시도 중…";}
  }
  poll();setInterval(poll,4000);

  // 총자산 실시간 추이(스파크라인). 서버 백그라운드 샘플러가 이력을 계속 적재하므로
  // 브라우저를 켜두지 않아도 흐름이 이어진다. /api/account/asset-history 는 SQLite 읽기라 가볍다.
  var assetPoints=[];
  function fmtWon(v){return "₩"+Math.round(Number(v||0)).toLocaleString("ko-KR");}
  function hhmm(iso){var d=new Date(iso);if(isNaN(d.getTime()))return "";return d.toLocaleTimeString("ko-KR",{hour12:false,hour:"2-digit",minute:"2-digit"});}
  function drawSpark(){
    var c=document.getElementById("asset-spark");if(!c)return;
    var ctx=c.getContext("2d");
    var w=Math.max(1,c.clientWidth||c.width),hh=c.height;
    if(c.width!==w)c.width=w;
    ctx.clearRect(0,0,w,hh);
    if(assetPoints.length<2)return;
    var vals=assetPoints.map(function(p){return Number(p.total_asset_krw||0);});
    var min=Math.min.apply(null,vals),max=Math.max.apply(null,vals);
    var pad=4,labelH=13,top=pad,bottom=hh-pad-labelH,span=Math.max(1,max-min),up=vals[vals.length-1]>=vals[0];
    var col=up?"#22c55e":"#ef4444";
    var pts=vals.map(function(v,i){return [pad+(i/(vals.length-1))*(w-pad*2),bottom-((v-min)/span)*(bottom-top)];});
    ctx.strokeStyle=col;ctx.lineWidth=2;ctx.beginPath();
    pts.forEach(function(p,i){if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);});
    ctx.stroke();
    var g=ctx.createLinearGradient(0,0,0,bottom);
    g.addColorStop(0,up?"rgba(34,197,94,.22)":"rgba(239,68,68,.22)");g.addColorStop(1,"rgba(0,0,0,0)");
    ctx.lineTo(pts[pts.length-1][0],bottom);ctx.lineTo(pts[0][0],bottom);ctx.closePath();ctx.fillStyle=g;ctx.fill();
    // 시간 축 라벨(HH:MM): 시작·중간·끝. 데이터가 적으면 시작·끝만.
    ctx.fillStyle="#8b98a9";ctx.font="10px system-ui,-apple-system,sans-serif";ctx.textBaseline="alphabetic";
    var ly=hh-3,n=assetPoints.length;
    var ticks=n>=5?[[0,"left",pad],[Math.floor((n-1)/2),"center",w/2],[n-1,"right",w-pad]]:[[0,"left",pad],[n-1,"right",w-pad]];
    ctx.strokeStyle="rgba(139,152,169,.18)";ctx.lineWidth=1;
    ticks.forEach(function(t){
      var label=hhmm(assetPoints[t[0]].created_at);if(!label)return;
      ctx.beginPath();ctx.moveTo(pts[t[0]][0],top);ctx.lineTo(pts[t[0]][0],bottom);ctx.stroke();
      ctx.textAlign=t[1];ctx.fillText(label,t[2],ly);
    });
  }
  async function pollAsset(){
    try{
      var r=await fetch("/api/account/asset-history?range=1D",{cache:"no-store"});
      var d=await r.json();assetPoints=(d&&d.points)||[];
      var valEl=document.getElementById("asset-val"),dEl=document.getElementById("asset-delta");
      if(!assetPoints.length){valEl.textContent="—";dEl.textContent="이력 수집 중";dEl.className="col delta";drawSpark();return;}
      var first=Number(assetPoints[0].total_asset_krw||0),last=Number(assetPoints[assetPoints.length-1].total_asset_krw||0);
      valEl.textContent=fmtWon(last);
      var diff=last-first,pct=first>0?(diff/first*100):0;
      dEl.textContent=(diff>=0?"▲ ":"▼ ")+fmtWon(Math.abs(diff))+" ("+(pct>=0?"+":"")+pct.toFixed(2)+"%)";
      dEl.className="col delta "+(diff>=0?"up":"down");
      drawSpark();
    }catch(e){}
  }
  window.addEventListener("resize",drawSpark);
  pollAsset();setInterval(pollAsset,15000);
})();
</script>
</body></html>"""


CANDIDATE_CHART_DISPLAY_HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>후보별 실시간 차트 분석</title>
  <style>
    :root { color-scheme: light; --bg:#f5f7fb; --panel:#fff; --text:#0f1f33; --muted:#64748b; --line:#dbe4f0; --up:#047857; --down:#c0261d; --warn:#b77900; --accent:#2563eb; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    header { position:sticky; top:0; z-index:3; display:flex; align-items:center; justify-content:space-between; gap:16px; padding:18px 24px; background:rgba(245,247,251,.94); border-bottom:1px solid var(--line); backdrop-filter: blur(10px); }
    h1 { margin:0; font-size:24px; letter-spacing:0; }
    .sub { color:var(--muted); font-size:13px; margin-top:4px; }
    .toolbar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; justify-content:flex-end; }
    .badge, button { min-height:34px; border-radius:8px; border:1px solid var(--line); background:var(--panel); color:var(--text); padding:7px 11px; font-size:13px; }
    button { cursor:pointer; font-weight:650; }
    main { padding:18px 24px 32px; }
    .summary { display:grid; grid-template-columns: repeat(4, minmax(140px,1fr)); gap:10px; margin-bottom:14px; }
    .metric { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }
    .metric .label { color:var(--muted); font-size:12px; }
    .metric .value { margin-top:6px; font-size:20px; font-weight:760; }
    .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap:14px; align-items:start; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; box-shadow:0 1px 2px rgba(15,31,51,.04); }
    .card-head { display:grid; grid-template-columns: 1fr auto; gap:8px; align-items:start; padding:14px 16px 10px; border-bottom:1px solid #edf2f7; }
    .symbol { font-size:20px; font-weight:800; }
    .price { text-align:right; font-size:20px; font-weight:760; }
    .age { color:var(--muted); font-size:12px; margin-top:2px; }
    .chart-wrap { height:250px; padding:8px 12px 0; }
    canvas { width:100%; height:100%; display:block; }
    .analysis { display:grid; grid-template-columns: repeat(4, 1fr); gap:8px; padding:10px 16px 14px; border-top:1px solid #edf2f7; }
    .cell { min-width:0; }
    .cell span { display:block; color:var(--muted); font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .cell strong { display:block; margin-top:3px; font-size:14px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .reasons { grid-column: 1 / -1; display:flex; flex-wrap:wrap; gap:6px; margin-top:2px; }
    .chip { border-radius:999px; padding:4px 8px; background:#eef2ff; color:#1e3a8a; font-size:12px; }
    .empty { padding:36px 18px; text-align:center; color:var(--muted); border:1px dashed var(--line); border-radius:8px; background:var(--panel); }
    .up { color:var(--up); } .down { color:var(--down); } .warn { color:var(--warn); }
    @media (max-width: 760px) {
      header { align-items:flex-start; flex-direction:column; padding:14px; }
      main { padding:12px; }
      .summary { grid-template-columns: repeat(2, minmax(0,1fr)); }
      .grid { grid-template-columns: 1fr; }
      .analysis { grid-template-columns: repeat(2, 1fr); }
      .chart-wrap { height:220px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>후보별 실시간 차트 분석</h1>
      <div class="sub">실시간 체결/호가로 만든 1분봉과 기술 판단 근거를 후보별로 표시합니다.</div>
    </div>
    <div class="toolbar">
      <span class="badge" id="updated">업데이트 -</span>
      <span class="badge" id="source">데이터 -</span>
      <button type="button" id="refresh">새로고침</button>
    </div>
  </header>
  <main>
    <section class="summary">
      <div class="metric"><div class="label">표시 후보</div><div class="value" id="mSymbols">-</div></div>
      <div class="metric"><div class="label">실시간 봉 보유</div><div class="value" id="mBars">-</div></div>
      <div class="metric"><div class="label">최근 체결</div><div class="value" id="mTicks">-</div></div>
      <div class="metric"><div class="label">평균 스프레드</div><div class="value" id="mSpread">-</div></div>
    </section>
    <section class="grid" id="cards"><div class="empty">실시간 후보 데이터를 불러오는 중입니다.</div></section>
  </main>
  <script>
    const cards = document.getElementById('cards');
    const fmt = (v, d=2) => Number.isFinite(Number(v)) ? Number(v).toLocaleString(undefined, {maximumFractionDigits:d}) : '-';
    const cls = (v) => Number(v) >= 0 ? 'up' : 'down';
    function ageLabel(iso){
      if(!iso) return '-';
      const sec = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
      if(sec < 60) return `${sec.toFixed(0)}초 전`;
      if(sec < 3600) return `${(sec/60).toFixed(1)}분 전`;
      return `${(sec/3600).toFixed(1)}시간 전`;
    }
    function drawChart(canvas, bars){
      const ctx = canvas.getContext('2d');
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr,0,0,dpr,0,0);
      const w = rect.width, h = rect.height;
      ctx.clearRect(0,0,w,h);
      ctx.fillStyle = '#ffffff'; ctx.fillRect(0,0,w,h);
      ctx.strokeStyle = '#e5edf6'; ctx.lineWidth = 1;
      for(let i=1;i<4;i++){ const y = i*h*.18; ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); }
      if(!bars || !bars.length){
        ctx.fillStyle = '#94a3b8'; ctx.font = '14px system-ui'; ctx.textAlign = 'center'; ctx.fillText('실시간 1분봉 없음', w/2, h/2);
        return;
      }
      const priceH = h * .72, volTop = priceH + 12, volH = h - volTop - 4;
      const highs = bars.map(b=>Number(b.high || b.close));
      const lows = bars.map(b=>Number(b.low || b.close));
      const maxP = Math.max(...highs), minP = Math.min(...lows);
      const pad = Math.max((maxP - minP) * .08, Math.abs(maxP) * .002, 0.01);
      const top = maxP + pad, bot = minP - pad;
      const maxV = Math.max(1, ...bars.map(b=>Number(b.volume || 0)));
      const gap = 3, bw = Math.max(3, (w - gap * (bars.length + 1)) / bars.length);
      const y = (p) => priceH - ((Number(p) - bot) / Math.max(0.000001, top - bot)) * priceH + 2;
      bars.forEach((b, i) => {
        const x = gap + i * (bw + gap);
        const open = Number(b.open), close = Number(b.close), high = Number(b.high), low = Number(b.low);
        const up = close >= open;
        ctx.strokeStyle = up ? '#059669' : '#dc2626';
        ctx.fillStyle = up ? 'rgba(5,150,105,.75)' : 'rgba(220,38,38,.75)';
        const cx = x + bw / 2;
        ctx.beginPath(); ctx.moveTo(cx, y(high)); ctx.lineTo(cx, y(low)); ctx.stroke();
        const bodyY = Math.min(y(open), y(close));
        const bodyH = Math.max(2, Math.abs(y(open) - y(close)));
        ctx.fillRect(x, bodyY, bw, bodyH);
        const vh = (Number(b.volume || 0) / maxV) * volH;
        ctx.fillStyle = up ? 'rgba(5,150,105,.28)' : 'rgba(220,38,38,.28)';
        ctx.fillRect(x, volTop + volH - vh, bw, vh);
      });
      ctx.fillStyle = '#64748b'; ctx.font = '11px system-ui'; ctx.textAlign = 'right';
      ctx.fillText(fmt(top), w - 4, 12); ctx.fillText(fmt(bot), w - 4, priceH - 4);
    }
    function cardHtml(item){
      const latest = item.latest_tick || {};
      const bars = item.bars || [];
      const first = bars[0], last = bars[bars.length - 1];
      const delta = first && last ? Number(last.close) - Number(first.open) : 0;
      const pct = first && Number(first.open) ? delta / Number(first.open) * 100 : 0;
      const tech = item.technical || {};
      const orderbook = item.latest_orderbook || {};
      const reasons = [...(tech.reason_codes || []), ...(item.micro_reasons || [])].slice(0, 5);
      return `<article class="card">
        <div class="card-head">
          <div><div class="symbol">${item.symbol}</div><div class="age">최근 체결 ${ageLabel(latest.received_at)} · 봉 ${bars.length}개</div></div>
          <div class="price">${fmt(latest.price ?? (last && last.close), 4)}<div class="${cls(delta)}" style="font-size:13px">${delta>=0?'+':''}${fmt(delta,4)} / ${pct>=0?'+':''}${fmt(pct,2)}%</div></div>
        </div>
        <div class="chart-wrap"><canvas data-symbol="${item.symbol}"></canvas></div>
        <div class="analysis">
          <div class="cell"><span>판단</span><strong>${tech.action || item.micro_regime || '-'}</strong></div>
          <div class="cell"><span>신뢰도</span><strong>${fmt(tech.confidence ?? item.confidence, 2)}</strong></div>
          <div class="cell"><span>스프레드</span><strong>${fmt(orderbook.spread_bps ?? item.spread_bps, 2)} bps</strong></div>
          <div class="cell"><span>호가 불균형</span><strong class="${cls(orderbook.imbalance ?? item.orderbook_imbalance)}">${fmt(orderbook.imbalance ?? item.orderbook_imbalance, 2)}</strong></div>
          <div class="cell"><span>VWAP</span><strong>${fmt(last && last.vwap, 4)}</strong></div>
          <div class="cell"><span>거래량</span><strong>${fmt(last && last.volume, 0)}</strong></div>
          <div class="cell"><span>유동성</span><strong>${fmt(last && last.liquidity_score, 2)}</strong></div>
          <div class="cell"><span>데이터</span><strong>${item.data_state || '-'}</strong></div>
          <div class="reasons">${reasons.length ? reasons.map(r=>`<span class="chip">${r}</span>`).join('') : '<span class="chip">근거 없음</span>'}</div>
        </div>
      </article>`;
    }
    let latestItems = [];
    async function load(){
      const res = await fetch('/api/realtime/candidate-charts?limit=12&minutes=90', {cache:'no-store'});
      const data = await res.json();
      latestItems = data.symbols || [];
      document.getElementById('updated').textContent = `업데이트 ${new Date(data.generated_at).toLocaleTimeString()}`;
      document.getElementById('source').textContent = data.mode || 'realtime_minute_bars';
      document.getElementById('mSymbols').textContent = fmt(latestItems.length,0);
      document.getElementById('mBars').textContent = fmt(latestItems.filter(x=>(x.bars||[]).length).length,0);
      document.getElementById('mTicks').textContent = fmt(latestItems.filter(x=>x.latest_tick).length,0);
      const spreads = latestItems.map(x=>Number((x.latest_orderbook||{}).spread_bps)).filter(Number.isFinite);
      document.getElementById('mSpread').textContent = spreads.length ? `${fmt(spreads.reduce((a,b)=>a+b,0)/spreads.length,2)} bps` : '-';
      cards.innerHTML = latestItems.length ? latestItems.map(cardHtml).join('') : '<div class="empty">표시할 후보가 없습니다. 실시간 수집이 아직 워밍업 중일 수 있습니다.</div>';
      latestItems.forEach(item => {
        const canvas = cards.querySelector(`canvas[data-symbol="${CSS.escape(item.symbol)}"]`);
        if(canvas) drawChart(canvas, item.bars || []);
      });
    }
    document.getElementById('refresh').addEventListener('click', load);
    window.addEventListener('resize', () => latestItems.forEach(item => {
      const canvas = cards.querySelector(`canvas[data-symbol="${CSS.escape(item.symbol)}"]`);
      if(canvas) drawChart(canvas, item.bars || []);
    }));
    load().catch(err => { cards.innerHTML = `<div class="empty">불러오기 실패: ${err}</div>`; });
    setInterval(load, 5000);
  </script>
</body>
</html>"""


@app.get("/display", response_class=HTMLResponse)
def trade_display() -> str:
    return TRADE_DISPLAY_HTML


@app.get("/display/ontology", response_class=HTMLResponse)
def ontology_display() -> str:
    return CANDIDATE_CHART_DISPLAY_HTML


@app.get("/display/candidates", response_class=HTMLResponse)
def candidate_chart_display() -> str:
    return CANDIDATE_CHART_DISPLAY_HTML


@app.post("/api/kiosk/exit")
def kiosk_exit() -> JSONResponse:
    """Close the local fullscreen kiosk browser (does not affect the server)."""
    import subprocess

    closed = False
    for pattern in ("127.0.0.1:8010/display", "--app=http://127.0.0.1:8010/display"):
        try:
            result = subprocess.run(
                ["pkill", "-f", pattern], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                closed = True
        except (OSError, subprocess.SubprocessError):
            pass
    return _json({"ok": True, "closed": closed})


@app.get("/api/gnn/realtime-trust")
def gnn_realtime_trust_status() -> JSONResponse:
  """Current forward-only confidence gate for ontology-weighted GNN routing."""
  return _json(_gnn_realtime_trust_evaluator.evaluate().as_dict())


@app.get("/api/status")
def status() -> JSONResponse:
  live_basis = _refresh_live_account_basis_for_auto() or _last_live_account_basis()
  if live_basis is not None:
    return _json(
      {
        "cash": live_basis["cash"],
        "cash_equivalent_krw": live_basis.get("cash_equivalent_krw", live_basis["cash"]),
        "krw_cash": live_basis["krw_cash"],
        "foreign_cash_krw": live_basis["foreign_cash_krw"],
        "cash_by_currency": live_basis["cash_by_currency"],
        "orderable_cash_by_currency": live_basis.get("orderable_cash_by_currency", live_basis["cash_by_currency"]),
        "orderable_cash_reconciliation": live_basis.get(
            "orderable_cash_reconciliation",
            {},
        ),
        "foreign_cash_by_currency": live_basis["foreign_cash_by_currency"],
        "base_currency": live_basis["base_currency"],
        "equity": live_basis["equity"],
        "cash_weight": live_basis["cash_weight"],
        "basis_source": live_basis["source"],
        "account_suffix": live_basis["account_suffix"],
        "account_checked": True,
        "holdings": len(tuple(live_basis.get("positions") or ())),
        "holdings_count": len(tuple(live_basis.get("positions") or ())),
        "positions": list(live_basis.get("positions") or ()),
        "daily_pnl_ratio": 0.0,
        "updated_at": datetime.now(timezone.utc),
        "last_error": None,
        "risk_rejections": [],
      }
    )
  snapshot = _live_snapshot()
  if snapshot["context"] is None or snapshot["research_result"] is None:
    _build_current_snapshot_from_store()
    snapshot = _live_snapshot()
  context = snapshot["context"]
  return _json(
    {
      "cash": context.account.cash,
      "equity": context.report.equity,
      "cash_weight": context.report.cash_weight,
      "basis_source": "realtime_model_account",
      "account_suffix": None,
      "daily_pnl_ratio": context.report.daily_pnl_ratio,
      "updated_at": _iso_or_none(snapshot["last_updated"]),
      "last_error": snapshot["last_error"],
      "risk_rejections": [
        {
          "ticker": result.ticker,
          "approved": result.approved,
          "rejection_reasons": result.rejection_reasons,
        }
        for result in context.risk_results
      ],
    }
  )


@app.get("/api/research")
def research() -> JSONResponse:
    snapshot = _get_or_refresh_live()
    research_result = snapshot["research_result"]
    context = snapshot["context"]
    return _json(
        {
            "configured_research": research_result,
            "events": context.events,
            "graph_triples": context.graph.triples(),
            "reasoning_paths": context.reasoning_paths,
            "updated_at": _iso_or_none(snapshot["last_updated"]),
            "last_error": snapshot["last_error"],
        }
    )


@app.post("/api/research/refresh")
def refresh_research() -> JSONResponse:
    _ensure_background_refresh()
    snapshot = _live_snapshot()
    return _json(
        {
            "ok": True,
            "status": "refresh_started" if snapshot["is_refreshing"] else "ready",
            "is_refreshing": snapshot["is_refreshing"],
            "progress": snapshot["progress"],
            "updated_at": _iso_or_none(snapshot["last_updated"]),
            "last_error": snapshot["last_error"],
        }
    )


@app.get("/api/research/configured")
def configured_research(config_path: str = "config/research_sources.example.json") -> JSONResponse:
    result = ResearchService().run_from_config(Path(config_path))
    return _json(result)


@app.get("/api/research/diagnostics")
def research_diagnostics() -> JSONResponse:
    snapshot = _live_snapshot()
    if snapshot["research_result"] is None or snapshot["context"] is None:
      return _json(_lightweight_diagnostics_response(snapshot))
    research_result = snapshot["research_result"]
    context = snapshot["context"]
    store_summary = dict(snapshot.get("store_summary") or {})
    return _json(
        {
            "research_config": str(DEFAULT_RESEARCH_CONFIG),
            "diagnostics": _diagnostics_with_collection_config(research_result.diagnostics),
            "skipped_sources": research_result.skipped_sources,
            "stored_new_records": snapshot["stored_new_records"],
            "store_summary": store_summary,
            "data_volume": _lightweight_data_volume(store_summary),
            "store_path": str(LocalResearchStore(root=_get_store_root()).db_path),
            "data_policy": _current_data_policy(),
            "events_sample": research_result.events[:5],
            "market_snapshots": research_result.market_snapshots[:25],
            "graph_triples_count": _graph_triples_count(snapshot, context),
            "reasoning_paths": context.reasoning_paths[:25],
            "ontology_runtime": context.ontology_runtime.as_dict(),
            "semantic_layer": _semantic_layer_diagnostics(context),
            "updated_at": _iso_or_none(snapshot["last_updated"]),
            "last_error": snapshot["last_error"],
            "is_refreshing": snapshot["is_refreshing"],
            "refresh_interval_seconds": LIVE_REFRESH_SECONDS,
        }
    )


def _lightweight_diagnostics_response(snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    snapshot = snapshot or _live_snapshot()
    store_summary = dict(snapshot.get("store_summary") or {})
    diagnostics = {
        "events_count": int(store_summary.get("events", 0) or 0),
        "raw_records_count": int(store_summary.get("raw_records", 0) or 0),
        "market_snapshots_count": int(store_summary.get("market_snapshots", 0) or 0),
        "macro_metrics_count": int(store_summary.get("macro_metrics", 0) or 0),
        "skipped_count": 0,
        "live_source_count": int(store_summary.get("raw_records", 0) or 0),
        "local_source_count": int(store_summary.get("market_snapshots", 0) or 0),
        "live_data_present": bool(store_summary.get("events") or store_summary.get("raw_records")),
        "collection_warnings": ("Analysis cache is still warming; showing stored data summary.",),
    }
    return {
        "research_config": str(DEFAULT_RESEARCH_CONFIG),
        "diagnostics": _diagnostics_with_collection_config(diagnostics),
        "skipped_sources": (),
        "stored_new_records": snapshot.get("stored_new_records", {}),
        "store_summary": store_summary,
        "data_volume": _lightweight_data_volume(store_summary),
        "store_path": str(LocalResearchStore(root=_get_store_root()).db_path),
        "data_policy": _current_data_policy(),
        "events_sample": (),
        "market_snapshots": (),
        "graph_triples_count": 0,
        "reasoning_paths": (),
        "ontology_runtime": get_ontology_runtime().as_dict(),
        "updated_at": _iso_or_none(snapshot.get("last_updated")),
        "last_error": snapshot.get("last_error"),
        "is_refreshing": snapshot.get("is_refreshing", False),
        "refresh_interval_seconds": LIVE_REFRESH_SECONDS,
    }


@app.get("/api/ontology/graph")
def ontology_graph(full: bool = False) -> JSONResponse:
    snapshot = _live_snapshot()
    context = snapshot.get("context")
    if context is None:
      _ensure_background_refresh()
      payload = _empty_graph_payload(snapshot)
      payload["live_trace"] = _current_live_reasoning_trace()
      return _json(payload)
    # The dashboard's 3D view explicitly requests the complete graph.  Keep the
    # compact cached payload as the default for lightweight API consumers.
    if full:
      payload = _graph_payload(context, trim_for_ui=False)
      payload["live_trace"] = _current_live_reasoning_trace()
      return _json(payload)
    payload = snapshot.get("graph_payload")
    if payload is None or snapshot.get("graph_payload_context_id") != id(context):
      payload = _graph_payload(context)
      with _live_lock:
        if _live_state["context"] is context:
          _live_state["graph_payload"] = payload
          _live_state["graph_payload_context_id"] = id(context)
    response_payload = dict(payload)
    response_payload["live_trace"] = _current_live_reasoning_trace()
    return _json(response_payload)


def _current_live_reasoning_trace() -> dict[str, Any] | None:
    """Latest stages actually reached by the live trading engine.

    This is deliberately separate from ``reasoning_steps``. Those steps explain
    the latest stored ontology result; ``live_trace`` is clocked by the running
    order engine and is the only source the realtime graph animation follows.
    """
    with _realtime_trading_lock:
      engine = _realtime_trading_engine
    if engine is None:
      return None
    try:
      status = engine.get_status() or {}
    except Exception:  # noqa: BLE001 - graph diagnostics must never stop trading.
      return None
    trace = status.get("live_trace")
    return dict(trace) if isinstance(trace, dict) else None


@app.get("/api/ontology/live-trace")
def ontology_live_trace() -> JSONResponse:
    """Lightweight actual engine trace for the graph's one-second follow mode."""
    trace = _current_live_reasoning_trace()
    return _json({"ok": trace is not None, "trace": trace})


@app.get("/api/ontology/runtime")
def ontology_runtime() -> JSONResponse:
    return _json(get_ontology_runtime().as_dict())


@app.get("/api/realtime/candidate-charts")
def realtime_candidate_charts(limit: int = 12, minutes: int = 90) -> JSONResponse:
    store = RealtimeMarketDataStore()
    now = datetime.now(timezone.utc)
    safe_limit = max(1, min(24, int(limit or 12)))
    safe_minutes = max(5, min(390, int(minutes or 90)))
    since = now - timedelta(minutes=safe_minutes)
    symbols = _candidate_chart_symbols(store, limit=safe_limit)
    technical_by_symbol = _latest_technical_by_symbol()
    micro_by_symbol = _latest_micro_by_symbol()
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        try:
            store.build_latest_minute_bar(symbol, now=now)
        except Exception:  # noqa: BLE001 - chart display is diagnostic-only.
            pass
        try:
            bars = store.recent_minute_bars(symbol, since, limit=safe_minutes)
        except Exception:  # noqa: BLE001
            bars = ()
        try:
            latest_tick = store.latest_tick(symbol)
        except Exception:  # noqa: BLE001
            latest_tick = None
        try:
            latest_orderbook = store.latest_orderbook(symbol)
        except Exception:  # noqa: BLE001
            latest_orderbook = None
        technical = technical_by_symbol.get(symbol) or {}
        micro = micro_by_symbol.get(symbol) or {}
        last_bar = bars[-1] if bars else None
        rows.append(
            {
                "symbol": symbol,
                "bars": [_realtime_bar_payload(bar) for bar in bars],
                "latest_tick": _realtime_tick_payload(latest_tick),
                "latest_orderbook": _realtime_orderbook_payload(latest_orderbook),
                "technical": _compact_technical_payload(technical),
                "micro_regime": micro.get("micro_regime"),
                "micro_reasons": list(micro.get("reason_codes") or micro.get("reasons") or ())[:5],
                "confidence": micro.get("confidence"),
                "spread_bps": getattr(last_bar, "spread_bps", None),
                "orderbook_imbalance": getattr(last_bar, "orderbook_imbalance", None),
                "data_state": "1분봉" if bars else ("최근 체결만" if latest_tick else "대기"),
            }
        )
    return _json(
        {
            "generated_at": now.isoformat(),
            "mode": "실시간 체결/호가 기반 1분봉",
            "lookback_minutes": safe_minutes,
            "symbols": rows,
        }
    )


@app.get("/api/realtime/runtime")
def realtime_runtime() -> JSONResponse:
    risk_policy = ShortHorizonRiskPolicy()
    training_status = _safe_live_training_status_fast()
    return _json(
        {
            "acceleration": _realtime_runtime_static_cache.get(
                "acceleration",
                RealtimeAccelerationPolicy().status(),
            ),
            "event_llm": _realtime_runtime_static_cache.get(
                "event_llm",
                _event_llm_runtime_status_fast(),
            ),
            "ontology_npu": _realtime_runtime_static_cache.get(
                "ontology_npu",
                {"status": "initializing"},
            ),
            "live_training": training_status,
            "short_horizon_policy": risk_policy,
            "operation_mode": _operation_mode_state.get("active"),
            "resource_allocation": {
                "model_inference": os.getenv("LLM_EVENT_DEVICE", "NPU"),
                "model_backend": os.getenv("LLM_EVENT_INFERENCE_BACKEND", "openvino"),
                "ontology_classification": "openvino_npu_with_cpu_fallback",
                "deterministic_paper_trading_engine": "cpu_worker_after_npu_screening",
                "risk_and_order_rules": "cpu_worker",
                "openvino_cache_dir": os.getenv("OPENVINO_CACHE_DIR", "data/runtime/openvino_cache"),
            },
        }
    )


def _event_llm_runtime_status_fast() -> dict[str, Any]:
    """Return UI-safe LLM status without network/model probes.

    ``event_llm_runtime_status`` may probe local OpenAI-compatible endpoints.
    This route is polled by the dashboard, so it must be a cheap snapshot rather
    than a liveness check that can hold the only local app worker.
    """
    enabled = os.getenv("LLM_EVENT_CLASSIFIER_ENABLED", "").lower() in {"1", "true", "yes"}
    provider = os.getenv("LLM_EVENT_PROVIDER", "remote").strip().lower()
    model = os.getenv("LLM_EVENT_MODEL", "")
    backend = os.getenv("LLM_EVENT_INFERENCE_BACKEND", "")
    device = os.getenv("LLM_EVENT_DEVICE", "")
    status: dict[str, Any] = {
        "enabled": enabled,
        "provider": provider,
        "model": model,
        "backend": backend,
        "device": device,
        "available": False,
        "reason": None,
        "probe_skipped": True,
    }
    if not enabled:
        status["reason"] = "LLM_EVENT_CLASSIFIER_ENABLED is false."
        return status
    if not model:
        status["reason"] = "LLM_EVENT_MODEL is not configured."
        return status
    if provider in {"local", "ollama", "llamacpp", "llama.cpp"}:
        status["backend"] = "openai-compatible"
        status["device"] = "local-server"
        status["endpoint"] = os.getenv("LLM_EVENT_LOCAL_ENDPOINT") or os.getenv(
            "LLM_EVENT_ENDPOINT",
            "http://127.0.0.1:11434/v1/chat/completions",
        )
        # configure_default_event_llm_env enables this provider only after a
        # successful local reachability check. Dashboard polls reuse that result.
        status["available"] = True
        status["reason"] = None
        return status
    if provider in {"embedded", "inprocess", "transformers", "local-model", "openvino-llm", "multimodal"}:
        exists = Path(model).exists()
        status["available"] = exists
        status["reason"] = None if exists else f"embedded model path does not exist: {model}"
        return status
    if os.getenv("LLM_EVENT_API_KEY") or os.getenv("OPENAI_API_KEY"):
        status["available"] = True
        return status
    status["reason"] = "remote provider needs LLM_EVENT_API_KEY or OPENAI_API_KEY."
    return status


def _safe_live_training_status_fast() -> dict[str, Any]:
    """Cheap dashboard status for the latest live-eligible model.

    The full training status scans larger runtime artifacts and is still
    available at ``/api/live-training/status``. The realtime runtime endpoint is
    polled frequently, so it only checks the registry pointer.
    """
    try:
        artifact = ModelArtifactRegistry().load_latest_live_eligible()
    except Exception as exc:  # noqa: BLE001 - status endpoint should report, not fail.
        return {
            "ok": False,
            "pipeline": "collect_features_train_save_predict",
            "model_saved": False,
            "latest_live_eligible_exists": False,
            "error": f"{exc.__class__.__name__}: {exc}",
            "fast_status": True,
        }
    return {
        "ok": True,
        "pipeline": "collect_features_train_save_predict",
        "model_saved": True,
        "latest_live_eligible_exists": bool(getattr(artifact, "live_eligible", False)),
        "latest_live_eligible_model_artifact_id": artifact.artifact_id,
        "feature_schema_hash": artifact.feature_schema_hash,
        "fast_status": True,
    }


@app.get("/api/ai/validation")
def ai_validation() -> JSONResponse:
    configure_default_event_llm_env()
    llm_status = event_llm_runtime_status()
    llm_validation: dict[str, Any] = {"ok": False, "status": llm_status}
    if llm_status.get("available"):
      try:
        classifier = build_event_llm_classifier_from_env()
        if classifier is None:
          llm_validation["error"] = "LLM_CLASSIFIER_NOT_CONFIGURED"
        else:
          sample = classifier.classify(
              "Example issuer wins a large supply contract",
              "The issuer announced a multi-year supply agreement. Analysts expect higher revenue and margins.",
              {"DEMO": "Example issuer"},
          )
          labels = tuple(str(item) for item in sample.event_labels)
          llm_validation.update(
              {
                  "ok": bool(sample.summary and sample.confidence >= 0.0),
                  "model": sample.model,
                  "sentiment": getattr(sample.sentiment, "value", str(sample.sentiment)),
                  "tickers": sample.tickers,
                  "event_labels": labels,
                  "confidence": sample.confidence,
                  "summary": sample.summary,
                  "schema_ok": isinstance(sample.key_facts, tuple) and isinstance(sample.tickers, tuple),
              }
          )
      except Exception as exc:  # noqa: BLE001 - validation must report, not fail the UI.
        llm_validation["error"] = f"{exc.__class__.__name__}: {exc}"
    training_status = _safe_live_training_status()
    predictor_validation = _validate_live_signal_predictor()
    ontology_status = get_ontology_npu_classifier().status()
    return _json(
        {
            "ok": bool(llm_validation.get("ok")) and bool(training_status.get("ok")) and bool(predictor_validation.get("ok")),
            "event_llm": llm_validation,
            "live_training": training_status,
            "live_signal_predictor": predictor_validation,
            "ontology_npu": ontology_status,
            "recommendation": _ai_validation_recommendation(llm_validation, training_status, predictor_validation),
        }
    )


def _validate_live_signal_predictor() -> dict[str, Any]:
    if not live_signal_model_inference_enabled():
      return {
          "ok": False,
          "disabled": True,
          "error": "LIVE_SIGNAL_MODEL_INFERENCE_DISABLED",
          "uses_live_eligible_model": False,
      }
    registry = ModelArtifactRegistry()
    try:
      artifact = registry.load_latest_live_eligible()
    except Exception as exc:  # noqa: BLE001
      return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}", "uses_live_eligible_model": False}
    metrics = dict(getattr(artifact, "metrics", {}) or {})
    finite_weights = all(math.isfinite(float(value)) for value in (*artifact.weights, *artifact.expected_return_weights))
    finite_bias = math.isfinite(float(artifact.bias)) and math.isfinite(float(artifact.expected_return_bias))
    return {
        "ok": bool(artifact.live_eligible and finite_weights and finite_bias),
        "artifact_id": artifact.artifact_id,
        "path": str(artifact.path),
        "uses_live_eligible_model": artifact.live_eligible,
        "feature_schema_hash": artifact.feature_schema_hash,
        "metrics": metrics,
        "finite_parameters": finite_weights and finite_bias,
    }


def _ai_validation_recommendation(
    llm_validation: dict[str, Any],
    training_status: dict[str, Any],
    predictor_validation: dict[str, Any],
) -> str:
    if not llm_validation.get("ok"):
      return "LLM classifier is reachable check failed; keep keyword fallback active."
    quality = training_status.get("quality") if isinstance(training_status.get("quality"), dict) else {}
    if not predictor_validation.get("ok"):
      return "Use ontology/rule fallback for live execution until a live-eligible predictor is available."
    if not quality.get("meaningful_for_live"):
      return "Model is useful for research diagnostics, but live edge is not proven because top-k forward return is not positive."
    return "LLM and live predictor validation passed; keep FinalTradeGate/RiskManager as mandatory execution gates."


@app.get("/api/live-training/status")
def live_training_status_api() -> JSONResponse:
    return _json(_safe_live_training_status())


@app.get("/api/npu/runtime")
def npu_runtime() -> JSONResponse:
    manager = get_npu_runtime_manager()
    modules = manager.status().get("modules", {})
    ontology_status = get_ontology_npu_classifier().status()
    payload_modules = {
        "ontology_candidate_scorer": _to_jsonable(ontology_status),
        "theory_vote_scorer": modules.get("theory_vote_scorer") or manager.status("theory_vote_scorer"),
        "evidence_cluster_compressor": modules.get("evidence_cluster_compressor") or manager.status("evidence_cluster_compressor"),
        "conflict_scorer": modules.get("conflict_scorer") or manager.status("conflict_scorer"),
        "short_horizon_predictor": modules.get("short_horizon_predictor") or manager.status("short_horizon_predictor"),
        "execution_edge_scorer": modules.get("execution_edge_scorer") or manager.status("execution_edge_scorer"),
    }
    return _json(
        {
            "available_devices": manager.available_devices,
            "selected_device": manager.device_preference if "NPU" in manager.available_devices else "CPU",
            "modules": payload_modules,
            "cpu_kept_deterministic": (
                "ontology_graph_traversal",
                "explanation_trace",
                "final_action_selection",
                "risk_manager",
                "broker_execution",
            ),
        }
    )


def _safe_live_training_status() -> dict[str, Any]:
    try:
        status = live_training_status()
    except Exception as exc:  # noqa: BLE001 - runtime status must remain available.
        return {
            "ok": False,
            "error": f"{exc.__class__.__name__}: {exc}",
            "pipeline": "collect_features_train_save_predict",
        }
    latest_saved = status.get("latest_saved_artifact") or status.get("latest_ineligible_artifact")
    latest_live = status.get("latest_live_eligible_artifact")
    latest_saved_id = latest_saved.get("artifact_id") if isinstance(latest_saved, dict) else None
    latest_live_id = latest_live.get("artifact_id") if isinstance(latest_live, dict) else None
    latest_live_eligible = bool(status.get("latest_live_eligible_exists"))
    inference_enabled = live_signal_model_inference_enabled()
    quality = _live_training_quality_summary(latest_saved, latest_live, status.get("training_rows"))
    return {
        **status,
        "ok": True,
        "pipeline": "collect_features_train_save_predict",
        "auto_training_enabled": True,
        "model_saved": bool(latest_saved_id or latest_live_eligible),
        "latest_model_artifact_id": latest_saved_id,
        "latest_live_eligible_model_artifact_id": latest_live_id,
        "inference_enabled": inference_enabled,
        "inference_uses_latest_live_eligible": inference_enabled and latest_live_eligible,
        "quality": quality,
    }


def _live_training_quality_summary(
    latest_saved: Any,
    latest_live: Any,
    training_rows: Any,
) -> dict[str, Any]:
    saved_metrics = (latest_saved or {}).get("metrics") if isinstance(latest_saved, dict) else {}
    live_metrics = (latest_live or {}).get("metrics") if isinstance(latest_live, dict) else {}
    metrics = saved_metrics if isinstance(saved_metrics, dict) else {}
    auc = _number_or_zero(metrics.get("auc"))
    precision_at_k = _number_or_zero(metrics.get("precision_at_k"))
    top_k_return = _number_or_zero(metrics.get("avg_forward_net_return_bps_top_k"))
    rows = int(_number_or_zero(training_rows))
    positive = _number_or_zero(metrics.get("positive_labels"))
    negative = _number_or_zero(metrics.get("negative_labels"))
    label_total = max(1.0, positive + negative)
    label_balance = positive / label_total
    latest_saved_live_eligible = bool((latest_saved or {}).get("live_eligible")) if isinstance(latest_saved, dict) else False
    live_available = bool(latest_live)
    meaningful_for_research = rows >= 500 and auc >= 0.55 and 0.005 <= label_balance <= 0.50
    meaningful_for_live = bool(live_available and live_metrics)
    if top_k_return <= 0:
      meaningful_for_live = False
    return {
        "training_rows": rows,
        "auc": auc,
        "precision_at_k": precision_at_k,
        "avg_forward_net_return_bps_top_k": top_k_return,
        "positive_label_ratio": label_balance,
        "latest_saved_live_eligible": latest_saved_live_eligible,
        "live_eligible_available": live_available,
        "meaningful_for_research": meaningful_for_research,
        "meaningful_for_live": meaningful_for_live,
        "assessment": (
            "live_candidate"
            if meaningful_for_live
            else "research_only_until_top_k_positive"
            if meaningful_for_research
            else "needs_more_or_better_labels"
        ),
    }




LIVE_FLAG_VALUES = {
    "LIVE_TRADING_ENABLED": "true",
    "KIS_LIVE_ENABLED": "true",
    "LIVE_ORDER_SUBMIT_ENABLED": "true",
    "LIVE_SIGNAL_MODEL_INFERENCE_ENABLED": "true",
    "REQUIRE_MANUAL_ARMING": "false",
    "KILL_SWITCH_ENABLED": "false",
}


def _manual_arming_required() -> bool:
    env_value = os.getenv("REQUIRE_MANUAL_ARMING")
    if env_value is not None:
      return env_value.strip().lower() in {"1", "true", "yes", "on"}
    try:
      safety = load_live_trading_safety_config()
      return bool(safety.require_manual_arming)
    except LiveConfigError:
      return True


@app.get("/api/live-flags/status")
def live_flags_status() -> JSONResponse:
    return _json(_live_flags_status_payload())


@app.post("/api/live-flags/apply")
async def live_flags_apply(request: Request) -> JSONResponse:
    payload = await request.json()
    confirmation = str(payload.get("confirmation") or "")
    if confirmation != "APPLY_LIVE_FLAGS":
      raise HTTPException(status_code=400, detail="confirmation must be APPLY_LIVE_FLAGS")
    for key, value in LIVE_FLAG_VALUES.items():
      os.environ[key] = value
    audit.record("live_flags_applied_from_ui", {"keys": sorted(LIVE_FLAG_VALUES)})
    return _json(_live_flags_status_payload(applied=True))


def _live_flags_status_payload(applied: bool = False) -> dict[str, Any]:
    readiness = _web_live_readiness_summary(include_kis_health=False)
    live_ready = bool(readiness["ok"])
    manual_arming_required = _manual_arming_required()
    if live_ready:
      if manual_arming_required:
        message = "Live flags are active. Orders still require readiness success and manual arming."
      else:
        message = "Live flags are active. Orders can be submitted when readiness gates pass."
    else:
      message = "Live flags are active. Live orders remain safely blocked until readiness gates pass."
    return {
        "ok": True,
        "applied": applied,
        "live_ready": live_ready,
        "flags": {key: os.getenv(key) for key in LIVE_FLAG_VALUES},
        "manual_arming_required": manual_arming_required,
        "orders_submitted": False,
        "readiness": readiness,
        "message": message,
    }


def _web_live_readiness_summary(*, include_kis_health: bool = False) -> dict[str, Any]:
    gates: dict[str, bool] = {}
    failures: dict[str, str] = {}

    def record(name: str, ok: bool, reason: str | None = None) -> None:
      gates[name] = ok
      if not ok and reason:
        failures[name] = reason

    try:
      load_live_trading_safety_config()
      record("live_trading_safety_config", True)
    except LiveConfigError as exc:
      record("live_trading_safety_config", False, str(exc))
    try:
      load_order_execution_config()
      record("order_execution_config", True)
    except LiveConfigError as exc:
      record("order_execution_config", False, str(exc))
    if live_signal_model_inference_enabled():
      try:
        artifact = ModelArtifactRegistry().load_latest_live_eligible()
        model_ok = artifact.feature_schema_hash == LIVE_SHORT_HORIZON_SCHEMA.schema_hash
        record(
            "live_eligible_model",
            model_ok,
            None
            if model_ok
            else f"FEATURE_SCHEMA_MISMATCH expected={LIVE_SHORT_HORIZON_SCHEMA.schema_hash} actual={artifact.feature_schema_hash}",
        )
      except Exception as exc:  # noqa: BLE001 - UI readiness should summarize every gate.
        record("live_eligible_model", False, _live_model_readiness_failure_message(exc))
    else:
      record("live_signal_model_inference_disabled", True)
    secrets = validate_live_secret_file()
    record("kis_secret_file", _kis_secret_file_gate_ok(secrets), "missing KIS secret file or required keys")
    runtime = evaluate_live_runtime_gates(require_manual_arming=False)
    record("live_flags", runtime.ok, ",".join(runtime.failures) if runtime.failures else None)
    if include_kis_health:
      try:
        client = build_kis_client(enabled=True)
        health = run_kis_health_check(client, include_account=True, include_websocket=True)
        record("kis_health", health.ok, ",".join(f"{key}:{value}" for key, value in health.failures.items()))
      except Exception as exc:  # noqa: BLE001 - never throw secrets or raw credential state to UI.
        record("kis_health", False, exc.__class__.__name__)
    else:
      record("kis_health_deferred", True)
    return {"ok": not failures, "gates": gates, "failures": failures}


def _kis_secret_file_gate_ok(secrets: dict[str, bool]) -> bool:
  return all(
      bool(secrets.get(key))
      for key in (
          "file_exists",
          "KIS_APP_KEY",
          "KIS_APP_SECRET",
          "KIS_ACCOUNT_NO",
          "KIS_ACCOUNT_PRODUCT_CODE",
      )
  )


def _live_model_readiness_failure_message(exc: Exception) -> str:
  base = str(exc) or exc.__class__.__name__
  if base not in {"NO_LIVE_ELIGIBLE_MODEL_ARTIFACT", "LATEST_MODEL_NOT_LIVE_ELIGIBLE"}:
    return base
  try:
    status = live_training_status()
  except Exception:  # noqa: BLE001 - readiness must not fail because diagnostics failed.
    return base
  detail = [
      base,
      f"training_rows={int(status.get('training_rows') or 0)}",
      f"feature_frames={int(status.get('feature_frame_lines') or 0)}",
      f"realtime_store={'present' if status.get('realtime_store_exists') else 'missing'}",
  ]
  artifact = status.get("latest_ineligible_artifact")
  if isinstance(artifact, dict) and artifact:
    reasons = ",".join(str(item) for item in artifact.get("reason_codes") or ())
    examples = int(artifact.get("example_count") or 0)
    detail.append(f"latest_ineligible={artifact.get('artifact_id')} examples={examples}")
    if reasons:
      detail.append(f"reasons={reasons}")
  else:
    detail.append("latest_ineligible=none")
  return " ".join(detail)


@app.post("/api/operation-mode/start")
async def operation_mode_start(request: Request) -> JSONResponse:
    try:
      payload = await request.json()
    except json.JSONDecodeError as exc:
      raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from exc
    if payload is None:
      payload = {}
    if not isinstance(payload, dict):
      raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    return _json(await run_in_threadpool(_operation_mode_start_response, payload))


def _operation_mode_start_response(payload: dict[str, Any]) -> dict[str, Any]:
    requested_mode = str(payload.get("mode", "live_trading"))
    deprecated_paper_modes = {"testing", "paper", "paper_trading", "paper_trading_test"}
    mode = "live_trading" if requested_mode in deprecated_paper_modes else requested_mode
    if not _operation_mode_lock.acquire(blocking=False):
      return {
          "ok": False,
          "status": "busy",
          "mode": mode,
          "requested_mode": requested_mode,
          "message": "Another operation-mode request is still being prepared.",
          "request": _operation_mode_request_snapshot(),
      }
    try:
      _set_operation_request(True, "starting", f"Starting {mode}", None)
      state = OperationModeManager().start(mode)
      with _live_lock:
        _operation_mode_state["active"] = state
        if mode != "live_trading":
          _clear_live_analysis_cache_unlocked()
      audit.record("operation_mode_started", {"mode_state": state})

      result = _to_jsonable(state)
      result["ok"] = True
      result["status"] = "started"
      result["requested_mode"] = requested_mode
      if requested_mode != mode:
        result["mode_normalized_from"] = requested_mode
        result["mode_normalization_message"] = "Paper trading mode has been removed; live trading mode was started instead."
      result["mode_state"] = state
      result["data_policy"] = _current_data_policy()

      if mode == "learning":
        _start_live_worker(mode)
        result["training_status"] = "continuous_collection_started"
        result["training_message"] = "Realtime learning and information collection continue while the server is running."

      if mode in {"live_readiness", "live_trading_test"}:
        # 거래/점검 플로우는 학습 워커와 독립적으로 동작한다.
        result["live_readiness_status"] = "checked"
        result["live_readiness_kind"] = "kis_live_readiness"
        kis_connection = _kis_connection_probe(paper=False, include_account=True)
        previous_basis = _last_live_account_basis()
        canonical_basis = _merge_live_account_basis_with_previous(
            _account_basis_from_kis_connection(kis_connection),
            previous_basis,
        )
        canonical_basis = _stabilize_account_basis(canonical_basis)
        if kis_connection.get("account_checked") and canonical_basis is not None:
          kis_connection = _connection_with_account_basis(kis_connection, canonical_basis)
        result["kis_connection"] = kis_connection
        with _live_lock:
          _operation_mode_state["last_kis_connection"] = kis_connection
          _operation_mode_state["last_kis_connection_checked_at"] = time.time()
        if kis_connection.get("ok"):
          result["live_readiness_message"] = (
              "KIS 실전 인증 점검이 완료되었습니다. 주문은 보내지 않았고 실전 주문 게이트는 계속 비활성화되어 있습니다."
          )
        else:
          result["live_readiness_message"] = (
              f"KIS 실전 인증 점검을 완료하지 못했습니다: {kis_connection.get('message') or kis_connection.get('error')}. "
              "주문은 보내지 않았고 실전 주문 게이트는 계속 비활성화되어 있습니다."
          )

      if mode == "live_trading":
        # 거래 플로우는 학습 워커와 독립적으로 동작한다(학습은 별도 제어).
        # 실시간 틱 수집기 + 독립 실시간 거래 엔진을 가동한다.
        # (주문의 실제 전송은 LiveExecutionCoordinator의 안전 게이트가 최종 결정한다.)
        _start_kis_realtime_collector()
        _start_realtime_trading_engine()
        with _realtime_trading_lock:
          active_engine = _realtime_trading_engine
        if active_engine is not None and hasattr(active_engine, "enable_buys"):
          active_engine.enable_buys("OPERATION_MODE_LIVE_TRADING")
        config = load_short_horizon_strategy_config()
        execution_config = config.get("execution", {})
        config_live_enabled = bool(execution_config.get("live_trading_enabled", False))
        env_live_enabled = _env_flag("LIVE_TRADING_ENABLED", False) and _env_flag("KIS_LIVE_ENABLED", False)
        kis_connection = _kis_connection_probe(paper=False, include_account=True)
        previous_basis = _last_live_account_basis()
        canonical_basis = _merge_live_account_basis_with_previous(
            _account_basis_from_kis_connection(kis_connection),
            previous_basis,
        )
        canonical_basis = _stabilize_account_basis(canonical_basis)
        if kis_connection.get("account_checked") and canonical_basis is not None:
          kis_connection = _connection_with_account_basis(kis_connection, canonical_basis)
        result["kis_connection"] = kis_connection
        with _live_lock:
          _operation_mode_state["last_kis_connection"] = kis_connection
          _operation_mode_state["last_kis_connection_checked_at"] = time.time()
        runtime_gate = evaluate_live_runtime_gates(require_manual_arming=_manual_arming_required())
        result["runtime_gate"] = {"ok": runtime_gate.ok, "failures": tuple(runtime_gate.failures)}
        result["live_order_journal"] = _live_order_journal_snapshot()
        result["live_trading_status"] = "armed" if config_live_enabled and env_live_enabled and kis_connection.get("ok") and runtime_gate.ok else "blocked"
        result["live_trading_enabled_by_config"] = config_live_enabled
        result["live_trading_enabled_by_env"] = env_live_enabled
        result["live_trading_message"] = (
            "Live auto-trading gate is armed. Orders still require StrategyCandidateFactory, "
            "RealityCheck, ontology checks, RiskManager, and FinalTradeGate approval."
            if result["live_trading_status"] == "armed"
            else (
                "Live auto-trading is blocked. Check config/env/runtime gate failures: "
                + ", ".join(tuple(runtime_gate.failures) or ("CONFIG_OR_KIS_CONNECTION_NOT_READY",))
            )
        )
        if kis_connection.get("account_checked"):
          with _live_lock:
            _operation_mode_state["live_trading_baseline_equity"] = float(
                kis_connection.get("actual_equity") or kis_connection.get("equity") or kis_connection.get("account_value") or 0.0
            )
        _ensure_background_refresh()

      _set_operation_request(False, "started", f"{mode} started", None)
      result["request"] = _operation_mode_request_snapshot()
      result["learning"] = _learning_state_snapshot()
      return result
    except Exception as exc:
      _set_operation_request(False, "error", f"{mode} failed", str(exc))
      audit.record("operation_mode_failed", {"mode": mode, "error": str(exc)})
      return {
          "ok": False,
          "status": "error",
          "mode": mode,
          "requested_mode": requested_mode,
          "message": str(exc),
          "request": _operation_mode_request_snapshot(),
      }
    finally:
      _operation_mode_lock.release()


def _kis_connection_probe(paper: bool, include_account: bool = False) -> dict[str, Any]:
    mode = "live"
    try:
      client = KisDevelopersApiClient(paper=False, enabled=include_account)
      token = client.issue_access_token()
      result: dict[str, Any] = {
          "ok": True,
          "mode": mode,
          "base_url": client.endpoints.base_url,
          "token_available": bool(token),
          "token_issued": client.token_source == "issued",
          "token_source": client.token_source or "unknown",
          "token_reused": client.token_source in {"cache", "env", "injected"},
          "token_length": len(token),
          "account_suffix": f"...{client.credentials.account_no[-2:]}" if client.credentials.account_no else "",
      }
      if include_account:
        portfolio = client.get_portfolio()
        result["account_checked"] = True
        result["holdings"] = len(portfolio.account.holdings)
        result["holdings_count"] = len(portfolio.account.holdings)
        result["positions"] = [
            {
                "ticker": holding.ticker,
                "market": holding.market,
                "quantity": holding.quantity,
                "sellable_quantity": getattr(holding, "sellable_quantity", None),
                "average_price": holding.average_price,
                "last_price": holding.last_price,
                "market_value": holding.market_value,
                "unrealized_pnl": holding.unrealized_pnl,
                "return_rate": (
                    holding.unrealized_pnl / (holding.quantity * holding.average_price)
                    if holding.quantity > 0 and holding.average_price > 0
                    else 0.0
                ),
                "currency": "KRW" if str(holding.ticker).isdigit() and len(str(holding.ticker)) == 6 else "USD",
            }
            for holding in portfolio.account.holdings
        ]
        cash_by_currency = _cash_by_currency_payload(
            getattr(portfolio.account, "cash_by_currency", None),
            portfolio.account.cash,
            getattr(portfolio.account, "base_currency", "KRW"),
        )
        orderable_cash_by_currency = _cash_by_currency_payload(
            getattr(portfolio.account, "orderable_cash_by_currency", None),
            portfolio.account.cash,
            getattr(portfolio.account, "base_currency", "KRW"),
        )
        portfolio_orderable_krw = _number_or_zero(
            orderable_cash_by_currency.get("KRW")
        )
        authoritative_orderable_krw: float | None = None
        reconciliation_error: str | None = None
        accessor = getattr(client, "_get_domestic_orderable_cash", None)
        if callable(accessor):
          try:
            authoritative_orderable_krw = max(
                0.0,
                _number_or_zero(accessor()),
            )
          except Exception as exc:  # noqa: BLE001 - account probe remains usable.
            reconciliation_error = str(exc)
        mismatch = bool(
            authoritative_orderable_krw is not None
            and authoritative_orderable_krw > 0.0
            and abs(authoritative_orderable_krw - portfolio_orderable_krw) > 1.0
        )
        if authoritative_orderable_krw is not None and authoritative_orderable_krw > 0.0:
          # Reuse the KIS adapter's single nrcvb_buy_amt authority. The
          # connection payload must never substitute settled deposit cash.
          orderable_cash_by_currency["KRW"] = authoritative_orderable_krw
        result["orderable_cash_reconciliation"] = {
            "portfolio_krw": portfolio_orderable_krw,
            "authoritative_krw": authoritative_orderable_krw,
            "difference_krw": (
                round(authoritative_orderable_krw - portfolio_orderable_krw, 4)
                if authoritative_orderable_krw is not None
                else None
            ),
            "mismatch": mismatch,
            "tolerance_krw": 1.0,
            "error": reconciliation_error,
        }
        if mismatch:
          audit.record(
              "kis_orderable_cash_mismatch",
              result["orderable_cash_reconciliation"],
          )
        krw_cash = cash_by_currency.get("KRW", portfolio.account.cash)
        usd_cash = _number_or_zero(cash_by_currency.get("USD", 0.0))
        fx_by_currency = dict(getattr(portfolio.account, "fx_rate_by_currency", {}) or {})
        try:
          default_usd_krw = float(os.getenv("KIS_USD_KRW_RATE", "1380"))
        except ValueError:
          default_usd_krw = 1380.0
        if not (900.0 <= default_usd_krw <= 2000.0):
          default_usd_krw = 1380.0
        # FX rate priority: (1) broker-reported FX if sane, (2) rate implied by the
        # broker cash-equivalent if sane, (3) configured default. Every candidate is
        # bounded to a sane band so a garbage rate can never inflate foreign cash —
        # the real account produced a 3,168 KRW/USD back-calc that blew 외화 예수금 up ~2.3x.
        usd_krw_rate = _number_or_zero(fx_by_currency.get("USD", 0.0))
        if not (900.0 <= usd_krw_rate <= 2000.0):
          broker_equiv = _number_or_zero(getattr(portfolio.account, "cash_equivalent_krw", None))
          usd_position_value = sum(
              _number_or_zero(position.get("market_value"))
              for position in result["positions"]
              if str(position.get("currency") or "").upper() == "USD"
          )
          base = usd_cash + usd_position_value
          implied = (broker_equiv - krw_cash) / base if (broker_equiv > krw_cash and base > 0) else 0.0
          usd_krw_rate = implied if 900.0 <= implied <= 2000.0 else default_usd_krw
        # Foreign cash in KRW = broker/orderable foreign-currency cash * sane FX.
        # Do NOT derive it as (cash_equivalent - krw_cash): that turns settlement
        # residuals and account-total adjustments into fake foreign cash.
        account_foreign_cash_krw = _number_or_zero(getattr(portfolio.account, "foreign_cash_krw", None))
        foreign_cash_krw = account_foreign_cash_krw if account_foreign_cash_krw > 0 else usd_cash * usd_krw_rate
        broker_cash_equivalent_krw = _number_or_zero(getattr(portfolio.account, "cash_equivalent_krw", None))
        cash_equivalent_krw = broker_cash_equivalent_krw if broker_cash_equivalent_krw > 0 else krw_cash + foreign_cash_krw
        invested_value_krw = 0.0
        for position in result["positions"]:
          currency = str(position.get("currency") or "KRW").upper()
          market_value = _number_or_zero(position.get("market_value"))
          if currency == "USD" and usd_krw_rate > 0:
            position["market_value_krw"] = market_value * usd_krw_rate
            position["unrealized_pnl_krw"] = _number_or_zero(position.get("unrealized_pnl")) * usd_krw_rate
          else:
            position["market_value_krw"] = market_value
            position["unrealized_pnl_krw"] = _number_or_zero(position.get("unrealized_pnl"))
          invested_value_krw += _number_or_zero(position.get("market_value_krw"))
        # Total equity from the CORRECTED parts (sane FX): KRW cash + foreign cash +
        # stock value. Do not trust a broker-provided total that can embed a bad FX
        # rate (which inflated both 외화 예수금 and the domestic-vs-total composition).
        corrected_parts_equity = krw_cash + foreign_cash_krw + invested_value_krw
        # portfolio.account.equity is total_equity_krw — KIS's integrated 총자산
        # (tot_asst_amt), the single value kis_real.get_portfolio reconciles across
        # settlement and the number the KIS app itself shows. Prefer it verbatim.
        #
        # corrected_parts_equity (krw_cash + foreign + ALL stock) DOUBLE-COUNTS a
        # pending same-day domestic BUY: until T+2 settlement the buy cash still sits
        # in the KRW deposit (krw_cash) while the purchased stock ALSO appears in
        # invested_value, so the parts sum overstates by the in-flight buy amount
        # (live: deposit 1,530,244 + stock 889,899 + ... = 2.51M vs true 1.67M). The
        # old max(broker, parts) then wrongly picked the inflated parts sum. Trust the
        # broker integrated total when present; fall back to the parts sum only when
        # the broker gives nothing, and never read below the cash we can see.
        broker_equity = _number_or_zero(getattr(portfolio.account, "equity", 0.0))
        if broker_equity > 0:
          actual_equity = broker_equity
        else:
          actual_equity = corrected_parts_equity
        result["cash"] = krw_cash
        result["cash_equivalent_krw"] = cash_equivalent_krw
        result["actual_deposit"] = krw_cash
        result["krw_cash"] = cash_by_currency.get("KRW", portfolio.account.cash)
        result["foreign_cash_krw"] = foreign_cash_krw
        result["cash_by_currency"] = cash_by_currency
        result["orderable_cash_by_currency"] = orderable_cash_by_currency
        result["foreign_cash_by_currency"] = _foreign_cash_by_currency(cash_by_currency)
        result["base_currency"] = getattr(portfolio.account, "base_currency", "KRW")
        result["invested_value"] = invested_value_krw
        result["actual_equity"] = actual_equity
        result["cash_weight"] = krw_cash / actual_equity if actual_equity > 0 else 0.0
        result["actual_deposit_currency"] = "KRW"
        # Today's realized (settled) P&L = domestic + overseas, both in KRW.
        # Fail-safe: each lookup degrades to 0 with a diagnostic rather than
        # raising, and the overseas leg failing must never drop the domestic
        # figure (accounts without overseas trading permission return an error).
        from zoneinfo import ZoneInfo

        today = datetime.now(ZoneInfo("Asia/Seoul")).date()
        domestic_realized_krw = 0.0
        overseas_realized_krw = 0.0
        try:
          domestic_realized_krw = float(client.get_domestic_realized_pnl(today, today))
        except Exception as exc:  # noqa: BLE001 - realized P&L is best-effort.
          result["realized_pnl_error"] = str(exc)
        try:
          overseas_settlement = client.get_overseas_settlement_summary(
              today,
              today,
          )
          overseas_realized_krw = float(
              overseas_settlement.get("realized_pnl_krw") or 0.0
          )
          result["broker_expenses_today_overseas_krw"] = float(
              overseas_settlement.get("broker_expenses_krw") or 0.0
          )
          result["gross_trading_difference_today_overseas_krw"] = float(
              overseas_settlement.get("gross_trading_difference_krw") or 0.0
          )
        except Exception as exc:  # noqa: BLE001 - overseas realized P&L is best-effort.
          result["realized_pnl_overseas_error"] = str(exc)
        result["realized_pnl_today_krw"] = domestic_realized_krw + overseas_realized_krw
        result["realized_pnl_today_domestic_krw"] = domestic_realized_krw
        result["realized_pnl_today_overseas_krw"] = overseas_realized_krw
        result["account_api_sources"] = {
            "domestic_balance": "TTTC8434R /uapi/domestic-stock/v1/trading/inquire-balance",
            "domestic_orderable_cash": "TTTC8908R /uapi/domestic-stock/v1/trading/inquire-psbl-order",
            "domestic_realized_pnl": "TTTC8715R /uapi/domestic-stock/v1/trading/inquire-period-trade-profit",
            "overseas_balance": "TTTS3012R /uapi/overseas-stock/v1/trading/inquire-balance",
            "overseas_present_balance": "CTRP6504R /uapi/overseas-stock/v1/trading/inquire-present-balance",
            "overseas_orderable_cash": "TTTS3007R /uapi/overseas-stock/v1/trading/inquire-psamount",
            "overseas_realized_pnl": "TTTS3039R /uapi/overseas-stock/v1/trading/inquire-period-profit",
        }
      else:
        result["account_checked"] = False
      return result
    except KisApiError as exc:
      return _kis_probe_error_payload(mode, exc)
    except (RuntimeError, OSError) as exc:
      return {
          "ok": False,
          "mode": mode,
          "error": str(exc),
          "message": str(exc),
      }


def _kis_probe_error_payload(mode: str, exc: KisApiError) -> dict[str, Any]:
    response = getattr(exc, "response", {}) or {}
    error_code = str(response.get("error_code") or response.get("rt_cd") or "")
    raw_message = str(
        response.get("error_description")
        or response.get("msg1")
        or response.get("message")
        or exc
    )
    message = raw_message
    retry_after_seconds = None
    if error_code == "EGW00133":
      message = (
          "KIS 접근토큰 발급 제한입니다. 이미 발급된 토큰이 있으면 "
          "config/secrets/kis_api_keys.env의 KIS_LIVE_ACCESS_TOKEN에 넣고 다시 점검하세요. "
          "캐시 토큰이 있으면 앱은 새 발급 없이 그 토큰으로 실계좌 읽기를 시도합니다."
      )
      retry_after_seconds = 60
    return {
        "ok": False,
        "mode": mode,
        "error_code": error_code,
        "error": str(exc),
        "raw_message": raw_message,
        "message": message,
        "retry_after_seconds": retry_after_seconds,
    }


@app.get("/api/operation-mode/status")
async def operation_mode_status() -> JSONResponse:
    streaming = []
    with _streaming_demos_lock:
      for demo_id, demo in list(_streaming_demos.items())[-5:]:
        streaming.append(
            {
                "demo_id": demo_id,
                "progress": demo.get_progress(),
                "complete": demo.is_complete(),
            }
        )
    with _live_lock:
      active = _to_jsonable(_operation_mode_state.get("active"))
      last_kis_connection = _to_jsonable(_operation_mode_state.get("last_kis_connection"))
    if isinstance(active, dict) and last_kis_connection and active.get("mode") in {"live_readiness", "live_trading"}:
      active["kis_connection"] = last_kis_connection
    return _json(
        {
            "active": active,
            "kis_connection": last_kis_connection,
            "request": _operation_mode_request_snapshot(),
            "learning": _learning_state_snapshot(),
            "collection_log": _live_snapshot()["collection_log"],
            "streaming": streaming,
            "auto_reliability": _auto_reliability_status(),
        }
    )


def _auto_reliability_status() -> dict[str, Any]:
  with _live_lock:
    return _to_jsonable(dict(_auto_reliability_state))


@app.get("/api/auto-reliability/status")
def auto_reliability_status() -> JSONResponse:
  return _json(_auto_reliability_status())


def _diagnostic_file_state(path: Path, now: datetime) -> dict[str, Any]:
  try:
    stat = path.stat()
  except OSError:
    return {
        "path": str(path),
        "exists": False,
        "size_bytes": 0,
        "updated_at": None,
        "age_seconds": None,
    }
  updated = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
  return {
      "path": str(path),
      "exists": True,
      "size_bytes": int(stat.st_size),
      "updated_at": updated.isoformat(),
      "age_seconds": round(max(0.0, (now - updated).total_seconds()), 1),
  }


def _latest_collection_entry(
  rows: list[dict[str, Any]],
  *,
  statuses: set[str] | None = None,
  message_contains: str | None = None,
) -> dict[str, Any] | None:
  needle = str(message_contains or "").lower()
  for row in reversed(rows):
    if statuses and str(row.get("status") or "").lower() not in statuses:
      continue
    if needle and needle not in str(row.get("message") or "").lower():
      continue
    return dict(row)
  return None


def _live_training_history(
  *,
  root: Path | None = None,
  limit: int = 48,
  use_cache: bool = True,
) -> dict[str, Any]:
  """Return real completed training cycles, never synthetic epoch progress."""
  model_root = root or Path("data/models/live_short_horizon")
  history_limit = max(2, min(240, int(limit)))
  cache_key = str(model_root.resolve())
  now_monotonic = time.monotonic()
  cached_payload = _live_training_history_cache.get("payload")
  if (
      use_cache
      and cached_payload is not None
      and _live_training_history_cache.get("root") == cache_key
      and int(_live_training_history_cache.get("limit") or 0) == history_limit
      and now_monotonic - float(_live_training_history_cache.get("loaded_at") or 0.0) < 60.0
  ):
    return dict(cached_payload)

  try:
    paths = sorted(
        (
            path
            for path in model_root.glob("live_short_horizon.*.json")
            if path.name != "latest.json"
        ),
        key=lambda path: path.name,
    )[-history_limit:]
  except OSError:
    paths = []

  points: list[dict[str, Any]] = []
  for path in paths:
    try:
      artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
      continue
    metrics = dict(artifact.get("metrics") or {})
    training_data = dict(artifact.get("training_data") or {})
    training_state = dict(artifact.get("training_state") or {})
    deployment = dict(artifact.get("deployment") or {})
    row_count = int(
        training_data.get("row_count")
        or metrics.get("example_count")
        or 0
    )
    materialized_row_count = int(
        training_data.get("materialized_row_count")
        or row_count
    )
    points.append(
        {
            "timestamp": artifact.get("created_at"),
            "artifact_id": artifact.get("artifact_id") or path.stem,
            "auc": float(metrics.get("auc") or 0.0),
            "precision_at_k": float(metrics.get("precision_at_k") or 0.0),
            "top_return_bps": float(
                metrics.get("avg_forward_net_return_bps_top_k") or 0.0
            ),
            "training_rows": row_count,
            "materialized_rows": materialized_row_count,
            "fresh_rows": int(training_data.get("fresh_row_count") or 0),
            "new_rows": int(training_data.get("new_materialized_row_count") or 0),
            "positive_labels": int(metrics.get("positive_labels") or 0),
            "negative_labels": int(metrics.get("negative_labels") or 0),
            "live_eligible": bool(artifact.get("live_eligible")),
            "promoted": bool(deployment.get("promoted")),
            "deployment_reason": deployment.get("reason"),
            "training_mode": (
                training_state.get("mode")
                or training_data.get("training_mode")
                or "full"
            ),
            "parent_artifact_id": training_state.get("parent_artifact_id"),
            "incremental_rows": int(
                training_state.get("incremental_example_count")
                or training_data.get("incremental_row_count")
                or row_count
            ),
            "full_retrain_reason": (
                training_state.get("full_retrain_reason")
                or training_data.get("full_retrain_reason")
            ),
        }
    )

  latest = points[-1] if points else {}
  previous = points[-2] if len(points) > 1 else {}
  rows = int(latest.get("training_rows") or 0)
  first_timestamp: datetime | None = None
  last_timestamp: datetime | None = None
  if points:
    try:
      first_timestamp = datetime.fromisoformat(
          str(points[0].get("timestamp") or "").replace("Z", "+00:00")
      )
      last_timestamp = datetime.fromisoformat(
          str(points[-1].get("timestamp") or "").replace("Z", "+00:00")
      )
      if first_timestamp.tzinfo is None:
        first_timestamp = first_timestamp.replace(tzinfo=timezone.utc)
      if last_timestamp.tzinfo is None:
        last_timestamp = last_timestamp.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
      first_timestamp = None
      last_timestamp = None
  elapsed_hours = (
      max(0.0, (last_timestamp - first_timestamp).total_seconds() / 3600.0)
      if first_timestamp is not None and last_timestamp is not None
      else 0.0
  )
  row_growth = int(latest.get("materialized_rows") or 0) - int(
      (points[0] if points else {}).get("materialized_rows") or 0
  )
  vectorized = rows >= 1_000
  incremental_mode = str(latest.get("training_mode") or "") == "incremental"
  optimizer = {
      "classification_family": "logistic_regression_sgd",
      "classification_learning_rate": (
          _env_float_web("LIVE_MODEL_BATCH_LOGISTIC_LR", 0.08)
          if vectorized
          else 0.08
      ),
      "classification_epochs": (
          _auto_reliability_int("LIVE_MODEL_INCREMENTAL_LOGISTIC_EPOCHS", 25, 1)
          if incremental_mode
          else _auto_reliability_int("LIVE_MODEL_BATCH_LOGISTIC_EPOCHS", 250, 20)
          if vectorized
          else 250
      ),
      "regression_family": "linear_regression_sgd",
      "regression_learning_rate": (
          _env_float_web("LIVE_MODEL_BATCH_LINEAR_LR", 0.01)
          if vectorized
          else 0.01
      ),
      "regression_epochs": (
          _auto_reliability_int("LIVE_MODEL_INCREMENTAL_LINEAR_EPOCHS", 18, 1)
          if incremental_mode
          else _auto_reliability_int("LIVE_MODEL_BATCH_LINEAR_EPOCHS", 180, 20)
          if vectorized
          else 180
      ),
      "l2": _env_float_web("LIVE_MODEL_L2", 0.001),
      "implementation": "vectorized_batch" if vectorized else "sample_sgd",
      "training_mode": "incremental" if incremental_mode else "full",
  }
  auc_delta = float(latest.get("auc") or 0.0) - float(previous.get("auc") or 0.0)
  precision_delta = float(latest.get("precision_at_k") or 0.0) - float(
      previous.get("precision_at_k") or 0.0
  )
  return_delta = float(latest.get("top_return_bps") or 0.0) - float(
      previous.get("top_return_bps") or 0.0
  )
  if not points:
    status = "waiting"
  elif int(latest.get("new_rows") or 0) <= 0:
    status = "collecting"
  elif bool(latest.get("promoted")):
    status = "promoted"
  elif auc_delta > 0 or precision_delta > 0 or return_delta > 0:
    status = "evaluated_improved"
  else:
    status = "evaluated"
  payload = {
      "points": points,
      "status": status,
      "cycle_interval_seconds": LIVE_TRAINING_INTERVAL_SECONDS,
      "optimizer": optimizer,
      "latest": latest or None,
      "change": {
          "auc": auc_delta,
          "precision_at_k": precision_delta,
          "top_return_bps": return_delta,
          "training_rows": int(latest.get("training_rows") or 0)
          - int(previous.get("training_rows") or 0),
          "materialized_rows": int(latest.get("materialized_rows") or 0)
          - int(previous.get("materialized_rows") or 0),
      },
      "rows_per_hour": row_growth / elapsed_hours if elapsed_hours > 0 else 0.0,
      "window_hours": elapsed_hours,
      "note": (
          "차트는 완료된 실제 학습 사이클을 표시합니다. 짧은 배치 학습이므로 "
          "실행 중 에포크 진행률은 생성하지 않습니다."
      ),
  }
  if use_cache:
    _live_training_history_cache.update(
        {
            "loaded_at": now_monotonic,
            "root": cache_key,
            "limit": history_limit,
            "payload": payload,
        }
    )
  return payload


def _diagnostic_blocker(code: str, components: dict[str, Any]) -> dict[str, Any]:
  details = {
      "RISK_POLICY_NOT_READY": (
          "위험 정책",
          "손절 정책이 비활성화되어 실거래 승격이 차단되었습니다.",
          ", ".join((components.get("risk_policy") or {}).get("failures") or ("정책 확인 필요",)),
      ),
      "MODEL_NOT_READY": (
          "모델 성능",
          "최신 재학습 모델이 실거래 성능 기준을 통과하지 못했습니다.",
          ", ".join((components.get("model") or {}).get("reason_codes") or ("모델 확인 필요",)),
      ),
      "MARKET_DATA_NOT_READY": (
          "실시간 시세·호가",
          "현재 열린 시장에서 거래급 체결과 호가가 충분히 들어오지 않습니다.",
          ", ".join((components.get("market_data") or {}).get("missing_markets") or ("시장 확인 필요",)),
      ),
      "BROKER_NOT_READY": (
          "KIS 연결",
          "KIS 실계좌 조회 또는 인증 상태가 준비되지 않았습니다.",
          "계좌 연결 확인 필요",
      ),
      "RUNTIME_NOT_READY": (
          "실행 게이트",
          "실거래 런타임 안전 게이트가 차단 상태입니다.",
          ", ".join((components.get("runtime") or {}).get("failures") or ("게이트 확인 필요",)),
      ),
      "CONFIG_NOT_READY": (
          "실거래 설정",
          "실거래 설정이 완전히 활성화되지 않았습니다.",
          "설정 확인 필요",
      ),
      "MARKET_CLOSED": (
          "시장 시간",
          "현재 실거래 가능한 시장이 닫혀 있습니다.",
          "개장 시간까지 대기",
      ),
  }
  label, message, detail = details.get(
      code,
      (code.replace("_", " "), "실거래 승격 조건을 통과하지 못했습니다.", code),
  )
  return {"code": code, "label": label, "message": message, "detail": detail}


def _intelligence_lineage_payload(
    snapshot: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
  context = snapshot.get("context")
  events = tuple(getattr(context, "events", ()) or ()) if context is not None else ()
  triples = tuple(getattr(getattr(context, "graph", None), "triples", lambda: ())())
  fresh_cutoff = now - timedelta(hours=24)
  fresh_events = 0
  llm_events = 0
  labeled_events = 0
  fact_events = 0
  for event in events:
    event_at = getattr(event, "event_date", None)
    if isinstance(event_at, datetime):
      if event_at.tzinfo is None:
        event_at = event_at.replace(tzinfo=timezone.utc)
      fresh_events += int(event_at >= fresh_cutoff)
    model = str(getattr(event, "classification_model", "") or "")
    llm_events += int(model not in {"", "keyword_v1", "keyword_v1_after_llm_error"})
    labeled_events += int(bool(getattr(event, "event_labels", ()) or ()))
    fact_events += int(bool(getattr(event, "key_facts", ()) or ()))
  event_predicates = {
      "hasRecentNews",
      "hasRecentDisclosure",
      "generatesSemanticFeature",
  }
  event_graph_links = sum(
      1 for triple in triples if str(getattr(triple, "predicate", "")) in event_predicates
  )
  try:
    from app.graph.macro_micro_feed import snapshot as macro_micro_snapshot

    macro_micro = macro_micro_snapshot() or {}
  except Exception:
    macro_micro = {}
  macro_diag = dict((macro_micro.get("macro_result") or {}).get("diagnostics") or {})
  micro_rows = list(macro_micro.get("micro_results") or ())
  micro_event_evidence = sum(
      int(((row.get("diagnostics") or {}).get("event_evidence_count") or 0))
      for row in micro_rows
      if isinstance(row, dict)
  )
  research_result = snapshot.get("research_result")
  research_reason = str(
      ((getattr(research_result, "diagnostics", None) or {}).get("reason") or "")
  )
  collection_active = research_reason != "live_trading_fast_path"
  ready = bool(events and event_graph_links)
  return {
      "ready": ready,
      "status": "ready" if ready else "degraded",
      "research_collection_enabled_in_live": collection_active,
      "research_last_collected_at": _iso_or_none(
          snapshot.get("research_last_collected_at")
      ),
      "research_interval_seconds": LIVE_RESEARCH_COLLECTION_INTERVAL_SECONDS,
      "context_events": len(events),
      "fresh_events_24h": fresh_events,
      "llm_classified_events": llm_events,
      "labeled_events": labeled_events,
      "events_with_key_facts": fact_events,
      "ontology_event_links": event_graph_links,
      "macro_event_evidence": int(macro_diag.get("macro_event_count") or 0),
      "micro_event_evidence": micro_event_evidence,
      "synthetic_event_count": sum(
          1
          for event in events
          if str(getattr(getattr(event, "source", None), "raw_url", "") or "").startswith(
              "local://sample"
          )
      ),
  }


def _system_diagnostics_payload() -> dict[str, Any]:
  now = datetime.now(timezone.utc)
  snapshot = _live_snapshot()
  learning = snapshot.get("learning") or {}
  rows = list(snapshot.get("collection_log") or ())
  reliability = _auto_reliability_status()
  components = dict(reliability.get("components") or {})
  latest_complete = _latest_collection_entry(rows, statuses={"complete"})
  latest_training = next(
      (
          dict(row)
          for row in reversed(rows)
          if any(
              marker in str(row.get("message") or "").lower()
              for marker in ("model retrained", "model training skipped", "model challenger")
          )
      ),
      None,
  )
  latest_market = next(
      (
          dict(row)
          for row in reversed(rows)
          if any(
              marker in str(row.get("message") or "").lower()
              for marker in ("realtime collector", "realtime bridge", "krx fully closed", "live market data")
          )
      ),
      None,
  )
  latest_collector_result = next(
      (
          dict(row)
          for row in reversed(rows)
          if int(((row.get("counts") or {}).get("control_messages") or 0)) > 0
      ),
      None,
  )
  with _live_lock:
    learning_worker_running = bool(_live_worker is not None and _live_worker.is_alive())
    training_worker_running = bool(
        _live_training_worker is not None
        and _live_training_worker.is_alive()
        and not _live_training_stop.is_set()
    )
    collector_running = bool(
        _kis_realtime_collector_worker is not None
        and _kis_realtime_collector_worker.is_alive()
        and not _kis_realtime_collector_stop.is_set()
    )
    krx_feature_frame_running = bool(
        _krx_feature_frame_thread is not None
        and _krx_feature_frame_thread.is_alive()
        and not _krx_feature_frame_stop.is_set()
    )
    krx_feature_symbols = list(_kis_realtime_complete_symbols)
    krx_feature_sampled_symbols = list(_krx_feature_last_signature)
    us_fast_poll_status = dict(_us_fast_poll_state)
    us_websocket_status = dict(_kis_overseas_realtime_state)
  with _realtime_trading_lock:
    trading_running = bool(
        _realtime_trading_worker is not None and _realtime_trading_worker.is_alive()
    )
    trading_engine = _realtime_trading_engine
  try:
    trading_engine_status = trading_engine.get_status() if trading_engine is not None else {}
  except Exception:
    trading_engine_status = {}
  us_fast_poll_running = bool(_us_fast_poll_thread is not None and _us_fast_poll_thread.is_alive())
  reasons = [str(code) for code in reliability.get("reasons") or ()]
  blockers = [_diagnostic_blocker(code, components) for code in reasons]
  # Reliability transitions are deliberately cadence-cached, but diagnostics must
  # show the artifact written moments ago rather than the previous 5-minute sample.
  model_health = _latest_model_reliability(now)
  challenger_health = dict(model_health.get("latest_challenger") or {})
  displayed_model = (
      challenger_health
      if not bool(model_health.get("schema_matches")) and challenger_health
      else model_health
  )
  training_counts = {
      **dict(displayed_model.get("metrics") or {}),
      **dict((latest_training or {}).get("counts") or {}),
  }
  try:
    with sqlite3.connect("data/store/live_training_rows.sqlite3") as conn:
      training_counts["materialized_training_rows"] = int(
          conn.execute(
              """
              select count(*) from live_training_rows
              where json_extract(payload, '$.feature_schema_hash') in ('', ?)
              """,
              (LIVE_SHORT_HORIZON_SCHEMA.schema_hash,),
          ).fetchone()[0]
      )
  except (sqlite3.Error, OSError, TypeError):
    pass
  evaluated_artifact_id = str(training_counts.get("artifact_id") or "")
  active_artifact_id = str(model_health.get("artifact_id") or "")
  training_counts["displayed_model_role"] = (
      "challenger"
      if (
          displayed_model is challenger_health
          or (evaluated_artifact_id and evaluated_artifact_id != active_artifact_id)
      )
      else "active"
  )
  collection_counts = dict((latest_complete or {}).get("counts") or {})
  market_counts = dict((latest_market or {}).get("counts") or {})
  collector_counts = dict((latest_collector_result or {}).get("counts") or {})
  subscription_requests = int(
      collector_counts.get("subscription_requests")
      or collector_counts.get("subscriptions")
      or 0
  )
  subscription_rejected = int(
      collector_counts.get("subscriptions_rejected")
      or collector_counts.get("control_errors")
      or 0
  )
  subscription_accepted = (
      int(collector_counts.get("subscriptions_accepted") or 0)
      if "subscriptions_accepted" in collector_counts
      else max(0, subscription_requests - subscription_rejected)
  )
  market_data = dict(components.get("market_data") or {})
  healthy = dict(market_data.get("healthy") or {})
  account = _last_live_account_basis() or {}
  orderable = account.get("orderable_cash_by_currency") or {}
  cash_by_currency = account.get("cash_by_currency") or {}
  usd_orderable = _number_or_zero(
      orderable.get("USD") if "USD" in orderable else cash_by_currency.get("USD")
  )
  active_markets = list(reliability.get("active_markets") or ())
  training_history = _live_training_history()

  research_active = bool(learning.get("active") and learning_worker_running)
  training_active = bool(training_worker_running)
  trade_data_active = bool(
      any(healthy.get(market) for market in active_markets)
      or int(market_counts.get("realtime_ticks") or 0) > 0
      or int(market_counts.get("live_us_realtime_bridge_ticks") or 0) > 0
  )
  current_mode = str(reliability.get("mode") or learning.get("mode") or "")
  if trading_running and current_mode == "live_trading" and bool(reliability.get("ready")):
    headline = "실시간 거래 엔진이 실행 중입니다."
    summary = "전략 판단과 주문 게이트가 실시간으로 평가되고 있습니다."
  elif trading_running:
    headline = "보유 종목 안전 감시는 실행 중이고, 신규 매수는 차단되어 있습니다."
    summary = "학습 모드에서는 매도 위험 감시만 유지하고 신규 주문 승격을 대기합니다."
  elif research_active and training_active:
    headline = "연구·재학습은 진행 중이고, 실거래 승격은 대기 중입니다."
    summary = "시스템 정지가 아니라 안전 게이트 차단 상태입니다."
  else:
    headline = "학습 또는 수집 워커 일부가 중지되어 있습니다."
    summary = "아래 워커 상태와 최근 활동을 확인하세요."

  file_states = {
      "research_store": _diagnostic_file_state(Path("data/store/research.sqlite3"), now),
      "realtime_store": _diagnostic_file_state(Path("data/store/realtime_market_data.sqlite3"), now),
      "feature_journal": _diagnostic_file_state(Path("logs/live-feature-frames.jsonl"), now),
      "decision_log": _diagnostic_file_state(Path("logs/decision-log.jsonl"), now),
  }
  return {
      "generated_at": now.isoformat(),
      "headline": headline,
      "summary": summary,
      "mode": reliability.get("mode") or learning.get("mode"),
      "score": float(reliability.get("score") or 0.0),
      "threshold": float(reliability.get("threshold") or 0.9),
      "ready": bool(reliability.get("ready")),
      "active_markets": active_markets,
      "account_context": {
          "krw_orderable": _number_or_zero(orderable.get("KRW")),
          "usd_orderable": usd_orderable,
          "us_collection_limited_by_cash": "US" in active_markets and usd_orderable <= 0,
      },
      "workers": [
          {
              "key": "research_collection",
              "label": "뉴스·시장 연구 수집",
              "running": research_active,
              "detail": "수집 사이클 실행/예약 중" if research_active else "수집 워커 중지",
          },
          {
              "key": "model_training",
              "label": "단기 모델 재학습",
              "running": training_active,
              "detail": (
                  f"AUC {float(training_counts.get('auc') or 0.0):.4f} · "
                  f"Precision@K {float(training_counts.get('precision_at_k') or 0.0):.4f}"
              ),
          },
          {
              "key": "krx_realtime",
              "label": "KIS 국내 실시간 수집",
              "running": collector_running,
              "detail": (
                  f"승인 {subscription_accepted} / 요청 {subscription_requests} · "
                  f"거절 {subscription_rejected}"
                  if collector_counts
                  else str((latest_market or {}).get("message") or "상태 기록 없음")
              ),
          },
          {
              "key": "us_realtime",
              "label": "미국 실시간 보강",
              "running": us_fast_poll_running or bool(healthy.get("US")),
              "detail": (
                  "USD 주문 가능 잔액 0원 · 미국 매수 후보 보강 제한"
                  if "US" in active_markets and usd_orderable <= 0
                  else f"건강한 종목 {len(healthy.get('US') or ())}개"
              ),
          },
          {
              "key": "trading_engine",
              "label": "실시간 거래 엔진",
              "running": trading_running,
              "detail": "실행 중" if trading_running else "신뢰도 게이트 통과 전 대기",
          },
      ],
      "flows": {
          "intelligence": _intelligence_lineage_payload(snapshot, now),
          "research_collection": {
              "active": research_active,
              "latest": latest_complete,
              "counts": collection_counts,
          },
          "market_data": {
              "active": trade_data_active,
              "healthy": healthy,
              "minimum_per_market": market_data.get("minimum_per_market"),
              "minimum_by_market": market_data.get("minimum_by_market"),
              "ready_markets": market_data.get("ready_markets"),
              "required_markets": market_data.get("required_markets"),
              "extended_order_markets": market_data.get("extended_order_markets"),
              "missing_markets": market_data.get("missing_markets"),
              "partial": bool(market_data.get("partial")),
              "latest": latest_market,
              "subscription": {
                  "requests": subscription_requests,
                  "accepted": subscription_accepted,
                  "rejected": subscription_rejected,
                  "error_codes": dict(collector_counts.get("subscription_errors_by_code") or {}),
                  "observed_capacity": _kis_realtime_observed_subscription_capacity,
                  "limit_reached": bool(collector_counts.get("subscription_limit_reached")),
                  "overseas_websocket": {
                      "running": bool(us_websocket_status.get("running")),
                      "last_attempt_at": us_websocket_status.get("last_attempt_at"),
                      "last_success_at": us_websocket_status.get("last_success_at"),
                      "symbols": list(us_websocket_status.get("symbols") or ()),
                      "counts": dict(us_websocket_status.get("counts") or {}),
                      "last_error": us_websocket_status.get("last_error"),
                      "observed_capacity": _kis_overseas_observed_subscription_capacity,
                      "trade_tr_id": "HDFSCNT0",
                      "orderbook_tr_id": "HDFSASP0",
                      "orderbook_levels": 1,
                  },
              },
          },
          "training": {
              "active": training_active,
              "latest": latest_training,
              "metrics": training_counts,
              "history": training_history,
              "feature_frames_built": int(training_counts.get("feature_frames_built") or 0),
              "feature_sampler": {
                  "krx_running": krx_feature_frame_running,
                  "krx_interval_seconds": max(
                      2.0,
                      _env_float_web("LIVE_KRX_FEATURE_FRAME_SECONDS", 5.0),
                  ),
                  "complete_subscription_symbols": krx_feature_symbols,
                  "sampled_symbols": krx_feature_sampled_symbols,
                  "us_running": us_fast_poll_running,
                  "us_last_attempt_at": us_fast_poll_status.get("last_attempt_at"),
                  "us_last_success_at": us_fast_poll_status.get("last_success_at"),
                  "us_symbols": list(us_fast_poll_status.get("symbols") or ()),
                  "us_saved_ticks": int(us_fast_poll_status.get("saved_ticks") or 0),
                  "us_saved_orderbooks": int(us_fast_poll_status.get("saved_orderbooks") or 0),
                  "us_errors": dict(us_fast_poll_status.get("errors") or {}),
                  "us_last_error": us_fast_poll_status.get("last_error"),
              },
              "active_model": {
                  "artifact_id": model_health.get("artifact_id"),
                  "metrics": dict(model_health.get("metrics") or {}),
                  "age_seconds": model_health.get("age_seconds"),
                  "training_age_seconds": model_health.get("training_age_seconds"),
                  "training_heartbeat_at": model_health.get("training_heartbeat_at"),
                  "training_heartbeat_ok": bool(model_health.get("training_heartbeat_ok")),
                  "training_in_progress": bool(model_health.get("training_in_progress")),
                  "live_eligible": bool(model_health.get("live_eligible")),
              },
              "challenger": dict(model_health.get("latest_challenger") or {}),
              "training_skipped": bool(training_counts.get("training_skipped")),
              "deployment": {
                  "deployed": bool(training_counts.get("deployed")),
                  "reason": training_counts.get("deployment_reason"),
                  "artifact_id": training_counts.get("artifact_id"),
              },
          },
          "strategy_session": dict(trading_engine_status.get("strategy_session") or {}),
      },
      "blockers": blockers,
      "files": file_states,
      "recent_activity": rows[-10:],
      "next_collection_at": learning.get("next_collection_at"),
      "last_error": reliability.get("last_error") or snapshot.get("last_error"),
  }


@app.get("/api/system-diagnostics")
def system_diagnostics() -> JSONResponse:
  payload = _cached_system_diagnostics_payload()
  payload["market_session_capabilities"] = _market_session_capability_payload()
  return _json(payload)


def _market_session_capability_payload() -> dict[str, Any]:
  """전 세션 capability matrix + 미지원 route 목록.

  운영 화면이 **데이터 수신 가능** 과 **주문 가능** 을 같은 상태로 표시하지 않게 하려면
  두 값을 분리해 내려 줘야 한다. NXT 처럼 시세는 오지만 세션별 실주문 승인이 없는 경우는
  ``is_data_only`` 로 구별된다.
  """
  try:
    from app.data.market_capabilities import default_service

    service = default_service()
    return {
        "ok": True,
        "matrix": service.capability_matrix(),
        "sessions": service.session_report(),
    }
  except Exception as exc:  # noqa: BLE001 - 진단 payload 는 절대 서버를 죽이지 않는다.
    return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _cached_system_diagnostics_payload() -> dict[str, Any]:
  """Serve dashboard polling without stacking expensive history reads.

  The terminal asks every five seconds, while a full diagnostic includes model
  history and several live stores. Once a payload exists, refresh it in one
  background thread and immediately return the last complete snapshot.
  """
  now_monotonic = time.monotonic()
  try:
    ttl = max(10.0, float(os.getenv("SYSTEM_DIAGNOSTICS_CACHE_SECONDS", "30")))
  except (TypeError, ValueError):
    ttl = 30.0
  with _system_diagnostics_cache_lock:
    payload = _system_diagnostics_cache.get("payload")
    loaded_at = float(_system_diagnostics_cache.get("loaded_at") or 0.0)
    if isinstance(payload, dict) and now_monotonic - loaded_at < ttl:
      return dict(payload)
    if isinstance(payload, dict):
      if not bool(_system_diagnostics_cache.get("refreshing")):
        _system_diagnostics_cache["refreshing"] = True
        threading.Thread(
          target=_refresh_system_diagnostics_cache,
          name="system-diagnostics-refresh",
          daemon=True,
        ).start()
      return dict(payload)
    # The first request computes once while holding the lock. Other first-wave
    # pollers wait here and reuse that result instead of launching duplicates.
    computed = _system_diagnostics_payload()
    _system_diagnostics_cache.update(
      {"payload": computed, "loaded_at": time.monotonic(), "refreshing": False}
    )
    return dict(computed)


def _refresh_system_diagnostics_cache() -> None:
  try:
    computed = _system_diagnostics_payload()
  except Exception:  # noqa: BLE001 - retain the last complete dashboard snapshot.
    with _system_diagnostics_cache_lock:
      _system_diagnostics_cache["refreshing"] = False
    return
  with _system_diagnostics_cache_lock:
    _system_diagnostics_cache.update(
      {"payload": computed, "loaded_at": time.monotonic(), "refreshing": False}
    )


@app.post("/api/operation-mode/stop-learning")
async def operation_mode_stop_learning() -> JSONResponse:
    _start_live_worker("learning")
    with _live_lock:
      _append_collection_log_unlocked("scheduled", "Continuous learning remains active while the server is running")
    audit.record("learning_collection_continues", {"checked_at": datetime.now(timezone.utc).isoformat()})
    return _json(
        {
            "ok": True,
            "status": "continuous",
            "message": "Learning and information collection continue while the server is running.",
            "learning": _learning_state_snapshot(),
            "progress": _live_snapshot()["progress"],
            "collection_log": _live_snapshot()["collection_log"],
        }
    )


@app.get("/api/live-progress")
async def live_progress() -> JSONResponse:
    snapshot = _live_snapshot()
    return _json(
        {
            "is_refreshing": snapshot["is_refreshing"],
            "learning": snapshot["learning"],
            "collection_log": snapshot["collection_log"],
            "progress": snapshot["progress"],
            "updated_at": _iso_or_none(snapshot["last_updated"]),
            "last_error": snapshot["last_error"],
        }
    )


@app.get("/api/research/volume")
def research_volume() -> JSONResponse:
    store = LocalResearchStore(root=_get_store_root())
    summary = store.summary(prune=False)
    by_kind = {
        key: int(summary.get(key, 0))
        for key in (
            "events",
            "raw_records",
            "market_snapshots",
            "macro_metrics",
            "realtime_quotes",
            "realtime_executions",
        )
    }
    return _json(
        {
            "store_path": str(store.db_path),
            "data_volume": {
                "by_kind": by_kind,
                "by_source": [],
                "by_day": [],
                "market_snapshot_sources": {},
                "top_market_tickers": [],
            },
        }
    )


@app.get("/api/risk/principal-protection/state")
def principal_protection_state() -> JSONResponse:
    config = _principal_config_with_live_account_basis(_load_principal_protection_config())
    account = _principal_protection_account_snapshot(config)
    state = PrincipalProtectionEngine().compute_state(
        account,
        account.holdings,
        realized_pnl=account.realized_pnl_today,
        unrealized_pnl=account.unrealized_pnl_today,
        config=config,
        high_watermark=_load_principal_high_watermark(account.equity, config),
    )
    _save_principal_high_watermark(state.high_watermark)
    return _json(
        {
            "config": config,
            "state": state,
            "capital_allocation": _principal_capital_allocation(state),
            "updated_at": datetime.now(timezone.utc),
        }
    )


@app.put("/api/risk/principal-protection/config")
async def principal_protection_config_update(request: Request) -> JSONResponse:
    payload = await request.json()
    current = _load_principal_protection_config()
    config = _principal_config_from_payload({**asdict(current), **dict(payload)})
    _save_principal_protection_config(config)
    return principal_protection_state()


@app.post("/api/risk/principal-protection/preview-order")
async def principal_protection_preview_order(request: Request) -> JSONResponse:
    payload = await request.json()
    snapshot = _get_or_refresh_live()
    context = snapshot["context"]
    config = _principal_config_with_live_account_basis(_load_principal_protection_config())
    ticker = str(payload.get("ticker") or (context.markets[0].ticker if context.markets else ""))
    market = next((item for item in context.markets if item.ticker == ticker), context.markets[0])
    action = OrderAction(str(payload.get("action", "BUY")).upper())
    suggested_weight = float(payload.get("suggested_weight", 0.01))
    expected_exit_price = float(payload.get("expected_exit_price") or market.last_price * 1.02)
    quantity = int(payload.get("quantity") or max(0, int(context.account.equity * suggested_weight / max(1e-9, market.last_price))))
    intent = OrderIntent(
        ticker=market.ticker,
        market=market.market,
        action=action,
        suggested_weight=suggested_weight,
        confidence=float(payload.get("confidence", 0.5)),
        valid_until=datetime.now(timezone.utc) + timedelta(minutes=5),
        reasoning_summary=("principal protection preview",),
        supporting_factors=(),
        contradicting_factors=(),
        source_data_ids=("preview",),
        strategy_family="preview",
        expected_exit_price=expected_exit_price,
        target_net_return=float(payload.get("target_net_return", 0.0)),
        strategy_metadata={"stop_loss_price": payload.get("stop_loss_price")},
    )
    cost = None
    if action == OrderAction.BUY and quantity > 0:
        cost = RiskManager().cost_engine.estimate(
            symbol=market.ticker,
            market=market.market,
            venue="KRX",
            instrument_type="domestic_stock",
            entry_price=market.last_price,
            expected_exit_price=expected_exit_price,
            quantity=quantity,
            target_net_return=intent.target_net_return or 0.0,
            average_daily_trading_value=market.average_daily_trading_value,
        )
    decision = PrincipalProtectionEngine().validate_order(
        intent,
        context.account,
        context.account.holdings,
        market,
        cost,
        config,
        proposed_quantity=quantity,
        high_watermark=_load_principal_high_watermark(context.account.equity, config),
    )
    return _json({"decision": decision, "cost_breakdown": cost.as_dict() if cost else None})


@app.post("/api/live-snapshot")
async def live_snapshot(request: Request) -> JSONResponse:
    payload = await request.json()
    goal_payload = payload.get("goal")
    force_refresh = bool(payload.get("force_refresh", False))
    include_graph = bool(payload.get("include_graph", False))
    return _json(await run_in_threadpool(_live_snapshot_response, goal_payload, force_refresh, include_graph))


def _live_snapshot_response(goal_payload: Any, force_refresh: bool, include_graph: bool = False) -> dict[str, Any]:
    snapshot = _live_snapshot()
    if not force_refresh and not include_graph and (snapshot["context"] is None or snapshot["research_result"] is None):
      return _lightweight_live_snapshot_response(snapshot)
    if force_refresh or include_graph:
      snapshot = _get_or_refresh_live(force_refresh=force_refresh)
    research_result = snapshot["research_result"]
    context = snapshot["context"]
    store_summary = dict(snapshot.get("store_summary") or {})
    lightweight_volume = _lightweight_data_volume(store_summary)
    graph_triples_count = _graph_triples_count(snapshot, context)
    graph_payload: dict[str, Any] = {
        "counts": {
            "nodes": len(getattr(context.graph, "nodes", {}) or {}),
            "links": graph_triples_count,
        },
        "summary_only": True,
    }
    if include_graph:
      cached_graph = snapshot.get("graph_payload")
      if cached_graph is None or snapshot.get("graph_payload_context_id") != id(context):
        cached_graph = _graph_payload(context)
        with _live_lock:
          if _live_state["context"] is context:
            _live_state["graph_payload"] = cached_graph
            _live_state["graph_payload_context_id"] = id(context)
      graph_payload = cached_graph
    live_basis = _last_live_account_basis()
    status_payload = {
        "cash": context.account.cash,
        "equity": context.report.equity,
        "cash_weight": context.report.cash_weight,
        "basis_source": "realtime_model_account",
        "account_suffix": None,
        "daily_pnl_ratio": context.report.daily_pnl_ratio,
        "updated_at": _iso_or_none(snapshot["last_updated"]),
        "last_error": snapshot["last_error"],
    }
    if live_basis is not None:
      status_payload.update(
          {
              "cash": live_basis["cash"],
              "cash_equivalent_krw": live_basis.get("cash_equivalent_krw", live_basis["cash"]),
              "krw_cash": live_basis["krw_cash"],
              "foreign_cash_krw": live_basis["foreign_cash_krw"],
              "cash_by_currency": live_basis["cash_by_currency"],
              "foreign_cash_by_currency": live_basis["foreign_cash_by_currency"],
              "base_currency": live_basis["base_currency"],
              "equity": live_basis["equity"],
              "cash_weight": live_basis["cash_weight"],
              "basis_source": live_basis["source"],
              "account_suffix": live_basis["account_suffix"],
              "daily_pnl_ratio": 0.0,
              "updated_at": datetime.now(timezone.utc),
          }
      )
    response: dict[str, Any] = {
        "status": status_payload,
        "diagnostics": {
            "research_config": str(DEFAULT_RESEARCH_CONFIG),
            "diagnostics": _diagnostics_with_collection_config(research_result.diagnostics),
            "skipped_sources": research_result.skipped_sources,
            "stored_new_records": snapshot["stored_new_records"],
            "store_summary": store_summary,
            "data_volume": lightweight_volume,
            "store_path": str(LocalResearchStore(root=_get_store_root()).db_path),
            "data_policy": _current_data_policy(),
            "graph_triples_count": graph_triples_count,
            "reasoning_paths": context.reasoning_paths[:25],
            "ontology_runtime": context.ontology_runtime.as_dict(),
            "is_refreshing": snapshot["is_refreshing"],
            "refresh_interval_seconds": LIVE_REFRESH_SECONDS,
        },
        "graph": graph_payload,
        "updated_at": _iso_or_none(snapshot["last_updated"]),
    }
    if isinstance(goal_payload, dict):
        goal_request = _parse_goal_request(goal_payload)
        assessment = assess_goal(
            goal_request,
            _goal_account_snapshot(context),
            context.markets,
            context.indicators,
            context.signals,
            context.graph,
        )
        response["assessment"] = assessment
        response["compromises"] = build_compromise_goals(assessment)
    return response


def _graph_triples_count(snapshot: dict[str, Any], context: Any) -> int:
    cached_graph = snapshot.get("graph_payload")
    if isinstance(cached_graph, dict):
      counts = cached_graph.get("counts")
      if isinstance(counts, dict):
        try:
          return int(counts.get("links", 0) or 0)
        except (TypeError, ValueError):
          return 0
    return len(context.graph.triples())


def _lightweight_live_snapshot_response(snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    snapshot = snapshot or _live_snapshot()
    basis = _last_live_account_basis()
    diagnostics = _lightweight_diagnostics_response(snapshot)
    cash = float(basis["cash"]) if basis is not None else 0.0
    equity = float(basis["equity"]) if basis is not None else cash
    return {
        "status": {
            "cash": cash,
            "cash_equivalent_krw": float(basis.get("cash_equivalent_krw") or cash) if basis is not None else cash,
            "krw_cash": float(basis["krw_cash"]) if basis is not None else cash,
            "foreign_cash_krw": float(basis["foreign_cash_krw"]) if basis is not None else 0.0,
            "cash_by_currency": basis["cash_by_currency"] if basis is not None else {"KRW": cash},
            "foreign_cash_by_currency": basis["foreign_cash_by_currency"] if basis is not None else {},
            "base_currency": basis["base_currency"] if basis is not None else "KRW",
            "equity": equity,
            "cash_weight": float(basis["cash_weight"]) if basis is not None else 0.0,
            "basis_source": basis["source"] if basis is not None else "warming_up",
            "account_suffix": basis["account_suffix"] if basis is not None else None,
            "daily_pnl_ratio": 0.0,
            "updated_at": _iso_or_none(snapshot.get("last_updated")),
            "last_error": snapshot.get("last_error"),
        },
        "diagnostics": diagnostics,
        "graph": {
            "counts": {"nodes": 0, "links": 0},
            "summary_only": True,
        },
        "updated_at": _iso_or_none(snapshot.get("last_updated")),
        "warming_up": True,
    }


def _lightweight_data_volume(summary: dict[str, Any]) -> dict[str, Any]:
    by_kind = {
        key: int(summary.get(key, 0) or 0)
        for key in (
            "events",
            "raw_records",
            "market_snapshots",
            "macro_metrics",
            "realtime_quotes",
            "realtime_executions",
        )
    }
    return {
        "by_kind": by_kind,
        "by_source": [],
        "by_day": [],
        "market_snapshot_sources": {},
        "top_market_tickers": [],
    }


@app.post("/api/assess-goal")
async def assess_goal_api(request: Request) -> JSONResponse:
    payload = await request.json()
    return _json(await run_in_threadpool(_assess_goal_response, payload))


def _assess_goal_response(payload: dict[str, Any]) -> dict[str, Any]:
    goal_request = _parse_goal_request(payload)
    snapshot = _live_snapshot()
    context = snapshot.get("context")
    if context is None:
      basis = _last_live_account_basis()
      account = AccountSnapshot(
          cash=max(1.0, float(basis["cash"])) if basis is not None else 10_000_000.0,
          holdings=(),
      )
      assessment = assess_goal(goal_request, account, (), {}, (), KnowledgeGraph())
      provisional = True
    else:
      assessment = assess_goal(
          goal_request,
          _goal_account_snapshot(context),
          context.markets,
          context.indicators,
          context.signals,
          context.graph,
      )
      provisional = False
    compromises = build_compromise_goals(assessment)
    session_id = str(uuid4())
    sessions[session_id] = {"assessment": assessment, "compromises": compromises, "started": False}
    audit.record("goal_assessment", {"session_id": session_id, "assessment": assessment})
    return {
        "session_id": session_id,
        "assessment": assessment,
        "compromises": compromises,
        "provisional": provisional,
    }


@app.post("/api/start")
async def start_program(request: Request) -> JSONResponse:
    payload = await request.json()
    session_id = str(payload.get("session_id", ""))
    selected = payload.get("selected_goal")
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Unknown negotiation session")
    if not isinstance(selected, dict):
        raise HTTPException(status_code=400, detail="selected_goal is required")

    goal = NegotiatedGoal(
        target_return_rate=float(selected["target_return_rate"]),
        target_profit_amount=float(selected["target_profit_amount"]),
        period_days=int(selected["period_days"]),
        feasibility_percent=int(selected["feasibility_percent"]),
        label=str(selected.get("label", "Accepted target")),
    )
    context = _get_or_refresh_live()["context"]
    mock_account = _mock_demo_account(context)
    broker = _reset_mock_kis_for_context(context, mock_account)
    run = run_mock_trading_cycle(
        goal,
        mock_account,
        context.markets,
        context.indicators,
        context.graph,
        broker=broker,
    )
    sessions[session_id]["started"] = True
    sessions[session_id]["goal"] = goal
    sessions[session_id]["mock_trading_run"] = run
    _mock_trading_state.update(
        {
            "active": True,
            "session_id": session_id,
            "goal": goal,
            "started_at": datetime.now(),
            "initial_equity": mock_account.equity,
            "last_run": run,
        }
    )
    audit.record("mock_program_started_after_goal_acceptance", {"session_id": session_id, "goal": goal, "run": run})
    return _json(
        {
            "started": True,
            "mode": "mock_kis_paper_trading",
            "message": "선택한 목표 기준으로 모의 KIS 자동매매 데모를 시작했습니다. 실거래는 비활성화되어 있습니다.",
            "accepted_goal": goal,
            "llm_judgment": run.llm_judgment,
            "ontology_evidence": run.ontology_evidence,
            "goal_execution_plan": run.goal_plan,
            "signals": run.goal_plan.signals,
            "order_intents": run.order_intents,
            "risk_results": run.risk_results,
            "kis_order_receipts": run.kis_order_receipts,
            "kis_executions": run.kis_executions,
            "portfolio": run.portfolio,
            "performance": _mock_performance(context),
        }
    )


@app.post("/api/mock-trading/run")
async def mock_trading_run(request: Request) -> JSONResponse:
    payload = await request.json()
    context = _get_or_refresh_live()["context"]
    goal = _goal_from_payload(payload, context)
    mock_account = _mock_demo_account(context)
    run = run_mock_trading_cycle(
        goal,
        mock_account,
        context.markets,
        context.indicators,
        context.graph,
        broker=_reset_mock_kis_for_context(context, mock_account),
    )
    audit.record("mock_trading_run", run)
    return _json(run)


@app.post("/api/mock-kis/orders")
async def mock_kis_place_order(request: Request) -> JSONResponse:
    payload = await request.json()
    context = _get_or_refresh_live()["context"]
    order = _parse_final_order(payload)
    broker = _mock_kis_for_context(context)
    receipt = broker.place_limit_order(order)
    execution = broker.get_order_status(receipt.order_id)
    return _json({"receipt": receipt, "execution": execution})


@app.get("/api/mock-kis/orders/{order_id}")
def mock_kis_order_status(order_id: str) -> JSONResponse:
    context = _get_or_refresh_live()["context"]
    broker = _mock_kis_for_context(context)
    return _json(broker.get_order_status(order_id))


@app.post("/api/paper-trading/terminate/{demo_id}")
async def streaming_demo_terminate(demo_id: str) -> JSONResponse:
    return _paper_trading_removed_response()


def _streaming_demo_terminate_response(demo_id: str) -> dict[str, Any]:
    step_lock = _streaming_demo_step_locks.setdefault(demo_id, threading.Lock())
    if not step_lock.acquire(blocking=False):
        demo = _streaming_demos.get(demo_id)
        return {
            "ok": False,
            "demo_id": demo_id,
            "status": "busy",
            "account": _streaming_demo_account_payload(demo),
            "message": "Paper trading is processing a step. Try termination again in a moment.",
        }
    try:
        demo = _streaming_demos.get(demo_id)
        if demo is None:
            return {"ok": False, "demo_id": demo_id, "status": "expired", "message": "Paper trading session not found."}
        current_step = int(getattr(demo, "_current_step", 0) or 0)
        prices = {
            ticker: float(bars[min(max(0, current_step), len(bars) - 1)].close)
            for ticker, bars in getattr(demo, "_bars_by_ticker", {}).items()
            if bars
        }
        trades = demo._liquidate_holdings(prices, datetime.now(timezone.utc))
        if hasattr(demo, "_current_step") and hasattr(demo, "_timestamps"):
            demo._current_step = len(getattr(demo, "_timestamps", ()))
        return {
            "ok": True,
            "demo_id": demo_id,
            "status": "terminated",
            "liquidated": True,
            "sell_order_count": len(trades),
            "trades": [_to_jsonable(trade) for trade in trades],
            "account": _streaming_demo_account_payload(demo),
            "final_results": demo.get_final_results(),
            "message": "Paper trading terminated after selling all current simulated holdings.",
        }
    finally:
        step_lock.release()


# ---- Human-readable trade explanations (Pi kiosk display) -------------------
# Maps the engine's terse reason codes into plain-Korean "why" phrases so the
# local monitor can explain, intuitively, why each stock was bought or sold at
# this price and time. Purely presentational — no decision logic here.
_BUY_REASON_TEXT = {
    "PositiveNewsConfirm": "긍정 뉴스가 매수를 뒷받침",
    "InformedOrderFlowImbalance": "외국인·기관 매수 우위(정보성 수급)",
    "ForeignInstitutionJointBuying": "외국인·기관 동반 매수",
    "RetailSupplyAbsorbedByInformedFlow": "개인 매물을 정보성 매수세가 흡수",
    "OrderFlowPriceConfirmation": "수급과 가격이 같은 방향",
    "SuspectedSmartMoneyAccumulation": "스마트머니 매집 의심",
    "NpuCompositeMomentum": "상승 모멘텀 포착",
    "LiquiditySupport": "충분한 유동성 확보",
    "RevenueGrowth": "매출 성장",
    "EarningsGrowth": "이익 성장",
    "ProfitabilityQuality": "수익성 양호",
    "FreshBrokerQuote": "실시간 시세 신선",
    "CashFitOneShare": "매수 가능 현금 확보",
    "ExecutableBuyCandidate": "실시간 실행 조건 충족",
    "RealtimeAdaptiveFallbackBuyCandidate": "실시간 적응형 매수 조건 충족",
    "RuntimeProbeBuyCandidate": "소량 탐색 매수 조건 충족",
}
_SELL_PROFIT_BASES = {
    "take_profit_amount": "목표 이익금액 달성",
    "quick_take_profit": "빠른 목표수익 도달",
    "profit_lock": "고점 대비 이익 반납 방지(이익 잠금)",
    "profit_time_exit": "보유시간 경과 후 순이익 실현",
    "profit_exit": "목표 수익 도달",
}
_SELL_LOSS_BASES = {
    "stop_loss": "손절 — 손실 최소화",
    "hard_stop_loss": "하드 손절 — 자본 보호",
    "domestic_emergency_exit": "긴급 청산",
    "loss_exit": "손실 청산",
    "domestic_drawdown_reduce": "낙폭 확대로 비중 축소",
    "domestic_concentration_reduce": "집중도 과다로 비중 축소",
    "trailing_exit": "추적 손절",
    "invalid_signal_exit": "매수 근거 약화(신호 무효)",
}
_SELL_NEUTRAL_BASES = {
    "time_exit": "보유시간 만료 청산",
    "profit_exit_ontology": "리스크 신호로 청산",
}
_HOLD_REASON_TEXT = {
    "HOLD_BELOW_PROFIT_TARGET": "아직 목표 수익 미달 → 보유",
    "WIDE_SPREAD": "호가 스프레드가 넓어 매수 보류",
    "LOW_LIQUIDITY": "유동성 부족으로 보류",
    "INSUFFICIENT_CASH_FOR_ONE_SHARE": "1주 매수 현금 부족",
    "FALLBACK_SCORE_BELOW_THRESHOLD": "매수 점수 기준 미달",
    "ONTOLOGY_REQUIRED_FOR_MODEL_FALLBACK": "근거 확인 부족으로 보류",
    "MODEL_AUXILIARY_ONLY_NEEDS_CONFIRMATION": "모델 단독 매수 불가(근거 필요)",
    "MODEL_FEATURE_UNAVAILABLE": "실시간 데이터 부족으로 판단 보류",
    "SELL_BELOW_BREAK_EVEN_BLOCKED": "손실 매도 방지(본전 미만)",
    "SMALL_ACCOUNT_ONE_SHARE_LOSS_BLOCK": "소액계좌 보호(1주 손실매도 차단)",
    "HOLD_UNPROFITABLE_ONTOLOGY_SELL_BLOCKED": "손실권 매도 보류",
    "LOSS_EXIT_DISABLED": "손실 청산 비활성",
    "MARKET_SESSION_CLOSED": "장 마감",
    "MISSING_MARKET_DATA": "시세 없음",
    "open_sell_kept": "기존 매도 주문 유지(중복 방지)",
    "BELOW_TARGET_NET_RETURN_AFTER_COST": "비용 차감 후 목표 순수익 미달",
    "BELOW_BREAK_EVEN_WITH_MARGIN": "본전(마진 포함) 미만 예상 → 매수 보류",
    "COST_BURDEN_HIGH": "거래비용 부담 과다",
    "SPREAD_TOO_WIDE": "호가 스프레드가 넓어 매수 보류",
    "SPREAD_CONSUMES_ALPHA": "스프레드가 기대수익을 잠식",
    "LIQUIDITY_TOO_LOW": "유동성 부족으로 보류",
    "SLIPPAGE_RISK_HIGH": "슬리피지 위험 과다",
    "PROFITABILITY_GATE_REJECTED": "수익성 게이트 거부(순기대수익 부족)",
    "RECENT_LOSS_SYMBOL_COOLDOWN": "최근 손실 종목 재매수 대기",
    "NO_SELLABLE_QUANTITY": "매도 가능 수량 없음",
    "OPEN_ORDER_OR_SETTLEMENT_LOCK": "미체결 주문/결제 잠금",
}


def _humanize_reason(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    base, _sep, detail = text.partition(":")
    base = base.strip()
    detail = detail.strip()
    if base == "VolumeSurge":
        return f"거래량 급증 {detail}".strip()
    if base in _BUY_REASON_TEXT:
        return _BUY_REASON_TEXT[base]
    for table in (_SELL_PROFIT_BASES, _SELL_LOSS_BASES, _SELL_NEUTRAL_BASES):
        if base in table:
            return f"{table[base]} ({detail})" if detail else table[base]
    if base in _HOLD_REASON_TEXT:
        return _HOLD_REASON_TEXT[base]
    return text.replace("_", " ")


def _reason_tone(kind: str, outcome: str, bases: list[str]) -> str:
    if outcome in ("blocked", "error"):
        return "warn"
    if kind == "BUY":
        return "buy"
    if kind == "SELL":
        for base in bases:
            if base in _SELL_PROFIT_BASES:
                return "profit"
        for base in bases:
            if base in _SELL_LOSS_BASES:
                return "loss"
        return "sell"
    return "hold"


def _time_ago_ko(at_iso: str, now: datetime) -> str:
    try:
        moment = datetime.fromisoformat(str(at_iso).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return ""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    seconds = max(0.0, (now - moment).total_seconds())
    if seconds < 60:
        return f"{int(seconds)}초 전"
    if seconds < 3600:
        return f"{int(seconds // 60)}분 전"
    if seconds < 86400:
        return f"{int(seconds // 3600)}시간 전"
    return f"{int(seconds // 86400)}일 전"


def _kst_hm(at_iso: str) -> str:
    try:
        moment = datetime.fromisoformat(str(at_iso).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return ""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return (moment + timedelta(hours=9)).strftime("%H:%M")


def _is_us_symbol_market(symbol: str, market: str) -> bool:
    s = str(symbol or "").strip()
    if s.isdigit() and len(s) == 6:
        return False
    return bool(str(market or "").strip()) or not s.isdigit()


def _format_price_display(price: Any, is_us: bool) -> str:
    try:
        value = float(price)
    except (TypeError, ValueError):
        return ""
    if value <= 0:
        return ""
    return f"${value:,.2f}" if is_us else f"{value:,.0f}원"


def _kiosk_orderable_cash() -> dict[str, Any]:
    """Orderable cash by currency for the Pi kiosk / mobile overview.

    Reuses the same cached KIS live-account basis the account dashboard reads.
    This endpoint is polled frequently, so it must not trigger broker account
    refreshes itself; background/account-dashboard refresh paths keep the cache
    warm.
    """
    basis = _last_live_account_basis()
    if basis is None:
        return {"available": False}
    orderable = {
        str(code).upper(): _number_or_zero(amount)
        for code, amount in dict(basis.get("orderable_cash_by_currency") or {}).items()
    }
    krw = _number_or_zero(orderable.get("KRW", basis.get("krw_cash")))
    foreign_native = {
        code: amount for code, amount in orderable.items() if code != "KRW" and amount > 0
    }
    # Primary foreign currency = the largest non-KRW orderable balance (USD for KR+US).
    primary_ccy = max(foreign_native, key=foreign_native.get) if foreign_native else "USD"
    return {
        "available": True,
        "krw": krw,
        "foreign_currency": primary_ccy,
        "foreign_native": foreign_native.get(primary_ccy, 0.0),
        "foreign_krw": _number_or_zero(basis.get("foreign_cash_krw")),
        "by_currency": orderable,
    }


def _kiosk_market_overview(now: datetime | None = None) -> dict[str, Any]:
    from datetime import time as _time
    from zoneinfo import ZoneInfo

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    seoul = current.astimezone(ZoneInfo("Asia/Seoul"))
    eastern = current.astimezone(ZoneInfo("America/New_York"))

    krx_core = _is_live_market_core_open("KRX", current)
    krx_ext = _is_live_market_extended_open("KRX", current)
    us_core = _is_live_market_core_open("US", current)
    us_ext = _is_live_market_extended_open("US", current)
    us_day = (
        seoul.weekday() < 5
        and _time(9, 0) <= seoul.time() <= _time(16, 50)
        and not us_core
        and not (_time(4, 0) <= eastern.time() < _time(9, 30))
        and not (_time(16, 0) < eastern.time() <= _time(20, 0))
    )

    if krx_core:
        primary = ("국내 정규장", "KRX 매매 가능", "open")
    elif us_core:
        primary = ("미국 정규장", "해외 매매 가능", "open")
    elif eastern.weekday() < 5 and not _is_us_market_holiday(eastern.date()) and _time(4, 0) <= eastern.time() < _time(9, 30):
        primary = ("미국 프리마켓", "프장 감시 중", "pre")
    elif eastern.weekday() < 5 and not _is_us_market_holiday(eastern.date()) and _time(16, 0) < eastern.time() <= _time(20, 0):
        primary = ("미국 애프터마켓", "시간외 감시 중", "after")
    elif us_day:
        primary = ("해외 데이마켓", "KIS 주간주문 시간", "day")
    elif krx_ext:
        primary = ("국내 시간외/장전", "국내 예약·시간외 구간", "pre")
    else:
        primary = ("양쪽 휴장", "거래보다 분석 대기", "closed")

    if krx_core:
        krx_label, krx_detail, krx_tone = "국내 장 오픈", "09:00-15:30", "open"
    elif krx_ext and seoul.weekday() < 5 and _time(8, 30) <= seoul.time() < _time(9, 0):
        krx_label, krx_detail, krx_tone = "국내 장전", "08:30-09:00", "pre"
    elif krx_ext:
        krx_label, krx_detail, krx_tone = "국내 시간외", "15:30 이후", "after"
    elif seoul.weekday() < 5 and seoul.time() < _time(8, 30):
        krx_label, krx_detail, krx_tone = "국내 장전 대기", "08:30부터 감시", "idle"
    else:
        krx_label, krx_detail, krx_tone = "국내 휴장", "KRX 대기", "closed"

    if us_core:
        us_label, us_detail, us_tone = "미국 장 오픈", "09:30-16:00 ET", "open"
    elif eastern.weekday() < 5 and not _is_us_market_holiday(eastern.date()) and _time(4, 0) <= eastern.time() < _time(9, 30):
        us_label, us_detail, us_tone = "미국 프리마켓", "04:00-09:30 ET", "pre"
    elif eastern.weekday() < 5 and not _is_us_market_holiday(eastern.date()) and _time(16, 0) < eastern.time() <= _time(20, 0):
        us_label, us_detail, us_tone = "미국 애프터", "16:00-20:00 ET", "after"
    elif us_day:
        us_label, us_detail, us_tone = "해외 데이마켓", "09:00-16:50 KST", "day"
    else:
        us_label, us_detail, us_tone = "미국 휴장", "해외 대기", "closed"

    snap = _live_snapshot()
    progress = snap.get("progress") or {}
    learning = snap.get("learning") or {}
    collection_log = snap.get("collection_log") or []
    latest_log = collection_log[-1] if collection_log else {}
    progress_active = bool(progress.get("active") or snap.get("is_refreshing"))
    if progress_active:
        work_label = "뉴스·데이터 분석 중"
        work_detail = str(progress.get("message") or "실시간 수집/분석 진행 중")
        work_tone = "busy"
    elif learning.get("active"):
        work_label = "뉴스 분석 예약"
        work_detail = "다음 수집 대기" if learning.get("next_collection_at") else "백그라운드 학습 대기"
        work_tone = "idle"
    elif latest_log:
        work_label = "최근 분석 완료"
        work_detail = str(latest_log.get("message") or latest_log.get("status") or "대기")
        work_tone = "idle"
        log_message = work_detail.lower()
        log_status = str(latest_log.get("status") or "").lower()
        with _live_lock:
            collector_running = (
                _kis_realtime_collector_worker is not None
                and _kis_realtime_collector_worker.is_alive()
                and not _kis_realtime_collector_stop.is_set()
            )
        if collector_running and "kis realtime collector" in log_message:
            if log_status == "reconnecting":
                work_label = "실시간 시세 재연결 중"
                work_tone = "busy"
            elif log_status in {"running", "complete", "market_closed"}:
                work_label = "실시간 시세 수집 중"
                work_tone = "busy" if log_status == "running" else "idle"
    else:
        work_label = "뉴스 분석 대기"
        work_detail = "새 수집 사이클 대기"
        work_tone = "idle"

    return {
        "primary": {"label": primary[0], "detail": primary[1], "tone": primary[2]},
        "markets": [
            {"key": "KRX", "label": krx_label, "detail": krx_detail, "tone": krx_tone},
            {"key": "US", "label": us_label, "detail": us_detail, "tone": us_tone},
        ],
        "work": {"label": work_label, "detail": work_detail[:90], "tone": work_tone},
        "times": {
            "kst": seoul.strftime("%H:%M"),
            "et": eastern.strftime("%H:%M"),
        },
    }


def _trade_explanation_cards(limit: int = 14) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    with _realtime_trading_lock:
        engine = _realtime_trading_engine
        running = _realtime_trading_worker is not None and _realtime_trading_worker.is_alive()
    if engine is None:
        return {
            "generated_at": now.isoformat(),
            "running": running,
            "buy_enabled": None,
            "overview": _kiosk_market_overview(now),
            "orderable_cash": _kiosk_orderable_cash(),
            "cards": [],
        }
    status = engine.get_status()
    diagnostics = engine.decision_engine.get_diagnostics() if hasattr(engine, "decision_engine") else None
    events = status.get("recent_events") or []
    summary = status.get("last_summary") or {}
    keep_outcomes = {"submitted", "amended", "filled", "partially_filled", "blocked", "error", "open_sell_kept"}
    action_label = {
        "submitted": "주문", "amended": "정정", "filled": "체결", "partially_filled": "일부 체결",
        "blocked": "차단됨", "error": "오류", "open_sell_kept": "주문 유지",
    }
    cards: list[dict[str, Any]] = []
    for event in events:
        kind = str(event.get("kind") or "").upper()
        outcome = str(event.get("outcome") or "")
        if kind not in ("BUY", "SELL") or outcome not in keep_outcomes:
            continue
        symbol = str(event.get("symbol") or "")
        market = str(event.get("market") or "")
        is_us = _is_us_symbol_market(symbol, market)
        raw_reason = str(event.get("reason") or event.get("detail") or "")
        parts = [p for p in raw_reason.split(";") if p.strip()]
        bases = [p.partition(":")[0].strip() for p in parts]
        reasons = []
        for part in parts:
            phrase = _humanize_reason(part)
            if phrase and phrase not in reasons:
                reasons.append(phrase)
        reasons = reasons[:4]
        price = event.get("limit_price") if event.get("limit_price") else event.get("average_fill_price")
        price_txt = _format_price_display(price, is_us)
        try:
            quantity = int(event.get("quantity") or 0)
        except (TypeError, ValueError):
            quantity = 0
        name = _resolve_instrument_label(symbol) if symbol else symbol
        verb = "매수" if kind == "BUY" else "매도"
        act = action_label.get(outcome, outcome)
        bits = [b for b in [f"{quantity}주" if quantity else "", price_txt] if b]
        headline = f"{name} · {' · '.join(bits)} {verb} {act}".strip() if bits else f"{name} {verb} {act}"
        cards.append(
            {
                "kind": kind,
                "outcome": outcome,
                "tone": _reason_tone(kind, outcome, bases),
                "symbol": symbol,
                "name": name,
                "quantity": quantity,
                "price_text": price_txt,
                "verb": verb,
                "action": act,
                "headline": headline,
                "reasons": reasons or [_humanize_reason(raw_reason)] if raw_reason else reasons,
                "time_ago": _time_ago_ko(event.get("at"), now),
                "time_hm": _kst_hm(event.get("at")),
            }
        )
        if len(cards) >= limit:
            break
    if len(cards) < limit:
        cards.extend(_recent_evaluation_cards(summary, now, limit - len(cards)))
    if len(cards) < limit:
        diag_card = _diagnostic_evaluation_card(diagnostics, now)
        if diag_card is not None:
            cards.append(diag_card)
    if not cards:
        cards.append(_engine_heartbeat_card(status, now, running))
    activity = _realtime_activity_payload(status, summary, diagnostics)
    return {
        "generated_at": now.isoformat(),
        "running": running,
        "buy_enabled": bool(status.get("buy_enabled")),
        "last_reason": status.get("last_reason"),
        "activity": activity,
        "overview": _kiosk_market_overview(now),
        "orderable_cash": _kiosk_orderable_cash(),
        "cards": cards,
    }


def _realtime_activity_payload(
    status: dict[str, Any],
    summary: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    diagnostics = diagnostics or {}
    policy_state = diagnostics.get("policy_state") if isinstance(diagnostics.get("policy_state"), dict) else {}
    model_health = policy_state.get("model_health") if isinstance(policy_state.get("model_health"), dict) else {}
    profitability = diagnostics.get("profitability_decision") if isinstance(diagnostics.get("profitability_decision"), dict) else {}
    return {
        "cycle": status.get("cycles"),
        "last_cycle_at": status.get("last_cycle_at"),
        "reason": summary.get("reason") or status.get("last_reason"),
        "buy_evaluated": summary.get("buy_evaluated", 0),
        "buy_rejected": summary.get("buy_rejected", 0),
        "sell_evaluated": summary.get("sell_evaluated", 0),
        "sell_rejected": summary.get("sell_rejected", 0),
        "submitted": summary.get("submitted", 0),
        "blocked": summary.get("blocked", 0),
        "errors": summary.get("errors", 0),
        "skipped_ignored": summary.get("skipped_ignored", 0),
        "ignored_symbols": list(summary.get("ignored_symbols") or ()),
        "current_symbol": policy_state.get("symbol") or profitability.get("symbol"),
        "current_action": profitability.get("action"),
        "model_status": model_health.get("status"),
        "quote_refresh_status": diagnostics.get("quote_refresh_status"),
    }


def _recent_evaluation_cards(summary: dict[str, Any], now: datetime, remaining: int) -> list[dict[str, Any]]:
    if remaining <= 0:
        return []
    at = str(summary.get("at") or now.isoformat())
    cards: list[dict[str, Any]] = []
    ignored = [str(symbol) for symbol in (summary.get("ignored_symbols") or ()) if str(symbol)]
    skipped_ignored = int(summary.get("skipped_ignored") or 0)
    if ignored and skipped_ignored:
        cards.append(
            {
                "kind": "CYCLE",
                "outcome": "ignored",
                "tone": "hold",
                "symbol": ", ".join(ignored[:4]),
                "name": ", ".join(ignored[:4]),
                "headline": f"{', '.join(ignored[:4])} ignored by realtime trading",
                "reasons": ["REALTIME_IGNORE_SYMBOLS", f"skipped {skipped_ignored} locked/special symbol"],
                "time_ago": _time_ago_ko(at, now),
                "time_hm": _kst_hm(at),
            }
        )
    for row in (summary.get("rejections") or []):
        if len(cards) >= remaining:
            break
        symbol = str(row.get("symbol") or "")
        side = str(row.get("side") or "").upper()
        codes = [str(code) for code in (row.get("reason_codes") or ()) if str(code)]
        primary = _humanize_reason(codes[0]) if codes else "evaluation held"
        reasons = []
        for code in codes[:4]:
            text = _humanize_reason(code)
            if text and text not in reasons:
                reasons.append(text)
        name = _resolve_instrument_label(symbol) if symbol else symbol
        verb = "BUY" if side == "BUY" else "SELL" if side == "SELL" else "CHECK"
        headline = f"{name or symbol} {verb} evaluated, no order"
        cards.append(
            {
                "kind": side or "EVAL",
                "outcome": "held",
                "tone": "hold",
                "symbol": symbol,
                "name": name or symbol,
                "headline": headline,
                "reasons": reasons or [primary],
                "time_ago": _time_ago_ko(at, now),
                "time_hm": _kst_hm(at),
            }
        )
    if not cards and any(int(summary.get(key) or 0) for key in ("buy_evaluated", "sell_evaluated")):
        cards.append(
            {
                "kind": "CYCLE",
                "outcome": "evaluated",
                "tone": "hold",
                "symbol": "",
                "name": "Realtime cycle",
                "headline": "Realtime cycle completed with no order",
                "reasons": [
                    f"buy evaluated {int(summary.get('buy_evaluated') or 0)} / rejected {int(summary.get('buy_rejected') or 0)}",
                    f"sell evaluated {int(summary.get('sell_evaluated') or 0)} / rejected {int(summary.get('sell_rejected') or 0)}",
                ],
                "time_ago": _time_ago_ko(at, now),
                "time_hm": _kst_hm(at),
            }
        )
    return cards[:remaining]


def _diagnostic_evaluation_card(diagnostics: dict[str, Any] | None, now: datetime) -> dict[str, Any] | None:
    from zoneinfo import ZoneInfo

    if not isinstance(diagnostics, dict) or not diagnostics:
        return None
    policy_state = diagnostics.get("policy_state") if isinstance(diagnostics.get("policy_state"), dict) else {}
    profitability = diagnostics.get("profitability_decision") if isinstance(diagnostics.get("profitability_decision"), dict) else {}
    model_health = policy_state.get("model_health") if isinstance(policy_state.get("model_health"), dict) else {}
    symbol = str(policy_state.get("symbol") or profitability.get("symbol") or "")
    if not symbol:
        return None
    action = str(profitability.get("action") or "CHECK").upper()
    allowed = bool(profitability.get("allowed"))
    reason_codes = []
    for code in profitability.get("rejection_reasons") or ():
        reason_codes.append(str(code))
    for code in model_health.get("reason_codes") or ():
        reason_codes.append(str(code))
    reasons = []
    for code in reason_codes[:4]:
        text = _humanize_reason(code)
        if text and text not in reasons:
            reasons.append(text)
    if diagnostics.get("quote_refresh_status"):
        reasons.append(f"quote {diagnostics.get('quote_refresh_status')}")
    name = _resolve_instrument_label(symbol)
    return {
        "kind": action,
        "outcome": "checking" if allowed else "held",
        "tone": "buy" if allowed and action == "BUY" else "hold",
        "symbol": symbol,
        "name": name or symbol,
        "headline": f"{name or symbol} {action} checking now",
        "reasons": reasons or ["decision diagnostics active"],
        "time_ago": "",
        "time_hm": now.astimezone(ZoneInfo("Asia/Seoul")).strftime("%H:%M"),
    }


def _engine_heartbeat_card(status: dict[str, Any], now: datetime, running: bool) -> dict[str, Any]:
    from zoneinfo import ZoneInfo

    cycles = int(status.get("cycles") or 0)
    last_cycle = status.get("last_cycle_at")
    headline = "Realtime trading engine is running" if running else "Realtime trading engine is stopped"
    reasons = [
        f"cycles completed {cycles}",
        "waiting for first completed decision cycle" if running and not last_cycle else f"last cycle {last_cycle}",
    ]
    if status.get("buy_enabled") is False:
        reasons.append(f"buy disabled: {status.get('buy_disabled_reason') or 'configured'}")
    return {
        "kind": "CYCLE",
        "outcome": "heartbeat" if running else "stopped",
        "tone": "hold",
        "symbol": "",
        "name": "Realtime engine",
        "headline": headline,
        "reasons": reasons,
        "time_ago": "",
        "time_hm": now.astimezone(ZoneInfo("Asia/Seoul")).strftime("%H:%M"),
    }


@app.get("/api/trade-explanations")
def trade_explanations() -> JSONResponse:
    return _json(_trade_explanation_cards())


@app.get("/api/realtime-trading/status")
def realtime_trading_status() -> JSONResponse:
  with _realtime_trading_lock:
    engine = _realtime_trading_engine
    running = _realtime_trading_worker is not None and _realtime_trading_worker.is_alive()
  diagnostics = engine.decision_engine.get_diagnostics() if engine is not None and hasattr(engine, "decision_engine") else None
  engine_status = engine.get_status() if engine is not None else None
  return _json(
    {
      "ok": True,
      "running": running,
      "auto_start": AUTO_START_REALTIME_TRADING,
      "status": engine_status,
      "buy_enabled": engine_status.get("buy_enabled") if isinstance(engine_status, dict) else None,
      "buy_disabled_reason": engine_status.get("buy_disabled_reason") if isinstance(engine_status, dict) else None,
      "liquidation_requested": engine_status.get("liquidation_requested") if isinstance(engine_status, dict) else None,
      "liquidation_reason": engine_status.get("liquidation_reason") if isinstance(engine_status, dict) else None,
      "buy_warmup_pending": list(_pending_krx_buy_candidate_warmup_symbols()),
      "decision_diagnostics": diagnostics,
    }
  )


def _buy_candidate_warmup_detail() -> dict:
  """How close is the best streaming symbol to the minute-bar requirement?

  ``_candidate_has_strategy_feature_history`` needs N completed bars. After a
  restart, or in the first minutes of a session, every streaming symbol fails it
  while looking perfectly healthy at the tick level — a symbol can have many
  ticks in ten minutes while still lacking the required completed bars.
  Reporting the shortfall and an ETA turns that from a mystery into a wait.
  """
  try:
    required = max(
        10, _auto_reliability_int("REALTIME_STRATEGY_MINUTE_HISTORY_BARS", 20, 10)
    )
    store = RealtimeMarketDataStore()
    now = datetime.now(timezone.utc)
    since = now - timedelta(seconds=120)
    best_symbol, best_bars = "", 0
    for symbol in tuple(store.active_symbols(since, limit=32)):
      try:
        bars = store.recent_minute_bars(
            symbol, now - timedelta(minutes=max(120, required * 3)), limit=max(120, required)
        )
      except Exception:  # noqa: BLE001 - one unreadable symbol must not hide the rest.
        continue
      if len(bars) > best_bars:
        best_symbol, best_bars = str(symbol), len(bars)
    # Affordability is checked alongside warm-up because it is the other reason a
    # perfectly healthy, fully warmed-up universe still yields zero candidates —
    # and the two are indistinguishable from a bare "0 candidates". Measured case:
    # KRW orderable 0원 with 073240 at 7,220원, while the only funded currency
    # (USD 67.57) belonged to a closed market.
    cheapest_symbol, cheapest_ask = "", 0.0
    for symbol in tuple(store.active_symbols(since, limit=32)):
      try:
        book = store.latest_orderbook(symbol)
      except Exception:  # noqa: BLE001
        continue
      ask = float(getattr(book, "best_ask", 0.0) or 0.0) if book is not None else 0.0
      if ask > 0 and (cheapest_ask <= 0 or ask < cheapest_ask):
        cheapest_symbol, cheapest_ask = str(symbol), ask
    # The engine's own account view, so the panel reports the same number the
    # affordability filter actually applied.
    try:
      account = _realtime_engine_account_snapshot()
      krw_orderable = _account_available_cash(account, "KRW") if account else 0.0
    except Exception:  # noqa: BLE001 - cash lookup is diagnostic only.
      krw_orderable = 0.0
    unaffordable = bool(cheapest_ask > 0 and krw_orderable < cheapest_ask)
    return {
        "warming_up": best_bars < required,
        "best_symbol": best_symbol,
        "best_bars": best_bars,
        "required_bars": required,
        "eta_minutes": max(0, required - best_bars),
        "krw_orderable": krw_orderable,
        "cheapest_candidate": cheapest_symbol,
        "cheapest_ask": cheapest_ask,
        "unaffordable": unaffordable,
    }
  except Exception:  # noqa: BLE001 - diagnostics must never raise.
    return {}


def _entry_blockade_chain() -> list[dict]:
  """Ordered "why is nothing trading" chain, first unmet link wins.

  Built because the operator-visible answer used to be a single reason code from
  whichever layer happened to fail last — for 11,614 consecutive cycles that was
  ``NO_POSITIVE_NET_GNN_EDGE``, which named the GNN while the actual constraint
  was that no scanned market was in its regular session. Each link reports its own
  verdict so the real blocker is identifiable at a glance.
  """
  from app.data.market_session import new_entry_session_report

  chain: list[dict] = []

  def _link(stage: str, ok: bool, detail: str, data: dict | None = None) -> None:
    chain.append({"stage": stage, "ok": bool(ok), "detail": detail, "data": data or {}})

  with _realtime_trading_lock:
    engine = _realtime_trading_engine
    running = _realtime_trading_worker is not None and _realtime_trading_worker.is_alive()
  _link("engine_running", running, "실시간 엔진 스레드" if running else "엔진이 실행 중이 아님")

  status = engine.get_status() if engine is not None else {}
  summary = (status or {}).get("last_summary") or {}
  session = summary.get("strategy_session") or {}

  # A cycle that returns early on MARKET_SESSION_CLOSED never reaches the line
  # that stamps live_armed, so a missing key means "not evaluated this cycle",
  # not "disarmed". Reading the absence as False made this chain report the
  # arming layer every night and weekend while the real constraint was the
  # market session — precisely the misattribution the chain exists to prevent.
  # Absent falls back to the engine's standing buy_enabled flag, same as the
  # terminal's live_trading view does.
  armed_value = summary.get("live_armed")
  armed = bool(status.get("buy_enabled")) if armed_value is None else bool(armed_value)
  disabled_reason = str(status.get("buy_disabled_reason") or "").strip()
  _link(
      "live_armed",
      armed,
      "라이브 제출 무장됨"
      if armed
      else f"라이브 제출이 무장되지 않음{f' ({disabled_reason})' if disabled_reason else ''}",
      {"evaluated_this_cycle": armed_value is not None, "buy_disabled_reason": disabled_reason or None},
  )

  sessions = new_entry_session_report()
  candidates = tuple(summary.get("buy_candidate_sample") or ())
  groups = {"KRX" if str(s).isdigit() and len(str(s)) == 6 else "US" for s in candidates}
  scanned = {g: sessions["groups"].get(g, {}) for g in groups} or sessions["groups"]
  # The cycle's own early-return verdict is authoritative.  Diagnostics may be
  # requested after a session boundary, at which point recomputing from the
  # wall clock can incorrectly rewrite a closed-market cycle as open.
  session_ok = (
      False
      if str(summary.get("reason") or "") == "MARKET_SESSION_CLOSED"
      else any(item.get("allows_new_entry") for item in scanned.values())
  )
  _link(
    "market_session",
    session_ok,
    "정규장 진행 중"
    if session_ok
    else "스캔 중인 시장이 정규장이 아님 — 신규 진입 보류(청산은 계속 동작)",
    {"scanned_groups": scanned, "all_groups": sessions["groups"],
     "extended_hours_entry_enabled": sessions["extended_hours_entry_enabled"]},
  )

  count = int(summary.get("buy_candidate_count") or 0)
  # A bare "0 candidates" is not actionable. The overwhelmingly common cause is
  # minute-bar warm-up: a candidate needs REALTIME_STRATEGY_MINUTE_HISTORY_BARS
  # completed bars, which after a restart (or right after the open) simply have
  # not accrued yet. Distinguishing "warming up" from "nothing qualifies" is the
  # difference between waiting and debugging.
  warmup = _buy_candidate_warmup_detail() if count == 0 else {}
  detail = f"매수 후보 {count}개"
  if count == 0 and warmup.get("warming_up"):
    detail = (
      f"후보 0개 — 분봉 워밍업 중 "
      f"({warmup['best_symbol']} {warmup['best_bars']}/{warmup['required_bars']}개, "
      f"약 {warmup['eta_minutes']}분 후 충족 예상)"
    )
  elif count == 0 and warmup.get("unaffordable"):
    detail = (
      f"후보 0개 — 주문가능 현금 부족 "
      f"(KRW {warmup['krw_orderable']:,.0f} < 최저가 {warmup['cheapest_candidate']} "
      f"{warmup['cheapest_ask']:,.0f})"
    )
  elif count == 0:
    detail = "후보 0개 — 스트리밍 종목 없음 또는 전 종목이 후보 필터에서 제외됨"
  _link("buy_candidates", count > 0, detail, {"sample": list(candidates), **warmup})

  diagnostics = list(session.get("candidate_diagnostics") or ())
  actionable = [
    item for item in diagnostics
    if str(item.get("selected_strategy") or "").lower() not in {"", "hold", "sell", "reduce_risk"}
  ]
  _link(
    "micro_buy_intents",
    bool(actionable),
    f"실행 가능한 마이크로 전략 {len(actionable)}/{len(diagnostics)}"
    if diagnostics
    else "마이크로 결과 없음",
    {"blocking_reason_codes": sorted(
      {code for item in diagnostics for code in (item.get("reason_codes") or ())}
    )[:12]},
  )

  # The bandit fields exist only on the post-refactor build, so their absence
  # from the ENGINE is the answer. Their absence from the last cycle summary is
  # not: a cycle that returned early (closed market) publishes no session block
  # at all, and reading that as "old code — restart the server" sent the
  # operator to restart a healthy process every time the market was shut.
  if "bandit_selected_arm" not in session:
    session = (status or {}).get("strategy_session") or session
  bandit_present = "bandit_selected_arm" in session
  if bandit_present:
    arm = session.get("bandit_selected_arm")
    picked = bool(arm) and arm != "no_trade"
    _link(
      "strategy_election",
      picked,
      f"선택된 arm: {arm}" if picked else "보수적 하단값이 양수인 전략 없음 → NO_TRADE",
      {"conservative_edge_bps": session.get("bandit_conservative_edge_bps"),
       "is_exploration": session.get("bandit_is_exploration"),
       "reason_codes": session.get("bandit_reason_codes"),
       "evaluations": session.get("bandit_evaluations")},
    )
  else:
    _link(
      "strategy_election",
      False,
      "구 코드가 실행 중입니다(보수적 bandit 필드 없음) — 서버 재시작 필요",
      {"session_last_reason": session.get("last_reason")},
    )

  phase = str(session.get("phase") or "")
  _link(
    "position",
    phase in {"ARMED", "ENTERING", "OWNED", "EXITING"},
    f"세션 단계 {phase or 'UNKNOWN'}",
    {"last_reason": session.get("last_reason"),
     "session_phases": session.get("session_phases")},
  )

  return chain


@app.get("/api/investor-flow/status")
def investor_flow_status() -> JSONResponse:
  """Coverage and last-refresh state of the daily investor-flow store.

  Exposed because a quietly stalled refresher is indistinguishable from a working
  one until ``residual_relative_strength`` has silently stopped being evaluable —
  the exact failure mode this whole feature was added to fix.
  """
  from app.data.investor_flow_store import InvestorFlowStore

  try:
    coverage = InvestorFlowStore().coverage()
  except Exception as exc:  # noqa: BLE001 - diagnostics must never 500.
    coverage = {"error": f"{type(exc).__name__}: {exc}"}
  with _live_lock:
    status = dict(_investor_flow_refresh_status)
  return _json(
    {
      "ok": True,
      "enabled": AUTO_START_INVESTOR_FLOW_REFRESH,
      "refresh_interval_seconds": _env_float_web(
        "INVESTOR_FLOW_REFRESH_SECONDS", 21_600.0
      ),
      "coverage": coverage,
      **status,
    }
  )


@app.post("/api/investor-flow/refresh")
def investor_flow_refresh() -> JSONResponse:
  """Run the investor-flow refresh now (read-only against the broker)."""
  try:
    payload = _refresh_investor_flow_once()
  except Exception as exc:  # noqa: BLE001 - report the failure, never 500 the UI.
    return _json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
  return _json({"ok": True, "result": payload})


@app.get("/api/weekend-brief")
def weekend_brief_status() -> JSONResponse:
  """The current Monday-open prior and the prior's historical accuracy.

  The track record is the point. A weekend analysis nobody grades is
  indistinguishable from a wrong one, so accuracy is reported next to the claim.
  """
  from app.research.weekend_brief import WeekendBriefStore, weekend_window

  try:
    store = WeekendBriefStore()
    latest = store.latest_prior()
    record = store.track_record()
  except Exception as exc:  # noqa: BLE001 - diagnostics must never 500.
    return _json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
  window = weekend_window(datetime.now(timezone.utc))
  with _live_lock:
    status = dict(_weekend_brief_status)
  return _json(
    {
      "ok": True,
      "enabled": AUTO_START_WEEKEND_BRIEF,
      "in_weekend_window": window is not None,
      "window_key": window.key if window else None,
      "latest_prior": latest,
      "track_record": record,
      **status,
    }
  )


@app.post("/api/weekend-brief/refresh")
def weekend_brief_refresh() -> JSONResponse:
  """Run the weekend research pass now (read-only against market data)."""
  try:
    return _json({"ok": True, "result": _run_weekend_brief_once()})
  except Exception as exc:  # noqa: BLE001
    return _json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


@app.get("/api/realtime-trading/entry-blockade")
def realtime_trading_entry_blockade() -> JSONResponse:
  """Single-call answer to "왜 아직 거래가 없나"."""
  try:
    chain = _entry_blockade_chain()
  except Exception as exc:  # noqa: BLE001 - diagnostics must never 500 the dashboard.
    return _json({"ok": False, "error": f"{type(exc).__name__}: {exc}", "chain": []})
  blocker = next((link for link in chain if not link["ok"]), None)
  return _json(
    {
      "ok": True,
      "trading_possible": blocker is None,
      "blocking_stage": blocker["stage"] if blocker else None,
      "blocking_detail": blocker["detail"] if blocker else None,
      "chain": chain,
    }
  )


def _schedule_app_process_shutdown(delay_seconds: float = 1.5) -> None:
    """Exit the process, stopping the trading engine and feeds on the way out.

    ``os._exit`` skips atexit handlers AND the FastAPI shutdown event, so the
    previous version killed the process with the realtime engine still running.
    The teardown is therefore invoked explicitly here rather than relying on an
    exit hook that this code path never reaches.
    """

    def _shutdown_later() -> None:
        time.sleep(max(0.2, delay_seconds))
        try:
            stopped = _graceful_teardown()
            audit.record("app_process_shutdown", {"stopped": stopped})
        except Exception as exc:  # noqa: BLE001 - never block the exit itself
            audit.record(
                "app_process_shutdown_teardown_failed",
                {"error": str(exc) or exc.__class__.__name__},
            )
        os._exit(0)

    threading.Thread(target=_shutdown_later, name="ui-requested-shutdown", daemon=True).start()


def _restart_safety_report() -> dict[str, Any]:
    """Is it safe to stop this process right now?

    A restart discards in-memory exit state: the armed stop, target, trailing
    high-watermark and holding clock for an open position. The broker keeps the
    position but nothing is left watching it, so restarting while holding is how a
    managed trade silently becomes an unmanaged one.

    Fails CLOSED. If the account cannot be read, the answer is "unsafe", not
    "probably fine" — an unreadable account is exactly when you least want to
    assume there is no position.
    """
    reasons: list[str] = []
    holdings_count: int | None = None
    positions: list[str] = []
    try:
        account = _realtime_engine_account_snapshot()
    except Exception as exc:  # noqa: BLE001
        account = None
        reasons.append(f"ACCOUNT_SNAPSHOT_UNAVAILABLE:{type(exc).__name__}")
    if account is None:
        if "ACCOUNT_SNAPSHOT_UNAVAILABLE" not in " ".join(reasons):
            reasons.append("ACCOUNT_SNAPSHOT_UNAVAILABLE")
    else:
        holdings = tuple(getattr(account, "holdings", ()) or ())
        holdings_count = len(holdings)
        positions = [
            str(getattr(holding, "ticker", "") or "").strip() for holding in holdings
        ]
        if holdings_count:
            reasons.append(f"OPEN_POSITIONS:{holdings_count}")

    session_phase: str | None = None
    engine_running = False
    try:
        with _realtime_trading_lock:
            engine = _realtime_trading_engine
        if engine is not None:
            engine_running = True
            status = engine.get_status() if hasattr(engine, "get_status") else {}
            session = (status or {}).get("strategy_session") or {}
            session_phase = str(session.get("phase") or "") or None
            # ARMED/EXITING mean an order is in flight or an exit is being worked.
            if session_phase in {"ARMED", "ENTERING", "EXITING"}:
                reasons.append(f"SESSION_PHASE_IN_FLIGHT:{session_phase}")
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"ENGINE_STATUS_UNAVAILABLE:{type(exc).__name__}")

    return {
        "safe": not reasons,
        "reasons": reasons,
        "holdings_count": holdings_count,
        "positions": positions,
        "engine_running": engine_running,
        "session_phase": session_phase,
    }


@app.get("/api/system/restart-safety")
def system_restart_safety() -> JSONResponse:
    """Read-only: would stopping this process abandon managed state?"""
    try:
        report = _restart_safety_report()
    except Exception as exc:  # noqa: BLE001 - a broken check must read as unsafe.
        return _json(
            {
                "ok": True,
                "safe": False,
                "reasons": [f"SAFETY_CHECK_FAILED:{type(exc).__name__}"],
            }
        )
    return _json({"ok": True, **report})


@app.post("/api/system/graceful-shutdown")
def system_graceful_shutdown(force: bool = False) -> JSONResponse:
    """Stop background workers, then exit. Refuses unsafe stops unless forced.

    Used by ``run.ps1`` so a relaunch replaces the server instead of killing it:
    a force-kill leaves the trading engine's in-flight state and the SQLite
    writers to be terminated mid-operation.
    """
    report = _restart_safety_report()
    if not report["safe"] and not force:
        audit.record("graceful_shutdown_refused", report)
        return _json(
            {
                "ok": False,
                "status": "refused",
                "message": (
                    "Refusing to stop: restarting now would abandon in-memory exit "
                    "state. Re-send with force=true to override."
                ),
                **report,
            }
        )
    audit.record("graceful_shutdown_accepted", {**report, "forced": bool(force)})
    _schedule_app_process_shutdown()
    return _json(
        {
            "ok": True,
            "status": "shutting_down",
            "forced": bool(force),
            **report,
        }
    )


def _is_domestic_holding(holding: Holding) -> bool:
    ticker = str(getattr(holding, "ticker", "") or "").strip().upper()
    market = str(getattr(holding, "market", "") or "").strip().upper()
    return (ticker.isdigit() and len(ticker) == 6) or market in {"KR", "KRX", "KOSPI", "KOSDAQ", "KONEX"}


def _profit_seeking_termination_price(holding: Holding) -> float:
    last_price = float(getattr(holding, "last_price", 0.0) or 0.0)
    average_price = float(getattr(holding, "average_price", 0.0) or 0.0)
    if last_price <= 0:
        last_price = average_price
    if average_price <= 0:
        return round(max(0.0, last_price), 4)
    try:
        target_return = max(0.0, float(os.getenv("LIVE_TERMINATION_TARGET_PROFIT_RATE", "0.0025")))
    except (TypeError, ValueError):
        target_return = 0.0025
    try:
        hard_loss_exit = abs(float(os.getenv("LIVE_TERMINATION_HARD_LOSS_EXIT_RATE", "0.03")))
    except (TypeError, ValueError):
        hard_loss_exit = 0.03
    pnl_rate = (last_price - average_price) / average_price
    if pnl_rate <= -hard_loss_exit:
        price = last_price
    else:
        price = max(last_price, average_price * (1.0 + target_return))
    if _is_domestic_holding(holding):
        return float(max(1, round(price)))
    return round(max(0.0001, price), 4)


@app.post("/api/live-trading/terminate")
async def live_trading_terminate(shutdown: bool = True) -> JSONResponse:
    return _json(await run_in_threadpool(_live_trading_terminate_response, shutdown))


def _live_trading_terminate_response(shutdown: bool = True) -> dict[str, Any]:
    with _live_lock:
        active = _operation_mode_state.get("active")
    active_mode = getattr(active, "mode", None)
    active_mode_value = getattr(active_mode, "value", active_mode)
    if active_mode_value != "live_trading":
        return {"ok": False, "status": "not_live_trading", "message": "Live trading is not the active operation mode."}
    # 청산 모드는 엔진을 멈추지 않는다. BUY 평가만 즉시 닫고, 엔진의 SELL
    # 경로를 계속 살려 보유분 전량 청산을 반복 평가/제출하게 한다.
    os.environ["REALTIME_BUY_ENABLED"] = "false"
    with _realtime_trading_lock:
        engine = _realtime_trading_engine
    if engine is not None and hasattr(engine, "request_full_liquidation"):
        engine.request_full_liquidation("LIVE_TERMINATION_FULL_LIQUIDATION")
        status = engine.get_status() if hasattr(engine, "get_status") else {}
        audit.record(
            "live_trading_liquidation_requested",
            {
                "buy_enabled": False,
                "shutdown_requested": shutdown,
                "engine_status": status,
            },
        )
        return {
            "ok": True,
            "status": "liquidation_started",
            "buy_enabled": False,
            "liquidation_requested": True,
            "shutdown_scheduled": False,
            "engine_status": status,
            "message": "BUY evaluation is disabled. Realtime engine will keep running sell-only full-liquidation cycles.",
        }
    if engine is not None and hasattr(engine, "disable_buys"):
        engine.disable_buys("LIVE_TERMINATION_FULL_LIQUIDATION")
    config = load_short_horizon_strategy_config()
    config_live_enabled = bool(config.get("execution", {}).get("live_trading_enabled", False))
    env_live_enabled = _env_flag("LIVE_TRADING_ENABLED", False) and _env_flag("KIS_LIVE_ENABLED", False)
    runtime = evaluate_live_runtime_gates(require_manual_arming=_manual_arming_required())
    if not (config_live_enabled and env_live_enabled and runtime.ok):
        return {
            "ok": False,
            "status": "blocked",
            "live_trading_enabled_by_config": config_live_enabled,
            "live_trading_enabled_by_env": env_live_enabled,
            "runtime_gate_failures": runtime.failures,
            "message": "Live termination sell orders are blocked until live config, env flags, and manual arming gates pass.",
        }
    client = KisDevelopersApiClient(paper=False, enabled=True)
    portfolio = client.get_portfolio()
    receipts = []
    executions = []
    skipped = []
    for holding in portfolio.account.holdings:
        quantity = int(getattr(holding, "quantity", 0) or 0)
        limit_price = _profit_seeking_termination_price(holding)
        if quantity <= 0 or limit_price <= 0:
            skipped.append({"ticker": holding.ticker, "quantity": quantity, "last_price": limit_price})
            continue
        order = FinalOrder(
            ticker=holding.ticker,
            market=holding.market or "KR",
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            quantity=quantity,
            limit_price=limit_price,
            manual_approval_required=False,
        )
        receipt = client.place_limit_order(order)
        receipts.append(receipt)
        try:
            executions.append(client.get_order_status(receipt.order_id))
        except Exception as exc:  # noqa: BLE001 - the sell order was submitted; status lookup is best effort.
            executions.append({"order_id": receipt.order_id, "status": "STATUS_LOOKUP_FAILED", "message": str(exc)})
    with _live_lock:
        _operation_mode_state["active"] = None
        _operation_mode_state["live_trading_baseline_equity"] = None
    audit.record(
        "live_trading_terminated",
        {
            "submitted_sell_orders": len(receipts),
            "skipped_holdings": skipped,
            "account_equity": portfolio.account.equity,
            "buy_enabled": False,
            "shutdown_scheduled": shutdown,
        },
    )
    if shutdown:
        _schedule_app_process_shutdown()
    return {
        "ok": True,
        "status": "terminated",
        "buy_enabled": False,
        "submitted_sell_orders": len(receipts),
        "skipped_holdings": skipped,
        "receipts": receipts,
        "executions": executions,
        "shutdown_scheduled": shutdown,
        "message": "Live trading termination disabled BUY and submitted profit-seeking limit SELL orders for current KIS holdings.",
    }


@app.get("/api/mock-kis/portfolio")
def mock_kis_portfolio() -> JSONResponse:
    context = _get_or_refresh_live()["context"]
    broker = _mock_kis_for_context(context)
    return _json(broker.get_portfolio())


@app.post("/api/paper-trading/start")
async def paper_trading_start(request: Request) -> JSONResponse:
    return _paper_trading_removed_response()


def _paper_trading_removed_response() -> JSONResponse:
    return JSONResponse(
        {
            "ok": False,
            "status": "removed",
            "mode": "live_trading",
            "message": "Paper trading has been removed. Use /api/operation-mode/start with mode=live_trading.",
        },
        status_code=410,
    )


def _paper_trading_start_response(payload: dict[str, Any]) -> dict[str, Any]:
    target_return_rate = _normalised_target_return(payload.get("target_return_rate", 0.02))
    if "period_minutes" in payload:
      period_minutes = int(payload.get("period_minutes", 390))
    elif "period_days" in payload:
      period_minutes = int(payload.get("period_days", 7)) * 390
    else:
      period_minutes = 390
    initial_cash = _resolve_operating_initial_cash(payload)
    initial_cash_source = _resolved_initial_cash_source(payload)
    _ensure_initial_principal_configured(initial_cash)
    seed = int(payload.get("seed", 42))
    acceleration_factor = float(payload.get("acceleration_factor", 60.0))
    max_speed = bool(payload.get("max_speed", True))
    profit_gain = _resolve_auto_profit_gain(payload, initial_cash)
    demo_id = _start_streaming_demo(
        target_return_rate=target_return_rate,
        period_minutes=period_minutes,
        initial_cash=initial_cash,
        seed=seed,
        acceleration_factor=acceleration_factor,
        profit_gain=profit_gain,
        max_speed=max_speed,
    )
    
    return {
        "demo_id": demo_id,
        "status": "initialized",
        "progress": 0.0,
        "target_return_rate": target_return_rate,
        "period_minutes": period_minutes,
        "initial_cash": initial_cash,
        "initial_cash_source": initial_cash_source,
        "acceleration_factor": acceleration_factor,
        "profit_gain": max(0.25, min(4.0, profit_gain)),
        "profit_gain_source": "auto_goal_account_liquidity",
        "message": "모의투자가 시작되었습니다. 단계 진행은 /api/paper-trading/step을 호출하세요.",
    }


@app.post("/api/paper-trading/step")
async def paper_trading_step(request: Request) -> JSONResponse:
    return _paper_trading_removed_response()


def _streaming_demo_step_response(payload: dict[str, Any]) -> dict[str, Any]:
    started_at = time.perf_counter()
    demo_id = str(payload.get("demo_id", ""))
    if demo_id not in _streaming_demos:
        return {
            "demo_id": demo_id,
            "status": "expired",
            "progress": 0.0,
            "message": "Paper trading session expired. Start a new paper trading run.",
            "step_latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
        }

    step_lock = _streaming_demo_step_locks.setdefault(demo_id, threading.Lock())
    if not step_lock.acquire(blocking=False):
        demo = _streaming_demos.get(demo_id)
        return {
            "demo_id": demo_id,
            "status": "busy",
            "progress": demo.get_progress() if demo is not None else 0.0,
            "account": _streaming_demo_account_payload(demo),
            "message": "Paper trading step is already running.",
            "step_latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
        }

    try:
        demo = _streaming_demos[demo_id]
        wait_seconds = demo.seconds_until_next_step()
        if wait_seconds > 0:
            return {
                "demo_id": demo_id,
                "status": "waiting",
                "progress": demo.get_progress(),
                "account": _streaming_demo_account_payload(demo),
                "seconds_until_next_step": round(wait_seconds, 1),
                "retry_after_seconds": round(wait_seconds, 1),
                "message": "Waiting for the next one-minute paper trading bar.",
                "step_latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
            }
        result = demo.run_step()
        candidate_selection = demo.get_candidate_selection()
        if result is None:
            return {
                "demo_id": demo_id,
                "status": "completed",
                "progress": 100.0,
                "message": "Paper trading completed.",
                "final_results": demo.get_final_results(),
                "account": _streaming_demo_account_payload(demo),
                "ontology_filter_1": _candidate_selection_payload(candidate_selection),
                "step_latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
            }

        time_scaler = demo.get_time_scaler()
        virtual_time = time_scaler.get_virtual_time() if time_scaler else None
        final = demo.get_final_results() if demo.is_complete() else None
        saved_realtime = _save_streaming_step_realtime_records(demo_id, result)
        return {
            "demo_id": demo_id,
            "status": "completed" if final is not None else "running",
            "compute_backend": "openvino_npu_screening_plus_cpu_rules",
            "ontology_backend": get_ontology_npu_classifier().status(),
            "universe_count": result.universe_ticker_count,
            "universe_scanned_count": result.universe_scanned_count,
            "active_ticker_count": result.active_ticker_count,
            "candidate_ticker_count": result.candidate_ticker_count,
            "ontology_npu": _to_jsonable(result.ontology_npu),
            "step": result.visible_step,
            "raw_step": result.step_index,
            "chart_bar": result.step_index,
            "progress": result.progress_percent,
            "seconds_until_next_step": round(demo.seconds_until_next_step(), 1),
            "timestamp": result.timestamp,
            "virtual_time": virtual_time,
            "prices": result.prices,
            "account": {
                "cash": result.cash,
                "account_value": result.account_value,
                "return_rate": result.return_rate,
                "base_currency": result.base_currency,
                "cash_by_currency": result.cash_by_currency,
                "account_value_krw": result.account_value_krw,
                "usd_krw_rate": result.usd_krw_rate,
            },
            "principal_protection": _to_jsonable(result.principal_protection),
            "profit_gain": _to_jsonable(result.profit_gain),
            "currency_by_ticker": result.currency_by_ticker,
            "holdings": result.holdings,
            "trades_in_step": len(result.trades_in_step),
            "cumulative_trades": result.cumulative_trades,
            "trades": [_to_jsonable(t) for t in result.trades_in_step],
            "stored_realtime_records": saved_realtime,
            "final_results": final,
            "ontology_filter_1": _candidate_selection_payload(candidate_selection),
            "step_latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
        }
    finally:
        step_lock.release()


def _streaming_demo_account_payload(demo: Any | None) -> dict[str, Any] | None:
    if demo is None:
        return None
    cash = float(getattr(demo, "_cash", 0.0) or 0.0)
    initial_cash = max(1.0, float(getattr(demo, "initial_cash", 1.0) or 1.0))
    cash_by_currency = dict(getattr(demo, "_cash_by_currency", {}) or {})
    account_value = cash
    return {
        "cash": round(cash, 2),
        "account_value": round(account_value, 2),
        "return_rate": round((account_value - initial_cash) / initial_cash, 6),
        "base_currency": "KRW",
        "cash_by_currency": {str(key): round(float(value), 2) for key, value in sorted(cash_by_currency.items())},
        "account_value_krw": round(account_value, 2),
        "usd_krw_rate": 0.0,
    }


def _candidate_selection_payload(selection: Any | None) -> dict[str, Any] | None:
    if selection is None:
        return None
    return {
        "stage": "ontology_filter_1",
        "full_universe_count": selection.full_universe_count,
        "candidate_count": len(selection.candidate_stocks),
        "chart_fetch_count": len(selection.chart_fetch_scope),
        "chart_fetch_scope": selection.chart_fetch_scope[:20],
        "rejected_count": len(selection.rejected_stocks),
        "latency_ms": selection.latency_ms,
        "api_call_count": selection.api_call_count,
        "sample_traces": [
            {
                "stock_code": trace.stock_code,
                "decision": trace.decision,
                "score": trace.score,
                "fired_rules": trace.fired_rules,
                "reason": trace.reason,
            }
            for trace in selection.traces[:10]
        ],
    }


def _save_streaming_step_realtime_records(demo_id: str, result: Any) -> dict[str, int]:
    source_time = result.timestamp
    quotes = tuple(
        RealtimeQuote(
            ticker=ticker,
            market="SIM",
            observed_at=source_time,
            last_price=price,
            source=SourceMetadata(
                source_name="accelerated_demo_stream",
                retrieved_at=source_time,
                raw_url=f"local://accelerated-demo/{demo_id}/quotes/{result.step_index}",
                source_id=f"demo-quote:{demo_id}:{result.step_index}:{ticker}",
            ),
        )
        for ticker, price in result.universe_prices.items()
    )
    executions = tuple(
        RealtimeExecution(
            ticker=trade.ticker,
            market="SIM",
            executed_at=trade.timestamp,
            price=trade.price,
            quantity=trade.quantity,
            side=trade.side,
            trade_id=f"{demo_id}:{result.step_index}:{index}:{trade.ticker}:{trade.side}",
            source=SourceMetadata(
                source_name="accelerated_demo_stream",
                retrieved_at=source_time,
                raw_url=f"local://accelerated-demo/{demo_id}/executions/{result.step_index}",
                source_id=f"demo-execution:{demo_id}:{result.step_index}:{index}",
            ),
        )
        for index, trade in enumerate(result.trades_in_step)
    )
    return {
        "realtime_quotes": 0,
        "realtime_executions": 0,
        "skipped_simulated_quotes": len(quotes),
        "skipped_simulated_executions": len(executions),
    }


@app.get("/api/paper-trading/status/{demo_id}")
def streaming_demo_status(demo_id: str) -> JSONResponse:
    """Return the current state of a streaming paper-trading demo."""
    return _paper_trading_removed_response()
    if demo_id not in _streaming_demos:
        raise HTTPException(status_code=404, detail="Demo not found")
    
    demo = _streaming_demos[demo_id]
    time_scaler = demo.get_time_scaler()
    
    return _json({
        "demo_id": demo_id,
        "progress": demo.get_progress(),
        "is_complete": demo.is_complete(),
        "is_paused": time_scaler.is_paused() if time_scaler else False,
        "scale_factor": time_scaler.get_scale_factor() if time_scaler else 1.0,
        "seconds_until_next_step": round(demo.seconds_until_next_step(), 1),
        "time_config": _to_jsonable(demo.config),
    })


@app.post("/api/paper-trading/pause/{demo_id}")
async def streaming_demo_pause(demo_id: str) -> JSONResponse:
    """Pause a streaming paper-trading demo."""
    return _paper_trading_removed_response()
    if demo_id not in _streaming_demos:
        raise HTTPException(status_code=404, detail="Demo not found")
    
    demo = _streaming_demos[demo_id]
    demo.pause()
    
    return _json({
        "demo_id": demo_id,
        "status": "paused",
        "is_paused": True,
    })


@app.post("/api/paper-trading/resume/{demo_id}")
async def streaming_demo_resume(demo_id: str) -> JSONResponse:
    """Start a temporary accelerated paper-trading demo."""
    return _paper_trading_removed_response()
    if demo_id not in _streaming_demos:
        raise HTTPException(status_code=404, detail="Demo not found")
    
    demo = _streaming_demos[demo_id]
    demo.resume()
    
    return _json({
        "demo_id": demo_id,
        "status": "resumed",
        "is_paused": False,
    })


@app.post("/api/paper-trading/cleanup/{demo_id}")
async def streaming_demo_cleanup(demo_id: str) -> JSONResponse:
    """Clean up a streaming paper-trading demo."""
    return _paper_trading_removed_response()
    with _streaming_demos_lock:
        if demo_id in _streaming_demos:
            del _streaming_demos[demo_id]
        _streaming_demo_step_locks.pop(demo_id, None)
    
    return _json({
        "demo_id": demo_id,
        "status": "cleaned_up",
        "message": "데모가 정리되었습니다.",
    })


@app.get("/api/mock-kis/portfolio")
def mock_kis_portfolio() -> JSONResponse:
    context = _get_or_refresh_live()["context"]
    broker = _mock_kis_for_context(context)
    return _json(broker.get_portfolio())


@app.get("/api/mock-trading/performance")
def mock_trading_performance() -> JSONResponse:
    context = _get_or_refresh_live()["context"]
    return _json(_mock_performance(context))


@app.get("/api/live-trading/progress")
def live_trading_progress() -> JSONResponse:
    try:
      connection = _cached_kis_connection_probe(paper=False, include_account=True)
    except Exception as exc:  # pragma: no cover - broker/network defensive boundary
      connection = {"ok": False, "mode": "live", "message": str(exc), "error": str(exc)}
    basis = _account_basis_from_kis_connection(connection) or _last_live_account_basis()
    if basis is not None and connection.get("account_checked"):
      connection = _connection_with_account_basis(connection, basis)
      with _live_lock:
        _operation_mode_state["last_kis_connection"] = connection
        _operation_mode_state["last_kis_connection_checked_at"] = time.time()
    positions = list(connection.get("positions") or (basis or {}).get("positions") or [])
    snapshot = _live_snapshot()
    execution_summary = snapshot.get("live_execution_summary") or {}
    realtime_summary = _realtime_engine_execution_summary()
    if realtime_summary is not None:
      execution_summary = realtime_summary
    runtime_gate = evaluate_live_runtime_gates(require_manual_arming=_manual_arming_required())
    journal = _live_order_journal_snapshot()
    positions, pending_positions = _reconciled_live_positions(positions, journal)
    if pending_positions and basis is not None:
      basis = dict(basis)
      basis["positions"] = positions
      basis["invested_value"] = sum(_number_or_zero(position.get("market_value_krw") or position.get("market_value")) for position in positions)
      connection = _connection_with_account_basis(connection, basis)
      connection["pending_positions"] = pending_positions
      with _live_lock:
        _operation_mode_state["stable_account_basis"] = basis
        _operation_mode_state["last_kis_connection"] = connection
        _operation_mode_state["last_kis_connection_checked_at"] = time.time()
    active_mode = None
    baseline_equity = None
    with _live_lock:
      active = _operation_mode_state.get("active")
      active_mode = getattr(getattr(active, "mode", None), "value", getattr(active, "mode", None))
      baseline_equity = _operation_mode_state.get("live_trading_baseline_equity")
    equity = float(basis["equity"]) if basis is not None else 0.0
    cash = float(basis["cash"]) if basis is not None else 0.0
    initial = float(baseline_equity or equity or 0.0)
    return_rate = (equity - initial) / initial if initial > 0 else 0.0
    realtime_engine_running = bool((realtime_summary or {}).get("engine_running"))
    return _json(
        {
            "active": active_mode == "live_trading" or realtime_engine_running,
            "mode": "live_trading",
            "account_checked": bool(connection.get("account_checked")),
            "connection": connection,
            "cash": cash,
            "cash_equivalent_krw": float(basis.get("cash_equivalent_krw") or cash) if basis is not None else cash,
            "krw_cash": float(basis["krw_cash"]) if basis is not None else 0.0,
            "foreign_cash_krw": float(basis.get("foreign_cash_krw") or 0.0) if basis is not None else 0.0,
            "cash_by_currency": basis["cash_by_currency"] if basis is not None else {},
            "foreign_cash_by_currency": basis["foreign_cash_by_currency"] if basis is not None else {},
            "equity": equity,
            "initial_equity": initial,
            "profit": equity - initial,
            "return_rate": return_rate,
            "positions": positions,
            "pending_positions": pending_positions,
            "execution_summary": execution_summary,
            "runtime_gate": {"ok": runtime_gate.ok, "failures": tuple(runtime_gate.failures)},
            "live_order_journal": journal,
            "orders_count": journal["orders_count"],
            "executions_count": journal["submitted_count"],
            "recent_orders": journal["recent_orders"],
            "recent_executions": journal["recent_executions"],
            "updated_at": datetime.now(timezone.utc),
            "message": (
                "Live order submission is enabled; approved FinalOrder records may be submitted."
                if runtime_gate.ok
                else "Live order submission is blocked by runtime gates: " + ", ".join(runtime_gate.failures)
            ),
        }
    )


def _live_order_journal_snapshot(path: str | Path = "logs/live-orders.jsonl", limit: int = 20) -> dict[str, Any]:
    journal_path = Path(path)
    if not journal_path.exists():
      return {
          "path": str(journal_path),
          "orders_count": 0,
          "submitted_count": 0,
          "blocked_count": 0,
          "error_count": 0,
          "recent_orders": [],
          "recent_executions": [],
      }
    events: list[dict[str, Any]] = []
    try:
      with journal_path.open("r", encoding="utf-8") as file:
        for line in file:
          try:
            event = json.loads(line)
          except json.JSONDecodeError:
            continue
          if isinstance(event, dict):
            events.append(event)
    except OSError:
      events = []
    enriched_events = _enrich_live_order_events(events)
    recent_events = enriched_events[-max(1, limit):]
    recent_orders = [_live_order_event_payload(event) for event in recent_events]
    recent_orders = [item for item in recent_orders if item]
    all_orders = [_live_order_event_payload(event) for event in enriched_events]
    all_orders = [item for item in all_orders if item]
    submitted_orders = [
        item
        for item in all_orders
        if item.get("event_type") in {"live_order_submitted", "live_order_status", "live_trading_order_submitted"}
    ]
    recent_executions = [
        item
        for item in recent_orders
        if item.get("event_type") in {"live_order_submitted", "live_order_status", "live_trading_order_submitted"}
    ]
    return {
        "path": str(journal_path),
        "orders_count": len(all_orders),
        "submitted_count": sum(1 for event in events if event.get("event_type") in {"live_order_submitted", "live_trading_order_submitted"}),
        "blocked_count": sum(1 for event in events if event.get("event_type") in {"live_order_blocked", "live_trading_execution_blocked"}),
        "error_count": sum(1 for event in events if "error" in str(event.get("event_type") or "")),
        "recent_orders": recent_orders[-limit:],
        "recent_executions": recent_executions[-limit:],
        "submitted_orders": submitted_orders[-50:],
    }


def _enrich_live_order_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attempts: dict[str, dict[str, Any]] = {}
    enriched: list[dict[str, Any]] = []
    for event in events:
      copied = dict(event)
      payload = dict(copied.get("payload") or {}) if isinstance(copied.get("payload"), dict) else {}
      event_type = str(copied.get("event_type") or "")
      primary_key = str(payload.get("idempotency_key") or "")
      execution_key = str(payload.get("execution_id") or "")
      if event_type == "live_order_submission_attempt":
        if primary_key:
          attempts[primary_key] = payload
        if execution_key:
          attempts[execution_key] = payload
      elif event_type in {"live_order_submitted", "live_order_status"}:
        attempt = attempts.get(primary_key) or attempts.get(execution_key)
        if attempt and "order" not in payload and isinstance(attempt.get("order"), dict):
          payload["order"] = attempt["order"]
          copied["payload"] = payload
      enriched.append(copied)
    return enriched


def _live_order_event_payload(event: dict[str, Any]) -> dict[str, Any] | None:
    event_type = str(event.get("event_type") or "")
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    if not event_type.startswith("live_"):
      return None
    order = payload.get("order") if isinstance(payload.get("order"), dict) else {}
    raw = payload.get("raw") if isinstance(payload.get("raw"), dict) else {}
    ticker = str(payload.get("ticker") or order.get("ticker") or "")
    if not ticker:
      ticker = str(raw.get("ticker") or "")
    market = str(payload.get("market") or order.get("market") or "")
    if not market:
      market = str(raw.get("market") or "")
    quantity = payload.get("quantity") or order.get("quantity") or ""
    raw_quantity = raw.get("quantity") or ""
    limit_price = payload.get("limit_price") or order.get("limit_price") or ""
    raw_price = raw.get("price") or ""
    status = str(payload.get("status") or raw.get("status") or "").upper()
    filled_quantity = raw_quantity if status in {"FILLED", "PARTIALLY_FILLED"} else payload.get("filled_quantity", "")
    average_fill_price = raw_price if filled_quantity not in ("", None) else payload.get("average_fill_price", "")
    currency = "USD" if _is_us_order_market(market, ticker) else "KRW"
    return {
        "event_type": event_type,
        "recorded_at": event.get("recorded_at"),
        "ticker": ticker,
        "market": market,
        "side": payload.get("side") or order.get("side") or raw.get("side") or "",
        "quantity": quantity or raw_quantity,
        "limit_price": limit_price,
        "notional": _number_or_zero(quantity or raw_quantity) * _number_or_zero(limit_price or raw_price),
        "currency": currency,
        "broker_order_id": payload.get("broker_order_id") or payload.get("order_id") or raw.get("order_id") or "",
        "status": status,
        "filled_quantity": filled_quantity,
        "average_fill_price": average_fill_price,
        "raw": raw,
        "reason_codes": payload.get("reason_codes") or payload.get("blocked") or (),
        "error": payload.get("error") or payload.get("message") or "",
    }


def _parse_event_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
      return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value or "").strip()
    if not text:
      return None
    try:
      parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
      return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _recent_live_buy_orders_for_reconciliation(
    journal: dict[str, Any],
    positions: list[dict[str, Any]],
    *,
    max_age_hours: float = 36.0,
    limit: int = 10,
) -> list[dict[str, Any]]:
    existing = {str(position.get("ticker") or "").upper() for position in positions if isinstance(position, dict)}
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    candidates: list[dict[str, Any]] = []
    seen_order_ids: set[str] = set()
    for item in reversed(list(journal.get("submitted_orders") or ())):
      if not isinstance(item, dict):
        continue
      order_id = str(item.get("broker_order_id") or "")
      ticker = str(item.get("ticker") or "").upper().strip()
      side = str(item.get("side") or "").upper()
      if not order_id or order_id in seen_order_ids or not ticker or ticker in existing or side != "BUY":
        continue
      recorded_at = _parse_event_datetime(item.get("recorded_at"))
      if recorded_at is not None and recorded_at < cutoff:
        continue
      seen_order_ids.add(order_id)
      candidates.append(item)
      if len(candidates) >= limit:
        break
    return list(reversed(candidates))


def _pending_position_from_order_status(order: dict[str, Any], execution: Any | None = None) -> dict[str, Any] | None:
    ticker = str(getattr(execution, "ticker", None) or order.get("ticker") or "").upper().strip()
    if not ticker:
      return None
    side = getattr(execution, "side", None) or order.get("side")
    side_value = getattr(side, "value", side)
    if str(side_value or "").upper() != "BUY":
      return None
    status = str(getattr(execution, "status", None) or order.get("status") or "ACCEPTED").upper()
    if status in {"CANCELED", "REJECTED", "EXPIRED"}:
      return None
    if status not in {"FILLED", "PARTIALLY_FILLED"}:
      return None
    quantity = int(_number_or_zero(getattr(execution, "quantity", None) or order.get("quantity") or 0))
    if quantity <= 0:
      return None
    price = _number_or_zero(getattr(execution, "price", None) or order.get("limit_price") or 0)
    if price <= 0:
      return None
    market = str(order.get("market") or "KR").upper()
    currency = "USD" if _is_us_order_market(market, ticker) else "KRW"
    value = quantity * price
    return {
        "ticker": ticker,
        "market": market,
        "quantity": quantity,
        "average_price": price,
        "last_price": price,
        "market_value": value,
        "unrealized_pnl": 0.0,
        "return_rate": 0.0,
        "currency": currency,
        "market_value_krw": value if currency == "KRW" else 0.0,
        "unrealized_pnl_krw": 0.0,
        "source": "order_reconciliation",
        "position_state": "pending_balance",
        "order_status": status,
        "broker_order_id": order.get("broker_order_id") or getattr(execution, "order_id", ""),
        "recorded_at": order.get("recorded_at"),
    }


def _reconciled_live_positions(
    positions: list[dict[str, Any]],
    journal: dict[str, Any],
    broker: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not journal.get("submitted_orders"):
      return positions, []
    reconciled = [dict(position) for position in positions if isinstance(position, dict)]
    existing = {str(position.get("ticker") or "").upper() for position in reconciled}
    pending: list[dict[str, Any]] = []
    owned_broker = broker
    for order in _recent_live_buy_orders_for_reconciliation(journal, reconciled):
      execution = None
      status_error = ""
      if owned_broker is None:
        try:
          owned_broker = KisDevelopersApiClient(paper=False, enabled=True)
        except Exception as exc:  # pragma: no cover - credential/runtime defensive boundary
          status_error = str(exc)
      if owned_broker is not None:
        try:
          order_id = str(order.get("broker_order_id") or "")
          if hasattr(owned_broker, "_orders") and order_id:
            side = OrderSide.BUY if str(order.get("side") or "").upper() == "BUY" else OrderSide.SELL
            owned_broker._orders.setdefault(  # type: ignore[attr-defined]
                order_id,
                FinalOrder(
                    ticker=str(order.get("ticker") or ""),
                    market=str(order.get("market") or "KR"),
                    order_type=OrderType.LIMIT,
                    side=side,
                    quantity=max(1, int(_number_or_zero(order.get("quantity") or 1))),
                    limit_price=_number_or_zero(order.get("limit_price") or 0),
                ),
            )
          execution = owned_broker.get_order_status(str(order.get("broker_order_id") or ""))
        except Exception as exc:  # pragma: no cover - broker/network defensive boundary
          status_error = str(exc)
      pending_position = _pending_position_from_order_status(order, execution)
      if pending_position is None:
        continue
      ticker = str(pending_position.get("ticker") or "").upper()
      if ticker in existing:
        continue
      if status_error:
        pending_position["order_status_error"] = status_error
      existing.add(ticker)
      pending.append(pending_position)
      reconciled.append(pending_position)
    return reconciled, pending


def _is_us_order_market(market: str, ticker: str) -> bool:
    market_upper = str(market or "").upper()
    if any(token in market_upper for token in ("US", "NASDAQ", "NYSE", "AMEX", "NASD")):
      return True
    return bool(ticker) and not (ticker.isdigit() and len(ticker) == 6)


def _parse_goal_request(payload: dict[str, Any]) -> GoalRequest:
    period_minutes = int(payload.get("period_minutes") or 0)
    period_days = int(payload.get("period_days") or 0)
    if period_days <= 0 and period_minutes > 0:
        period_days = max(1, (period_minutes + 389) // 390)
    goal_mode = str(payload.get("goal_mode") or "").strip()
    target_return_rate = payload.get("target_return_rate")
    has_rate = target_return_rate not in (None, "")
    target_profit_amount = payload.get("target_profit_amount")
    has_amount = target_profit_amount not in (None, "")

    if has_rate and has_amount:
        raise HTTPException(
            status_code=400,
            detail="Use either target_return_rate or target_profit_amount, not both.",
        )
    if goal_mode and goal_mode not in {"rate", "amount"}:
        raise HTTPException(status_code=400, detail="Unsupported goal_mode.")
    if goal_mode == "rate" and not has_rate:
        raise HTTPException(status_code=400, detail="target_return_rate is required for rate mode.")
    if goal_mode == "amount" and not has_amount:
        raise HTTPException(status_code=400, detail="target_profit_amount is required for amount mode.")
    if not has_rate and not has_amount:
        raise HTTPException(status_code=400, detail="A target return rate or profit amount is required.")

    return GoalRequest(
        target_return_rate=float(target_return_rate) / 100.0 if has_rate else None,
        target_profit_amount=float(target_profit_amount) if has_amount else None,
        period_days=period_days,
        period_minutes=period_minutes if period_minutes > 0 else None,
    )

    if goal_mode and goal_mode != "rate":
        raise HTTPException(
            status_code=400,
            detail="목표 수익률만 사용할 수 있습니다.",
        )
    if not has_rate:
        raise HTTPException(
            status_code=400,
            detail="목표 수익률을 입력하세요.",
        )

    parsed_rate = float(target_return_rate) / 100.0
    parsed_amount = None

    return GoalRequest(
        target_return_rate=parsed_rate,
        target_profit_amount=parsed_amount,
        period_days=period_days,
        period_minutes=period_minutes if period_minutes > 0 else None,
    )


def _goal_from_payload(payload: dict[str, Any], context: Any) -> NegotiatedGoal:
    selected = payload.get("selected_goal") if isinstance(payload.get("selected_goal"), dict) else payload
    if "target_return_rate" in selected or "target_profit_amount" in selected:
        raw_rate = selected.get("target_return_rate")
        target_return_rate = None
        if raw_rate not in (None, ""):
            numeric_rate = float(raw_rate)
            target_return_rate = numeric_rate if numeric_rate <= 1 else numeric_rate / 100.0
        request = GoalRequest(
            target_return_rate=target_return_rate,
            target_profit_amount=(
                float(selected["target_profit_amount"])
                if selected.get("target_profit_amount") not in (None, "")
                else None
            ),
            period_days=int(selected.get("period_days") or 30),
            period_minutes=int(selected["period_minutes"]) if selected.get("period_minutes") not in (None, "") else None,
        )
        assessment = assess_goal(
            request,
            _goal_account_snapshot(context),
            context.markets,
            context.indicators,
            context.signals,
            context.graph,
        )
        return NegotiatedGoal(
            target_return_rate=assessment.requested_return_rate,
            target_profit_amount=assessment.requested_profit_amount,
            period_days=assessment.period_days,
            feasibility_percent=assessment.feasibility_percent,
            label=str(selected.get("label", "Mock API target")),
        )
    return NegotiatedGoal(
        target_return_rate=0.02,
        target_profit_amount=context.report.equity * 0.02,
        period_days=30,
        feasibility_percent=65,
        label="Default mock API target",
    )


def _parse_final_order(payload: dict[str, Any]) -> FinalOrder:
    return FinalOrder(
        ticker=str(payload["ticker"]),
        market=str(payload.get("market", "MOCK")),
        order_type=OrderType(str(payload.get("order_type", "LIMIT"))),
        side=OrderSide(str(payload["side"])),
        quantity=int(payload["quantity"]),
        limit_price=float(payload["limit_price"]),
        time_in_force=str(payload.get("time_in_force", "DAY")),
        manual_approval_required=bool(payload.get("manual_approval_required", True)),
    )


def _mock_kis_for_context(context: Any) -> MockKisDevelopersApi:
    global _mock_kis
    with _mock_kis_lock:
        if _mock_kis is None:
            _mock_kis = MockKisDevelopersApi(
                account=context.account,
                market_prices={market.ticker: market.last_price for market in context.markets},
                sectors={market.ticker: market.sector for market in context.markets},
                company_names={market.ticker: market.company_name for market in context.markets},
            )
        else:
            _mock_kis.market_prices.update(
                {market.ticker: market.last_price for market in context.markets}
            )
        return _mock_kis


def _reset_mock_kis_for_context(context: Any, account: AccountSnapshot | None = None) -> MockKisDevelopersApi:
    global _mock_kis
    account = account or context.account
    with _mock_kis_lock:
        _mock_kis = MockKisDevelopersApi(
            account=account,
            market_prices={market.ticker: market.last_price for market in context.markets},
            sectors={market.ticker: market.sector for market in context.markets},
            company_names={market.ticker: market.company_name for market in context.markets},
        )
        return _mock_kis


def _mock_demo_account(context: Any) -> AccountSnapshot:
  return context.account


def _mock_performance(context: Any) -> dict[str, Any]:
    broker = _mock_kis_for_context(context)
    prices = {market.ticker: market.last_price for market in context.markets}
    broker.market_prices.update(prices)
    portfolio = broker.get_portfolio()
    initial_equity = float(_mock_trading_state.get("initial_equity") or context.report.equity)
    positions = []
    position_value = 0.0
    for holding in portfolio.account.holdings:
        price = prices.get(holding.ticker, holding.last_price)
        market_value = holding.quantity * price
        cost = holding.quantity * holding.average_price
        position_value += market_value
        positions.append(
            {
                "ticker": holding.ticker,
                "quantity": holding.quantity,
                "average_price": holding.average_price,
                "last_price": price,
                "market_value": market_value,
                "unrealized_pnl": market_value - cost,
                "return_rate": (market_value - cost) / cost if cost else 0.0,
            }
        )
    equity = portfolio.account.cash + position_value
    profit_amount = equity - initial_equity
    goal = _mock_trading_state.get("goal")
    target_return_rate = float(goal.target_return_rate) if goal is not None else None
    executions = [
        {
            "order_id": execution.order_id,
            "ticker": execution.ticker,
            "side": execution.side.value,
            "quantity": execution.quantity,
            "price": execution.price,
            "executed_value": execution.executed_value,
            "status": execution.status,
            "message": execution.message,
            "executed_at": execution.executed_at,
        }
        for execution in broker.list_executions()
    ]
    return {
        "active": bool(_mock_trading_state.get("active")),
        "session_id": _mock_trading_state.get("session_id"),
        "started_at": _iso_or_none(_mock_trading_state.get("started_at")),
        "cash": portfolio.account.cash,
        "position_value": position_value,
        "equity": equity,
        "initial_equity": initial_equity,
        "profit_amount": profit_amount,
        "return_rate": profit_amount / initial_equity if initial_equity else 0.0,
        "target_return_rate": target_return_rate,
        "target_achieved": (
            profit_amount / initial_equity >= target_return_rate
            if initial_equity and target_return_rate is not None
            else False
        ),
        "positions": sorted(positions, key=lambda item: abs(item["market_value"]), reverse=True),
        "orders_count": len(broker.list_orders()),
        "executions_count": len(executions),
        "recent_executions": executions[-20:],
        "updated_at": datetime.now().isoformat(),
    }


def _load_default_research() -> ResearchRunResult:
    return ResearchService(progress_callback=_research_progress).run_from_config(DEFAULT_RESEARCH_CONFIG)


def _live_research_collection_due(now: datetime | None = None) -> bool:
  current = now or datetime.now(timezone.utc)
  with _live_lock:
    last = _live_state.get("research_last_collected_at")
  if last is None:
    return True
  if isinstance(last, str):
    last = _parse_iso_datetime(last)
  if not isinstance(last, datetime):
    return True
  if last.tzinfo is None:
    last = last.replace(tzinfo=timezone.utc)
  return (current - last).total_seconds() >= LIVE_RESEARCH_COLLECTION_INTERVAL_SECONDS


def _diagnostics_with_collection_config(diagnostics: dict[str, Any]) -> dict[str, Any]:
  result = dict(diagnostics or {})
  try:
    config = json.loads(DEFAULT_RESEARCH_CONFIG.read_text(encoding="utf-8"))
  except Exception:
    return result
  stooq_count = len(config.get("stooq_symbols", []))
  yahoo_chart_count = len(config.get("yahoo_chart_symbols", []))
  alpha_vantage_count = len(config.get("alpha_vantage_symbols", []))
  configured_counts = {
      "rss_feeds": len(config.get("rss_feeds", [])),
      "rss_fetch_articles": int(bool(config.get("rss_fetch_articles", False))),
      "rss_article_fetch_limit_per_feed": int(config.get("rss_article_fetch_limit_per_feed", 0) or 0),
      "html_pages": len(config.get("html_pages", [])),
      "dynamic_pages": len(config.get("dynamic_pages", [])),
      "stooq_symbols": stooq_count,
      "yahoo_chart_symbols": yahoo_chart_count,
      "alpha_vantage_symbols": alpha_vantage_count,
      "fred_series": len(config.get("fred_series", [])),
      "ecos_series": len(config.get("ecos_series", [])),
      "opendart_disclosures": len(config.get("opendart_disclosures", [])),
  }
  warnings = list(result.get("collection_warnings") or [])
  if stooq_count + yahoo_chart_count + alpha_vantage_count == 0:
    warnings.append(
        "No external stock chart source is configured; market snapshots will be limited to listed-universe reference records."
    )
  elif yahoo_chart_count > 0:
    warnings.append("Yahoo chart endpoints may be blocked by robots.txt in the built-in HTTP client.")
  result["configured_source_counts"] = {
      **configured_counts,
      **dict(result.get("configured_source_counts") or {}),
  }
  result["external_chart_sources_configured"] = stooq_count + yahoo_chart_count + alpha_vantage_count
  result["collection_warnings"] = tuple(dict.fromkeys(warnings))
  return result


def _research_progress(source_key: str, completed: int, total: int) -> None:
    is_retry = source_key.startswith("retry:")
    percent = 50 if is_retry else 18 + int((min(completed, total) / max(1, total)) * 30)
    message = _format_research_progress_message(source_key, completed, total)
    _set_live_progress(
        percent,
        "research",
        message,
    )


def _format_research_progress_message(source_key: str, completed: int, total: int) -> str:
  label = _format_source_label(source_key)
  if source_key.startswith("retry:"):
    retry_target, attempt = _split_retry_source_key(source_key)
    retry_label = _format_source_label(retry_target)
    if attempt:
      return f"재시도 중 · {retry_label} · {attempt}"
    return f"재시도 중 · {retry_label}"
  return f"자료 수집 중 · {label} · {completed}/{max(1, total)}"


def _split_retry_source_key(source_key: str) -> tuple[str, str | None]:
  if not source_key.startswith("retry:"):
    return source_key, None
  payload = source_key[6:]
  attempt = None
  if ":attempt " in payload:
    payload, attempt = payload.rsplit(":attempt ", 1)
  return payload, attempt


def _format_source_label(source_key: str) -> str:
  raw = source_key
  if source_key.startswith("retry:"):
    raw, _ = _split_retry_source_key(source_key)
  prefix, _, remainder = raw.partition(":")
  labels = {
    "rss": "RSS 뉴스",
    "html": "HTML 페이지",
    "dynamic": "동적 페이지",
    "stooq": "Stooq 시세",
    "yahoo_chart": "Yahoo 차트",
    "fred": "FRED",
    "ecos": "ECOS",
    "opendart": "OpenDART",
  }
  if not remainder:
    return labels.get(prefix, raw)
  tail = remainder
  if prefix == "dynamic":
    tail = remainder.rsplit("/", 1)[-1]
  return f"{labels.get(prefix, prefix)} {tail}"


def _build_web_context():
  return _get_or_refresh_live()["context"]


def _set_operation_request(busy: bool, stage: str, message: str, error: str | None) -> None:
  now = datetime.now(timezone.utc)
  with _live_lock:
    request = dict(_operation_mode_state.get("request") or {})
    if busy and not request.get("busy"):
      request["started_at"] = now
    request.update(
        {
            "busy": busy,
            "stage": stage,
            "message": message,
            "updated_at": now,
            "last_error": error,
        }
    )
    _operation_mode_state["request"] = request


def _operation_mode_request_snapshot() -> dict[str, Any]:
  with _live_lock:
    request = dict(_operation_mode_state.get("request") or {})
  for key in ("started_at", "updated_at"):
    if isinstance(request.get(key), datetime):
      request[key] = _iso_or_none(request[key])
  return request


def _ensure_background_refresh() -> None:
  global _refresh_worker
  with _live_lock:
    if bool(_live_state["is_refreshing"]):
      _live_state["refresh_requested_after_current"] = True
      return
    if _refresh_worker is not None and _refresh_worker.is_alive():
      return
    _refresh_worker = threading.Thread(target=_refresh_live_cache, name="operation-mode-refresh", daemon=True)
    _refresh_worker.start()


def _start_kis_realtime_collector() -> None:
  global _kis_realtime_collector_worker
  with _live_lock:
    if _kis_realtime_collector_worker is not None and _kis_realtime_collector_worker.is_alive():
      return
    _kis_realtime_collector_stop.clear()
    _append_collection_log_unlocked(
        "scheduled",
        "KIS realtime tick and orderbook collector is starting in parallel",
    )
    _kis_realtime_collector_worker = threading.Thread(
        target=_kis_realtime_collector_loop,
        name="kis-realtime-collector",
        daemon=True,
    )
    _kis_realtime_collector_worker.start()


def _start_kis_overseas_realtime_collector() -> None:
  global _kis_overseas_realtime_worker
  if not _env_flag("KIS_OVERSEAS_REALTIME_ENABLED", True):
    return
  with _live_lock:
    if _kis_overseas_realtime_worker is not None and _kis_overseas_realtime_worker.is_alive():
      return
    _kis_overseas_realtime_stop.clear()
    _kis_overseas_realtime_state.update({"running": True, "last_error": None})
    _append_collection_log_unlocked(
        "scheduled",
        "KIS overseas HDFSCNT0/HDFSASP0 WebSocket collector is starting",
    )
    _kis_overseas_realtime_worker = threading.Thread(
        target=_kis_overseas_realtime_collector_loop,
        name="kis-overseas-realtime-collector",
        daemon=True,
    )
    _kis_overseas_realtime_worker.start()


def _refresh_us_exchange_map_if_stale() -> None:
  """Rebuild the US ticker→exchange map when missing or older than the max age.

  Best-effort: a network failure leaves the existing CSV in place (build returns {}).
  """
  from app.backtesting.accelerated_demo import US_EXCHANGE_MAP_CACHE, build_us_exchange_map

  max_age_days = max(0.0, float(os.getenv("US_EXCHANGE_MAP_MAX_AGE_DAYS", "7")))
  try:
    age_days = (time.time() - os.path.getmtime(US_EXCHANGE_MAP_CACHE)) / 86400.0
    if age_days < max_age_days:
      return
  except OSError:
    pass  # missing → rebuild
  mapping = build_us_exchange_map()
  audit.record("us_exchange_map_refreshed", {"rows": len(mapping)})


def _start_us_exchange_map_refresher() -> None:
  def worker() -> None:
    interval = max(3600.0, float(os.getenv("US_EXCHANGE_MAP_REFRESH_SECONDS", "86400")))
    while True:
      try:
        _refresh_us_exchange_map_if_stale()
      except Exception as exc:  # noqa: BLE001 - refresher must never kill the process.
        audit.record("us_exchange_map_refresh_failed", {"error": str(exc) or exc.__class__.__name__})
      time.sleep(interval)

  threading.Thread(target=worker, name="us-exchange-map-refresher", daemon=True).start()


_investor_flow_refresh_status: dict[str, Any] = {
    "last_run_at": None,
    "last_result": None,
    "last_error": None,
    "runs": 0,
}


def _refresh_investor_flow_once() -> dict[str, Any]:
  """Refresh the daily investor-flow store (개인/외국인/기관 순매수).

  ``residual_relative_strength`` requires informed-flow evidence, and KIS reports
  it per business day, so the store needs a daily top-up rather than a tick feed.
  A single call per symbol returns ~30 business days, which is why this is cheap
  enough to run on a schedule and still self-heals a gap after downtime.
  """
  from app.data.investor_flow_collector import refresh_investor_flow

  result = refresh_investor_flow(
      minimum_bars=_auto_reliability_int("INVESTOR_FLOW_MINIMUM_BARS", 100),
      delay_seconds=max(0.0, _env_float_web("INVESTOR_FLOW_REQUEST_DELAY_SEC", 0.3)),
  )
  payload = result.as_dict()
  with _live_lock:
    _investor_flow_refresh_status["last_run_at"] = datetime.now(timezone.utc).isoformat()
    _investor_flow_refresh_status["last_result"] = payload
    _investor_flow_refresh_status["last_error"] = None
    _investor_flow_refresh_status["runs"] = int(
        _investor_flow_refresh_status.get("runs") or 0
    ) + 1
  audit.record("investor_flow_refreshed", payload)
  return payload


def _start_investor_flow_refresher() -> None:
  """Daily background refresh of the investor-flow store.

  Runs once shortly after boot so a fresh install backfills without an operator
  step, then on ``INVESTOR_FLOW_REFRESH_SECONDS`` (default 6h). Six hours rather
  than exactly 24 because the CURRENT business day's figures keep changing while
  the session runs, so a once-a-day read would leave today's flow stale for most
  of the day it is needed.
  """

  def worker() -> None:
    # Let the KIS client, token and bar store settle before the first call; a
    # refresh racing startup would fail on a half-initialised client and look
    # like a broker error.
    startup_delay = max(0.0, _env_float_web("INVESTOR_FLOW_STARTUP_DELAY_SEC", 45.0))
    if _live_training_stop.wait(startup_delay):
      return
    interval = max(600.0, _env_float_web("INVESTOR_FLOW_REFRESH_SECONDS", 21_600.0))
    while True:
      try:
        _refresh_investor_flow_once()
      except Exception as exc:  # noqa: BLE001 - refresher must never kill the process.
        message = str(exc) or exc.__class__.__name__
        with _live_lock:
          _investor_flow_refresh_status["last_error"] = message
        audit.record("investor_flow_refresh_failed", {"error": message})
      if _live_training_stop.wait(interval):
        return

  threading.Thread(target=worker, name="investor-flow-refresher", daemon=True).start()


_weekend_brief_status: dict[str, Any] = {
    "last_run_at": None,
    "last_action": None,
    "last_error": None,
    "runs": 0,
}


def _krx_reference_symbols() -> tuple[str, ...]:
  """Liquid KRX names used as the market proxy for the Monday gap."""
  anchors = _realtime_session_anchor_symbols()
  if anchors:
    return anchors
  limit = max(1, _auto_reliability_int("REALTIME_SESSION_ANCHOR_MAX", 2, 1))
  candidates: list[str] = []
  try:
    candidates.extend(_cached_domestic_ranking_symbols())
  except Exception:  # noqa: BLE001 - reference discovery is best-effort.
    pass
  try:
    candidates.extend(
        RealtimeMarketDataStore().active_symbols(
            datetime.now(timezone.utc) - timedelta(days=1),
            limit=limit * 4,
        )
    )
  except Exception:  # noqa: BLE001 - listed-universe fallback remains available.
    pass
  if not candidates:
    try:
      candidates.extend(load_krx_listed_universe(limit=limit * 4))
    except Exception:  # noqa: BLE001 - unknown is safer than fabricated evidence.
      pass
  return tuple(
      dict.fromkeys(
          str(symbol).strip().zfill(6)
          for symbol in candidates
          if str(symbol).strip().isdigit() and len(str(symbol).strip()) <= 6
      )
  )[:limit]


def _session_move_bps(symbols: tuple[str, ...], since: datetime, until: datetime) -> float | None:
  """Average close-to-close move across ``symbols`` inside a window, in bps.

  Returns ``None`` when no symbol has both ends — an unmeasurable move must not be
  reported as 0.0, which would read as "flat" rather than "unknown".
  """
  moves: list[float] = []
  try:
    store = RealtimeMarketDataStore()
  except Exception:  # noqa: BLE001
    return None
  for symbol in symbols:
    try:
      bars = store.recent_minute_bars(symbol, since, limit=2000)
    except Exception:  # noqa: BLE001
      continue
    inside = [bar for bar in bars or () if since <= bar.minute_start < until]
    if len(inside) < 2:
      continue
    first = float(getattr(inside[0], "open", 0.0) or getattr(inside[0], "close", 0.0) or 0.0)
    last = float(getattr(inside[-1], "close", 0.0) or 0.0)
    if first > 0 and last > 0:
      moves.append((last / first - 1.0) * 10_000.0)
  if not moves:
    return None
  return sum(moves) / len(moves)


def _monday_open_gap_bps(window: Any) -> float | None:
  """Realized KRX gap: Monday's opening print against Friday's closing print."""
  symbols = _krx_reference_symbols()
  gaps: list[float] = []
  try:
    store = RealtimeMarketDataStore()
  except Exception:  # noqa: BLE001
    return None
  for symbol in symbols:
    try:
      bars = store.recent_minute_bars(
          symbol, window.start - timedelta(hours=8), limit=4000
      )
    except Exception:  # noqa: BLE001
      continue
    before = [bar for bar in bars or () if bar.minute_start < window.start]
    after = [bar for bar in bars or () if bar.minute_start >= window.end]
    if not before or not after:
      continue
    friday_close = float(getattr(before[-1], "close", 0.0) or 0.0)
    monday_open = float(getattr(after[0], "open", 0.0) or getattr(after[0], "close", 0.0) or 0.0)
    if friday_close > 0 and monday_open > 0:
      gaps.append((monday_open / friday_close - 1.0) * 10_000.0)
  if not gaps:
    return None
  return sum(gaps) / len(gaps)


def _run_weekend_brief_once() -> dict[str, Any]:
  """Refresh the weekend prior, or score it once Monday has opened.

  Both markets are shut from the KRX Friday close to the KRX Monday open, so this
  is the one window with spare compute and no trading to disturb. The output is a
  committed claim about the Monday gap that is graded afterwards — an unscored
  weekend analysis is indistinguishable from a wrong one.
  """
  from app.research.weekend_brief import (
      WeekendBriefStore,
      build_monday_prior,
      collect_weekend_signals,
      weekend_window,
  )

  now = datetime.now(timezone.utc)
  store = WeekendBriefStore()
  window = weekend_window(now)

  if window is None:
    # Outside the closed window: the only outstanding work is grading the last
    # prior once Monday's opening print exists.
    latest = store.latest_prior()
    if not latest or latest.get("scored_at"):
      return {"action": "IDLE"}
    from app.research.weekend_brief import weekend_window as _ww

    reopened = _ww(datetime.fromisoformat(latest["computed_at"]))
    if reopened is None:
      return {"action": "IDLE"}
    gap = _monday_open_gap_bps(reopened)
    if gap is None:
      return {"action": "AWAITING_OPEN_PRICE", "window_key": latest["window_key"]}
    score = store.record_score(latest["window_key"], gap)
    payload = {"action": "SCORED", "score": score, "track_record": store.track_record()}
    audit.record("weekend_brief_scored", payload)
    return payload

  # Inside the window: deepen the inputs first, then (re)build the prior.
  # Enrichment runs here because both venues are shut — the LLM pass is the
  # heaviest thing this process does and must not compete with a trading session.
  enrichment = _run_weekend_enrichment(window)

  from app.research.weekend_brief import us_session_move_bps

  us_move, proxy = us_session_move_bps(window)
  signals = collect_weekend_signals(window, us_session_move_bps=us_move)
  prior = build_monday_prior(signals, computed_at=now)
  store.save_prior(prior)
  payload = {
      "action": "PRIOR_UPDATED",
      "prior": prior.as_dict(),
      "us_proxy": proxy,
      "enrichment": enrichment,
  }
  audit.record("weekend_brief_updated", payload)
  return payload


def _run_weekend_enrichment(window: Any) -> dict[str, Any]:
  """Deepen macro history and de-saturate event sentiment, weekend only.

  Both are best-effort: the brief must still be produced if an external source is
  unreachable, and a partial enrichment is reported rather than hidden.
  """
  from app.research.weekend_enrichment import (
      backfill_macro_history,
      load_saturated_events,
      reclassify_events,
  )

  summary: dict[str, Any] = {}

  # --- FRED history -------------------------------------------------------
  try:
    config = json.loads(DEFAULT_RESEARCH_CONFIG.read_text(encoding="utf-8"))
    series = [
        (str(item.get("series_id")), str(item.get("name")))
        for item in (config.get("fred_series") or [])
        if item.get("series_id") and item.get("name")
    ]
    if series:
      from app.storage import LocalResearchStore

      backfill = backfill_macro_history(
          series=series,
          store=LocalResearchStore(),
          limit=max(10, _auto_reliability_int("WEEKEND_MACRO_HISTORY_LIMIT", 120, 10)),
      )
      summary["macro_backfill"] = backfill.as_dict()
  except Exception as exc:  # noqa: BLE001 - enrichment must never break the brief
    summary["macro_backfill_error"] = f"{type(exc).__name__}: {exc}"

  # --- LLM re-classification ----------------------------------------------
  try:
    # Deliberately small per pass. qwen2.5:1.5b on CPU takes seconds per headline,
    # so a 120-event batch ran past 10 minutes and made the manual endpoint appear
    # hung. The worker fires hourly, so ~40 per pass still clears far more than a
    # weekend's event volume across Sat-Sun while keeping any single pass bounded.
    limit = max(5, _auto_reliability_int("WEEKEND_RECLASSIFY_LIMIT", 40, 5))
    events = load_saturated_events(
        since_iso=window.start.astimezone(timezone.utc).isoformat(),
        until_iso=window.end.astimezone(timezone.utc).isoformat(),
        limit=limit,
    )
    if events:
      result = reclassify_events(events)
      summary["reclassification"] = result.as_dict()
  except Exception as exc:  # noqa: BLE001
    summary["reclassification_error"] = f"{type(exc).__name__}: {exc}"

  return summary


def _start_weekend_brief_worker() -> None:
  """Sat-Mon weekend research loop.

  Runs on a plain interval rather than a cron: the interesting states (inside the
  closed window / Monday open reached) are decided from the clock inside
  ``_run_weekend_brief_once``, so a missed tick self-heals on the next one instead
  of skipping a weekend entirely.
  """

  def worker() -> None:
    startup_delay = max(0.0, _env_float_web("WEEKEND_BRIEF_STARTUP_DELAY_SEC", 60.0))
    if _live_training_stop.wait(startup_delay):
      return
    interval = max(300.0, _env_float_web("WEEKEND_BRIEF_INTERVAL_SECONDS", 3600.0))
    while True:
      try:
        result = _run_weekend_brief_once()
        with _live_lock:
          _weekend_brief_status["last_run_at"] = datetime.now(timezone.utc).isoformat()
          _weekend_brief_status["last_action"] = result.get("action")
          _weekend_brief_status["last_error"] = None
          _weekend_brief_status["runs"] = int(
              _weekend_brief_status.get("runs") or 0
          ) + 1
      except Exception as exc:  # noqa: BLE001 - research must never kill the process.
        message = str(exc) or exc.__class__.__name__
        with _live_lock:
          _weekend_brief_status["last_error"] = message
        audit.record("weekend_brief_failed", {"error": message})
      if _live_training_stop.wait(interval):
        return

  threading.Thread(target=worker, name="weekend-brief", daemon=True).start()


def _stop_kis_realtime_collector() -> None:
  worker: threading.Thread | None
  _kis_realtime_collector_stop.set()
  with _live_lock:
    worker = _kis_realtime_collector_worker
    _append_collection_log_unlocked("stopped", "KIS realtime collector stopped")
  if worker is not None:
    worker.join(timeout=2.0)


def _stop_kis_overseas_realtime_collector() -> None:
  _kis_overseas_realtime_stop.set()
  with _live_lock:
    worker = _kis_overseas_realtime_worker
    _kis_overseas_realtime_state["running"] = False
    _append_collection_log_unlocked("stopped", "KIS overseas realtime collector stopped")
  if worker is not None:
    worker.join(timeout=2.0)


def _live_training_loop() -> None:
  """실시간 수집 데이터로 라이브 단기 모델을 주기적으로 재학습·배포한다.

  - 데이터 수집: KIS 실시간 수집기 + 트레이딩 엔진 평가 프레임 저널링(실시간·상시).
  - 학습: 이 루프가 주기(LIVE_TRAINING_INTERVAL_SECONDS)마다 수행 — 트레이딩과 독립 스레드.
  registry.save는 적격 모델만 latest를 원자적으로 교체하므로, 부적격 재학습은 기존 모델을 보존하고
  라이브 예측기(매 예측마다 latest 재로딩)는 재시작 없이 개선된 모델을 자동 반영한다.
  """
  startup_delay = max(
      0.0,
      _env_float_web("LIVE_TRAINING_STARTUP_DELAY_SECONDS", 20.0),
  )
  if _live_training_stop.wait(startup_delay):
    return
  while not _live_training_stop.is_set():
    next_wait_seconds = float(LIVE_TRAINING_INTERVAL_SECONDS)
    with _live_lock:
      _live_training_heartbeat.update(
          {
              "started_at": datetime.now(timezone.utc).isoformat(),
              "error": None,
          }
      )
    try:
      minimum_rows_before_skip = max(
          30,
          _auto_reliability_int("LIVE_TRAINING_BACKFILL_MIN_MATERIALIZED_ROWS", 1_000, 30),
      )
      stored_training_rows = materialized_training_row_count()
      if stored_training_rows >= minimum_rows_before_skip:
        backfill = {
            "built": 0,
            "attempted": 0,
            "errors": {},
            "reason": "MATERIALIZED_HISTORY_SUFFICIENT",
            "materialized_training_rows": stored_training_rows,
        }
      else:
        backfill = backfill_live_feature_frames_from_realtime_store()
      collection = collect_live_feature_frames_from_realtime_store()
      artifact = train_live_short_horizon_from_collected_features()
      metrics = artifact.get("metrics") or {}
      deployment = artifact.get("deployment") or {}
      training_data = artifact.get("training_data") or {}
      skipped = bool(artifact.get("training_skipped"))
      promoted = bool(deployment.get("promoted"))
      if skipped:
        status = "waiting"
        message = "Live model training skipped (no newly labelled data)"
      elif promoted:
        status = "complete"
        message = "Live short-horizon model retrained and deployed"
      elif artifact.get("live_eligible"):
        status = "complete"
        message = "Live model challenger qualified but active model remained better"
      else:
        status = "running"
        message = "Live short-horizon model retrained (kept previous eligible model)"
      with _live_lock:
        _live_training_heartbeat.update(
            {
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "ok": True,
                "skipped": skipped,
                "artifact_id": artifact.get("artifact_id"),
                "error": None,
            }
        )
        _append_collection_log_unlocked(
            status,
            message,
            counts={
                "feature_frames_built": (
                    int(backfill.get("built", 0) or 0)
                    + int(collection.get("built", 0) or 0)
                ),
                "historical_frames_backfilled": int(backfill.get("built", 0) or 0),
                "training_rows": int(float(metrics.get("example_count", 0) or 0)),
                "fresh_training_rows": int(
                    training_data.get("new_materialized_row_count") or 0
                ),
                "evaluated_training_rows": int(training_data.get("fresh_row_count") or 0),
                "auc": round(float(metrics.get("auc", 0.0) or 0.0), 4),
                "precision_at_k": round(float(metrics.get("precision_at_k", 0.0) or 0.0), 4),
                "avg_forward_net_return_bps_top_k": round(
                    float(metrics.get("avg_forward_net_return_bps_top_k", 0.0) or 0.0),
                    4,
                ),
                "live_eligible": bool(artifact.get("live_eligible")),
                "training_skipped": skipped,
                "skip_reason": artifact.get("skip_reason"),
                "deployed": promoted,
                "deployment_reason": deployment.get("reason"),
                "artifact_id": artifact.get("artifact_id"),
            },
        )
    except Exception as exc:  # noqa: BLE001 - 학습 실패가 서버/트레이딩을 죽여서는 안 된다.
      error_message = str(exc) or exc.__class__.__name__
      if "database is locked" in error_message.lower():
        # Startup and periodic store maintenance can briefly own SQLite's writer
        # lock. Retry promptly; waiting the normal five-minute cadence would make
        # one transient collision look like a stopped learner.
        next_wait_seconds = max(
            5.0,
            min(
                30.0,
                _env_float_web("LIVE_TRAINING_DB_LOCK_RETRY_SECONDS", 15.0),
            ),
        )
      with _live_lock:
        _live_training_heartbeat.update(
            {
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "ok": False,
                "skipped": False,
                "error": error_message,
            }
        )
        _append_collection_log_unlocked(
            "error",
            f"Live training cycle failed: {error_message}",
        )
    _live_training_stop.wait(next_wait_seconds)


def _start_live_training_worker() -> None:
  global _live_training_worker
  with _live_lock:
    if _live_training_worker is not None and _live_training_worker.is_alive():
      return
    _live_training_stop.clear()
    _append_collection_log_unlocked(
        "scheduled",
        f"Periodic live model training starting (every {LIVE_TRAINING_INTERVAL_SECONDS}s)",
    )
    _live_training_worker = threading.Thread(
        target=_live_training_loop,
        name="live-model-training",
        daemon=True,
    )
    _live_training_worker.start()


def _stop_live_training_worker() -> None:
  worker: threading.Thread | None
  _live_training_stop.set()
  with _live_lock:
    worker = _live_training_worker
  if worker is not None:
    worker.join(timeout=2.0)


def _build_realtime_trading_engine() -> RealtimeTradingEngine:
  from app.trading.strategy_session import StrategySessionManager

  store = RealtimeMarketDataStore()
  account = _live_account_snapshot_for_analysis()
  rules = _live_risk_rules_for_account(account)
  broker_client = KisDevelopersApiClient(paper=False, enabled=True)

  def _refresh_market_snapshot(symbol: str, market: str, decision_time: datetime) -> MarketSnapshot | None:
    try:
      return broker_client.get_market_snapshot(symbol, market, company_name=symbol, sector="Unknown")
    except Exception:
      return None

  decision_engine = SharedLiveDecisionEngine(
      store,
      risk_manager=RiskManager(rules),
      market_refresher=_refresh_market_snapshot,
  )
  coordinator = LiveExecutionCoordinator(broker_client)
  _ensure_us_fast_poll_started()
  macro_micro_observer = _build_macro_micro_observer(decision_engine)
  strategy_session_manager = StrategySessionManager(
      selection_evidence_provider=_strategy_session_selection_evidence,
  )
  return RealtimeTradingEngine(
      decision_engine=decision_engine,
      coordinator=coordinator,
      account_provider=_realtime_engine_account_snapshot,
      candidate_symbols_provider=_realtime_engine_buy_candidates,
      session_open_provider=lambda: bool(_active_live_market_groups()),
      ontology_graph_provider=_latest_ontology_graph,
      market_open_provider=_is_open_live_market_ticker,
      cycle_observer=_record_realtime_trading_cycle,
      macro_micro_observer=macro_micro_observer,
      strategy_session_manager=strategy_session_manager,
  )


def _strategy_session_selection_evidence(symbols: tuple[str, ...]) -> dict[str, Any]:
  """Generate and return current ontology/GNN evidence for election candidates.

  The live engine's US fast-poll path does not traverse the websocket event
  runtime, so merely reading its shadow log leaves US evidence frozen at the
  last domestic websocket event.  Build from the same validated live feature
  frame used by the decision engine before reading the persisted comparison.
  """
  wanted = {str(symbol or "").upper() for symbol in symbols}
  if not wanted:
    return {}
  observed_at = datetime.now(timezone.utc)
  frames: dict[str, Any] = {}
  try:
    from app.features.live_feature_frame import LiveFeatureFrameBuilder

    builder = LiveFeatureFrameBuilder(RealtimeMarketDataStore())
    for symbol in sorted(wanted):
      try:
        frames[symbol] = builder.build(symbol, decision_time=observed_at)
      except Exception as exc:  # closed-world: no valid frame means no new evidence.
        _record_live_shadow_error(symbol, exc, observed_at)
    try:
      # Collect the new absorption thesis independently of ontology/GNN election.
      # This is a shadow-only journal/simulator path and cannot submit an order.
      from app.trading.mechanical_shadow import default_mechanical_shadow_collector

      default_mechanical_shadow_collector().collect(
          frames.values(),
          observed_at=observed_at,
      )
    except Exception as exc:
      # The experimental collector is never allowed to starve the established
      # ontology/GNN evidence refresh.
      _record_live_shadow_error("_mechanical_shadow", exc, observed_at)
    _refresh_live_candidate_shadow(frames, observed_at)
  except Exception as exc:
    _record_live_shadow_error("_pipeline", exc, observed_at)
  try:
    latest = ((build_refactor_dashboard().get("shadow") or {}).get("latest_by_symbol") or {})
  except Exception:
    latest = {}
  rows = {
      symbol: dict(row)
      for symbol, row in latest.items()
      if str(symbol).upper() in wanted and isinstance(row, dict)
  }
  # The shadow comparison predates slow completed-bar features. Enrich its
  # ontology evidence at election time from the same validated KIS live frame
  # used by SharedDecisionEngine; no synthetic/default RVGI values are allowed.
  try:
    from app.features.live_feature_frame import LiveFeatureFrameBuilder
    from app.routing.actions import StrategyRoutingAction
    from app.technical.feature_builder import technical_feature_set_from_live_frame

    for symbol in wanted:
      row = rows.setdefault(symbol, {"symbol": symbol, "decisions": []})
      try:
        frame = frames[symbol]
        features = technical_feature_set_from_live_frame(frame, symbol)
      except Exception:
        continue
      box_width_ok = (
          features.box_width_pct is not None
          and 0.002 <= features.box_width_pct <= 0.04
      )
      above_box = (
          features.price is not None
          and features.box_high is not None
          and features.price >= features.box_high
      )
      volume_confirmed = bool(
          features.volume_spike_ratio is not None
          and features.volume_spike_ratio >= 1.5
      )
      eligible = bool(
          features.rvgi is not None
          and features.rvgi_signal is not None
          and features.rvgi > features.rvgi_signal
          and features.rvgi_bullish_cross
          and box_width_ok
          and above_box
          and volume_confirmed
          and (features.liquidity_score or 0.0) > 0
          and features.spread_bps is not None
      )
      context = {
          "ontology_eligible": eligible,
          "rvgi": features.rvgi,
          "rvgi_signal": features.rvgi_signal,
          "rvgi_diff": features.rvgi_diff,
          "rvgi_bullish_cross": features.rvgi_bullish_cross,
          "box_high": features.box_high,
          "box_low": features.box_low,
          "box_mid": features.box_mid,
          "box_width_pct": features.box_width_pct,
          "box_position": features.box_position,
          "box_context_timestamp": features.box_context_timestamp,
          "box_previous_close": features.box_previous_close,
          "volume_confirmed": volume_confirmed,
          "breakout_distance_bps": features.breakout_distance_bps,
      }
      row["rvgi_box_context"] = context
      row["rvgi_box_as_of"] = observed_at.isoformat()
      if eligible:
        row["as_of"] = row.get("as_of") or observed_at.isoformat()
        decisions = [
            item for item in list(row.get("decisions") or ())
            if not (
                isinstance(item, dict)
                and item.get("path") == "ontology"
                and item.get("strategy_id") == "rvgi_box_breakout"
            )
        ]
        decisions.insert(
            0,
            {
                "path": "ontology",
                "action": StrategyRoutingAction.ACTIVATE_STRATEGY.value,
                "strategy_id": "rvgi_box_breakout",
                "utility": None,
                "reason_codes": [
                    "RVGI_BULLISH_CROSS",
                    "CAUSAL_BOX_BREAKOUT",
                    "BREAKOUT_VOLUME_CONFIRMED",
                    "EVIDENCE_CLUSTER:breakout_cluster",
                ],
            },
        )
        row["decisions"] = decisions
  except Exception:
    pass
  return rows


def _refresh_live_candidate_shadow(
    frames: dict[str, Any],
    observed_at: datetime,
) -> None:
  """Run the shared ontology+GNN pipeline for valid live election frames."""
  global _live_shadow_service
  if not frames:
    return
  flags = RefactorFeatureFlags.from_env()
  require_live_gnn = os.getenv(
      "STRATEGY_SESSION_REQUIRE_LIVE_GNN",
      "true",
  ).strip().lower() in {"1", "true", "yes", "on"}
  enabled = bool(flags.ontology_router or flags.gnn_shadow or require_live_gnn)
  with _live_shadow_lock:
    _live_shadow_state["enabled"] = enabled
    _live_shadow_state["last_attempt_at"] = observed_at.isoformat()
    if not enabled:
      return
    if _live_shadow_service is None:
      from app.routing.shadow_intelligence import ShadowIntelligenceService

      try:
        interval = max(
            1.0,
            float(os.getenv("REALTIME_STRATEGY_SHADOW_INTERVAL_SECONDS", "5.0")),
        )
      except (TypeError, ValueError):
        interval = 5.0
      _live_shadow_service = ShadowIntelligenceService(
          feature_dim=28,
          minimum_interval_seconds=interval,
          enable_npu_comparison=flags.npu_inference,
      )

    from app.routing.shadow_intelligence import slow_snapshot_from_live_feature_frame

    for symbol, frame in sorted(frames.items()):
      try:
        snapshot = slow_snapshot_from_live_feature_frame(frame)
        result = _live_shadow_service.evaluate(snapshot)
      except Exception as exc:  # one symbol must never kill the trading cycle.
        _record_live_shadow_error(symbol, exc, observed_at, lock_held=True)
        continue
      if result is None:
        continue
      _live_shadow_state["last_success_at"] = observed_at.isoformat()
      _live_shadow_state["last_symbol"] = symbol
      _live_shadow_state["generated"] = int(
          _live_shadow_state.get("generated") or 0
      ) + 1
      errors = dict(_live_shadow_state.get("errors") or {})
      errors.pop(symbol, None)
      _live_shadow_state["errors"] = errors


def _record_live_shadow_error(
    symbol: str,
    exc: Exception,
    observed_at: datetime,
    *,
    lock_held: bool = False,
) -> None:
  """Keep diagnostics without allowing advisory inference to stop execution."""
  def update() -> None:
    errors = dict(_live_shadow_state.get("errors") or {})
    errors[str(symbol)] = {
        "code": type(exc).__name__,
        "detail": str(exc)[:240],
        "at": observed_at.isoformat(),
    }
    _live_shadow_state["errors"] = errors
    _live_shadow_state["last_attempt_at"] = observed_at.isoformat()

  if lock_held:
    update()
  else:
    with _live_shadow_lock:
      update()


def _realtime_engine_account_snapshot() -> AccountSnapshot | None:
  """Use the controller's recent authoritative account without blocking a 1s cycle."""
  cached = _account_snapshot_from_live_basis(_last_live_account_basis())
  return cached if cached is not None else _live_account_snapshot_for_analysis()


def _realtime_engine_buy_candidates() -> tuple[str, ...]:
  """Return only already-streaming candidates; never run broker scans in the engine.

  Cached rankings/watchlists are ordering hints, not proof of live data.  A
  previously ranked symbol must also appear in ``active_symbols`` inside the
  freshness window; otherwise macro reasoning receives an empty time series and
  permanently reports ``MACRO_INSUFFICIENT_DATA`` despite a healthy websocket.
  """
  max_age = max(5.0, _env_float_web("REALTIME_BUY_CANDIDATE_MAX_AGE_SEC", 120.0))
  limit = max(1, _auto_reliability_int("REALTIME_MAX_BUY_EVALUATIONS_PER_CYCLE", 8))
  try:
    since = datetime.now(timezone.utc) - timedelta(seconds=max_age)
    fresh = tuple(RealtimeMarketDataStore().active_symbols(since, limit=max(32, limit * 4)))
  except Exception:
    fresh = ()
  fresh_set = {str(symbol or "").upper() for symbol in fresh}
  try:
    account = _realtime_engine_account_snapshot()
    store = RealtimeMarketDataStore()
  except Exception:
    account = None
    store = None
  with _live_lock:
    sticky_us = tuple(_us_learning_watchlist_cache.get("symbols") or ())
  cached_context = _cached_context_buy_candidates(
      limit=max(16, limit * 2),
      account=account,
  )
  domestic_ranked = tuple(_cached_domestic_ranking_symbols() or ())
  ordered = _prioritize_realtime_buy_candidates(
      tuple(dict.fromkeys((*cached_context, *domestic_ranked, *sticky_us, *fresh))),
      account=account,
  )
  selected: list[str] = []
  for symbol in ordered:
    if str(symbol or "").upper() not in fresh_set:
      continue
    if not _is_live_buy_candidate_symbol(symbol) or not _is_open_live_market_ticker(symbol):
      continue
    group = _ticker_market_group_for_live_trading(symbol)
    if group == "KRX" and not _is_live_market_core_open("KRX"):
      continue
    if (
        store is not None
        and group == "KRX"
        and not _candidate_has_strategy_feature_history(symbol, store)
    ):
      continue
    if (
        store is not None
        and group == "KRX"
        and not _candidate_has_fresh_buy_orderbook(symbol, store)
    ):
      continue
    if account is not None and store is not None and not _cached_candidate_affordable(
        symbol,
        account,
        store,
    ):
      continue
    selected.append(symbol)
    if len(selected) >= limit:
      break
  return tuple(selected)


def _cached_candidate_affordable(
    symbol: str,
    account: AccountSnapshot,
    store: RealtimeMarketDataStore,
) -> bool:
  """Cheap pre-slot affordability check using the already-persisted live tick."""
  try:
    tick = store.latest_tick(symbol)
  except Exception:
    return True
  price = float(getattr(tick, "price", 0.0) or 0.0) if tick is not None else 0.0
  if price <= 0:
    return True
  group = _ticker_market_group_for_live_trading(symbol, "")
  currency = "KRW" if group == "KRX" else "USD"
  cash = _account_available_cash(account, currency)
  buffer_rate = max(1.0, _env_float_web("REALTIME_AFFORDABILITY_BUFFER_RATE", 1.01))
  return cash + 1e-9 >= price * buffer_rate


_us_fast_poll_thread = None
_us_fast_poll_state: dict[str, Any] = {
    "last_attempt_at": None,
    "last_success_at": None,
    "symbols": (),
    "saved_ticks": 0,
    "saved_orderbooks": 0,
    "errors": {},
    "last_error": None,
}
_krx_feature_frame_thread: threading.Thread | None = None
_krx_feature_frame_stop = threading.Event()
_krx_feature_last_signature: dict[str, tuple[str, str]] = {}


def _ensure_us_fast_poll_started() -> None:
  """Start a US-only fast polling loop (separate from the ~minutes-long collection
  cycle) so HELD US symbols get sub-minute realtime rows in the store, enough for
  live feature frames / micro reasoning / exit signals. Idempotent; daemon thread."""
  global _us_fast_poll_thread
  import threading
  if os.getenv("REALTIME_US_FAST_POLL_ENABLED", "true").strip().lower() in {"0", "false", "no", "off"}:
    return
  if _us_fast_poll_thread is not None and _us_fast_poll_thread.is_alive():
    return
  t = threading.Thread(target=_us_fast_poll_loop, name="us-fast-poll", daemon=True)
  _us_fast_poll_thread = t
  t.start()


def _ensure_krx_feature_frame_started() -> None:
  """Journal fresh KRX feature frames independently from the 5-minute trainer."""
  global _krx_feature_frame_thread
  if os.getenv("LIVE_KRX_FEATURE_FRAME_ENABLED", "true").strip().lower() in {
      "0",
      "false",
      "no",
      "off",
  }:
    return
  if _krx_feature_frame_thread is not None and _krx_feature_frame_thread.is_alive():
    return
  _krx_feature_frame_stop.clear()
  _krx_feature_frame_thread = threading.Thread(
      target=_krx_feature_frame_loop,
      name="krx-feature-frame",
      daemon=True,
  )
  _krx_feature_frame_thread.start()


def _stop_krx_feature_frame_worker() -> None:
  _krx_feature_frame_stop.set()


def _krx_feature_frame_symbols() -> tuple[str, ...]:
  with _live_lock:
    complete = tuple(_kis_realtime_complete_symbols)
  watched = _dashboard_krx_watch_symbols()
  limit = max(1, _auto_reliability_int("LIVE_KRX_FEATURE_FRAME_SYMBOL_LIMIT", 4))
  return tuple(dict.fromkeys((*watched, *complete)))[:limit]


def _krx_feature_frame_loop() -> None:
  interval = max(2.0, _env_float_web("LIVE_KRX_FEATURE_FRAME_SECONDS", 5.0))
  # Startup launches several SQLite-backed workers together. Initial schema
  # maintenance can briefly hold the writer lock; retry instead of letting this
  # one daemon die permanently before the first feature frame is sampled.
  store = None
  while store is None and not _krx_feature_frame_stop.is_set():
    try:
      store = RealtimeMarketDataStore()
    except Exception:  # noqa: BLE001 - transient SQLite startup contention.
      if _krx_feature_frame_stop.wait(min(5.0, interval)):
        return
  if store is None:
    return
  while not _krx_feature_frame_stop.is_set():
    for symbol in _krx_feature_frame_symbols():
      try:
        tick = store.latest_tick(symbol)
        book = store.latest_orderbook(symbol)
        signature = (
            str(getattr(tick, "record_id", "") or ""),
            str(getattr(book, "record_id", "") or ""),
        )
        if not any(signature) or _krx_feature_last_signature.get(symbol) == signature:
          continue
        result = collect_live_feature_frames_from_realtime_store(symbols=(symbol,))
        if int(result.get("built", 0) or 0) > 0:
          _krx_feature_last_signature[symbol] = signature
      except Exception:  # noqa: BLE001 - feature sampling must not stop market ingest.
        continue
    _krx_feature_frame_stop.wait(interval)


def _us_fast_poll_target_symbols(held: tuple[str, ...]) -> tuple[str, ...]:
  """Return stable US poll targets without ever dropping held positions."""
  if _active_operation_mode() == "live_trading":
    limit = max(1, _auto_reliability_int("AUTO_RELIABILITY_US_WARM_SYMBOLS", 4))
    # Always use the TTL-aware selector. Reading the cache directly forever
    # pinned the first few symbols and starved the wider scanner of live data.
    watched = _sticky_us_learning_symbols(limit)
    return tuple(dict.fromkeys((*held, *watched)))
  warm = _sticky_us_learning_symbols(
      _auto_reliability_int("AUTO_RELIABILITY_US_WARM_SYMBOLS", 4)
  )
  return tuple(dict.fromkeys((*held, *warm)))


def _held_us_realtime_symbols() -> tuple[str, ...]:
  try:
    acct = _account_snapshot_from_live_basis(_last_live_account_basis())
    if acct is None:
      return ()
    return tuple(
        dict.fromkeys(
            str(getattr(holding, "ticker", "") or "").upper().strip()
            for holding in (getattr(acct, "holdings", ()) or ())
            if _ticker_market_group_for_live_trading(
                str(getattr(holding, "ticker", "") or ""),
                str(getattr(holding, "market", "") or ""),
            )
            == "US"
        )
    )
  except Exception:
    return ()


def _kis_overseas_realtime_symbols() -> tuple[str, ...]:
  limit = max(1, min(20, _auto_reliability_int("KIS_OVERSEAS_REALTIME_MAX_SYMBOLS", 6)))
  if (
      _kis_overseas_observed_subscription_capacity
      and time.monotonic() - _kis_overseas_observed_capacity_at < 600.0
  ):
    limit = min(limit, max(1, _kis_overseas_observed_subscription_capacity // 2))
  return _us_fast_poll_target_symbols(_held_us_realtime_symbols())[:limit]


def _kis_overseas_realtime_collector_loop() -> None:
  global _kis_overseas_observed_subscription_capacity, _kis_overseas_observed_capacity_at
  from app.data.kis_realtime import run_kis_overseas_realtime_websocket_collector
  from app.data.market_session import MarketPhase, market_phase

  runtime_seconds = max(
      30.0,
      _env_float_web("KIS_OVERSEAS_REALTIME_RESUBSCRIBE_SECONDS", 120.0),
  )
  from app.data.kis_realtime import is_us_daytime_quote_session

  while not _kis_overseas_realtime_stop.is_set():
    if _kis_realtime_session_owner() not in {"US", "BOTH"}:
      with _live_lock:
        _kis_overseas_realtime_state.update(
            {"running": True, "symbols": (), "last_error": None, "session": "standby"}
        )
      if _kis_overseas_realtime_stop.wait(15.0):
        return
      continue
    phase = market_phase("US")
    # US daytime trading (주간거래) runs 10:00-16:00 KST, which is 21:00-03:00
    # ET — i.e. CLOSED on the US clock. Skipping on that alone meant the
    # daytime session never streamed, even though HDFSCNT0 serves it.
    daytime_quotes = is_us_daytime_quote_session()
    if phase is MarketPhase.CLOSED and not daytime_quotes:
      with _live_lock:
        _kis_overseas_realtime_state.update(
            {"running": True, "symbols": (), "last_error": None, "session": "closed"}
        )
      if _kis_overseas_realtime_stop.wait(60.0):
        return
      continue
    symbols = _kis_overseas_realtime_symbols()
    if not symbols:
      if _kis_overseas_realtime_stop.wait(10.0):
        return
      continue
    attempted_at = datetime.now(timezone.utc).isoformat()
    with _live_lock:
      _kis_overseas_realtime_state.update(
          {
              "running": True,
              "last_attempt_at": attempted_at,
              "symbols": symbols,
              "last_error": None,
              "session": "daytime" if daytime_quotes else phase.value,
          }
      )
    try:
      # Persistent session, same as the domestic collector: a resubscribe
      # re-diffs in place. This also swaps the tr_key family automatically when
      # the daytime window opens or closes (DNASAAPL <-> RBAQAAPL), because the
      # key factory is session-aware and the diff is computed on tr_keys.
      counts = asyncio.run(
          run_kis_overseas_realtime_websocket_collector(
              symbols=symbols,
              symbols_provider=_kis_overseas_realtime_symbols,
              store=RealtimeMarketDataStore(),
              client=_kis_realtime_collector_client(),
              stop_event=_kis_overseas_realtime_stop,
              resubscribe_event=_kis_overseas_realtime_resubscribe,
              max_runtime_seconds=runtime_seconds,
              progress_callback=_record_kis_overseas_realtime_progress,
              session_active_provider=lambda: (
                  _kis_realtime_session_owner() in {"US", "BOTH"}
              ),
          )
      )
      status, cycle_message = _classify_kis_overseas_collector_cycle(counts)
      succeeded = status in {"running", "complete"}
      accepted = int(counts.get("subscriptions_accepted") or 0)
      if counts.get("subscription_limit_reached") and accepted >= 2:
        _kis_overseas_observed_subscription_capacity = max(2, accepted - (accepted % 2))
        _kis_overseas_observed_capacity_at = time.monotonic()
      with _live_lock:
        _kis_overseas_realtime_state.update(
            {
                "last_success_at": (
                    datetime.now(timezone.utc).isoformat()
                    if succeeded
                    else _kis_overseas_realtime_state.get("last_success_at")
                ),
                "counts": dict(counts),
                "last_error": None,
            }
        )
        _append_collection_log_unlocked(
            status,
            cycle_message,
            counts={
                "phase": phase.value,
                "symbols": len(symbols),
                "symbol_sample": list(symbols),
                "subscriptions_accepted": accepted,
                "subscriptions_rejected": int(counts.get("subscriptions_rejected") or 0),
                "observed_capacity": _kis_overseas_observed_subscription_capacity,
                "ticks": int(counts.get("ticks") or 0),
                "orderbooks": int(counts.get("orderbooks") or 0),
            },
        )
      if counts.get("appkey_already_in_use"):
        if _kis_overseas_realtime_stop.wait(90.0):
          return
      elif counts.get("subscription_limit_reached"):
        # Reconnect once with complete trade+book pairs only. Without this
        # pause, a constrained account repeatedly opens partial subscriptions.
        if _kis_overseas_realtime_stop.wait(5.0):
          return
      elif counts.get("connection_closed"):
        if _kis_overseas_realtime_stop.wait(10.0):
          return
    except Exception as exc:
      with _live_lock:
        _kis_overseas_realtime_state.update(
            {
                "last_error": f"{exc.__class__.__name__}: {exc}",
                "counts": {},
            }
        )
        _append_collection_log_unlocked(
            "error",
            f"KIS overseas realtime collector failed: {exc.__class__.__name__}: {exc}",
        )
      if _kis_overseas_realtime_stop.wait(20.0):
        return


def _record_kis_overseas_realtime_progress(counts: dict[str, Any]) -> None:
  """Expose persistent-session control replies without waiting for disconnect."""
  global _kis_overseas_observed_subscription_capacity, _kis_overseas_observed_capacity_at
  accepted = int(counts.get("subscriptions_accepted") or 0)
  has_events = int(counts.get("ticks") or 0) + int(counts.get("orderbooks") or 0) > 0
  limit_reached = bool(counts.get("subscription_limit_reached"))
  notice: tuple[str, str] | None = None
  with _live_lock:
    _kis_overseas_realtime_state["counts"] = dict(counts)
    if accepted > 0:
      _kis_overseas_realtime_state["last_success_at"] = (
          datetime.now(timezone.utc).isoformat()
      )
    _kis_overseas_realtime_state["last_error"] = (
        None
        if accepted > 0 or has_events
        else _kis_overseas_realtime_state.get("last_error")
    )
    capacity_is_stale = (
        _kis_overseas_observed_subscription_capacity is None
        or time.monotonic() - _kis_overseas_observed_capacity_at >= 600.0
    )
    if limit_reached and accepted >= 2 and capacity_is_stale:
      capacity = max(2, accepted - (accepted % 2))
      _kis_overseas_observed_subscription_capacity = capacity
      _kis_overseas_observed_capacity_at = time.monotonic()
      _kis_overseas_realtime_state["observed_capacity"] = capacity
      notice = (
          "waiting",
          (
              "KIS overseas subscription capacity detected; "
              f"reducing to {capacity // 2} complete trade+orderbook symbol"
          ),
      )
    if _kis_overseas_observed_subscription_capacity:
      _kis_overseas_realtime_state["observed_capacity"] = (
          _kis_overseas_observed_subscription_capacity
      )
  current_symbols = (
      _kis_overseas_realtime_symbols()
      if _kis_overseas_observed_subscription_capacity
      else ()
  )
  if current_symbols:
    with _live_lock:
      _kis_overseas_realtime_state["symbols"] = current_symbols
      live_counts = dict(_kis_overseas_realtime_state.get("counts") or {})
      live_counts["active_complete_subscriptions"] = len(current_symbols) * 2
      live_counts["active_complete_symbols"] = list(current_symbols)
      _kis_overseas_realtime_state["counts"] = live_counts
  if notice is not None:
    _kis_overseas_realtime_resubscribe.set()
  if notice is not None:
    with _live_lock:
      _append_collection_log_unlocked(notice[0], notice[1], counts=dict(counts))


def _classify_kis_overseas_collector_cycle(
    counts: dict[str, Any],
) -> tuple[str, str]:
  accepted = int(counts.get("subscriptions_accepted") or 0)
  ticks = int(counts.get("ticks") or 0)
  orderbooks = int(counts.get("orderbooks") or 0)
  if counts.get("appkey_already_in_use"):
    return (
        "waiting",
        "KIS overseas websocket waiting: AppKey is held by another realtime session",
    )
  if counts.get("subscription_limit_reached"):
    return (
        "waiting" if accepted > 0 else "error",
        (
            "KIS overseas websocket partially subscribed; account subscription limit reached"
            if accepted > 0
            else "KIS overseas websocket subscription rejected: account limit reached"
        ),
    )
  if accepted > 0:
    if counts.get("connection_closed"):
      return (
          "waiting",
          "KIS overseas websocket disconnected after a valid subscription; reconnecting",
      )
    return (
        "running",
        f"KIS overseas websocket active ({accepted} subscriptions, {ticks} ticks, {orderbooks} books)",
    )
  errors = dict(counts.get("subscription_errors_by_code") or {})
  if errors:
    detail = ", ".join(f"{code}={count}" for code, count in sorted(errors.items()))
    return "error", f"KIS overseas websocket subscriptions rejected ({detail})"
  if counts.get("connection_closed"):
    return "waiting", "KIS overseas websocket closed before confirmation; reconnecting"
  return "waiting", "KIS overseas websocket cycle ended without a subscription confirmation"


def _has_fresh_us_websocket_book(symbol: str, *, max_age_seconds: float = 30.0) -> bool:
  try:
    book = RealtimeMarketDataStore().latest_orderbook(symbol)
  except Exception:
    return False
  sequence = str(getattr(book, "sequence_key", "") or "")
  received_at = getattr(book, "received_at", None)
  return bool(
      sequence.startswith("us-kis-ws:")
      and isinstance(received_at, datetime)
      and (datetime.now(timezone.utc) - received_at).total_seconds() <= max_age_seconds
  )


def _us_fast_poll_loop() -> None:
  import time as _t
  from types import SimpleNamespace

  from app.data.market_session import is_market_fully_closed

  try:
    interval = max(5, int(os.getenv("REALTIME_US_FAST_POLL_SEC", "12")))
  except (TypeError, ValueError):
    interval = 12
  try:
    closed_interval = max(60, int(os.getenv("REALTIME_US_CLOSED_FALLBACK_SEC", "300")))
  except (TypeError, ValueError):
    closed_interval = 300
  while True:
    sleep_for = interval
    try:
      held = ()
      us_open = "US" in set(_active_live_market_groups())
      us_closed = is_market_fully_closed("US")
      if us_open or us_closed:
        held = _held_us_realtime_symbols()
      if us_open:
        # Holdings must remain first-class poll targets in learning mode too.
        # Previously the learning candidate assignment discarded ``held``
        # entirely, so a held symbol could go stale while warm names stayed fresh.
        symbols = _us_fast_poll_target_symbols(held)
        # This thread is independent from the account/reliability controller, so a
        # slow portfolio request cannot stop sub-minute US market-data collection.
        from app.trading.us_realtime_bridge import refresh_us_realtime_for_context_buy_candidates
        # REST is now a watchdog/fallback. Do not overwrite a fresh WebSocket
        # event from HDFSASP0 with a polling snapshot.
        symbols = tuple(
            symbol for symbol in symbols if not _has_fresh_us_websocket_book(symbol)
        )
        if symbols:
          result = refresh_us_realtime_for_context_buy_candidates(
              SimpleNamespace(markets=(), reasoning_paths=()),
              symbols=symbols,
          )
          collect_live_feature_frames_from_realtime_store(symbols=symbols)
          saved = dict(result.get("saved") or {})
          with _live_lock:
            _us_fast_poll_state.update(
                {
                    "last_attempt_at": datetime.now(timezone.utc).isoformat(),
                    "last_success_at": (
                        datetime.now(timezone.utc).isoformat()
                        if result.get("ok")
                        else _us_fast_poll_state.get("last_success_at")
                    ),
                    "symbols": tuple(symbols),
                    "saved_ticks": int(saved.get("realtime_ticks") or 0),
                    "saved_orderbooks": int(saved.get("orderbooks") or 0),
                    "errors": dict(result.get("errors") or {}),
                    "last_error": None,
                }
            )
      elif held and us_closed:
        # Fully closed: slower REST snapshot fallback (distinct source, not
        # live-buy eligible) so held US marks stay fresh for valuation/display.
        _rest_snapshot_fallback_refresh(held, "US")
        sleep_for = closed_interval
    except Exception as exc:  # noqa: BLE001 - best-effort; never crash the poll loop.
      with _live_lock:
        _us_fast_poll_state.update(
            {
                "last_attempt_at": datetime.now(timezone.utc).isoformat(),
                "last_error": f"{exc.__class__.__name__}: {exc}",
            }
        )
    _t.sleep(sleep_for)


def _event_severity_halflife_seconds() -> float:
  """Half-life for news severity decay. A 24h TTL with no decay meant a
  half-day-old headline still blocked trading at full strength."""
  try:
    hours = float(os.getenv("LIVE_EVENT_SEVERITY_HALFLIFE_HOURS", "3"))
  except (TypeError, ValueError):
    hours = 3.0
  return max(60.0, hours * 3600.0)


def _event_keyword_severity_factor() -> float:
  """Discount applied to keyword/fallback classifications.

  ``keyword_v1_after_llm_error`` means the LLM failed and a keyword matcher
  guessed. That degraded path must not be able to halt all trading by itself.
  """
  try:
    factor = float(os.getenv("LIVE_EVENT_KEYWORD_SEVERITY_FACTOR", "0.5"))
  except (TypeError, ValueError):
    factor = 0.5
  return max(0.0, min(1.0, factor))


def _live_event_evidence(
    symbols: tuple[str, ...],
    decision_time: datetime,
) -> tuple[tuple[dict[str, Any], ...], dict[str, tuple[dict[str, Any], ...]]]:
  """Project recent classified events into macro and per-symbol ontology facts."""

  try:
    ttl_hours = max(1, int(os.getenv("LIVE_EVENT_EVIDENCE_TTL_HOURS", "24")))
  except (TypeError, ValueError):
    ttl_hours = 24
  cutoff = decision_time - timedelta(hours=ttl_hours)
  symbol_set = {str(symbol or "").upper().strip() for symbol in symbols}
  with _live_lock:
    context = _live_state.get("context")
  events = tuple(getattr(context, "events", ()) or ()) if context is not None else ()
  macro: list[dict[str, Any]] = []
  by_symbol: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in symbol_set}

  for event in events:
    event_at = getattr(event, "event_date", None)
    if not isinstance(event_at, datetime):
      continue
    if event_at.tzinfo is None:
      event_at = event_at.replace(tzinfo=timezone.utc)
    if event_at < cutoff or event_at > decision_time + timedelta(minutes=5):
      continue
    sentiment = str(
        getattr(getattr(event, "sentiment", None), "value", getattr(event, "sentiment", ""))
    ).upper()
    confidence = max(
        0.0,
        min(1.0, float(getattr(event, "classification_confidence", 0.0) or 0.0)),
    )
    age_seconds = max(0.0, (decision_time - event_at).total_seconds())
    classification_model = str(getattr(event, "classification_model", "") or "")
    event_type = str(
        getattr(getattr(event, "event_type", None), "value", getattr(event, "event_type", ""))
    )
    # Severity is NOT the classifier's confidence. Confidence says how sure the
    # model is about the *label*; it says nothing about how material the event
    # is. Decay it with age and discount degraded classifier paths so a single
    # stale or keyword-fallback headline cannot halt the whole market.
    decay = 0.5 ** (age_seconds / max(60.0, _event_severity_halflife_seconds()))
    source_factor = (
        _event_keyword_severity_factor()
        if ("keyword" in classification_model.lower() or "error" in classification_model.lower())
        else 1.0
    )
    severity = (confidence * decay * source_factor) if sentiment == "NEGATIVE" else 0.0
    evidence = {
        "event_id": str(getattr(event, "event_id", "") or ""),
        "event_type": event_type,
        "sentiment": sentiment,
        "severity": severity,
        "raw_severity": confidence if sentiment == "NEGATIVE" else 0.0,
        "age_decay": decay,
        "source_factor": source_factor,
        "confidence": confidence,
        "labels": list(getattr(event, "event_labels", ()) or ()),
        "classification_model": classification_model,
        "age_seconds": age_seconds,
    }
    tickers = {
        str(ticker or "").upper().strip()
        for ticker in tuple(getattr(event, "tickers", ()) or ())
    }
    for symbol in tickers & symbol_set:
      by_symbol.setdefault(symbol, []).append(evidence)
    # Only an explicitly MACRO-typed event is market-wide. A story whose tickers
    # failed to resolve is *unresolved*, not market-wide: treating it as macro
    # turned single-stock headlines into a market-wide BLOCK_BUY.
    if event_type.upper() == "MACRO":
      macro.append(evidence)

  return (
      tuple(sorted(macro, key=lambda item: float(item["age_seconds"]))[:50]),
      {
          symbol: tuple(sorted(rows, key=lambda item: float(item["age_seconds"]))[:20])
          for symbol, rows in by_symbol.items()
      },
  )


def _build_macro_micro_observer(decision_engine):
  """Per-cycle macro/micro ontology reasoning for live candidate control and GUI.

  Throttled to the macro loop interval. Best-effort: it builds a
  MacroMicroReasoningBundle, records it to macro_micro_feed for the dashboard,
  and returns it so the realtime engine/candidate selector can use macro blocks
  and ranked micro BUY candidates. It still never submits an order; the
  ProfitabilityGate, RiskManager, FinalTradeGate, and broker execution path stay
  authoritative.
  Returns ``None`` (disabling the hook) if the layer is disabled or unavailable.
  """
  import time as _time

  try:
    from app.graph import macro_micro_feed
    from app.graph.macro_micro_config import load_macro_micro_policy
    from app.graph.macro_reasoner import MacroMarketReasoner, MacroReasoningInput
    from app.graph.micro_reasoner import MicroSymbolReasoner, MicroReasoningInput
    from app.graph.ontology_coordinator import OntologyCoordinator
    from app.technical.feature_builder import technical_feature_set_from_live_frame
    from app.features.macro_feature_frame import macro_feature_frame_from_store

    policy = load_macro_micro_policy()
  except Exception:  # noqa: BLE001 - advisory layer optional; never break engine build.
    return None
  if not policy.enabled:
    return None

  coordinator = OntologyCoordinator(
      macro_reasoner=MacroMarketReasoner(policy.macro_config),
      micro_reasoner=MicroSymbolReasoner(policy.micro_config),
      config=policy.coordinator_config,
  )
  interval = max(1, int(policy.macro_loop_interval_seconds))
  last_run = [0.0]
  last_bundle = [None]

  def _observer(account, held_symbols, candidates, decision_time):
    now = _time.monotonic()
    if now - last_run[0] < interval:
      return last_bundle[0]  # throttle compute, but keep latest live control bundle available
    last_run[0] = now
    holdings_by_symbol = {str(getattr(h, "ticker", "")): h for h in (getattr(account, "holdings", ()) or ())}
    # Real market context aggregated across the tracked universe (no index feed
    # needed). Off-hours / no ticks -> None fields -> conservative macro regime.
    sector_of = {
        s: str(getattr(h, "sector", "") or "")
        for s, h in holdings_by_symbol.items()
        if getattr(h, "sector", None)
    }
    universe = tuple(dict.fromkeys((*candidates, *holdings_by_symbol.keys())))
    macro_event_evidence, symbol_event_evidence = _live_event_evidence(
        universe,
        decision_time,
    )
    macro_kwargs: dict = {}
    try:
        store = getattr(decision_engine, "store", None)
        if store is not None:
            frame = macro_feature_frame_from_store(store, universe, now=decision_time, sector_of=sector_of)
            macro_kwargs = frame.as_macro_kwargs()
            # Per-symbol residuals feed the REAL within-sector ranking (which
            # replaced the arbiter's global BUY rank) and the residual
            # relative-strength thesis.
            macro_kwargs["symbol_residual_returns"] = dict(frame.per_symbol_residual_return)
            macro_kwargs["symbol_long_residual_returns"] = dict(
                frame.per_symbol_residual_return_long
            )
            macro_kwargs["symbol_market_betas"] = dict(frame.per_symbol_market_beta)
            # Change-point detection runs BEFORE any strategy is scored: it decides
            # whether the models and the accumulated per-strategy history may still
            # be believed at all. A detected break narrows the macro regime to
            # DISLOCATED and widens every bandit uncertainty penalty.
            try:
                from app.graph.change_point import default_detector

                verdict = default_detector().update(
                    frame.as_change_point_channels(), timestamp=decision_time
                )
                macro_kwargs["change_point_probability"] = verdict.change_point_probability
                macro_kwargs["regime_stability"] = verdict.regime_stability
            except Exception:  # noqa: BLE001 - detection is advisory; never fatal.
                pass
    except Exception:  # noqa: BLE001 - macro features are best-effort.
        macro_kwargs = {}
    candidate_markets = {
        "KR"
        if _ticker_market_group_for_live_trading(symbol, "") == "KRX"
        else "US"
        for symbol in candidates
    }
    reasoning_market = (
        next(iter(candidate_markets))
        if len(candidate_markets) == 1
        else "GLOBAL"
    )
    macro_input = MacroReasoningInput(
        timestamp=decision_time,
        market=reasoning_market,
        candidate_universe=tuple(candidates),
        macro_news_evidence=macro_event_evidence,
        provenance={"sector_of": sector_of},
        **macro_kwargs,
    )

    def _builder(symbol, macro_result):
      features = None
      frame = None
      technical_feature_error_code = None
      broker_quote = None
      realtime_tick = None
      quote_age_seconds = None
      try:
        frame = decision_engine.feature_builder.build(symbol, decision_time=decision_time)
        features = technical_feature_set_from_live_frame(frame, symbol)
      except Exception as exc:  # noqa: BLE001 - off-hours / missing data -> NO_TRADE micro result.
        features = None
        raw_error = str(exc).split(":", 1)[0].strip().upper()
        technical_feature_error_code = (
            raw_error
            if raw_error and all(char.isalnum() or char == "_" for char in raw_error)
            else type(exc).__name__.upper()
        )
      try:
        store = getattr(decision_engine, "store", None)
        if store is not None and hasattr(store, "latest_tick"):
          realtime_tick = store.latest_tick(symbol)
          received_at = getattr(realtime_tick, "received_at", None) if realtime_tick is not None else None
          if received_at is not None:
            quote_age_seconds = max(0.0, (decision_time - received_at).total_seconds())
      except Exception:  # noqa: BLE001 - advisory only.
        realtime_tick = None
      if features is None:
        try:
          refresher = getattr(decision_engine, "market_refresher", None)
          if refresher is not None:
            from app.trading.shared_decision_engine import _resolve_order_market

            broker_quote = refresher(symbol, _resolve_order_market(symbol, account), decision_time)
            if broker_quote is not None:
              quote_age_seconds = 0.0
        except Exception:  # noqa: BLE001 - broker quote fallback is advisory only.
          broker_quote = None
      h = holdings_by_symbol.get(symbol)
      holding_state = None
      if h is not None:
        holding_state = {"quantity": int(getattr(h, "quantity", 0) or 0),
                         "average_price": float(getattr(h, "average_price", 0.0) or 0.0)}
      return MicroReasoningInput(
          timestamp=decision_time, symbol=symbol,
          allowed_micro_strategies=macro_result.allowed_micro_strategies,
          blocked_micro_strategies=macro_result.blocked_micro_strategies,
          technical_features=features,
          live_feature_frame=frame,
          technical_feature_error_code=technical_feature_error_code,
          holding_state=holding_state,
          realtime_tick=realtime_tick, broker_quote=broker_quote,
          quote_age_seconds=quote_age_seconds,
          event_evidence=symbol_event_evidence.get(symbol, ()),
      )

    bundle = coordinator.run(macro_input, micro_input_builder=_builder, held_symbols=held_symbols)
    last_bundle[0] = bundle
    macro_micro_feed.record_bundle(bundle.as_dict())
    return bundle

  return _observer


def _record_realtime_trading_cycle(summary: dict[str, Any]) -> None:
  _apply_live_buy_candidate_backoff(summary)
  compact = {
      "at": summary.get("at"),
      "reason": summary.get("reason"),
      "submitted": summary.get("submitted", 0),
      "buy_submitted": summary.get("buy_submitted", 0),
      "sell_submitted": summary.get("sell_submitted", 0),
      "buy_submit_attempted": summary.get("buy_submit_attempted", 0),
      "buy_evaluated": summary.get("buy_evaluated", 0),
      "buy_rejected": summary.get("buy_rejected", 0),
      "sell_evaluated": summary.get("sell_evaluated", 0),
      "sell_rejected": summary.get("sell_rejected", 0),
      "skipped_market_closed": summary.get("skipped_market_closed", 0),
      "skipped_cooldown": summary.get("skipped_cooldown", 0),
      "skipped_ignored": summary.get("skipped_ignored", 0),
      "errors": summary.get("errors", 0),
      "blocked": summary.get("blocked", 0),
      "live_armed": summary.get("live_armed"),
      "buy_disabled": summary.get("buy_disabled", False),
      "buy_disabled_reason": summary.get("buy_disabled_reason"),
      "strategy_session": dict(summary.get("strategy_session") or {}),
      "rejections": list(summary.get("rejections") or ())[:5],
  }
  audit.record("realtime_trading_cycle", compact)


def _apply_live_buy_candidate_backoff(summary: dict[str, Any]) -> None:
  now = time.monotonic()
  hard_cooldown = max(30.0, _env_float_web("REALTIME_US_BAD_CANDIDATE_COOLDOWN_SEC", 900.0))
  soft_cooldown = max(30.0, _env_float_web("REALTIME_US_WEAK_CANDIDATE_COOLDOWN_SEC", 300.0))
  updates: dict[str, float] = {}
  for rejection in tuple(summary.get("rejections") or ()):
    if str(rejection.get("side") or "").upper() != "BUY":
      continue
    symbol = str(rejection.get("symbol") or "").upper().strip()
    if not symbol or _is_krx_ticker(symbol):
      continue
    reason_codes = {str(code) for code in tuple(rejection.get("reason_codes") or ())}
    warnings = {str(code) for code in tuple(rejection.get("warnings") or ())}
    reason_text = ",".join((*reason_codes, *warnings))
    spread_or_liquidity_block = any(
        code.startswith("WIDE_SPREAD")
        or code.startswith("SPREAD_TOO_WIDE")
        for code in reason_codes
    )
    hard_block = (
        "EXEC_NO_ORDERBOOK_BLOCKED" in reason_codes
        or "INSUFFICIENT_CASH_FOR_ONE_SHARE" in reason_codes
        or "EMPTY_OR_INVALID_ORDERBOOK" in warnings
        or "QUOTE_COUNT_ZERO" in reason_text
        or "ORDERBOOK_COUNT_ZERO" in reason_text
        or "ORDERBOOK_STALE" in reason_text
    )
    soft_block = (
        {"SPREAD_TOO_WIDE", "SLIPPAGE_RISK_HIGH"} <= reason_codes
        or spread_or_liquidity_block
    )
    if hard_block:
      updates[symbol] = max(updates.get(symbol, 0.0), now + hard_cooldown)
    elif soft_block:
      updates[symbol] = max(updates.get(symbol, 0.0), now + soft_cooldown)
  if not updates:
    return
  with _live_lock:
    expired = [symbol for symbol, until in _live_buy_candidate_backoff_until.items() if float(until or 0.0) <= now]
    for symbol in expired:
      _live_buy_candidate_backoff_until.pop(symbol, None)
    _live_buy_candidate_backoff_until.update(updates)


def _realtime_buy_candidates() -> tuple[str, ...]:
  """실시간 매수 후보 = 설정 심볼 + 스토어에 신선한 틱이 있는 종목(시장 개장 종목만 엔진이 추림).

  설정 심볼만 쓰면 후보가 KR 2종목뿐이라 미국 장중에 매수가 전혀 평가되지 않는다.
  스토어에 데이터가 흐르는 종목을 후보로 넓혀 매수·매도가 함께 판단되게 한다.
  """
  _apply_latest_realtime_candidate_backoff()
  macro_micro_bundle = _fresh_macro_micro_bundle()
  if _macro_micro_enforces_live_trading() and _macro_micro_blocks_new_buys(macro_micro_bundle):
    return ()
  max_age = float(os.getenv("REALTIME_BUY_CANDIDATE_MAX_AGE_SEC", "120"))
  limit = max(1, int(float(os.getenv("REALTIME_BUY_CANDIDATE_LIMIT", "120"))))
  config_symbols = _load_realtime_collection_symbols()
  fresh: tuple[str, ...] = ()
  try:
    since = datetime.now(timezone.utc) - timedelta(seconds=max_age)
    fresh = RealtimeMarketDataStore().active_symbols(since, limit=limit)
  except Exception:  # noqa: BLE001 - candidate discovery is best-effort.
    fresh = ()
  cached_context = _cached_context_buy_candidates(limit=limit)
  affordable = tuple(_live_affordable_buy_candidate_symbols(limit=limit) or ())
  pending_warmup = tuple(_pending_krx_buy_candidate_warmup_symbols(clean_ready=False) or ())
  domestic_ranked = tuple(_cached_domestic_ranking_symbols() or ())
  surge = tuple(_cached_volume_surge_symbols() or ())
  macro_micro_ranked = _macro_micro_ranked_buy_candidates(macro_micro_bundle)
  macro_micro_blocked = set(_macro_micro_blocked_buy_candidates(macro_micro_bundle))
  us_scan_fallback = _us_open_cash_scan_fallback_candidates(limit=limit)
  warmed_us = _warm_us_volume_surge_candidates_for_buy_filter(
      tuple(dict.fromkeys((*macro_micro_ranked, *surge, *us_scan_fallback)))
  )
  try:
    us_scan_priority_limit = max(0, int(os.getenv("REALTIME_US_SCAN_FALLBACK_PRIORITY_LIMIT", "8")))
  except (TypeError, ValueError):
    us_scan_priority_limit = 8
  us_scan_priority = tuple(us_scan_fallback[:us_scan_priority_limit])
  us_scan_tail = tuple(symbol for symbol in us_scan_fallback if symbol not in set(us_scan_priority))
  ordered = _prioritize_realtime_buy_candidates(
      tuple(
          symbol
          for symbol in dict.fromkeys((
              *macro_micro_ranked,
              *us_scan_priority,
              *domestic_ranked,
              *surge,
              *cached_context,
              *fresh,
              *pending_warmup,
              *config_symbols,
              *affordable,
              *us_scan_tail,
          ))
          if symbol not in macro_micro_blocked
      )
  )
  return _filter_realtime_buy_candidates_by_affordability(
      tuple(symbol for symbol in ordered if _is_live_buy_candidate_symbol(symbol)),
      limit=limit,
      prevalidated_symbols=tuple(dict.fromkeys((*affordable, *warmed_us, *us_scan_fallback))),
      backoff_bypass_symbols=us_scan_fallback,
  )


def _apply_latest_realtime_candidate_backoff() -> None:
  with _realtime_trading_lock:
    engine = _realtime_trading_engine
  if engine is None:
    return
  try:
    status = engine.get_status()
    summary = status.get("last_summary") if isinstance(status, dict) else None
  except Exception:  # noqa: BLE001 - advisory only.
    return
  if isinstance(summary, dict) and summary:
    _apply_live_buy_candidate_backoff(summary)


def _macro_micro_enforces_live_trading() -> bool:
  return os.getenv("REALTIME_MACRO_MICRO_ENFORCE", "true").strip().lower() in {"1", "true", "yes", "on"}


def _env_bool_web(name: str, default: bool) -> bool:
  raw = os.getenv(name)
  if raw is None:
    return default
  return raw.strip().lower() in {"1", "true", "yes", "on"}


def _fresh_macro_micro_bundle() -> dict[str, Any] | None:
  if not _macro_micro_enforces_live_trading():
    return None
  bundle = _latest_macro_micro_bundle()
  if not isinstance(bundle, dict) or not bundle:
    return None
  timestamp = _parse_macro_micro_timestamp(bundle.get("timestamp"))
  if timestamp is None:
    return None
  max_age = max(5.0, _env_float_web("REALTIME_MACRO_MICRO_MAX_AGE_SEC", 180.0))
  age = max(0.0, (datetime.now(timezone.utc) - timestamp).total_seconds())
  return bundle if age <= max_age else None


def _parse_macro_micro_timestamp(raw: Any) -> datetime | None:
  if isinstance(raw, datetime):
    value = raw
  else:
    try:
      text = str(raw or "").strip()
      if not text:
        return None
      value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
      return None
  if value.tzinfo is None:
    value = value.replace(tzinfo=timezone.utc)
  return value.astimezone(timezone.utc)


def _macro_micro_blocks_new_buys(bundle: dict[str, Any] | None) -> bool:
  if not bundle:
    return False
  macro = bundle.get("macro_result") if isinstance(bundle.get("macro_result"), dict) else {}
  reason_codes = {str(code) for code in tuple(macro.get("reason_codes") or ())}
  if (
      "MACRO_INSUFFICIENT_DATA" in reason_codes
      and not _env_bool_web("REALTIME_MACRO_MICRO_BLOCK_ON_INSUFFICIENT_DATA", False)
  ):
    return False
  return bool(macro.get("blocks_buy"))


def _macro_micro_ranked_buy_candidates(bundle: dict[str, Any] | None) -> tuple[str, ...]:
  if not bundle:
    return ()
  if _macro_micro_blocks_new_buys(bundle):
    return ()
  ranked: list[str] = []
  for item in tuple(bundle.get("ranked_trade_intents") or ()):
    if not isinstance(item, dict):
      continue
    if str(item.get("side") or "").upper() != "BUY":
      continue
    symbol = str(item.get("symbol") or "").upper().strip()
    if symbol and _is_live_buy_candidate_symbol(symbol):
      ranked.append(symbol)
  if ranked:
    return tuple(dict.fromkeys(ranked))
  return tuple(
      dict.fromkeys(
          symbol
          for symbol in (str(raw or "").upper().strip() for raw in tuple(bundle.get("buy_candidates") or ()))
          if symbol and _is_live_buy_candidate_symbol(symbol)
      )
  )


def _macro_micro_blocked_buy_candidates(bundle: dict[str, Any] | None) -> tuple[str, ...]:
  if not bundle:
    return ()
  macro = bundle.get("macro_result") if isinstance(bundle.get("macro_result"), dict) else {}
  reason_codes = {str(code) for code in tuple(macro.get("reason_codes") or ())}
  if (
      "MACRO_INSUFFICIENT_DATA" in reason_codes
      and not _env_bool_web("REALTIME_MACRO_MICRO_BLOCK_ON_INSUFFICIENT_DATA", False)
  ):
    return ()
  micro_results = tuple(row for row in tuple(bundle.get("micro_results") or ()) if isinstance(row, dict))
  if micro_results:
    hard_blocked: list[str] = []
    hard_reason_prefixes = (
        "LOW_LIQUIDITY_TECHNICAL_BLOCK",
        "HIGH_VOLATILITY_TECHNICAL_BLOCK",
        "SPREAD_CONSUMES_TECHNICAL_ALPHA",
        "EXECUTION_QUALITY_BLOCK",
    )
    soft_reasons = {"MICRO_TECHNICAL_HISTORY_INSUFFICIENT", "WAIT_CONFIRMATION"}
    for row in micro_results:
      symbol = str(row.get("symbol") or "").upper().strip()
      if not symbol or not _is_live_buy_candidate_symbol(symbol):
        continue
      reason_codes = {str(code) for code in tuple(row.get("reason_codes") or ())}
      if reason_codes and reason_codes <= soft_reasons:
        continue
      micro_regime = str(row.get("micro_regime") or "").upper()
      execution_quality = str(row.get("execution_quality") or "").upper()
      has_hard_reason = any(any(code.startswith(prefix) for prefix in hard_reason_prefixes) for code in reason_codes)
      if micro_regime == "NO_TRADE_SYMBOL" or execution_quality == "BLOCKED" or has_hard_reason:
        hard_blocked.append(symbol)
    return tuple(dict.fromkeys(hard_blocked))
  return tuple(
      dict.fromkeys(
          symbol
          for symbol in (str(raw or "").upper().strip() for raw in tuple(bundle.get("blocked_candidates") or ()))
          if symbol and _is_live_buy_candidate_symbol(symbol)
      )
  )


def _us_open_cash_scan_fallback_candidates(limit: int = 8) -> tuple[str, ...]:
  if "US" not in set(_active_live_market_groups()):
    return ()
  try:
    account = _live_account_snapshot_for_analysis()
  except Exception:
    return ()
  if account is None or _account_available_cash(account, "USD") <= 0:
    return ()
  try:
    cap = max(0, min(int(limit), int(os.getenv("REALTIME_US_SCAN_FALLBACK_LIMIT", "8"))))
  except (TypeError, ValueError):
    cap = min(int(limit), 8)
  if cap <= 0:
    return ()
  raw = os.getenv(
      "REALTIME_US_SCAN_FALLBACK_SYMBOLS",
      "F,SOFI,INTC,PFE,T,BAC,WBD,SNAP,PLTR,NIO,RIVN,LCID,OPEN,VALE,NU,CCL,KVUE,LYFT,HOOD",
  )
  configured = tuple(
      dict.fromkeys(
          symbol
          for symbol in (item.strip().upper() for item in raw.replace(";", ",").split(","))
          if symbol and _is_live_buy_candidate_symbol(symbol)
      )
  )
  if configured:
    return configured[:cap]
  return tuple(
      symbol
      for symbol in _load_us_nasdaq_universe()
      if _is_live_buy_candidate_symbol(symbol) and not _live_buy_candidate_in_backoff(symbol)
  )[:cap]


def _warm_us_volume_surge_candidates_for_buy_filter(symbols: tuple[str, ...]) -> tuple[str, ...]:
  if "US" not in set(_active_live_market_groups()):
    return ()
  try:
    limit = max(0, int(os.getenv("REALTIME_US_VOLUME_SURGE_WARM_LIMIT", "5")))
  except (TypeError, ValueError):
    limit = 5
  if limit <= 0:
    return ()
  target = tuple(
      dict.fromkeys(
          symbol
          for symbol in (str(raw or "").upper().strip() for raw in symbols)
          if symbol
          and _ticker_market_group_for_live_trading(symbol, "") == "US"
          and _is_live_buy_candidate_symbol(symbol)
          and not _live_buy_candidate_in_backoff(symbol)
      )
  )[:limit]
  if not target:
    return ()
  now = time.monotonic()
  try:
    interval = max(15.0, float(os.getenv("REALTIME_US_VOLUME_SURGE_WARM_INTERVAL_SEC", "60")))
  except (TypeError, ValueError):
    interval = 60.0
  with _live_lock:
    last_at = float(_volume_surge_warm_cache.get("at") or 0.0)
    last_symbols = tuple(_volume_surge_warm_cache.get("symbols") or ())
    if now - last_at < interval and set(target).issubset(set(last_symbols)):
      return target
    _volume_surge_warm_cache["at"] = now
    _volume_surge_warm_cache["symbols"] = target
  try:
    from app.trading.us_realtime_bridge import refresh_us_realtime_for_context_buy_candidates

    refresh_us_realtime_for_context_buy_candidates(SimpleNamespace(markets=(), reasoning_paths=()), symbols=target)
    return target
  except Exception:  # noqa: BLE001 - advisory warmup only; normal filters still protect execution.
    return target


def _prioritize_realtime_buy_candidates(
    symbols: tuple[str, ...],
    *,
    account: AccountSnapshot | None = None,
) -> tuple[str, ...]:
  """Keep currently tradeable cash buckets inside the engine's first evaluation window."""
  open_groups = set(_active_live_market_groups())
  if not ({"KRX", "US"} & open_groups):
    return symbols
  if account is None:
    try:
      account = _live_account_snapshot_for_analysis()
    except Exception:  # noqa: BLE001 - preserve original ordering on account lookup failure.
      return symbols
  krw_cash = _account_available_cash(account, "KRW") if account is not None else 0.0
  usd_cash = _account_available_cash(account, "USD") if account is not None else 0.0
  if krw_cash <= 0 and usd_cash <= 0:
    return symbols
  kr_symbols: list[str] = []
  us_symbols: list[str] = []
  other_symbols: list[str] = []
  seen: set[str] = set()
  for raw in symbols:
    symbol = str(raw or "").upper().strip()
    if not symbol or symbol in seen or not _is_live_buy_candidate_symbol(symbol):
      continue
    seen.add(symbol)
    group = _ticker_market_group_for_live_trading(symbol, "")
    if group == "KRX" and "KRX" in open_groups and krw_cash > 0:
      kr_symbols.append(symbol)
    elif group == "US" and "US" in open_groups and usd_cash > 0:
      us_symbols.append(symbol)
    else:
      other_symbols.append(symbol)
  if (
      "KRX" in open_groups
      and "US" in open_groups
      and krw_cash > 0
      and usd_cash > 0
      and kr_symbols
      and us_symbols
  ):
    # Reserve evaluation capacity for both open markets. A long KRX list must
    # not consume the whole cycle and starve otherwise healthy US candidates.
    interleaved: list[str] = []
    width = max(len(kr_symbols), len(us_symbols))
    for index in range(width):
      if index < len(kr_symbols):
        interleaved.append(kr_symbols[index])
      if index < len(us_symbols):
        interleaved.append(us_symbols[index])
    return tuple((*interleaved, *other_symbols))
  if "KRX" in open_groups and krw_cash > 0:
    return tuple((*kr_symbols, *us_symbols, *other_symbols))
  return tuple((*us_symbols, *kr_symbols, *other_symbols))


def _is_excluded_us_live_candidate(symbol: str) -> bool:
  text = str(symbol or "").upper().strip()
  if not text or _ticker_market_group_for_live_trading(text, "") != "US":
    return False
  suffixes = tuple(
      item.strip().upper()
      for item in os.getenv("REALTIME_US_EXCLUDE_SYMBOL_SUFFIXES", "U,WS,WT,W,R,P").split(",")
      if item.strip()
  )
  return any(text.endswith(suffix) and len(text) > len(suffix) for suffix in suffixes)


def _is_live_buy_candidate_symbol(symbol: str, market: str = "") -> bool:
  """Reject symbols that cannot map to a live cash-stock order route.

  KRX cash equities are six digits. US live candidates are alphabetic common-stock
  style tickers after the warrant/unit suffix filter. Values such as ``0015G0``
  come from non-cash instruments and otherwise waste quote calls/evaluation slots.
  """
  text = str(symbol or "").upper().strip().split(".", 1)[0]
  if not text:
    return False
  group = _ticker_market_group_for_live_trading(text, market)
  if group == "KRX":
    return text.isdigit() and len(text) == 6
  if group == "US":
    return text[0].isalpha() and not _is_excluded_us_live_candidate(text)
  return False


_affordable_candidate_cache: dict[str, Any] = {"key": None, "at": 0.0, "symbols": ()}
_broker_quote_backoff_until: dict[str, float] = {}
_broker_quote_cache: dict[tuple[str, str], tuple[float, MarketSnapshot]] = {}
_broker_quote_rate_lock = threading.Lock()
_broker_quote_next_allowed_at = 0.0
_live_buy_candidate_backoff_until: dict[str, float] = {}
_volume_surge_warm_cache: dict[str, Any] = {"at": 0.0, "symbols": ()}
_us_nasdaq_universe_cache: dict[str, Any] = {"mtime": None, "symbols": ()}
_us_learning_watchlist_cache: dict[str, Any] = {
    "at": 0.0,
    "cash_usd": None,
    "symbols": (),
    "pool": (),
    "rotation_index": 0,
}


def _account_available_cash(account: AccountSnapshot, currency: str) -> float:
  code = str(currency or "").upper().strip() or "KRW"
  orderable = getattr(account, "orderable_cash_by_currency", None) or {}
  if code in orderable:
    return float(orderable.get(code) or 0.0)
  cash_by_currency = getattr(account, "cash_by_currency", None) or {}
  if code in cash_by_currency:
    return float(cash_by_currency.get(code) or 0.0)
  if code == "KRW":
    return float(getattr(account, "cash", 0.0) or 0.0)
  if str(getattr(account, "base_currency", "") or "").upper() == code:
    return float(getattr(account, "cash", 0.0) or 0.0)
  return 0.0


def _recent_affordable_us_watchlist(
    account: AccountSnapshot,
    *,
    limit: int,
    database: Path | None = None,
) -> tuple[str, ...]:
  """Rank affordable US symbols by sustained, execution-usable market activity."""
  cash_usd = _account_available_cash(account, "USD")
  if cash_usd <= 0 or limit <= 0:
    return ()
  database = database or Path(
      os.getenv("REALTIME_MARKET_DATA_DB", "data/store/realtime_market_data.sqlite3")
  )
  if not database.exists():
    return ()
  now = datetime.now(timezone.utc)
  since = (
      now
      - timedelta(hours=max(1.0, _env_float_web("REALTIME_US_WATCHLIST_LOOKBACK_HOURS", 6.0)))
  ).isoformat()
  fresh_since = (
      now
      - timedelta(
          seconds=max(
              60.0,
              _env_float_web("REALTIME_US_WATCHLIST_MAX_TICK_AGE_SEC", 180.0),
          )
      )
  ).isoformat()
  minimum_ticks = max(
      2,
      _auto_reliability_int("REALTIME_US_WATCHLIST_MIN_TICKS", 3, 2),
  )
  sample_rows = max(
      1_000,
      _auto_reliability_int("REALTIME_US_WATCHLIST_SAMPLE_ROWS", 20_000, 1_000),
  )
  try:
    with closing(sqlite3.connect(database, timeout=5.0)) as connection:
      rows = connection.execute(
          """
          with recent_ticks as (
            select symbol, price, volume, received_at
            from realtime_ticks
            where received_at >= ?
              and symbol not glob '[0-9][0-9][0-9][0-9][0-9][0-9]'
              and price > 0
            order by received_at desc
            limit ?
          ),
          tick_stats as (
            select
              symbol,
              count(*) as tick_count,
              max(received_at) as latest_tick_at,
              sum(price * max(volume, 1)) as observed_notional,
              min(price) as minimum_price,
              max(price) as maximum_price
            from recent_ticks
            group by symbol
          ),
          latest_prices as (
            select ticks.symbol, max(ticks.price) as latest_price
            from recent_ticks ticks
            join tick_stats stats
              on stats.symbol = ticks.symbol
             and stats.latest_tick_at = ticks.received_at
            group by ticks.symbol
          ),
          book_stats as (
            select
              symbol,
              count(*) as book_count,
              avg(spread_bps) as average_spread_bps
            from realtime_orderbook
            where received_at >= ?
              and symbol in (select symbol from tick_stats)
            group by symbol
          )
          select
            stats.symbol,
            prices.latest_price,
            stats.latest_tick_at,
            stats.tick_count,
            stats.observed_notional,
            coalesce(books.book_count, 0) as book_count,
            coalesce(books.average_spread_bps, 1000000.0) as average_spread_bps,
            10000.0 * (stats.maximum_price - stats.minimum_price)
              / max(prices.latest_price, 0.000001) as observed_range_bps
          from tick_stats stats
          join latest_prices prices on prices.symbol = stats.symbol
          left join book_stats books on books.symbol = stats.symbol
          where stats.tick_count >= ?
            and stats.latest_tick_at >= ?
            and prices.latest_price <= ?
          order by
            case when coalesce(books.average_spread_bps, 1000000.0) <= 60.0
              then 0 else 1 end asc,
            observed_range_bps desc,
            book_count desc,
            average_spread_bps asc,
            stats.tick_count desc,
            stats.observed_notional desc,
            stats.latest_tick_at desc
          limit ?
          """,
          (
              since,
              sample_rows,
              since,
              minimum_ticks,
              fresh_since,
              cash_usd,
              max(limit * 8, limit),
          ),
      ).fetchall()
  except sqlite3.Error:
    return ()
  selected: list[str] = []
  excluded = _held_or_recent_buy_tickers(account)
  for symbol, _price, _received_at, *_quality in rows:
    ticker = str(symbol or "").upper().strip()
    if (
        not ticker
        or ticker in excluded
        or _is_excluded_us_live_candidate(ticker)
        or not _is_live_buy_candidate_symbol(ticker, "US")
    ):
      continue
    selected.append(ticker)
    if len(selected) >= limit:
      break
  return tuple(selected)


def _liquid_affordable_us_seed_symbols(
    account: AccountSnapshot,
    *,
    limit: int,
) -> tuple[str, ...]:
  """Broker-verify a domain-prior list used when learned activity is insufficient."""
  if limit <= 0:
    return ()
  raw = os.getenv(
      "REALTIME_US_LIQUIDITY_SEED_SYMBOLS",
      "F,SOFI,INTC,PFE,T,BAC,WBD,SNAP,PLTR,NIO,RIVN,LCID,NU,CCL,KVUE,LYFT,HOOD",
  )
  excluded = _held_or_recent_buy_tickers(account)
  configured = tuple(
      dict.fromkeys(
          ticker
          for ticker in (item.strip().upper() for item in raw.replace(";", ",").split(","))
          if ticker
          and ticker not in excluded
          and _is_live_buy_candidate_symbol(ticker, "US")
      )
  )
  if not configured:
    return ()
  try:
    exchange_map = _load_us_listed_exchange_map()
  except Exception:
    exchange_map = {}
  return _broker_affordable_candidate_symbols(
      configured,
      "US",
      account,
      max_symbols=limit,
      exchange_resolver=lambda symbol: exchange_map.get(symbol, "NASDAQ"),
  )


def _sticky_us_learning_symbols(limit: int) -> tuple[str, ...]:
  """Hold and rotate a broker-verified US watchlist to form causal labels."""
  safe_limit = max(1, int(limit))
  account = _account_snapshot_from_live_basis(_last_live_account_basis())
  if account is None:
    return ()
  cash_usd = round(_account_available_cash(account, "USD"), 2)
  if cash_usd <= 0:
    return ()
  now = time.monotonic()
  ttl = max(300.0, _env_float_web("REALTIME_US_WATCHLIST_TTL_SEC", 1800.0))
  quality_recheck = max(
      180.0,
      _env_float_web("REALTIME_US_WATCHLIST_RECHECK_SEC", 300.0),
  )
  with _live_lock:
    cached_symbols = tuple(_us_learning_watchlist_cache.get("symbols") or ())
    cached_at = float(_us_learning_watchlist_cache.get("at") or 0.0)
    cached_cash = _us_learning_watchlist_cache.get("cash_usd")
  if (
      cached_symbols
      and cached_cash == cash_usd
      and now - cached_at < min(ttl, quality_recheck)
  ):
    return cached_symbols[:safe_limit]

  try:
    pool_multiplier = max(
        1,
        min(6, int(os.getenv("REALTIME_US_ROTATION_POOL_MULTIPLIER", "1"))),
    )
  except (TypeError, ValueError):
    pool_multiplier = 1
  pool_limit = safe_limit * pool_multiplier
  pool = list(_recent_affordable_us_watchlist(account, limit=pool_limit))
  if len(pool) < pool_limit:
    pool.extend(
        symbol
        for symbol in _liquid_affordable_us_seed_symbols(
            account,
            limit=pool_limit - len(pool),
        )
        if symbol not in pool
    )
  if len(pool) < pool_limit:
    discovered = tuple(
        _live_affordable_buy_candidate_symbols(limit=pool_limit) or ()
    )
    pool.extend(
        symbol
        for symbol in discovered
        if _ticker_market_group_for_live_trading(symbol, "") == "US"
    )
  pool_result = tuple(dict.fromkeys(pool))[:pool_limit]
  with _live_lock:
    previous_pool = tuple(_us_learning_watchlist_cache.get("pool") or ())
    previous_index = int(
        _us_learning_watchlist_cache.get("rotation_index") or 0
    )
  if pool_multiplier > 1 and len(pool_result) > safe_limit:
    rotation_index = (
        previous_index + 1
        if previous_pool == pool_result and cached_symbols
        else 0
    )
    start = (rotation_index * safe_limit) % len(pool_result)
    result = tuple(
        pool_result[(start + offset) % len(pool_result)]
        for offset in range(min(safe_limit, len(pool_result)))
    )
  else:
    rotation_index = 0
    result = pool_result[:safe_limit]
  with _live_lock:
    _us_learning_watchlist_cache.update(
        {
            "at": now,
            "cash_usd": cash_usd,
            "symbols": result,
            "pool": pool_result,
            "rotation_index": rotation_index,
        }
    )
  return result


def _load_us_nasdaq_universe() -> tuple[str, ...]:
  """NASDAQ-listed-only US universe for live discovery.

  Live US buys quote with EXCD=NAS and route orders with OVRS_EXCG_CD=NASD, so the
  discovery universe MUST be NASDAQ-listed: a NYSE/AMEX/Arca name quoted as NASDAQ
  returns price 0 (silently dropped), and if one slipped through, its order would be
  rejected for a wrong exchange. The merged nasdaqtraded universe is ~66% non-NASDAQ,
  which starved US discovery to ~0 affordable candidates. This reads a NASDAQ-only
  cache (data/universe/us_nasdaq_listed.csv, built from nasdaqlisted.txt) and falls
  back to the merged universe only when that cache is missing.
  """
  import csv
  from pathlib import Path as _Path

  path = _Path("data/universe/us_nasdaq_listed.csv")
  try:
    mtime = path.stat().st_mtime
  except OSError:
    return tuple(load_us_listed_universe(limit=None) or ())
  if _us_nasdaq_universe_cache.get("mtime") != mtime:
    symbols: list[str] = []
    try:
      with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
          symbol = str(row.get("symbol") or "").upper().strip()
          if symbol:
            symbols.append(symbol)
    except OSError:
      return tuple(load_us_listed_universe(limit=None) or ())
    _us_nasdaq_universe_cache["symbols"] = tuple(symbols)
    _us_nasdaq_universe_cache["mtime"] = mtime
  symbols = tuple(_us_nasdaq_universe_cache.get("symbols") or ())
  return symbols or tuple(load_us_listed_universe(limit=None) or ())


def _live_affordable_buy_candidate_symbols(limit: int = 120) -> tuple[str, ...]:
  """Return open-session symbols that this small live account can plausibly buy.

  This feeds the low-latency realtime engine. The slower broker quote overlay still
  verifies prices before order creation; this list only makes sure the engine has
  a useful scan universe when cached ontology BuyCandidate paths are empty.
  """
  open_groups = set(_active_live_market_groups())
  if not open_groups:
    return ()
  try:
    account = _live_account_snapshot_for_analysis()
  except Exception:  # noqa: BLE001 - candidates are best-effort.
    account = None
  if account is None:
    return ()
  excluded = _held_or_recent_buy_tickers(account)
  symbols: list[str] = []
  max_count = max(1, int(limit))
  cache_ttl = max(5.0, float(os.getenv("REALTIME_AFFORDABLE_CANDIDATE_TTL_SEC", "45")))
  cache_key = (
      tuple(sorted(open_groups)),
      max_count,
      round(_account_available_cash(account, "USD"), 2),
      round(_account_available_cash(account, "KRW"), 0),
      tuple(sorted(excluded))[:20],
  )
  now = time.monotonic()
  if (
      _affordable_candidate_cache.get("key") == cache_key
      and now - float(_affordable_candidate_cache.get("at") or 0.0) < cache_ttl
  ):
    return tuple(_affordable_candidate_cache.get("symbols") or ())[:max_count]

  if "US" in open_groups and _account_available_cash(account, "USD") > 0:
    try:
      us_limit = max(0, int(os.getenv("REALTIME_US_DISCOVERY_CANDIDATE_LIMIT", "8")))
    except ValueError:
      us_limit = 8
    # Prefer prices already arriving through the websocket. Re-quoting random
    # discovery symbols while a live universe exists wastes the shared KIS
    # request budget and was the main source of "초당 거래건수 초과" errors.
    try:
      live_store = RealtimeMarketDataStore()
      fresh_since = datetime.now(timezone.utc) - timedelta(
          seconds=max(30.0, _env_float_web("REALTIME_BUY_CANDIDATE_MAX_AGE_SEC", 120.0))
      )
      fresh_us = tuple(live_store.active_symbols(fresh_since, limit=max(32, us_limit * 4)))
    except Exception:
      live_store = None
      fresh_us = ()
    for ticker in fresh_us:
      if len(symbols) >= min(max_count, us_limit):
        break
      ticker = str(ticker or "").upper()
      if (
          ticker in excluded
          or _ticker_market_group_for_live_trading(ticker, "") != "US"
          or _is_excluded_us_live_candidate(ticker)
      ):
        continue
      if live_store is not None and _cached_candidate_affordable(ticker, account, live_store):
        symbols.append(ticker)

    # Once at least one affordable streaming symbol exists, let websocket
    # rotation grow that universe naturally. Do not fill the remaining slots
    # with synchronous random quotes competing with account/order endpoints.
    remaining_us = (
        0
        if symbols
        else max(0, min(max_count - len(symbols), us_limit - len(symbols)))
    )
    if remaining_us > 0:
      try:
        us_exchange_map = _load_us_listed_exchange_map()
      except Exception as exc:  # noqa: BLE001 - discovery is optional.
        audit.record("realtime_us_discovery_candidate_load_failed", {"error": str(exc)})
        us_exchange_map = {}
      us_universe = tuple(us_exchange_map.keys()) or tuple(_load_us_nasdaq_universe() or ())
      us_symbols: list[str] = []
      for symbol in _rotated_symbols(us_universe):
        ticker = str(symbol or "").upper().strip().split(".", 1)[0]
        if (
            ticker
            and ticker not in excluded
            and ticker not in symbols
            and not _is_excluded_us_live_candidate(ticker)
        ):
          us_symbols.append(ticker)
        if len(us_symbols) >= remaining_us:
          break
      symbols.extend(
          _broker_affordable_candidate_symbols(
              tuple(us_symbols),
              "US",
              account,
              max_symbols=remaining_us,
              exchange_resolver=lambda s: us_exchange_map.get(s, "NASDAQ"),
          )
        )

  if "KRX" in open_groups and _account_available_cash(account, "KRW") > 0 and len(symbols) < max_count:
    try:
      krx_limit = max(0, int(os.getenv("REALTIME_KRX_DISCOVERY_CANDIDATE_LIMIT", "6")))
    except ValueError:
      krx_limit = 6
    try:
      krx_scan_limit = max(krx_limit, int(os.getenv("REALTIME_KRX_DISCOVERY_SCAN_LIMIT", "24")))
    except ValueError:
      krx_scan_limit = max(krx_limit, 24)
    try:
      krx_universe = tuple(load_krx_listed_universe(limit=None) or ())
    except Exception as exc:  # noqa: BLE001 - discovery is optional.
      audit.record("realtime_krx_discovery_candidate_load_failed", {"error": str(exc)})
      krx_universe = ()
    krx_symbols: list[str] = []
    for symbol in _rotated_symbols(krx_universe):
      ticker = str(symbol or "").upper().strip().split(".", 1)[0]
      if ticker and ticker not in excluded and ticker.isdigit() and len(ticker) == 6:
        krx_symbols.append(ticker)
      if len(krx_symbols) >= min(max_count, krx_scan_limit):
        break
    symbols.extend(
        _broker_affordable_candidate_symbols(
            tuple(krx_symbols),
            "KOSPI",
            account,
            max_symbols=max(0, min(max_count, krx_limit)),
        )
    )

  result = tuple(dict.fromkeys(symbols))[:max_count]
  _affordable_candidate_cache.update({"key": cache_key, "at": now, "symbols": result})
  return result


def _broker_affordable_candidate_symbols(
    symbols: tuple[str, ...],
    market: str,
    account: AccountSnapshot,
    *,
    max_symbols: int,
    exchange_resolver: Callable[[str], str] | None = None,
) -> tuple[str, ...]:
  if max_symbols <= 0 or not symbols:
    return ()
  now = time.monotonic()
  market_key = market.upper().strip() or "UNKNOWN"
  if now < float(_broker_quote_backoff_until.get(market_key, 0.0) or 0.0):
    return ()
  client = KisDevelopersApiClient(paper=False, enabled=True)
  selected: list[str] = []
  errors: list[dict[str, str]] = []
  quote_delay = max(0.05, float(os.getenv("REALTIME_BROKER_QUOTE_DELAY_SEC", "0.35")))
  quote_cache_ttl = max(5.0, float(os.getenv("REALTIME_BROKER_QUOTE_CACHE_TTL_SEC", "60")))
  for symbol in symbols:
    if len(selected) >= max_symbols:
      break
    if now < float(_broker_quote_backoff_until.get(symbol, 0.0) or 0.0):
      continue
    # US discovery must quote each ticker on its REAL exchange (NASD/NYSE/AMEX): a
    # NYSE name quoted as NASDAQ returns price 0. exchange_resolver supplies the
    # per-ticker exchange; without it, the fixed market is used (e.g. KRX).
    quote_market = exchange_resolver(symbol) if exchange_resolver is not None else market
    cache_key = (str(quote_market).upper(), symbol)
    cached = _broker_quote_cache.get(cache_key)
    try:
      if cached is not None and time.monotonic() - cached[0] < quote_cache_ttl:
        snapshot = cached[1]
      else:
        global _broker_quote_next_allowed_at
        with _broker_quote_rate_lock:
          wait_seconds = _broker_quote_next_allowed_at - time.monotonic()
          if wait_seconds > 0:
            time.sleep(wait_seconds)
          snapshot = client.get_market_snapshot(
              symbol,
              quote_market,
              company_name=symbol,
              sector="Unknown",
          )
          _broker_quote_next_allowed_at = time.monotonic() + quote_delay
        _broker_quote_cache[cache_key] = (time.monotonic(), snapshot)
    except Exception as exc:  # noqa: BLE001 - one failed quote should not stop discovery.
      error_text = str(exc) or exc.__class__.__name__
      errors.append({"ticker": symbol, "market": quote_market, "error": error_text})
      cooldown = _broker_quote_error_cooldown_seconds(error_text)
      if cooldown > 0.0:
        until = time.monotonic() + cooldown
        _broker_quote_backoff_until[symbol] = until
        if _is_market_wide_broker_quote_error(error_text):
          _broker_quote_backoff_until[market_key] = until
          break
      continue
    if snapshot.last_price > 0 and is_market_affordable_for_account(snapshot, account):
      selected.append(snapshot.ticker)
  if errors:
    audit.record(
        "realtime_affordable_candidate_quote_errors",
        {
            "market": market,
            "errors": errors[:10],
            "selected": selected[:10],
            "backoff_seconds": round(max(0.0, float(_broker_quote_backoff_until.get(market_key, 0.0) or 0.0) - time.monotonic()), 2),
        },
    )
  return tuple(selected)


def _broker_quote_error_cooldown_seconds(error_text: str) -> float:
  lowered = error_text.lower()
  if "초당 거래건수" in error_text or "rate" in lowered or "too many" in lowered:
    return max(5.0, float(os.getenv("REALTIME_BROKER_RATE_LIMIT_BACKOFF_SEC", "20")))
  if "temporary failure in name resolution" in lowered:
    return max(5.0, float(os.getenv("REALTIME_BROKER_DNS_BACKOFF_SEC", "30")))
  if "timed out" in lowered or "closed connection" in lowered or "remote end closed" in lowered:
    return max(3.0, float(os.getenv("REALTIME_BROKER_TRANSIENT_BACKOFF_SEC", "10")))
  return 0.0


def _is_market_wide_broker_quote_error(error_text: str) -> bool:
  lowered = error_text.lower()
  return (
      "초당 거래건수" in error_text
      or "temporary failure in name resolution" in lowered
      or "timed out" in lowered
      or "closed connection" in lowered
      or "remote end closed" in lowered
  )


def _cached_context_buy_candidates(
    limit: int = 30,
    *,
    account: AccountSnapshot | None = None,
) -> tuple[str, ...]:
  with _live_lock:
    context = _live_state.get("context")
  if context is None:
    return ()

  markets = tuple(getattr(context, "markets", ()) or ())
  market_by_ticker = {str(getattr(market, "ticker", "") or "").upper().strip(): market for market in markets}
  open_groups = set(_active_live_market_groups())
  if account is None:
    try:
      account = _live_account_snapshot_for_analysis()
    except Exception:  # noqa: BLE001 - fail closed for buy candidates.
      account = None
  selected: list[str] = []

  for path in tuple(getattr(context, "reasoning_paths", ()) or ()):
    if str(getattr(path, "conclusion", "") or "") != "BuyCandidate":
      continue
    ticker = str(getattr(path, "ticker", "") or "").upper().strip()
    if not ticker:
      continue
    market = market_by_ticker.get(ticker)
    if account is not None and (market is None or not is_market_affordable_for_account(market, account)):
      continue
    group = _ticker_market_group_for_live_trading(ticker, getattr(market, "market", "") if market is not None else "")
    if not open_groups or group in open_groups:
      selected.append(ticker)

  selection = getattr(context, "candidate_selection", None)
  for ticker_value in tuple(getattr(selection, "candidate_stocks", ()) or ()):
    ticker = str(ticker_value or "").upper().strip()
    if not ticker:
      continue
    market = market_by_ticker.get(ticker)
    if account is not None and (market is None or not is_market_affordable_for_account(market, account)):
      continue
    group = _ticker_market_group_for_live_trading(ticker, getattr(market, "market", "") if market is not None else "")
    if not open_groups or group in open_groups:
      selected.append(ticker)

  return tuple(dict.fromkeys(selected))[: max(1, int(limit))]


def _filter_realtime_buy_candidates_by_affordability(
    symbols: tuple[str, ...],
    *,
    limit: int,
    prevalidated_symbols: tuple[str, ...] = (),
    backoff_bypass_symbols: tuple[str, ...] = (),
) -> tuple[str, ...]:
  try:
    account = _live_account_snapshot_for_analysis()
  except Exception:  # noqa: BLE001 - fail closed for live buy candidates.
    return ()
  if account is None:
    return ()
  with _live_lock:
    context = _live_state.get("context")
  context_markets = {
      str(getattr(market, "ticker", "") or "").upper().strip(): market
      for market in tuple(getattr(context, "markets", ()) or ())
  }
  store = RealtimeMarketDataStore()
  selected: list[str] = []
  seen: set[str] = set()
  prevalidated = {str(symbol or "").upper().strip() for symbol in prevalidated_symbols}
  backoff_bypass = {str(symbol or "").upper().strip() for symbol in backoff_bypass_symbols}
  for symbol in symbols:
    ticker = str(symbol or "").upper().strip()
    if not ticker or ticker in seen:
      continue
    if ticker not in backoff_bypass and _live_buy_candidate_in_backoff(ticker):
      continue
    if _is_excluded_us_live_candidate(ticker):
      continue
    if ticker not in prevalidated:
      market = _candidate_affordability_market(ticker, context_markets.get(ticker), store)
      if market is None or not _candidate_affordable_with_buffer(ticker, market, account):
        continue
      if not _candidate_has_usable_live_liquidity(ticker, store):
        continue
    if _is_krx_ticker(ticker) and not _candidate_has_fresh_buy_orderbook(ticker, store):
      _queue_krx_buy_candidate_warmup(ticker)
    selected.append(ticker)
    seen.add(ticker)
    if len(selected) >= max(1, int(limit)):
      break
  return tuple(selected)


def _live_buy_candidate_in_backoff(ticker: str) -> bool:
  symbol = str(ticker or "").upper().strip()
  if not symbol:
    return False
  if _is_krx_ticker(symbol):
    return False
  now = time.monotonic()
  with _live_lock:
    until = float(_live_buy_candidate_backoff_until.get(symbol) or 0.0)
    if until <= now:
      _live_buy_candidate_backoff_until.pop(symbol, None)
      return False
  if _us_candidate_orderbook_recovered(symbol):
    with _live_lock:
      _live_buy_candidate_backoff_until.pop(symbol, None)
    return False
  return True


def _us_candidate_orderbook_recovered(symbol: str) -> bool:
  if _ticker_market_group_for_live_trading(symbol, "") != "US":
    return False
  try:
    orderbook = RealtimeMarketDataStore().latest_orderbook(symbol)
  except Exception:  # noqa: BLE001
    return False
  if orderbook is None or not _is_recent_realtime_item(orderbook, "REALTIME_US_CANDIDATE_ORDERBOOK_MAX_AGE_SEC", 180.0):
    return False
  bid = _number_or_zero(getattr(orderbook, "best_bid", 0.0))
  ask = _number_or_zero(getattr(orderbook, "best_ask", 0.0))
  return bid > 0 and ask >= bid


def _is_krx_ticker(ticker: str) -> bool:
  text = str(ticker or "").upper().strip()
  return text.isdigit() and len(text) == 6


def _candidate_has_fresh_buy_orderbook(ticker: str, store: RealtimeMarketDataStore) -> bool:
  if not _is_krx_ticker(ticker):
    return True
  try:
    orderbook = store.latest_orderbook(ticker)
  except Exception:  # noqa: BLE001 - fail closed for live BUY, collector will warm it.
    return False
  if orderbook is None:
    return False
  bid = _number_or_zero(getattr(orderbook, "best_bid", 0.0))
  ask = _number_or_zero(getattr(orderbook, "best_ask", 0.0))
  if bid <= 0 or ask <= 0 or ask < bid:
    return False
  if str(getattr(orderbook, "source", "") or "") != "kis_realtime_websocket":
    return False
  received_at = getattr(orderbook, "received_at", None)
  if received_at is None:
    return False
  try:
    age_seconds = max(
        0.0,
        (datetime.now(timezone.utc) - received_at).total_seconds(),
    )
  except (TypeError, ValueError):
    return False
  # Candidate admission must be at least as strict as the feature builder.
  # Otherwise a symbol reaches ontology/micro reasoning and is immediately
  # rejected as MARKET_DATA_NOT_LIVE_BUY_ELIGIBLE, creating a false impression
  # that strategy election itself is too restrictive.
  candidate_max_age = max(
      1.0,
      _env_float_web("REALTIME_KRX_BUY_CANDIDATE_ORDERBOOK_MAX_AGE_SEC", 180.0),
  )
  feature_max_age = max(
      1.0,
      _env_float_web("LIVE_FEATURE_MAX_ORDERBOOK_AGE_MS", 15_000.0) / 1_000.0,
  )
  return age_seconds <= min(candidate_max_age, feature_max_age)


def _queue_krx_buy_candidate_warmup(ticker: str) -> None:
  symbol = str(ticker or "").upper().strip()
  if not _is_krx_ticker(symbol):
    return
  try:
    if _candidate_has_fresh_buy_orderbook(symbol, RealtimeMarketDataStore()):
      with _live_lock:
        _pending_krx_buy_candidate_warmup.pop(symbol, None)
      return
  except Exception:  # noqa: BLE001 - queue on uncertainty; collector will refresh it.
    pass
  now = time.monotonic()
  ttl = max(30.0, _env_float_web("REALTIME_BUY_CANDIDATE_WARMUP_TTL_SEC", 180.0))
  request_resubscribe = False
  with _live_lock:
    request_resubscribe = symbol not in _pending_krx_buy_candidate_warmup
    _pending_krx_buy_candidate_warmup[symbol] = now + ttl
  if request_resubscribe:
    _request_kis_realtime_collector_resubscribe("warmup_candidate", (symbol,))


def _observe_dashboard_market_stream(ticker: str) -> None:
  """Keep the KRX symbol visible in the dashboard on the scarce WS pair."""
  symbol = str(ticker or "").upper().strip()
  if not _is_krx_ticker(symbol):
    return
  from app.data.market_session import MarketPhase, market_phase

  regular_feed_open = market_phase("KRX") is MarketPhase.REGULAR
  now = time.monotonic()
  ttl = max(10.0, _env_float_web("REALTIME_DASHBOARD_WATCH_TTL_SEC", 20.0))
  request_resubscribe = False
  stale_recovery = False
  try:
    store = RealtimeMarketDataStore()
    tick = store.latest_tick(symbol)
    orderbook = store.latest_orderbook(symbol)
    stale_after = max(
        3.0,
        _env_float_web("REALTIME_DASHBOARD_STALE_RECOVERY_SEC", 8.0),
    )
    tick_fresh = (
        tick is not None
        and _is_recent_realtime_item(
            tick,
            "REALTIME_DASHBOARD_STALE_RECOVERY_SEC",
            stale_after,
        )
    )
    book_fresh = (
        orderbook is not None
        and _is_recent_realtime_item(
            orderbook,
            "REALTIME_DASHBOARD_STALE_RECOVERY_SEC",
            stale_after,
        )
    )
    # H0STCNT0/H0STASP0 do not produce this application's tradeable feed outside
    # the regular session. Reconnecting to an intentionally quiet stream caused
    # stale subscriptions to accumulate at KIS and eventually OPSP0008.
    stale_recovery = regular_feed_open and not tick_fresh and not book_fresh
  except Exception:  # noqa: BLE001 - observer remains best-effort.
    stale_recovery = regular_feed_open
  with _live_lock:
    new_watch = symbol not in _dashboard_krx_watch
    request_resubscribe = new_watch and regular_feed_open
    _dashboard_krx_watch[symbol] = now + ttl
    last_recovery = _dashboard_krx_stale_recovery_at.get(symbol, 0.0)
    if stale_recovery and (new_watch or now - last_recovery >= 10.0):
      _dashboard_krx_stale_recovery_at[symbol] = now
      request_resubscribe = request_resubscribe or not new_watch
  if request_resubscribe:
    _request_kis_realtime_collector_resubscribe(
        "dashboard_stream_stale" if stale_recovery and not new_watch else "dashboard_stream",
        (symbol,),
    )


def _dashboard_krx_watch_symbols() -> tuple[str, ...]:
  now = time.monotonic()
  with _live_lock:
    expired = [
        symbol
        for symbol, until in _dashboard_krx_watch.items()
        if float(until or 0.0) <= now
    ]
    for symbol in expired:
      _dashboard_krx_watch.pop(symbol, None)
      _dashboard_krx_stale_recovery_at.pop(symbol, None)
    return tuple(
        symbol
        for symbol, _until in sorted(
            _dashboard_krx_watch.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )


def _pending_krx_buy_candidate_warmup_symbols(*, clean_ready: bool = True) -> tuple[str, ...]:
  now = time.monotonic()
  with _live_lock:
    pending = dict(_pending_krx_buy_candidate_warmup)
  ready: list[str] = []
  if clean_ready:
    try:
      store = RealtimeMarketDataStore()
      ready = [symbol for symbol in pending if _candidate_has_fresh_buy_orderbook(symbol, store)]
    except Exception:  # noqa: BLE001 - status cleanup is advisory only.
      ready = []
  with _live_lock:
    expired = [symbol for symbol, until in _pending_krx_buy_candidate_warmup.items() if float(until or 0.0) <= now]
    for symbol in ready:
      _pending_krx_buy_candidate_warmup.pop(symbol, None)
    for symbol in expired:
      _pending_krx_buy_candidate_warmup.pop(symbol, None)
    return tuple(_pending_krx_buy_candidate_warmup.keys())


def _request_kis_realtime_collector_resubscribe(reason: str, symbols: tuple[str, ...] = ()) -> None:
  global _kis_realtime_last_resubscribe_request_at
  now = time.monotonic()
  debounce_seconds = max(
      5.0,
      _env_float_web("REALTIME_BUY_CANDIDATE_RESUBSCRIBE_DEBOUNCE_SEC", 30.0),
  )
  with _live_lock:
    if _kis_realtime_collector_resubscribe.is_set():
      return
    if now - _kis_realtime_last_resubscribe_request_at < debounce_seconds:
      _append_collection_log_unlocked(
          "scheduled",
          "KIS realtime collector resubscribe deferred for warmup batching",
          counts={
              "reason": reason,
              "symbols": len(symbols),
              "symbol_sample": list(symbols[:8]),
              "debounce_seconds": debounce_seconds,
          },
      )
      return
    _kis_realtime_last_resubscribe_request_at = now
    _kis_realtime_collector_resubscribe.set()
    _append_collection_log_unlocked(
        "scheduled",
        "KIS realtime collector resubscribe requested",
        counts={
            "reason": reason,
            "symbols": len(symbols),
            "symbol_sample": list(symbols[:8]),
        },
    )


def _candidate_affordability_market(
    ticker: str,
    context_market: MarketSnapshot | None,
    store: RealtimeMarketDataStore,
) -> MarketSnapshot | None:
  tick = store.latest_tick(ticker)
  tick_price = (
      float(getattr(tick, "price", 0.0) or 0.0)
      if tick is not None and _is_recent_realtime_item(tick, "REALTIME_BUY_CANDIDATE_MAX_AGE_SEC", 240.0)
      else 0.0
  )
  book_price = _candidate_orderbook_price(ticker, store) if tick_price <= 0 else 0.0
  live_price = tick_price if tick_price > 0 else book_price
  market = context_market
  if market is None:
    if live_price <= 0:
      return None
    market_name = "KOSPI" if ticker.isdigit() and len(ticker) == 6 else "NASDAQ"
    return replace(_placeholder_live_market(ticker, market_name), last_price=live_price)
  if live_price > 0:
    return replace(market, last_price=live_price)
  return market


def _candidate_orderbook_price(ticker: str, store: RealtimeMarketDataStore) -> float:
  if not (ticker.isdigit() and len(ticker) == 6):
    return 0.0
  try:
    orderbook = store.latest_orderbook(ticker)
  except Exception:  # noqa: BLE001
    return 0.0
  if orderbook is None or not _is_recent_realtime_item(orderbook, "REALTIME_KRX_CANDIDATE_ORDERBOOK_MAX_AGE_SEC", 300.0):
    return 0.0
  bid = _number_or_zero(getattr(orderbook, "best_bid", 0.0))
  ask = _number_or_zero(getattr(orderbook, "best_ask", 0.0))
  if bid > 0 and ask >= bid:
    return (bid + ask) / 2.0
  if ask > 0:
    return ask
  if bid > 0:
    return bid
  return 0.0


def _candidate_has_usable_live_liquidity(ticker: str, store: RealtimeMarketDataStore) -> bool:
  if not (ticker.isdigit() and len(ticker) == 6):
    return True
  try:
    orderbook = store.latest_orderbook(ticker)
  except Exception:  # noqa: BLE001
    return True
  if orderbook is None or not _is_recent_realtime_item(orderbook, "REALTIME_KRX_CANDIDATE_ORDERBOOK_MAX_AGE_SEC", 300.0):
    return True
  depth = float(getattr(orderbook, "total_bid_volume", 0.0) or 0.0) + float(getattr(orderbook, "total_ask_volume", 0.0) or 0.0)
  mid_price = _candidate_orderbook_price(ticker, store)
  orderbook_value = depth * mid_price if mid_price > 0 else 0.0
  min_value = max(0.0, _env_float_web("REALTIME_KRX_MIN_CANDIDATE_ORDERBOOK_VALUE_KRW", 5_000_000.0))
  if orderbook_value >= min_value:
    return True
  liquidity_score = min(1.0, depth / 1_000_000.0)
  min_score = max(0.0, _env_float_web("REALTIME_KRX_MIN_CANDIDATE_LIQUIDITY_SCORE", 0.30))
  return liquidity_score >= min_score


def _candidate_has_strategy_feature_history(
    ticker: str,
    store: RealtimeMarketDataStore,
) -> bool:
  """Pre-filter candidates that cannot build the strategy feature frame."""
  minimum_bars = max(
      10,
      _auto_reliability_int("REALTIME_STRATEGY_MINUTE_HISTORY_BARS", 20, 10),
  )
  maximum_age_seconds = max(
      60.0,
      _env_float_web("REALTIME_STRATEGY_HISTORY_MAX_AGE_SEC", 180.0),
  )
  now = datetime.now(timezone.utc)
  try:
    bars = store.recent_minute_bars(
        ticker,
        now - timedelta(minutes=max(120, minimum_bars * 3)),
        limit=max(120, minimum_bars),
    )
  except Exception:
    return False
  if len(bars) < minimum_bars:
    return False
  latest = getattr(bars[-1], "minute_start", None)
  if latest is None:
    return False
  try:
    age = max(0.0, (now - latest).total_seconds())
  except (TypeError, ValueError):
    return False
  return age <= maximum_age_seconds


def _candidate_affordable_with_buffer(ticker: str, market: MarketSnapshot, account: AccountSnapshot) -> bool:
  if not is_market_affordable_for_account(market, account):
    return False
  price = float(getattr(market, "last_price", 0.0) or 0.0)
  if price <= 0:
    return False
  buffer = max(1.0, _env_float_web("REALTIME_ONE_SHARE_CASH_BUFFER", 1.03))
  available = cash_available_for_market(account, market)
  if available < price * buffer:
    return False
  if ticker.isdigit() and len(ticker) == 6:
    max_price = max(0.0, _env_float_web("REALTIME_KRX_MAX_ONE_SHARE_PRICE_KRW", 0.0))
    if max_price > 0.0 and price > max_price:
      return False
  return True


def _is_recent_realtime_item(item: Any, env_name: str, default_seconds: float) -> bool:
  received_at = getattr(item, "received_at", None)
  if received_at is None:
    return False
  try:
    age = max(0.0, (datetime.now(timezone.utc) - received_at).total_seconds())
  except Exception:  # noqa: BLE001
    return False
  return age <= max(1.0, _env_float_web(env_name, default_seconds))


def _env_float_web(name: str, default: float) -> float:
  try:
    return float(os.getenv(name, str(default)))
  except (TypeError, ValueError):
    return default


_volume_surge_cache: dict[str, Any] = {"at": 0.0, "symbols": ()}
_domestic_ranking_cache: dict[str, Any] = {"at": 0.0, "symbols": ()}


def _cached_volume_surge_symbols() -> tuple[str, ...]:
  """KIS 해외주식 거래량급증 종목을 TTL 캐시로 받아 매수 후보에 더한다(미국장 개장 시).

  매 사이클(~1s) API를 때리지 않도록 TTL(기본 60s) 캐시. 비활성/오류 시 빈 결과.
  """
  if os.getenv("REALTIME_USE_VOLUME_SURGE_API", "true").lower() in {"0", "false", "no", "off"}:
    return ()
  if "US" not in _active_live_market_groups():
    return ()
  ttl = float(os.getenv("REALTIME_VOLUME_SURGE_TTL_SEC", "60"))
  now = time.monotonic()
  with _live_lock:
    if now - float(_volume_surge_cache.get("at") or 0.0) < ttl and _volume_surge_cache.get("symbols"):
      return tuple(_volume_surge_cache.get("symbols") or ())
  try:
    from app.trading.us_realtime_bridge import fetch_overseas_volume_surge_symbols

    limit = max(1, int(float(os.getenv("REALTIME_VOLUME_SURGE_LIMIT", "20"))))
    result = fetch_overseas_volume_surge_symbols(max_symbols=limit)
    symbols = tuple(result.get("symbols") or ())
  except Exception:  # noqa: BLE001 - best-effort; never break candidate discovery.
    symbols = ()
  with _live_lock:
    _volume_surge_cache["at"] = now
    _volume_surge_cache["symbols"] = symbols
  return symbols


def _cached_domestic_ranking_symbols() -> tuple[str, ...]:
  """KIS domestic ranking APIs supply fresh KRX buy-discovery candidates."""
  if os.getenv("REALTIME_USE_KRX_RANKING_API", "true").lower() in {"0", "false", "no", "off"}:
    return ()
  if "KRX" not in _active_live_market_groups():
    return ()
  ttl = float(os.getenv("REALTIME_KRX_RANKING_TTL_SEC", "30"))
  now = time.monotonic()
  with _live_lock:
    if now - float(_domestic_ranking_cache.get("at") or 0.0) < ttl and _domestic_ranking_cache.get("symbols"):
      return tuple(_domestic_ranking_cache.get("symbols") or ())
  try:
    from app.trading.domestic_realtime_bridge import fetch_domestic_ranking_symbols

    raw_sources = os.getenv("REALTIME_KRX_RANKING_SOURCES", "volume_rank,fluctuation,volume_power")
    sources = tuple(item.strip().lower() for item in raw_sources.split(",") if item.strip())
    limit = max(1, int(float(os.getenv("REALTIME_KRX_RANKING_CANDIDATE_LIMIT", "24"))))
    result = fetch_domestic_ranking_symbols(sources=sources, max_symbols=limit)
    symbols = tuple(result.get("symbols") or ())
  except Exception:  # noqa: BLE001 - best-effort; never break candidate discovery.
    symbols = ()
  with _live_lock:
    _domestic_ranking_cache["at"] = now
    _domestic_ranking_cache["symbols"] = symbols
  return symbols


def _realtime_engine_execution_summary() -> dict[str, Any] | None:
  """실시간 거래 엔진의 실제 활동을 대시보드 요약 형식으로 매핑한다.

  옛 _run_live_trading_execution_cycle 요약을 대체해, '실전 투자 성과' 패널이
  엔진의 제출/차단/오류·매수/매도 평가 수를 그대로 보여주도록 한다.
  """
  with _realtime_trading_lock:
    engine = _realtime_trading_engine
    running = _realtime_trading_worker is not None and _realtime_trading_worker.is_alive()
  if engine is None:
    return None
  status = engine.get_status()
  last = status.get("last_summary") or {}
  buy_eval = int(last.get("buy_evaluated", 0) or 0)
  sell_eval = int(last.get("sell_evaluated", 0) or 0)
  runtime = evaluate_live_runtime_gates(require_manual_arming=_manual_arming_required())
  return {
      "attempted": bool(int(status.get("submitted", 0) or 0) + int(status.get("blocked", 0) or 0) + int(status.get("errors", 0) or 0)),
      "source": "realtime_trading_engine",
      "engine_running": running,
      "signals": buy_eval + sell_eval,
      "buy_signals": buy_eval,
      "sell_signals": sell_eval,
      "intents": buy_eval + sell_eval,
      "approved_buy_orders": int(status.get("buy_submitted", 0) or 0),
      "approved_sell_orders": int(status.get("sell_submitted", 0) or 0),
      "executable_buy_orders": int(status.get("buy_submitted", 0) or 0),
      "executable_sell_orders": int(status.get("sell_submitted", 0) or 0),
      "submitted": int(status.get("submitted", 0) or 0),
      "amended": int(status.get("amended", 0) or 0),
      "buy_submitted": int(status.get("buy_submitted", 0) or 0),
      "sell_submitted": int(status.get("sell_submitted", 0) or 0),
      "blocked": int(status.get("blocked", 0) or 0),
      "errors": int(status.get("errors", 0) or 0),
      "skipped_market_closed": int(last.get("skipped_market_closed", 0) or 0),
      "skipped_cooldown": int(last.get("skipped_cooldown", 0) or 0),
      "buy_rejected": int(last.get("buy_rejected", 0) or 0),
      "sell_rejected": int(last.get("sell_rejected", 0) or 0),
      "rejections": tuple(last.get("rejections", ()) or ())[:12],
      "cycles": int(status.get("cycles", 0) or 0),
      "last_reason": last.get("reason"),
      "runtime_gate": {"ok": runtime.ok, "failures": tuple(runtime.failures)},
      "last_cycle_at": status.get("last_cycle_at"),
      "recent_events": list(status.get("recent_events") or ())[:10],
  }


def _latest_ontology_graph() -> Any | None:
  """최신 분석 컨텍스트(학습/새로고침 루프 산출물)의 온톨로지 그래프를 읽어
  실시간 매도 판단에 반영한다. 없으면 None(매도는 TP/SL로 동작)."""
  with _live_lock:
    context = _live_state.get("context")
  return getattr(context, "graph", None)


def _realtime_trading_loop() -> None:
  global _realtime_trading_engine
  try:
    engine = _build_realtime_trading_engine()
  except Exception as exc:  # noqa: BLE001 - surface build failure, keep server alive.
    audit.record("realtime_trading_engine_start_failed", {"error": str(exc) or exc.__class__.__name__})
    with _live_lock:
      _append_collection_log_unlocked(
          "error",
          f"Realtime trading engine failed to start: {str(exc) or exc.__class__.__name__}",
      )
    return
  with _realtime_trading_lock:
    _realtime_trading_engine = engine
  if os.getenv("LIVE_TERMINATION_SELL_ONLY_ON_START", "false").strip().lower() in {"1", "true", "yes", "on"}:
    try:
      engine.request_full_liquidation("LIVE_TERMINATION_SELL_ONLY_ON_START")
    except Exception as exc:  # noqa: BLE001 - keep server alive; submit guard still blocks BUY.
      audit.record("realtime_trading_sell_only_start_failed", {"error": str(exc) or exc.__class__.__name__})
  audit.record("realtime_trading_engine_started", {"auto_start": AUTO_START_REALTIME_TRADING})
  engine.run_forever(_realtime_trading_stop)


def _start_realtime_trading_engine() -> None:
  global _realtime_trading_worker
  with _realtime_trading_lock:
    if _realtime_trading_worker is not None and _realtime_trading_worker.is_alive():
      return
    _realtime_trading_stop.clear()
    audit.record("realtime_trading_engine_starting", {"auto_start": AUTO_START_REALTIME_TRADING})
    _realtime_trading_worker = threading.Thread(
        target=_realtime_trading_loop,
        name="realtime-trading-engine",
        daemon=True,
    )
    _realtime_trading_worker.start()


def _stop_realtime_trading_engine() -> None:
  _realtime_trading_stop.set()
  with _realtime_trading_lock:
    worker = _realtime_trading_worker
  if worker is not None:
    worker.join(timeout=3.0)


def _kis_realtime_collector_symbols() -> tuple[str, ...]:
  """Symbols the realtime collector subscribes to = static config + today's affordable
  KR buy candidates.

  The KIS realtime WebSocket used here speaks DOMESTIC TR_IDs only (H0STCNT0/H0STASP0),
  so only Korean 6-digit tickers produce data — US affordable names are excluded (they
  would need a separate overseas-realtime subscription). Feeding the affordable KR
  candidates gives THOSE names live ticks/orderbook, so volume-surge/momentum signals
  and model feature frames can form on names the account can actually buy — instead of
  only the two unaffordable config blue-chips.
  """
  base = list(_load_realtime_collection_symbols())
  try:
    requested_max_syms = max(2, int(os.getenv("REALTIME_COLLECTOR_MAX_SYMBOLS", "18")))
  except (TypeError, ValueError):
    requested_max_syms = 18
  try:
    configured_max_subscriptions = max(2, int(os.getenv("KIS_REALTIME_MAX_SUBSCRIPTIONS", "40")))
  except (TypeError, ValueError):
    configured_max_subscriptions = 40
  observed_capacity = _kis_realtime_effective_subscription_capacity()
  max_subscriptions = (
      min(configured_max_subscriptions, observed_capacity)
      if observed_capacity
      else configured_max_subscriptions
  )
  max_syms = max(1, min(requested_max_syms, max_subscriptions // 2))
  held_domestic: list[str] = []
  context_domestic: list[str] = []
  try:
    account = _live_account_snapshot_for_analysis()
  except Exception:  # noqa: BLE001 - collector can still run with static config.
    account = None
  if account is not None:
    for holding in tuple(getattr(account, "holdings", ()) or ()):
      ticker = str(getattr(holding, "ticker", "") or "").upper().strip()
      market = str(getattr(holding, "market", "") or "").upper().strip()
      if ticker.isdigit() and len(ticker) == 6 and _ticker_market_group_for_live_trading(ticker, market) == "KRX":
        held_domestic.append(ticker)
  with _live_lock:
    context = _live_state.get("context")
  for ticker in _cached_context_symbols(context):
    if ticker.isdigit() and len(ticker) == 6:
      context_domestic.append(ticker)
  try:
    extra = _live_affordable_buy_candidate_symbols(limit=max_syms)
  except Exception:  # noqa: BLE001 - candidate discovery is best-effort.
    extra = ()
  try:
    domestic_ranked = _cached_domestic_ranking_symbols()
  except Exception:  # noqa: BLE001 - candidate discovery is best-effort.
    domestic_ranked = ()
  pending_warmup = _pending_krx_buy_candidate_warmup_symbols()
  dashboard_watch = _dashboard_krx_watch_symbols()
  kr_extra = [s for s in extra if str(s).isdigit() and len(str(s)) == 6]
  kr_ranked = [s for s in domestic_ranked if str(s).isdigit() and len(str(s)) == 6]
  candidate_pool = list(
      dict.fromkeys(
          [
              *pending_warmup,
              *kr_extra,
              *kr_ranked,
              *context_domestic,
              *base,
          ]
      )
  )
  if observed_capacity:
    store = RealtimeMarketDataStore()
    candidate_pool = [
        symbol
        for symbol in candidate_pool
        if _kis_realtime_training_candidate_viable(symbol, store)
    ]
    if candidate_pool:
      rotation_seconds = max(
          60,
          _auto_reliability_int("KIS_REALTIME_SYMBOL_ROTATION_SECONDS", 300, 60),
      )
      offset = int(time.time() // rotation_seconds) % len(candidate_pool)
      candidate_pool = candidate_pool[offset:] + candidate_pool[:offset]
  # Session anchors are never rotated or displaced. Session-structure strategies
  # need the SAME symbol at BOTH ends of one session — market intraday momentum
  # compares the 09:00-09:30 window with 14:50-15:20 — and the rotation above
  # reshuffles the pool every KIS_REALTIME_SYMBOL_ROTATION_SECONDS (default 300s).
  # Measured consequence of not pinning: of 360 stored KRX symbol-days, exactly 2
  # carried both windows, so no session-structure strategy could be evaluated at all.
  anchors = _realtime_session_anchor_symbols()
  # Priority order matters and is deliberate:
  #   1. HELD positions — a position with no live feed cannot be priced to exit. This
  #      outranks everything, including the anchors; a research feature must never
  #      displace the data an open trade needs.
  #   2. dashboard watch — the operator is looking at it right now.
  #   3. session anchors — placed ABOVE the rotating pool so they survive rotation,
  #      but below the two above so they can never starve them.
  #   4. affordable/pending candidates, which get the remaining scarce quote+trade
  #      pairs before static blue chips the account may not be able to buy.
  merged = list(
      dict.fromkeys(
          [
              *held_domestic,
              *dashboard_watch,
              *anchors,
              *candidate_pool,
          ]
      )
  )
  return tuple(merged[:max_syms])


def _realtime_session_anchor_symbols() -> tuple[str, ...]:
  """KRX symbols kept subscribed for a whole session, whatever else rotates.

  Deliberately a SMALL set: each anchor consumes one of the scarce quote+trade
  subscription pairs that candidate discovery also needs. Two liquid names are
  enough to produce one session-structure observation per day each, which is what
  turns an unmeasurable strategy into a measurable one.
  """
  raw = os.getenv("REALTIME_SESSION_ANCHOR_SYMBOLS", "")
  symbols = [token.strip() for token in raw.split(",")]
  anchors = [s for s in symbols if s.isdigit() and len(s) == 6]
  try:
    limit = max(0, int(os.getenv("REALTIME_SESSION_ANCHOR_MAX", "2")))
  except (TypeError, ValueError):
    limit = 2
  if anchors or limit == 0:
    return tuple(dict.fromkeys(anchors))[:limit]

  # No implicit issuer preference: choose from observed/ranked/configured data,
  # then the listed universe. The process cache pins the result for the session
  # so intraday strategies see both the opening and closing windows.
  session_key = datetime.now(timezone.utc).date().isoformat()
  cached = _realtime_session_anchor_cache.get("symbols")
  if _realtime_session_anchor_cache.get("session") == session_key and cached:
    return tuple(cached)[:limit]
  candidates: list[str] = []
  for provider in (_cached_domestic_ranking_symbols, _load_realtime_collection_symbols):
    try:
      candidates.extend(provider())
    except Exception:  # noqa: BLE001 - the next dynamic source may still work.
      continue
  if not candidates:
    try:
      candidates.extend(load_krx_listed_universe(limit=max(limit * 4, limit)))
    except Exception:  # noqa: BLE001 - no anchor is safer than inventing a symbol.
      pass
  resolved = tuple(
      dict.fromkeys(
          str(symbol).strip().zfill(6)
          for symbol in candidates
          if str(symbol).strip().isdigit() and len(str(symbol).strip()) <= 6
      )
  )[:limit]
  _realtime_session_anchor_cache.update({"session": session_key, "symbols": resolved})
  return resolved


_realtime_session_anchor_cache: dict[str, Any] = {"session": None, "symbols": ()}


def _kis_realtime_training_candidate_viable(
    symbol: str,
    store: RealtimeMarketDataStore,
) -> bool:
  try:
    tick = store.latest_tick(symbol)
    orderbook = store.latest_orderbook(symbol)
  except Exception:  # noqa: BLE001 - unknown candidates deserve one discovery cycle.
    return True
  minimum_price = max(0.0, _env_float_web("KIS_REALTIME_TRAINING_MIN_PRICE_KRW", 500.0))
  maximum_spread = max(1.0, _env_float_web("LIVE_TRAINING_MAX_SPREAD_BPS", 80.0))
  if tick is not None and float(tick.price) < minimum_price:
    return False
  if orderbook is not None and float(orderbook.spread_bps) > maximum_spread:
    return False
  return True


def _cached_context_symbols(context: Any | None, *, limit: int = 80) -> tuple[str, ...]:
  if context is None:
    return ()
  selected: list[str] = []
  for path in tuple(getattr(context, "reasoning_paths", ()) or ()):
    ticker = str(getattr(path, "ticker", "") or "").upper().strip()
    if ticker:
      selected.append(ticker)
  selection = getattr(context, "candidate_selection", None)
  for ticker_value in tuple(getattr(selection, "candidate_stocks", ()) or ()):
    ticker = str(ticker_value or "").upper().strip()
    if ticker:
      selected.append(ticker)
  return tuple(dict.fromkeys(selected))[: max(1, int(limit))]


def _rest_snapshot_fallback_refresh(symbols: tuple[str, ...], group: str) -> dict[str, int]:
  """Keep last-known prices fresh via REST when a market is fully closed.

  Uses the broker's REST quote (routed by market) and writes a distinct
  ``KIS_REST_SNAPSHOT_SOURCE`` tick, so closed-market prices feed valuation/display
  without ever being treated as a live-tradeable realtime quote.
  """
  from app.data.rest_snapshot_fallback import refresh_rest_snapshot_into_store

  client = _kis_realtime_collector_client()
  market_name = "KRX" if str(group).upper() == "KRX" else "NASDAQ"

  def _refresh(symbol: str, market: str, when: datetime):
    try:
      return client.get_market_snapshot(symbol, market, company_name=symbol, sector="Unknown")
    except Exception:  # noqa: BLE001 - closed-market snapshot is best-effort.
      return None

  return refresh_rest_snapshot_into_store(
      symbols,
      store=RealtimeMarketDataStore(),
      refresher=_refresh,
      market_of=lambda _symbol: market_name,
  )


def _kis_realtime_collector_loop() -> None:
  from app.data.market_session import MarketPhase, market_phase

  resubscribe_seconds = max(30.0, float(os.getenv("REALTIME_COLLECTOR_RESUBSCRIBE_SECONDS", "300")))
  closed_fallback_seconds = max(60.0, float(os.getenv("REALTIME_COLLECTOR_CLOSED_FALLBACK_SECONDS", "300")))
  while not _kis_realtime_collector_stop.is_set():
    if _kis_realtime_session_owner() not in {"KRX", "BOTH"}:
      if _kis_realtime_collector_stop.wait(15.0):
        return
      continue
    symbols = _kis_realtime_collector_symbols()
    if not symbols:
      with _live_lock:
        _append_collection_log_unlocked("error", "KIS realtime collector has no symbols configured")
      if _kis_realtime_collector_stop.wait(30.0):
        return
      continue
    # The 통합 (KRX+NXT) feed carries NXT, whose session runs 08:00-20:00, so
    # PRE and AFTER now have real streaming data. Only fully CLOSED falls back
    # to REST snapshots. On the KRX-only feed there is nothing outside the
    # regular session, so the old behaviour is kept for that configuration.
    from app.data.kis_realtime import _domestic_subscription_tr_ids
    from app.data.market_session import streaming_phase

    subscription_tr_ids = _domestic_subscription_tr_ids()
    # During the regular session the KRX-only trade channel is the dependable
    # source of prints.  The unified channel can continue publishing books while
    # its trade leg is silent, which leaves every candidate stale.  Keep an
    # explicit KIS_REALTIME_FEED override authoritative; otherwise use KRX-only
    # in the core session and unified outside it for NXT coverage.
    if not os.getenv("KIS_REALTIME_FEED", "").strip() and market_phase("KRX") is MarketPhase.REGULAR:
      subscription_tr_ids = ("H0STCNT0", "H0STASP0")
    unified_feed = subscription_tr_ids[0] in {"H0UNCNT0", "H0NXCNT0"}
    streaming_phases = (
        {MarketPhase.REGULAR, MarketPhase.PRE, MarketPhase.AFTER}
        if unified_feed
        else {MarketPhase.REGULAR}
    )
    krx_phase = streaming_phase("KRX", include_nxt=unified_feed)
    if krx_phase not in streaming_phases:
      fallback = _rest_snapshot_fallback_refresh(symbols, "KRX")
      with _live_lock:
        _append_collection_log_unlocked(
            "market_closed",
            f"KRX {krx_phase.value} phase; regular realtime feed inactive, REST snapshot fallback",
            counts={
                "phase": krx_phase.value,
                "symbols": len(symbols),
                "snapshots_saved": int(fallback.get("saved", 0) or 0),
            },
        )
      if _kis_realtime_collector_stop.wait(closed_fallback_seconds):
        return
      continue
    with _live_lock:
      _append_collection_log_unlocked(
          "running",
          "KIS realtime collector subscribed symbols",
          counts={
              "symbols": len(symbols),
              "symbol_sample": list(symbols[:12]),
              "skipped_subscriptions": len(_kis_realtime_collector_skip_pairs()),
              "subscription_budget": int(os.getenv("KIS_REALTIME_MAX_SUBSCRIPTIONS", "40")),
              "phase": market_phase("KRX").value,
          },
      )
    try:
      _kis_realtime_collector_resubscribe.clear()
      use_event_driven = RefactorFeatureFlags.from_env().websocket_market_data
      collector_fn = run_kis_realtime_websocket_collector
      if use_event_driven:
        from app.data.event_runtime import run_event_driven_kis_websocket_collector
        collector_fn = run_event_driven_kis_websocket_collector
      # One persistent session: a resubscribe re-diffs the symbol set in place
      # with tr_type 1/2 instead of reconnecting. Reconnecting minted a new
      # approval key each time and KIS bills registrations per session, which
      # is what drained the account down to a single subscribable symbol.
      counts = asyncio.run(
          collector_fn(
              symbols=symbols,
              symbols_provider=_kis_realtime_collector_symbols,
              store=RealtimeMarketDataStore(),
              client=_kis_realtime_collector_client(),
              stop_event=_kis_realtime_collector_stop,
              resubscribe_event=_kis_realtime_collector_resubscribe,
              skip_subscriptions=_kis_realtime_collector_skip_pairs(),
              max_runtime_seconds=resubscribe_seconds,
              session_active_provider=lambda: (
                  _kis_realtime_session_owner() in {"KRX", "BOTH"}
              ),
              subscription_tr_ids=subscription_tr_ids,
          )
      )
      _record_kis_realtime_collector_result(counts)
      if not _kis_realtime_collector_stop.is_set():
        if counts.get("appkey_already_in_use"):
          time.sleep(max(30.0, _env_float_web("KIS_REALTIME_APPKEY_IN_USE_BACKOFF_SEC", 90.0)))
        elif counts.get("connection_closed") and int(counts.get("messages") or 0) <= 0:
          time.sleep(max(2.0, _env_float_web("KIS_REALTIME_RECONNECT_BACKOFF_SEC", 20.0)))
        else:
          time.sleep(2.0)
    except Exception as exc:  # noqa: BLE001 - keep app startup alive and surface collector failures.
      with _live_lock:
        _append_collection_log_unlocked(
            "error",
            f"KIS realtime collector failed: {str(exc) or exc.__class__.__name__}",
        )
      if _kis_realtime_collector_stop.wait(30.0):
        return


def _kis_realtime_collector_client() -> KisDevelopersApiClient:
  return KisDevelopersApiClient(paper=False, enabled=True)


def _kis_realtime_effective_subscription_capacity() -> int | None:
  """Learned KIS capacity, expired after a retry window.

  KIS answers OPSP0008 while another (often already-dead) session still holds
  the approval key's realtime slots. Honouring that answer forever would keep
  the collector at one symbol long after the slots freed up, so the learned
  value is re-probed periodically.
  """
  if _kis_realtime_observed_subscription_capacity is None:
    return None
  try:
    retry_seconds = float(os.getenv("KIS_REALTIME_CAPACITY_RETRY_SECONDS", "900"))
  except (TypeError, ValueError):
    retry_seconds = 900.0
  if retry_seconds <= 0:
    return _kis_realtime_observed_subscription_capacity
  age = time.monotonic() - _kis_realtime_observed_capacity_at
  if age >= retry_seconds:
    return None  # re-probe the configured maximum on the next subscribe
  return _kis_realtime_observed_subscription_capacity


def _record_kis_realtime_collector_result(counts: dict[str, Any]) -> None:
  global _kis_realtime_observed_subscription_capacity, _kis_realtime_complete_symbols
  global _kis_realtime_observed_capacity_at
  disconnected = bool(counts.get("connection_closed"))
  resubscribe_requested = bool(counts.get("resubscribe_requested"))
  stopping = _kis_realtime_collector_stop.is_set()
  appkey_in_use = bool(counts.get("appkey_already_in_use"))
  accepted = int(counts.get("subscriptions_accepted") or 0)
  if counts.get("subscription_limit_reached") and accepted >= 2:
    # Keep complete quote+trade pairs only. An odd third slot is not enough to
    # make another symbol trade-ready and repeatedly causes OPSP0008 noise.
    _kis_realtime_observed_subscription_capacity = max(2, accepted - (accepted % 2))
    _kis_realtime_observed_capacity_at = time.monotonic()
  accepted_by_symbol: dict[str, set[str]] = {}
  for item in tuple(counts.get("accepted_subscription_pairs") or ()):
    symbol = str(item.get("symbol") or "").strip()
    tr_id = str(item.get("tr_id") or "").strip()
    if symbol and tr_id:
      accepted_by_symbol.setdefault(symbol, set()).add(tr_id)
  # KRX-only, unified KRX+NXT and NXT feeds use different TR IDs for the
  # same trade/orderbook pair.  Treat every supported domestic pair as
  # complete; otherwise the default unified feed is subscribed successfully
  # but the fast feature sampler sees zero symbols.
  domestic_trade_tr_ids = {"H0STCNT0", "H0UNCNT0", "H0NXCNT0"}
  domestic_orderbook_tr_ids = {"H0STASP0", "H0UNASP0", "H0NXASP0"}
  complete_symbols = tuple(
      symbol
      for symbol, tr_ids in accepted_by_symbol.items()
      if tr_ids.intersection(domestic_trade_tr_ids)
      and tr_ids.intersection(domestic_orderbook_tr_ids)
  )
  if complete_symbols:
    with _live_lock:
      _kis_realtime_complete_symbols = complete_symbols
  for item in tuple(counts.get("rejected_subscription_pairs") or ()):
    symbol = str(item.get("symbol") or "").strip()
    tr_id = str(item.get("tr_id") or "").strip()
    if symbol and tr_id and str(item.get("msg_cd") or "").upper() != "OPSP0008":
      _kis_realtime_collector_skipped_subscriptions[(symbol, tr_id)] = time.time()
  if disconnected and not stopping and not appkey_in_use:
    _record_kis_realtime_collector_bad_subscription(counts)
  if appkey_in_use and not stopping:
    status = "reconnecting"
    message = "KIS realtime collector backing off because appkey is already in use"
  elif disconnected and not stopping:
    status = "reconnecting"
    message = "KIS realtime collector reconnecting after WebSocket close"
  elif resubscribe_requested and not stopping:
    status = "running"
    message = "KIS realtime collector resubscribing with refreshed candidate symbols"
  elif disconnected:
    status = "stopped"
    message = "KIS realtime collector stopped after WebSocket close"
  else:
    status = "complete"
    message = "KIS realtime collector ended"
  with _live_lock:
    _append_collection_log_unlocked(status, message, counts=counts)


def _kis_realtime_collector_skip_pairs(now: float | None = None) -> tuple[tuple[str, str], ...]:
  now = time.time() if now is None else now
  ttl_seconds = max(60.0, float(os.getenv("KIS_REALTIME_BAD_SUBSCRIPTION_TTL_SEC", "600")))
  expired = [key for key, recorded_at in _kis_realtime_collector_skipped_subscriptions.items() if now - recorded_at > ttl_seconds]
  for key in expired:
    _kis_realtime_collector_skipped_subscriptions.pop(key, None)
  return tuple(sorted(_kis_realtime_collector_skipped_subscriptions))


def _record_kis_realtime_collector_bad_subscription(counts: dict[str, Any]) -> None:
  symbol = str(counts.get("last_subscription_symbol") or "").strip()
  tr_id = str(counts.get("last_subscription_tr_id") or "").strip()
  if not symbol or not tr_id:
    return
  key = (symbol, tr_id)
  _kis_realtime_collector_skipped_subscriptions[key] = time.time()
  counts["bad_subscription_quarantined"] = {"symbol": symbol, "tr_id": tr_id}
  counts["bad_subscription_skip_ttl_sec"] = max(
      60,
      int(float(os.getenv("KIS_REALTIME_BAD_SUBSCRIPTION_TTL_SEC", "600"))),
  )


def _load_realtime_collection_symbols(path: str | Path = "config/realtime_market_data.json") -> tuple[str, ...]:
  config_path = Path(path)
  data: dict[str, Any] = {}
  for candidate in (config_path,):
    if not candidate.exists():
      continue
    try:
      loaded = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
      continue
    if isinstance(loaded, dict):
      data = loaded
      break
  symbols = data.get("symbols") or os.getenv("KIS_REALTIME_SYMBOLS", "").split(",")
  normalized: list[str] = []
  for item in symbols if isinstance(symbols, list) else []:
    text = str(item).strip()
    if not text:
      continue
    normalized.append(text.zfill(6) if text.isdigit() else text)
  if not normalized:
    for item in os.getenv("KIS_REALTIME_SYMBOLS", "").split(","):
      text = item.strip()
      if text:
        normalized.append(text.zfill(6) if text.isdigit() else text)
  return tuple(dict.fromkeys(normalized))


def _start_live_worker(learning_mode: str | None = None) -> None:
  global _live_worker
  with _live_lock:
    _live_state["stop"] = False
    worker_alive = _live_worker is not None and _live_worker.is_alive()
    if worker_alive:
      if learning_mode is not None:
        _live_state["learning_active"] = True
        _live_state["learning_mode"] = learning_mode
        _live_state["learning_stopped_at"] = None
      return
    if learning_mode is not None:
      now = datetime.now(timezone.utc)
      _live_state["learning_active"] = True
      _live_state["learning_mode"] = learning_mode
      _live_state["learning_started_at"] = now
      _live_state["learning_stopped_at"] = None
      _live_state["learning_next_collection_at"] = now
      _live_state["collection_cycle"] = 0
      _live_state["collection_log"] = []
      _live_state["last_error"] = None
      _append_collection_log_unlocked(
          "scheduled",
          "Learning collection started; first cycle is running now",
          mode=learning_mode,
      )
    _live_worker = threading.Thread(target=_live_worker_loop, name="live-research-refresh", daemon=True)
    _live_worker.start()


def _stop_live_worker() -> None:
  worker: threading.Thread | None
  with _live_lock:
    _live_state["stop"] = True
    _live_state["learning_active"] = False
    _live_state["learning_stopped_at"] = datetime.now(timezone.utc)
    _live_state["learning_next_collection_at"] = None
    _append_collection_log_unlocked("stopped", "Learning collection stopped by user")
    worker = _live_worker
  if worker is not None:
    worker.join(timeout=2.0)
  _set_live_progress(0, "idle", "Learning data collection stopped", active=False)


def _refresh_us_realtime_bridge_dense(context: Any, *, symbols: tuple[str, ...] | None) -> dict[str, Any]:
  """Sample US REST quotes more than once per live cycle to reduce flat feature rows."""
  from app.trading.us_realtime_bridge import refresh_us_realtime_for_context_buy_candidates

  try:
    passes = max(1, min(6, int(os.getenv("REALTIME_US_REST_WARM_PASSES", "2"))))
  except (TypeError, ValueError):
    passes = 2
  try:
    delay_seconds = max(0.0, min(10.0, float(os.getenv("REALTIME_US_REST_WARM_DELAY_SECONDS", "1.5"))))
  except (TypeError, ValueError):
    delay_seconds = 1.5

  summaries: list[dict[str, Any]] = []
  for index in range(passes):
    summaries.append(
        refresh_us_realtime_for_context_buy_candidates(
            context,
            symbols=symbols,
        )
    )
    if index < passes - 1 and delay_seconds > 0.0:
      with _live_lock:
        stopping = bool(_live_state.get("stop"))
      if stopping:
        break
      time.sleep(delay_seconds)

  if not summaries:
    return {
        "ok": True,
        "symbols": tuple(symbols or ()),
        "saved": {"realtime_ticks": 0, "orderbooks": 0},
        "touched": {},
        "errors": {},
        "passes": 0,
        "target_source": "context.reasoning_paths",
    }

  merged_saved = {"realtime_ticks": 0, "orderbooks": 0}
  merged_touched: dict[str, int] = {}
  merged_errors: dict[str, str] = {}
  merged_symbols: list[str] = []
  for summary in summaries:
    saved = summary.get("saved") or {}
    merged_saved["realtime_ticks"] += int(saved.get("realtime_ticks", 0) or 0)
    merged_saved["orderbooks"] += int(saved.get("orderbooks", 0) or 0)
    for symbol in tuple(summary.get("symbols") or ()):
      merged_symbols.append(str(symbol))
    for key, value in (summary.get("touched") or {}).items():
      merged_touched[str(key)] = merged_touched.get(str(key), 0) + int(value or 0)
    for key, value in (summary.get("errors") or {}).items():
      merged_errors[str(key)] = str(value)

  last = summaries[-1]
  return {
      **last,
      "ok": all(bool(summary.get("ok", False)) for summary in summaries),
      "symbols": tuple(dict.fromkeys(merged_symbols)),
      "saved": merged_saved,
      "touched": merged_touched,
      "errors": merged_errors,
      "passes": len(summaries),
      "delay_seconds": delay_seconds,
  }


def _live_worker_loop() -> None:
  _refresh_live_cache()
  while True:
    with _live_lock:
      if _live_state["stop"]:
        break
      learning_active = bool(_live_state.get("learning_active"))
      active_mode = _active_operation_mode()
      interval_seconds = _live_worker_interval_seconds(learning_active, active_mode)
      next_at = datetime.now(timezone.utc) + timedelta(seconds=interval_seconds)
      _live_state["learning_next_collection_at"] = next_at if learning_active else None
    if learning_active:
      _set_live_progress(
          100,
          "waiting",
          f"Next internet and chart data collection starts at {next_at.astimezone().strftime('%H:%M')}",
          active=False,
      )
    slept = 0.0
    while slept < interval_seconds:
      time.sleep(0.5)
      slept += 0.5
      with _live_lock:
        if _live_state["stop"]:
          return
    with _live_lock:
      should_refresh = not _live_state["is_refreshing"]
    if should_refresh:
      _refresh_live_cache()


def _live_worker_interval_seconds(learning_active: bool, active_mode: str | None) -> int:
  if active_mode == "live_trading":
    return LIVE_REFRESH_SECONDS
  return LEARNING_COLLECTION_INTERVAL_SECONDS if learning_active else LIVE_REFRESH_SECONDS


def _refresh_live_cache() -> None:
  with _refresh_guard:
    with _live_lock:
      _live_state["is_refreshing"] = True
      _live_state["collection_cycle"] = int(_live_state.get("collection_cycle") or 0) + 1
      cycle = int(_live_state["collection_cycle"])
      learning_mode = _live_state.get("learning_mode")
      _append_collection_log_unlocked(
          "running",
          "Collecting internet sources and stock chart data",
          cycle=cycle,
          mode=learning_mode,
      )
    started_at = time.monotonic()
    _set_live_progress(5, "starting", "Starting live data refresh")
    try:
      store = LocalResearchStore(root=_get_store_root())
      active_mode = _active_operation_mode()
      research_collected = False
      if active_mode == "live_trading":
        if _live_research_collection_due():
          _set_live_progress(18, "research", "Collecting live news, events, and macro sources")
          research_result = _load_default_research()
          _set_live_progress(48, "storage", "Saving live event and macro research")
          stored_counts = store.save_research_result(research_result)
          research_collected = True
          with _live_lock:
            _live_state["research_last_collected_at"] = datetime.now(timezone.utc)
        else:
          _set_live_progress(18, "broker", "Using recent stored event research and live broker quotes")
          with _live_lock:
            cached_research = _live_state.get("research_result")
          research_result = (
              cached_research
              if isinstance(cached_research, ResearchRunResult)
              else _empty_live_research_result("live_research_interval_not_due")
          )
          stored_counts = {
              "events": 0,
              "raw_records": 0,
              "market_snapshots": 0,
              "macro_metrics": 0,
              "realtime_quotes": 0,
              "realtime_executions": 0,
          }
      else:
        _set_live_progress(18, "research", "Collecting configured market, news, and macro sources")
        research_result = _load_default_research()
        _set_live_progress(48, "storage", "Saving research records")
        stored_counts = store.save_research_result(research_result)
      _set_live_progress(64, "analysis", "Building indicators, ontology graph, and reasoning paths")
      live_account = _live_account_snapshot_for_analysis() if active_mode == "live_trading" else None
      live_risk_rules = (
          _live_risk_rules_for_account(live_account)
          if active_mode == "live_trading"
          else None
      )
      active_market_groups = _active_live_market_groups() if active_mode == "live_trading" else ()
      analysis_research = _analysis_research_for_current_mode(store)
      context_research_result = research_result
      live_broker_quote_summary: dict[str, Any] | None = None
      live_affordable_broker_market_count = 0
      if active_mode == "live_trading":
        if active_market_groups:
          live_broker_targets = _live_broker_targets_for_active_session(analysis_research)
          live_broker_targets = _merge_market_targets(
              live_broker_targets,
              _live_holding_quote_targets(live_account),
              _live_affordable_krx_discovery_targets(analysis_research, live_account, live_broker_targets),
              _live_affordable_us_discovery_targets(analysis_research, live_account, live_broker_targets),
          )
          if live_broker_targets:
            analysis_research, live_broker_quote_summary = _with_live_broker_market_snapshots_for_targets(
                analysis_research,
                live_broker_targets,
            )
          else:
            analysis_research, live_broker_quote_summary = _with_live_broker_market_snapshots(analysis_research)
        else:
          live_broker_quote_summary = {
              "quotes": 0,
              "requested": 0,
              "errors": [],
              "reason": "MARKET_SESSION_CLOSED",
              "active_groups": (),
          }
          analysis_research = replace(analysis_research, market_snapshots=())
        context_research_result = replace(research_result, market_snapshots=())
        analysis_research = _live_broker_only_research(analysis_research, account=live_account)
        live_affordable_broker_market_count = len(tuple(getattr(analysis_research, "market_snapshots", ()) or ()))
      context = build_analysis_context(
          context_research_result,
          analysis_research,
          account_override=live_account,
          risk_rules=live_risk_rules,
      )
      if active_mode == "live_trading":
        analysis_research, buy_quote_summary = _with_live_broker_quotes_for_context_intents(analysis_research, context)
        if int(buy_quote_summary.get("quotes", 0) or 0) > 0:
          live_broker_quote_summary = _merge_quote_summaries(live_broker_quote_summary, buy_quote_summary)
          analysis_research = _live_broker_only_research(analysis_research, account=live_account)
          live_affordable_broker_market_count = len(tuple(getattr(analysis_research, "market_snapshots", ()) or ()))
          context = build_analysis_context(
              context_research_result,
              analysis_research,
              account_override=live_account,
              risk_rules=live_risk_rules,
          )
      model_paths: dict[str, str] = {}
      realtime_examples = build_realtime_supervised_examples(context.temporal_frames, context.signals)
      if learning_mode == "learning":
        _set_live_progress(76, "learning", "Updating realtime supervised model artifacts")
        model_paths = update_realtime_model_artifacts(ModelArtifactStore(), realtime_examples)
      elif learning_mode in {"testing", "paper_trading", "paper_trading_test"}:
        _set_live_progress(76, "paper_trading", "Calculating paper trading realized PnL")
        test_result = run_hypothetical_realtime_test(context.temporal_frames, context.signals)
        model_paths = update_realtime_model_artifacts(ModelArtifactStore(), realtime_examples, test_result)
      _set_live_progress(80, "learning", "Publishing ontology context and current approved model")
      live_feature_symbols = _live_realtime_feature_symbols_for_active_session(context) if active_mode == "live_trading" else None
      if active_mode == "live_trading" and "US" in set(_active_live_market_groups()):
        # ALWAYS warm US realtime data for HELD US symbols. The collection context
        # (reasoning_paths/markets) is KR-centric, so account-held US names never
        # enter the selection above -> the US REST bridge got 0 symbols and no US
        # ticks/orderbooks were ever stored (breaking US feature frames, micro
        # reasoning, and model training). Held symbols must be warmed for exit
        # signals regardless. Independent of the (separately-broken) affordable
        # discovery path. Best-effort; bounded by the small holdings count.
        try:
          _acct = _live_account_snapshot_for_analysis()
          _held_us = tuple(
              str(getattr(h, "ticker", "") or "").upper().strip()
              for h in (getattr(_acct, "holdings", ()) or ())
              if _ticker_market_group_for_live_trading(
                  str(getattr(h, "ticker", "") or ""), str(getattr(h, "market", "") or "")
              ) == "US"
          )
          if _held_us:
            live_feature_symbols = tuple(dict.fromkeys((*(live_feature_symbols or ()), *_held_us)))
        except Exception:  # noqa: BLE001 - warming held US symbols is best-effort.
          pass
        # Warm US realtime data for AFFORDABLE discovery candidates too, not just the
        # ontology BuyCandidates (which skew to unaffordable big-caps). This feeds the
        # REST bridge + feature collection below so cheap US names the account can
        # actually buy accumulate ticks/orderbook → momentum/model signals. Capped to
        # protect the KIS REST rate limit.
        try:
          warm_cap = max(0, int(os.getenv("REALTIME_US_FEATURE_WARM_LIMIT", "10")))
        except (TypeError, ValueError):
          warm_cap = 10
        if warm_cap > 0:
          try:
            affordable_us = tuple(
                s for s in _live_affordable_buy_candidate_symbols(limit=warm_cap)
                if _ticker_market_group_for_live_trading(s, "") == "US"
            )[:warm_cap]
            live_feature_symbols = tuple(dict.fromkeys((*(live_feature_symbols or ()), *affordable_us)))
          except Exception:  # noqa: BLE001 - warming is best-effort.
            pass
      live_us_realtime_bridge_summary = {}
      if active_mode == "live_trading":
        try:
          live_us_realtime_bridge_summary = _refresh_us_realtime_bridge_dense(
              context,
              symbols=live_feature_symbols,
          )
        except Exception as exc:  # noqa: BLE001 - quote bridge failure should be surfaced through feature errors, not crash refresh.
          live_us_realtime_bridge_summary = {
              "ok": False,
              "symbols": tuple(live_feature_symbols or ()),
              "saved": {"realtime_ticks": 0, "orderbooks": 0},
              "touched": {},
              "errors": {"us_realtime_bridge": f"{exc.__class__.__name__}: {exc}"},
              "target_source": "context.reasoning_paths",
          }
      model_registry = ModelArtifactRegistry()
      if AUTO_START_LIVE_TRAINING:
        # The dedicated periodic trainer owns feature collection, fitting, and
        # promotion. Repeating either operation here waits on its cache/training
        # locks and delays publication of the newly built ontology context.
        live_feature_collection = {
            "built": 0,
            "attempted": 0,
            "symbols": tuple(live_feature_symbols or ()),
            "errors": {},
            "deferred_to_periodic_trainer": True,
        }
        try:
          live_model_artifact = json.loads(model_registry.latest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
          live_model_artifact = {
              "artifact_id": None,
              "live_eligible": False,
              "metrics": {},
              "training_skipped": True,
              "skip_reason": "PERIODIC_TRAINER_OWNS_MODEL_UPDATE",
          }
      else:
        live_feature_collection = collect_live_feature_frames_from_realtime_store(symbols=live_feature_symbols)
        live_model_artifact = train_live_short_horizon_from_collected_features()
      artifact_id = str(live_model_artifact.get("artifact_id") or "")
      if live_model_artifact.get("live_eligible") and model_registry.latest_path.exists():
        model_paths["live_short_horizon"] = str(model_registry.latest_path)
      elif artifact_id:
        model_paths["live_short_horizon"] = str(model_registry.root / f"{artifact_id}.json")
      _set_live_progress(84, "graph", "Persisting ontology graph and reasoning paths")
      graph_counts = (
          store.save_graph_and_reasoning(context.graph.triples(), context.reasoning_paths)
          if active_mode != "live_trading" or research_collected
          else {"graph_triples": 0, "reasoning_paths": 0}
      )
      typed_projection_counts = {
          "typed_ohlcv_bars": store.sync_realtime_ohlcv(),
          "typed_realtime_quotes": store.sync_realtime_quotes(),
          "typed_candidate_scores": store.save_typed_candidate_scores(
              tuple(
                  {
                      "ticker": path.ticker,
                      "observed_at": datetime.now(timezone.utc),
                      "stage": "ontology_reasoning",
                      "score": path.confidence,
                      "reason_mask": 0,
                      "backend": "ontology",
                  }
                  for path in context.reasoning_paths
              )
          ),
      }
      # 주문 실행은 독립 실시간 거래 엔진(_realtime_trading_*)이 단독 수행한다.
      # 학습/새로고침 루프는 데이터·모델 수집만 담당하되, 대시보드 요약은 엔진 실제 활동을 반영한다.
      live_execution_summary = _realtime_engine_execution_summary() if active_mode == "live_trading" else None
      with _live_lock:
        _live_state["research_result"] = research_result
        _live_state["context"] = context
        _live_state["context_mode"] = active_mode
        _live_state["graph_payload"] = _graph_payload(context)
        _live_state["graph_payload_context_id"] = id(context)
        _live_state["live_execution_summary"] = live_execution_summary
        _live_state["store_summary"] = store.summary()
        _live_state["stored_new_records"] = {
            **stored_counts,
            **graph_counts,
            **typed_projection_counts,
        }
        _live_state["last_updated"] = datetime.now()
        _live_state["last_error"] = None
        duration_ms = int((time.monotonic() - started_at) * 1000)
        _append_collection_log_unlocked(
            "complete",
            "Collection cycle saved and reflected in analysis",
            cycle=cycle,
            mode=learning_mode,
            duration_ms=duration_ms,
            counts={
                **stored_counts,
                **graph_counts,
                **typed_projection_counts,
                "events_seen": len(research_result.events),
                "raw_records_seen": len(research_result.raw_records),
                "market_snapshots_seen": len(research_result.market_snapshots),
                "macro_metrics_seen": len(research_result.macro_metrics),
                "temporal_frames": len(context.temporal_frames),
                "supervised_examples": len(realtime_examples),
                "model_artifacts": len(model_paths),
                "live_feature_frames_built": int(live_feature_collection.get("built", 0) or 0),
        "live_us_realtime_bridge_symbols": len(tuple((live_us_realtime_bridge_summary or {}).get("symbols", ()) or ())),
        "live_us_realtime_bridge_passes": int((live_us_realtime_bridge_summary or {}).get("passes", 0) or 0),
        "live_us_realtime_bridge_ticks": int(((live_us_realtime_bridge_summary or {}).get("saved", {}) or {}).get("realtime_ticks", 0) or 0),
        "live_us_realtime_bridge_orderbooks": int(((live_us_realtime_bridge_summary or {}).get("saved", {}) or {}).get("orderbooks", 0) or 0),
        "live_us_realtime_bridge_errors": len(((live_us_realtime_bridge_summary or {}).get("errors", {}) or {})),
                "live_affordable_broker_markets": live_affordable_broker_market_count,
                "live_short_horizon_live_eligible": bool(live_model_artifact.get("live_eligible")),
                "live_short_horizon_examples": int(live_model_artifact.get("metrics", {}).get("example_count", 0)),
                "live_broker_quotes": int((live_broker_quote_summary or {}).get("quotes", 0) or 0),
                "live_execution_attempted": bool(live_execution_summary and live_execution_summary.get("attempted")),
                "live_orders_submitted": int((live_execution_summary or {}).get("submitted", 0) or 0),
            },
        )
      _set_live_progress(
          100,
          "waiting",
          "Live analysis cache is ready; background learning will run at the next scheduled cycle",
          active=False,
      )
    except Exception as exc:
      error_traceback = traceback.format_exc()
      with _live_lock:
        _live_state["last_error"] = str(exc)
        _live_state["last_traceback"] = error_traceback
        _append_collection_log_unlocked(
            "error",
            str(exc),
            cycle=cycle,
            mode=learning_mode,
            duration_ms=int((time.monotonic() - started_at) * 1000),
        )
      audit.record("live_refresh_failed", {"error": str(exc), "traceback": error_traceback})
      _set_live_progress(100, "error", str(exc), active=False)
    finally:
      start_followup_refresh = False
      with _live_lock:
        _live_state["is_refreshing"] = False
        start_followup_refresh = bool(_live_state.get("refresh_requested_after_current")) and not bool(_live_state.get("stop"))
        _live_state["refresh_requested_after_current"] = False
      if start_followup_refresh:
        with _live_lock:
          global _refresh_worker
          _refresh_worker = threading.Thread(target=_refresh_live_cache, name="operation-mode-refresh-followup", daemon=True)
          _refresh_worker.start()


def _empty_live_research_result(reason: str) -> ResearchRunResult:
  return ResearchRunResult(
      events=(),
      raw_records=(),
      market_snapshots=(),
      macro_metrics=(),
      skipped_sources=(),
      archived_paths=(),
      diagnostics={
          "events_count": 0,
          "raw_records_count": 0,
          "market_snapshots_count": 0,
          "macro_metrics_count": 0,
          "skipped_count": 0,
          "live_source_count": 0,
          "local_source_count": 0,
          "live_data_present": False,
          "latest_observed_at": None,
          "source_names": [],
          "per_ticker": {},
          "reason": reason,
      },
  )


def _empty_stored_research() -> StoredResearch:
  return StoredResearch(
      events=(),
      raw_records=(),
      market_snapshots=(),
      macro_metrics=(),
      realtime_quotes=(),
      realtime_executions=(),
      graph_triples=(),
      reasoning_paths=(),
  )



_US_LIVE_MARKETS = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS", "CBOE", "IEX", "US"}
_KRX_LIVE_MARKETS = {"KRX", "KOSPI", "KOSDAQ", "KONEX"}


def _ticker_market_group_for_live_trading(ticker: str, market: str = "") -> str:
  """Classify a ticker into the market session used by live execution.

  Numeric six-digit symbols are treated as Korean equities.
  Alphabetic ETF/equity symbols such as AAPL, MSFT, NVDA, QQQ, SOXX are treated as US.
  """
  symbol = str(ticker or "").upper().strip()
  market_name = str(market or "").upper().strip()
  if market_name in _KRX_LIVE_MARKETS or (symbol.isdigit() and len(symbol) == 6):
    return "KRX"
  if symbol and symbol[0].isdigit():
    return "UNKNOWN"
  if market_name in _US_LIVE_MARKETS or (symbol and not (symbol.isdigit() and len(symbol) == 6)):
    return "US"
  return "UNKNOWN"


def _is_live_market_core_open(group: str, now_utc: Any | None = None) -> bool:
  """Return True only during the regular core session for the target market.

  This intentionally does not bypass stale quote/orderbook checks.
  It only prevents KRX symbols from being used while the US market is the active live session.

  세션 경계는 계산하지 않고 ``app.data.market_capabilities`` 에 위임한다 — 이 함수가
  자체 시각창을 갖고 있던 것이 세션 판정 중복의 원인이었다
  (``docs/realtime_session_gap_analysis.md`` §3 항목 2).
  """
  from app.data.market_session import MarketPhase, market_phase

  return market_phase(str(group or ""), _coerce_utc(now_utc)) is MarketPhase.REGULAR


def _is_live_market_extended_open(group: str, now_utc: Any | None = None) -> bool:
  """Return True when KIS has a plausible cash-stock order route open.

  KRX supports pre/post after-hours order divisions on the domestic cash order
  endpoint (``ORD_DVSN`` 05/06/07), NXT has its own verified route, and US-listed
  stocks use the normal overseas route during pre/core/after-market hours plus a
  separate ``daytime-order`` endpoint during the Korean daytime session.

  "주문 route 가 존재하는가" 는 canonical capability 의 ``trade_available`` 과 같은
  질문이므로 그대로 위임한다. **이것은 "신규 진입해도 되는가" 가 아니다** — 그 판정은
  ``MarketSessionService.new_entry_allowed`` 이며 세션별 실주문 승인까지 요구한다.
  """
  from app.data.market_capabilities import default_service, normalize_market_group

  market = normalize_market_group(str(group or ""))
  if market is None:
    return False
  return default_service().trade_available(market, _coerce_utc(now_utc))


def _is_us_market_holiday(day: Any) -> bool:
  """NYSE/Nasdaq 전일 휴장일 여부.

  휴장일 집합은 버전이 기록된 ``config/market_sessions.yaml`` 캘린더 스냅샷에 있다.
  이전에는 이 함수와 ``market_session.py`` 가 각각 사본을 갖고 있어 어긋날 수 있었다.
  """
  from datetime import date as _date

  from app.data.market_capabilities import MarketGroup, default_service

  if isinstance(day, _date):
    target = day
  else:
    try:
      target = _date.fromisoformat(str(day))
    except (TypeError, ValueError):
      return False
  return default_service().calendar.is_holiday(MarketGroup.US, target)


def _coerce_utc(now_utc: Any | None = None) -> Any:
  from datetime import datetime as _datetime
  from datetime import timezone as _timezone

  current = now_utc or _datetime.now(_timezone.utc)
  if getattr(current, "tzinfo", None) is None:
    current = current.replace(tzinfo=_timezone.utc)
  return current


def _active_live_market_groups(now_utc: Any | None = None) -> tuple[str, ...]:
  groups = []
  for group in ("US", "KRX"):
    if _is_live_market_extended_open(group, now_utc):
      groups.append(group)
  return tuple(groups)


def _kis_realtime_session_owner(now_utc: Any | None = None) -> str:
  """Elect the sole market-data WebSocket allowed to use the KIS AppKey.

  KIS rejects a second concurrent realtime socket with OPSP8996. Prefer the
  actual US exchange session when it is open, KRX during its core session, and
  US daytime quotes only after the KRX core session has ended. Setting
  ``KIS_REALTIME_SINGLE_SESSION=false`` restores the legacy parallel behavior
  for accounts that explicitly support multiple sockets.
  """
  if not _env_flag("KIS_REALTIME_SINGLE_SESSION", True):
    return "BOTH"

  from app.data.kis_realtime import is_us_daytime_quote_session
  from app.data.market_session import MarketPhase, market_phase, streaming_phase

  current = now_utc or datetime.now(timezone.utc)
  if market_phase("US", current) is not MarketPhase.CLOSED:
    return "US"
  if _is_live_market_core_open("KRX", current):
    return "KRX"
  if is_us_daytime_quote_session(current):
    return "US"
  if streaming_phase("KRX", current, include_nxt=True) is not MarketPhase.CLOSED:
    return "KRX"
  return "NONE"


def _is_open_live_market_ticker(ticker: str, market: str = "", now_utc: Any | None = None) -> bool:
  group = _ticker_market_group_for_live_trading(ticker, market)
  return group in set(_active_live_market_groups(now_utc))


def _market_name_by_ticker(records: Any) -> dict[str, str]:
  mapping: dict[str, str] = {}
  for market in tuple(records or ()):
    ticker = str(getattr(market, "ticker", "") or "").upper().strip()
    if not ticker:
      continue
    mapping[ticker] = str(getattr(market, "market", "") or "").upper().strip()
  return mapping


def _live_broker_targets_for_active_session(stored: Any, now_utc: Any | None = None) -> tuple[str, ...]:
  """Select domestic and overseas BuyCandidate tickers for broker quote overlay."""
  market_by_ticker = _market_name_by_ticker(getattr(stored, "market_snapshots", ()) or ())
  open_groups = set(_active_live_market_groups(now_utc))
  if not open_groups:
    return ()
  selected: list[str] = []

  for path in tuple(getattr(stored, "reasoning_paths", ()) or ()):
    conclusion = str(getattr(path, "conclusion", "") or "")
    if conclusion != "BuyCandidate":
      continue
    ticker = str(getattr(path, "ticker", "") or "").upper().strip()
    if not ticker:
      continue
    market_group = _ticker_market_group_for_live_trading(ticker, market_by_ticker.get(ticker, ""))
    if market_group in open_groups:
      selected.append(ticker)

  return tuple(dict.fromkeys(selected))


def _merge_market_targets(*groups: tuple[Any, ...]) -> tuple[MarketSnapshot, ...]:
  merged: list[MarketSnapshot] = []
  seen: set[str] = set()
  for group in groups:
    for item in tuple(group or ()):
      if isinstance(item, MarketSnapshot):
        market = item
      else:
        ticker = str(item or "").upper().strip()
        if not ticker:
          continue
        market = _placeholder_live_market(ticker)
      key = market.ticker.upper().strip()
      if not key or key in seen:
        continue
      merged.append(market)
      seen.add(key)
  return tuple(merged)


def _placeholder_live_market(ticker: str, market: str | None = None) -> MarketSnapshot:
  now = datetime.now(timezone.utc)
  symbol = str(ticker or "").upper().strip()
  return MarketSnapshot(
      ticker=symbol,
      market=market or ("KOSDAQ" if symbol.isdigit() and len(symbol) == 6 else "NASDAQ"),
      company_name=symbol,
      sector="Unknown",
      last_price=0.0,
      average_daily_trading_value=0.0,
      volatility_20d=0.03,
      source=SourceMetadata(
          source_name="live_quote_target",
          source_id=f"live-quote-target:{symbol}",
          raw_url=f"local://live-quote-target/{symbol}",
          retrieved_at=now,
      ),
  )


def _live_holding_quote_targets(account: AccountSnapshot | None) -> tuple[MarketSnapshot, ...]:
  if account is None:
    return ()
  targets: list[MarketSnapshot] = []
  for holding in tuple(getattr(account, "holdings", ()) or ()):
    ticker = str(getattr(holding, "ticker", "") or "").upper().strip()
    if not ticker:
      continue
    market = str(getattr(holding, "market", "") or "")
    group = _ticker_market_group_for_live_trading(ticker, market)
    if group not in set(_active_live_market_groups()):
      continue
    targets.append(_placeholder_live_market(ticker, "KOSDAQ" if group == "KRX" else "NASDAQ"))
  return tuple(targets)


def _live_affordable_krx_discovery_targets(
    stored: Any,
    account: AccountSnapshot | None,
    existing_targets: tuple[Any, ...] = (),
) -> tuple[MarketSnapshot, ...]:
  """Add KRX 6-digit symbols for small-cash live quote discovery.

  BuyCandidate paths can be empty or dominated by expensive mega-caps. In live mode,
  quote a bounded set of domestic symbols too so the account-level one-share filter
  can discover genuinely affordable KRX names from broker prices.
  """
  if account is None:
    return ()
  if not _is_live_market_extended_open("KRX"):
    return ()
  krw_cash = _account_available_cash(account, "KRW")
  if krw_cash <= 0:
    return ()
  try:
    limit = max(0, int(os.getenv("LIVE_KRX_AFFORDABLE_DISCOVERY_LIMIT", "300")))
  except ValueError:
    limit = 300
  if limit <= 0:
    return ()

  seen = {
      str(getattr(item, "ticker", item) or "").upper().strip()
      for item in tuple(existing_targets or ())
  }
  candidates: list[MarketSnapshot] = []

  stored_markets = tuple(getattr(stored, "market_snapshots", ()) or ())
  for market in sorted(
      stored_markets,
      key=lambda item: float(getattr(item, "last_price", 0.0) or 0.0),
  ):
    ticker = str(getattr(market, "ticker", "") or "").upper().strip()
    if (
        ticker in seen
        or not (ticker.isdigit() and len(ticker) == 6)
        or _ticker_market_group_for_live_trading(ticker, getattr(market, "market", "")) != "KRX"
    ):
      continue
    candidates.append(market)
    seen.add(ticker)
    if len(candidates) >= limit:
      break

  if len(candidates) < limit:
    try:
      universe = load_krx_listed_universe(limit=None)
    except Exception as exc:  # noqa: BLE001 - discovery is an optional expansion.
      audit.record("live_affordable_krx_universe_load_failed", {"error": str(exc)})
      universe = ()
    for symbol in universe:
      ticker = str(symbol or "").upper().split(".", 1)[0]
      if ticker in seen or not (ticker.isdigit() and len(ticker) == 6):
        continue
      candidates.append(_placeholder_live_market(ticker, "KOSDAQ" if symbol.endswith(".KQ") else "KOSPI"))
      seen.add(ticker)
      if len(candidates) >= limit:
        break

  if candidates:
    audit.record(
        "live_affordable_krx_discovery_targets_added",
        {"targets": len(candidates), "krw_cash": krw_cash, "limit": limit},
    )
  return tuple(candidates)


def _live_affordable_us_discovery_targets(
    stored: Any,
    account: AccountSnapshot | None,
    existing_targets: tuple[Any, ...] = (),
) -> tuple[MarketSnapshot, ...]:
  """Add US symbols for tiny USD balances so live mode can find one-share fits."""
  if account is None:
    return ()
  if not _is_live_market_extended_open("US"):
    return ()
  usd_cash = _account_available_cash(account, "USD")
  if usd_cash <= 0:
    return ()
  try:
    limit = max(0, int(os.getenv("LIVE_US_AFFORDABLE_DISCOVERY_LIMIT", "120")))
  except ValueError:
    limit = 120
  if limit <= 0:
    return ()

  seen = {
      str(getattr(item, "ticker", item) or "").upper().strip()
      for item in tuple(existing_targets or ())
  }
  excluded = _held_or_recent_buy_tickers(account)
  candidates: list[MarketSnapshot] = []
  stored_markets = tuple(getattr(stored, "market_snapshots", ()) or ())
  for market in _rotated_affordable_markets(stored_markets):
    ticker = str(getattr(market, "ticker", "") or "").upper().strip()
    if (
        ticker in seen
        or ticker in excluded
        or not ticker
        or _ticker_market_group_for_live_trading(ticker, getattr(market, "market", "")) != "US"
    ):
      continue
    candidates.append(market)
    seen.add(ticker)
    if len(candidates) >= limit:
      break

  if len(candidates) < limit:
    try:
      us_exchange_map = _load_us_listed_exchange_map()
    except Exception as exc:  # noqa: BLE001 - discovery is optional.
      audit.record("live_affordable_us_universe_load_failed", {"error": str(exc)})
      us_exchange_map = {}
    universe = tuple(us_exchange_map.keys()) or tuple(_load_us_nasdaq_universe() or ())
    for symbol in _rotated_symbols(tuple(universe)):
      ticker = str(symbol or "").upper().strip().split(".", 1)[0]
      if ticker in seen or ticker in excluded or not ticker or _is_excluded_us_live_candidate(ticker):
        continue
      candidates.append(_placeholder_live_market(ticker, us_exchange_map.get(ticker, "NASDAQ")))
      seen.add(ticker)
      if len(candidates) >= limit:
        break

  if candidates:
    audit.record(
        "live_affordable_us_discovery_targets_added",
        {"targets": len(candidates), "usd_cash": usd_cash, "limit": limit},
    )
  return tuple(candidates)


def _held_or_recent_buy_tickers(account: AccountSnapshot | None) -> set[str]:
  excluded = {
      str(getattr(holding, "ticker", "") or "").upper().strip()
      for holding in tuple(getattr(account, "holdings", ()) or ())
      if str(getattr(holding, "ticker", "") or "").strip()
  }
  excluded.update(_recent_live_buy_tickers())
  return excluded


def _recent_live_buy_tickers(path: str | Path = "logs/live-orders.jsonl") -> set[str]:
  try:
    cooldown_seconds = max(0, int(os.getenv("LIVE_BUY_TICKER_COOLDOWN_SECONDS", "21600")))
  except ValueError:
    cooldown_seconds = 21600
  if cooldown_seconds <= 0:
    return set()
  cutoff = datetime.now(timezone.utc) - timedelta(seconds=cooldown_seconds)
  tickers: set[str] = set()
  try:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
  except OSError:
    return tickers
  for line in lines[-200:]:
    try:
      event = json.loads(line)
    except json.JSONDecodeError:
      continue
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    order = payload.get("order") if isinstance(payload.get("order"), dict) else {}
    side = str(payload.get("side") or order.get("side") or "").upper()
    if side != "BUY":
      continue
    recorded_at = _parse_iso_datetime(event.get("recorded_at"))
    if recorded_at is not None and recorded_at < cutoff:
      continue
    ticker = str(payload.get("ticker") or order.get("ticker") or "").upper().strip()
    if ticker:
      tickers.add(ticker)
  return tickers


def _parse_iso_datetime(value: Any) -> datetime | None:
  if not value:
    return None
  try:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
  except ValueError:
    return None
  if parsed.tzinfo is None:
    parsed = parsed.replace(tzinfo=timezone.utc)
  return parsed.astimezone(timezone.utc)


def _rotated_affordable_markets(markets: tuple[MarketSnapshot, ...]) -> tuple[MarketSnapshot, ...]:
  ordered = tuple(
      sorted(
          markets,
          key=lambda item: (
              round(float(getattr(item, "last_price", 0.0) or 0.0), 2),
              str(getattr(item, "ticker", "") or ""),
          ),
      )
  )
  if not ordered:
    return ()
  offset = datetime.now(timezone.utc).toordinal() % len(ordered)
  return ordered[offset:] + ordered[:offset]


def _rotated_symbols(symbols: tuple[Any, ...]) -> tuple[Any, ...]:
  if not symbols:
    return ()
  count = len(symbols)
  bucket = int(time.time() // max(30, int(os.getenv("REALTIME_UNIVERSE_ROTATION_SECONDS", "60"))))
  offset = bucket % count
  rotated = symbols[offset:] + symbols[:offset]
  if count < 80:
    return rotated
  step = max(1, min(count - 1, int(os.getenv("REALTIME_UNIVERSE_SPREAD_STEP", "37"))))
  while math.gcd(step, count) != 1 and step > 1:
    step -= 1
  return tuple(rotated[(index * step) % count] for index in range(count))


def _live_realtime_feature_symbols_for_active_session(context: Any, now_utc: Any | None = None) -> tuple[str, ...]:
  """Limit live feature frame collection to the market that is actually open.

  This prevents stale domestic realtime rows from blocking US live trading cycles.
  """
  open_groups = set(_active_live_market_groups(now_utc))
  if not open_groups:
    return ()

  markets = tuple(getattr(context, "markets", ()) or ())
  market_by_ticker = _market_name_by_ticker(markets)
  selected: list[str] = []

  for path in tuple(getattr(context, "reasoning_paths", ()) or ()):
    conclusion = str(getattr(path, "conclusion", "") or "")
    if conclusion != "BuyCandidate":
      continue
    ticker = str(getattr(path, "ticker", "") or "").upper().strip()
    if not ticker:
      continue
    market_group = _ticker_market_group_for_live_trading(ticker, market_by_ticker.get(ticker, ""))
    if market_group in open_groups:
      selected.append(ticker)

  if not selected:
    for market in markets:
      ticker = str(getattr(market, "ticker", "") or "").upper().strip()
      if not ticker:
        continue
      market_group = _ticker_market_group_for_live_trading(ticker, getattr(market, "market", ""))
      if market_group in open_groups:
        selected.append(ticker)

  return tuple(dict.fromkeys(selected))

def _with_live_broker_market_snapshots(stored: StoredResearch) -> tuple[StoredResearch, dict[str, Any]]:
  markets = tuple(getattr(stored, "market_snapshots", ()) or ())
  if not markets:
    return stored, {"quotes": 0, "errors": [], "message": "no stored markets available for broker quote overlay"}
  try:
    limit = max(1, int(os.getenv("LIVE_BROKER_QUOTE_LIMIT", "240")))
  except ValueError:
    limit = 240
  client = KisDevelopersApiClient(paper=False, enabled=True)
  quoted: list[MarketSnapshot] = []
  errors: list[dict[str, str]] = []
  targets = _candidate_live_quote_markets(markets)
  for market in targets[:limit]:
    try:
      snapshot = client.get_market_snapshot(
          market.ticker,
          market.market,
          company_name=market.company_name,
          sector=market.sector,
      )
    except Exception as exc:  # noqa: BLE001 - one quote failure should not stop the live refresh.
      errors.append({"ticker": market.ticker, "market": market.market, "error": str(exc)})
      continue
    if snapshot.last_price > 0:
      quoted.append(snapshot)
  if not quoted:
    audit.record(
        "live_broker_quote_overlay_empty",
        {"requested": min(limit, len(markets)), "errors": errors[:10]},
    )
    return stored, {"quotes": 0, "errors": errors[:10], "message": "KIS broker quote overlay did not return usable quotes"}
  by_ticker = {market.ticker: market for market in markets}
  for snapshot in quoted:
    by_ticker[snapshot.ticker] = snapshot
  merged = tuple(by_ticker.values())
  summary = {"quotes": len(quoted), "requested": min(limit, len(markets)), "errors": errors[:10]}
  audit.record("live_broker_quote_overlay_applied", summary)
  return replace(stored, market_snapshots=merged), summary


def _candidate_live_quote_markets(markets: tuple[MarketSnapshot, ...]) -> tuple[MarketSnapshot, ...]:
  prioritized = _prioritized_live_quote_markets(markets)
  by_ticker = {market.ticker: market for market in prioritized}
  candidate_tickers: list[str] = []
  try:
    snapshots = build_lightweight_market_snapshots_from_markets(markets)
    selection = ontology_filter_1(snapshots, target_count=max(80, min(160, len(markets))))
    candidate_tickers.extend(selection.candidate_stocks)
  except Exception as exc:  # noqa: BLE001 - quote overlay can still use priority fallback.
    audit.record("live_broker_quote_candidate_selection_failed", {"error": str(exc)})
  selected: list[MarketSnapshot] = []
  seen: set[str] = set()
  for ticker in candidate_tickers:
    market = by_ticker.get(ticker)
    if market is not None and ticker not in seen:
      selected.append(market)
      seen.add(ticker)
  for market in prioritized:
    if market.ticker not in seen:
      selected.append(market)
      seen.add(market.ticker)
  return tuple(selected)


def _with_live_broker_quotes_for_context_intents(stored: StoredResearch, context: Any) -> tuple[StoredResearch, dict[str, Any]]:
  markets_by_ticker = {market.ticker: market for market in tuple(getattr(context, "markets", ()) or ())}
  targets: list[MarketSnapshot] = []
  seen: set[str] = set()
  for intent in tuple(getattr(context, "intents", ()) or ()):
    ticker = str(getattr(intent, "ticker", "") or "")
    market = markets_by_ticker.get(ticker)
    if (
        market is None
        or ticker in seen
        or market.source.source_type == "broker_api"
        or not _is_open_live_market_ticker(ticker, market.market)
    ):
      continue
    targets.append(market)
    seen.add(ticker)
  if not targets:
    return stored, {"quotes": 0, "requested": 0, "errors": []}
  updated, summary = _with_live_broker_market_snapshots_for_targets(stored, tuple(targets))
  audit.record("live_broker_quote_intent_overlay_applied", summary)
  return updated, summary


def _live_broker_only_research(stored: StoredResearch, *, account: AccountSnapshot | None = None) -> StoredResearch:
  source_markets = tuple(getattr(stored, "market_snapshots", ()) or ())
  active_groups = _active_live_market_groups()
  if not active_groups:
    if source_markets:
      audit.record(
          "live_broker_only_research_empty",
          {
              "input_markets": len(source_markets),
              "broker_rejections": (),
              "active_groups": active_groups,
              "reason": "MARKET_SESSION_CLOSED",
          },
      )
    return replace(stored, market_snapshots=())
  broker_markets: list[MarketSnapshot] = []
  rejected: list[dict[str, Any]] = []
  held_tickers = {
      str(getattr(holding, "ticker", "") or "").upper().strip()
      for holding in tuple(getattr(account, "holdings", ()) or ())
  }
  for market in source_markets:
    reason = ""
    ticker = str(getattr(market, "ticker", "") or "").upper().strip()
    is_held_position = ticker in held_tickers
    if market.source.source_type != "broker_api":
      reason = "NOT_BROKER_API"
    elif market.source.trust_level < 5:
      reason = "LOW_SOURCE_TRUST"
    elif market.source.quality_score < 0.8:
      reason = "LOW_SOURCE_QUALITY"
    elif not market.source.is_realtime:
      reason = "NOT_REALTIME"
    elif market.last_price <= 0:
      reason = "PRICE_NOT_POSITIVE"
    elif not is_held_position and not is_market_affordable_for_account(market, account):
      reason = "INSUFFICIENT_CASH_FOR_ONE_SHARE"
    elif not _is_open_live_market_ticker(market.ticker, market.market):
      reason = "MARKET_SESSION_CLOSED"
    if reason:
      if market.source.source_type == "broker_api":
        rejected.append(
            {
                "ticker": market.ticker,
                "market": market.market,
                "last_price": market.last_price,
                "reason": reason,
            }
        )
      continue
    broker_markets.append(market)
  if source_markets and not broker_markets:
    audit.record(
        "live_broker_only_research_empty",
        {
            "input_markets": len(source_markets),
            "broker_rejections": rejected[:20],
            "active_groups": active_groups,
        },
    )
  return replace(stored, market_snapshots=broker_markets)


def _is_market_affordable_for_account(market: MarketSnapshot, account: AccountSnapshot | None) -> bool:
  return is_market_affordable_for_account(market, account)


def _cash_available_for_market_web(account: AccountSnapshot, market: MarketSnapshot) -> float:
  return cash_available_for_market(account, market)


def _market_currency_for_web(market: MarketSnapshot) -> str:
  return market_currency(market)


def _with_live_broker_market_snapshots_for_targets(
  stored: StoredResearch,
  targets: tuple[MarketSnapshot, ...],
) -> tuple[StoredResearch, dict[str, Any]]:
  client = KisDevelopersApiClient(paper=False, enabled=True)
  quoted: list[MarketSnapshot] = []
  errors: list[dict[str, str]] = []
  for market in targets:
    try:
      snapshot = client.get_market_snapshot(
          market.ticker,
          market.market,
          company_name=market.company_name,
          sector=market.sector,
      )
    except Exception as exc:  # noqa: BLE001 - one quote failure should not stop the live refresh.
      errors.append({"ticker": market.ticker, "market": market.market, "error": str(exc)})
      continue
    if snapshot.last_price > 0:
      quoted.append(snapshot)
  if not quoted:
    return stored, {"quotes": 0, "requested": len(targets), "errors": errors[:10]}
  by_ticker = {market.ticker: market for market in tuple(getattr(stored, "market_snapshots", ()) or ())}
  for snapshot in quoted:
    by_ticker[snapshot.ticker] = snapshot
  return replace(stored, market_snapshots=tuple(by_ticker.values())), {
      "quotes": len(quoted),
      "requested": len(targets),
      "errors": errors[:10],
  }


def _merge_quote_summaries(base: dict[str, Any] | None, extra: dict[str, Any]) -> dict[str, Any]:
  merged = dict(base or {})
  merged["quotes"] = int(merged.get("quotes", 0) or 0) + int(extra.get("quotes", 0) or 0)
  merged["requested"] = int(merged.get("requested", 0) or 0) + int(extra.get("requested", 0) or 0)
  merged["errors"] = [*(merged.get("errors", []) or []), *(extra.get("errors", []) or [])][:10]
  return merged


def _prioritized_live_quote_markets(markets: tuple[MarketSnapshot, ...]) -> tuple[MarketSnapshot, ...]:
  def priority(market: MarketSnapshot) -> tuple[int, float, str]:
    source = market.source
    trusted = source.source_type == "broker_api" or "kis" in source.source_name.lower()
    reference = source.source_name == "listed_universe_reference"
    overseas = _is_overseas_market_for_web(market)
    return (
        0 if trusted else 1 if overseas else 2 if not reference else 3,
        -float(market.average_daily_trading_value or 0.0),
        market.ticker,
    )

  deduped: dict[str, MarketSnapshot] = {}
  for market in sorted(markets, key=priority):
    deduped.setdefault(market.ticker, market)
  return tuple(deduped.values())


def _is_overseas_market_for_web(market: MarketSnapshot) -> bool:
  return is_overseas_market(market)


def _run_live_trading_execution_cycle(context: Any) -> dict[str, Any]:
  intents_count = len(getattr(context, "intents", ()) or ())
  risk_results_count = len(getattr(context, "risk_results", ()) or ())
  signals = tuple(getattr(context, "signals", ()) or ())
  buy_signals_count = sum(1 for signal in signals if getattr(signal, "action", None) == OrderAction.BUY)
  sell_signals_count = sum(1 for signal in signals if getattr(signal, "action", None) in {OrderAction.SELL, OrderAction.REDUCE})
  account = getattr(context, "account", None)
  account_cash = float(getattr(account, "cash", 0.0) or 0.0)
  account_cash_equivalent = float(getattr(account, "cash_equivalent_krw", 0.0) or 0.0)
  if account_cash_equivalent <= 0:
    account_cash_equivalent = account_cash
  approved_orders = [
      result.final_order
      for result in getattr(context, "risk_results", ()) or ()
      if getattr(result, "approved", False) and getattr(result, "final_order", None) is not None
  ]
  executable_orders = [
      order
      for order in approved_orders
      if _is_live_executable_order(order)
  ]
  executable_orders, cash_fit_skipped_orders = _cash_fit_executable_orders(executable_orders, account)
  executable_orders, session_fit_skipped_orders = _session_fit_executable_orders(executable_orders)
  approved_buy_orders = [order for order in approved_orders if _order_side_name(order) == "BUY"]
  approved_sell_orders = [order for order in approved_orders if _order_side_name(order) == "SELL"]
  executable_buy_orders = [order for order in executable_orders if _order_side_name(order) == "BUY"]
  executable_sell_orders = [order for order in executable_orders if _order_side_name(order) == "SELL"]
  summary: dict[str, Any] = {
      "attempted": False,
      "approved_orders": len(approved_orders),
      "approved_buy_orders": len(approved_buy_orders),
      "approved_sell_orders": len(approved_sell_orders),
      "executable_orders": len(executable_orders),
      "executable_buy_orders": len(executable_buy_orders),
      "executable_sell_orders": len(executable_sell_orders),
      "markets": len(getattr(context, "markets", ()) or ()),
      "signals": len(signals),
      "buy_signals": buy_signals_count,
      "sell_signals": sell_signals_count,
      "hold_signals": max(0, len(signals) - buy_signals_count - sell_signals_count),
      "intents": intents_count,
      "risk_results": risk_results_count,
      "account_cash": account_cash,
      "account_cash_equivalent_krw": account_cash_equivalent,
      "holdings_count": len(getattr(account, "holdings", ()) or ()),
      "submitted": 0,
      "blocked": [],
      "errors": [],
      "cash_fit_skipped_orders": cash_fit_skipped_orders,
        "session_fit_skipped_orders": session_fit_skipped_orders,
  }
  runtime_snapshot = evaluate_live_runtime_gates(require_manual_arming=_manual_arming_required())
  summary["runtime_gate"] = {"ok": runtime_snapshot.ok, "failures": tuple(runtime_snapshot.failures)}
  if intents_count <= 0:
    active_groups = _active_live_market_groups()
    summary["active_market_groups"] = active_groups
    summary["reason"] = "MARKET_SESSION_CLOSED" if not active_groups else "NO_ORDER_INTENTS"
    summary["diagnostics"] = {
        "signals_present": summary["signals"] > 0,
        "buy_signals_present": buy_signals_count > 0,
        "markets_present": summary["markets"] > 0,
        "active_market_groups": active_groups,
        "message": (
            "No supported KIS live trading session is open."
            if not active_groups
            else "Realtime signals were collected, but none became live order intents for the current "
            "market/account state."
        ),
    }
    audit.record("live_trading_execution_skipped", summary)
    return summary
  if risk_results_count <= 0:
    summary["reason"] = "NO_RISK_RESULTS"
    summary["diagnostics"] = {
        "message": "Order intents were not converted into RiskManager results for live execution.",
    }
    audit.record("live_trading_execution_skipped", summary)
    return summary
  if not approved_orders:
    rejections = [
        {
            "ticker": getattr(result, "ticker", ""),
            "rejection_reasons": tuple(getattr(result, "rejection_reasons", ()) or ()),
        }
        for result in (getattr(context, "risk_results", ()) or ())[:10]
        if not getattr(result, "approved", False)
    ]
    summary["rejections"] = rejections
    summary["reason"] = "NO_APPROVED_FINAL_ORDERS"
    audit.record("live_trading_execution_skipped", summary)
    return summary
  if not executable_orders:
    summary["skipped_orders"] = [
        {"ticker": getattr(order, "ticker", ""), "market": getattr(order, "market", "")}
        for order in approved_orders
    ]
    summary["reason"] = "NO_EXECUTABLE_LIVE_FINAL_ORDERS"
    audit.record("live_trading_execution_skipped", summary)
    return summary

  runtime = runtime_snapshot
  if not runtime.ok:
    summary["blocked"] = list(runtime.failures)
    audit.record("live_trading_execution_blocked", summary)
    return summary

  coordinator = LiveExecutionCoordinator(KisDevelopersApiClient(paper=False, enabled=True))
  session_key = _live_trading_session_key()
  max_submissions = _live_max_auto_order_submissions_per_cycle()
  for order in executable_orders:
    if summary["submitted"] >= max_submissions:
      summary.setdefault("submission_fit_skipped_orders", []).append(
          {
              "ticker": getattr(order, "ticker", ""),
              "market": getattr(order, "market", ""),
              "reason": "MAX_AUTO_ORDERS_PER_CYCLE_REACHED",
              "limit": max_submissions,
          }
      )
      continue
    summary["attempted"] = True
    key = (
        f"auto-live:{session_key}:{getattr(order, 'ticker', '')}:{_order_side_name(order)}:"
        f"{getattr(order, 'quantity', '')}:{getattr(order, 'limit_price', '')}"
    )
    try:
      submission = coordinator.submit_final_order(order, idempotency_key=key)
    except LiveExecutionBlocked as exc:
      summary["blocked"].append({"ticker": order.ticker, "reason_codes": tuple(exc.reason_codes)})
      continue
    except Exception as exc:  # noqa: BLE001 - live cycle must log and continue safely.
      summary["errors"].append({"ticker": order.ticker, "error_type": exc.__class__.__name__, "message": str(exc)})
      continue
    summary["submitted"] += 1
    audit.record(
        "live_trading_order_submitted",
        {
            "ticker": order.ticker,
            "quantity": order.quantity,
            "limit_price": order.limit_price,
            "broker_order_id": submission.broker_order_id,
            "status": submission.status,
            "execution_id": submission.execution_id,
        },
    )
  event = "live_trading_execution_completed" if summary["submitted"] else "live_trading_execution_blocked"
  audit.record(event, summary)
  return summary


def _live_max_auto_order_submissions_per_cycle() -> int:
  try:
    return max(1, int(os.getenv("LIVE_MAX_AUTO_ORDERS_PER_CYCLE", "1")))
  except ValueError:
    return 1


def _live_trading_session_key() -> str:
  with _live_lock:
    active = _operation_mode_state.get("active")
  if isinstance(active, dict):
    started_at = str(active.get("started_at") or "")
    if started_at:
      return re.sub(r"[^0-9A-Za-z]+", "", started_at)[:32]
  return datetime.now(timezone.utc).strftime("%Y%m%d%H%M")


def _is_live_executable_order(order: FinalOrder) -> bool:
  ticker = str(getattr(order, "ticker", "") or "")
  market = str(getattr(order, "market", "") or "").upper()
  if ticker.isdigit() and len(ticker) == 6:
    return True
  if not ticker.replace(".", "").replace("-", "").isalnum():
    return False
  return any(
      token in market
      for token in ("US", "NASDAQ", "NASD", "NYSE", "AMEX", "SEHK", "SHAA", "SZAA", "TKSE", "HASE", "VNSE", "OVERSEAS")
  )


def _order_side_name(order: Any) -> str:
  side = getattr(order, "side", "")
  return str(getattr(side, "value", side) or "").upper()


def _cash_fit_executable_orders(
    orders: list[FinalOrder],
    account: Any,
) -> tuple[list[FinalOrder], list[dict[str, Any]]]:
  if not orders:
    return orders, []
  cash_by_currency = dict(getattr(account, "cash_by_currency", {}) or {})
  remaining: dict[str, float] = {
      "KRW": float(cash_by_currency.get("KRW", getattr(account, "cash", 0.0)) or 0.0),
      "USD": float(cash_by_currency.get("USD", 0.0) or 0.0),
  }
  kept: list[FinalOrder] = []
  skipped: list[dict[str, Any]] = []
  for order in sorted(orders, key=lambda item: float(item.limit_price) * int(item.quantity)):
    if _order_side_name(order) != "BUY":
      kept.append(order)
      continue
    currency = "KRW" if str(order.ticker).isdigit() and len(str(order.ticker)) == 6 else "USD"
    required = float(order.limit_price) * int(order.quantity)
    if required <= max(0.0, remaining.get(currency, 0.0)):
      kept.append(order)
      remaining[currency] = max(0.0, remaining.get(currency, 0.0) - required)
    else:
      skipped.append(
          {
              "ticker": order.ticker,
              "quantity": order.quantity,
              "limit_price": order.limit_price,
              "currency": currency,
              "required_cash": required,
              "remaining_cash": remaining.get(currency, 0.0),
              "reason": "WOULD_EXCEED_REMAINING_CASH",
          }
      )
  return kept, skipped


def _session_fit_executable_orders(
    orders: list[FinalOrder],
    now_utc: Any | None = None,
) -> tuple[list[FinalOrder], list[dict[str, Any]]]:
  if not orders:
    return orders, []
  active_groups = set(_active_live_market_groups(now_utc))
  kept: list[FinalOrder] = []
  skipped: list[dict[str, Any]] = []
  for order in orders:
    ticker = str(getattr(order, "ticker", "") or "")
    market = str(getattr(order, "market", "") or "")
    group = _ticker_market_group_for_live_trading(ticker, market)
    if group in active_groups:
      kept.append(order)
      continue
    skipped.append(
        {
            "ticker": ticker,
            "market": market,
            "side": _order_side_name(order),
            "quantity": int(getattr(order, "quantity", 0) or 0),
            "limit_price": float(getattr(order, "limit_price", 0.0) or 0.0),
            "required_group": group,
            "active_groups": tuple(sorted(active_groups)),
            "reason": "MARKET_SESSION_CLOSED",
        }
    )
  return kept, skipped


def _get_or_refresh_live(force_refresh: bool = False) -> dict[str, Any]:
  snapshot = _live_snapshot()
  current_mode = _active_operation_mode()
  cache_matches_mode = snapshot.get("context_mode") == current_mode
  if snapshot["context"] is not None and cache_matches_mode and not force_refresh:
    return snapshot
  if force_refresh:
    _ensure_background_refresh()
    if snapshot["context"] is not None and cache_matches_mode:
      return snapshot
  if (snapshot["context"] is None or not cache_matches_mode) and not force_refresh:
    _ensure_background_refresh()
    _build_current_snapshot_from_store()
    return _live_snapshot()
  last_updated = snapshot["last_updated"]
  stale = (
    last_updated is None
    or (datetime.now() - last_updated).total_seconds() > LIVE_STALE_SECONDS
  )
  if force_refresh or stale or snapshot["context"] is None:
    _ensure_background_refresh()
    snapshot = _live_snapshot()

  if snapshot["context"] is None or snapshot["research_result"] is None:
    _build_current_snapshot_from_store()
    snapshot = _live_snapshot()
  return snapshot


def _build_current_snapshot_from_store() -> None:
  store = LocalResearchStore(root=_get_store_root())
  context = build_analysis_context(stored_research=_analysis_research_for_current_mode(store))
  current_mode = _active_operation_mode()
  with _live_lock:
    if _live_state["context"] is not None and _live_state.get("context_mode") == current_mode:
      return
    _live_state["context"] = context
    _live_state["context_mode"] = current_mode
    _live_state["graph_payload"] = _graph_payload(context)
    _live_state["graph_payload_context_id"] = id(context)
    _live_state["research_result"] = ResearchRunResult(
      events=(),
      raw_records=(),
      market_snapshots=(),
      macro_metrics=(),
      skipped_sources=(),
      archived_paths=(),
      diagnostics={
        "events_count": 0,
        "raw_records_count": 0,
        "market_snapshots_count": 0,
        "macro_metrics_count": 0,
        "skipped_count": 0,
        "live_source_count": 0,
        "local_source_count": 0,
        "live_data_present": False,
        "latest_observed_at": None,
        "source_names": [],
        "per_ticker": {},
      },
    )
    _live_state["store_summary"] = store.summary()
    _live_state["stored_new_records"] = {}
    _live_state["last_updated"] = datetime.now()
    _live_state["last_error"] = None


def _live_snapshot() -> dict[str, Any]:
  with _live_lock:
    return {
      "context": _live_state["context"],
      "research_result": _live_state["research_result"],
      "context_mode": _live_state.get("context_mode"),
      "store_summary": dict(_live_state["store_summary"]),
      "stored_new_records": dict(_live_state["stored_new_records"]),
      "last_updated": _live_state["last_updated"],
      "last_error": _live_state["last_error"],
      "is_refreshing": bool(_live_state["is_refreshing"]),
      "progress": dict(_live_state["progress"]),
      "learning": _learning_state_snapshot_unlocked(),
      "collection_log": list(_live_state.get("collection_log") or ()),
      "graph_payload": _live_state.get("graph_payload"),
      "graph_payload_context_id": _live_state.get("graph_payload_context_id"),
      "live_execution_summary": _live_state.get("live_execution_summary"),
    }


def _learning_state_snapshot() -> dict[str, Any]:
  with _live_lock:
    return _learning_state_snapshot_unlocked()


def _learning_state_snapshot_unlocked() -> dict[str, Any]:
  return {
    "active": bool(_live_state.get("learning_active")),
    "mode": _live_state.get("learning_mode"),
    "started_at": _iso_or_none(_live_state.get("learning_started_at")),
    "stopped_at": _iso_or_none(_live_state.get("learning_stopped_at")),
    "next_collection_at": _iso_or_none(_live_state.get("learning_next_collection_at")),
    "refresh_interval_seconds": LEARNING_COLLECTION_INTERVAL_SECONDS,
  }


def _append_collection_log_unlocked(
  status: str,
  message: str,
  *,
  cycle: int | None = None,
  mode: str | None = None,
  duration_ms: int | None = None,
  counts: dict[str, Any] | None = None,
) -> None:
  log = list(_live_state.get("collection_log") or [])
  log.append(
      {
          "timestamp": datetime.now(timezone.utc).isoformat(),
          "cycle": cycle if cycle is not None else _live_state.get("collection_cycle"),
          "mode": mode or _live_state.get("learning_mode"),
          "status": status,
          "message": message,
          "duration_ms": duration_ms,
          "counts": counts or {},
      }
  )
  _live_state["collection_log"] = log[-80:]


def _set_live_progress(
  percent: int,
  stage: str,
  message: str,
  active: bool = True,
) -> None:
  now = datetime.now()
  with _live_lock:
    previous = dict(_live_state["progress"])
    started_at = previous.get("started_at") if active else previous.get("started_at")
    if active and not previous.get("active"):
      started_at = now
    _live_state["progress"] = {
      "active": active,
      "percent": max(0, min(100, int(percent))),
      "stage": stage,
      "message": message,
      "started_at": started_at,
      "updated_at": now,
    }


def _iso_or_none(value: datetime | None) -> str | None:
  return value.isoformat() if value is not None else None


def _candidate_chart_symbols(store: RealtimeMarketDataStore, *, limit: int = 12) -> tuple[str, ...]:
    symbols: list[str] = []

    def add_many(values: Any) -> None:
        for value in tuple(values or ()):
            symbol = str(value or "").upper().strip()
            if symbol and symbol not in symbols:
                symbols.append(symbol)

    add_many(item.get("symbol") for item in _latest_technical_by_symbol().values())
    try:
        macro = _latest_macro_micro_bundle()
        macro_result = (macro or {}).get("macro_result") or {}
        add_many(macro_result.get("candidate_symbols"))
        add_many((item or {}).get("symbol") for item in ((macro or {}).get("micro_results") or ()))
    except Exception:  # noqa: BLE001
        pass
    try:
        basis = _last_live_account_basis()
        add_many((p or {}).get("ticker") or (p or {}).get("symbol") for p in ((basis or {}).get("positions") or ()))
    except Exception:  # noqa: BLE001
        pass
    try:
        add_many(store.active_symbols(datetime.now(timezone.utc) - timedelta(minutes=30), limit=limit * 3))
    except Exception:  # noqa: BLE001
        pass
    return tuple(symbols[: max(1, int(limit))])


def _latest_technical_by_symbol() -> dict[str, dict[str, Any]]:
    try:
        from app.technical.decision_feed import snapshot

        rows = snapshot()
    except Exception:  # noqa: BLE001
        rows = ()
    result: dict[str, dict[str, Any]] = {}
    for row in rows or ():
        symbol = str((row or {}).get("symbol") or "").upper().strip()
        if symbol:
            result[symbol] = dict(row or {})
    return result


def _latest_macro_micro_bundle() -> dict[str, Any] | None:
    try:
        from app.graph import macro_micro_feed

        bundle = macro_micro_feed.snapshot()
    except Exception:  # noqa: BLE001
        return None
    return dict(bundle or {}) if bundle else None


def _latest_micro_by_symbol() -> dict[str, dict[str, Any]]:
    bundle = _latest_macro_micro_bundle() or {}
    result: dict[str, dict[str, Any]] = {}
    for row in bundle.get("micro_results") or ():
        symbol = str((row or {}).get("symbol") or "").upper().strip()
        if symbol:
            result[symbol] = dict(row or {})
    return result


def _realtime_bar_payload(bar: Any) -> dict[str, Any]:
    return {
        "symbol": getattr(bar, "symbol", None),
        "minute_start": _iso_or_none(getattr(bar, "minute_start", None)),
        "open": getattr(bar, "open", None),
        "high": getattr(bar, "high", None),
        "low": getattr(bar, "low", None),
        "close": getattr(bar, "close", None),
        "volume": getattr(bar, "volume", None),
        "vwap": getattr(bar, "vwap", None),
        "trade_count": getattr(bar, "trade_count", None),
        "spread_bps": getattr(bar, "spread_bps", None),
        "orderbook_imbalance": getattr(bar, "orderbook_imbalance", None),
        "liquidity_score": getattr(bar, "liquidity_score", None),
        "volatility": getattr(bar, "volatility", None),
        "last_update_age_ms": getattr(bar, "last_update_age_ms", None),
    }


def _realtime_tick_payload(tick: Any | None) -> dict[str, Any] | None:
    if tick is None:
        return None
    return {
        "symbol": getattr(tick, "symbol", None),
        "exchange_timestamp": _iso_or_none(getattr(tick, "exchange_timestamp", None)),
        "received_at": _iso_or_none(getattr(tick, "received_at", None)),
        "source": getattr(tick, "source", None),
        "price": getattr(tick, "price", None),
        "volume": getattr(tick, "volume", None),
        "trade_direction": getattr(tick, "trade_direction", None),
        "latency_ms": getattr(tick, "latency_ms", None),
    }


def _realtime_orderbook_payload(orderbook: Any | None) -> dict[str, Any] | None:
    if orderbook is None:
        return None
    return {
        "symbol": getattr(orderbook, "symbol", None),
        "exchange_timestamp": _iso_or_none(getattr(orderbook, "exchange_timestamp", None)),
        "received_at": _iso_or_none(getattr(orderbook, "received_at", None)),
        "source": getattr(orderbook, "source", None),
        "best_bid": getattr(orderbook, "best_bid", None),
        "best_ask": getattr(orderbook, "best_ask", None),
        "spread_bps": getattr(orderbook, "spread_bps", None),
        "total_bid_volume": getattr(orderbook, "total_bid_volume", None),
        "total_ask_volume": getattr(orderbook, "total_ask_volume", None),
        "imbalance": getattr(orderbook, "imbalance", None),
    }


def _compact_technical_payload(row: dict[str, Any]) -> dict[str, Any]:
    tech = dict(row.get("technical") or {})
    profit = dict(row.get("profitability") or {})
    confidence = tech.get("confidence")
    if confidence is None:
        confidence = tech.get("score")
    return {
        "symbol": row.get("symbol"),
        "action": row.get("action"),
        "approved": row.get("approved"),
        "reason_codes": list(row.get("reason_codes") or ())[:5],
        "regime": row.get("technical_regime") or tech.get("regime"),
        "methodology": row.get("technical_methodology") or tech.get("methodology"),
        "confidence": confidence,
        "expected_edge_bps": tech.get("expected_edge_bps") or profit.get("expected_net_bps"),
        "profitability": profit.get("decision") or profit.get("status"),
    }


def _graph_payload(context: Any, *, trim_for_ui: bool = True) -> dict[str, Any]:
    triples = context.graph.triples()
    event_meta = _event_metadata_map(context.events, context.reasoning_paths)
    links: list[dict[str, Any]] = []
    seen_links: set[tuple[str, str, str]] = set()

    for triple in triples:
        if not _include_graph_triple(triple, event_meta):
            continue
        key = (str(triple.subject), str(triple.predicate), str(triple.object))
        if key in seen_links:
            continue
        seen_links.add(key)
        links.append(
            {
                "source": triple.subject,
                "target": triple.object,
                "predicate": triple.predicate,
                "evidence_id": triple.evidence_id,
            }
        )

    market_tickers = {str(market.ticker) for market in getattr(context, "markets", ())}
    for event_node, meta in event_meta.items():
        predicate = "hasRecentDisclosure" if event_node.startswith("DISCLOSURE:") else "hasRecentNews"
        for ticker in meta.get("tickers", ()):
            if ticker not in market_tickers:
                continue
            key = (ticker, predicate, event_node)
            if key in seen_links:
                continue
            seen_links.add(key)
            links.append(
                {
                    "source": ticker,
                    "target": event_node,
                    "predicate": predicate,
                    "evidence_id": "event:time-sensitive",
                }
            )

    importance = _node_importance_map(links)
    kind_overrides = _semantic_node_kind_overrides(links)
    nodes: dict[str, dict[str, Any]] = {}
    for link in links:
        for node_id in (link["source"], link["target"]):
            if node_id in nodes:
                continue
            score = round(importance.get(node_id, 0.0), 4)
            nodes[node_id] = _node_payload(node_id, score, event_meta.get(node_id), kind_overrides.get(str(node_id)))

    reasoning_steps = _build_reasoning_steps(context.reasoning_paths)
    if trim_for_ui:
        display_nodes, display_links, display_steps = _trim_graph_payload_for_ui(
            list(nodes.values()),
            links,
            reasoning_steps,
        )
    else:
        display_nodes = list(nodes.values())
        display_links = links
        display_steps = reasoning_steps
    # Overlay the hierarchical macro–micro reasoning (advisory) onto the graph so
    # the visualization reflects market regime -> candidates -> per-symbol micro.
    # Added AFTER trimming so it always renders; additive (existing graph intact).
    display_nodes, display_links, macro_micro_summary = _apply_macro_micro_overlay(display_nodes, display_links)
    payload_node_count = len(display_nodes) if not trim_for_ui else len(nodes)
    payload_link_count = len(display_links) if not trim_for_ui else len(links)
    return {
        "nodes": display_nodes,
        "links": display_links,
        "reasoning_steps": display_steps,
        "macro_micro": macro_micro_summary,
        "counts": {"nodes": payload_node_count, "links": payload_link_count},
        "display_counts": {"nodes": len(display_nodes), "links": len(display_links)},
        "truncated": trim_for_ui and (len(display_nodes) < len(nodes) or len(display_links) < len(links)),
        "runtime": context.ontology_runtime.as_dict(),
        "candidate_selection": _candidate_selection_payload(getattr(context, "candidate_selection", None)),
        "parameter_tuning": tuple(getattr(context, "parameter_tuning", ()) or ()),
        "temporal_frame_count": len(tuple(getattr(context, "temporal_frames", ()) or ())),
        # Additive standards-based RDF/OWL/SHACL view. Separates asserted from
        # inferred triples and reports SHACL validation. Never replaces the
        # fields above; the GUI renders it in its own diagnostic panel.
        "semantic_layer": _semantic_layer_payload(context),
    }


def _semantic_layer_diagnostics(context: Any) -> dict[str, Any] | None:
    """Compact (graph-free) semantic-layer summary for the diagnostics panel."""
    layer = getattr(context, "ontology_layer", None)
    if layer is None:
        return None
    try:
        data = layer.as_dict()
    except Exception:  # pragma: no cover - defensive
        return None
    validation = data.get("validation", {}) or {}
    return {
        "ok": data.get("ok", True),
        "errors": data.get("errors", []),
        "reasoning_profile": data.get("reasoning_profile"),
        "counts": data.get("counts", {}),
        "timings_ms": data.get("timings_ms", {}),
        "validation": {
            "mode": validation.get("mode"),
            "conforms": validation.get("conforms"),
            "blocking": validation.get("blocking"),
            "violation_count": len(validation.get("violations", []) or []),
            "violations": (validation.get("violations", []) or [])[:25],
            "error": validation.get("error"),
        },
    }


def _semantic_layer_payload(context: Any, *, node_limit: int = 400, link_limit: int = 500) -> dict[str, Any] | None:
    """Compact GUI payload for the RDF/OWL/SHACL ontology layer.

    Returns None when the layer is disabled/unavailable. Asserted and inferred
    triples are separated, SHACL results are kept apart from OWL results, and
    the semantic graph is trimmed so the GUI never receives oversized payloads.
    """
    layer = getattr(context, "ontology_layer", None)
    if layer is None:
        return None
    try:
        data = layer.as_dict()
    except Exception:  # pragma: no cover - defensive
        return None
    graph = data.get("graph") or {}
    nodes = list(graph.get("nodes", []))[:node_limit]
    links = list(graph.get("links", []))[:link_limit]
    inferred_types = data.get("inferred_types", {}) or {}
    return {
        "ok": data.get("ok", True),
        "errors": data.get("errors", []),
        "reasoning_profile": data.get("reasoning_profile"),
        "counts": data.get("counts", {}),
        "timings_ms": data.get("timings_ms", {}),
        # OWL logical results (kept separate from Python policy scores).
        "owl": {
            "inferred_types": dict(list(inferred_types.items())[:node_limit]),
            "inferred_type_count": len(inferred_types),
        },
        # SHACL validation (rendered in the diagnostic/log section, not the
        # primary portfolio view).
        "validation": data.get("validation", {}),
        # Frames for grouped visualization: asset facts / evidence / signals /
        # risk / inferred classes / validation.
        "graph": {
            "nodes": nodes,
            "links": links,
            "truncated": len(graph.get("nodes", [])) > node_limit or len(graph.get("links", [])) > link_limit,
            "node_count": len(graph.get("nodes", [])),
            "link_count": len(graph.get("links", [])),
        },
    }


def _empty_graph_payload(snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    snapshot = snapshot or {}
    return {
        "nodes": [],
        "links": [],
        "reasoning_steps": [],
        "counts": {"nodes": 0, "links": 0},
        "display_counts": {"nodes": 0, "links": 0},
        "truncated": False,
        "runtime": get_ontology_runtime().as_dict(),
        "candidate_selection": None,
        "parameter_tuning": (),
        "temporal_frame_count": 0,
        "semantic_layer": None,
        "warming": True,
        "is_refreshing": bool(snapshot.get("is_refreshing")),
        "last_error": snapshot.get("last_error"),
    }


def _trim_graph_payload_for_ui(
    nodes: list[dict[str, Any]],
    links: list[dict[str, Any]],
    reasoning_steps: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if len(nodes) <= ONTOLOGY_UI_NODE_LIMIT and len(links) <= ONTOLOGY_UI_LINK_LIMIT:
        return nodes, links, reasoning_steps[:ONTOLOGY_UI_REASONING_STEP_LIMIT]

    node_by_id = {str(node.get("id")): node for node in nodes}
    degree: dict[str, int] = {node_id: 0 for node_id in node_by_id}
    for link in links:
        source = str(link.get("source"))
        target = str(link.get("target"))
        degree[source] = degree.get(source, 0) + 1
        degree[target] = degree.get(target, 0) + 1

    preferred_kinds = {
        "ticker": 7.0,
        "support": 6.5,
        "risk": 6.5,
        "contradiction": 6.5,
        "pipeline": 5.5,
        "tuning": 5.0,
        "parameter": 4.5,
        "metric": 4.0,
        "sector": 3.5,
        "event": 3.0,
        "temporal": 2.0,
        "entity": 1.0,
    }

    def node_score(node: dict[str, Any]) -> float:
        node_id = str(node.get("id"))
        return (
            float(node.get("importance_score") or 0.0)
            + degree.get(node_id, 0) * 0.18
            + preferred_kinds.get(str(node.get("kind") or ""), 0.0)
            + (8.0 if node.get("highlight") else 0.0)
        )

    selected_ids = {
        str(node.get("id"))
        for node in sorted(nodes, key=node_score, reverse=True)[:ONTOLOGY_UI_NODE_LIMIT]
    }

    def link_score(link: dict[str, Any]) -> float:
        predicate = str(link.get("predicate") or "")
        source = str(link.get("source"))
        target = str(link.get("target"))
        predicate_boost = 10.0 if predicate in {
            "supportsSignal",
            "increasesRiskOf",
            "contradictsSignal",
            "hasRecentNews",
            "hasRecentDisclosure",
            "selectsCandidate",
            "feedsStage",
            "tunesParameter",
        } else 0.0
        return predicate_boost + node_score(node_by_id.get(source, {})) + node_score(node_by_id.get(target, {}))

    display_links = [
        link
        for link in sorted(links, key=link_score, reverse=True)
        if str(link.get("source")) in selected_ids and str(link.get("target")) in selected_ids
    ][:ONTOLOGY_UI_LINK_LIMIT]

    display_nodes = [node for node in sorted(nodes, key=node_score, reverse=True) if str(node.get("id")) in selected_ids]
    display_ids = {str(node.get("id")) for node in display_nodes}
    display_steps = [
        step for step in reasoning_steps
        if any(str(node_id) in display_ids for node_id in step.get("nodes", ()))
    ][:ONTOLOGY_UI_REASONING_STEP_LIMIT]
    return display_nodes, display_links, display_steps


def _resolve_instrument_label(node_id: str) -> str:
    """Map a bare ticker code to a configured display name when known."""
    from app.graph.rdf_adapter import _instrument_name_map

    text = str(node_id)
    base = text.split(".", 1)[0]
    names = _instrument_name_map()
    return names.get(base) or names.get(text) or text


def _apply_macro_micro_overlay(display_nodes: list, display_links: list) -> tuple[list, list, dict | None]:
    """Overlay the latest macro/micro reasoning bundle onto the graph payload.

    Adds market-regime -> candidate -> micro-regime nodes/links (advisory) using
    existing node kinds, and returns a compact summary for a dedicated panel.
    Best-effort: no bundle -> graph unchanged, summary None.
    """
    try:
        from app.graph import macro_micro_feed
        from app.account_dashboard import build_macro_micro_panel

        bundle = macro_micro_feed.snapshot()
    except Exception:  # noqa: BLE001 - overlay is advisory; never break the graph.
        return display_nodes, display_links, None
    if not bundle:
        return display_nodes, display_links, None

    macro = bundle.get("macro_result") or {}
    node_index = {n["id"]: n for n in display_nodes}
    seen_links = {(str(l.get("source")), str(l.get("predicate")), str(l.get("target"))) for l in display_links}

    def _add_node(nid, label, kind, size=13.0):
        if nid not in node_index:
            node_index[nid] = {"id": nid, "label": label, "kind": kind, "importance_score": 0.5, "size": size}

    def _add_link(src, predicate, tgt, evidence):
        key = (str(src), str(predicate), str(tgt))
        if key not in seen_links:
            seen_links.add(key)
            display_links.append({"source": src, "target": tgt, "predicate": predicate, "evidence_id": evidence})

    regime = macro.get("market_regime") or "NO_TRADE_MARKET"
    market_id, regime_id = "MacroMarket", f"MarketRegime:{regime}"
    _add_node(market_id, "시장(거시)", "pipeline", 18.0)
    _add_node(regime_id, regime, "support", 16.0)
    _add_link(market_id, "hasMarketRegime", regime_id, "macro")
    for sym in (macro.get("candidate_symbols") or [])[:12]:
        _add_node(str(sym), str(sym), "ticker", 12.0)
        _add_link(market_id, "selectsCandidateSymbol", str(sym), "macro")
    for strat in (macro.get("blocked_micro_strategies") or [])[:6]:
        bid = f"BlockedStrategy:{strat}"
        _add_node(bid, str(strat), "risk", 10.0)
        _add_link(regime_id, "blocksMicroStrategy", bid, "macro")
    for m in (bundle.get("micro_results") or [])[:12]:
        sym = str(m.get("symbol") or "")
        mr = m.get("micro_regime") or "NO_TRADE_SYMBOL"
        if not sym:
            continue
        mrid = f"MicroRegime:{mr}"
        _add_node(sym, sym, "ticker", 12.0)
        _add_node(mrid, mr, "support", 11.0)
        _add_link(sym, "hasMicroRegime", mrid, "micro")

    summary = None
    try:
        summary = build_macro_micro_panel(bundle)
    except Exception:  # noqa: BLE001
        summary = None
    return list(node_index.values()), display_links, summary


def _node_payload(
    node_id: str,
    importance_score: float,
    event_meta: dict[str, Any] | None = None,
    kind_override: str | None = None,
) -> dict[str, Any]:
    kind = kind_override or _node_kind(node_id)
    payload = {
        "id": node_id,
        "label": event_meta.get("title", node_id) if event_meta else _resolve_instrument_label(node_id),
        "kind": kind,
        "importance_score": round(importance_score + float(event_meta.get("boost", 0.0) if event_meta else 0.0), 4),
        "size": _node_size(kind, importance_score + float(event_meta.get("boost", 0.0) if event_meta else 0.0)),
    }
    if event_meta:
        payload.update(event_meta)
    return payload


def _semantic_node_kind_overrides(links: list[dict[str, Any]]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    relation_kind = {
        "supportsSignal": "support",
        "decreasesRiskOf": "support",
        "increasesRiskOf": "risk",
        "contradictsSignal": "contradiction",
    }
    for link in links:
        kind = relation_kind.get(str(link.get("predicate", "")))
        if not kind:
            continue
        for field in ("source", "target"):
            node_id = str(link.get(field, ""))
            if not node_id or _node_kind(node_id) in {"ticker", "event", "temporal", "pipeline", "tuning", "parameter", "metric", "sector"}:
                continue
            if overrides.get(node_id) == "risk":
                continue
            if kind == "risk" or node_id not in overrides:
                overrides[node_id] = kind
    return overrides


def _node_kind(node_id: str) -> str:
    if re.match(r"^(NEWS|DISCLOSURE|MACRO|MARKET|FINANCIAL):", node_id):
        return "event"
    if node_id.startswith((
        "TimeBucket:",
        "TemporalFrame:",
        "ImpactScore:",
        "MarketSnapshot:",
        "RealtimeQuote:",
        "RealtimeExecution:",
        "RawSource:",
        "MacroMetric:",
    )):
        return "temporal"
    if node_id == "OntologyMultiStagePipeline" or node_id.startswith("OntologyFilter") or node_id in {
        "CandidateStock",
        "SelectiveChartFetching",
        "SemanticFeatureExtraction",
        "AIPredictionSmallSet",
        "NoTradeSignal",
        "FinalTradeGate",
    }:
        return "pipeline"
    if node_id.startswith("OntologyTuningMode:") or node_id == "MarketInterpretationParameterTuning":
        return "tuning"
    if node_id.startswith("Parameter:") or node_id.startswith("TunedValue:"):
        return "parameter"
    if node_id.startswith("UniverseCount:") or node_id.startswith("CandidateCount:") or node_id in {
        "TradingCost",
        "BrokerageFee",
        "SellTax",
        "Slippage",
        "BidAskSpread",
        "MarketImpact",
        "BreakEvenReturn",
        "RequiredExitPrice",
        "NetExpectedReturn",
        "CostToAlphaRatio",
    }:
        return "metric"
    if node_id in {"Semiconductor", "Battery", "Finance"}:
        return "sector"
    if node_id in {
        "EarningsGrowth",
        "ProfitabilityQuality",
        "PositiveEventImpact",
        "PositiveInvestorFlow",
        "InformedOrderFlowImbalance",
        "ForeignInstitutionJointBuying",
        "RetailSupplyAbsorbedByInformedFlow",
        "OrderFlowPriceConfirmation",
        "SuspectedSmartMoneyAccumulation",
        "OrderFlowConfirmedBuyCandidate",
        "SectorMomentum",
        "BuyCandidate",
        "HoldWithTrailingStop",
        "BreakoutWatch",
        "Watchlist",
        "RiskAdjustedSizing",
        "NetProfitability",
    }:
        return "support"
    if node_id in {
        "MacroRateRisk",
        "NegativeEventRisk",
        "VolatilityRisk",
        "LiquidityRisk",
        "OrderFlowDistributionRisk",
        "ThinLiquidityPriceImpactRisk",
        "CostBurden",
        "SlippageRisk",
        "SpreadRisk",
        "SellCandidate",
        "ReduceRiskCandidate",
        "WaitOrTakeProfit",
    }:
        return "risk"
    if node_id in {
        "ValuationDiscipline",
        "AggressiveBuy",
        "ValuationSlightlyHigh",
        "InformedOrderFlowDistribution",
        "ForeignInstitutionJointSelling",
        "RetailDemandMeetsInformedSelling",
        "OrderFlowPriceDivergence",
        "SuspectedSmartMoneyDistribution",
    }:
        return "contradiction"
    if _looks_like_ticker(node_id):
        return "ticker"
    return "entity"


def _event_metadata_map(
    events: tuple[Any, ...],
    reasoning_paths: tuple[Any, ...],
) -> dict[str, dict[str, Any]]:
    now = datetime.now(timezone.utc)
    used_event_nodes = _reasoning_event_nodes(reasoning_paths)
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for event in events:
        node_id = f"{event.event_type}:{event.event_id}"
        age_hours = max(0.0, (now - _aware_datetime(event.event_date)).total_seconds() / 3600)
        confidence = max(0.0, min(1.0, float(getattr(event, "classification_confidence", 0.0) or 0.0)))
        sentiment = str(getattr(event, "sentiment", "NEUTRAL"))
        is_directional = sentiment in {"POSITIVE", "NEGATIVE"}
        is_used = node_id in used_event_nodes
        recency = math.exp(-age_hours / 72.0)
        score = recency * 2.2 + confidence * 1.2
        if is_directional:
            score += 1.4
        if getattr(event, "event_labels", ()):
            score += 0.45
        if getattr(event, "key_facts", ()):
            score += 0.25
        if is_used:
            score += 3.0

        keep = is_used or age_hours <= 168 or (is_directional and age_hours <= 720 and score >= 2.2)
        if not keep:
            continue
        scored.append(
            (
                score,
                node_id,
                {
                    "time_sensitive": True,
                    "highlight": is_used or score >= 2.2,
                    "used_in_reasoning": is_used,
                    "event_age_hours": round(age_hours, 1),
                    "event_date": _aware_datetime(event.event_date).isoformat(),
                    "sentiment": sentiment,
                    "title": str(getattr(event, "title", node_id))[:120],
                    "summary": str(getattr(event, "summary", ""))[:240],
                    "tickers": tuple(str(ticker) for ticker in getattr(event, "tickers", ()) if str(ticker)),
                    "boost": min(5.0, score),
                },
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    return {node_id: meta for _score, node_id, meta in scored[:140]}


def _reasoning_event_nodes(reasoning_paths: tuple[Any, ...]) -> set[str]:
    nodes: set[str] = set()
    for path in reasoning_paths:
        for attr in ("supporting_triples", "contradicting_triples", "risk_triples"):
            for triple_text in getattr(path, attr, ()):
                triple = _parse_reasoning_triple(triple_text)
                if triple is None:
                    continue
                subject, _predicate, obj = triple
                if _node_kind(subject) == "event":
                    nodes.add(subject)
                if _node_kind(obj) == "event":
                    nodes.add(obj)
    return nodes


def _include_graph_triple(triple: Any, event_meta: dict[str, dict[str, Any]]) -> bool:
    subject_is_event = _node_kind(str(triple.subject)) == "event"
    target_is_event = _node_kind(str(triple.object)) == "event"
    if not subject_is_event and not target_is_event:
        return True
    if subject_is_event and str(triple.subject) not in event_meta:
        return False
    if target_is_event and str(triple.object) not in event_meta:
        return False
    if str(triple.predicate) == "generatesSemanticFeature":
        return bool(subject_is_event and event_meta.get(str(triple.subject), {}).get("used_in_reasoning"))
    return True


def _aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _looks_like_ticker(node_id: str) -> bool:
    if re.fullmatch(r"\d{6}(?:\.[A-Z]{1,4})?", node_id):
        return True
    return re.fullmatch(r"[A-Z][A-Z0-9]{0,6}(?:[.-][A-Z0-9]{1,4})?", node_id) is not None


def _node_importance_map(links: list[dict[str, Any]]) -> dict[str, float]:
    if not links:
        return {}

    scores: dict[str, float] = {}
    degrees: dict[str, int] = {}
    for link in links:
        weight = _triple_weight(str(link.get("predicate", "")))
        for node_id in (str(link["source"]), str(link["target"])):
            scores[node_id] = scores.get(node_id, 0.0) + weight
            degrees[node_id] = degrees.get(node_id, 0) + 1

    for node_id, degree in degrees.items():
        kind = _node_kind(node_id)
        kind_bias = {
            "ticker": 1.35,
            "event": 1.35,
            "pipeline": 1.20,
            "tuning": 1.18,
            "parameter": 1.05,
            "temporal": 1.22,
            "metric": 0.90,
            "sector": 1.10,
            "risk": 1.05,
            "support": 1.00,
            "contradiction": 1.00,
            "entity": 0.95,
        }.get(kind, 1.0)
        scores[node_id] = (scores.get(node_id, 0.0) + math.log1p(degree)) * kind_bias

    return scores


def _triple_weight(predicate: str) -> float:
    return {
        "supportsSignal": 1.15,
        "increasesRiskOf": 1.10,
        "contradictsSignal": 1.05,
        "hasRecentNews": 0.95,
        "hasRecentDisclosure": 0.90,
        "selectsCandidate": 1.15,
        "feedsStage": 1.05,
        "usesCostModel": 1.12,
        "contains": 1.04,
        "produces": 1.10,
        "blocksTradeBelow": 1.16,
        "tunesParameter": 1.20,
        "hasTunedValue": 1.00,
        "containsFrame": 1.16,
        "hasTimeFrame": 1.18,
        "observesTicker": 1.18,
        "containsEvent": 1.22,
        "occursInTimeBucket": 1.12,
        "usesMarketSnapshot": 1.05,
        "containsQuote": 1.08,
        "containsExecution": 1.08,
        "usesRawSource": 1.02,
        "hasMacroContext": 1.00,
        "hasImpactScore": 1.15,
        "hasTuningMode": 1.10,
        "adjustsStage": 1.15,
        "producesTunedValue": 1.16,
        "appliesToStage": 1.12,
        "usesOntologySignal": 1.18,
        "calibratesSignal": 1.18,
        "raisesTuningPressure": 1.25,
        "requiresApprovalFrom": 1.05,
        "observedUniverseCount": 0.78,
        "selectedCandidateCount": 0.84,
        "fetchesChartsFor": 0.84,
        "belongsToSector": 0.85,
        "hasTicker": 0.75,
        "isListedOn": 0.70,
        "hasExposureTo": 0.70,
    }.get(predicate, 0.65)


def _node_size(kind: str, score: float) -> float:
    base = {
        "ticker": 10.0,
        "event": 7.0,
        "temporal": 6.0,
        "pipeline": 7.5,
        "tuning": 7.0,
        "parameter": 6.0,
        "metric": 5.2,
        "sector": 8.0,
        "support": 8.0,
        "risk": 8.0,
        "contradiction": 8.0,
        "entity": 7.0,
    }.get(kind, 7.0)
    scaled = base + min(16.0, math.log1p(max(score, 0.0)) * 4.6)
    return round(scaled, 2)


def _build_reasoning_steps(reasoning_paths: tuple[Any, ...]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for path in reasoning_paths:
        confidence = round(float(path.confidence) * 100, 1)
        groups = (
            ("supporting_triples", "긍정 근거 확인", "support"),
            ("contradicting_triples", "상충 근거 확인", "contradiction"),
            ("risk_triples", "리스크 근거 확인", "risk"),
        )
        for attr, title, tone in groups:
            for triple_text in getattr(path, attr, ()):
                triple = _parse_reasoning_triple(triple_text)
                if triple is None:
                    continue
                subject, predicate, obj = triple
                steps.append(
                    {
                        "path_id": path.path_id,
                        "ticker": path.ticker,
                        "title": title,
                        "description": f"{path.ticker}: {subject} --{predicate}--> {obj}",
                        "nodes": [subject, obj],
                        "links": [{"source": subject, "target": obj, "predicate": predicate}],
                        "tone": tone,
                        "confidence_percent": confidence,
                    }
                )
        steps.append(
            {
                "path_id": path.path_id,
                "ticker": path.ticker,
                "title": "결론 도출",
                "description": f"{path.ticker}: {path.conclusion} · 신뢰도 {confidence:.1f}%",
                "nodes": [path.ticker, path.conclusion],
                "links": [{"source": path.ticker, "target": path.conclusion, "predicate": "supportsSignal"}],
                "tone": "conclusion" if path.conclusion == "BuyCandidate" else "watch",
                "confidence_percent": confidence,
            }
        )
    return steps


def _parse_reasoning_triple(value: str) -> tuple[str, str, str] | None:
    marker = " --"
    arrow = "--> "
    if marker not in value or arrow not in value:
        return None
    subject, rest = value.split(marker, 1)
    predicate, obj = rest.split(arrow, 1)
    return subject.strip(), predicate.strip(), obj.strip()


def _json(value: Any) -> JSONResponse:
    return JSONResponse(_to_jsonable(value))


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


HTML = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" type="image/png" href="/static/icon.png">
  <link rel="apple-touch-icon" href="/static/icon.png">
  <title>개인 투자 분석 시스템</title>
  <style>
    :root {
      --bg: #f6f7f9; --panel: #ffffff; --ink: #1d2430; --muted: #667085;
      --line: #d9dee7; --accent: #0f766e; --accent-strong: #0b5f59;
      --warn: #b45309; --chip: #eef6f4;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); font-family: Arial, Helvetica, sans-serif; letter-spacing: 0; }
    .shell { display: grid; grid-template-columns: 330px minmax(0, 1fr); min-height: 100vh; }
    aside { border-right: 1px solid var(--line); background: #eef2f5; padding: 20px; }
    main { padding: 22px; }
    h1 { font-size: 22px; margin: 0 0 18px; }
    h2 { font-size: 16px; margin: 0 0 12px; }
    label { display: block; font-size: 13px; color: var(--muted); margin-bottom: 6px; }
    input { width: 100%; height: 40px; border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; font-size: 14px; background: white; }
    button { height: 40px; border: 0; border-radius: 6px; padding: 0 14px; background: var(--accent); color: white; font-weight: 700; cursor: pointer; }
    button:hover { background: var(--accent-strong); }
    button:disabled { opacity: .45; cursor: not-allowed; }
    button.secondary { background: white; color: var(--ink); border: 1px solid var(--line); }
    .segmented { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 14px; }
    .segmented label { margin: 0; }
    .segmented input { position: absolute; opacity: 0; pointer-events: none; }
    .segmented span { display: block; border: 1px solid var(--line); border-radius: 6px; padding: 10px; background: white; color: var(--ink); font-weight: 700; text-align: center; cursor: pointer; }
    .segmented input:checked + span { background: var(--accent); border-color: var(--accent); color: white; }
    .field { margin-bottom: 14px; }
    .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 14px; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }
    .span-4 { grid-column: span 4; } .span-8 { grid-column: span 8; } .span-12 { grid-column: span 12; }
    .metric { font-size: 26px; font-weight: 800; }
    .muted { color: var(--muted); font-size: 13px; }
    .bar { width: 100%; height: 12px; background: #e6e9ef; border-radius: 999px; overflow: hidden; margin-top: 10px; }
    .bar > span { display: block; height: 100%; background: var(--accent); width: 0%; }
    .bar.good > span { background: #067647; }
    .bar.warn > span { background: #b45309; }
    .bar.bad > span { background: #b42318; }
    .score-row { display: grid; grid-template-columns: 140px minmax(160px, 1fr) 58px; gap: 10px; align-items: center; margin: 10px 0; }
    .score-label { color: var(--muted); font-size: 13px; }
    .score-value { font-weight: 800; text-align: right; }
    .stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
    .stat { border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fbfcfe; }
    .stat strong { display: block; font-size: 22px; margin-bottom: 4px; }
    .ticker-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .ticker-card { border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: white; }
    .sentiment { display: flex; height: 12px; border-radius: 999px; overflow: hidden; background: #e6e9ef; margin: 10px 0; }
    .sentiment span { display: block; height: 100%; }
    .sentiment .pos { background: #067647; }
    .sentiment .neu { background: #98a2b3; }
    .sentiment .neg { background: #b42318; }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    .choice { border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: white; cursor: pointer; }
    .choice.selected { border-color: var(--accent); outline: 2px solid #99d5cc; }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; }
    .chip { background: var(--chip); color: var(--accent-strong); border-radius: 999px; padding: 6px 9px; font-size: 12px; }
    .asset-table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }
    .asset-table th, .asset-table td { padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: right; vertical-align: top; }
    .asset-table th { color: var(--muted); font-weight: 700; text-align: left; width: 48%; }
    .asset-table tr:last-child th, .asset-table tr:last-child td { border-bottom: 0; }
    .holding-lists { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }
    .holding-list { border: 1px solid var(--line); border-radius: 8px; padding: 10px; min-height: 116px; }
    .holding-list strong { display: block; font-size: 12px; margin-bottom: 8px; color: var(--muted); }
    .holding-list ul { list-style: none; padding: 0; margin: 0; display: grid; gap: 7px; }
    .holding-list li { display: flex; justify-content: space-between; gap: 8px; font-size: 12px; }
    .holding-list .empty { color: var(--muted); }
    .log { white-space: pre-wrap; background: #111827; color: #e5e7eb; border-radius: 8px; padding: 14px; min-height: 160px; overflow: auto; font-size: 12px; }
    .table-wrap { width: 100%; overflow: auto; border: 1px solid var(--line); border-radius: 8px; }
    table.live-table { width: 100%; border-collapse: collapse; font-size: 12px; background: white; }
    .live-table th, .live-table td { padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
    .live-table th:first-child, .live-table td:first-child { text-align: left; }
    .live-table tr:last-child td { border-bottom: 0; }
    .tone-pos { color: #067647; font-weight: 700; }
    .tone-neg { color: #b42318; font-weight: 700; }
    .side-buy { color: #067647; font-weight: 800; }
    .side-sell { color: #b42318; font-weight: 800; }
    .status { padding: 10px 12px; border-radius: 8px; background: #fff7ed; color: var(--warn); border: 1px solid #fed7aa; margin-bottom: 14px; }
    .work-status { margin-top: 14px; padding: 12px; border: 1px solid var(--line); border-radius: 8px; background: #f8fafc; display: none; }
    .work-status.active { display: block; }
    .work-status strong { display: block; margin-bottom: 6px; }
    .work-status .bar { margin-top: 8px; height: 10px; }
    .collection-log-chart { width: 100%; height: 64px; margin-top: 10px; display: block; border: 1px solid var(--line); border-radius: 6px; background: #fff; }
    .collection-log-list { margin-top: 8px; display: grid; gap: 6px; max-height: 170px; overflow: auto; }
    .collection-log-item { display: grid; grid-template-columns: 58px 1fr auto; gap: 8px; align-items: center; padding: 7px 8px; border: 1px solid var(--line); border-radius: 6px; background: #fff; font-size: 12px; }
    .collection-log-item strong { display: inline; margin: 0; font-size: 12px; }
    .collection-log-status { width: 9px; height: 9px; border-radius: 50%; display: inline-block; margin-right: 5px; background: #94a3b8; }
    .collection-log-status.running { background: #0f766e; }
    .collection-log-status.complete { background: #16a34a; }
    .collection-log-status.error { background: #dc2626; }
    .collection-log-status.scheduled, .collection-log-status.stopped { background: #64748b; }
    .data-volume-wrap { margin-top: 12px; display: grid; grid-template-columns: minmax(260px, 1fr) minmax(220px, 320px); gap: 12px; align-items: stretch; }
    .data-volume-chart { width: 100%; height: 180px; display: block; border: 1px solid var(--line); border-radius: 6px; background: #fff; }
    .source-volume-list { border: 1px solid var(--line); border-radius: 6px; background: #fff; padding: 10px; max-height: 180px; overflow: auto; display: grid; gap: 6px; }
    .source-volume-row { display: grid; grid-template-columns: 1fr auto; gap: 8px; font-size: 12px; align-items: center; }
    .warning-list { margin-top: 10px; display: grid; gap: 6px; }
    .warning-item { padding: 8px 10px; border-radius: 6px; border: 1px solid #f59e0b; background: #fffbeb; color: #92400e; font-size: 12px; }
    .mode-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin: 10px 0 14px; }
    .mode-grid button { height: auto; min-height: 44px; padding: 8px 10px; text-align: left; }
    .mode-grid small { display: block; margin-top: 3px; font-weight: 400; opacity: .9; }
    .mode-step-label { margin: 10px 0 6px; font-size: 12px; color: var(--muted); font-weight: 700; letter-spacing: .02em; }
    .mode-grid button.active { border-color: var(--accent); box-shadow: 0 0 0 1px rgba(15,118,110,.12); }
    .mode-grid button:disabled { opacity: .55; cursor: not-allowed; }
    .flow-panel { display: grid; gap: 7px; margin-top: 14px; }
    .flow-step { display: grid; grid-template-columns: 22px minmax(0, 1fr); gap: 8px; align-items: center; padding: 8px 9px; border: 1px solid var(--line); border-radius: 6px; background: #fff; font-size: 12px; }
    .flow-dot { width: 18px; height: 18px; border-radius: 50%; background: #cbd5e1; border: 3px solid #edf2f7; }
    .flow-step strong { display: block; font-size: 12px; margin: 0 0 2px; }
    .flow-step span { display: block; color: var(--muted); overflow-wrap: anywhere; }
    .flow-step.active { border-color: #8bc7be; background: #f0fdfa; }
    .flow-step.active .flow-dot { background: var(--accent); }
    .flow-step.done .flow-dot { background: #067647; }
    .flow-step.error { border-color: #fecaca; background: #fff1f2; }
    .flow-step.error .flow-dot { background: #b42318; }
    .mini-chart { width: 100%; height: 74px; margin-top: 10px; border: 1px solid var(--line); border-radius: 6px; background: #fff; display: block; }
    .score-row .mini-chart { grid-column: 1 / -1; }
    .ontology-scene { grid-column: 1 / -1; order: -1; min-height: 760px; position: relative; overflow: hidden; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); background: #0f172a; }
    .ontology-wide-layout { width: 100%; margin: 0 0 14px; }
    .ontology-toolbar { position: absolute; z-index: 2; top: 12px; left: 12px; right: 12px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; color: #e5e7eb; }
    .ontology-toolbar button { height: 34px; background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.22); color: #fff; }
    .ontology-toolbar button:hover { background: rgba(255,255,255,.18); }
    .ontology-badge { display: inline-flex; align-items: center; min-height: 34px; padding: 0 10px; border-radius: 6px; background: rgba(15,23,42,.72); border: 1px solid rgba(255,255,255,.18); font-size: 13px; }
    .ontology-filter { display: inline-flex; gap: 4px; padding: 4px; border-radius: 6px; background: rgba(15,23,42,.72); border: 1px solid rgba(255,255,255,.18); }
    .ontology-filter label { margin: 0; color: #e5e7eb; font-size: 12px; display: inline-flex; align-items: center; gap: 4px; padding: 4px 6px; }
    .ontology-filter input { width: auto; height: auto; }
    .reasoning-strip { position: absolute; z-index: 2; left: 12px; right: 12px; bottom: 54px; display: grid; grid-template-columns: minmax(160px, 1fr) minmax(220px, 2fr); gap: 8px; align-items: stretch; pointer-events: none; }
    .reasoning-strip > div { padding: 9px 10px; border-radius: 6px; background: rgba(15,23,42,.78); border: 1px solid rgba(255,255,255,.18); color: #e5e7eb; font-size: 12px; }
    .reasoning-strip strong { display: block; color: #fff; margin-bottom: 3px; }
    .reasoning-progress { height: 5px; margin-top: 7px; border-radius: 999px; background: rgba(255,255,255,.16); overflow: hidden; }
    .reasoning-progress span { display: block; height: 100%; width: 0%; background: #facc15; }
    .ontology-legend { position: absolute; z-index: 2; left: 12px; bottom: 12px; display: flex; flex-wrap: wrap; gap: 8px; max-width: calc(100% - 24px); }
    .legend-item { display: inline-flex; align-items: center; gap: 6px; min-height: 28px; padding: 0 8px; border-radius: 6px; background: rgba(15,23,42,.72); border: 1px solid rgba(255,255,255,.18); color: #e5e7eb; font-size: 12px; }
    .legend-dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block; border: 1px solid rgba(255,255,255,.5); box-shadow: 0 0 0 1px rgba(15,23,42,.35); }
    .ontology-panel { position: absolute; z-index: 2; top: 58px; right: 12px; width: 260px; max-width: calc(100% - 24px); padding: 12px; border-radius: 8px; background: rgba(15,23,42,.86); border: 1px solid rgba(255,255,255,.18); color: #e5e7eb; font-size: 12px; }
    .ontology-panel strong { display: block; font-size: 15px; margin-bottom: 6px; color: #fff; }
    .ontology-panel .muted { color: #cbd5e1; }
    #ontologyCanvas { width: 100%; height: 760px; display: block; }
    #ontologyTooltip { position: absolute; z-index: 3; pointer-events: none; min-width: 160px; max-width: 260px; padding: 8px 10px; border-radius: 6px; background: rgba(15,23,42,.92); color: #fff; border: 1px solid rgba(255,255,255,.18); font-size: 12px; transform: translate(12px, 12px); display: none; }
    @media (min-width: 901px) {
      .shell { grid-template-columns: 300px minmax(0, 1fr); }
      aside { padding: 14px; max-height: 100vh; overflow: auto; }
      main { padding: 14px; }
      h1 { font-size: 20px; margin-bottom: 10px; }
      h2 { font-size: 14px; margin-bottom: 8px; }
      .grid { gap: 10px; align-items: start; }
      .panel { padding: 12px; }
      .metric { font-size: 22px; }
      .stats { grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }
      .stat { padding: 9px; }
      .stat strong { font-size: 18px; }
      .field { margin-bottom: 9px; }
      input, button { height: 34px; font-size: 13px; }
      .mode-grid { grid-template-columns: 1fr; gap: 6px; }
      .mode-grid button { min-height: 38px; padding: 7px 9px; }
      .work-status { margin-top: 8px; padding: 9px; }
      .flow-panel { grid-template-columns: 1fr 1fr; gap: 6px; }
      .flow-step { grid-template-columns: 16px minmax(0, 1fr); padding: 7px; }
      .flow-dot { width: 14px; height: 14px; border-width: 2px; }
      .collection-log-chart,
      .collection-log-list,
      .data-volume-wrap,
      .warning-list,
      #sourceVolumeList,
      #output,
      #relations { display: none !important; }
      #learningStatusCard .bar { height: 8px; }
      #streamingDemoContainer[hidden] { display: none !important; }
      #streamingDemoContainer .score-row { grid-template-columns: 84px minmax(0, 1fr); margin: 6px 0; }
      #streamingDemoContainer .score-row .mini-chart { display: none; }
      .cards { grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }
      .choice { padding: 10px; min-height: 96px; }
      .choice .metric { font-size: 20px; }
      .table-wrap { max-height: 160px; }
      table.live-table { font-size: 11px; }
      .live-table th, .live-table td { padding: 6px 7px; }
      .ontology-scene { min-height: 700px; }
      #ontologyCanvas { height: 700px; }
    }
    @media (max-width: 900px) {
      .shell { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .span-4, .span-8 { grid-column: span 12; }
      .cards { grid-template-columns: 1fr; }
      .holding-lists { grid-template-columns: 1fr; }
      .stats, .ticker-grid { grid-template-columns: 1fr; }
      .data-volume-wrap { grid-template-columns: 1fr; }
      .ontology-scene { min-height: 560px; }
      #ontologyCanvas { height: 560px; }
      .reasoning-strip { grid-template-columns: 1fr; bottom: 84px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <a href="/account" style="display:block; margin:0 0 16px; padding:12px 14px; border-radius:8px; background:#176b87; color:#fff; text-decoration:none; font-weight:700; text-align:center;">계좌·자산 대시보드 열기</a>
      <h1>개인 투자 분석 시스템</h1>
      <div class="status" id="gate">목표가 확정될 때까지 프로그램은 시작되지 않습니다.</div>
      <h2>운영 모드</h2>
      <div class="mode-step-label">실시간 통합 데이터 기준</div>
      <div class="mode-grid" id="modeActionGrid">
        <button type="button" id="modeTestingButton" onclick="window.startSelectedOperationMode && window.startSelectedOperationMode('live')">실전 투자<small>KIS 실전 서버 전용</small></button>
        <button type="button" id="liveFlagsButton" class="secondary" onclick="window.applyLiveFlags && window.applyLiveFlags()">실전 플래그 적용<small>주문 없이 게이트만 전환</small></button>
        <button type="button" id="modeLiveTradingButton" class="danger" onclick="window.startSelectedOperationMode && window.startSelectedOperationMode('live')">실전 투자<small>자동매매 게이트</small></button>
      </div>
      <button type="button" id="terminateTradingButton" class="danger" onclick="window.terminateActiveTrading && window.terminateActiveTrading()" disabled>조기 종료<small>보유분 전량 매도 후 종료</small></button>
      <div class="work-status active">
        <strong id="operationModeStatus">모드 대기</strong>
        <div class="muted" id="runtimeStatus">NPU 상태 확인 중</div>
      </div>
      <div class="work-status active" id="learningStatusCard">
        <strong id="learningStatusTitle">학습 현황</strong>
        <div class="muted" id="learningStatusMessage">실시간 상태를 확인하는 중입니다.</div>
        <div class="bar"><span id="learningStatusProgress" style="width:0%"></span></div>
        <div class="muted" id="learningStatusMeta" style="margin-top:8px;">대기 중</div>
        <canvas class="collection-log-chart" id="learningCollectionChart" width="280" height="64"></canvas>
        <div class="collection-log-list" id="learningCollectionLog"></div>
      </div>
      <div class="work-status active">
        <strong>시스템 흐름</strong>
        <div class="flow-panel" id="systemFlowPanel">
          <div class="flow-step" data-flow-step="mode"><i class="flow-dot"></i><div><strong>모드</strong><span>학습/모의투자 선택 대기</span></div></div>
          <div class="flow-step" data-flow-step="data"><i class="flow-dot"></i><div><strong>데이터</strong><span>실시간 자료 상태 확인 중</span></div></div>
          <div class="flow-step" data-flow-step="analysis"><i class="flow-dot"></i><div><strong>분석</strong><span>목표 가능성 또는 전략 계산 대기</span></div></div>
          <div class="flow-step" data-flow-step="simulation"><i class="flow-dot"></i><div><strong>실전 판단</strong><span>실전 게이트 대기</span></div></div>
        </div>
      </div>
      <form id="goalForm">
        <div class="field"><label for="targetReturn">목표 수익률 (%)</label><input id="targetReturn" name="target_return_rate" type="number" step="0.1" min="0" value="2" placeholder="예: 5"></div>
        <div class="field"><label for="targetMinutes">목표 시간 (분)</label><input id="targetMinutes" name="period_minutes" type="number" min="1" step="1" value="390" placeholder="예: 390"></div>
        <div class="note" id="autoSimulationBasis">실전 판단 기준자금과 수익률 게인은 실전 계좌 기준으로 자동 산정됩니다.</div>
        <button type="submit">가능성 분석</button>
        <div class="work-status" id="workStatus">
          <strong id="workTitle">작업 대기 중</strong>
          <div class="muted" id="workMessage">버튼을 누르면 진행 현황을 표시합니다.</div>
          <div class="bar"><span id="workProgress"></span></div>
        </div>
      </form>
      <hr style="margin: 20px 0; border: none; border-top: 1px solid var(--line);">
      <h2>실전 게이트 상태</h2>
      <div class="note">모의투자 흐름은 제거되었고, 실전 KIS 게이트 상태만 표시합니다.</div>
      <div class="work-status active" id="streamingDemoContainer" hidden>
        <strong id="streamingDemoStatus">대기 중</strong>
        <div class="bar"><span id="streamingDemoProgress" style="width:0%"></span></div>
        <div style="margin-top: 10px; font-size: 12px;">
          <div class="score-row">
            <span class="score-label">예수금</span>
            <span class="score-value" id="streamingDeposit">-</span>
          </div>
          <div class="score-row">
            <span class="score-label">투자금</span>
            <span class="score-value" id="streamingInvested">-</span>
          </div>
          <div class="score-row">
            <span class="score-label">수익금</span>
            <span class="score-value" id="streamingProfit">-</span>
          </div>
          <div class="score-row">
            <span class="score-label">수익률</span>
            <span class="score-value" id="streamingReturnRate">0%</span>
            <canvas class="mini-chart" id="streamingReturnChart" width="280" height="74"></canvas>
          </div>
        </div>
      </div>
      </form>
    </aside>
    <main>
      <div class="grid">
        <section class="panel span-4"><h2>자산평가</h2><div class="metric" id="equity">-</div><table class="asset-table"><tbody><tr><th>총 자산</th><td id="totalAssets">-</td></tr><tr><th>주문 가능 원화</th><td id="cash">-</td></tr><tr><th>국내 보유 주식</th><td id="domesticInvestedValue">-</td></tr><tr><th>주문 가능 달러</th><td id="usdCash">-</td></tr><tr><th>해외 보유 주식</th><td id="foreignInvestedValue">-</td></tr></tbody></table><span id="investedValue" hidden></span><span id="krwCash" hidden></span><span id="foreignCash" hidden></span><span id="cashWeight" hidden></span></section>
        <section class="panel span-4"><h2 id="performancePanelTitle">실전 투자 성과</h2><div class="metric" id="mockReturn">대기 중</div><div class="bar"><span id="mockReturnBar"></span></div><div class="chips" style="margin-top:12px;"><span class="chip" id="mockProfit">실전 손익 -</span><span class="chip" id="mockEquity">실전 평가금 -</span><span class="chip" id="mockTarget">실전 목표 -</span></div><p class="muted" id="mockStatus" style="margin-bottom:0;">KIS 실전 계좌 기준 성과와 게이트 상태를 표시합니다.</p></section>
        <section class="panel span-4">
          <h2>보유 주식 현황</h2>
          <div class="metric" id="brokerDeposit">읽기 전</div>
          <div class="chips" style="margin-top:12px;">
            <span class="chip" id="brokerAccount">계좌번호 -</span>
            <span class="chip" id="brokerHoldings">보유 종목 -</span>
            <span class="chip" id="brokerEquity" hidden></span>
            <span class="chip" id="brokerInvested" hidden></span>
            <span class="chip" id="brokerKrwCash" hidden></span>
            <span class="chip" id="brokerForeignCash" hidden></span>
          </div>
          <div class="holding-lists">
            <div class="holding-list"><strong>국내 보유 주식 목록</strong><ul id="brokerDomesticHoldings"><li class="empty">-</li></ul></div>
            <div class="holding-list"><strong>해외 보유 주식 목록</strong><ul id="brokerForeignHoldings"><li class="empty">-</li></ul></div>
          </div>
          <p class="muted" id="brokerStatus" style="margin-bottom:0;">실전 준비 점검에서 읽기 전용으로 실제 예수금과 보유 종목 수만 확인합니다. 주문은 제출하지 않습니다.</p>
        </section>
        <section class="panel span-4">
          <h2>원금 보호</h2>
          <div class="metric" id="principalMode">대기 중</div>
          <div class="bar"><span id="principalCushionBar"></span></div>
          <div class="chips" style="margin-top:12px;">
            <span class="chip" id="principalFloor">보호 바닥 -</span>
            <span class="chip" id="principalGrowth">성장 자본 -</span>
            <span class="chip" id="principalBudget">위험 예산 -</span>
          </div>
          <p class="muted" id="principalStatus" style="margin-bottom:0;">초기 원금을 설정하면 BUY 주문 전 원금 보호 게이트가 적용됩니다.</p>
        </section>
        <section class="panel span-8">
          <h2>달성 가능성</h2>
          <div class="metric" id="feasibility">대기 중</div>
          <div class="bar"><span id="feasibilityBar"></span></div>
          <div id="scoreBreakdown" style="margin-top:14px;"></div>
          <p class="muted" id="summary">목표를 입력하면 시장 자료, 온톨로지 관계, 리스크 정렬을 바탕으로 달성 가능성을 계산합니다.</p>
        </section>
        <section class="panel span-12">
          <h2>실시간 자료 진단 <span class="muted" id="liveRefreshBadge">갱신 대기</span></h2>
          <div class="stats" id="diagnosticStats"></div>
          <div class="stats" id="storeStats" style="margin-top:10px;"></div>
          <div class="warning-list" id="collectionWarnings"></div>
          <div class="data-volume-wrap">
            <canvas class="data-volume-chart" id="dataVolumeChart" width="640" height="180"></canvas>
            <div class="source-volume-list" id="sourceVolumeList"></div>
          </div>
        </section>
        <section class="ontology-scene ontology-wide-layout">
          <div class="ontology-toolbar">
            <span class="ontology-badge">실시간 3D GNN 네트워크</span>
            <span class="ontology-badge" id="ontologyCounts">노드 - | 관계 -</span>
            <button id="resetGraph" type="button">전체 그래프 맞춤</button>
            <button id="toggleLabels" type="button">라벨 켜기</button>
            <button id="toggleReasoning" type="button">추론 일시정지</button>
            <span class="ontology-badge" id="reasoningBadge">추론 단계 -</span>
            <div class="ontology-filter" id="ontologyFilters">
              <label><input type="checkbox" value="ticker" checked>종목</label>
              <label><input type="checkbox" value="event" checked>이벤트</label>
              <label><input type="checkbox" value="temporal" checked>시간축</label>
              <label><input type="checkbox" value="support" checked>긍정</label>
              <label><input type="checkbox" value="risk" checked>리스크</label>
              <label><input type="checkbox" value="contradiction" checked>상충</label>
              <label><input type="checkbox" value="sector" checked>섹터</label>
              <label><input type="checkbox" value="pipeline" checked>Pipeline</label>
              <label><input type="checkbox" value="tuning" checked>Tuning</label>
              <label><input type="checkbox" value="parameter" checked>Parameter</label>
              <label><input type="checkbox" value="metric" checked>Metric</label>
              <label><input type="checkbox" value="entity" checked>개체</label>
            </div>
          </div>
          <canvas id="ontologyCanvas"></canvas>
          <div class="ontology-panel" id="ontologyPanel">
            <strong>노드를 선택하세요</strong>
            <div class="muted">마우스를 올리거나 클릭하면 노드 종류와 연결 관계를 확인할 수 있습니다.</div>
          </div>
          <div class="reasoning-strip">
            <div>
              <strong id="reasoningTitle">실시간 추론 대기</strong>
              <span id="reasoningMeta">그래프를 불러오면 추론 경로가 순차적으로 강조됩니다.</span>
              <div class="reasoning-progress"><span id="reasoningProgress"></span></div>
            </div>
            <div id="reasoningDescription">활성화된 노드와 엣지가 밝게 빛나며 현재 판단 근거를 보여줍니다.</div>
          </div>
          <div class="ontology-legend">
            <span class="legend-item"><span class="legend-dot" style="background:#38bdf8"></span>종목</span>
            <span class="legend-item"><span class="legend-dot" style="background:#f97316"></span>뉴스/이벤트</span>
            <span class="legend-item"><span class="legend-dot" style="background:#06b6d4"></span>시간축</span>
            <span class="legend-item"><span class="legend-dot" style="background:#22c55e"></span>긍정 신호</span>
            <span class="legend-item"><span class="legend-dot" style="background:#ef4444"></span>리스크</span>
            <span class="legend-item"><span class="legend-dot" style="background:#d946ef"></span>상충 요인</span>
            <span class="legend-item"><span class="legend-dot" style="background:#84cc16"></span>섹터</span>
            <span class="legend-item"><span class="legend-dot" style="background:#2563eb"></span>Pipeline</span>
            <span class="legend-item"><span class="legend-dot" style="background:#eab308"></span>Tuning</span>
            <span class="legend-item"><span class="legend-dot" style="background:#ec4899"></span>Parameter</span>
            <span class="legend-item"><span class="legend-dot" style="background:#94a3b8"></span>Metric</span>
            <span class="legend-item"><span class="legend-dot" style="background:#f8fafc"></span>개체</span>
          </div>
          <div id="ontologyTooltip"></div>
        </section>
        <section class="panel span-12"><h2>목표 대안</h2><div class="cards" id="choices"></div><div style="margin-top:14px;"><button id="startButton" disabled>선택한 목표로 시작</button> <button class="secondary" id="resetButton" type="button">초기화</button></div></section>
        <section class="panel span-12"><h2 id="runProgressTitle">실시간 모의 진행</h2><div class="stats" id="mockRunStats"></div><div class="grid" style="margin-top:12px;"><div class="span-12"><h2 id="executionTableTitle">최근 체결 및 종료 청산</h2><div class="table-wrap"><table class="live-table"><thead><tr><th>구분</th><th>종목</th><th>수량</th><th>가격/금액</th></tr></thead><tbody id="mockExecutions"><tr><td colspan="4">체결 내역 없음</td></tr></tbody></table></div><div style="margin-top: 12px;"><h2>스트리밍 데모 거래</h2><div class="table-wrap"><table class="live-table"><thead><tr><th>종목</th><th>구분</th><th>수량</th><th>금액</th></tr></thead><tbody id="streamingTradeList"><tr><td colspan="4">거래 없음</td></tr></tbody></table></div></div></div></div></section>
        <section class="panel span-4"><h2>온톨로지 신호</h2><div class="chips" id="relations"></div></section>
        <section class="panel span-8"><h2>자료 및 프로그램 출력</h2><div class="log" id="output">아직 실행하지 않았습니다.</div></section>
      </div>
    </main>
  </div>
  <script>
    let sessionId = null;
    let selectedGoal = null;
    let graphState = null;
    let lastGoalPayload = null;
    let lastGraphSignature = '';
    let liveRefreshBusy = false;
    let statusBusy = false;
    let diagnosticsBusy = false;
    let learningStatusBusy = false;
    let learningStatusTimer = null;
    let lastRenderedCollectionCycle = null;
    let mockPerformanceTimer = null;
    let operationModeBusy = false;
    let liveTradingProgressBusy = false;
    let operationRequestActive = false;
    let activeOperationMode = null;
    let lastBrokerConnection = null;
    let liveAccountBasis = null;
    
    let streamingDemoId = null;
    let streamingDemoRunning = false;
    let streamingDemoHistory = [];
    let streamingDemoPrices = {};
    let streamingInitialCash = 0;
    let streamingStepBusy = false;
    let streamingDemoTimer = null;
    let streamingStepFailures = 0;
    let streamingReturnSeries = [];
    const fmtWon = new Intl.NumberFormat('ko-KR', { style: 'currency', currency: 'KRW', maximumFractionDigits: 0 });
    const fmtUsd = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 });

    function formatMoney(value, currency = 'KRW') {
      const numeric = Number(value || 0);
      return String(currency || 'KRW').toUpperCase() === 'USD' ? fmtUsd.format(numeric) : fmtWon.format(numeric);
    }

    function translateGoalLabel(label) {
      const translations = {
        'Requested target': '요청된 목표',
        'Lower return': '낮은 수익',
        'Longer period': '연장된 기간',
        'Balanced compromise': '균형잡힌 절충',
      };
      return translations[label] || label;
    }

    function formatCashByCurrency(account = {}) {
      const cashByCurrency = account && account.cash_by_currency && typeof account.cash_by_currency === 'object'
        ? account.cash_by_currency
        : null;
      if (!cashByCurrency) {
        return formatMoney(Number(account.cash || 0), account.base_currency || 'KRW');
      }
      const preferred = ['KRW', 'USD'];
      const currencies = [
        ...preferred.filter((currency) => Object.prototype.hasOwnProperty.call(cashByCurrency, currency)),
        ...Object.keys(cashByCurrency).filter((currency) => !preferred.includes(currency)).sort(),
      ];
      if (!currencies.length) {
        return formatMoney(Number(account.cash || 0), account.base_currency || 'KRW');
      }
      return currencies.map((currency) => formatMoney(Number(cashByCurrency[currency] || 0), currency)).join(' / ');
    }

    function currencyMap(account = {}) {
      const source = account && account.cash_by_currency && typeof account.cash_by_currency === 'object'
        ? account.cash_by_currency
        : {};
      const result = {};
      Object.keys(source).forEach((currency) => {
        const code = String(currency || '').toUpperCase();
        if (code) result[code] = Number(source[currency] || 0);
      });
      if (!Object.prototype.hasOwnProperty.call(result, 'KRW')) {
        result.KRW = Number(account.krw_cash ?? account.actual_deposit ?? account.cash ?? 0);
      }
      return result;
    }

    function formatForeignCash(account = {}) {
      const balances = currencyMap(account);
      const currencies = Object.keys(balances).filter((currency) => currency !== 'KRW').sort();
      if (!currencies.length) return '-';
      return currencies.map((currency) => formatMoney(balances[currency], currency)).join(' / ');
    }

    function accountSnapshotSummary(data = {}) {
      const equity = Number(data.equity ?? data.actual_equity ?? data.account_value ?? data.total_evaluation_amount ?? 0);
      const balances = currencyMap(data);
      const krwCash = Number(data.krw_cash ?? balances.KRW ?? data.actual_deposit ?? 0);
      const foreignCashKrw = Number(data.foreign_cash_krw ?? 0);
      const cashFromCurrency = krwCash + Math.max(0, foreignCashKrw);
      const cash = Number(data.cash ?? data.actual_deposit ?? krwCash);
      const cashEquivalentKrw = Number(data.cash_equivalent_krw ?? cashFromCurrency);
      const invested = Math.max(0, equity - cashEquivalentKrw);
      const cashWeight = Number(data.cash_weight ?? (equity > 0 ? cash / equity : 0));
      return { equity, cash, cashEquivalentKrw, invested, krwCash, foreignCashKrw, cashByCurrency: balances, foreignCashByCurrency: data.foreign_cash_by_currency || {}, cashWeight };
    }

    function positionCurrency(position = {}) {
      const explicit = String(position.currency || '').toUpperCase();
      if (explicit) return explicit;
      const ticker = String(position.ticker || '');
      return /^\d{6}$/.test(ticker) ? 'KRW' : 'USD';
    }

    function splitInvestmentSummary(data = {}, summary = accountSnapshotSummary(data)) {
      const positions = Array.isArray(data.positions) ? data.positions : [];
      let domestic = 0;
      let foreign = 0;
      let foreignUsd = 0;
      positions.forEach((position) => {
        const rawValue = Number(position.market_value || 0);
        const krwValue = Number(position.market_value_krw || 0);
        if (positionCurrency(position) === 'KRW') domestic += krwValue > 0 ? krwValue : rawValue;
        else {
          foreign += krwValue > 0 ? krwValue : 0;
          foreignUsd += rawValue;
        }
      });
      if (!positions.length) domestic = Number(summary.invested || 0);
      return {
        domesticInvested: Math.max(0, domestic),
        foreignInvested: Math.max(0, foreign),
        foreignInvestedUsd: Math.max(0, foreignUsd),
        usdCash: Number((summary.cashByCurrency || {}).USD ?? (data.cash_by_currency || {}).USD ?? 0),
      };
    }

    function renderHoldingList(targetId, positions = []) {
      const target = document.getElementById(targetId);
      if (!target) return;
      const rows = positions.slice(0, 8).map((position) => {
        const ticker = String(position.ticker || '-');
        const qty = Number(position.quantity || 0);
        const currency = positionCurrency(position);
        const pending = String(position.position_state || '') === 'pending_balance';
        const label = pending ? `${ticker} · 잔고반영 대기` : ticker;
        const value = currency === 'KRW'
          ? formatMoney(Number(position.market_value_krw ?? position.market_value ?? 0), 'KRW')
          : formatMoney(Number(position.market_value ?? 0), 'USD');
        return `<li><span>${label} x ${qty}</span><span>${value}</span></li>`;
      }).join('');
      target.innerHTML = rows || '<li class="empty">-</li>';
    }

    function applyCompactKoreanDashboard() {
      document.title = 'Live Trading Dashboard';
      const title = document.querySelector('aside h1');
      if (title) title.textContent = 'Live Trading Dashboard';
      const gate = document.getElementById('gate');
      if (gate) gate.textContent = 'Learning and collection continue automatically while the server is running.';
      const stepLabel = document.querySelector('.mode-step-label');
      if (stepLabel) stepLabel.textContent = 'Operation Mode';
      const learningButton = document.getElementById('modeLearningButton');
      const paperButton = document.getElementById('modeTestingButton');
      const liveButton = document.getElementById('modeLiveButton');
      const liveTradingButton = document.getElementById('modeLiveTradingButton');
      const refreshButton = document.getElementById('modeLearningStopButton');
      if (learningButton) {
        learningButton.innerHTML = 'Learning status<small>Runs while server is open</small>';
        learningButton.onclick = () => loadLearningStatus();
      }
      if (paperButton) {
        paperButton.innerHTML = '실전 투자<small>KIS 실전 서버 전용</small>';
        paperButton.onclick = () => startSelectedOperationMode('live');
      }
      if (liveButton) {
        liveButton.innerHTML = 'Live readiness<small>Auth check without orders</small>';
        liveButton.onclick = () => startSelectedOperationMode('live_test');
      }
      if (liveTradingButton) {
        liveTradingButton.innerHTML = 'Live auto trading<small>Risk-gated execution</small>';
        liveTradingButton.onclick = () => startSelectedOperationMode('live');
      }
      if (refreshButton) {
        refreshButton.innerHTML = 'Refresh status<small>Collection stays active</small>';
        refreshButton.disabled = false;
        refreshButton.onclick = () => loadLearningStatus();
      }
      const flowCopy = {
        mode: ['Mode', 'Live auto trading'],
        data: ['Collection', 'Auto refresh while server runs'],
        analysis: ['Learning', 'Realtime store updates'],
        simulation: ['Live gate', 'Waiting for KIS/API state'],
      };
      document.querySelectorAll('#systemFlowPanel .flow-step').forEach((step) => {
        const copy = flowCopy[step.dataset.flowStep];
        if (!copy) return;
        const strong = step.querySelector('strong');
        const span = step.querySelector('span');
        if (strong) strong.textContent = copy[0];
        if (span) span.textContent = copy[1];
      });
    }
    async function loadStatus() {
      if (statusBusy) return;
      statusBusy = true;
      try {
        const data = await (await fetch('/api/status')).json();
        renderStatus(data);
        if (data.basis_source === 'kis_live_account') {
          renderBrokerAccountCard({
            ...data,
            ok: true,
            mode: 'live',
            account_checked: true,
            holdings_count: data.holdings_count ?? (Array.isArray(data.positions) ? data.positions.length : 0),
            holdings: data.holdings ?? data.holdings_count ?? (Array.isArray(data.positions) ? data.positions.length : 0),
          });
        }
      } finally {
        statusBusy = false;
      }
    }

    async function loadLearningStatus() {
      if (learningStatusBusy) return;
      learningStatusBusy = true;
      try {
        const data = await (await fetch('/api/live-progress')).json();
        renderLearningStatus(data);
        updateLearningStopButton(data.learning);
        renderCollectionLog(data.collection_log || []);
        maybeRefreshDiagnosticsAfterCollection(data.collection_log || []);
        const progress = data.progress || {};
        if (progress.stage === 'error') {
          renderSystemFlow({ data: 'error' }, { data: progress.message || 'Data refresh failed' });
        } else if (data.is_refreshing || progress.active) {
          renderSystemFlow({ data: 'active' }, { data: progress.message || 'Refreshing data' });
        } else if ((progress.percent || 0) >= 100) {
          renderSystemFlow({ data: 'done', analysis: 'done' }, {
            data: progress.message || 'Realtime data collection complete; waiting for next scheduled cycle',
            analysis: 'Analysis cache ready',
          });
        }
      } catch (error) {
        renderLearningStatus({
          is_refreshing: false,
          progress: {
            active: false,
            percent: 0,
            stage: 'error',
            message: String(error && error.message ? error.message : error),
          },
        });
      } finally {
        learningStatusBusy = false;
      }
    }

    async function loadDiagnostics() {
      if (diagnosticsBusy) return;
      diagnosticsBusy = true;
      try {
        const data = await (await fetch('/api/research/diagnostics')).json();
        renderDiagnostics(data);
      } catch (error) {
        const message = `Diagnostics error: ${error.message || error}`;
        const badge = document.getElementById('liveRefreshBadge');
        if (badge) badge.textContent = message;
        renderSystemFlow({ data: 'error' }, { data: message });
        throw error;
      } finally {
        diagnosticsBusy = false;
      }
    }

    async function loadOntologyGraph() {
      const canvas = document.getElementById('ontologyCanvas');
      if (!canvas || canvas.offsetParent === null) return;
      try {
        const data = await (await fetch('/api/ontology/graph?full=true', { cache: 'no-store' })).json();
        if (!data || !data.counts) return;
        const display = data.display_counts || {};
        const suffix = data.truncated ? ` | shown ${display.nodes || 0}/${data.counts.nodes} nodes, ${display.links || 0}/${data.counts.links} links` : '';
        document.getElementById('ontologyCounts').textContent = `Nodes ${data.counts.nodes} | Links ${data.counts.links}${suffix}`;
        // 그래프가 비었거나(컨텍스트 미준비) 직전과 동일하면 재렌더 생략 — 준비되면 자동으로 그려진다.
        const signature = graphSignature(data);
        if (signature === lastGraphSignature && Number(data.counts.nodes || 0) > 0) {
          if (graphState && graphState.applyLiveTrace) graphState.applyLiveTrace(data.live_trace);
          return;
        }
        if (Number(data.counts.nodes || 0) <= 0) return;
        lastGraphSignature = signature;
        await renderOntologyGraph(data);
        renderSystemFlow({ analysis: 'done' }, { analysis: 'Ontology graph ready' });
      } catch (error) {
        console.error('ontology graph load failed', error);
      }
    }

    async function loadOntologyLiveTrace() {
      if (!graphState || !graphState.applyLiveTrace) return;
      try {
        const payload = await (await fetch('/api/ontology/live-trace', { cache: 'no-store' })).json();
        if (payload && payload.trace) graphState.applyLiveTrace(payload.trace);
      } catch (error) {
        console.debug('live ontology trace unavailable', error);
      }
    }

    async function loadRealtimeRuntime() {
      const data = await (await fetch('/api/realtime/runtime')).json();
      const accel = data.acceleration || {};
      const policy = data.short_horizon_policy || {};
      const eventLlm = data.event_llm || {};
      const ontologyNpu = data.ontology_npu || {};
      const training = data.live_training || {};
      const npuLabel = accel.uses_npu ? 'NPU active' : `CPU fallback (${accel.active_backend || '-'})`;
      const llmLabel = eventLlm.available ? `LLM ${eventLlm.provider || '-'} ready` : `LLM waiting ${eventLlm.reason || 'not configured'}`;
      const ontologyNpuLabel = ontologyNpu.uses_npu
        ? (Number(ontologyNpu.last_items || 0) > 0
          ? `Ontology NPU ${ontologyNpu.last_items} rows ${ontologyNpu.last_latency_ms ? `${ontologyNpu.last_latency_ms}ms` : 'ready'}`
          : 'Ontology NPU ready, no batch yet')
        : `Ontology fallback ${ontologyNpu.backend || '-'}`;
      const trainingLabel = training.ok
        ? `training rows ${training.training_rows || 0} | model ${training.inference_uses_latest_live_eligible ? 'live' : (training.model_saved ? 'saved' : 'waiting')}`
        : 'training status waiting';
      document.getElementById('runtimeStatus').textContent =
        `${npuLabel} | ${ontologyNpuLabel} | ${trainingLabel} | ${llmLabel} | ${accel.latency_profile || 'low_latency'} | horizons ${((accel.prediction_horizons_seconds || []).join('/'))}s | cap ${((policy.max_position_weight_intraday || 0) * 100).toFixed(1)}%`;
      if (data.operation_mode) renderOperationMode(data.operation_mode);
    }
    function selectedOperationMode(action) {
      if (action === 'training') return 'learning';
      if (action === 'testing' || action === 'paper') return 'live_trading';
      if (action === 'live_test') return 'live_readiness';
      return 'live_trading';
    }

    function updateModeButtons() {
      const learningButton = document.getElementById('modeLearningButton');
      const refreshButton = document.getElementById('modeLearningStopButton');
      const paperButton = document.getElementById('modeTestingButton');
      const liveButton = document.getElementById('modeLiveButton');
      const liveTradingButton = document.getElementById('modeLiveTradingButton');
      const liveFlagsButton = document.getElementById('liveFlagsButton');
      if (!paperButton) return;
      if (learningButton) learningButton.disabled = operationRequestActive;
      paperButton.disabled = operationRequestActive;
      if (liveButton) liveButton.disabled = operationRequestActive;
      if (liveTradingButton) liveTradingButton.disabled = operationRequestActive;
      if (liveFlagsButton) liveFlagsButton.disabled = operationRequestActive;
      updateTerminateTradingButton();
      if (refreshButton) refreshButton.disabled = false;
      if (learningButton) learningButton.innerHTML = '수집/학습 상태<small>서버 실행 중 자동 진행</small>';
      paperButton.innerHTML = '실전 투자<small>KIS 실전 서버 전용</small>';
      if (liveButton) liveButton.innerHTML = '실전 준비 점검<small>주문 없이 인증 확인</small>';
      if (liveTradingButton) liveTradingButton.innerHTML = '실전 투자<small>자동매매 게이트</small>';
      if (refreshButton) refreshButton.innerHTML = '상태 새로고침<small>수집은 중단하지 않음</small>';
    }

    function updateTerminateTradingButton() {
      const button = document.getElementById('terminateTradingButton');
      if (!button) return;
      const canTerminate = streamingDemoRunning || activeOperationMode === 'live_trading';
      button.disabled = operationRequestActive || !canTerminate;
      button.innerHTML = activeOperationMode === 'live_trading'
        ? '조기 종료<small>실전 보유분 지정가 전량 매도</small>'
        : '조기 종료<small>실전 투자 세션 없음</small>';
    }

    function setModeButtonsLocked(locked) {
      const learningButton = document.getElementById('modeLearningButton');
      const refreshButton = document.getElementById('modeLearningStopButton');
      const paperButton = document.getElementById('modeTestingButton');
      const liveButton = document.getElementById('modeLiveButton');
      const liveTradingButton = document.getElementById('modeLiveTradingButton');
      const liveFlagsButton = document.getElementById('liveFlagsButton');
      const terminateButton = document.getElementById('terminateTradingButton');
      const enabled = !locked;
      if (learningButton) learningButton.disabled = !enabled;
      if (paperButton) paperButton.disabled = !enabled;
      if (liveButton) liveButton.disabled = !enabled;
      if (liveTradingButton) liveTradingButton.disabled = !enabled;
      if (liveFlagsButton) liveFlagsButton.disabled = !enabled;
      if (terminateButton) terminateButton.disabled = locked || !(streamingDemoRunning || activeOperationMode === 'live_trading');
      if (refreshButton) refreshButton.disabled = false;
    }

    function renderSystemFlow(states = {}, messages = {}) {
      document.querySelectorAll('#systemFlowPanel .flow-step').forEach((step) => {
        const key = step.dataset.flowStep;
        const state = Object.prototype.hasOwnProperty.call(states, key)
          ? states[key]
          : (step.dataset.flowState || 'idle');
        step.classList.remove('active', 'done', 'error');
        if (state === 'active' || state === 'done' || state === 'error') step.classList.add(state);
        step.dataset.flowState = state;
        const label = step.querySelector('span');
        if (label && messages[key]) label.textContent = messages[key];
      });
    }

    function modeLabel(mode) {
      return {
        learning: '수집/학습',
        testing: '실전 투자 진행',
        paper_trading: '실전 투자 진행',
        live_readiness: '실전 준비 점검',
        live_trading: '실전 투자 진행',
      }[mode] || mode || '운영 모드';
    }

    function isPaperTradingMode(mode) {
      return false;
    }

    function isTradingCheckMode(mode) {
      return isPaperTradingMode(mode) || mode === 'live_readiness' || mode === 'live_trading';
    }

    function operationStarted(data) {
      return Boolean(data.paper_trading_status || data.live_readiness_status || data.live_trading_status || data.training_status);
    }

    function startsStreamingLoop(mode, data) {
      return false;
    }

    function setPerformancePanelMode(mode, statusMessage = '') {
      const isLive = mode === 'live_trading' || mode === 'live';
      const title = document.getElementById('performancePanelTitle');
      if (title) title.textContent = '실전 투자 성과';
      const profit = document.getElementById('mockProfit');
      const equity = document.getElementById('mockEquity');
      const target = document.getElementById('mockTarget');
      if (profit) profit.textContent = '실전 손익 -';
      if (equity) equity.textContent = '실전 평가금 -';
      if (target) target.textContent = '실전 목표 -';
      if (statusMessage) {
        const status = document.getElementById('mockStatus');
        if (status) status.textContent = statusMessage;
      }
    }

    function stopStreamingDemoLoop(message = '') {
      if (streamingDemoTimer) {
        window.clearTimeout(streamingDemoTimer);
        streamingDemoTimer = null;
      }
      streamingDemoRunning = false;
      streamingStepBusy = false;
      streamingDemoId = null;
      if (activeOperationMode !== 'live_trading') activeOperationMode = null;
      updateTerminateTradingButton();
      const container = document.getElementById('streamingDemoContainer');
      if (container) container.hidden = true;
      if (message) {
        const statusNode = document.getElementById('streamingDemoStatus');
        const mockNode = document.getElementById('mockStatus');
        if (statusNode) statusNode.textContent = message;
        if (mockNode && activeOperationMode !== 'live_trading') mockNode.textContent = message;
      }
    }
    async function fetchJsonWithTimeout(url, options = {}, timeoutMs = 20000) {
      if (!timeoutMs || timeoutMs <= 0) {
        const response = await fetch(url, options);
        if (!response.ok) throw new Error(await response.text());
        return await response.json();
      }
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
      try {
        const response = await fetch(url, { ...options, signal: controller.signal });
        if (!response.ok) throw new Error(await response.text());
        return await response.json();
      } finally {
        window.clearTimeout(timeout);
      }
    }

    async function fetchWithOptionalTimeout(url, options = {}, timeoutMs = 0) {
      if (!timeoutMs || typeof AbortController === 'undefined') {
        return await fetch(url, options);
      }
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
      try {
        return await fetch(url, { ...options, signal: controller.signal });
      } finally {
        window.clearTimeout(timeout);
      }
    }

    function renderLiveFlagResult(data) {
      const failures = data && data.readiness && data.readiness.failures ? data.readiness.failures : {};
      const failureText = Object.entries(failures).map(([key, value]) => `${key}: ${value}`).join('\\n');
      const liveReady = Boolean(data.live_ready);
      document.getElementById('operationModeStatus').textContent = liveReady
        ? 'Live flags applied | readiness gates passed; manual arming is still required.'
        : 'Live flags applied | waiting for readiness gates.';
      document.getElementById('runtimeStatus').textContent = data.message || 'Live flag state updated.';
      document.getElementById('gate').textContent = liveReady
        ? 'Live flags are active. Orders still require manual arming and FinalOrder approval.'
        : 'Live flags are active. Live orders remain safely blocked until readiness checks pass.';
      document.getElementById('output').textContent = JSON.stringify(data, null, 2);
      renderSystemFlow(
        { mode: liveReady ? 'done' : 'active' },
        { mode: liveReady ? 'Live flags active' : `Readiness pending${failureText ? `: ${failureText.split('\\n')[0]}` : ''}` }
      );
    }

    async function applyLiveFlags() {
      if (operationRequestActive) return;
      operationRequestActive = true;
      setModeButtonsLocked(true);
      document.getElementById('operationModeStatus').textContent = 'Applying live flags...';
      document.getElementById('runtimeStatus').textContent = 'No order will be submitted by this action.';
      try {
        const data = await fetchJsonWithTimeout('/api/live-flags/apply', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ confirmation: 'APPLY_LIVE_FLAGS' })
        }, 30000);
        renderLiveFlagResult(data);
      } catch (error) {
        const message = String(error && error.message ? error.message : error);
        document.getElementById('operationModeStatus').textContent = 'Live flag apply failed';
        document.getElementById('runtimeStatus').textContent = message;
        document.getElementById('output').textContent = message;
      } finally {
        operationRequestActive = false;
        setModeButtonsLocked(false);
        updateModeButtons();
      }
    }

    async function startOperationMode(mode, options = {}) {
      if (operationRequestActive) {
        document.getElementById('runtimeStatus').textContent = 'Another operation request is already running.';
        return;
      }
      if (mode === 'live_trading') {
        activeOperationMode = 'live_trading';
        setPerformancePanelMode('live_trading', '실전 투자 성과 대기 중 · KIS 계좌와 실주문 게이트를 확인합니다.');
      }
      if (!isPaperTradingMode(mode)) {
        stopStreamingDemoLoop(`${modeLabel(mode)} requested; paper trading loop stopped.`);
      }
      operationRequestActive = true;
      renderSystemFlow({
        mode: 'active',
        data: 'idle',
        analysis: 'idle',
        simulation: isTradingCheckMode(mode) ? 'active' : 'idle',
      }, {
        mode: `${modeLabel(mode)} start requested`,
        simulation: isPaperTradingMode(mode) ? 'Preparing paper trading' : (mode === 'live_trading' ? 'Checking live auto-trading gate' : 'Checking live readiness'),
      });
      document.getElementById('operationModeStatus').textContent = `${modeLabel(mode)} start requested...`;
      document.getElementById('runtimeStatus').textContent = 'Waiting for server response.';
      document.getElementById('output').textContent = `${modeLabel(mode)} request sent.`;
      setModeButtonsLocked(true);
      try {
        const data = await fetchJsonWithTimeout('/api/operation-mode/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mode, ...options })
        }, 45000);
        if (data.ok === false) throw new Error(data.message || data.status || 'operation mode request failed');
        renderOperationMode(data);
        if (mode === 'live_trading') {
          activeOperationMode = 'live_trading';
          loadMockPerformance().catch(() => {});
        }
        updateLearningStopButton(data.learning);
        const started = operationStarted(data);
        renderSystemFlow({
          mode: 'done',
          data: mode === 'learning' || started ? 'active' : 'done',
          analysis: mode === 'learning' ? 'active' : 'idle',
          simulation: started && isTradingCheckMode(mode) ? 'active' : 'idle',
        }, {
          mode: `${modeLabel(mode)} started`,
          data: 'Collection and learning continue',
          simulation: isPaperTradingMode(mode) ? 'Paper trading running' : (mode === 'live_trading' ? 'Live auto-trading gate checked' : 'Live readiness checked'),
        });
        document.getElementById('output').textContent = JSON.stringify(data, null, 2);
        if (startsStreamingLoop(mode, data)) {
          document.getElementById('streamingDemoContainer').hidden = false;
          streamingDemoId = data.demo_id;
          streamingDemoRunning = true;
          activeOperationMode = mode;
          updateTerminateTradingButton();
          streamingInitialCash = Number(data.initial_cash || options.initial_cash || 10000000);
          streamingTargetReturnRate = Number(data.target_return_rate || options.target_return_rate || 0);
          if (streamingTargetReturnRate > 1) streamingTargetReturnRate /= 100.0;
          streamingTargetMinutes = Number(data.period_minutes || options.period_minutes || 0);
          streamingDemoHistory = [];
          streamingDemoPrices = {};
          streamingReturnSeries = [];
          streamingStepFailures = 0;
          if (streamingDemoTimer) window.clearTimeout(streamingDemoTimer);
          renderStreamingPerformance({
            progress: 0,
            account: { cash: streamingInitialCash, account_value: streamingInitialCash, return_rate: 0 },
            status: 'running',
          });
          updateStreamingAccount({
            account: { cash: streamingInitialCash, account_value: streamingInitialCash, return_rate: 0, base_currency: 'KRW', cash_by_currency: { KRW: streamingInitialCash } },
          });
          drawStreamingReturnChart();
          autoRunStreamingDemo(true);
        }
        loadRealtimeRuntime().catch((error) => {
          document.getElementById('runtimeStatus').textContent = `Runtime status failed: ${error.message || error}`;
        });
      } catch (error) {
        const message = error.name === 'AbortError'
          ? 'Server response timed out. Try again shortly.'
          : String(error && error.message ? error.message : error);
        document.getElementById('operationModeStatus').textContent = `${modeLabel(mode)} start failed`;
        document.getElementById('runtimeStatus').textContent = message;
        document.getElementById('output').textContent = message;
        renderSystemFlow({ mode: 'error' }, { mode: message });
        loadOperationModeStatus().catch(() => {});
      } finally {
        operationRequestActive = false;
        setModeButtonsLocked(false);
        updateModeButtons();
      }
    }
    async function startSelectedOperationMode(action) {
      const mode = selectedOperationMode(action);
      if (!mode) return;
      const goal = currentGoalPayload();
      if (!goal && (action === 'testing' || action === 'paper')) {
        document.getElementById('output').textContent = 'Enter target return and target time first.';
        return;
      }
      const options = goal ? {
        target_return_rate: goal.target_return_rate,
        period_minutes: goal.period_minutes,
        initial_cash_source: 'auto',
      } : {};
      await startOperationMode(mode, options);
    }

    async function terminateActiveTrading() {
      if (operationRequestActive) return;
      if (!streamingDemoRunning && activeOperationMode !== 'live_trading') {
        document.getElementById('runtimeStatus').textContent = 'No active paper or live trading session to terminate.';
        return;
      }
      operationRequestActive = true;
      setModeButtonsLocked(true);
      document.getElementById('operationModeStatus').textContent = 'Early termination requested...';
      document.getElementById('runtimeStatus').textContent = 'Submitting final liquidation request.';
      try {
        const isLiveTermination = activeOperationMode === 'live_trading' && !streamingDemoRunning;
        const url = isLiveTermination ? '/api/live-trading/terminate' : `/api/paper-trading/terminate/${streamingDemoId}`;
        const data = await fetchJsonWithTimeout(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        }, 45000);
        document.getElementById('output').textContent = JSON.stringify(data, null, 2);
        if (data.ok === false) {
          document.getElementById('operationModeStatus').textContent = 'Early termination blocked';
          document.getElementById('runtimeStatus').textContent = data.message || data.status || 'Termination failed.';
          return;
        }
        if (data.account) updateStreamingAccount(data);
        if (data.final_results) renderStreamingPerformance({ progress: 100, account: data.account || {}, status: 'terminated' });
        streamingDemoRunning = false;
        streamingStepBusy = false;
        streamingDemoId = null;
        activeOperationMode = null;
        document.getElementById('operationModeStatus').textContent = 'Trading terminated';
        document.getElementById('runtimeStatus').textContent = data.message || 'All current holdings were submitted for liquidation.';
        const statusNode = document.getElementById('streamingDemoStatus');
        if (statusNode) statusNode.textContent = '조기 종료 완료';
      } catch (error) {
        const message = String(error && error.message ? error.message : error);
        document.getElementById('operationModeStatus').textContent = 'Early termination failed';
        document.getElementById('runtimeStatus').textContent = message;
        document.getElementById('output').textContent = message;
      } finally {
        operationRequestActive = false;
        setModeButtonsLocked(false);
        updateTerminateTradingButton();
      }
    }

    async function stopLearningCollection() {
      const stopLearningButton = document.getElementById('modeLearningStopButton');
      if (stopLearningButton) stopLearningButton.disabled = true;
      document.getElementById('learningStatusMessage').textContent = '학습 데이터 수집 상태를 새로고침하는 중입니다.';
      try {
        const data = await fetchJsonWithTimeout('/api/operation-mode/stop-learning', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        }, 10000);
        updateLearningStopButton(data.learning);
        renderCollectionLog(data.collection_log || []);
        renderLearningStatus({
          is_refreshing: false,
          learning: data.learning,
          progress: data.progress || { active: false, percent: 0, stage: 'idle', message: data.message },
        });
        document.getElementById('output').textContent = JSON.stringify(data, null, 2);
        loadOperationModeStatus().catch(() => {});
      } catch (error) {
        document.getElementById('learningStatusMessage').textContent = String(error && error.message ? error.message : error);
        loadLearningStatus().catch(() => {});
      }
    }

    function renderOperationMode(data) {
      activeOperationMode = data.mode || null;
      updateTerminateTradingButton();
      const labels = {
        learning: 'Learning',
        testing: 'Live trading',
        paper_trading: 'Live trading',
        live_readiness: 'Live readiness',
        live_trading: 'Live trading',
      };
      const mode = labels[data.mode] || data.mode || 'Mode waiting';
      const message = data.paper_trading_message || data.live_readiness_message || data.live_trading_message || data.training_message || data.execution_label || '';
      document.getElementById('operationModeStatus').textContent = `${mode} | ${message}`;
      if (data.mode === 'live_trading') {
        setPerformancePanelMode('live_trading', '실전 투자 성과 대기 중 · 실행 요약을 갱신합니다.');
      } else if (data.mode === 'paper_trading') {
        setPerformancePanelMode('paper_trading');
      }
      if (data.kis_connection && (data.kis_connection.account_checked || data.kis_connection.ok)) {
        lastBrokerConnection = data.kis_connection;
      }
      const connection = data.kis_connection || lastBrokerConnection;
      if ((data.mode === 'live_readiness' || data.mode === 'live_trading') && connection) {
        renderBrokerAccountCard(connection);
      }
      document.getElementById('gate').textContent = data.mode === 'paper_trading'
        ? 'Paper trading mode uses the KIS virtual broker and blocks live orders.'
        : data.mode === 'live_readiness'
          ? 'Live readiness checks authentication and state without submitting orders.'
          : data.mode === 'live_trading'
            ? `Live auto-trading gate ${data.live_trading_status || 'checked'}; RiskManager and FinalTradeGate remain mandatory.`
            : 'Learning and collection continue while the server is running.';
    }

    /*
    function renderBrokerAccountCard(connection = {}) {
      const depositTarget = document.getElementById('brokerDeposit');
      const holdingsTarget = document.getElementById('brokerHoldings');
      const accountTarget = document.getElementById('brokerAccount');
      const statusTarget = document.getElementById('brokerStatus');
      if (!depositTarget || !holdingsTarget || !accountTarget || !statusTarget) return;
      const accountSuffix = connection.account_suffix || '-';
      accountTarget.textContent = `계좌 ${accountSuffix}`;
      if (connection.account_checked) {
        const deposit = Number(connection.actual_deposit ?? connection.cash ?? 0);
        depositTarget.textContent = fmtWon.format(deposit);
        holdingsTarget.textContent = `실보유 종목 ${connection.holdings || 0}개`;
        statusTarget.textContent = 'KIS 실계좌 읽기 완료 · 주문 제출 없음';
        document.getElementById('runtimeStatus').textContent =
          `실계좌 예수금 ${fmtWon.format(deposit)} · 실보유 종목 ${connection.holdings || 0}개 · 주문 제출 없음`;
        return;
      }
      if (connection.ok === false) {
        depositTarget.textContent = '조회 실패';
        holdingsTarget.textContent = '실보유 종목 -';
        statusTarget.textContent = connection.message || connection.error || 'KIS 계좌 조회에 실패했습니다.';
        if (connection.retry_after_seconds) {
          statusTarget.textContent += ` ${connection.retry_after_seconds}초 후 다시 시도하세요.`;
        }
        return;
      }
      depositTarget.textContent = '읽기 전';
      holdingsTarget.textContent = '실보유 종목 -';
      statusTarget.textContent = '실전 준비 점검에서 읽기 전용으로 실제 예수금과 보유 종목 수만 확인합니다. 주문은 제출하지 않습니다.';
    }
    */

    /*
    function renderBrokerAccountCard(connection = {}) {
      const depositTarget = document.getElementById('brokerDeposit');
      const holdingsTarget = document.getElementById('brokerHoldings');
      const accountTarget = document.getElementById('brokerAccount');
      const statusTarget = document.getElementById('brokerStatus');
      if (!depositTarget || !holdingsTarget || !accountTarget || !statusTarget) return;
      const accountSuffix = connection.account_suffix || '-';
      accountTarget.textContent = `계좌 ${accountSuffix}`;
      if (connection.account_checked) {
        const deposit = Number(connection.actual_deposit ?? connection.cash ?? 0);
        const basis = brokerBasisAmount(connection);
        const holdings = connection.holdings_count ?? connection.holdings ?? 0;
        const submitted = Number((connection.live_order_journal || {}).submitted_count ?? connection.submitted_count ?? 0);
        const updatedAt = connection.updated_at ? new Date(connection.updated_at).toLocaleTimeString('ko-KR') : new Date().toLocaleTimeString('ko-KR');
        depositTarget.textContent = fmtWon.format(deposit);
        holdingsTarget.textContent = `실보유 종목 ${holdings}개`;
        statusTarget.textContent = 'KIS 실계좌 읽기 완료 · 주문 제출 없음';
        applyLiveAccountBasis(connection);
        document.getElementById('runtimeStatus').textContent =
          `KIS live account read complete | basis ${fmtWon.format(basis || deposit)} | holdings ${holdings}`;
        return;
      }
      if (connection.ok === false) {
        depositTarget.textContent = '조회 실패';
        holdingsTarget.textContent = '실보유 종목 -';
        statusTarget.textContent = connection.message || connection.error || 'KIS 계좌 조회에 실패했습니다.';
        if (connection.retry_after_seconds) {
          statusTarget.textContent += ` ${connection.retry_after_seconds}초 후 다시 시도하세요.`;
        }
        return;
      }
      depositTarget.textContent = '읽기 전';
      holdingsTarget.textContent = '실보유 종목 -';
      statusTarget.textContent = '실전 준비 점검에서 읽기 전용으로 실제 예수금과 보유 종목 수만 확인합니다. 주문은 제출하지 않습니다.';
    }

    */

    function renderBrokerAccountCard(connection = {}) {
      const depositTarget = document.getElementById('brokerDeposit');
      const holdingsTarget = document.getElementById('brokerHoldings');
      const accountTarget = document.getElementById('brokerAccount');
      const equityTarget = document.getElementById('brokerEquity');
      const investedTarget = document.getElementById('brokerInvested');
      const krwCashTarget = document.getElementById('brokerKrwCash');
      const foreignCashTarget = document.getElementById('brokerForeignCash');
      const statusTarget = document.getElementById('brokerStatus');
      if (!depositTarget || !holdingsTarget || !accountTarget || !statusTarget) return;
      const accountSuffix = connection.account_suffix || '-';
      accountTarget.textContent = `Account ${accountSuffix}`;
      if (connection.account_checked) {
        const summary = accountSnapshotSummary(connection);
        const holdings = connection.holdings_count ?? connection.holdings ?? 0;
        depositTarget.textContent = fmtWon.format(summary.cash);
        if (equityTarget) equityTarget.textContent = `총자산 ${fmtWon.format(summary.equity)}`;
        if (investedTarget) investedTarget.textContent = `보유주식 ${fmtWon.format(summary.invested)}`;
        if (krwCashTarget) krwCashTarget.textContent = `원화 ${fmtWon.format(summary.krwCash)}`;
        if (foreignCashTarget) foreignCashTarget.textContent = `외화 ${formatForeignCash(connection)}`;
        holdingsTarget.textContent = `Holdings ${holdings}`;
        statusTarget.textContent = `KIS 실계좌 실시간 갱신 ${updatedAt} · 제출 ${submitted}건`;
        applyLiveAccountBasis(connection);
        document.getElementById('runtimeStatus').textContent =
          `KIS live account read complete | 총자산 ${fmtWon.format(summary.equity)} | KRW ${fmtWon.format(summary.krwCash)} | 외화 ${formatForeignCash(connection)} | 보유주식 ${fmtWon.format(summary.invested)} | holdings ${holdings}`;
        return;
      }
      if (connection.ok === false) {
        depositTarget.textContent = 'Read failed';
        holdingsTarget.textContent = 'Holdings -';
        if (equityTarget) equityTarget.textContent = '총자산 -';
        if (investedTarget) investedTarget.textContent = '보유주식 -';
        if (krwCashTarget) krwCashTarget.textContent = 'KRW -';
        if (foreignCashTarget) foreignCashTarget.textContent = '외화 -';
        statusTarget.textContent = connection.message || connection.error || 'KIS account lookup failed.';
        if (connection.retry_after_seconds) {
          statusTarget.textContent += ` Retry after ${connection.retry_after_seconds}s.`;
        }
        return;
      }
      depositTarget.textContent = 'Before read';
      holdingsTarget.textContent = 'Holdings -';
      if (krwCashTarget) krwCashTarget.textContent = 'KRW -';
      if (foreignCashTarget) foreignCashTarget.textContent = '외화 -';
      statusTarget.textContent = 'KIS 실계좌 조회 대기 중';
    }

    function brokerBasisAmount(connection = {}) {
      const summary = accountSnapshotSummary(connection);
      return Number(summary.cash || 0);
    }

    function applyLiveAccountBasis(connection = {}) {
      const summary = accountSnapshotSummary(connection);
      const basis = Number(summary.cash || 0);
      if (!basis) return;
      const cash = Number(summary.cash || 0);
      liveAccountBasis = {
        cash,
        cash_equivalent_krw: Number(summary.cashEquivalentKrw || cash),
        krw_cash: Number(summary.krwCash),
        foreign_cash_krw: Number(summary.foreignCashKrw),
        cash_by_currency: summary.cashByCurrency,
        foreign_cash_by_currency: connection.foreign_cash_by_currency || {},
        base_currency: connection.base_currency || 'KRW',
        equity: Number(summary.equity || basis),
        cash_weight: Number(summary.cashWeight || 0),
        account_suffix: connection.account_suffix || '',
      };
      const initialCashInput = document.getElementById('initialCash');
      if (initialCashInput) initialCashInput.value = String(Math.round(basis));
      const autoBasisTarget = document.getElementById('autoSimulationBasis');
      if (autoBasisTarget) {
        autoBasisTarget.textContent = `시뮬레이션 기준 현금 ${fmtWon.format(basis)} · 총자산 ${fmtWon.format(summary.equity || basis)}`;
      }
      renderStatus({
        cash: liveAccountBasis.cash,
        cash_equivalent_krw: liveAccountBasis.cash_equivalent_krw,
        krw_cash: liveAccountBasis.krw_cash,
        foreign_cash_krw: liveAccountBasis.foreign_cash_krw,
        cash_by_currency: liveAccountBasis.cash_by_currency,
        foreign_cash_by_currency: liveAccountBasis.foreign_cash_by_currency,
        base_currency: liveAccountBasis.base_currency,
        equity: liveAccountBasis.equity,
        cash_weight: liveAccountBasis.cash_weight,
        basis_source: 'kis_live_account',
        account_suffix: liveAccountBasis.account_suffix,
      });
      const goal = currentGoalPayload();
      if (goal) lastGoalPayload = goal;
    }

    function renderPrincipalProtection(data = {}) {
      const state = data.state || {};
      const modeTarget = document.getElementById('principalMode');
      const floorTarget = document.getElementById('principalFloor');
      const growthTarget = document.getElementById('principalGrowth');
      const budgetTarget = document.getElementById('principalBudget');
      const statusTarget = document.getElementById('principalStatus');
      const barTarget = document.getElementById('principalCushionBar');
      if (!modeTarget || !floorTarget || !growthTarget || !budgetTarget || !statusTarget || !barTarget) return;
      const mode = state.current_mode || 'NOT_CONFIGURED';
      const equity = Number(state.current_equity || 0);
      const floorValue = Number(state.protected_floor || 0);
      const cushion = Number(state.cushion || 0);
      const riskBudget = Number(state.risk_budget || 0);
      const growth = Number(state.available_growth_capital || 0);
      const distance = equity - floorValue;
      const pct = equity > 0 ? Math.max(0, Math.min(100, (cushion / equity) * 100)) : 0;
      modeTarget.textContent = mode;
      floorTarget.textContent = `보호 바닥 ${fmtWon.format(floorValue)}`;
      growthTarget.textContent = `성장 자본 ${fmtWon.format(growth)}`;
      budgetTarget.textContent = `위험 예산 ${fmtWon.format(riskBudget)}`;
      barTarget.style.width = `${pct.toFixed(1)}%`;
      if (mode === 'PRINCIPAL_LOCKDOWN') {
        statusTarget.textContent = `BUY 차단: 보호 바닥까지 ${fmtWon.format(distance)} 남았습니다. SELL/축소만 허용됩니다.`;
      } else if (mode === 'DE_RISK') {
        statusTarget.textContent = 'DE_RISK: 신규 매수는 제한되고 노출 축소를 우선합니다.';
      } else if (mode === 'NOT_CONFIGURED') {
        statusTarget.textContent = '초기 원금을 설정하면 원금 보호 바닥과 이익 재투자 예산이 계산됩니다.';
      } else {
        statusTarget.textContent = `보호 여유 ${fmtWon.format(cushion)} · 고점 대비 낙폭 ${(Number(state.drawdown_from_high_watermark || 0) * 100).toFixed(2)}%`;
      }
    }

    async function loadPrincipalProtectionState() {
      const data = await fetchJsonWithTimeout('/api/risk/principal-protection/state', {}, 8000);
      renderPrincipalProtection(data);
      return data;
    }

    async function loadOperationModeStatus() {
      if (operationModeBusy) return;
      operationModeBusy = true;
      try {
      const data = await fetchJsonWithTimeout('/api/operation-mode/status', {}, 8000);
      const request = data.request || {};
      if (request.busy) {
        renderSystemFlow({ mode: 'active' }, { mode: request.message || 'Request running' });
      } else if (request.stage === 'error') {
        renderSystemFlow({ mode: 'error' }, { mode: request.last_error || request.message || 'Request failed' });
      }
      updateLearningStopButton(data.learning);
      if (Array.isArray(data.collection_log)) {
        renderCollectionLog(data.collection_log);
        maybeRefreshDiagnosticsAfterCollection(data.collection_log);
      }
      if (data.kis_connection && (data.kis_connection.account_checked || data.kis_connection.ok)) {
        lastBrokerConnection = data.kis_connection;
        renderBrokerAccountCard(data.kis_connection);
        if (!data.active) {
          const suffix = data.kis_connection.account_suffix || '';
          const basis = brokerBasisAmount(data.kis_connection);
          document.getElementById('operationModeStatus').textContent =
            `Auto live readiness checked | account ${suffix || '-'} | basis ${fmtWon.format(basis || 0)}`;
          document.getElementById('gate').textContent =
            'KIS live account is connected read-only. Paper trading and live gate use this account basis.';
        }
      }
      if (data.active) renderOperationMode(data.active);
      return data;
      } finally {
        operationModeBusy = false;
      }
    }
    document.getElementById('goalForm').addEventListener('submit', async (event) => {
      event.preventDefault();
      const payload = currentGoalPayload();
      if (!payload) {
        document.getElementById('output').textContent = 'Enter target return and target time.';
        return;
      }
      const stopProgress = startLocalProgress('Assessing feasibility', 'Calculating goal feasibility from current data and ontology.');
      setBusy(true);
      try {
        lastGoalPayload = payload;
        const data = await fetchJsonWithTimeout('/api/assess-goal', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        }, 45000);
        renderAssessment(data);
        setWorkStatus('Assessment complete', 'Goal feasibility and alternatives were calculated.', 100, true);
      } catch (error) {
        document.getElementById('output').textContent = String(error && error.message ? error.message : error);
        setWorkStatus('Assessment failed', String(error && error.message ? error.message : error), 100, true);
      } finally {
        stopProgress();
        setBusy(false);
      }
    });

    const loadResearchButton = document.getElementById('loadResearch');
    if (loadResearchButton) {
      loadResearchButton.addEventListener('click', async () => {
        const stopProgress = startProgressPolling('Starting collection', 'Information collection continues in the background.');
        setBusy(true);
        try {
          const data = await (await fetch('/api/research/refresh', { method: 'POST' })).json();
          document.getElementById('output').textContent = JSON.stringify(data, null, 2);
          await loadDiagnostics();
          setWorkStatus('Collection started', 'Collection is running in the background.', 35, false);
        } catch (error) {
          document.getElementById('output').textContent = String(error && error.message ? error.message : error);
          setWorkStatus('Collection failed', String(error && error.message ? error.message : error), 100, true);
        } finally {
          stopProgress();
          setBusy(false);
        }
      });
    }

    function scoreRow(label, value, tone = 'good', valueText = null) {
      const numeric = Number(value || 0);
      const bounded = Math.max(0, Math.min(100, numeric));
      const text = valueText ?? `${bounded.toFixed(1)}%`;
      const toneClass = tone === 'bad' ? 'score-bad' : tone === 'warn' ? 'score-warn' : 'score-good';
      return `
        <div class="score-row">
          <span>${label}</span>
          <span class="${toneClass}">${text}</span>
        </div>
      `;
    }

    function renderAssessment(data, options = {}) {
      if (data.session_id) sessionId = data.session_id;
      const previousSelection = selectedGoal;
      if (!options.preserveSelection) selectedGoal = null;
      const assessment = data.assessment;
      document.getElementById('feasibility').textContent = `${assessment.feasibility_percent}%`;
      document.getElementById('feasibilityBar').style.width = `${assessment.feasibility_percent}%`;
      document.getElementById('scoreBreakdown').innerHTML = `
        ${scoreRow('Market support', assessment.market_support_percent, 'good')}
        ${scoreRow('Risk pressure', assessment.risk_pressure_percent, 'warn')}
        ${scoreRow('Goal drag', assessment.annualized_drag_percent, 'bad')}
        ${scoreRow('Annual target', Math.min(100, assessment.annualized_required_return * 100), 'bad', `${(assessment.annualized_required_return * 100).toFixed(1)}%`)}
      `;
      document.getElementById('summary').textContent = assessment.reasoning.join(' ');
      document.getElementById('relations').innerHTML = assessment.ontology_relations.slice(0, 10).map((item) => `<span class="chip">${item}</span>`).join('');
      const choices = document.getElementById('choices');
      choices.innerHTML = '';
      data.compromises.forEach((goal, index) => {
        const div = document.createElement('div');
        div.className = 'choice';
        div.innerHTML = `<strong>${translateGoalLabel(goal.label)}</strong><div class="metric">${goal.feasibility_percent}%</div><div class="muted">Return ${(goal.target_return_rate * 100).toFixed(2)}%</div><div class="muted">Profit ${fmtWon.format(goal.target_profit_amount)}</div><div class="muted">Period ${goal.period_days}d</div>`;
        div.addEventListener('click', () => {
          document.querySelectorAll('.choice').forEach((node) => node.classList.remove('selected'));
          div.classList.add('selected');
          selectedGoal = goal;
          document.getElementById('startButton').disabled = false;
        });
        choices.appendChild(div);
        const sameAsPrevious = previousSelection && goal.label === previousSelection.label
          && goal.period_days === previousSelection.period_days
          && Math.abs(goal.target_return_rate - previousSelection.target_return_rate) < 0.000001;
        if ((options.preserveSelection && sameAsPrevious) || (!options.preserveSelection && index === 0)) div.click();
      });
    }
    function currentGoalPayload() {
      const form = document.getElementById('goalForm');
      const payload = Object.fromEntries(new FormData(form).entries());
      const targetReturnRate = Number(payload.target_return_rate || 0);
      const periodMinutes = Number(payload.period_minutes || 0);
      if (!targetReturnRate || targetReturnRate < 0) return null;
      if (!periodMinutes || periodMinutes < 1) return null;
      return {
        target_return_rate: targetReturnRate,
        period_minutes: periodMinutes,
        period_days: Math.max(1, Math.ceil(periodMinutes / 390)),
      };
    }

    function applyUrlGoalParams() {
      const params = new URLSearchParams(window.location.search);
      const targetReturn = params.get('target_return_rate');
      const periodMinutes = params.get('period_minutes');
      if (targetReturn !== null) {
        document.getElementById('targetReturn').value = targetReturn;
      }
      if (periodMinutes !== null) {
        document.getElementById('targetMinutes').value = periodMinutes;
      }
      const goal = currentGoalPayload();
      if (goal) lastGoalPayload = goal;
    }

    function applyGoalPayloadToMode(payload) {
      if (!payload) return {};
      return {
        target_return_rate: payload.target_return_rate,
        period_minutes: payload.period_minutes,
        initial_cash_source: 'auto',
      };
    }

    function currentGoalPayloadForAssessment() {
      const payload = currentGoalPayload();
      if (!payload) return null;
      return {
        target_return_rate: payload.target_return_rate,
        period_minutes: payload.period_minutes,
        period_days: payload.period_days,
      };
    }

    function updateModeActionCopy() {
      updateModeButtons();
      document.getElementById('operationModeStatus').textContent =
        'Choose learning status, paper trading, or live readiness.';
    }

    document.querySelectorAll('#unusedEnvironmentGrid button').forEach((button) => {
      button.addEventListener('click', () => {
        document.querySelectorAll('#unusedEnvironmentGrid button').forEach((node) => node.classList.remove('active'));
        button.classList.add('active');
      });
    });

    updateModeButtons();
    updateModeActionCopy();

    function setBusy(isBusy) {
      document.querySelector('#goalForm button[type="submit"]').disabled = isBusy;
      document.getElementById('loadResearch').disabled = isBusy;
    }

    function setWorkStatus(title, message, percent, visible = true) {
      const box = document.getElementById('workStatus');
      const bounded = Math.max(0, Math.min(100, Number(percent) || 0));
      box.classList.toggle('active', visible);
      document.getElementById('workTitle').textContent = title;
      document.getElementById('workMessage').textContent = message;
      document.getElementById('workProgress').style.width = `${bounded}%`;
    }

    function startProgressPolling(title, fallbackMessage) {
      let stopped = false;
      let localPercent = 8;
      setWorkStatus(title, fallbackMessage, localPercent, true);
      const timer = window.setInterval(async () => {
        if (stopped) return;
        try {
          const data = await (await fetch('/api/live-progress')).json();
          const progress = data.progress || {};
          if (progress.active || data.is_refreshing) {
            setWorkStatus(title, progress.message || fallbackMessage, progress.percent || localPercent, true);
            return;
          }
        } catch (error) {
          // Keep the local progress moving even if the progress endpoint is briefly busy.
        }
        localPercent = Math.min(92, localPercent + 4);
        setWorkStatus(title, fallbackMessage, localPercent, true);
      }, 700);
      return () => {
        stopped = true;
        window.clearInterval(timer);
      };
    }

    function startLocalProgress(title, message) {
      let stopped = false;
      let percent = 18;
      setWorkStatus(title, message, percent, true);
      const timer = window.setInterval(() => {
        if (stopped) return;
        percent = Math.min(92, percent + 11);
        setWorkStatus(title, message, percent, true);
      }, 250);
      return () => {
        stopped = true;
        window.clearInterval(timer);
      };
    }

    async function refreshLiveSnapshot() {
      if (streamingDemoRunning || operationRequestActive) return;
      if (liveRefreshBusy) return;
      liveRefreshBusy = true;
      const badge = document.getElementById('liveRefreshBadge');
      if (badge) badge.textContent = 'Refreshing';
      try {
        const goal = currentGoalPayload();
        if (goal) lastGoalPayload = goal;
        const res = await fetchWithOptionalTimeout('/api/live-snapshot', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ goal: lastGoalPayload, force_refresh: false, include_graph: false }),
        }, 12000);
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        renderStatus(data.status);
        renderDiagnostics(data.diagnostics);
        const ontologyCounts = document.getElementById('ontologyCounts');
        if (ontologyCounts && data.graph && data.graph.counts) {
          ontologyCounts.textContent = `Nodes ${data.graph.counts.nodes} | Links ${data.graph.counts.links}`;
        }
        const ontologyCanvas = document.getElementById('ontologyCanvas');
        if (ontologyCanvas && ontologyCanvas.offsetParent !== null && data.graph && Array.isArray(data.graph.nodes) && Array.isArray(data.graph.links)) {
          const signature = graphSignature(data.graph);
          if (signature !== lastGraphSignature) {
            lastGraphSignature = signature;
            await renderOntologyGraph(data.graph);
          }
        }
        if (data.assessment && data.compromises) {
          renderAssessment({ assessment: data.assessment, compromises: data.compromises }, { preserveSelection: true });
        }
        const updatedText = data.updated_at ? new Date(data.updated_at).toLocaleTimeString('ko-KR') : 'waiting';
        const errorText = data.status && data.status.last_error ? ` | error ${data.status.last_error}` : '';
        if (badge) badge.textContent = `Last refresh ${updatedText}${errorText}`;
      } catch (error) {
        if (badge) badge.textContent = 'Refresh failed';
        console.error(error);
      } finally {
        liveRefreshBusy = false;
      }
    }

    function startLearningStatusPolling() {
      if (learningStatusTimer) window.clearInterval(learningStatusTimer);
      loadLearningStatus();
      learningStatusTimer = window.setInterval(loadLearningStatus, 1500);
    }

    function renderStatus(data) {
      if (liveAccountBasis && (!data || data.basis_source !== 'kis_live_account')) {
        data = {
          ...(data || {}),
          cash: liveAccountBasis.cash,
          cash_equivalent_krw: liveAccountBasis.cash_equivalent_krw,
          krw_cash: liveAccountBasis.krw_cash,
          foreign_cash_krw: liveAccountBasis.foreign_cash_krw,
          cash_by_currency: liveAccountBasis.cash_by_currency,
          foreign_cash_by_currency: liveAccountBasis.foreign_cash_by_currency,
          base_currency: liveAccountBasis.base_currency,
          equity: liveAccountBasis.equity,
          cash_weight: liveAccountBasis.cash_weight,
          basis_source: 'kis_live_account',
          account_suffix: liveAccountBasis.account_suffix,
        };
      }
      const summary = accountSnapshotSummary(data);
      const split = splitInvestmentSummary(data, summary);
      document.getElementById('equity').textContent = fmtWon.format(summary.cash);
      const totalAssetsTarget = document.getElementById('totalAssets');
      if (totalAssetsTarget) totalAssetsTarget.textContent = fmtWon.format(summary.equity);
      document.getElementById('cash').textContent = fmtWon.format(summary.cash);
      const domesticInvestedTarget = document.getElementById('domesticInvestedValue');
      if (domesticInvestedTarget) domesticInvestedTarget.textContent = fmtWon.format(split.domesticInvested);
      const usdCashTarget = document.getElementById('usdCash');
      if (usdCashTarget) usdCashTarget.textContent = formatMoney(split.usdCash, 'USD');
      const foreignInvestedTarget = document.getElementById('foreignInvestedValue');
      if (foreignInvestedTarget) foreignInvestedTarget.textContent = formatMoney(split.foreignInvestedUsd, 'USD');
      const investedValueTarget = document.getElementById('investedValue');
      if (investedValueTarget) investedValueTarget.textContent = `보유주식 ${fmtWon.format(summary.invested)}`;
      const krwCashTarget = document.getElementById('krwCash');
      if (krwCashTarget) krwCashTarget.textContent = `KRW 현금 ${fmtWon.format(summary.krwCash)}`;
      const foreignCashTarget = document.getElementById('foreignCash');
      if (foreignCashTarget) foreignCashTarget.textContent = `외화/해외평가 ${fmtWon.format(summary.foreignCashKrw)}`;
      document.getElementById('cashWeight').textContent = `현금 비중 ${(summary.cashWeight * 100).toFixed(1)}%`;
      if (data.basis_source === 'kis_live_account' && Number(data.equity || 0) > 0) {
        liveAccountBasis = {
          cash: Number(summary.cash),
          cash_equivalent_krw: Number(summary.cashEquivalentKrw),
          krw_cash: Number(summary.krwCash),
          foreign_cash_krw: Number(summary.foreignCashKrw),
          cash_by_currency: summary.cashByCurrency,
          foreign_cash_by_currency: summary.foreignCashByCurrency,
          base_currency: data.base_currency || 'KRW',
          equity: Number(summary.equity),
          cash_weight: Number(summary.cashWeight || 0),
          account_suffix: data.account_suffix || '',
        };
        const initialCashInput = document.getElementById('initialCash');
        if (initialCashInput) initialCashInput.value = String(Math.round(liveAccountBasis.cash));
        const autoBasisTarget = document.getElementById('autoSimulationBasis');
        if (autoBasisTarget) {
          autoBasisTarget.textContent = `시뮬레이션 기준 현금 ${fmtWon.format(liveAccountBasis.cash)} · 총자산 ${fmtWon.format(liveAccountBasis.equity)}`;
        }
      }
    }

    function renderLearningStatus(data) {
      const progress = data.progress || {};
      const learning = data.learning || {};
      const percent = Math.max(0, Math.min(100, Number(progress.percent) || 0));
      const active = Boolean(progress.active || data.is_refreshing);
      const stageLabels = {
        idle: 'idle', starting: 'starting', research: 'collecting', storage: 'storing',
        analysis: 'analysis', graph: 'graph', waiting: 'waiting', complete: 'complete', error: 'error',
      };
      const stage = stageLabels[progress.stage] || progress.stage || 'idle';
      const message = prettyLearningMessage(progress.message || 'Checking status');
      document.getElementById('learningStatusTitle').textContent = active
        ? 'Collection running'
        : (learning.active ? 'Collection complete; background learning scheduled' : 'Collection waiting');
      document.getElementById('learningStatusMessage').textContent = message;
      document.getElementById('learningStatusProgress').style.width = `${percent}%`;
      document.getElementById('learningStatusMeta').textContent = `${stage} | ${percent.toFixed(1)}%`;
      document.getElementById('learningStatusCard').classList.toggle('active', true);
    }

    function updateLearningStopButton(learning = {}) {
      const stopLearningButton = document.getElementById('modeLearningStopButton');
      if (stopLearningButton) stopLearningButton.disabled = false;
    }

    function renderCollectionLog(log = []) {
      const entries = Array.isArray(log) ? log.slice(-8) : [];
      const list = document.getElementById('learningCollectionLog');
      if (list) {
        list.innerHTML = entries.length
          ? entries.slice().reverse().map((entry) => {
              const when = entry.timestamp ? new Date(entry.timestamp).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' }) : '-';
              const status = String(entry.status || 'scheduled');
              const detail = entry.message || '';
              return `<div class="collection-log-item"><span>${when}</span><span><i class="collection-log-status ${status}"></i><strong>${collectionStatusLabel(status)}</strong> ${detail}</span><span></span></div>`;
            }).join('')
          : '<div class="muted">No collection log yet</div>';
      }
      drawCollectionLogChart(entries);
    }

    function maybeRefreshDiagnosticsAfterCollection(log = []) {
      if (!Array.isArray(log) || !log.length) return;
      const completed = log.filter((entry) => entry && entry.status === 'complete' && entry.cycle != null).slice(-1)[0];
      if (!completed || completed.cycle === lastRenderedCollectionCycle) return;
      lastRenderedCollectionCycle = completed.cycle;
      loadDiagnostics().catch((error) => console.error(error));
    }

    function collectionStatusLabel(status) {
      return { scheduled: 'scheduled', running: 'running', complete: 'complete', error: 'error', stopped: 'stopped' }[status] || status || 'status';
    }
    function drawCollectionLogChart(entries) {
      const canvas = document.getElementById('learningCollectionChart');
      if (!canvas || !canvas.getContext) return;
      const ctx = canvas.getContext('2d');
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = '#e2e8f0';
      ctx.beginPath();
      ctx.moveTo(8, height - 14);
      ctx.lineTo(width - 8, height - 14);
      ctx.stroke();
      if (!entries.length) {
        ctx.fillStyle = '#64748b';
        ctx.font = '12px Arial';
        ctx.fillText('수집 로그 대기 중', 12, 34);
        return;
      }
      const bars = entries.slice(-18);
      const gap = 4;
      const barWidth = Math.max(6, Math.floor((width - 24 - gap * (bars.length - 1)) / Math.max(1, bars.length)));
      bars.forEach((entry, index) => {
        const counts = entry.counts || {};
        const totalSeen = Number(counts.events_seen || 0) + Number(counts.raw_records_seen || 0) + Number(counts.market_snapshots_seen || 0) + Number(counts.macro_metrics_seen || 0);
        const value = Math.max(1, totalSeen);
        const barHeight = entry.status === 'complete' ? Math.min(height - 22, 10 + Math.log10(value + 1) * 18) : entry.status === 'running' ? height - 28 : 12;
        const x = 12 + index * (barWidth + gap);
        const y = height - 14 - barHeight;
        ctx.fillStyle = collectionStatusColor(entry.status);
        ctx.fillRect(x, y, barWidth, barHeight);
      });
    }
    function collectionStatusColor(status) {
      return {
        running: '#0f766e',
        complete: '#16a34a',
        error: '#dc2626',
        scheduled: '#64748b',
        stopped: '#64748b',
      }[status] || '#94a3b8';
    }

    function prettyLearningMessage(message) {
      const text = String(message || '').trim();
      if (!text) return '실시간 상태를 확인하는 중입니다.';
      if (text.startsWith('Retrying failed source:')) {
        return `재시도 중 · ${text.slice('Retrying failed source:'.length).trim()}`;
      }
      return text;
    }

    function liveExecutionReasonLabel(summary = {}) {
      if (!summary || typeof summary !== 'object') return '대기';
      const submitted = Number(summary.submitted || 0);
      const approvedBuy = Number(summary.approved_buy_orders || 0);
      const approvedSell = Number(summary.approved_sell_orders || 0);
      const buySignals = Number(summary.buy_signals || 0);
      const sellSignals = Number(summary.sell_signals || 0);
      const blocked = Array.isArray(summary.blocked) ? summary.blocked.length : 0;
      const errors = Array.isArray(summary.errors) ? summary.errors.length : 0;

      if (submitted > 0) return '주문 제출';
      if (approvedBuy + approvedSell > 0) return '승인 완료';
      if (buySignals + sellSignals > 0) return '게이트 심사중';
      if (blocked > 0) return '게이트 차단';
      if (errors > 0) return '실행 오류';
      return '대기';
    }

    function liveExecutionSummaryText(summary = {}, data = {}) {
      const runtimeGate = data.runtime_gate || summary.runtime_gate || {};
      const failures = Array.isArray(runtimeGate.failures) ? runtimeGate.failures : [];
      const skipped = Array.isArray(summary.cash_fit_skipped_orders) ? summary.cash_fit_skipped_orders.length : 0;
      const blocked = Array.isArray(summary.blocked) ? summary.blocked.length : 0;
      const errors = Array.isArray(summary.errors) ? summary.errors.length : 0;
      const reason = liveExecutionReasonLabel(summary);
      const gateText = runtimeGate.ok ? '실주문 가능' : `실주문 차단${failures.length ? `: ${failures.join(', ')}` : ''}`;
      return `실행 ${reason} · 신호 ${Number(summary.signals || 0)} · BUY ${Number(summary.buy_signals || 0)} · SELL/REDUCE ${Number(summary.sell_signals || 0)} · 후보 ${Number(summary.intents || 0)} · 승인 매수 ${Number(summary.approved_buy_orders || 0)} · 승인 매도 ${Number(summary.approved_sell_orders || 0)} · 실행가능 매수 ${Number(summary.executable_buy_orders || 0)} · 실행가능 매도 ${Number(summary.executable_sell_orders || 0)} · 제출 ${Number(summary.submitted || 0)} · 현금부족 제외 ${skipped} · 차단 ${blocked} · 오류 ${errors} · ${gateText}`;
    }

    const HOLD_REASON_TEXT = {
      HOLD_BELOW_PROFIT_TARGET: '아직 목표 수익 미달 → 보유',
      WIDE_SPREAD: '호가 스프레드가 넓어 매수 보류',
      LOW_LIQUIDITY: '유동성 부족으로 보류',
      INSUFFICIENT_CASH_FOR_ONE_SHARE: '1주 매수 현금 부족',
      FALLBACK_SCORE_BELOW_THRESHOLD: '매수 점수 기준 미달',
      ONTOLOGY_REQUIRED_FOR_MODEL_FALLBACK: '근거 확인 부족으로 보류',
      MODEL_AUXILIARY_ONLY_NEEDS_CONFIRMATION: '모델 단독 매수 불가(근거 필요)',
      MODEL_FEATURE_UNAVAILABLE: '실시간 데이터 부족으로 판단 보류',
      SELL_BELOW_BREAK_EVEN_BLOCKED: '손실 매도 방지(본전 미만)',
      SMALL_ACCOUNT_ONE_SHARE_LOSS_BLOCK: '소액계좌 보호(1주 손실매도 차단)',
      HOLD_UNPROFITABLE_ONTOLOGY_SELL_BLOCKED: '손실권 매도 보류',
      LOSS_EXIT_DISABLED: '손실 청산 비활성',
      MARKET_SESSION_CLOSED: '장 마감',
      MISSING_MARKET_DATA: '시세 없음',
      open_sell_kept: '기존 매도 주문 유지(중복 방지)',
      BELOW_TARGET_NET_RETURN_AFTER_COST: '비용 차감 후 목표 순수익 미달',
      BELOW_BREAK_EVEN_WITH_MARGIN: '본전(마진 포함) 미만 예상 → 매수 보류',
      COST_BURDEN_HIGH: '거래비용 부담 과다',
      SPREAD_TOO_WIDE: '호가 스프레드가 넓어 매수 보류',
      SPREAD_CONSUMES_ALPHA: '스프레드가 기대수익을 잠식',
      LIQUIDITY_TOO_LOW: '유동성 부족으로 보류',
      SLIPPAGE_RISK_HIGH: '슬리피지 위험 과다',
      PROFITABILITY_GATE_REJECTED: '수익성 게이트 거부(순기대수익 부족)',
      RECENT_LOSS_SYMBOL_COOLDOWN: '최근 손실 종목 재매수 대기',
      NO_SELLABLE_QUANTITY: '매도 가능 수량 없음',
      OPEN_ORDER_OR_SETTLEMENT_LOCK: '미체결 주문/결제 잠금',
    };
    function humanizeReasonCode(code) {
      const raw = String(code || '').trim();
      if (!raw) return '';
      const base = raw.split(':')[0].trim();
      return HOLD_REASON_TEXT[base] || base.replaceAll('_', ' ');
    }

    function liveExecutionRejectionText(summary = {}) {
      const rejections = Array.isArray(summary.rejections) ? summary.rejections.slice(0, 5) : [];
      if (!rejections.length) return '';
      return rejections.map((item) => {
        const side = String(item.side || '').toUpperCase();
        const symbol = item.symbol || item.ticker || '-';
        const reasons = Array.isArray(item.reason_codes) ? item.reason_codes : [];
        const reason = reasons.length ? reasons.map(humanizeReasonCode).join(' / ') : '사유 없음';
        return `${side} ${symbol}: ${reason}`;
      }).join(' | ');
    }

    function renderLivePerformanceSummary(data = {}) {
      setPerformancePanelMode('live_trading');
      const percent = Number(data.return_rate || 0) * 100;
      document.getElementById('mockReturn').textContent = `${percent.toFixed(2)}%`;
      document.getElementById('mockReturnBar').style.width = `${Math.max(0, Math.min(100, Math.abs(percent)))}%`;
      document.getElementById('mockProfit').textContent = `실전 손익 ${fmtWon.format(data.profit || 0)}`;
      document.getElementById('mockEquity').textContent = `실전 평가금 ${fmtWon.format(data.equity || 0)}`;
      document.getElementById('mockTarget').textContent = '실전 목표 -';
      const accountSummary = accountSnapshotSummary(data);
      const executionSummary = data.execution_summary || {};
      const rejectionText = liveExecutionRejectionText(executionSummary);
      document.getElementById('mockStatus').textContent =
        `KIS 실계좌 기준 갱신 · 총자산 ${fmtWon.format(accountSummary.equity)} · 주문가능 원화 ${fmtWon.format(accountSummary.cash)} · 보유주식 ${fmtWon.format(accountSummary.invested)} · 외화 ${fmtWon.format(accountSummary.foreignCashKrw)} · ${liveExecutionSummaryText(executionSummary, data)}`;
      if (rejectionText) document.getElementById('mockStatus').textContent += ` · 사유 ${rejectionText}`;
    }

    async function loadMockPerformance() {
      if (activeOperationMode === 'live_trading') {
        setPerformancePanelMode('live_trading');
        const data = await (await fetch('/api/live-trading/progress')).json();
        data.display_mode = 'live';
        window.setTimeout(() => renderLivePerformanceSummary(data), 0);
        return;
      }

      const data = await (await fetch('/api/mock-trading/performance')).json();
      const panelTitle = document.getElementById('runProgressTitle');
      const tableTitle = document.getElementById('executionTableTitle');
      if (panelTitle) panelTitle.textContent = '실시간 모의 진행';
      if (tableTitle) tableTitle.textContent = '최근 체결 및 종료 청산';
      const percent = Number(data.return_rate || 0) * 100;
      document.getElementById('mockReturn').textContent = data.active ? `${percent.toFixed(2)}%` : '대기 중';
      document.getElementById('mockReturnBar').style.width = `${Math.max(0, Math.min(100, Math.abs(percent)))}%`;
      document.getElementById('mockProfit').textContent = `모의 손익 ${fmtWon.format(data.profit || 0)}`;
      document.getElementById('mockEquity').textContent = `모의 평가 ${fmtWon.format(data.equity || 0)}`;
      document.getElementById('mockTarget').textContent = data.goal ? `목표 ${(Number(data.goal.target_return_rate || 0) * 100).toFixed(2)}%` : '목표 -';
      document.getElementById('mockStatus').textContent = data.active
        ? `Paper trading 진행 중 · 주문 ${data.order_count || 0}건 · 체결 ${data.execution_count || 0}건`
        : '실제 계좌와 분리된 paper trading 성과가 표시됩니다.';
    }
    function renderStreamingPerformance(data) {
      if (!data || !data.account) return;
      const account = data.account || {};
      const returnRate = Number(account.return_rate || 0) * 100;
      const accountValue = Number(account.account_value || streamingInitialCash || 0);
      const cash = Number(account.cash || 0);
      const initial = Number(streamingInitialCash || accountValue || 0);
      const profit = accountValue - initial;
      const simulatedProgress = Number(data.progress || 0);
      streamingReturnSeries.push({ progress: simulatedProgress, returnRate });
      if (streamingReturnSeries.length > 160) streamingReturnSeries.shift();
      document.getElementById('mockReturn').textContent = `${returnRate.toFixed(2)}%`;
      document.getElementById('mockReturnBar').style.width = `${Math.max(0, Math.min(100, Math.abs(returnRate)))}%`;
      document.getElementById('mockProfit').textContent = `모의 손익 ${fmtWon.format(profit)}`;
      document.getElementById('mockEquity').textContent = `모의 평가 ${fmtWon.format(accountValue)}`;
      document.getElementById('mockStatus').textContent =
        `Paper trading ${simulatedProgress.toFixed(1)}% 진행 · 목표 시간 ${streamingTargetMinutes || '-'}분 · 모의 현금 ${fmtWon.format(cash)}`;
      drawStreamingReturnChart();
    }
    function drawStreamingReturnChart() {
      const canvas = document.getElementById('streamingReturnChart');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = '#e6e9ef';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
      const points = streamingReturnSeries.length ? streamingReturnSeries : [{ progress: 0, returnRate: 0 }];
      const values = points.map((p) => Number(p.returnRate || 0) * 100);
      const target = Number(streamingTargetReturnRate || 0) * 100;
      const min = Math.min(-0.5, target, ...values);
      const max = Math.max(0.5, target, ...values);
      const yFor = (value) => height - 10 - ((value - min) / Math.max(0.001, max - min)) * (height - 20);
      if (target) {
        const y = yFor(target);
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = '#b45309';
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.strokeStyle = '#0f766e';
      ctx.lineWidth = 2;
      ctx.beginPath();
      points.forEach((point, index) => {
        const x = points.length === 1 ? 8 : (index / (points.length - 1)) * (width - 16) + 8;
        const y = yFor(Number(point.returnRate || 0) * 100);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.fillStyle = '#667085';
      ctx.font = '11px Arial';
      ctx.fillText(`현재 ${(values[values.length - 1] || 0).toFixed(2)}%`, 8, 15);
      if (target) ctx.fillText(`목표 ${target.toFixed(2)}%`, width - 82, 15);
    }

    function renderMockRunTables(data) {
      const positions = data.positions || [];
      const executions = data.recent_executions || [];
      const recentOrders = data.recent_orders || [];
      const isLive = data.display_mode === 'live' || data.mode === 'live_trading' || activeOperationMode === 'live_trading';
      const orderRows = isLive && recentOrders.length ? recentOrders : executions;
      document.getElementById('mockRunStats').innerHTML = `
        <div class="stat"><strong>${data.active ? '진행 중' : '대기'}</strong><span class="muted">모의투자 상태</span></div>
        <div class="stat"><strong>${data.orders_count || 0}</strong><span class="muted">주문</span></div>
        <div class="stat"><strong>${data.executions_count || 0}</strong><span class="muted">체결</span></div>
        <div class="stat"><strong>${positions.length}</strong><span class="muted">보유 종목</span></div>
      `;
      if (isLive) {
        document.getElementById('mockRunStats').innerHTML = `
          <div class="stat"><strong>${data.active ? '진행 중' : '실계좌 조회'}</strong><span class="muted">실전투자 상태</span></div>
          <div class="stat"><strong>${data.orders_count || 0}</strong><span class="muted">주문</span></div>
          <div class="stat"><strong>${data.executions_count || 0}</strong><span class="muted">체결</span></div>
          <div class="stat"><strong>${positions.length}</strong><span class="muted">보유 종목</span></div>
        `;
      }
      if (isLive) {
        const runtimeGate = data.runtime_gate || {};
        const journal = data.live_order_journal || {};
        document.getElementById('mockRunStats').innerHTML = `
          <div class="stat"><strong>${data.active ? '진행 중' : '실계좌 조회'}</strong><span class="muted">실전투자 상태</span></div>
          <div class="stat"><strong>${runtimeGate.ok ? '가능' : '차단'}</strong><span class="muted">실주문 게이트</span></div>
          <div class="stat"><strong>${journal.submitted_count || data.executions_count || 0}</strong><span class="muted">제출</span></div>
          <div class="stat"><strong>${journal.blocked_count || 0}</strong><span class="muted">차단</span></div>
        `;
      }
      const positionTarget = document.getElementById('mockPositions');
      if (positionTarget) positionTarget.innerHTML = positions.length ? positions.map((item) => {
        const pnl = Number(item.unrealized_pnl || 0);
        const rate = Number(item.return_rate || 0) * 100;
        const tone = pnl >= 0 ? 'tone-pos' : 'tone-neg';
        return `<tr>
          <td>${item.ticker}</td>
          <td>${item.quantity}</td>
          <td>${formatMoney(item.average_price || 0, item.currency)}</td>
          <td>${formatMoney(item.last_price || 0, item.currency)}</td>
          <td>${formatMoney(item.market_value || 0, item.currency)}</td>
          <td class="${tone}">${fmtWon.format(pnl)}</td>
          <td class="${tone}">${rate.toFixed(2)}%</td>
        </tr>`;
      }).join('') : '<tr><td colspan="7">보유 종목 없음</td></tr>';
      document.getElementById('mockExecutions').innerHTML = orderRows.length ? orderRows.slice().reverse().map((item) => {
        const side = orderSideLabel(item);
        const sideClass = side === 'BUY' ? 'side-buy' : side === 'SELL' ? 'side-sell' : '';
        const price = Number(item.price ?? item.limit_price ?? 0);
        const quantityValue = Number(item.quantity || 0);
        const quantity = quantityValue > 0 ? quantityValue : (item.status || '-');
        const ticker = item.ticker || item.market || item.event_type || '-';
        const currency = orderCurrency(item);
        const amount = Number(item.notional ?? (price * quantityValue) ?? 0);
        const reasonText = Array.isArray(item.reason_codes) ? item.reason_codes.map(humanizeReasonCode).join(', ') : (item.reason_codes ? humanizeReasonCode(item.reason_codes) : '');
        const detail = item.broker_order_id ? `주문 ${item.broker_order_id}` : (item.error || reasonText || '');
        const priceAmount = price > 0
          ? `${formatMoney(price, currency)} / ${formatMoney(amount, currency)}${detail ? ` · ${detail}` : ''}`
          : detail || '-';
        return `<tr>
          <td class="${sideClass}">${side}</td>
          <td>${ticker}</td>
          <td>${quantity}</td>
          <td>${priceAmount}</td>
        </tr>`;
      }).join('') : '<tr><td colspan="4">체결 이력 없음</td></tr>';
      if (isLive && !orderRows.length) {
        const failures = ((data.runtime_gate || {}).failures || []).join(', ');
        const emptyLiveOrderText = `<tr><td colspan="4">실전 주문 내역 없음${failures ? ` · 게이트 차단: ${failures}` : ''}</td></tr>`;
        document.getElementById('mockExecutions').innerHTML = emptyLiveOrderText;
        document.getElementById('mockExecutions').innerHTML = '<tr><td colspan="4">실시간 체결 내역은 주문 추적기에 기록되면 표시됩니다.</td></tr>';
        document.getElementById('mockExecutions').innerHTML = emptyLiveOrderText;
      }
    }

    function orderSideLabel(item = {}) {
      const side = String(item.side || '').toUpperCase();
      if (side === 'BUY' || side === 'SELL') return side;
      if (item.event_type === 'live_order_submitted') return item.status || 'SUBMITTED';
      return item.event_type || item.status || '-';
    }

    function orderCurrency(item = {}) {
      const explicit = String(item.currency || '').toUpperCase();
      if (explicit) return explicit;
      const market = String(item.market || '').toUpperCase();
      const ticker = String(item.ticker || '');
      if (market.includes('US') || market.includes('NASDAQ') || market.includes('NYSE') || market.includes('AMEX') || market.includes('NASD')) return 'USD';
      if (ticker && !(ticker.match(/^[0-9]{6}$/))) return 'USD';
      return 'KRW';
    }

    function renderDiagnostics(data = {}) {
      const d = data.diagnostics || data || {};
      const liveLabel = d.live_data_present ? '실시간' : '로컬';
      const stats = document.getElementById('diagnosticStats');
      if (stats) {
        stats.innerHTML = `
          <div class="stat"><strong>${liveLabel}</strong><span class="muted">자료 모드</span></div>
          <div class="stat"><strong>${d.live_source_count || 0}</strong><span class="muted">실시간 출처</span></div>
          <div class="stat"><strong>${d.external_chart_sources_configured || 0}</strong><span class="muted">차트 출처</span></div>
          <div class="stat"><strong>${data.graph_triples_count || 0}</strong><span class="muted">온톨로지 관계</span></div>
        `;
      }
      const store = data.store_summary || {};
      const storeStats = document.getElementById('storeStats');
      if (storeStats) {
        storeStats.innerHTML = `
          <div class="stat"><strong>${store.events || 0}</strong><span class="muted">이벤트</span></div>
          <div class="stat"><strong>${store.raw_records || 0}</strong><span class="muted">원문</span></div>
          <div class="stat"><strong>${store.market_snapshots || 0}</strong><span class="muted">시세</span></div>
          <div class="stat"><strong>${store.realtime_quotes || 0}</strong><span class="muted">실시간 quote</span></div>
        `;
      }
      const warnings = [...(d.collection_warnings || []), ...(data.skipped_sources || [])].slice(0, 4);
      const warningList = document.getElementById('collectionWarnings');
      if (warningList) warningList.innerHTML = warnings.map((item) => `<div class="warning-item">${String(item)}</div>`).join('');
      renderDataVolume(data.data_volume || {});
    }
    function renderDataVolume(volume = {}) {
      drawDataVolumeChart(volume.by_kind || {});
      const list = document.getElementById('sourceVolumeList');
      if (!list) return;
      const rows = Object.entries(volume.market_snapshot_sources || {}).slice(0, 8);
      list.innerHTML = rows.length
        ? rows.map(([name, count]) => `<div class="source-volume-row"><span>${name}</span><strong>${count}</strong></div>`).join('')
        : '<div class="muted">출처별 데이터 없음</div>';
    }

    function drawDataVolumeChart(byKind = {}) {
      const canvas = document.getElementById('dataVolumeChart');
      if (!canvas || !canvas.getContext) return;
      const ctx = canvas.getContext('2d');
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, width, height);
      const items = [
        ['events', '이벤트', '#0f766e'],
        ['raw_records', '원문', '#2563eb'],
        ['market_snapshots', '시세', '#f59e0b'],
        ['macro_metrics', '매크로', '#7c3aed'],
        ['realtime_quotes', 'Quote', '#0891b2'],
        ['realtime_executions', '체결', '#dc2626'],
      ];
      const maxValue = Math.max(1, ...items.map(([key]) => Number(byKind[key] || 0)));
      const barWidth = Math.max(24, Math.floor((width - 36) / items.length) - 10);
      items.forEach(([key, label, color], index) => {
        const value = Number(byKind[key] || 0);
        const h = Math.max(4, Math.round((height - 44) * value / maxValue));
        const x = 18 + index * (barWidth + 10);
        const y = height - 24 - h;
        ctx.fillStyle = color;
        ctx.fillRect(x, y, barWidth, h);
        ctx.fillStyle = '#334155';
        ctx.font = '11px Arial';
        ctx.fillText(label, x, height - 8);
        ctx.fillText(String(value), x, Math.max(12, y - 4));
      });
    }
    async function renderOntologyGraph(data) {
      const canvas = document.getElementById('ontologyCanvas');
      const tooltip = document.getElementById('ontologyTooltip');
      const THREE = await loadThree();
      if (!THREE) {
        renderOntologyGraph2d(data, canvas, tooltip);
        return;
      }

      if (graphState) {
        graphState.stop = true;
        if (graphState.cleanup) graphState.cleanup();
        if (graphState.renderer) graphState.renderer.dispose();
      }

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x050914);
      scene.fog = new THREE.FogExp2(0x050914, 0.00048);
      const camera = new THREE.PerspectiveCamera(52, 1, 0.1, 5000);
      camera.position.set(0, 0, 760);
      let renderer;
      try {
        renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: 'high-performance' });
      } catch (error) {
        console.warn('WebGL unavailable; using 2D graph fallback.', error);
        renderOntologyGraph2d(data, canvas, tooltip);
        return;
      }
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.18;

      const root = new THREE.Group();
      scene.add(root);
      scene.add(new THREE.HemisphereLight(0xbfe8ff, 0x11152b, 1.35));
      const light = new THREE.PointLight(0x8fdcff, 75, 1800);
      light.position.set(300, 280, 500);
      scene.add(light);
      const rimLight = new THREE.PointLight(0xa855f7, 55, 1500);
      rimLight.position.set(-420, -220, -280);
      scene.add(rimLight);

      // A sparse star field gives the rotating graph stable depth cues without
      // competing with the actual GNN edges.
      const starPositions = [];
      for (let index = 0; index < 900; index += 1) {
        const radius = 720 + seededUnit(`star:${index}:r`) * 1200;
        const theta = seededUnit(`star:${index}:t`) * Math.PI * 2;
        const phi = Math.acos(2 * seededUnit(`star:${index}:p`) - 1);
        starPositions.push(
          radius * Math.sin(phi) * Math.cos(theta),
          radius * Math.sin(phi) * Math.sin(theta),
          radius * Math.cos(phi),
        );
      }
      const starGeometry = new THREE.BufferGeometry();
      starGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starPositions, 3));
      scene.add(new THREE.Points(starGeometry, new THREE.PointsMaterial({
        color: 0x8db9dc,
        size: 1.4,
        transparent: true,
        opacity: 0.42,
        depthWrite: false,
      })));

      const renderGraph = prepareRenderableGraph(data.nodes || [], data.links || []);
      const nodes = computeGraphLayout(renderGraph.nodes, renderGraph.links);
      const graphMetrics = buildGraphMetrics(nodes, renderGraph.links);
      document.getElementById('ontologyCounts').textContent =
        `노드 ${data.counts.nodes} · 관계 ${data.counts.links} · 표시 ${nodes.length}/${renderGraph.links.length}`;
      const nodeMap = new Map(nodes.map((node) => [node.id, node]));
      const raycaster = new THREE.Raycaster();
      const pointer = new THREE.Vector2(99, 99);
      const interactionController = new AbortController();
      const interactionSignal = interactionController.signal;
      const nodeMeshes = [];
      const labelSprites = [];
      const linkLines = [];
      const linkGlowLines = [];
      const nodeMeshById = new Map();
      const nodeGlowById = new Map();
      const labelById = new Map();
      const lineByKey = new Map();
      const labelState = { visible: false };
      const reasoningState = {
        steps: (((data.live_trace || {}).stages) || []).filter((step) => (step.nodes || []).some((id) => nodeMap.has(id))),
        playing: false,
        followLive: true,
        currentIndex: -1,
        activeNodeIds: new Set(),
        activeLinkKeys: new Set(),
      };
      const activeKinds = new Set(['ticker', 'event', 'temporal', 'sector', 'support', 'risk', 'contradiction', 'pipeline', 'tuning', 'parameter', 'metric', 'entity']);

      for (const link of renderGraph.links) {
        const source = nodeMap.get(link.source);
        const target = nodeMap.get(link.target);
        if (!source || !target) continue;
        const geometry = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(...source.position),
          new THREE.Vector3(...target.position),
        ]);
        const material = new THREE.LineBasicMaterial({
          color: edgeColor(link.predicate),
          transparent: true,
          opacity: 0.42,
          depthWrite: false,
        });
        const line = new THREE.Line(geometry, material);
        line.userData = { source: link.source, target: link.target, predicate: link.predicate };
        line.userData.baseColor = edgeColor(link.predicate);
        line.userData.baseOpacity = 0.42;
        root.add(line);
        linkLines.push(line);
        lineByKey.set(linkKey(link.source, link.target, link.predicate), line);
        const glowMaterial = new THREE.LineBasicMaterial({
          color: 0x67e8f9,
          transparent: true,
          opacity: 0,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
        });
        const glowLine = new THREE.Line(geometry.clone(), glowMaterial);
        glowLine.visible = false;
        glowLine.userData = { source: link.source, target: link.target, predicate: link.predicate };
        root.add(glowLine);
        linkGlowLines.push(glowLine);
      }

      for (const node of nodes) {
        const radius = nodeRadius(node, graphMetrics);
        const sphereSegments = nodes.length > 500 ? 8 : nodes.length > 250 ? 12 : 20;
        const geometry = new THREE.SphereGeometry(radius, sphereSegments, sphereSegments);
        const highlighted = Boolean(node.highlight);
        const material = new THREE.MeshStandardMaterial({
          color: nodeColor(node.kind),
          emissive: nodeColor(node.kind),
          emissiveIntensity: highlighted ? 0.34 : 0.08,
          roughness: 0.34,
          metalness: 0.12,
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(...node.position);
        mesh.userData = node;
        mesh.userData.baseRadius = radius;
        mesh.userData.baseEmissiveIntensity = highlighted ? 0.34 : 0.08;
        root.add(mesh);
        nodeMeshes.push(mesh);
        nodeMeshById.set(node.id, mesh);
        const glowGeometry = new THREE.SphereGeometry(radius * 2.35, 18, 18);
        const glowMaterial = new THREE.MeshBasicMaterial({
          color: nodeColor(node.kind),
          transparent: true,
          opacity: 0,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
        });
        const glow = new THREE.Mesh(glowGeometry, glowMaterial);
        glow.position.copy(mesh.position);
        glow.visible = highlighted;
        glow.userData = node;
        root.add(glow);
        nodeGlowById.set(node.id, glow);
        const label = createTextSprite(THREE, shortLabel(node.label), nodeColor(node.kind));
        label.position.set(node.position[0] + 12, node.position[1] + 12, node.position[2]);
        label.visible = false;
        label.userData = node;
        root.add(label);
        labelSprites.push(label);
        labelById.set(node.id, label);
      }

      let dragging = false;
      let lastX = 0;
      let lastY = 0;
      let rotationX = -0.18;
      let rotationY = 0.34;
      let targetZoom = 760;
      let pausedUntil = 0;
      let visibleCenter = new THREE.Vector3(0, 0, 0);
      const nodePhysics = {
        cellSize: 86,
        homePull: 0.018,
        damping: 0.82,
        push: 2.25,
        maxVelocity: 6.2,
        velocities: new Map(),
        homePositions: new Map(),
      };
      for (const mesh of nodeMeshes) {
        nodePhysics.velocities.set(mesh.userData.id, new THREE.Vector3());
        nodePhysics.homePositions.set(mesh.userData.id, mesh.position.clone());
      }

      function resize() {
        const rect = canvas.getBoundingClientRect();
        renderer.setSize(rect.width, rect.height, false);
        camera.aspect = rect.width / Math.max(1, rect.height);
        camera.updateProjectionMatrix();
      }

      function updatePointer(event) {
        const rect = canvas.getBoundingClientRect();
        pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      }

      canvas.addEventListener('pointerdown', (event) => {
        updatePointer(event);
        dragging = true;
        lastX = event.clientX;
        lastY = event.clientY;
        pausedUntil = performance.now() + 2500;
        canvas.setPointerCapture(event.pointerId);
      }, { signal: interactionSignal });
      canvas.addEventListener('pointermove', (event) => {
        updatePointer(event);
        if (!dragging) return;
        rotationY += (event.clientX - lastX) * 0.008;
        rotationX += (event.clientY - lastY) * 0.008;
        lastX = event.clientX;
        lastY = event.clientY;
      }, { signal: interactionSignal });
      canvas.addEventListener('pointerup', () => { dragging = false; }, { signal: interactionSignal });
      canvas.addEventListener('wheel', (event) => {
        event.preventDefault();
        targetZoom = Math.max(260, Math.min(1300, targetZoom + event.deltaY * 0.7));
        pausedUntil = performance.now() + 2200;
      }, { passive: false, signal: interactionSignal });
      document.getElementById('resetGraph').onclick = () => {
        rotationX = -0.18;
        rotationY = 0.34;
        targetZoom = 760;
        fitVisibleGraph();
      };
      document.getElementById('toggleLabels').onclick = () => {
        labelState.visible = !labelState.visible;
        document.getElementById('toggleLabels').textContent = labelState.visible ? '라벨 끄기' : '라벨 켜기';
        updateVisibility(false);
      };
      document.getElementById('toggleReasoning').onclick = () => {
        reasoningState.followLive = !reasoningState.followLive;
        document.getElementById('toggleReasoning').textContent = reasoningState.followLive ? '실시간 추적 중' : '화면 고정됨';
      };
      document.querySelectorAll('#ontologyFilters input').forEach((input) => {
        input.onchange = () => {
          if (input.checked) activeKinds.add(input.value);
          else activeKinds.delete(input.value);
          updateVisibility();
        };
      });

      canvas.addEventListener('click', (event) => {
        updatePointer(event);
        raycaster.setFromCamera(pointer, camera);
        const hit = raycaster.intersectObjects(nodeMeshes.filter((mesh) => mesh.visible), false)[0];
        if (hit) renderNodePanel(hit.object.userData, data.links);
      }, { signal: interactionSignal });

      function updateVisibility(refit = true) {
        let visibleCount = 0;
        let visibleLinkCount = 0;
        for (const mesh of nodeMeshes) {
          mesh.visible = activeKinds.has(mesh.userData.kind);
          if (mesh.visible) visibleCount += 1;
        }
        for (const glow of nodeGlowById.values()) {
          glow.visible = (glow.userData.highlight || reasoningState.activeNodeIds.has(glow.userData.id)) && activeKinds.has(glow.userData.kind);
        }
        for (const sprite of labelSprites) {
          sprite.visible = (labelState.visible || reasoningState.activeNodeIds.has(sprite.userData.id)) && activeKinds.has(sprite.userData.kind);
        }
        for (const line of linkLines) {
          const source = nodeMap.get(line.userData.source);
          const target = nodeMap.get(line.userData.target);
          line.visible = Boolean(source && target && activeKinds.has(source.kind) && activeKinds.has(target.kind));
          if (line.visible) visibleLinkCount += 1;
        }
        for (const line of linkGlowLines) {
          const source = nodeMap.get(line.userData.source);
          const target = nodeMap.get(line.userData.target);
          line.visible = Boolean(
            source
            && target
            && activeKinds.has(source.kind)
            && activeKinds.has(target.kind)
            && reasoningState.activeLinkKeys.has(linkKey(line.userData.source, line.userData.target, line.userData.predicate))
          );
        }
        if (refit) fitVisibleGraph();
        document.getElementById('ontologyCounts').textContent =
          `전체 노드 ${data.counts.nodes} · 전체 연결 ${data.counts.links} · 현재 ${visibleCount}/${visibleLinkCount}`;
      }

      function fitVisibleGraph() {
        const visibleMeshes = nodeMeshes.filter((mesh) => activeKinds.has(mesh.userData.kind));
        if (!visibleMeshes.length) {
          visibleCenter.set(0, 0, 0);
          return;
        }
        const center = new THREE.Vector3(0, 0, 0);
        let maxDistance = 1;
        for (const mesh of visibleMeshes) center.add(mesh.position);
        center.multiplyScalar(1 / visibleMeshes.length);
        for (const mesh of visibleMeshes) maxDistance = Math.max(maxDistance, mesh.position.distanceTo(center));
        visibleCenter.copy(center);
        targetZoom = Math.max(260, Math.min(1300, maxDistance * 2.35 + 280));
      }

      function gridKeyFor(position) {
        const cell = nodePhysics.cellSize;
        return `${Math.floor(position.x / cell)},${Math.floor(position.y / cell)},${Math.floor(position.z / cell)}`;
      }

      function applyNodeProximityRepulsion() {
        const visibleMeshes = nodeMeshes.filter((mesh) => mesh.visible);
        if (visibleMeshes.length < 2) return;
        const grid = new Map();
        for (const mesh of visibleMeshes) {
          const key = gridKeyFor(mesh.position);
          if (!grid.has(key)) grid.set(key, []);
          grid.get(key).push(mesh);
        }

        for (let index = 0; index < visibleMeshes.length; index += 1) {
          const mesh = visibleMeshes[index];
          const velocity = nodePhysics.velocities.get(mesh.userData.id);
          const home = nodePhysics.homePositions.get(mesh.userData.id);
          if (!velocity || !home) continue;

          velocity.addScaledVector(home.clone().sub(mesh.position), nodePhysics.homePull);

          const cx = Math.floor(mesh.position.x / nodePhysics.cellSize);
          const cy = Math.floor(mesh.position.y / nodePhysics.cellSize);
          const cz = Math.floor(mesh.position.z / nodePhysics.cellSize);
          for (let ox = -1; ox <= 1; ox += 1) {
            for (let oy = -1; oy <= 1; oy += 1) {
              for (let oz = -1; oz <= 1; oz += 1) {
                const bucket = grid.get(`${cx + ox},${cy + oy},${cz + oz}`);
                if (!bucket) continue;
                for (const other of bucket) {
                  if (other === mesh || other.userData.index <= mesh.userData.index) continue;
                  const otherVelocity = nodePhysics.velocities.get(other.userData.id);
                  if (!otherVelocity) continue;
                  const minDistance = mesh.userData.baseRadius + other.userData.baseRadius + 34;
                  const dx = mesh.position.x - other.position.x;
                  const dy = mesh.position.y - other.position.y;
                  const dz = mesh.position.z - other.position.z;
                  const distanceSq = dx * dx + dy * dy + dz * dz;
                  if (distanceSq >= minDistance * minDistance) continue;
                  const distance = Math.sqrt(distanceSq) || 0.001;
                  const pressure = (1 - distance / minDistance) * nodePhysics.push;
                  const nx = dx / distance;
                  const ny = dy / distance;
                  const nz = dz / distance;
                  velocity.x += nx * pressure;
                  velocity.y += ny * pressure;
                  velocity.z += nz * pressure;
                  otherVelocity.x -= nx * pressure;
                  otherVelocity.y -= ny * pressure;
                  otherVelocity.z -= nz * pressure;
                }
              }
            }
          }
        }

        for (const mesh of visibleMeshes) {
          const velocity = nodePhysics.velocities.get(mesh.userData.id);
          if (!velocity) continue;
          velocity.multiplyScalar(nodePhysics.damping);
          if (velocity.length() > nodePhysics.maxVelocity) velocity.setLength(nodePhysics.maxVelocity);
          mesh.position.add(velocity);
        }
      }

      function syncGraphGeometryPositions() {
        for (const glow of nodeGlowById.values()) {
          const mesh = nodeMeshById.get(glow.userData.id);
          if (mesh) glow.position.copy(mesh.position);
        }
        for (const label of labelSprites) {
          const mesh = nodeMeshById.get(label.userData.id);
          if (mesh) label.position.set(mesh.position.x + 12, mesh.position.y + 12, mesh.position.z);
        }
        for (const line of linkLines) updateLineGeometry(line);
        for (const line of linkGlowLines) updateLineGeometry(line);
      }

      function updateLineGeometry(line) {
        const source = nodeMeshById.get(line.userData.source);
        const target = nodeMeshById.get(line.userData.target);
        if (!source || !target) return;
        const positions = line.geometry.attributes.position;
        positions.setXYZ(0, source.position.x, source.position.y, source.position.z);
        positions.setXYZ(1, target.position.x, target.position.y, target.position.z);
        positions.needsUpdate = true;
        line.geometry.computeBoundingSphere();
      }

      function updateReasoning(now) {
        if (!reasoningState.steps.length) {
          document.getElementById('reasoningBadge').textContent = '실제 추론 대기 중';
          return;
        }
      }

      function setActiveReasoningStep(index) {
        if (index < 0 || index >= reasoningState.steps.length) return;
        reasoningState.currentIndex = index;
        const step = reasoningState.steps[index];
        reasoningState.activeNodeIds = new Set(step.nodes || []);
        reasoningState.activeLinkKeys = new Set((step.links || []).map((link) => linkKey(link.source, link.target, link.predicate)));
        document.getElementById('reasoningBadge').textContent = `실제 단계 ${index + 1}/${reasoningState.steps.length}`;
        document.getElementById('reasoningTitle').textContent = step.title || '추론 단계';
        document.getElementById('reasoningMeta').textContent = `${step.ticker || '-'} · ${formatLiveTraceTime(step.observed_at)}`;
        document.getElementById('reasoningDescription').textContent = step.description || '';
        document.getElementById('reasoningProgress').style.width = `${((index + 1) / reasoningState.steps.length) * 100}%`;
        updateVisibility(false);
      }

      function applyReasoningGlow(now) {
        const pulse = 0.62 + Math.sin(now / 135) * 0.38;
        for (const mesh of nodeMeshes) {
          const active = reasoningState.activeNodeIds.has(mesh.userData.id);
          mesh.scale.setScalar(active ? 1.12 + pulse * 0.22 : 1);
          mesh.material.emissiveIntensity = active ? 1.35 + pulse * 0.95 : mesh.userData.baseEmissiveIntensity;
          mesh.material.color.setHex(active ? neonColor(mesh.userData.kind) : nodeColor(mesh.userData.kind));
        }
        for (const glow of nodeGlowById.values()) {
          const active = reasoningState.activeNodeIds.has(glow.userData.id);
          const highlighted = Boolean(glow.userData.highlight);
          glow.visible = (active || highlighted) && activeKinds.has(glow.userData.kind);
          glow.scale.setScalar(active ? 1.05 + pulse * 0.35 : 1);
          glow.material.opacity = active ? 0.22 + pulse * 0.18 : highlighted ? 0.12 + pulse * 0.08 : 0;
          glow.material.color.setHex(neonColor(glow.userData.kind));
        }
        for (const line of linkLines) {
          const active = reasoningState.activeLinkKeys.has(linkKey(line.userData.source, line.userData.target, line.userData.predicate));
          line.material.opacity = active ? 1 : line.userData.baseOpacity;
          line.material.color.setHex(active ? neonEdgeColor(line.userData.predicate) : line.userData.baseColor);
        }
        for (const line of linkGlowLines) {
          const active = reasoningState.activeLinkKeys.has(linkKey(line.userData.source, line.userData.target, line.userData.predicate));
          const source = nodeMap.get(line.userData.source);
          const target = nodeMap.get(line.userData.target);
          line.visible = Boolean(active && source && target && activeKinds.has(source.kind) && activeKinds.has(target.kind));
          line.material.opacity = active ? 0.42 + pulse * 0.28 : 0;
          line.material.color.setHex(neonEdgeColor(line.userData.predicate));
        }
        for (const [nodeId, label] of labelById.entries()) {
          if (reasoningState.activeNodeIds.has(nodeId)) {
            label.material.opacity = 1;
          }
        }
      }

      function applyLiveTrace(trace) {
        if (!reasoningState.followLive || !trace) return;
        const steps = (trace.stages || []).filter((step) => (step.nodes || []).some((id) => nodeMap.has(id)));
        reasoningState.steps = steps;
        if (steps.length) setActiveReasoningStep(steps.length - 1);
      }

      const cleanup = () => {
        interactionController.abort();
        window.removeEventListener('resize', resize);
        scene.traverse((object) => {
          if (object.geometry) object.geometry.dispose();
          const materials = Array.isArray(object.material) ? object.material : object.material ? [object.material] : [];
          for (const material of materials) {
            if (material.map) material.map.dispose();
            material.dispose();
          }
        });
      };
      graphState = { renderer, stop: false, cleanup, applyLiveTrace };
      resize();
      window.addEventListener('resize', resize);
      document.getElementById('toggleReasoning').textContent = '실시간 추적 중';
      if (reasoningState.steps.length) setActiveReasoningStep(reasoningState.steps.length - 1);

      function animate(now) {
        if (!graphState || graphState.stop) return;
        requestAnimationFrame(animate);
        if (!dragging && now > pausedUntil) rotationY += 0.0022;
        root.rotation.x = rotationX;
        root.rotation.y = rotationY;
        root.position.x += (-visibleCenter.x - root.position.x) * 0.08;
        root.position.y += (-visibleCenter.y - root.position.y) * 0.08;
        root.position.z += (-visibleCenter.z - root.position.z) * 0.08;
        camera.position.z += (targetZoom - camera.position.z) * 0.08;
        updateReasoning(now);
        applyNodeProximityRepulsion();
        syncGraphGeometryPositions();
        applyReasoningGlow(now);

        raycaster.setFromCamera(pointer, camera);
        const hit = raycaster.intersectObjects(nodeMeshes.filter((mesh) => mesh.visible), false)[0];
        if (hit) {
          const rect = canvas.getBoundingClientRect();
          tooltip.style.display = 'block';
          tooltip.style.left = `${Math.min(rect.width - 280, Math.max(8, (pointer.x + 1) * rect.width / 2))}px`;
          tooltip.style.top = `${Math.min(rect.height - 80, Math.max(50, (-pointer.y + 1) * rect.height / 2))}px`;
          tooltip.innerHTML = `<strong>${hit.object.userData.label}</strong><br>${kindLabel(hit.object.userData.kind)} · 연결 ${degree(hit.object.userData.id, renderGraph.links)}개 · 중요도 ${Number(hit.object.userData.importance_score || 0).toFixed(2)}`;
        } else {
          tooltip.style.display = 'none';
        }

        renderer.render(scene, camera);
      }
      requestAnimationFrame(animate);
      updateVisibility();
    }

    function renderOntologyGraph2d(data, canvas, tooltip) {
      if (graphState) {
        graphState.stop = true;
        if (graphState.cleanup) graphState.cleanup();
        if (graphState.renderer) graphState.renderer.dispose();
      }
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const renderGraph = prepareRenderableGraph(data.nodes || [], data.links || []);
      const nodes = computeSemanticLayout(
        renderGraph.nodes.map((node, index) => ({ ...node, index })),
        renderGraph.links
      );
      const graphMetrics = buildGraphMetrics(nodes, renderGraph.links);
      const nodeMap = new Map(nodes.map((node) => [node.id, node]));
      const activeKinds = new Set(['ticker', 'event', 'temporal', 'sector', 'support', 'risk', 'contradiction', 'pipeline', 'tuning', 'parameter', 'metric', 'entity']);

      // adjacency for Obsidian-style hover neighbour highlighting
      const adj = new Map(nodes.map((node) => [node.id, new Set()]));
      for (const link of renderGraph.links) {
        if (adj.has(link.source) && adj.has(link.target)) {
          adj.get(link.source).add(link.target);
          adj.get(link.target).add(link.source);
        }
      }
      // seed live-simulation coordinates from the precomputed layout
      nodes.forEach((node, i) => {
        const p = node.position || [];
        node.x = Number.isFinite(p[0]) ? p[0] : Math.cos(i * 1.7) * 170;
        node.y = Number.isFinite(p[1]) ? p[1] : Math.sin(i * 1.7) * 170;
        node.vx = 0; node.vy = 0; node.fixed = false;
      });

      const reasoningState = {
        steps: (((data.live_trace || {}).stages) || []).filter((step) => (step.nodes || []).some((id) => nodeMap.has(id))),
        playing: false,
        followLive: true,
        currentIndex: -1,
        activeNodeIds: new Set(),
        activeLinkKeys: new Set(),
      };
      const view = { scale: 1, tx: 0, ty: 0, labels: false, pointerX: -9999, pointerY: -9999 };
      const sim = { repel: 6500, linkLen: 84 };
      let hoveredNode = null, selectedId = null, frozen = false;
      // Large graphs are seeded from the server layout, so relax gently (low alpha)
      // to avoid an expensive O(n^2) churn on modest devices; small graphs settle fully.
      let alpha = nodes.length > 260 ? 0.45 : 1;
      const reheat = (a) => { alpha = Math.max(alpha, a); };
      if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) alpha = 0.3;

      document.getElementById('ontologyCounts').textContent =
        `노드 ${data.counts.nodes} · 관계 ${data.counts.links} · 표시 ${nodes.length}/${renderGraph.links.length}`;

      function resize() {
        const rect = canvas.getBoundingClientRect();
        const ratio = Math.min(window.devicePixelRatio || 1, 2);
        canvas.width = Math.max(1, Math.floor(rect.width * ratio));
        canvas.height = Math.max(1, Math.floor(rect.height * ratio));
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      }

      const worldToScreen = (x, y) => ({ x: x * view.scale + view.tx, y: y * view.scale + view.ty });
      const screenToWorld = (sx, sy) => ({ x: (sx - view.tx) / view.scale, y: (sy - view.ty) / view.scale });
      const visibleNode = (node) => activeKinds.has(node.kind);
      function screenRadius(node) {
        const base = nodeRadius(node, graphMetrics);
        return Math.max(2.4, base * Math.max(0.6, Math.min(view.scale, 1.8)));
      }

      function fitVisibleGraph2d() {
        const rect = canvas.getBoundingClientRect();
        const vis = nodes.filter(visibleNode);
        if (!vis.length) { view.scale = 1; view.tx = rect.width / 2; view.ty = rect.height / 2; return; }
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        vis.forEach((n) => { if (n.x < minX) minX = n.x; if (n.x > maxX) maxX = n.x; if (n.y < minY) minY = n.y; if (n.y > maxY) maxY = n.y; });
        const w = Math.max(1, maxX - minX), h = Math.max(1, maxY - minY);
        view.scale = Math.max(0.3, Math.min(2.6, Math.min(rect.width / (w + 150), rect.height / (h + 150))));
        view.tx = rect.width / 2 - ((minX + maxX) / 2) * view.scale;
        view.ty = rect.height / 2 - ((minY + maxY) / 2) * view.scale;
      }

      function pickNode(sx, sy) {
        let best = null, bd = Infinity;
        for (const node of nodes) {
          if (!visibleNode(node)) continue;
          const p = worldToScreen(node.x, node.y);
          const r = screenRadius(node) + 5;
          const dx = p.x - sx, dy = p.y - sy, d = dx * dx + dy * dy;
          if (d < r * r && d < bd) { bd = d; best = node; }
        }
        return best;
      }

      function setActiveReasoningStep(index) {
        if (index < 0 || index >= reasoningState.steps.length) return;
        reasoningState.currentIndex = index;
        const step = reasoningState.steps[index];
        reasoningState.activeNodeIds = new Set(step.nodes || []);
        reasoningState.activeLinkKeys = new Set((step.links || []).map((link) => linkKey(link.source, link.target, link.predicate)));
        document.getElementById('reasoningBadge').textContent = `실제 단계 ${index + 1}/${reasoningState.steps.length}`;
        document.getElementById('reasoningTitle').textContent = step.title || '추론 단계';
        document.getElementById('reasoningMeta').textContent = `${step.ticker || '-'} · ${formatLiveTraceTime(step.observed_at)}`;
        document.getElementById('reasoningDescription').textContent = step.description || '';
        document.getElementById('reasoningProgress').style.width = `${((index + 1) / reasoningState.steps.length) * 100}%`;
      }

      function updateReasoning(now) {
        if (!reasoningState.steps.length) {
          document.getElementById('reasoningBadge').textContent = '실제 추론 대기 중';
          return;
        }
      }

      function applyLiveTrace(trace) {
        if (!reasoningState.followLive || !trace) return;
        const steps = (trace.stages || []).filter((step) => (step.nodes || []).some((id) => nodeMap.has(id)));
        reasoningState.steps = steps;
        if (steps.length) setActiveReasoningStep(steps.length - 1);
      }

      function stepSim() {
        if (frozen || alpha < 0.02) return;
        const list = nodes.filter(visibleNode);
        for (let i = 0; i < list.length; i++) {
          const a = list[i];
          for (let j = i + 1; j < list.length; j++) {
            const b = list[j];
            let dx = a.x - b.x, dy = a.y - b.y, d2 = dx * dx + dy * dy;
            if (d2 < 0.01) { d2 = 0.01; dx = Math.random() - 0.5; dy = Math.random() - 0.5; }
            const d = Math.sqrt(d2), f = Math.min(sim.repel / d2, 42);
            const fx = dx / d * f, fy = dy / d * f;
            a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
          }
        }
        for (const link of renderGraph.links) {
          const a = nodeMap.get(link.source), b = nodeMap.get(link.target);
          if (!a || !b || !visibleNode(a) || !visibleNode(b)) continue;
          let dx = b.x - a.x, dy = b.y - a.y;
          const d = Math.sqrt(dx * dx + dy * dy) || 0.01, f = (d - sim.linkLen) * 0.045;
          const fx = dx / d * f, fy = dy / d * f;
          a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
        }
        for (const n of list) {
          n.vx += (-n.x) * 0.003; n.vy += (-n.y) * 0.003;
          if (n.fixed) { n.vx = 0; n.vy = 0; continue; }
          n.vx *= 0.86; n.vy *= 0.86;
          n.x += n.vx * alpha * 1.4; n.y += n.vy * alpha * 1.4;
        }
        alpha *= 0.992;
      }

      function draw(now) {
        if (!graphState || graphState.stop) return;
        stepSim();
        const rect = canvas.getBoundingClientRect();
        ctx.clearRect(0, 0, rect.width, rect.height);
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, rect.width, rect.height);
        updateReasoning(now);
        const pulse = 0.55 + Math.sin(now / 150) * 0.45;
        const focus = hoveredNode;
        const neigh = focus ? adj.get(focus.id) : null;
        const DIM = 0.12;

        for (const link of renderGraph.links) {
          const s = nodeMap.get(link.source), t = nodeMap.get(link.target);
          if (!s || !t || !visibleNode(s) || !visibleNode(t)) continue;
          const a = worldToScreen(s.x, s.y), b = worldToScreen(t.x, t.y);
          const reasonActive = reasoningState.activeLinkKeys.has(linkKey(link.source, link.target, link.predicate));
          const near = focus ? (focus === s || focus === t) : true;
          ctx.save();
          ctx.strokeStyle = intColorToCss(reasonActive ? neonEdgeColor(link.predicate) : edgeColor(link.predicate));
          ctx.globalAlpha = focus ? (near ? 0.85 : DIM) : (reasonActive ? 0.78 : 0.22);
          ctx.lineWidth = reasonActive ? 2.4 + pulse * 1.5 : (focus && near ? 1.6 : 0.8);
          ctx.shadowBlur = reasonActive ? 16 : 0;
          ctx.shadowColor = ctx.strokeStyle;
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
          ctx.restore();
        }

        for (const node of nodes) {
          if (!visibleNode(node)) continue;
          const p = worldToScreen(node.x, node.y);
          const reasonActive = reasoningState.activeNodeIds.has(node.id);
          const near = !focus || node === focus || (neigh && neigh.has(node.id));
          const radius = screenRadius(node) * (reasonActive ? 1.3 : 1);
          ctx.save();
          ctx.globalAlpha = focus ? (near ? 1 : DIM) : 0.95;
          ctx.fillStyle = intColorToCss(reasonActive ? neonColor(node.kind) : nodeColor(node.kind));
          ctx.shadowBlur = reasonActive ? 24 : (node === focus ? 14 : 0);
          ctx.shadowColor = intColorToCss(neonColor(node.kind));
          ctx.beginPath(); ctx.arc(p.x, p.y, radius, 0, Math.PI * 2); ctx.fill();
          if (node === focus || node.id === selectedId) {
            ctx.shadowBlur = 0; ctx.lineWidth = 2; ctx.strokeStyle = '#e5e7eb'; ctx.stroke();
          }
          const showLabel = view.labels || reasonActive || (focus && near) || view.scale > 1.4;
          if (showLabel) {
            ctx.shadowBlur = 0;
            ctx.globalAlpha = focus ? (near ? 1 : DIM) : 0.95;
            ctx.fillStyle = '#e5e7eb';
            ctx.font = '12px Arial';
            ctx.textBaseline = 'top';
            ctx.fillText(shortLabel(node.label), p.x + radius + 4, p.y - radius - 3);
          }
          ctx.restore();
        }

        if (hoveredNode) {
          const p = worldToScreen(hoveredNode.x, hoveredNode.y);
          tooltip.style.display = 'block';
          tooltip.style.left = `${Math.min(rect.width - 280, Math.max(8, p.x + 12))}px`;
          tooltip.style.top = `${Math.min(rect.height - 80, Math.max(50, p.y + 12))}px`;
          tooltip.innerHTML = `<strong>${hoveredNode.label}</strong><br>${kindLabel(hoveredNode.kind)} · 연결 ${degree(hoveredNode.id, renderGraph.links)}개 · 중요도 ${Number(hoveredNode.importance_score || 0).toFixed(2)}`;
        } else {
          tooltip.style.display = 'none';
        }

        requestAnimationFrame(draw);
      }

      let dragNode = null, panning = false, lastX = 0, lastY = 0, moved = false;
      canvas.onpointerdown = (event) => {
        const rect = canvas.getBoundingClientRect();
        const sx = event.clientX - rect.left, sy = event.clientY - rect.top;
        moved = false; lastX = event.clientX; lastY = event.clientY;
        const n = pickNode(sx, sy);
        if (n) { dragNode = n; n.fixed = true; reheat(0.5); } else { panning = true; }
        canvas.setPointerCapture(event.pointerId);
      };
      canvas.onpointermove = (event) => {
        const rect = canvas.getBoundingClientRect();
        const sx = event.clientX - rect.left, sy = event.clientY - rect.top;
        view.pointerX = sx; view.pointerY = sy;
        if (dragNode) {
          const w = screenToWorld(sx, sy);
          dragNode.x = w.x; dragNode.y = w.y; dragNode.vx = 0; dragNode.vy = 0;
          moved = true; reheat(0.4); return;
        }
        if (panning) {
          view.tx += event.clientX - lastX; view.ty += event.clientY - lastY;
          lastX = event.clientX; lastY = event.clientY; moved = true; return;
        }
        hoveredNode = pickNode(sx, sy);
        canvas.style.cursor = hoveredNode ? 'pointer' : 'grab';
      };
      canvas.onpointerup = () => {
        if (dragNode) dragNode.fixed = false;
        if (!moved && hoveredNode) { selectedId = hoveredNode.id; renderNodePanel(hoveredNode, data.links); }
        dragNode = null; panning = false;
      };
      canvas.onpointerleave = () => { view.pointerX = -9999; view.pointerY = -9999; hoveredNode = null; };
      canvas.onwheel = (event) => {
        event.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const sx = event.clientX - rect.left, sy = event.clientY - rect.top;
        const f = event.deltaY > 0 ? 0.9 : 1.1;
        const ns = Math.max(0.3, Math.min(4, view.scale * f));
        view.tx = sx - (sx - view.tx) * (ns / view.scale);
        view.ty = sy - (sy - view.ty) * (ns / view.scale);
        view.scale = ns;
      };
      document.getElementById('resetGraph').onclick = () => { fitVisibleGraph2d(); reheat(0.6); };
      document.getElementById('toggleLabels').onclick = () => {
        view.labels = !view.labels;
        document.getElementById('toggleLabels').textContent = view.labels ? '라벨 끄기' : '라벨 켜기';
      };
      document.getElementById('toggleReasoning').onclick = () => {
        reasoningState.followLive = !reasoningState.followLive;
        document.getElementById('toggleReasoning').textContent = reasoningState.followLive ? '실시간 추적 중' : '화면 고정됨';
      };
      document.querySelectorAll('#ontologyFilters input').forEach((input) => {
        input.onchange = () => {
          if (input.checked) activeKinds.add(input.value);
          else activeKinds.delete(input.value);
          reheat(0.5);
        };
      });

      const cleanup = () => window.removeEventListener('resize', resize);
      graphState = { stop: false, renderer: null, cleanup, applyLiveTrace };
      resize();
      window.addEventListener('resize', resize, { passive: true });
      fitVisibleGraph2d();
      document.getElementById('toggleReasoning').textContent = '실시간 추적 중';
      if (reasoningState.steps.length) setActiveReasoningStep(reasoningState.steps.length - 1);
      requestAnimationFrame(draw);
    }
    function intColorToCss(value) {
      return `#${Number(value || 0).toString(16).padStart(6, '0')}`;
    }

    async function loadThree() {
      if (window.__threeModule) return window.__threeModule;
      try {
        window.__threeModule = await import('https://unpkg.com/three@0.165.0/build/three.module.js');
        return window.__threeModule;
      } catch (error) {
        console.error(error);
        return null;
      }
    }

    function prepareRenderableGraph(rawNodes, rawLinks) {
      const kindOverrides = inferRenderableKinds(rawNodes, rawLinks);
      const normalizedNodes = rawNodes.map((node) => kindOverrides.has(node.id) ? { ...node, kind: kindOverrides.get(node.id) } : node);
      const degreeMap = new Map(normalizedNodes.map((node) => [node.id, 0]));
      for (const link of rawLinks) {
        const boost = importantPredicate(link.predicate) ? 7 : 1;
        degreeMap.set(link.source, (degreeMap.get(link.source) || 0) + boost);
        degreeMap.set(link.target, (degreeMap.get(link.target) || 0) + boost);
      }
      const scoreNode = (node) => Number(node.importance_score || 0) + (degreeMap.get(node.id) || 0) * 0.02 + (node.highlight ? 3 : 0);
      const byScore = (a, b) => scoreNode(b) - scoreNode(a);
      const priorityKind = { support: 9, risk: 9, contradiction: 9, pipeline: 7.2, tuning: 7, parameter: 6.6, metric: 6, sector: 5.7, ticker: 5.2, event: 4.8, temporal: 4.4, entity: 1 };
      const nodes = normalizedNodes
        .slice()
        .sort((a, b) => {
          const kindScore = (priorityKind[b.kind] || 0) - (priorityKind[a.kind] || 0);
          if (kindScore) return kindScore;
          return byScore(a, b);
        });
      const selected = new Set(nodes.map((node) => node.id));
      const links = rawLinks
        .filter((link) => selected.has(link.source) && selected.has(link.target))
        .sort((a, b) => Number(importantPredicate(b.predicate)) - Number(importantPredicate(a.predicate)));
      return { nodes, links };
    }

    function inferRenderableKinds(rawNodes, rawLinks) {
      const fixedKinds = new Set(['ticker', 'event', 'temporal', 'pipeline', 'tuning', 'parameter', 'metric', 'sector']);
      const nodeById = new Map(rawNodes.map((node) => [node.id, node]));
      const overrides = new Map();
      const assign = (id, kind) => {
        const node = nodeById.get(id);
        if (!node || fixedKinds.has(node.kind)) return;
        if (overrides.get(id) === 'risk') return;
        if (kind === 'risk' || !overrides.has(id)) overrides.set(id, kind);
      };
      for (const link of rawLinks) {
        if (link.predicate === 'supportsSignal' || link.predicate === 'decreasesRiskOf') {
          assign(link.source, 'support');
          assign(link.target, 'support');
        } else if (link.predicate === 'increasesRiskOf') {
          assign(link.source, 'risk');
          assign(link.target, 'risk');
        } else if (link.predicate === 'contradictsSignal') {
          assign(link.source, 'contradiction');
          assign(link.target, 'contradiction');
        }
      }
      return overrides;
    }

    function importantPredicate(predicate) {
      return [
        'supportsSignal',
        'increasesRiskOf',
        'contradictsSignal',
        'decreasesRiskOf',
        'generatesSemanticFeature',
        'hasTechnicalIndicator',
        'hasRecentNews',
        'hasRecentDisclosure',
        'selectsCandidate',
        'feedsStage',
        'tunesParameter',
        'hasTunedValue',
        'containsFrame',
        'hasTimeFrame',
        'observesTicker',
        'containsEvent',
        'occursInTimeBucket',
        'usesMarketSnapshot',
        'containsQuote',
        'containsExecution',
        'usesRawSource',
        'hasMacroContext',
        'hasImpactScore',
        'hasTuningMode',
        'adjustsStage',
        'producesTunedValue',
        'appliesToStage',
        'usesOntologySignal',
        'calibratesSignal',
        'raisesTuningPressure',
        'requiresApprovalFrom',
        'observedUniverseCount',
        'selectedCandidateCount',
        'fetchesChartsFor',
      ].includes(predicate);
    }

    function computeGraphLayout(rawNodes, rawLinks) {
      const nodes = rawNodes.map((node, index) => ({ ...node, index }));
      if (!nodes.length) return nodes;
      if (nodes.length > 240) return computeFastClusterLayout(nodes, rawLinks);

      const nodeMap = new Map(nodes.map((node) => [node.id, node]));
      const links = rawLinks.filter((link) => nodeMap.has(link.source) && nodeMap.has(link.target));
      const adjacency = new Map(nodes.map((node) => [node.id, new Set()]));
      const degreeMap = new Map(nodes.map((node) => [node.id, 0]));
      for (const link of links) {
        adjacency.get(link.source).add(link.target);
        adjacency.get(link.target).add(link.source);
        degreeMap.set(link.source, degreeMap.get(link.source) + 1);
        degreeMap.set(link.target, degreeMap.get(link.target) + 1);
      }

      const anchors = {
        ticker: [0, 0, 0],
        support: [-130, 95, 95],
        risk: [155, 90, -95],
        contradiction: [160, -85, 115],
        sector: [-165, -105, -100],
        entity: [0, -165, 120],
        event: [-15, 150, -125],
        temporal: [-95, -155, 20],
        pipeline: [145, -130, -35],
        tuning: [135, 20, 150],
        parameter: [205, -35, 125],
        metric: [215, 115, 25],
      };
      const positions = new Map();
      const velocities = new Map();

      nodes.forEach((node) => {
        const seed = seededUnit(node.id || node.label || String(node.index));
        const anchor = anchors[node.kind] || [0, 0, 0];
        const importance = Math.max(0, Math.min(1, Number(node.importance_score || 0)));
        const spread = node.kind === 'ticker' ? 95 : 185 - importance * 55;
        const angleA = seed * Math.PI * 2;
        const angleB = seededUnit(`${node.id}:z`) * Math.PI * 2;
        positions.set(node.id, {
          x: anchor[0] + Math.cos(angleA) * spread * (0.35 + seededUnit(`${node.id}:rx`) * 0.65),
          y: anchor[1] + Math.sin(angleA) * spread * (0.35 + seededUnit(`${node.id}:ry`) * 0.65),
          z: anchor[2] + Math.sin(angleB) * spread * (0.25 + seededUnit(`${node.id}:rz`) * 0.55),
        });
        velocities.set(node.id, { x: 0, y: 0, z: 0 });
      });

      const iterations = Math.max(45, Math.min(230, Math.floor(14000 / Math.max(1, nodes.length))));
      for (let iteration = 0; iteration < iterations; iteration += 1) {
        const cooling = 1 - iteration / iterations;
        for (let i = 0; i < nodes.length; i += 1) {
          const a = nodes[i];
          const pa = positions.get(a.id);
          const va = velocities.get(a.id);
          for (let j = i + 1; j < nodes.length; j += 1) {
            const b = nodes[j];
            const pb = positions.get(b.id);
            const vb = velocities.get(b.id);
            let dx = pa.x - pb.x;
            let dy = pa.y - pb.y;
            let dz = pa.z - pb.z;
            let distance = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.001;
            const connected = adjacency.get(a.id).has(b.id);
            const minDistance = connected ? 48 : 82;
            if (distance < minDistance) distance = minDistance;
            const repulsion = (connected ? 1200 : 4200) / (distance * distance);
            dx /= distance; dy /= distance; dz /= distance;
            va.x += dx * repulsion; va.y += dy * repulsion; va.z += dz * repulsion;
            vb.x -= dx * repulsion; vb.y -= dy * repulsion; vb.z -= dz * repulsion;
          }
        }

        for (const link of links) {
          const source = positions.get(link.source);
          const target = positions.get(link.target);
          const vs = velocities.get(link.source);
          const vt = velocities.get(link.target);
          const dx = target.x - source.x;
          const dy = target.y - source.y;
          const dz = target.z - source.z;
          const distance = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.001;
          const desired = linkLength(link.predicate);
          const force = (distance - desired) * (0.006 + cooling * 0.004) * linkStrength(link.predicate);
          const nx = dx / distance;
          const ny = dy / distance;
          const nz = dz / distance;
          vs.x += nx * force; vs.y += ny * force; vs.z += nz * force;
          vt.x -= nx * force; vt.y -= ny * force; vt.z -= nz * force;
        }

        for (const node of nodes) {
          const p = positions.get(node.id);
          const v = velocities.get(node.id);
          const anchor = anchors[node.kind] || [0, 0, 0];
          const degree = degreeMap.get(node.id) || 0;
          const anchorPull = node.kind === 'ticker' ? 0.004 : 0.0018;
          const centerPull = 0.0009 + Math.min(0.002, degree * 0.00018);
          v.x += (anchor[0] - p.x) * anchorPull - p.x * centerPull;
          v.y += (anchor[1] - p.y) * anchorPull - p.y * centerPull;
          v.z += (anchor[2] - p.z) * anchorPull - p.z * centerPull;
          v.x *= 0.78; v.y *= 0.78; v.z *= 0.78;
          p.x += v.x * (0.8 + cooling * 0.45);
          p.y += v.y * (0.8 + cooling * 0.45);
          p.z += v.z * (0.8 + cooling * 0.45);
        }
      }

      let cx = 0, cy = 0, cz = 0;
      for (const node of nodes) {
        const p = positions.get(node.id);
        cx += p.x; cy += p.y; cz += p.z;
      }
      cx /= nodes.length; cy /= nodes.length; cz /= nodes.length;

      let maxRadius = 1;
      for (const node of nodes) {
        const p = positions.get(node.id);
        p.x -= cx; p.y -= cy; p.z -= cz;
        maxRadius = Math.max(maxRadius, Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z));
      }
      const scale = Math.min(1.55, 340 / maxRadius);
      return nodes.map((node) => {
        const p = positions.get(node.id);
        return {
          ...node,
          position: [
            p.x * scale,
            p.y * scale,
            p.z * scale,
          ],
        };
      });
    }

    function computeSemanticLayout(nodes, rawLinks) {
      const nodeMap = new Map(nodes.map((node) => [node.id, node]));
      const links = (rawLinks || []).filter((link) => nodeMap.has(link.source) && nodeMap.has(link.target));
      const adjacency = new Map(nodes.map((node) => [node.id, new Set()]));
      const degreeMap = new Map(nodes.map((node) => [node.id, 0]));
      for (const link of links) {
        adjacency.get(link.source).add(link.target);
        adjacency.get(link.target).add(link.source);
        degreeMap.set(link.source, degreeMap.get(link.source) + 1);
        degreeMap.set(link.target, degreeMap.get(link.target) + 1);
      }

      const positions = new Map();
      const velocities = new Map();
      const semanticDrift = {
        ticker: [0, 0],
        event: [-120, -80],
        temporal: [-40, -135],
        pipeline: [125, -75],
        tuning: [130, 92],
        parameter: [190, 118],
        support: [-95, 95],
        risk: [65, 112],
        contradiction: [15, 150],
        sector: [-160, 40],
        metric: [210, -20],
        entity: [0, 0],
      };
      for (const node of nodes) {
        const seed = seededUnit(node.id || node.label || String(node.index));
        const angle = seed * Math.PI * 2;
        const radius = 45 + seededUnit(`${node.id}:semantic-radius`) * 310;
        const drift = semanticDrift[node.kind] || [0, 0];
        const importance = Math.min(1, Math.max(0, Number(node.importance_score || 0) / 10));
        positions.set(node.id, {
          x: Math.cos(angle) * radius * (1 - importance * 0.35) + drift[0],
          y: Math.sin(angle) * radius * (1 - importance * 0.35) + drift[1],
        });
        velocities.set(node.id, { x: 0, y: 0 });
      }

      const iterations = Math.max(55, Math.min(120, Math.floor(18000 / Math.max(120, nodes.length))));
      const sampleStep = Math.max(1, Math.floor(nodes.length / 260));
      for (let iteration = 0; iteration < iterations; iteration += 1) {
        const cooling = 1 - iteration / iterations;
        for (let i = 0; i < nodes.length; i += 1) {
          const a = nodes[i];
          const pa = positions.get(a.id);
          const va = velocities.get(a.id);
          for (let j = i + 1; j < nodes.length; j += sampleStep) {
            const b = nodes[j];
            if (a.id === b.id) continue;
            const pb = positions.get(b.id);
            const vb = velocities.get(b.id);
            let dx = pa.x - pb.x;
            let dy = pa.y - pb.y;
            let distance = Math.sqrt(dx * dx + dy * dy) || 0.001;
            const connected = adjacency.get(a.id).has(b.id);
            const minDistance = connected ? 30 : 48;
            if (distance < minDistance) distance = minDistance;
            const crowd = Math.max(0.62, Math.min(1.15, Math.log10(nodes.length + 10) / 2.6));
            const force = (connected ? 900 : 2600 * sampleStep) * crowd / (distance * distance);
            dx /= distance;
            dy /= distance;
            va.x += dx * force;
            va.y += dy * force;
            vb.x -= dx * force;
            vb.y -= dy * force;
          }
        }

        for (const link of links) {
          const source = positions.get(link.source);
          const target = positions.get(link.target);
          const vs = velocities.get(link.source);
          const vt = velocities.get(link.target);
          const dx = target.x - source.x;
          const dy = target.y - source.y;
          const distance = Math.sqrt(dx * dx + dy * dy) || 0.001;
          const desired = linkLength(link.predicate);
          const force = (distance - desired) * (0.010 + cooling * 0.007) * linkStrength(link.predicate);
          const nx = dx / distance;
          const ny = dy / distance;
          vs.x += nx * force;
          vs.y += ny * force;
          vt.x -= nx * force;
          vt.y -= ny * force;
        }

        for (const node of nodes) {
          const p = positions.get(node.id);
          const v = velocities.get(node.id);
          const drift = semanticDrift[node.kind] || [0, 0];
          const degree = degreeMap.get(node.id) || 0;
          const centerPull = 0.0011 + Math.min(0.0024, degree * 0.00016);
          const driftPull = node.kind === 'ticker' ? 0.0005 : 0.0012;
          v.x += (drift[0] - p.x) * driftPull - p.x * centerPull;
          v.y += (drift[1] - p.y) * driftPull - p.y * centerPull;
          v.x *= 0.74;
          v.y *= 0.74;
          p.x += v.x * (0.85 + cooling * 0.28);
          p.y += v.y * (0.85 + cooling * 0.28);
        }
      }

      let cx = 0;
      let cy = 0;
      for (const node of nodes) {
        const p = positions.get(node.id);
        cx += p.x;
        cy += p.y;
      }
      cx /= nodes.length;
      cy /= nodes.length;
      let maxRadius = 1;
      for (const node of nodes) {
        const p = positions.get(node.id);
        p.x -= cx;
        p.y -= cy;
        maxRadius = Math.max(maxRadius, Math.sqrt(p.x * p.x + p.y * p.y));
      }
      const scale = Math.min(1.35, 360 / maxRadius);
      return nodes.map((node) => {
        const p = positions.get(node.id);
        return { ...node, position: [p.x * scale, p.y * scale, 0] };
      });
    }

    function computeFastClusterLayout(nodes, rawLinks) {
      const degreeMap = new Map(nodes.map((node) => [node.id, 0]));
      for (const link of rawLinks || []) {
        degreeMap.set(link.source, (degreeMap.get(link.source) || 0) + 1);
        degreeMap.set(link.target, (degreeMap.get(link.target) || 0) + 1);
      }
      const shells = {
        ticker: { center: [0, 0, 0], radius: 245, xScale: 1.18, yScale: 0.88, zScale: 0.92, offset: 0.2 },
        support: { center: [-135, 90, 120], radius: 95, xScale: 0.95, yScale: 0.85, zScale: 1.25, offset: 0.9 },
        risk: { center: [145, 90, -115], radius: 115, xScale: 1.0, yScale: 0.9, zScale: 1.18, offset: 1.7 },
        contradiction: { center: [155, -90, 125], radius: 125, xScale: 0.95, yScale: 0.95, zScale: 1.22, offset: 2.5 },
        sector: { center: [-180, -115, -105], radius: 130, xScale: 1.05, yScale: 0.86, zScale: 1.08, offset: 3.1 },
        event: { center: [-10, 165, -145], radius: 165, xScale: 1.22, yScale: 0.72, zScale: 1.28, offset: 3.8 },
        temporal: { center: [-110, -170, 35], radius: 110, xScale: 0.8, yScale: 0.95, zScale: 1.3, offset: 4.3 },
        pipeline: { center: [140, -150, -45], radius: 105, xScale: 0.9, yScale: 0.9, zScale: 1.25, offset: 4.8 },
        tuning: { center: [135, 20, 165], radius: 105, xScale: 0.85, yScale: 0.85, zScale: 1.3, offset: 5.2 },
        parameter: { center: [215, -45, 135], radius: 95, xScale: 0.75, yScale: 0.9, zScale: 1.2, offset: 5.6 },
        metric: { center: [225, 115, 30], radius: 100, xScale: 0.82, yScale: 0.9, zScale: 1.12, offset: 6.0 },
        entity: { center: [0, -165, 130], radius: 165, xScale: 1.15, yScale: 0.78, zScale: 1.22, offset: 6.4 },
      };
      const grouped = new Map();
      for (const node of nodes) {
        if (!grouped.has(node.kind)) grouped.set(node.kind, []);
        grouped.get(node.kind).push(node);
      }
      const positioned = [];
      const goldenAngle = Math.PI * (3 - Math.sqrt(5));
      for (const [kind, group] of grouped.entries()) {
        const shell = shells[kind] || shells.entity;
        group.sort((a, b) => (degreeMap.get(b.id) || 0) - (degreeMap.get(a.id) || 0));
        group.forEach((node, index) => {
          const t = group.length === 1 ? 0.5 : (index + 0.5) / group.length;
          const zUnit = 1 - 2 * t;
          const radial = Math.sqrt(Math.max(0, 1 - zUnit * zUnit));
          const angle = shell.offset + index * goldenAngle + seededUnit(`${node.id}:angle`) * 0.38;
          const degree = degreeMap.get(node.id) || 0;
          const importance = Math.min(1, Math.log1p(Math.max(0, Number(node.importance_score || 0))) / 5.2);
          const corePull = Math.min(0.42, degree * 0.018 + importance * 0.22);
          const radius = shell.radius * (1 - corePull) + seededUnit(`${node.id}:radius`) * 52;
          const [cx, cy, cz] = shell.center;
          positioned.push({
            ...node,
            position: [
              cx + Math.cos(angle) * radial * radius * shell.xScale,
              cy + Math.sin(angle) * radial * radius * shell.yScale,
              cz + zUnit * radius * shell.zScale + (seededUnit(`${node.id}:zfast`) - 0.5) * 42,
            ],
          });
        });
      }
      return positioned;
    }

    function linkLength(predicate) {
      if (predicate === 'supportsSignal') return 86;
      if (predicate === 'increasesRiskOf') return 108;
      if (predicate === 'contradictsSignal') return 112;
      if (predicate === 'hasRecentNews' || predicate === 'hasRecentDisclosure') return 64;
      if (predicate === 'containsFrame' || predicate === 'hasTimeFrame' || predicate === 'observesTicker') return 62;
      if (predicate === 'containsEvent' || predicate === 'occursInTimeBucket') return 58;
      if (predicate === 'containsQuote' || predicate === 'containsExecution' || predicate === 'usesMarketSnapshot' || predicate === 'usesRawSource' || predicate === 'hasMacroContext' || predicate === 'hasImpactScore') return 60;
      if (predicate === 'selectsCandidate') return 70;
      if (predicate === 'feedsStage' || predicate === 'requiresApprovalFrom' || predicate === 'usesCostModel') return 82;
      if (predicate === 'contains' || predicate === 'produces' || predicate === 'blocksTradeBelow') return 72;
      if (predicate === 'hasTuningMode' || predicate === 'adjustsStage' || predicate === 'appliesToStage') return 72;
      if (predicate === 'tunesParameter' || predicate === 'hasTunedValue' || predicate === 'producesTunedValue') return 58;
      if (predicate === 'usesOntologySignal' || predicate === 'calibratesSignal') return 66;
      if (predicate === 'raisesTuningPressure') return 62;
      if (predicate === 'belongsToSector' || predicate === 'hasTicker') return 78;
      return 124;
    }

    function linkStrength(predicate) {
      if (predicate === 'supportsSignal') return 1.25;
      if (predicate === 'increasesRiskOf') return 1.05;
      if (predicate === 'contradictsSignal') return 1.1;
      if (predicate === 'hasRecentNews' || predicate === 'hasRecentDisclosure') return 1.55;
      if (predicate === 'containsFrame' || predicate === 'hasTimeFrame' || predicate === 'observesTicker') return 1.65;
      if (predicate === 'containsEvent' || predicate === 'occursInTimeBucket') return 1.72;
      if (predicate === 'containsQuote' || predicate === 'containsExecution' || predicate === 'usesMarketSnapshot' || predicate === 'usesRawSource' || predicate === 'hasMacroContext' || predicate === 'hasImpactScore') return 1.58;
      if (predicate === 'selectsCandidate') return 1.5;
      if (predicate === 'feedsStage' || predicate === 'requiresApprovalFrom' || predicate === 'usesCostModel') return 1.35;
      if (predicate === 'contains' || predicate === 'produces' || predicate === 'blocksTradeBelow') return 1.32;
      if (predicate === 'hasTuningMode' || predicate === 'adjustsStage' || predicate === 'appliesToStage') return 1.45;
      if (predicate === 'tunesParameter' || predicate === 'hasTunedValue' || predicate === 'producesTunedValue') return 1.7;
      if (predicate === 'usesOntologySignal' || predicate === 'calibratesSignal') return 1.55;
      if (predicate === 'raisesTuningPressure') return 1.65;
      if (predicate === 'belongsToSector' || predicate === 'hasTicker') return 1.35;
      return 0.9;
    }

    function seededUnit(value) {
      const text = String(value);
      let hash = 2166136261;
      for (let i = 0; i < text.length; i += 1) {
        hash ^= text.charCodeAt(i);
        hash = Math.imul(hash, 16777619);
      }
      return ((hash >>> 0) % 100000) / 100000;
    }

    function nodeColor(kind) {
      return {
        ticker: 0x38bdf8,
        event: 0xf97316,
        temporal: 0x06b6d4,
        pipeline: 0x2563eb,
        tuning: 0xeab308,
        parameter: 0xec4899,
        metric: 0x94a3b8,
        sector: 0x84cc16,
        support: 0x22c55e,
        risk: 0xef4444,
        contradiction: 0xd946ef,
        entity: 0xf8fafc
      }[kind] || 0xf8fafc;
    }

    function edgeColor(predicate) {
      if (predicate === 'supportsSignal') return 0x22c55e;
      if (predicate === 'increasesRiskOf') return 0xef4444;
      if (predicate === 'contradictsSignal') return 0xd946ef;
      if (predicate === 'hasRecentNews' || predicate === 'hasRecentDisclosure') return 0xf97316;
      if (predicate === 'containsFrame' || predicate === 'hasTimeFrame' || predicate === 'observesTicker' || predicate === 'containsEvent' || predicate === 'occursInTimeBucket' || predicate === 'containsQuote' || predicate === 'containsExecution' || predicate === 'usesMarketSnapshot' || predicate === 'usesRawSource' || predicate === 'hasMacroContext' || predicate === 'hasImpactScore') return 0x06b6d4;
      if (predicate === 'selectsCandidate' || predicate === 'feedsStage' || predicate === 'requiresApprovalFrom' || predicate === 'usesCostModel' || predicate === 'contains' || predicate === 'produces' || predicate === 'blocksTradeBelow') return 0x2563eb;
      if (predicate === 'tunesParameter' || predicate === 'hasTunedValue' || predicate === 'producesTunedValue' || predicate === 'hasTuningMode' || predicate === 'adjustsStage' || predicate === 'appliesToStage' || predicate === 'usesOntologySignal' || predicate === 'calibratesSignal' || predicate === 'raisesTuningPressure') return 0xeab308;
      return 0x94a3b8;
    }

    function neonColor(kind) {
      return {
        ticker: 0x67e8f9,
        event: 0xfdba74,
        temporal: 0x67e8f9,
        pipeline: 0x93c5fd,
        tuning: 0xfef08a,
        parameter: 0xf9a8d4,
        metric: 0xcbd5e1,
        sector: 0xd9f99d,
        support: 0x86efac,
        risk: 0xfca5a5,
        contradiction: 0xf0abfc,
        entity: 0xffffff
      }[kind] || 0xffffff;
    }

    function neonEdgeColor(predicate) {
      if (predicate === 'supportsSignal') return 0x86efac;
      if (predicate === 'increasesRiskOf') return 0xfca5a5;
      if (predicate === 'contradictsSignal') return 0xf0abfc;
      if (predicate === 'hasRecentNews' || predicate === 'hasRecentDisclosure') return 0xfdba74;
      if (predicate === 'containsFrame' || predicate === 'hasTimeFrame' || predicate === 'observesTicker' || predicate === 'containsEvent' || predicate === 'occursInTimeBucket' || predicate === 'containsQuote' || predicate === 'containsExecution' || predicate === 'usesMarketSnapshot' || predicate === 'usesRawSource' || predicate === 'hasMacroContext' || predicate === 'hasImpactScore') return 0x67e8f9;
      if (predicate === 'selectsCandidate' || predicate === 'feedsStage' || predicate === 'requiresApprovalFrom' || predicate === 'usesCostModel' || predicate === 'contains' || predicate === 'produces' || predicate === 'blocksTradeBelow') return 0x93c5fd;
      if (predicate === 'tunesParameter' || predicate === 'hasTunedValue' || predicate === 'producesTunedValue' || predicate === 'hasTuningMode' || predicate === 'adjustsStage' || predicate === 'appliesToStage' || predicate === 'usesOntologySignal' || predicate === 'calibratesSignal' || predicate === 'raisesTuningPressure') return 0xfef08a;
      return 0x67e8f9;
    }

    function buildGraphMetrics(nodes, links) {
      const degreeMap = new Map(nodes.map((node) => [node.id, 0]));
      for (const link of links || []) {
        degreeMap.set(link.source, (degreeMap.get(link.source) || 0) + 1);
        degreeMap.set(link.target, (degreeMap.get(link.target) || 0) + 1);
      }
      const degrees = [...degreeMap.values()];
      const maxDegree = Math.max(1, ...degrees);
      const averageDegree = degrees.length
        ? degrees.reduce((sum, value) => sum + value, 0) / degrees.length
        : 0;
      return {
        nodeCount: Math.max(1, nodes.length),
        linkCount: Math.max(0, (links || []).length),
        maxDegree,
        averageDegree,
        degreeMap,
      };
    }

    function nodeRadius(node, metrics = null) {
      const size = Number(node && node.size);
      const kind = node && node.kind;
      const nodeCount = metrics ? metrics.nodeCount : 1;
      const degree = metrics ? (metrics.degreeMap.get(node.id) || 0) : 0;
      const maxDegree = metrics ? metrics.maxDegree : 1;
      const density = metrics ? metrics.linkCount / Math.max(1, metrics.nodeCount) : 0;
      const crowdScale = Math.max(0.46, Math.min(0.86, 1.02 - Math.log10(nodeCount + 10) * 0.18 - Math.min(0.18, density * 0.018)));
      const kindBase = kind === 'ticker'
        ? 8.2
        : kind === 'event'
          ? 4.4
          : kind === 'temporal'
            ? 4.8
          : kind === 'sector'
            ? 5.8
            : kind === 'pipeline'
              ? 5.8
              : kind === 'tuning'
                ? 5.2
                : kind === 'parameter'
                  ? 4.5
                  : kind === 'metric'
                    ? 3.8
                    : 5.1;
      const backendSize = Number.isFinite(size) && size > 0 ? Math.min(1.35, Math.max(0.82, size / 12)) : 1;
      const importance = Math.max(0, Math.min(1, Number(node && node.importance_score || 0)));
      const degreeBoost = Math.log1p(degree) / Math.log1p(maxDegree);
      const radius = kindBase * backendSize * crowdScale * (0.82 + importance * 0.32 + degreeBoost * 0.42);
      return Math.max(2.8, Math.min(13.5, radius));
    }

    function kindLabel(kind) {
      return {
        ticker: '종목',
        event: '뉴스/공시 이벤트',
        temporal: '시간축',
        pipeline: '분석 파이프라인',
        tuning: '파라미터 조정',
        parameter: '파라미터',
        metric: '지표',
        sector: '섹터',
        support: '긍정 신호',
        risk: '리스크 요인',
        contradiction: '상충 요인',
        entity: '개체'
      }[kind] || kind;
    }

    function shortLabel(label) {
      if (label.startsWith('NEWS:')) return `뉴스 ${label.slice(5, 11)}`;
      if (label.length > 22) return `${label.slice(0, 20)}...`;
      return label;
    }
    function degree(nodeId, links) {
      return links.filter((link) => link.source === nodeId || link.target === nodeId).length;
    }

    function linkKey(source, target, predicate) {
      return `${source}::${predicate}::${target}`;
    }

    function formatLiveTraceTime(value) {
      if (!value) return '실제 시각 대기';
      const parsed = new Date(value);
      if (Number.isNaN(parsed.getTime())) return String(value);
      return `실제 ${parsed.toLocaleTimeString('ko-KR', { hour12: false })}`;
    }

    function graphSignature(graph) {
      const counts = graph.counts || {};
      const latestStep = (graph.reasoning_steps || []).map((step) => `${step.path_id}:${step.title}:${step.description}`).join('|');
      let topologyHash = 2166136261;
      for (const link of graph.links || []) {
        const value = `${link.source}>${link.predicate}>${link.target}|`;
        for (let index = 0; index < value.length; index += 1) {
          topologyHash ^= value.charCodeAt(index);
          topologyHash = Math.imul(topologyHash, 16777619);
        }
      }
      return `${counts.nodes || 0}:${counts.links || 0}:${topologyHash >>> 0}:${latestStep}`;
    }

    function renderNodePanel(node, links) {
      const nodeLinks = links.filter((link) => link.source === node.id || link.target === node.id);
      const related = nodeLinks
        .slice(0, 20)
        .map((link) => {
          const other = link.source === node.id ? link.target : link.source;
          const direction = link.source === node.id ? '나감' : '들어옴';
          return `<div>${direction} <strong>${link.predicate}</strong> ${shortLabel(other)}</div>`;
        })
        .join('');
      const hiddenCount = Math.max(0, nodeLinks.length - 20);
      document.getElementById('ontologyPanel').innerHTML = `
        <strong>${node.label}</strong>
        <div class="muted">종류: ${kindLabel(node.kind)} · 연결 ${degree(node.id, links)}개 · 중요도 ${Number(node.importance_score || 0).toFixed(2)}</div>
        <div style="margin-top:10px;">${related || '<span class="muted">연결 관계 없음</span>'}</div>
        <div class="muted" style="margin-top:8px;">${hiddenCount > 0 ? `추가 관계 ${hiddenCount}개는 생략했습니다.` : ''}</div>
      `;
    }
    function createTextSprite(THREE, text, color) {
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      canvas.width = 512;
      canvas.height = 128;
      context.font = 'bold 42px Arial';
      context.fillStyle = 'rgba(15, 23, 42, 0.78)';
      context.fillRect(0, 18, 512, 72);
      context.strokeStyle = `#${color.toString(16).padStart(6, '0')}`;
      context.lineWidth = 4;
      context.strokeRect(2, 20, 508, 68);
      context.fillStyle = '#ffffff';
      context.fillText(text, 18, 67);
      const texture = new THREE.CanvasTexture(canvas);
      const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
      const sprite = new THREE.Sprite(material);
      sprite.scale.set(130, 32, 1);
      return sprite;
    }

    document.getElementById('startButton').addEventListener('click', async () => {
      if (!sessionId || !selectedGoal) return;
      const data = await (await fetch('/api/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: sessionId, selected_goal: selectedGoal }) })).json();
      document.getElementById('gate').textContent = data.started ? '선택한 목표 기준으로 모의투자를 시작했습니다. 실전 주문은 비활성화되어 있습니다.' : '시작하지 못했습니다.';
      document.getElementById('output').textContent = JSON.stringify(data, null, 2);
      if (data.performance) renderMockPerformance(data.performance);
      if (data.started) startMockPerformancePolling();
    });

    document.getElementById('resetButton').addEventListener('click', () => {
      sessionId = null; selectedGoal = null;
      document.getElementById('choices').innerHTML = '';
      document.getElementById('relations').innerHTML = '';
      document.getElementById('feasibility').textContent = '대기 중';
      document.getElementById('feasibilityBar').style.width = '0%';
      document.getElementById('summary').textContent = '목표를 입력하면 시장 자료와 온톨로지 근거를 바탕으로 가능성을 계산합니다.';
      document.getElementById('gate').textContent = '학습과 정보 수집은 서버가 켜져 있는 동안 자동으로 계속 진행됩니다.';
      document.getElementById('output').textContent = '아직 실행하지 않았습니다.';
      document.getElementById('startButton').disabled = true;
      if (mockPerformanceTimer) window.clearInterval(mockPerformanceTimer);
    });


    let streamingTargetReturnRate = 0;
    let streamingTargetMinutes = 0;

    async function startStreamingDemo() {
      let targetReturn = parseFloat(document.getElementById('targetReturn')?.value || 0.02);
      if (targetReturn > 1) targetReturn = targetReturn / 100.0;
      const periodMinutes = parseInt(document.getElementById('targetMinutes')?.value || 390);
      streamingDemoHistory = [];
      streamingDemoPrices = {};
      streamingReturnSeries = [];
      streamingInitialCash = Math.max(1, Number(liveAccountBasis?.cash || 10000000));
      streamingTargetReturnRate = targetReturn;
      streamingTargetMinutes = periodMinutes;
      try {
        const response = await fetch('/api/paper-trading/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            target_return_rate: targetReturn,
            period_minutes: periodMinutes,
            initial_cash_source: 'auto'
          })
        });
        const data = await response.json();
        streamingDemoId = data.demo_id;
        streamingDemoRunning = true;
        activeOperationMode = 'paper_trading';
        updateTerminateTradingButton();
        streamingInitialCash = Number(data.initial_cash || streamingInitialCash || 10000000);
        document.getElementById('streamingDemoContainer').hidden = false;
        document.getElementById('streamingDemoStatus').textContent = '모의투자 실행 중...';
        document.getElementById('streamingDemoProgress').style.width = '0%';
      } catch (error) {
        document.getElementById('streamingDemoStatus').textContent = '오류: ' + error.message;
      }
    }

    async function runStreamingDemoStep() {
      if (!streamingDemoId || !streamingDemoRunning) return;
      if (streamingStepBusy) return { status: 'busy', progress: 0 };
      streamingStepBusy = true;
      const requestedDemoId = streamingDemoId;
      try {
        document.getElementById('streamingDemoStatus').textContent = '모의투자 종목 스캔 및 매매 판단 중...';
        document.getElementById('mockStatus').textContent = '모의투자 종목 스캔 및 매매 판단 중...';
        const response = await fetchWithOptionalTimeout('/api/paper-trading/step', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ demo_id: requestedDemoId }),
        }, 120000);
        if (!response.ok) {
          streamingStepFailures += 1;
          const message = response.status === 404 ? '모의투자 세션이 만료되었습니다. 다시 시작하세요.' : `모의투자 업데이트 실패 (${response.status})`;
          if (streamingStepFailures < 3 && response.status >= 500) {
            const retryMessage = `모의투자 응답 지연, 재시도 중 (${streamingStepFailures}/3)`;
            document.getElementById('streamingDemoStatus').textContent = retryMessage;
            document.getElementById('mockStatus').textContent = retryMessage;
            return { status: 'retrying', progress: 0 };
          }
          streamingDemoRunning = false;
          streamingDemoId = null;
          document.getElementById('streamingDemoStatus').textContent = message;
          document.getElementById('mockStatus').textContent = message;
          return { status: 'stopped', progress: 0 };
        }
        const data = await response.json();
        if (!streamingDemoRunning || requestedDemoId !== streamingDemoId) {
          return { status: 'stale', progress: 0 };
        }
        streamingStepFailures = 0;
        updateStreamingAccount(data);
        if (data.status === 'waiting') {
          const remaining = Number(data.seconds_until_next_step || 0);
          const message = `모의투자 대기 중 · 다음 1분 bar까지 ${remaining.toFixed(1)}초`;
          document.getElementById('streamingDemoStatus').textContent = message;
          document.getElementById('mockStatus').textContent = message;
          return data;
        }
        if (data.status === 'expired') {
          streamingDemoRunning = false;
          streamingDemoId = null;
          const message = '모의투자 세션이 만료되었습니다. 다시 시작하세요.';
          document.getElementById('streamingDemoStatus').textContent = message;
          document.getElementById('mockStatus').textContent = message;
          return data;
        }
        document.getElementById('streamingDemoProgress').style.width = data.progress + '%';
        if (data.prices && typeof data.prices === 'object') {
          Object.entries(data.prices).forEach(([ticker, price]) => { streamingDemoPrices[ticker] = Number(price || 0); });
        }
        document.getElementById('streamingDemoStatus').textContent =
          `모의투자 ${data.step || '완료'}분 진행 · 전체 ${data.universe_scanned_count || data.universe_count || '-'}개 스캔 · 후보 ${data.candidate_ticker_count || data.active_ticker_count || '-'}개 · ${(data.progress || 0).toFixed(1)}%`;
        if (data.account) {
          renderStreamingPerformance(data);
          loadPrincipalProtectionState().catch(() => {});
        }
        if (data.trades && data.trades.length > 0) {
          data.trades.forEach(t => { streamingDemoHistory.unshift(t); streamingDemoPrices[t.ticker] = t.price; });
          const executionList = document.getElementById('mockExecutions');
          if (executionList) executionList.innerHTML = streamingDemoHistory.slice(0, 20).map(t => `<tr><td class="side-${t.side.toLowerCase()}">${t.side}</td><td>${t.ticker}</td><td>${t.quantity}</td><td>${formatMoney(t.price, t.currency)}</td></tr>`).join('');
          const tradeList = document.getElementById('streamingTradeList');
          if (tradeList) tradeList.innerHTML = data.trades.map(t => `<tr><td>${t.ticker}</td><td class="side-${t.side.toLowerCase()}">${t.side}</td><td>${t.quantity}</td><td>${formatMoney(t.value, t.currency)}${t.currency === 'USD' ? ` (${fmtWon.format(t.value_krw || 0)})` : ''}</td></tr>`).join('');
        }
        if (data.status === 'completed') {
          streamingDemoRunning = false;
          document.getElementById('streamingDemoStatus').textContent = '모의투자 완료';
        }
        return data;
      } catch (error) {
        streamingStepFailures += 1;
        const message = `모의투자 응답 지연, 재시도 중 (${Math.min(streamingStepFailures, 3)}/3)`;
        document.getElementById('output').textContent = `Paper trading step error: ${error.message || error}`;
        if (streamingStepFailures < 3) {
          document.getElementById('streamingDemoStatus').textContent = message;
          document.getElementById('mockStatus').textContent = message;
          return { status: 'retrying', progress: 0 };
        }
        streamingDemoRunning = false;
        document.getElementById('streamingDemoStatus').textContent = `모의투자 업데이트 실패: ${error.message || error}`;
        return null;
      } finally {
        streamingStepBusy = false;
      }
    }

    function updateStreamingAccount(data) {
      if (!data || !data.account) return;
      const cash = Number(data.account.cash || 0);
      const accountValue = Number(data.account.account_value || cash);
      const invested = Math.max(0, accountValue - cash);
      const profit = accountValue - streamingInitialCash;
      document.getElementById('streamingDeposit').textContent = formatCashByCurrency(data.account);
      document.getElementById('streamingInvested').textContent = fmtWon.format(invested);
      document.getElementById('streamingProfit').textContent = fmtWon.format(profit);
      document.getElementById('streamingReturnRate').textContent = (Number(data.account.return_rate || 0) * 100).toFixed(2) + '%';
    }

    async function autoRunStreamingDemo(isFirstTick = false) {
      if (!streamingDemoRunning) return;
      const intervalMs = 60000;
      const data = await runStreamingDemoStep();
      if (streamingDemoRunning) {
        const nextStepSeconds = Number(data && data.seconds_until_next_step);
        const paceByBackend = (data && (data.status === 'waiting' || data.status === 'running')) && Number.isFinite(nextStepSeconds);
        const waitMs = paceByBackend ? Math.max(50, nextStepSeconds * 1000) : intervalMs;
        streamingDemoTimer = setTimeout(() => autoRunStreamingDemo(false), waitMs);
      }
    }
    function renderStatus(data = {}) {
      if (liveAccountBasis && (!data || data.basis_source !== 'kis_live_account')) {
        data = {
          ...(data || {}),
          cash: liveAccountBasis.cash,
          cash_equivalent_krw: liveAccountBasis.cash_equivalent_krw,
          krw_cash: liveAccountBasis.krw_cash,
          foreign_cash_krw: liveAccountBasis.foreign_cash_krw,
          cash_by_currency: liveAccountBasis.cash_by_currency,
          foreign_cash_by_currency: liveAccountBasis.foreign_cash_by_currency,
          base_currency: liveAccountBasis.base_currency,
          equity: liveAccountBasis.equity,
          cash_weight: liveAccountBasis.cash_weight,
          basis_source: 'kis_live_account',
          account_suffix: liveAccountBasis.account_suffix,
        };
      }
      const summary = accountSnapshotSummary(data || {});
      const setText = (id, text) => {
        const node = document.getElementById(id);
        if (node) node.textContent = text;
      };
      const split = splitInvestmentSummary(data, summary);
      setText('equity', fmtWon.format(summary.cash));
      setText('totalAssets', fmtWon.format(summary.equity));
      setText('cash', fmtWon.format(summary.cash));
      setText('domesticInvestedValue', fmtWon.format(split.domesticInvested));
      setText('usdCash', formatMoney(split.usdCash, 'USD'));
      setText('foreignInvestedValue', formatMoney(split.foreignInvestedUsd, 'USD'));
      setText('investedValue', `보유주식 ${fmtWon.format(summary.invested)}`);
      setText('krwCash', `KRW 현금 ${fmtWon.format(summary.krwCash)}`);
      setText('foreignCash', `외화/해외평가 ${fmtWon.format(summary.foreignCashKrw)}`);
      setText('cashWeight', `현금 비중 ${(summary.cashWeight * 100).toFixed(1)}%`);
      if (data.basis_source === 'kis_live_account' && Number(data.equity || 0) > 0) {
        liveAccountBasis = {
          cash: Number(summary.cash),
          cash_equivalent_krw: Number(summary.cashEquivalentKrw),
          krw_cash: Number(summary.krwCash),
          foreign_cash_krw: Number(summary.foreignCashKrw),
          cash_by_currency: summary.cashByCurrency,
          foreign_cash_by_currency: summary.foreignCashByCurrency,
          base_currency: data.base_currency || 'KRW',
          equity: Number(summary.equity),
          cash_weight: Number(summary.cashWeight || 0),
          account_suffix: data.account_suffix || '',
        };
      }
    }

    function renderBrokerAccountCard(connection = {}) {
      const depositTarget = document.getElementById('brokerDeposit');
      const holdingsTarget = document.getElementById('brokerHoldings');
      const accountTarget = document.getElementById('brokerAccount');
      const equityTarget = document.getElementById('brokerEquity');
      const investedTarget = document.getElementById('brokerInvested');
      const krwCashTarget = document.getElementById('brokerKrwCash');
      const foreignCashTarget = document.getElementById('brokerForeignCash');
      const statusTarget = document.getElementById('brokerStatus');
      if (!depositTarget || !holdingsTarget || !accountTarget || !statusTarget) return;
      accountTarget.textContent = `Account ${connection.account_suffix || '-'}`;
      if (connection.account_checked) {
        const summary = accountSnapshotSummary(connection);
        const holdings = connection.holdings_count ?? connection.holdings ?? 0;
        const submitted = Number((connection.live_order_journal || {}).submitted_count ?? connection.submitted_count ?? 0);
        const updatedAt = connection.updated_at ? new Date(connection.updated_at).toLocaleTimeString('ko-KR') : new Date().toLocaleTimeString('ko-KR');
        const accountLabel = `계좌번호 ${connection.account_suffix || '-'}`;
        const positions = Array.isArray(connection.positions) ? connection.positions : [];
        const domesticPositions = positions.filter((position) => positionCurrency(position) === 'KRW');
        const foreignPositions = positions.filter((position) => positionCurrency(position) !== 'KRW');
        depositTarget.textContent = accountLabel;
        accountTarget.textContent = accountLabel;
        if (equityTarget) equityTarget.textContent = `총자산 ${fmtWon.format(summary.equity)}`;
        if (investedTarget) investedTarget.textContent = `보유주식 ${fmtWon.format(summary.invested)}`;
        if (krwCashTarget) krwCashTarget.textContent = `원화 ${fmtWon.format(summary.krwCash)}`;
        if (foreignCashTarget) foreignCashTarget.textContent = `외화 ${formatForeignCash(connection)}`;
        holdingsTarget.textContent = `보유 종목 ${holdings}개`;
        renderHoldingList('brokerDomesticHoldings', domesticPositions);
        renderHoldingList('brokerForeignHoldings', foreignPositions);
        statusTarget.textContent = `KIS 실계좌 실시간 갱신 ${updatedAt} · 제출 ${submitted}건`;
        applyLiveAccountBasis(connection);
        return;
      }
      if (connection.ok === false) {
        depositTarget.textContent = '조회 실패';
        accountTarget.textContent = '계좌번호 -';
        holdingsTarget.textContent = '보유 종목 -';
        renderHoldingList('brokerDomesticHoldings', []);
        renderHoldingList('brokerForeignHoldings', []);
        if (equityTarget) equityTarget.textContent = '총자산 -';
        if (investedTarget) investedTarget.textContent = '보유주식 -';
        if (krwCashTarget) krwCashTarget.textContent = '원화 -';
        if (foreignCashTarget) foreignCashTarget.textContent = '외화 -';
        statusTarget.textContent = connection.message || connection.error || 'KIS 계좌 조회 실패';
        return;
      }
      depositTarget.textContent = '조회 대기';
      accountTarget.textContent = '계좌번호 -';
      holdingsTarget.textContent = '보유 종목 -';
      renderHoldingList('brokerDomesticHoldings', []);
      renderHoldingList('brokerForeignHoldings', []);
      if (equityTarget) equityTarget.textContent = '총자산 -';
      if (investedTarget) investedTarget.textContent = '보유주식 -';
      if (krwCashTarget) krwCashTarget.textContent = '원화 -';
      if (foreignCashTarget) foreignCashTarget.textContent = '외화 -';
      statusTarget.textContent = 'KIS 실계좌 조회 대기 중';
    }

    async function refreshLiveTradingProgress() {
      if (activeOperationMode !== 'live_trading') return;
      if (liveTradingProgressBusy) return;
      liveTradingProgressBusy = true;
      try {
      const data = await (await fetch('/api/live-trading/progress')).json();
      if (data.connection) {
        const connection = {
          ...data.connection,
          updated_at: data.updated_at,
          live_order_journal: data.live_order_journal,
          submitted_count: data.executions_count,
        };
        renderBrokerAccountCard(connection);
      }
      renderStatus({ ...data, basis_source: 'kis_live_account' });
      renderLivePerformanceSummary(data);
      renderMockRunTables({ ...data, display_mode: 'live' });
      } finally {
        liveTradingProgressBusy = false;
      }
    }

    window.applyLiveFlags = applyLiveFlags;
    window.startSelectedOperationMode = startSelectedOperationMode;
    window.terminateActiveTrading = terminateActiveTrading;
    applyCompactKoreanDashboard();
    applyUrlGoalParams();
    updateModeButtons();
    updateModeActionCopy();
    loadStatus();
    loadPrincipalProtectionState().catch(() => {});
    startLearningStatusPolling();
    loadRealtimeRuntime();
    loadOperationModeStatus().catch(() => {});
    loadDiagnostics();
    loadOntologyGraph();
    loadMockPerformance();
    refreshLiveSnapshot();
    setInterval(() => loadOperationModeStatus().catch(() => {}), 3000);
    setInterval(() => loadStatus().catch(() => {}), 3000);
    setInterval(() => refreshLiveTradingProgress().catch(() => {}), 3000);
    setInterval(refreshLiveSnapshot, 5000);
    setInterval(() => loadMockPerformance().catch(() => {}), 3000);
    setInterval(() => loadOntologyGraph().catch(() => {}), 15000);
    setInterval(() => loadOntologyLiveTrace().catch(() => {}), 1000);
  </script>
</body>
</html>
"""


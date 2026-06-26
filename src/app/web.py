from __future__ import annotations

import math
import json
import os
import re
import threading
import time
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse

from app.audit import AuditLogger
from app.backtesting import StreamingAcceleratedDemo, TimeScalerConfig, TimeMode
from app.data.llm_classifier import event_llm_runtime_status
from app.execution import MockKisDevelopersApi, PaperOrderExecutor
from app.graph import get_ontology_runtime
from app.goals import GoalRequest, NegotiatedGoal, assess_goal, build_compromise_goals
from app.pipeline import build_analysis_context
from app.research import ResearchRunResult, ResearchService
from app.realtime import OperationModeManager, RealtimeAccelerationPolicy, ShortHorizonRiskPolicy
from app.realtime.learning import (
    build_realtime_supervised_examples,
    run_hypothetical_realtime_test,
    update_realtime_model_artifacts,
)
from app.graph.npu_classifier import get_ontology_npu_classifier
from app.risk import RiskManager
from app.schemas.domain import (
    AccountSnapshot,
    FinalOrder,
    OrderSide,
    OrderType,
    RealtimeExecution,
    RealtimeQuote,
    SourceMetadata,
)
from app.storage import LocalResearchStore, ModelArtifactStore, StoredResearch
from app.strategy import build_goal_execution_plan
from app.trading import run_mock_trading_cycle

app = FastAPI(title="개인 투자 분석 시스템")
audit = AuditLogger(Path("logs/web-audit.jsonl"))
sessions: dict[str, dict[str, Any]] = {}
DEFAULT_RESEARCH_CONFIG = Path(os.getenv("RESEARCH_CONFIG", "config/research_sources.live.json"))
LIVE_REFRESH_SECONDS = max(5, int(os.getenv("LIVE_REFRESH_SECONDS", "15")))
LIVE_STALE_SECONDS = max(LIVE_REFRESH_SECONDS * 2, int(os.getenv("LIVE_STALE_SECONDS", "45")))
LEARNING_COLLECTION_INTERVAL_SECONDS = max(
    60,
    int(os.getenv("LEARNING_COLLECTION_INTERVAL_SECONDS", "3600")),
)
AUTO_START_LIVE_WORKER = os.getenv("AUTO_START_LIVE_WORKER", "false").lower() in {"1", "true", "yes", "on"}

_live_lock = threading.Lock()
_refresh_guard = threading.Lock()
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
    "request": {
        "busy": False,
        "stage": "idle",
        "message": "Waiting",
        "started_at": None,
        "updated_at": None,
        "last_error": None,
    },
}


def _get_store_root() -> Path:
  return Path(os.getenv("REALTIME_STORE_ROOT", "data/store"))


def _active_operation_mode() -> str:
  mode = _operation_mode_state.get("active")
  if mode is None:
    return "learning"
  mode_value = getattr(mode, "mode", mode)
  return str(getattr(mode_value, "value", mode_value))


def _is_simulation_mode(mode: Any | None = None) -> bool:
  return False


def _simulation_can_use_live_store() -> bool:
  return False


def _analysis_research_for_current_mode(current_store: LocalResearchStore) -> StoredResearch:
  return current_store.load_analysis_inputs()


def _current_data_policy() -> dict[str, Any]:
  return {
      "mode": _active_operation_mode(),
      "primary_store": _get_store_root().as_posix(),
      "analysis_input_stores": [_get_store_root().as_posix()],
      "synthetic_data_allowed": False,
      "orders_in_testing": False,
      "model_root": "data/models",
      "rule": "Learning, testing, and live trading all use the unified realtime data store only.",
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
) -> str:
  """Start a streaming accelerated demo and return demo_id."""
  demo_id = str(uuid4())
  
  if target_return_rate > 1:
    target_return_rate /= 100.0
  initial_cash = max(100_000.0, float(initial_cash))
  
  demo = StreamingAcceleratedDemo(
      config=TimeScalerConfig(mode=TimeMode.REALTIME, acceleration_factor=1.0),
      target_return_rate=target_return_rate,
      period_minutes=period_minutes,
      initial_cash=initial_cash,
      seed=seed,
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
  })
  
  return demo_id


_live_worker: threading.Thread | None = None
_live_state: dict[str, Any] = {
    "context": None,
    "research_result": None,
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
}


def _clear_live_analysis_cache_unlocked() -> None:
  _live_state["context"] = None
  _live_state["research_result"] = None
  _live_state["context_mode"] = None
  _live_state["graph_payload"] = None
  _live_state["graph_payload_context_id"] = None
  _live_state["store_summary"] = {}
  _live_state["stored_new_records"] = {}


@app.on_event("startup")
def _startup_live_worker() -> None:
    RealtimeAccelerationPolicy().apply_process_hints()
    if AUTO_START_LIVE_WORKER:
      _start_live_worker()


@app.on_event("shutdown")
def _shutdown_live_worker() -> None:
    _stop_live_worker()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.get("/api/status")
def status() -> JSONResponse:
  snapshot = _get_or_refresh_live()
  context = snapshot["context"]
  return _json(
    {
      "cash": context.account.cash,
      "equity": context.report.equity,
      "cash_weight": context.report.cash_weight,
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
    snapshot = _get_or_refresh_live()
    research_result = snapshot["research_result"]
    context = snapshot["context"]
    return _json(
        {
            "research_config": str(DEFAULT_RESEARCH_CONFIG),
            "diagnostics": _diagnostics_with_collection_config(research_result.diagnostics),
            "skipped_sources": research_result.skipped_sources,
            "stored_new_records": snapshot["stored_new_records"],
            "store_summary": snapshot["store_summary"],
            "data_volume": LocalResearchStore(root=_get_store_root()).data_volume(),
            "store_path": str(LocalResearchStore(root=_get_store_root()).db_path),
            "data_policy": _current_data_policy(),
            "events_sample": research_result.events[:5],
            "market_snapshots": research_result.market_snapshots,
            "graph_triples_count": len(context.graph.triples()),
            "reasoning_paths": context.reasoning_paths,
            "ontology_runtime": context.ontology_runtime.as_dict(),
            "updated_at": _iso_or_none(snapshot["last_updated"]),
            "last_error": snapshot["last_error"],
            "is_refreshing": snapshot["is_refreshing"],
            "refresh_interval_seconds": LIVE_REFRESH_SECONDS,
        }
    )


@app.get("/api/ontology/graph")
def ontology_graph() -> JSONResponse:
    snapshot = _get_or_refresh_live()
    context = snapshot["context"]
    payload = snapshot.get("graph_payload")
    if payload is None or snapshot.get("graph_payload_context_id") != id(context):
      payload = _graph_payload(context)
      with _live_lock:
        if _live_state["context"] is context:
          _live_state["graph_payload"] = payload
          _live_state["graph_payload_context_id"] = id(context)
    return _json(payload)


@app.get("/api/ontology/runtime")
def ontology_runtime() -> JSONResponse:
    return _json(get_ontology_runtime().as_dict())


@app.get("/api/realtime/runtime")
def realtime_runtime() -> JSONResponse:
    acceleration = RealtimeAccelerationPolicy().status()
    risk_policy = ShortHorizonRiskPolicy()
    return _json(
        {
            "acceleration": acceleration,
            "event_llm": event_llm_runtime_status(),
            "ontology_npu": get_ontology_npu_classifier().status(),
            "short_horizon_policy": risk_policy,
            "operation_mode": _operation_mode_state.get("active"),
            "resource_allocation": {
                "model_inference": os.getenv("LLM_EVENT_DEVICE", "NPU"),
                "model_backend": os.getenv("LLM_EVENT_INFERENCE_BACKEND", "openvino"),
                "ontology_classification": "openvino_npu_with_cpu_fallback",
                "deterministic_simulation": "cpu_worker_after_npu_screening",
                "risk_and_order_rules": "cpu_worker",
                "openvino_cache_dir": os.getenv("OPENVINO_CACHE_DIR", "data/runtime/openvino_cache"),
            },
        }
    )


@app.post("/api/operation-mode/start")
async def operation_mode_start(request: Request) -> JSONResponse:
    payload = await request.json()
    return _json(_operation_mode_start_response(payload))
    mode = str(payload.get("mode", "testing"))
    state = OperationModeManager().start(mode)
    _operation_mode_state["active"] = state
    audit.record("operation_mode_started", {"mode_state": state})
    
    result = _to_jsonable(state)
    result["mode_state"] = state
    
    if mode == "testing":
      target_return_rate = float(payload.get("target_return_rate", 0.02))
      if target_return_rate > 1:
        target_return_rate /= 100.0
      period_minutes = int(payload.get("period_minutes", 390))
      demo_id = _start_streaming_demo(
          target_return_rate=target_return_rate,
          period_minutes=period_minutes,
          initial_cash=float(payload.get("initial_cash", 10_000_000)),
          seed=int(payload.get("seed", 42)),
      )
      demo = _streaming_demos.get(demo_id)
      if demo:
        result["demo_id"] = demo_id
        result["demo_status"] = "initialized"
        result["target_return_rate"] = target_return_rate
        result["period_minutes"] = period_minutes
        result["initial_cash"] = float(payload.get("initial_cash", 10_000_000))
        result["universe_count"] = len(demo._bars_by_ticker)
        result["demo_message"] = "시뮬레이션 테스트가 시작되었습니다. 목표 수익률과 목표 시간으로 자동 진행됩니다."
    
    return _json(result)


def _operation_mode_start_response(payload: dict[str, Any]) -> dict[str, Any]:
    mode = str(payload.get("mode", "testing"))
    if not _operation_mode_lock.acquire(blocking=False):
      return {
          "ok": False,
          "status": "busy",
          "mode": mode,
          "message": "Another operation-mode request is still being prepared.",
          "request": _operation_mode_request_snapshot(),
      }
    try:
      _set_operation_request(True, "starting", f"Starting {mode}", None)
      state = OperationModeManager().start(mode)
      with _live_lock:
        _operation_mode_state["active"] = state
        _clear_live_analysis_cache_unlocked()
      audit.record("operation_mode_started", {"mode_state": state})

      result = _to_jsonable(state)
      result["ok"] = True
      result["status"] = "started"
      result["mode_state"] = state
      result["data_policy"] = _current_data_policy()

      if mode == "learning":
        _start_live_worker(mode)
        result["training_status"] = "continuous_collection_started"
        result["training_message"] = "Realtime learning will update model artifacts until you press the stop button."

      if mode == "testing":
        snapshot = _get_or_refresh_live(force_refresh=True)
        context = snapshot["context"]
        test_result = run_hypothetical_realtime_test(context.temporal_frames, context.signals)
        model_paths = update_realtime_model_artifacts(
            ModelArtifactStore(),
            build_realtime_supervised_examples(context.temporal_frames, context.signals),
            test_result,
        )
        result["test_status"] = "completed"
        result["test_result"] = test_result
        result["model_artifacts"] = model_paths
        result["test_message"] = "Realtime test completed with hypothetical trades only; no broker orders were submitted."

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
          "message": str(exc),
          "request": _operation_mode_request_snapshot(),
      }
    finally:
      _operation_mode_lock.release()


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
    return _json(
        {
            "active": _operation_mode_state.get("active"),
            "request": _operation_mode_request_snapshot(),
            "learning": _learning_state_snapshot(),
            "collection_log": _live_snapshot()["collection_log"],
            "streaming": streaming,
        }
    )


@app.post("/api/operation-mode/stop-learning")
async def operation_mode_stop_learning() -> JSONResponse:
    _stop_live_worker()
    with _live_lock:
      _operation_mode_state["active"] = None
      _clear_live_analysis_cache_unlocked()
    audit.record("learning_collection_stopped", {"stopped_at": datetime.now(timezone.utc).isoformat()})
    return _json(
        {
            "ok": True,
            "status": "stopped",
            "message": "Learning data collection has stopped.",
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


@app.post("/api/live-snapshot")
async def live_snapshot(request: Request) -> JSONResponse:
    payload = await request.json()
    goal_payload = payload.get("goal")
    force_refresh = bool(payload.get("force_refresh", False))
    return _json(await run_in_threadpool(_live_snapshot_response, goal_payload, force_refresh))


def _live_snapshot_response(goal_payload: Any, force_refresh: bool) -> dict[str, Any]:
    snapshot = _get_or_refresh_live(force_refresh=force_refresh)
    research_result = snapshot["research_result"]
    context = snapshot["context"]
    store_summary = dict(snapshot.get("store_summary") or {})
    lightweight_volume = _lightweight_data_volume(store_summary)
    graph_payload = snapshot.get("graph_payload")
    if graph_payload is None or snapshot.get("graph_payload_context_id") != id(context):
      graph_payload = _graph_payload(context)
      with _live_lock:
        if _live_state["context"] is context:
          _live_state["graph_payload"] = graph_payload
          _live_state["graph_payload_context_id"] = id(context)
    response: dict[str, Any] = {
        "status": {
            "cash": context.account.cash,
            "equity": context.report.equity,
            "cash_weight": context.report.cash_weight,
            "daily_pnl_ratio": context.report.daily_pnl_ratio,
            "updated_at": _iso_or_none(snapshot["last_updated"]),
            "last_error": snapshot["last_error"],
        },
        "diagnostics": {
            "research_config": str(DEFAULT_RESEARCH_CONFIG),
            "diagnostics": _diagnostics_with_collection_config(research_result.diagnostics),
            "skipped_sources": research_result.skipped_sources,
            "stored_new_records": snapshot["stored_new_records"],
            "store_summary": store_summary,
            "data_volume": lightweight_volume,
            "store_path": str(LocalResearchStore(root=_get_store_root()).db_path),
            "data_policy": _current_data_policy(),
            "graph_triples_count": len(context.graph.triples()),
            "reasoning_paths": context.reasoning_paths,
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
            context.account,
            context.markets,
            context.indicators,
            context.signals,
            context.graph,
        )
        response["assessment"] = assessment
        response["compromises"] = build_compromise_goals(assessment)
    return response


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
    goal_request = _parse_goal_request(payload)
    context = _get_or_refresh_live()["context"]
    assessment = assess_goal(
        goal_request,
        context.account,
        context.markets,
        context.indicators,
        context.signals,
        context.graph,
    )
    compromises = build_compromise_goals(assessment)
    session_id = str(uuid4())
    sessions[session_id] = {"assessment": assessment, "compromises": compromises, "started": False}
    audit.record("goal_assessment", {"session_id": session_id, "assessment": assessment})
    return _json({"session_id": session_id, "assessment": assessment, "compromises": compromises})


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


@app.get("/api/mock-kis/portfolio")
def mock_kis_portfolio() -> JSONResponse:
    context = _get_or_refresh_live()["context"]
    broker = _mock_kis_for_context(context)
    return _json(broker.get_portfolio())


@app.post("/api/streaming-demo/start")
async def streaming_demo_start(request: Request) -> JSONResponse:
    """스트리밍 데모를 시작합니다 (10배 가속)."""
    payload = await request.json()
    
    target_return_rate = float(payload.get("target_return_rate", 0.02))
    if "period_minutes" in payload:
      period_minutes = int(payload.get("period_minutes", 390))
    elif "period_days" in payload:
      period_minutes = int(payload.get("period_days", 7)) * 390
    else:
      period_minutes = 390
    initial_cash = max(100_000.0, float(payload.get("initial_cash", 10_000_000)))
    seed = int(payload.get("seed", 42))
    demo_id = _start_streaming_demo(
        target_return_rate=target_return_rate,
        period_minutes=period_minutes,
        initial_cash=initial_cash,
        seed=seed,
    )
    
    return _json({
        "demo_id": demo_id,
        "status": "initialized",
        "progress": 0.0,
        "message": "시뮬레이션 데모가 시작되었습니다. 단계별 진행하기 위해 /api/streaming-demo/step을 호출하세요.",
    })


@app.post("/api/streaming-demo/step")
async def streaming_demo_step(request: Request) -> JSONResponse:
    """스트리밍 데모를 한 스텝 진행합니다."""
    payload = await request.json()
    return _json(await run_in_threadpool(_streaming_demo_step_response, payload))
    demo_id = str(payload.get("demo_id", ""))
    
    if demo_id not in _streaming_demos:
        return _json({
            "demo_id": demo_id,
            "status": "expired",
            "progress": 0.0,
            "message": "Simulation session expired. Start a new simulation test.",
        })
    
    demo = _streaming_demos[demo_id]
    result = demo.run_step()
    
    if result is None:
        # 완료됨
        final = demo.get_final_results()
        return _json({
            "demo_id": demo_id,
            "status": "completed",
            "progress": 100.0,
            "message": "데모가 완료되었습니다.",
            "final_results": final,
        })
    
    # 스텝 결과 반환
    time_scaler = demo.get_time_scaler()
    virtual_time = time_scaler.get_virtual_time() if time_scaler else None
    final = demo.get_final_results() if demo.is_complete() else None
    
    return _json({
        "demo_id": demo_id,
        "status": "completed" if final is not None else "running",
        "universe_count": result.universe_ticker_count,
        "active_ticker_count": result.active_ticker_count,
        "step": result.visible_step,
        "raw_step": result.step_index,
        "chart_bar": result.step_index,
        "progress": result.progress_percent,
        "timestamp": result.timestamp,
        "virtual_time": virtual_time,
        "prices": result.prices,
        "account": {
            "cash": result.cash,
            "account_value": result.account_value,
            "return_rate": result.return_rate,
        },
        "holdings": result.holdings,
        "trades_in_step": len(result.trades_in_step),
        "cumulative_trades": result.cumulative_trades,
        "trades": [_to_jsonable(t) for t in result.trades_in_step],
        "final_results": final,
    })


def _streaming_demo_step_response(payload: dict[str, Any]) -> dict[str, Any]:
    demo_id = str(payload.get("demo_id", ""))
    if demo_id not in _streaming_demos:
        return {
            "demo_id": demo_id,
            "status": "expired",
            "progress": 0.0,
            "message": "Simulation session expired. Start a new simulation test.",
        }

    step_lock = _streaming_demo_step_locks.setdefault(demo_id, threading.Lock())
    if not step_lock.acquire(blocking=False):
        demo = _streaming_demos.get(demo_id)
        return {
            "demo_id": demo_id,
            "status": "busy",
            "progress": demo.get_progress() if demo is not None else 0.0,
            "message": "Simulation step is already running.",
        }

    try:
        demo = _streaming_demos[demo_id]
        wait_seconds = demo.seconds_until_next_step()
        if wait_seconds > 0:
            return {
                "demo_id": demo_id,
                "status": "waiting",
                "progress": demo.get_progress(),
                "seconds_until_next_step": round(wait_seconds, 1),
                "retry_after_seconds": round(wait_seconds, 1),
                "message": "Waiting for the next real-time one-minute simulation bar.",
            }
        result = demo.run_step()
        candidate_selection = demo.get_candidate_selection()
        if result is None:
            return {
                "demo_id": demo_id,
                "status": "completed",
                "progress": 100.0,
                "message": "Simulation completed.",
                "final_results": demo.get_final_results(),
                "ontology_filter_1": _candidate_selection_payload(candidate_selection),
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
            },
            "holdings": result.holdings,
            "trades_in_step": len(result.trades_in_step),
            "cumulative_trades": result.cumulative_trades,
            "trades": [_to_jsonable(t) for t in result.trades_in_step],
            "stored_realtime_records": saved_realtime,
            "final_results": final,
            "ontology_filter_1": _candidate_selection_payload(candidate_selection),
        }
    finally:
        step_lock.release()


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


@app.get("/api/streaming-demo/status/{demo_id}")
def streaming_demo_status(demo_id: str) -> JSONResponse:
    """스트리밍 데모의 현재 상태를 조회합니다."""
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


@app.post("/api/streaming-demo/pause/{demo_id}")
async def streaming_demo_pause(demo_id: str) -> JSONResponse:
    """스트리밍 데모를 일시 정지합니다."""
    if demo_id not in _streaming_demos:
        raise HTTPException(status_code=404, detail="Demo not found")
    
    demo = _streaming_demos[demo_id]
    demo.pause()
    
    return _json({
        "demo_id": demo_id,
        "status": "paused",
        "is_paused": True,
    })


@app.post("/api/streaming-demo/resume/{demo_id}")
async def streaming_demo_resume(demo_id: str) -> JSONResponse:
    """일시 정지된 스트리밍 데모를 다시 시작합니다."""
    if demo_id not in _streaming_demos:
        raise HTTPException(status_code=404, detail="Demo not found")
    
    demo = _streaming_demos[demo_id]
    demo.resume()
    
    return _json({
        "demo_id": demo_id,
        "status": "resumed",
        "is_paused": False,
    })


@app.post("/api/streaming-demo/cleanup/{demo_id}")
async def streaming_demo_cleanup(demo_id: str) -> JSONResponse:
    """스트리밍 데모를 정리합니다."""
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
    )

    if goal_mode and goal_mode != "rate":
        raise HTTPException(
            status_code=400,
            detail="목표 수익률만 사용합니다.",
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
        )
        assessment = assess_goal(
            request,
            context.account,
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
    if context.account.equity >= 5_000_000:
        return context.account
    return AccountSnapshot(
        cash=10_000_000,
        holdings=context.account.holdings,
        realized_pnl_today=context.account.realized_pnl_today,
        unrealized_pnl_today=context.account.unrealized_pnl_today,
        captured_at=datetime.now(timezone.utc),
    )


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
  snapshot = _live_snapshot()
  if snapshot["is_refreshing"]:
    return
  worker = threading.Thread(target=_refresh_live_cache, name="operation-mode-refresh", daemon=True)
  worker.start()


def _start_live_worker(learning_mode: str | None = None) -> None:
  global _live_worker
  with _live_lock:
    _live_state["stop"] = False
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
    if _live_worker is not None and _live_worker.is_alive():
      return
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


def _live_worker_loop() -> None:
  _refresh_live_cache()
  while True:
    with _live_lock:
      if _live_state["stop"]:
        break
      learning_active = bool(_live_state.get("learning_active"))
      interval_seconds = LEARNING_COLLECTION_INTERVAL_SECONDS if learning_active else LIVE_REFRESH_SECONDS
      next_at = datetime.now(timezone.utc) + timedelta(seconds=interval_seconds)
      _live_state["learning_next_collection_at"] = next_at if learning_active else None
    if learning_active:
      _set_live_progress(100, "waiting", f"Next internet and chart data collection starts at {next_at.astimezone().strftime('%H:%M')}")
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
      _set_live_progress(18, "research", "Collecting configured market, news, and macro sources")
      research_result = _load_default_research()
      _set_live_progress(48, "storage", "Saving research records")
      stored_counts = store.save_research_result(research_result)
      _set_live_progress(64, "analysis", "Building indicators, ontology graph, and reasoning paths")
      context = build_analysis_context(research_result, _analysis_research_for_current_mode(store))
      model_paths: dict[str, str] = {}
      realtime_examples = build_realtime_supervised_examples(context.temporal_frames, context.signals)
      if learning_mode == "learning":
        _set_live_progress(76, "learning", "Updating realtime supervised model artifacts")
        model_paths = update_realtime_model_artifacts(ModelArtifactStore(), realtime_examples)
      elif learning_mode == "testing":
        _set_live_progress(76, "testing", "Calculating hypothetical realized PnL without orders")
        test_result = run_hypothetical_realtime_test(context.temporal_frames, context.signals)
        model_paths = update_realtime_model_artifacts(ModelArtifactStore(), realtime_examples, test_result)
      _set_live_progress(84, "graph", "Persisting ontology graph and reasoning paths")
      graph_counts = store.save_graph_and_reasoning(context.graph.triples(), context.reasoning_paths)
      with _live_lock:
        _live_state["research_result"] = research_result
        _live_state["context"] = context
        _live_state["context_mode"] = _active_operation_mode()
        _live_state["graph_payload"] = _graph_payload(context)
        _live_state["graph_payload_context_id"] = id(context)
        _live_state["store_summary"] = store.summary()
        _live_state["stored_new_records"] = {**stored_counts, **graph_counts}
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
                "events_seen": len(research_result.events),
                "raw_records_seen": len(research_result.raw_records),
                "market_snapshots_seen": len(research_result.market_snapshots),
                "macro_metrics_seen": len(research_result.macro_metrics),
                "temporal_frames": len(context.temporal_frames),
                "supervised_examples": len(realtime_examples),
                "model_artifacts": len(model_paths),
            },
        )
      with _live_lock:
        keep_active = bool(_live_state.get("learning_active")) and not bool(_live_state.get("stop"))
      _set_live_progress(
          100,
          "complete",
          "Learning collection cycle completed; continuing until stop is pressed"
          if keep_active
          else "Live analysis cache is ready",
          active=keep_active,
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
      with _live_lock:
        _live_state["is_refreshing"] = False


def _get_or_refresh_live(force_refresh: bool = False) -> dict[str, Any]:
  snapshot = _live_snapshot()
  current_mode = _active_operation_mode()
  cache_matches_mode = snapshot.get("context_mode") == current_mode
  if snapshot["context"] is not None and cache_matches_mode and not force_refresh:
    return snapshot
  if (snapshot["context"] is None or not cache_matches_mode) and not force_refresh:
    _build_current_snapshot_from_store()
    return _live_snapshot()
  last_updated = snapshot["last_updated"]
  stale = (
    last_updated is None
    or (datetime.now() - last_updated).total_seconds() > LIVE_STALE_SECONDS
  )
  if force_refresh or stale or snapshot["context"] is None:
    _refresh_live_cache()
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


def _graph_payload(context: Any) -> dict[str, Any]:
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

    return {
        "nodes": list(nodes.values()),
        "links": links,
        "reasoning_steps": _build_reasoning_steps(context.reasoning_paths),
        "counts": {"nodes": len(nodes), "links": len(links)},
        "runtime": context.ontology_runtime.as_dict(),
        "candidate_selection": _candidate_selection_payload(getattr(context, "candidate_selection", None)),
        "parameter_tuning": tuple(getattr(context, "parameter_tuning", ()) or ()),
        "temporal_frame_count": len(tuple(getattr(context, "temporal_frames", ()) or ())),
    }


def _node_payload(
    node_id: str,
    importance_score: float,
    event_meta: dict[str, Any] | None = None,
    kind_override: str | None = None,
) -> dict[str, Any]:
    kind = kind_override or _node_kind(node_id)
    payload = {
        "id": node_id,
        "label": event_meta.get("title", node_id) if event_meta else node_id,
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
    }:
        return "pipeline"
    if node_id.startswith("OntologyTuningMode:") or node_id == "MarketInterpretationParameterTuning":
        return "tuning"
    if node_id.startswith("Parameter:") or node_id.startswith("TunedValue:"):
        return "parameter"
    if node_id.startswith("UniverseCount:") or node_id.startswith("CandidateCount:"):
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
    }:
        return "support"
    if node_id in {
        "MacroRateRisk",
        "NegativeEventRisk",
        "VolatilityRisk",
        "LiquidityRisk",
        "OrderFlowDistributionRisk",
        "ThinLiquidityPriceImpactRisk",
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
                "title": "결론 산출",
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
  <link rel="icon" href="data:,">
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
    @media (max-width: 900px) {
      .shell { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .span-4, .span-8 { grid-column: span 12; }
      .cards { grid-template-columns: 1fr; }
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
      <h1>개인 투자 분석 시스템</h1>
      <div class="status" id="gate">목표가 확정될 때까지 프로그램은 시작되지 않습니다.</div>
      <h2>운영 모드</h2>
      <div class="mode-step-label">실시간 통합 데이터 기준</div>
      <div class="mode-grid" id="modeActionGrid">
        <button type="button" id="modeLearningButton" onclick="startSelectedOperationMode('training')">학습<small>실시간 데이터 + 손익 라벨</small></button>
        <button type="button" id="modeLearningStopButton" class="secondary" onclick="stopLearningCollection()" disabled>학습 종료<small>수집 중지</small></button>
        <button type="button" id="modeTestingButton" onclick="startSelectedOperationMode('testing')">테스트<small>실제 주문 없이 가상 손익</small></button>
        <button type="button" id="modeLiveButton" onclick="startOperationMode('live_trading')">실전<small>리스크 게이트 적용</small></button>
      </div>
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
          <div class="flow-step" data-flow-step="mode"><i class="flow-dot"></i><div><strong>모드</strong><span>학습/테스트 선택 대기</span></div></div>
          <div class="flow-step" data-flow-step="data"><i class="flow-dot"></i><div><strong>데이터</strong><span>실시간 자료 상태 확인 중</span></div></div>
          <div class="flow-step" data-flow-step="analysis"><i class="flow-dot"></i><div><strong>분석</strong><span>목표 가능성 또는 전략 계산 대기</span></div></div>
          <div class="flow-step" data-flow-step="simulation"><i class="flow-dot"></i><div><strong>모의 진행</strong><span>시뮬레이션 대기</span></div></div>
        </div>
      </div>
      <form id="goalForm">
        <div class="field"><label for="targetReturn">목표 수익률 (%)</label><input id="targetReturn" name="target_return_rate" type="number" step="0.1" min="0" value="2" placeholder="예: 5"></div>
        <div class="field"><label for="targetMinutes">목표 시간 (분)</label><input id="targetMinutes" name="period_minutes" type="number" min="1" step="1" value="390" placeholder="예: 390"></div>
        <div class="field"><label for="initialCash">시뮬레이션 예수금 (원)</label><input id="initialCash" name="initial_cash" type="number" min="100000" step="100000" value="10000000" placeholder="예: 10000000"></div>
        <button type="submit">가능성 분석</button>
        <button class="secondary" id="loadResearch" type="button">자료 불러오기</button>
        <div class="work-status" id="workStatus">
          <strong id="workTitle">작업 대기 중</strong>
          <div class="muted" id="workMessage">버튼을 누르면 진행 현황이 표시됩니다.</div>
          <div class="bar"><span id="workProgress"></span></div>
        </div>
      </form>
      <hr style="margin: 20px 0; border: none; border-top: 1px solid var(--line);">
      <h2>시뮬레이션 테스트</h2>
      <div class="note">시뮬레이션 테스트는 위에서 입력한 목표 수익률과 목표 시간으로 자동 진행됩니다.</div>
      <div class="work-status active" id="streamingDemoContainer" style="display:none;">
        <strong id="streamingDemoStatus">대기 중</strong>
        <div class="bar"><span id="streamingDemoProgress" style="width:0%"></span></div>
        <div style="margin-top: 10px; font-size: 12px;">
          <div class="score-row">
            <span class="score-label">예수금:</span>
            <span class="score-value" id="streamingDeposit">₩0</span>
          </div>
          <div class="score-row">
            <span class="score-label">투자금:</span>
            <span class="score-value" id="streamingInvested">₩0</span>
          </div>
          <div class="score-row">
            <span class="score-label">수익금:</span>
            <span class="score-value" id="streamingProfit">₩0</span>
          </div>
          <div class="score-row">
            <span class="score-label">수익률:</span>
            <span class="score-value" id="streamingReturnRate">0%</span>
            <canvas class="mini-chart" id="streamingReturnChart" width="280" height="74"></canvas>
          </div>
        </div>
      </div>
      </form>
    </aside>
    <main>
      <div class="grid">
        <section class="panel span-4"><h2>포트폴리오</h2><div class="metric" id="equity">-</div><div class="muted">총 평가금액</div><div class="chips" style="margin-top:12px;"><span class="chip" id="cash">예치금 -</span><span class="chip" id="cashWeight">현금 비중 -</span></div></section>
        <section class="panel span-4"><h2>모의 수익률</h2><div class="metric" id="mockReturn">대기 중</div><div class="bar"><span id="mockReturnBar"></span></div><div class="chips" style="margin-top:12px;"><span class="chip" id="mockProfit">수익금 -</span><span class="chip" id="mockEquity">평가금 -</span><span class="chip" id="mockTarget">목표 -</span></div><p class="muted" id="mockStatus" style="margin-bottom:0;">선택한 목표로 시작하면 모의 KIS 포트폴리오 수익률이 표시됩니다.</p></section>
        <section class="panel span-8">
          <h2>달성 가능성</h2>
          <div class="metric" id="feasibility">대기 중</div>
          <div class="bar"><span id="feasibilityBar"></span></div>
          <div id="scoreBreakdown" style="margin-top:14px;"></div>
          <p class="muted" id="summary">목표를 입력하면 시장 자료, 온톨로지 관계, 리스크 압력을 바탕으로 달성 가능성을 계산합니다.</p>
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
            <span class="ontology-badge">3D 온톨로지 네트워크</span>
            <span class="ontology-badge" id="ontologyCounts">노드 - · 관계 -</span>
            <button id="resetGraph" type="button">시점 초기화</button>
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
            <div id="reasoningDescription">활성화되는 노드와 엣지가 밝게 빛나며 현재 판단 근거를 보여줍니다.</div>
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
        <section class="panel span-12"><h2>목표 타협안</h2><div class="cards" id="choices"></div><div style="margin-top:14px;"><button id="startButton" disabled>선택한 목표로 시작</button> <button class="secondary" id="resetButton" type="button">초기화</button></div></section>
        <section class="panel span-12"><h2>실시간 모의 진행</h2><div class="stats" id="mockRunStats"></div><div class="grid" style="margin-top:12px;"><div class="span-12"><h2>최근 체결 및 종료 청산</h2><div class="table-wrap"><table class="live-table"><thead><tr><th>구분</th><th>종목</th><th>수량</th><th>가격/금액</th></tr></thead><tbody id="mockExecutions"><tr><td colspan="4">체결 내역 없음</td></tr></tbody></table></div><div style="margin-top: 12px;"><h2>스트리밍 데모 거래</h2><div class="table-wrap"><table class="live-table"><thead><tr><th>종목</th><th>구분</th><th>수량</th><th>금액</th></tr></thead><tbody id="streamingTradeList"><tr><td colspan="4">거래 없음</td></tr></tbody></table></div></div></div></div></section>
        <section class="panel span-4"><h2>온톨로지 신호</h2><div class="chips" id="relations"></div></section>
        <section class="panel span-8"><h2>자료 및 프로그램 출력</h2><div class="log" id="output">아직 실행되지 않았습니다.</div></section>
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
    let learningStatusBusy = false;
    let learningStatusTimer = null;
    let lastRenderedCollectionCycle = null;
    let mockPerformanceTimer = null;
    let operationRequestActive = false;
    let streamingStepBusy = false;
    let streamingStepFailures = 0;
    let streamingDemoTimer = null;
    let streamingReturnSeries = [];
    const fmtWon = new Intl.NumberFormat('ko-KR', { style: 'currency', currency: 'KRW', maximumFractionDigits: 0 });

    async function loadStatus() {
      const data = await (await fetch('/api/status')).json();
      renderStatus(data);
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
          renderSystemFlow({ data: 'done', analysis: 'done' }, { data: 'Data cache ready', analysis: 'Analysis cache ready' });
        }
        /*
        if (progress.stage === 'error') {
          renderSystemFlow({ data: 'error' }, { data: progress.message || '데이터 갱신 실패' });
        } else if (data.is_refreshing || progress.active) {
          renderSystemFlow({ data: 'active' }, { data: progress.message || '데이터 갱신 중' });
        } else if ((progress.percent || 0) >= 100) {
          renderSystemFlow({ data: 'done', analysis: 'done' }, { data: '데이터 캐시 준비 완료', analysis: '분석 캐시 준비 완료' });
        }
        */
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
      const data = await (await fetch('/api/research/diagnostics')).json();
      renderDiagnostics(data);
    }

    async function loadOntologyGraph() {
      const data = await (await fetch('/api/ontology/graph')).json();
      document.getElementById('ontologyCounts').textContent = `노드 ${data.counts.nodes} · 관계 ${data.counts.links}`;
      lastGraphSignature = graphSignature(data);
      await renderOntologyGraph(data);
    }

    async function loadRealtimeRuntime() {
      const data = await (await fetch('/api/realtime/runtime')).json();
      const accel = data.acceleration || {};
      const policy = data.short_horizon_policy || {};
      const eventLlm = data.event_llm || {};
      const ontologyNpu = data.ontology_npu || {};
      const npuLabel = accel.uses_npu ? 'NPU 사용 중' : `CPU fallback (${accel.active_backend || '-'})`;
      const llmLabel = eventLlm.available ? `LLM ${eventLlm.provider || '-'} ready` : `LLM 대기: ${eventLlm.reason || 'not configured'}`;
      const ontologyNpuLabel = ontologyNpu.uses_npu
        ? `온톨로지 NPU ${ontologyNpu.last_items || 0}건 ${ontologyNpu.last_latency_ms ? `${ontologyNpu.last_latency_ms}ms` : 'ready'}`
        : `온톨로지 NPU fallback ${ontologyNpu.backend || '-'}`;
      document.getElementById('runtimeStatus').textContent =
        `${npuLabel} · ${ontologyNpuLabel} · ${llmLabel} · ${accel.latency_profile || 'low_latency'} · 예측 ${((accel.prediction_horizons_seconds || []).join('/'))}초 · 포지션 cap ${((policy.max_position_weight_intraday || 0) * 100).toFixed(1)}%`;
      if (data.operation_mode) renderOperationMode(data.operation_mode);
    }

    function selectedOperationMode(action) {
      if (action === 'training') return 'learning';
      if (action === 'testing') return 'testing';
      return 'live_trading';
    }

    function updateModeButtons() {
      const learningButton = document.getElementById('modeLearningButton');
      const stopLearningButton = document.getElementById('modeLearningStopButton');
      const testingButton = document.getElementById('modeTestingButton');
      const liveButton = document.getElementById('modeLiveButton');
      if (!learningButton || !testingButton) return;
      learningButton.disabled = operationRequestActive;
      if (stopLearningButton) stopLearningButton.disabled = true;
      testingButton.disabled = operationRequestActive;
      if (liveButton) liveButton.disabled = operationRequestActive;
      learningButton.innerHTML = '학습<small>실시간 데이터 + 손익 라벨</small>';
      testingButton.innerHTML = '테스트<small>실제 주문 없이 가상 손익</small>';
    }

    function setModeButtonsLocked(locked) {
      const learningButton = document.getElementById('modeLearningButton');
      const stopLearningButton = document.getElementById('modeLearningStopButton');
      const testingButton = document.getElementById('modeTestingButton');
      const liveButton = document.getElementById('modeLiveButton');
      if (!learningButton || !testingButton) return;
      const enabled = !locked;
      learningButton.disabled = !enabled;
      if (stopLearningButton && locked) stopLearningButton.disabled = true;
      testingButton.disabled = !enabled;
      if (liveButton) liveButton.disabled = !enabled;
    }

    function renderSystemFlow(states = {}, messages = {}) {
      document.querySelectorAll('#systemFlowPanel .flow-step').forEach((step) => {
        const key = step.dataset.flowStep;
        const state = states[key] || 'idle';
        step.classList.remove('active', 'done', 'error');
        if (state === 'active' || state === 'done' || state === 'error') {
          step.classList.add(state);
        }
        const label = step.querySelector('span');
        if (label && messages[key]) label.textContent = messages[key];
      });
    }

    function modeLabel(mode) {
      return {
        learning: '학습',
        testing: '테스트',
        live_trading: '실전 테스트'
      }[mode] || mode || '운영 모드';
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

    async function startOperationMode(mode, options = {}) {
      if (operationRequestActive) {
        document.getElementById('runtimeStatus').textContent = '이미 모드 시작 요청을 처리 중입니다.';
        return;
      }
      operationRequestActive = true;
      renderSystemFlow({
        environment: 'done',
        mode: 'active',
        data: 'idle',
        analysis: 'idle',
        simulation: mode === 'testing' ? 'active' : 'idle',
      }, {
        mode: `${modeLabel(mode)} 시작 요청 중`,
        simulation: mode === 'testing' ? '테스트 손익 계산 준비 중' : '테스트 대기',
      });
      document.getElementById('operationModeStatus').textContent = `${modeLabel(mode)} 시작 요청 중...`;
      document.getElementById('runtimeStatus').textContent = '서버 응답을 기다리는 중입니다.';
      document.getElementById('output').textContent = `${modeLabel(mode)} 시작 요청을 보냈습니다.`;
      const learningButton = document.getElementById('modeLearningButton');
      const testingButton = document.getElementById('modeTestingButton');
      setModeButtonsLocked(true);
      try {
        const data = await fetchJsonWithTimeout('/api/operation-mode/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode, ...options })
        }, 45000);
        if (data.ok === false) {
          throw new Error(data.message || data.status || 'operation mode request failed');
        }
        renderOperationMode(data);
        updateLearningStopButton(data.learning);
        renderSystemFlow({
          environment: 'done',
          mode: 'done',
          data: mode === 'learning' ? 'active' : 'done',
          analysis: 'idle',
          simulation: mode === 'testing' ? 'done' : 'idle',
        }, {
          mode: `${modeLabel(mode)} 시작됨`,
          data: mode.includes('training') ? '학습 데이터 갱신 중' : '데이터 준비 완료',
          simulation: mode === 'testing' ? '가상 실현손익 계산 완료' : '테스트 대기',
        });
        document.getElementById('output').textContent = JSON.stringify(data, null, 2);
        if (false && mode === 'testing' && data.demo_id) {
          document.getElementById('streamingDemoContainer').style.display = 'block';
          streamingDemoId = data.demo_id;
          streamingDemoRunning = true;
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
            account: {
              cash: streamingInitialCash,
              account_value: streamingInitialCash,
              return_rate: 0,
            },
            status: 'running',
          });
          drawStreamingReturnChart();
          autoRunStreamingDemo(true);
        }
        loadRealtimeRuntime().catch((error) => {
          document.getElementById('runtimeStatus').textContent = `NPU 상태 확인 실패: ${error.message || error}`;
        });
      } catch (error) {
        const message = error.name === 'AbortError'
          ? '서버 응답 시간이 초과되었습니다. 잠시 후 다시 눌러주세요.'
          : String(error && error.message ? error.message : error);
        document.getElementById('operationModeStatus').textContent = `${modeLabel(mode)} 시작 실패`;
        document.getElementById('runtimeStatus').textContent = message;
        document.getElementById('output').textContent = message;
        renderSystemFlow({ mode: 'error' }, { mode: message });
        loadOperationModeStatus().catch(() => {});
      } finally {
        operationRequestActive = false;
        setModeButtonsLocked(false);
      }
    }

    async function startSelectedOperationMode(action) {
      const mode = selectedOperationMode(action);
      if (!mode) return;
      const goal = currentGoalPayload();
      if (!goal && action === 'testing') {
        document.getElementById('output').textContent = '목표 수익률과 목표 시간(분)을 먼저 입력하세요.';
        return;
      }
      const options = goal ? {
        target_return_rate: goal.target_return_rate,
        period_minutes: goal.period_minutes,
        initial_cash: goal.initial_cash,
      } : {};
      await startOperationMode(mode, options);
    }

    async function stopLearningCollection() {
      const stopLearningButton = document.getElementById('modeLearningStopButton');
      if (stopLearningButton) stopLearningButton.disabled = true;
      document.getElementById('learningStatusMessage').textContent = '학습 데이터 수집을 종료하는 중입니다.';
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
      const labels = {
        learning: '학습',
        testing: '테스트',
        live_trading: '실제 투자 진행'
      };
      const mode = labels[data.mode] || data.mode || '모드 대기';
      document.getElementById('operationModeStatus').textContent =
        `${mode} · ${data.execution_label || ''}` + (data.demo_message ? ` · ${data.demo_message}` : '');
      document.getElementById('gate').textContent =
        data.mode === 'learning'
          ? '실제 투자 학습 모드입니다. 현 시장 데이터는 학습에만 사용되고 주문은 금지됩니다.'
          : data.mode === 'live_trading'
            ? '실제 투자 진행 모드입니다. 자동 주문은 차단되고 리스크/승인 게이트가 우선합니다.'
            : '테스트 모드입니다. 실제 주문 없이 가상 체결 손익만 계산합니다.';
    }

    async function loadOperationModeStatus() {
      const data = await fetchJsonWithTimeout('/api/operation-mode/status', {}, 8000);
      const request = data.request || {};
      if (request.busy) {
        renderSystemFlow({ mode: 'active' }, { mode: request.message || '요청 처리 중' });
      } else if (request.stage === 'error') {
        renderSystemFlow({ mode: 'error' }, { mode: request.last_error || request.message || '요청 실패' });
      }
      updateLearningStopButton(data.learning);
      if (Array.isArray(data.collection_log)) {
        renderCollectionLog(data.collection_log);
        maybeRefreshDiagnosticsAfterCollection(data.collection_log);
      }
      if (data.active) renderOperationMode(data.active);
      return data;
    }

    document.getElementById('goalForm').addEventListener('submit', async (event) => {
      event.preventDefault();
      const payload = currentGoalPayload();
      if (!payload) {
        document.getElementById('output').textContent = '목표 수익률과 목표 시간(분)을 입력하세요.';
        return;
      }
      const stopProgress = startLocalProgress('가능성 분석 중', '현재 확보된 스냅샷으로 목표 가능성을 계산하고 있습니다.');
      setBusy(true);
      try {
        lastGoalPayload = payload;
        const res = await fetch('/api/assess-goal', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        if (!res.ok) throw new Error(await res.text());
        renderAssessment(await res.json());
        setWorkStatus('가능성 분석 완료', '목표 달성 가능성과 타협안을 계산했습니다.', 100, true);
      } catch (error) {
        document.getElementById('output').textContent = String(error && error.message ? error.message : error);
        setWorkStatus('가능성 분석 실패', String(error && error.message ? error.message : error), 100, true);
      } finally {
        stopProgress();
        setBusy(false);
      }
    });

    document.getElementById('loadResearch').addEventListener('click', async () => {
      const stopProgress = startProgressPolling('자료 수집 시작 중', '전체 상장 universe와 설정된 자료 수집을 준비하고 있습니다.');
      setBusy(true);
      try {
        const data = await (await fetch('/api/research/refresh', { method: 'POST' })).json();
        document.getElementById('output').textContent = JSON.stringify(data, null, 2);
        await loadDiagnostics();
        setWorkStatus('자료 수집 시작', '수집은 백그라운드에서 진행됩니다. 진행률 패널에서 상태를 확인하세요.', 35, false);
      } catch (error) {
        document.getElementById('output').textContent = String(error && error.message ? error.message : error);
        setWorkStatus('자료 불러오기 실패', String(error && error.message ? error.message : error), 100, true);
      } finally {
        stopProgress();
        setBusy(false);
      }
    });

    function renderAssessment(data, options = {}) {
      if (data.session_id) sessionId = data.session_id;
      const previousSelection = selectedGoal;
      if (!options.preserveSelection) selectedGoal = null;
      const assessment = data.assessment;
      document.getElementById('feasibility').textContent = `${assessment.feasibility_percent}%`;
      document.getElementById('feasibilityBar').style.width = `${assessment.feasibility_percent}%`;
      document.getElementById('scoreBreakdown').innerHTML = `
        ${scoreRow('시장 지지', assessment.market_support_percent, 'good')}
        ${scoreRow('리스크 압력', assessment.risk_pressure_percent, 'warn')}
        ${scoreRow('목표 난이도', assessment.annualized_drag_percent, 'bad')}
        ${scoreRow('연환산 목표', Math.min(100, assessment.annualized_required_return * 100), 'bad', `${(assessment.annualized_required_return * 100).toFixed(1)}%`)}
      `;
      document.getElementById('summary').textContent = assessment.reasoning.join(' ');
      document.getElementById('relations').innerHTML = assessment.ontology_relations.slice(0, 10).map((item) => `<span class="chip">${item}</span>`).join('');
      const choices = document.getElementById('choices');
      choices.innerHTML = '';
      data.compromises.forEach((goal, index) => {
        const div = document.createElement('div');
        div.className = 'choice';
        div.innerHTML = `<strong>${translateGoalLabel(goal.label)}</strong><div class="metric">${goal.feasibility_percent}%</div><div class="muted">수익률 ${(goal.target_return_rate * 100).toFixed(2)}%</div><div class="muted">수익금 ${fmtWon.format(goal.target_profit_amount)}</div><div class="muted">기간 ${goal.period_days}일</div>`;
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
      const initialCash = Number(payload.initial_cash || 0);
      if (!targetReturnRate || targetReturnRate < 0) return null;
      if (!periodMinutes || periodMinutes < 1) return null;
      if (!initialCash || initialCash < 100000) return null;
      return {
        target_return_rate: targetReturnRate,
        period_minutes: periodMinutes,
        initial_cash: initialCash,
        period_days: Math.max(1, Math.ceil(periodMinutes / 390)),
      };
    }

    function applyUrlGoalParams() {
      const params = new URLSearchParams(window.location.search);
      const targetReturn = params.get('target_return_rate');
      const periodMinutes = params.get('period_minutes');
      const initialCash = params.get('initial_cash');
      if (targetReturn !== null) {
        document.getElementById('targetReturn').value = targetReturn;
      }
      if (periodMinutes !== null) {
        document.getElementById('targetMinutes').value = periodMinutes;
      }
      if (initialCash !== null) {
        document.getElementById('initialCash').value = initialCash;
      }
      const goal = currentGoalPayload();
      if (goal) lastGoalPayload = goal;
    }

    function applyGoalPayloadToMode(payload) {
      if (!payload) return {};
      return {
        target_return_rate: payload.target_return_rate,
        period_minutes: payload.period_minutes,
        initial_cash: payload.initial_cash,
      };
    }

    function currentGoalPayloadForAssessment() {
      const payload = currentGoalPayload();
      if (!payload) return null;
      return {
        target_return_rate: payload.target_return_rate,
        period_days: payload.period_days,
      };
    }

    function updateModeActionCopy() {
      updateModeButtons();
      document.getElementById('operationModeStatus').textContent =
        '실시간 통합 데이터 기준으로 학습, 테스트, 실전 중 하나를 선택하세요.';
      return;
      /*
      const state = 'realtime'
        ? '시뮬레이션'
        : 'realtime' === 'unused'
          ? '실전'
          : '환경';
      const label = true
        ? `${state} 환경을 선택했습니다. 다음으로 학습 또는 테스트를 고르세요.`
        : '먼저 실전 또는 시뮬레이션을 선택하세요.';
      document.getElementById('operationModeStatus').textContent = label;
      */
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
      if (streamingDemoRunning || operationRequestActive) {
        return;
      }
      if (liveRefreshBusy) return;
      liveRefreshBusy = true;
      const badge = document.getElementById('liveRefreshBadge');
      try {
        const goal = currentGoalPayload();
        if (goal) lastGoalPayload = goal;
        const res = await fetch('/api/live-snapshot', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ goal: lastGoalPayload, force_refresh: false }),
          signal: AbortSignal.timeout(12000),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        renderStatus(data.status);
        renderDiagnostics(data.diagnostics);
        const signature = graphSignature(data.graph);
        document.getElementById('ontologyCounts').textContent = `노드 ${data.graph.counts.nodes} · 관계 ${data.graph.counts.links}`;
        if (signature !== lastGraphSignature) {
          lastGraphSignature = signature;
          await renderOntologyGraph(data.graph);
        }
        if (data.assessment && data.compromises) {
          renderAssessment({ assessment: data.assessment, compromises: data.compromises }, { preserveSelection: true });
        }
        const updatedText = data.updated_at ? new Date(data.updated_at).toLocaleTimeString('ko-KR') : '대기 중';
        const errorText = data.status && data.status.last_error ? ` · 오류 ${data.status.last_error}` : '';
        badge.textContent = `마지막 갱신 ${updatedText}${errorText}`;
      } catch (error) {
        badge.textContent = '갱신 실패';
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
      document.getElementById('equity').textContent = fmtWon.format(data.equity);
      document.getElementById('cash').textContent = `예치금 ${fmtWon.format(data.cash)}`;
      document.getElementById('cashWeight').textContent = `현금 비중 ${(data.cash_weight * 100).toFixed(1)}%`;
    }

    function renderLearningStatus(data) {
      const progress = data.progress || {};
      const learning = data.learning || {};
      const percent = Math.max(0, Math.min(100, Number(progress.percent) || 0));
      const active = Boolean(progress.active || data.is_refreshing || learning.active);
      const stageLabels = {
        idle: '대기',
        starting: '시작',
        research: '자료 수집',
        storage: '저장',
        analysis: '분석',
        graph: '그래프 반영',
        waiting: '다음 수집 대기',
        complete: '완료',
        error: '오류',
      };
      const stage = stageLabels[progress.stage] || progress.stage || '대기';
      const message = prettyLearningMessage(progress.message || '실시간 상태를 확인하는 중입니다.');
      document.getElementById('learningStatusTitle').textContent = active ? '학습 진행 중' : '학습 대기 중';
      document.getElementById('learningStatusMessage').textContent = message;
      document.getElementById('learningStatusProgress').style.width = `${percent}%`;
      document.getElementById('learningStatusMeta').textContent = active
        ? `${stage} · ${percent.toFixed(1)}% · ${data.updated_at ? new Date(data.updated_at).toLocaleTimeString('ko-KR') : '갱신 중'}`
        : `${stage} · ${data.updated_at ? new Date(data.updated_at).toLocaleTimeString('ko-KR') : '대기 중'}`;
      document.getElementById('learningStatusCard').classList.toggle('active', active || percent > 0);
    }

    function updateLearningStopButton(learning = {}) {
      const stopLearningButton = document.getElementById('modeLearningStopButton');
      if (stopLearningButton) {
        stopLearningButton.disabled = !Boolean(learning && learning.active);
      }
    }

    function renderCollectionLog(log = []) {
      const entries = Array.isArray(log) ? log.slice(-24) : [];
      const list = document.getElementById('learningCollectionLog');
      if (list) {
        if (!entries.length) {
          list.innerHTML = '<div class="muted">수집 로그가 아직 없습니다.</div>';
        } else {
          list.innerHTML = entries.slice().reverse().slice(0, 8).map((entry) => {
            const counts = entry.counts || {};
            const totalSeen = Number(counts.events_seen || 0)
              + Number(counts.raw_records_seen || 0)
              + Number(counts.market_snapshots_seen || 0)
              + Number(counts.macro_metrics_seen || 0);
            const totalStored = Number(counts.events || 0)
              + Number(counts.raw_records || 0)
              + Number(counts.market_snapshots || 0)
              + Number(counts.macro_metrics || 0);
            const when = entry.timestamp ? new Date(entry.timestamp).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' }) : '-';
            const duration = entry.duration_ms ? `${(Number(entry.duration_ms) / 1000).toFixed(1)}s` : '';
            const detail = totalSeen > 0
              ? `신규 ${totalStored} · 확인 뉴스 ${counts.events_seen || 0} · 원문 ${counts.raw_records_seen || 0} · 시세 ${counts.market_snapshots_seen || 0}`
              : (entry.message || '');
            const status = String(entry.status || 'scheduled');
            return `<div class="collection-log-item">
              <span>${when}</span>
              <span><i class="collection-log-status ${status}"></i><strong>${collectionStatusLabel(status)}</strong> ${detail}</span>
              <span class="muted">${duration}</span>
            </div>`;
          }).join('');
        }
      }
      drawCollectionLogChart(entries);
    }

    function maybeRefreshDiagnosticsAfterCollection(log = []) {
      if (!Array.isArray(log) || !log.length) return;
      const completed = log
        .filter((entry) => entry && entry.status === 'complete' && entry.cycle != null)
        .slice(-1)[0];
      if (!completed || completed.cycle === lastRenderedCollectionCycle) return;
      lastRenderedCollectionCycle = completed.cycle;
      loadDiagnostics().catch((error) => console.error(error));
    }

    function collectionStatusLabel(status) {
      return {
        scheduled: '예약',
        running: '수집 중',
        complete: '완료',
        error: '오류',
        stopped: '종료',
      }[status] || status || '상태';
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
        const totalSeen = Number(counts.events_seen || 0)
          + Number(counts.raw_records_seen || 0)
          + Number(counts.market_snapshots_seen || 0)
          + Number(counts.macro_metrics_seen || 0);
        const value = Math.max(1, totalSeen);
        const barHeight = entry.status === 'complete'
          ? Math.min(height - 22, 10 + Math.log10(value + 1) * 18)
          : entry.status === 'running'
            ? height - 28
            : 12;
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

    async function loadMockPerformance() {
      try {
        const data = await (await fetch('/api/mock-trading/performance')).json();
        renderMockPerformance(data);
      } catch (error) {
        console.error(error);
      }
    }

    function renderMockPerformance(data) {
      const returnRate = Number(data.return_rate || 0);
      const targetRate = Number(data.target_return_rate || 0);
      const percent = returnRate * 100;
      const targetPercent = targetRate * 100;
      const progress = targetRate > 0 ? Math.max(0, Math.min(100, (returnRate / targetRate) * 100)) : 0;
      document.getElementById('mockReturn').textContent = data.active ? `${percent.toFixed(2)}%` : '대기 중';
      document.getElementById('mockReturnBar').style.width = `${progress}%`;
      document.getElementById('mockProfit').textContent = `수익금 ${fmtWon.format(data.profit_amount || 0)}`;
      document.getElementById('mockEquity').textContent = `평가금 ${fmtWon.format(data.equity || 0)}`;
      document.getElementById('mockTarget').textContent = `목표 ${targetPercent ? targetPercent.toFixed(2) : '-'}%`;
      document.getElementById('mockStatus').textContent = data.active
        ? (data.target_achieved ? '목표 수익률을 달성했습니다.' : `목표까지 ${(targetPercent - percent).toFixed(2)}%p 남았습니다.`)
        : '선택한 목표로 시작하면 모의 KIS 포트폴리오 수익률이 표시됩니다.';
      renderMockRunTables(data);
    }

    function renderStreamingPerformance(data) {
      const account = data.account || {};
      const accountValue = Number(account.account_value || streamingInitialCash || 0);
      const cash = Number(account.cash || 0);
      const returnRate = Number(account.return_rate || 0);
      const targetRate = Number(streamingTargetReturnRate || 0);
      const targetPercent = targetRate * 100;
      const profit = accountValue - Number(streamingInitialCash || 0);
      const progress = targetRate > 0 ? Math.max(0, Math.min(100, (returnRate / targetRate) * 100)) : 0;
      const returnPercent = returnRate * 100;
      const simulatedProgress = Math.max(0, Math.min(100, Number(data.progress || 0)));
      streamingReturnSeries.push({ progress: simulatedProgress, returnRate });
      if (streamingReturnSeries.length > 160) streamingReturnSeries.shift();
      drawStreamingReturnChart();

      document.getElementById('mockReturn').textContent = `${returnPercent.toFixed(2)}%`;
      document.getElementById('mockReturnBar').style.width = `${progress}%`;
      document.getElementById('mockProfit').textContent = `수익금 ${fmtWon.format(profit)}`;
      document.getElementById('mockEquity').textContent = `평가금 ${fmtWon.format(accountValue)}`;
      document.getElementById('mockTarget').textContent = `목표 ${targetPercent ? targetPercent.toFixed(2) : '-'}%`;
      document.getElementById('mockStatus').textContent =
        `가상 차트 기준 ${simulatedProgress.toFixed(1)}% 진행 · 목표 시간 ${streamingTargetMinutes || '-'}분 · 현금 ${fmtWon.format(cash)}`;
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
      document.getElementById('mockRunStats').innerHTML = `
        <div class="stat"><strong>${data.active ? '진행 중' : '대기'}</strong><span class="muted">모의 상태</span></div>
        <div class="stat"><strong>${data.orders_count || 0}</strong><span class="muted">주문</span></div>
        <div class="stat"><strong>${data.executions_count || 0}</strong><span class="muted">체결</span></div>
        <div class="stat"><strong>${positions.length}</strong><span class="muted">미청산 종목</span></div>
      `;
      const positionTarget = document.getElementById('mockPositions');
      if (positionTarget) positionTarget.innerHTML = positions.length ? positions.map((item) => {
        const pnl = Number(item.unrealized_pnl || 0);
        const rate = Number(item.return_rate || 0) * 100;
        const tone = pnl >= 0 ? 'tone-pos' : 'tone-neg';
        return `<tr>
          <td>${item.ticker}</td>
          <td>${item.quantity}</td>
          <td>${fmtWon.format(item.average_price || 0)}</td>
          <td>${fmtWon.format(item.last_price || 0)}</td>
          <td>${fmtWon.format(item.market_value || 0)}</td>
          <td class="${tone}">${fmtWon.format(pnl)}</td>
          <td class="${tone}">${rate.toFixed(2)}%</td>
        </tr>`;
      }).join('') : '<tr><td colspan="7">보유 종목 없음</td></tr>';
      document.getElementById('mockExecutions').innerHTML = executions.length ? executions.slice().reverse().map((item) => {
        const sideClass = item.side === 'BUY' ? 'side-buy' : 'side-sell';
        return `<tr>
          <td class="${sideClass}">${item.side}</td>
          <td>${item.ticker}</td>
          <td>${item.quantity}</td>
          <td>${fmtWon.format(item.price || 0)}</td>
        </tr>`;
      }).join('') : '<tr><td colspan="4">체결 내역 없음</td></tr>';
    }

    function startMockPerformancePolling() {
      if (mockPerformanceTimer) window.clearInterval(mockPerformanceTimer);
      loadMockPerformance();
      mockPerformanceTimer = window.setInterval(loadMockPerformance, 2000);
    }

    function scoreRow(label, percent, tone, displayValue = null) {
      const value = Math.max(0, Math.min(100, Number(percent) || 0));
      return `<div class="score-row"><div class="score-label">${label}</div><div class="bar ${tone}"><span style="width:${value}%"></span></div><div class="score-value">${displayValue || `${Math.round(value)}%`}</div></div>`;
    }

    function renderDiagnostics(data) {
      const d = data.diagnostics;
      const liveLabel = d.live_data_present ? '실시간' : '로컬';
      document.getElementById('diagnosticStats').innerHTML = `
        <div class="stat"><strong>${liveLabel}</strong><span class="muted">자료 모드</span></div>
        <div class="stat"><strong>${d.events_count}</strong><span class="muted">뉴스 이벤트</span></div>
        <div class="stat"><strong>${d.live_source_count}</strong><span class="muted">실시간 출처 URL</span></div>
        <div class="stat"><strong>${d.market_snapshots_count}</strong><span class="muted">시세 스냅샷</span></div>
      `;
      const store = data.store_summary || {};
      const storedNew = data.stored_new_records || {};
      const storedNewTotal = Number(storedNew.events || 0)
        + Number(storedNew.raw_records || 0)
        + Number(storedNew.market_snapshots || 0)
        + Number(storedNew.macro_metrics || 0);
      document.getElementById('storeStats').innerHTML = `
        <div class="stat"><strong>${store.events || 0}</strong><span class="muted">저장된 이벤트</span></div>
        <div class="stat"><strong>${store.market_snapshots || 0}</strong><span class="muted">저장된 시세 스냅샷</span></div>
        <div class="stat"><strong>${storedNewTotal}</strong><span class="muted">최근 신규 저장</span></div>
        <div class="stat"><strong>${data.store_path || '-'}</strong><span class="muted">저장 위치</span></div>
      `;
      renderCollectionWarnings(d);
      renderDataVolume(data.data_volume || {});
      return;
      /*
      const entries = Object.entries(d.per_ticker || {});
      document.getElementById('tickerMetrics').innerHTML = entries.map(([ticker, item]) => {
        const total = Math.max(1, item.events || 0);
        const pos = item.positive / total * 100;
        const neu = item.neutral / total * 100;
        const neg = item.negative / total * 100;
        return `<div class="ticker-card">
          <strong>${ticker}</strong>
          <div class="muted">이벤트 ${item.events}개 · 최신 ${item.latest_event_at || '-'}</div>
          <div class="sentiment"><span class="pos" style="width:${pos}%"></span><span class="neu" style="width:${neu}%"></span><span class="neg" style="width:${neg}%"></span></div>
          <div class="chips"><span class="chip">긍정 ${item.positive}</span><span class="chip">중립 ${item.neutral}</span><span class="chip">부정 ${item.negative}</span><span class="chip">실시간 URL ${item.live_source_urls}</span></div>
        </div>`;
      }).join('');
      */
    }

    function renderCollectionWarnings(d = {}) {
      const warnings = Array.isArray(d.collection_warnings) ? d.collection_warnings : [];
      const marketSources = d.external_chart_sources_configured || 0;
      const items = [...warnings];
      if (marketSources === 0) {
        items.push('외부 주식 차트 수집원이 꺼져 있습니다. 현재 시세 스냅샷은 상장 종목 참고값 위주입니다.');
      }
      const target = document.getElementById('collectionWarnings');
      if (!target) return;
      target.innerHTML = items.map((message) => `<div class="warning-item">${message}</div>`).join('');
    }

    function renderDataVolume(volume = {}) {
      drawDataVolumeChart(volume.by_kind || {});
      const rows = Array.isArray(volume.by_source) ? volume.by_source : [];
      const marketSources = volume.market_snapshot_sources || {};
      const sourceList = document.getElementById('sourceVolumeList');
      if (sourceList) {
        const topRows = rows
          .filter((row) => ['events', 'raw_records', 'market_snapshots', 'macro_metrics'].includes(row.kind))
          .sort((a, b) => Number(b.count || 0) - Number(a.count || 0))
          .slice(0, 12);
        const marketSourceText = Object.entries(marketSources)
          .map(([name, count]) => `${name}: ${count}`)
          .join(' · ');
        sourceList.innerHTML = `
          <div class="muted"><strong>출처별 저장량</strong></div>
          ${topRows.map((row) => `<div class="source-volume-row"><span>${kindLabelForVolume(row.kind)} · ${row.source_name}</span><strong>${row.count}</strong></div>`).join('')}
          <div class="muted" style="margin-top:6px;">시세 출처: ${marketSourceText || '없음'}</div>
        `;
      }
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
        ['events', '뉴스/이벤트', '#0f766e'],
        ['raw_records', '원문', '#2563eb'],
        ['market_snapshots', '시세', '#16a34a'],
        ['macro_metrics', '매크로', '#b45309'],
      ].map(([key, label, color]) => ({ key, label, color, value: Number(byKind[key] || 0) }));
      const maxValue = Math.max(1, ...items.map((item) => item.value));
      ctx.strokeStyle = '#e2e8f0';
      ctx.beginPath();
      ctx.moveTo(54, height - 34);
      ctx.lineTo(width - 16, height - 34);
      ctx.stroke();
      const barArea = width - 90;
      const step = barArea / items.length;
      items.forEach((item, index) => {
        const barWidth = Math.min(72, Math.max(34, step * 0.5));
        const barHeight = Math.max(2, (height - 62) * (item.value / maxValue));
        const x = 60 + index * step + (step - barWidth) / 2;
        const y = height - 34 - barHeight;
        ctx.fillStyle = item.color;
        ctx.fillRect(x, y, barWidth, barHeight);
        ctx.fillStyle = '#1d2430';
        ctx.font = 'bold 13px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(String(item.value), x + barWidth / 2, y - 7);
        ctx.fillStyle = '#667085';
        ctx.font = '12px Arial';
        ctx.fillText(item.label, x + barWidth / 2, height - 12);
      });
      ctx.textAlign = 'left';
      ctx.fillStyle = '#667085';
      ctx.font = '12px Arial';
      ctx.fillText('저장된 수집 데이터 양', 12, 18);
    }

    function kindLabelForVolume(kind) {
      return {
        events: '뉴스',
        raw_records: '원문',
        market_snapshots: '시세',
        macro_metrics: '매크로',
      }[kind] || kind;
    }

    function translateGoalLabel(label) {
      const labels = {
        'Requested target': '요청 목표',
        'Lower return': '수익률 낮춤',
        'Longer period': '기간 연장',
        'Balanced compromise': '균형 타협안'
      };
      return labels[label] || label;
    }

    async function renderOntologyGraph(data) {
      const canvas = document.getElementById('ontologyCanvas');
      const tooltip = document.getElementById('ontologyTooltip');
      const THREE = await loadThree();
      if (!THREE) {
        renderOntologyGraph2d(data, canvas, tooltip);
        tooltip.style.display = 'block';
        tooltip.style.left = '12px';
        tooltip.style.top = '52px';
        tooltip.textContent = '3D 라이브러리를 불러오지 못해 2D로 표시합니다.';
        return;
      }

      if (graphState) {
        graphState.stop = true;
        if (graphState.renderer) graphState.renderer.dispose();
      }

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0f172a);
      const camera = new THREE.PerspectiveCamera(52, 1, 0.1, 5000);
      camera.position.set(0, 0, 760);
      const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

      const root = new THREE.Group();
      scene.add(root);
      scene.add(new THREE.AmbientLight(0xffffff, 0.78));
      const light = new THREE.PointLight(0xffffff, 1.1);
      light.position.set(300, 280, 500);
      scene.add(light);

      const renderGraph = prepareRenderableGraph(data.nodes || [], data.links || []);
      const nodes = computeGraphLayout(renderGraph.nodes, renderGraph.links);
      const graphMetrics = buildGraphMetrics(nodes, renderGraph.links);
      document.getElementById('ontologyCounts').textContent =
        `노드 ${data.counts.nodes} · 관계 ${data.counts.links} · 표시 ${nodes.length}/${renderGraph.links.length}`;
      const nodeMap = new Map(nodes.map((node) => [node.id, node]));
      const raycaster = new THREE.Raycaster();
      const pointer = new THREE.Vector2(99, 99);
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
        steps: (data.reasoning_steps || []).filter((step) => (step.nodes || []).some((id) => nodeMap.has(id))),
        playing: true,
        currentIndex: -1,
        startedAt: performance.now(),
        stepMs: 1450,
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
        const material = new THREE.LineBasicMaterial({ color: edgeColor(link.predicate), transparent: true, opacity: 0.26 });
        const line = new THREE.Line(geometry, material);
        line.userData = { source: link.source, target: link.target, predicate: link.predicate };
        line.userData.baseColor = edgeColor(link.predicate);
        line.userData.baseOpacity = 0.26;
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
        const geometry = new THREE.SphereGeometry(radius, 16, 16);
        const highlighted = Boolean(node.highlight);
        const material = new THREE.MeshStandardMaterial({
          color: nodeColor(node.kind),
          emissive: nodeColor(node.kind),
          emissiveIntensity: highlighted ? 0.34 : 0.08,
          roughness: 0.62,
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

      function resize() {
        const rect = canvas.getBoundingClientRect();
        renderer.setSize(rect.width, rect.height, false);
        camera.aspect = rect.width / Math.max(1, rect.height);
        camera.updateProjectionMatrix();
      }

      canvas.addEventListener('pointerdown', (event) => {
        dragging = true;
        lastX = event.clientX;
        lastY = event.clientY;
        pausedUntil = performance.now() + 2500;
        canvas.setPointerCapture(event.pointerId);
      });
      canvas.addEventListener('pointermove', (event) => {
        const rect = canvas.getBoundingClientRect();
        pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        if (!dragging) return;
        rotationY += (event.clientX - lastX) * 0.008;
        rotationX += (event.clientY - lastY) * 0.008;
        lastX = event.clientX;
        lastY = event.clientY;
      });
      canvas.addEventListener('pointerup', () => { dragging = false; });
      canvas.addEventListener('wheel', (event) => {
        event.preventDefault();
        targetZoom = Math.max(260, Math.min(1300, targetZoom + event.deltaY * 0.7));
        pausedUntil = performance.now() + 2200;
      }, { passive: false });
      document.getElementById('resetGraph').onclick = () => {
        rotationX = -0.18;
        rotationY = 0.34;
        targetZoom = 760;
        fitVisibleGraph();
      };
      document.getElementById('toggleLabels').onclick = () => {
        labelState.visible = !labelState.visible;
        document.getElementById('toggleLabels').textContent = labelState.visible ? '라벨 끄기' : '라벨 켜기';
        updateVisibility();
      };
      document.getElementById('toggleReasoning').onclick = () => {
        reasoningState.playing = !reasoningState.playing;
        reasoningState.startedAt = performance.now() - Math.max(0, reasoningState.currentIndex) * reasoningState.stepMs;
        document.getElementById('toggleReasoning').textContent = reasoningState.playing ? '추론 일시정지' : '추론 재생';
      };
      document.querySelectorAll('#ontologyFilters input').forEach((input) => {
        input.onchange = () => {
          if (input.checked) activeKinds.add(input.value);
          else activeKinds.delete(input.value);
          updateVisibility();
        };
      });

      canvas.addEventListener('click', () => {
        raycaster.setFromCamera(pointer, camera);
        const hit = raycaster.intersectObjects(nodeMeshes.filter((mesh) => mesh.visible), false)[0];
        if (hit) renderNodePanel(hit.object.userData, data.links);
      });

      function updateVisibility() {
        let visibleCount = 0;
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
        fitVisibleGraph();
        document.getElementById('ontologyCounts').textContent =
          `노드 ${data.counts.nodes} · 관계 ${data.counts.links} · 표시 ${visibleCount}/${renderGraph.links.length}`;
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

      function updateReasoning(now) {
        if (!reasoningState.steps.length) {
          document.getElementById('reasoningBadge').textContent = '추론 단계 0/0';
          return;
        }
        if (reasoningState.playing) {
          const elapsed = Math.max(0, now - reasoningState.startedAt);
          const index = Math.floor(elapsed / reasoningState.stepMs) % reasoningState.steps.length;
          if (index !== reasoningState.currentIndex) setActiveReasoningStep(index);
        }
      }

      function setActiveReasoningStep(index) {
        if (index < 0 || index >= reasoningState.steps.length) return;
        reasoningState.currentIndex = index;
        const step = reasoningState.steps[index];
        reasoningState.activeNodeIds = new Set(step.nodes || []);
        reasoningState.activeLinkKeys = new Set((step.links || []).map((link) => linkKey(link.source, link.target, link.predicate)));
        document.getElementById('reasoningBadge').textContent = `추론 단계 ${index + 1}/${reasoningState.steps.length}`;
        document.getElementById('reasoningTitle').textContent = step.title || '추론 단계';
        document.getElementById('reasoningMeta').textContent = `${step.ticker || '-'} · 신뢰도 ${step.confidence_percent ?? '-'}%`;
        document.getElementById('reasoningDescription').textContent = step.description || '';
        document.getElementById('reasoningProgress').style.width = `${((index + 1) / reasoningState.steps.length) * 100}%`;
        updateVisibility();
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

      graphState = { renderer, stop: false };
      resize();
      window.addEventListener('resize', resize);
      if (reasoningState.steps.length) setActiveReasoningStep(0);

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
      if (graphState) graphState.stop = true;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const renderGraph = prepareRenderableGraph(data.nodes || [], data.links || []);
      const nodes = computeGraphLayout(renderGraph.nodes, renderGraph.links);
      const graphMetrics = buildGraphMetrics(nodes, renderGraph.links);
      const nodeMap = new Map(nodes.map((node) => [node.id, node]));
      const activeKinds = new Set(['ticker', 'event', 'temporal', 'sector', 'support', 'risk', 'contradiction', 'pipeline', 'tuning', 'parameter', 'metric', 'entity']);
      const reasoningState = {
        steps: (data.reasoning_steps || []).filter((step) => (step.nodes || []).some((id) => nodeMap.has(id))),
        playing: true,
        currentIndex: -1,
        startedAt: performance.now(),
        stepMs: 1450,
        activeNodeIds: new Set(),
        activeLinkKeys: new Set(),
      };
      const view = { scale: 1, offsetX: 0, offsetY: 0, dragging: false, lastX: 0, lastY: 0, labels: false, pointerX: -9999, pointerY: -9999 };
      let hoveredNode = null;

      document.getElementById('ontologyCounts').textContent =
        `노드 ${data.counts.nodes} · 관계 ${data.counts.links} · 표시 ${nodes.length}/${renderGraph.links.length}`;

      function resize() {
        const rect = canvas.getBoundingClientRect();
        const ratio = Math.min(window.devicePixelRatio || 1, 2);
        canvas.width = Math.max(1, Math.floor(rect.width * ratio));
        canvas.height = Math.max(1, Math.floor(rect.height * ratio));
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      }

      function project(node) {
        const rect = canvas.getBoundingClientRect();
        const [x, y] = node.position || [0, 0, 0];
        return {
          x: rect.width / 2 + (x + view.offsetX) * view.scale,
          y: rect.height / 2 + (y + view.offsetY) * view.scale,
        };
      }

      function visibleNode(node) {
        return activeKinds.has(node.kind);
      }

      function fitVisibleGraph2d() {
        const visibleNodes = nodes.filter(visibleNode);
        if (!visibleNodes.length) {
          view.scale = 1;
          view.offsetX = 0;
          view.offsetY = 0;
          return;
        }
        const xs = visibleNodes.map((node) => (node.position || [0, 0, 0])[0]);
        const ys = visibleNodes.map((node) => (node.position || [0, 0, 0])[1]);
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);
        const width = Math.max(1, maxX - minX);
        const height = Math.max(1, maxY - minY);
        const rect = canvas.getBoundingClientRect();
        view.offsetX = -((minX + maxX) / 2);
        view.offsetY = -((minY + maxY) / 2);
        view.scale = Math.max(0.45, Math.min(2.7, Math.min(rect.width / (width + 120), rect.height / (height + 120))));
      }

      function setActiveReasoningStep(index) {
        if (index < 0 || index >= reasoningState.steps.length) return;
        reasoningState.currentIndex = index;
        const step = reasoningState.steps[index];
        reasoningState.activeNodeIds = new Set(step.nodes || []);
        reasoningState.activeLinkKeys = new Set((step.links || []).map((link) => linkKey(link.source, link.target, link.predicate)));
        document.getElementById('reasoningBadge').textContent = `추론 단계 ${index + 1}/${reasoningState.steps.length}`;
        document.getElementById('reasoningTitle').textContent = step.title || '추론 단계';
        document.getElementById('reasoningMeta').textContent = `${step.ticker || '-'} · 신뢰도 ${step.confidence_percent ?? '-'}%`;
        document.getElementById('reasoningDescription').textContent = step.description || '';
        document.getElementById('reasoningProgress').style.width = `${((index + 1) / reasoningState.steps.length) * 100}%`;
      }

      function updateReasoning(now) {
        if (!reasoningState.steps.length) {
          document.getElementById('reasoningBadge').textContent = '추론 단계 0/0';
          return;
        }
        if (!reasoningState.playing) return;
        const index = Math.floor(Math.max(0, now - reasoningState.startedAt) / reasoningState.stepMs) % reasoningState.steps.length;
        if (index !== reasoningState.currentIndex) setActiveReasoningStep(index);
      }

      function draw(now) {
        if (!graphState || graphState.stop) return;
        const rect = canvas.getBoundingClientRect();
        ctx.clearRect(0, 0, rect.width, rect.height);
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, rect.width, rect.height);
        updateReasoning(now);
        const pulse = 0.55 + Math.sin(now / 150) * 0.45;

        for (const link of renderGraph.links) {
          const source = nodeMap.get(link.source);
          const target = nodeMap.get(link.target);
          if (!source || !target || !visibleNode(source) || !visibleNode(target)) continue;
          const a = project(source);
          const b = project(target);
          const active = reasoningState.activeLinkKeys.has(linkKey(link.source, link.target, link.predicate));
          ctx.save();
          ctx.strokeStyle = intColorToCss(active ? neonEdgeColor(link.predicate) : edgeColor(link.predicate));
          ctx.globalAlpha = active ? 0.78 : 0.22;
          ctx.lineWidth = active ? 2.4 + pulse * 1.5 : 0.8;
          ctx.shadowBlur = active ? 16 : 0;
          ctx.shadowColor = ctx.strokeStyle;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
          ctx.restore();
        }

        hoveredNode = null;
        let nearestDistance = 18;
        for (const node of nodes) {
          if (!visibleNode(node)) continue;
          const p = project(node);
          const active = reasoningState.activeNodeIds.has(node.id);
          const highlighted = Boolean(node.highlight);
          const radius = nodeRadius(node, graphMetrics) * view.scale * (active ? 1.35 : 1);
          const color = intColorToCss(active ? neonColor(node.kind) : nodeColor(node.kind));
          ctx.save();
          ctx.fillStyle = color;
          ctx.globalAlpha = 0.92;
          ctx.shadowBlur = active ? 24 : highlighted ? 14 : 0;
          ctx.shadowColor = intColorToCss(neonColor(node.kind));
          ctx.beginPath();
          ctx.arc(p.x, p.y, Math.max(2.2, radius), 0, Math.PI * 2);
          ctx.fill();
          if (highlighted || active) {
            ctx.globalAlpha = active ? 0.28 + pulse * 0.16 : 0.16;
            ctx.beginPath();
            ctx.arc(p.x, p.y, Math.max(6, radius * 2.5), 0, Math.PI * 2);
            ctx.fill();
          }
          if (view.labels || active) {
            ctx.globalAlpha = 0.95;
            ctx.shadowBlur = 0;
            ctx.fillStyle = '#e5e7eb';
            ctx.font = '12px Arial';
            ctx.fillText(shortLabel(node.label), p.x + radius + 4, p.y - radius - 3);
          }
          ctx.restore();
          const d = Math.hypot(p.x - view.pointerX, p.y - view.pointerY);
          if (d < nearestDistance) {
            nearestDistance = d;
            hoveredNode = node;
          }
        }

        if (hoveredNode) {
          const p = project(hoveredNode);
          tooltip.style.display = 'block';
          tooltip.style.left = `${Math.min(rect.width - 280, Math.max(8, p.x + 12))}px`;
          tooltip.style.top = `${Math.min(rect.height - 80, Math.max(50, p.y + 12))}px`;
          tooltip.innerHTML = `<strong>${hoveredNode.label}</strong><br>${kindLabel(hoveredNode.kind)} · 연결 ${degree(hoveredNode.id, renderGraph.links)}개 · 중요도 ${Number(hoveredNode.importance_score || 0).toFixed(2)}`;
        } else {
          tooltip.style.display = 'none';
        }

        requestAnimationFrame(draw);
      }

      canvas.onpointerdown = (event) => {
        view.dragging = true;
        view.lastX = event.clientX;
        view.lastY = event.clientY;
        canvas.setPointerCapture(event.pointerId);
      };
      canvas.onpointermove = (event) => {
        const rect = canvas.getBoundingClientRect();
        view.pointerX = event.clientX - rect.left;
        view.pointerY = event.clientY - rect.top;
        if (!view.dragging) return;
        view.offsetX += (event.clientX - view.lastX) / Math.max(0.1, view.scale);
        view.offsetY += (event.clientY - view.lastY) / Math.max(0.1, view.scale);
        view.lastX = event.clientX;
        view.lastY = event.clientY;
      };
      canvas.onpointerup = () => { view.dragging = false; };
      canvas.onpointerleave = () => { view.pointerX = -9999; view.pointerY = -9999; };
      canvas.onwheel = (event) => {
        event.preventDefault();
        view.scale = Math.max(0.45, Math.min(2.7, view.scale * (event.deltaY > 0 ? 0.9 : 1.1)));
      };
      canvas.onclick = () => {
        if (hoveredNode) renderNodePanel(hoveredNode, data.links);
      };
      document.getElementById('resetGraph').onclick = () => {
        fitVisibleGraph2d();
      };
      document.getElementById('toggleLabels').onclick = () => {
        view.labels = !view.labels;
        document.getElementById('toggleLabels').textContent = view.labels ? '라벨 끄기' : '라벨 켜기';
      };
      document.getElementById('toggleReasoning').onclick = () => {
        reasoningState.playing = !reasoningState.playing;
        reasoningState.startedAt = performance.now() - Math.max(0, reasoningState.currentIndex) * reasoningState.stepMs;
        document.getElementById('toggleReasoning').textContent = reasoningState.playing ? '추론 일시정지' : '추론 재생';
      };
      document.querySelectorAll('#ontologyFilters input').forEach((input) => {
        input.onchange = () => {
          if (input.checked) activeKinds.add(input.value);
          else activeKinds.delete(input.value);
          fitVisibleGraph2d();
        };
      });

      graphState = { stop: false, renderer: null };
      resize();
      window.addEventListener('resize', resize, { passive: true });
      fitVisibleGraph2d();
      if (reasoningState.steps.length) setActiveReasoningStep(0);
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
      } finally {
        streamingStepBusy = false;
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
      if (nodes.length > 700) return computeFastClusterLayout(nodes, rawLinks);

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
      if (predicate === 'feedsStage' || predicate === 'requiresApprovalFrom') return 82;
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
      if (predicate === 'feedsStage' || predicate === 'requiresApprovalFrom') return 1.35;
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
      if (predicate === 'selectsCandidate' || predicate === 'feedsStage' || predicate === 'requiresApprovalFrom') return 0x2563eb;
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
      if (predicate === 'selectsCandidate' || predicate === 'feedsStage' || predicate === 'requiresApprovalFrom') return 0x93c5fd;
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
        temporal: '시간 동기화 프레임',
        pipeline: '분석 파이프라인',
        tuning: '파라미터 튜닝 모드',
        parameter: '튜닝 파라미터',
        metric: '파이프라인 지표',
        sector: '섹터',
        support: '긍정 신호',
        risk: '리스크 요인',
        contradiction: '상충 요인',
        entity: '개체'
      }[kind] || kind;
    }

    function shortLabel(label) {
      if (label.startsWith('NEWS:')) return `뉴스 ${label.slice(5, 11)}`;
      if (label.length > 22) return `${label.slice(0, 20)}…`;
      return label;
    }

    function degree(nodeId, links) {
      return links.filter((link) => link.source === nodeId || link.target === nodeId).length;
    }

    function linkKey(source, target, predicate) {
      return `${source}::${predicate}::${target}`;
    }

    function graphSignature(graph) {
      const counts = graph.counts || {};
      const latestStep = (graph.reasoning_steps || []).map((step) => `${step.path_id}:${step.title}:${step.description}`).join('|');
      return `${counts.nodes || 0}:${counts.links || 0}:${latestStep}`;
    }

    function renderNodePanel(node, links) {
      const nodeLinks = links.filter((link) => link.source === node.id || link.target === node.id);
      const related = nodeLinks
        .slice(0, 20)
        .map((link) => {
          const other = link.source === node.id ? link.target : link.source;
          const direction = link.source === node.id ? '→' : '←';
          return `<div>${direction} <strong>${link.predicate}</strong> ${shortLabel(other)}</div>`;
        })
        .join('');
      const hiddenCount = Math.max(0, nodeLinks.length - 20);
      document.getElementById('ontologyPanel').innerHTML = `
        <strong>${node.label}</strong>
        <div class="muted">종류: ${kindLabel(node.kind)} · 연결 ${degree(node.id, links)}개 · 중요도 ${Number(node.importance_score || 0).toFixed(2)}</div>
        <div style="margin-top:10px;">${related || '<span class="muted">연결 관계 없음</span>'}</div>
        <div class="muted" style="margin-top:8px;">${hiddenCount > 0 ? `추가 관계 ${hiddenCount}개는 생략되었습니다.` : ''}</div>
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
      document.getElementById('gate').textContent = data.started ? '분석 및 모의 실행 모드로 시작했습니다. 실거래는 비활성화되어 있습니다.' : '시작되지 않았습니다.';
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
      document.getElementById('summary').textContent = '목표를 입력하면 시장 자료, 온톨로지 관계, 리스크 압력을 바탕으로 달성 가능성을 계산합니다.';
      document.getElementById('gate').textContent = '목표가 확정될 때까지 프로그램은 시작되지 않습니다.';
      document.getElementById('output').textContent = '아직 실행되지 않았습니다.';
      document.getElementById('startButton').disabled = true;
      if (mockPerformanceTimer) window.clearInterval(mockPerformanceTimer);
    });
    
    // ===== 스트리밍 데모 관련 함수 =====
    let streamingDemoId = null;
    let streamingDemoRunning = false;
    let streamingDemoHistory = [];  // 거래 내역 누적
    let streamingDemoPrices = {};   // 종목별 마지막 가격
    let streamingInitialCash = 0;   // 초기 자본 (예수금)
    let streamingTargetReturnRate = 0;
    let streamingTargetMinutes = 0;
    
    async function startStreamingDemo() {
      let targetReturn = parseFloat(document.getElementById('targetReturn')?.value || 0.02);
      if (targetReturn > 1) targetReturn = targetReturn / 100.0;
      const periodMinutes = parseInt(document.getElementById('targetMinutes')?.value || 390);
      const initialCash = Math.max(100000, Number(document.getElementById('initialCash')?.value || 10000000));
      
      streamingDemoHistory = [];  // 초기화
      streamingDemoPrices = {};
      streamingReturnSeries = [];
      streamingInitialCash = initialCash;  // 초기 자본 저장
      streamingTargetReturnRate = targetReturn;
      streamingTargetMinutes = periodMinutes;
      
      try {
        const response = await fetch('/api/streaming-demo/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            target_return_rate: targetReturn,
            period_minutes: periodMinutes,
            initial_cash: initialCash,
          })
        });
        const data = await response.json();
        streamingDemoId = data.demo_id;
        streamingDemoRunning = true;
        document.getElementById('streamingDemoStatus').textContent = '실행 중...';
        document.getElementById('streamingDemoProgress').style.width = '0%';
        console.log('Streaming demo started:', data);
      } catch (error) {
        console.error('Failed to start streaming demo:', error);
        document.getElementById('streamingDemoStatus').textContent = '오류: ' + error.message;
      }
    }
    
    async function runStreamingDemoStep() {
      if (!streamingDemoId || !streamingDemoRunning) {
        console.warn('Demo not running');
        return;
      }
      if (streamingStepBusy) {
        return { status: 'busy', progress: 0 };
      }
      streamingStepBusy = true;
      
      try {
        document.getElementById('streamingDemoStatus').textContent =
          '전체 상장 종목 NPU 전수 스캔 및 온톨로지/전략/리스크 분석 중...';
        document.getElementById('mockStatus').textContent =
          '전체 상장 종목 NPU 전수 스캔 및 온톨로지/전략/리스크 분석 중...';
        const response = await fetch('/api/streaming-demo/step', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ demo_id: streamingDemoId }),
          signal: AbortSignal.timeout(120000),
        });
        if (!response.ok) {
          streamingStepFailures += 1;
          if (streamingStepFailures < 3 && response.status >= 500) {
            const retryMessage = `시뮬레이션 응답 지연, 재시도 중 (${streamingStepFailures}/3)`;
            document.getElementById('streamingDemoStatus').textContent = retryMessage;
            document.getElementById('mockStatus').textContent = retryMessage;
            return { status: 'retrying', progress: 0 };
          }
          streamingDemoRunning = false;
          streamingDemoId = null;
          const message = response.status === 404
            ? '시뮬레이션 세션이 만료되었습니다. 시뮬레이션 테스트를 다시 시작하세요.'
            : `시뮬레이션 업데이트 실패 (${response.status})`;
          document.getElementById('streamingDemoStatus').textContent = message;
          document.getElementById('mockStatus').textContent = message;
          return { status: 'stopped', progress: 0 };
        }
        const data = await response.json();
        streamingStepFailures = 0;
        if (data.status === 'waiting') {
          const remaining = Number(data.seconds_until_next_step || 0);
          const message = `실시간 대기 중 · 다음 1분 bar까지 ${remaining.toFixed(1)}초`;
          document.getElementById('streamingDemoStatus').textContent = message;
          document.getElementById('mockStatus').textContent = message;
          return data;
        }
        if (data.status === 'expired') {
          streamingDemoRunning = false;
          streamingDemoId = null;
          const message = '시뮬레이션 세션이 만료되었습니다. 시뮬레이션 테스트를 다시 시작하세요.';
          document.getElementById('streamingDemoStatus').textContent = message;
          document.getElementById('mockStatus').textContent = message;
          return data;
        }
        
        document.getElementById('streamingDemoProgress').style.width = data.progress + '%';
        if (data.prices && typeof data.prices === 'object') {
          Object.entries(data.prices).forEach(([ticker, price]) => {
            streamingDemoPrices[ticker] = Number(price || 0);
          });
        }
        document.getElementById('streamingDemoStatus').textContent =
          `가상 차트 ${data.step || '완료'}분 진행 · 전체 ${data.universe_scanned_count || data.universe_count || '-'}개 NPU 전수 스캔 · 후보 ${data.candidate_ticker_count || data.active_ticker_count || '-'}개 정밀 분석 · bar ${data.chart_bar ?? '-'} · ${(data.progress || 0).toFixed(1)}%`;
        
        // 계정 정보 업데이트
        if (data.account) {
          const cash = data.account.cash;  // 예수금 = 현재 남아있는 현금
          const invested = data.account.account_value - data.account.cash;  // 투자금 = 총자산 - 현금
          const profit = data.account.account_value - streamingInitialCash;  // 수익금 = 총자산 - 초기자본
          
          document.getElementById('streamingDeposit').textContent = 
            fmtWon.format(cash);
          document.getElementById('streamingInvested').textContent = 
            fmtWon.format(invested);
          document.getElementById('streamingProfit').textContent = 
            fmtWon.format(profit);
          document.getElementById('streamingReturnRate').textContent = 
            (data.account.return_rate * 100).toFixed(2) + '%';
          renderStreamingPerformance(data);
        }
        
        // 거래 내역 누적
        if (data.trades && data.trades.length > 0) {
          data.trades.forEach(t => {
            streamingDemoHistory.unshift(t);
            streamingDemoPrices[t.ticker] = t.price;
          });
          
          // 최근 체결 테이블 업데이트 (최대 20개)
          const executionList = document.getElementById('mockExecutions');
          if (executionList && streamingDemoHistory.length > 0) {
            const executionHtml = streamingDemoHistory.slice(0, 20).map(t =>
              `<tr>
                <td class="side-${t.side.toLowerCase()}">${t.side}</td>
                <td>${t.ticker}</td>
                <td>${t.quantity}</td>
                <td>${fmtWon.format(t.price)}</td>
              </tr>`
            ).join('');
            executionList.innerHTML = executionHtml;
          }
          
          // 스트리밍 데모 거래 테이블 (최신 거래)
          const tradeList = document.getElementById('streamingTradeList');
          if (tradeList) {
            const tradeHtml = data.trades.map(t => 
              `<tr><td>${t.ticker}</td><td class="side-${t.side.toLowerCase()}">${t.side}</td>
               <td>${t.quantity}</td><td>${fmtWon.format(t.value)}</td></tr>`
            ).join('');
            tradeList.innerHTML = tradeHtml;
          }
        }
        
        // 보유 종목 테이블 업데이트
        if (data.holdings && typeof data.holdings === 'object') {
          const positionList = document.getElementById('mockPositions');
          if (positionList) {
            const holdings = Object.entries(data.holdings);
            if (holdings.length === 0) {
              positionList.innerHTML = '<tr><td colspan="7">보유 종목 없음</td></tr>';
            } else {
              const positionHtml = holdings.map(([ticker, quantity]) => {
                const price = streamingDemoPrices[ticker] || 0;
                const marketValue = quantity * price;
                return `<tr>
                  <td>${ticker}</td>
                  <td>${quantity}</td>
                  <td>${fmtWon.format(price)}</td>
                  <td>${fmtWon.format(price)}</td>
                  <td>${fmtWon.format(marketValue)}</td>
                  <td>-</td>
                  <td>-</td>
                </tr>`;
              }).join('');
              positionList.innerHTML = positionHtml;
            }
          }
        }
        
        if (data.status === 'completed') {
          streamingDemoRunning = false;
          document.getElementById('streamingDemoStatus').textContent = '완료!';
          console.log('Final results:', data.final_results);
        }
        return data;
      } catch (error) {
        console.error('Step execution failed:', error);
        streamingStepFailures += 1;
        const message = `시뮬레이션 응답 지연, 재시도 중 (${Math.min(streamingStepFailures, 3)}/3)`;
        if (streamingStepFailures < 3) {
          document.getElementById('streamingDemoStatus').textContent = message;
          document.getElementById('mockStatus').textContent = message;
          return { status: 'retrying', progress: 0 };
        }
        streamingDemoRunning = false;
        document.getElementById('streamingDemoStatus').textContent = `시뮬레이션 업데이트 실패: ${error.message || error}`;
        return null;
      } finally {
        streamingStepBusy = false;
      }
    }
    
    async function autoRunStreamingDemo(isFirstTick = false) {
      if (!streamingDemoRunning) return;
      const intervalMs = 60000;
      const data = await runStreamingDemoStep();
      if (streamingDemoRunning) {
        const waitMs = data && data.status === 'waiting'
          ? Math.max(1000, Number(data.seconds_until_next_step || 60) * 1000)
          : intervalMs;
        streamingDemoTimer = setTimeout(() => autoRunStreamingDemo(false), waitMs);
      }
    }
    
    // 로드 시 초기화
    applyUrlGoalParams();
    updateModeButtons();
    updateModeActionCopy();
    loadStatus();
    startLearningStatusPolling();
    loadRealtimeRuntime();
    loadOperationModeStatus().catch(() => {});
    loadDiagnostics();
    loadOntologyGraph();
    loadMockPerformance();
    refreshLiveSnapshot();
    setInterval(() => loadOperationModeStatus().catch(() => {}), 3000);
    setInterval(refreshLiveSnapshot, 5000);
  </script>
</body>
</html>
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from app.audit import AuditLogger
from app.graph import get_ontology_runtime
from app.goals import GoalRequest, NegotiatedGoal, assess_goal, build_compromise_goals
from app.pipeline import build_analysis_context
from app.research import ResearchRunResult, ResearchService
from app.storage import LocalResearchStore

app = FastAPI(title="개인 투자 분석 시스템")
audit = AuditLogger(Path("logs/web-audit.jsonl"))
sessions: dict[str, dict[str, Any]] = {}
DEFAULT_RESEARCH_CONFIG = Path("config/research_sources.live.json")
LIVE_REFRESH_SECONDS = max(5, int(os.getenv("LIVE_REFRESH_SECONDS", "15")))
LIVE_STALE_SECONDS = max(LIVE_REFRESH_SECONDS * 2, int(os.getenv("LIVE_STALE_SECONDS", "45")))

_live_lock = threading.Lock()
_refresh_guard = threading.Lock()
_live_worker: threading.Thread | None = None
_live_state: dict[str, Any] = {
    "context": None,
    "research_result": None,
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
}


@app.on_event("startup")
def _startup_live_worker() -> None:
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
            "diagnostics": research_result.diagnostics,
            "skipped_sources": research_result.skipped_sources,
            "stored_new_records": snapshot["stored_new_records"],
            "store_summary": snapshot["store_summary"],
            "store_path": str(LocalResearchStore().db_path),
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
    context = _get_or_refresh_live()["context"]
    return _json(_graph_payload(context))


@app.get("/api/ontology/runtime")
def ontology_runtime() -> JSONResponse:
    return _json(get_ontology_runtime().as_dict())


@app.get("/api/live-progress")
def live_progress() -> JSONResponse:
    snapshot = _live_snapshot()
    return _json(
        {
            "is_refreshing": snapshot["is_refreshing"],
            "progress": snapshot["progress"],
            "updated_at": _iso_or_none(snapshot["last_updated"]),
            "last_error": snapshot["last_error"],
        }
    )


@app.post("/api/live-snapshot")
async def live_snapshot(request: Request) -> JSONResponse:
    payload = await request.json()
    goal_payload = payload.get("goal")
    force_refresh = bool(payload.get("force_refresh", False))
    snapshot = _get_or_refresh_live(force_refresh=force_refresh)
    research_result = snapshot["research_result"]
    context = snapshot["context"]
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
            "diagnostics": research_result.diagnostics,
            "skipped_sources": research_result.skipped_sources,
            "stored_new_records": snapshot["stored_new_records"],
            "store_summary": snapshot["store_summary"],
            "store_path": str(LocalResearchStore().db_path),
            "graph_triples_count": len(context.graph.triples()),
            "reasoning_paths": context.reasoning_paths,
            "ontology_runtime": context.ontology_runtime.as_dict(),
            "is_refreshing": snapshot["is_refreshing"],
            "refresh_interval_seconds": LIVE_REFRESH_SECONDS,
        },
        "graph": _graph_payload(context),
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
    return _json(response)


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
    sessions[session_id]["started"] = True
    sessions[session_id]["goal"] = goal
    audit.record("program_started_after_goal_acceptance", {"session_id": session_id, "goal": goal})

    return _json(
        {
            "started": True,
            "mode": "analysis_and_paper_only",
            "message": "협의된 목표를 확정한 뒤 프로그램을 시작했습니다. 실거래는 비활성화되어 있습니다.",
            "accepted_goal": goal,
            "portfolio_report": context.report,
            "events": context.events,
            "reasoning_paths": context.reasoning_paths,
            "signals": context.signals,
            "order_intents": context.intents,
            "risk_results": context.risk_results,
            "cash_guardrail": "Buy validation includes deposit_limit_check and cash reserve checks.",
        }
    )


def _parse_goal_request(payload: dict[str, Any]) -> GoalRequest:
    period_days = int(payload.get("period_days") or 0)
    goal_mode = str(payload.get("goal_mode") or "").strip()
    target_return_rate = payload.get("target_return_rate")
    target_profit_amount = payload.get("target_profit_amount")
    has_rate = target_return_rate not in (None, "")
    has_amount = target_profit_amount not in (None, "")

    if goal_mode not in {"rate", "amount"}:
        if has_rate == has_amount:
            raise HTTPException(
                status_code=400,
                detail="목표 수익률 또는 목표 수익금 중 하나만 입력하세요.",
            )
        goal_mode = "rate" if has_rate else "amount"

    if goal_mode == "rate":
        if not has_rate or has_amount:
            raise HTTPException(
                status_code=400,
                detail="목표 수익률 모드에서는 목표 수익률만 입력해야 합니다.",
            )
        parsed_rate = float(target_return_rate) / 100.0
        parsed_amount = None
    else:
        if not has_amount or has_rate:
            raise HTTPException(
                status_code=400,
                detail="목표 수익금 모드에서는 목표 수익금만 입력해야 합니다.",
            )
        parsed_rate = None
        parsed_amount = float(target_profit_amount)

    return GoalRequest(
        target_return_rate=parsed_rate,
        target_profit_amount=parsed_amount,
        period_days=period_days,
    )


def _load_default_research() -> ResearchRunResult:
    return ResearchService(progress_callback=_research_progress).run_from_config(DEFAULT_RESEARCH_CONFIG)


def _research_progress(source_key: str, completed: int, total: int) -> None:
    is_retry = source_key.startswith("retry:")
    percent = 50 if is_retry else 18 + int((min(completed, total) / max(1, total)) * 30)
    message = (
        f"Retrying failed source: {source_key[6:]}"
        if is_retry
        else f"Collecting source {completed}/{total}: {source_key}"
    )
    _set_live_progress(
        percent,
        "research",
        message,
    )


def _build_web_context():
  return _get_or_refresh_live()["context"]


def _start_live_worker() -> None:
  global _live_worker
  with _live_lock:
    _live_state["stop"] = False
    if _live_worker is not None and _live_worker.is_alive():
      return
    _live_worker = threading.Thread(target=_live_worker_loop, name="live-research-refresh", daemon=True)
    _live_worker.start()


def _stop_live_worker() -> None:
  worker: threading.Thread | None
  with _live_lock:
    _live_state["stop"] = True
    worker = _live_worker
  if worker is not None:
    worker.join(timeout=2.0)


def _live_worker_loop() -> None:
  _refresh_live_cache()
  while True:
    with _live_lock:
      if _live_state["stop"]:
        break
    slept = 0.0
    while slept < LIVE_REFRESH_SECONDS:
      time.sleep(0.5)
      slept += 0.5
      with _live_lock:
        if _live_state["stop"]:
          return
    _refresh_live_cache()


def _refresh_live_cache() -> None:
  with _refresh_guard:
    with _live_lock:
      _live_state["is_refreshing"] = True
    _set_live_progress(5, "starting", "Starting live data refresh")
    try:
      store = LocalResearchStore()
      _set_live_progress(18, "research", "Collecting configured market, news, and macro sources")
      research_result = _load_default_research()
      _set_live_progress(48, "storage", "Saving research records")
      stored_counts = store.save_research_result(research_result)
      _set_live_progress(64, "analysis", "Building indicators, ontology graph, and reasoning paths")
      context = build_analysis_context(research_result, store.load())
      _set_live_progress(84, "graph", "Persisting ontology graph and reasoning paths")
      graph_counts = store.save_graph_and_reasoning(context.graph.triples(), context.reasoning_paths)
      with _live_lock:
        _live_state["research_result"] = research_result
        _live_state["context"] = context
        _live_state["store_summary"] = store.summary()
        _live_state["stored_new_records"] = {**stored_counts, **graph_counts}
        _live_state["last_updated"] = datetime.now()
        _live_state["last_error"] = None
      _set_live_progress(100, "complete", "Live analysis cache is ready", active=False)
    except Exception as exc:
      with _live_lock:
        _live_state["last_error"] = str(exc)
      _set_live_progress(100, "error", str(exc), active=False)
    finally:
      with _live_lock:
        _live_state["is_refreshing"] = False


def _get_or_refresh_live(force_refresh: bool = False) -> dict[str, Any]:
  snapshot = _live_snapshot()
  if snapshot["is_refreshing"] and snapshot["context"] is not None and not force_refresh:
    return snapshot
  last_updated = snapshot["last_updated"]
  stale = (
    last_updated is None
    or (datetime.now() - last_updated).total_seconds() > LIVE_STALE_SECONDS
  )
  if force_refresh or stale or snapshot["context"] is None:
    _refresh_live_cache()
    snapshot = _live_snapshot()

  if snapshot["context"] is None or snapshot["research_result"] is None:
    raise HTTPException(status_code=503, detail="Live research cache is not ready yet.")
  return snapshot


def _live_snapshot() -> dict[str, Any]:
  with _live_lock:
    return {
      "context": _live_state["context"],
      "research_result": _live_state["research_result"],
      "store_summary": dict(_live_state["store_summary"]),
      "stored_new_records": dict(_live_state["stored_new_records"]),
      "last_updated": _live_state["last_updated"],
      "last_error": _live_state["last_error"],
      "is_refreshing": bool(_live_state["is_refreshing"]),
      "progress": dict(_live_state["progress"]),
    }


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
    links: list[dict[str, Any]] = []

    for triple in triples:
        links.append(
            {
                "source": triple.subject,
                "target": triple.object,
                "predicate": triple.predicate,
                "evidence_id": triple.evidence_id,
            }
        )

    importance = _node_importance_map(links)
    nodes: dict[str, dict[str, Any]] = {}
    for link in links:
        for node_id in (link["source"], link["target"]):
            if node_id in nodes:
                continue
            score = round(importance.get(node_id, 0.0), 4)
            nodes[node_id] = _node_payload(node_id, score)

    return {
        "nodes": list(nodes.values()),
        "links": links,
        "reasoning_steps": _build_reasoning_steps(context.reasoning_paths),
        "counts": {"nodes": len(nodes), "links": len(links)},
        "runtime": context.ontology_runtime.as_dict(),
    }


def _node_payload(node_id: str, importance_score: float) -> dict[str, Any]:
    kind = _node_kind(node_id)
    return {
        "id": node_id,
        "label": node_id,
        "kind": kind,
        "importance_score": importance_score,
        "size": _node_size(kind, importance_score),
    }


def _node_kind(node_id: str) -> str:
    if node_id.startswith("NEWS:"):
        return "event"
    if node_id in {"Semiconductor", "Battery", "Finance"}:
        return "sector"
    if node_id in {
        "EarningsGrowth",
        "ProfitabilityQuality",
        "PositiveEventImpact",
        "BuyCandidate",
        "RiskAdjustedSizing",
    }:
        return "support"
    if node_id in {"MacroRateRisk", "NegativeEventRisk", "VolatilityRisk", "LiquidityRisk"}:
        return "risk"
    if node_id in {"ValuationDiscipline", "AggressiveBuy", "ValuationSlightlyHigh"}:
        return "contradiction"
    if _looks_like_ticker(node_id):
        return "ticker"
    return "entity"


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
            "event": 1.15,
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
        "belongsToSector": 0.85,
        "hasTicker": 0.75,
        "isListedOn": 0.70,
        "hasExposureTo": 0.70,
    }.get(predicate, 0.65)


def _node_size(kind: str, score: float) -> float:
    base = {
        "ticker": 10.0,
        "event": 7.0,
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
    .status { padding: 10px 12px; border-radius: 8px; background: #fff7ed; color: var(--warn); border: 1px solid #fed7aa; margin-bottom: 14px; }
    .work-status { margin-top: 14px; padding: 12px; border: 1px solid var(--line); border-radius: 8px; background: #f8fafc; display: none; }
    .work-status.active { display: block; }
    .work-status strong { display: block; margin-bottom: 6px; }
    .work-status .bar { margin-top: 8px; height: 10px; }
    .ontology-scene { grid-column: span 12; min-height: 430px; position: relative; overflow: hidden; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); background: #0f172a; }
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
    .legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
    .ontology-panel { position: absolute; z-index: 2; top: 58px; right: 12px; width: 260px; max-width: calc(100% - 24px); padding: 12px; border-radius: 8px; background: rgba(15,23,42,.86); border: 1px solid rgba(255,255,255,.18); color: #e5e7eb; font-size: 12px; }
    .ontology-panel strong { display: block; font-size: 15px; margin-bottom: 6px; color: #fff; }
    .ontology-panel .muted { color: #cbd5e1; }
    #ontologyCanvas { width: 100%; height: 430px; display: block; }
    #ontologyTooltip { position: absolute; z-index: 3; pointer-events: none; min-width: 160px; max-width: 260px; padding: 8px 10px; border-radius: 6px; background: rgba(15,23,42,.92); color: #fff; border: 1px solid rgba(255,255,255,.18); font-size: 12px; transform: translate(12px, 12px); display: none; }
    @media (max-width: 900px) {
      .shell { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .span-4, .span-8 { grid-column: span 12; }
      .cards { grid-template-columns: 1fr; }
      .stats, .ticker-grid { grid-template-columns: 1fr; }
      .ontology-scene { min-height: 360px; }
      #ontologyCanvas { height: 360px; }
      .reasoning-strip { grid-template-columns: 1fr; bottom: 84px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <h1>개인 투자 분석 시스템</h1>
      <div class="status" id="gate">목표가 확정될 때까지 프로그램은 시작되지 않습니다.</div>
      <form id="goalForm">
        <div class="segmented">
          <label><input type="radio" name="goal_mode" value="rate" checked><span>목표 수익률</span></label>
          <label><input type="radio" name="goal_mode" value="amount"><span>목표 수익금</span></label>
        </div>
        <div class="field" id="rateField"><label for="targetReturn">목표 수익률 (%)</label><input id="targetReturn" name="target_return_rate" type="number" step="0.1" min="0" placeholder="예: 5"></div>
        <div class="field" id="amountField" style="display:none;"><label for="targetAmount">목표 수익금 (원)</label><input id="targetAmount" name="target_profit_amount" type="number" step="10000" min="0" placeholder="예: 50000"></div>
        <div class="field"><label for="periodDays">목표 기간 (일)</label><input id="periodDays" name="period_days" type="number" min="1" value="90"></div>
        <button type="submit">가능성 분석</button>
        <button class="secondary" id="loadResearch" type="button">자료 불러오기</button>
        <div class="work-status" id="workStatus">
          <strong id="workTitle">작업 대기 중</strong>
          <div class="muted" id="workMessage">버튼을 누르면 진행 현황이 표시됩니다.</div>
          <div class="bar"><span id="workProgress"></span></div>
        </div>
      </form>
    </aside>
    <main>
      <div class="grid">
        <section class="panel span-4"><h2>포트폴리오</h2><div class="metric" id="equity">-</div><div class="muted">총 평가금액</div><div class="chips" style="margin-top:12px;"><span class="chip" id="cash">예치금 -</span><span class="chip" id="cashWeight">현금 비중 -</span></div></section>
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
          <div class="ticker-grid" id="tickerMetrics" style="margin-top:12px;"></div>
        </section>
        <section class="ontology-scene">
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
              <label><input type="checkbox" value="support" checked>긍정</label>
              <label><input type="checkbox" value="risk" checked>리스크</label>
              <label><input type="checkbox" value="contradiction" checked>상충</label>
              <label><input type="checkbox" value="sector" checked>섹터</label>
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
            <span class="legend-item"><span class="legend-dot" style="background:#f59e0b"></span>뉴스/이벤트</span>
            <span class="legend-item"><span class="legend-dot" style="background:#22c55e"></span>긍정 신호</span>
            <span class="legend-item"><span class="legend-dot" style="background:#ef4444"></span>리스크</span>
            <span class="legend-item"><span class="legend-dot" style="background:#fb7185"></span>상충 요인</span>
            <span class="legend-item"><span class="legend-dot" style="background:#a78bfa"></span>섹터</span>
          </div>
          <div id="ontologyTooltip"></div>
        </section>
        <section class="panel span-12"><h2>목표 타협안</h2><div class="cards" id="choices"></div><div style="margin-top:14px;"><button id="startButton" disabled>선택한 목표로 시작</button> <button class="secondary" id="resetButton" type="button">초기화</button></div></section>
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
    const fmtWon = new Intl.NumberFormat('ko-KR', { style: 'currency', currency: 'KRW', maximumFractionDigits: 0 });

    async function loadStatus() {
      const data = await (await fetch('/api/status')).json();
      renderStatus(data);
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

    document.querySelectorAll('input[name="goal_mode"]').forEach((node) => {
      node.addEventListener('change', syncGoalMode);
    });

    function syncGoalMode() {
      const mode = document.querySelector('input[name="goal_mode"]:checked').value;
      const rateField = document.getElementById('rateField');
      const amountField = document.getElementById('amountField');
      const rateInput = document.getElementById('targetReturn');
      const amountInput = document.getElementById('targetAmount');
      const rateMode = mode === 'rate';
      rateField.style.display = rateMode ? 'block' : 'none';
      amountField.style.display = rateMode ? 'none' : 'block';
      rateInput.disabled = !rateMode;
      amountInput.disabled = rateMode;
      if (rateMode) amountInput.value = '';
      else rateInput.value = '';
    }

    document.getElementById('goalForm').addEventListener('submit', async (event) => {
      event.preventDefault();
      const payload = currentGoalPayload();
      if (!payload) {
        document.getElementById('output').textContent = '목표 수익률 또는 목표 수익금 중 하나와 목표 기간을 입력하세요.';
        return;
      }
      const stopProgress = startProgressPolling('가능성 분석 중', '시장 자료와 온톨로지 캐시를 확인하고 있습니다.');
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
      const stopProgress = startProgressPolling('자료 불러오는 중', '뉴스, 시세, 온톨로지 관계를 불러오고 있습니다.');
      setBusy(true);
      try {
        const data = await (await fetch('/api/research')).json();
        document.getElementById('output').textContent = JSON.stringify(data, null, 2);
        document.getElementById('relations').innerHTML = data.graph_triples
          .filter((item) => ['supportsSignal', 'contradictsSignal', 'increasesRiskOf'].includes(item.predicate))
          .slice(0, 14)
          .map((item) => `<span class="chip">${item.subject} ${item.predicate} ${item.object}</span>`)
          .join('');
        setWorkStatus('자료 불러오기 완료', '분석 자료를 화면에 반영했습니다.', 100, true);
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
      const mode = payload.goal_mode;
      const period = Number(payload.period_days || 0);
      if (!period || period < 1) return null;
      if (mode === 'rate' && payload.target_return_rate !== '') return payload;
      if (mode === 'amount' && payload.target_profit_amount !== '') return payload;
      return null;
    }

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

    async function refreshLiveSnapshot() {
      if (liveRefreshBusy) return;
      liveRefreshBusy = true;
      const badge = document.getElementById('liveRefreshBadge');
      try {
        const goal = currentGoalPayload();
        if (goal) lastGoalPayload = goal;
        const res = await fetch('/api/live-snapshot', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ goal: lastGoalPayload, force_refresh: false })
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

    function renderStatus(data) {
      document.getElementById('equity').textContent = fmtWon.format(data.equity);
      document.getElementById('cash').textContent = `예치금 ${fmtWon.format(data.cash)}`;
      document.getElementById('cashWeight').textContent = `현금 비중 ${(data.cash_weight * 100).toFixed(1)}%`;
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
      document.getElementById('storeStats').innerHTML = `
        <div class="stat"><strong>${store.events || 0}</strong><span class="muted">저장된 이벤트</span></div>
        <div class="stat"><strong>${store.graph_triples || 0}</strong><span class="muted">저장된 그래프 관계</span></div>
        <div class="stat"><strong>${store.reasoning_paths || 0}</strong><span class="muted">저장된 추론 경로</span></div>
        <div class="stat"><strong>${data.store_path || '-'}</strong><span class="muted">저장 위치</span></div>
      `;
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
        tooltip.style.display = 'block';
        tooltip.style.left = '12px';
        tooltip.style.top = '52px';
        tooltip.textContent = '3D 라이브러리를 불러오지 못했습니다.';
        return;
      }

      if (graphState) {
        graphState.stop = true;
        graphState.renderer.dispose();
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

      const nodes = data.nodes.map((node, index) => ({ ...node, index, position: nodePosition(index, data.nodes.length) }));
      const nodeMap = new Map(nodes.map((node) => [node.id, node]));
      const raycaster = new THREE.Raycaster();
      const pointer = new THREE.Vector2(99, 99);
      const nodeMeshes = [];
      const labelSprites = [];
      const linkLines = [];
      const nodeMeshById = new Map();
      const labelById = new Map();
      const lineByKey = new Map();
      const labelState = { visible: false };
      const reasoningState = {
        steps: data.reasoning_steps || [],
        playing: true,
        currentIndex: -1,
        startedAt: performance.now(),
        stepMs: 1450,
        activeNodeIds: new Set(),
        activeLinkKeys: new Set(),
      };
      const activeKinds = new Set(['ticker', 'event', 'sector', 'support', 'risk', 'contradiction', 'entity']);

      for (const link of data.links) {
        const source = nodeMap.get(link.source);
        const target = nodeMap.get(link.target);
        if (!source || !target) continue;
        const geometry = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(...source.position),
          new THREE.Vector3(...target.position),
        ]);
        const material = new THREE.LineBasicMaterial({ color: edgeColor(link.predicate), transparent: true, opacity: 0.42 });
        const line = new THREE.Line(geometry, material);
        line.userData = { source: link.source, target: link.target, predicate: link.predicate };
        line.userData.baseColor = edgeColor(link.predicate);
        line.userData.baseOpacity = 0.42;
        root.add(line);
        linkLines.push(line);
        lineByKey.set(linkKey(link.source, link.target, link.predicate), line);
      }

      for (const node of nodes) {
        const geometry = new THREE.SphereGeometry(nodeRadius(node), 18, 18);
        const material = new THREE.MeshStandardMaterial({
          color: nodeColor(node.kind),
          emissive: nodeColor(node.kind),
          emissiveIntensity: 0.18,
          roughness: 0.5,
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(...node.position);
        mesh.userData = node;
        root.add(mesh);
        nodeMeshes.push(mesh);
        nodeMeshById.set(node.id, mesh);
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
        for (const mesh of nodeMeshes) {
          mesh.visible = activeKinds.has(mesh.userData.kind);
        }
        for (const sprite of labelSprites) {
          sprite.visible = (labelState.visible || reasoningState.activeNodeIds.has(sprite.userData.id)) && activeKinds.has(sprite.userData.kind);
        }
        for (const line of linkLines) {
          const source = nodeMap.get(line.userData.source);
          const target = nodeMap.get(line.userData.target);
          line.visible = Boolean(source && target && activeKinds.has(source.kind) && activeKinds.has(target.kind));
        }
      }

      function updateReasoning(now) {
        if (!reasoningState.steps.length) {
          document.getElementById('reasoningBadge').textContent = '추론 단계 0/0';
          return;
        }
        if (reasoningState.playing) {
          const index = Math.floor((now - reasoningState.startedAt) / reasoningState.stepMs) % reasoningState.steps.length;
          if (index !== reasoningState.currentIndex) setActiveReasoningStep(index);
        }
      }

      function setActiveReasoningStep(index) {
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
          mesh.scale.setScalar(active ? 1.25 + pulse * 0.32 : 1);
          mesh.material.emissiveIntensity = active ? 0.95 + pulse * 0.65 : 0.18;
        }
        for (const line of linkLines) {
          const active = reasoningState.activeLinkKeys.has(linkKey(line.userData.source, line.userData.target, line.userData.predicate));
          line.material.opacity = active ? 0.95 : line.userData.baseOpacity;
          line.material.color.setHex(active ? 0xfacc15 : line.userData.baseColor);
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
          tooltip.innerHTML = `<strong>${hit.object.userData.label}</strong><br>${kindLabel(hit.object.userData.kind)} · 연결 ${degree(hit.object.userData.id, data.links)}개 · 중요도 ${Number(hit.object.userData.importance_score || 0).toFixed(2)}`;
        } else {
          tooltip.style.display = 'none';
        }

        renderer.render(scene, camera);
      }
      requestAnimationFrame(animate);
      updateVisibility();
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

    function nodePosition(index, total) {
      const golden = Math.PI * (3 - Math.sqrt(5));
      const y = 1 - (index / Math.max(1, total - 1)) * 2;
      const radius = Math.sqrt(Math.max(0, 1 - y * y));
      const theta = golden * index;
      const scale = 260;
      return [Math.cos(theta) * radius * scale, y * scale, Math.sin(theta) * radius * scale];
    }

    function nodeColor(kind) {
      return {
        ticker: 0x38bdf8,
        event: 0xf59e0b,
        sector: 0xa78bfa,
        support: 0x22c55e,
        risk: 0xef4444,
        contradiction: 0xfb7185,
        entity: 0xe5e7eb
      }[kind] || 0xe5e7eb;
    }

    function edgeColor(predicate) {
      if (predicate === 'supportsSignal') return 0x22c55e;
      if (predicate === 'increasesRiskOf') return 0xef4444;
      if (predicate === 'contradictsSignal') return 0xfb7185;
      return 0x94a3b8;
    }

    function nodeRadius(node) {
      const size = Number(node && node.size);
      if (Number.isFinite(size) && size > 0) return Math.max(5, Math.min(30, size));
      const kind = node && node.kind;
      return kind === 'ticker' ? 13 : kind === 'event' ? 8 : 10;
    }

    function kindLabel(kind) {
      return {
        ticker: '종목',
        event: '뉴스/공시 이벤트',
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
    });
    loadStatus();
    loadDiagnostics();
    loadOntologyGraph();
    refreshLiveSnapshot();
    setInterval(refreshLiveSnapshot, 5000);
    syncGoalMode();
  </script>
</body>
</html>
"""

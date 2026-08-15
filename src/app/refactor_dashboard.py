from __future__ import annotations

from contextlib import closing
import json
import os
import re
import sqlite3
import threading
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.config.refactor_profile import load_refactor_profile
from app.features import session_structure
from app.strategy.experts import ALL_EXPERT_TYPES


_SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,20}$")
_MARKET_SERIES_CACHE_CONDITION = threading.Condition()
_MARKET_SERIES_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
_MARKET_SERIES_REFRESHING: set[tuple[str, str]] = set()


def build_refactor_dashboard(root: str | Path = ".") -> dict[str, Any]:
    """Build a read-only operations view of the refactored trading path."""
    base = Path(root)
    profile, profile_error = _profile(base)
    counterfactual = _json(base / "data/reports/refactor_counterfactual_evaluation.json")
    benchmark = _json(base / "data/reports/strategy_utility_openvino.json")
    gnn_training = _json(base / "data/models/strategy_utility/rgcn_shadow.json")
    readiness, readiness_meta = _latest_json_with_freshness(
        base / "data/reports",
        "live_readiness_*.json",
        max_age_seconds=max(
            60.0,
            _env_float("REFACTOR_READINESS_MAX_AGE_SEC", 900.0),
        ),
    )
    lifecycle = _lifecycle(base / "data/store/trading-lifecycle.sqlite3")
    shadow = _shadow(base / "logs/refactor-shadow-comparison.jsonl")
    journal = _journal(base / "data/store/causal-order-journal.jsonl")

    flags = dict((profile or {}).get("flags") or {})
    mode = str((profile or {}).get("mode") or "invalid")
    broker_submission = bool((profile or {}).get("broker_submission_enabled"))
    model_trained = (
        (base / "data/models/strategy_utility/rgcn_shadow.npz").exists()
        and int(gnn_training.get("rows") or 0) > 0
    )
    data_promoted = bool(counterfactual.get("promotion_eligible"))
    npu_promoted = bool(benchmark.get("promotion_eligible"))
    policy_ready = (
        bool(readiness)
        and bool(readiness_meta.get("fresh"))
        and not bool(readiness.get("failures"))
    )
    execution_ready = bool(flags.get("strategy_owned_execution"))
    live_enabled = bool(flags.get("live_enabled"))

    gates = [
        _gate(
            "Point-in-time 데이터",
            data_promoted,
            "대표 KRX/NXT 데이터와 이벤트·섹터·세션 이력이 부족합니다.",
            counterfactual.get("status") or "보고서 없음",
        ),
        _gate(
            "효용 모델 학습·교정",
            model_trained,
            (
                f"{int(gnn_training.get('rows') or 0):,}개 인과 라벨로 교정했습니다. "
                "관찰 전용이며 실거래 권한은 부여하지 않습니다."
                if model_trained
                else "학습·교정된 shadow 체크포인트가 없습니다."
            ),
            "CALIBRATED SHADOW" if model_trained else "UNTRAINED",
        ),
        _gate(
            "NPU 승격",
            npu_promoted,
            "CPU보다 느리고 엄격한 utility parity 기준을 통과하지 못했습니다.",
            "CPU 유지",
        ),
        _gate(
            "거래 위험 정책",
            policy_ready,
            _readiness_reason(readiness, readiness_meta),
            (
                "READY"
                if policy_ready
                else (
                    "STALE_REPORT"
                    if readiness_meta.get("available") and not readiness_meta.get("fresh")
                    else "BLOCKED"
                )
            ),
        ),
        _gate(
            "전략 소유 실행",
            execution_ready and live_enabled and broker_submission,
            "명시적 live·소유 실행·broker submission이 모두 켜져야 합니다.",
            "DISARMED" if not broker_submission else "ARMED",
        ),
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "runtime_scope": "refactor_shadow" if mode == "shadow" else "refactor_execution",
        "authoritative_for_live_orders": bool(mode in {"canary", "live"}),
        "profile_valid": profile_error is None,
        "profile_error": profile_error,
        "broker_submission_enabled": broker_submission,
        "live_order_capable": bool(
            mode in {"canary", "live"}
            and broker_submission
            and live_enabled
            and execution_ready
            and all(gate["passed"] for gate in gates)
        ),
        "flags": flags,
        "readiness_report": readiness_meta,
        "pipeline": [
            {
                "id": "market",
                "label": "시장 이벤트",
                "detail": "KIS WebSocket → 로컬 상태·증분 봉",
                "active": bool(flags.get("websocket_market_data")),
            },
            {
                "id": "ontology",
                "label": "온톨로지 게이트",
                "detail": "TTL·필수 사실·전략 호환성",
                "active": bool(flags.get("ontology_router")),
            },
            {
                "id": "utility",
                "label": "전략 효용",
                "detail": "CPU shadow · NoTrade 포함",
                "active": bool(flags.get("gnn_shadow") or flags.get("gnn_rerank")),
            },
            {
                "id": "owner",
                "label": "전략 소유권",
                "detail": "진입부터 청산까지 단일 StrategyInstance",
                "active": execution_ready,
            },
            {
                "id": "risk",
                "label": "위험 판정",
                "detail": "승인·축소·거절·비상청산",
                "active": execution_ready,
            },
            {
                "id": "broker",
                "label": "KIS 실행",
                "detail": "인과 저널·멱등키·broker 원장",
                "active": broker_submission,
            },
        ],
        "promotion_gates": gates,
        "lifecycle": lifecycle,
        "shadow": shadow,
        "journal": journal,
        "evaluation": {
            "status": counterfactual.get("status") or "NO_REPORT",
            "snapshots": (counterfactual.get("labels") or {}).get("snapshots", 0),
            "strategy_labels": (counterfactual.get("labels") or {}).get("strategy_labels", 0),
            "dates": (counterfactual.get("coverage") or {}).get("distinct_utc_dates", 0),
            "symbols": (counterfactual.get("coverage") or {}).get("symbols", 0),
            "walk_forward_observations": (
                counterfactual.get("walk_forward_tabular_baseline") or {}
            ).get("observations", 0),
            "selected_trades": (
                counterfactual.get("walk_forward_tabular_baseline") or {}
            ).get("selected_trades", 0),
            "strategy_metrics": counterfactual.get("strategy_metrics") or {},
        },
        "gnn_training": gnn_training,
        "devices": {
            "selected": "CPU",
            "cpu": benchmark.get("cpu") or {},
            "npu": benchmark.get("npu") or {},
            "parity": benchmark.get("parity") or {},
            "npu_promotion_eligible": npu_promoted,
        },
    }


def build_strategy_market_view(
    symbol: str | None = None,
    *,
    limit: int = 180,
    root: str | Path = ".",
    selection_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read-only chart, ontology, strategy, and execution view for one symbol."""
    base = Path(root)
    dashboard = build_refactor_dashboard(base)
    shadow = dashboard.get("shadow") or {}
    lifecycle = dashboard.get("lifecycle") or {}
    requested = str(symbol or "").strip().upper()
    if requested and not _SYMBOL_PATTERN.fullmatch(requested):
        raise ValueError("invalid symbol")
    market_database = base / "data/store/realtime_market_data.sqlite3"
    selected = requested or _default_symbol(shadow, lifecycle, market_database) or ""
    safe_limit = max(30, min(390, int(limit or 180)))
    market = _market_series(
        market_database,
        selected,
        safe_limit,
    )
    candidates = _candidate_symbols(
        market_database,
        shadow,
        selected,
    )
    selection = _selection_for_symbol(shadow, selected)
    if selection_override:
        selection = {**selection, **dict(selection_override)}
    if not selection.get("sector_rank_table"):
        selection["sector_rank_table"] = _live_sector_rank_table()
    strategy_id = str(
        selection.get("strategy_id")
        or selection.get("ontology_strategy_id")
        or ""
    )
    algorithm = _algorithm(strategy_id)
    execution = _execution_for_symbol(
        base / "data/store/causal-order-journal.jsonl",
        selected,
        selection,
    )
    decision_ontology = _decision_ontology(
        market,
        selection,
        event_feed=_event_feed_status(
            base / "data/store/news_sentiment.sqlite3", selected
        ),
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": dashboard.get("mode"),
        "live_order_capable": dashboard.get("live_order_capable"),
        "symbol": selected,
        "candidates": candidates,
        "market": market,
        "selection": selection,
        "algorithm": algorithm,
        "decision_ontology": decision_ontology,
        "execution": execution,
        "promotion_gates": dashboard.get("promotion_gates") or [],
    }


def build_strategy_market_stream(
    symbol: str,
    *,
    limit: int = 30,
    root: str | Path = ".",
) -> dict[str, Any]:
    """Lightweight one-second polling payload for the moving trading chart."""
    selected = str(symbol or "").strip().upper()
    if not selected or not _SYMBOL_PATTERN.fullmatch(selected):
        raise ValueError("invalid symbol")
    safe_limit = max(2, min(90, int(limit or 30)))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": selected,
        "market": _market_series(
            Path(root) / "data/store/realtime_market_data.sqlite3",
            selected,
            safe_limit,
        ),
    }


def _profile(base: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        profile = load_refactor_profile(base / "config/refactor_profile.json")
        return {
            "mode": profile.mode.value,
            "broker_submission_enabled": profile.broker_submission_enabled,
            "maximum_order_notional": profile.maximum_order_notional,
            "allowed_symbols": list(profile.allowed_symbols),
            "flags": asdict(profile.flags),
        }, None
    except Exception as exc:  # noqa: BLE001 - diagnostics must degrade safely.
        return None, f"{type(exc).__name__}: {exc}"


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _latest_json(directory: Path, pattern: str) -> dict[str, Any]:
    try:
        candidates = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    except OSError:
        return {}
    return _json(candidates[-1]) if candidates else {}


def _latest_json_with_freshness(
    directory: Path,
    pattern: str,
    *,
    max_age_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the newest report plus provenance; an old file is never current state."""
    try:
        candidates = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    except OSError:
        candidates = []
    if not candidates:
        return {}, {
            "available": False,
            "fresh": False,
            "reason": "READINESS_REPORT_MISSING",
            "max_age_seconds": float(max_age_seconds),
        }
    latest = candidates[-1]
    try:
        modified_at = datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return {}, {
            "available": False,
            "fresh": False,
            "reason": "READINESS_REPORT_UNREADABLE",
            "path": str(latest),
            "max_age_seconds": float(max_age_seconds),
        }
    age_seconds = max(0.0, (datetime.now(timezone.utc) - modified_at).total_seconds())
    payload = _json(latest)
    fresh = bool(payload) and age_seconds <= float(max_age_seconds)
    return payload, {
        "available": bool(payload),
        "fresh": fresh,
        "reason": "CURRENT" if fresh else "STALE_READINESS_REPORT",
        "path": str(latest),
        "modified_at": modified_at.isoformat(),
        "age_seconds": round(age_seconds, 3),
        "max_age_seconds": float(max_age_seconds),
    }


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _lifecycle(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "instances": 0, "open_positions": 0, "states": {}}
    try:
        connection = sqlite3.connect(path)
        try:
            states = {
                str(status): int(count)
                for status, count in connection.execute(
                    "select status, count(*) from strategy_instances group by status"
                )
            }
            positions = [
                {
                    "symbol": symbol,
                    "quantity": int(quantity),
                    "strategy_id": strategy_id,
                    "strategy_instance_id": instance_id,
                    "opened_at": opened_at,
                }
                for symbol, quantity, strategy_id, instance_id, opened_at in connection.execute(
                    """
                    select symbol, quantity, origin_strategy_id,
                           strategy_instance_id, opened_at
                    from positions where quantity != 0
                    order by opened_at desc limit 20
                    """
                )
            ]
        finally:
            connection.close()
        return {
            "available": True,
            "instances": sum(states.values()),
            "open_positions": len(positions),
            "states": states,
            "positions": positions,
        }
    except sqlite3.Error as exc:
        return {
            "available": False,
            "instances": 0,
            "open_positions": 0,
            "states": {},
            "error": str(exc),
        }


def _shadow(path: Path) -> dict[str, Any]:
    rows = _json_lines(path, 200)
    latest = rows[-1] if rows else {}
    latest_by_symbol: dict[str, dict[str, Any]] = {}
    for row in reversed(rows):
        symbol = str(row.get("symbol") or "").upper()
        if symbol and symbol not in latest_by_symbol:
            latest_by_symbol[symbol] = row
    return {
        "available": bool(rows),
        "samples": len(rows),
        "action_agreement_rate": (
            sum(bool(row.get("action_agreement")) for row in rows) / len(rows) if rows else None
        ),
        "strategy_agreement_rate": (
            sum(bool(row.get("strategy_agreement")) for row in rows) / len(rows)
            if rows
            else None
        ),
        "latest": latest,
        "latest_by_symbol": latest_by_symbol,
    }


def _journal(path: Path) -> dict[str, Any]:
    rows = _json_lines(path, 500)
    counts = Counter(str(row.get("event_type") or "unknown") for row in rows)
    return {
        "available": bool(rows),
        "events": len(rows),
        "event_counts": dict(sorted(counts.items())),
        "latest_event_type": rows[-1].get("event_type") if rows else None,
    }


def _json_lines(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = _tail_lines(path, limit)
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _tail_lines(path: Path, limit: int, *, maximum_bytes: int = 512 * 1024) -> list[str]:
    """Read a bounded diagnostic tail so dashboard polling stays cheap."""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        end = handle.tell()
        start = max(0, end - maximum_bytes)
        handle.seek(start)
        payload = handle.read()
    if start:
        _, _, payload = payload.partition(b"\n")
    return payload.decode("utf-8", errors="replace").splitlines()[-limit:]


def _gate(label: str, passed: bool, reason: str, value: str) -> dict[str, Any]:
    return {"label": label, "passed": passed, "reason": reason, "value": value}


def _readiness_reason(
    readiness: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> str:
    report_meta = meta or {}
    if report_meta.get("available") and not report_meta.get("fresh"):
        age = float(report_meta.get("age_seconds") or 0.0)
        return (
            f"준비도 보고서가 {age / 60.0:.1f}분 전 자료라 현재 장애 판정에 사용하지 않습니다. "
            "현재 LIVE_TRADING 상태는 운영 진단 API에서 별도로 확인해야 합니다."
        )
    if not readiness:
        return "현재 유효한 준비도 보고서가 없습니다."
    failures = readiness.get("failures") or {}
    if failures:
        return " / ".join(str(value) for value in failures.values())
    return "최근 readiness 보고서가 없거나 모든 정책 검사를 통과했습니다."


def _default_symbol(
    shadow: dict[str, Any],
    lifecycle: dict[str, Any],
    market_database: Path | None = None,
) -> str | None:
    positions = lifecycle.get("positions") or []
    for position in positions:
        symbol = str(position.get("symbol") or "").strip().upper()
        if _SYMBOL_PATTERN.fullmatch(symbol):
            return symbol
    latest = shadow.get("latest") or {}
    symbol = str(latest.get("symbol") or "").strip().upper()
    if _SYMBOL_PATTERN.fullmatch(symbol):
        return symbol
    for candidate in (shadow.get("latest_by_symbol") or {}):
        symbol = str(candidate or "").strip().upper()
        if _SYMBOL_PATTERN.fullmatch(symbol):
            return symbol
    if market_database is None or not market_database.exists():
        return None
    try:
        connection = sqlite3.connect(market_database)
        try:
            row = connection.execute(
                """
                select symbol from realtime_minute_bars
                where symbol is not null and trim(symbol) <> ''
                group by symbol
                order by max(minute_start) desc, count(*) desc
                limit 1
                """
            ).fetchone()
        finally:
            connection.close()
    except sqlite3.Error:
        return None
    symbol = str(row[0] if row else "").strip().upper()
    return symbol if _SYMBOL_PATTERN.fullmatch(symbol) else None


def _selection_for_symbol(shadow: dict[str, Any], symbol: str) -> dict[str, Any]:
    row = (shadow.get("latest_by_symbol") or {}).get(symbol) or {}
    decisions = list(row.get("decisions") or [])
    ontology = next(
        (decision for decision in decisions if decision.get("path") == "ontology"),
        {},
    )
    preferred = next(
        (
            decision
            for path in ("cpu_gnn", "ontology", "legacy")
            for decision in decisions
            if decision.get("path") == path
        ),
        {},
    )
    ontology_action = str(ontology.get("action") or "NO_TRADE").upper()
    ontology_strategy_id = ontology.get("strategy_id")
    ontology_allowed = bool(ontology_strategy_id) and ontology_action in {
        "ACTIVATE_STRATEGY",
        "ADMISSIBLE",
        "ALLOW",
        "ALLOWED",
        "BUY",
    }
    as_of = row.get("as_of")
    evidence_age_seconds: float | None = None
    if as_of:
        try:
            parsed_as_of = datetime.fromisoformat(str(as_of).replace("Z", "+00:00"))
            if parsed_as_of.tzinfo is None:
                parsed_as_of = parsed_as_of.replace(tzinfo=timezone.utc)
            evidence_age_seconds = max(
                0.0,
                (datetime.now(timezone.utc) - parsed_as_of).total_seconds(),
            )
        except (TypeError, ValueError):
            evidence_age_seconds = None
    try:
        maximum_age = max(
            10.0,
            float(os.getenv("STRATEGY_SESSION_EVIDENCE_MAX_AGE_SEC", "120")),
        )
    except (TypeError, ValueError):
        maximum_age = 120.0
    evidence_missing = not bool(row)
    evidence_stale = bool(
        row
        and (
            evidence_age_seconds is None
            or evidence_age_seconds > maximum_age
        )
    )
    reason_codes = list(preferred.get("reason_codes") or [])
    if evidence_missing:
        reason_codes.append("DECISION_EVIDENCE_MISSING")
    elif evidence_stale:
        reason_codes.append("DECISION_EVIDENCE_STALE")
    return {
        "as_of": as_of,
        "action": preferred.get("action") or "NO_TRADE",
        "strategy_id": preferred.get("strategy_id"),
        "utility": preferred.get("utility"),
        "realtime_trust_score": preferred.get("realtime_trust_score"),
        "realtime_trust_samples": preferred.get("realtime_trust_samples"),
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "path": preferred.get("path") or "ontology",
        "evidence_status": (
            "MISSING" if evidence_missing else "STALE" if evidence_stale else "CURRENT"
        ),
        "evidence_stale": evidence_stale,
        "evidence_age_seconds": evidence_age_seconds,
        "evidence_max_age_seconds": maximum_age,
        "ontology_allowed": ontology_allowed,
        "ontology_action": ontology_action,
        "ontology_strategy_id": ontology_strategy_id,
        "ontology_reason_codes": ontology.get("reason_codes") or [],
        "all_decisions": decisions,
        "sector_rank_table": row.get("sector_rank_table") or next(
            (
                (decision.get("diagnostics") or {}).get("sector_rank_table")
                for decision in decisions
                if isinstance(decision.get("diagnostics"), dict)
                and (decision.get("diagnostics") or {}).get("sector_rank_table")
            ),
            None,
        ),
    }


def _event_feed_status(path: Path, symbol: str) -> dict[str, Any]:
    """Report an actual event-store query, including an honest zero-row result."""
    result = {"queried": False, "samples": 0, "updated_at": None, "score": None}
    if not path.exists():
        return result
    try:
        with closing(sqlite3.connect(path)) as connection:
            row = connection.execute(
                "select count(*), max(observed_at) from news_sentiment where ticker = ?",
                (symbol,),
            ).fetchone()
            latest = connection.execute(
                "select score from news_sentiment where ticker = ? order by observed_at desc limit 1",
                (symbol,),
            ).fetchone()
    except sqlite3.Error:
        return result
    return {
        "queried": True,
        "samples": int(row[0] or 0) if row else 0,
        "updated_at": row[1] if row else None,
        "score": float(latest[0]) if latest else None,
    }


def _live_sector_rank_table() -> dict[str, Any] | None:
    """Read the macro reasoner's delivered table from the live diagnostic feed."""
    try:
        from app.graph.macro_micro_feed import snapshot

        bundle = snapshot() or {}
    except Exception:  # noqa: BLE001 - dashboard diagnostics degrade safely.
        return None
    macro = bundle.get("macro_result") if isinstance(bundle, dict) else None
    table = macro.get("sector_rank_table") if isinstance(macro, dict) else None
    return dict(table) if isinstance(table, dict) else None


def _candidate_symbols(
    database: Path,
    shadow: dict[str, Any],
    selected: str,
) -> list[dict[str, Any]]:
    symbols = [selected]
    symbols.extend((shadow.get("latest_by_symbol") or {}).keys())
    if database.exists():
        try:
            connection = sqlite3.connect(database)
            try:
                symbols.extend(
                    str(row[0]).upper()
                    for row in connection.execute(
                        """
                        select symbol from realtime_minute_bars
                        group by symbol
                        order by max(minute_start) desc, count(*) desc
                        limit 12
                        """
                    )
                )
            finally:
                connection.close()
        except sqlite3.Error:
            pass
    seen: set[str] = set()
    result = []
    for candidate in symbols:
        normalized = str(candidate).upper()
        if not normalized or normalized in seen or not _SYMBOL_PATTERN.fullmatch(normalized):
            continue
        seen.add(normalized)
        selection = _selection_for_symbol(shadow, normalized)
        ontology_allowed = bool(selection["ontology_allowed"])
        result.append(
            {
                "symbol": normalized,
                "action": (
                    selection["ontology_action"]
                    if ontology_allowed
                    else selection["action"]
                ),
                "strategy_id": (
                    selection["ontology_strategy_id"]
                    if ontology_allowed
                    else selection["strategy_id"]
                ),
                "ontology_allowed": ontology_allowed,
                "selected": normalized == selected,
            }
        )
        if len(result) >= 12:
            break
    return result


def _market_series(database: Path, symbol: str, limit: int) -> dict[str, Any]:
    """Cached UI read model over the live SQLite journal.

    The terminal polls both the full view and the lightweight stream. They used
    to launch identical 2,500-tick/600-book reads independently, so a slow read
    made the next poll overlap and created an unbounded queue of duplicate
    queries. Trading does not consume this function; a short stale-while-refresh
    cache is therefore safe and keeps the GUI responsive without weakening any
    execution freshness gate.
    """
    safe_limit = max(1, min(390, int(limit or 180)))
    key = (str(database.resolve()), str(symbol or "").upper())
    try:
        cache_seconds = max(
            0.25, float(os.getenv("MARKET_VIEW_CACHE_SECONDS", "1.0"))
        )
    except (TypeError, ValueError):
        cache_seconds = 1.0
    now = time.monotonic()
    with _MARKET_SERIES_CACHE_CONDITION:
        cached = _MARKET_SERIES_CACHE.get(key)
        if cached is not None:
            cached_at, payload = cached
            if now - cached_at >= cache_seconds and key not in _MARKET_SERIES_REFRESHING:
                _MARKET_SERIES_REFRESHING.add(key)
                threading.Thread(
                    target=_refresh_market_series_cache,
                    args=(database, key),
                    name=f"market-view-refresh-{key[1]}",
                    daemon=True,
                ).start()
            return _project_market_series(payload, safe_limit)
        if key in _MARKET_SERIES_REFRESHING:
            _MARKET_SERIES_CACHE_CONDITION.wait(timeout=10.0)
            cached = _MARKET_SERIES_CACHE.get(key)
            if cached is not None:
                return _project_market_series(cached[1], safe_limit)
        _MARKET_SERIES_REFRESHING.add(key)
    try:
        payload = _market_series_uncached(database, key[1], 390)
    finally:
        # ``payload`` may not exist only on an unexpected non-SQLite exception.
        with _MARKET_SERIES_CACHE_CONDITION:
            if "payload" in locals():
                _MARKET_SERIES_CACHE[key] = (time.monotonic(), payload)
                if len(_MARKET_SERIES_CACHE) > 32:
                    oldest = min(
                        _MARKET_SERIES_CACHE,
                        key=lambda item: _MARKET_SERIES_CACHE[item][0],
                    )
                    if oldest != key:
                        _MARKET_SERIES_CACHE.pop(oldest, None)
            _MARKET_SERIES_REFRESHING.discard(key)
            _MARKET_SERIES_CACHE_CONDITION.notify_all()
    return _project_market_series(payload, safe_limit)


def _refresh_market_series_cache(
    database: Path, key: tuple[str, str]
) -> None:
    try:
        payload = _market_series_uncached(database, key[1], 390)
    except Exception:  # noqa: BLE001 - retain the last good UI snapshot.
        payload = None
    with _MARKET_SERIES_CACHE_CONDITION:
        if payload is not None:
            _MARKET_SERIES_CACHE[key] = (time.monotonic(), payload)
        _MARKET_SERIES_REFRESHING.discard(key)
        _MARKET_SERIES_CACHE_CONDITION.notify_all()


def _project_market_series(payload: dict[str, Any], limit: int) -> dict[str, Any]:
    projected = dict(payload)
    projected["bars"] = list(payload.get("bars") or ())[-limit:]
    return projected


def _market_series_uncached(database: Path, symbol: str, limit: int) -> dict[str, Any]:
    empty = {
        "symbol": symbol,
        "bars": [],
        "second_bars": [],
        "microstructure": {},
        "latest_tick": None,
        "latest_orderbook": None,
        "last_price": None,
        "change_rate": None,
        "stale": True,
        "trade_stale": True,
        "orderbook_stale": True,
        "feed_state": "STALE",
        "last_event_at": None,
        "last_price_source": None,
    }
    if not database.exists():
        return empty
    try:
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        try:
            orderbook_columns = {
                str(row[1])
                for row in connection.execute(
                    "pragma table_info(realtime_orderbook)"
                ).fetchall()
            }
            received_at_expr = (
                "received_at"
                if "received_at" in orderbook_columns
                else "exchange_timestamp as received_at"
            )
            source_expr = (
                "source"
                if "source" in orderbook_columns
                else "'unknown' as source"
            )
            levels_expr = (
                "levels_json"
                if "levels_json" in orderbook_columns
                else "'[]' as levels_json"
            )
            bars = list(
                reversed(
                    connection.execute(
                        """
                        select symbol, minute_start, open, high, low, close, volume,
                               vwap, trade_count, spread_bps, orderbook_imbalance,
                               liquidity_score, volatility, last_update_age_ms
                        from realtime_minute_bars
                        where symbol = ?
                        order by minute_start desc limit ?
                        """,
                        (symbol, limit),
                    ).fetchall()
                )
            )
            tick = connection.execute(
                """
                select exchange_timestamp, received_at, price, volume,
                       trade_direction, latency_ms
                from realtime_ticks where symbol = ?
                order by exchange_timestamp desc limit 1
                """,
                (symbol,),
            ).fetchone()
            recent_ticks = list(
                reversed(
                    connection.execute(
                        """
                        select exchange_timestamp, received_at, price, volume,
                               trade_direction, latency_ms
                        from realtime_ticks where symbol = ?
                        order by exchange_timestamp desc limit 2500
                        """,
                        (symbol,),
                    ).fetchall()
                )
            )
            book = connection.execute(
                f"""
                select exchange_timestamp, {received_at_expr}, {source_expr},
                       best_bid, best_ask, spread_bps,
                       total_bid_volume, total_ask_volume, imbalance, latency_ms,
                       {levels_expr}
                from realtime_orderbook where symbol = ?
                order by exchange_timestamp desc limit 1
                """,
                (symbol,),
            ).fetchone()
            recent_books = list(
                reversed(
                    connection.execute(
                        """
                        select exchange_timestamp, best_bid, best_ask, spread_bps,
                               total_bid_volume, total_ask_volume, imbalance, latency_ms
                        from realtime_orderbook where symbol = ?
                        order by exchange_timestamp desc limit 600
                        """,
                        (symbol,),
                    ).fetchall()
                )
            )
        finally:
            connection.close()
    except sqlite3.Error as exc:
        return {**empty, "error": str(exc)}
    payload_bars = [
        {
            "time": row["minute_start"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
            "vwap": row["vwap"],
            "trade_count": row["trade_count"],
            "spread_bps": row["spread_bps"],
            "orderbook_imbalance": row["orderbook_imbalance"],
            "liquidity_score": row["liquidity_score"],
            "volatility": row["volatility"],
        }
        for row in bars
    ]
    first_close = float(payload_bars[0]["close"]) if payload_bars else None
    last_close = float(payload_bars[-1]["close"]) if payload_bars else None
    now = datetime.now(timezone.utc)

    def event_age(row: sqlite3.Row | None) -> tuple[str | None, float | None]:
        if row is None:
            return None, None
        value = str(row["exchange_timestamp"])
        try:
            moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            return value, max(0.0, (now - moment.astimezone(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return value, None

    tick_at, tick_age = event_age(tick)
    book_at, book_age = event_age(book)
    trade_stale = tick_age is None or tick_age > 90
    orderbook_stale = book_age is None or book_age > 90
    stale = trade_stale and orderbook_stale
    event_candidates = [
        (age, at, source)
        for age, at, source in (
            (tick_age, tick_at, "trade"),
            (book_age, book_at, "quote_mid"),
        )
        if age is not None and at is not None
    ]
    latest_event = min(
        event_candidates,
        default=(None, None, None),
        key=lambda item: item[0],
    )
    event_at = latest_event[1]
    last_price_source = latest_event[2]
    quote_mid = None
    if book is not None:
        bid = float(book["best_bid"] or 0)
        ask = float(book["best_ask"] or 0)
        if bid > 0 and ask >= bid:
            quote_mid = (bid + ask) / 2
    display_price = (
        quote_mid
        if last_price_source == "quote_mid" and quote_mid is not None
        else (float(tick["price"]) if tick is not None else last_close)
    )
    second_bars, microstructure = _second_level_market_series(
        recent_ticks,
        recent_books,
    )
    microstructure["ready"] = bool(
        microstructure.get("temporal_ready")
        and not trade_stale
        and not orderbook_stale
    )
    microstructure["trade_age_seconds"] = tick_age
    microstructure["orderbook_age_seconds"] = book_age
    microstructure["display_feed_state"] = (
        "LIVE_TRADE"
        if not trade_stale
        else ("LIVE_QUOTE_ONLY" if not orderbook_stale else "STALE")
    )
    if trade_stale:
        microstructure["block_reason"] = "STALE_SECOND_DATA"
    elif not microstructure.get("temporal_ready"):
        microstructure["block_reason"] = "INSUFFICIENT_SECOND_SAMPLES"
    elif orderbook_stale:
        microstructure["block_reason"] = "STALE_OR_MISSING_ORDERBOOK"
    else:
        microstructure["block_reason"] = None
    book_payload = dict(book) if book is not None else None
    if book_payload is not None:
        raw_levels = book_payload.pop("levels_json", "[]")
        try:
            parsed_levels = json.loads(str(raw_levels or "[]"))
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed_levels = []
        book_payload["levels"] = [
            {
                "bid_price": float(level.get("bid_price") or 0),
                "bid_size": max(0.0, float(level.get("bid_size") or 0)),
                "ask_price": float(level.get("ask_price") or 0),
                "ask_size": max(0.0, float(level.get("ask_size") or 0)),
            }
            for level in parsed_levels
            if isinstance(level, dict)
            and float(level.get("bid_price") or 0) > 0
            and float(level.get("ask_price") or 0) > 0
            and float(level.get("ask_price") or 0) >= float(level.get("bid_price") or 0)
        ][:10]
        book_payload["age_seconds"] = book_age
        book_payload["stale"] = orderbook_stale
        book_payload["valid"] = bool(
            float(book_payload.get("best_bid") or 0) > 0
            and float(book_payload.get("best_ask") or 0)
            >= float(book_payload.get("best_bid") or 0)
            and (
                float(book_payload.get("total_bid_volume") or 0) > 0
                or float(book_payload.get("total_ask_volume") or 0) > 0
            )
        )
    return {
        "symbol": symbol,
        "bars": payload_bars,
        "second_bars": second_bars,
        "microstructure": microstructure,
        "latest_tick": dict(tick) if tick is not None else None,
        "latest_orderbook": book_payload,
        "last_price": display_price,
        "change_rate": (
            last_close / first_close - 1
            if first_close and last_close is not None
            else None
        ),
        "stale": stale,
        "trade_stale": trade_stale,
        "orderbook_stale": orderbook_stale,
        "feed_state": microstructure["display_feed_state"],
        "last_event_at": event_at,
        "last_price_source": last_price_source,
    }


def _second_level_market_series(
    ticks: list[sqlite3.Row],
    books: list[sqlite3.Row],
    *,
    window_seconds: int = 120,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parsed_ticks: list[tuple[datetime, sqlite3.Row]] = []
    for row in ticks:
        try:
            timestamp = datetime.fromisoformat(str(row["exchange_timestamp"]))
        except (TypeError, ValueError):
            continue
        parsed_ticks.append((timestamp, row))
    parsed_books: list[tuple[datetime, sqlite3.Row]] = []
    for row in books:
        try:
            timestamp = datetime.fromisoformat(
                str(row["exchange_timestamp"]).replace("Z", "+00:00")
            )
        except (TypeError, ValueError):
            continue
        parsed_books.append((timestamp, row))
    event_times = [item[0] for item in parsed_ticks] + [item[0] for item in parsed_books]
    if not event_times:
        return [], {
            "temporal_ready": False,
            "tick_count_1s": 0,
            "tick_count_5s": 0,
            "tick_count_10s": 0,
            "trade_bar_count": 0,
            "quote_bar_count": 0,
        }
    display_latest_at = max(event_times)
    trade_latest_at = parsed_ticks[-1][0] if parsed_ticks else display_latest_at
    cutoff = display_latest_at.timestamp() - max(10, window_seconds)
    buckets: dict[datetime, dict[str, Any]] = {}
    for timestamp, row in parsed_ticks:
        if timestamp.timestamp() < cutoff:
            continue
        second = timestamp.replace(microsecond=0)
        price = float(row["price"])
        volume = max(0, int(row["volume"] or 0))
        direction = str(row["trade_direction"] or "").upper()
        bucket = buckets.setdefault(
            second,
            {
                "time": second.isoformat(),
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 0,
                "_notional": 0.0,
                "trade_count": 0,
                "buy_volume": 0,
                "sell_volume": 0,
                "bar_source": "trade",
            },
        )
        bucket["high"] = max(float(bucket["high"]), price)
        bucket["low"] = min(float(bucket["low"]), price)
        bucket["close"] = price
        bucket["volume"] += volume
        bucket["_notional"] += price * volume
        bucket["trade_count"] += 1
        if direction in {"BUY", "B"}:
            bucket["buy_volume"] += volume
        elif direction in {"SELL", "S"}:
            bucket["sell_volume"] += volume
    for timestamp, row in parsed_books:
        if timestamp.timestamp() < cutoff:
            continue
        second = timestamp.replace(microsecond=0)
        if second in buckets:
            continue
        bid = float(row["best_bid"] or 0)
        ask = float(row["best_ask"] or 0)
        if bid <= 0 or ask < bid:
            continue
        midpoint = (bid + ask) / 2
        buckets[second] = {
            "time": second.isoformat(),
            "open": midpoint,
            "high": midpoint,
            "low": midpoint,
            "close": midpoint,
            "volume": 0,
            "vwap": midpoint,
            "trade_count": 0,
            "buy_volume": 0,
            "sell_volume": 0,
            "aggressor_imbalance": 0.0,
            "bar_source": "quote_mid",
        }
    second_bars = []
    for second in sorted(buckets):
        bucket = buckets[second]
        directed = bucket["buy_volume"] + bucket["sell_volume"]
        if bucket["bar_source"] == "trade":
            bucket["vwap"] = (
                bucket["_notional"] / bucket["volume"]
                if bucket["volume"]
                else bucket["close"]
            )
            del bucket["_notional"]
            bucket["aggressor_imbalance"] = (
                (bucket["buy_volume"] - bucket["sell_volume"]) / directed
                if directed
                else 0.0
            )
        second_bars.append(bucket)

    def window_rows(seconds: int) -> list[tuple[datetime, sqlite3.Row]]:
        start = trade_latest_at.timestamp() - seconds
        return [item for item in parsed_ticks if item[0].timestamp() >= start]

    def window_return(seconds: int) -> float | None:
        rows = window_rows(seconds)
        if len(rows) < 2:
            return None
        first = float(rows[0][1]["price"])
        last = float(rows[-1][1]["price"])
        return last / first - 1 if first > 0 else None

    last_5s = window_rows(5)
    buy_volume = sum(
        max(0, int(row["volume"] or 0))
        for _, row in last_5s
        if str(row["trade_direction"] or "").upper() in {"BUY", "B"}
    )
    sell_volume = sum(
        max(0, int(row["volume"] or 0))
        for _, row in last_5s
        if str(row["trade_direction"] or "").upper() in {"SELL", "S"}
    )
    directed_volume = buy_volume + sell_volume
    recent_books = []
    five_second_cutoff = trade_latest_at.timestamp() - 5
    for timestamp, row in parsed_books:
        if five_second_cutoff <= timestamp.timestamp() <= trade_latest_at.timestamp():
            recent_books.append(row)
    spread_change = None
    imbalance_change = None
    if len(recent_books) >= 2:
        spread_change = float(recent_books[-1]["spread_bps"]) - float(recent_books[0]["spread_bps"])
        imbalance_change = float(recent_books[-1]["imbalance"]) - float(recent_books[0]["imbalance"])
    last_10s = window_rows(10)
    unique_seconds = {timestamp.replace(microsecond=0) for timestamp, _ in last_10s}
    return second_bars, {
        "as_of": display_latest_at.isoformat(),
        "temporal_ready": len(last_10s) >= 3 and len(unique_seconds) >= 2,
        "tick_count_1s": len(window_rows(1)),
        "tick_count_5s": len(last_5s),
        "tick_count_10s": len(last_10s),
        "return_1s": window_return(1),
        "return_5s": window_return(5),
        "return_10s": window_return(10),
        "volume_1s": sum(max(0, int(row["volume"] or 0)) for _, row in window_rows(1)),
        "volume_5s": sum(max(0, int(row["volume"] or 0)) for _, row in last_5s),
        "aggressor_imbalance_5s": (
            (buy_volume - sell_volume) / directed_volume if directed_volume else 0.0
        ),
        "spread_change_5s_bps": spread_change,
        "orderbook_imbalance_change_5s": imbalance_change,
        "trade_bar_count": sum(bar["bar_source"] == "trade" for bar in second_bars),
        "quote_bar_count": sum(bar["bar_source"] == "quote_mid" for bar in second_bars),
        "latest_bar_source": second_bars[-1]["bar_source"] if second_bars else None,
    }


def _algorithm(strategy_id: str) -> dict[str, Any] | None:
    for expert_type in ALL_EXPERT_TYPES:
        expert = expert_type()
        if expert.strategy_id != strategy_id:
            continue
        config = expert.config
        return {
            "strategy_id": expert.strategy_id,
            "thesis": expert.thesis,
            "entry_quantile": config.entry_quantile,
            "confirmation_quantile": config.confirmation_quantile,
            "stop_bps": config.stop_bps,
            "profit_bps": config.profit_bps,
            "trailing_bps": config.trailing_bps,
            "max_holding_seconds": config.max_holding_seconds,
            "visual_indicators": _visual_indicators(expert.strategy_id),
        }
    return None


def _visual_indicators(strategy_id: str) -> list[str]:
    mapping = {
        "intraday_momentum": ["MA5", "MA20", "VWAP", "Volume"],
        "breakout_volume": ["20-bar High", "MA20", "VWAP", "Volume"],
        "vwap_mean_reversion": ["VWAP", "MA5", "MA20", "Deviation"],
        "liquidity_shock_reversal": ["Spread", "Orderbook Imbalance", "VWAP", "Volume"],
        "event_momentum": ["Event Time", "VWAP", "MA5", "Volume"],
        "cross_sectional_relative_strength": ["Relative Strength", "MA20", "VWAP"],
        "gap_context": ["Session Open", "Gap", "VWAP", "Volume"],
        "rvgi_box_breakout": ["RVGI", "RVGI Signal", "Box High", "Box Low", "Volume"],
        "opening_range_breakout": ["Opening Range High", "Opening Range Low", "Relative Volume"],
        "market_intraday_momentum": ["First 30m Return", "First 30m Volatility", "Close Clock"],
        "market_intraday_momentum_short": ["First 30m Return", "First 30m Volatility", "Borrow"],
        "opening_range_breakdown": ["Opening Range High", "Opening Range Low", "Sell Flow", "Borrow"],
        "residual_relative_weakness": ["Sector Rank", "Residual Return", "Borrow"],
        "overnight_gap_carry": ["Close Clock", "VWAP Premium", "Day Volatility"],
        "range_support_reversion": ["20-bar Range Low", "Range Mid", "ATR", "Volume"],
        "bar_trend_continuation": ["EMA Fast/Slow", "MACD", "VWAP", "Relative Volume"],
    }
    return mapping.get(strategy_id, ["MA5", "MA20", "VWAP", "Volume"])


def _decision_ontology(
    market: dict[str, Any],
    selection: dict[str, Any],
    *,
    event_feed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact, explainable data-to-decision graph for the live terminal.

    Decision nodes come from the persisted shadow comparison. Indicator values are
    explicitly marked as dashboard reconstructions because the shadow record does
    not currently persist the exact feature tensor used by the router/GNN.
    """
    bars = list(market.get("bars") or [])
    latest = bars[-1] if bars else {}
    previous = bars[-2] if len(bars) > 1 else {}
    closes = [_finite(row.get("close")) for row in bars]
    closes = [value for value in closes if value is not None]
    volumes = [_finite(row.get("volume")) for row in bars]
    volumes = [value for value in volumes if value is not None]
    returns = [
        closes[index] / closes[index - 1] - 1
        for index in range(1, len(closes))
        if closes[index - 1]
    ]
    deviations = [
        float(row["close"]) / float(row["vwap"]) - 1
        for row in bars
        if _finite(row.get("close")) is not None
        and _finite(row.get("vwap")) not in {None, 0.0}
    ]
    latest_return = (
        float(latest["close"]) / float(previous["close"]) - 1
        if _finite(latest.get("close")) is not None
        and _finite(previous.get("close")) not in {None, 0.0}
        else None
    )
    latest_volume = _finite(latest.get("volume"))
    latest_deviation = (
        float(latest["close"]) / float(latest["vwap"]) - 1
        if _finite(latest.get("close")) is not None
        and _finite(latest.get("vwap")) not in {None, 0.0}
        else None
    )
    recent = bars[-20:]
    recent_highs = [
        value for row in recent if (value := _finite(row.get("high"))) is not None
    ]
    recent_lows = [
        value for row in recent if (value := _finite(row.get("low"))) is not None
    ]
    recent_high = max(recent_highs, default=None)
    recent_low = min(recent_lows, default=None)
    breakout_position = None
    if (
        recent_high is not None
        and recent_low is not None
        and recent_high > recent_low
        and _finite(latest.get("close")) is not None
    ):
        breakout_position = (
            float(latest["close"]) - recent_low
        ) / (recent_high - recent_low)
    book = market.get("latest_orderbook") or {}
    spread = _finite(book.get("spread_bps"))
    imbalance = _finite(book.get("imbalance"))
    liquidity = _finite(latest.get("liquidity_score"))
    volatility = _finite(latest.get("volatility"))
    micro = market.get("microstructure") or {}
    event_feed = event_feed or {}
    sector_table = selection.get("sector_rank_table")
    sector_table = sector_table if isinstance(sector_table, dict) else {}
    symbol = str(market.get("symbol") or "").upper()
    ranks = sector_table.get("ranks") if isinstance(sector_table.get("ranks"), dict) else {}
    sizes = sector_table.get("sector_sizes") if isinstance(sector_table.get("sector_sizes"), dict) else {}
    sectors = sector_table.get("sector_of") if isinstance(sector_table.get("sector_of"), dict) else {}
    sector = sectors.get(symbol)
    sector_rank = _finite(ranks.get(symbol))
    sector_size = _finite(sizes.get(sector)) if sector else None
    relative_strength_score = (
        1.0 - (sector_rank - 1.0) / max(1.0, sector_size - 1.0)
        if sector_rank is not None and sector_size is not None and sector_size > 1
        else None
    )
    session_values = _dashboard_session_context(bars, symbol)
    rvgi_result = None
    box_result = None
    try:
        from app.features.schemas import OHLCVBar
        from app.technical import indicators as technical_indicators

        typed_bars = tuple(
            OHLCVBar(
                ticker=str(market.get("symbol") or ""),
                as_of=datetime.fromisoformat(
                    str(row.get("minute_start")).replace("Z", "+00:00")
                ),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume") or 0.0),
            )
            for row in bars
            if row.get("minute_start")
        )
        rvgi_result = technical_indicators.rvgi(typed_bars, 10)
        box_result = technical_indicators.causal_box_geometry(typed_bars, 20)
    except (TypeError, ValueError, KeyError):
        rvgi_result = None
        box_result = None

    indicators = {
        "rvgi_cross": _indicator(
            "RVGI bullish cross",
            1.0 if rvgi_result and rvgi_result.bullish_cross else (
                0.0 if rvgi_result and rvgi_result.ok else None
            ),
            (
                float(rvgi_result.main)
                if rvgi_result and rvgi_result.main is not None
                else None
            ),
            (
                f"signal={rvgi_result.signal:.6f}"
                if rvgi_result and rvgi_result.signal is not None
                else "completed bar history unavailable"
            ),
            "minute_bars",
        ),
        "rvgi_diff": _indicator(
            "RVGI - signal",
            _bounded(
                0.5 + (rvgi_result.main - rvgi_result.signal) * 5.0
                if rvgi_result
                and rvgi_result.main is not None
                and rvgi_result.signal is not None
                else None
            ),
            (
                rvgi_result.main - rvgi_result.signal
                if rvgi_result
                and rvgi_result.main is not None
                and rvgi_result.signal is not None
                else None
            ),
            "pure causal RVGI",
            "minute_bars",
        ),
        "box_position": _indicator(
            "Frozen box position",
            box_result.position if box_result and box_result.ok else None,
            box_result.high if box_result and box_result.ok else None,
            (
                f"low={box_result.low}, high={box_result.high}"
                if box_result and box_result.ok
                else "causal box history unavailable"
            ),
            "minute_bars",
        ),
        "return_1s": _indicator(
            "1초 수익률", _bounded(.5 + float(micro.get("return_1s") or 0.0) * 100),
            _finite(micro.get("return_1s")), "직전 1초 체결가 변화", "tick",
        ),
        "return_5s": _indicator(
            "5초 수익률", _bounded(.5 + float(micro.get("return_5s") or 0.0) * 50),
            _finite(micro.get("return_5s")), "직전 5초 체결가 변화", "tick",
        ),
        "tick_rate_5s": _indicator(
            "5초 체결 빈도", _bounded(float(micro.get("tick_count_5s") or 0) / 25),
            _finite(micro.get("tick_count_5s")), "직전 5초 체결 건수", "tick",
        ),
        "aggressor_imbalance_5s": _indicator(
            "5초 체결강도", _bounded((float(micro.get("aggressor_imbalance_5s") or 0.0) + 1) / 2),
            _finite(micro.get("aggressor_imbalance_5s")), "매수·매도 주도 체결량 불균형", "tick",
        ),
        "spread_change_5s": _indicator(
            "5초 스프레드 변화", _bounded(.5 + float(micro.get("spread_change_5s_bps") or 0.0) / 20),
            _finite(micro.get("spread_change_5s_bps")), "직전 5초 호가 스프레드 변화", "orderbook",
        ),
        "second_data_ready": _indicator(
            "초단위 데이터 게이트", 1.0 if micro.get("ready") else 0.0,
            1.0 if micro.get("ready") else 0.0,
            str(micro.get("block_reason") or "READY"), "tick",
        ),
        "return": _indicator(
            "수익률 분위", _percentile(latest_return, returns), latest_return,
            "최근 1분 수익률의 관측 구간 내 분위", "minute_bars",
        ),
        "volume": _indicator(
            "거래량 분위", _percentile(latest_volume, volumes), latest_volume,
            "최근 1분 거래량의 관측 구간 내 분위", "minute_bars",
        ),
        "breakout": _indicator(
            "20봉 돌파 위치", breakout_position, breakout_position,
            "최근 20봉 저가~고가 범위 내 종가 위치", "minute_bars",
        ),
        "vwap_deviation": _indicator(
            "VWAP 괴리 분위", _percentile(latest_deviation, deviations), latest_deviation,
            "종가와 VWAP 괴리의 관측 구간 내 분위", "minute_bars",
        ),
        "reversion": _indicator(
            "회귀 확인", _percentile(latest_return, returns), latest_return,
            "최근 가격 반등 강도", "minute_bars",
        ),
        "liquidity_shock": _indicator(
            "유동성 충격", _bounded(spread / 20 if spread is not None else None), spread,
            "호가 스프레드 확대 강도", "orderbook",
        ),
        "price_drop": _indicator(
            "가격 하락", _percentile(-latest_return if latest_return is not None else None, [-x for x in returns]),
            latest_return, "최근 음의 수익률 강도", "minute_bars",
        ),
        "recovery": _indicator(
            "충격 회복", _percentile(latest_return, returns), latest_return,
            "충격 이후 양의 수익률 강도", "minute_bars",
        ),
        "event_relevance": _indicator(
            "이벤트 관련성",
            _bounded(abs(float(event_feed["score"]))) if event_feed.get("score") is not None else None,
            _finite(event_feed.get("score")), "뉴스 이벤트 저장소의 실제 조회 결과", "event_feed",
        ),
        "event_direction": _indicator(
            "이벤트 방향",
            _bounded((float(event_feed["score"]) + 1.0) / 2.0) if event_feed.get("score") is not None else None,
            _finite(event_feed.get("score")), "뉴스 이벤트 저장소의 최신 방향 점수", "event_feed",
        ),
        "relative_strength": _indicator(
            "횡단면 상대강도", relative_strength_score, sector_rank,
            f"sector={sector or 'unknown'}, rank={sector_rank}, size={sector_size}", "cross_section",
        ),
        "liquidity": _indicator(
            "유동성 점수", _bounded(liquidity), liquidity,
            "분봉 유동성 점수", "minute_bars",
        ),
        "gap": _indicator(
            "시가 갭", None, None, "세션 기준 이전 종가 특성이 이 응답에 없음", "session_context",
        ),
        "opening_range": _indicator(
            "시초 범위 위치", session_values.get("opening_range_position"),
            session_values.get("opening_range_high"),
            f"low={session_values.get('opening_range_low')}", "session_context",
        ),
        "first_half_hour_return": _indicator(
            "첫 30분 수익률",
            _bounded(.5 + float(session_values["first_half_hour_return_bps"]) / 200.0)
            if session_values.get("first_half_hour_return_bps") is not None else None,
            session_values.get("first_half_hour_return_bps"),
            "전일 종가 대비 첫 30분 종료 가격 수익률(bps)", "session_context",
        ),
        "first_half_hour_volatility": _indicator(
            "첫 30분 변동성 분위",
            session_values.get("first_half_hour_volatility_percentile"),
            session_values.get("first_half_hour_volatility_percentile"),
            "동일 종목의 과거 첫 30분 변동성 대비 경험적 분위", "session_context",
        ),
        "overnight_carry_window": _indicator(
            "마감 30분 진입 창",
            session_values.get("overnight_carry_window"),
            session_values.get("minutes_to_continuous_close"),
            "종목 거래소의 정규장 연속매매 종료까지 남은 시간", "session_context",
        ),
        "opening_confirmation": _indicator(
            "시가 확인", _percentile(latest_return, returns), latest_return,
            "최근 수익률을 이용한 가격발견 확인", "minute_bars",
        ),
        "spread": _indicator(
            "스프레드", _bounded(spread / 20 if spread is not None else None), spread,
            "최우선 호가 스프레드", "orderbook",
        ),
        "imbalance": _indicator(
            "호가 불균형", _bounded((imbalance + 1) / 2 if imbalance is not None else None),
            imbalance, "매수·매도 잔량 불균형", "orderbook",
        ),
        "volatility": _indicator(
            "변동성", _bounded(volatility), volatility,
            "분봉 실현 변동성", "minute_bars",
        ),
    }
    requirements: dict[str, list[tuple[str, str, float]]] = {
        "intraday_momentum": [("return", ">=", .8), ("volume", ">=", .65)],
        "breakout_volume": [("breakout", ">=", .8), ("volume", ">=", .8)],
        "vwap_mean_reversion": [("vwap_deviation", "<=", .2), ("reversion", ">=", .65)],
        "liquidity_shock_reversal": [
            ("liquidity_shock", ">=", .8), ("price_drop", ">=", .8), ("recovery", ">=", .65),
        ],
        "event_momentum": [("event_relevance", ">=", .8), ("event_direction", ">=", .65)],
        "cross_sectional_relative_strength": [
            ("relative_strength", ">=", .8), ("liquidity", ">=", .65),
        ],
        "gap_context": [("gap", ">=", .8), ("opening_confirmation", ">=", .65)],
        "rvgi_box_breakout": [
            ("rvgi_cross", ">=", .8),
            ("box_position", ">=", .8),
            ("volume", ">=", .65),
        ],
        "opening_range_breakout": [("opening_range", ">=", .8), ("volume", ">=", .65)],
        "market_intraday_momentum": [
            ("first_half_hour_return", ">=", .8),
            ("first_half_hour_volatility", ">=", .6),
        ],
        "market_intraday_momentum_short": [
            ("first_half_hour_return", "<=", .2),
            ("first_half_hour_volatility", ">=", .6),
        ],
        "opening_range_breakdown": [("opening_range", "<=", .2), ("volume", ">=", .65)],
        "residual_relative_weakness": [("relative_strength", "<=", .2), ("liquidity", ">=", .65)],
        "overnight_gap_carry": [
            ("overnight_carry_window", ">=", .8),
            ("vwap_deviation", ">=", .8),
            ("first_half_hour_volatility", ">=", .65),
        ],
        # ``<=`` and not ``>=``: box_position is the raw position in the 20-bar box
        # (0 = at the floor), and rvgi_box_breakout reads the same key the other way
        # up. Displaying it with the wrong comparator would show the operator the
        # opposite of the condition the algorithm actually applies.
        "range_support_reversion": [
            ("box_position", "<=", .2),
            ("liquidity", ">=", .65),
        ],
        "bar_trend_continuation": [
            ("return", ">=", .8),
            ("volume", ">=", .65),
            ("liquidity", ">=", .65),
        ],
    }
    ontology_strategy = str(
        selection.get("ontology_strategy_id") or selection.get("strategy_id") or ""
    )
    final_strategy = str(selection.get("strategy_id") or "")
    final_action = str(selection.get("action") or "NO_TRADE").upper()
    algorithms = []
    for expert_type in ALL_EXPERT_TYPES:
        expert = expert_type()
        checks = []
        for feature, operator, threshold in requirements.get(expert.strategy_id, ()):
            item = indicators.get(
                feature,
                _indicator(feature, None, None, "RVGI Box context unavailable", "minute_bars"),
            )
            value = item["score"]
            passed = (
                value >= threshold if value is not None and operator == ">="
                else value <= threshold if value is not None else None
            )
            checks.append(
                {
                    "indicator_id": feature,
                    "operator": operator,
                    "threshold": threshold,
                    "passed": passed,
                }
            )
        algorithms.append(
            {
                "id": expert.strategy_id,
                "label": expert.strategy_id.replace("_", " "),
                "thesis": expert.thesis,
                "ontology_selected": expert.strategy_id == ontology_strategy,
                "final_selected": expert.strategy_id == final_strategy and final_action != "NO_TRADE",
                "requirements": checks,
                "visual_indicators": _visual_indicators(expert.strategy_id),
            }
        )
    sources = [
        _source("minute_bars", "1분봉", bool(bars), len(bars), market.get("last_event_at")),
        _source(
            "tick",
            "초단위 체결 틱",
            bool(market.get("latest_tick")),
            int(micro.get("tick_count_10s") or 0),
            micro.get("as_of") or market.get("last_event_at"),
        ),
        _source("orderbook", "실시간 호가", bool(book), 1 if book else 0, book.get("exchange_timestamp")),
        _source(
            "event_feed", "이벤트·뉴스", int(event_feed.get("samples") or 0) > 0,
            int(event_feed.get("samples") or 0), event_feed.get("updated_at"),
            status="AVAILABLE" if int(event_feed.get("samples") or 0) > 0 else "MISSING",
            queried=bool(event_feed.get("queried")),
        ),
        _source(
            "cross_section", "횡단면 종목군", bool(sector_table), len(ranks),
            selection.get("as_of"),
            status="AVAILABLE" if relative_strength_score is not None else "PARTIAL" if sector_table else "MISSING",
        ),
        _source(
            "session_context", "세션 컨텍스트", True,
            int(session_values.get("price_field_count") or 0), market.get("last_event_at"),
            status="AVAILABLE" if session_values.get("price_complete") else "PARTIAL",
            clock_available=True,
            price_available=bool(session_values.get("price_complete")),
        ),
    ]
    reasons = list(
        dict.fromkeys(
            str(reason)
            for decision in selection.get("all_decisions") or []
            for reason in decision.get("reason_codes") or []
            if str(reason).strip()
        )
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "decision": selection.get("decision_source")
            or "logs/refactor-shadow-comparison.jsonl",
            "indicators": "dashboard reconstruction from realtime_market_data.sqlite3",
            "warning": "지표값은 화면용 재구성값이며 저장된 모델 입력 텐서의 직접 기여도는 아닙니다.",
        },
        "fresh": not bool(market.get("stale")),
        "sources": sources,
        "indicators": [
            {"id": indicator_id, **payload}
            for indicator_id, payload in indicators.items()
        ],
        "algorithms": algorithms,
        "ontology_selection": {
            "strategy_id": ontology_strategy or None,
            "allowed": bool(selection.get("ontology_allowed")),
            "action": selection.get("ontology_action") or "NO_TRADE",
        },
        "final_decision": {
            "strategy_id": final_strategy or None,
            "action": final_action,
            "path": selection.get("path") or "ontology",
            "utility": selection.get("utility"),
            "reason_codes": reasons,
        },
    }


def _dashboard_session_context(
    bars: list[dict[str, Any]], symbol: str = ""
) -> dict[str, Any]:
    """Reconstruct price session facts with the same causal authority as serving.

    ``symbol`` selects the exchange clock; without it the KRX window would be
    applied to US bars, which is the mismatch this replay exists to detect.
    """
    parsed: list[tuple[datetime, dict[str, Any]]] = []
    for row in bars:
        raw = row.get("time") or row.get("minute_start")
        try:
            start = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        parsed.append((start, row))
    if not parsed:
        return {"price_complete": False, "price_field_count": 0}
    parsed.sort(key=lambda item: item[0])
    session = session_structure.regular_session(symbol)
    kst = session.zone
    now = parsed[-1][0] + timedelta(minutes=1)
    minutes_to_close = session.minutes_to_continuous_close(now)
    clock_result: dict[str, Any] = {
        "overnight_carry_window": (
            1.0 if session.in_last_continuous_half_hour(now) else 0.0
        ),
        "minutes_to_continuous_close": round(minutes_to_close, 3),
    }
    latest_day = session.trading_day(now)
    current = [row for start, row in parsed if start.astimezone(kst).date() == latest_day]
    prior = [(start, row) for start, row in parsed if start.astimezone(kst).date() < latest_day]
    session_open = datetime(
        latest_day.year,
        latest_day.month,
        latest_day.day,
        session.open_time.hour,
        session.open_time.minute,
        tzinfo=kst,
    )
    observed = session_structure.opening_range(
        current, session_open=session_open, minutes=30, now=now
    )
    if observed is None:
        return {**clock_result, "price_complete": False, "price_field_count": 0}
    result: dict[str, Any] = {
        **clock_result,
        "opening_range_high": observed.high,
        "opening_range_low": observed.low,
    }
    previous_close = float(prior[-1][1].get("close") or 0.0) if prior else 0.0
    opening_return = session_structure.first_half_hour_return_bps(
        current,
        previous_close=previous_close,
        session_open=session_open,
        minutes=30,
        now=now,
    )
    if opening_return is not None:
        result["first_half_hour_return_bps"] = opening_return
    history: list[float] = []
    prior_days = sorted({start.astimezone(kst).date() for start, _ in prior})
    for day in prior_days:
        day_bars = [row for start, row in prior if start.astimezone(kst).date() == day]
        day_open = datetime(
            day.year,
            day.month,
            day.day,
            session.open_time.hour,
            session.open_time.minute,
            tzinfo=kst,
        )
        past = session_structure.opening_range(
            day_bars, session_open=day_open, minutes=30,
            now=day_open + timedelta(days=1),
        )
        if past is not None:
            history.append(past.volatility)
    volatility = session_structure.first_half_hour_volatility_percentile(
        history, observed.volatility, minimum_samples=3
    )
    if volatility is not None:
        result["first_half_hour_volatility_percentile"] = volatility
    latest_close = _finite(current[-1].get("close")) if current else None
    span = observed.high - observed.low
    if latest_close is not None and span > 0:
        result["opening_range_position"] = _bounded(
            (latest_close - observed.low) / span
        )
    result["price_field_count"] = sum(
        result.get(key) is not None
        for key in (
            "opening_range_high", "opening_range_low",
            "first_half_hour_return_bps", "first_half_hour_volatility_percentile",
        )
    )
    result["price_complete"] = result["price_field_count"] == 4
    return result


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _bounded(value: float | None) -> float | None:
    return None if value is None else max(0.0, min(1.0, float(value)))


def _percentile(value: float | None, observations: list[float]) -> float | None:
    if value is None or not observations:
        return None
    return sum(item <= value for item in observations) / len(observations)


def _indicator(
    label: str,
    score: float | None,
    raw_value: float | None,
    detail: str,
    source_id: str,
) -> dict[str, Any]:
    return {
        "label": label,
        "score": _bounded(score),
        "raw_value": raw_value,
        "detail": detail,
        "source_id": source_id,
        "available": score is not None,
        "provenance": "reconstructed",
    }


def _source(
    source_id: str,
    label: str,
    available: bool,
    samples: int,
    updated_at: Any,
    **metadata: Any,
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "available": available,
        "samples": samples,
        "updated_at": updated_at,
        **metadata,
    }


def _execution_for_symbol(
    path: Path,
    symbol: str,
    selection: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    for row in _json_lines(path, 500):
        payload = row.get("payload") or {}
        if str(payload.get("symbol") or "").upper() == symbol:
            rows.append(row)
    event_types = {str(row.get("event_type") or "") for row in rows}
    selected = bool(selection.get("ontology_allowed"))
    stage_specs = [
        ("ontology", "온톨로지 허용", selected, selection.get("action") or "NO_TRADE"),
        ("strategy", "전략 인스턴스", selected, selection.get("strategy_id") or "대기"),
        ("intent", "OrderIntent 저장", "order_intent_persisted" in event_types, "인과 저널"),
        ("risk", "RiskVerdict", "risk_verdict_persisted" in event_types, "승인·축소·거절"),
        (
            "broker",
            "KIS 주문 접수",
            bool(event_types & {"broker_order_submitted", "broker_order_acknowledged"}),
            "멱등키 제출",
        ),
        (
            "fill",
            "체결·포지션 반영",
            bool(event_types & {"fill_recorded", "order_filled"}),
            "broker 원장",
        ),
    ]
    first_pending_seen = False
    stages = []
    for stage_id, label, completed, detail in stage_specs:
        if completed:
            status = "complete"
        elif not first_pending_seen:
            status = "current"
            first_pending_seen = True
        else:
            status = "pending"
        stages.append(
            {"id": stage_id, "label": label, "status": status, "detail": detail}
        )
    return {
        "stages": stages,
        "events": [
            {
                "event_type": row.get("event_type"),
                "payload": row.get("payload"),
            }
            for row in rows[-30:]
        ],
        "event_count": len(rows),
    }

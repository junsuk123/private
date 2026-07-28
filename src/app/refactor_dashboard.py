from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config.refactor_profile import load_refactor_profile
from app.strategy.experts import ALL_EXPERT_TYPES


_SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,20}$")


def build_refactor_dashboard(root: str | Path = ".") -> dict[str, Any]:
    """Build a read-only operations view of the refactored trading path."""
    base = Path(root)
    profile, profile_error = _profile(base)
    counterfactual = _json(base / "data/reports/refactor_counterfactual_evaluation.json")
    benchmark = _json(base / "data/reports/strategy_utility_openvino.json")
    gnn_training = _json(base / "data/models/strategy_utility/rgcn_shadow.json")
    readiness = _latest_json(base / "data/reports", "live_readiness_*.json")
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
    policy_ready = bool(readiness) and not bool(readiness.get("failures"))
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
            _readiness_reason(readiness),
            "READY" if policy_ready else "BLOCKED",
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
) -> dict[str, Any]:
    """Read-only chart, ontology, strategy, and execution view for one symbol."""
    base = Path(root)
    dashboard = build_refactor_dashboard(base)
    shadow = dashboard.get("shadow") or {}
    lifecycle = dashboard.get("lifecycle") or {}
    requested = str(symbol or "").strip().upper()
    if requested and not _SYMBOL_PATTERN.fullmatch(requested):
        raise ValueError("invalid symbol")
    selected = requested or _default_symbol(shadow, lifecycle) or "005930"
    safe_limit = max(30, min(390, int(limit or 180)))
    market = _market_series(
        base / "data/store/realtime_market_data.sqlite3",
        selected,
        safe_limit,
    )
    candidates = _candidate_symbols(
        base / "data/store/realtime_market_data.sqlite3",
        shadow,
        selected,
    )
    selection = _selection_for_symbol(shadow, selected)
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
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": dashboard.get("mode"),
        "live_order_capable": dashboard.get("live_order_capable"),
        "symbol": selected,
        "candidates": candidates,
        "market": market,
        "selection": selection,
        "algorithm": algorithm,
        "execution": execution,
        "promotion_gates": dashboard.get("promotion_gates") or [],
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


def _readiness_reason(readiness: dict[str, Any]) -> str:
    failures = readiness.get("failures") or {}
    if failures:
        return " / ".join(str(value) for value in failures.values())
    return "최근 readiness 보고서가 없거나 모든 정책 검사를 통과했습니다."


def _default_symbol(shadow: dict[str, Any], lifecycle: dict[str, Any]) -> str | None:
    positions = lifecycle.get("positions") or []
    if positions:
        return str(positions[0].get("symbol") or "").upper() or None
    latest = shadow.get("latest") or {}
    return str(latest.get("symbol") or "").upper() or None


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
        "ADMISSIBLE",
        "ALLOW",
        "ALLOWED",
        "BUY",
    }
    return {
        "as_of": row.get("as_of"),
        "action": preferred.get("action") or "NO_TRADE",
        "strategy_id": preferred.get("strategy_id"),
        "utility": preferred.get("utility"),
        "reason_codes": preferred.get("reason_codes") or [],
        "path": preferred.get("path") or "ontology",
        "ontology_allowed": ontology_allowed,
        "ontology_action": ontology_action,
        "ontology_strategy_id": ontology_strategy_id,
        "ontology_reason_codes": ontology.get("reason_codes") or [],
        "all_decisions": decisions,
    }


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
    empty = {
        "bars": [],
        "latest_tick": None,
        "latest_orderbook": None,
        "last_price": None,
        "change_rate": None,
        "stale": True,
        "last_event_at": None,
    }
    if not database.exists():
        return empty
    try:
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        try:
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
            book = connection.execute(
                """
                select exchange_timestamp, best_bid, best_ask, spread_bps,
                       total_bid_volume, total_ask_volume, imbalance, latency_ms
                from realtime_orderbook where symbol = ?
                order by exchange_timestamp desc limit 1
                """,
                (symbol,),
            ).fetchone()
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
    event_at = (
        str(tick["exchange_timestamp"])
        if tick is not None
        else (str(payload_bars[-1]["time"]) if payload_bars else None)
    )
    stale = True
    if event_at:
        try:
            stale = (
                datetime.now(timezone.utc) - datetime.fromisoformat(event_at)
            ).total_seconds() > 90
        except ValueError:
            stale = True
    return {
        "bars": payload_bars,
        "latest_tick": dict(tick) if tick is not None else None,
        "latest_orderbook": dict(book) if book is not None else None,
        "last_price": float(tick["price"]) if tick is not None else last_close,
        "change_rate": (
            last_close / first_close - 1
            if first_close and last_close is not None
            else None
        ),
        "stale": stale,
        "last_event_at": event_at,
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
    }
    return mapping.get(strategy_id, ["MA5", "MA20", "VWAP", "Volume"])


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

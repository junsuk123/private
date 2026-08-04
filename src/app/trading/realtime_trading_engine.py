"""실시간(틱 기반) 단타 트레이딩 엔진.

학습 플로우와 완전히 독립된 전용 스레드에서 동작한다. KIS 실시간 시세 틱을
소비해 매수/빠른 매도(익절·손절·모델 청산)를 즉시 판단하고, 가드된
LiveExecutionCoordinator를 통해 주문을 제출한다.

실제 자금 이동 여부는 LiveExecutionCoordinator 내부의 안전 게이트
(evaluate_live_runtime_gates + 수동 무장 파일)가 최종적으로 결정한다.
무장 전에는 submit 시 LiveExecutionBlocked가 발생하며, 엔진은 이를 잡아
"blocked"로 기록하고 계속 동작한다(=실주문 없음).
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, time as day_time, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Deque
from zoneinfo import ZoneInfo

from app.execution.kis_errors import LiveExecutionBlocked
from app.execution.execution_quality import ExecutionQualityEngine, ExecutionQualityInput
from app.execution.order_pricing_policy import (
    ExecutionPricingPolicy,
    PricingContext,
    classify_action_reason,
    is_urgent_sell,
    tick_size_for,
)
from app.execution.exchange_resolver import ExchangeResolver
from app.cost import TradingCostEngine
from app.data.realtime_types import KIS_REALTIME_SOURCE
from app.storage.execution_quality_store import ExecutionQualityStore
from app.schemas.domain import AccountSnapshot, FinalOrder, Holding, OrderSide, OrderType
from app.trading.strategy_supervisor import (
    StrategySupervisor,
    SupervisorObservation,
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _is_domestic_symbol_or_market(symbol: str, market: str = "") -> bool:
    ticker = str(symbol or "").strip().upper()
    market_name = str(market or "").strip().upper()
    return (ticker.isdigit() and len(ticker) == 6) or market_name in {"KR", "KRX", "KOSPI", "KOSDAQ", "KONEX"}


def _is_krx_core_buy_session(now_utc: datetime | None = None) -> bool:
    current = now_utc or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    local = current.astimezone(ZoneInfo("Asia/Seoul"))
    return local.weekday() < 5 and day_time(9, 0) <= local.time() <= day_time(15, 20)


def _is_no_available_sell_quantity_error(exc: Exception) -> bool:
    raw = str(exc)
    message = raw.lower()
    return (
        "매매가능한 수량이 없습니다" in raw
        or "매매가능 수량" in raw
        or "no quantity" in message
        or "available quantity" in message
        or "apbk0988" in message
    )


def _is_market_closed_order_error(exc: Exception) -> bool:
    raw = str(exc)
    message = raw.lower()
    return (
        "장운영시간이 아닙니다" in raw
        or "market closed" in message
        or "not market" in message
        or "not trading" in message
        or "apbk2995" in message
    )


def _cost_context_for_liquidation_holding(holding: Holding) -> tuple[str, str, str]:
    symbol = str(getattr(holding, "ticker", "") or "").strip().upper()
    market = str(getattr(holding, "market", "") or "").strip().upper()
    if (symbol.isdigit() and len(symbol) == 6) or market in {"KR", "KRX", "KOSPI", "KOSDAQ", "KONEX"}:
        descriptor = f"{getattr(holding, 'company_name', '')} {getattr(holding, 'sector', '')}".lower()
        if any(token in descriptor for token in ("etf", "etn", "elw", "상장지수", "인버스", "레버리지")):
            return "KR", "KRX", "domestic_etf"
        return "KR", "KRX", "domestic_stock"
    venue = market or "NASD"
    return venue, venue, "overseas_stock"


def _is_loss_minimizing_liquidation(reason_codes: tuple[str, ...]) -> bool:
    text = " ".join(str(code or "").upper() for code in reason_codes or ())
    return "LOSS_MINIMIZING_LIQUIDATION" in text or "BREAKEVEN_OR_BETTER" in text


def _record_technical_decision(symbol: str, action: str, result) -> None:
    """Best-effort push of a decision's technical context to the GUI feed."""
    try:
        from app.technical.decision_feed import record_decision

        record_decision(
            symbol,
            action,
            getattr(result, "approved", False),
            getattr(result, "reason_codes", ()),
            getattr(result, "diagnostics", None),
        )
    except Exception:  # noqa: BLE001 - advisory GUI feed must never affect trading.
        pass


def _begin_technical_decision_cycle() -> None:
    """Collect dashboard technical decisions for this engine cycle."""
    try:
        from app.technical.decision_feed import begin_cycle

        begin_cycle()
    except Exception:  # noqa: BLE001 - advisory GUI feed must never affect trading.
        pass


def _commit_technical_decision_cycle() -> None:
    """Publish this cycle's technical decisions to the GUI feed."""
    try:
        from app.technical.decision_feed import commit_cycle

        commit_cycle()
    except Exception:  # noqa: BLE001 - advisory GUI feed must never affect trading.
        pass


def _rejection_reason_counts(summary: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rejection in tuple(summary.get("rejections") or ()):
        if not isinstance(rejection, dict):
            continue
        for raw in tuple(rejection.get("reason_codes") or ()):
            code = str(raw or "").split(":", 1)[0].strip()
            if code:
                counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _record_rejection_reason_counts(summary: dict[str, Any], reason_codes: tuple[str, ...]) -> None:
    counts = summary.setdefault("rejection_reason_counts", {})
    if not isinstance(counts, dict):
        counts = {}
        summary["rejection_reason_counts"] = counts
    for raw in tuple(reason_codes or ()):
        code = str(raw or "").split(":", 1)[0].strip()
        if code:
            counts[code] = int(counts.get(code, 0) or 0) + 1


def _macro_micro_blocks_buy(bundle: Any | None) -> bool:
    if bundle is None:
        return False
    macro = getattr(bundle, "macro_result", None)
    if macro is not None and hasattr(macro, "blocks_buy"):
        reason_codes = {str(code) for code in tuple(getattr(macro, "reason_codes", ()) or ())}
        if "MACRO_INSUFFICIENT_DATA" in reason_codes and not _env_bool(
            "REALTIME_MACRO_MICRO_BLOCK_ON_INSUFFICIENT_DATA", False
        ):
            return False
        return bool(getattr(macro, "blocks_buy", False))
    if isinstance(bundle, dict):
        macro_dict = bundle.get("macro_result") if isinstance(bundle.get("macro_result"), dict) else {}
        reason_codes = {str(code) for code in tuple(macro_dict.get("reason_codes") or ())}
        if "MACRO_INSUFFICIENT_DATA" in reason_codes and not _env_bool(
            "REALTIME_MACRO_MICRO_BLOCK_ON_INSUFFICIENT_DATA", False
        ):
            return False
        return bool(macro_dict.get("blocks_buy"))
    return False


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _ignored_realtime_symbols() -> set[str]:
    raw = os.getenv("REALTIME_IGNORE_SYMBOLS", "")
    return {
        symbol.strip().upper()
        for symbol in raw.replace(";", ",").split(",")
        if symbol.strip()
    }


@dataclass
class RealtimeTradingConfig:
    interval_ms: int = field(default_factory=lambda: max(100, _env_int("REALTIME_TRADING_INTERVAL_MS", 1000)))
    # Non-session holdings use the same cost-clearing floor as the dynamic exit
    # policy. Strategy-owned positions keep their model-cost-aware target.
    take_profit: float = field(default_factory=lambda: _env_float("REALTIME_TAKE_PROFIT", 0.0080))
    stop_loss: float = field(default_factory=lambda: _env_float("REALTIME_STOP_LOSS", 0.010))
    buy_weight: float = field(default_factory=lambda: _env_float("REALTIME_BUY_WEIGHT", 0.01))
    max_orders_per_cycle: int = field(default_factory=lambda: max(1, _env_int("REALTIME_MAX_ORDERS_PER_CYCLE", 8)))
    # Keep fresh-account live trading conservative: after one accepted buy, wait for
    # the next broker account snapshot before sizing another buy.
    max_buy_orders_per_cycle: int = field(default_factory=lambda: max(1, _env_int("REALTIME_MAX_BUY_ORDERS_PER_CYCLE", 1)))
    max_buy_evaluations_per_cycle: int = field(default_factory=lambda: max(1, _env_int("REALTIME_MAX_BUY_EVALUATIONS_PER_CYCLE", 8)))
    max_cycle_seconds: float = field(default_factory=lambda: max(1.0, _env_float("REALTIME_MAX_CYCLE_SECONDS", 12.0)))
    # 같은 종목을 매 사이클(~1s) 재제출해 중복 주문/에러가 쌓이는 것을 막는 쿨다운.
    submit_cooldown_sec: float = field(default_factory=lambda: _env_float("REALTIME_SUBMIT_COOLDOWN_SEC", 20.0))
    # 하드 거부(브로커 에러/게이트 차단) 종목은 더 길게 쉬어 에러 폭주를 막는다(ETP 미신청·자금부족 등).
    error_cooldown_sec: float = field(default_factory=lambda: _env_float("REALTIME_ERROR_COOLDOWN_SEC", 300.0))
    # 장운영시간 거절은 종목 문제가 아니라 세션 문제이므로 더 길게 쉬어 API 거절 폭주를 막는다.
    market_closed_error_cooldown_sec: float = field(default_factory=lambda: _env_float("REALTIME_MARKET_CLOSED_ERROR_COOLDOWN_SEC", 1800.0))
    # 매도 주문을 낸 종목은 그 주문이 처리될 때까지 재매도 금지(가능수량 초과 APBK0988 방지).
    sell_inflight_cooldown_sec: float = field(default_factory=lambda: _env_float("REALTIME_SELL_INFLIGHT_COOLDOWN_SEC", 600.0))
    sell_amend_min_price_delta: float = field(default_factory=lambda: _env_float("REALTIME_SELL_AMEND_MIN_PRICE_DELTA", 0.0005))
    # 방금 매도한 종목을 곧바로 되사는 회전(churn)을 막는다 — 왕복 수수료·스프레드만
    # 반복 지출하며 엣지 없이 자산을 깎는 것을 방지. 이 시간 내 같은 종목 신규매수 보류.
    rebuy_cooldown_sec: float = field(default_factory=lambda: _env_float("REALTIME_REBUY_COOLDOWN_SEC", 3600.0))
    # Recent losing round trips get a longer symbol-level buy cooldown. This is
    # seeded from logs/live-orders.jsonl on startup and updated from accepted
    # live orders during the current process.
    loss_rebuy_cooldown_sec: float = field(default_factory=lambda: _env_float("REALTIME_LOSS_REBUY_COOLDOWN_SEC", 86400.0))
    loss_rebuy_return_threshold: float = field(default_factory=lambda: _env_float("REALTIME_LOSS_REBUY_RETURN_THRESHOLD", -0.004))
    order_log_path: str = field(default_factory=lambda: os.getenv("REALTIME_ORDER_LOG_PATH", "logs/live-orders.jsonl"))


class RealtimeTradingEngine:
    """Independent real-time day-trading loop. Pure orchestration over injected deps."""

    def __init__(
        self,
        *,
        decision_engine: Any,
        coordinator: Any,
        account_provider: Callable[[], AccountSnapshot | None],
        candidate_symbols_provider: Callable[[], tuple[str, ...]],
        session_open_provider: Callable[[], bool],
        ontology_graph_provider: Callable[[], Any] | None = None,
        market_open_provider: Callable[[str, str], bool] | None = None,
        cycle_observer: Callable[[dict[str, Any]], None] | None = None,
        macro_micro_observer: Callable[[AccountSnapshot, tuple[str, ...], tuple[str, ...], datetime], Any] | None = None,
        strategy_session_manager: Any | None = None,
        strategy_supervisor: Any | None = None,
        config: RealtimeTradingConfig | None = None,
        recent_events_max: int = 50,
    ) -> None:
        self.decision_engine = decision_engine
        self.coordinator = coordinator
        self.account_provider = account_provider
        self.candidate_symbols_provider = candidate_symbols_provider
        self.session_open_provider = session_open_provider
        self.ontology_graph_provider = ontology_graph_provider
        # 종목별 시장 세션 게이트: 해당 종목의 거래소가 지금 열려 있는지(닫혀 있으면 주문 보류).
        self.market_open_provider = market_open_provider
        self.cycle_observer = cycle_observer
        # Advisory-only per-cycle hook: runs macro/micro ontology reasoning and
        # records the bundle for the GUI panel. It NEVER submits or gates orders.
        self.macro_micro_observer = macro_micro_observer
        # Closed-world owner: elects and locks one symbol + one algorithm until
        # the broker account confirms that the owned position is flat.
        self.strategy_session_manager = strategy_session_manager
        # Background observer. The elected algorithm carries no regime,
        # liquidity or volatility analysis; this is what watches those and can
        # halt it (SOFT: no new entries, HARD: exit now).
        if strategy_supervisor is None and strategy_session_manager is not None:
            strategy_supervisor = StrategySupervisor()
        self.strategy_supervisor = strategy_supervisor
        self.config = config or RealtimeTradingConfig()
        # Execution-quality layer (Phase 3): rejects buys whose alpha would be consumed
        # by spread/slippage and records realized slippage per symbol/strategy.
        self.execution_quality = ExecutionQualityEngine(store=ExecutionQualityStore())
        # Executable-price authority: re-prices every order from the live book, side, and
        # exit urgency before submission (last_price is a reference, not an executable price).
        self.pricing_policy = ExecutionPricingPolicy()
        # Routing-exchange resolution + validation (blocks unknown US BUYs in live strict mode).
        self.exchange_resolver = ExchangeResolver()
        # No-chase guard: last price at which a buy failed to fill, per symbol.
        self._last_failed_entry_price: dict[str, float] = {}
        self._lock = threading.Lock()
        self._last_submit_monotonic: dict[str, float] = {}
        self._error_backoff_until: dict[str, float] = {}
        self._open_sell_orders: dict[str, dict[str, Any]] = {}
        self._sell_lock_until: dict[str, float] = {}
        # 최근 매도 시각(monotonic) — 재매수 쿨다운(churn 억제)에 사용.
        self._recent_sell_monotonic: dict[str, float] = {}
        self._recent_buy_orders: dict[str, Deque[tuple[float, float]]] = {}
        self._loss_cooldown_until: dict[str, float] = {}
        self._recent: Deque[dict[str, Any]] = deque(maxlen=recent_events_max)
        self._buy_enabled = os.getenv("REALTIME_BUY_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
        self._buy_disabled_reason: str | None = None
        self._liquidation_requested = False
        self._liquidation_reason: str | None = None
        self._last_observed_cycle_monotonic = 0.0
        self._last_observed_cycle_reason: str | None = None
        self._status: dict[str, Any] = {
            "cycles": 0,
            "last_cycle_at": None,
            "submitted": 0,
            "amended": 0,
            "buy_submitted": 0,
            "buy_submit_attempted": 0,
            "sell_submitted": 0,
            "blocked": 0,
            "errors": 0,
            "last_reason": None,
            "last_summary": None,
            "live_trace": None,
        }
        self._seed_loss_cooldowns_from_order_log()

    # ---- status ---------------------------------------------------------
    def disable_buys(self, reason: str = "BUY_DISABLED") -> None:
        # Engine control is instance-local. Process-wide runtime policy belongs
        # to the orchestration layer (app.web explicitly updates the environment
        # during live termination). Mutating os.environ here disabled every
        # subsequently constructed engine and made recovery/tests order-dependent.
        self._buy_enabled = False
        self._buy_disabled_reason = reason
        self._record(
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "kind": "CONTROL",
                "outcome": "buy_disabled",
                "detail": reason,
            }
        )

    def enable_buys(self, reason: str = "BUY_REENABLED") -> None:
        """Re-enable entries after a transient reliability demotion.

        Position monitoring never stops during a demotion; this control only
        restores new entries after the orchestration layer has revalidated the
        broker, policy, model, and at least one live market.
        """
        if self._liquidation_requested:
            return
        configured = os.getenv("REALTIME_BUY_ENABLED", "true").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self._buy_enabled = configured
        self._buy_disabled_reason = None if configured else "REALTIME_BUY_ENABLED=false"
        self._record(
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "kind": "CONTROL",
                "outcome": "buy_enabled" if configured else "buy_remains_disabled",
                "detail": reason,
            }
        )

    def request_full_liquidation(self, reason: str = "LIVE_TERMINATION_FULL_LIQUIDATION") -> None:
        """Switch the engine into sell-only liquidation mode.

        BUY discovery/evaluation is skipped, existing holdings are evaluated every
        cycle, and any HOLD result is upgraded to a full-quantity SELL intent.
        """
        self.disable_buys(reason)
        self._liquidation_requested = True
        self._liquidation_reason = reason
        self._last_submit_monotonic.clear()
        self._error_backoff_until.clear()
        self._sell_lock_until.clear()
        self._record(
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "kind": "CONTROL",
                "outcome": "full_liquidation_requested",
                "detail": reason,
            }
        )

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            status = dict(self._status)
            status["recent_events"] = list(self._recent)
            status["config"] = {
                "interval_ms": self.config.interval_ms,
                "take_profit": self.config.take_profit,
                "stop_loss": self.config.stop_loss,
                "buy_weight": self.config.buy_weight,
                "max_orders_per_cycle": self.config.max_orders_per_cycle,
                "max_buy_evaluations_per_cycle": self.config.max_buy_evaluations_per_cycle,
            }
            status["buy_enabled"] = self._buy_enabled
            status["buy_disabled_reason"] = self._buy_disabled_reason
            status["liquidation_requested"] = self._liquidation_requested
            status["liquidation_reason"] = self._liquidation_reason
            status["loss_cooldown_symbols"] = sorted(
                symbol for symbol, until in self._loss_cooldown_until.items() if until > time.monotonic()
            )
            if self.strategy_session_manager is not None:
                status["strategy_session"] = self.strategy_session_manager.snapshot()
            if self.strategy_supervisor is not None:
                status["strategy_supervisor"] = self.strategy_supervisor.snapshot()
            return status

    def _record(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._recent.appendleft(event)

    def _start_live_trace(self, decision_time: datetime) -> None:
        with self._lock:
            self._status["live_trace"] = {
                "cycle_id": decision_time.isoformat(),
                "started_at": decision_time.isoformat(),
                "finished_at": None,
                "completed": False,
                "current_stage": "cycle_start",
                "stages": [],
            }

    def _trace_stage(
        self,
        stage_id: str,
        title: str,
        description: str,
        *,
        nodes: tuple[str, ...] = (),
        links: tuple[dict[str, str], ...] = (),
        ticker: str | None = None,
    ) -> None:
        observed_at = datetime.now(timezone.utc).isoformat()
        stage = {
            "stage_id": stage_id,
            "path_id": f"live:{stage_id}",
            "title": title,
            "description": description,
            "observed_at": observed_at,
            "nodes": list(dict.fromkeys(str(node) for node in nodes if node)),
            "links": [dict(link) for link in links],
            "ticker": ticker,
            "tone": "live",
            "actual": True,
        }
        with self._lock:
            trace = dict(self._status.get("live_trace") or {})
            stages = list(trace.get("stages") or ())
            stages.append(stage)
            trace["stages"] = stages[-16:]
            trace["current_stage"] = stage_id
            self._status["live_trace"] = trace

    # ---- one cycle ------------------------------------------------------
    def run_once(self, decision_time: datetime | None = None) -> dict[str, Any]:
        decision_time = decision_time or datetime.now(timezone.utc)
        self._start_live_trace(decision_time)
        _begin_technical_decision_cycle()
        summary: dict[str, Any] = {
            "at": decision_time.isoformat(),
            "submitted": 0,
            "amended": 0,
            "buy_submitted": 0,
            "sell_submitted": 0,
            "blocked": 0,
            "errors": 0,
            "sell_evaluated": 0,
            "buy_evaluated": 0,
            "sell_rejected": 0,
            "buy_rejected": 0,
            "rejections": [],
            "skipped_market_closed": 0,
            "skipped_cooldown": 0,
            "reason": None,
        }

        if not self.session_open_provider():
            self._trace_stage(
                "market_session",
                "시장 세션 확인",
                "실제 거래 엔진이 신규 진입 가능 세션이 아님을 확인했습니다.",
                nodes=("OntologyMultiStagePipeline", "NoTradeSignal"),
                links=({"source": "OntologyMultiStagePipeline", "target": "NoTradeSignal", "predicate": "blockedBySession"},),
            )
            summary["reason"] = "MARKET_SESSION_CLOSED"
            self._finish_cycle(summary)
            return summary

        self._trace_stage(
            "market_session",
            "시장 세션 확인",
            "실제 거래 엔진이 주문 평가 가능 세션을 확인했습니다.",
            nodes=("OntologyMultiStagePipeline", "OntologyFilter1:LightweightScreening"),
        )
        account = self.account_provider()
        if account is None:
            self._trace_stage(
                "account",
                "계좌 상태 확인",
                "실제 계좌 스냅샷을 읽지 못해 사이클을 중단했습니다.",
                nodes=("OntologyFilter1:LightweightScreening", "NoTradeSignal"),
            )
            summary["reason"] = "NO_ACCOUNT_SNAPSHOT"
            self._finish_cycle(summary)
            return summary

        held_tickers = {h.ticker for h in (account.holdings or ())}
        self._trace_stage(
            "account",
            "계좌 상태 확인",
            f"실제 계좌 확인 완료 · 보유 {len(held_tickers)}종목",
            nodes=("OntologyFilter1:LightweightScreening",),
        )
        # 매도·매수는 독립 예산을 갖는다 — 매도가 사이클 한도를 다 써서 매수를 굶기면 안 된다.
        sell_submitted = 0
        buy_submitted = 0
        buy_submit_attempted = 0
        liquidation_mode = self._liquidation_requested
        buy_enabled = self._buy_enabled and os.getenv("REALTIME_BUY_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
        if liquidation_mode:
            buy_enabled = False
            self._buy_disabled_reason = self._liquidation_reason or "LIVE_TERMINATION_FULL_LIQUIDATION"
        realized_pnl_today = float(getattr(account, "realized_pnl_today", 0.0) or 0.0)
        account_equity = max(1.0, float(getattr(account, "equity", 0.0) or 0.0))
        daily_loss_stop_krw = max(0.0, _env_float("REALTIME_DAILY_REALIZED_LOSS_BUY_STOP_KRW", 0.0))
        daily_loss_stop_rate = max(0.0, _env_float("REALTIME_DAILY_REALIZED_LOSS_BUY_STOP_RATE", 0.0))
        daily_loss_threshold = max(daily_loss_stop_krw, account_equity * daily_loss_stop_rate)
        if buy_enabled and daily_loss_threshold > 0.0 and realized_pnl_today <= -daily_loss_threshold:
            buy_enabled = False
            self._buy_disabled_reason = (
                f"DAILY_REALIZED_LOSS_BUY_STOP:{realized_pnl_today:.0f}<={-daily_loss_threshold:.0f}"
            )
        # Display-only telemetry for the account dashboard profitability panel.
        # These are read-model fields; they do not influence any trading decision.
        summary["realized_pnl_today_krw"] = realized_pnl_today
        summary["daily_loss_budget_krw"] = daily_loss_threshold if daily_loss_threshold > 0.0 else None
        summary["daily_loss_budget_remaining_krw"] = (
            max(0.0, daily_loss_threshold + realized_pnl_today) if daily_loss_threshold > 0.0 else None
        )
        summary["live_armed"] = buy_enabled
        summary["liquidation_requested"] = liquidation_mode
        summary["liquidation_reason"] = self._liquidation_reason

        # 최신 온톨로지 추론 그래프(분석 컨텍스트)를 1회 조회해 매도 판단에 반영한다.
        ontology_graph = None
        if self.ontology_graph_provider is not None:
            try:
                ontology_graph = self.ontology_graph_provider()
            except Exception:  # noqa: BLE001 - ontology is best-effort.
                ontology_graph = None

        ignored_symbols = _ignored_realtime_symbols()
        summary["ignored_symbols"] = sorted(ignored_symbols)
        summary["skipped_ignored"] = 0
        if liquidation_mode:
            cycle_buy_candidates = ()
        else:
            try:
                cycle_buy_candidates = tuple(
                    str(symbol or "").upper()
                    for symbol in tuple(self.candidate_symbols_provider() or ())
                    if str(symbol or "").strip()
                )
            except Exception as exc:  # noqa: BLE001 - candidate discovery failure should be visible, not fatal.
                cycle_buy_candidates = ()
                summary["errors"] += 1
                summary["reason"] = summary["reason"] or f"BUY_CANDIDATE_PROVIDER_ERROR:{exc.__class__.__name__}"
        summary["buy_candidate_count"] = len(cycle_buy_candidates)
        summary["buy_candidate_sample"] = list(cycle_buy_candidates[:10])
        self._trace_stage(
            "candidates",
            "실시간 후보 탐색",
            f"실제 스트리밍 후보 {len(cycle_buy_candidates)}개를 확인했습니다.",
            nodes=("OntologyFilter1:LightweightScreening", "CandidateStock", *cycle_buy_candidates[:6]),
            links=({"source": "OntologyFilter1:LightweightScreening", "target": "CandidateStock", "predicate": "producesCandidate"},),
        )

        # Macro/micro ontology reasoning for live candidate control and diagnostics.
        # It can only rank or block new BUY candidates; it never submits an order.
        macro_micro_bundle = None
        if self.macro_micro_observer is not None:
            try:
                macro_micro_bundle = self.macro_micro_observer(
                    account,
                    tuple(sorted(held_tickers)),
                    cycle_buy_candidates,
                    decision_time,
                )
            except Exception:  # noqa: BLE001 - advisory panel must never affect trading.
                macro_micro_bundle = None
        if self.macro_micro_observer is not None:
            macro_result = (
                getattr(macro_micro_bundle, "macro_result", None)
                or getattr(macro_micro_bundle, "macro", None)
            )
            raw_macro_regime = getattr(macro_result, "market_regime", "") or ""
            macro_regime = str(getattr(raw_macro_regime, "value", raw_macro_regime))
            self._trace_stage(
                "macro_micro",
                "매크로·마이크로 추론",
                f"실제 추론 완료 · 시장 국면 {macro_regime or '미확정'}",
                nodes=("MacroMarket", "OntologyFilter2:EntryDecision"),
                links=({"source": "MacroMarket", "target": "OntologyFilter2:EntryDecision", "predicate": "feedsDecision"},),
            )
        if buy_enabled and _macro_micro_blocks_buy(macro_micro_bundle):
            buy_enabled = False
            self._buy_disabled_reason = "MACRO_MICRO_BLOCK_BUY"

        # A single closed-world session owns all new risk. Selection occurs only
        # after macro/micro ontology reasoning, and remains locked through entry,
        # monitoring, exit submission, and broker-confirmed flattening.
        if self.strategy_session_manager is not None:
            try:
                strategy_session = self.strategy_session_manager.evaluate(
                    account,
                    cycle_buy_candidates,
                    macro_micro_bundle,
                    decision_time,
                )
                cycle_buy_candidates = self.strategy_session_manager.allowed_buy_candidates(
                    cycle_buy_candidates,
                    account,
                )
                summary["strategy_session"] = strategy_session
                summary["owned_buy_candidates"] = list(cycle_buy_candidates)
                selected_symbol = str(strategy_session.get("selected_symbol") or "")
                selected_strategy = str(strategy_session.get("selected_strategy") or "")
                self._trace_stage(
                    "strategy_election",
                    "전략 선출",
                    (
                        f"실제 선출 결과 · {selected_symbol} / {selected_strategy}"
                        if selected_symbol and selected_strategy
                        else f"실제 선출 결과 · 미선택 ({strategy_session.get('last_reason') or '조건 미충족'})"
                    ),
                    nodes=("OntologyFilter2:EntryDecision", selected_symbol, "NoTradeSignal" if not selected_symbol else "FinalTradeGate"),
                    ticker=selected_symbol or None,
                )
            except Exception as exc:  # noqa: BLE001 - ownership failure must fail closed.
                buy_enabled = False
                self._buy_disabled_reason = f"STRATEGY_SESSION_ERROR:{exc.__class__.__name__}"
                summary["errors"] += 1
                summary["strategy_session"] = {
                    "phase": "ERROR",
                    "last_reason": self._buy_disabled_reason,
                }

        # The elected algorithm contains no market-condition analysis of its
        # own. The supervisor is what keeps watching: SOFT stops new entries,
        # HARD also hands the open position to the exit path.
        if self.strategy_supervisor is not None and self.strategy_session_manager is not None:
            try:
                verdict = self._supervise_session(
                    account,
                    macro_micro_bundle,
                    decision_time,
                )
            except Exception as exc:  # noqa: BLE001 - supervision must fail closed.
                buy_enabled = False
                self._buy_disabled_reason = f"SUPERVISOR_ERROR:{exc.__class__.__name__}"
                summary["errors"] += 1
                verdict = None
            if verdict is not None:
                summary["strategy_supervisor"] = verdict.as_dict()
                if verdict.blocks_new_entries:
                    cycle_buy_candidates = ()
                    buy_enabled = False
                    self._buy_disabled_reason = (
                        f"SUPERVISOR_{verdict.level.value}_HALT:"
                        f"{','.join(verdict.reason_codes) or 'UNSPECIFIED'}"
                    )
                    summary["owned_buy_candidates"] = []

        # 1) 매도: 보유 포지션의 빠른 청산.
        for holding in tuple(account.holdings or ()):
            holding_symbol = str(getattr(holding, "ticker", "") or "").upper()
            if holding_symbol in ignored_symbols:
                summary["skipped_ignored"] += 1
                continue
            if not liquidation_mode and sell_submitted >= self.config.max_orders_per_cycle:
                break
            if self.market_open_provider is not None and not self.market_open_provider(holding.ticker, holding.market or ""):
                summary["skipped_market_closed"] += 1
                continue  # 거래소 마감: 지금 주문하면 브로커가 거부하므로 보류.
            has_open_sell = holding.ticker in self._open_sell_orders
            sell_lock_until = self._sell_lock_until.get(holding.ticker)
            if sell_lock_until is not None and time.monotonic() < sell_lock_until and not has_open_sell:
                summary["skipped_cooldown"] += 1
                continue  # 결제/가능수량 잠금이 풀릴 때까지 반복 재시도하지 않는다.
            if self._in_cooldown(holding.ticker) and not has_open_sell:
                summary["skipped_cooldown"] += 1
                continue  # 최근 제출한 종목은 쿨다운 동안 재제출하지 않는다(중복/에러 방지).
            summary["sell_evaluated"] += 1
            session_exit_reason = (
                self.strategy_session_manager.exit_reason_for(holding)
                if self.strategy_session_manager is not None
                else None
            )
            session_owned = bool(
                self.strategy_session_manager is not None
                and self.strategy_session_manager.owns_position(holding.ticker)
            )
            if session_owned and not liquidation_mode:
                if not session_exit_reason:
                    # Once a strategy owns the position, only that strategy's
                    # target/stop/trailing/time rules may initiate an exit.
                    continue
                result = self._forced_strategy_session_exit_result(
                    holding,
                    SimpleNamespace(
                        approved=False,
                        final_order=None,
                        reason_codes=(),
                        diagnostics={
                            "selected_strategy": (
                                self.strategy_session_manager.snapshot().get(
                                    "selected_strategy"
                                )
                            ),
                        },
                    ),
                    session_exit_reason,
                )
            else:
                try:
                    result = self.decision_engine.evaluate_exit_for_holding(
                        holding,
                        account,
                        take_profit=self.config.take_profit,
                        stop_loss=self.config.stop_loss,
                        ontology_graph=ontology_graph,
                        decision_time=decision_time,
                    )
                except Exception as exc:  # noqa: BLE001 - one symbol must not kill the loop.
                    summary["errors"] += 1
                    self._record({"at": decision_time.isoformat(), "symbol": holding.ticker, "kind": "SELL", "outcome": "eval_error", "detail": f"{exc.__class__.__name__}: {exc}"})
                    continue
            _record_technical_decision(holding.ticker, "SELL", result)
            if liquidation_mode:
                result = self._forced_liquidation_result(holding, result)
            if result.approved and result.final_order is not None:
                final_order = self._fit_sell_order_to_available_quantity(result.final_order, holding)
                if final_order is None:
                    self._sell_lock_until[holding.ticker] = time.monotonic() + max(300.0, self.config.sell_inflight_cooldown_sec)
                    summary["sell_rejected"] += 1
                    self._append_rejection(
                        summary,
                        holding.ticker,
                        "SELL",
                        ("NO_SELLABLE_QUANTITY", "OPEN_ORDER_OR_SETTLEMENT_LOCK"),
                    )
                    continue
                # Resolve exchange, re-price from the live book (best_bid / marketable
                # stop), and run the execution-quality gate for the SELL too.
                priced_order, exec_ok, exec_reason, exec_diag = self._prepare_order_for_execution(
                    holding.ticker, "SELL", final_order, getattr(result, "diagnostics", None), result.reason_codes, account, decision_time
                )
                if not exec_ok or priced_order is None:
                    summary["sell_rejected"] += 1
                    self._append_rejection(summary, holding.ticker, "SELL", (exec_reason,))
                    continue
                # Record the sell so we don't immediately re-buy the same name (churn).
                self._recent_sell_monotonic[holding.ticker] = time.monotonic()
                if has_open_sell:
                    if self._amend_open_sell(priced_order, result.reason_codes, decision_time, summary, pricing_diag=exec_diag):
                        sell_submitted += 1
                elif self._submit(priced_order, "SELL", result.reason_codes, decision_time, summary, pricing_diag=exec_diag):
                    sell_submitted += 1
                    if self.strategy_session_manager is not None:
                        self.strategy_session_manager.mark_exit_submitted(
                            holding.ticker,
                            decision_time,
                        )
            else:
                summary["sell_rejected"] += 1
                self._append_rejection(summary, holding.ticker, "SELL", result.reason_codes, getattr(result, "diagnostics", None))

        # 2) 매수: 미보유 후보 진입(매도와 독립 예산).
        if not buy_enabled:
            self._trace_stage(
                "execution_gate",
                "최종 주문 게이트",
                f"실제 주문 차단 · {self._buy_disabled_reason or 'BUY_DISABLED'}",
                nodes=("OntologyFilter3:FinalRiskApproval", "NoTradeSignal"),
            )
            summary["reason"] = summary["reason"] or (self._buy_disabled_reason or "BUY_DISABLED")
            summary["buy_disabled"] = True
            summary["buy_disabled_reason"] = self._buy_disabled_reason or "REALTIME_BUY_ENABLED=false"
            summary["buy_submitted"] = buy_submitted
            summary["buy_submit_attempted"] = buy_submit_attempted
            summary["sell_submitted"] = sell_submitted
            self._finish_cycle(summary)
            return summary

        skipped_market_closed_symbols: list[str] = []
        buy_loop_started = time.monotonic()
        for symbol in cycle_buy_candidates:
            symbol = str(symbol or "").upper()
            if time.monotonic() - buy_loop_started >= self.config.max_cycle_seconds:
                summary["reason"] = summary["reason"] or "CYCLE_TIME_BUDGET_REACHED"
                break
            if symbol in ignored_symbols:
                summary["skipped_ignored"] += 1
                continue
            if summary["buy_evaluated"] >= self.config.max_buy_evaluations_per_cycle:
                summary["reason"] = summary["reason"] or "BUY_EVALUATION_LIMIT_REACHED"
                break
            if buy_submit_attempted >= min(self.config.max_orders_per_cycle, self.config.max_buy_orders_per_cycle):
                break
            if symbol in held_tickers:
                continue  # 보유 종목은 매도 감시 대상이므로 신규 매수에서 제외.
            loss_until = self._loss_cooldown_until.get(symbol)
            if loss_until is not None and time.monotonic() < loss_until:
                summary["skipped_cooldown"] += 1
                self._append_rejection(summary, symbol, "BUY", ("RECENT_LOSS_SYMBOL_COOLDOWN",))
                continue
            if self.config.rebuy_cooldown_sec > 0:
                last_sell = self._recent_sell_monotonic.get(symbol)
                if last_sell is not None and (time.monotonic() - last_sell) < self.config.rebuy_cooldown_sec:
                    summary["skipped_cooldown"] += 1
                    continue  # 방금 판 종목 재매수 보류(churn 억제).
            if (
                _is_domestic_symbol_or_market(symbol)
                and os.getenv("REALTIME_DOMESTIC_BUY_CORE_SESSION_ONLY", "false").strip().lower() in {"1", "true", "yes", "on"}
                and not _is_krx_core_buy_session(decision_time)
            ):
                # 기본은 정규장 이외(장전/장후/시간외 단일가)에도 국내 매수를 허용한다.
                # 실제 거래소 개장 여부는 아래 market_open_provider(확장시간 게이트)가 판단한다.
                summary["skipped_market_closed"] += 1
                continue
            if self.market_open_provider is not None and not self.market_open_provider(symbol, ""):
                summary["skipped_market_closed"] += 1
                if len(skipped_market_closed_symbols) < 10:
                    skipped_market_closed_symbols.append(symbol)
                continue  # 거래소 마감: 신규 매수 보류.
            if self._in_cooldown(symbol):
                summary["skipped_cooldown"] += 1
                continue
            summary["buy_evaluated"] += 1
            selected_strategy = (
                self.strategy_session_manager.selected_strategy_for(symbol)
                if self.strategy_session_manager is not None
                else None
            )
            try:
                buy_kwargs = {
                    "suggested_weight": self.config.buy_weight,
                    "ontology_graph": None if selected_strategy else ontology_graph,
                    "decision_time": decision_time,
                }
                if selected_strategy:
                    buy_kwargs["selected_strategy"] = selected_strategy
                    # Hand the algorithm the slow context frozen at election
                    # time instead of letting it re-derive market conditions.
                    election_context = self.strategy_session_manager.election_context_for(symbol)
                    if election_context:
                        buy_kwargs["election_context"] = election_context
                result = self.decision_engine.evaluate_buy(
                    symbol,
                    account,
                    **buy_kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                summary["errors"] += 1
                self._record({"at": decision_time.isoformat(), "symbol": symbol, "kind": "BUY", "outcome": "eval_error", "detail": f"{exc.__class__.__name__}: {exc}"})
                continue
            _record_technical_decision(symbol, "BUY", result)
            if result.approved and result.final_order is not None:
                priced_order, exec_ok, exec_reason, exec_diag = self._prepare_order_for_execution(
                    symbol, "BUY", result.final_order, getattr(result, "diagnostics", None), result.reason_codes, account, decision_time
                )
                if not exec_ok or priced_order is None:
                    summary["buy_rejected"] += 1
                    self._append_rejection(summary, symbol, "BUY", (exec_reason,))
                    continue
                buy_submit_attempted += 1
                summary["buy_submit_attempted"] = buy_submit_attempted
                if self._submit(priced_order, "BUY", result.reason_codes, decision_time, summary, pricing_diag=exec_diag):
                    buy_submitted += 1
                    if self.strategy_session_manager is not None:
                        self.strategy_session_manager.mark_entry_submitted(symbol, decision_time)
            else:
                summary["buy_rejected"] += 1
                self._append_rejection(summary, symbol, "BUY", result.reason_codes, getattr(result, "diagnostics", None))

        summary["buy_submitted"] = buy_submitted
        summary["buy_submit_attempted"] = buy_submit_attempted
        summary["sell_submitted"] = sell_submitted
        self._trace_stage(
            "execution_gate",
            "최종 주문 게이트",
            (
                f"실제 주문 처리 · 평가 {summary['buy_evaluated']}건, 제출 시도 {buy_submit_attempted}건, 제출 {buy_submitted}건"
            ),
            nodes=(
                "OntologyFilter3:FinalRiskApproval",
                "FinalTradeGate",
                "NoTradeSignal" if buy_submitted <= 0 else "CandidateStock",
            ),
        )
        if skipped_market_closed_symbols:
            summary["skipped_market_closed_symbols"] = skipped_market_closed_symbols
        # Short-side bookkeeping. Runs AFTER every order decision so it can never
        # influence one, and swallows its own errors so a shadow-accounting fault
        # cannot stop live LONG trading.
        self._run_short_cycle(summary, cycle_buy_candidates, decision_time)
        self._finish_cycle(summary)
        return summary

    # ------------------------------------------------------------------ #
    # Short-side cycle (no orders, no gating)                             #
    # ------------------------------------------------------------------ #
    def _run_short_cycle(
        self,
        summary: dict[str, Any],
        candidates: tuple[str, ...],
        decision_time: datetime,
    ) -> None:
        """Keeps the short deployment ladder fed and moving.

        Four jobs, none of which can create an order:

        1. poll 대주 availability for short-eligible candidates, so a locate EXISTS
           when a short signal fires (without this the store stays empty and every
           short candidate is dropped before it becomes a proposal);
        2. hand newly journaled shadow plans to the evaluation service;
        3. walk those plans against the current book and persist resolved outcomes;
        4. re-evaluate deployment states on the configured interval.

        Entirely wrapped in try/except: this is bookkeeping for a subsystem that
        submits nothing, and it must not be able to break the live LONG path.
        """
        if not _env_bool("SHORT_STRATEGY_CYCLE_ENABLED", True):
            return
        try:
            from app.trading.borrow_polling import default_borrow_poller
            from app.trading.shadow_evaluation_service import (
                default_shadow_evaluation_service,
            )

            poller = default_borrow_poller()
            service = default_shadow_evaluation_service()
        except Exception as exc:  # noqa: BLE001
            summary["short_cycle"] = {"error": f"{type(exc).__name__}: {exc}"}
            return

        short_summary: dict[str, Any] = {}
        try:
            # Demand-driven: only the symbols a short could actually elect this cycle.
            poller.track(candidates)
            short_summary["borrow_poll"] = poller.poll_once(now=decision_time).as_dict()
        except Exception as exc:  # noqa: BLE001
            short_summary["borrow_poll_error"] = f"{type(exc).__name__}: {exc}"

        try:
            plans = (
                self.strategy_session_manager.drain_shadow_plans()
                if self.strategy_session_manager is not None
                else ()
            )
            # Candidate rotation must not orphan an already-open shadow plan.
            # Keep walking every adopted symbol until it resolves or expires,
            # even after it leaves the current election universe.
            observed_symbols = tuple(
                dict.fromkeys((*candidates, *service.open_symbols))
            )
            quotes = self._short_cycle_quotes(observed_symbols, decision_time)
            short_summary["evaluation"] = service.evaluate_tick(
                quotes, now=decision_time, new_plans=plans
            ).as_dict()
            # The LONG/SHORT/NO_TRADE comparison from this cycle's election feeds
            # ``short_rescue_rate`` — the gate that asks whether adding shorts bought
            # anything at all, rather than only adding exposure.
            session = summary.get("strategy_session")
            if isinstance(session, dict):
                service.record_directional_comparison(
                    session.get("directional_comparison")
                )
            short_summary["short_rescue_rate"] = service.short_rescue_rate
        except Exception as exc:  # noqa: BLE001
            short_summary["evaluation_error"] = f"{type(exc).__name__}: {exc}"

        try:
            decisions = self._maybe_evaluate_promotions(service, decision_time)
            if decisions is not None:
                short_summary["promotion"] = decisions
        except Exception as exc:  # noqa: BLE001
            short_summary["promotion_error"] = f"{type(exc).__name__}: {exc}"
        summary["short_cycle"] = short_summary

    def _short_cycle_quotes(
        self, candidates: tuple[str, ...], decision_time: datetime
    ) -> dict[str, dict[str, Any]]:
        """Current bid/ask per candidate, for walking shadow barriers.

        Reuses the book the engine already holds — shadow evaluation must not add a
        market-data fetch. A symbol whose book is missing or crossed is simply omitted:
        the simulator would refuse the quote anyway, and stamping a synthetic one is
        how a fabricated price path gets in.

        ``observed_at`` is the BOOK's own timestamp where available, not
        ``decision_time``. That distinction is the temporal leak defence — using the
        cycle clock for a book received earlier would let a plan signalled this cycle
        treat an older book as post-signal data.
        """
        quotes: dict[str, dict[str, Any]] = {}
        for symbol in candidates:
            # Reuses the engine's own book accessor, so shadow evaluation sees exactly
            # the book the order path would have seen — a second lookup could disagree
            # and make shadow results unreproducible live.
            orderbook = self._latest_orderbook(symbol)
            if orderbook is None:
                continue
            bid = float(getattr(orderbook, "best_bid", 0.0) or 0.0)
            ask = float(getattr(orderbook, "best_ask", 0.0) or 0.0)
            if bid <= 0 or ask < bid:
                continue
            received_at = getattr(orderbook, "received_at", None)
            observed_at = (
                received_at
                if isinstance(received_at, datetime)
                else decision_time
            )
            if observed_at.tzinfo is None:
                observed_at = observed_at.replace(tzinfo=timezone.utc)
            quotes[str(symbol).upper()] = {
                "bid_price": bid,
                "ask_price": ask,
                "observed_at": observed_at,
            }
        return quotes

    def _maybe_evaluate_promotions(
        self, service: Any, decision_time: datetime
    ) -> list[dict[str, Any]] | None:
        """Re-evaluate deployment states, at most once per configured interval.

        Rate-limited because it is a multi-table read per arm and the trading loop runs
        every second, while the ladder's own config asks for 300s. Returns ``None`` when
        the interval has not elapsed.
        """
        from app.trading.short_strategy_promotion import default_promotion_controller

        controller = default_promotion_controller()
        interval = max(30, int(controller.config.evaluation_interval_seconds))
        last = getattr(self, "_last_promotion_eval_at", None)
        if last is not None and (decision_time - last).total_seconds() < interval:
            return None
        self._last_promotion_eval_at = decision_time
        from app.trading.short_strategy_promotion import RuntimeHealth

        health = RuntimeHealth(
            # Model calibration is a SEPARATE precondition and is not asserted here;
            # it stays False until the directional GNN head exists, which keeps every
            # arm blocked at the MODEL_NOT_CALIBRATED gate. That is the honest state.
            model_calibrated=False,
            short_rescue_rate=service.short_rescue_rate,
            change_point_probability=float(
                getattr(self, "_last_change_point_probability", 0.0) or 0.0
            ),
        )
        decisions = controller.evaluate_all(health=health, now=decision_time)
        return [
            item.as_dict()
            for item in decisions
            if item.changed or item.failed_gates
        ][:8]

    def _in_cooldown(self, symbol: str) -> bool:
        now = time.monotonic()
        # 하드 거부(차단/에러)된 종목은 더 긴 백오프 동안 재시도하지 않는다(에러 폭주 방지).
        backoff = self._error_backoff_until.get(symbol)
        if backoff is not None and now < backoff:
            return True
        cooldown = self.config.submit_cooldown_sec
        if cooldown <= 0:
            return False
        last = self._last_submit_monotonic.get(symbol)
        return last is not None and (now - last) < cooldown

    def _append_rejection(
        self,
        summary: dict[str, Any],
        symbol: str,
        side: str,
        reason_codes: tuple[str, ...],
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        _record_rejection_reason_counts(summary, tuple(reason_codes or ()))
        rejections = summary.setdefault("rejections", [])
        if len(rejections) >= 12:
            return
        entry: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "reason_codes": tuple(reason_codes or ()),
        }
        # Surface the already-computed profitability decision for the dashboard
        # decision cards. Read-only projection; no decision logic here.
        prof = (diagnostics or {}).get("profitability_decision") if isinstance(diagnostics, dict) else None
        if isinstance(prof, dict):
            entry["profitability"] = {
                key: prof.get(key)
                for key in (
                    "entry_price",
                    "expected_exit_price",
                    "break_even_exit_price",
                    "all_in_cost_rate",
                    "gross_expected_return",
                    "net_expected_return",
                    "required_min_net_return",
                    "spread_rate",
                    "expected_slippage_rate",
                    "liquidity_score",
                    "cost_to_alpha_ratio",
                )
            }
            warnings = prof.get("warnings") or prof.get("data_quality_flags")
            if warnings:
                entry["warnings"] = list(warnings)
        rejections.append(entry)

    def _fit_sell_order_to_available_quantity(self, order: FinalOrder, holding: Any) -> FinalOrder | None:
        sellable = getattr(holding, "sellable_quantity", None)
        if sellable is None:
            return order
        try:
            available = int(sellable)
        except (TypeError, ValueError):
            return order
        if available <= 0:
            return None
        if available >= int(order.quantity):
            return order
        return replace(order, quantity=available)

    def _live_mode(self) -> bool:
        """True when the coordinator is bound to a live (non-paper) broker.

        When we cannot tell, assume LIVE so exchange resolution stays strict (blocking
        an unknown US BUY is the safe default; a false-positive block never loses money).
        """
        broker = getattr(self.coordinator, "broker", None)
        return not bool(getattr(broker, "paper", False))

    def _latest_orderbook(self, symbol: str) -> Any | None:
        store = getattr(self.decision_engine, "store", None)
        if store is None or not hasattr(store, "latest_orderbook"):
            return None
        try:
            return store.latest_orderbook(symbol)
        except Exception:  # noqa: BLE001 - book fetch is best-effort.
            return None

    def _refresh_us_orderbook_for_execution(self, symbol: str, decision_time: datetime) -> Any | None:
        if _is_domestic_symbol_or_market(symbol):
            return None
        try:
            from app.trading.us_realtime_bridge import refresh_us_realtime_for_context_buy_candidates

            refresh_us_realtime_for_context_buy_candidates(object(), symbols=(symbol,), max_symbols=1)
        except Exception:  # noqa: BLE001 - execution prep still fails closed if refresh fails.
            return None
        orderbook = self._latest_orderbook(symbol)
        if orderbook is None:
            return None
        received_at = getattr(orderbook, "received_at", None)
        if isinstance(received_at, datetime):
            ref_dt = received_at if received_at.tzinfo else received_at.replace(tzinfo=timezone.utc)
            max_age = max(1.0, _env_float("REALTIME_EXEC_US_ORDERBOOK_MAX_AGE_SEC", 180.0))
            if max(0.0, (decision_time - ref_dt).total_seconds()) > max_age:
                return None
        return orderbook

    def _prepare_order_for_execution(
        self,
        symbol: str,
        side: str,
        final_order: FinalOrder,
        diagnostics: dict[str, Any] | None,
        reason_codes: tuple[str, ...],
        account: AccountSnapshot | None,
        decision_time: datetime,
    ) -> tuple[FinalOrder | None, bool, str, dict[str, Any]]:
        """Resolve routing exchange, re-price from the live book, and run the
        execution-quality gate — for BOTH buys and sells. Returns
        ``(priced_order, ok, reason_code, diagnostics)``. Best-effort; never raises.

        * BUY: uses best_ask (chase-capped); a missing/stale book blocks the order so it
          is never priced as if the spread were zero. Keeps the no-chase-after-failure guard.
        * SELL: uses best_bid (take-profit / model / reduce) or best_bid minus a tick
          offset (stop / hard-stop / emergency). An urgent stop is still allowed to exit
          with a discounted reference price when no book exists; a non-urgent no-book SELL
          is blocked. Exit-edge is never vetoed by BUY-oriented spread/slippage math.
        * Exchange: unknown US BUY is blocked in live strict mode (never silently NASD).
        """
        side_u = str(side).upper()
        diag: dict[str, Any] = {}
        try:
            reference_price = float(getattr(final_order, "limit_price", 0.0) or 0.0)
            if reference_price <= 0:
                return None, False, "EXEC_INVALID_REFERENCE_PRICE", diag
            diagnostics = diagnostics or {}
            pd = diagnostics.get("profitability_decision") or {}
            exit_reason = str(diagnostics.get("exit_reason") or "")
            action_reason = classify_action_reason(side_u, exit_reason, tuple(reason_codes or ()))
            urgent_exit = side_u == "SELL" and is_urgent_sell(action_reason)
            is_domestic = _is_domestic_symbol_or_market(symbol, getattr(final_order, "market", ""))
            # The execution-risk gate needs an order-book data source to assess
            # spread/pricing/exchange. Missing source is FAIL-CLOSED for BUY (never priced
            # as if the spread were zero) and for non-urgent SELL (skip). Only an urgent
            # stop/hard-stop/emergency SELL is allowed to exit at the reference price so a
            # wiring gap can never trap a losing position.
            store = getattr(self.decision_engine, "store", None)
            if store is None or not hasattr(store, "latest_orderbook"):
                diag["exec_prepare_no_book_source"] = True
                if urgent_exit:
                    return final_order, True, "EXEC_NO_BOOK_SOURCE_EMERGENCY_SELL", diag
                return None, False, "EXEC_NO_BOOK_SOURCE", diag

            # No-chase guard (BUY only): do not re-enter above the last failed entry price.
            if side_u == "BUY":
                last_failed = self._last_failed_entry_price.get(symbol)
                if last_failed is not None and reference_price > last_failed * 1.001:
                    return None, False, "EXEC_NO_CHASE_AFTER_FAILED_ENTRY", diag

            # 1) Routing exchange resolution + validation (before pricing/submit).
            live = self._live_mode()
            resolution = self.exchange_resolver.resolve(symbol, side_u, account=account, live=live)
            diag["exchange_resolution_source"] = resolution.source
            diag["resolved_exchange"] = resolution.exchange
            diag["exchange_confidence"] = resolution.confidence
            if not resolution.allowed:
                return None, False, resolution.reason_code or "US_EXCHANGE_UNKNOWN", diag
            resolved_market = getattr(final_order, "market", "") or ""
            if resolution.exchange and resolution.exchange != "KR":
                resolved_market = resolution.exchange

            # 2) Fetch the order book ONCE; share it with pricing and quality.
            orderbook = self._latest_orderbook(symbol)
            if side_u == "BUY" and not is_domestic:
                stale_or_missing_book = orderbook is None
                if orderbook is not None:
                    received_at = getattr(orderbook, "received_at", None)
                    if isinstance(received_at, datetime):
                        ref_dt = received_at if received_at.tzinfo else received_at.replace(tzinfo=timezone.utc)
                        stale_or_missing_book = max(0.0, (decision_time - ref_dt).total_seconds()) > max(
                            1.0,
                            _env_float("REALTIME_EXEC_US_ORDERBOOK_MAX_AGE_SEC", 180.0),
                        )
                if stale_or_missing_book:
                    refreshed_book = self._refresh_us_orderbook_for_execution(symbol, decision_time)
                    if refreshed_book is not None:
                        orderbook = refreshed_book
            best_bid = float(getattr(orderbook, "best_bid", 0.0) or 0.0) if orderbook is not None else 0.0
            best_ask = float(getattr(orderbook, "best_ask", 0.0) or 0.0) if orderbook is not None else 0.0
            bid_depth = float(getattr(orderbook, "total_bid_volume", 0.0) or 0.0) if orderbook is not None else 0.0
            ask_depth = float(getattr(orderbook, "total_ask_volume", 0.0) or 0.0) if orderbook is not None else 0.0
            orderbook_age_sec: float | None = None
            if orderbook is not None:
                received_at = getattr(orderbook, "received_at", None)
                if isinstance(received_at, datetime):
                    ref_dt = received_at if received_at.tzinfo else received_at.replace(tzinfo=timezone.utc)
                    orderbook_age_sec = max(0.0, (decision_time - ref_dt).total_seconds())
                # A non-live (REST-snapshot) book is not a fresh tradeable book for a BUY.
                if str(getattr(orderbook, "source", "")) != KIS_REALTIME_SOURCE:
                    orderbook_age_sec = (orderbook_age_sec or 0.0) + 1e6

            # 3) Executable limit price (side- and urgency-aware).
            pricing = self.pricing_policy.price(
                PricingContext(
                    symbol=symbol,
                    side=side_u,
                    action_reason=action_reason,
                    reference_price=reference_price,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    is_domestic=is_domestic,
                    min_net_exit_return=float(pd.get("required_min_net_return", 0.0) or 0.0),
                    expected_net_return=float(pd.get("net_expected_return", 0.0) or 0.0),
                    orderbook_age_sec=orderbook_age_sec,
                )
            )
            diag.update(
                {
                    "pricing_policy": pricing.pricing_policy,
                    "original_reference_price": reference_price,
                    "final_limit_price": pricing.limit_price,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "spread_rate": ((best_ask - best_bid) / ((best_ask + best_bid) / 2.0))
                    if best_bid > 0 and best_ask >= best_bid
                    else None,
                    "orderbook_age_sec": orderbook_age_sec,
                    "action_reason": action_reason,
                    "pricing_warnings": list(pricing.warnings),
                }
            )
            if side_u == "SELL" and _is_loss_minimizing_liquidation(tuple(reason_codes or ())):
                protected_limit = max(float(pricing.limit_price or 0.0), reference_price)
                if protected_limit > 0:
                    tick = tick_size_for(protected_limit, is_domestic)
                    protected_limit = max(tick, math.ceil(protected_limit / max(tick, 1e-9)) * tick)
                    pricing = replace(
                        pricing,
                        limit_price=round(protected_limit, 4),
                        pricing_policy=f"{pricing.pricing_policy}_BREAKEVEN_OR_BETTER",
                        warnings=tuple(pricing.warnings) + ("LOSS_MINIMIZING_LIMIT_PRESERVED",),
                    )
                    diag["final_limit_price"] = pricing.limit_price
                    diag["loss_minimizing_limit_preserved"] = True

            # 4) Execution-quality gate (side-aware; no-orderbook blocking).
            assessment = self.execution_quality.assess(
                ExecutionQualityInput(
                    symbol=symbol,
                    strategy_family="live_short_horizon",
                    decision_reference_price=reference_price,
                    gross_expected_return=float(pd.get("gross_expected_return", 0.0) or 0.0),
                    net_expected_return=float(pd.get("net_expected_return", 0.0) or 0.0),
                    required_min_net_return=float(pd.get("required_min_net_return", 0.0) or 0.0),
                    best_bid=best_bid,
                    best_ask=best_ask,
                    bid_depth=bid_depth,
                    ask_depth=ask_depth,
                    order_quantity=int(getattr(final_order, "quantity", 1) or 1),
                    side=side_u,
                    action_reason=action_reason,
                    orderbook_age_sec=orderbook_age_sec,
                )
            )
            diag["exec_quality_warnings"] = list(assessment.warnings)
            if not assessment.allowed:
                if side_u == "BUY":
                    self._last_failed_entry_price[symbol] = reference_price
                return None, False, assessment.reject_reason or "EXEC_QUALITY_REJECTED", diag

            # 5) Final priced order. If pricing declined (BUY without a usable book) the
            #    quality gate should already have blocked; guard defensively.
            if not pricing.priced or pricing.limit_price <= 0:
                if side_u == "BUY":
                    self._last_failed_entry_price[symbol] = reference_price
                    return None, False, "EXEC_NO_ORDERBOOK_BLOCKED", diag
                # Non-urgent SELL that could not be priced: skip rather than send garbage.
                return None, False, "EXEC_SELL_NOT_PRICEABLE", diag

            if side_u == "BUY":
                self._last_failed_entry_price.pop(symbol, None)
            priced_order = replace(final_order, limit_price=pricing.limit_price, market=resolved_market)
            return priced_order, True, "EXEC_OK", diag
        except Exception as exc:  # noqa: BLE001 - preparation is best-effort; never kill the loop.
            self._record(
                {
                    "at": decision_time.isoformat(),
                    "symbol": symbol,
                    "kind": side_u,
                    "outcome": "exec_prepare_error",
                    "detail": f"{exc.__class__.__name__}: {exc}",
                }
            )
            diag["exec_prepare_error"] = f"{exc.__class__.__name__}: {exc}"
            # FAIL-CLOSED: a bug in exchange/pricing/quality prep must NOT submit an order
            # priced at the raw reference (last) price. BUY and non-urgent SELL are blocked.
            # Only an urgent stop/hard-stop/emergency SELL is still allowed to exit so a
            # preparation bug can never trap a losing position.
            try:
                _reason = classify_action_reason(
                    side_u, str((diagnostics or {}).get("exit_reason") or ""), tuple(reason_codes or ())
                )
                _urgent = side_u == "SELL" and is_urgent_sell(_reason)
            except Exception:  # noqa: BLE001 - classification must never raise here.
                _urgent = False
            if _urgent:
                return final_order, True, "EXEC_PREPARE_SKIPPED_EMERGENCY_SELL", diag
            return None, False, "EXEC_PREPARE_FAILED", diag

    def _forced_liquidation_result(self, holding: Holding, original_result: Any) -> Any:
        last_price = float(getattr(holding, "last_price", 0.0) or 0.0)
        average_price = float(getattr(holding, "average_price", 0.0) or 0.0)
        reference_price = last_price or average_price
        quantity = int(getattr(holding, "sellable_quantity", None) or getattr(holding, "quantity", 0) or 0)
        try:
            target_net_return = max(0.0, _env_float("LIVE_TERMINATION_TARGET_PROFIT_RATE", 0.0025))
            market, venue, instrument_type = _cost_context_for_liquidation_holding(holding)
            cost = TradingCostEngine().estimate(
                symbol=str(getattr(holding, "ticker", "") or ""),
                market=market,
                venue=venue,
                instrument_type=instrument_type,
                entry_price=average_price,
                expected_exit_price=max(reference_price, average_price),
                quantity=max(1, quantity),
                target_net_return=target_net_return,
            )
            reference_price = max(reference_price, cost.required_exit_price)
        except Exception:  # noqa: BLE001 - fallback still avoids knowingly selling below average cost.
            reference_price = max(reference_price, average_price * (1.0 + _env_float("LIVE_TERMINATION_TARGET_PROFIT_RATE", 0.0025)))
        order = FinalOrder(
            ticker=str(getattr(holding, "ticker", "") or ""),
            market=str(getattr(holding, "market", "") or "KR"),
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            quantity=max(0, quantity),
            limit_price=reference_price,
            manual_approval_required=False,
        )
        original_codes = tuple(getattr(original_result, "reason_codes", ()) or ())
        reason_codes = (
            "LIVE_TERMINATION_LOSS_MINIMIZING_LIQUIDATION",
            "SELL_BREAKEVEN_OR_BETTER_REQUESTED",
            *original_codes,
        )
        diagnostics = dict(getattr(original_result, "diagnostics", None) or {})
        diagnostics["exit_reason"] = "loss_minimizing_liquidation"
        diagnostics["liquidation_override"] = True
        diagnostics["loss_minimizing_reference_price"] = reference_price
        try:
            return replace(
                original_result,
                approved=True,
                final_order=order,
                reason_codes=reason_codes,
                diagnostics=diagnostics,
            )
        except TypeError:
            return SimpleNamespace(
                approved=True,
                final_order=order,
                reason_codes=reason_codes,
                diagnostics=diagnostics,
            )

    def _supervise_session(
        self,
        account: AccountSnapshot,
        macro_micro_bundle: Any,
        decision_time: datetime,
    ) -> Any:
        """Observe the elected/owned symbol and grade any violation.

        Everything read here was previously re-derived inside the strategy on
        every tick. It now lives in exactly one place, and the only thing it can
        do is halt — it can never open, price, or size an order.
        """
        session = self.strategy_session_manager.snapshot()
        symbol = str(session.get("selected_symbol") or "").upper()
        if not symbol:
            return None
        phase = str(session.get("phase") or "")
        holdings = {
            str(getattr(item, "ticker", "") or "").upper(): item
            for item in tuple(getattr(account, "holdings", ()) or ())
        }
        position_open = symbol in holdings

        tick = None
        orderbook = None
        store = getattr(self.decision_engine, "store", None)
        if store is not None:
            try:
                tick = store.latest_tick(symbol)
                orderbook = store.latest_orderbook(symbol)
            except Exception:  # noqa: BLE001 - a missing read is itself an observation.
                tick = orderbook = None
        data_age = None
        if tick is not None:
            received = getattr(tick, "received_at", None)
            if isinstance(received, datetime):
                reference = received if received.tzinfo else received.replace(tzinfo=timezone.utc)
                data_age = max(0.0, (decision_time - reference).total_seconds())

        session_tradable = None
        if self.market_open_provider is not None:
            holding = holdings.get(symbol)
            market = str(getattr(holding, "market", "") or "") if holding is not None else ""
            try:
                session_tradable = bool(self.market_open_provider(symbol, market))
            except Exception:  # noqa: BLE001 - treat an unusable provider as unknown.
                session_tradable = None

        macro = getattr(macro_micro_bundle, "macro_result", None)
        macro_risk = getattr(getattr(macro, "risk_level", None), "value", None)
        allowed = tuple(getattr(macro, "allowed_micro_strategies", ()) or ())
        blocked = tuple(getattr(macro, "blocked_micro_strategies", ()) or ())
        strategy_id = str(session.get("selected_strategy") or "") or None
        ontology_allows = None
        if strategy_id:
            # The macro layer uses a coarser strategy vocabulary, so translate
            # before comparing. An unanswerable check stays None rather than
            # being read as a withdrawal.
            from app.technical.strategy_algorithms import macro_strategy_permitted

            ontology_allows = macro_strategy_permitted(strategy_id, allowed, blocked)

        realized_pnl_today = float(getattr(account, "realized_pnl_today", 0.0) or 0.0)
        account_equity = max(1.0, float(getattr(account, "equity", 0.0) or 0.0))
        daily_limit = max(
            max(0.0, _env_float("REALTIME_DAILY_REALIZED_LOSS_BUY_STOP_KRW", 0.0)),
            account_equity * max(0.0, _env_float("REALTIME_DAILY_REALIZED_LOSS_BUY_STOP_RATE", 0.0)),
        )

        verdict = self.strategy_supervisor.evaluate(
            SupervisorObservation(
                symbol=symbol,
                as_of=decision_time,
                strategy_id=strategy_id,
                position_open=position_open,
                data_age_seconds=data_age,
                session_tradable=session_tradable,
                broker_healthy=None,
                ontology_allows_strategy=ontology_allows,
                macro_blocks_buy=_macro_micro_blocks_buy(macro_micro_bundle),
                macro_risk_level=macro_risk,
                liquidity_score=None,
                spread_bps=(
                    float(getattr(orderbook, "spread_bps", 0.0) or 0.0)
                    if orderbook is not None
                    else None
                ),
                realized_volatility=self._symbol_realtime_volatility_safe(symbol, decision_time),
                daily_realized_loss=realized_pnl_today,
                daily_loss_limit=daily_limit or None,
            )
        )
        if verdict.forces_exit and phase in {"ARMED", "ENTERING", "OWNED"}:
            self.strategy_session_manager.request_halt(
                symbol, verdict.level.value, verdict.hard_reason_codes
            )
            self._record(
                {
                    "at": decision_time.isoformat(),
                    "symbol": symbol,
                    "kind": "SUPERVISOR",
                    "outcome": "hard_halt",
                    "detail": ",".join(verdict.hard_reason_codes) or "UNSPECIFIED",
                }
            )
        elif verdict.level.blocks_new_entries and phase == "ARMED":
            self.strategy_session_manager.request_halt(
                symbol, verdict.level.value, verdict.soft_reason_codes
            )
        return verdict

    def _symbol_realtime_volatility_safe(self, symbol: str, decision_time: datetime) -> float | None:
        getter = getattr(self.decision_engine, "_symbol_realtime_volatility", None)
        if getter is None:
            return None
        try:
            return float(getter(symbol, decision_time))
        except Exception:  # noqa: BLE001 - volatility is observational only.
            return None

    def _forced_strategy_session_exit_result(
        self,
        holding: Holding,
        original_result: Any,
        exit_reason: str,
    ) -> Any:
        """Convert a session invalidation/target into a full-position SELL.

        This override does not bypass exchange, executable-price, quantity,
        account, or broker runtime gates. It only turns the owned strategy's
        explicit lifecycle exit into the same FinalOrder shape those gates use.
        """
        reference_price = float(
            getattr(holding, "last_price", 0.0)
            or getattr(holding, "average_price", 0.0)
            or 0.0
        )
        quantity = int(
            getattr(holding, "sellable_quantity", None)
            or getattr(holding, "quantity", 0)
            or 0
        )
        order = FinalOrder(
            ticker=str(getattr(holding, "ticker", "") or ""),
            market=str(getattr(holding, "market", "") or "KR"),
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            quantity=max(0, quantity),
            limit_price=reference_price,
            manual_approval_required=False,
        )
        original_codes = tuple(getattr(original_result, "reason_codes", ()) or ())
        reason_codes = (f"STRATEGY_SESSION_EXIT:{exit_reason}", exit_reason, *original_codes)
        diagnostics = dict(getattr(original_result, "diagnostics", None) or {})
        diagnostics["exit_reason"] = exit_reason.lower()
        diagnostics["strategy_session_override"] = True
        try:
            return replace(
                original_result,
                approved=True,
                final_order=order,
                reason_codes=reason_codes,
                diagnostics=diagnostics,
            )
        except TypeError:
            return SimpleNamespace(
                approved=True,
                final_order=order,
                reason_codes=reason_codes,
                diagnostics=diagnostics,
            )

    def _submit(
        self,
        order: FinalOrder,
        side: str,
        reason_codes: tuple[str, ...],
        decision_time: datetime,
        summary: dict[str, Any],
        pricing_diag: dict[str, Any] | None = None,
    ) -> bool:
        # 제출을 시도한 순간부터 쿨다운 시작(성공/차단/에러 무관) — 매초 재제출 방지.
        self._last_submit_monotonic[order.ticker] = time.monotonic()
        event: dict[str, Any] = {
            "at": decision_time.isoformat(),
            "symbol": order.ticker,
            "market": order.market,
            "kind": side,
            "quantity": order.quantity,
            "limit_price": order.limit_price,
            "reason": ";".join(reason_codes or ()),
        }
        if pricing_diag:
            # Execution diagnostics for GUI/logging: pricing policy, reference vs final
            # limit price, book, spread, orderbook age, and exchange resolution.
            event["execution"] = dict(pricing_diag)
        if str(side).upper() == "BUY" and (
            self._liquidation_requested
            or not self._buy_enabled
            or os.getenv("REALTIME_BUY_ENABLED", "true").strip().lower() in {"0", "false", "no", "off"}
        ):
            summary["blocked"] += 1
            summary["buy_rejected"] = int(summary.get("buy_rejected") or 0) + 1
            event["outcome"] = "blocked"
            event["detail"] = self._buy_disabled_reason or self._liquidation_reason or "BUY_SUBMIT_DISABLED"
            self._record(event)
            return False
        try:
            submission = self.coordinator.submit_final_order(order)
        except LiveExecutionBlocked as exc:
            summary["blocked"] += 1
            event["outcome"] = "blocked"
            event["detail"] = ";".join(getattr(exc, "reason_codes", ()) or ()) or str(exc)
            self._error_backoff_until[order.ticker] = time.monotonic() + self.config.error_cooldown_sec
            self._record(event)
            return False
        except Exception as exc:  # noqa: BLE001 - surface broker/API errors, keep looping.
            summary["errors"] += 1
            event["outcome"] = "error"
            event["detail"] = f"{exc.__class__.__name__}: {exc}"
            cooldown = (
                self.config.market_closed_error_cooldown_sec
                if _is_market_closed_order_error(exc)
                else self.config.error_cooldown_sec
            )
            self._error_backoff_until[order.ticker] = time.monotonic() + cooldown
            self._record(event)
            return False
        summary["submitted"] += 1
        # 매도 주문을 성공적으로 제출했으면, 그 주문이 처리될 때까지 같은 종목 재매도를 막는다
        # (미체결 매도 주문이 보유분을 묶어 가능수량=0 → 재매도 시 APBK0988 반복).
        if side == "SELL":
            self._error_backoff_until[order.ticker] = max(
                self._error_backoff_until.get(order.ticker, 0.0),
                time.monotonic() + self.config.sell_inflight_cooldown_sec,
            )
        event["outcome"] = "submitted"
        event["execution_id"] = getattr(submission, "execution_id", None)
        event["status"] = getattr(submission, "status", None)
        event["broker_order_id"] = getattr(submission, "broker_order_id", None)
        if side == "SELL" and getattr(submission, "broker_order_id", None):
            self._open_sell_orders[order.ticker] = {
                "broker_order_id": getattr(submission, "broker_order_id"),
                "order": order,
                "updated_at": decision_time.isoformat(),
            }
        self._record_submitted_order_for_performance(order, side)
        self._record(event)
        broker_order_id = str(getattr(submission, "broker_order_id", None) or "")
        if broker_order_id:
            self._poll_submitted_order_status_async(broker_order_id, order.ticker)
        return True

    def _poll_submitted_order_status_async(self, broker_order_id: str, symbol: str) -> None:
        thread = threading.Thread(
            target=self._poll_submitted_order_status,
            args=(broker_order_id, symbol),
            name=f"order-status-{symbol}-{broker_order_id}",
            daemon=True,
        )
        thread.start()

    def _poll_submitted_order_status(self, broker_order_id: str, symbol: str) -> None:
        try:
            snapshot = self.coordinator.poll_status(broker_order_id)
        except Exception as exc:  # noqa: BLE001 - status polling must not stop trading.
            self._record(
                {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "kind": "STATUS",
                    "outcome": "error",
                    "broker_order_id": broker_order_id,
                    "detail": f"{exc.__class__.__name__}: {exc}",
                }
            )
            return
        raw = getattr(snapshot, "raw", None)
        observed_status = str(getattr(snapshot, "status", "UNKNOWN") or "UNKNOWN").upper()
        if observed_status in {"FILLED", "CANCELLED", "CANCELED", "REJECTED", "EXPIRED"}:
            self._open_sell_orders.pop(symbol, None)
            # Broker account snapshots can lag a terminal order status. Keep the
            # symbol locked briefly so a stale holding row cannot create/amend a
            # second SELL after the first order has already filled.
            self._sell_lock_until[symbol] = time.monotonic() + max(
                30.0,
                min(120.0, self.config.sell_inflight_cooldown_sec),
            )
        self._record(
            {
                "at": getattr(snapshot, "observed_at", datetime.now(timezone.utc)).isoformat(),
                "symbol": symbol,
                "kind": "STATUS",
                "outcome": observed_status.lower(),
                "broker_order_id": broker_order_id,
                "filled_quantity": getattr(raw, "quantity", None),
                "average_fill_price": getattr(raw, "price", None),
            }
        )

    def _record_submitted_order_for_performance(self, order: FinalOrder, side: str) -> None:
        price = float(getattr(order, "limit_price", 0.0) or 0.0)
        quantity = float(getattr(order, "quantity", 0.0) or 0.0)
        if price <= 0.0 or quantity <= 0.0:
            return
        now = time.monotonic()
        if side == "BUY":
            queue = self._recent_buy_orders.setdefault(order.ticker, deque(maxlen=20))
            queue.append((price, quantity))
            return
        if side != "SELL":
            return
        queue = self._recent_buy_orders.get(order.ticker)
        if not queue:
            return
        buy_price, _buy_quantity = queue.popleft()
        if buy_price <= 0.0:
            return
        gross_return = price / buy_price - 1.0
        if gross_return <= self.config.loss_rebuy_return_threshold and self.config.loss_rebuy_cooldown_sec > 0.0:
            self._loss_cooldown_until[order.ticker] = now + self.config.loss_rebuy_cooldown_sec
            self._record(
                {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "symbol": order.ticker,
                    "kind": "CONTROL",
                    "outcome": "loss_symbol_cooldown",
                    "detail": f"gross_return={gross_return:.4f}",
                }
            )

    def _seed_loss_cooldowns_from_order_log(self) -> None:
        if self.config.loss_rebuy_cooldown_sec <= 0.0:
            return
        path = Path(self.config.order_log_path)
        if not path.exists():
            return
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        buys: dict[str, list[tuple[datetime, float, float]]] = {}
        latest_loss: dict[str, datetime] = {}
        for line in lines[-5000:]:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event_type") != "live_order_submitted":
                continue
            payload = record.get("payload") or {}
            order = payload.get("order") or {}
            symbol = str(order.get("ticker") or "")
            side = str(order.get("side") or "").upper()
            if not symbol or side not in {"BUY", "SELL"}:
                continue
            try:
                recorded_at = datetime.fromisoformat(str(record.get("recorded_at")).replace("Z", "+00:00"))
                price = float(order.get("limit_price") or 0.0)
                quantity = float(order.get("quantity") or 0.0)
            except (TypeError, ValueError):
                continue
            if price <= 0.0 or quantity <= 0.0:
                continue
            if side == "BUY":
                buys.setdefault(symbol, []).append((recorded_at, price, quantity))
                continue
            queue = buys.get(symbol)
            if not queue:
                continue
            buy_time, buy_price, _buy_quantity = queue.pop(0)
            del buy_time
            if buy_price <= 0.0:
                continue
            if price / buy_price - 1.0 <= self.config.loss_rebuy_return_threshold:
                latest_loss[symbol] = recorded_at
        now_wall = datetime.now(timezone.utc)
        now_mono = time.monotonic()
        for symbol, loss_time in latest_loss.items():
            if loss_time.tzinfo is None:
                loss_time = loss_time.replace(tzinfo=timezone.utc)
            elapsed = max(0.0, (now_wall - loss_time.astimezone(timezone.utc)).total_seconds())
            remaining = self.config.loss_rebuy_cooldown_sec - elapsed
            if remaining > 0.0:
                self._loss_cooldown_until[symbol] = now_mono + remaining

    def _amend_open_sell(
        self,
        order: FinalOrder,
        reason_codes: tuple[str, ...],
        decision_time: datetime,
        summary: dict[str, Any],
        pricing_diag: dict[str, Any] | None = None,
    ) -> bool:
        existing = self._open_sell_orders.get(order.ticker)
        broker_order_id = str((existing or {}).get("broker_order_id") or "")
        if not broker_order_id:
            self._open_sell_orders.pop(order.ticker, None)
            return self._submit(order, "SELL", reason_codes, decision_time, summary, pricing_diag=pricing_diag)
        event: dict[str, Any] = {
            "at": decision_time.isoformat(),
            "symbol": order.ticker,
            "market": order.market,
            "kind": "SELL",
            "quantity": order.quantity,
            "limit_price": order.limit_price,
            "reason": ";".join(reason_codes or ()),
            "broker_order_id": broker_order_id,
            "action": "amend_existing_sell",
        }
        if pricing_diag:
            event["execution"] = dict(pricing_diag)
        previous_order = (existing or {}).get("order")
        previous_price = float(getattr(previous_order, "limit_price", 0.0) or 0.0)
        price_delta = abs(float(order.limit_price or 0.0) - previous_price) / max(previous_price, 1e-9)
        if previous_order is not None and price_delta < self.config.sell_amend_min_price_delta:
            event["outcome"] = "open_sell_kept"
            event["detail"] = "existing_sell_order_same_price"
            self._record(event)
            return False
        try:
            amended = self.coordinator.amend_final_order(broker_order_id, order)
        except LiveExecutionBlocked as exc:
            summary["blocked"] += 1
            event["outcome"] = "blocked"
            event["detail"] = ";".join(getattr(exc, "reason_codes", ()) or ()) or str(exc)
            self._record(event)
            return False
        except Exception as exc:  # noqa: BLE001 - cancel and reorder if KIS refuses revision.
            if _is_no_available_sell_quantity_error(exc):
                self._open_sell_orders.pop(order.ticker, None)
                event["outcome"] = "open_sell_dropped"
                event["detail"] = f"amend_not_available={exc.__class__.__name__}: {exc}"
                self._record(event)
                return False
            if "원주문정보가 존재하지않습니다" in str(exc):
                self._open_sell_orders.pop(order.ticker, None)
                event["outcome"] = "open_sell_dropped"
                event["detail"] = f"amend_missing_origin={exc.__class__.__name__}: {exc}"
                self._record(event)
                return self._submit(order, "SELL", reason_codes, decision_time, summary, pricing_diag=pricing_diag)
            if "정정취소 가능수량" in str(exc) or "no quantity" in str(exc).lower():
                self._open_sell_orders.pop(order.ticker, None)
                event["outcome"] = "open_sell_dropped"
                event["detail"] = f"amend_not_available={exc.__class__.__name__}: {exc}"
                self._record(event)
                return False
            try:
                self.coordinator.cancel_final_order(broker_order_id, (existing or {}).get("order") or order)
                self._open_sell_orders.pop(order.ticker, None)
            except Exception as cancel_exc:  # noqa: BLE001
                summary["errors"] += 1
                event["outcome"] = "error"
                event["detail"] = (
                    f"amend_failed={exc.__class__.__name__}: {exc}; "
                    f"cancel_failed={cancel_exc.__class__.__name__}: {cancel_exc}"
                )
                self._record(event)
                return False
            event["outcome"] = "canceled_for_reorder"
            event["detail"] = f"amend_failed={exc.__class__.__name__}: {exc}"
            self._record(event)
            return self._submit(order, "SELL", reason_codes, decision_time, summary, pricing_diag=pricing_diag)
        new_order_id = getattr(amended, "broker_order_id", None) or broker_order_id
        self._open_sell_orders[order.ticker] = {
            "broker_order_id": new_order_id,
            "order": order,
            "updated_at": decision_time.isoformat(),
        }
        self._last_submit_monotonic[order.ticker] = time.monotonic()
        self._error_backoff_until[order.ticker] = max(
            self._error_backoff_until.get(order.ticker, 0.0),
            time.monotonic() + self.config.sell_inflight_cooldown_sec,
        )
        summary["amended"] += 1
        event["outcome"] = "amended"
        event["execution_id"] = getattr(amended, "execution_id", None)
        event["status"] = getattr(amended, "status", None)
        event["broker_order_id"] = new_order_id
        self._record(event)
        return True

    def _finish_cycle(self, summary: dict[str, Any]) -> None:
        if not isinstance(summary.get("rejection_reason_counts"), dict):
            summary["rejection_reason_counts"] = _rejection_reason_counts(summary)
        else:
            summary["rejection_reason_counts"] = dict(
                sorted(summary["rejection_reason_counts"].items(), key=lambda item: (-int(item[1] or 0), item[0]))
            )
        _commit_technical_decision_cycle()
        should_observe = self._should_observe_cycle(summary)
        with self._lock:
            self._status["cycles"] += 1
            self._status["last_cycle_at"] = summary["at"]
            self._status["submitted"] += summary["submitted"]
            self._status["amended"] += summary.get("amended", 0)
            self._status["buy_submitted"] += summary.get("buy_submitted", 0)
            self._status["sell_submitted"] += summary.get("sell_submitted", 0)
            self._status["blocked"] += summary["blocked"]
            self._status["errors"] += summary["errors"]
            self._status["last_reason"] = summary["reason"]
            self._status["last_summary"] = summary
            trace = dict(self._status.get("live_trace") or {})
            trace["finished_at"] = datetime.now(timezone.utc).isoformat()
            trace["completed"] = True
            trace["outcome"] = {
                "reason": summary.get("reason"),
                "buy_evaluated": int(summary.get("buy_evaluated") or 0),
                "buy_submit_attempted": int(summary.get("buy_submit_attempted") or 0),
                "buy_submitted": int(summary.get("buy_submitted") or 0),
                "sell_submitted": int(summary.get("sell_submitted") or 0),
            }
            self._status["live_trace"] = trace
        if should_observe and self.cycle_observer is not None:
            try:
                self.cycle_observer(dict(summary))
            except Exception:  # noqa: BLE001 - telemetry must never stop trading.
                pass

    def _should_observe_cycle(self, summary: dict[str, Any]) -> bool:
        reason = str(summary.get("reason") or "")
        submitted = int(summary.get("submitted") or 0)
        amended = int(summary.get("amended") or 0)
        blocked = int(summary.get("blocked") or 0)
        errors = int(summary.get("errors") or 0)
        attempted = int(summary.get("buy_submit_attempted") or 0)
        if submitted > 0 or amended > 0 or blocked > 0 or errors > 0 or attempted > 0:
            self._last_observed_cycle_monotonic = time.monotonic()
            self._last_observed_cycle_reason = reason
            return True
        if reason != self._last_observed_cycle_reason:
            self._last_observed_cycle_monotonic = time.monotonic()
            self._last_observed_cycle_reason = reason
            return True
        interval = max(5.0, _env_float("REALTIME_CYCLE_AUDIT_INTERVAL_SEC", 30.0))
        if time.monotonic() - self._last_observed_cycle_monotonic >= interval:
            self._last_observed_cycle_monotonic = time.monotonic()
            self._last_observed_cycle_reason = reason
            return True
        return False

    # ---- thread loop ----------------------------------------------------
    def run_forever(self, stop_event: threading.Event) -> None:
        interval_seconds = max(0.1, self.config.interval_ms / 1000.0)
        while not stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001 - never let the trading thread die silently.
                self._record({"at": datetime.now(timezone.utc).isoformat(), "kind": "CYCLE", "outcome": "error", "detail": f"{exc.__class__.__name__}: {exc}"})
            stop_event.wait(interval_seconds)

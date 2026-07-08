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
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, time as day_time, timezone
from pathlib import Path
from typing import Any, Callable, Deque
from zoneinfo import ZoneInfo

from app.execution.kis_errors import LiveExecutionBlocked
from app.execution.execution_quality import ExecutionQualityEngine, ExecutionQualityInput
from app.storage.execution_quality_store import ExecutionQualityStore
from app.schemas.domain import AccountSnapshot, FinalOrder, Holding


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
    take_profit: float = field(default_factory=lambda: _env_float("REALTIME_TAKE_PROFIT", 0.0025))
    stop_loss: float = field(default_factory=lambda: _env_float("REALTIME_STOP_LOSS", 0.010))
    buy_weight: float = field(default_factory=lambda: _env_float("REALTIME_BUY_WEIGHT", 0.01))
    max_orders_per_cycle: int = field(default_factory=lambda: max(1, _env_int("REALTIME_MAX_ORDERS_PER_CYCLE", 8)))
    # Keep fresh-account live trading conservative: after one accepted buy, wait for
    # the next broker account snapshot before sizing another buy.
    max_buy_orders_per_cycle: int = field(default_factory=lambda: max(1, _env_int("REALTIME_MAX_BUY_ORDERS_PER_CYCLE", 1)))
    max_buy_evaluations_per_cycle: int = field(default_factory=lambda: max(1, _env_int("REALTIME_MAX_BUY_EVALUATIONS_PER_CYCLE", 30)))
    # 같은 종목을 매 사이클(~1s) 재제출해 중복 주문/에러가 쌓이는 것을 막는 쿨다운.
    submit_cooldown_sec: float = field(default_factory=lambda: _env_float("REALTIME_SUBMIT_COOLDOWN_SEC", 20.0))
    # 하드 거부(브로커 에러/게이트 차단) 종목은 더 길게 쉬어 에러 폭주를 막는다(ETP 미신청·자금부족 등).
    error_cooldown_sec: float = field(default_factory=lambda: _env_float("REALTIME_ERROR_COOLDOWN_SEC", 300.0))
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
        self.config = config or RealtimeTradingConfig()
        # Execution-quality layer (Phase 3): rejects buys whose alpha would be consumed
        # by spread/slippage and records realized slippage per symbol/strategy.
        self.execution_quality = ExecutionQualityEngine(store=ExecutionQualityStore())
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
        }
        self._seed_loss_cooldowns_from_order_log()

    # ---- status ---------------------------------------------------------
    def disable_buys(self, reason: str = "BUY_DISABLED") -> None:
        self._buy_enabled = False
        self._buy_disabled_reason = reason
        os.environ["REALTIME_BUY_ENABLED"] = "false"
        self._record(
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "kind": "CONTROL",
                "outcome": "buy_disabled",
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
            status["loss_cooldown_symbols"] = sorted(
                symbol for symbol, until in self._loss_cooldown_until.items() if until > time.monotonic()
            )
            return status

    def _record(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._recent.appendleft(event)

    # ---- one cycle ------------------------------------------------------
    def run_once(self, decision_time: datetime | None = None) -> dict[str, Any]:
        decision_time = decision_time or datetime.now(timezone.utc)
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
            summary["reason"] = "MARKET_SESSION_CLOSED"
            self._finish_cycle(summary)
            return summary

        account = self.account_provider()
        if account is None:
            summary["reason"] = "NO_ACCOUNT_SNAPSHOT"
            self._finish_cycle(summary)
            return summary

        held_tickers = {h.ticker for h in (account.holdings or ())}
        # 매도·매수는 독립 예산을 갖는다 — 매도가 사이클 한도를 다 써서 매수를 굶기면 안 된다.
        sell_submitted = 0
        buy_submitted = 0
        buy_submit_attempted = 0
        buy_enabled = self._buy_enabled and os.getenv("REALTIME_BUY_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
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

        # 1) 매도: 보유 포지션의 빠른 청산.
        for holding in tuple(account.holdings or ()):
            holding_symbol = str(getattr(holding, "ticker", "") or "").upper()
            if holding_symbol in ignored_symbols:
                summary["skipped_ignored"] += 1
                continue
            if sell_submitted >= self.config.max_orders_per_cycle:
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
                # Record the sell so we don't immediately re-buy the same name (churn).
                self._recent_sell_monotonic[holding.ticker] = time.monotonic()
                if has_open_sell:
                    if self._amend_open_sell(final_order, result.reason_codes, decision_time, summary):
                        sell_submitted += 1
                elif self._submit(final_order, "SELL", result.reason_codes, decision_time, summary):
                    sell_submitted += 1
            else:
                summary["sell_rejected"] += 1
                self._append_rejection(summary, holding.ticker, "SELL", result.reason_codes, getattr(result, "diagnostics", None))

        # 2) 매수: 미보유 후보 진입(매도와 독립 예산).
        if not buy_enabled:
            summary["reason"] = summary["reason"] or (self._buy_disabled_reason or "BUY_DISABLED")
            summary["buy_disabled"] = True
            summary["buy_disabled_reason"] = self._buy_disabled_reason or "REALTIME_BUY_ENABLED=false"
            summary["buy_submitted"] = buy_submitted
            summary["buy_submit_attempted"] = buy_submit_attempted
            summary["sell_submitted"] = sell_submitted
            self._finish_cycle(summary)
            return summary

        for symbol in self.candidate_symbols_provider():
            symbol = str(symbol or "").upper()
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
                continue  # 거래소 마감: 신규 매수 보류.
            if self._in_cooldown(symbol):
                summary["skipped_cooldown"] += 1
                continue
            summary["buy_evaluated"] += 1
            try:
                result = self.decision_engine.evaluate_buy(
                    symbol,
                    account,
                    suggested_weight=self.config.buy_weight,
                    ontology_graph=ontology_graph,
                    decision_time=decision_time,
                )
            except Exception as exc:  # noqa: BLE001
                summary["errors"] += 1
                self._record({"at": decision_time.isoformat(), "symbol": symbol, "kind": "BUY", "outcome": "eval_error", "detail": f"{exc.__class__.__name__}: {exc}"})
                continue
            _record_technical_decision(symbol, "BUY", result)
            if result.approved and result.final_order is not None:
                buy_submit_attempted += 1
                summary["buy_submit_attempted"] = buy_submit_attempted
                exec_ok, exec_reason = self._execution_quality_gate(symbol, result, decision_time)
                if not exec_ok:
                    summary["buy_rejected"] += 1
                    self._append_rejection(summary, symbol, "BUY", (exec_reason,))
                    continue
                if self._submit(result.final_order, "BUY", result.reason_codes, decision_time, summary):
                    buy_submitted += 1
            else:
                summary["buy_rejected"] += 1
                self._append_rejection(summary, symbol, "BUY", result.reason_codes, getattr(result, "diagnostics", None))

        summary["buy_submitted"] = buy_submitted
        summary["buy_submit_attempted"] = buy_submit_attempted
        summary["sell_submitted"] = sell_submitted
        self._finish_cycle(summary)
        return summary

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

    def _execution_quality_gate(
        self, symbol: str, result: Any, decision_time: datetime
    ) -> tuple[bool, str]:
        """Execution-quality pre-submission gate (Phase 3). Best-effort; never raises.

        Rejects a buy whose alpha would be consumed by spread/expected slippage, blocks
        chasing a symbol upward after a failed/blocked entry, and records the reference
        price for realized-slippage tracking. Returns (ok, reason_code).
        """
        try:
            final_order = result.final_order
            reference_price = float(getattr(final_order, "limit_price", 0.0) or 0.0)
            if reference_price <= 0:
                return True, "EXEC_OK"  # nothing to assess; defer to downstream gates
            # No-chase guard: do not re-enter above the last blocked/failed entry price.
            last_failed = self._last_failed_entry_price.get(symbol)
            if last_failed is not None and reference_price > last_failed * 1.001:
                return False, "EXEC_NO_CHASE_AFTER_FAILED_ENTRY"
            pd = (result.diagnostics or {}).get("profitability_decision") or {}
            orderbook = None
            store = getattr(self.decision_engine, "store", None)
            if store is not None and hasattr(store, "latest_orderbook"):
                try:
                    orderbook = store.latest_orderbook(symbol)
                except Exception:  # noqa: BLE001
                    orderbook = None
            assessment = self.execution_quality.assess(
                ExecutionQualityInput(
                    symbol=symbol,
                    strategy_family=str(getattr(final_order, "strategy_family", "") or "live_short_horizon"),
                    decision_reference_price=reference_price,
                    gross_expected_return=float(pd.get("gross_expected_return", 0.0) or 0.0),
                    net_expected_return=float(pd.get("net_expected_return", 0.0) or 0.0),
                    required_min_net_return=float(pd.get("required_min_net_return", 0.0) or 0.0),
                    best_bid=float(getattr(orderbook, "best_bid", 0.0) or 0.0) if orderbook is not None else 0.0,
                    best_ask=float(getattr(orderbook, "best_ask", 0.0) or 0.0) if orderbook is not None else 0.0,
                    bid_depth=float(getattr(orderbook, "total_bid_volume", 0.0) or 0.0) if orderbook is not None else 0.0,
                    ask_depth=float(getattr(orderbook, "total_ask_volume", 0.0) or 0.0) if orderbook is not None else 0.0,
                    order_quantity=int(getattr(final_order, "quantity", 1) or 1),
                    side="BUY",
                )
            )
            if not assessment.allowed:
                # Record the rejected price so we do not chase this symbol upward.
                self._last_failed_entry_price[symbol] = reference_price
                return False, assessment.reject_reason or "EXEC_QUALITY_REJECTED"
            # Clear any prior no-chase marker once a clean entry passes.
            self._last_failed_entry_price.pop(symbol, None)
            return True, "EXEC_OK"
        except Exception as exc:  # noqa: BLE001 - execution-quality is best-effort.
            self._record(
                {
                    "at": decision_time.isoformat(),
                    "symbol": symbol,
                    "kind": "BUY",
                    "outcome": "exec_quality_error",
                    "detail": f"{exc.__class__.__name__}: {exc}",
                }
            )
            return True, "EXEC_QUALITY_SKIPPED"

    def _submit(
        self,
        order: FinalOrder,
        side: str,
        reason_codes: tuple[str, ...],
        decision_time: datetime,
        summary: dict[str, Any],
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
            self._error_backoff_until[order.ticker] = time.monotonic() + self.config.error_cooldown_sec
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
        self._record(
            {
                "at": getattr(snapshot, "observed_at", datetime.now(timezone.utc)).isoformat(),
                "symbol": symbol,
                "kind": "STATUS",
                "outcome": str(getattr(snapshot, "status", "UNKNOWN") or "UNKNOWN").lower(),
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
    ) -> bool:
        existing = self._open_sell_orders.get(order.ticker)
        broker_order_id = str((existing or {}).get("broker_order_id") or "")
        if not broker_order_id:
            self._open_sell_orders.pop(order.ticker, None)
            return self._submit(order, "SELL", reason_codes, decision_time, summary)
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
                return self._submit(order, "SELL", reason_codes, decision_time, summary)
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
            return self._submit(order, "SELL", reason_codes, decision_time, summary)
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

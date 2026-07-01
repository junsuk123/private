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

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque

from app.execution.kis_errors import LiveExecutionBlocked
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


@dataclass
class RealtimeTradingConfig:
    interval_ms: int = field(default_factory=lambda: max(100, _env_int("REALTIME_TRADING_INTERVAL_MS", 1000)))
    take_profit: float = field(default_factory=lambda: _env_float("REALTIME_TAKE_PROFIT", 0.006))
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
        self.config = config or RealtimeTradingConfig()
        self._lock = threading.Lock()
        self._last_submit_monotonic: dict[str, float] = {}
        self._error_backoff_until: dict[str, float] = {}
        self._open_sell_orders: dict[str, dict[str, Any]] = {}
        self._recent: Deque[dict[str, Any]] = deque(maxlen=recent_events_max)
        self._status: dict[str, Any] = {
            "cycles": 0,
            "last_cycle_at": None,
            "submitted": 0,
            "amended": 0,
            "buy_submitted": 0,
            "sell_submitted": 0,
            "blocked": 0,
            "errors": 0,
            "last_reason": None,
            "last_summary": None,
        }

    # ---- status ---------------------------------------------------------
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

        # 최신 온톨로지 추론 그래프(분석 컨텍스트)를 1회 조회해 매도 판단에 반영한다.
        ontology_graph = None
        if self.ontology_graph_provider is not None:
            try:
                ontology_graph = self.ontology_graph_provider()
            except Exception:  # noqa: BLE001 - ontology is best-effort.
                ontology_graph = None

        # 1) 매도: 보유 포지션의 빠른 청산.
        for holding in tuple(account.holdings or ()):
            if sell_submitted >= self.config.max_orders_per_cycle:
                break
            if self.market_open_provider is not None and not self.market_open_provider(holding.ticker, holding.market or ""):
                summary["skipped_market_closed"] += 1
                continue  # 거래소 마감: 지금 주문하면 브로커가 거부하므로 보류.
            has_open_sell = holding.ticker in self._open_sell_orders
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
            if result.approved and result.final_order is not None:
                if has_open_sell:
                    if self._amend_open_sell(result.final_order, result.reason_codes, decision_time, summary):
                        sell_submitted += 1
                elif self._submit(result.final_order, "SELL", result.reason_codes, decision_time, summary):
                    sell_submitted += 1
            else:
                summary["sell_rejected"] += 1
                self._append_rejection(summary, holding.ticker, "SELL", result.reason_codes)

        # 2) 매수: 미보유 후보 진입(매도와 독립 예산).
        for symbol in self.candidate_symbols_provider():
            if summary["buy_evaluated"] >= self.config.max_buy_evaluations_per_cycle:
                summary["reason"] = summary["reason"] or "BUY_EVALUATION_LIMIT_REACHED"
                break
            if buy_submitted >= min(self.config.max_orders_per_cycle, self.config.max_buy_orders_per_cycle):
                break
            if symbol in held_tickers:
                continue  # 보유 종목은 매도 감시 대상이므로 신규 매수에서 제외.
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
            if result.approved and result.final_order is not None:
                if self._submit(result.final_order, "BUY", result.reason_codes, decision_time, summary):
                    buy_submitted += 1
            else:
                summary["buy_rejected"] += 1
                self._append_rejection(summary, symbol, "BUY", result.reason_codes)

        summary["buy_submitted"] = buy_submitted
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
    ) -> None:
        rejections = summary.setdefault("rejections", [])
        if len(rejections) >= 12:
            return
        rejections.append(
            {
                "symbol": symbol,
                "side": side,
                "reason_codes": tuple(reason_codes or ()),
            }
        )

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
        self._record(event)
        return True

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
        try:
            amended = self.coordinator.amend_final_order(broker_order_id, order)
        except LiveExecutionBlocked as exc:
            summary["blocked"] += 1
            event["outcome"] = "blocked"
            event["detail"] = ";".join(getattr(exc, "reason_codes", ()) or ()) or str(exc)
            self._record(event)
            return False
        except Exception as exc:  # noqa: BLE001 - cancel and reorder if KIS refuses revision.
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

    # ---- thread loop ----------------------------------------------------
    def run_forever(self, stop_event: threading.Event) -> None:
        interval_seconds = max(0.1, self.config.interval_ms / 1000.0)
        while not stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001 - never let the trading thread die silently.
                self._record({"at": datetime.now(timezone.utc).isoformat(), "kind": "CYCLE", "outcome": "error", "detail": f"{exc.__class__.__name__}: {exc}"})
            stop_event.wait(interval_seconds)

"""Small read-only dashboard contract. Never obtains broker data or starts work."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any


STAGE_LABELS = {
    "engine": "엔진", "live_armed": "주문 권한", "market_session": "거래 시간",
    "buy_candidates": "후보 종목", "micro_buy_intents": "진입 조건",
    "strategy_election": "전략 선택", "position": "포지션",
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _order_rows(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize actual submission/status events; strategy signals are not orders."""
    accepted = {"SUBMITTED", "ACCEPTED", "OPEN", "WORKING", "FILLED", "PARTIALLY_FILLED", "REJECTED", "BLOCKED", "CANCELED", "CANCELLED", "ERROR"}
    normalized = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        symbol = event.get("ticker") or event.get("symbol")
        order_id = event.get("broker_order_id") or event.get("order_id")
        side = str(event.get("side") or event.get("kind") or "").upper()
        state = str(event.get("order_status") or event.get("status") or event.get("outcome") or "").upper()
        quantity = event.get("ordered_quantity", event.get("quantity"))
        if not symbol or state not in accepted:
            continue
        if not order_id and not (side in {"BUY", "SELL"} and quantity is not None):
            continue
        normalized.append({
            "occurred_at": event.get("occurred_at") or event.get("recorded_at") or event.get("at"),
            "ticker": symbol, "side": side if side in {"BUY", "SELL"} else None,
            "order_status": state, "broker_order_id": order_id,
            "ordered_quantity": quantity, "filled_quantity": event.get("filled_quantity"),
        })
    normalized.sort(key=lambda item: str(item.get("occurred_at") or ""))
    merged: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(normalized):
        key = str(row.get("broker_order_id") or f"unsubmitted:{index}")
        merged[key] = {**merged.get(key, {}), **{k: v for k, v in row.items() if v is not None}}
    return sorted(merged.values(), key=lambda item: str(item.get("occurred_at") or ""))[-20:]


def build_operations_overview(
    *,
    trading: Mapping[str, Any] | None = None,
    blockade: Mapping[str, Any] | None = None,
    markets: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    mode: Any = None,
    orders: Sequence[Mapping[str, Any]] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Normalize cached snapshots; an unknown state never means permission.

    KR and US observations are independent. The existing single execution owner
    is explicitly reported rather than implying two simultaneous order engines.
    """
    trading = _mapping(trading)
    status = _mapping(trading.get("status")) or trading
    session = _mapping(status.get("strategy_session")) or _mapping(_mapping(status.get("last_summary")).get("strategy_session"))
    blockade = _mapping(blockade)
    chain = [dict(item) for item in blockade.get("chain", []) if isinstance(item, Mapping)]
    first_blocker = next((item for item in chain if item.get("ok") is not True), None)
    session_link = next((item for item in chain if item.get("stage") == "market_session"), {})
    scanned = _mapping(_mapping(session_link.get("data")).get("scanned_groups"))
    markets = _mapping(markets)
    market_rows = {}
    for code in ("KR", "US"):
        raw = _mapping(markets.get(code)) or _mapping(scanned.get(code))
        market_rows[code] = {
            "market": code,
            "phase": raw.get("phase") or raw.get("session_phase") or "UNKNOWN",
            "allows_new_entry": raw.get("allows_new_entry"),
            "reason": raw.get("reason") or raw.get("detail"),
            "regime": raw.get("regime"),
            "source": raw.get("source"),
            "sources": raw.get("sources") or [],
            "quote_age_seconds": raw.get("quote_age_seconds"),
            "healthy_symbols": raw.get("healthy_symbols"),
            "strategy_ids": raw.get("strategy_ids") or raw.get("strategies") or [],
            "updated_at": raw.get("updated_at"),
        }
    pipeline = []
    blocked = False
    for link in chain:
        # The diagnostic checks may run independently: do not label a later
        # successful result 'unevaluated' solely because an earlier gate failed.
        state = "passed" if link.get("ok") is True else "blocked"
        pipeline.append({
            "stage": link.get("stage"),
            "label": STAGE_LABELS.get(link.get("stage"), link.get("stage") or "진단"),
            "state": state,
            "first_blocker": state == "blocked" and not blocked,
            "detail": link.get("detail") or "",
        })
        blocked = blocked or state == "blocked"
    armed_link = next((item for item in chain if item.get("stage") == "live_armed"), {})
    blocker = {
        "stage": first_blocker.get("stage") if first_blocker else blockade.get("blocking_stage"),
        "detail": first_blocker.get("detail") if first_blocker else blockade.get("blocking_detail"),
    }
    runtime = _mapping(runtime)
    return {
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(),
        "markets": market_rows,
        "trading": {
            "running": trading.get("running", status.get("running")),
            "mode": mode if isinstance(mode, str) else _mapping(_mapping(mode).get("active")).get("mode"),
            "armed": armed_link.get("ok"),
            "trading_possible": bool(chain)
            and trading.get("running", status.get("running")) is True
            and armed_link.get("ok") is True
            and blockade.get("ok") is not False
            and all(item.get("ok") is True for item in chain),
            "last_cycle_at": status.get("last_cycle_at") or session.get("last_cycle_at"),
            "selected_symbol": session.get("selected_symbol"),
            "selected_strategy": session.get("selected_strategy"),
            "phase": session.get("phase"),
            "execution_scope": "single_account_owner",
            "blocker": blocker,
            "pipeline": pipeline,
            "orders": _order_rows(orders or []),
        },
        "runtime": {key: runtime.get(key) for key in (
            "backend", "requested_backend", "device", "latency_ms", "model",
            "fallback_reason", "learning", "host", "state_path",
        )},
    }


OPERATIONS_PAGE = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>OBAITS · 트레이딩 현황</title>
  <link rel="icon" href="/static/icon.png">
  <link rel="stylesheet" href="/static/operations_dashboard.css?v=20260920">
  <script defer src="/static/operations_dashboard.js?v=20260920"></script>
</head>
<body>
  <a class="skip-link" href="#profit">손익으로 이동</a>
  <main class="dashboard">
    <header class="page-header">
      <div class="brand"><span class="brand-icon" aria-hidden="true">O</span><div><p class="eyebrow">OBAITS / TRADING DESK</p><h1>트레이딩 현황</h1></div></div>
      <nav aria-label="운영 메뉴"><a href="/account/advanced">전략 상세</a><a href="/display/ontology">관계 그래프</a><button id="refresh" type="button">새로고침</button></nav>
    </header>
    <div class="connection-bar"><span id="connection" role="status">상태 확인 중</span><span id="updated">—</span></div>

    <section id="profit" aria-labelledby="profit-title">
      <div class="section-heading"><h2 id="profit-title">계좌와 손익</h2><span id="account-freshness" class="badge">확인 대기</span></div>
      <div class="kpis">
        <article class="kpi primary"><h3>오늘 실현 손익</h3><strong id="pnl-today">—</strong><p>계좌 기준 · 체결된 거래</p></article>
        <article class="kpi"><h3>평가 손익</h3><strong id="pnl-unrealized">—</strong><p>현재 보유 종목</p></article>
        <article class="kpi"><h3>총자산</h3><strong id="assets">—</strong><p id="assets-detail">국내·해외 합산 · 원화 환산</p></article>
        <article class="kpi"><h3>주문 가능 현금</h3><strong id="cash-krw">—</strong><p id="cash-usd">USD —</p></article>
      </div>
      <p class="source-note" id="account-note">마지막으로 확인된 실계좌 데이터를 불러옵니다.</p>
    </section>

    <section aria-labelledby="markets-title">
      <div class="section-heading"><h2 id="markets-title">지금의 시장</h2><p>국내·미국 독립 관측</p></div>
      <div class="markets" id="markets"><article class="card empty">시장 상태 확인 중</article></div>
    </section>

    <section class="card execution" aria-labelledby="execution-title">
      <div class="section-heading"><h2 id="execution-title">거래는 어디까지 진행됐나요?</h2><span class="badge" id="execution-mode">확인 대기</span></div>
      <div class="verdict" id="verdict" role="status"><span class="verdict-indicator" aria-hidden="true"></span><div><h3 id="blocker-title">주문 경로 확인 중</h3><p id="blocker-detail">엔진 상태와 진입 조건을 확인합니다.</p></div></div>
      <ol class="pipeline" id="pipeline"></ol>
      <div class="execution-footer"><span id="execution-owner">선택된 전략 확인 대기</span><span>시장별 관측 · 계좌당 한 실행 주체</span></div>
    </section>

    <div class="lower-grid">
      <section class="card" aria-labelledby="holdings-title">
        <div class="section-heading"><h2 id="holdings-title">보유 종목</h2><span id="holding-count">—</span></div>
        <div class="table-scroll"><table><thead><tr><th scope="col">종목 / 시장</th><th scope="col">수량</th><th scope="col">현재가</th><th scope="col">평가 손익</th></tr></thead><tbody id="holdings"><tr><td colspan="4" class="empty">계좌 확인 대기</td></tr></tbody></table></div>
      </section>
      <section class="card" aria-labelledby="runtime-title">
        <div class="section-heading"><h2 id="runtime-title">추론과 모델</h2><span class="badge" id="backend-badge">확인 대기</span></div>
        <dl class="runtime-list" id="runtime"><div><dt>실제 추론 장치</dt><dd>확인 대기</dd></div></dl>
        <p class="source-note" id="runtime-note">실행 중인 장치와 검증된 모델을 표시합니다.</p>
      </section>
    </div>

    <section class="card" aria-labelledby="orders-title">
      <div class="section-heading"><h2 id="orders-title">최근 주문과 체결</h2><span id="order-count">—</span></div>
      <div class="table-scroll"><table><thead><tr><th scope="col">시각</th><th scope="col">종목</th><th scope="col">방향</th><th scope="col">주문 / 체결 수량</th><th scope="col">상태</th></tr></thead><tbody id="orders"><tr><td colspan="5" class="empty">주문 이력 확인 대기</td></tr></tbody></table></div>
    </section>
    <footer class="page-footer"><span>현금 주식 중심 · 관측과 체결을 구분합니다.</span><a href="/account/advanced">차트·학습·상세 진단 열기 ↗</a></footer>
  </main>
</body>
</html>"""

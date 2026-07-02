from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

from app.account_dashboard import AccountDashboardService


def create_account_router(
    *,
    status_provider: Callable[[], dict[str, Any] | None] | None = None,
    logs_provider: Callable[[], dict[str, Any] | None] | None = None,
) -> APIRouter:
    router = APIRouter()
    service = AccountDashboardService(status_provider=status_provider, logs_provider=logs_provider)

    @router.get("/account", response_class=HTMLResponse)
    def account_dashboard_page() -> HTMLResponse:
        return HTMLResponse(_ACCOUNT_PAGE)

    @router.get("/api/account/dashboard")
    def account_dashboard() -> JSONResponse:
        return JSONResponse(service.build_dashboard())

    @router.get("/api/account/holdings")
    def account_holdings() -> JSONResponse:
        return JSONResponse({"holdings": service.holdings()})

    @router.get("/api/account/cash")
    def account_cash() -> JSONResponse:
        return JSONResponse({"cash": service.cash()})

    @router.get("/api/account/profit")
    def account_profit() -> JSONResponse:
        dashboard = service.build_dashboard(persist=False)
        snapshot = dashboard.get("snapshot") or {}
        return JSONResponse(
            {
                "realized_pnl_today_krw": snapshot.get("realized_pnl_today_krw", 0),
                "realized_pnl_period_krw": snapshot.get("realized_pnl_period_krw", 0),
                "unrealized_pnl_krw": snapshot.get("unrealized_pnl_krw", 0),
                "total_pnl_krw": snapshot.get("total_pnl_krw", 0),
                "total_pnl_rate": snapshot.get("total_pnl_rate", 0),
            }
        )

    @router.get("/api/account/trades")
    def account_trades() -> JSONResponse:
        return JSONResponse({"trades": service.trades()})

    @router.get("/api/account/asset-history")
    def account_asset_history(range: str = "1D") -> JSONResponse:  # noqa: A002 - query parameter name.
        return JSONResponse({"range": range, "points": service.asset_history(range)})

    @router.get("/api/account/logs")
    def account_logs() -> JSONResponse:
        return JSONResponse(service.logs())

    return router


_ACCOUNT_PAGE = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Account Dashboard</title>
  <link rel="stylesheet" href="/static/account_dashboard.css" />
</head>
<body>
  <main class="account-dashboard" id="account-dashboard">
    <header class="account-header">
      <div>
        <h1>Account Dashboard</h1>
        <p id="account-source">loading</p>
      </div>
      <div class="account-actions">
        <button type="button" id="account-refresh">새로고침</button>
        <a href="/">매매 대시보드</a>
      </div>
    </header>

    <section class="dashboard-frame">
      <div class="frame-title">
        <h2>계좌 통합 요약</h2>
        <span id="account-stale-badge" class="badge">-</span>
      </div>
      <div class="kpi-grid" id="account-kpis"></div>
    </section>

    <section class="main-grid">
      <article class="dashboard-frame">
        <div class="frame-title">
          <h2>총자산 추이</h2>
          <div class="segmented" id="history-range">
            <button data-range="1D">1D</button>
            <button data-range="1W">1W</button>
            <button data-range="1M">1M</button>
            <button data-range="3M">3M</button>
          </div>
        </div>
        <canvas class="chart-frame" id="asset-chart" width="900" height="320"></canvas>
      </article>
      <article class="dashboard-frame">
        <div class="frame-title">
          <h2>자산 배분</h2>
        </div>
        <canvas class="chart-frame" id="allocation-chart" width="420" height="320"></canvas>
        <div class="allocation-list" id="allocation-list"></div>
      </article>
    </section>

    <section class="dashboard-frame">
      <div class="frame-title">
        <h2>보유 주식 현황</h2>
        <div class="account-filters">
          <input id="holding-search" placeholder="종목 검색" />
          <select id="holding-market">
            <option value="all">전체</option>
            <option value="domestic">국내주식</option>
            <option value="overseas">해외주식</option>
          </select>
        </div>
      </div>
      <div class="table-frame">
        <table>
          <thead><tr><th>종목</th><th>시장</th><th>수량</th><th>평단</th><th>현재가</th><th>평가금액</th><th>평가손익</th><th>수익률</th><th>비중</th><th>통화</th></tr></thead>
          <tbody id="holdings-body"></tbody>
        </table>
      </div>
    </section>

    <section class="main-grid">
      <article class="dashboard-frame">
        <div class="frame-title"><h2>최근 거래 및 손익</h2></div>
        <div class="table-frame compact">
          <table>
            <thead><tr><th>일시</th><th>시장</th><th>종목</th><th>구분</th><th>주문</th><th>체결</th><th>금액</th><th>상태</th></tr></thead>
            <tbody id="trades-body"></tbody>
          </table>
        </div>
      </article>
      <article class="dashboard-frame">
        <div class="frame-title"><h2>통화별 예수금</h2></div>
        <div class="table-frame compact">
          <table>
            <thead><tr><th>통화</th><th>잔고</th><th>주문가능</th><th>원화환산</th><th>환율</th></tr></thead>
            <tbody id="cash-body"></tbody>
          </table>
        </div>
      </article>
    </section>

    <section class="dashboard-frame system-strip" id="system-strip"></section>

    <details class="dashboard-frame log-details">
      <summary>진단 로그 및 오류</summary>
      <div class="log-panel" id="account-logs"></div>
    </details>
  </main>
  <script src="/static/account_dashboard.js"></script>
</body>
</html>
"""

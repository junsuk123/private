from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

from app.account_dashboard import AccountDashboardService


def create_account_router(
    *,
    status_provider: Callable[[], dict[str, Any] | None] | None = None,
    logs_provider: Callable[[], dict[str, Any] | None] | None = None,
    refactor_provider: Callable[[], dict[str, Any]] | None = None,
    market_view_provider: Callable[[str | None, int], dict[str, Any]] | None = None,
    market_stream_provider: Callable[[str, int], dict[str, Any]] | None = None,
    market_stream_observer: Callable[[str], None] | None = None,
    service: AccountDashboardService | None = None,
) -> APIRouter:
    router = APIRouter()
    # A shared service can be injected so a background sampler and the HTTP routes
    # write to the same snapshot store; otherwise build one from the providers.
    service = service or AccountDashboardService(status_provider=status_provider, logs_provider=logs_provider)

    @router.get("/account", response_class=HTMLResponse)
    def account_dashboard_page() -> HTMLResponse:
        return HTMLResponse(_STRATEGY_TERMINAL_PAGE)

    @router.get("/api/account/dashboard")
    def account_dashboard() -> JSONResponse:
        return JSONResponse(service.build_dashboard())

    @router.get("/api/account/summary")
    def account_summary() -> JSONResponse:
        return JSONResponse(service.cached_asset_summary())

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
      return JSONResponse({"trades": service.holding_orders()})

    @router.get("/api/account/asset-history")
    def account_asset_history(range: str = "1D") -> JSONResponse:  # noqa: A002 - query parameter name.
        return JSONResponse({"range": range, "points": service.asset_history(range)})

    @router.get("/api/account/logs")
    def account_logs() -> JSONResponse:
        return JSONResponse(service.logs())

    @router.get("/api/account/technical")
    def account_technical() -> JSONResponse:
        return JSONResponse(service.technical())

    @router.get("/api/account/macro-micro")
    def account_macro_micro() -> JSONResponse:
        return JSONResponse(service.macro_micro())

    @router.get("/api/refactor/dashboard")
    def refactor_dashboard() -> JSONResponse:
        return JSONResponse(refactor_provider() if refactor_provider else {})

    @router.get("/api/refactor/market-view")
    def refactor_market_view(symbol: str | None = None, limit: int = 180) -> JSONResponse:
        if market_view_provider is None:
            return JSONResponse({})
        try:
            return JSONResponse(market_view_provider(symbol, limit))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @router.get("/api/refactor/market-stream")
    def refactor_market_stream(symbol: str, limit: int = 30) -> JSONResponse:
        if market_stream_provider is None:
            return JSONResponse({})
        try:
            if market_stream_observer is not None:
                market_stream_observer(symbol)
            return JSONResponse(market_stream_provider(symbol, limit))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    return router


_ACCOUNT_PAGE = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Account Dashboard</title>
  <link rel="icon" type="image/png" href="/static/icon.png" />
  <link rel="apple-touch-icon" href="/static/icon.png" />
  <link rel="stylesheet" href="/static/account_dashboard.css?v=20260727-strategy-owner" />
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
        <button type="button" id="account-terminate" class="danger">&#51333;&#47308;</button>
        <a href="/">매매 대시보드</a>
        <a href="/api/realtime-trading/status" target="_blank" rel="noreferrer">자동거래 상태</a>
      </div>
    </header>

    <section class="dashboard-frame">
      <div class="frame-title">
        <h2>계좌 통합 요약</h2>
        <span id="account-stale-badge" class="badge">-</span>
      </div>
      <div class="kpi-grid" id="account-kpis"></div>
    </section>

    <section class="dashboard-frame">
      <div class="frame-title">
        <h2>수익성 요약 (비용 차감 기준)</h2>
        <span id="profitability-armed" class="badge">-</span>
      </div>
      <div class="kpi-grid" id="profitability-kpis"></div>
      <div class="profitability-note" id="profitability-note"></div>
    </section>

    <section class="dashboard-frame refactor-console" id="refactor-console">
      <div class="frame-title refactor-title">
        <div>
          <p class="eyebrow">ONTOLOGY-GATED · STRATEGY-OWNED</p>
          <h2>전략 운영 제어판</h2>
        </div>
        <div class="refactor-title-actions">
          <span id="refactor-device-badge" class="badge neutral">CPU</span>
          <span id="refactor-mode-badge" class="badge warn">불러오는 중</span>
        </div>
      </div>
      <div class="refactor-safety" id="refactor-safety"></div>
      <div class="refactor-kpis" id="refactor-kpis"></div>
      <div class="refactor-section-title">
        <h3>결정·실행 경로</h3>
        <span>활성 단계만 주문 경로에 참여합니다</span>
      </div>
      <div class="refactor-pipeline" id="refactor-pipeline"></div>
      <div class="refactor-grid">
        <article class="refactor-subpanel">
          <div class="subpanel-heading">
            <h3>승격 게이트</h3>
            <span id="refactor-gate-count">-</span>
          </div>
          <div class="promotion-gates" id="refactor-promotion-gates"></div>
        </article>
        <article class="refactor-subpanel">
          <div class="subpanel-heading">
            <h3>전략 소유 포지션</h3>
            <span id="refactor-owner-count">-</span>
          </div>
          <div class="owner-positions" id="refactor-owner-positions"></div>
        </article>
      </div>
      <div class="refactor-grid">
        <article class="refactor-subpanel">
          <div class="subpanel-heading">
            <h3>전략별 반사실 평가</h3>
            <span id="refactor-eval-status">-</span>
          </div>
          <div class="strategy-evaluation" id="refactor-strategy-evaluation"></div>
        </article>
        <article class="refactor-subpanel">
          <div class="subpanel-heading">
            <h3>Shadow 비교</h3>
            <span id="refactor-shadow-status">-</span>
          </div>
          <div class="shadow-comparison" id="refactor-shadow-comparison"></div>
        </article>
      </div>
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
        <h2>실시간 판단 흐름</h2>
        <span class="badge" id="decision-cycle-badge">-</span>
      </div>
      <div class="decision-live-strip" id="decision-live-strip"></div>
      <div class="decision-flow" id="decision-flow"></div>
      <div class="decision-grid">
        <article>
          <h3>최근 실행 판단</h3>
          <div class="decision-events" id="decision-events"></div>
        </article>
        <article>
          <h3>최근 보류 사유</h3>
          <div class="decision-rejections" id="decision-rejections"></div>
        </article>
      </div>
    </section>

    <section class="dashboard-frame">
      <div class="frame-title">
        <h2>온톨로지 지식 그래프</h2>
        <a href="/display/ontology" target="_blank" rel="noreferrer">전체화면으로 보기</a>
      </div>
      <iframe src="/display/ontology?embed=1" title="온톨로지 지식 그래프" loading="lazy"
        style="width:100%;height:min(68vh,440px);border:0;border-radius:10px;background:#0b0f16;display:block"></iframe>
    </section>

    <section class="dashboard-frame">
      <div class="frame-title">
        <h2>보유 주식 관리</h2>
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
          <thead><tr><th>종목</th><th>시장</th><th>수량</th><th>평단</th><th>현재가</th><th>주문 상태</th><th>주문 요약</th><th>평가손익</th><th>예상순손익</th><th>수익률</th><th>비중</th><th>통화</th></tr></thead>
          <tbody id="holdings-body"></tbody>
        </table>
      </div>
    </section>

    <section class="main-grid">
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

    <section class="dashboard-frame" id="macro-micro-panel">
      <div class="frame-title">
        <h2>거시–미시 온톨로지 (자문 전용)</h2>
        <span class="badge" id="macro-micro-badge">-</span>
      </div>
      <p class="tech-note">거시 온톨로지가 시장 레짐·후보·허용전략을 정하고, 미시 온톨로지가 종목별로 병렬 추론합니다. 최종 승인은 RiskManager·ProfitabilityGate가 가집니다.</p>
      <div class="mm-graph" id="mm-graph"></div>
      <div class="mm-macro" id="mm-macro"></div>
      <div class="mm-grid">
        <article><h3>후보 종목 미시 추론</h3><div class="tech-cards" id="mm-micro-cards"></div></article>
        <article><h3>통합 랭킹 (SELL/REDUCE 우선)</h3><div class="tech-cards" id="mm-ranked"></div></article>
      </div>
    </section>

    <section class="dashboard-frame" id="technical-panel">
      <div class="frame-title">
        <h2>기술적 예측 (자문 전용)</h2>
        <span class="badge" id="technical-badge">-</span>
      </div>
      <p class="tech-note">RiskManager·ProfitabilityGate가 최종 권한을 가지며, 아래는 근거·설명일 뿐입니다.</p>
      <div class="tech-grid">
        <article><h3>매수 승인</h3><div class="tech-cards" id="tech-buy-approved"></div></article>
        <article><h3>매수 보류/거부</h3><div class="tech-cards" id="tech-buy-rejected"></div></article>
        <article><h3>매도/축소</h3><div class="tech-cards" id="tech-sell-reduce"></div></article>
        <article><h3>보유/관망</h3><div class="tech-cards" id="tech-hold"></div></article>
      </div>
    </section>

    <section class="dashboard-frame system-strip" id="system-strip"></section>

    <details class="dashboard-frame log-details">
      <summary>진단 로그 및 오류</summary>
      <div class="log-panel" id="account-logs"></div>
    </details>
  </main>
  <script src="/static/account_dashboard.js?v=20260730-expenses-gnn-trust"></script>
</body>
</html>
"""


_STRATEGY_TERMINAL_PAGE = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Ontology Strategy Terminal</title>
  <link rel="icon" type="image/png" href="/static/icon.png" />
  <link rel="stylesheet" href="/static/strategy_terminal.css?v=20260730-ops-overview-1" />
  <link rel="stylesheet" href="/static/operations_overview.css?v=20260730-ops-overview-1" />
</head>
<body>
  <main class="terminal-shell">
    <header class="terminal-header">
      <div class="brand">
        <span class="brand-mark">O</span>
        <div>
          <p>ONTOLOGY-GATED EXECUTION</p>
          <h1>Strategy Trading Terminal</h1>
        </div>
      </div>
      <div class="header-status">
        <span class="connection"><i></i><b id="feed-state">데이터 연결 확인 중</b></span>
        <span class="status-chip" id="terminal-mode">SHADOW</span>
        <time id="terminal-clock">--:--:--</time>
        <button type="button" id="terminal-refresh" aria-label="새로고침">↻</button>
      </div>
    </header>

    <section class="ops-overview" id="ops-overview" aria-labelledby="ops-overview-title">
      <div class="ops-overview-head">
        <div>
          <p class="panel-kicker">LIVE OPERATIONS OVERVIEW</p>
          <h2 id="ops-overview-title">통합 운영 관제</h2>
          <p id="ops-overall-summary">서버와 실거래 파이프라인 상태를 불러오는 중입니다.</p>
        </div>
        <div class="ops-overall-state waiting" id="ops-overall-state">
          <i></i>
          <div><strong>확인 중</strong><small id="ops-updated-at">갱신 대기</small></div>
        </div>
      </div>

      <div class="ops-alert waiting" id="ops-alert">
        <span>현재 핵심 상태</span>
        <strong id="ops-alert-title">상태 확인 중</strong>
        <p id="ops-alert-detail">각 운영 게이트의 최신 값을 확인하고 있습니다.</p>
      </div>

      <div class="ops-gate-grid" id="ops-gate-grid">
        <article class="waiting" data-ops-gate="server"><span>01 서버</span><strong>확인 중</strong><small>-</small></article>
        <article class="waiting" data-ops-gate="broker"><span>02 계좌·브로커</span><strong>확인 중</strong><small>-</small></article>
        <article class="waiting" data-ops-gate="market"><span>03 실시간 시세</span><strong>확인 중</strong><small>-</small></article>
        <article class="waiting" data-ops-gate="ontology"><span>04 온톨로지</span><strong>확인 중</strong><small>-</small></article>
        <article class="waiting" data-ops-gate="gnn"><span>05 GNN 신뢰도</span><strong>확인 중</strong><small>-</small></article>
        <article class="waiting" data-ops-gate="execution"><span>06 전략·주문</span><strong>확인 중</strong><small>-</small></article>
      </div>

      <div class="ops-detail-grid">
        <article class="ops-card">
          <div class="ops-card-head"><h3>실시간 시장 데이터</h3><span id="ops-feed-state">-</span></div>
          <div class="ops-kpi-grid">
            <div><span>구독 종목</span><strong id="ops-feed-symbols">-</strong></div>
            <div><span>구독 승인</span><strong id="ops-feed-accepted">-</strong></div>
            <div><span>체결 / 호가</span><strong id="ops-feed-events">-</strong></div>
            <div><span>건강 종목</span><strong id="ops-feed-healthy">-</strong></div>
          </div>
          <p class="ops-card-note" id="ops-feed-note">실시간 수집 상태 확인 중</p>
        </article>

        <article class="ops-card">
          <div class="ops-card-head"><h3>온톨로지·연구 파이프라인</h3><span id="ops-ontology-state">-</span></div>
          <div class="ops-kpi-grid">
            <div><span>컨텍스트 이벤트</span><strong id="ops-context-events">-</strong></div>
            <div><span>온톨로지 연결</span><strong id="ops-ontology-links">-</strong></div>
            <div><span>연구 사이클</span><strong id="ops-research-cycle">-</strong></div>
            <div><span>모델 상태</span><strong id="ops-model-state">-</strong></div>
          </div>
          <p class="ops-card-note" id="ops-ontology-note">연구 및 그래프 상태 확인 중</p>
        </article>

        <article class="ops-card ops-gnn-card">
          <div class="ops-card-head"><h3>GNN 실시간 신뢰도</h3><span id="ops-gnn-state">-</span></div>
          <div class="ops-score-line">
            <strong id="ops-gnn-score">-</strong>
            <div><span id="ops-gnn-samples">표본 -</span><small id="ops-gnn-trusted">신뢰 전략 -</small></div>
          </div>
          <div class="ops-progress"><i id="ops-gnn-progress"></i></div>
          <div class="ops-mini-metrics">
            <span>양수 순효율 <b id="ops-gnn-positive">-</b></span>
            <span>평균 순효율 <b id="ops-gnn-net">-</b></span>
            <span>불확실성 <b id="ops-gnn-uncertainty">-</b></span>
          </div>
          <div class="ops-strategy-list" id="ops-strategy-list"></div>
        </article>

        <article class="ops-card">
          <div class="ops-card-head"><h3>전략 선택·실거래 엔진</h3><span id="ops-engine-state">-</span></div>
          <div class="ops-kpi-grid">
            <div><span>세션 단계</span><strong id="ops-session-phase">-</strong></div>
            <div><span>선택 전략</span><strong id="ops-selected-strategy">-</strong></div>
            <div><span>매수 후보</span><strong id="ops-buy-candidates">-</strong></div>
            <div><span>주문 / 오류</span><strong id="ops-order-errors">-</strong></div>
          </div>
          <p class="ops-card-note" id="ops-engine-note">전략 세션 상태 확인 중</p>
        </article>
      </div>

      <div class="ops-footer">
        <div><span>운영 모드</span><strong id="ops-mode">-</strong></div>
        <div><span>자동 신뢰도</span><strong id="ops-reliability">-</strong></div>
        <div><span>GNN 실거래 필수</span><strong id="ops-gnn-required">-</strong></div>
        <div><span>누적 엔진 주기</span><strong id="ops-engine-cycles">-</strong></div>
        <div class="ops-footer-wide"><span>현재 판단 사유</span><strong id="ops-current-reason">-</strong></div>
      </div>
    </section>

    <section class="live-owner-strip" aria-labelledby="live-owner-title">
      <div class="live-owner-heading">
        <p class="panel-kicker">LIVE STRATEGY EXECUTION</p>
        <h2 id="live-owner-title">실시간 자동 트레이딩</h2>
        <small>온톨로지·GNN이 채택한 종목과 전략부터 KIS 주문 이행·체결까지 자동 추적합니다.</small>
      </div>
      <div class="live-owner-metrics">
        <article><span>실거래 세션</span><strong id="live-owner-state">확인 중</strong><small id="live-owner-session">-</small></article>
        <article><span>채택 종목</span><strong id="live-owner-symbol">대기</strong><small id="live-owner-market">자동 전환</small></article>
        <article><span>채택 알고리즘</span><strong id="live-owner-strategy">대기</strong><small id="live-owner-source">온톨로지 판단 대기</small></article>
        <article><span>주문 이행</span><strong id="live-owner-order">대기</strong><small id="live-owner-cycle">마지막 주기 -</small></article>
      </div>
    </section>

    <section class="asset-overview" aria-labelledby="asset-overview-title">
      <div class="asset-overview-head">
        <div>
          <p class="panel-kicker">MY ACCOUNT</p>
          <h2 id="asset-overview-title">내 자산</h2>
        </div>
        <div class="asset-trust">
          <span class="status-chip blocked" id="asset-status">확인 중</span>
          <small id="asset-verified-at">마지막 확인 -</small>
        </div>
      </div>
      <div class="asset-metrics" id="asset-metrics">
        <article class="asset-primary">
          <span>총자산</span>
          <strong id="asset-total">-</strong>
          <small id="asset-source">계좌 정보를 불러오는 중입니다.</small>
        </article>
        <article><span>현금성 자산</span><strong id="asset-cash">-</strong><small id="asset-cash-detail">원화 - · 외화 -</small></article>
        <article><span>보유주식 평가</span><strong id="asset-stocks">-</strong><small id="asset-stock-detail">국내 - · 해외 -</small></article>
        <article><span>평가·실현 손익</span><strong id="asset-pnl">-</strong><small id="asset-pnl-rate">수익률 -</small></article>
      </div>
      <div class="asset-warning" id="asset-warning" role="status"></div>
    </section>

    <section class="diagnostics-panel terminal-panel" aria-labelledby="diagnostics-title">
      <div class="diagnostics-head">
        <div>
          <p class="panel-kicker">SYSTEM LEARNING &amp; TRADE READINESS</p>
          <h2 id="diagnostics-title">시스템 진행 상태</h2>
          <p id="diagnostics-summary">학습·수집·실거래 게이트를 확인하는 중입니다.</p>
        </div>
        <div class="diagnostics-score">
          <span id="diagnostics-mode">확인 중</span>
          <strong id="diagnostics-score">-</strong>
          <small id="diagnostics-threshold">승격 기준 -</small>
        </div>
      </div>
      <div class="diagnostics-progress" aria-label="실거래 준비도">
        <i id="diagnostics-progress-bar"></i>
      </div>
      <div class="diagnostics-grid">
        <article>
          <div class="diagnostics-subhead">
            <h3>실행 워커</h3>
            <span id="diagnostics-worker-count">-</span>
          </div>
          <div class="worker-grid" id="diagnostics-workers"></div>
        </article>
        <article>
          <div class="diagnostics-subhead">
            <h3>실거래 정체 원인</h3>
            <span id="diagnostics-blocker-count">-</span>
          </div>
          <div class="blocker-list" id="diagnostics-blockers"></div>
        </article>
      </div>
      <div class="diagnostics-grid diagnostics-lower">
        <article>
          <div class="diagnostics-subhead">
            <h3>수집·학습 증거</h3>
            <span id="diagnostics-generated-at">-</span>
          </div>
          <div class="evidence-grid" id="diagnostics-evidence"></div>
        </article>
        <article>
          <div class="diagnostics-subhead">
            <h3>최근 활동</h3>
            <span id="diagnostics-next-run">-</span>
          </div>
          <div class="activity-list" id="diagnostics-activity"></div>
        </article>
      </div>
      <section class="training-monitor" aria-labelledby="training-monitor-title">
        <div class="training-monitor-head">
          <div>
            <p class="panel-kicker">REAL-TIME MODEL LEARNING</p>
            <h3 id="training-monitor-title">실시간 학습 과정</h3>
            <small id="training-monitor-note">완료된 학습 사이클을 불러오는 중입니다.</small>
          </div>
          <div class="training-monitor-state">
            <span id="training-cycle-state">WAITING</span>
            <time id="training-last-cycle">-</time>
          </div>
        </div>
        <div class="training-kpis">
          <article>
            <span>분류 학습률</span>
            <strong id="training-learning-rate">-</strong>
            <small id="training-optimizer">옵티마이저 확인 중</small>
          </article>
          <article>
            <span>AUC 변화</span>
            <strong id="training-auc-change">-</strong>
            <small id="training-current-auc">현재 AUC -</small>
          </article>
          <article>
            <span>학습 데이터 증가</span>
            <strong id="training-row-rate">-</strong>
            <small id="training-new-rows">최근 신규 행 -</small>
          </article>
          <article>
            <span>모델 교체</span>
            <strong id="training-deployment-state">-</strong>
            <small id="training-deployment-reason">평가 기록 확인 중</small>
          </article>
        </div>
        <div class="training-chart-grid">
          <article class="training-chart-card">
            <div class="training-chart-title">
              <b>검증 성능 추이</b>
              <span><i class="auc"></i>AUC <i class="precision"></i>Precision@K <i class="returns"></i>상위 수익(bp)</span>
            </div>
            <canvas id="training-performance-chart" aria-label="검증 성능 추이 차트"></canvas>
          </article>
          <article class="training-chart-card">
            <div class="training-chart-title">
              <b>학습 데이터 누적</b>
              <span><i class="rows"></i>누적 행 <i class="new-rows"></i>신규 행</span>
            </div>
            <canvas id="training-data-chart" aria-label="학습 데이터 누적 차트"></canvas>
          </article>
        </div>
        <div class="training-cycle-list" id="training-cycle-list"></div>
      </section>
    </section>

    <section class="selection-strip">
      <div class="section-label">
        <span>01</span>
        <div><b>온톨로지 후보</b><small>전략 게이트가 평가한 종목</small></div>
      </div>
      <div class="candidate-list" id="candidate-list"></div>
    </section>

    <section class="instrument-hero">
      <div class="instrument-title">
        <span class="market-pill" id="instrument-market">ONTOLOGY</span>
        <h2 id="instrument-symbol">선택 대기</h2>
        <div class="price-line">
          <strong id="instrument-price">-</strong>
          <span id="instrument-change">-</span>
        </div>
      </div>
      <div class="instrument-stats" id="instrument-stats"></div>
      <article class="algorithm-card">
        <div class="algorithm-head">
          <span>SELECTED ALGORITHM</span>
          <i id="algorithm-state">NO TRADE</i>
        </div>
        <h3 id="algorithm-name">온톨로지 선택 대기</h3>
        <p id="algorithm-thesis">필수 사실과 기대 순효용을 통과한 전략만 표시됩니다.</p>
        <div class="algorithm-tags" id="algorithm-tags"></div>
      </article>
    </section>

    <section class="terminal-panel second-analysis-panel">
      <div class="panel-head">
        <div>
          <p class="panel-kicker">SECOND-LEVEL MICROSTRUCTURE</p>
          <h2>1초 체결·호가 분석</h2>
        </div>
        <span class="status-chip blocked" id="second-data-status">SECOND DATA WAITING</span>
      </div>
      <div class="second-analysis-grid">
        <div class="second-chart-wrap">
          <canvas id="second-price-chart"></canvas>
          <div class="chart-empty" id="second-chart-empty">초단위 체결 데이터를 기다리고 있습니다.</div>
        </div>
        <div class="second-metrics" id="second-metrics"></div>
      </div>
      <div class="second-analysis-footer">
        <span id="second-analysis-time">최근 갱신 -</span>
        <span id="second-analysis-gate">실매수 진입에는 최근 10초 동안 여러 초 구간의 체결과 실시간 호가가 필요합니다.</span>
      </div>
    </section>

    <section class="terminal-panel decision-ontology-panel">
      <div class="panel-head ontology-graph-head">
        <div>
          <p class="panel-kicker">LIVE DECISION ONTOLOGY</p>
          <h2>데이터 → 지표 → 알고리즘 → 최종 판단</h2>
          <p class="ontology-graph-summary" id="decision-ontology-summary">실시간 선택 경로를 불러오는 중입니다.</p>
        </div>
        <div class="ontology-graph-actions">
          <span class="status-chip" id="decision-ontology-live">WAITING</span>
          <button type="button" class="graph-filter active" data-graph-filter="active">선택 경로</button>
          <button type="button" class="graph-filter" data-graph-filter="all">전체 관계</button>
        </div>
      </div>
      <div class="ontology-legend">
        <span class="active-path">활성 선택 경로</span>
        <span class="pass-path">조건 충족</span>
        <span class="block-path">조건 미충족·최종 차단</span>
        <span class="unknown-path">데이터 없음</span>
      </div>
      <div class="decision-ontology-layout">
        <div class="decision-ontology-canvas">
          <svg id="decision-ontology-graph" role="img" aria-label="실시간 데이터와 알고리즘 선택 관계 그래프"></svg>
        </div>
        <aside class="ontology-inspector" id="ontology-inspector">
          <span>NODE INSPECTOR</span>
          <h3>노드를 선택하세요</h3>
          <p>데이터, 지표, 연결선 또는 알고리즘을 누르면 값·조건·출처를 확인할 수 있습니다.</p>
        </aside>
      </div>
      <div class="algorithm-catalog" id="algorithm-catalog"></div>
      <p class="ontology-provenance" id="ontology-provenance"></p>
    </section>

    <section class="workspace-grid">
      <article class="terminal-panel chart-panel">
        <div class="panel-head">
          <div>
            <p class="panel-kicker">REAL-TIME MARKET</p>
            <h2>가격·알고리즘 시각화</h2>
          </div>
          <div class="chart-head-tools">
            <div class="chart-stream-state" id="chart-stream-state"><i></i><span>1초 스트림 연결 중</span></div>
            <div class="chart-timeframe">
              <button type="button" class="active" data-chart-mode="seconds">1초</button>
              <button type="button" data-chart-mode="minutes">1분</button>
            </div>
            <div class="chart-legend">
              <span class="candle-up">상승봉</span>
              <span class="candle-down">하락봉</span>
              <span class="ma-fast">MA5</span>
              <span class="ma-slow">MA20</span>
              <span class="vwap-line">VWAP</span>
            </div>
          </div>
        </div>
        <div class="chart-wrap">
          <canvas id="price-chart"></canvas>
          <div class="chart-empty" id="chart-empty">선택 종목의 분봉을 기다리고 있습니다.</div>
        </div>
        <div class="volume-wrap"><canvas id="volume-chart"></canvas></div>
        <div class="chart-footer">
          <span id="chart-range">최근 180개 1분봉</span>
          <span id="chart-updated">마지막 이벤트 -</span>
        </div>
      </article>

      <aside class="terminal-panel ontology-panel">
        <div class="panel-head">
          <div>
            <p class="panel-kicker">CLOSED-WORLD GATE</p>
            <h2>온톨로지 결정 근거</h2>
          </div>
          <span class="status-chip blocked" id="ontology-status">BLOCKED</span>
        </div>
        <div class="ontology-flow" id="ontology-flow"></div>
        <div class="reason-box">
          <h3>판정 근거</h3>
          <div id="ontology-reasons"></div>
        </div>
        <div class="decision-compare" id="decision-compare"></div>
      </aside>
    </section>

    <section class="terminal-panel execution-panel">
      <div class="panel-head">
        <div>
          <p class="panel-kicker">CAUSAL ORDER LIFECYCLE</p>
          <h2>주문·체결 과정</h2>
        </div>
        <span class="status-chip" id="execution-count">0 EVENTS</span>
      </div>
      <div class="execution-track" id="execution-track"></div>
      <div class="execution-bottom">
        <div class="execution-tape" id="execution-tape"></div>
        <div class="orderbook-card" id="orderbook-card"></div>
      </div>
    </section>

    <footer class="terminal-footer">
      <div><span>SAFETY</span><strong id="safety-state">실주문 차단</strong></div>
      <p id="safety-reason">승격 게이트와 전략 소유 실행 상태를 확인하고 있습니다.</p>
      <a href="/api/refactor/market-view" target="_blank" rel="noreferrer">RAW DATA ↗</a>
    </footer>
  </main>
  <script src="/static/strategy_terminal.js?v=20260730-entry-trust-v2"></script>
  <script src="/static/operations_overview.js?v=20260730-entry-trust-v2"></script>
</body>
</html>
"""

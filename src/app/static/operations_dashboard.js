/* Read-only cached monitoring. Missing observations must never look like zero P&L
   or live order permission. No broker refresh, POST, WebGL, or animation loop. */
(() => {
  'use strict';
  const state = { busy: false, timer: null, operations: null, account: null, failures: [], lastSuccess: null };
  const $ = (id) => document.getElementById(id);
  const text = (id, value) => { const element = $(id); if (element) element.textContent = value ?? '—'; };
  const escape = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
  const numeric = (value) => value === null || value === undefined || value === '' ? null : Number.isFinite(Number(value)) ? Number(value) : null;
  const number = (value, digits = 0) => numeric(value) === null ? '—' : numeric(value).toLocaleString('ko-KR', { maximumFractionDigits: digits });
  const money = (value) => numeric(value) === null ? '—' : `${number(value)}원`;
  const clock = (value) => !value || Number.isNaN(new Date(value).getTime()) ? '확인 대기' : new Date(value).toLocaleString('ko-KR', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
  const signed = (value) => numeric(value) > 0 ? `+${money(value)}` : money(value);
  const tone = (value) => numeric(value) > 0 ? 'positive' : numeric(value) < 0 ? 'negative' : '';
  const phases = { UNKNOWN: '확인 대기', REGULAR: '정규장', PRE: '프리마켓', PREMARKET: '프리마켓', AFTER: '애프터마켓', AFTERHOURS: '애프터마켓', CLOSED: '휴장', DAY: '주간 거래', DAYTIME: '주간 거래', AUCTION: '동시호가', NXT: 'NXT', KR_REGULAR: '정규장', US_REGULAR: '정규장', KR_CLOSED: '휴장', US_CLOSED: '휴장', US_PREMARKET: '프리마켓', US_AFTERHOURS: '애프터마켓', US_DAYTIME: '주간 거래' };
  Object.assign(phases, { KRX_PREOPEN: 'KRX 장전', KRX_OPENING_AUCTION: 'KRX 시초 동시호가', KRX_REGULAR: 'KRX 정규장', KRX_CLOSING_AUCTION: 'KRX 종가 동시호가', KRX_AFTER_CLOSE: 'KRX 장후', KRX_AFTER_SINGLE_PRICE: 'KRX 시간외 단일가', NXT_PRE: 'NXT 프리마켓', NXT_REGULAR: 'NXT 정규장', NXT_POST: 'NXT 애프터마켓', US_AFTERMARKET: '애프터마켓' });
  const regimes = { bull: '상승', bullish: '상승', trend_up: '상승 추세', bear: '하락', bearish: '하락', trend_down: '하락 추세', sideways: '횡보', range: '횡보', volatile: '변동성 확대', high_volatility: '변동성 확대', high_vol: '변동성 확대', stress: '위험 회피', risk_off: '위험 회피', neutral: '중립', unknown: '확인 대기' };
  const reasons = { MARKET_POLICY_CASH_REGIME: '시장 프로필: 현금 대기 권고', SESSION_LIVE_AUTHORIZATION_REQUIRED: '세션 실주문 권한 확인 필요', SESSION_CLOSED: '현재 거래 가능 세션 없음', NO_ACTIVE_SESSION: '현재 거래 가능 세션 없음' };
  const stages = { engine: '엔진', live_armed: '주문 권한', market_session: '거래 시간', buy_candidates: '후보 종목', micro_buy_intents: '진입 조건', strategy_election: '전략 선택', position: '포지션' };
  const orderStates = { FILLED: '체결', PARTIALLY_FILLED: '일부 체결', ACCEPTED: '접수', SUBMITTED: '제출', REJECTED: '거절', BLOCKED: '차단', CANCELED: '취소', CANCELLED: '취소', OPEN: '대기', WORKING: '대기', NO_ORDER: '주문 없음' };

  function badge(id, value, className = '') { const node = $(id); node.textContent = value; node.className = `badge ${className}`; }
  function renderAccount(account) {
    const snapshot = account?.snapshot || null;
    const checked = snapshot && account?.status !== 'unavailable';
    const stamp = account?.last_verified_at || snapshot?.updated_at;
    const stampMs = Date.parse(stamp || '');
    const fresh = checked && account?.authoritative === true && account?.is_stale !== true && Number.isFinite(stampMs) && Date.now() - stampMs <= 120000 && stampMs <= Date.now() + 5000;
    $('profit').classList.toggle('account-stale', !fresh);
    badge('account-freshness', fresh ? '실계좌 확인값' : checked ? '마지막 확인값' : '계좌 확인 대기', fresh ? 'good' : 'warn');
    for (const [id, key] of [['pnl-today', 'realized_pnl_today_krw'], ['pnl-unrealized', 'unrealized_pnl_krw']]) {
      const value = checked ? snapshot[key] : null;
      text(id, signed(value)); $(id).className = tone(value);
    }
    text('assets', money(checked ? snapshot.total_asset_krw : null));
    const cash = checked ? snapshot.orderable_cash_by_currency || {} : {};
    text('cash-krw', money(cash.KRW)); text('cash-usd', `USD ${number(cash.USD, 2)}`);
    text('account-note', `${fresh ? '실계좌' : checked ? '보관된 실계좌' : '계좌 미확인'} · ${clock(stamp)}${checked && !fresh ? ' · 현재 잔고와 차이가 있을 수 있습니다.' : ''}`);
    const rows = checked && Array.isArray(account.holdings) ? account.holdings : [];
    text('holding-count', checked ? `${rows.length}종목` : '확인 대기');
    $('holdings').innerHTML = rows.length ? rows.map((row) => `<tr><td><strong>${escape(row.name || row.ticker)}</strong><small>${escape(row.ticker)} · ${escape(row.market || row.market_group || '—')}</small></td><td>${number(row.quantity, 4)}</td><td>${number(row.current_price ?? row.last_price, row.currency === 'USD' ? 2 : 0)} ${escape(row.currency || '')}</td><td class="${tone(row.unrealized_pnl_krw)}">${signed(row.unrealized_pnl_krw)}<small>${numeric(row.unrealized_pnl_rate) === null ? '—' : `${number(row.unrealized_pnl_rate * 100, 2)}%`}</small></td></tr>`).join('') : `<tr><td colspan="4" class="empty">${checked ? '보유 중인 종목이 없습니다.' : '확인된 실계좌 정보가 없습니다.'}</td></tr>`;
  }

  function renderMarkets(markets) {
    $('markets').innerHTML = ['KR', 'US'].map((code) => {
      const row = markets?.[code] || {};
      const phase = String(row.phase || 'UNKNOWN').toUpperCase();
      const allowed = row.allows_new_entry;
      const strategies = Array.isArray(row.strategy_ids) ? row.strategy_ids.map((item) => typeof item === 'object' ? item.name || item.strategy_id || '' : item).filter(Boolean) : [];
      const source = row.source || (Array.isArray(row.sources) ? row.sources.join(' · ') : '') || '데이터 출처 확인 대기';
      const summary = row.reason ? String(row.reason).split(/,\s*/).map((item) => reasons[item] || item).join(' · ') : (allowed === true ? '세션 진입 가능 · 종목별 비용과 리스크 조건 평가' : allowed === false ? '현재 세션에서 신규 진입 대기' : '시장 시간과 주문 가능 여부를 확인하고 있습니다.');
      const phaseLabel = phase.split(/,\s*/).map((item) => phases[item] || item).join(' · ');
      return `<article class="card"><div class="market-heading"><div><p class="market-code">${code} / CASH EQUITY</p><h3>${code === 'KR' ? '국내 시장' : '미국 시장'}</h3></div><span class="badge ${allowed === true ? 'good' : allowed === false ? 'warn' : ''}">${escape(phaseLabel)}</span></div><p class="market-summary">${escape(summary)}</p><dl class="market-metrics"><div><dt>시장 국면</dt><dd>${escape(regimes[String(row.regime || 'unknown').toLowerCase()] || row.regime)}</dd></div><div><dt>시세 경과</dt><dd>${numeric(row.quote_age_seconds) === null ? '확인 대기' : `${number(row.quote_age_seconds, 1)}초`}</dd></div><div><dt>신선한 종목</dt><dd>${numeric(row.healthy_symbols) === null ? '확인 대기' : `${number(row.healthy_symbols)}개`}</dd></div></dl><p class="market-source">${escape(source)}${strategies.length ? `<br>전략 · ${escape(strategies.join(' / '))}` : ''}</p></article>`;
    }).join('');
  }

  function renderTrading(trading) {
    const row = trading || {};
    const pipeline = Array.isArray(row.pipeline) ? row.pipeline : [];
    const first = pipeline.find((item) => item.first_blocker) || pipeline.find((item) => item.state === 'blocked');
    const ready = row.trading_possible === true && row.running === true && row.armed === true;
    const blocker = row.blocker || {};
    let title = ready ? '전략 평가 가능' : blocker.stage ? `${stages[blocker.stage] || blocker.stage} · 확인 필요` : row.running === false ? '거래 엔진 대기' : '진입 상태 확인 대기';
    text('blocker-title', title);
    text('blocker-detail', blocker.detail || first?.detail || (ready ? '관측된 평가 조건을 충족했습니다. 종목별 주문 직전 검증이 남아 있으며, 실제 제출과 체결은 아래 이력에서 확인하세요.' : '실행 상태를 확인할 수 있는 진단 정보가 아직 없습니다.'));
    $('verdict').className = `verdict ${ready ? 'good' : ''}`;
    const modes = { live_trading: '실거래', live: '실거래', paper: '모의거래', shadow: '관측', analysis_only: '분석' };
    badge('execution-mode', `${modes[row.mode] || row.mode || '모드 확인 대기'} · ${row.armed === true ? '주문 권한 확인' : row.armed === false ? '주문 잠금' : '권한 확인 대기'}`, row.armed === true ? 'good' : 'warn');
    $('pipeline').innerHTML = pipeline.map((step, index) => `<li class="${step.state === 'passed' ? 'passed' : 'blocked'} ${step.first_blocker ? 'first-blocker' : ''}" title="${escape(step.detail)}"><span class="stage-index">${String(index + 1).padStart(2, '0')}</span><strong>${escape(step.label || stages[step.stage] || step.stage)}</strong><small>${step.state === 'passed' ? '통과' : step.first_blocker ? '첫 대기 지점' : '조건 미충족'}</small></li>`).join('');
    text('execution-owner', row.selected_strategy ? `${row.selected_symbol || '—'} · ${row.selected_strategy} · ${row.phase || '선택됨'}` : '선택된 실행 전략 없음');
    const orders = Array.isArray(row.orders) ? row.orders : [];
    text('order-count', `${orders.length}건`);
    $('orders').innerHTML = orders.length ? orders.slice().reverse().map((order) => `<tr><td>${escape(clock(order.occurred_at || order.recorded_at || order.updated_at))}</td><td>${escape(order.ticker || order.symbol || '—')}</td><td>${escape(({ BUY: '매수', SELL: '매도' })[String(order.side).toUpperCase()] || order.side || '—')}</td><td>${number(order.ordered_quantity ?? order.quantity, 4)} / ${number(order.filled_quantity, 4)}</td><td>${escape(orderStates[order.order_status || order.status] || order.order_status || order.status || '확인 대기')}</td></tr>`).join('') : '<tr><td colspan="5" class="empty">확인된 최근 주문이 없습니다. 신호 발생은 주문 체결과 다릅니다.</td></tr>';
  }

  function renderRuntime(runtime) {
    const row = runtime || {};
    const model = row.model || {};
    const learning = row.learning || {};
    badge('backend-badge', row.backend || '확인 대기', row.backend ? '' : 'warn');
    const predictionStatus = ({ READY: '진입 예측 조건 통과', WAITING: '진입 예측 조건 대기' })[model.status] || model.status;
    const pairs = [['실제 추론 장치', row.device || row.backend], ['요청한 장치', row.requested_backend], ['추론 지연', numeric(row.latency_ms) === null ? null : `${number(row.latency_ms, 2)} ms`], ['모델 버전', model.version], ['모델 갱신', model.updated_at ? clock(model.updated_at) : null], ['최근 예측 판정', predictionStatus], ['학습 상태', learning.status || (learning.active === true ? '학습 중' : learning.active === false ? '대기' : null)]];
    $('runtime').innerHTML = pairs.map(([label, value]) => `<div><dt>${escape(label)}</dt><dd>${escape(value || '확인 대기')}</dd></div>`).join('');
    text('runtime-note', row.fallback_reason || model.reason || '관측된 실행 장치 기준입니다. 새 모델은 검증 결과에 따라 반영됩니다.');
  }

  async function fetchJson(url) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 8000);
    try {
      const response = await fetch(url, { cache: 'no-store', credentials: 'same-origin', signal: controller.signal });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      if (!payload || typeof payload !== 'object' || Array.isArray(payload)) throw new Error('Invalid dashboard response');
      return payload;
    } finally { window.clearTimeout(timeout); }
  }

  async function refresh() {
    if (state.busy || document.hidden) return;
    state.busy = true; $('refresh').disabled = true;
    try {
      const results = await Promise.allSettled([fetchJson('/api/operations/overview'), fetchJson('/api/account/summary')]);
      state.failures = [];
      results.forEach((result, index) => {
        if (result.status === 'fulfilled') state[index === 0 ? 'operations' : 'account'] = result.value;
        else state.failures.push(index === 0 ? '운영 상태' : '계좌');
      });
      if (results[0].status === 'fulfilled') {
        renderMarkets(state.operations.markets); renderTrading(state.operations.trading); renderRuntime(state.operations.runtime);
        state.lastSuccess = state.operations.generated_at || new Date().toISOString();
      }
      renderAccount(state.failures.includes('계좌') && state.account ? { ...state.account, authoritative: false, is_stale: true } : state.account);
      text('updated', `운영 상태 갱신 ${clock(state.lastSuccess)}`);
      text('connection', state.failures.length ? `${state.failures.join(' · ')} 연결 지연 · 마지막 확인값 표시` : '연결됨 · 5초 간격 확인');
      $('connection').className = state.failures.length ? 'stale' : '';
      if (state.failures.includes('운영 상태')) {
        $('verdict').className = 'verdict';
        text('blocker-title', '운영 상태 연결 지연');
        text('blocker-detail', '마지막 진단을 표시하고 있습니다. 현재 주문 권한과 시장 상태는 확인할 수 없습니다.');
        badge('execution-mode', '연결 지연 · 현재 권한 확인 불가', 'warn');
      }
    } finally { state.busy = false; $('refresh').disabled = false; }
  }
  function schedule() { window.clearTimeout(state.timer); state.timer = window.setTimeout(async () => { await refresh(); schedule(); }, 5000); }
  // Small pure formatting hooks support offline contract tests without a server.
  if (typeof module !== 'undefined' && module.exports) module.exports = { numeric, number, money, escape, renderAccount, renderMarkets, renderTrading, renderRuntime, refresh, state };
  if (typeof document !== 'undefined' && $('refresh')) {
    $('refresh').addEventListener('click', refresh);
    document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
    refresh(); schedule();
  }
})();

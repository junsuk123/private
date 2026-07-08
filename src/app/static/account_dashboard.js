const state = { dashboard: null, range: '1D', market: 'all', query: '' };

const fmtKrw = (value) => `₩${Math.round(Number(value || 0)).toLocaleString('ko-KR')}`;
const fmtMoney = (value, currency = 'KRW') => currency === 'KRW'
  ? fmtKrw(value)
  : `${currency} ${Number(value || 0).toLocaleString('en-US', { maximumFractionDigits: 2 })}`;
const fmtPct = (value) => `${(Number(value || 0) * 100).toFixed(2)}%`;
const clsPnl = (value) => Number(value || 0) >= 0 ? 'positive' : 'negative';

async function fetchJson(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function refreshDashboard() {
  const data = await fetchJson('/api/account/dashboard');
  let trading = null;
  try {
    trading = await fetchJson('/api/realtime-trading/status');
  } catch (_) {
    trading = null;
  }
  let runtime = null;
  try {
    runtime = await fetchJson('/api/realtime/runtime');
  } catch (_) {
    runtime = null;
  }
  state.dashboard = data;
  state.runtime = runtime;
  renderDashboard(data, trading, runtime);
  await refreshHistory();
}

async function refreshHistory() {
  const data = await fetchJson(`/api/account/asset-history?range=${encodeURIComponent(state.range)}`);
  renderAssetChart(data.points || []);
}

function renderDashboard(data, trading, runtime) {
  const snapshot = data.snapshot || {};
  document.getElementById('account-source').textContent = `${snapshot.source || 'unknown'} | updated ${formatTime(snapshot.updated_at)}`;
  const badge = document.getElementById('account-stale-badge');
  badge.textContent = snapshot.is_stale ? `stale ${Math.round(snapshot.stale_seconds || 0)}s` : 'live';
  badge.className = snapshot.is_stale ? 'badge warn' : 'badge';

  renderKpis(snapshot);
  renderProfitability(data.profitability || {}, snapshot, trading);
  renderAllocation(snapshot.asset_allocations || []);
  renderHoldings(mergeHoldingsWithOrders(data.holdings || [], data.holding_orders || []));
  renderCash(data.cash || []);
  renderSystem(snapshot, data.logs || {}, trading, runtime);
  renderDecisionFlow(trading);
  renderTechnical(data.technical || {});
  renderMacroMicro(data.macro_micro || {});
  renderLogs(data.logs || {});
}

function renderMacroMicro(mm) {
  const badge = document.getElementById('macro-micro-badge');
  if (badge) {
    badge.textContent = mm.available ? (mm.market_regime || '-') : '데이터 없음';
    badge.className = mm.available && !mm.blocks_buy ? 'badge' : 'badge warn';
  }
  const macro = document.getElementById('mm-macro');
  if (macro) {
    if (!mm.available) {
      macro.innerHTML = '<div class="tech-card empty">거시 추론 데이터 없음</div>';
    } else {
      const sectors = (mm.sector_rankings || []).slice(0, 5)
        .map((s) => `${s.sector}(${techNum(s.score, 2)})`).join(', ') || '-';
      const rows = [
        ['시장 레짐', mm.market_regime || '-'],
        ['거시 리스크', mm.macro_risk_level || '-'],
        ['거시 신뢰도', techNum(mm.macro_confidence, 2)],
        ['신규매수 차단', mm.blocks_buy ? '예' : '아니오'],
        ['허용 전략', (mm.allowed_micro_strategies || []).join(', ') || '-'],
        ['차단 전략', (mm.blocked_micro_strategies || []).join(', ') || '-'],
        ['후보 종목', (mm.candidate_symbols || []).join(', ') || '-'],
        ['섹터 상위', sectors],
      ];
      macro.innerHTML = `<div class="tech-card ${mm.blocks_buy ? 'rejected' : 'approved'}">`
        + `<div class="tech-detail">${rows.map(([k, v]) => `<span><em>${k}</em>${v}</span>`).join('')}</div></div>`;
    }
  }
  const microContainer = document.getElementById('mm-micro-cards');
  if (microContainer) {
    const rows = mm.micro || [];
    microContainer.innerHTML = rows.length ? rows.map((m) => {
      const buy = m.entry_signal === 'BUY_CANDIDATE';
      const exit = (m.exit_signal && m.exit_signal !== 'NONE');
      const kind = buy ? 'approved' : exit ? 'sell' : 'rejected';
      const detail = [
        ['미시 레짐', m.micro_regime || '-'],
        ['전략', m.selected_strategy || '-'],
        ['진입', m.entry_signal || '-'],
        ['청산', m.exit_signal || '-'],
        ['예상순수익', techNum(m.expected_net_return_bps, 1, 'bps')],
        ['예상청산가', techNum(m.expected_exit_price, 2)],
        ['체결품질', m.execution_quality || '-'],
        ['신뢰도', techNum(m.confidence, 2)],
      ].map(([k, v]) => `<span><em>${k}</em>${v}</span>`).join('');
      return `<div class="tech-card ${kind}"><div class="tech-symbol">${m.symbol || '-'}</div><div class="tech-detail">${detail}</div></div>`;
    }).join('') : '<div class="tech-card empty">미시 추론 결과 없음</div>';
  }
  const rankedContainer = document.getElementById('mm-ranked');
  if (rankedContainer) {
    const ranked = mm.ranked_intents || [];
    rankedContainer.innerHTML = ranked.length ? ranked.map((i) => {
      const kind = i.side === 'BUY' ? 'approved' : 'sell';
      return `<div class="tech-card ${kind}"><div class="tech-symbol">#${i.rank} ${i.side} ${i.symbol}</div>`
        + `<div class="tech-detail"><span><em>전략</em>${i.selected_strategy || '-'}</span>`
        + `<span><em>순수익</em>${techNum(i.expected_net_return_bps, 1, 'bps')}</span>`
        + `<span><em>신뢰도</em>${techNum(i.confidence, 2)}</span></div></div>`;
    }).join('') : '<div class="tech-card empty">랭킹된 의도 없음</div>';
  }
}

const TECH_REJECT_TEXT = {
  below_net_edge: '순수익 부족(비용 차감 후 목표 미달)',
  spread_consumes_alpha: '스프레드가 기대수익을 잠식',
  low_liquidity: '유동성 부족',
  high_volatility: '변동성 위험 과다',
  model_feature_unavailable: '모델/시세 데이터 미가용',
  no_ontology_support: '온톨로지 근거 부족',
  other: '기타',
};

function techNum(value, digits, suffix) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
  return Number(value).toFixed(digits === undefined ? 2 : digits) + (suffix || '');
}

function renderTechnicalCards(containerId, cards, kind) {
  const container = document.getElementById(containerId);
  if (!container) return;
  if (!cards || !cards.length) {
    container.innerHTML = '<div class="tech-card empty">해당 없음</div>';
    return;
  }
  container.innerHTML = cards
    .map((c) => {
      const rows = [
        ['국면', c.regime || '-'],
        ['방법론', c.methodology || '-'],
        ['기대엣지', techNum(c.expected_edge_bps, 1, 'bps')],
        ['기대보유', c.expected_horizon_seconds ? `${c.expected_horizon_seconds}s` : '-'],
        ['예상청산가', techNum(c.expected_exit_price, 2)],
        ['하방위험', techNum(c.downside_risk_bps, 1, 'bps')],
        ['VWAP거리', techNum(c.vwap_distance_bps, 1, 'bps')],
        ['신뢰도', techNum(c.confidence, 2)],
        ['게이트순수익', techNum(c.net_expected_return === null ? null : c.net_expected_return * 10000, 1, 'bps')],
      ];
      const reject = c.reject_reason ? `<div class="tech-reject">${TECH_REJECT_TEXT[c.reject_reason] || c.reject_reason}</div>` : '';
      const explain = c.explanation ? `<div class="tech-explain">${c.explanation}</div>` : '';
      const detail = rows.map(([k, v]) => `<span><em>${k}</em>${v}</span>`).join('');
      return `<div class="tech-card ${kind}"><div class="tech-symbol">${c.symbol || '-'}</div>${reject}<div class="tech-detail">${detail}</div>${explain}</div>`;
    })
    .join('');
}

function renderTechnical(technical) {
  const badge = document.getElementById('technical-badge');
  if (badge) {
    badge.textContent = technical.available ? `${technical.count} 종목` : '데이터 없음';
    badge.className = technical.available ? 'badge' : 'badge warn';
  }
  renderTechnicalCards('tech-buy-approved', technical.buy_approved, 'approved');
  renderTechnicalCards('tech-buy-rejected', technical.buy_rejected, 'rejected');
  renderTechnicalCards('tech-sell-reduce', [...(technical.sell || []), ...(technical.reduce || [])], 'sell');
  renderTechnicalCards('tech-hold', technical.hold, 'hold');
}

// Human-readable hold/rejection reasons. Mirrors app.web._HOLD_REASON_TEXT so the
// account dashboard shows the same explanations as the kiosk.
const REASON_TEXT = {
  HOLD_BELOW_PROFIT_TARGET: '아직 목표 수익 미달 → 보유',
  WIDE_SPREAD: '호가 스프레드가 넓어 매수 보류',
  LOW_LIQUIDITY: '유동성 부족으로 보류',
  INSUFFICIENT_CASH_FOR_ONE_SHARE: '1주 매수 현금 부족',
  FALLBACK_SCORE_BELOW_THRESHOLD: '매수 점수 기준 미달',
  ONTOLOGY_REQUIRED_FOR_MODEL_FALLBACK: '근거 확인 부족으로 보류',
  MODEL_AUXILIARY_ONLY_NEEDS_CONFIRMATION: '모델 단독 매수 불가(근거 필요)',
  MODEL_FEATURE_UNAVAILABLE: '실시간 데이터 부족으로 판단 보류',
  SELL_BELOW_BREAK_EVEN_BLOCKED: '손실 매도 방지(본전 미만)',
  SMALL_ACCOUNT_ONE_SHARE_LOSS_BLOCK: '소액계좌 보호(1주 손실매도 차단)',
  HOLD_UNPROFITABLE_ONTOLOGY_SELL_BLOCKED: '손실권 매도 보류',
  LOSS_EXIT_DISABLED: '손실 청산 비활성',
  MARKET_SESSION_CLOSED: '장 마감',
  MISSING_MARKET_DATA: '시세 없음',
  open_sell_kept: '기존 매도 주문 유지(중복 방지)',
  BELOW_TARGET_NET_RETURN_AFTER_COST: '비용 차감 후 목표 순수익 미달',
  BELOW_BREAK_EVEN_WITH_MARGIN: '본전(마진 포함) 미만 예상 → 매수 보류',
  COST_BURDEN_HIGH: '거래비용 부담 과다',
  SPREAD_TOO_WIDE: '호가 스프레드가 넓어 매수 보류',
  SPREAD_CONSUMES_ALPHA: '스프레드가 기대수익을 잠식',
  LIQUIDITY_TOO_LOW: '유동성 부족으로 보류',
  SLIPPAGE_RISK_HIGH: '슬리피지 위험 과다',
  PROFITABILITY_GATE_REJECTED: '수익성 게이트 거부(순기대수익 부족)',
  RECENT_LOSS_SYMBOL_COOLDOWN: '최근 손실 종목 재매수 대기',
  NO_SELLABLE_QUANTITY: '매도 가능 수량 없음',
  OPEN_ORDER_OR_SETTLEMENT_LOCK: '미체결 주문/결제 잠금',
  NO_ACCOUNT_SNAPSHOT: '계좌 스냅샷 없음',
};

function humanizeReason(code) {
  const raw = String(code || '').trim();
  if (!raw) return '';
  const base = raw.split(':')[0].trim();
  if (REASON_TEXT[base]) return REASON_TEXT[base];
  return base.replaceAll('_', ' ');
}

const fmtOrNa = (value, fn) => (value === null || value === undefined) ? 'n/a' : fn(value);

function renderProfitability(prof, snapshot, trading) {
  const status = (trading && trading.status) ? trading.status : {};
  const summary = status.last_summary || {};
  const armed = summary.live_armed !== undefined ? summary.live_armed : status.buy_enabled;
  const running = Boolean(trading && trading.running);
  const armedBadge = document.getElementById('profitability-armed');
  if (armedBadge) {
    const isArmed = running && Boolean(armed);
    armedBadge.textContent = isArmed ? '라이브 무장' : (running ? '매수 정지' : '대기');
    armedBadge.className = isArmed ? 'badge' : 'badge warn';
  }
  const closedTradeCount = Number(prof.closed_trade_count || 0);
  const zeroTradeState = closedTradeCount === 0;
  const fallbackValue = (value, formatter, fallback = '0') => {
    if (value === null || value === undefined) return fallback;
    return formatter(value);
  };
  const budgetRemaining = summary.daily_loss_budget_remaining_krw;
  const rows = [
    ['오늘 순손익(실현−비용)', fmtKrw(prof.net_after_cost_krw), `실현 ${fmtKrw(prof.realized_pnl_today_krw)} · 비용 ${fmtKrw(prof.trade_cost_krw)}`, clsPnl(prof.net_after_cost_krw)],
    ['승률', zeroTradeState ? '0.00%' : fallbackValue(prof.win_rate, fmtPct, '0.00%'), `${prof.win_count || 0}승 / ${prof.loss_count || 0}패 (${closedTradeCount}건)`],
    ['평균 수익', zeroTradeState ? '₩0' : fallbackValue(prof.avg_win_krw, fmtKrw, '₩0'), zeroTradeState ? '청산 거래 없음' : '이익 거래 평균', 'positive'],
    ['평균 손실', zeroTradeState ? '₩0' : fallbackValue(prof.avg_loss_krw, fmtKrw, '₩0'), zeroTradeState ? '청산 거래 없음' : '손실 거래 평균', 'negative'],
    ['손익비 (Payoff)', zeroTradeState ? '0.00' : fallbackValue(prof.payoff_ratio, (v) => Number(v).toFixed(2), '0.00'), zeroTradeState ? '청산 거래 없음' : '평균수익 / |평균손실|'],
    ['기대값 (Expectancy)', zeroTradeState ? '₩0' : fallbackValue(prof.expectancy_krw, fmtKrw, '₩0'), zeroTradeState ? '청산 거래 없음' : '거래당 기대손익', clsPnl(prof.expectancy_krw)],
    ['거래 비용', fmtKrw(prof.trade_cost_krw), `수수료 ${fmtKrw(prof.fees_krw)} · 세금 ${fmtKrw(prof.tax_krw)}`, 'negative'],
    ['일일 손실 한도 잔여', budgetRemaining === null || budgetRemaining === undefined ? 'n/a' : fmtKrw(budgetRemaining), summary.daily_loss_budget_krw ? `한도 ${fmtKrw(summary.daily_loss_budget_krw)}` : '한도 미설정'],
  ];
  const grid = document.getElementById('profitability-kpis');
  if (grid) {
    grid.innerHTML = rows.map(([label, value, note, className]) => `
      <div class="kpi-card">
        <span>${escapeHtml(label)}</span>
        <strong class="${className || ''}">${escapeHtml(value)}</strong>
        <small>${escapeHtml(note || '')}</small>
      </div>
    `).join('');
  }
  const note = document.getElementById('profitability-note');
  if (note) {
    note.textContent = zeroTradeState
      ? '체결된 청산 거래가 없어 승률·손익비·기대값은 0 기준으로 표시됩니다.'
      : '승률·평균손익·손익비·기대값은 최근 청산 거래(실현손익) 기준입니다.';
  }
}

function renderKpis(snapshot) {
  const settlementCash = Number(snapshot.settlement_cash_krw || 0);
  const cashNote = settlementCash > 0
    ? `외화 ${fmtKrw(snapshot.foreign_cash_krw)} · 결제예정 ${fmtKrw(settlementCash)}`
    : `외화 ${fmtKrw(snapshot.foreign_cash_krw)}`;
  const rows = [
    ['총자산', fmtKrw(snapshot.total_asset_krw), `순자산 ${fmtKrw(snapshot.net_asset_krw)}`],
    ['평가손익', fmtKrw(snapshot.unrealized_pnl_krw), fmtPct(snapshot.total_pnl_rate), clsPnl(snapshot.unrealized_pnl_krw)],
    ['실현손익', fmtKrw(snapshot.realized_pnl_period_krw), '기간 기준', clsPnl(snapshot.realized_pnl_period_krw)],
    ['주문가능 KRW', fmtMoney((snapshot.orderable_cash_by_currency || {}).KRW || snapshot.krw_cash, 'KRW'), '원화'],
    ['주문가능 USD', fmtMoney((snapshot.orderable_cash_by_currency || {}).USD || 0, 'USD'), '외화'],
    ['현금성 자산', fmtKrw(snapshot.cash_equivalent_krw), cashNote],
  ];
  document.getElementById('account-kpis').innerHTML = rows.map(([label, value, note, className]) => `
    <div class="kpi-card">
      <span>${label}</span>
      <strong class="${className || ''}">${value}</strong>
      <small>${note || ''}</small>
    </div>
  `).join('');
}

function renderAllocation(rows) {
  const list = document.getElementById('allocation-list');
  list.innerHTML = rows.map((row) => `
    <div class="allocation-row">
      <span>${row.label}</span>
      <strong>${fmtKrw(row.value_krw)} · ${fmtPct(row.weight)}</strong>
    </div>
  `).join('');
  drawDonut(document.getElementById('allocation-chart'), rows);
}

function renderHoldings(rows) {
  const filtered = rows.filter((row) => {
    const marketOk = state.market === 'all' || row.market_group === state.market;
    const q = state.query.toLowerCase();
    const queryOk = !q || `${row.ticker} ${row.name}`.toLowerCase().includes(q);
    return marketOk && queryOk;
  });
  const body = document.getElementById('holdings-body');
  body.innerHTML = filtered.length ? filtered.map((row) => {
    const hasCost = Number(row.round_trip_cost_rate || 0) > 0;
    const breakEven = hasCost ? fmtMoney(row.break_even_price, row.currency) : 'n/a';
    const estNet = hasCost
      ? `<span class="${clsPnl(row.estimated_net_pnl_krw)}">${fmtKrw(row.estimated_net_pnl_krw)}</span>`
      : 'n/a';
    const orderState = row.order_state ? escapeHtml(row.order_state) : '주문 없음';
    const orderSummary = escapeHtml(row.order_summary || '현재 걸린 주문 없음');
    return `
    <tr>
      <td><strong>${row.ticker}</strong><br><small>${row.name || ''}</small></td>
      <td>${row.market_group === 'domestic' ? '국내' : '해외'}<br><small>${row.exchange || row.market}</small></td>
      <td>${Number(row.quantity || 0).toLocaleString()}</td>
      <td>${fmtMoney(row.average_price, row.currency)}</td>
      <td>${breakEven}</td>
      <td>${fmtMoney(row.current_price, row.currency)}</td>
      <td>${orderState}</td>
      <td>${orderSummary}</td>
      <td>${fmtKrw(row.evaluation_amount_krw)}</td>
      <td class="${clsPnl(row.unrealized_pnl_krw)}">${fmtKrw(row.unrealized_pnl_krw)}</td>
      <td>${estNet}</td>
      <td class="${clsPnl(row.unrealized_pnl_rate)}">${fmtPct(row.unrealized_pnl_rate)}</td>
      <td>${fmtPct(row.weight_of_total_asset)}</td>
      <td>${row.currency}</td>
    </tr>`;
  }).join('') : `<tr class="empty-row"><td colspan="13">현재 보유 종목 없음</td></tr>`;
}

function mergeHoldingsWithOrders(holdings, orders) {
  const orderByTicker = new Map();
  (orders || []).forEach((order) => {
    const ticker = String(order && order.ticker ? order.ticker : '').toUpperCase().trim();
    if (!ticker) return;
    const current = orderByTicker.get(ticker);
    const currentTime = current ? Number(new Date(current.occurred_at || 0)) : -Infinity;
    const nextTime = Number(new Date(order.occurred_at || 0));
    if (!current || nextTime >= currentTime) {
      orderByTicker.set(ticker, order);
    }
  });
  return (holdings || []).map((holding) => {
    const order = orderByTicker.get(String(holding.ticker || '').toUpperCase().trim()) || {};
    return {
      ...holding,
      order_state: order.order_state || order.order_status || '',
      order_status: order.order_status || '',
      order_summary: order.order_summary || '',
      order_id: order.order_id || '',
      filled_quantity: order.filled_quantity || 0,
      occurred_at: order.occurred_at || '',
    };
  });
}

function renderCash(rows) {
  const body = document.getElementById('cash-body');
  body.innerHTML = rows.length ? rows.map((row) => {
    const isSettlement = String(row.currency || '').toUpperCase() === 'KRW_SETTLEMENT';
    const label = isSettlement ? '결제예정' : row.currency;
    const balance = isSettlement ? fmtKrw(row.cash_balance) : fmtMoney(row.cash_balance, row.currency);
    const orderable = isSettlement ? '-' : fmtMoney(row.orderable_amount, row.currency);
    const fx = isSettlement ? '-' : Number(row.fx_rate_to_krw || 0).toLocaleString('ko-KR', { maximumFractionDigits: 4 });
    return `
    <tr>
      <td>${escapeHtml(label)}</td>
      <td>${balance}</td>
      <td>${orderable}</td>
      <td>${fmtKrw(row.krw_equivalent)}</td>
      <td>${fx}</td>
    </tr>`;
  }).join('') : `<tr class="empty-row"><td colspan="5">예수금 정보 수집 중</td></tr>`;
}

function renderSystem(snapshot, logs, trading, runtime) {
  const warnings = snapshot.data_quality_warnings || [];
  const tradingStatus = trading && trading.status ? trading.status : {};
  const accel = (runtime && runtime.acceleration) || runtime || {};
  const eventLlm = (runtime && runtime.event_llm) || {};
  const providerLabel = accel.uses_npu
    ? `NPU (${accel.active_backend || accel.selected_device || 'NPU'})`
    : `CPU fallback${accel.active_backend ? ` (${accel.active_backend})` : ''}`;
  const llmLabel = eventLlm.available
    ? `LLM ${eventLlm.provider || '-'}`
    : (runtime ? `LLM 대기 (${eventLlm.reason || '미구성'})` : 'n/a');
  const items = [
    ['KIS 상태', snapshot.is_live ? '연결됨' : 'fallback'],
    ['자동거래', trading && trading.running ? '실행 중' : '대기'],
    ['자동시작', trading && trading.auto_start ? '켜짐' : '꺼짐'],
    ['추론 장치', runtime ? providerLabel : 'n/a'],
    ['이벤트 LLM', llmLabel],
    ['주문 제출', String((tradingStatus.submitted || 0))],
    ['마지막 계좌 갱신', formatTime(snapshot.updated_at)],
    ['데이터 stale', snapshot.is_stale ? '주의' : '정상'],
    ['데이터 품질 경고', String(warnings.length)],
    ['보유 종목', String((state.dashboard.holdings || []).length)],
    ['로그 오류', logs.last_error ? '있음' : '없음'],
  ];
  const pills = items.map(([label, value]) => `
    <div class="system-pill"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>
  `).join('');
  const warnBlock = warnings.length ? `
    <div class="system-warnings">
      ${warnings.map((w) => `<span class="warn-chip">${escapeHtml(humanizeReason(String(w).split(':')[0]) || String(w))}</span>`).join('')}
    </div>` : '';
  document.getElementById('system-strip').innerHTML = pills + warnBlock;
}

function renderDecisionFlow(trading) {
  const flow = document.getElementById('decision-flow');
  const events = document.getElementById('decision-events');
  const rejections = document.getElementById('decision-rejections');
  const badge = document.getElementById('decision-cycle-badge');
  const liveStrip = document.getElementById('decision-live-strip');
  if (!flow || !events || !rejections || !badge) return;

  const status = trading && trading.status ? trading.status : {};
  const summary = status.last_summary || {};
  const diagnostics = trading && trading.decision_diagnostics ? trading.decision_diagnostics : {};
  const policyState = diagnostics.policy_state || {};
  const modelHealth = policyState.model_health || {};
  const profitabilityDecision = diagnostics.profitability_decision || {};
  const running = Boolean(trading && trading.running);
  badge.textContent = running
    ? `cycle ${Number(status.cycles || 0).toLocaleString('ko-KR')}`
    : 'stopped';
  badge.className = running ? 'badge' : 'badge warn';

  const buyEvaluated = Number(summary.buy_evaluated || 0);
  const sellEvaluated = Number(summary.sell_evaluated || 0);
  const submitted = Number(summary.submitted || 0);
  const amended = Number(summary.amended || 0);
  const blocked = Number(summary.blocked || 0);
  const errors = Number(summary.errors || 0);
  const buyRejected = Number(summary.buy_rejected || 0);
  const sellRejected = Number(summary.sell_rejected || 0);
  const totalEvaluated = buyEvaluated + sellEvaluated;
  const ignoredSymbols = summary.ignored_symbols || [];
  if (liveStrip) {
    const chips = [
      ['마지막 사이클', status.last_cycle_at ? formatTime(status.last_cycle_at) : '-'],
      ['상태', running ? '실행 중' : '정지'],
      ['매수 평가/보류', `${buyEvaluated}/${buyRejected}`],
      ['매도 평가/보류', `${sellEvaluated}/${sellRejected}`],
      ['주문 제출', String(submitted)],
      ['차단/오류', `${blocked}/${errors}`],
      ['무시 종목', ignoredSymbols.length ? ignoredSymbols.join(', ') : '-'],
      ['무시 건수', String(Number(summary.skipped_ignored || 0))],
      ['현재 평가', policyState.symbol ? `${policyState.symbol} ${profitabilityDecision.action || ''}` : '-'],
      ['모델/시세', `${modelHealth.status || '-'} / ${diagnostics.quote_refresh_status || '-'}`],
      ['마지막 사유', summary.reason || status.last_reason || '-'],
    ];
    liveStrip.innerHTML = chips.map(([label, value]) => `
      <div class="decision-live-chip">
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(value)}</strong>
      </div>
    `).join('');
  }
  const stages = [
    ['계좌', running ? '자동 실행' : '대기', status.last_cycle_at ? formatTime(status.last_cycle_at) : '-'],
    ['매도 평가', `${sellEvaluated}건`, `${Number(summary.sell_submitted || 0)} 제출 · ${sellRejected} 보류`],
    ['매수 평가', `${buyEvaluated}건`, `${Number(summary.buy_submitted || 0)} 제출 · ${buyRejected} 보류`],
    ['주문 처리', `${submitted} 제출`, `${amended} 정정 · ${blocked} 차단`],
    ['오류', `${errors}건`, summary.reason || status.last_reason || '정상'],
  ];
  flow.innerHTML = stages.map(([label, value, note], index) => {
    const ratio = index === 0 ? (running ? 1 : 0) : Math.min(1, Number.parseFloat(String(value)) / Math.max(1, totalEvaluated));
    const cls = errors && label === '오류' ? 'danger' : submitted && label === '주문 처리' ? 'active' : '';
    return `
      <div class="decision-stage ${cls}">
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(value)}</strong>
        <small>${escapeHtml(note)}</small>
        <div class="decision-meter"><i style="width:${Math.max(8, Math.round(ratio * 100))}%"></i></div>
      </div>
    `;
  }).join('');

  const recentEvents = (status.recent_events || []).slice(0, 12);
  events.innerHTML = recentEvents.length ? recentEvents.map((event) => {
    const kind = String(event.kind || '');
    const outcome = String(event.outcome || event.status || '');
    const rawReason = String(event.reason || event.detail || '');
    const reason = rawReason
      ? rawReason.split(';').filter(Boolean).map(humanizeReason).join(' · ')
      : formatTime(event.at);
    const cls = outcome.includes('error') || outcome.includes('blocked') ? 'danger' : outcome.includes('submitted') ? 'active' : '';
    return `
      <div class="decision-event ${cls}">
        <div><strong>${escapeHtml(kind)} ${escapeHtml(event.symbol || '-')}</strong><span>${escapeHtml(outcome || '-')}</span></div>
        <small>${escapeHtml(reason)}</small>
      </div>
    `;
  }).join('') : `<div class="decision-empty">최근 실행 판단 없음</div>`;

  const rows = (summary.rejections || []).slice(0, 12);
  rejections.innerHTML = rows.length ? rows.map((row) => {
    const codes = (row.reason_codes || []).map((code) => String(code));
    const primaryHuman = humanizeReason(codes[0]) || '보류';
    const rawCodes = codes.join(' · ');
    const p = row.profitability || {};
    const chips = [];
    if (p.expected_exit_price !== undefined && p.expected_exit_price !== null) chips.push(`예상청산 ${Number(p.expected_exit_price).toLocaleString()}`);
    if (p.break_even_exit_price !== undefined && p.break_even_exit_price !== null) chips.push(`손익분기 ${Number(p.break_even_exit_price).toLocaleString()}`);
    if (p.net_expected_return !== undefined && p.net_expected_return !== null) chips.push(`순기대 ${fmtPct(p.net_expected_return)}`);
    if (p.required_min_net_return !== undefined && p.required_min_net_return !== null) chips.push(`요구 ${fmtPct(p.required_min_net_return)}`);
    if (p.all_in_cost_rate !== undefined && p.all_in_cost_rate !== null) chips.push(`비용 ${fmtPct(p.all_in_cost_rate)}`);
    if (p.spread_rate !== undefined && p.spread_rate !== null) chips.push(`스프레드 ${fmtPct(p.spread_rate)}`);
    if (p.liquidity_score !== undefined && p.liquidity_score !== null) chips.push(`유동성 ${Number(p.liquidity_score).toFixed(2)}`);
    const chipHtml = chips.length ? `<div class="rejection-metrics">${chips.map((c) => `<span>${escapeHtml(c)}</span>`).join('')}</div>` : '';
    return `
      <div class="decision-rejection">
        <strong>${escapeHtml(row.side || '-')} ${escapeHtml(row.symbol || '-')}</strong>
        <span>${escapeHtml(primaryHuman)}</span>
        ${chipHtml}
        <small>${escapeHtml(rawCodes)}</small>
      </div>
    `;
  }).join('') : `<div class="decision-empty">최근 보류 사유 없음</div>`;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function renderLogs(logs) {
  document.getElementById('account-logs').textContent = JSON.stringify(logs, null, 2);
}

function drawDonut(canvas, rows) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const total = rows.reduce((sum, row) => sum + Math.max(0, Number(row.value_krw || 0)), 0);
  if (!total) {
    drawEmpty(ctx, w, h, '자산 배분 수집 중');
    return;
  }
  const colors = ['#176b87', '#8f5f2a', '#2e7d5b', '#7a5cbd', '#8a94a6'];
  let start = -Math.PI / 2;
  rows.forEach((row, index) => {
    const value = Math.max(0, Number(row.value_krw || 0));
    const angle = (value / total) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(w / 2, h / 2);
    ctx.arc(w / 2, h / 2, Math.min(w, h) * 0.38, start, start + angle);
    ctx.fillStyle = colors[index % colors.length];
    ctx.fill();
    start += angle;
  });
  ctx.beginPath();
  ctx.arc(w / 2, h / 2, Math.min(w, h) * 0.20, 0, Math.PI * 2);
  ctx.fillStyle = '#fbfcfe';
  ctx.fill();
  ctx.fillStyle = '#132238';
  ctx.textAlign = 'center';
  ctx.font = '600 15px Segoe UI';
  ctx.fillText('자산 배분', w / 2, h / 2 + 5);
}

function renderAssetChart(points) {
  const canvas = document.getElementById('asset-chart');
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!points.length) {
    drawEmpty(ctx, w, h, '이력을 수집 중');
    return;
  }
  const values = points.map((p) => Number(p.total_asset_krw || 0));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = 34;
  ctx.strokeStyle = '#d7dee8';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, h - pad);
  ctx.lineTo(w - pad, h - pad);
  ctx.stroke();
  ctx.strokeStyle = '#176b87';
  ctx.lineWidth = 3;
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = pad + (index / Math.max(1, values.length - 1)) * (w - pad * 2);
    const y = h - pad - ((value - min) / Math.max(1, max - min)) * (h - pad * 2);
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = '#667085';
  ctx.font = '12px Segoe UI';
  ctx.fillText(fmtKrw(max), pad + 4, pad + 12);
  ctx.fillText(fmtKrw(min), pad + 4, h - pad - 6);
}

function drawEmpty(ctx, w, h, text) {
  ctx.fillStyle = '#667085';
  ctx.textAlign = 'center';
  ctx.font = '15px Segoe UI';
  ctx.fillText(text, w / 2, h / 2);
}

function formatTime(value) {
  if (!value) return '-';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleString('ko-KR', { hour12: false });
}

async function terminateProgram() {
  const button = document.getElementById('account-terminate');
  const source = document.getElementById('account-source');
  if (button) button.disabled = true;
  if (source) source.textContent = 'Termination requested: BUY disabled, submitting profit-seeking SELL orders...';
  try {
    const data = await fetchJson('/api/live-trading/terminate?shutdown=true');
    if (source) {
      source.textContent = data.ok
        ? `Termination complete: SELL orders ${Number(data.submitted_sell_orders || 0)}, server shutdown scheduled`
        : `Termination blocked: ${data.message || data.status || 'unknown'}`;
    }
    if (data.ok) {
      setTimeout(() => {
        window.open('', '_self');
        window.close();
      }, 800);
    } else if (button) {
      button.disabled = false;
    }
  } catch (error) {
    if (source) source.textContent = `Termination failed: ${error.message}`;
    if (button) button.disabled = false;
  }
}

document.getElementById('account-refresh').addEventListener('click', refreshDashboard);
const terminateButton = document.getElementById('account-terminate');
if (terminateButton) terminateButton.addEventListener('click', terminateProgram);
document.getElementById('holding-search').addEventListener('input', (event) => {
  state.query = event.target.value || '';
  renderHoldings((state.dashboard || {}).holdings || []);
});
document.getElementById('holding-market').addEventListener('change', (event) => {
  state.market = event.target.value;
  renderHoldings((state.dashboard || {}).holdings || []);
});
document.querySelectorAll('#history-range button').forEach((button) => {
  button.addEventListener('click', async () => {
    state.range = button.dataset.range || '1D';
    document.querySelectorAll('#history-range button').forEach((item) => item.classList.toggle('active', item === button));
    await refreshHistory();
  });
});
document.querySelector('#history-range button[data-range="1D"]').classList.add('active');

refreshDashboard().catch((error) => {
  document.getElementById('account-logs').textContent = `dashboard load failed: ${error.message}`;
});
setInterval(() => refreshDashboard().catch(() => {}), 5000);

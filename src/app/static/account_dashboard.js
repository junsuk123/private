const state = { dashboard: null, refactor: null, range: '1D', market: 'all', query: '' };

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

async function postJson(url) {
  const response = await fetch(url, { method: 'POST', cache: 'no-store' });
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
  let refactor = null;
  try {
    refactor = await fetchJson('/api/refactor/dashboard');
  } catch (_) {
    refactor = null;
  }
  let blockade = null;
  try {
    blockade = await fetchJson('/api/realtime-trading/entry-blockade');
  } catch (_) {
    blockade = null;
  }
  let shortLadder = null;
  try {
    shortLadder = await fetchJson('/api/short-strategies/status');
  } catch (_) {
    shortLadder = null;
  }
  let directional = null;
  try {
    directional = await fetchJson('/api/directional-bandit/evaluations');
  } catch (_) {
    directional = null;
  }
  state.dashboard = data;
  state.runtime = runtime;
  state.refactor = refactor;
  state.blockade = blockade;
  state.shortLadder = shortLadder;
  state.directional = directional;
  renderDashboard(data, trading, runtime);
  renderRefactorDashboard(refactor);
  renderEntryBlockade(blockade);
  renderShortLadder(shortLadder, directional);
  await refreshHistory();
}

// Deployment states, weakest first. The ladder is rendered as a fixed track rather
// than just the current label, so an operator can see BOTH where an arm is and how
// far it still has to go — a bare "SHADOW" badge reads as a failure when it is in
// fact the correct starting state.
const SHORT_LADDER_STATES = ['SHADOW', 'LIVE_PROBE', 'LIVE_LIMITED', 'LIVE_FULL'];
const SHORT_STATE_LABELS = {
  DISABLED: '비활성',
  SHADOW: '섀도우',
  LIVE_PROBE: '실거래 탐침',
  LIVE_LIMITED: '실거래 제한',
  LIVE_FULL: '실거래 정상',
  SUSPENDED: '중단',
};
// Only the codes an operator can act on differently. Everything else falls through
// to the raw code, which is better than a wrong friendly name.
const SHORT_REASON_LABELS = {
  SHORT_PROMOTION_SAMPLE_INSUFFICIENT: '표본 부족',
  SHORT_PROMOTION_TRADING_DAYS_INSUFFICIENT: '거래일 부족',
  SHORT_PROMOTION_SYMBOL_BREADTH_INSUFFICIENT: '종목 수 부족',
  SHORT_CONFIDENCE_BELOW_THRESHOLD: '신뢰도 미달',
  SHORT_CONSERVATIVE_EDGE_NON_POSITIVE: '보수적 엣지 미달',
  SHORT_COST_COVERAGE_INSUFFICIENT: '비용 대비 엣지 부족',
  SHORT_BORROW_AVAILABILITY_RATE_LOW: '대주 가용률 낮음',
  SHORT_RESCUE_RATE_INSUFFICIENT: '숏 기여도 부족',
  SHORT_PROMOTION_HOLDOUT_NOT_PASSED: 'holdout 미통과',
  SHORT_MODEL_NOT_CALIBRATED: '모델 보정 미완',
  SHORT_CALIBRATION_ERROR_HIGH: '예측 오차 큼',
  SHORT_SLIPPAGE_ERROR_HIGH: '슬리피지 오차 큼',
  SHORT_PROFIT_FACTOR_INSUFFICIENT: 'profit factor 미달',
  SHORT_DRAWDOWN_EXCEEDED: '낙폭 초과',
  SHORT_LOSS_STREAK_EXCEEDED: '연속 손실 초과',
  SHORT_DEPLOYMENT_SUSPENDED: '중단됨',
  SHORT_STRATEGY_SHADOW_ONLY: '섀도우 전용',
  STRATEGY_DEPLOYMENT_SHADOW_ONLY: '실적 미달 · 섀도우 강등',
};

function shortReasonLabel(code) {
  return SHORT_REASON_LABELS[code] || code;
}

function renderShortLadder(payload, directional) {
  const grid = document.getElementById('short-ladder-arms');
  const headline = document.getElementById('short-ladder-headline');
  const verdict = document.getElementById('short-ladder-verdict');
  if (!grid || !headline || !verdict) return;
  if (!payload || payload.ok === false) {
    headline.textContent = '숏 배포 상태를 불러오지 못했습니다.';
    verdict.textContent = '-';
    verdict.className = 'badge neutral';
    grid.innerHTML = '';
    return;
  }
  const arms = Array.isArray(payload.arms) ? payload.arms : [];
  const authorized = arms.filter((arm) => arm.submits_orders);
  if (!arms.length) {
    verdict.textContent = '숏 없음';
    verdict.className = 'badge neutral';
    headline.textContent = '활성화된 숏 전략이 없습니다.';
  } else if (authorized.length) {
    // Deliberately styled as a WARNING, not a success. A live short is the state that
    // deserves attention, not the state that deserves a green tick.
    verdict.textContent = `실주문 ${authorized.length}개`;
    verdict.className = 'badge blocked';
    headline.textContent = `${authorized.length}개 arm이 실주문 권한을 가지고 있습니다.`;
  } else {
    verdict.textContent = '실주문 0개';
    verdict.className = 'badge';
    headline.textContent = `${arms.length}개 arm 전부 섀도우 — 실주문 권한이 없습니다. 이것이 정상 상태입니다.`;
  }

  grid.innerHTML = '';
  arms.forEach((arm) => {
    const card = document.createElement('div');
    card.className = `short-arm ${arm.submits_orders ? 'live' : 'shadow'}`;

    const name = document.createElement('p');
    name.className = 'short-arm-name';
    name.textContent = arm.strategy_id || arm.strategy_key || '-';
    card.appendChild(name);

    const track = document.createElement('div');
    track.className = 'short-arm-track';
    const current = String(arm.state || 'SHADOW');
    const currentIndex = SHORT_LADDER_STATES.indexOf(current);
    if (current === 'SUSPENDED') {
      const step = document.createElement('span');
      step.className = 'short-step suspended';
      step.textContent = SHORT_STATE_LABELS.SUSPENDED;
      track.appendChild(step);
    } else {
      SHORT_LADDER_STATES.forEach((stateName, index) => {
        const step = document.createElement('span');
        const reached = currentIndex >= 0 && index <= currentIndex;
        step.className = `short-step${reached ? ' reached' : ''}${
          index === currentIndex ? ' current' : ''
        }`;
        step.textContent = SHORT_STATE_LABELS[stateName] || stateName;
        track.appendChild(step);
      });
    }
    card.appendChild(track);

    const meta = document.createElement('p');
    meta.className = 'short-arm-meta';
    const score = typeof arm.confidence_score === 'number' ? arm.confidence_score.toFixed(3) : '-';
    const passes = arm.consecutive_passes ?? 0;
    const required = arm.required_consecutive_cycles ?? '-';
    meta.textContent = `신뢰도 ${score} · 연속통과 ${passes}/${required}`;
    card.appendChild(meta);

    const remaining = Array.isArray(arm.remaining_conditions) ? arm.remaining_conditions : [];
    if (remaining.length) {
      const list = document.createElement('p');
      list.className = 'short-arm-blockers';
      // Cap the list: a brand-new arm fails every gate, and rendering fifteen chips
      // makes the panel unreadable without telling the operator anything more.
      const shown = remaining.slice(0, 4).map(shortReasonLabel).join(' · ');
      const extra = remaining.length > 4 ? ` 외 ${remaining.length - 4}건` : '';
      list.textContent = `남은 조건: ${shown}${extra}`;
      card.appendChild(list);
    }
    grid.appendChild(card);
  });

  const health = document.getElementById('short-borrow-health');
  if (health) {
    const borrow = payload.borrow_health || {};
    const lookups = borrow.lookup_count ?? 0;
    if (!lookups) {
      // "We asked nothing" must not read as "nothing was available".
      health.textContent = '대주 데스크: 최근 조회 없음 (가용률 미측정)';
    } else {
      const rate = typeof borrow.availability_rate === 'number'
        ? `${(borrow.availability_rate * 100).toFixed(0)}%`
        : '-';
      health.textContent = `대주 데스크: 조회 ${lookups}건 · 가용률 ${rate} · 종목 ${borrow.distinct_symbols ?? 0}개`;
    }
  }

  const compare = document.getElementById('short-direction-compare');
  if (compare) {
    const cmp = (directional && directional.directional_comparison) || {};
    const long = cmp.best_long_conservative_edge_bps;
    const short = cmp.best_short_conservative_edge_bps;
    const fmt = (value) => (typeof value === 'number' ? `${value.toFixed(1)}bps` : '-');
    let note = `방향 비교: LONG ${fmt(long)} · SHORT ${fmt(short)}`;
    if (cmp.short_rescued) note += ' · 숏이 NO_TRADE를 구제했을 상황';
    else if (cmp.both_directions_negative) note += ' · 양방향 모두 음수';
    compare.textContent = note;
  }
}

// Ordered "why is nothing trading" chain. The FIRST unmet link is the answer;
// everything after it is unreachable, so it is rendered muted rather than as an
// additional failure. Showing every layer's verdict at once is the point: the old
// dashboard surfaced one reason code from whichever layer failed last, which named
// the GNN for 11,614 consecutive cycles while the market session was the blocker.
const BLOCKADE_STAGE_LABELS = {
  engine_running: '엔진 실행',
  live_armed: '라이브 무장',
  market_session: '시장 세션',
  buy_candidates: '매수 후보',
  micro_buy_intents: '마이크로 전략',
  strategy_election: '전략 선택',
  position: '포지션',
};

function renderEntryBlockade(payload) {
  const list = document.getElementById('blockade-chain');
  const headline = document.getElementById('blockade-headline');
  const verdict = document.getElementById('blockade-verdict');
  if (!list || !headline || !verdict) return;
  if (!payload || payload.ok === false) {
    headline.textContent = '진단을 불러오지 못했습니다.';
    verdict.textContent = '-';
    verdict.className = 'badge neutral';
    list.innerHTML = '';
    return;
  }
  const chain = Array.isArray(payload.chain) ? payload.chain : [];
  const blockedAt = chain.findIndex((link) => !link.ok);
  if (payload.trading_possible) {
    verdict.textContent = '진입 가능';
    verdict.className = 'badge';
    headline.textContent = '모든 단계 통과 — 진입을 막는 요인이 없습니다.';
  } else {
    const label = BLOCKADE_STAGE_LABELS[payload.blocking_stage] || payload.blocking_stage || '알 수 없음';
    verdict.textContent = `차단: ${label}`;
    verdict.className = 'badge blocked';
    headline.textContent = payload.blocking_detail || '';
  }
  list.innerHTML = '';
  chain.forEach((link, index) => {
    const item = document.createElement('li');
    const unreachable = blockedAt >= 0 && index > blockedAt;
    item.className = `blockade-step ${link.ok ? 'pass' : 'fail'}${unreachable ? ' unreachable' : ''}`;
    const mark = document.createElement('span');
    mark.className = 'blockade-mark';
    mark.textContent = unreachable ? '·' : link.ok ? '✓' : '✕';
    const body = document.createElement('div');
    const title = document.createElement('p');
    title.className = 'blockade-stage';
    title.textContent = BLOCKADE_STAGE_LABELS[link.stage] || link.stage;
    const detail = document.createElement('p');
    detail.className = 'blockade-detail';
    detail.textContent = unreachable ? '앞 단계에서 막혀 평가되지 않음' : (link.detail || '');
    body.appendChild(title);
    body.appendChild(detail);
    const extra = blockadeExtra(link);
    if (extra) body.appendChild(extra);
    item.appendChild(mark);
    item.appendChild(body);
    list.appendChild(item);
  });
}

function blockadeExtra(link) {
  const data = link.data || {};
  const parts = [];
  if (link.stage === 'market_session' && data.scanned_groups) {
    Object.entries(data.scanned_groups).forEach(([group, info]) => {
      parts.push(`${group}: ${info.phase}${info.allows_new_entry ? ' (진입 가능)' : ''}`);
    });
    if (data.extended_hours_entry_enabled) parts.push('시간외 진입 허용됨');
  }
  if (link.stage === 'buy_candidates' && Array.isArray(data.sample) && data.sample.length) {
    parts.push(data.sample.join(', '));
  }
  if (link.stage === 'micro_buy_intents' && Array.isArray(data.blocking_reason_codes)) {
    const counts = data.reason_code_counts || {};
    parts.push(...data.blocking_reason_codes.map((code) => (
      counts[code] > 1 ? `${code} ×${counts[code]}` : code
    )));
    if (Array.isArray(data.hard_blocked_symbols) && data.hard_blocked_symbols.length) {
      parts.push(`하드 차단: ${data.hard_blocked_symbols.join(', ')}`);
    }
  }
  if (link.stage === 'strategy_election') {
    if (typeof data.algorithm_evaluated_count === 'number') {
      parts.push(`실제 트리거 ${data.algorithm_triggered_count || 0}/${data.algorithm_evaluated_count}`);
    }
    if (data.best_pair) {
      parts.push(`최상위 ${data.best_pair.symbol}×${data.best_pair.arm}`);
    }
    if (typeof data.change_point_probability === 'number') {
      parts.push(`변화점 ${(data.change_point_probability * 100).toFixed(1)}%`);
    }
    if (typeof data.conservative_edge_bps === 'number') {
      // The same field carries two different quantities. Under the bandit it is
      // a pessimistic lower bound; under GNN-direct election it is the model's
      // raw forward edge with no shrinkage at all. Labelling both "보수적" would
      // make an unhedged number read as a conservative one.
      const direct = (data.reason_codes || []).includes('GNN_DIRECT_ELECTION');
      const label = direct ? 'GNN 예측 엣지(무보정)' : '보수적 엣지';
      parts.push(`${label} ${data.conservative_edge_bps.toFixed(1)}bp`);
    }
    if (data.is_exploration) parts.push('탐색 진입(최소 비중)');
    (data.reason_codes || []).forEach((code) => parts.push(BLOCKADE_REASON_LABELS[code] || code));
    Object.entries(data.algorithm_rejection_counts || {})
      .sort((a, b) => b[1] - a[1])
      .slice(0, 2)
      .forEach(([code, count]) => parts.push(`${code} ×${count}`));
    if (data.session_last_reason) parts.push(data.session_last_reason);
  }
  if (link.stage === 'position' && data.last_reason) parts.push(data.last_reason);
  if (!parts.length) return null;
  const wrap = document.createElement('div');
  wrap.className = 'blockade-tags';
  parts.slice(0, 10).forEach((text) => {
    const tag = document.createElement('span');
    tag.className = 'blockade-tag';
    tag.textContent = text;
    wrap.appendChild(tag);
  });
  return wrap;
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
  renderMacroMicroVisual(data.macro_micro || {});
  renderLogs(data.logs || {});
}

function renderRefactorDashboard(data) {
  const modeBadge = document.getElementById('refactor-mode-badge');
  const deviceBadge = document.getElementById('refactor-device-badge');
  const safety = document.getElementById('refactor-safety');
  const kpis = document.getElementById('refactor-kpis');
  const pipeline = document.getElementById('refactor-pipeline');
  const gates = document.getElementById('refactor-promotion-gates');
  const gateCount = document.getElementById('refactor-gate-count');
  const positions = document.getElementById('refactor-owner-positions');
  const ownerCount = document.getElementById('refactor-owner-count');
  const evaluation = document.getElementById('refactor-strategy-evaluation');
  const evaluationStatus = document.getElementById('refactor-eval-status');
  const shadow = document.getElementById('refactor-shadow-comparison');
  const shadowStatus = document.getElementById('refactor-shadow-status');
  if (!modeBadge || !data || !Object.keys(data).length) {
    if (modeBadge) {
      modeBadge.textContent = '상태 없음';
      modeBadge.className = 'badge blocked';
    }
    if (safety) safety.innerHTML = '<strong>진단 불가</strong><span>리팩터링 상태 API를 불러오지 못했습니다.</span>';
    return;
  }

  const mode = String(data.mode || 'invalid').toUpperCase();
  modeBadge.textContent = `${mode} · ${data.live_order_capable ? '주문 가능' : '주문 차단'}`;
  modeBadge.className = data.live_order_capable ? 'badge' : 'badge blocked';
  const devices = data.devices || {};
  deviceBadge.textContent = devices.selected || 'CPU';
  deviceBadge.className = 'badge neutral';

  const failedGates = (data.promotion_gates || []).filter((gate) => !gate.passed);
  if (safety) {
    safety.className = data.live_order_capable ? 'refactor-safety safe' : 'refactor-safety';
    safety.innerHTML = data.live_order_capable
      ? '<strong>실행 준비</strong><span>모든 승격 게이트가 통과했고 제한된 주문 경계가 활성화되었습니다.</span>'
      : `<strong>안전 차단</strong><span>실주문 비활성 · ${failedGates.length}개 승격 게이트 미통과 · NoTrade 우선</span>`;
  }

  const evalData = data.evaluation || {};
  const lifecycle = data.lifecycle || {};
  const cpu = devices.cpu || {};
  const npu = devices.npu || {};
  const kpiRows = [
    ['운영 모드', mode, data.broker_submission_enabled ? 'broker 제출 켜짐' : 'broker 제출 꺼짐'],
    ['전략 소유 포지션', String(lifecycle.open_positions || 0), `인스턴스 ${lifecycle.instances || 0}`],
    ['반사실 라벨', Number(evalData.strategy_labels || 0).toLocaleString('ko-KR'), `시점 ${Number(evalData.snapshots || 0).toLocaleString('ko-KR')}`],
    ['외표본 선택', String(evalData.selected_trades || 0), `관측 ${Number(evalData.walk_forward_observations || 0).toLocaleString('ko-KR')}`],
    ['추론 장치', devices.selected || 'CPU', `CPU p95 ${formatMs(cpu.p95_ms)} · NPU p95 ${formatMs(npu.p95_ms)}`],
  ];
  if (kpis) {
    kpis.innerHTML = kpiRows.map(([label, value, note]) => `
      <div class="refactor-metric">
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(value)}</strong>
        <small>${escapeHtml(note)}</small>
      </div>
    `).join('');
  }

  if (pipeline) {
    pipeline.innerHTML = (data.pipeline || []).map((node, index) => `
      <div class="pipeline-node ${node.active ? 'active' : 'inactive'}">
        <span class="pipeline-index">${index + 1}</span>
        <strong>${escapeHtml(node.label || '-')}</strong>
        <small>${escapeHtml(node.detail || '-')}</small>
        <span class="pipeline-state">${node.active ? 'ACTIVE' : 'STANDBY'}</span>
      </div>
    `).join('');
  }

  const promotionGates = data.promotion_gates || [];
  if (gateCount) gateCount.textContent = `${promotionGates.filter((gate) => gate.passed).length}/${promotionGates.length} 통과`;
  if (gates) {
    gates.innerHTML = promotionGates.map((gate) => `
      <div class="promotion-gate ${gate.passed ? 'pass' : 'fail'}" title="${escapeHtml(gate.reason || '')}">
        <i class="gate-dot"></i>
        <strong>${escapeHtml(gate.label || '-')}</strong>
        <small>${escapeHtml(gate.reason || '-')}</small>
        <em>${escapeHtml(gate.value || (gate.passed ? 'PASS' : 'BLOCK'))}</em>
      </div>
    `).join('');
  }

  const owned = lifecycle.positions || [];
  if (ownerCount) ownerCount.textContent = `${owned.length}개`;
  if (positions) {
    positions.innerHTML = owned.length ? owned.map((position) => `
      <div class="owner-row">
        <strong>${escapeHtml(position.symbol || '-')}</strong>
        <span title="${escapeHtml(position.strategy_instance_id || '')}">${escapeHtml(position.strategy_id || '-')} · ${escapeHtml(position.strategy_instance_id || '-')}</span>
        <small>${Number(position.quantity || 0).toLocaleString('ko-KR')}주</small>
      </div>
    `).join('') : '<div class="owner-empty">열린 전략 소유 포지션이 없습니다.<br>새 경로는 현재 주문을 생성하지 않습니다.</div>';
  }

  const strategyNames = {
    intraday_momentum: '장중 모멘텀',
    breakout_volume: '거래량 돌파',
    vwap_mean_reversion: 'VWAP 평균회귀',
    bar_confirmed_vwap_recovery: '1분봉 확인 VWAP 회복',
    liquidity_shock_reversal: '유동성 충격 반전',
    event_momentum: '이벤트 모멘텀',
    cross_sectional_relative_strength: '횡단면 상대강도',
    gap_context: '갭 컨텍스트',
  };
  const metrics = evalData.strategy_metrics || {};
  if (evaluationStatus) {
    evaluationStatus.textContent = `${evalData.status || 'NO REPORT'} · ${evalData.dates || 0}일`;
  }
  if (evaluation) {
    evaluation.innerHTML = Object.entries(strategyNames).map(([id, name]) => {
      const metric = metrics[id] || {};
      const triggered = Number(metric.triggered || 0);
      const filled = Number(metric.filled || 0);
      const rate = metric.fill_rate_when_triggered;
      const net = metric.mean_net_return_bps_when_filled;
      return `
        <div class="strategy-row ${triggered ? '' : 'disabled'}">
          <strong title="${escapeHtml(id)}">${escapeHtml(name)}</strong>
          <span>발동 ${triggered.toLocaleString('ko-KR')}</span>
          <span>체결 ${filled.toLocaleString('ko-KR')}</span>
          <div class="strategy-bar" title="평균 순수익 ${formatBps(net)}"><i style="width:${rate === null || rate === undefined ? 0 : Math.max(2, Math.min(100, Number(rate) * 100))}%"></i></div>
        </div>
      `;
    }).join('');
  }

  const shadowData = data.shadow || {};
  if (shadowStatus) shadowStatus.textContent = shadowData.available ? `${shadowData.samples || 0} samples` : '대기';
  if (shadow) {
    if (!shadowData.available) {
      shadow.innerHTML = '<div class="shadow-empty">Shadow 비교 로그가 아직 없습니다.<br>주문 없이 legacy·ontology·CPU/NPU 결정을 비교합니다.</div>';
    } else {
      const latest = shadowData.latest || {};
      const decisions = latest.decisions || [];
      shadow.innerHTML = `
        <div class="shadow-summary">
          <div class="shadow-tile"><span>행동 일치율</span><strong>${formatRatio(shadowData.action_agreement_rate)}</strong></div>
          <div class="shadow-tile"><span>전략 일치율</span><strong>${formatRatio(shadowData.strategy_agreement_rate)}</strong></div>
        </div>
        <div class="shadow-decisions">
          ${decisions.map((decision) => `
            <div class="shadow-decision">
              <strong>${escapeHtml(decision.path || '-')}</strong>
              <span>${escapeHtml(decision.action || '-')}</span>
              <span>${escapeHtml(decision.strategy_id || 'NoTrade')}</span>
            </div>
          `).join('')}
        </div>
      `;
    }
  }
}

function formatMs(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(2)}ms` : '-';
}

function formatBps(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(1)}bps` : '-';
}

function formatRatio(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : '-';
}

function renderMacroMicro(mm) {
  const badge = document.getElementById('macro-micro-badge');
  if (badge) {
    badge.textContent = mm.available ? (mm.market_regime || '-') : '데이터 없음';
    badge.className = mm.available && !mm.blocks_buy ? 'badge' : 'badge warn';
  }
  const MM_REASON_TEXT = {
    MICRO_SIGNAL_UNAVAILABLE: '실시간 시세 없음(피처 미생성)',
    MACRO_INSUFFICIENT_DATA: '실시간 시장 데이터 없음',
    MICRO_STRATEGY_BLOCKED_BY_MACRO: '거시 전략 차단',
    STALE_QUOTE: '시세 지연',
    LOW_LIQUIDITY: '유동성 부족',
    SPREAD_CONSUMES_ALPHA: '스프레드가 기대수익 잠식',
    EXPECTED_NET_RETURN_NON_POSITIVE: '순수익 미달',
    MICRO_EXIT_DETERIORATION: '청산 악화 신호',
  };
  function mmReasons(codes) {
    return (codes || []).map((c) => MM_REASON_TEXT[c] || c).join(', ');
  }
  const macro = document.getElementById('mm-macro');
  const noLiveData = mm.available && mm.data_status === 'no_live_data';
  if (macro) {
    if (!mm.available) {
      macro.innerHTML = '<div class="tech-card empty">거시 추론 데이터 없음</div>';
    } else if (noLiveData) {
      macro.innerHTML = '<div class="tech-card rejected"><div class="tech-symbol">⏳ 실시간 시세 미수신</div>'
        + '<div class="tech-explain">국내·해외장 어느 쪽도 실시간 시세(틱/호가)가 스토어에 없어 시장 레짐·종목 신호를 아직 산출할 수 없습니다. '
        + '장중에 시세가 들어오면 자동으로 레짐·후보·매수/매도 신호가 표시됩니다. (온톨로지/게이트는 정상, 데이터 대기 상태)</div></div>';
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
        + `<div class="tech-detail">${visibleDetailRows(rows)}</div></div>`;
    }
  }
  const microContainer = document.getElementById('mm-micro-cards');
  if (microContainer) {
    const rows = mm.micro || [];
    microContainer.innerHTML = rows.length ? rows.map((m) => {
      const buy = m.entry_signal === 'BUY_CANDIDATE';
      const exit = (m.exit_signal && m.exit_signal !== 'NONE');
      const kind = buy ? 'approved' : exit ? 'sell' : 'rejected';
      const detailRows = [
        ['미시 레짐', m.micro_regime || '-'],
        ['전략', m.selected_strategy || '-'],
        ['진입', m.entry_signal || '-'],
        ['청산', m.exit_signal || '-'],
        ['예상순수익', techNum(m.expected_net_return_bps, 1, 'bps')],
        ['예상청산가', techNum(m.expected_exit_price, 2)],
        ['체결품질', m.execution_quality || '-'],
        ['신뢰도', techNum(m.confidence, 2)],
      ];
      const detail = visibleDetailRows(detailRows);
      const reasons = mmReasons(m.reason_codes);
      const reasonHtml = reasons ? `<div class="tech-explain">사유: ${reasons}</div>` : '';
      return `<div class="tech-card ${kind}"><div class="tech-symbol">${m.symbol || '-'}</div><div class="tech-detail">${detail}</div>${reasonHtml}</div>`;
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

function renderMacroMicroVisual(mm) {
  const badge = document.getElementById('macro-micro-badge');
  let graph = document.getElementById('mm-graph');
  const macro = document.getElementById('mm-macro');
  const microContainer = document.getElementById('mm-micro-cards');
  const rankedContainer = document.getElementById('mm-ranked');
  const panel = document.getElementById('macro-micro-panel');
  if (!graph && macro) {
    graph = document.createElement('div');
    graph.className = 'mm-graph';
    graph.id = 'mm-graph';
    macro.parentNode.insertBefore(graph, macro);
  }
  if (panel) {
    const title = panel.querySelector('.frame-title h2');
    const note = panel.querySelector('.tech-note');
    const headings = panel.querySelectorAll('.mm-grid article h3');
    if (title) title.textContent = '거시·미시 온톨로지';
    if (note) note.textContent = '거시 레짐, 미시 종목 판단, 최종 게이트 흐름을 한 화면에서 확인합니다.';
    if (headings[0]) headings[0].textContent = '종목별 미시 추론';
    if (headings[1]) headings[1].textContent = '통합 랭킹';
  }
  const available = Boolean(mm && mm.available);
  const blocked = Boolean(mm && mm.blocks_buy);
  if (badge) {
    badge.textContent = available ? `${mm.market_regime || 'regime'} · ${blocked ? '매수 차단' : '매수 허용'}` : '데이터 없음';
    badge.className = available && !blocked ? 'badge' : 'badge warn';
  }
  if (!available) {
    const empty = '<div class="tech-card empty">거시·미시 온톨로지 데이터 대기 중</div>';
    if (graph) graph.innerHTML = renderOntologyEmptyGraph('데이터 대기');
    if (macro) macro.innerHTML = empty;
    if (microContainer) microContainer.innerHTML = empty;
    if (rankedContainer) rankedContainer.innerHTML = empty;
    return;
  }

  const microRows = Array.isArray(mm.micro) ? mm.micro : [];
  const ranked = Array.isArray(mm.ranked_intents) ? mm.ranked_intents : [];
  const sectorRows = Array.isArray(mm.sector_rankings) ? mm.sector_rankings : [];
  const candidateSymbols = Array.isArray(mm.candidate_symbols) ? mm.candidate_symbols : [];
  const noLiveData = mm.data_status === 'no_live_data';
  const buyCount = microRows.filter((m) => m.entry_signal === 'BUY_CANDIDATE').length;
  const exitCount = microRows.filter((m) => m.exit_signal && m.exit_signal !== 'NONE').length;
  const holdCount = Math.max(0, microRows.length - buyCount - exitCount);

  if (graph) {
    graph.innerHTML = renderOntologyGraphPanel({
      mm,
      microRows,
      sectorRows,
      ranked,
      candidateSymbols,
      blocked,
      noLiveData,
      buyCount,
      exitCount,
      holdCount,
    });
  }

  if (macro) {
    const topSectors = sectorRows.slice(0, 4);
    const allowed = (mm.allowed_micro_strategies || []).slice(0, 4);
    const blockedStrategies = (mm.blocked_micro_strategies || []).slice(0, 4);
    const contextSymbols = Array.isArray(mm.market_context_symbols) ? mm.market_context_symbols : [];
    const contextCount = Number(mm.market_context_symbol_count || contextSymbols.length || 0);
    macro.innerHTML = `
      <div class="mm-summary-grid">
        ${metricTile('시장 레짐', mm.market_regime || '-', noLiveData ? '실시간 시세 대기' : `신뢰도 ${techNum(mm.macro_confidence, 2)}`, blocked ? 'warn' : 'ok')}
        ${metricTile('거시 리스크', mm.macro_risk_level || '-', blocked ? '신규 매수 차단' : '신규 매수 가능', blocked ? 'warn' : 'ok')}
        ${metricTile('후보 종목', String(candidateSymbols.length), candidateSymbols.slice(0, 6).join(', ') || '-', 'info')}
        ${metricTile('시장 컨텍스트', String(contextCount), contextSymbols.slice(0, 6).join(', ') || '세션 앵커 대기', 'info')}
        ${metricTile('랭킹 의도', String(ranked.length), ranked.slice(0, 3).map((r) => `${r.side}:${r.symbol}`).join(', ') || '-', 'info')}
      </div>
      <div class="mm-lanes">
        <div>${laneTitle('상위 섹터')}${topSectors.length ? topSectors.map((s) => progressRow(s.sector, Number(s.score || 0), Number(s.confidence || 0))).join('') : '<span class="muted">섹터 데이터 없음</span>'}</div>
        <div>${laneTitle('허용 전략')}${pillList(allowed, 'ok')}${laneTitle('차단 전략')}${pillList(blockedStrategies, 'warn')}</div>
      </div>
    `;
  }

  if (microContainer) {
    microContainer.innerHTML = microRows.length ? microRows.slice(0, 10).map((m) => {
      const action = m.entry_signal === 'BUY_CANDIDATE' ? 'BUY' : (m.exit_signal && m.exit_signal !== 'NONE') ? 'SELL' : 'HOLD';
      const kind = action === 'BUY' ? 'approved' : action === 'SELL' ? 'sell' : 'hold';
      const reasons = (m.reason_codes || []).slice(0, 3).map(humanizeReason).join(' · ');
      const detail = visibleDetailRows([
        ['레짐', m.micro_regime || '-'],
        ['전략', m.selected_strategy || '-'],
        ['순기대', techNum(m.expected_net_return_bps, 1, 'bps')],
        ['청산가', techNum(m.expected_exit_price, 2)],
        ['체결', m.execution_quality || '-'],
        ['신뢰', techNum(m.confidence, 2)],
      ]);
      return `<div class="tech-card ${kind}">
        <div class="tech-symbol">${escapeHtml(m.symbol || '-')} <span class="mm-action ${kind}">${action}</span></div>
        ${renderMicroOntologySvg(m, { compact: true })}
        <div class="tech-detail">${detail}</div>
        ${reasons ? `<div class="tech-explain">${escapeHtml(reasons)}</div>` : ''}
      </div>`;
    }).join('') : '<div class="tech-card empty">미시 추론 결과 없음</div>';
  }

  if (rankedContainer) {
    rankedContainer.innerHTML = ranked.length ? ranked.slice(0, 10).map((i) => {
      const kind = i.side === 'BUY' ? 'approved' : 'sell';
      return `<div class="mm-rank-card ${kind}">
        <strong>#${escapeHtml(i.rank || '-')} ${escapeHtml(i.side || '-')} ${escapeHtml(i.symbol || '-')}</strong>
        <span>${escapeHtml(i.selected_strategy || i.micro_regime || '-')}</span>
        <div class="mm-rank-bars">
          <i style="width:${clampPct(Number(i.confidence || 0) * 100)}%"></i>
        </div>
        <small>순기대 ${techNum(i.expected_net_return_bps, 1, 'bps')} · 신뢰 ${techNum(i.confidence, 2)}</small>
      </div>`;
    }).join('') : '<div class="tech-card empty">랭킹된 의도 없음</div>';
  }
}

function renderOntologyGraphPanel(data) {
  const microRows = (data.microRows || []).slice(0, 12);
  const microGraphs = microRows.length
    ? microRows.map((row) => `<article class="mm-micro-ontology">${renderMicroOntologySvg(row)}</article>`).join('')
    : '<div class="tech-card empty">미시 온톨로지 그래프 대기 중</div>';
  return `<div class="mm-ontology-panel">
    <div class="mm-ontology-header">
      <h3>거시 온톨로지 노드·엣지</h3>
      <span>${escapeHtml(data.mm.market_regime || '-')} · 후보 ${data.candidateSymbols.length} · 미시 ${data.microRows.length}</span>
    </div>
    ${renderMacroOntologySvg(data)}
    <div class="mm-ontology-header micro">
      <h3>종목별 미시 온톨로지 노드·엣지</h3>
      <span>N×M 레이아웃 · 최대 12개 표시</span>
    </div>
    <div class="mm-micro-graph-grid">${microGraphs}</div>
  </div>`;
}

function renderMacroOntologySvg(data) {
  const mm = data.mm || {};
  const allowed = (mm.allowed_micro_strategies || []).slice(0, 2).join(', ') || '-';
  const blockedStrategies = (mm.blocked_micro_strategies || []).slice(0, 2).join(', ') || '-';
  const topSector = (data.sectorRows || [])[0];
  const nodes = [
    { id: 'market', x: 98, y: 118, label: 'Market', value: mm.market_regime || '-', tone: data.blocked ? 'warn' : 'ok' },
    { id: 'risk', x: 276, y: 58, label: 'Risk', value: mm.macro_risk_level || '-', tone: data.blocked ? 'warn' : 'info' },
    { id: 'confidence', x: 276, y: 178, label: 'Confidence', value: `${(Number(mm.macro_confidence || 0) * 100).toFixed(0)}%`, tone: 'info' },
    { id: 'sector', x: 460, y: 58, label: 'Sector', value: topSector ? topSector.sector : '-', tone: 'info' },
    { id: 'allow', x: 460, y: 178, label: 'Allows', value: allowed, tone: 'ok' },
    { id: 'block', x: 640, y: 58, label: 'Blocks', value: blockedStrategies, tone: data.blocked ? 'warn' : 'hold' },
    { id: 'candidates', x: 640, y: 178, label: 'Candidates', value: `${data.candidateSymbols.length} symbols`, tone: data.noLiveData ? 'warn' : 'info' },
    { id: 'gate', x: 820, y: 118, label: 'Gate', value: data.blocked ? 'BUY BLOCK' : 'BUY OPEN', tone: data.blocked ? 'warn' : 'ok' },
  ];
  const edges = [
    ['market', 'risk', 'has_risk'],
    ['market', 'confidence', 'measured_by'],
    ['risk', 'sector', 'weights'],
    ['confidence', 'allow', 'supports'],
    ['sector', 'block', 'constrains'],
    ['allow', 'candidates', 'permits'],
    ['block', 'gate', data.blocked ? 'blocks' : 'monitors'],
    ['candidates', 'gate', 'feeds'],
  ];
  return `<svg class="mm-ontology-svg macro" viewBox="0 0 920 260" role="img" aria-label="거시 온톨로지 노드 엣지 그래프">
    <defs>
      <marker id="mmArrowMacro" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L8,3 z" fill="#8091a7"></path>
      </marker>
    </defs>
    <rect x="0" y="0" width="920" height="260" fill="#f8fafc"></rect>
    ${edges.map((edge) => ontologyEdge(edge, nodes, 'mmArrowMacro')).join('')}
    ${nodes.map((node) => ontologyNode(node)).join('')}
    <g transform="translate(42 226)">
      <text x="0" y="0" class="mm-edge-note">BUY ${data.buyCount} · SELL ${data.exitCount} · HOLD ${data.holdCount} · ranked ${(data.ranked || []).length}</text>
    </g>
  </svg>`;
}

function renderMicroOntologySvg(row, options) {
  const compact = Boolean(options && options.compact);
  const width = compact ? 470 : 520;
  const height = compact ? 190 : 230;
  const markerId = compact ? 'mmArrowMicroCompact' : 'mmArrowMicroFull';
  const reasons = (row.reason_codes || []).slice(0, 2).map(humanizeReason).join(', ') || '-';
  const entry = row.entry_signal || 'NONE';
  const exit = row.exit_signal || 'NONE';
  const actionTone = entry === 'BUY_CANDIDATE' ? 'ok' : exit !== 'NONE' ? 'warn' : 'hold';
  const nodes = [
    { id: 'symbol', x: 78, y: compact ? 88 : 108, label: 'Symbol', value: row.symbol || '-', tone: actionTone },
    { id: 'regime', x: 220, y: 48, label: 'Regime', value: row.micro_regime || '-', tone: actionTone },
    { id: 'strategy', x: 220, y: compact ? 132 : 168, label: 'Strategy', value: row.selected_strategy || '-', tone: 'info' },
    { id: 'entry', x: 365, y: 48, label: 'Entry', value: entry, tone: entry === 'BUY_CANDIDATE' ? 'ok' : 'hold' },
    { id: 'exit', x: 365, y: compact ? 132 : 168, label: 'Exit', value: exit, tone: exit !== 'NONE' ? 'warn' : 'hold' },
  ];
  if (!compact) {
    nodes.push(
      { id: 'net', x: 78, y: 188, label: 'Net Edge', value: techNum(row.expected_net_return_bps, 1, 'bps'), tone: Number(row.expected_net_return_bps || 0) > 0 ? 'ok' : 'warn' },
      { id: 'quality', x: 365, y: 108, label: 'Quality', value: row.execution_quality || '-', tone: row.execution_quality === 'WEAK' ? 'warn' : 'info' },
      { id: 'reason', x: 220, y: 108, label: 'Reason', value: reasons, tone: 'hold' },
    );
  }
  const edges = compact
    ? [
        ['symbol', 'regime', 'classified_as'],
        ['symbol', 'strategy', 'selects'],
        ['regime', 'entry', 'emits'],
        ['strategy', 'exit', 'guards'],
      ]
    : [
        ['symbol', 'regime', 'classified_as'],
        ['symbol', 'net', 'expects'],
        ['regime', 'reason', 'explained_by'],
        ['reason', 'quality', 'constrained_by'],
        ['regime', 'entry', 'emits'],
        ['strategy', 'exit', 'guards'],
        ['net', 'strategy', 'selects'],
        ['quality', 'exit', 'routes'],
      ];
  return `<svg class="mm-ontology-svg micro${compact ? ' compact' : ''}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(row.symbol || 'symbol')} 미시 온톨로지 그래프">
    <defs>
      <marker id="${markerId}" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L7,3 z" fill="#8091a7"></path>
      </marker>
    </defs>
    <rect x="0" y="0" width="${width}" height="${height}" fill="#ffffff"></rect>
    ${edges.map((edge) => ontologyEdge(edge, nodes, markerId)).join('')}
    ${nodes.map((node) => ontologyNode(node, compact ? 42 : 46)).join('')}
  </svg>`;
}

function ontologyEdge(edge, nodes, markerId) {
  const from = nodes.find((node) => node.id === edge[0]);
  const to = nodes.find((node) => node.id === edge[1]);
  if (!from || !to) return '';
  const midX = (from.x + to.x) / 2;
  const midY = (from.y + to.y) / 2;
  return `<g class="mm-ontology-edge">
    <line x1="${from.x}" y1="${from.y}" x2="${to.x}" y2="${to.y}" marker-end="url(#${markerId})"></line>
    <text x="${midX}" y="${midY - 5}" text-anchor="middle">${escapeHtml(edge[2])}</text>
  </g>`;
}

function ontologyNode(node, radius) {
  const r = radius || 50;
  return `<g class="mm-ontology-node ${node.tone || 'info'}" transform="translate(${node.x} ${node.y})">
    <circle r="${r}"></circle>
    <text y="-7" text-anchor="middle" class="mm-node-label">${escapeHtml(node.label)}</text>
    <text y="14" text-anchor="middle" class="mm-node-value">${escapeHtml(shortNodeValue(node.value))}</text>
  </g>`;
}

function shortNodeValue(value) {
  const text = String(value == null ? '-' : value);
  return text.length > 18 ? `${text.slice(0, 17)}...` : text;
}

function renderOntologyEmptyGraph(label) {
  return `<svg class="mm-graph-svg" viewBox="0 0 920 260" role="img" aria-label="${escapeHtml(label)}">
    <rect x="0" y="0" width="920" height="260" rx="0" fill="#f8fafc"></rect>
    <text x="460" y="132" text-anchor="middle" fill="#667085" font-size="16" font-weight="700">${escapeHtml(label)}</text>
  </svg>`;
}

function renderOntologyFlowSvg(data) {
  const nodes = [
    { id: 'macro', x: 90, y: 78, label: '거시', value: data.regime, tone: data.blocked ? 'warn' : 'ok' },
    { id: 'risk', x: 265, y: 78, label: '리스크', value: data.risk, tone: data.blocked ? 'warn' : 'info' },
    { id: 'candidates', x: 440, y: 78, label: '후보', value: `${data.candidates}종목`, tone: data.noLiveData ? 'warn' : 'info' },
    { id: 'micro', x: 615, y: 78, label: '미시', value: `${data.micro}건`, tone: 'info' },
    { id: 'gate', x: 790, y: 78, label: '최종 게이트', value: data.blocked ? '차단' : '검증', tone: data.blocked ? 'warn' : 'ok' },
  ];
  const bars = [
    { label: 'BUY', value: data.buyCount, color: '#138a5b' },
    { label: 'SELL', value: data.exitCount, color: '#c2413a' },
    { label: 'HOLD', value: data.holdCount, color: '#64748b' },
    { label: 'RANK', value: data.rankedCount, color: '#2563eb' },
  ];
  const max = Math.max(1, ...bars.map((b) => b.value));
  const links = nodes.slice(0, -1).map((n, idx) => {
    const next = nodes[idx + 1];
    return `<path d="M ${n.x + 58} ${n.y} C ${n.x + 95} ${n.y}, ${next.x - 95} ${next.y}, ${next.x - 58} ${next.y}" class="mm-link ${data.blocked && idx >= 1 ? 'blocked' : ''}"></path>`;
  }).join('');
  return `<svg class="mm-graph-svg" viewBox="0 0 920 260" role="img" aria-label="거시 미시 온톨로지 흐름">
    <defs>
      <filter id="mmShadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="4" stdDeviation="5" flood-color="#0f172a" flood-opacity="0.10"/></filter>
    </defs>
    <rect x="0" y="0" width="920" height="260" rx="0" fill="#f8fafc"></rect>
    ${links}
    ${nodes.map((n) => graphNode(n)).join('')}
    <g transform="translate(54 178)">
      ${bars.map((b, idx) => {
        const y = idx * 18;
        const w = Math.max(10, (b.value / max) * 180);
        return `<g transform="translate(0 ${y})">
          <text x="0" y="11" fill="#475569" font-size="11" font-weight="700">${b.label}</text>
          <rect x="48" y="2" width="180" height="10" rx="5" fill="#e2e8f0"></rect>
          <rect x="48" y="2" width="${w}" height="10" rx="5" fill="${b.color}"></rect>
          <text x="238" y="11" fill="#0f172a" font-size="11" font-weight="700">${b.value}</text>
        </g>`;
      }).join('')}
    </g>
    <text x="790" y="204" text-anchor="middle" fill="#475569" font-size="12">거시 신뢰도 ${(Number(data.confidence || 0) * 100).toFixed(0)}%</text>
  </svg>`;
}

function graphNode(n) {
  return `<g class="mm-node ${n.tone}" transform="translate(${n.x} ${n.y})" filter="url(#mmShadow)">
    <circle r="58"></circle>
    <text y="-10" text-anchor="middle" class="mm-node-label">${escapeHtml(n.label)}</text>
    <text y="14" text-anchor="middle" class="mm-node-value">${escapeHtml(n.value)}</text>
  </g>`;
}

function metricTile(label, value, note, tone) {
  return `<div class="mm-metric ${tone || ''}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(note || '')}</small></div>`;
}

function laneTitle(text) {
  return `<h4 class="mm-lane-title">${escapeHtml(text)}</h4>`;
}

function progressRow(label, score, confidence) {
  const pct = clampPct(Math.abs(score) * 100);
  const cls = score >= 0 ? 'positive' : 'negative';
  return `<div class="mm-progress-row">
    <span>${escapeHtml(label || '-')}</span>
    <div><i class="${cls}" style="width:${pct}%"></i></div>
    <strong>${score.toFixed(2)} · ${(confidence * 100).toFixed(0)}%</strong>
  </div>`;
}

function pillList(items, tone) {
  return items && items.length
    ? `<div class="mm-pill-list">${items.map((item) => `<span class="${tone || ''}">${escapeHtml(item)}</span>`).join('')}</div>`
    : '<span class="muted">없음</span>';
}

function clampPct(value) {
  return Math.max(0, Math.min(100, Number.isFinite(value) ? value : 0));
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

function visibleDetailRows(rows) {
  return rows
    .filter(([, value]) => value !== null && value !== undefined && String(value).trim() !== '' && String(value).trim() !== '-')
    .slice(0, 6)
    .map(([key, value]) => `<span><em>${key}</em>${value}</span>`)
    .join('');
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
      const detail = visibleDetailRows(rows);
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
  BAR_VWAP_RECOVERY_INPUTS_MISSING: '1분봉 회복 판단 데이터 부족',
  BAR_VWAP_DISPLACEMENT_TOO_SMALL: 'VWAP 하락 이격이 진입 기준보다 작음',
  BAR_VWAP_VOLATILITY_SCALE_MISSING: '변동성 정규화 값 부족',
  BAR_VWAP_DISLOCATION_TOO_EXTREME: '정상 회귀보다 구조적 재가격 가능성이 큼',
  BAR_VWAP_NOT_OVERSOLD: '과매도 조건 미충족',
  BAR_VWAP_FAST_EMA_NOT_RECLAIMED: '완료 1분봉이 단기 EMA를 아직 회복하지 못함',
  BAR_VWAP_MACD_NOT_TURNED: 'MACD 회복 전환 미확인',
  BAR_VWAP_RECOVERY_NOT_PERSISTENT: '1분봉 회복 지속성 부족',
  BAR_VWAP_LIQUIDITY_TOO_LOW: '회복 거래에 필요한 유동성 부족',
  BAR_VWAP_SPREAD_TOO_WIDE: '스프레드가 넓어 회복 기대수익 잠식',
  BAR_CONFIRMED_VWAP_RECOVERY: '완료 1분봉 기준 VWAP 회복 진입 확인',
  COMPLETED_MINUTE_TREND_TURNED: '완료 1분봉 추세가 상승 전환',
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
  // GNN-direct posture. These must not render as raw codes: they are the only
  // on-screen sign that the pessimistic bound and the NO_TRADE option are not
  // in play for this election.
  PROFITABILITY_GATE_OVERRULED: '⚠ 수익성 게이트 거부를 무시하고 진행(GNN 직결)',
  GNN_DIRECT_ELECTION: 'GNN 직결 채택(밴딧 하한·NO_TRADE 미적용)',
  GNN_ESTIMATE_PRESENT: 'GNN 예측 있음',
  GNN_ESTIMATE_UNAVAILABLE_RANKED_LAST: '⚠ GNN 예측 없이 채택(후순위였으나 단독 후보)',
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
    ['오늘 순손익(제비용 반영)', fmtKrw(prof.net_after_cost_krw), `KIS 정산손익 ${fmtKrw(prof.realized_pnl_today_krw)} · 제비용 ${fmtKrw(prof.broker_expenses_krw)}`, clsPnl(prof.net_after_cost_krw)],
    ['승률', zeroTradeState ? '0.00%' : fallbackValue(prof.win_rate, fmtPct, '0.00%'), `${prof.win_count || 0}승 / ${prof.loss_count || 0}패 (${closedTradeCount}건)`],
    ['평균 수익', zeroTradeState ? '₩0' : fallbackValue(prof.avg_win_krw, fmtKrw, '₩0'), zeroTradeState ? '청산 거래 없음' : '이익 거래 평균', 'positive'],
    ['평균 손실', zeroTradeState ? '₩0' : fallbackValue(prof.avg_loss_krw, fmtKrw, '₩0'), zeroTradeState ? '청산 거래 없음' : '손실 거래 평균', 'negative'],
    ['손익비 (Payoff)', zeroTradeState ? '0.00' : fallbackValue(prof.payoff_ratio, (v) => Number(v).toFixed(2), '0.00'), zeroTradeState ? '청산 거래 없음' : '평균수익 / |평균손실|'],
    ['기대값 (Expectancy)', zeroTradeState ? '₩0' : fallbackValue(prof.expectancy_krw, fmtKrw, '₩0'), zeroTradeState ? '청산 거래 없음' : '거래당 기대손익', clsPnl(prof.expectancy_krw)],
    ['거래 제비용', fmtKrw(prof.broker_expenses_krw), `매입·매도수수료 + 유관기관 제비용 + 제세금 · ${prof.broker_expense_source || '미수집'}`, 'negative'],
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
      <td>${fmtMoney(row.current_price, row.currency)}</td>
      <td class="order-state">${orderState}</td>
      <td class="order-summary" title="${orderSummary}">${orderSummary}</td>
      <td class="${clsPnl(row.unrealized_pnl_krw)}">${fmtKrw(row.unrealized_pnl_krw)}</td>
      <td>${estNet}</td>
      <td class="${clsPnl(row.unrealized_pnl_rate)}">${fmtPct(row.unrealized_pnl_rate)}</td>
      <td>${fmtPct(row.weight_of_total_asset)}</td>
      <td>${row.currency}</td>
    </tr>`;
  }).join('') : `<tr class="empty-row"><td colspan="12">현재 보유 종목 없음</td></tr>`;
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
  const collectorRows = Array.isArray(logs.collection_log) ? logs.collection_log : [];
  const collector = [...collectorRows].reverse().find((row) => {
    const counts = row && row.counts ? row.counts : {};
    return Number(counts.control_messages || 0) > 0;
  });
  const collectorCounts = collector && collector.counts ? collector.counts : {};
  const subscriptionRequests = Number(
    collectorCounts.subscription_requests || collectorCounts.subscriptions || 0
  );
  const subscriptionRejected = Number(
    collectorCounts.subscriptions_rejected || collectorCounts.control_errors || 0
  );
  const subscriptionAccepted = Object.prototype.hasOwnProperty.call(
    collectorCounts,
    "subscriptions_accepted"
  )
    ? Number(collectorCounts.subscriptions_accepted || 0)
    : Math.max(0, subscriptionRequests - subscriptionRejected);
  const subscriptionLabel = subscriptionRequests
    ? `${subscriptionAccepted}/${subscriptionRequests} 승인 · ${subscriptionRejected} 거절`
    : '수집 대기';
  const sellOnly = Boolean(trading && trading.running && trading.buy_enabled === false);
  const items = [
    ['KIS 구독', subscriptionLabel],
    ['포지션 감시', trading && trading.running ? (sellOnly ? '매도 전용 실행' : '매수·매도 실행') : '중지'],
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
    if (p.all_in_cost_rate !== undefined && p.all_in_cost_rate !== null) chips.push(`총 거래비용 ${fmtPct(p.all_in_cost_rate)}`);
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
  if (source) source.textContent = 'Termination requested: BUY evaluation disabled, sell-only liquidation mode starting...';
  try {
    const data = await postJson('/api/live-trading/terminate?shutdown=false');
    if (source) {
      const engine = data.engine_status || {};
      const sellSubmitted = Number(engine.sell_submitted || (data.cycle_summary || {}).sell_submitted || data.submitted_sell_orders || 0);
      source.textContent = data.ok
        ? `Sell-only liquidation active: BUY disabled, submitted SELL total ${sellSubmitted}`
        : `Termination blocked: ${data.message || data.status || 'unknown'}`;
    }
    if (data.ok) {
      await refreshDashboard();
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

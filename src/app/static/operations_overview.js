const operationsOverviewState = {
  busy: false,
  timer: null,
};

const opsReasonLabels = {
  MARKET_DATA_NOT_READY: '실시간 체결·호가 신선도 부족',
  GNN_NOT_LIVE_AUTHORIZED: 'GNN 실시간 신뢰도 미승격',
  NO_POSITIVE_NET_GNN_EDGE: 'GNN 검증 완료 · 현재 양의 순효율 후보 없음',
  GNN_POSITIVE_EDGE_AWAITING_ENTRY_VALIDATION: '양의 순효율 후보 실시간 결과 검증 중',
  GNN_REALTIME_MODEL_TRUST_PASSED: 'GNN 실시간 모델 검증 통과',
  GNN_TRUST_NET_SIGN_ACCURACY_TOO_LOW: 'GNN 순효율 방향 정확도 부족',
  GNN_TRUST_NET_MAE_TOO_HIGH: 'GNN 순효율 오차 과다',
  NO_FRESH_STRATEGY_ELECTION: '신선한 전략 선택 근거 대기',
  MACRO_INSUFFICIENT_DATA: '전략 계산용 분봉 이력 축적 중',
  GNN_TRUST_INSUFFICIENT_REALTIME_SAMPLES: 'GNN 실시간 검증 표본 부족',
  GNN_TRUST_POSITIVE_NET_RATE_TOO_LOW: '양수 순효율 비율 미달',
  GNN_TRUST_REALIZED_NET_EDGE_NON_POSITIVE: '실현 순효율이 양수가 아님',
  GNN_TRUST_SCORE_BELOW_THRESHOLD: 'GNN 신뢰 점수 기준 미달',
  GNN_TRUST_NO_STRATEGY_PASSED: '신뢰 기준 통과 전략 없음',
  GNN_TRUST_NO_POSITIVE_EDGE_VALIDATED_STRATEGY: '양의 진입 신호 검증 완료 전략 없음',
};

function opsText(id, value) {
  const node = document.getElementById(id);
  if (node) node.textContent = value ?? '-';
}

function opsNumber(value, digits = 0) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '-';
  return numeric.toLocaleString('ko-KR', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function opsPercent(value, digits = 1) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '-';
  return `${(numeric * 100).toFixed(digits)}%`;
}

function opsClock(value) {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '-';
  return date.toLocaleTimeString('ko-KR', { hour12: false });
}

function opsReason(code) {
  const raw = String(code || '').trim();
  return opsReasonLabels[raw] || raw || '-';
}

function opsSetGate(key, tone, title, detail) {
  const gate = document.querySelector(`[data-ops-gate="${key}"]`);
  if (!gate) return;
  gate.className = tone;
  const strong = gate.querySelector('strong');
  const small = gate.querySelector('small');
  if (strong) strong.textContent = title;
  if (small) small.textContent = detail;
}

function opsSetBadge(id, tone, text) {
  const node = document.getElementById(id);
  if (!node) return;
  node.className = tone;
  node.textContent = text;
}

async function opsFetchJson(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${url} · HTTP ${response.status}`);
  return response.json();
}

function opsPrimaryBlocker(diag, reliability, gnn, trading) {
  const systemError = diag?.last_error;
  if (systemError) {
    return {
      tone: 'error',
      title: '시스템 오류 확인 필요',
      detail: String(systemError),
    };
  }
  const market = reliability?.components?.market_data || {};
  if (!market.ok) {
    const missing = (market.missing_markets || []).join(', ') || '활성 시장';
    return {
      tone: 'blocked',
      title: '실시간 시장 데이터 대기',
      detail: `${missing} 시장에서 신선한 체결과 호가가 기준 종목 수만큼 필요합니다.`,
    };
  }
  if (!gnn?.passed) {
    const sampleCount = Number(gnn?.sample_count || 0);
    const minimum = Number(gnn?.minimum_samples || 0);
    const mainReason = (gnn?.reason_codes || [])[0];
    return {
      tone: 'waiting',
      title: 'GNN 실시간 검증 진행 중',
      detail: `표본 ${sampleCount}/${minimum} · ${opsReason(mainReason)}. 통과 전에는 전략 소유권과 주문이 차단됩니다.`,
    };
  }
  if (!(gnn?.trusted_strategy_ids || []).length) {
    const calibrated = gnn?.calibrated_strategy_ids || [];
    return {
      tone: 'waiting',
      title: 'GNN 양의 진입 신호 검증 중',
      detail: `보정 신뢰 통과 ${calibrated.length}개 · 실제 시장에서 양의 순효율이 반복 검증된 전략만 주문 권한을 받습니다.`,
    };
  }
  const session = trading?.status?.strategy_session || {};
  if (!trading?.running) {
    return {
      tone: 'waiting',
      title: '실거래 엔진 승격 대기',
      detail: '데이터와 GNN 게이트는 통과했지만 실거래 엔진이 아직 시작되지 않았습니다.',
    };
  }
  if (!session.selected_strategy) {
    return {
      tone: 'waiting',
      title: '전략 선택 근거 탐색 중',
      detail: `${opsReason(session.last_reason)} · 실시간 데이터에서 온톨로지와 GNN이 합의할 전략을 찾고 있습니다.`,
    };
  }
  return {
    tone: 'good',
    title: '실거래 전략 활성',
    detail: `${session.selected_symbol || '-'} · ${session.selected_strategy} · ${session.phase || 'ACTIVE'}`,
  };
}

function renderOperationsOverview({ diag, reliability, trading, gnn, mode }) {
  const market = reliability?.components?.market_data || {};
  const healthy = market.healthy || {};
  const ws = diag?.flows?.market_data?.subscription?.overseas_websocket || {};
  const wsCounts = ws.counts || {};
  const intelligence = diag?.flows?.intelligence || {};
  const research = diag?.flows?.research_collection || {};
  const training = diag?.flows?.training || {};
  const activeModel = training.active_model || {};
  const session = trading?.status?.strategy_session || {};
  const summary = trading?.status?.last_summary || {};
  const trusted = gnn?.trusted_strategy_ids || [];
  const calibrated = gnn?.calibrated_strategy_ids || [];
  const overall = opsPrimaryBlocker(diag, reliability, gnn, trading);

  const overallNode = document.getElementById('ops-overall-state');
  if (overallNode) overallNode.className = `ops-overall-state ${overall.tone}`;
  const overallStrong = overallNode?.querySelector('strong');
  if (overallStrong) overallStrong.textContent = overall.title;
  opsText('ops-updated-at', `갱신 ${opsClock(diag?.generated_at)}`);
  opsText(
    'ops-overall-summary',
    `운영 모드 ${String(mode?.active?.mode || reliability?.mode || '-').toUpperCase()} · 자동 신뢰도 ${opsNumber(reliability?.score, 2)} · 엔진 ${trading?.running ? '실행 중' : '대기'}`,
  );

  const alert = document.getElementById('ops-alert');
  if (alert) alert.className = `ops-alert ${overall.tone}`;
  opsText('ops-alert-title', overall.title);
  opsText('ops-alert-detail', overall.detail);

  opsSetGate('server', diag?.last_error ? 'error' : 'pass', diag?.last_error ? '오류' : '정상', `PID 연결 · ${opsClock(diag?.generated_at)}`);
  const brokerOk = Boolean(reliability?.components?.broker?.ok);
  opsSetGate('broker', brokerOk ? 'pass' : 'block', brokerOk ? '연결 정상' : '연결 실패', brokerOk ? '실계좌 조회 가능' : 'KIS 계좌 연결 확인 필요');
  opsSetGate(
    'market',
    market.ok ? 'pass' : 'block',
    market.ok ? '거래급 데이터' : '신선도 미달',
    `KRX ${(healthy.KRX || []).length} · US ${(healthy.US || []).length}`,
  );
  const ontologyOk = Boolean(intelligence.ready) && !diag?.last_error;
  opsSetGate(
    'ontology',
    ontologyOk ? 'pass' : 'warn',
    ontologyOk ? '그래프 정상' : '수집·구축 중',
    `연결 ${opsNumber(intelligence.ontology_event_links)} · 이벤트 ${opsNumber(intelligence.context_events)}`,
  );
  opsSetGate(
    'gnn',
    trusted.length ? 'pass' : (gnn?.passed ? 'warn' : 'block'),
    trusted.length ? '진입 신뢰 통과' : (gnn?.passed ? '모델 보정 통과' : '검증 대기'),
    `점수 ${opsNumber(gnn?.score, 3)} · 보정 ${calibrated.length} · 진입 ${trusted.length}`,
  );
  const executionReady = Boolean(
    trading?.running
    && trading?.buy_enabled
    && gnn?.passed
    && session?.selected_strategy,
  );
  opsSetGate(
    'execution',
    executionReady ? 'pass' : (trading?.running ? 'warn' : 'block'),
    executionReady ? '전략 활성' : (trading?.running ? '엔진 감시 중' : '엔진 대기'),
    executionReady ? `${session.selected_symbol} · ${session.selected_strategy}` : opsReason(session.last_reason || overall.title),
  );

  const symbols = ws.symbols || [];
  opsSetBadge('ops-feed-state', market.ok ? 'pass' : 'block', market.ok ? 'READY' : 'NOT READY');
  opsText('ops-feed-symbols', symbols.length ? symbols.join(' · ') : '-');
  opsText('ops-feed-accepted', opsNumber(wsCounts.subscriptions_accepted || 0));
  opsText('ops-feed-events', `${opsNumber(wsCounts.ticks || 0)} / ${opsNumber(wsCounts.orderbooks || 0)}`);
  opsText('ops-feed-healthy', `KRX ${(healthy.KRX || []).length} · US ${(healthy.US || []).length}`);
  opsText(
    'ops-feed-note',
    ws.last_error
      ? `오류: ${ws.last_error}`
      : `마지막 구독 성공 ${opsClock(ws.last_success_at)} · 필수 시장 ${(market.required_markets || []).join(', ') || '-'}`,
  );

  opsSetBadge('ops-ontology-state', ontologyOk ? 'pass' : 'warn', ontologyOk ? 'READY' : 'BUILDING');
  opsText('ops-context-events', opsNumber(intelligence.context_events || 0));
  opsText('ops-ontology-links', opsNumber(intelligence.ontology_event_links || 0));
  opsText('ops-research-cycle', research.latest?.status || (research.active ? 'running' : 'waiting'));
  opsText('ops-model-state', activeModel.live_eligible ? '운영 가능' : '검증 중');
  opsText(
    'ops-ontology-note',
    `연구 ${research.latest?.status || (research.active ? '진행 중' : '대기')} · 활성 모델 AUC ${opsNumber(activeModel.metrics?.auc, 3)} · 오류 ${diag?.last_error ? '있음' : '없음'}`,
  );

  opsSetBadge(
    'ops-gnn-state',
    trusted.length ? 'pass' : 'warn',
    trusted.length ? 'ENTRY READY' : (gnn?.passed ? 'CALIBRATED' : 'VALIDATING'),
  );
  opsText('ops-gnn-score', opsNumber(gnn?.score, 3));
  opsText('ops-gnn-samples', `표본 ${opsNumber(gnn?.sample_count)}/${opsNumber(gnn?.minimum_samples)}`);
  opsText('ops-gnn-trusted', `진입 허용 ${trusted.length ? trusted.join(', ') : '없음'} · 보정 통과 ${calibrated.length}`);
  const progress = Math.min(100, Math.max(0, Number(gnn?.sample_count || 0) / Math.max(1, Number(gnn?.minimum_samples || 1)) * 100));
  const progressNode = document.getElementById('ops-gnn-progress');
  if (progressNode) progressNode.style.width = `${progress}%`;
  opsText('ops-gnn-positive', opsPercent(gnn?.positive_net_rate));
  opsText('ops-gnn-net', `${opsNumber(gnn?.mean_realized_net_bps, 2)} bp`);
  opsText('ops-gnn-uncertainty', opsNumber(gnn?.mean_uncertainty, 3));
  const strategyMetrics = gnn?.strategy_metrics || {};
  const strategyList = document.getElementById('ops-strategy-list');
  if (strategyList) {
    const rows = Object.entries(strategyMetrics);
    strategyList.innerHTML = rows.length
      ? rows.map(([strategyId, metrics]) => `
          <div class="ops-strategy-row">
            <b title="${strategyId}">${strategyId}</b>
            <span>${Number(metrics.sample_count || 0)}표본</span>
            <span class="${metrics.entry_authorized ? 'pass' : (metrics.calibration_passed ? 'warn' : 'block')}">${metrics.entry_authorized ? '진입 허용' : (metrics.calibration_passed ? '보정 통과' : Number(metrics.score || 0).toFixed(3))}</span>
          </div>
        `).join('')
      : '<div class="ops-strategy-row"><b>검증 전략 없음</b><span>-</span><span class="block">대기</span></div>';
  }

  opsSetBadge(
    'ops-engine-state',
    trading?.running ? (session.last_reason === 'GNN_NOT_LIVE_AUTHORIZED' ? 'warn' : 'pass') : 'block',
    trading?.running ? 'RUNNING' : 'STOPPED',
  );
  opsText('ops-session-phase', session.phase || '-');
  opsText('ops-selected-strategy', session.selected_strategy || '미선택');
  const candidates = summary.buy_candidate_sample || [];
  opsText('ops-buy-candidates', candidates.length ? candidates.join(' · ') : '없음');
  opsText('ops-order-errors', `${opsNumber(trading?.status?.submitted || 0)} / ${opsNumber(trading?.status?.errors || 0)}`);
  opsText(
    'ops-engine-note',
    `${opsReason(session.last_reason)} · 선택 종목 ${session.selected_symbol || '-'} · GNN 판단 ${session.gnn_action || '-'}`,
  );

  opsText('ops-mode', String(mode?.active?.mode || reliability?.mode || '-').toUpperCase());
  opsText('ops-reliability', `${opsNumber(reliability?.score, 2)} / ${opsNumber(reliability?.threshold, 2)} · ${reliability?.ready ? '통과' : '대기'}`);
  opsText('ops-gnn-required', session.require_live_gnn === true ? '필수 · 강제' : (trading?.running ? '비활성' : '엔진 시작 후 확인'));
  opsText('ops-engine-cycles', opsNumber(trading?.status?.cycles || 0));
  opsText('ops-current-reason', opsReason(session.last_reason || (reliability?.reasons || [])[0] || (gnn?.reason_codes || [])[0]));
}

function renderOperationsOverviewError(error) {
  const overall = document.getElementById('ops-overall-state');
  if (overall) overall.className = 'ops-overall-state error';
  const strong = overall?.querySelector('strong');
  if (strong) strong.textContent = '상태 조회 실패';
  const alert = document.getElementById('ops-alert');
  if (alert) alert.className = 'ops-alert error';
  opsText('ops-alert-title', '운영 API 연결 오류');
  opsText('ops-alert-detail', error?.message || String(error));
  opsSetGate('server', 'error', '조회 실패', '서버와 API 상태 확인 필요');
}

async function refreshOperationsOverview() {
  if (operationsOverviewState.busy) return;
  operationsOverviewState.busy = true;
  try {
    const [diag, reliability, trading, gnn, mode] = await Promise.all([
      opsFetchJson('/api/system-diagnostics'),
      opsFetchJson('/api/auto-reliability/status'),
      opsFetchJson('/api/realtime-trading/status'),
      opsFetchJson('/api/gnn/realtime-trust'),
      opsFetchJson('/api/operation-mode/status'),
    ]);
    renderOperationsOverview({ diag, reliability, trading, gnn, mode });
  } catch (error) {
    renderOperationsOverviewError(error);
  } finally {
    operationsOverviewState.busy = false;
  }
}

window.refreshOperationsOverview = refreshOperationsOverview;
window.addEventListener('DOMContentLoaded', () => {
  refreshOperationsOverview();
  operationsOverviewState.timer = window.setInterval(refreshOperationsOverview, 5000);
});

const terminalState = {
  symbol: null,
  data: null,
  busy: false,
  assetBusy: false,
  diagnosticsBusy: false,
  diagnostics: null,
  ontologyFilter: 'active',
  streamBusy: false,
  streamReceivedAt: 0,
  lastSecondBarTime: null,
  chartTransitionActive: false,
  chartMode: 'seconds',
  chartAnimationFrame: null,
  chartLastPaint: 0,
};

const strategyLabels = {
  intraday_momentum: '장중 모멘텀',
  breakout_volume: '거래량 돌파',
  vwap_mean_reversion: 'VWAP 평균회귀',
  liquidity_shock_reversal: '유동성 충격 반전',
  event_momentum: '이벤트 모멘텀',
  cross_sectional_relative_strength: '횡단면 상대강도',
  gap_context: '갭 컨텍스트',
};

async function fetchMarketView(symbol = null) {
  if (terminalState.busy) return;
  terminalState.busy = true;
  const refresh = document.getElementById('terminal-refresh');
  if (refresh) refresh.classList.add('loading');
  try {
    const query = symbol ? `?symbol=${encodeURIComponent(symbol)}&limit=180` : '?limit=180';
    const response = await fetch(`/api/refactor/market-view${query}`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const data = await response.json();
    const symbolChanged = terminalState.symbol && terminalState.symbol !== data.symbol;
    terminalState.data = data;
    terminalState.symbol = data.symbol;
    registerSecondBarArrival(data.market || {}, Boolean(symbolChanged));
    renderTerminal(data);
  } catch (error) {
    document.getElementById('feed-state').textContent = `데이터 오류 · ${error.message}`;
    document.querySelector('.connection')?.classList.add('stale');
  } finally {
    terminalState.busy = false;
    if (refresh) refresh.classList.remove('loading');
  }
}

async function fetchAssetSummary() {
  if (terminalState.assetBusy) return;
  terminalState.assetBusy = true;
  try {
    const response = await fetch('/api/account/summary', { cache: 'no-store' });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    renderAssetSummary(await response.json());
  } catch (error) {
    renderAssetSummary({
      status: 'unavailable',
      message: `계좌 정보를 불러오지 못했습니다: ${error.message}`,
      snapshot: null,
    });
  } finally {
    terminalState.assetBusy = false;
  }
}

async function fetchMarketStream() {
  if (terminalState.streamBusy || !terminalState.symbol || !terminalState.data) return;
  terminalState.streamBusy = true;
  try {
    const response = await fetch(
      `/api/refactor/market-stream?symbol=${encodeURIComponent(terminalState.symbol)}&limit=30`,
      { cache: 'no-store' },
    );
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const stream = await response.json();
    if (stream.symbol !== terminalState.symbol || !stream.market) return;
    terminalState.data.market = {
      ...(terminalState.data.market || {}),
      ...stream.market,
    };
    terminalState.data.generated_at = stream.generated_at || terminalState.data.generated_at;
    registerSecondBarArrival(terminalState.data.market);
    renderHeader(terminalState.data, terminalState.data.market);
    renderInstrument(
      terminalState.symbol,
      terminalState.data.market,
      terminalState.data.selection || {},
      terminalState.data.algorithm || null,
    );
    renderTradingChart(terminalState.data.market, terminalState.data.algorithm || null);
    renderSecondAnalysis(terminalState.data.market);
    renderExecution(
      terminalState.data.execution || {},
      terminalState.data.market.latest_orderbook,
    );
    setChartStreamState(true, terminalState.data.market);
  } catch (error) {
    setChartStreamState(false, null, error.message);
  } finally {
    terminalState.streamBusy = false;
  }
}

async function fetchSystemDiagnostics() {
  if (terminalState.diagnosticsBusy) return;
  terminalState.diagnosticsBusy = true;
  try {
    const response = await fetch('/api/system-diagnostics', { cache: 'no-store' });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    renderSystemDiagnostics(await response.json());
  } catch (error) {
    renderSystemDiagnostics({
      headline: '시스템 진단 정보를 불러오지 못했습니다.',
      summary: error.message,
      workers: [],
      blockers: [{ label: '진단 API', message: '상태 조회 실패', detail: error.message }],
      files: {},
      recent_activity: [],
    });
  } finally {
    terminalState.diagnosticsBusy = false;
  }
}

function renderSystemDiagnostics(data) {
  terminalState.diagnostics = data;
  const score = Number(data.score || 0);
  const threshold = Number(data.threshold || .9);
  const scorePercent = Math.max(0, Math.min(100, score * 100));
  document.getElementById('diagnostics-summary').textContent =
    `${data.headline || '상태 확인 중'} ${data.summary || ''}`.trim();
  document.getElementById('diagnostics-mode').textContent =
    String(data.mode || 'unknown').toUpperCase();
  document.getElementById('diagnostics-score').textContent =
    `${scorePercent.toFixed(0)}%`;
  document.getElementById('diagnostics-threshold').textContent =
    `실거래 승격 기준 ${(threshold * 100).toFixed(0)}%`;
  document.getElementById('diagnostics-progress-bar').style.width = `${scorePercent}%`;

  const workers = data.workers || [];
  const runningWorkers = workers.filter((worker) => worker.running).length;
  document.getElementById('diagnostics-worker-count').textContent =
    `${runningWorkers}/${workers.length} 실행`;
  document.getElementById('diagnostics-workers').innerHTML = workers.length
    ? workers.map((worker) => `
      <div class="worker-card ${worker.running ? 'running' : ''}">
        <span class="worker-state">${worker.running ? 'RUNNING' : 'WAITING'}</span>
        <b>${escapeHtml(worker.label || worker.key || '-')}</b>
        <small>${escapeHtml(worker.detail || '-')}</small>
      </div>
    `).join('')
    : '<div class="blocker-clear">워커 상태가 없습니다.</div>';

  const blockers = data.blockers || [];
  document.getElementById('diagnostics-blocker-count').textContent =
    blockers.length ? `${blockers.length}개 차단` : '모두 통과';
  document.getElementById('diagnostics-blockers').innerHTML = blockers.length
    ? blockers.map((item) => `
      <div class="blocker-card">
        <b>${escapeHtml(item.label || item.code || '-')}</b>
        <span>${escapeHtml(item.message || '-')}</span>
        <small>${escapeHtml(item.detail || item.code || '-')}</small>
      </div>
    `).join('')
    : '<div class="blocker-clear">실거래 승격 차단 사유가 없습니다.</div>';

  const flows = data.flows || {};
  const researchCounts = (flows.research_collection || {}).counts || {};
  const training = flows.training || {};
  const metrics = training.metrics || {};
  const activeModel = training.active_model || {};
  const activeMetrics = activeModel.metrics || {};
  const deployment = training.deployment || {};
  const market = flows.market_data || {};
  const healthy = market.healthy || {};
  const account = data.account_context || {};
  const files = data.files || {};
  const evidence = [
    [
      '운영 모델',
      `AUC ${Number(activeMetrics.auc || 0).toFixed(4)}`,
      `Precision@K ${Number(activeMetrics.precision_at_k || 0).toFixed(4)} · 실거래 적용 중`,
    ],
    [
      '모델 갱신 상태',
      training.training_skipped ? '새 라벨 대기' : (deployment.deployed ? '개선 모델 교체' : '기존 모델 유지'),
      deployment.reason || (training.training_skipped ? '동일 데이터 반복 학습 생략' : '후보 모델 검증 중'),
    ],
    [
      '누적 학습 행',
      formatInteger(metrics.materialized_training_rows || metrics.training_rows || metrics.example_count),
      `모델 윈도우 ${formatInteger(metrics.training_rows || metrics.example_count)}행 · 이번 주기 성숙 라벨 ${formatInteger(metrics.fresh_training_rows || 0)}개`,
    ],
    [
      '연구 데이터 수집',
      flows.research_collection?.active ? '진행 중' : '중지',
      `이벤트 ${formatInteger(researchCounts.events || researchCounts.events_seen)} · 원문 ${formatInteger(researchCounts.raw_records || researchCounts.raw_records_seen)}`,
    ],
    [
      '시장 스냅샷',
      formatInteger(researchCounts.market_snapshots || researchCounts.market_snapshots_seen),
      `연구 DB ${formatFileSize(files.research_store?.size_bytes)} · ${formatAge(files.research_store?.age_seconds)}`,
    ],
    [
      '새 실시간 학습 프레임',
      formatInteger(training.feature_frames_built),
      `저널 ${formatFileSize(files.feature_journal?.size_bytes)} · ${formatAge(files.feature_journal?.age_seconds)}`,
    ],
    [
      '최신 모델',
      `AUC ${Number(metrics.auc || 0).toFixed(4)}`,
      `Precision@K ${Number(metrics.precision_at_k || 0).toFixed(4)} · ${metrics.live_eligible ? '승격 가능' : '기준 미달'}`,
    ],
    [
      '거래급 시세 종목',
      `KRX ${(healthy.KRX || []).length} · US ${(healthy.US || []).length}`,
      `시장별 최소 ${market.minimum_per_market || '-'}개 필요`,
    ],
    [
      '주문 가능 현금',
      `KRW ${formatCompactNumber(account.krw_orderable)}`,
      `USD ${formatCompactNumber(account.usd_orderable)}${account.us_collection_limited_by_cash ? ' · 미국 후보 제한' : ''}`,
    ],
    [
      '실시간 저장소',
      formatFileSize(files.realtime_store?.size_bytes),
      formatAge(files.realtime_store?.age_seconds),
    ],
    [
      '다음 연구 수집',
      data.next_collection_at ? shortDateTime(data.next_collection_at) : '-',
      `상태 생성 ${shortDateTime(data.generated_at)}`,
    ],
  ];
  document.getElementById('diagnostics-evidence').innerHTML = evidence.map(([label, value, note]) => `
    <div class="evidence-card">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      <small>${escapeHtml(note)}</small>
    </div>
  `).join('');

  const activity = [...(data.recent_activity || [])].reverse().slice(0, 8);
  document.getElementById('diagnostics-activity').innerHTML = activity.length
    ? activity.map((item) => `
      <div class="activity-row">
        <time>${escapeHtml(shortClock(item.timestamp))}</time>
        <b>${escapeHtml(item.status || '-')}</b>
        <span title="${escapeHtml(item.message || '')}">${escapeHtml(item.message || '-')}</span>
      </div>
    `).join('')
    : '<div class="blocker-clear">최근 활동 기록이 없습니다.</div>';
  document.getElementById('diagnostics-generated-at').textContent =
    `갱신 ${shortClock(data.generated_at)}`;
  document.getElementById('diagnostics-next-run').textContent =
    data.next_collection_at ? `다음 ${shortDateTime(data.next_collection_at)}` : '다음 일정 없음';
  renderTrainingMonitor((data.flows || {}).training || {});
}

function renderTrainingMonitor(training) {
  const history = training.history || {};
  const points = Array.isArray(history.points) ? history.points : [];
  const latest = history.latest || points.at(-1) || {};
  const change = history.change || {};
  const optimizer = history.optimizer || {};
  const state = document.getElementById('training-cycle-state');
  if (!state) return;

  const statusLabels = {
    waiting: 'WAITING',
    collecting: 'COLLECTING',
    promoted: 'MODEL PROMOTED',
    evaluated_improved: 'IMPROVED',
    evaluated: 'EVALUATED',
  };
  state.textContent = statusLabels[history.status] || String(history.status || 'WAITING').toUpperCase();
  document.getElementById('training-last-cycle').textContent =
    latest.timestamp ? `최근 학습 ${shortDateTime(latest.timestamp)}` : '학습 기록 없음';
  document.getElementById('training-monitor-note').textContent =
    history.note || '완료된 실제 학습 사이클을 표시합니다.';

  const learningRate = Number(optimizer.classification_learning_rate);
  document.getElementById('training-learning-rate').textContent =
    Number.isFinite(learningRate) ? learningRate.toFixed(4) : '-';
  document.getElementById('training-optimizer').textContent = optimizer.classification_family
    ? `${String(latest.training_mode || 'full').toUpperCase()} · ${optimizer.classification_family} · ${formatInteger(optimizer.classification_epochs)} epochs · L2 ${optimizer.l2}`
    : '옵티마이저 기록 없음';

  const aucChange = Number(change.auc || 0);
  const aucChangeNode = document.getElementById('training-auc-change');
  aucChangeNode.textContent = points.length > 1 ? formatSignedMetric(aucChange, 4) : '-';
  aucChangeNode.className = aucChange > 0 ? 'positive' : aucChange < 0 ? 'negative' : '';
  document.getElementById('training-current-auc').textContent =
    `현재 AUC ${Number(latest.auc || 0).toFixed(4)} · Precision ${Number(latest.precision_at_k || 0).toFixed(4)}`;

  const rowsPerHour = Number(history.rows_per_hour || 0);
  document.getElementById('training-row-rate').textContent =
    `${rowsPerHour.toFixed(rowsPerHour >= 10 ? 0 : 1)} 행/시간`;
  document.getElementById('training-new-rows').textContent =
    `${latest.training_mode === 'incremental' ? '증분 학습' : '전체 학습'} ${formatInteger(latest.incremental_rows || latest.training_rows || 0)}행 · 누적 ${formatInteger(latest.materialized_rows || latest.training_rows || 0)}행 · 모델 윈도우 ${formatInteger(latest.training_rows || 0)}행`;

  const promoted = Boolean(latest.promoted);
  const eligible = Boolean(latest.live_eligible);
  const deploymentNode = document.getElementById('training-deployment-state');
  deploymentNode.textContent = promoted ? '교체 완료' : eligible ? '승격 후보' : '기존 모델 유지';
  deploymentNode.className = promoted || eligible ? 'positive' : '';
  document.getElementById('training-deployment-reason').textContent =
    latest.deployment_reason || '평가 기록 없음';

  drawTrainingPerformanceChart(points);
  drawTrainingDataChart(points);
  const cycles = points.slice(-10).reverse();
  document.getElementById('training-cycle-list').innerHTML = cycles.length
    ? cycles.map((point) => `
      <article class="training-cycle ${point.promoted ? 'promoted' : point.live_eligible ? 'eligible' : ''}">
        <time>${escapeHtml(shortClock(point.timestamp))}</time>
        <b>AUC ${Number(point.auc || 0).toFixed(4)}</b>
        <small>${String(point.training_mode || 'full').toUpperCase()} ${formatInteger(point.incremental_rows || point.training_rows || 0)}행 · P@K ${Number(point.precision_at_k || 0).toFixed(3)}</small>
        <small>${point.promoted ? '모델 교체' : point.live_eligible ? '승격 가능' : escapeHtml(point.deployment_reason || '평가 완료')}</small>
      </article>
    `).join('')
    : '<div class="blocker-clear">아직 완료된 학습 사이클이 없습니다.</div>';
}

function drawTrainingPerformanceChart(points) {
  const canvas = document.getElementById('training-performance-chart');
  if (!canvas) return;
  const { ctx, width, height } = prepareCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 34, right: 36, top: 12, bottom: 22 };
  const plotWidth = Math.max(1, width - pad.left - pad.right);
  const plotHeight = Math.max(1, height - pad.top - pad.bottom);
  drawTrainingGrid(ctx, width, height, pad, ['1.00', '0.75', '0.50', '0.25', '0.00']);
  if (!points.length) {
    drawEmptyTrainingChart(ctx, width, height);
    return;
  }
  const xAt = (index) => pad.left + (points.length === 1 ? plotWidth / 2 : index * plotWidth / (points.length - 1));
  const scoreY = (value) => pad.top + (1 - Math.max(0, Math.min(1, Number(value) || 0))) * plotHeight;
  drawTrainingLine(ctx, points, xAt, (point) => scoreY(point.auc), '#39d7e7');
  drawTrainingLine(ctx, points, xAt, (point) => scoreY(point.precision_at_k), '#9b8cff');

  const returns = points.map((point) => Number(point.top_return_bps || 0));
  let returnMin = Math.min(0, ...returns);
  let returnMax = Math.max(0, ...returns);
  if (returnMax === returnMin) returnMax = returnMin + 1;
  const returnY = (value) => pad.top + (returnMax - Number(value || 0)) / (returnMax - returnMin) * plotHeight;
  drawTrainingLine(ctx, points, xAt, (point) => returnY(point.top_return_bps), '#f3b95f', [4, 3]);
  ctx.fillStyle = '#708196';
  ctx.font = '7px Consolas';
  ctx.textAlign = 'left';
  ctx.fillText(`${returnMax.toFixed(1)}bp`, width - pad.right + 3, pad.top + 3);
  ctx.fillText(`${returnMin.toFixed(1)}bp`, width - pad.right + 3, height - pad.bottom);
  drawTrainingTimeLabels(ctx, points, xAt, height);
}

function drawTrainingDataChart(points) {
  const canvas = document.getElementById('training-data-chart');
  if (!canvas) return;
  const { ctx, width, height } = prepareCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 42, right: 12, top: 12, bottom: 22 };
  const plotWidth = Math.max(1, width - pad.left - pad.right);
  const plotHeight = Math.max(1, height - pad.top - pad.bottom);
  if (!points.length) {
    drawEmptyTrainingChart(ctx, width, height);
    return;
  }
  const rowValues = points.map((point) => Number(point.materialized_rows || point.training_rows || 0));
  const newValues = points.map((point) => Number(point.new_rows || 0));
  const rowMin = Math.min(...rowValues);
  const rowMax = Math.max(...rowValues);
  const rowSpan = Math.max(1, rowMax - rowMin);
  const newMax = Math.max(1, ...newValues);
  const xAt = (index) => pad.left + (points.length === 1 ? plotWidth / 2 : index * plotWidth / (points.length - 1));
  const rowY = (value) => pad.top + (rowMax - Number(value || 0)) / rowSpan * plotHeight;
  const barWidth = Math.max(2, Math.min(9, plotWidth / Math.max(1, points.length) * .55));
  ctx.fillStyle = 'rgba(57, 215, 231, .25)';
  points.forEach((point, index) => {
    const barHeight = Number(point.new_rows || 0) / newMax * plotHeight * .42;
    ctx.fillRect(xAt(index) - barWidth / 2, pad.top + plotHeight - barHeight, barWidth, barHeight);
  });
  drawTrainingLine(ctx, points, xAt, (point) => rowY(point.materialized_rows || point.training_rows), '#42d392');
  ctx.fillStyle = '#708196';
  ctx.font = '7px Consolas';
  ctx.textAlign = 'right';
  ctx.fillText(formatInteger(rowMax), pad.left - 4, pad.top + 3);
  ctx.fillText(formatInteger(rowMin), pad.left - 4, height - pad.bottom);
  ctx.strokeStyle = '#1c2937';
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, height - pad.bottom);
  ctx.lineTo(width - pad.right, height - pad.bottom);
  ctx.stroke();
  drawTrainingTimeLabels(ctx, points, xAt, height);
}

function drawTrainingGrid(ctx, width, height, pad, labels) {
  const plotHeight = height - pad.top - pad.bottom;
  ctx.font = '7px Consolas';
  ctx.textAlign = 'right';
  labels.forEach((label, index) => {
    const y = pad.top + index * plotHeight / (labels.length - 1);
    ctx.strokeStyle = '#172330';
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillStyle = '#66778c';
    ctx.fillText(label, pad.left - 5, y + 2);
  });
}

function drawTrainingLine(ctx, points, xAt, yAt, color, dash = []) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.setLineDash(dash);
  ctx.beginPath();
  points.forEach((point, index) => {
    const x = xAt(index);
    const y = yAt(point);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.restore();
}

function drawTrainingTimeLabels(ctx, points, xAt, height) {
  const indices = [...new Set([0, Math.floor((points.length - 1) / 2), points.length - 1])];
  ctx.fillStyle = '#66778c';
  ctx.font = '7px Consolas';
  indices.forEach((index) => {
    ctx.textAlign = index === 0 ? 'left' : index === points.length - 1 ? 'right' : 'center';
    ctx.fillText(shortClock(points[index]?.timestamp), xAt(index), height - 6);
  });
}

function drawEmptyTrainingChart(ctx, width, height) {
  ctx.fillStyle = '#66778c';
  ctx.font = '9px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('학습 이력 대기 중', width / 2, height / 2);
}

function formatSignedMetric(value, digits = 4) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '-';
  return `${number > 0 ? '+' : ''}${number.toFixed(digits)}`;
}

function renderAssetSummary(data) {
  const snapshot = data.snapshot || null;
  const live = data.status === 'live' && data.authoritative;
  const lastKnown = data.status === 'last_known';
  const status = document.getElementById('asset-status');
  status.textContent = live ? '실계좌 LIVE' : lastKnown ? '마지막 확인값' : '확인 불가';
  status.className = live ? 'status-chip' : 'status-chip blocked';
  document.getElementById('asset-verified-at').textContent = snapshot
    ? `마지막 확인 ${formatTime(data.last_verified_at)}`
    : '마지막 확인 -';
  document.getElementById('asset-source').textContent = data.message || '계좌 상태를 확인할 수 없습니다.';
  document.getElementById('asset-total').textContent = snapshot ? formatKrw(snapshot.total_asset_krw) : '-';
  document.getElementById('asset-cash').textContent = snapshot ? formatKrw(snapshot.cash_equivalent_krw) : '-';
  document.getElementById('asset-cash-detail').textContent = snapshot
    ? `원화 ${formatKrw(snapshot.krw_cash)} · 외화 ${formatKrw(snapshot.foreign_cash_krw)}`
    : '원화 - · 외화 -';
  const stockValue = snapshot
    ? Number(snapshot.domestic_stock_value_krw || 0) + Number(snapshot.overseas_stock_value_krw || 0)
    : null;
  document.getElementById('asset-stocks').textContent = snapshot ? formatKrw(stockValue) : '-';
  document.getElementById('asset-stock-detail').textContent = snapshot
    ? `국내 ${formatKrw(snapshot.domestic_stock_value_krw)} · 해외 ${formatKrw(snapshot.overseas_stock_value_krw)}`
    : '국내 - · 해외 -';
  const pnl = snapshot ? Number(snapshot.total_pnl_krw || 0) : null;
  const pnlElement = document.getElementById('asset-pnl');
  pnlElement.textContent = snapshot ? formatSignedKrw(pnl) : '-';
  pnlElement.className = snapshot ? (pnl > 0 ? 'positive' : pnl < 0 ? 'negative' : '') : '';
  document.getElementById('asset-pnl-rate').textContent = snapshot
    ? `수익률 ${formatPercent(snapshot.total_pnl_rate)}`
    : '수익률 -';
  const warning = document.getElementById('asset-warning');
  warning.textContent = live ? '' : (data.message || '실계좌 연결 상태를 확인해 주세요.');
  warning.hidden = live;
  document.querySelector('.asset-overview')?.classList.toggle('asset-degraded', !live);
}

function renderTerminal(data) {
  const market = data.market || {};
  const selection = data.selection || {};
  const algorithm = data.algorithm || null;
  renderHeader(data, market);
  renderLiveOwner(data);
  renderCandidates(data.candidates || []);
  renderInstrument(data.symbol, market, selection, algorithm);
  renderTradingChart(market, algorithm);
  renderSecondAnalysis(market);
  renderDecisionOntology(data.decision_ontology || {});
  renderOntology(selection, market);
  renderExecution(data.execution || {}, market.latest_orderbook);
  renderSafety(data);
}

function renderLiveOwner(data) {
  const live = data.live_trading || {};
  const session = data.strategy_session || {};
  const execution = data.execution || {};
  const events = execution.events || [];
  const latestEvent = events[0] || {};
  const latestPayload = latestEvent.payload || {};
  const adopted = Boolean(live.adopted && live.symbol && live.strategy_id);
  const running = Boolean(live.engine_running);
  const capable = Boolean(data.live_order_capable);
  const container = document.querySelector('.live-owner-strip');
  container?.classList.toggle('scanning', running && !adopted);
  container?.classList.toggle('blocked', !running || !capable);

  document.getElementById('live-owner-state').textContent = running
    ? (adopted ? String(live.phase || 'ACTIVE') : 'SCANNING')
    : 'ENGINE OFFLINE';
  document.getElementById('live-owner-session').textContent = adopted
    ? `SESSION ${String(live.session_id || '-').slice(-12)}`
    : (live.last_reason || live.buy_disabled_reason || '채택 가능한 신호 탐색 중');
  document.getElementById('live-owner-symbol').textContent = adopted ? live.symbol : '채택 대기';
  document.getElementById('live-owner-market').textContent = adopted
    ? `${session.macro_regime || 'MACRO'} · ${session.micro_regime || 'MICRO'}`
    : `현재 화면 ${data.symbol || '-'}`;
  document.getElementById('live-owner-strategy').textContent = adopted
    ? (strategyLabels[live.strategy_id] || live.strategy_id)
    : '채택 대기';
  document.getElementById('live-owner-source').textContent = adopted
    ? `${live.selection_source || 'ONTOLOGY'} 선택 · ${live.execution_authority || live.strategy_id} 단독 실행`
    : '후보 판단은 주문 권한과 분리';
  const orderState = String(latestPayload.status || latestEvent.event_type || '').toUpperCase();
  document.getElementById('live-owner-order').textContent =
    orderState || (capable ? 'KIS ARMED' : 'ORDER BLOCKED');
  document.getElementById('live-owner-cycle').textContent =
    `마지막 주기 ${shortClock(live.last_cycle_at)} · ${execution.event_count || 0} events`;
}

function renderHeader(data, market) {
  const stale = Boolean(market.stale);
  const connection = document.querySelector('.connection');
  connection?.classList.toggle('stale', stale);
  document.getElementById('feed-state').textContent = stale
    ? '시세 정체 · 자동 재연결 중'
    : market.feed_state === 'LIVE_QUOTE_ONLY'
      ? '실시간 호가 수신 · 체결 대기'
      : '실시간 체결·호가 수신';
  const mode = document.getElementById('terminal-mode');
  mode.textContent = `${String(data.mode || 'unknown').toUpperCase()} · ${data.live_order_capable ? 'ORDER ARMED' : 'ORDER BLOCKED'}`;
  mode.className = data.live_order_capable ? 'status-chip' : 'status-chip blocked';
}

function setChartStreamState(connected, market = null, error = '') {
  const target = document.getElementById('chart-stream-state');
  if (!target) return;
  const fresh = connected && market && !market.stale;
  target.classList.toggle('connected', Boolean(fresh));
  target.classList.toggle('stale', !fresh);
  const label = target.querySelector('span');
  if (label) {
    label.textContent = fresh
      ? `${market.feed_state === 'LIVE_QUOTE_ONLY' ? 'QUOTE MID' : 'LIVE TRADE'} · ${shortClock(market.last_event_at)}`
      : connected
        ? `스트림 대기 · ${market?.microstructure?.block_reason || 'STALE'}`
        : `연결 재시도 · ${error || 'WAITING'}`;
  }
}

function registerSecondBarArrival(market, reset = false) {
  const bars = market.second_bars || [];
  const latestTime = bars[bars.length - 1]?.time || null;
  if (reset || terminalState.lastSecondBarTime === null) {
    terminalState.lastSecondBarTime = latestTime;
    terminalState.chartTransitionActive = false;
    terminalState.streamReceivedAt = performance.now() - 1000;
    return;
  }
  if (latestTime && latestTime !== terminalState.lastSecondBarTime) {
    terminalState.lastSecondBarTime = latestTime;
    terminalState.chartTransitionActive = true;
    terminalState.streamReceivedAt = performance.now();
  }
}

function renderCandidates(rows) {
  const container = document.getElementById('candidate-list');
  container.innerHTML = rows.length ? rows.map((row) => `
    <button type="button" class="candidate ${row.selected ? 'selected' : ''}" data-symbol="${escapeHtml(row.symbol)}">
      <strong>${escapeHtml(row.symbol)}</strong>
      <em>${row.ontology_allowed ? 'ALLOWED' : 'WATCH'}</em>
      <small>${escapeHtml(strategyLabels[row.strategy_id] || row.strategy_id || row.action || 'NO_TRADE')}</small>
    </button>
  `).join('') : '<span class="tape-empty">온톨로지 후보를 기다리고 있습니다.</span>';
  container.querySelectorAll('.candidate').forEach((button) => {
    button.addEventListener('click', () => fetchMarketView(button.dataset.symbol));
  });
}

function renderInstrument(symbol, market, selection, algorithm) {
  const session = terminalState.data?.strategy_session || {};
  const live = terminalState.data?.live_trading || {};
  document.getElementById('instrument-symbol').textContent = symbol || '선택 대기';
  document.getElementById('instrument-price').textContent = formatPrice(market.last_price);
  const change = document.getElementById('instrument-change');
  const rate = Number(market.change_rate);
  change.textContent = Number.isFinite(rate) ? `${rate >= 0 ? '+' : ''}${(rate * 100).toFixed(2)}% / chart range` : '-';
  change.className = Number.isFinite(rate) ? (rate >= 0 ? 'positive' : 'negative') : '';
  const bars = market.bars || [];
  const latest = bars[bars.length - 1] || {};
  const tick = market.latest_tick || {};
  const book = market.latest_orderbook || {};
  const stats = [
    ['OPEN', latest.open],
    ['HIGH', latest.high],
    ['LOW', latest.low],
    ['VOLUME', latest.volume],
    ['SPREAD', book.spread_bps === null || book.spread_bps === undefined ? '-' : `${Number(book.spread_bps).toFixed(1)} bps`],
    ['LATENCY', tick.latency_ms === null || tick.latency_ms === undefined ? '-' : `${Number(tick.latency_ms).toFixed(1)} ms`],
  ];
  document.getElementById('instrument-stats').innerHTML = stats.map(([label, value]) => `
    <div class="instrument-stat"><span>${label}</span><strong>${typeof value === 'number' ? formatCompact(value) : escapeHtml(value ?? '-')}</strong></div>
  `).join('');

  const adoptedStrategyId = live.adopted ? (live.strategy_id || session.selected_strategy) : null;
  const active = Boolean(live.adopted && adoptedStrategyId);
  document.getElementById('algorithm-state').textContent = String(live.phase || session.phase || 'SCANNING');
  document.getElementById('algorithm-name').textContent = active
    ? (strategyLabels[adoptedStrategyId] || adoptedStrategyId)
    : '실거래 전략 채택 대기';
  document.getElementById('algorithm-thesis').textContent = active
    ? `${algorithm?.thesis || '온톨로지·GNN 채택 전략'} · ${session.last_reason || '실시간 감시 중'}`
    : `${session.last_reason || '매수·매도 조건을 평가 중입니다.'} · 후보 모델의 판단은 실제 채택 전까지 주문되지 않습니다.`;
  const tags = algorithm ? [
    ...(algorithm.visual_indicators || []),
    `STOP ${algorithm.stop_bps}bps`,
    `TARGET ${algorithm.profit_bps}bps`,
    `HOLD ${algorithm.max_holding_seconds}s`,
  ] : ['CLOSED-WORLD', 'NoTrade', 'OWNER-LOCK'];
  if (session.session_id) {
    tags.unshift(
      `OWNER ${String(session.session_id).slice(-8)}`,
      session.selection_source || 'ONTOLOGY',
    );
    if (Number(session.target_price) > 0) tags.push(`목표 ${formatPrice(session.target_price)}`);
    if (Number(session.stop_loss_rate) > 0) tags.push(`전략 손절 ${(Number(session.stop_loss_rate) * 100).toFixed(2)}%`);
    if (Number(session.trailing_stop_rate) > 0) tags.push(`추적 손절 ${(Number(session.trailing_stop_rate) * 100).toFixed(2)}%`);
    if (session.exit_reason) tags.push(`EXIT ${session.exit_reason}`);
  } else if (session.phase) {
    tags.unshift(session.phase, session.macro_regime || 'MACRO WAITING');
    const candidateReason = session.candidate_diagnostics?.[0]?.reason_codes?.[0];
    if (candidateReason) tags.push(candidateReason);
  }
  document.getElementById('algorithm-tags').innerHTML = tags.map((tag) => `<span>${escapeHtml(tag)}</span>`).join('');
}

function renderTradingChart(market, algorithm = null) {
  const secondBars = market.second_bars || [];
  const useSeconds = terminalState.chartMode === 'seconds' && secondBars.length > 0;
  const bars = useSeconds ? secondBars.slice(-90) : (market.bars || []);
  renderChart(bars, algorithm, market.last_price, useSeconds ? 'seconds' : 'minutes');
}

function renderChart(bars, algorithm = null, referencePrice = null, timeframe = 'minutes') {
  const empty = document.getElementById('chart-empty');
  empty.style.display = bars.length ? 'none' : 'grid';
  const live = timeframe === 'seconds';
  const progress = live && terminalState.chartTransitionActive
    ? Math.min(1, Math.max(0, (performance.now() - terminalState.streamReceivedAt) / 420))
    : 1;
  if (progress >= 1) terminalState.chartTransitionActive = false;
  drawPriceChart(
    document.getElementById('price-chart'),
    bars,
    algorithm,
    referencePrice,
    { live, progress, maximumPoints: live ? 90 : bars.length },
  );
  drawVolumeChart(
    document.getElementById('volume-chart'),
    bars,
    { live, progress, maximumPoints: live ? 90 : bars.length },
  );
  document.getElementById('chart-range').textContent = live
    ? `실시간 이동 창 · 최근 ${bars.length}개 1초봉 · 체결봉 + 실제 호가 중간값봉 · MA5 / MA20 / VWAP`
    : `최근 ${bars.length}개 1분봉 · MA5 / MA20 / VWAP`;
  const last = bars[bars.length - 1];
  document.getElementById('chart-updated').textContent = `${live ? (last?.bar_source === 'quote_mid' ? 'QUOTE MID' : 'LIVE TRADE') : '마지막 이벤트'} ${formatTime(last?.time)}`;
}

function drawPriceChart(canvas, bars, algorithm, referencePrice, options = {}) {
  const { ctx, width, height } = prepareCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#0d131c';
  ctx.fillRect(0, 0, width, height);
  if (!bars.length) return;
  const pad = { left: 12, right: 62, top: 15, bottom: 24 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const lows = bars.map((bar) => Number(bar.low));
  const highs = bars.map((bar) => Number(bar.high));
  const reference = Number(referencePrice || bars[bars.length - 1]?.close);
  if (algorithm && Number.isFinite(reference)) {
    lows.push(reference * (1 - Number(algorithm.stop_bps || 0) / 10000));
    highs.push(reference * (1 + Number(algorithm.profit_bps || 0) / 10000));
  }
  const priceMin = Math.min(...lows);
  const priceMax = Math.max(...highs);
  const span = Math.max(priceMax - priceMin, Math.abs(priceMax) * .001, 1e-6);
  const min = priceMin - span * .05;
  const max = priceMax + span * .05;
  const y = (value) => pad.top + (max - Number(value)) / (max - min) * plotHeight;
  const maximumPoints = Math.max(1, Number(options.maximumPoints || bars.length));
  const step = plotWidth / maximumPoints;
  const progress = Number(options.progress || 0);
  const xForIndex = (index) => options.live
    ? width - pad.right - step * (bars.length - index - .5) + step * (1 - progress)
    : pad.left + step * (index + .5);

  ctx.strokeStyle = '#1c2734';
  ctx.fillStyle = '#6f8095';
  ctx.font = '9px Consolas';
  ctx.textAlign = 'left';
  for (let i = 0; i <= 5; i += 1) {
    const yy = pad.top + plotHeight * i / 5;
    ctx.beginPath();
    ctx.moveTo(pad.left, yy);
    ctx.lineTo(width - pad.right, yy);
    ctx.stroke();
    ctx.fillText(formatPrice(max - (max - min) * i / 5), width - pad.right + 7, yy + 3);
  }
  for (let i = 0; i < bars.length; i += 1) {
    const bar = bars[i];
    const x = xForIndex(i);
    const quoteOnly = bar.bar_source === 'quote_mid';
    const up = Number(bar.close) >= Number(bar.open);
    const color = quoteOnly ? '#39d7e7' : (up ? '#42d392' : '#ff6678');
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, y(bar.high));
    ctx.lineTo(x, y(bar.low));
    ctx.stroke();
    const bodyTop = Math.min(y(bar.open), y(bar.close));
    const bodyHeight = Math.max(1, Math.abs(y(bar.open) - y(bar.close)));
    const bodyWidth = Math.max(1, Math.min(7, step * .66));
    if (quoteOnly) {
      ctx.strokeRect(x - bodyWidth / 2, bodyTop - 1, bodyWidth, Math.max(2, bodyHeight + 2));
    } else {
      ctx.fillRect(x - bodyWidth / 2, bodyTop, bodyWidth, bodyHeight);
    }
  }
  drawLine(ctx, movingAverage(bars.map((bar) => Number(bar.close)), 5), y, pad.left, step, '#39d7e7', 1.3, [], xForIndex);
  drawLine(ctx, movingAverage(bars.map((bar) => Number(bar.close)), 20), y, pad.left, step, '#9b8cff', 1.3, [], xForIndex);
  drawLine(ctx, bars.map((bar) => Number(bar.vwap) || null), y, pad.left, step, '#f3b95f', 1, [4, 3], xForIndex);
  if (algorithm && Number.isFinite(reference)) {
    drawLevel(ctx, width, height, pad, y, reference * (1 + Number(algorithm.profit_bps || 0) / 10000), '#42d392', 'TARGET');
    drawLevel(ctx, width, height, pad, y, reference * (1 - Number(algorithm.stop_bps || 0) / 10000), '#ff6678', 'STOP');
  }

  ctx.fillStyle = '#6f8095';
  ctx.textAlign = 'center';
  const labels = Math.min(6, bars.length);
  for (let i = 0; i < labels; i += 1) {
    const index = Math.round(i * (bars.length - 1) / Math.max(1, labels - 1));
    const x = xForIndex(index);
    ctx.fillText(options.live ? shortClock(bars[index].time) : shortTime(bars[index].time), x, height - 7);
  }
  if (options.live && bars.length) {
    const lastIndex = bars.length - 1;
    const livePrice = Number(referencePrice || bars[lastIndex].close);
    const liveX = xForIndex(lastIndex);
    const liveY = y(livePrice);
    const pulse = .55 + Math.sin(performance.now() / 170) * .25;
    ctx.save();
    ctx.setLineDash([3, 4]);
    ctx.strokeStyle = `rgba(57,215,231,${pulse})`;
    ctx.beginPath();
    ctx.moveTo(pad.left, liveY);
    ctx.lineTo(width - pad.right, liveY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#39d7e7';
    ctx.shadowColor = '#39d7e7';
    ctx.shadowBlur = 8 + pulse * 7;
    ctx.beginPath();
    ctx.arc(liveX, liveY, 2.8 + pulse * 1.8, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.font = '700 9px Consolas';
    ctx.textAlign = 'right';
    ctx.fillText(`LIVE ${formatPrice(livePrice)}`, width - pad.right - 4, liveY - 6);
    ctx.restore();
  }
}

function drawLevel(ctx, width, height, pad, y, value, color, label) {
  const yy = y(value);
  if (yy < pad.top || yy > height - pad.bottom) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(pad.left, yy);
  ctx.lineTo(width - pad.right, yy);
  ctx.stroke();
  ctx.font = '8px Consolas';
  ctx.textAlign = 'right';
  ctx.fillText(`${label} ${formatPrice(value)}`, width - pad.right - 4, yy - 3);
  ctx.restore();
}

function drawLine(ctx, values, y, left, step, color, width, dash = [], xForIndex = null) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.beginPath();
  let started = false;
  values.forEach((value, index) => {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return;
    const x = xForIndex ? xForIndex(index) : left + step * (index + .5);
    if (!started) {
      ctx.moveTo(x, y(value));
      started = true;
    } else {
      ctx.lineTo(x, y(value));
    }
  });
  if (started) ctx.stroke();
  ctx.restore();
}

function drawVolumeChart(canvas, bars, options = {}) {
  const { ctx, width, height } = prepareCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#0d131c';
  ctx.fillRect(0, 0, width, height);
  if (!bars.length) return;
  const left = 12;
  const right = 62;
  const plotWidth = width - left - right;
  const maxVolume = Math.max(...bars.map((bar) => Number(bar.volume || 0)), 1);
  const maximumPoints = Math.max(1, Number(options.maximumPoints || bars.length));
  const step = plotWidth / maximumPoints;
  const progress = Number(options.progress || 0);
  bars.forEach((bar, index) => {
    const value = Number(bar.volume || 0);
    const h = value / maxVolume * (height - 10);
    const x = options.live
      ? width - right - step * (bars.length - index) + step * (1 - progress)
      : left + step * index;
    if (bar.bar_source !== 'quote_mid') {
      ctx.fillStyle = Number(bar.close) >= Number(bar.open) ? 'rgba(66,211,146,.5)' : 'rgba(255,102,120,.5)';
      ctx.fillRect(x, height - h, Math.max(1, step * .7), h);
    }
  });
}

function renderSecondAnalysis(market) {
  const bars = market.second_bars || [];
  const micro = market.microstructure || {};
  const status = document.getElementById('second-data-status');
  if (!status) return;
  status.textContent = micro.ready ? 'SECOND DATA READY' : `BLOCKED · ${micro.block_reason || 'WAITING'}`;
  status.className = micro.ready ? 'status-chip' : 'status-chip blocked';
  const metrics = [
    ['표시 피드', micro.display_feed_state || 'STALE', 'display_feed_state'],
    ['체결/호가 봉', `${Number(micro.trade_bar_count || 0)} / ${Number(micro.quote_bar_count || 0)}`, 'trade_bar_count'],
    ['1초 수익률', formatSecondReturn(micro.return_1s), 'return_1s'],
    ['5초 수익률', formatSecondReturn(micro.return_5s), 'return_5s'],
    ['10초 수익률', formatSecondReturn(micro.return_10s), 'return_10s'],
    ['1초 체결', `${Number(micro.tick_count_1s || 0)}건`, 'tick_count_1s'],
    ['5초 체결', `${Number(micro.tick_count_5s || 0)}건`, 'tick_count_5s'],
    ['5초 거래량', formatCompact(micro.volume_5s), 'volume_5s'],
    ['5초 체결강도', formatSignedDecimal(micro.aggressor_imbalance_5s), 'aggressor_imbalance_5s'],
    ['스프레드 변화', micro.spread_change_5s_bps === null || micro.spread_change_5s_bps === undefined ? '-' : `${formatSignedDecimal(micro.spread_change_5s_bps)} bps`, 'spread_change_5s_bps'],
    ['호가 불균형 변화', formatSignedDecimal(micro.orderbook_imbalance_change_5s), 'orderbook_imbalance_change_5s'],
  ];
  document.getElementById('second-metrics').innerHTML = metrics.map(([label, value, key]) => {
    const numeric = Number(micro[key]);
    const direction = Number.isFinite(numeric) ? (numeric > 0 ? 'positive' : numeric < 0 ? 'negative' : '') : '';
    return `<article><span>${escapeHtml(label)}</span><strong class="${direction}">${escapeHtml(value)}</strong></article>`;
  }).join('');
  document.getElementById('second-analysis-time').textContent = `최근 초단위 이벤트 ${formatTime(micro.as_of)}`;
  document.getElementById('second-analysis-gate').textContent = micro.ready
    ? '초단위 진입 게이트 통과 · 모델과 온톨로지가 이 데이터로 진입 시점을 재평가합니다.'
    : `실매수 차단: ${micro.block_reason || 'SECOND_LEVEL_DATA_NOT_READY'}`;
  const empty = document.getElementById('second-chart-empty');
  empty.style.display = bars.length ? 'none' : 'grid';
  drawSecondPriceChart(document.getElementById('second-price-chart'), bars);
}

function drawSecondPriceChart(canvas, allBars) {
  const bars = allBars.slice(-90);
  const { ctx, width, height } = prepareCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#090f17';
  ctx.fillRect(0, 0, width, height);
  if (!bars.length) return;
  const pad = { left: 10, right: 58, top: 13, bottom: 24 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const prices = bars.flatMap((bar) => [Number(bar.high), Number(bar.low)]).filter(Number.isFinite);
  const minimum = Math.min(...prices);
  const maximum = Math.max(...prices);
  const span = Math.max(maximum - minimum, Math.abs(maximum) * .0001, 1e-6);
  const low = minimum - span * .08;
  const high = maximum + span * .08;
  const y = (value) => pad.top + (high - Number(value)) / (high - low) * plotHeight;
  const step = plotWidth / Math.max(1, bars.length);
  ctx.strokeStyle = '#1b2836';
  ctx.fillStyle = '#66788e';
  ctx.font = '8px Consolas';
  for (let index = 0; index <= 4; index += 1) {
    const yy = pad.top + plotHeight * index / 4;
    ctx.beginPath();
    ctx.moveTo(pad.left, yy);
    ctx.lineTo(width - pad.right, yy);
    ctx.stroke();
    ctx.fillText(formatPrice(high - (high - low) * index / 4), width - pad.right + 6, yy + 3);
  }
  bars.forEach((bar, index) => {
    const x = pad.left + step * (index + .5);
    const quoteOnly = bar.bar_source === 'quote_mid';
    const up = Number(bar.close) >= Number(bar.open);
    const imbalance = Number(bar.aggressor_imbalance || 0);
    ctx.strokeStyle = quoteOnly ? '#39d7e7' : (up ? '#42d392' : '#ff6678');
    ctx.fillStyle = quoteOnly ? '#39d7e7' : (up ? '#42d392' : '#ff6678');
    ctx.beginPath();
    ctx.moveTo(x, y(bar.high));
    ctx.lineTo(x, y(bar.low));
    ctx.stroke();
    const bodyTop = Math.min(y(bar.open), y(bar.close));
    const bodyWidth = Math.max(1, step * .56);
    const bodyHeight = Math.max(1, Math.abs(y(bar.open) - y(bar.close)));
    if (quoteOnly) {
      ctx.strokeRect(x - Math.max(1, step * .28), bodyTop - 1, bodyWidth, Math.max(2, bodyHeight + 2));
    } else {
      ctx.fillRect(x - Math.max(1, step * .28), bodyTop, bodyWidth, bodyHeight);
    }
    const flowHeight = Math.min(12, Math.abs(imbalance) * 12);
    ctx.fillStyle = imbalance >= 0 ? 'rgba(66,211,146,.5)' : 'rgba(255,102,120,.5)';
    ctx.fillRect(x - Math.max(1, step * .28), height - pad.bottom - flowHeight, Math.max(1, step * .56), flowHeight);
  });
  ctx.fillStyle = '#66788e';
  ctx.textAlign = 'center';
  const labelIndexes = [0, Math.floor((bars.length - 1) / 2), bars.length - 1];
  labelIndexes.forEach((index) => {
    const x = pad.left + step * (index + .5);
    ctx.fillText(shortClock(bars[index].time), x, height - 7);
  });
}

function formatSecondReturn(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number >= 0 ? '+' : ''}${(number * 100).toFixed(3)}%` : '-';
}

function formatSignedDecimal(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number >= 0 ? '+' : ''}${number.toFixed(3)}` : '-';
}

function renderOntology(selection, market) {
  const allowed = Boolean(selection.ontology_allowed);
  const ontologyStrategyId = selection.ontology_strategy_id || selection.strategy_id || null;
  const finalAction = String(selection.action || 'NO_TRADE').toUpperCase();
  const utilityPassed = finalAction !== 'NO_TRADE';
  const status = document.getElementById('ontology-status');
  status.textContent = allowed ? 'ONTOLOGY ALLOWED' : 'NO TRADE';
  status.className = allowed ? 'status-chip' : 'status-chip blocked';
  const nodes = [
    ['데이터 신선도', market.stale ? 'STALE · 신규 진입 차단' : 'FRESH', !market.stale],
    ['운영 사실 검증', allowed ? '필수 사실 충족' : '필수 사실 미충족', allowed],
    ['전략 호환성', ontologyStrategyId || '허용 전략 없음', allowed],
    ['순효용·불확실성', selection.utility === null || selection.utility === undefined ? 'NON_POSITIVE_NET_EDGE' : `utility ${Number(selection.utility).toFixed(3)}`, utilityPassed],
    ['최종 라우팅', finalAction, utilityPassed],
  ];
  document.getElementById('ontology-flow').innerHTML = nodes.map(([label, detail, pass]) => `
    <div class="ontology-node ${pass ? 'pass' : 'block'}"><i></i><b>${escapeHtml(label)}</b><small>${escapeHtml(detail)}</small></div>
  `).join('');
  const reasons = selection.reason_codes || [];
  document.getElementById('ontology-reasons').innerHTML = reasons.length
    ? reasons.map((reason) => `<span class="reason-chip">${escapeHtml(reason)}</span>`).join('')
    : `<span class="reason-chip">${allowed ? 'ALL_REQUIRED_FACTS_VALID' : 'NO_ADMISSIBLE_STRATEGY'}</span>`;
  document.getElementById('decision-compare').innerHTML = (selection.all_decisions || []).map((decision) => `
    <div class="decision-row">
      <b>${escapeHtml(decision.path || '-')}</b>
      <span>${escapeHtml(decision.action || '-')}</span>
      <span>${escapeHtml(strategyLabels[decision.strategy_id] || decision.strategy_id || 'NoTrade')}</span>
    </div>
  `).join('');
}

function renderDecisionOntology(trace) {
  const svg = document.getElementById('decision-ontology-graph');
  if (!svg) return;
  const sources = trace.sources || [];
  const indicators = trace.indicators || [];
  const algorithms = trace.algorithms || [];
  const ontology = trace.ontology_selection || {};
  const finalDecision = trace.final_decision || {};
  const activeAlgorithm = algorithms.find((item) => item.ontology_selected);
  const activeIndicatorIds = new Set(
    (activeAlgorithm?.requirements || []).map((item) => item.indicator_id),
  );
  const visibleIndicators = terminalState.ontologyFilter === 'all'
    ? indicators
    : indicators.filter((item) => activeIndicatorIds.has(item.id));
  const visibleSources = terminalState.ontologyFilter === 'all'
    ? sources
    : sources.filter((source) => visibleIndicators.some((item) => item.source_id === source.id));
  const graphSources = visibleSources.length ? visibleSources : sources.filter((source) => source.available).slice(0, 3);
  const graphIndicators = visibleIndicators.length ? visibleIndicators : indicators.filter((item) => item.available).slice(0, 4);
  const graphAlgorithms = terminalState.ontologyFilter === 'all'
    ? algorithms
    : algorithms.filter((item) => item.ontology_selected || item.final_selected);
  const displayedAlgorithms = graphAlgorithms.length ? graphAlgorithms : algorithms;
  const height = Math.max(
    510,
    78 + graphSources.length * 72,
    78 + graphIndicators.length * 62,
    78 + displayedAlgorithms.length * 72,
  );
  const width = 1100;
  const positions = new Map();
  const nodeWidth = 188;
  const layerX = { source: 30, indicator: 300, algorithm: 635, decision: 905 };
  svg.replaceChildren();
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('height', String(height));

  const addText = (x, y, value, className) => {
    const textNode = createSvgElement('text', { x, y, class: className });
    textNode.textContent = value;
    svg.appendChild(textNode);
  };
  addText(layerX.source, 27, '01  SOURCE DATA', 'graph-layer-label');
  addText(layerX.indicator, 27, '02  INDICATORS / FACTS', 'graph-layer-label');
  addText(layerX.algorithm, 27, '03  STRATEGY EXPERTS', 'graph-layer-label');
  addText(layerX.decision, 27, '04  ROUTING', 'graph-layer-label');

  const layoutNodes = (items, layer, gap, formatter) => {
    const total = (items.length - 1) * gap;
    const start = Math.max(52, (height - total) / 2 - 27);
    items.forEach((item, index) => {
      const y = start + index * gap;
      positions.set(`${layer}:${item.id}`, { x: layerX[layer], y, width: nodeWidth, height: 48 });
      drawOntologyNode(svg, {
        ...formatter(item),
        x: layerX[layer],
        y,
        width: nodeWidth,
        height: 48,
      });
    });
  };
  layoutNodes(graphSources, 'source', 72, (item) => ({
    id: `source:${item.id}`,
    title: item.label,
    value: item.available ? `${item.samples || 0} samples` : 'DATA MISSING',
    className: item.available ? '' : 'unavailable',
    detail: {
      kind: 'SOURCE DATA',
      title: item.label,
      value: item.available ? '수신 중' : '데이터 없음',
      description: item.available ? '실시간 지표 계산에 사용되는 원천 데이터입니다.' : '현재 시장 응답에서 이 데이터가 제공되지 않습니다.',
      rows: [['ID', item.id], ['샘플', item.samples || 0], ['갱신', item.updated_at || '-']],
    },
  }));
  layoutNodes(graphIndicators, 'indicator', 62, (item) => ({
    id: `indicator:${item.id}`,
    title: item.label,
    value: item.available ? `score ${formatOntologyScore(item.score)}` : 'UNKNOWN',
    className: item.available ? '' : 'unavailable',
    detail: {
      kind: 'RECONSTRUCTED INDICATOR',
      title: item.label,
      value: item.available ? formatOntologyScore(item.score) : 'N/A',
      description: item.detail,
      rows: [
        ['원시값', formatOntologyRaw(item.raw_value)],
        ['출처', item.source_id],
        ['계보', item.provenance || 'reconstructed'],
      ],
    },
  }));
  layoutNodes(displayedAlgorithms, 'algorithm', 72, (item) => ({
    id: `algorithm:${item.id}`,
    title: strategyLabels[item.id] || item.label || item.id,
    value: item.ontology_selected ? 'ONTOLOGY SELECTED' : item.final_selected ? 'FINAL SELECTED' : 'CANDIDATE',
    className: item.ontology_selected || item.final_selected ? 'selected' : '',
    detail: {
      kind: 'STRATEGY EXPERT',
      title: strategyLabels[item.id] || item.label || item.id,
      value: item.ontology_selected ? '온톨로지 선택' : item.final_selected ? '최종 선택' : '후보',
      description: item.thesis,
      rows: (item.requirements || []).map((rule) => [
        indicators.find((indicator) => indicator.id === rule.indicator_id)?.label || rule.indicator_id,
        `${rule.operator} ${formatOntologyScore(rule.threshold)} · ${rule.passed === null ? 'UNKNOWN' : rule.passed ? 'PASS' : 'FAIL'}`,
      ]),
    },
  }));

  const decisionId = 'decision:final';
  const decisionY = Math.max(52, height / 2 - 24);
  positions.set(decisionId, { x: layerX.decision, y: decisionY, width: 165, height: 55 });
  drawOntologyNode(svg, {
    id: decisionId,
    x: layerX.decision,
    y: decisionY,
    width: 165,
    height: 55,
    title: finalDecision.action || 'NO_TRADE',
    value: `${String(finalDecision.path || 'ontology').toUpperCase()} ROUTER`,
    className: finalDecision.action === 'NO_TRADE' ? 'blocked' : 'selected',
    detail: {
      kind: 'AUTHORITATIVE DECISION',
      title: finalDecision.action || 'NO_TRADE',
      value: finalDecision.strategy_id || '주문 없음',
      description: (finalDecision.reason_codes || []).join(' · ') || '기록된 차단 사유가 없습니다.',
      rows: [
        ['경로', finalDecision.path || '-'],
        ['효용', finalDecision.utility ?? '-'],
        ['기록', trace.provenance?.decision || '-'],
      ],
    },
  });

  const edges = [];
  graphIndicators.forEach((indicator) => {
    if (positions.has(`source:${indicator.source_id}`)) {
      edges.push({
        from: `source:${indicator.source_id}`,
        to: `indicator:${indicator.id}`,
        className: indicator.available ? 'pass' : 'unknown',
        detail: {
          kind: 'DATA → INDICATOR',
          title: `${sources.find((item) => item.id === indicator.source_id)?.label || indicator.source_id} → ${indicator.label}`,
          value: indicator.available ? formatOntologyScore(indicator.score) : 'UNKNOWN',
          description: indicator.detail,
          rows: [['출처', indicator.source_id], ['산출 방식', 'dashboard reconstruction']],
        },
      });
    }
  });
  displayedAlgorithms.forEach((algorithm) => {
    (algorithm.requirements || []).forEach((rule) => {
      if (!positions.has(`indicator:${rule.indicator_id}`)) return;
      const indicator = indicators.find((item) => item.id === rule.indicator_id);
      const isActive = algorithm.ontology_selected && rule.passed !== false;
      edges.push({
        from: `indicator:${rule.indicator_id}`,
        to: `algorithm:${algorithm.id}`,
        className: isActive ? 'active' : rule.passed === true ? 'pass' : rule.passed === false ? 'block' : 'unknown',
        detail: {
          kind: 'ONTOLOGY REQUIREMENT',
          title: `${indicator?.label || rule.indicator_id} → ${strategyLabels[algorithm.id] || algorithm.id}`,
          value: `${formatOntologyScore(indicator?.score)} ${rule.operator} ${formatOntologyScore(rule.threshold)}`,
          description: rule.passed === null ? '필요 데이터가 없어 이 화면에서는 조건을 재검증할 수 없습니다.' : rule.passed ? '화면 재구성값 기준으로 조건을 충족합니다.' : '화면 재구성값 기준으로 조건을 충족하지 않습니다.',
          rows: [['결과', rule.passed === null ? 'UNKNOWN' : rule.passed ? 'PASS' : 'FAIL'], ['선택 기록', algorithm.ontology_selected ? 'SELECTED' : 'NOT SELECTED']],
        },
      });
    });
    const blockedByUtility = (finalDecision.reason_codes || []).some((reason) => String(reason).includes(algorithm.id));
    if (algorithm.ontology_selected || terminalState.ontologyFilter === 'all') {
      edges.push({
        from: `algorithm:${algorithm.id}`,
        to: decisionId,
        className: algorithm.final_selected ? 'active' : blockedByUtility || finalDecision.action === 'NO_TRADE' ? 'block' : 'unknown',
        detail: {
          kind: 'STRATEGY → ROUTER',
          title: `${strategyLabels[algorithm.id] || algorithm.id} → ${finalDecision.action || 'NO_TRADE'}`,
          value: algorithm.final_selected ? 'ROUTED' : blockedByUtility ? 'NON-POSITIVE NET EDGE' : 'NOT ROUTED',
          description: algorithm.ontology_selected && !algorithm.final_selected
            ? '온톨로지는 전략을 허용했지만 최종 효용 라우터가 주문을 차단했습니다.'
            : '최종 의사결정 로그에 기록된 라우팅 관계입니다.',
          rows: [['온톨로지', algorithm.ontology_selected ? 'SELECTED' : 'NOT SELECTED'], ['최종', algorithm.final_selected ? 'SELECTED' : 'BLOCKED']],
        },
      });
    }
  });
  edges.forEach((edge) => drawOntologyEdge(svg, positions, edge));
  [...svg.querySelectorAll('.graph-node')].forEach((node) => svg.appendChild(node));

  const allowedName = strategyLabels[ontology.strategy_id] || ontology.strategy_id || '선택 없음';
  document.getElementById('decision-ontology-summary').textContent =
    `온톨로지: ${allowedName} ${ontology.allowed ? '허용' : '차단'} · 최종: ${finalDecision.action || 'NO_TRADE'} (${String(finalDecision.path || '-').toUpperCase()})`;
  const liveBadge = document.getElementById('decision-ontology-live');
  liveBadge.textContent = `${trace.fresh ? 'LIVE' : 'STALE'} · ${shortClock(trace.generated_at)}`;
  liveBadge.className = trace.fresh ? 'status-chip' : 'status-chip blocked';
  document.getElementById('ontology-provenance').textContent =
    `${trace.provenance?.warning || ''} 결정 출처: ${trace.provenance?.decision || '-'} · 지표 출처: ${trace.provenance?.indicators || '-'}`;
  document.getElementById('algorithm-catalog').innerHTML = algorithms.map((item) => {
    const failed = (item.requirements || []).filter((rule) => rule.passed === false).length;
    const unknown = (item.requirements || []).filter((rule) => rule.passed === null).length;
    const state = item.ontology_selected ? 'ONTOLOGY SELECTED' : failed ? `${failed} RULE BLOCKED` : unknown ? `${unknown} DATA UNKNOWN` : 'RULES PASS';
    return `<article class="algorithm-catalog-card ${item.ontology_selected ? 'selected' : failed ? 'blocked' : ''}">
      <b>${escapeHtml(strategyLabels[item.id] || item.label || item.id)}</b>
      <small>${escapeHtml((item.visual_indicators || []).join(' · '))}</small>
      <em>${escapeHtml(state)}</em>
    </article>`;
  }).join('');
  document.querySelectorAll('.graph-filter').forEach((button) => {
    button.classList.toggle('active', button.dataset.graphFilter === terminalState.ontologyFilter);
    button.onclick = () => {
      terminalState.ontologyFilter = button.dataset.graphFilter;
      renderDecisionOntology(trace);
    };
  });
}

function createSvgElement(name, attributes = {}) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

function drawOntologyNode(svg, node) {
  const group = createSvgElement('g', {
    class: `graph-node ${node.className || ''}`,
    transform: `translate(${node.x} ${node.y})`,
    tabindex: 0,
    role: 'button',
  });
  group.appendChild(createSvgElement('rect', { width: node.width, height: node.height, rx: 7 }));
  const title = createSvgElement('text', { x: 11, y: 20, class: 'node-title' });
  title.textContent = truncateGraphText(node.title, 25);
  const value = createSvgElement('text', { x: 11, y: 36, class: 'node-value' });
  value.textContent = truncateGraphText(node.value, 29);
  group.append(title, value);
  group.addEventListener('click', () => showOntologyInspector(node.detail));
  group.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') showOntologyInspector(node.detail);
  });
  svg.appendChild(group);
}

function drawOntologyEdge(svg, positions, edge) {
  const from = positions.get(edge.from);
  const to = positions.get(edge.to);
  if (!from || !to) return;
  const x1 = from.x + from.width;
  const y1 = from.y + from.height / 2;
  const x2 = to.x;
  const y2 = to.y + to.height / 2;
  const curve = Math.max(45, (x2 - x1) * .46);
  const path = createSvgElement('path', {
    d: `M ${x1} ${y1} C ${x1 + curve} ${y1}, ${x2 - curve} ${y2}, ${x2} ${y2}`,
    class: `graph-edge ${edge.className || ''}`,
  });
  path.addEventListener('click', () => showOntologyInspector(edge.detail));
  svg.insertBefore(path, svg.firstChild);
}

function showOntologyInspector(detail = {}) {
  const inspector = document.getElementById('ontology-inspector');
  inspector.innerHTML = `
    <span>${escapeHtml(detail.kind || 'ONTOLOGY')}</span>
    <h3>${escapeHtml(detail.title || '-')}</h3>
    <div class="inspector-value">${escapeHtml(String(detail.value ?? '-'))}</div>
    <p>${escapeHtml(detail.description || '')}</p>
    <div class="inspector-table">${(detail.rows || []).map(([label, value]) => `
      <div><span>${escapeHtml(String(label))}</span><b>${escapeHtml(String(value ?? '-'))}</b></div>
    `).join('')}</div>
  `;
}

function formatOntologyScore(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : 'N/A';
}

function formatOntologyRaw(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString('ko-KR', { maximumFractionDigits: 6 }) : 'N/A';
}

function truncateGraphText(value, maximum) {
  const text = String(value ?? '-');
  return text.length > maximum ? `${text.slice(0, maximum - 1)}…` : text;
}

function renderExecution(execution, book) {
  const stages = execution.stages || [];
  document.getElementById('execution-count').textContent = `${execution.event_count || 0} EVENTS`;
  document.getElementById('execution-track').innerHTML = stages.map((stage, index) => `
    <div class="execution-stage ${escapeHtml(stage.status)}">
      <span class="stage-number">${String(index + 1).padStart(2, '0')}</span>
      <b>${escapeHtml(stage.label)}</b>
      <small>${escapeHtml(stage.detail)}</small>
    </div>
  `).join('');
  const events = execution.events || [];
  document.getElementById('execution-tape').innerHTML = events.length ? events.map((event) => `
    <div class="tape-row">
      <b>${escapeHtml(shortClock(event.recorded_at))} · ${escapeHtml(event.event_type || '-')}</b>
      <span>${escapeHtml(compactPayload(event.payload))}</span>
    </div>
  `).join('') : '<div class="tape-empty">실제 주문 기록이 없습니다. 전략과 종목이 채택되면 매수·매도 판단부터 KIS 전송, 접수, 체결까지 여기에 표시됩니다.</div>';
  renderOrderbookAnalysis(book);
}

function renderOrderbookAnalysis(book) {
  const container = document.getElementById('orderbook-card');
  const ageSeconds = Number(book?.age_seconds);
  const stale = Boolean(book?.stale) || (Number.isFinite(ageSeconds) && ageSeconds > 90);
  const valid = Boolean(book?.valid)
    && Number(book?.best_bid) > 0
    && Number(book?.best_ask) >= Number(book?.best_bid);
  if (!book || !valid || stale) {
    const state = stale ? 'STALE ORDERBOOK' : 'NO VALID ORDERBOOK';
    const detail = stale
      ? `마지막 호가 ${formatAge(ageSeconds)} 전 · 오래된 잔량은 분석에 사용하지 않습니다.`
      : '유효한 매수·매도 호가가 들어오면 깊이 그래프가 표시됩니다.';
    container.className = `orderbook-card ${stale ? 'stale' : 'invalid'}`;
    container.innerHTML = `
      <div class="book-title">${state}</div>
      <div class="book-state-detail">${escapeHtml(detail)}</div>
      ${book ? `<div class="book-row last-known"><span>LAST BID ${formatPrice(book.best_bid)}</span><span>LAST ASK ${formatPrice(book.best_ask)}</span></div>` : ''}
    `;
    return;
  }

  const bidTotal = Math.max(0, Number(book.total_bid_volume) || 0);
  const askTotal = Math.max(0, Number(book.total_ask_volume) || 0);
  const total = bidTotal + askTotal;
  const bidShare = total > 0 ? Math.max(0, Math.min(100, bidTotal / total * 100)) : 50;
  const levels = (Array.isArray(book.levels) ? book.levels : []).slice(0, 10);
  const maximumDepth = Math.max(
    1,
    ...levels.flatMap((level) => [
      Math.max(0, Number(level.bid_size) || 0),
      Math.max(0, Number(level.ask_size) || 0),
    ]),
  );
  const depthRows = levels.map((level, index) => {
    const bidSize = Math.max(0, Number(level.bid_size) || 0);
    const askSize = Math.max(0, Number(level.ask_size) || 0);
    const bidWidth = bidSize / maximumDepth * 100;
    const askWidth = askSize / maximumDepth * 100;
    return `
      <div class="book-depth-row">
        <div class="book-depth-side bid">
          <i style="width:${bidWidth.toFixed(2)}%"></i>
          <span>${formatPrice(level.bid_price)}</span><b>${formatCompact(bidSize)}</b>
        </div>
        <em>L${index + 1}</em>
        <div class="book-depth-side ask">
          <i style="width:${askWidth.toFixed(2)}%"></i>
          <span>${formatPrice(level.ask_price)}</span><b>${formatCompact(askSize)}</b>
        </div>
      </div>
    `;
  }).join('');
  const spread = Number(book.spread_bps);
  const spreadClass = Number.isFinite(spread) && spread > 80 ? 'wide' : '';
  container.className = 'orderbook-card live';
  container.innerHTML = `
    <div class="book-title">LIVE ORDERBOOK DEPTH <small>${levels.length || 1} LEVELS</small></div>
    <div class="book-spread ${spreadClass}">SPREAD ${Number.isFinite(spread) ? `${spread.toFixed(2)} bps` : '-'}</div>
    <div class="book-row"><span>BID ${formatPrice(book.best_bid)}</span><span>ASK ${formatPrice(book.best_ask)}</span></div>
    <div class="book-row"><span>${formatCompact(bidTotal)} (${bidShare.toFixed(1)}%)</span><span>${formatCompact(askTotal)} (${(100 - bidShare).toFixed(1)}%)</span></div>
    <div class="book-imbalance"><i style="width:${bidShare}%"></i></div>
    <div class="book-depth-head"><span>매수 가격 · 잔량</span><span>매도 가격 · 잔량</span></div>
    <div class="book-depth">${depthRows || '<div class="book-state-detail">단계별 잔량 없음</div>'}</div>
  `;
}

function renderSafety(data) {
  const passed = (data.promotion_gates || []).filter((gate) => gate.passed);
  const failed = (data.promotion_gates || []).filter((gate) => !gate.passed);
  document.getElementById('safety-state').textContent = data.live_order_capable ? '제한 실행 가능' : '실주문 차단';
  document.getElementById('safety-reason').textContent = data.live_order_capable
    ? '모든 승격 게이트 통과'
    : `${passed.length}/${passed.length + failed.length} 게이트 통과 · ${failed.map((gate) => gate.label).join(' / ')}`;
}

function movingAverage(values, window) {
  return values.map((_, index) => {
    if (index + 1 < window) return null;
    const slice = values.slice(index + 1 - window, index + 1);
    return slice.reduce((sum, value) => sum + value, 0) / window;
  });
}

function prepareCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width, height };
}

function formatPrice(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '-';
  const digits = Math.abs(number) >= 1000 ? 0 : Math.abs(number) >= 10 ? 2 : 4;
  return number.toLocaleString('ko-KR', { maximumFractionDigits: digits });
}
function formatCompact(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '-';
  return Intl.NumberFormat('ko-KR', { notation: 'compact', maximumFractionDigits: 2 }).format(number);
}
function formatKrw(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '-';
  return `${Math.round(number).toLocaleString('ko-KR')}원`;
}
function formatSignedKrw(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '-';
  return `${number > 0 ? '+' : ''}${Math.round(number).toLocaleString('ko-KR')}원`;
}
function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '-';
  return `${number > 0 ? '+' : ''}${(number * 100).toFixed(2)}%`;
}
function formatInteger(value) {
  return Math.round(Number(value || 0)).toLocaleString('ko-KR');
}
function formatCompactNumber(value) {
  return Number(value || 0).toLocaleString('ko-KR', { maximumFractionDigits: 2 });
}
function formatFileSize(value) {
  const bytes = Number(value || 0);
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / (1024 ** index)).toFixed(index >= 3 ? 2 : 1)} ${units[index]}`;
}
function formatAge(value) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds)) return '갱신 기록 없음';
  if (seconds < 60) return `${Math.round(seconds)}초 전 갱신`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}분 전 갱신`;
  return `${(seconds / 3600).toFixed(1)}시간 전 갱신`;
}
function shortClock(value) {
  if (!value) return '-';
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? String(value).slice(11, 19)
    : date.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
}
function shortDateTime(value) {
  if (!value) return '-';
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? String(value)
    : date.toLocaleString('ko-KR', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false });
}
function formatTime(value) {
  if (!value) return '-';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString('ko-KR', { hour12: false });
}
function shortTime(value) {
  if (!value) return '-';
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? String(value).slice(11, 16)
    : date.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', hour12: false });
}
function compactPayload(payload) {
  if (!payload || typeof payload !== 'object') return String(payload ?? '-');
  return Object.entries(payload)
    .filter(([key]) => [
      'symbol',
      'ticker',
      'action',
      'side',
      'quantity',
      'filled_quantity',
      'limit_price',
      'average_fill_price',
      'reason_code',
      'status',
      'intent_id',
      'verdict_id',
      'broker_order_id',
    ].includes(key))
    .map(([key, value]) => `${key}=${value}`)
    .join(' · ') || JSON.stringify(payload).slice(0, 180);
}
function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

document.getElementById('terminal-refresh').addEventListener('click', () => {
  fetchMarketView();
  fetchAssetSummary();
  fetchSystemDiagnostics();
});
window.addEventListener('resize', () => {
  if (terminalState.data) {
    renderTradingChart(
      terminalState.data.market || {},
      terminalState.data.algorithm,
    );
  }
  if (terminalState.diagnostics) {
    renderTrainingMonitor((terminalState.diagnostics.flows || {}).training || {});
  }
});
document.querySelectorAll('[data-chart-mode]').forEach((button) => {
  button.addEventListener('click', () => {
    terminalState.chartMode = button.dataset.chartMode || 'seconds';
    document.querySelectorAll('[data-chart-mode]').forEach((item) => {
      item.classList.toggle('active', item === button);
    });
    if (terminalState.data) {
      renderTradingChart(terminalState.data.market || {}, terminalState.data.algorithm);
    }
  });
});
function animateTradingChart(timestamp) {
  if (
    terminalState.chartMode === 'seconds'
    && terminalState.data
    && timestamp - terminalState.chartLastPaint >= 120
  ) {
    terminalState.chartLastPaint = timestamp;
    renderTradingChart(terminalState.data.market || {}, terminalState.data.algorithm);
  }
  terminalState.chartAnimationFrame = requestAnimationFrame(animateTradingChart);
}
setInterval(() => {
  document.getElementById('terminal-clock').textContent = new Date().toLocaleTimeString('ko-KR', { hour12: false });
}, 1000);
setInterval(() => fetchMarketStream(), 1000);
setInterval(() => fetchMarketView(), 3000);
setInterval(() => fetchSystemDiagnostics(), 5000);
setInterval(() => fetchAssetSummary(), 15000);
fetchMarketView();
fetchAssetSummary();
fetchSystemDiagnostics();
terminalState.chartAnimationFrame = requestAnimationFrame(animateTradingChart);

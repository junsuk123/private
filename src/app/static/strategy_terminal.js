const terminalState = {
  symbol: null,
  data: null,
  busy: false,
  assetBusy: false,
  diagnosticsBusy: false,
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

async function fetchMarketView(symbol = terminalState.symbol) {
  if (terminalState.busy) return;
  terminalState.busy = true;
  const refresh = document.getElementById('terminal-refresh');
  if (refresh) refresh.classList.add('loading');
  try {
    const query = symbol ? `?symbol=${encodeURIComponent(symbol)}&limit=180` : '?limit=180';
    const response = await fetch(`/api/refactor/market-view${query}`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const data = await response.json();
    terminalState.data = data;
    terminalState.symbol = data.symbol;
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
  renderCandidates(data.candidates || []);
  renderInstrument(data.symbol, market, selection, algorithm);
  renderChart(market.bars || [], algorithm, market.last_price);
  renderOntology(selection, market);
  renderExecution(data.execution || {}, market.latest_orderbook);
  renderSafety(data);
}

function renderHeader(data, market) {
  const stale = Boolean(market.stale);
  const connection = document.querySelector('.connection');
  connection?.classList.toggle('stale', stale);
  document.getElementById('feed-state').textContent = stale ? '저장 시세 · 최신 이벤트 지연' : '실시간 이벤트 수신';
  const mode = document.getElementById('terminal-mode');
  mode.textContent = `${String(data.mode || 'unknown').toUpperCase()} · ${data.live_order_capable ? 'ORDER ARMED' : 'ORDER BLOCKED'}`;
  mode.className = data.live_order_capable ? 'status-chip' : 'status-chip blocked';
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

  const ontologyStrategyId = selection.ontology_strategy_id || selection.strategy_id || null;
  const active = Boolean(selection.ontology_allowed && ontologyStrategyId);
  document.getElementById('algorithm-state').textContent = String(selection.action || 'NO TRADE');
  document.getElementById('algorithm-name').textContent = active
    ? (strategyLabels[ontologyStrategyId] || ontologyStrategyId)
    : '온톨로지 선택 대기';
  document.getElementById('algorithm-thesis').textContent = algorithm?.thesis
    || '필수 사실과 기대 순효용을 통과한 전략만 활성화됩니다.';
  const tags = algorithm ? [
    ...(algorithm.visual_indicators || []),
    `STOP ${algorithm.stop_bps}bps`,
    `TARGET ${algorithm.profit_bps}bps`,
    `HOLD ${algorithm.max_holding_seconds}s`,
  ] : ['CLOSED-WORLD', 'NoTrade', 'OWNER-LOCK'];
  document.getElementById('algorithm-tags').innerHTML = tags.map((tag) => `<span>${escapeHtml(tag)}</span>`).join('');
}

function renderChart(bars, algorithm = null, referencePrice = null) {
  const empty = document.getElementById('chart-empty');
  empty.style.display = bars.length ? 'none' : 'grid';
  drawPriceChart(document.getElementById('price-chart'), bars, algorithm, referencePrice);
  drawVolumeChart(document.getElementById('volume-chart'), bars);
  document.getElementById('chart-range').textContent = `최근 ${bars.length}개 1분봉 · MA5 / MA20 / VWAP`;
  const last = bars[bars.length - 1];
  document.getElementById('chart-updated').textContent = `마지막 이벤트 ${formatTime(last?.time)}`;
}

function drawPriceChart(canvas, bars, algorithm, referencePrice) {
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
  const step = plotWidth / Math.max(1, bars.length);

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
    const x = pad.left + step * (i + .5);
    const up = Number(bar.close) >= Number(bar.open);
    const color = up ? '#42d392' : '#ff6678';
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
    ctx.fillRect(x - bodyWidth / 2, bodyTop, bodyWidth, bodyHeight);
  }
  drawLine(ctx, movingAverage(bars.map((bar) => Number(bar.close)), 5), y, pad.left, step, '#39d7e7', 1.3);
  drawLine(ctx, movingAverage(bars.map((bar) => Number(bar.close)), 20), y, pad.left, step, '#9b8cff', 1.3);
  drawLine(ctx, bars.map((bar) => Number(bar.vwap) || null), y, pad.left, step, '#f3b95f', 1, [4, 3]);
  if (algorithm && Number.isFinite(reference)) {
    drawLevel(ctx, width, height, pad, y, reference * (1 + Number(algorithm.profit_bps || 0) / 10000), '#42d392', 'TARGET');
    drawLevel(ctx, width, height, pad, y, reference * (1 - Number(algorithm.stop_bps || 0) / 10000), '#ff6678', 'STOP');
  }

  ctx.fillStyle = '#6f8095';
  ctx.textAlign = 'center';
  const labels = Math.min(6, bars.length);
  for (let i = 0; i < labels; i += 1) {
    const index = Math.round(i * (bars.length - 1) / Math.max(1, labels - 1));
    const x = pad.left + step * (index + .5);
    ctx.fillText(shortTime(bars[index].time), x, height - 7);
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

function drawLine(ctx, values, y, left, step, color, width, dash = []) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.beginPath();
  let started = false;
  values.forEach((value, index) => {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return;
    const x = left + step * (index + .5);
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

function drawVolumeChart(canvas, bars) {
  const { ctx, width, height } = prepareCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#0d131c';
  ctx.fillRect(0, 0, width, height);
  if (!bars.length) return;
  const left = 12;
  const right = 62;
  const plotWidth = width - left - right;
  const maxVolume = Math.max(...bars.map((bar) => Number(bar.volume || 0)), 1);
  const step = plotWidth / bars.length;
  bars.forEach((bar, index) => {
    const value = Number(bar.volume || 0);
    const h = value / maxVolume * (height - 10);
    const x = left + step * index;
    ctx.fillStyle = Number(bar.close) >= Number(bar.open) ? 'rgba(66,211,146,.5)' : 'rgba(255,102,120,.5)';
    ctx.fillRect(x, height - h, Math.max(1, step * .7), h);
  });
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
      <b>${escapeHtml(event.event_type || '-')}</b>
      <span>${escapeHtml(compactPayload(event.payload))}</span>
    </div>
  `).join('') : '<div class="tape-empty">저장된 OrderIntent가 없습니다. 온톨로지가 전략을 선택하면 인과 체결 로그가 여기에 표시됩니다.</div>';
  const imbalance = Number(book?.imbalance);
  const bidShare = Number.isFinite(imbalance) ? Math.max(0, Math.min(100, (imbalance + 1) * 50)) : 50;
  document.getElementById('orderbook-card').innerHTML = `
    <div class="book-title">LEVEL 1 ORDERBOOK</div>
    <div class="book-spread">SPREAD ${book?.spread_bps === null || book?.spread_bps === undefined ? '-' : `${Number(book.spread_bps).toFixed(2)} bps`}</div>
    <div class="book-row"><span>BID ${formatPrice(book?.best_bid)}</span><span>ASK ${formatPrice(book?.best_ask)}</span></div>
    <div class="book-row"><span>${formatCompact(book?.total_bid_volume)}</span><span>${formatCompact(book?.total_ask_volume)}</span></div>
    <div class="book-imbalance"><i style="width:${bidShare}%"></i></div>
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
    .filter(([key]) => ['symbol', 'action', 'quantity', 'reason_code', 'status', 'intent_id', 'verdict_id', 'broker_order_id'].includes(key))
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
    renderChart(
      terminalState.data.market?.bars || [],
      terminalState.data.algorithm,
      terminalState.data.market?.last_price,
    );
  }
});
setInterval(() => {
  document.getElementById('terminal-clock').textContent = new Date().toLocaleTimeString('ko-KR', { hour12: false });
}, 1000);
setInterval(() => fetchMarketView(), 3000);
setInterval(() => fetchSystemDiagnostics(), 5000);
setInterval(() => fetchAssetSummary(), 15000);
fetchMarketView();
fetchAssetSummary();
fetchSystemDiagnostics();

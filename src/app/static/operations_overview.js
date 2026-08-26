const operationsOverviewState = {
  busy: false,
  timer: null,
};

const opsReasonLabels = {
  MARKET_DATA_NOT_READY: '실시간 체결·호가 신선도 부족',
  GNN_NOT_LIVE_AUTHORIZED: 'GNN 실시간 신뢰도 미승격',
  NO_MECHANICAL_STRATEGY_TRIGGER: '전략별 실제 진입 조건 대기',
  NO_ADMISSIBLE_TRIGGERED_STRATEGY: '발동 전략의 실행 권한·차입 조건 미충족',
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

function opsOperatingModelMetric(activeModel, metric, digits = 3) {
  const rawDirect = activeModel?.metrics?.[metric];
  const direct = rawDirect == null ? Number.NaN : Number(rawDirect);
  if (Number.isFinite(direct)) return direct.toFixed(digits);

  const marketValues = Object.entries(activeModel?.market_models || {})
    .map(([market, report]) => {
      const rawValue = report?.metrics?.[metric];
      const value = rawValue == null ? Number.NaN : Number(rawValue);
      return Number.isFinite(value) ? `${market} ${value.toFixed(digits)}` : null;
    })
    .filter(Boolean);
  return marketValues.length ? marketValues.join(' · ') : 'N/A';
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
  let firstError = null;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const response = await fetch(url, { cache: 'no-store' });
      if (response.ok) return response.json();
      const error = new Error(`${url} · HTTP ${response.status}`);
      if (![502, 503, 504].includes(response.status) || attempt > 0) throw error;
      firstError = error;
    } catch (error) {
      if (attempt > 0) {
        const detail = firstError ? `${firstError.message}; retry: ${error.message}` : error.message;
        throw new Error(detail);
      }
      firstError = error;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 150));
  }
  throw firstError || new Error(`${url} · unknown fetch failure`);
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
  if (gnn?.checkpoint_live_authorized !== true) {
    return {
      tone: 'waiting',
      title: 'GNN SHADOW ONLY',
      detail: 'The checkpoint is not live-authorized. Inference may continue, but order authority remains blocked.',
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

function renderOperationsOverview({ diag, reliability, trading, gnn, mode, macro }) {
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
  const gnnHasScore = gnn?.score_available === true
    || Number(gnn?.sample_count || 0) > 0;
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
    `점수 ${gnnHasScore ? opsNumber(gnn?.score, 3) : 'N/A'} · 보정 ${calibrated.length} · 진입 ${trusted.length}`,
  );
  const executionReady = Boolean(
    trading?.running
    && trading?.buy_enabled
    && gnn?.checkpoint_live_authorized === true
    && trusted.length > 0
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
    `연구 ${research.latest?.status || (research.active ? '진행 중' : '대기')} · 활성 모델 AUC ${opsOperatingModelMetric(activeModel, 'auc', 3)} · 오류 ${diag?.last_error ? '있음' : '없음'}`,
  );

  opsSetBadge(
    'ops-gnn-state',
    trusted.length ? 'pass' : 'warn',
    trusted.length
      ? 'ENTRY READY'
      : (gnn?.checkpoint_live_authorized === false
        ? 'SHADOW ONLY'
        : (gnn?.passed ? 'CALIBRATED' : 'COLLECTING')),
  );
  opsText('ops-gnn-score', gnnHasScore ? opsNumber(gnn?.score, 3) : 'N/A');
  opsText('ops-gnn-samples', `표본 ${opsNumber(gnn?.sample_count)}/${opsNumber(gnn?.minimum_samples)}`);
  const strategyRows = Object.values(gnn?.strategy_metrics || {});
  const upsideSupervised = (gnn?.upside_supervised_strategy_ids || []).length;
  opsText(
    'ops-gnn-trusted',
    `진입 허용 ${trusted.length ? trusted.join(', ') : '없음'} · 보정 통과 ${calibrated.length}`
      + ` · 상승 학습 ${upsideSupervised}/${strategyRows.length || '-'}`
      + ` · ${gnn?.outcome_validation_method === 'directional_strategy_policy_replay_v2' ? '방향별 실행정책 재현' : '구형 예측 검증'}`,
  );
  const progress = Math.min(100, Math.max(0, Number(gnn?.sample_count || 0) / Math.max(1, Number(gnn?.minimum_samples || 1)) * 100));
  const progressNode = document.getElementById('ops-gnn-progress');
  if (progressNode) progressNode.style.width = `${progress}%`;
  // "0.0%" and "no positive-edge forecast was ever made" are different facts and
  // the second one is the one that is true right now. A rate over an empty set
  // rendered as 0% reads as "the model tried and lost every time".
  const positiveForecasts = strategyRows.reduce(
    (total, row) => total + Number(row?.trade_sample_count || 0),
    0,
  );
  opsText(
    'ops-gnn-positive',
    positiveForecasts ? `${opsPercent(gnn?.positive_net_rate)} (${positiveForecasts}건)` : '양의 예측 표본 없음',
  );
  opsText(
    'ops-gnn-net',
    positiveForecasts ? `${opsNumber(gnn?.mean_realized_net_bps, 2)} bp` : '-',
  );
  opsText('ops-gnn-uncertainty', opsNumber(gnn?.mean_uncertainty, 3));
  const strategyMetrics = gnn?.strategy_metrics || {};
  const trainingMarketMetrics = gnn?.training_strategy_market_metrics || {};
  const liveMarketMetrics = gnn?.strategy_market_metrics || {};
  const strategyList = document.getElementById('ops-strategy-list');
  if (strategyList) {
    // Include strategies present only in the latest training checkpoint. A
    // strategy with no current live sample still needs to expose whether it is
    // profitable, cost-bound, under-sampled, or structurally unreachable.
    const strategyIds = Array.from(new Set([
      ...Object.keys(trainingMarketMetrics),
      ...Object.keys(strategyMetrics),
    ])).sort();
    const rows = strategyIds.map((strategyId) => [strategyId, strategyMetrics[strategyId] || {}]);
    strategyList.innerHTML = rows.length
      ? rows.map(([strategyId, metrics]) => {
          // Three outcomes that used to collapse into one "보정 통과" badge:
          //   entry allowed / waiting for live evidence / upside head untaught.
          // The last one never resolves on its own, so it must not look like
          // progress. Requires a retrain, not patience.
          const unsupervised = metrics.upside_supervised === false;
          const rowsTaught = metrics.upside_training_rows;
          const minimum = Number(gnn?.minimum_upside_supervision_rows || 20);
          const hasLiveMetrics = Object.prototype.hasOwnProperty.call(strategyMetrics, strategyId);
          const trainedMarkets = trainingMarketMetrics[strategyId] || {};
          let cls = 'warn';
          let label = '보정 통과';
          let hint = '실시간 양엣지 표본 대기';
          if (!hasLiveMetrics) {
            const trainedRows = Object.values(trainedMarkets);
            const positiveMarket = trainedRows.find((row) => (
              Number(row.filled || 0) >= minimum
              && Number(row.mean_net_return_bps_when_filled || 0) > 0
            ));
            const unreachable = trainedRows.length && trainedRows.every((row) => (
              String(row.performance_diagnosis || '').startsWith('STRUCTURALLY_UNREACHABLE')
              || Number(row.filled || 0) === 0
            ));
            cls = positiveMarket ? 'warn' : 'block';
            label = positiveMarket ? '학습 양엣지·운영 대기' : (unreachable ? '컨텍스트 부족' : '학습 순손실/표본 부족');
            hint = '체크포인트 진단만 존재하며 실시간 전방 검증 표본은 아직 없음';
          } else if (metrics.entry_authorized) {
            cls = 'pass';
            label = '진입 허용';
            hint = '양엣지 검증 통과';
          } else if (unsupervised) {
            cls = 'block';
            label = `상승 미학습 ${rowsTaught == null ? '' : `${rowsTaught}/${minimum}`}`.trim();
            hint = '상승(MFE) 헤드 학습 부족 → 양엣지 예보 억제됨. 재학습 필요';
          } else if (!metrics.calibration_passed) {
            cls = 'block';
            label = Number(metrics.score || 0).toFixed(3);
            hint = '보정 미달';
          } else if (metrics.execution_validation_stage === 'POSITIVE_EDGE_VALIDATION_FAILED') {
            cls = 'block';
            label = '양엣지 검증 실패';
            hint = `양의 예측 ${Number(metrics.trade_sample_count || 0)}건을 방향별 목표·손절·트레일링·시간 종료로 재현한 순효율 ${opsNumber(metrics.mean_realized_net_bps, 1)}bp`;
          }
          const marketEvidence = liveMarketMetrics[strategyId] || trainingMarketMetrics[strategyId] || {};
          const marketSummary = ['KRX', 'US']
            .filter((market) => marketEvidence[market])
            .map((market) => {
              const row = marketEvidence[market];
              const net = row.mean_realized_net_bps ?? row.mean_net_return_bps_when_filled;
              const count = row.trade_sample_count ?? row.filled ?? 0;
              return `${market} ${net == null ? '-' : `${opsNumber(net, 1)}bp`}(${Number(count)}건)`;
            })
            .join(' · ');
          const marketDiagnostics = ['KRX', 'US']
            .filter((market) => marketEvidence[market])
            .map((market) => {
              const row = marketEvidence[market];
              const gross = row.mean_gross_return_bps_when_filled;
              const cost = row.mean_cost_bps_when_filled;
              const win = row.positive_net_rate_when_filled;
              const factor = row.profit_factor_when_filled;
              const exits = row.exit_reason_counts
                ? Object.entries(row.exit_reason_counts).map(([reason, count]) => `${reason} ${count}`).join('/')
                : '청산 표본 대기';
              return `${market}: 총 ${gross == null ? '-' : `${opsNumber(gross, 1)}bp`} · 비용 ${cost == null ? '-' : `${opsNumber(cost, 1)}bp`} · 승률 ${win == null ? '-' : opsPercent(win)} · PF ${factor == null ? '-' : opsNumber(factor, 2)} · ${exits}`;
            })
            .join(' | ');
          const authorizedMarkets = metrics.upside_authorized_markets
            || gnn?.upside_authorized_strategy_markets?.[strategyId]
            || [];
          const marketHint = marketSummary
            ? `${marketSummary} · ${marketDiagnostics} · 양의 헤드 허용 ${authorizedMarkets.length ? authorizedMarkets.join('/') : '없음'}`
            : '시장별 실행 표본 대기';
          return `
          <div class="ops-strategy-row" title="${hint} · ${marketHint}">
            <b title="${strategyId}">${strategyId}</b>
            <span>${marketSummary || `${Number(metrics.sample_count || 0)}표본`}</span>
            <span class="${cls}">${label}</span>
          </div>
        `;
        }).join('')
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
  const contextSymbols = Array.isArray(macro?.market_context_symbols)
    ? macro.market_context_symbols
    : [];
  const contextCount = Number(macro?.market_context_symbol_count || contextSymbols.length || 0);
  opsText(
    'ops-market-context',
    contextCount
      ? `${opsNumber(contextCount)}종목 · 후보와 독립`
      : '앵커 데이터 대기',
  );
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
    const [diag, reliability, trading, gnn, mode, blockade, macro] = await Promise.all([
      opsFetchJson('/api/system-diagnostics'),
      opsFetchJson('/api/auto-reliability/status'),
      opsFetchJson('/api/realtime-trading/status'),
      opsFetchJson('/api/gnn/realtime-trust'),
      opsFetchJson('/api/operation-mode/status'),
      opsFetchJson('/api/realtime-trading/entry-blockade'),
      opsFetchJson('/api/account/macro-micro'),
    ]);
    renderOperationsOverview({ diag, reliability, trading, gnn, mode, macro });
    renderEntryBlockade(blockade);
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


// --- Entry blockade -------------------------------------------------------
// Ordered "why is nothing trading" chain. The FIRST unmet link is the answer;
// links after it are dimmed because they were never reached, which is a
// different statement from having failed.
//
// This exists because the terminal used to surface one reason code from
// whichever layer failed last. For 11,614 consecutive cycles that read
// NO_POSITIVE_NET_GNN_EDGE — naming the GNN, while the actual constraint was
// that no scanned market was in its regular session and every candidate was
// failing on after-hours book liquidity.
const BLOCKADE_STAGE_LABELS = {
  engine_running: '엔진 실행',
  live_armed: '라이브 무장',
  market_session: '시장 세션',
  buy_candidates: '매수 후보',
  micro_buy_intents: '마이크로 전략',
  strategy_election: '전략 선택',
  position: '포지션',
};

function blockadeTags(link) {
  const data = link.data || {};
  const parts = [];
  if (link.stage === 'market_session' && data.scanned_groups) {
    Object.entries(data.scanned_groups).forEach(([group, info]) => {
      parts.push(`${group} ${info.phase}${info.allows_new_entry ? ' · 진입가능' : ''}`);
    });
    if (data.extended_hours_entry_enabled) parts.push('시간외 진입 허용');
  }
  if (link.stage === 'buy_candidates') {
    if (data.warming_up) {
      parts.push(`분봉 ${data.best_bars}/${data.required_bars} (${data.best_symbol})`);
      parts.push(`약 ${data.eta_minutes}분 후 충족`);
    }
    if (data.unaffordable) {
      parts.push(`주문가능 KRW ${Math.round(data.krw_orderable).toLocaleString()}원`);
      parts.push(`최저가 ${data.cheapest_candidate} ${Math.round(data.cheapest_ask).toLocaleString()}원`);
    }
    if (Array.isArray(data.sample)) parts.push(...data.sample.slice(0, 6));
  }
  if (link.stage === 'micro_buy_intents' && Array.isArray(data.blocking_reason_codes)) {
    const counts = data.reason_code_counts || {};
    parts.push(...data.blocking_reason_codes.slice(0, 6).map((code) => (
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
      // Under GNN-direct election this field is the model's raw forward edge,
      // not a pessimistic bound -- see the same guard in account_dashboard.js.
      const direct = (data.reason_codes || []).includes('GNN_DIRECT_ELECTION');
      const label = direct ? 'GNN 예측 엣지(무보정)' : '보수적 엣지';
      parts.push(`${label} ${data.conservative_edge_bps.toFixed(1)}bp`);
    }
    if (data.is_exploration) parts.push('탐색 진입(최소 비중)');
    (data.reason_codes || []).slice(0, 4).forEach((code) => parts.push(code));
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
  parts.slice(0, 8).forEach((text) => {
    const tag = document.createElement('span');
    tag.className = 'blockade-tag';
    tag.textContent = text;
    wrap.appendChild(tag);
  });
  return wrap;
}

function renderEntryBlockade(payload) {
  const list = document.getElementById('blockade-chain');
  const headline = document.getElementById('blockade-headline');
  const verdict = document.getElementById('blockade-verdict');
  if (!list || !headline || !verdict) return;
  if (!payload || payload.ok === false) {
    headline.textContent = '진단을 불러오지 못했습니다.';
    verdict.textContent = '확인 불가';
    verdict.className = 'status-chip waiting';
    list.innerHTML = '';
    return;
  }
  const chain = Array.isArray(payload.chain) ? payload.chain : [];
  const blockedAt = chain.findIndex((link) => !link.ok);
  if (payload.trading_possible) {
    verdict.textContent = '진입 가능';
    verdict.className = 'status-chip';
    headline.textContent = '모든 단계 통과 — 신규 진입을 막는 요인이 없습니다.';
  } else {
    const label = BLOCKADE_STAGE_LABELS[payload.blocking_stage] || payload.blocking_stage || '알 수 없음';
    verdict.textContent = `차단 · ${label}`;
    verdict.className = 'status-chip blocked';
    headline.textContent = payload.blocking_detail || '';
  }
  list.innerHTML = '';
  chain.forEach((link, index) => {
    const unreachable = blockedAt >= 0 && index > blockedAt;
    const item = document.createElement('li');
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
    const tags = unreachable ? null : blockadeTags(link);
    if (tags) body.appendChild(tags);
    item.appendChild(mark);
    item.appendChild(body);
    list.appendChild(item);
  });
}

window.renderEntryBlockade = renderEntryBlockade;

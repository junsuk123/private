(function () {
  'use strict';

  const STAGE_LABELS = {
    engine_running: '엔진 실행',
    live_armed: '라이브 무장',
    market_session: '시장 세션',
    buy_candidates: '후보 준비',
    micro_buy_intents: '마이크로 판단',
    strategy_election: '전략 선출',
    position: '주문·포지션',
  };

  const HARD_REASONS = new Set([
    'MICRO_HARD_RISK_BLOCK',
    'INSTRUMENT_NOT_LIVE_BUY_ELIGIBLE',
    'MARKET_NOT_OPEN_FOR_NEW_ENTRY',
    'PRICE_EXCEEDS_ORDERABLE_CASH',
    'INSUFFICIENT_USD_ORDERABLE_CASH',
    'INSUFFICIENT_KRW_ORDERABLE_CASH',
  ]);

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function statusForLink(link) {
    if (!link) return 'pending';
    return link.ok ? 'pass' : 'blocked';
  }

  function stageNode(link, stage) {
    const status = statusForLink(link);
    const node = element('article', `entry-flow-node ${status}`);
    node.dataset.stage = stage;
    const head = element('div', 'entry-flow-node-head');
    head.appendChild(element('span', 'entry-flow-state-dot'));
    head.appendChild(element('strong', '', STAGE_LABELS[stage] || stage));
    head.appendChild(element('small', '', status === 'pass' ? '통과' : status === 'blocked' ? '차단' : '대기'));
    node.appendChild(head);
    node.appendChild(element('p', '', link?.detail || '이전 단계 통과 대기'));
    return node;
  }

  function invertSamples(samples) {
    const result = {};
    Object.entries(samples || {}).forEach(([reason, symbols]) => {
      (symbols || []).forEach((symbol) => {
        const key = String(symbol || '').toUpperCase();
        if (!key) return;
        if (!result[key]) result[key] = [];
        if (!result[key].includes(reason)) result[key].push(reason);
      });
    });
    return result;
  }

  function candidateRows(data) {
    const queueRows = data?.warmup_queue?.readiness?.symbols || {};
    const selected = new Set([
      ...(data?.candidate_filter_selected_symbols || []),
      ...(data?.sample || []),
    ].map((item) => String(item || '').toUpperCase()));
    const rejectionBySymbol = invertSamples(data?.candidate_filter_reason_samples);
    const admissionBySymbol = invertSamples(data?.candidate_filter_admission_note_samples);
    const symbols = new Set([...selected, ...Object.keys(queueRows), ...Object.keys(rejectionBySymbol)]);
    const rows = [];

    symbols.forEach((rawKey) => {
      const queue = queueRows[rawKey] || {};
      const symbol = String(queue.symbol || rawKey.split(':').pop() || '').toUpperCase();
      if (!symbol) return;
      const reasons = rejectionBySymbol[symbol] || [];
      const admissions = admissionBySymbol[symbol] || [];
      const observations = Number(queue.observations || 0);
      const required = Number(queue.minimum_observations || 0);
      let status = 'warming';
      let stateLabel = required ? `분봉 ${observations}/${required}` : '데이터 준비 중';

      if (selected.has(symbol)) {
        status = 'pass';
        stateLabel = '다음 단계 진행';
      } else if (reasons.some((reason) => HARD_REASONS.has(reason))) {
        status = 'blocked';
        stateLabel = '종목별 하드 차단';
      } else if (reasons.includes('NOT_FRESH_IN_TICK_WINDOW')) {
        status = 'stale';
        stateLabel = required && observations >= required ? '분봉 준비 · 실시간 틱 대기' : stateLabel;
      } else if (String(queue.state || '').toUpperCase() === 'DATA_READY') {
        status = 'bar-ready';
        stateLabel = '분봉 준비 · 틱/호가 대기';
      } else if (reasons.includes('STRATEGY_TICK_WINDOW_NOT_READY')) {
        status = 'warming';
        stateLabel = required ? `틱 대기 · 분봉 ${observations}/${required}` : '전략용 틱 대기';
      }

      rows.push({ symbol, status, stateLabel, observations, required, reasons, admissions });
    });

    const priority = { pass: 0, 'bar-ready': 1, warming: 2, stale: 3, blocked: 4 };
    return rows.sort((a, b) => (
      (priority[a.status] ?? 9) - (priority[b.status] ?? 9)
      || b.observations - a.observations
      || a.symbol.localeCompare(b.symbol)
    )).slice(0, 14);
  }

  function candidateNode(row) {
    const node = element('article', `entry-candidate-node ${row.status}`);
    const head = element('div', 'entry-candidate-head');
    head.appendChild(element('strong', '', row.symbol));
    head.appendChild(element('span', '', row.status === 'pass' ? 'READY' : row.status === 'bar-ready' ? 'BAR READY' : row.status.toUpperCase()));
    node.appendChild(head);
    node.appendChild(element('p', '', row.stateLabel));
    if (row.required > 0) {
      const track = element('div', 'entry-candidate-progress');
      const fill = element('i');
      fill.style.width = `${Math.min(100, Math.max(2, row.observations / row.required * 100))}%`;
      track.appendChild(fill);
      node.appendChild(track);
    }
    const details = [...row.admissions, ...row.reasons].slice(0, 2);
    if (details.length) {
      const reason = element('small', 'entry-candidate-reason', details.join(' · '));
      reason.title = [...row.admissions, ...row.reasons].join('\n');
      node.appendChild(reason);
    }
    return node;
  }

  function arrow(label) {
    const node = element('div', 'entry-flow-arrow');
    node.appendChild(element('span', '', '↓'));
    if (label) node.appendChild(element('small', '', label));
    return node;
  }

  function globalLayer(byStage) {
    const section = element('section', 'entry-flow-layer global');
    const heading = element('div', 'entry-flow-layer-title');
    heading.appendChild(element('strong', '', 'GLOBAL SAFETY · ALL'));
    heading.appendChild(element('span', '', '전역 안전 노드는 모두 순서대로 통과'));
    section.appendChild(heading);
    const rail = element('div', 'entry-flow-rail');
    ['engine_running', 'live_armed', 'market_session'].forEach((stage, index) => {
      if (index) rail.appendChild(element('span', 'entry-flow-inline-arrow', '→'));
      rail.appendChild(stageNode(byStage[stage], stage));
    });
    section.appendChild(rail);
    return section;
  }

  function candidateLayer(link) {
    const data = link?.data || {};
    const rows = candidateRows(data);
    const selectedCount = Number(data.candidate_filter_selected_count ?? data.sample?.length ?? 0);
    const inputCount = Number(data.candidate_filter_input_count || rows.length || 0);
    const section = element('section', `entry-flow-layer candidates ${selectedCount > 0 ? 'has-ready' : 'waiting'}`);
    const heading = element('div', 'entry-flow-layer-title');
    heading.appendChild(element('strong', '', 'CANDIDATE PREPARATION · ANY'));
    heading.appendChild(element('span', '', `병렬 ${inputCount}개 입력 · ${selectedCount}개 다음 단계 진행`));
    section.appendChild(heading);
    const rule = element('div', 'entry-any-rule');
    rule.appendChild(element('b', '', 'ANY'));
    rule.appendChild(element('span', '', '한 종목만 준비돼도 아래 전략 선출로 진행'));
    section.appendChild(rule);
    const grid = element('div', 'entry-candidate-grid');
    rows.forEach((row) => grid.appendChild(candidateNode(row)));
    if (!rows.length) {
      grid.appendChild(candidateNode({
        symbol: 'NO STREAM', status: 'stale', stateLabel: '표시할 실시간 후보가 없습니다.',
        observations: 0, required: 0, reasons: [], admissions: [],
      }));
    }
    section.appendChild(grid);
    const counts = data.candidate_filter_reason_counts || {};
    const summary = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 4);
    if (summary.length) {
      const tags = element('div', 'entry-flow-summary-tags');
      summary.forEach(([reason, count]) => tags.appendChild(element('span', '', `${reason} ×${count}`)));
      section.appendChild(tags);
    }
    return section;
  }

  function downstreamLayer(byStage) {
    const section = element('section', 'entry-flow-layer downstream');
    const heading = element('div', 'entry-flow-layer-title');
    heading.appendChild(element('strong', '', 'ELECTION & EXECUTION'));
    heading.appendChild(element('span', '', '준비된 종목만 합류해 독립 전략을 평가'));
    section.appendChild(heading);
    const rail = element('div', 'entry-flow-rail');
    ['micro_buy_intents', 'strategy_election', 'position'].forEach((stage, index) => {
      if (index) rail.appendChild(element('span', 'entry-flow-inline-arrow', '→'));
      rail.appendChild(stageNode(byStage[stage], stage));
    });
    section.appendChild(rail);
    return section;
  }

  function dagNode(item) {
    const status = ['pass', 'blocked', 'running', 'probe'].includes(item?.status)
      ? item.status : 'pending';
    const node = element('article', `entry-dag-node ${status}`);
    const head = element('div', 'entry-candidate-head');
    head.appendChild(element('span', 'entry-flow-state-dot'));
    head.appendChild(element('strong', '', item?.label || item?.id || 'NODE'));
    const metadata = item?.metadata || {};
    const statusLabel = status === 'probe'
      ? `PROBE ${Math.round(Number(metadata.risk_size_fraction || 0.10) * 100)}%`
      : status.toUpperCase();
    head.appendChild(element('span', '', statusLabel));
    node.appendChild(head);
    node.appendChild(element('p', '', item?.detail || '평가 대기'));
    const reasons = Array.isArray(metadata.reason_codes) ? metadata.reason_codes : [];
    if (reasons.length) {
      const reason = element('small', 'entry-candidate-reason', reasons.slice(0, 2).join(' · '));
      reason.title = reasons.join('\n');
      node.appendChild(reason);
    }
    return node;
  }

  function dagLayer(layer) {
    const parallel = Boolean(layer?.parallel);
    const section = element(
      'section',
      `entry-flow-layer decision-dag-layer ${parallel ? 'parallel' : 'merge'} ${layer?.id || ''}`,
    );
    const heading = element('div', 'entry-flow-layer-title');
    const title = element('strong', '', layer?.label || layer?.id || 'LAYER');
    const policy = element('b', `entry-dag-policy ${parallel ? 'parallel' : 'merge'}`, layer?.policy || 'MAP');
    title.appendChild(policy);
    heading.appendChild(title);
    heading.appendChild(element('span', '', layer?.description || ''));
    section.appendChild(heading);
    const grid = element('div', 'entry-dag-grid');
    (layer?.nodes || []).forEach((item) => grid.appendChild(dagNode(item)));
    section.appendChild(grid);
    return section;
  }

  function decisionDagGraph(dag) {
    const graph = element('div', 'entry-flow-graph entry-decision-dag');
    graph.setAttribute('role', 'img');
    graph.setAttribute('aria-label', '전체 실시간 거래 의사결정 계층과 독립 병렬 분기 그래프');
    const telemetry = dag?.telemetry || {};
    const advancingSymbols = Array.isArray(dag?.advancing_symbols) ? dag.advancing_symbols : [];
    const banner = element('div', 'entry-dag-telemetry');
    if (dag?.cycle_id) banner.setAttribute('title', `cycle ${dag.cycle_id}`);
    banner.appendChild(element('strong', '', dag?.execution_model || 'HIERARCHICAL PARALLEL DAG'));
    banner.appendChild(element(
      'span',
      '',
      `전략 ${Number(telemetry.strategy_evaluation_branch_count || 0)}분기 · `
        + `${Number(telemetry.strategy_evaluation_workers || 1)} workers · `
        + `${telemetry.strategy_evaluation_duration_ms ?? '-'}ms`
        + (telemetry.macro_mismatch_probe_enabled
          ? ` · MACRO SOFT-GATE · PROBE ${Math.round(Number(telemetry.macro_mismatch_probe_size_fraction || 0.10) * 100)}%`
          : ''),
    ));
    banner.appendChild(element('span', '', `SAME-CYCLE ${advancingSymbols.length} symbols`));
    graph.appendChild(banner);
    (dag?.layers || []).forEach((layer, index) => {
      if (index) {
        const previous = dag.layers[index - 1];
        const label = previous?.parallel
          ? `${previous.policy} merge · 완료된 분기별 전달`
          : `${previous?.policy || 'ALL'} 통과`;
        graph.appendChild(arrow(label));
      }
      graph.appendChild(dagLayer(layer));
    });
    return graph;
  }

  function render(payload) {
    const container = document.getElementById('blockade-chain');
    const headline = document.getElementById('blockade-headline');
    const verdict = document.getElementById('blockade-verdict');
    if (!container || !headline || !verdict) return;
    const terminal = Boolean(document.querySelector('.terminal-shell'));
    const baseBadge = terminal ? 'status-chip' : 'badge';
    container.innerHTML = '';

    if (!payload || payload.ok === false) {
      headline.textContent = '진입 흐름 진단을 불러오지 못했습니다.';
      verdict.textContent = '확인 불가';
      verdict.className = `${baseBadge} waiting`;
      return;
    }

    const chain = Array.isArray(payload.chain) ? payload.chain : [];
    const byStage = Object.fromEntries(chain.map((link) => [link.stage, link]));
    if (payload.trading_possible) {
      verdict.textContent = '진입 가능';
      verdict.className = baseBadge;
      headline.textContent = '전역 가드와 후보·전략 흐름이 모두 주문 단계까지 연결됐습니다.';
    } else {
      const label = STAGE_LABELS[payload.blocking_stage] || payload.blocking_stage || '대기';
      verdict.textContent = `현재 정지 · ${label}`;
      verdict.className = `${baseBadge} blocked`;
      headline.textContent = payload.blocking_detail || '다음 조건을 기다리고 있습니다.';
    }

    const decisionDag = payload.decision_dag;
    if (decisionDag && Array.isArray(decisionDag.layers) && decisionDag.layers.length) {
      container.appendChild(decisionDagGraph(decisionDag));
      return;
    }

    const graph = element('div', 'entry-flow-graph');
    graph.setAttribute('role', 'img');
    graph.setAttribute('aria-label', '전역 안전 가드와 종목별 병렬 준비 상태를 표시하는 진입 흐름 그래프');
    graph.appendChild(globalLayer(byStage));
    graph.appendChild(arrow('전역 가드 통과 후 후보를 병렬 준비'));
    graph.appendChild(candidateLayer(byStage.buy_candidates));
    graph.appendChild(arrow('ANY merge · 준비 완료 종목만 합류'));
    graph.appendChild(downstreamLayer(byStage));
    container.appendChild(graph);
  }

  window.renderEntryBlockadeGraph = render;
}());

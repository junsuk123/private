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
  state.dashboard = data;
  renderDashboard(data);
  await refreshHistory();
}

async function refreshHistory() {
  const data = await fetchJson(`/api/account/asset-history?range=${encodeURIComponent(state.range)}`);
  renderAssetChart(data.points || []);
}

function renderDashboard(data) {
  const snapshot = data.snapshot || {};
  document.getElementById('account-source').textContent = `${snapshot.source || 'unknown'} | updated ${formatTime(snapshot.updated_at)}`;
  const badge = document.getElementById('account-stale-badge');
  badge.textContent = snapshot.is_stale ? `stale ${Math.round(snapshot.stale_seconds || 0)}s` : 'live';
  badge.className = snapshot.is_stale ? 'badge warn' : 'badge';

  renderKpis(snapshot);
  renderAllocation(snapshot.asset_allocations || []);
  renderHoldings(data.holdings || []);
  renderTrades(data.trades || []);
  renderCash(data.cash || []);
  renderSystem(snapshot, data.logs || {});
  renderLogs(data.logs || {});
}

function renderKpis(snapshot) {
  const rows = [
    ['총자산', fmtKrw(snapshot.total_asset_krw), `순자산 ${fmtKrw(snapshot.net_asset_krw)}`],
    ['평가손익', fmtKrw(snapshot.unrealized_pnl_krw), fmtPct(snapshot.total_pnl_rate), clsPnl(snapshot.unrealized_pnl_krw)],
    ['실현손익', fmtKrw(snapshot.realized_pnl_period_krw), '기간 기준', clsPnl(snapshot.realized_pnl_period_krw)],
    ['주문가능 KRW', fmtMoney((snapshot.orderable_cash_by_currency || {}).KRW || snapshot.krw_cash, 'KRW'), '원화'],
    ['주문가능 USD', fmtMoney((snapshot.orderable_cash_by_currency || {}).USD || 0, 'USD'), '외화'],
    ['현금성 자산', fmtKrw(snapshot.cash_equivalent_krw), `외화 ${fmtKrw(snapshot.foreign_cash_krw)}`],
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
  body.innerHTML = filtered.length ? filtered.map((row) => `
    <tr>
      <td><strong>${row.ticker}</strong><br><small>${row.name || ''}</small></td>
      <td>${row.market_group === 'domestic' ? '국내' : '해외'}<br><small>${row.exchange || row.market}</small></td>
      <td>${Number(row.quantity || 0).toLocaleString()}</td>
      <td>${fmtMoney(row.average_price, row.currency)}</td>
      <td>${fmtMoney(row.current_price, row.currency)}</td>
      <td>${fmtKrw(row.evaluation_amount_krw)}</td>
      <td class="${clsPnl(row.unrealized_pnl_krw)}">${fmtKrw(row.unrealized_pnl_krw)}</td>
      <td class="${clsPnl(row.unrealized_pnl_rate)}">${fmtPct(row.unrealized_pnl_rate)}</td>
      <td>${fmtPct(row.weight_of_total_asset)}</td>
      <td>${row.currency}</td>
    </tr>
  `).join('') : `<tr class="empty-row"><td colspan="10">현재 보유 종목 없음</td></tr>`;
}

function renderTrades(rows) {
  const body = document.getElementById('trades-body');
  body.innerHTML = rows.length ? rows.slice(0, 20).map((row) => `
    <tr>
      <td>${formatTime(row.occurred_at)}</td>
      <td>${row.market_group || row.market}</td>
      <td>${row.ticker}</td>
      <td>${row.side}</td>
      <td>${Number(row.ordered_quantity || 0).toLocaleString()}</td>
      <td>${Number(row.filled_quantity || 0).toLocaleString()}</td>
      <td>${fmtKrw(row.amount_krw)}</td>
      <td>${row.order_status}</td>
    </tr>
  `).join('') : `<tr class="empty-row"><td colspan="8">거래 이력 수집 중</td></tr>`;
}

function renderCash(rows) {
  const body = document.getElementById('cash-body');
  body.innerHTML = rows.length ? rows.map((row) => `
    <tr>
      <td>${row.currency}</td>
      <td>${fmtMoney(row.cash_balance, row.currency)}</td>
      <td>${fmtMoney(row.orderable_amount, row.currency)}</td>
      <td>${fmtKrw(row.krw_equivalent)}</td>
      <td>${Number(row.fx_rate_to_krw || 0).toLocaleString('ko-KR', { maximumFractionDigits: 4 })}</td>
    </tr>
  `).join('') : `<tr class="empty-row"><td colspan="5">예수금 정보 수집 중</td></tr>`;
}

function renderSystem(snapshot, logs) {
  const warnings = snapshot.data_quality_warnings || [];
  const items = [
    ['KIS 상태', snapshot.is_live ? '연결됨' : 'fallback'],
    ['마지막 계좌 갱신', formatTime(snapshot.updated_at)],
    ['데이터 stale', snapshot.is_stale ? '주의' : '정상'],
    ['API 경고', String(warnings.length)],
    ['보유 종목', String((state.dashboard.holdings || []).length)],
    ['로그 오류', logs.last_error ? '있음' : '없음'],
  ];
  document.getElementById('system-strip').innerHTML = items.map(([label, value]) => `
    <div class="system-pill"><span>${label}</span><strong>${value}</strong></div>
  `).join('');
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

document.getElementById('account-refresh').addEventListener('click', refreshDashboard);
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
setInterval(() => refreshDashboard().catch(() => {}), 15000);

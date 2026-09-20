from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.account_dashboard import AccountDashboardService
from app.account_snapshot_store import AccountSnapshotStore
from app.operations_dashboard import build_operations_overview
from app.web_account_routes import create_account_router


def test_independent_markets_and_first_blocker_do_not_claim_order_permission():
    payload = build_operations_overview(
        trading={"running": True, "status": {"strategy_session": {"selected_symbol": "005930", "selected_strategy": "KR_LONG"}}},
        markets={"KR": {"phase": "REGULAR", "allows_new_entry": True}, "US": {"phase": "DAYTIME", "allows_new_entry": True}},
        blockade={"ok": True, "trading_possible": True, "chain": [
            {"stage": "engine", "ok": True},
            {"stage": "live_armed", "ok": False, "detail": "명시적 실거래 권한 없음"},
            {"stage": "market_session", "ok": True},
        ]},
    )
    assert payload["markets"]["KR"]["allows_new_entry"] is True
    assert payload["markets"]["US"]["allows_new_entry"] is True
    assert payload["trading"]["trading_possible"] is False
    assert payload["trading"]["blocker"]["stage"] == "live_armed"
    assert payload["trading"]["pipeline"][1]["first_blocker"] is True
    assert payload["trading"]["pipeline"][2]["state"] == "passed"
    assert payload["trading"]["execution_scope"] == "single_account_owner"


def test_absent_observations_stay_unknown():
    payload = build_operations_overview()
    assert payload["markets"]["KR"]["allows_new_entry"] is None
    assert payload["markets"]["US"]["quote_age_seconds"] is None
    assert payload["trading"]["armed"] is None
    assert payload["trading"]["trading_possible"] is False
    assert payload["runtime"]["backend"] is None


def test_ontology_policy_snapshot_is_passed_through_without_io_or_permission_changes():
    snapshot = {"policy_count": 1, "policies": {"KR:005930": {"policy": {
        "symbol": "005930", "market": "KR", "valid_for_entry": True,
        "hard_stop_rate": .007, "daily_loss_budget_rate": .004,
    }}}}
    payload = build_operations_overview(trading={"status": {"ontology_policy": snapshot}})
    assert payload["trading"]["ontology_policy"] == snapshot
    assert payload["trading"]["trading_possible"] is False
    assert build_operations_overview()["trading"]["ontology_policy"] == {}


def test_engine_events_cannot_masquerade_as_orders_and_fills_keep_submission_side():
    payload = build_operations_overview(orders=[
        {"at": "2026-09-20T02:00:00+00:00", "symbol": "AAPL", "kind": "STATUS", "outcome": "filled", "broker_order_id": "order-1", "filled_quantity": 1},
        {"at": "2026-09-20T01:59:59+00:00", "symbol": "AAPL", "kind": "BUY", "outcome": "submitted", "broker_order_id": "order-1", "quantity": 1},
        {"at": "2026-09-20T01:59:58+00:00", "symbol": "AAPL", "kind": "BUY", "outcome": "eval_error"},
        {"at": "2026-09-20T01:59:57+00:00", "kind": "CYCLE", "outcome": "error"},
        {"symbol": "MSFT", "kind": "BUY", "outcome": "signal"},
    ])
    assert len(payload["trading"]["orders"]) == 1
    order = payload["trading"]["orders"][0]
    assert order["order_status"] == "FILLED"
    assert order["side"] == "BUY"
    assert order["ordered_quantity"] == order["filled_quantity"] == 1


def test_wired_operations_endpoint_does_not_refresh_broker_or_heavy_diagnostics(monkeypatch):
    from app import web

    def forbidden(*args, **kwargs):
        raise AssertionError("read-only dashboard called an expensive refresh")

    monkeypatch.setattr(web, "_realtime_trading_engine", None)
    monkeypatch.setattr(web, "_realtime_trading_worker", None)
    monkeypatch.setattr(web, "_context_runtime", None)
    monkeypatch.setattr(web, "_refresh_live_account_basis_for_auto", forbidden)
    monkeypatch.setattr(web, "_entry_blockade_chain", forbidden)
    response = TestClient(web.app).get("/api/operations/overview")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload["markets"]) == {"KR", "US"}
    assert payload["trading"]["running"] is False
    assert payload["trading"]["orders"] == []


def test_cached_account_summary_never_refreshes_broker_and_expires(tmp_path):
    now = datetime.now(timezone.utc)
    store = AccountSnapshotStore(tmp_path / "account.sqlite3")
    calls = []
    service = AccountDashboardService(status_provider=lambda: calls.append("broker"), store=store)
    for age, expected in ((1, "live"), (180, "last_known")):
        stamp = (now - timedelta(seconds=age)).isoformat()
        store.save_dashboard({"snapshot": {"created_at": (now + timedelta(seconds=age)).isoformat(), "updated_at": stamp, "is_live": True, "source": "kis_live_account", "total_asset_krw": 0}})
        summary = service.cached_asset_summary()
        assert summary["status"] == expected
        assert summary["authoritative"] is (expected == "live")
        assert summary["snapshot"]["total_asset_krw"] == 0
    assert calls == []


def test_default_dashboard_and_advanced_terminal_keep_independent_routes(tmp_path):
    service = AccountDashboardService(store=AccountSnapshotStore(tmp_path / "account.sqlite3"))
    app = FastAPI()
    app.include_router(create_account_router(service=service, operations_provider=lambda: build_operations_overview()))
    client = TestClient(app)
    page = client.get("/account")
    assert page.status_code == 200
    assert "계좌와 손익" in page.text and "지금의 시장" in page.text
    assert "operations_dashboard.js" in page.text
    assert "strategy_terminal.js" not in page.text
    assert "strategy_terminal.js" in client.get("/account/advanced").text
    assert client.get("/api/operations/overview").json()["markets"]["US"]["phase"] == "UNKNOWN"
    assert client.get("/api/account/summary").json()["status"] == "unavailable"


def test_browser_rendering_and_partial_fetch_failure_offline():
    """Execute actual rendering against a DOM double, without any network access."""
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed for browser contract tests")
    script_path = Path(__file__).parents[1] / "src/app/static/operations_dashboard.js"
    script = r'''
const assert = require('node:assert/strict');
const nodes = new Map();
const get = id => {
  if (!nodes.has(id)) nodes.set(id, {textContent:'', innerHTML:'', className:'', classList:{toggle(){}}, addEventListener(){}});
  return nodes.get(id);
};
global.document = {hidden:true, getElementById: id => id === 'refresh' ? null : get(id)};
global.window = {setTimeout:()=>0, clearTimeout(){}};
const ui = require(SCRIPT_PATH);
document.getElementById = get;
assert.equal(ui.money(null), '—');
assert.equal(ui.money(0), '0원');
assert.equal(ui.money(''), '—');
assert.equal(ui.numeric(undefined), null);
ui.renderAccount({status:'unavailable', snapshot:null});
assert.equal(get('pnl-today').textContent, '—');
const account = {status:'live', authoritative:true, last_verified_at:new Date().toISOString(), snapshot:{total_asset_krw:0, realized_pnl_today_krw:0, unrealized_pnl_krw:-20, orderable_cash_by_currency:{KRW:0}}, holdings:[{ticker:'<script>alert(1)</script>', quantity:1, current_price:10, unrealized_pnl_krw:-20}]};
ui.renderAccount(account);
assert.equal(get('assets').textContent, '0원');
assert.equal(get('pnl-today').textContent, '0원');
assert.equal(get('pnl-unrealized').className, 'negative');
assert(!get('holdings').innerHTML.includes('<script>'));
ui.renderMarkets({KR:{phase:'REGULAR',allows_new_entry:true},US:{phase:'PREMARKET',allows_new_entry:true}});
assert(get('markets').innerHTML.includes('국내 시장'));
assert(get('markets').innerHTML.includes('프리마켓'));
ui.renderTrading({running:true, armed:false, trading_possible:true, orders:[]});
assert(!get('blocker-title').textContent.includes('조건 통과'));
ui.renderRuntime({requested_backend:'NPU',backend:'CPU',fallback_reason:'NPU_UNAVAILABLE'});
assert.equal(get('backend-badge').textContent, 'CPU');
assert.equal(get('runtime-note').textContent, 'NPU_UNAVAILABLE');
const policy = {symbol:'005930',market:'KR',as_of:new Date(Date.now()-1000).toISOString(),expires_at:new Date(Date.now()+10000).toISOString(),valid_for_entry:true,target_return_rate:.015,all_in_cost_rate:.003,hard_stop_rate:.007,trailing_stop_rate:.004,maximum_holding_seconds:900,position_cap:.04,daily_loss_budget_rate:.004,confidence:.8,early_exit_confirmations:2,policy_id:'policy-1',evidence_id:'cycle-1',evidence:[{metric:'spread_rate',source:'kis_realtime',derived_from:['<script>unsafe</script>']}]};
ui.renderPolicies({ontology_policy:{policies:{'KR:005930':{policy}}}});
assert(get('ontology-policies').innerHTML.includes('1.2%'));
assert(get('ontology-policies').innerHTML.includes('0.7%'));
assert(get('ontology-policies').innerHTML.includes('정책 조건 충족'));
assert(!get('ontology-policies').innerHTML.includes('<script>'));
ui.renderPolicies({ontology_policy:{policies:{'KR:005930':{policy}}}},false);
assert(get('ontology-policies').innerHTML.includes('재확인 필요'));
assert(!get('ontology-policies').innerHTML.includes('정책 조건 충족'));
ui.renderPolicies({ontology_policy:{policies:{'KR:005930':{policy:{...policy,expires_at:new Date(Date.now()-1).toISOString()}}}}});
assert(get('ontology-policies').innerHTML.includes('재분석 필요'));
ui.renderPolicies({});
assert(get('ontology-policies').innerHTML.includes('분석된 정책이 아직 없습니다'));
ui.state.operations = {generated_at:new Date().toISOString(), markets:{}};
ui.state.account = account;
document.hidden = false;
global.fetch = async url => {
  if(url.includes('/operations/')) throw Error('offline');
  return {ok:true,json:async()=>account};
};
(async()=>{
  await ui.refresh();
  assert.equal(get('blocker-title').textContent, '운영 상태 연결 지연');
  assert.equal(get('assets').textContent, '0원');
  assert(get('connection').textContent.includes('운영 상태'));
  assert.equal(ui.state.busy,false);
})().catch(error=>{ console.error(error); process.exitCode=1; });
'''.replace("SCRIPT_PATH", json.dumps(str(script_path)))
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr

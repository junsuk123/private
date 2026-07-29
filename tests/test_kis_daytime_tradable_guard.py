"""US 주간거래 eligibility guard (dtm_tr_psbl_yn).

KIS supports only a subset of US names for daytime trading; the rest are
rejected at the broker. 해외주식 상품기본정보 (CTPF1702R) exposes
``dtm_tr_psbl_yn``, so the check happens locally and produces an explainable
block instead of a broker error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.execution.kis_errors import LiveExecutionBlocked
from app.execution.kis_real import US_PRODUCT_TYPE_CODES, KisDevelopersApiClient
from app.schemas.domain import FinalOrder, OrderSide, OrderType


class RecordingTransport:
    def __init__(self, dtm_yn: str = "Y", *, fail: bool = False):
        self.dtm_yn = dtm_yn
        self.fail = fail
        self.calls: list[dict] = []

    def request(self, method, url, headers=None, params=None, body=None, timeout=None):
        self.calls.append({"method": method, "url": url, "params": params or {}, "body": body or {}})
        if "/quotations/search-info" in url:
            if self.fail:
                raise RuntimeError("reference data unavailable")
            return {"rt_cd": "0", "msg_cd": "OK", "msg1": "", "output": {"dtm_tr_psbl_yn": self.dtm_yn}}
        if "/oauth2/tokenP" in url or "/oauth2/Approval" in url:
            return {"access_token": "t", "expires_in": 86400, "approval_key": "k"}
        if "hashkey" in url:
            return {"HASH": "h"}
        return {"rt_cd": "0", "msg_cd": "OK", "msg1": "accepted", "output": {"ODNO": "123", "KRX_FWDG_ORD_ORGNO": "1"}}

    def search_info_calls(self):
        return [c for c in self.calls if "/quotations/search-info" in c["url"]]

    def order_calls(self):
        return [c for c in self.calls if "/trading/daytime-order" in c["url"] or c["url"].endswith("/trading/order")]


def client(tmp_path: Path, transport: RecordingTransport) -> KisDevelopersApiClient:
    return KisDevelopersApiClient(
        app_key="app",
        app_secret="secret",
        account_no="12345678-01",
        paper=False,
        enabled=True,
        transport=transport,
        token_cache_path=tmp_path / "token.json",
    )


def us_order(ticker: str = "AAPL") -> FinalOrder:
    return FinalOrder(
        ticker=ticker,
        market="NASDAQ",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=1,
        limit_price=212.34,
        manual_approval_required=False,
    )


@pytest.fixture(autouse=True)
def _daytime_session(monkeypatch):
    monkeypatch.setenv("KIS_FORCE_OVERSEAS_DAYTIME_ORDER", "true")
    monkeypatch.delenv("KIS_ENFORCE_US_DAYTIME_TRADABLE", raising=False)
    monkeypatch.setenv("KIS_LIVE_ENABLED", "true")


class TestProductInfoLookup:
    def test_uses_the_documented_tr_and_product_type(self, tmp_path):
        transport = RecordingTransport("Y")
        api = client(tmp_path, transport)
        assert api.is_us_daytime_tradable("AAPL", "NASD") is True
        call = transport.search_info_calls()[0]
        assert call["params"]["PRDT_TYPE_CD"] == US_PRODUCT_TYPE_CODES["NASD"] == "512"
        assert call["params"]["PDNO"] == "AAPL"

    @pytest.mark.parametrize("exchange,code", [("NASD", "512"), ("NYSE", "513"), ("AMEX", "529")])
    def test_product_type_per_us_exchange(self, tmp_path, exchange, code):
        transport = RecordingTransport("Y")
        client(tmp_path, transport).is_us_daytime_tradable("AAPL", exchange)
        assert transport.search_info_calls()[0]["params"]["PRDT_TYPE_CD"] == code

    def test_answer_is_cached(self, tmp_path):
        transport = RecordingTransport("Y")
        api = client(tmp_path, transport)
        for _ in range(5):
            api.is_us_daytime_tradable("AAPL", "NASD")
        assert len(transport.search_info_calls()) == 1

    def test_unknown_exchange_returns_none_without_calling(self, tmp_path):
        transport = RecordingTransport("Y")
        assert client(tmp_path, transport).is_us_daytime_tradable("0700", "SEHK") is None
        assert transport.search_info_calls() == []

    def test_lookup_failure_is_unknown_not_a_block(self, tmp_path):
        transport = RecordingTransport(fail=True)
        assert client(tmp_path, transport).is_us_daytime_tradable("AAPL", "NASD") is None


class TestOrderGuard:
    def test_daytime_ineligible_symbol_is_blocked_locally(self, tmp_path):
        transport = RecordingTransport("N")
        api = client(tmp_path, transport)
        with pytest.raises(LiveExecutionBlocked) as excinfo:
            api.place_limit_order(us_order("XYZ"))
        # reason_codes must stay a tuple of codes; passing a bare string here
        # made the message split into single characters.
        assert excinfo.value.reason_codes == ("US_DAYTIME_TRADING_NOT_SUPPORTED:XYZ",)
        assert "US_DAYTIME_TRADING_NOT_SUPPORTED:XYZ" in str(excinfo.value)
        # Nothing was sent to the order endpoint.
        assert transport.order_calls() == []

    def test_daytime_eligible_symbol_reaches_the_daytime_endpoint(self, tmp_path):
        transport = RecordingTransport("Y")
        receipt = client(tmp_path, transport).place_limit_order(us_order())
        assert receipt.accepted
        assert any("/trading/daytime-order" in c["url"] for c in transport.order_calls())

    def test_unknown_eligibility_still_submits(self, tmp_path):
        """A reference-data outage must not stop all daytime trading."""
        transport = RecordingTransport(fail=True)
        receipt = client(tmp_path, transport).place_limit_order(us_order())
        assert receipt.accepted

    def test_enforcement_can_be_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KIS_ENFORCE_US_DAYTIME_TRADABLE", "false")
        transport = RecordingTransport("N")
        receipt = client(tmp_path, transport).place_limit_order(us_order("XYZ"))
        assert receipt.accepted

    def test_regular_session_does_not_consult_daytime_eligibility(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KIS_FORCE_OVERSEAS_DAYTIME_ORDER", "false")
        transport = RecordingTransport("N")
        receipt = client(tmp_path, transport).place_limit_order(us_order("XYZ"))
        assert receipt.accepted
        assert transport.search_info_calls() == []

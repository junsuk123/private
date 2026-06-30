from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from app.execution.kis_mock import MockKisExecution, MockKisOrderReceipt, MockKisPortfolio
from app.schemas.domain import AccountSnapshot, FinalOrder, Holding, MarketSnapshot, OrderSide, SourceMetadata


KIS_LIVE_BASE_URL = "https://openapi.koreainvestment.com:9443"
KIS_PAPER_BASE_URL = "https://openapivts.koreainvestment.com:29443"
KIS_SECRETS_FILE = Path("config/secrets/kis_api_keys.env")
KIS_TOKEN_CACHE_SKEW_SECONDS = 60
_KIS_ENV_FILE_LOADED = False


class KisApiError(RuntimeError):
    def __init__(self, message: str, response: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.response = response or {}


class KisTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """Send one KIS REST request and return the decoded JSON payload."""


class UrllibKisTransport:
    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        data = None
        request_headers = dict(headers)
        if body is not None:
            data = json.dumps(body, separators=(",", ":")).encode("utf-8")
            request_headers["Content-Length"] = str(len(data))
        request = urllib.request.Request(
            url,
            data=data,
            headers=request_headers,
            method=method.upper(),
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError:
                decoded = {"rt_cd": "1", "msg1": payload, "http_status": exc.code}
            message = f"KIS HTTP {exc.code}: {decoded.get('msg1', payload)}"
            raise KisApiError(message, decoded) from exc
        try:
            return json.loads(payload) if payload else {}
        except json.JSONDecodeError as exc:
            raise KisApiError(f"KIS returned non-JSON response: {payload[:200]}") from exc


@dataclass(frozen=True)
class KisCredentials:
    app_key: str
    app_secret: str
    account_no: str
    account_product_code: str = "01"

    @classmethod
    def from_env(cls, paper: bool = False) -> "KisCredentials":
        load_kis_env_file()
        prefix = "KIS_PAPER_" if paper else "KIS_"
        app_key = os.getenv(f"{prefix}APP_KEY") or os.getenv("KIS_APP_KEY", "")
        app_secret = os.getenv(f"{prefix}APP_SECRET") or os.getenv("KIS_APP_SECRET", "")
        account_no = os.getenv(f"{prefix}ACCOUNT_NO") or os.getenv("KIS_ACCOUNT_NO", "")
        product_code = (
            os.getenv(f"{prefix}ACCOUNT_PRODUCT_CODE")
            or os.getenv("KIS_ACCOUNT_PRODUCT_CODE")
            or "01"
        )
        return cls.from_values(app_key, app_secret, account_no, product_code)

    @classmethod
    def from_values(
        cls,
        app_key: str | None,
        app_secret: str | None,
        account_no: str | None,
        account_product_code: str | None = None,
    ) -> "KisCredentials":
        account = (account_no or "").replace("-", "").strip()
        product_code = (account_product_code or "").strip()
        if len(account) >= 10 and not product_code:
            product_code = account[8:10]
            account = account[:8]
        if len(account) == 10:
            product_code = account[8:10]
            account = account[:8]
        return cls(
            app_key=(app_key or "").strip(),
            app_secret=(app_secret or "").strip(),
            account_no=account,
            account_product_code=product_code or "01",
        )

    def validate(self) -> None:
        missing = []
        if not self.app_key:
            missing.append("app_key")
        if not self.app_secret:
            missing.append("app_secret")
        if not self.account_no:
            missing.append("account_no")
        if missing:
            raise RuntimeError(f"Missing KIS credentials: {', '.join(missing)}")


@dataclass(frozen=True)
class KisEndpointSet:
    base_url: str
    paper: bool = False

    @classmethod
    def for_mode(cls, paper: bool, base_url: str | None = None) -> "KisEndpointSet":
        default_base_url = KIS_PAPER_BASE_URL if paper else KIS_LIVE_BASE_URL
        return cls(base_url=(base_url or default_base_url), paper=paper)

    def tr_id_for_order(self, side: OrderSide) -> str:
        if side == OrderSide.BUY:
            return "VTTC0012U" if self.paper else "TTTC0012U"
        return "VTTC0011U" if self.paper else "TTTC0011U"

    def overseas_tr_id_for_order(self, exchange_code: str, side: OrderSide) -> str:
        exchange = exchange_code.upper()
        if side == OrderSide.BUY:
            tr_id = (
                "TTTT1002U"
                if exchange in {"NASD", "NYSE", "AMEX"}
                else "TTTS1002U"
                if exchange == "SEHK"
                else "TTTS0202U"
                if exchange == "SHAA"
                else "TTTS0305U"
                if exchange == "SZAA"
                else "TTTS0308U"
                if exchange == "TKSE"
                else "TTTS0311U"
                if exchange in {"HASE", "VNSE"}
                else ""
            )
        else:
            tr_id = (
                "TTTT1006U"
                if exchange in {"NASD", "NYSE", "AMEX"}
                else "TTTS1001U"
                if exchange == "SEHK"
                else "TTTS1005U"
                if exchange == "SHAA"
                else "TTTS0304U"
                if exchange == "SZAA"
                else "TTTS0307U"
                if exchange == "TKSE"
                else "TTTS0310U"
                if exchange in {"HASE", "VNSE"}
                else ""
            )
        if not tr_id:
            raise ValueError(f"unsupported overseas exchange for KIS order: {exchange_code}")
        return "V" + tr_id[1:] if self.paper else tr_id

    def overseas_daytime_tr_id_for_order(self, side: OrderSide) -> str:
        if side == OrderSide.BUY:
            return "VTTT6036U" if self.paper else "TTTS6036U"
        return "VTTT6037U" if self.paper else "TTTS6037U"

    @property
    def order_revise_cancel_tr_id(self) -> str:
        return "VTTC0013U" if self.paper else "TTTC0013U"

    def overseas_revise_cancel_tr_id(self, exchange_code: str) -> str:
        exchange = exchange_code.upper()
        if exchange in {"NASD", "NYSE", "AMEX"}:
            return "VTTT1004U" if self.paper else "TTTT1004U"
        tr_id = (
            "TTTS1003U"
            if exchange == "SEHK"
            else "TTTS0302U"
            if exchange == "SHAA"
            else "TTTS0306U"
            if exchange == "SZAA"
            else "TTTS0309U"
            if exchange == "TKSE"
            else "TTTS0312U"
            if exchange in {"HASE", "VNSE"}
            else ""
        )
        if not tr_id:
            raise ValueError(f"unsupported overseas exchange for KIS revise/cancel: {exchange_code}")
        return "V" + tr_id[1:] if self.paper else tr_id

    @property
    def overseas_daytime_revise_cancel_tr_id(self) -> str:
        return "VTTT6038U" if self.paper else "TTTS6038U"

    @property
    def order_status_tr_id(self) -> str:
        return "VTTC8001R" if self.paper else "TTTC8001R"

    @property
    def balance_tr_id(self) -> str:
        return "VTTC8434R" if self.paper else "TTTC8434R"

    @property
    def orderable_cash_tr_id(self) -> str:
        return "VTTC8908R" if self.paper else "TTTC8908R"

    @property
    def overseas_present_balance_tr_id(self) -> str:
        return "VTRP6504R" if self.paper else "CTRP6504R"

    @property
    def overseas_balance_tr_id(self) -> str:
        return "VTTS3012R" if self.paper else "TTTS3012R"

    @property
    def overseas_orderable_cash_tr_id(self) -> str:
        return "VTTS3007R" if self.paper else "TTTS3007R"

    @property
    def overseas_order_status_tr_id(self) -> str:
        return "VTTS3035R" if self.paper else "TTTS3035R"


class KisDevelopersApiClient:
    """KIS Developers REST broker adapter for domestic cash stock orders.

    The same request builder is used for paper and live modes. Tests can inject
    a fake KisTransport, while production uses urllib and real KIS credentials.
    """

    def __init__(
        self,
        app_key: str | None = None,
        app_secret: str | None = None,
        account_no: str | None = None,
        account_product_code: str | None = None,
        base_url: str | None = None,
        enabled: bool | None = None,
        paper: bool | None = None,
        transport: KisTransport | None = None,
        access_token: str | None = None,
        token_expires_at: datetime | None = None,
        token_cache_path: str | Path | None = None,
    ) -> None:
        load_kis_env_file()
        self.paper = _env_bool("KIS_PAPER_TRADING", False) if paper is None else paper
        self.credentials = (
            KisCredentials.from_env(self.paper)
            if app_key is None and app_secret is None and account_no is None
            else KisCredentials.from_values(app_key, app_secret, account_no, account_product_code)
        )
        self.endpoints = KisEndpointSet.for_mode(self.paper, base_url or os.getenv("KIS_BASE_URL"))
        self.enabled = (
            _env_bool("KIS_LIVE_ENABLED", False)
            if enabled is None
            else bool(enabled)
        )
        self.transport = transport or UrllibKisTransport()
        self.timeout = float(os.getenv("KIS_TIMEOUT_SECONDS", "10"))
        self._access_token = access_token
        self._token_expires_at = token_expires_at
        self._token_source = "injected" if access_token else None
        self._token_cache_path = (
            Path(token_cache_path)
            if token_cache_path is not None
            else _default_token_cache_path(self.paper)
        )
        self._orders: dict[str, FinalOrder] = {}
        self._order_org_numbers: dict[str, str] = {}

    def place_limit_order(self, order: FinalOrder) -> MockKisOrderReceipt:
        self._ensure_enabled()
        if order.side not in {OrderSide.BUY, OrderSide.SELL}:
            raise ValueError(f"unsupported KIS order side: {order.side}")
        if _is_overseas_order(order):
            return self._place_overseas_limit_order(order)
        body = self._order_body(order)
        response = self._post(
            "/uapi/domestic-stock/v1/trading/order-cash",
            tr_id=self.endpoints.tr_id_for_order(order.side),
            body=body,
            include_hashkey=True,
        )
        self._ensure_success(response, "KIS order rejected")
        output = response.get("output") or {}
        order_id = str(output.get("ODNO") or output.get("odno") or "")
        if not order_id:
            order_id = f"KIS-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        self._orders[order_id] = order
        self._order_org_numbers[order_id] = str(output.get("KRX_FWDG_ORD_ORGNO") or output.get("krx_fwdg_ord_orgno") or "")
        return MockKisOrderReceipt(
            order_id=order_id,
            accepted=True,
            status="ACCEPTED",
            message=str(response.get("msg1") or "KIS accepted the order."),
            order=order,
            submitted_at=datetime.now(timezone.utc),
        )

    def _place_overseas_limit_order(self, order: FinalOrder) -> MockKisOrderReceipt:
        exchange_code = _overseas_exchange_code(order.market)
        body = self._overseas_order_body(order, exchange_code)
        path = "/uapi/overseas-stock/v1/trading/order"
        tr_id = self.endpoints.overseas_tr_id_for_order(exchange_code, order.side)
        if _is_us_daytime_order_session(order.market):
            path = "/uapi/overseas-stock/v1/trading/daytime-order"
            tr_id = self.endpoints.overseas_daytime_tr_id_for_order(order.side)
        response = self._post(
            path,
            tr_id=tr_id,
            body=body,
            include_hashkey=True,
        )
        self._ensure_success(response, "KIS overseas order rejected")
        output = response.get("output") or {}
        order_id = str(output.get("ODNO") or output.get("odno") or output.get("KRX_FWDG_ORD_ORGNO") or "")
        if not order_id:
            order_id = f"KISOVRS-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        self._orders[order_id] = order
        self._order_org_numbers[order_id] = str(output.get("KRX_FWDG_ORD_ORGNO") or output.get("krx_fwdg_ord_orgno") or "")
        return MockKisOrderReceipt(
            order_id=order_id,
            accepted=True,
            status="ACCEPTED",
            message=str(response.get("msg1") or "KIS accepted the overseas order."),
            order=order,
            submitted_at=datetime.now(timezone.utc),
        )

    def amend_limit_order(self, order_id: str, replacement: FinalOrder) -> MockKisOrderReceipt:
        """Revise an existing unfilled KIS limit order to the replacement quantity/price."""
        self._ensure_enabled()
        if _is_overseas_order(replacement):
            return self._amend_overseas_limit_order(order_id, replacement)
        body = self._revise_cancel_body(order_id, replacement, revise=True)
        response = self._post(
            "/uapi/domestic-stock/v1/trading/order-rvsecncl",
            tr_id=self.endpoints.order_revise_cancel_tr_id,
            body=body,
            include_hashkey=True,
        )
        self._ensure_success(response, "KIS order revise rejected")
        return self._receipt_from_revise_cancel_response(response, replacement, fallback_order_id=order_id)

    def cancel_order(self, order_id: str, order: FinalOrder) -> MockKisOrderReceipt:
        self._ensure_enabled()
        if _is_overseas_order(order):
            return self._cancel_overseas_order(order_id, order)
        body = self._revise_cancel_body(order_id, order, revise=False)
        response = self._post(
            "/uapi/domestic-stock/v1/trading/order-rvsecncl",
            tr_id=self.endpoints.order_revise_cancel_tr_id,
            body=body,
            include_hashkey=True,
        )
        self._ensure_success(response, "KIS order cancel rejected")
        return self._receipt_from_revise_cancel_response(response, order, fallback_order_id=order_id, status="CANCELED")

    def _amend_overseas_limit_order(self, order_id: str, replacement: FinalOrder) -> MockKisOrderReceipt:
        exchange_code = _overseas_exchange_code(replacement.market)
        body = self._overseas_revise_cancel_body(order_id, replacement, exchange_code, revise=True)
        path = "/uapi/overseas-stock/v1/trading/order-rvsecncl"
        tr_id = self.endpoints.overseas_revise_cancel_tr_id(exchange_code)
        if _is_us_daytime_order_session(replacement.market):
            path = "/uapi/overseas-stock/v1/trading/daytime-order-rvsecncl"
            tr_id = self.endpoints.overseas_daytime_revise_cancel_tr_id
        response = self._post(path, tr_id=tr_id, body=body, include_hashkey=True)
        self._ensure_success(response, "KIS overseas order revise rejected")
        return self._receipt_from_revise_cancel_response(response, replacement, fallback_order_id=order_id)

    def _cancel_overseas_order(self, order_id: str, order: FinalOrder) -> MockKisOrderReceipt:
        exchange_code = _overseas_exchange_code(order.market)
        body = self._overseas_revise_cancel_body(order_id, order, exchange_code, revise=False)
        path = "/uapi/overseas-stock/v1/trading/order-rvsecncl"
        tr_id = self.endpoints.overseas_revise_cancel_tr_id(exchange_code)
        if _is_us_daytime_order_session(order.market):
            path = "/uapi/overseas-stock/v1/trading/daytime-order-rvsecncl"
            tr_id = self.endpoints.overseas_daytime_revise_cancel_tr_id
        response = self._post(path, tr_id=tr_id, body=body, include_hashkey=True)
        self._ensure_success(response, "KIS overseas order cancel rejected")
        return self._receipt_from_revise_cancel_response(response, order, fallback_order_id=order_id, status="CANCELED")

    def get_order_status(self, order_id: str) -> MockKisExecution:
        self._ensure_enabled()
        order = self._orders.get(order_id)
        if order is not None and _is_overseas_market_name(order.market, order.ticker):
            return self._get_overseas_order_status(order_id, order)
        params = self._order_status_params(order_id, order)
        response = self._get(
            "/uapi/domestic-stock/v1/trading/inquire-daily-ccld",
            tr_id=self.endpoints.order_status_tr_id,
            params=params,
        )
        self._ensure_success(response, "KIS order-status lookup failed")
        row = _first_response_row(response)
        return self._execution_from_status(order_id, row, order)

    def _get_overseas_order_status(self, order_id: str, order: FinalOrder) -> MockKisExecution:
        response = self._get(
            "/uapi/overseas-stock/v1/trading/inquire-ccnl",
            tr_id=self.endpoints.overseas_order_status_tr_id,
            params=self._overseas_order_status_params(order),
        )
        self._ensure_success(response, "KIS overseas order-status lookup failed")
        rows = [
            row
            for row in _response_rows(response)
            if str(row.get("odno") or row.get("ODNO") or "") == order_id
        ]
        row = rows[0] if rows else _first_response_row(response)
        return self._overseas_execution_from_status(order_id, row, order)

    def get_portfolio(self) -> MockKisPortfolio:
        self._ensure_enabled()
        domestic_error: KisApiError | None = None
        holdings: tuple[Holding, ...] = ()
        cash = 0.0
        cash_by_currency: dict[str, float] = {"KRW": 0.0}
        try:
            response = self._get(
                "/uapi/domestic-stock/v1/trading/inquire-balance",
                tr_id=self.endpoints.balance_tr_id,
                params=self._balance_params(),
            )
            self._ensure_success(response, "KIS portfolio lookup failed")
            holdings = tuple(self._holding_from_balance(row) for row in response.get("output1") or ())
            summary = response.get("output2") or response.get("output3") or []
            summary_row = summary[0] if isinstance(summary, list) and summary else summary
            cash = _domestic_cash_from_balance_summary(summary_row, holdings)
            try:
                orderable_cash = self._get_domestic_orderable_cash()
            except KisApiError:
                orderable_cash = 0.0
            if orderable_cash > 0:
                cash = orderable_cash
            cash_by_currency = _cash_by_currency_from_summary(summary_row, cash)
        except KisApiError as exc:
            domestic_error = exc
        overseas_holdings: tuple[Holding, ...] = ()
        try:
            overseas_holdings = self._get_overseas_holdings()
        except Exception:
            overseas_holdings = ()
        try:
            foreign_cash_by_currency, foreign_cash_krw, total_assets_krw = self._get_overseas_cash_balance()
        except KisApiError:
            if domestic_error is not None:
                raise domestic_error
            raise
        try:
            foreign_orderable = self._get_overseas_orderable_cash_by_currency()
        except Exception:
            foreign_orderable = {}
        foreign_cash_by_currency.update(
            {currency: amount for currency, amount in foreign_orderable.items() if amount > 0}
        )
        if domestic_error is not None and not foreign_cash_by_currency and foreign_cash_krw <= 0:
            raise domestic_error
        cash_by_currency.update(foreign_cash_by_currency)
        all_holdings = holdings + overseas_holdings
        domestic_position_value = sum(max(0.0, holding.market_value) for holding in holdings)
        raw_position_value = sum(max(0.0, holding.market_value) for holding in all_holdings)
        cash_equivalent_krw = cash + foreign_cash_krw
        has_overseas_assets = bool(overseas_holdings) or foreign_cash_krw > 0
        if has_overseas_assets and total_assets_krw >= cash + domestic_position_value:
            cash_equivalent_krw = max(0.0, total_assets_krw - raw_position_value)
        account = AccountSnapshot(
            cash=cash,
            holdings=all_holdings,
            base_currency="KRW",
            cash_by_currency=cash_by_currency,
            cash_equivalent_krw=cash_equivalent_krw,
        )
        return MockKisPortfolio(
            account=account,
            market_prices={holding.ticker: holding.last_price for holding in holdings},
            updated_at=datetime.now(timezone.utc),
        )

    def get_market_snapshot(
        self,
        ticker: str,
        market: str,
        *,
        company_name: str | None = None,
        sector: str | None = None,
    ) -> MarketSnapshot:
        self._ensure_enabled()
        symbol = ticker.upper().strip()
        market_name = market.upper().strip()
        if _is_overseas_market_name(market_name, symbol):
            return self._get_overseas_market_snapshot(symbol, market_name, company_name=company_name, sector=sector)
        return self._get_domestic_market_snapshot(symbol, market_name, company_name=company_name, sector=sector)

    def _get_domestic_market_snapshot(
        self,
        ticker: str,
        market: str,
        *,
        company_name: str | None = None,
        sector: str | None = None,
    ) -> MarketSnapshot:
        response = self._get(
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            tr_id="FHKST01010100",
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker},
        )
        self._ensure_success(response, "KIS domestic quote lookup failed")
        output = response.get("output") or {}
        now = datetime.now(timezone.utc)
        price = _first_float(output, "stck_prpr", "prpr", "last", "close")
        volume = _first_float(output, "acml_vol", "cntg_vol", "tvol")
        trading_value = _first_float(output, "acml_tr_pbmn", "hts_avls", "tamt")
        if trading_value <= 0:
            trading_value = price * max(0.0, volume)
        volatility = abs(_first_float(output, "prdy_ctrt", "rate", "prdy_vrss_sign")) / 100.0
        return MarketSnapshot(
            ticker=ticker,
            market=market or "KRX",
            company_name=company_name or ticker,
            sector=sector or "Unknown",
            last_price=price,
            average_daily_trading_value=trading_value,
            volatility_20d=max(0.005, min(0.20, volatility or 0.03)),
            source=_broker_quote_source(ticker, "domestic", now),
        )

    def _get_overseas_market_snapshot(
        self,
        ticker: str,
        market: str,
        *,
        company_name: str | None = None,
        sector: str | None = None,
    ) -> MarketSnapshot:
        exchange_code = _overseas_quote_exchange_code(market)
        response = self._get(
            "/uapi/overseas-price/v1/quotations/price",
            tr_id="HHDFS00000300",
            params={"AUTH": "", "EXCD": exchange_code, "SYMB": ticker},
        )
        self._ensure_success(response, "KIS overseas quote lookup failed")
        output = response.get("output") or {}
        now = datetime.now(timezone.utc)
        price = _first_float(output, "last", "ovrs_nmix_prpr", "stck_prpr", "base")
        volume = _first_float(output, "tvol", "acml_vol", "pvol")
        trading_value = _first_float(output, "tamt", "acml_tr_pbmn")
        if trading_value <= 0:
            trading_value = price * max(0.0, volume)
        volatility = abs(_first_float(output, "rate", "prdy_ctrt")) / 100.0
        return MarketSnapshot(
            ticker=ticker,
            market=market or "US-LISTED",
            company_name=company_name or ticker,
            sector=sector or "Unknown",
            last_price=price,
            average_daily_trading_value=trading_value,
            volatility_20d=max(0.005, min(0.20, volatility or 0.03)),
            source=_broker_quote_source(ticker, "overseas", now),
        )

    def issue_access_token(self, force_refresh: bool = False) -> str:
        self.credentials.validate()
        if not force_refresh:
            cached = self._load_env_token() or self._load_cached_token()
            if cached:
                return cached
        self._ensure_token_cache_writable()
        response = self.transport.request(
            "POST",
            self._url("/oauth2/tokenP"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            body={
                "grant_type": "client_credentials",
                "appkey": self.credentials.app_key,
                "appsecret": self.credentials.app_secret,
            },
            timeout=self.timeout,
        )
        token = str(response.get("access_token") or "")
        if not token:
            raise KisApiError("KIS token response did not include access_token.", response)
        expires_in = int(response.get("expires_in") or 60 * 60 * 24)
        self._access_token = token
        self._token_expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=max(60, expires_in - KIS_TOKEN_CACHE_SKEW_SECONDS)
        )
        self._token_source = "issued"
        self._write_cached_token()
        return token

    @property
    def token_source(self) -> str | None:
        return self._token_source

    def _get(self, path: str, tr_id: str, params: dict[str, Any]) -> dict[str, Any]:
        return self.transport.request(
            "GET",
            self._url(path),
            headers=self._headers(tr_id),
            params=params,
            timeout=self.timeout,
        )

    def _post(
        self,
        path: str,
        tr_id: str,
        body: dict[str, Any],
        include_hashkey: bool = False,
    ) -> dict[str, Any]:
        headers = self._headers(tr_id)
        if include_hashkey:
            headers["hashkey"] = self._hashkey(body)
        return self.transport.request(
            "POST",
            self._url(path),
            headers=headers,
            body=body,
            timeout=self.timeout,
        )

    def _headers(self, tr_id: str) -> dict[str, str]:
        self.credentials.validate()
        return {
            "Content-Type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self._valid_token()}",
            "appkey": self.credentials.app_key,
            "appsecret": self.credentials.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }

    def _hashkey(self, body: dict[str, Any]) -> str:
        response = self.transport.request(
            "POST",
            self._url("/uapi/hashkey"),
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "appkey": self.credentials.app_key,
                "appsecret": self.credentials.app_secret,
            },
            body=body,
            timeout=self.timeout,
        )
        value = str(response.get("HASH") or response.get("hash") or "")
        if not value:
            raise KisApiError("KIS hashkey response did not include HASH.", response)
        return value

    def _valid_token(self) -> str:
        now = datetime.now(timezone.utc)
        if self._access_token and self._token_expires_at and self._token_expires_at > now:
            return self._access_token
        if self._access_token and self._token_expires_at is None:
            return self._access_token
        cached = self._load_env_token(now) or self._load_cached_token(now)
        if cached:
            return cached
        return self.issue_access_token()

    def _load_env_token(self, now: datetime | None = None) -> str | None:
        mode_prefix = "KIS_PAPER_" if self.paper else "KIS_LIVE_"
        token = (
            os.getenv(f"{mode_prefix}ACCESS_TOKEN")
            or (None if self.paper else os.getenv("KIS_ACCESS_TOKEN"))
            or ""
        ).strip()
        if not token:
            return None
        expires_at = _parse_datetime(
            os.getenv(f"{mode_prefix}ACCESS_TOKEN_EXPIRES_AT")
            or (None if self.paper else os.getenv("KIS_ACCESS_TOKEN_EXPIRES_AT"))
        )
        if expires_at is not None and expires_at <= (now or datetime.now(timezone.utc)):
            return None
        self._access_token = token
        self._token_expires_at = expires_at
        self._token_source = "env"
        return token

    def _load_cached_token(self, now: datetime | None = None) -> str | None:
        if not self._token_cache_path.exists():
            return None
        try:
            payload = json.loads(self._token_cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        token = str(payload.get("access_token") or "")
        expires_at = _parse_datetime(payload.get("expires_at"))
        mode = str(payload.get("mode") or "")
        if not token or expires_at is None or mode != ("paper" if self.paper else "live"):
            return None
        if expires_at <= (now or datetime.now(timezone.utc)):
            return None
        self._access_token = token
        self._token_expires_at = expires_at
        self._token_source = "cache"
        return token

    def _write_cached_token(self) -> None:
        if not self._access_token or not self._token_expires_at:
            raise RuntimeError("KIS token cache write requested without an access token.")
        payload = {
            "access_token": self._access_token,
            "expires_at": self._token_expires_at.isoformat(),
            "mode": "paper" if self.paper else "live",
            "base_url": self.endpoints.base_url,
            "account_suffix": self.credentials.account_no[-2:],
        }
        try:
            self._token_cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._token_cache_path.with_suffix(f"{self._token_cache_path.suffix}.tmp")
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(self._token_cache_path)
        except OSError as exc:
            raise RuntimeError(
                f"KIS access token was issued but could not be saved to {self._token_cache_path}."
            ) from exc
        saved = self._load_cached_token()
        if saved != self._access_token:
            raise RuntimeError(
                f"KIS access token was issued but cache verification failed at {self._token_cache_path}."
            )

    def _ensure_token_cache_writable(self) -> None:
        if self._token_cache_path.exists() and self._token_cache_path.is_dir():
            raise RuntimeError(f"KIS token cache path is a directory: {self._token_cache_path}")
        try:
            self._token_cache_path.parent.mkdir(parents=True, exist_ok=True)
            probe_path = self._token_cache_path.with_suffix(f"{self._token_cache_path.suffix}.probe")
            probe_path.write_text("ok", encoding="utf-8")
            probe_path.unlink()
        except OSError as exc:
            raise RuntimeError(
                f"KIS token cache is not writable at {self._token_cache_path}; "
                "fix this before requesting a new access token."
            ) from exc

    def _order_body(self, order: FinalOrder) -> dict[str, str]:
        return {
            "CANO": self.credentials.account_no,
            "ACNT_PRDT_CD": self.credentials.account_product_code,
            "PDNO": order.ticker,
            "ORD_DVSN": _domestic_order_division_code(),
            "ORD_QTY": str(int(order.quantity)),
            "ORD_UNPR": str(int(round(order.limit_price))),
            "EXCG_ID_DVSN_CD": "KRX",
            "SLL_TYPE": "",
            "CNDT_PRIC": "",
        }

    def _overseas_order_body(self, order: FinalOrder, exchange_code: str) -> dict[str, str]:
        return {
            "CANO": self.credentials.account_no,
            "ACNT_PRDT_CD": self.credentials.account_product_code,
            "OVRS_EXCG_CD": exchange_code,
            "PDNO": order.ticker.upper(),
            "ORD_QTY": str(int(order.quantity)),
            "OVRS_ORD_UNPR": _format_overseas_price(order.limit_price),
            "CTAC_TLNO": "",
            "MGCO_APTM_ODNO": "",
            "SLL_TYPE": "00" if order.side == OrderSide.SELL else "",
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": "00",
        }

    def _revise_cancel_body(self, order_id: str, order: FinalOrder, *, revise: bool) -> dict[str, str]:
        return {
            "CANO": self.credentials.account_no,
            "ACNT_PRDT_CD": self.credentials.account_product_code,
            "KRX_FWDG_ORD_ORGNO": self._order_org_numbers.get(order_id, ""),
            "ORGN_ODNO": order_id,
            "ORD_DVSN": _domestic_order_division_code(),
            "RVSE_CNCL_DVSN_CD": "01" if revise else "02",
            "ORD_QTY": str(int(order.quantity)),
            "ORD_UNPR": str(int(round(order.limit_price))) if revise else "0",
            "QTY_ALL_ORD_YN": "N" if revise else "Y",
            "CNDT_PRIC": "",
            "EXCG_ID_DVSN_CD": "KRX",
        }

    def _overseas_revise_cancel_body(
        self,
        order_id: str,
        order: FinalOrder,
        exchange_code: str,
        *,
        revise: bool,
    ) -> dict[str, str]:
        return {
            "CANO": self.credentials.account_no,
            "ACNT_PRDT_CD": self.credentials.account_product_code,
            "OVRS_EXCG_CD": exchange_code,
            "PDNO": order.ticker.upper(),
            "ORGN_ODNO": order_id,
            "RVSE_CNCL_DVSN_CD": "01" if revise else "02",
            "ORD_QTY": str(int(order.quantity)),
            "OVRS_ORD_UNPR": _format_overseas_price(order.limit_price) if revise else "0",
            "CTAC_TLNO": "",
            "MGCO_APTM_ODNO": "",
            "ORD_SVR_DVSN_CD": "0",
        }

    def _receipt_from_revise_cancel_response(
        self,
        response: dict[str, Any],
        order: FinalOrder,
        *,
        fallback_order_id: str,
        status: str = "ACCEPTED",
    ) -> MockKisOrderReceipt:
        output = response.get("output") or {}
        order_id = str(output.get("ODNO") or output.get("odno") or fallback_order_id)
        org_no = str(output.get("KRX_FWDG_ORD_ORGNO") or output.get("krx_fwdg_ord_orgno") or "")
        self._orders[order_id] = order
        if org_no:
            self._order_org_numbers[order_id] = org_no
        if order_id != fallback_order_id:
            self._orders.pop(fallback_order_id, None)
            if org_no:
                self._order_org_numbers.pop(fallback_order_id, None)
        return MockKisOrderReceipt(
            order_id=order_id,
            accepted=True,
            status=status,
            message=str(response.get("msg1") or "KIS accepted the order revise/cancel request."),
            order=order,
            submitted_at=datetime.now(timezone.utc),
        )

    def _order_status_params(self, order_id: str, order: FinalOrder | None) -> dict[str, str]:
        today = datetime.now().strftime("%Y%m%d")
        return {
            "CANO": self.credentials.account_no,
            "ACNT_PRDT_CD": self.credentials.account_product_code,
            "INQR_STRT_DT": today,
            "INQR_END_DT": today,
            "SLL_BUY_DVSN_CD": "02" if order and order.side == OrderSide.BUY else "00",
            "INQR_DVSN": "00",
            "PDNO": order.ticker if order else "",
            "CCLD_DVSN": "00",
            "ORD_GNO_BRNO": "",
            "ODNO": order_id,
            "INQR_DVSN_3": "00",
            "INQR_DVSN_1": "",
            "EXCG_ID_DVSN_CD": "KRX",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

    def _overseas_order_status_params(self, order: FinalOrder) -> dict[str, str]:
        today = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=2)).strftime("%Y%m%d")
        exchange_code = _overseas_exchange_code(order.market)
        side = "02" if order.side == OrderSide.BUY else "01"
        return {
            "CANO": self.credentials.account_no,
            "ACNT_PRDT_CD": self.credentials.account_product_code,
            "PDNO": "%",
            "ORD_STRT_DT": start,
            "ORD_END_DT": today,
            "SLL_BUY_DVSN": side,
            "CCLD_NCCS_DVSN": "00",
            "OVRS_EXCG_CD": exchange_code,
            "SORT_SQN": "DS",
            "ORD_DT": "",
            "ORD_GNO_BRNO": "",
            "ODNO": "",
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        }

    def _balance_params(self) -> dict[str, str]:
        return {
            "CANO": self.credentials.account_no,
            "ACNT_PRDT_CD": self.credentials.account_product_code,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

    def _domestic_orderable_cash_params(self) -> dict[str, str]:
        return {
            "CANO": self.credentials.account_no,
            "ACNT_PRDT_CD": self.credentials.account_product_code,
            "PDNO": "",
            "ORD_UNPR": "",
            "ORD_DVSN": "00",
            "CMA_EVLU_AMT_ICLD_YN": "N",
            "OVRS_ICLD_YN": "N",
        }

    def _get_domestic_orderable_cash(self) -> float:
        response = self._get(
            "/uapi/domestic-stock/v1/trading/inquire-psbl-order",
            tr_id=self.endpoints.orderable_cash_tr_id,
            params=self._domestic_orderable_cash_params(),
        )
        self._ensure_success(response, "KIS domestic orderable-cash lookup failed")
        output = response.get("output") or {}
        return _first_float(output, "ord_psbl_cash")

    def _overseas_present_balance_params(self, nation_code: str = "000") -> dict[str, str]:
        return {
            "CANO": self.credentials.account_no,
            "ACNT_PRDT_CD": self.credentials.account_product_code,
            "WCRC_FRCR_DVSN_CD": "02",
            "NATN_CD": nation_code,
            "TR_MKET_CD": "00",
            "INQR_DVSN_CD": "00",
        }

    def _overseas_balance_params(self, exchange_code: str = "") -> dict[str, str]:
        return {
            "CANO": self.credentials.account_no,
            "ACNT_PRDT_CD": self.credentials.account_product_code,
            "OVRS_EXCG_CD": exchange_code,
            "TR_CRCY_CD": "USD",
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        }

    def _get_overseas_holdings(self) -> tuple[Holding, ...]:
        response = self._get(
            "/uapi/overseas-stock/v1/trading/inquire-balance",
            tr_id=self.endpoints.overseas_balance_tr_id,
            params=self._overseas_balance_params(),
        )
        self._ensure_success(response, "KIS overseas balance lookup failed")
        holdings = tuple(
            holding
            for row in _response_rows(response)
            if (holding := self._overseas_holding_from_balance(row)) is not None
        )
        return holdings

    def _get_overseas_cash_balance(self) -> tuple[dict[str, float], float, float]:
        balances: dict[str, float] = {}
        foreign_cash_krw = 0.0
        total_assets_krw = 0.0
        for nation_code in ("000", "840"):
            try:
                response = self._get(
                    "/uapi/overseas-stock/v1/trading/inquire-present-balance",
                    tr_id=self.endpoints.overseas_present_balance_tr_id,
                    params=self._overseas_present_balance_params(nation_code),
                )
                self._ensure_success(response, "KIS overseas present balance lookup failed")
            except KisApiError:
                if nation_code == "840":
                    return balances, foreign_cash_krw, total_assets_krw
                continue
            balances.update(_foreign_orderable_cash_by_currency_from_overseas_response(response, nation_code))
            foreign_cash_krw = max(foreign_cash_krw, _foreign_cash_krw_from_overseas_response(response))
            total_assets_krw = max(total_assets_krw, _total_assets_krw_from_overseas_response(response))
            if balances or foreign_cash_krw > 0 or total_assets_krw > 0:
                break
        return balances, foreign_cash_krw, total_assets_krw

    def _overseas_orderable_cash_params(self, currency: str = "USD", exchange_code: str = "NASD") -> dict[str, str]:
        default_item = os.getenv("KIS_OVERSEAS_ORDERABLE_ITEM_CD", "AAPL").strip().upper() or "AAPL"
        return {
            "CANO": self.credentials.account_no,
            "ACNT_PRDT_CD": self.credentials.account_product_code,
            "OVRS_EXCG_CD": exchange_code,
            "TR_CRCY_CD": currency,
            "OVRS_ORD_UNPR": "1",
            "ITEM_CD": default_item,
        }

    def _get_overseas_orderable_cash_by_currency(self) -> dict[str, float]:
        balances: dict[str, float] = {}
        for currency, exchange in (("USD", "NASD"),):
            try:
                response = self._get(
                    "/uapi/overseas-stock/v1/trading/inquire-psamount",
                    tr_id=self.endpoints.overseas_orderable_cash_tr_id,
                    params=self._overseas_orderable_cash_params(currency, exchange),
                )
                self._ensure_success(response, "KIS overseas orderable-cash lookup failed")
            except KisApiError:
                continue
            amount = _overseas_orderable_amount_from_response(response)
            if amount > 0:
                balances[currency] = amount
        return balances

    def _execution_from_status(
        self,
        order_id: str,
        row: dict[str, Any],
        order: FinalOrder | None,
    ) -> MockKisExecution:
        quantity = int(_to_float(row.get("tot_ccld_qty") or row.get("ord_qty") or 0))
        price = _to_float(row.get("avg_prvs") or row.get("ord_unpr") or 0)
        ticker = str(row.get("pdno") or (order.ticker if order else ""))
        side_code = str(row.get("sll_buy_dvsn_cd") or "")
        side = order.side if order else (OrderSide.SELL if side_code == "01" else OrderSide.BUY)
        status = "FILLED" if quantity > 0 else "OPEN"
        return MockKisExecution(
            order_id=order_id,
            ticker=ticker,
            side=side,
            quantity=quantity,
            price=price,
            executed_value=quantity * price,
            status=status,
            message=str(row.get("ord_tmd") or "KIS order status received."),
            executed_at=datetime.now(timezone.utc),
        )

    def _overseas_execution_from_status(
        self,
        order_id: str,
        row: dict[str, Any],
        order: FinalOrder | None,
    ) -> MockKisExecution:
        ordered_quantity = int(_to_float(row.get("ft_ord_qty") or row.get("ord_qty") or 0))
        filled_quantity = int(_to_float(row.get("ft_ccld_qty") or row.get("tot_ccld_qty") or 0))
        open_quantity = int(_to_float(row.get("nccs_qty") or max(0, ordered_quantity - filled_quantity)))
        price = _first_float(row, "ft_ccld_unpr3", "ft_ord_unpr3")
        ticker = str(row.get("pdno") or (order.ticker if order else ""))
        side_code = str(row.get("sll_buy_dvsn_cd") or "")
        side = order.side if order else (OrderSide.SELL if side_code == "01" else OrderSide.BUY)
        status = "FILLED" if filled_quantity > 0 and open_quantity <= 0 else "OPEN"
        if filled_quantity > 0 and open_quantity > 0:
            status = "PARTIALLY_FILLED"
        return MockKisExecution(
            order_id=order_id,
            ticker=ticker,
            side=side,
            quantity=filled_quantity,
            price=price,
            executed_value=filled_quantity * price,
            status=status,
            message=str(row.get("prcs_stat_name") or row.get("ord_tmd") or "KIS overseas order status received."),
            executed_at=datetime.now(timezone.utc),
        )

    def _holding_from_balance(self, row: dict[str, Any]) -> Holding:
        quantity = int(_to_float(row.get("hldg_qty") or row.get("ord_psbl_qty") or 0))
        average_price = _to_float(row.get("pchs_avg_pric") or 0)
        last_price = _to_float(row.get("prpr") or row.get("bfdy_cprs_icdc") or average_price)
        opened_at = self._holding_opened_at_from_balance(row)
        return Holding(
            ticker=str(row.get("pdno") or ""),
            market="KR",
            company_name=str(row.get("prdt_name") or row.get("pdno") or ""),
            sector="Unknown",
            quantity=quantity,
            average_price=average_price,
            last_price=last_price,
            opened_at=opened_at,
        )

    def _overseas_holding_from_balance(self, row: dict[str, Any]) -> Holding | None:
        ticker = str(
            row.get("ovrs_pdno")
            or row.get("pdno")
            or row.get("symb")
            or row.get("prdt_code")
            or ""
        ).upper().strip()
        quantity = int(_to_float(row.get("ovrs_cblc_qty") or row.get("hldg_qty") or row.get("ord_psbl_qty") or 0))
        if not ticker or quantity <= 0:
            return None
        average_price = _first_float(row, "pchs_avg_pric", "avg_unpr", "frcr_pchs_amt1")
        last_price = _first_float(row, "now_pric2", "ovrs_now_pric1", "last", "prpr", "evlu_pfls_rt")
        if last_price <= 0:
            market_value = _first_float(row, "ovrs_stck_evlu_amt", "frcr_evlu_amt", "evlu_amt", "pchs_amt")
            last_price = market_value / quantity if market_value > 0 else average_price
        return Holding(
            ticker=ticker,
            market=_overseas_exchange_code(str(row.get("ovrs_excg_cd") or row.get("tr_mket_name") or "")),
            company_name=str(row.get("ovrs_item_name") or row.get("prdt_name") or ticker),
            sector="Unknown",
            quantity=quantity,
            average_price=average_price,
            last_price=last_price,
            opened_at=self._holding_opened_at_from_balance(row),
        )

    @staticmethod
    def _holding_opened_at_from_balance(row: dict[str, Any]) -> datetime | None:
        for date_key in ("pchs_dt", "acqs_dt", "bns_dt", "ord_dt", "buy_dt"):
            value = str(row.get(date_key) or "").strip()
            if len(value) == 8 and value.isdigit():
                try:
                    return datetime.strptime(value, "%Y%m%d").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
        return None

    def _url(self, path: str) -> str:
        return f"{self.endpoints.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _ensure_enabled(self) -> None:
        if not self.enabled:
            raise RuntimeError(
                "KIS trading is disabled. Set KIS_LIVE_ENABLED=true after approval gates are ready."
            )

    @staticmethod
    def _ensure_success(response: dict[str, Any], prefix: str) -> None:
        if str(response.get("rt_cd", "0")) != "0":
            message = str(response.get("msg1") or response.get("msg_cd") or response)
            raise KisApiError(f"{prefix}: {message}", response)


def _first_response_row(response: dict[str, Any]) -> dict[str, Any]:
    output = response.get("output1") or response.get("output") or []
    if isinstance(output, list) and output:
        return dict(output[0])
    if isinstance(output, dict):
        return output
    return {}


def _to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return 0.0


def _first_float(data: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = _to_float(data.get(key))
        if value > 0:
            return value
    return 0.0


def _domestic_cash_from_balance_summary(summary_row: dict[str, Any], holdings: tuple[Holding, ...]) -> float:
    explicit_cash = _first_float(
        summary_row,
        "dnca_tot_amt",
        "prvs_rcdl_excc_amt",
        "d2_auto_rdpt_amt",
        "cash",
    )
    if explicit_cash > 0:
        return explicit_cash
    total_evaluation = _first_float(summary_row, "tot_evlu_amt", "tot_asst_amt", "real_nass_amt")
    stock_evaluation = _first_float(
        summary_row,
        "scts_evlu_amt",
        "evlu_amt_smtl_amt",
        "pchs_amt_smtl_amt",
        "stock_evlu_amt",
    )
    if stock_evaluation <= 0:
        stock_evaluation = sum(max(0.0, holding.market_value) for holding in holdings)
    if total_evaluation > 0 and stock_evaluation > 0:
        return max(0.0, total_evaluation - stock_evaluation)
    return 0.0


def _broker_quote_source(ticker: str, scope: str, observed_at: datetime) -> SourceMetadata:
    return SourceMetadata(
        source_name="KIS broker quote",
        retrieved_at=observed_at,
        raw_url=f"kis://quotations/{scope}/{ticker}",
        source_id=f"kis-quote:{scope}:{ticker}:{observed_at.isoformat()}",
        source_type="broker_api",
        trust_level=5,
        observed_at=observed_at,
        latency_sec=0.0,
        is_realtime=True,
        license_policy="broker_account",
        quality_score=1.0,
    )


def _is_overseas_market_name(market: str, ticker: str) -> bool:
    return not (ticker.isdigit() and len(ticker) == 6) or any(
        token in market
        for token in ("US", "NASDAQ", "NASD", "NYSE", "AMEX", "SEHK", "SHAA", "SZAA", "TKSE", "HASE", "VNSE", "OVERSEAS")
    )


def _is_overseas_order(order: FinalOrder) -> bool:
    market = str(order.market or "").upper()
    return not (order.ticker.isdigit() and len(order.ticker) == 6) or any(
        token in market
        for token in ("US", "NASDAQ", "NASD", "NYSE", "AMEX", "SEHK", "SHAA", "SZAA", "TKSE", "HASE", "VNSE")
    )


def _overseas_exchange_code(market: str) -> str:
    value = str(market or "").upper()
    if "NASDAQ" in value or "NASD" in value:
        return "NASD"
    if "NYSE" in value:
        return "NYSE"
    if "AMEX" in value:
        return "AMEX"
    if "SEHK" in value or "HONG" in value:
        return "SEHK"
    if "SHAA" in value or "SHANGHAI" in value:
        return "SHAA"
    if "SZAA" in value or "SHENZHEN" in value:
        return "SZAA"
    if "TKSE" in value or "JAPAN" in value or "TOKYO" in value:
        return "TKSE"
    if "HASE" in value or "HANOI" in value:
        return "HASE"
    if "VNSE" in value or "VIETNAM" in value or "HOCHIMINH" in value:
        return "VNSE"
    if value in {"US", "US-LISTED", "GLOBAL", "OVERSEAS"}:
        return os.getenv("KIS_DEFAULT_US_EXCHANGE", "NASD").upper()
    return value or os.getenv("KIS_DEFAULT_US_EXCHANGE", "NASD").upper()


def _overseas_quote_exchange_code(market: str) -> str:
    value = _overseas_exchange_code(market)
    return {
        "NASD": "NAS",
        "NYSE": "NYS",
        "AMEX": "AMS",
        "SEHK": "HKS",
        "SHAA": "SHS",
        "SZAA": "SZS",
        "TKSE": "TSE",
        "HASE": "HNX",
        "VNSE": "HSX",
    }.get(value, value)


def _format_overseas_price(value: float) -> str:
    # KIS 미국주식 호가 단위: $1 이상은 소수점 2자리, $1 미만은 소수점 4자리까지만 허용(APTR0057).
    price = float(value)
    decimals = 2 if price >= 1.0 else 4
    text = f"{price:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if text else "0"


def _domestic_order_division_code(now: datetime | None = None) -> str:
    forced = os.getenv("KIS_DOMESTIC_ORD_DVSN", "").strip()
    if forced:
        return forced
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        return "00"
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    local = current.astimezone(ZoneInfo("Asia/Seoul"))
    minute = local.hour * 60 + local.minute
    if 8 * 60 + 30 <= minute < 8 * 60 + 40:
        return "05"
    if 15 * 60 + 40 <= minute < 16 * 60:
        return "06"
    if 16 * 60 <= minute <= 18 * 60:
        return "07"
    return "00"


def _is_us_daytime_order_session(market: str, now: datetime | None = None) -> bool:
    market_name = str(market or "").upper()
    if not any(token in market_name for token in ("US", "NASDAQ", "NASD", "NYSE", "AMEX", "OVERSEAS")):
        return False
    if os.getenv("KIS_FORCE_OVERSEAS_DAYTIME_ORDER", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        return False
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    local = current.astimezone(ZoneInfo("Asia/Seoul"))
    minute = local.hour * 60 + local.minute
    return local.weekday() < 5 and 9 * 60 <= minute <= 16 * 60 + 50


def _cash_by_currency_from_summary(summary_row: dict[str, Any], krw_cash: float) -> dict[str, float]:
    cash_by_currency: dict[str, float] = {"KRW": float(krw_cash or 0.0)}
    explicit = summary_row.get("cash_by_currency")
    if isinstance(explicit, dict):
        for currency, amount in explicit.items():
            code = str(currency or "").upper().strip()
            if code:
                cash_by_currency[code] = _to_float(amount)
    foreign = summary_row.get("foreign_cash_by_currency")
    if isinstance(foreign, dict):
        for currency, amount in foreign.items():
            code = str(currency or "").upper().strip()
            if code and code != "KRW":
                cash_by_currency[code] = _to_float(amount)
    aliases = {
        "USD": ("usd_cash", "usd_deposit", "usd_dnca_amt", "frcr_dnca_amt", "frcr_dncl_amt"),
        "JPY": ("jpy_cash", "jpy_deposit", "jpy_dnca_amt"),
        "EUR": ("eur_cash", "eur_deposit", "eur_dnca_amt"),
        "CNY": ("cny_cash", "cny_deposit", "cny_dnca_amt"),
        "HKD": ("hkd_cash", "hkd_deposit", "hkd_dnca_amt"),
    }
    for currency, keys in aliases.items():
        for key in keys:
            if key in summary_row:
                cash_by_currency[currency] = _to_float(summary_row.get(key))
                break
    return cash_by_currency


def _foreign_cash_by_currency_from_overseas_response(
    response: dict[str, Any],
    nation_code: str = "000",
) -> dict[str, float]:
    balances: dict[str, float] = {}
    for row in _response_rows(response):
        currency = _currency_from_row(row, nation_code)
        if not currency or currency == "KRW":
            continue
        amount = _foreign_cash_amount_from_row(row)
        if amount is None:
            continue
        balances[currency] = balances.get(currency, 0.0) + amount
    return balances


def _foreign_orderable_cash_by_currency_from_overseas_response(
    response: dict[str, Any],
    nation_code: str = "000",
) -> dict[str, float]:
    balances: dict[str, float] = {}
    for row in _response_rows(response):
        currency = _currency_from_row(row, nation_code)
        if not currency or currency == "KRW":
            continue
        amount = _foreign_orderable_cash_amount_from_row(row)
        if amount is None:
            continue
        balances[currency] = max(balances.get(currency, 0.0), amount)
    return balances


def _foreign_cash_krw_from_overseas_response(response: dict[str, Any]) -> float:
    best = 0.0
    for row in _response_rows(response):
        amount = _foreign_cash_amount_from_row(row)
        rate = _exchange_rate_from_row(row)
        if amount is not None and rate > 0:
            best = max(best, amount * rate)
    return best


def _total_assets_krw_from_overseas_response(response: dict[str, Any]) -> float:
    best = 0.0
    for row in _response_rows(response):
        for key in ("tot_asst_amt", "tot_asst_amt2", "tot_frcr_cblc_smtl"):
            if key in row:
                best = max(best, _to_float(row.get(key)))
    return best


def _response_rows(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if value and any(
            key in value
            for key in (
                "crcy_cd",
                "tr_crcy_cd",
                "frcr_dncl_amt",
                "frcr_dncl_amt_2",
                "frcr_drwg_psbl_amt_1",
                "nxdy_frcr_drwg_psbl_amt",
                "tot_asst_amt",
                "ovrs_tot_asst_amt",
                "frcr_evlu_tota",
                "tot_evlu_amt",
                "frcr_evlu_amt2",
                "tot_frcr_cblc_smtl",
                "wcrc_frcr_evlu_amt",
                "krw_evlu_amt",
                "evlu_amt_wcrc",
                "wcrc_tot_evlu_amt",
                "wcrc_tot_asst_amt",
                "bass_exrt",
                "aply_exrt",
                "frst_bltn_exrt",
                "ovrs_pdno",
                "ovrs_item_name",
                "ovrs_cblc_qty",
                "ovrs_stck_evlu_amt",
                "ord_psbl_amt",
                "ovrs_ord_psbl_amt",
                "max_ord_psbl_amt",
                "frcr_ord_psbl_amt1",
                "odno",
                "nccs_qty",
                "ft_ord_qty",
                "ft_ccld_qty",
                "ft_ord_unpr3",
            )
        ):
            rows.append(value)
        for item in value.values():
            rows.extend(_response_rows(item))
    elif isinstance(value, list):
        for item in value:
            rows.extend(_response_rows(item))
    return rows


def _currency_from_row(row: dict[str, Any], nation_code: str) -> str | None:
    for key in ("crcy_cd", "tr_crcy_cd", "ovrs_crcy_cd", "curr_cd", "currency", "bass_exrt_curr_cd"):
        value = str(row.get(key) or "").upper().strip()
        if value:
            return value
    if nation_code == "840":
        return "USD"
    return None


def _foreign_cash_amount_from_row(row: dict[str, Any]) -> float | None:
    cash_keys = (
        "frcr_dncl_amt",
        "frcr_dncl_amt_2",
        "frcr_dnca_amt",
        "dnca_frcr_amt",
        "ord_psbl_frcr_amt",
        "frcr_ord_psbl_amt",
        "frcr_buy_psbl_amt",
        "buy_psbl_frcr_amt",
        "withdrawable_frcr_amt",
        "frcr_drwg_psbl_amt_1",
        "nxdy_frcr_drwg_psbl_amt",
    )
    for key in cash_keys:
        if key in row:
            return _to_float(row.get(key))
    return None


def _foreign_orderable_cash_amount_from_row(row: dict[str, Any]) -> float | None:
    orderable_keys = (
        "frcr_drwg_psbl_amt_1",
        "nxdy_frcr_drwg_psbl_amt",
        "ord_psbl_frcr_amt",
        "frcr_ord_psbl_amt",
        "frcr_ord_psbl_amt1",
        "frcr_buy_psbl_amt",
        "buy_psbl_frcr_amt",
        "withdrawable_frcr_amt",
    )
    for key in orderable_keys:
        if key in row:
            return _to_float(row.get(key))
    return None


def _overseas_orderable_amount_from_response(response: dict[str, Any]) -> float:
    best = 0.0
    for row in _response_rows(response):
        for key in (
            "ord_psbl_amt",
            "ovrs_ord_psbl_amt",
            "max_ord_psbl_amt",
            "frcr_ord_psbl_amt1",
            "frcr_ord_psbl_amt",
            "buy_psbl_amt",
            "ord_psbl_frcr_amt",
            "frcr_buy_psbl_amt",
        ):
            if key in row:
                best = max(best, _to_float(row.get(key)))
    return best


def _exchange_rate_from_row(row: dict[str, Any]) -> float:
    for key in (
        "bass_exrt",
        "aply_exrt",
        "frst_bltn_exrt",
        "exrt",
        "exchange_rate",
        "usd_krw_rate",
    ):
        if key in row:
            rate = _to_float(row.get(key))
            if rate > 0:
                return rate
    return 0.0


def _default_token_cache_path(paper: bool) -> Path:
    mode = "paper" if paper else "live"
    return Path(os.getenv("KIS_TOKEN_CACHE_DIR", "config/secrets")) / f"kis_access_token.{mode}.json"


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def load_kis_env_file(path: str | Path | None = None, override: bool = False) -> bool:
    """Load local KIS secrets from an ignored env file without printing values."""
    global _KIS_ENV_FILE_LOADED
    secrets_path = Path(path) if path is not None else KIS_SECRETS_FILE
    if _KIS_ENV_FILE_LOADED and path is None and not override:
        return secrets_path.exists()
    if not secrets_path.exists():
        _KIS_ENV_FILE_LOADED = True
        return False
    for raw_line in secrets_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        key = name.strip()
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")
    _KIS_ENV_FILE_LOADED = True
    return True

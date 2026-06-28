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
from app.schemas.domain import AccountSnapshot, FinalOrder, Holding, OrderSide


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

    @property
    def order_status_tr_id(self) -> str:
        return "VTTC8001R" if self.paper else "TTTC8001R"

    @property
    def balance_tr_id(self) -> str:
        return "VTTC8434R" if self.paper else "TTTC8434R"


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
        self._token_cache_path = (
            Path(token_cache_path)
            if token_cache_path is not None
            else _default_token_cache_path(self.paper)
        )
        self._orders: dict[str, FinalOrder] = {}

    def place_limit_order(self, order: FinalOrder) -> MockKisOrderReceipt:
        self._ensure_enabled()
        if order.side not in {OrderSide.BUY, OrderSide.SELL}:
            raise ValueError(f"unsupported KIS order side: {order.side}")
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
        return MockKisOrderReceipt(
            order_id=order_id,
            accepted=True,
            status="ACCEPTED",
            message=str(response.get("msg1") or "KIS accepted the order."),
            order=order,
            submitted_at=datetime.now(timezone.utc),
        )

    def get_order_status(self, order_id: str) -> MockKisExecution:
        self._ensure_enabled()
        order = self._orders.get(order_id)
        params = self._order_status_params(order_id, order)
        response = self._get(
            "/uapi/domestic-stock/v1/trading/inquire-daily-ccld",
            tr_id=self.endpoints.order_status_tr_id,
            params=params,
        )
        self._ensure_success(response, "KIS order-status lookup failed")
        row = _first_response_row(response)
        return self._execution_from_status(order_id, row, order)

    def get_portfolio(self) -> MockKisPortfolio:
        self._ensure_enabled()
        response = self._get(
            "/uapi/domestic-stock/v1/trading/inquire-balance",
            tr_id=self.endpoints.balance_tr_id,
            params=self._balance_params(),
        )
        self._ensure_success(response, "KIS portfolio lookup failed")
        holdings = tuple(self._holding_from_balance(row) for row in response.get("output1") or ())
        summary = response.get("output2") or response.get("output3") or []
        summary_row = summary[0] if isinstance(summary, list) and summary else summary
        cash = _to_float(
            summary_row.get("dnca_tot_amt")
            or summary_row.get("prvs_rcdl_excc_amt")
            or summary_row.get("tot_evlu_amt")
        )
        account = AccountSnapshot(cash=cash, holdings=holdings)
        return MockKisPortfolio(
            account=account,
            market_prices={holding.ticker: holding.last_price for holding in holdings},
            updated_at=datetime.now(timezone.utc),
        )

    def issue_access_token(self, force_refresh: bool = False) -> str:
        self.credentials.validate()
        if not force_refresh:
            cached = self._load_cached_token()
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
        self._write_cached_token()
        return token

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
        cached = self._load_cached_token(now)
        if cached:
            return cached
        return self.issue_access_token()

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
            "ORD_DVSN": "00",
            "ORD_QTY": str(int(order.quantity)),
            "ORD_UNPR": str(int(round(order.limit_price))),
            "EXCG_ID_DVSN_CD": "KRX",
            "SLL_TYPE": "",
            "CNDT_PRIC": "",
        }

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

    def _holding_from_balance(self, row: dict[str, Any]) -> Holding:
        quantity = int(_to_float(row.get("hldg_qty") or row.get("ord_psbl_qty") or 0))
        average_price = _to_float(row.get("pchs_avg_pric") or 0)
        last_price = _to_float(row.get("prpr") or row.get("bfdy_cprs_icdc") or average_price)
        return Holding(
            ticker=str(row.get("pdno") or ""),
            market="KR",
            company_name=str(row.get("prdt_name") or row.get("pdno") or ""),
            sector="Unknown",
            quantity=quantity,
            average_price=average_price,
            last_price=last_price,
        )

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

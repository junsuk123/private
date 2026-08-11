from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from app.execution.kis_errors import LiveExecutionBlocked
from app.execution.kis_mock import MockKisExecution, MockKisOrderReceipt, MockKisPortfolio
from app.schemas.domain import AccountSnapshot, FinalOrder, Holding, MarketSnapshot, OrderSide, SourceMetadata


KIS_LIVE_BASE_URL = "https://openapi.koreainvestment.com:9443"
KIS_PAPER_BASE_URL = "https://openapivts.koreainvestment.com:29443"
KIS_SECRETS_FILE = Path("config/secrets/kis_api_keys.env")
KIS_TOKEN_CACHE_SKEW_SECONDS = 60
_KIS_ENV_FILE_LOADED = False
_KIS_GET_RATE_LOCK = threading.Lock()
_KIS_GET_NEXT_ALLOWED_AT = 0.0
_KIS_TOKEN_REFRESH_LOCK = threading.Lock()


class KisApiError(RuntimeError):
    def __init__(self, message: str, response: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.response = response or {}


def _is_expired_token_error(exc: KisApiError) -> bool:
    response = getattr(exc, "response", {}) or {}
    message = " ".join(
        str(value or "")
        for value in (
            exc,
            response.get("msg1"),
            response.get("message"),
            response.get("error_description"),
        )
    ).lower()
    return "token" in message and ("expired" in message or "만료" in message)


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
        app_key = os.getenv("KIS_APP_KEY", "")
        app_secret = os.getenv("KIS_APP_SECRET", "")
        account_no = os.getenv("KIS_ACCOUNT_NO", "")
        product_code = os.getenv("KIS_ACCOUNT_PRODUCT_CODE") or "01"
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
        return cls(base_url=(base_url or KIS_LIVE_BASE_URL), paper=False)

    def tr_id_for_order(self, side: OrderSide) -> str:
        if side == OrderSide.BUY:
            return "TTTC0012U"
        return "TTTC0011U"

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
        return tr_id

    def overseas_daytime_tr_id_for_order(self, side: OrderSide) -> str:
        if side == OrderSide.BUY:
            return "TTTS6036U"
        return "TTTS6037U"

    @property
    def order_revise_cancel_tr_id(self) -> str:
        return "TTTC0013U"

    def overseas_revise_cancel_tr_id(self, exchange_code: str) -> str:
        exchange = exchange_code.upper()
        if exchange in {"NASD", "NYSE", "AMEX"}:
            return "TTTT1004U"
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
        return tr_id

    @property
    def overseas_daytime_revise_cancel_tr_id(self) -> str:
        return "TTTS6038U"

    # --- 신용/대주 (credit borrow) ------------------------------------------- #
    # Separate TR ids from the cash path. Routing a short through ``order-cash``
    # would place an ordinary SELL, which for an account holding none of the stock
    # is a rejection at best; the dangerous case is the mirror — routing a
    # buy-to-cover as a cash BUY, which *succeeds* and leaves the account long the
    # stock while still owing the borrow.
    def credit_tr_id_for_order(self, side: OrderSide) -> str:
        """KIS domestic credit order: SELL / BUY.

        These ids are the ones published by KIS in ``order_credit.py``.  They are
        side ids, not credit-product ids; ``CRDT_TYPE`` carries the latter.
        """
        return "TTTC0051U" if side == OrderSide.SELL else "TTTC0052U"

    @property
    def credit_order_revise_cancel_tr_id(self) -> str:
        return "TTTC0083U"

    @property
    def credit_borrowable_tr_id(self) -> str:
        """대주 가능 종목 조회."""
        return "CTSC2702R"

    @property
    def credit_borrow_quantity_tr_id(self) -> str:
        """대주 가능 수량/이용료 조회."""
        return "TTTC8909R"

    @property
    def credit_balance_tr_id(self) -> str:
        """신용/대주 잔고 조회 (대출일 포함)."""
        return "CTRP6504R"

    @property
    def order_status_tr_id(self) -> str:
        return "TTTC8001R"

    @property
    def balance_tr_id(self) -> str:
        return "TTTC8434R"

    @property
    def account_balance_tr_id(self) -> str:
        return "CTRP6548R"

    @property
    def orderable_cash_tr_id(self) -> str:
        return "TTTC8908R"

    @property
    def overseas_present_balance_tr_id(self) -> str:
        return "CTRP6504R"

    @property
    def overseas_balance_tr_id(self) -> str:
        return "TTTS3012R"

    @property
    def overseas_orderable_cash_tr_id(self) -> str:
        return "TTTS3007R"

    @property
    def overseas_order_status_tr_id(self) -> str:
        return "TTTS3035R"


class KisDevelopersApiClient:
    """KIS Developers REST broker adapter for domestic cash stock orders.

    Tests can inject a fake KisTransport, while production uses urllib and real
    KIS live credentials.
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
        self.paper = False
        self.credentials = (
            KisCredentials.from_env(False)
            if app_key is None and app_secret is None and account_no is None
            else KisCredentials.from_values(app_key, app_secret, account_no, account_product_code)
        )
        self.endpoints = KisEndpointSet.for_mode(False, base_url or os.getenv("KIS_BASE_URL_REAL") or os.getenv("KIS_BASE_URL"))
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
        # order_id -> 원주문 route (endpoint family, venue, session, ORD_DVSN).
        #
        # 정정·취소는 **이 값**으로 라우팅한다. 이전에는 정정 시점의 시각으로
        # daytime/regular 를 다시 판정했고, 그래서 주간거래로 접수한 주문을 세션 경계
        # 이후에 정정하면 일반 order-rvsecncl 로 전송됐다 (원주문 불일치).
        self._order_routes: dict[str, dict[str, Any]] = {}
        # (ticker, exchange) -> (dtm_tr_psbl_yn as bool, monotonic timestamp)
        self._daytime_tradable_cache: dict[tuple[str, str], tuple[bool, float]] = {}

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
        self._record_order_route(
            order_id,
            route_family="DOMESTIC_CASH",
            endpoint="/uapi/domestic-stock/v1/trading/order-cash",
            exchange_id_code=body.get("EXCG_ID_DVSN_CD", "KRX"),
            order_division=body.get("ORD_DVSN", "00"),
            venue="NXT" if body.get("EXCG_ID_DVSN_CD") == "NXT" else "KRX",
        )
        return MockKisOrderReceipt(
            order_id=order_id,
            accepted=True,
            status="ACCEPTED",
            message=str(response.get("msg1") or "KIS accepted the order."),
            order=order,
            submitted_at=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------ #
    # 대주 (credit borrow / short) — read-only surface                     #
    # ------------------------------------------------------------------ #
    # VERIFICATION STATUS. Three guessed endpoints were disproven on 2026-08-02, but
    # KIS subsequently published the actual company inventory endpoint:
    #
    #   TTTC8909R  /trading/inquire-credit-psamount  -> "조회종목은 신용종목이
    #       아닙니다.(융자신규매수)" — the TR exists but reports 융자 (margin BUY)
    #       purchasing power, not 대주 (stock-loan) availability. Wrong question.
    #   CTSC0271R  /quotations/credit-by-company     -> "잘못된 TR 코드 입니다"
    #       — the TR id does not exist.
    #   CTRP6504R  /trading/inquire-credit-balance   -> HTTP 404 — the path does
    #       not exist.
    #
    #   CTSC2702R  /quotations/lendable-by-company -> official [domestic-stock-195]
    #       inventory query. It is company inventory, not a guarantee of the final
    #       account-sized order; account/risk limits remain separate gates.
    #
    # The balance read below reuses the SAME
    # ``inquire-balance`` / TTTC8434R call the portfolio path already makes every cycle
    # in production, so it introduces no new endpoint — 대주 lots are simply the rows
    # carrying credit metadata.
    def get_lendable_by_company(self, symbol: str) -> dict[str, Any]:
        """Return a normalized point-in-time KIS borrow inventory answer.

        ``CTSC2702R`` has no account parameters: it reports inventory KIS can make
        available, while the account and strategy gates independently reduce the
        eventual order size.  A successful but schema-incompatible response raises
        instead of fabricating a no-locate.
        """
        code = str(symbol or "").strip().upper()
        if not code:
            raise ValueError("borrow inventory lookup requires a symbol")
        response = self._get(
            "/uapi/domestic-stock/v1/quotations/lendable-by-company",
            tr_id=self.endpoints.credit_borrowable_tr_id,
            params={
                "EXCG_DVSN_CD": "00",
                "PDNO": code,
                "THCO_STLN_PSBL_YN": "Y",
                "INQR_DVSN_1": "0",
                "CTX_AREA_FK200": "",
                "CTX_AREA_NK100": "",
            },
        )
        self._ensure_success(response, "KIS lendable inventory lookup failed")

        rows: list[dict[str, Any]] = []
        for key in ("output1", "output2", "output"):
            value = response.get(key)
            if isinstance(value, dict):
                rows.append(dict(value))
            elif isinstance(value, list):
                rows.extend(dict(item) for item in value if isinstance(item, dict))

        matching = [
            row
            for row in rows
            if str(row.get("pdno") or row.get("PDNO") or "").strip().upper() == code
        ]
        # KIS splits this response: output1 may carry the overall Y/N flag while
        # output2 carries the per-symbol quantity. Preserve metadata rows without a
        # PDNO, plus the exact symbol row; never borrow fields from another symbol.
        metadata = [
            row for row in rows if not str(row.get("pdno") or row.get("PDNO") or "").strip()
        ]
        candidates = metadata + matching
        if not candidates:
            # An exact-symbol query with an empty successful result is an explicit
            # absence from KIS's lendable inventory, not a transport failure.
            return {
                "symbol": code,
                "available": False,
                "available_quantity": 0,
                "reject_reason": "KIS_BORROW_SYMBOL_NOT_LISTED",
                "raw": response,
            }

        possible: str | None = None
        quantity: int | None = None
        selected: dict[str, Any] = {}
        for row in candidates:
            selected.update(row)
            for key in ("psbl_yn", "thco_stln_psbl_yn", "PSBL_YN", "THCO_STLN_PSBL_YN"):
                if key in row and str(row.get(key) or "").strip():
                    possible = str(row.get(key)).strip().upper()
                    break
            # Requestable quantity is the executable inventory field identified by
            # KIS. trad_psbl_qty2 is retained as a documented fallback.
            for key in ("rqst_psbl_qty", "RQST_PSBL_QTY", "trad_psbl_qty2", "TRAD_PSBL_QTY2"):
                if key in row and str(row.get(key) or "").strip() != "":
                    quantity = _to_int(row.get(key))
                    break

        if not matching and quantity == 0 and str(response.get("msg_cd") or "") == "KIOK0560":
            # Actual KIS contract for an exact symbol absent from inventory:
            # rt_cd=0, msg_cd=KIOK0560 ("조회할 내용이 없습니다"), empty output1,
            # and an output2 aggregate whose quantities are all zero.
            return {
                "symbol": code,
                "available": False,
                "available_quantity": 0,
                "reject_reason": "KIS_BORROW_SYMBOL_NOT_LISTED",
                "raw": selected,
            }
        if possible not in {"Y", "N"}:
            raise KisApiError(
                f"KIS lendable response for {code} omitted PSBL_YN",
                response,
            )
        if possible == "Y" and quantity is None:
            raise KisApiError(
                f"KIS lendable response for {code} omitted requestable quantity",
                response,
            )
        resolved_quantity = max(0, int(quantity or 0))
        return {
            "symbol": code,
            "available": possible == "Y" and resolved_quantity > 0,
            "available_quantity": resolved_quantity,
            "reject_reason": (
                "" if possible == "Y" and resolved_quantity > 0 else "KIS_BORROW_INVENTORY_EMPTY"
            ),
            "raw": selected,
        }

    def get_borrow_balance(self) -> tuple[dict[str, Any], ...]:
        """Open 대주 positions as the BROKER sees them, one row per loan lot.

        Reads the already-verified ``inquire-balance`` response rather than a separate
        credit-balance endpoint — the 404 above showed that endpoint does not exist, and
        the balance response already carries ``loan_dt`` on credit rows.

        ``loan_date`` (대출일) is load-bearing: it identifies WHICH borrow lot a
        buy-to-cover repays. A row without one is returned WITH the flag set rather than
        dropped, because a position we cannot describe is exactly what has to reach the
        reconciliation logic and trigger a suspension.
        """
        lots: list[dict[str, Any]] = []
        for page in self._get_domestic_balance_pages():
            for row in page.get("output1") or []:
                if not isinstance(row, dict):
                    continue
                quantity = _to_int(row.get("hldg_qty") or row.get("cblc_qty"))
                if quantity <= 0:
                    continue
                loan_date = str(row.get("loan_dt") or "").strip()
                loan_amount = _to_float(row.get("loan_amt"))
                credit_type = str(row.get("crdt_type") or "").strip()
                # A plain cash holding has no loan date, no loan amount and no credit
                # type. Only rows with credit metadata can be a 대주 lot.
                if not loan_date and not loan_amount and not credit_type:
                    continue
                lots.append(
                    {
                        "symbol": str(row.get("pdno") or "").strip().upper(),
                        "quantity": quantity,
                        "average_price": _to_float(row.get("pchs_avg_pric")),
                        "loan_date": loan_date or None,
                        "loan_date_missing": not loan_date,
                        "loan_amount": loan_amount,
                        "credit_type": credit_type or None,
                        # Only 05 (대주) is a SHORT. A 융자 row (01) is a LEVERAGED
                        # LONG; counting it as short exposure would invert the
                        # net-exposure calculation.
                        "direction": "SHORT" if credit_type == "05" else "LONG",
                    }
                )
        return tuple(lots)

    def reconcile_credit_positions(
        self, internal_lots: tuple[dict[str, Any], ...] = ()
    ) -> dict[str, Any]:
        """Compare broker 대주 state against internal state. Broker wins.

        Returns a verdict the promotion controller reads as
        :class:`~app.trading.short_strategy_promotion.RuntimeHealth` flags. The three
        disagreements it can find, and why each is fail-closed:

        * **orphan** — the broker holds a short we have no record of. We cannot manage
          an exit for a position whose thesis we do not know, so new entries stop and
          the position goes to close-only management.
        * **phantom** — we believe we hold a short the broker does not. Our exit logic
          would send a buy-to-cover for stock we do not owe, which OPENS a long.
        * **missing loan date** — the lot exists but cannot be repaid through the
          매수상환 contract, which requires 대출일.

        A transport failure surfaces as ``broker_state_restored=False`` rather than as
        "no discrepancies": an unanswered reconciliation is not a clean one.
        """
        try:
            broker_lots = self.get_borrow_balance()
            restored = True
            error = ""
        except (KisApiError, RuntimeError, ValueError) as exc:
            broker_lots = ()
            restored = False
            error = str(exc)

        broker_shorts = {
            (lot["symbol"], lot.get("loan_date") or ""): lot
            for lot in broker_lots
            if lot.get("direction") == "SHORT"
        }
        internal_shorts = {
            (
                str(lot.get("symbol") or "").upper(),
                str(lot.get("loan_date") or ""),
            ): lot
            for lot in internal_lots
            if str(lot.get("direction") or "LONG").upper() == "SHORT"
        }
        orphans = [key for key in broker_shorts if key not in internal_shorts]
        phantoms = [key for key in internal_shorts if key not in broker_shorts]
        loan_date_missing = any(
            lot.get("loan_date_missing") for lot in broker_shorts.values()
        )
        quantity_mismatch = [
            key
            for key in broker_shorts
            if key in internal_shorts
            and int(internal_shorts[key].get("quantity") or 0)
            != int(broker_shorts[key].get("quantity") or 0)
        ]
        return {
            "broker_state_restored": restored,
            "error": error,
            "broker_short_lot_count": len(broker_shorts),
            "internal_short_lot_count": len(internal_shorts),
            "orphan_lots": [{"symbol": s, "loan_date": d} for s, d in orphans],
            "phantom_lots": [{"symbol": s, "loan_date": d} for s, d in phantoms],
            "quantity_mismatch_lots": [
                {"symbol": s, "loan_date": d} for s, d in quantity_mismatch
            ],
            "loan_date_missing": loan_date_missing,
            # Any disagreement blocks new short entries. Existing shorts stay
            # manageable so an orphan can still be closed.
            "position_direction_mismatch": bool(
                orphans or phantoms or quantity_mismatch
            ),
            "new_short_entries_blocked": bool(
                not restored
                or orphans
                or phantoms
                or quantity_mismatch
                or loan_date_missing
            ),
            "close_only_mode": bool(orphans),
            "broker_lots": list(broker_lots),
        }

    def place_credit_borrow_open_order(self, order: FinalOrder) -> MockKisOrderReceipt:
        """대주매도 신규 — opens a SHORT. Real money.

        Refuses anything whose contract does not say exactly that. The checks look
        redundant against the caller's own gates, and they are deliberately
        duplicated here: this is the last function before a live short exists, and
        every one of these mismatches would otherwise place a *different* order than
        the one intended.
        """
        self._ensure_enabled()
        _require_credit_contract(order, direction="SHORT", effect="OPEN", side=OrderSide.SELL)
        return self._place_credit_order(order, repay=False)

    def place_credit_borrow_close_order(self, order: FinalOrder) -> MockKisOrderReceipt:
        """대주 상환 매수 (buy-to-cover) — closes a SHORT. Real money.

        ``loan_date`` is mandatory and there is no fallback. Submitting a cover
        without it either fails, or — depending on the broker's defaulting — repays a
        DIFFERENT loan lot than intended, leaving one lot doubled and another still
        open. Both are worse than not sending the order.
        """
        self._ensure_enabled()
        _require_credit_contract(order, direction="SHORT", effect="CLOSE", side=OrderSide.BUY)
        if not str(order.loan_date or "").strip():
            raise ValueError(
                "credit-borrow close order requires loan_date (대출일); "
                "refusing to guess which loan lot to repay"
            )
        return self._place_credit_order(order, repay=True)

    def cancel_credit_order(self, order_id: str, order: FinalOrder) -> MockKisOrderReceipt:
        self._ensure_enabled()
        body = self._revise_cancel_body(order_id, order, revise=False)
        body["CRDT_TYPE"] = order.credit_type or (
            "26" if order.side == OrderSide.BUY else "22"
        )
        if order.loan_date:
            body["LOAN_DT"] = str(order.loan_date)
        response = self._post(
            "/uapi/domestic-stock/v1/trading/order-resv-rvsecncl",
            tr_id=self.endpoints.credit_order_revise_cancel_tr_id,
            body=body,
            include_hashkey=True,
        )
        self._ensure_success(response, "KIS credit cancel rejected")
        return self._receipt_from_revise_cancel_response(response, order_id, order)

    def _place_credit_order(self, order: FinalOrder, *, repay: bool) -> MockKisOrderReceipt:
        body = self._credit_order_body(order, repay=repay)
        response = self._post(
            "/uapi/domestic-stock/v1/trading/order-credit",
            tr_id=self.endpoints.credit_tr_id_for_order(order.side),
            body=body,
            include_hashkey=True,
        )
        self._ensure_success(response, "KIS credit order rejected")
        output = response.get("output") or {}
        # Cross-check what the broker says it booked against what we asked for. A
        # credit classification we did not request means the order that exists is not
        # the order we designed, and continuing to manage it as if it were is how an
        # unhedged, mis-typed position accumulates.
        booked_type = str(output.get("crdt_type") or output.get("CRDT_TYPE") or "").strip()
        expected_type = body.get("CRDT_TYPE", "")
        if booked_type and expected_type and booked_type != expected_type:
            raise KisApiError(
                f"KIS booked credit type {booked_type!r} but {expected_type!r} was requested "
                f"for {order.ticker}; refusing to treat this as the intended position",
                response,
            )
        order_id = str(output.get("ODNO") or output.get("odno") or "")
        if not order_id:
            order_id = f"KIS-CR-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        self._orders[order_id] = order
        self._order_org_numbers[order_id] = str(
            output.get("KRX_FWDG_ORD_ORGNO") or output.get("krx_fwdg_ord_orgno") or ""
        )
        self._record_order_route(
            order_id,
            route_family="DOMESTIC_CASH",
            endpoint="/uapi/domestic-stock/v1/trading/order-credit",
            exchange_id_code="KRX",
            order_division=_domestic_order_division_code(exchange_id_code="KRX"),
        )
        return MockKisOrderReceipt(
            order_id=order_id,
            accepted=True,
            status="ACCEPTED",
            message=str(response.get("msg1") or "KIS accepted the credit order."),
            order=order,
            submitted_at=datetime.now(timezone.utc),
        )

    def _credit_order_body(self, order: FinalOrder, *, repay: bool) -> dict[str, str]:
        # Official KIS codes: 22=유통대주신규(SELL), 26=유통대주상환(BUY).
        default_credit_type = "26" if repay else "22"
        body = {
            "CANO": self.credentials.account_no,
            "ACNT_PRDT_CD": self.credentials.account_product_code,
            "PDNO": order.ticker,
            "CRDT_TYPE": order.credit_type or default_credit_type,
            "ORD_DVSN": _domestic_order_division_code(),
            "ORD_QTY": str(int(order.quantity)),
            "ORD_UNPR": str(int(round(order.limit_price))),
            "EXCG_ID_DVSN_CD": "KRX",
            "LOAN_DT": "",
            "RVSE_CNCL_DVSN_CD": "",
            "CNDT_PRIC": "",
        }
        if repay:
            # Identifies WHICH loan lot this cover repays. Validated non-empty by the
            # caller; asserted again here because an empty LOAN_DT on a repayment is
            # silently accepted by the API and applied to an arbitrary lot.
            loan_date = str(order.loan_date or "").strip()
            if not loan_date:
                raise ValueError("credit repayment body requires LOAN_DT")
            body["LOAN_DT"] = loan_date
        return body

    def overseas_product_info(self, ticker: str, exchange_code: str) -> dict[str, Any]:
        """해외주식 상품기본정보 (CTPF1702R)."""
        product_type = US_PRODUCT_TYPE_CODES.get(str(exchange_code or "").upper())
        if not product_type:
            raise ValueError(f"no KIS product type code for exchange {exchange_code}")
        response = self._get(
            "/uapi/overseas-price/v1/quotations/search-info",
            tr_id="CTPF1702R",
            params={"PRDT_TYPE_CD": product_type, "PDNO": str(ticker or "").upper().strip()},
        )
        self._ensure_success(response, "KIS overseas product info lookup failed")
        return dict(response.get("output") or {})

    def is_us_daytime_tradable(self, ticker: str, exchange_code: str) -> bool | None:
        """``dtm_tr_psbl_yn`` for a US symbol, cached; ``None`` when unknown.

        KIS only supports a subset of US names for 주간거래, and submitting one
        of the others is rejected by the broker. ``None`` (lookup failed) is not
        treated as "not tradable" — the caller decides, so a reference-data
        outage cannot silently stop all daytime trading.
        """
        symbol = str(ticker or "").upper().strip()
        exchange = str(exchange_code or "").upper().strip()
        if not symbol or exchange not in US_PRODUCT_TYPE_CODES:
            return None
        key = (symbol, exchange)
        cached = self._daytime_tradable_cache.get(key)
        if cached is not None:
            value, cached_at = cached
            if time.monotonic() - cached_at < _daytime_tradable_cache_ttl_seconds():
                return value
        try:
            output = self.overseas_product_info(symbol, exchange)
        except Exception:  # noqa: BLE001 - reference data must never break the order path.
            return None
        raw = str(output.get("dtm_tr_psbl_yn") or "").strip().upper()
        if raw not in {"Y", "N"}:
            return None
        allowed = raw == "Y"
        self._daytime_tradable_cache[key] = (allowed, time.monotonic())
        return allowed

    def _place_overseas_limit_order(self, order: FinalOrder) -> MockKisOrderReceipt:
        exchange_code = _overseas_exchange_code(order.market)
        body = self._overseas_order_body(order, exchange_code)
        path = "/uapi/overseas-stock/v1/trading/order"
        tr_id = self.endpoints.overseas_tr_id_for_order(exchange_code, order.side)
        daytime = _is_us_daytime_order_session(order.market)
        if daytime:
            # KIS supports only a subset of US names for 주간거래; the rest are
            # rejected at the broker. Checking here turns a broker rejection
            # into a local, explainable block.
            if _enforce_daytime_tradable() and self.is_us_daytime_tradable(
                order.ticker, exchange_code
            ) is False:
                raise LiveExecutionBlocked(
                    (f"US_DAYTIME_TRADING_NOT_SUPPORTED:{order.ticker}",)
                )
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
        self._record_order_route(
            order_id,
            route_family=(
                "OVERSEAS_DAYTIME" if daytime else "OVERSEAS_REGULAR"
            ),
            endpoint=path,
            overseas_exchange_code=exchange_code,
        )
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
        path, tr_id = self._overseas_revise_cancel_route(
            order_id, exchange_code, replacement
        )
        response = self._post(path, tr_id=tr_id, body=body, include_hashkey=True)
        self._ensure_success(response, "KIS overseas order revise rejected")
        return self._receipt_from_revise_cancel_response(response, replacement, fallback_order_id=order_id)

    def _cancel_overseas_order(self, order_id: str, order: FinalOrder) -> MockKisOrderReceipt:
        exchange_code = _overseas_exchange_code(order.market)
        body = self._overseas_revise_cancel_body(order_id, order, exchange_code, revise=False)
        path, tr_id = self._overseas_revise_cancel_route(order_id, exchange_code, order)
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
        if not rows:
            return MockKisExecution(
                order_id=order_id,
                ticker=order.ticker,
                side=order.side,
                quantity=0,
                price=0.0,
                executed_value=0.0,
                status="OPEN",
                message="KIS overseas order status not found for this order id.",
                executed_at=datetime.now(timezone.utc),
            )
        row = rows[0]
        return self._overseas_execution_from_status(order_id, row, order)

    def get_portfolio(self) -> MockKisPortfolio:
        self._ensure_enabled()
        domestic_error: KisApiError | None = None
        holdings: tuple[Holding, ...] = ()
        cash = 0.0
        domestic_total_assets_krw = 0.0
        account_asset_summary: dict[str, float] = {}
        cash_by_currency: dict[str, float] = {"KRW": 0.0}
        orderable_cash_by_currency: dict[str, float] = {"KRW": 0.0}
        domestic_orderable_cash = 0.0
        try:
            responses = self._get_domestic_balance_pages()
            response = responses[0] if responses else {}
            rows: list[dict[str, Any]] = []
            for page in responses:
                rows.extend(row for row in (page.get("output1") or ()) if isinstance(row, dict))
            holdings = tuple(
                holding
                for row in rows
                if (holding := self._holding_from_balance(row)) is not None
            )
            summary = response.get("output2") or response.get("output3") or []
            summary_row = summary[0] if isinstance(summary, list) and summary else summary
            cash = _domestic_cash_from_balance_summary(summary_row, holdings)
            cash_by_currency = _cash_by_currency_from_summary(summary_row, cash)
            domestic_total_assets_krw = _domestic_total_assets_from_balance_summary(summary_row)
            try:
                account_asset_summary = self._get_account_asset_balance()
            except Exception:
                account_asset_summary = {}
        except KisApiError as exc:
            domestic_error = exc
        # Buying power is served by a separate KIS endpoint and must remain
        # available even when the domestic balance inquiry is temporarily
        # rejected. Keeping this call inside the balance try-block silently
        # converted a valid nrcvb_buy_amt into KRW=0 and starved affordability.
        try:
            domestic_orderable_cash = self._get_domestic_orderable_cash()
        except KisApiError:
            domestic_orderable_cash = 0.0
        orderable_cash_by_currency = dict(cash_by_currency)
        orderable_cash_by_currency["KRW"] = (
            domestic_orderable_cash
            if domestic_orderable_cash > 0
            else max(0.0, float(cash_by_currency.get("KRW", cash) or 0.0))
        )
        overseas_holdings: tuple[Holding, ...] = ()
        try:
            overseas_holdings = self._get_overseas_holdings()
        except Exception:
            overseas_holdings = ()
        try:
            foreign_cash_by_currency, foreign_cash_krw, total_assets_krw, foreign_fx_by_currency = self._get_overseas_cash_balance()
        except KisApiError:
            if domestic_error is not None:
                raise domestic_error
            raise
        try:
            foreign_orderable = self._get_overseas_orderable_cash_by_currency()
        except Exception:
            foreign_orderable = {}
        orderable_cash_by_currency.update(
            {currency: amount for currency, amount in foreign_orderable.items() if amount > 0}
        )
        # Foreign buying power is not the settled foreign-currency balance.
        # Integrated-margin accounts can expose KRW-backed USD buying power here;
        # copying it into cash_by_currency double-counts the same capital.
        # Display the true foreign (USD etc.) cash valued from currency balances × FX.
        # KIS's foreign_cash_krw re-includes the domestic KRW deposit (통합증거금), so it
        # must NOT be shown as "외화" — use the per-currency computation and only fall
        # back to the broker figure when no currency breakdown is available.
        _ccy_foreign_krw = _foreign_cash_krw_from_currency_balances(
            foreign_cash_by_currency,
            foreign_fx_by_currency,
        )
        # Some KIS integrated-margin responses put the domestic KRW deposit in a
        # field named like a foreign-cash total.  That summary is not evidence of
        # actual foreign cash by itself: require an explicit non-KRW currency
        # balance before using it as the display fallback.  Otherwise a KRW-only
        # account is rendered as KRW cash + the same amount again as foreign cash.
        has_explicit_foreign_cash = any(
            str(currency or "").upper() != "KRW" and _to_float(amount) > 0
            for currency, amount in foreign_cash_by_currency.items()
        )
        display_foreign_cash_krw = (
            _ccy_foreign_krw
            if _ccy_foreign_krw > 0
            else foreign_cash_krw if has_explicit_foreign_cash else 0.0
        )
        if domestic_error is not None and not foreign_cash_by_currency and foreign_cash_krw <= 0:
            raise domestic_error
        cash_by_currency.update(foreign_cash_by_currency)
        all_holdings = holdings + overseas_holdings
        domestic_position_value = sum(max(0.0, holding.market_value) for holding in holdings)
        overseas_position_value_krw = _overseas_holdings_value_krw(
            overseas_holdings,
            foreign_cash_by_currency,
            foreign_cash_krw,
            foreign_fx_by_currency,
        )
        # Pure overseas cash (USD etc.) valued directly from currency balances × FX.
        # Do NOT use foreign_cash_krw here: KIS's overseas present-balance re-includes the
        # domestic KRW deposit (통합증거금 cross-view), so adding it to the domestic total
        # double-counts the KRW cash (the old branch logic tried to dedup this and drifted,
        # over-reporting the account by ~그 KRW deposit). Total assets is therefore the
        # KIS domestic total-assets (tot_evlu_amt: D+2 settlement deposit + domestic stock)
        # + overseas stock + overseas cash — each bucket counted exactly once.
        usd_cash_krw = 0.0
        for _ccy, _amt in (cash_by_currency or {}).items():
            if str(_ccy).upper() == "KRW":
                continue
            _rate = float((foreign_fx_by_currency or {}).get(_ccy) or 0.0)
            if _rate > 0:
                usd_cash_krw += max(0.0, float(_amt or 0.0)) * _rate
        if usd_cash_krw <= 0.0:
            # No per-currency FX available: fall back to the broker's foreign-cash figure
            # minus the domestic settled deposit it double-counts.
            usd_cash_krw = max(0.0, foreign_cash_krw - cash)
        if total_assets_krw > 0 and total_assets_krw >= cash:
            # KIS overseas present-balance output3.tot_asst_amt is the broker's own
            # integrated 총자산 (settled KRW deposit + overseas stock + usable foreign
            # cash) — exactly what the KIS app shows. Prefer it verbatim so our total
            # matches the app. It is built from overseas-endpoint fields only, so it
            # EXCLUDES domestic equity holdings; add those back (counted once).
            #
            # Guard `>= cash`: only trust it as the *integrated* total when it actually
            # covers the domestic KRW deposit (the real integrated view carries the KRW
            # deposit — tot_dncl_amt — inside tot_asst_amt). Some responses report
            # tot_asst_amt as a foreign-only figure that omits the KRW deposit; those must
            # fall through to the component sum below, or the KRW deposit gets dropped.
            #
            # Why this beats tot_evlu_amt: while overseas buys are still settling (T+2)
            # the domestic D+2 deposit (prvs_rcdl_excc_amt, which feeds tot_evlu_amt) is
            # already reduced by the pending buy, yet we also hold the purchased overseas
            # stock — so tot_evlu_amt + overseas stock UNDERcounts by the in-flight amount
            # (e.g. settled 159,638 vs D+2 73,992 → ~85k missing). tot_asst_amt uses the
            # settled KRW deposit, matching the app through settlement.
            total_equity_krw = total_assets_krw + domestic_position_value
            cash_equivalent_krw = max(0.0, total_equity_krw - domestic_position_value - overseas_position_value_krw)
        elif domestic_total_assets_krw > 0:
            total_equity_krw = domestic_total_assets_krw + overseas_position_value_krw + usd_cash_krw
            cash_equivalent_krw = max(0.0, total_equity_krw - domestic_position_value - overseas_position_value_krw)
        else:
            cash_equivalent_krw = cash + usd_cash_krw
            total_equity_krw = cash_equivalent_krw + domestic_position_value + overseas_position_value_krw
        broker_total_assets_krw = _to_float(account_asset_summary.get("total_assets_krw"))
        if broker_total_assets_krw > 0:
            broker_cash_krw = _to_float(account_asset_summary.get("cash_krw"))
            broker_evaluation_krw = _to_float(account_asset_summary.get("evaluation_amount_krw"))
            if broker_evaluation_krw > 0:
                broker_cash_krw = max(0.0, broker_total_assets_krw - broker_evaluation_krw)
            total_equity_krw = broker_total_assets_krw
            cash_equivalent_krw = (
                broker_cash_krw
                if broker_cash_krw > 0
                else max(0.0, total_equity_krw - domestic_position_value - overseas_position_value_krw)
            )
            cash = max(0.0, cash_equivalent_krw - display_foreign_cash_krw)
            cash_by_currency["KRW"] = cash
        account = AccountSnapshot(
            cash=cash,
            holdings=all_holdings,
            base_currency="KRW",
            cash_by_currency=cash_by_currency,
            orderable_cash_by_currency=orderable_cash_by_currency,
            fx_rate_by_currency=foreign_fx_by_currency,
            cash_equivalent_krw=cash_equivalent_krw,
            foreign_cash_krw=display_foreign_cash_krw,
            total_equity_krw=total_equity_krw,
        )
        return MockKisPortfolio(
            account=account,
            market_prices={holding.ticker: holding.last_price for holding in holdings},
            updated_at=datetime.now(timezone.utc),
        )

    def _get_account_asset_balance(self) -> dict[str, float]:
        response = self._get(
            "/uapi/domestic-stock/v1/trading/inquire-account-balance",
            tr_id=self.endpoints.account_balance_tr_id,
            params=self._account_balance_params(),
        )
        self._ensure_success(response, "KIS account asset-balance lookup failed")
        return _account_asset_summary_from_response(response)

    def _get_domestic_balance_pages(self) -> list[dict[str, Any]]:
        pages: list[dict[str, Any]] = []
        params = self._balance_params()
        tr_cont: str | None = None
        seen_contexts: set[tuple[str, str]] = set()
        for _ in range(10):
            headers = self._headers(self.endpoints.balance_tr_id)
            if tr_cont:
                headers["tr_cont"] = tr_cont
            response = self.transport.request(
                "GET",
                self._url("/uapi/domestic-stock/v1/trading/inquire-balance"),
                headers=headers,
                params=params,
                timeout=self.timeout,
            )
            self._ensure_success(response, "KIS portfolio lookup failed")
            pages.append(response)
            next_fk = str(response.get("ctx_area_fk100") or "").strip()
            next_nk = str(response.get("ctx_area_nk100") or "").strip()
            if not next_fk or not next_nk:
                break
            context = (next_fk, next_nk)
            if context in seen_contexts:
                break
            seen_contexts.add(context)
            params = self._balance_params()
            params["CTX_AREA_FK100"] = next_fk
            params["CTX_AREA_NK100"] = next_nk
            tr_cont = "N"
        return pages

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

    def get_domestic_investor_flow(self, ticker: str) -> tuple[dict[str, Any], ...]:
        """Daily net buying by investor type (개인/외국인/기관) for a KRX symbol.

        ``inquire-investor`` (FHKST01010900) returns roughly the last 30 business
        days in one call, newest first, which is why a single request per symbol is
        enough to backfill a training window rather than needing months of
        collection first.

        Quantities are shares (``*_ntby_qty``) and values are in units of one
        million KRW (``*_ntby_tr_pbmn``); both are signed, negative meaning net
        selling. Returned as parsed rows in ascending date order — read-only, and
        deliberately not cached, because the caller persists it.
        """
        symbol = str(ticker or "").strip()
        if not (symbol.isdigit() and len(symbol) == 6):
            return ()
        response = self._get(
            "/uapi/domestic-stock/v1/quotations/inquire-investor",
            tr_id="FHKST01010900",
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol},
        )
        self._ensure_success(response, "KIS investor-flow lookup failed")
        raw = response.get("output") or response.get("output1") or []
        if isinstance(raw, dict):
            raw = [raw]
        rows: list[dict[str, Any]] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            business_date = str(entry.get("stck_bsop_date") or "").strip()
            if len(business_date) != 8 or not business_date.isdigit():
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "business_date": business_date,
                    "close_price": _to_float(entry.get("stck_clpr")),
                    "retail_net_buy_qty": _to_float(entry.get("prsn_ntby_qty")),
                    "foreign_net_buy_qty": _to_float(entry.get("frgn_ntby_qty")),
                    "institution_net_buy_qty": _to_float(entry.get("orgn_ntby_qty")),
                    # Value fields are the ones worth comparing across symbols; a
                    # share count means nothing next to a 1,700,000원 name.
                    "retail_net_buy_value": _to_float(entry.get("prsn_ntby_tr_pbmn")),
                    "foreign_net_buy_value": _to_float(entry.get("frgn_ntby_tr_pbmn")),
                    "institution_net_buy_value": _to_float(entry.get("orgn_ntby_tr_pbmn")),
                }
            )
        rows.sort(key=lambda row: row["business_date"])
        return tuple(rows)

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

    def inquire_domestic_period_profit(self, start_date: Any, end_date: Any) -> dict[str, Any]:
        """Raw KIS domestic period trade-profit inquiry (기간별매매손익현황조회).

        TR TTTC8715R returns realized (settled) trade profit/loss over
        [start_date, end_date] for domestic stocks.
        """
        params = {
            "CANO": self.credentials.account_no,
            "ACNT_PRDT_CD": self.credentials.account_product_code,
            "SORT_DVSN": "00",
            "PDNO": "",
            "INQR_STRT_DT": _kis_yyyymmdd(start_date),
            "INQR_END_DT": _kis_yyyymmdd(end_date),
            "CBLC_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        return self._get(
            "/uapi/domestic-stock/v1/trading/inquire-period-trade-profit",
            tr_id="TTTC8715R",
            params=params,
        )

    def get_domestic_realized_pnl(self, start_date: Any, end_date: Any) -> float:
        """Total realized (settled) domestic P&L in KRW over the given dates.

        Reads the KIS period trade-profit summary. Field names are parsed
        defensively (KIS returns the total realized P&L under one of a few
        documented keys); if the summary is absent it falls back to summing the
        per-trade rows. Raises KisApiError on an unsuccessful response.
        """
        response = self.inquire_domestic_period_profit(start_date, end_date)
        self._ensure_success(response, "KIS period trade-profit lookup failed")
        summary = response.get("output2")
        summary_row: dict[str, Any] = {}
        if isinstance(summary, list) and summary:
            summary_row = summary[0] if isinstance(summary[0], dict) else {}
        elif isinstance(summary, dict):
            summary_row = summary
        for key in ("tot_rlzt_pfls", "rlzt_pfls_smtl_amt", "rlzt_pfls", "rlzt_pfls_amt"):
            value = summary_row.get(key)
            if value is not None and str(value).strip() not in ("", "None"):
                return _to_float(value)
        # Fallback: sum per-trade realized P&L rows (output1).
        total = 0.0
        found = False
        for row in response.get("output1") or ():
            if not isinstance(row, dict):
                continue
            for key in ("rlzt_pfls", "trad_pfls", "pfls_amt", "evlu_pfls_amt"):
                if key in row and str(row.get(key)).strip() not in ("", "None"):
                    total += _to_float(row.get(key))
                    found = True
                    break
        return total if found else 0.0

    def inquire_overseas_period_profit(
        self,
        start_date: Any,
        end_date: Any,
        *,
        currency_division: str = "02",
    ) -> dict[str, Any]:
        """Raw KIS overseas period-profit inquiry (해외주식 기간손익).

        TR TTTS3039R returns realized (settled) trade profit/loss over
        [start_date, end_date] for overseas stocks across all exchanges/
        currencies (blank OVRS_EXCG_CD / CRCY_CD). ``WCRC_FRCR_DVSN_CD`` "01"
        returns amounts already converted to KRW (원화), "02" in the foreign
        currency — we default to "01" so no manual FX conversion is needed.
        """
        params = {
            "CANO": self.credentials.account_no,
            "ACNT_PRDT_CD": self.credentials.account_product_code,
            "OVRS_EXCG_CD": "",
            "NATN_CD": "",
            "CRCY_CD": "",
            "PDNO": "",
            "INQR_STRT_DT": _kis_yyyymmdd(start_date),
            "INQR_END_DT": _kis_yyyymmdd(end_date),
            "WCRC_FRCR_DVSN_CD": currency_division,
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        }
        return self._get(
            "/uapi/overseas-stock/v1/trading/inquire-period-profit",
            tr_id="TTTS3039R",
            params=params,
        )

    def get_overseas_realized_pnl(self, start_date: Any, end_date: Any) -> float:
        """Total realized (settled) overseas P&L in KRW over the given dates.

        Mirrors :meth:`get_domestic_realized_pnl`: reads the period-profit
        summary in KRW (WCRC_FRCR_DVSN_CD "01"), parsing field names
        defensively, and falls back to summing per-trade rows. Returns 0.0 when
        the account has no overseas activity. Raises KisApiError on a hard API
        failure (callers wrap this so it never breaks the domestic figure).
        """
        response = self.inquire_overseas_period_profit(start_date, end_date)
        self._ensure_success(response, "KIS overseas period-profit lookup failed")
        summary = response.get("output2")
        summary_row: dict[str, Any] = {}
        if isinstance(summary, list) and summary:
            summary_row = summary[0] if isinstance(summary[0], dict) else {}
        elif isinstance(summary, dict):
            summary_row = summary
        for key in (
            "ovrs_rlzt_pfls_tot_amt",
            "ovrs_rlzt_pfls_amt",
            "tot_rlzt_pfls",
            "rlzt_pfls_smtl_amt",
            "rlzt_pfls_amt",
            "rlzt_pfls",
        ):
            value = summary_row.get(key)
            if value is not None and str(value).strip() not in ("", "None"):
                return _to_float(value)
        # Fallback: sum per-trade realized P&L rows (output1).
        total = 0.0
        found = False
        for row in response.get("output1") or ():
            if not isinstance(row, dict):
                continue
            for key in ("ovrs_rlzt_pfls_amt", "rlzt_pfls", "trad_pfls", "pfls_amt", "evlu_pfls_amt"):
                if key in row and str(row.get(key)).strip() not in ("", "None"):
                    total += _to_float(row.get(key))
                    found = True
                    break
        return total if found else 0.0

    def get_overseas_settlement_summary(
        self,
        start_date: Any,
        end_date: Any,
    ) -> dict[str, float]:
        """Return settled overseas P&L and KIS 제비용 in KRW.

        Live KIS responses use WCRC_FRCR_DVSN_CD=02 for won-converted
        amounts. ``smtl_fee1`` is explicit broker/exchange/tax expense; it
        does not include spread or slippage.
        """
        response = self.inquire_overseas_period_profit(
            start_date,
            end_date,
            currency_division="02",
        )
        self._ensure_success(response, "KIS overseas period-profit lookup failed")
        summary = response.get("output2")
        row = (
            summary[0]
            if isinstance(summary, list)
            and summary
            and isinstance(summary[0], dict)
            else summary if isinstance(summary, dict) else {}
        )
        return {
            "sell_amount_krw": _to_float(row.get("stck_sll_amt_smtl")),
            "purchase_amount_krw": _to_float(row.get("stck_buy_amt_smtl")),
            "gross_trading_difference_krw": _to_float(
                row.get("excc_dfrm_amt")
            ),
            "broker_expenses_krw": _to_float(row.get("smtl_fee1")),
            "realized_pnl_krw": _first_float(
                row,
                "ovrs_rlzt_pfls_tot_amt",
                "ovrs_rlzt_pfls_amt",
                "rlzt_pfls_amt",
            ),
        }

    def _get(self, path: str, tr_id: str, params: dict[str, Any]) -> dict[str, Any]:
        if isinstance(self.transport, UrllibKisTransport):
            _throttle_kis_get()
        try:
            return self.transport.request(
                "GET",
                self._url(path),
                headers=self._headers(tr_id),
                params=params,
                timeout=self.timeout,
            )
        except KisApiError as exc:
            if not _is_expired_token_error(exc):
                raise
            self._refresh_rejected_access_token()
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
        try:
            return self.transport.request(
                "POST",
                self._url(path),
                headers=headers,
                body=body,
                timeout=self.timeout,
            )
        except KisApiError as exc:
            if not _is_expired_token_error(exc):
                raise
            self._refresh_rejected_access_token()
            retry_headers = self._headers(tr_id)
            if include_hashkey:
                retry_headers["hashkey"] = self._hashkey(body)
            return self.transport.request(
                "POST",
                self._url(path),
                headers=retry_headers,
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

    def _refresh_rejected_access_token(self) -> str:
        """Replace a token KIS rejects before its locally cached expiry time."""
        rejected = self._access_token
        with _KIS_TOKEN_REFRESH_LOCK:
            self._access_token = None
            self._token_expires_at = None
            # Another client/thread may already have replaced the shared token.
            cached = self._load_cached_token()
            if cached and cached != rejected:
                return cached
            self._access_token = None
            self._token_expires_at = None
            return self.issue_access_token(force_refresh=True)

    def _load_env_token(self, now: datetime | None = None) -> str | None:
        mode_prefix = "KIS_LIVE_"
        token = (
            os.getenv(f"{mode_prefix}ACCESS_TOKEN")
            or os.getenv("KIS_ACCESS_TOKEN")
            or ""
        ).strip()
        if not token:
            return None
        expires_at = _parse_datetime(
            os.getenv(f"{mode_prefix}ACCESS_TOKEN_EXPIRES_AT")
            or os.getenv("KIS_ACCESS_TOKEN_EXPIRES_AT")
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
        credential_fingerprint = str(payload.get("credential_fingerprint") or "")
        if (
            not token
            or expires_at is None
            or mode != "live"
            or credential_fingerprint != self._credential_fingerprint()
        ):
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
            "mode": "live",
            "base_url": self.endpoints.base_url,
            "account_suffix": self.credentials.account_no[-2:],
            # Bind a cached bearer token to the exact app/account credential set.
            # KIS accepts a token issued for an old app key until expiry, but rejects
            # subsequent REST headers when the local appSecret has been rotated.
            "credential_fingerprint": self._credential_fingerprint(),
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
        self._token_source = "issued"

    def _credential_fingerprint(self) -> str:
        material = "\0".join(
            (
                self.credentials.app_key,
                self.credentials.app_secret,
                self.credentials.account_no,
                self.credentials.account_product_code,
                self.endpoints.base_url,
            )
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

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

    def _domestic_exchange_id_code(self, order: FinalOrder) -> str:
        """``EXCG_ID_DVSN_CD``. 공식 허용값은 ``KRX`` / ``NXT`` / ``SOR`` 뿐이다.

        문서: "미입력시 KRX로 진행되며, 모의투자는 KRX만 가능". 이전 구현은 이 값을
        ``"KRX"`` 로 하드코딩해 NXT/SOR 라우팅이 아예 불가능했다. 여기서는 주문에 실린
        venue 힌트를 쓰고, 해석할 수 없으면 문서상 기본값인 KRX 로 둔다.
        """
        from app.data.market_capabilities import VERIFIED_ORDER_DIVISIONS

        for candidate in (
            getattr(order, "exchange_code", ""),
            getattr(order, "execution_venue", ""),
        ):
            code = str(candidate or "").upper().strip()
            if code == "NEXTRADE":
                code = "NXT"
            if code in VERIFIED_ORDER_DIVISIONS:
                return code
        return "KRX"

    def _record_order_route(
        self,
        order_id: str,
        *,
        route_family: str,
        endpoint: str,
        exchange_id_code: str | None = None,
        overseas_exchange_code: str | None = None,
        order_division: str | None = None,
        session: str = "",
        venue: str = "",
    ) -> None:
        """원주문이 실제로 어느 엔드포인트 family 로 갔는지 기록한다.

        정정·취소가 이 값을 읽어 **같은 family** 로만 라우팅한다. 세션이 바뀌었다고 다른
        venue 로 자동 정정하지 않는다 — 그런 전이는 "원주문 취소 + 신규 주문" 이라는
        명시적 상태 전이로만 허용된다.
        """
        self._order_routes[str(order_id)] = {
            "route_family": route_family,
            "endpoint": endpoint,
            "exchange_id_code": exchange_id_code,
            "overseas_exchange_code": overseas_exchange_code,
            "order_division": order_division,
            "session": session,
            "venue": venue,
        }

    def order_route(self, order_id: str) -> dict[str, Any] | None:
        """저널/코디네이터가 원주문 route 를 읽어 갈 수 있게 노출한다."""
        route = self._order_routes.get(str(order_id))
        return dict(route) if route else None

    def _overseas_revise_cancel_route(
        self, order_id: str, exchange_code: str, order: FinalOrder | None = None
    ) -> tuple[str, str]:
        """정정·취소 엔드포인트 + TR. **원주문 family 를 그대로 따른다.**

        원주문 route 를 모르는 경우(프로세스 재시작 등) 현재 시각으로 추정하지 않고
        현재 세션 판정으로 되돌아가되, 그 사실을 사유코드로 남길 수 있게 한다. 추정이
        위험한 이유는 주간거래 ODNO 를 일반 endpoint 로 보내면 원주문 불일치가 되기
        때문이다.
        """
        recorded = self._order_routes.get(str(order_id)) or {}
        family = str(recorded.get("route_family") or "")
        if not family:
            # 원주문 route 기록이 없다 (프로세스 재시작, 또는 이 프로세스가 접수하지 않은
            # 주문). 정정 대상 주문의 market 으로 현재 세션을 판정한다 — 기록이 있을 때는
            # 절대 이 경로로 오지 않으므로, 기록된 family 가 시각 재판정에 밀리지 않는다.
            known = order or self._orders.get(str(order_id))
            market = str(getattr(known, "market", "") or "")
            family = (
                "OVERSEAS_DAYTIME"
                if market and _is_us_daytime_order_session(market)
                else "OVERSEAS_REGULAR"
            )
        if family == "OVERSEAS_DAYTIME":
            return (
                "/uapi/overseas-stock/v1/trading/daytime-order-rvsecncl",
                self.endpoints.overseas_daytime_revise_cancel_tr_id,
            )
        return (
            "/uapi/overseas-stock/v1/trading/order-rvsecncl",
            self.endpoints.overseas_revise_cancel_tr_id(exchange_code),
        )

    def _order_body(self, order: FinalOrder) -> dict[str, str]:
        exchange_id_code = self._domestic_exchange_id_code(order)
        return {
            "CANO": self.credentials.account_no,
            "ACNT_PRDT_CD": self.credentials.account_product_code,
            "PDNO": order.ticker,
            "ORD_DVSN": _domestic_order_division_code(exchange_id_code=exchange_id_code),
            "ORD_QTY": str(int(order.quantity)),
            "ORD_UNPR": str(int(round(order.limit_price))),
            "EXCG_ID_DVSN_CD": exchange_id_code,
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
        """국내 정정·취소 body.

        ``ORD_DVSN`` 과 ``EXCG_ID_DVSN_CD`` 는 **원주문 값을 그대로** 쓴다. 시간외 단일가
        (07) 로 접수한 주문을 정정 시점 시각으로 다시 계산해 지정가(00) 로 보내면 거래소가
        원주문과 주문구분 불일치로 거부한다. 원주문 기록이 없으면(프로세스 재시작 등)
        현재 세션값으로 되돌아가되 거래소는 문서상 기본값인 KRX 를 유지한다.
        """
        recorded = self._order_routes.get(str(order_id)) or {}
        exchange_id_code = str(
            recorded.get("exchange_id_code") or self._domestic_exchange_id_code(order)
        )
        division = str(
            recorded.get("order_division")
            or _domestic_order_division_code(exchange_id_code=exchange_id_code)
        )
        return {
            "CANO": self.credentials.account_no,
            "ACNT_PRDT_CD": self.credentials.account_product_code,
            "KRX_FWDG_ORD_ORGNO": self._order_org_numbers.get(order_id, ""),
            "ORGN_ODNO": order_id,
            "ORD_DVSN": division,
            "RVSE_CNCL_DVSN_CD": "01" if revise else "02",
            "ORD_QTY": str(int(order.quantity)),
            "ORD_UNPR": str(int(round(order.limit_price))) if revise else "0",
            "QTY_ALL_ORD_YN": "N" if revise else "Y",
            "CNDT_PRIC": "",
            "EXCG_ID_DVSN_CD": exchange_id_code,
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

    def _account_balance_params(self) -> dict[str, str]:
        return {
            "CANO": self.credentials.account_no,
            "ACNT_PRDT_CD": self.credentials.account_product_code,
            "INQR_DVSN_1": "",
            "BSPR_BF_DT_APLY_YN": "",
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
        # Prefer nrcvb_buy_amt (미수없는매수금액 = cash + reusable sell proceeds, NO margin)
        # over ord_psbl_cash (주문가능현금 = settled deposit only). In KRX, sell proceeds are
        # immediately reusable for buying without going into 미수/credit, so basing orderable
        # cash on ord_psbl_cash alone locked the account into its tiny settled deposit
        # (e.g. 16,445) while 85,869 of reusable proceeds sat unused. nrcvb_buy_amt is
        # KIS-authoritative "how much you can buy without margin", so orders stay margin-free.
        return _first_float(output, "nrcvb_buy_amt", "ord_psbl_cash")

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

    def _get_overseas_cash_balance(self) -> tuple[dict[str, float], float, float, dict[str, float]]:
        balances: dict[str, float] = {}
        fx_rates: dict[str, float] = {}
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
                    return balances, foreign_cash_krw, total_assets_krw, fx_rates
                continue
            response_balances = _foreign_cash_by_currency_from_overseas_response(
                response,
                nation_code,
            )
            response_rates = _foreign_fx_by_currency_from_overseas_response(response, nation_code)
            balances.update(response_balances)
            fx_rates.update(response_rates)
            foreign_cash_krw = max(
                foreign_cash_krw,
                _foreign_cash_krw_from_overseas_response(response),
                _foreign_cash_summary_krw_from_overseas_response(response),
            )
            has_foreign_balance_context = bool(response_balances or response_rates or response.get("output1") or response.get("output2"))
            if has_foreign_balance_context:
                total_assets_krw = max(total_assets_krw, _total_assets_krw_from_overseas_response(response))
            if balances or foreign_cash_krw > 0 or total_assets_krw > 0:
                break
        return balances, foreign_cash_krw, total_assets_krw, fx_rates

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

    def _holding_from_balance(self, row: dict[str, Any]) -> Holding | None:
        ticker = str(row.get("pdno") or "").strip()
        quantity = int(_to_float(row.get("hldg_qty") or row.get("ord_psbl_qty") or 0))
        if not ticker or quantity <= 0:
            return None
        sellable_quantity = int(_to_float(row.get("ord_psbl_qty") or quantity))
        average_price = _to_float(row.get("pchs_avg_pric") or 0)
        last_price = _to_float(row.get("prpr") or row.get("bfdy_cprs_icdc") or average_price)
        opened_at = self._holding_opened_at_from_balance(row)
        return Holding(
            ticker=ticker,
            market="KR",
            company_name=str(row.get("prdt_name") or row.get("pdno") or ""),
            sector="Unknown",
            quantity=quantity,
            average_price=average_price,
            last_price=last_price,
            opened_at=opened_at,
            sellable_quantity=max(0, min(quantity, sellable_quantity)),
        )

    def _overseas_holding_from_balance(self, row: dict[str, Any]) -> Holding | None:
        ticker = str(
            row.get("ovrs_pdno")
            or row.get("pdno")
            or row.get("symb")
            or row.get("prdt_code")
            or ""
        ).upper().strip()
        quantity = int(_to_float(row.get("ovrs_cblc_qty") or row.get("hldg_qty") or row.get("ord_psbl_qty") or row.get("cblc_qty13") or row.get("ord_psbl_qty1") or 0))
        if not ticker or quantity <= 0:
            return None
        average_price = _first_float(row, "pchs_avg_pric", "avg_unpr", "avg_unpr3", "frcr_pchs_amt1")
        market_value = _first_float(row, "ovrs_stck_evlu_amt", "frcr_evlu_amt", "frcr_evlu_amt2", "evlu_amt")
        last_price = _first_float(row, "now_pric2", "ovrs_now_pric1", "ovrs_stck_prpr", "last", "prpr")
        if market_value > 0:
            implied_price = market_value / quantity
            if last_price <= 0 or abs((last_price * quantity) - market_value) > max(market_value, 1.0) * 0.5:
                last_price = implied_price
        if last_price <= 0:
            purchase_amount = _first_float(row, "pchs_amt", "frcr_pchs_amt", "frcr_pchs_amt1")
            market_value = market_value if market_value > 0 else purchase_amount
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


def _to_float_value(value: Any) -> float:
    """Alias kept explicit at the 대주 call sites (KIS returns comma-grouped text)."""
    return _to_float(value)


def _to_int(value: Any) -> int:
    """KIS quantities arrive as comma-grouped strings; unparseable means zero.

    Zero, not None: this feeds ``available_quantity``, and a quantity we cannot read
    must behave as "no locate" rather than as "unknown but maybe fine".
    """
    try:
        return int(float(str(value or "0").replace(",", "").strip() or "0"))
    except (TypeError, ValueError):
        return 0


def _to_optional_float(value: Any) -> float | None:
    """``None``-preserving float. An unreadable borrow RATE must stay unknown.

    Distinct from :func:`_to_float`, which returns 0.0. For a borrow fee, 0.0 means
    "free to borrow" and would let an unpriced short pass its cost gate — so this
    variant exists specifically so the fee can be absent rather than free.
    """
    if value in (None, ""):
        return None
    try:
        number = float(str(value).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # NaN-safe


def _parse_kis_deadline(value: Any) -> datetime | None:
    """Parse a KIS 'YYYYMMDD' repayment date into an aware datetime.

    Resolved at 15:30 KST, the end of the KRX session: a borrow due "on" a date must
    be repaid during that day's trading, not at midnight after it. Anchoring to
    midnight would make the recall-deadline guard think there was a full extra
    session available.
    """
    text = str(value or "").strip().replace("-", "")
    if len(text) != 8 or not text.isdigit():
        return None
    try:
        parsed = datetime.strptime(text, "%Y%m%d")
    except ValueError:
        return None
    return parsed.replace(hour=15, minute=30, tzinfo=timezone(timedelta(hours=9)))


def _require_credit_contract(
    order: FinalOrder, *, direction: str, effect: str, side: OrderSide
) -> None:
    """Fail closed unless the order's contract is exactly the intended one.

    Checks the direction, the effect, the execution product AND the broker side, all
    four. Any single mismatch means the order about to be submitted is not the order
    that was designed — most dangerously a SHORT/CLOSE contract carrying
    ``side=SELL``, which would sell stock the account does not hold instead of
    covering the borrow.
    """
    actual_direction = str(order.position_direction or "").strip().upper()
    # Resolved, not raw: an order whose effect was left to inference is still a valid
    # contract, and demanding the literal field would reject it for being implicit
    # rather than for being wrong.
    actual_effect = str(order.resolved_position_effect or "").strip().upper()
    product = str(order.execution_product or "").strip().upper()
    if actual_direction != direction or actual_effect != effect:
        raise ValueError(
            f"credit order contract mismatch for {order.ticker}: expected "
            f"{direction}/{effect}, got {actual_direction or '?'}/{actual_effect or '?'}"
        )
    if product != "CREDIT_BORROW":
        raise ValueError(
            f"credit order for {order.ticker} must declare execution_product="
            f"CREDIT_BORROW, got {product or '?'}"
        )
    if order.side != side:
        raise ValueError(
            f"credit order for {order.ticker} declares {direction}/{effect} but carries "
            f"broker side {order.side}; expected {side}"
        )
    expected_credit_type = "22" if effect.upper() == "OPEN" else "26"
    if str(order.credit_type or "").strip() != expected_credit_type:
        raise ValueError(
            f"credit order for {order.ticker} must use KIS CRDT_TYPE "
            f"{expected_credit_type} for SHORT/{effect.upper()}"
        )
    if order.quantity <= 0:
        raise ValueError(f"credit order for {order.ticker} must have positive quantity")


def _kis_yyyymmdd(value: Any) -> str:
    """Format a date/datetime/str as KIS 'YYYYMMDD'."""
    if hasattr(value, "strftime"):
        return value.strftime("%Y%m%d")
    return str(value).replace("-", "").strip()


def _first_float(data: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = _to_float(data.get(key))
        if value > 0:
            return value
    return 0.0


def _domestic_cash_from_balance_summary(summary_row: dict[str, Any], holdings: tuple[Holding, ...]) -> float:
    # Prefer the settlement-inclusive deposit (prvs_rcdl_excc_amt = 가수도정산금액 / D+2,
    # nxdy_excc_amt = 익일정산금액 / D+1) over dnca_tot_amt (예수금총액, settled-only).
    # In KRX, sell proceeds are immediately usable for re-buying and are part of total
    # assets; KIS's own tot_evlu_amt already counts them, so basing cash on dnca_tot_amt
    # undercounted the total by the pending sell proceeds (e.g. 16,445 vs 89,638) and
    # made the displayed total wrong (~136k instead of ~205k). Orderable-for-withdrawal
    # is fetched separately, so trading cash is unaffected.
    explicit_cash = _first_float(
        summary_row,
        "prvs_rcdl_excc_amt",
        "nxdy_excc_amt",
        "dnca_tot_amt",
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


def _domestic_total_assets_from_balance_summary(summary_row: dict[str, Any]) -> float:
    for key in ("tot_evlu_amt", "nass_amt", "bfdy_tot_asst_evlu_amt"):
        value = _to_float(summary_row.get(key))
        if value > 0:
            return value
    return 0.0


def _account_asset_summary_from_response(response: dict[str, Any]) -> dict[str, float]:
    summary = response.get("output2") or {}
    if isinstance(summary, list):
        summary_row = next((row for row in summary if isinstance(row, dict)), {})
    else:
        summary_row = summary if isinstance(summary, dict) else {}
    if not summary_row:
        return {}
    return {
        "total_assets_krw": _first_float(summary_row, "tot_asst_amt", "nass_tot_amt", "real_nass_amt"),
        "cash_krw": _first_float(summary_row, "tot_dncl_amt", "dncl_amt", "cma_evlu_amt"),
        "purchase_amount_krw": _first_float(summary_row, "pchs_amt_smtl"),
        "evaluation_amount_krw": _first_float(summary_row, "evlu_amt_smtl", "ovrs_stck_evlu_amt1"),
        "unrealized_pnl_krw": _first_float(summary_row, "evlu_pfls_amt_smtl"),
    }


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


# KIS 상품유형코드 for 해외주식 상품기본정보 (CTPF1702R).
US_PRODUCT_TYPE_CODES = {"NASD": "512", "NYSE": "513", "AMEX": "529"}


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


def _domestic_order_division_code(
    now: datetime | None = None, *, exchange_id_code: str = "KRX"
) -> str:
    """세션에 맞는 국내 ``ORD_DVSN``.

    거래소별 허용값이 **다르다** (공식 문서의 별도 표):

    * KRX — 00 지정가, 05 장전 시간외, 06 장후 시간외, 07 시간외 단일가 ...
    * NXT — 00 지정가, 03, 04, 11~16, 21~24. **05/06/07 없음.**
    * SOR — 00, 01, 03, 04, 11~16. **05/06/07·21~24 없음.**

    이전 구현은 거래소를 무시하고 시각만 보고 05/06/07 을 골랐다. ``EXCG_ID_DVSN_CD`` 가
    NXT 가 되는 순간 그 값은 비허용이 되어 조용히 거부된다. 그래서 여기서는 canonical
    service 가 계산한 세션별 값을 쓰고, 거래소 허용 집합으로 한 번 더 검증한다.
    """
    forced = os.getenv("KIS_DOMESTIC_ORD_DVSN", "").strip()
    from app.data.market_capabilities import (
        MarketGroup,
        VERIFIED_ORDER_DIVISIONS,
        Venue,
        default_service,
    )

    exchange = str(exchange_id_code or "KRX").upper().strip() or "KRX"
    allowed = VERIFIED_ORDER_DIVISIONS.get(exchange, VERIFIED_ORDER_DIVISIONS["KRX"])
    if forced:
        # 운영자가 명시한 값도 공식 허용 집합을 벗어나면 쓰지 않는다 (fail-closed).
        return forced if forced in allowed else "00"

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    target_venue = Venue.NXT if exchange == "NXT" else Venue.KRX
    for capability in default_service().active_capabilities(MarketGroup.KR, current):
        if capability.venue is not target_venue:
            continue
        division = capability.limit_order_division()
        if division in allowed:
            return division
    return "00"


def _daytime_tradable_cache_ttl_seconds() -> float:
    """How long a ``dtm_tr_psbl_yn`` answer stays fresh. It is reference data."""
    try:
        hours = float(os.getenv("KIS_DAYTIME_TRADABLE_CACHE_HOURS", "12"))
    except (TypeError, ValueError):
        hours = 12.0
    return max(60.0, hours * 3600.0)


def _enforce_daytime_tradable() -> bool:
    raw = os.getenv("KIS_ENFORCE_US_DAYTIME_TRADABLE", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _is_us_daytime_order_session(market: str, now: datetime | None = None) -> bool:
    """미국 주간거래(장전거래) 주문 세션인지.

    이전에는 여기서 KST 09:00-16:50 을 직접 계산했다. KIS 공식 문서의 주간거래 시간은
    **10:00 ~ 18:00 (한국시간, Summer Time 동일)** 이므로 두 방향으로 틀렸다:

    * 09:00-10:00 KST — 주간거래 시간이 아닌데 ``daytime-order`` 를 호출해 거부당함.
    * 16:50-18:00 KST — 주간거래 시간인데 일반 ``order`` 로 라우팅해 거부당함.

    판정을 canonical capability service 로 위임한다.
    근거: ``docs/kis_market_session_capability_matrix.md`` §5.1
    """
    market_name = str(market or "").upper()
    if not any(token in market_name for token in ("US", "NASDAQ", "NASD", "NYSE", "AMEX", "OVERSEAS")):
        return False
    override = os.getenv("KIS_FORCE_OVERSEAS_DAYTIME_ORDER", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    from app.data.market_capabilities import MarketGroup, SessionId, default_service

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return any(
        capability.session is SessionId.US_DAYTIME
        for capability in default_service().active_capabilities(MarketGroup.US, current)
    )


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


def _foreign_fx_by_currency_from_overseas_response(
    response: dict[str, Any],
    nation_code: str = "000",
) -> dict[str, float]:
    rates: dict[str, float] = {}
    for row in _response_rows(response):
        currency = _currency_from_row(row, nation_code)
        if not currency or currency == "KRW":
            continue
        rate = _exchange_rate_from_row(row)
        if rate > 0:
            rates[currency] = max(rates.get(currency, 0.0), rate)
    return rates


def _foreign_cash_krw_from_overseas_response(response: dict[str, Any]) -> float:
    best = 0.0
    for row in _response_rows(response):
        amount = _foreign_cash_amount_from_row(row)
        rate = _exchange_rate_from_row(row)
        if amount is not None and rate > 0:
            best = max(best, amount * rate)
    return best


def _foreign_cash_summary_krw_from_overseas_response(response: dict[str, Any]) -> float:
    best = 0.0
    for row in _response_rows(response):
        for key in ("tot_frcr_cblc_smtl", "frcr_use_psbl_amt", "frcr_evlu_tota", "frcr_evlu_amt2"):
            if key in row:
                best = max(best, _to_float(row.get(key)))
    return best


def _foreign_cash_krw_from_currency_balances(
    cash_by_currency: dict[str, float],
    fx_by_currency: dict[str, float],
) -> float:
    total = 0.0
    for currency, amount in cash_by_currency.items():
        code = str(currency or "").upper()
        if code == "KRW":
            continue
        rate = _to_float(fx_by_currency.get(code))
        if amount > 0 and rate > 0:
            total += amount * rate
    return total


def _overseas_holdings_value_krw(
    holdings: tuple[Holding, ...],
    cash_by_currency: dict[str, float],
    foreign_cash_krw: float,
    fx_by_currency: dict[str, float],
) -> float:
    if not holdings:
        return 0.0
    usd_rate = fx_by_currency.get("USD", 0.0)
    usd_cash = cash_by_currency.get("USD", 0.0)
    if usd_rate <= 0 and usd_cash > 0 and foreign_cash_krw > 0:
        usd_rate = foreign_cash_krw / usd_cash
    total = 0.0
    for holding in holdings:
        market = str(getattr(holding, "market", "") or "").upper()
        value = max(0.0, float(getattr(holding, "market_value", 0.0) or 0.0))
        if market and market != "KR":
            total += value * usd_rate if usd_rate > 0 else value
        else:
            total += value
    return total


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


def _default_token_cache_path(paper: bool = False) -> Path:
    return Path(os.getenv("KIS_TOKEN_CACHE_DIR", "config/secrets")) / "kis_access_token.live.json"


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


def _throttle_kis_get() -> None:
    """Serialize live KIS GETs across client instances to stay below burst quotas."""
    global _KIS_GET_NEXT_ALLOWED_AT
    minimum_interval = max(0.05, float(os.getenv("KIS_GLOBAL_GET_INTERVAL_SEC", "0.25")))
    with _KIS_GET_RATE_LOCK:
        wait_seconds = _KIS_GET_NEXT_ALLOWED_AT - time.monotonic()
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        _KIS_GET_NEXT_ALLOWED_AT = time.monotonic() + minimum_interval


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

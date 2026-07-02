from __future__ import annotations

from datetime import date
from typing import Any

from app.execution.kis_real import KisDevelopersApiClient


class KisOverseasAccountClient:
    """Read-only overseas account endpoints for the account dashboard."""

    def __init__(self, client: KisDevelopersApiClient) -> None:
        self.client = client

    @classmethod
    def from_client(cls, client: KisDevelopersApiClient) -> "KisOverseasAccountClient":
        return cls(client)

    def inquire_overseas_balance(self, exchange: str = "", currency: str = "USD") -> dict[str, Any]:
        params = {
            "CANO": self.client.credentials.account_no,
            "ACNT_PRDT_CD": self.client.credentials.account_product_code,
            "OVRS_EXCG_CD": exchange,
            "TR_CRCY_CD": currency,
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        }
        return self.client._get(  # noqa: SLF001 - reuses the existing authenticated KIS request wrapper.
            "/uapi/overseas-stock/v1/trading/inquire-balance",
            tr_id=self.client.endpoints.overseas_balance_tr_id,
            params=params,
        )

    def inquire_overseas_present_balance(
        self,
        basis: str = "KRW",
        nation: str = "000",
        market: str = "00",
    ) -> dict[str, Any]:
        params = {
            "CANO": self.client.credentials.account_no,
            "ACNT_PRDT_CD": self.client.credentials.account_product_code,
            "WCRC_FRCR_DVSN_CD": "02" if basis.upper() == "KRW" else "01",
            "NATN_CD": nation,
            "TR_MKET_CD": market,
            "INQR_DVSN_CD": "00",
        }
        return self.client._get(  # noqa: SLF001
            "/uapi/overseas-stock/v1/trading/inquire-present-balance",
            tr_id=self.client.endpoints.overseas_present_balance_tr_id,
            params=params,
        )

    def inquire_overseas_buyable_amount(
        self,
        exchange: str,
        ticker: str,
        order_price: float,
        currency: str = "USD",
    ) -> dict[str, Any]:
        params = {
            "CANO": self.client.credentials.account_no,
            "ACNT_PRDT_CD": self.client.credentials.account_product_code,
            "OVRS_EXCG_CD": exchange,
            "TR_CRCY_CD": currency,
            "OVRS_ORD_UNPR": _fmt(order_price),
            "ITEM_CD": ticker.upper(),
        }
        return self.client._get(  # noqa: SLF001
            "/uapi/overseas-stock/v1/trading/inquire-psamount",
            tr_id=self.client.endpoints.overseas_orderable_cash_tr_id,
            params=params,
        )

    def inquire_overseas_order_fills(
        self,
        start_date: date | str,
        end_date: date | str,
        ticker: str = "%",
        exchange: str = "%",
    ) -> dict[str, Any]:
        params = self._date_range_params(start_date, end_date)
        params.update({"PDNO": ticker, "OVRS_EXCG_CD": exchange, "SLL_BUY_DVSN": "00", "CCLD_NCCS_DVSN": "00"})
        return self.client._get(  # noqa: SLF001
            "/uapi/overseas-stock/v1/trading/inquire-ccnl",
            tr_id=self.client.endpoints.overseas_order_status_tr_id,
            params=params,
        )

    def inquire_overseas_period_profit(
        self,
        start_date: date | str,
        end_date: date | str,
        exchange: str = "",
        currency: str = "",
        ticker: str = "",
    ) -> dict[str, Any]:
        params = self._date_range_params(start_date, end_date)
        params.update({"OVRS_EXCG_CD": exchange, "TR_CRCY_CD": currency, "PDNO": ticker})
        return self.client._get(  # noqa: SLF001
            "/uapi/overseas-stock/v1/trading/inquire-period-profit",
            tr_id="TTTS3039R" if not self.client.paper else "VTTS3039R",
            params=params,
        )

    def inquire_overseas_period_transactions(self, start_date: date | str, end_date: date | str) -> dict[str, Any]:
        return self.client._get(  # noqa: SLF001
            "/uapi/overseas-stock/v1/trading/inquire-period-trans",
            tr_id="CTOS4001R" if not self.client.paper else "VTOS4001R",
            params=self._date_range_params(start_date, end_date),
        )

    def inquire_foreign_margin(self) -> dict[str, Any]:
        return self.client._get(  # noqa: SLF001
            "/uapi/overseas-stock/v1/trading/foreign-margin",
            tr_id="TTTC2101R" if not self.client.paper else "VTTC2101R",
            params={"CANO": self.client.credentials.account_no, "ACNT_PRDT_CD": self.client.credentials.account_product_code},
        )

    def inquire_overseas_open_orders(self) -> dict[str, Any]:
        today = date.today()
        return self.inquire_overseas_order_fills(today, today, ticker="%", exchange="%")

    def _date_range_params(self, start_date: date | str, end_date: date | str) -> dict[str, str]:
        return {
            "CANO": self.client.credentials.account_no,
            "ACNT_PRDT_CD": self.client.credentials.account_product_code,
            "INQR_STRT_DT": _date_text(start_date),
            "INQR_END_DT": _date_text(end_date),
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        }


def _date_text(value: date | str) -> str:
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    return str(value).replace("-", "")


def _fmt(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".")

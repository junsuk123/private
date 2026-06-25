from __future__ import annotations

import os

from app.execution.kis_mock import MockKisExecution, MockKisOrderReceipt, MockKisPortfolio
from app.schemas.domain import FinalOrder


class KisDevelopersApiClient:
    """Replaceable real KIS Developers adapter skeleton.

    The trading pipeline depends only on the BrokerClient protocol. This class is
    intentionally disabled until real KIS credentials, token handling, endpoint
    IDs, and audit controls are wired in.
    """

    def __init__(
        self,
        app_key: str | None = None,
        app_secret: str | None = None,
        account_no: str | None = None,
        base_url: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.app_key = app_key or os.getenv("KIS_APP_KEY")
        self.app_secret = app_secret or os.getenv("KIS_APP_SECRET")
        self.account_no = account_no or os.getenv("KIS_ACCOUNT_NO")
        self.base_url = base_url or os.getenv("KIS_BASE_URL", "https://openapi.koreainvestment.com:9443")
        self.enabled = bool(enabled) if enabled is not None else os.getenv("KIS_LIVE_ENABLED") == "1"

    def place_limit_order(self, order: FinalOrder) -> MockKisOrderReceipt:
        self._ensure_enabled()
        raise NotImplementedError("Real KIS limit-order submission is not implemented yet.")

    def get_order_status(self, order_id: str) -> MockKisExecution:
        self._ensure_enabled()
        raise NotImplementedError("Real KIS order-status polling is not implemented yet.")

    def get_portfolio(self) -> MockKisPortfolio:
        self._ensure_enabled()
        raise NotImplementedError("Real KIS portfolio retrieval is not implemented yet.")

    def _ensure_enabled(self) -> None:
        if not self.enabled:
            raise RuntimeError("Real KIS API is disabled. Use MockKisDevelopersApi for simulation.")

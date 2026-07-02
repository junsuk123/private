from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.account_dashboard import AccountDashboardService
from app.account_snapshot_store import AccountSnapshotStore
from app.execution.kis_overseas import KisOverseasAccountClient
from app.execution.kis_real import KisDevelopersApiClient


class AccountDashboardTest(unittest.TestCase):
    def test_dashboard_normalizes_cash_holdings_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = AccountDashboardService(
                status_provider=lambda: {
                    "basis_source": "kis_live_account",
                    "account_checked": True,
                    "updated_at": "2026-07-01T00:00:00+00:00",
                    "krw_cash": 100_000,
                    "foreign_cash_krw": 50_000,
                    "cash_equivalent_krw": 150_000,
                    "equity": 250_000,
                    "cash_by_currency": {"KRW": 100_000, "USD": 40.0},
                    "positions": [
                        {
                            "ticker": "005930",
                            "market": "KRX",
                            "currency": "KRW",
                            "quantity": 1,
                            "average_price": 70_000,
                            "last_price": 80_000,
                            "market_value_krw": 80_000,
                        },
                        {
                            "ticker": "AAPL",
                            "market": "NASDAQ",
                            "currency": "USD",
                            "quantity": 1,
                            "average_price": 100,
                            "last_price": 110,
                            "market_value_krw": 20_000,
                        },
                    ],
                },
                logs_provider=lambda: {"collection_log": [], "last_error": None},
                store=AccountSnapshotStore(Path(tmp) / "account.sqlite3"),
            )

            dashboard = service.build_dashboard()
            history = service.asset_history("1D")

        snapshot = dashboard["snapshot"]
        self.assertEqual(snapshot["total_asset_krw"], 250_000)
        self.assertEqual(snapshot["domestic_stock_value_krw"], 80_000)
        self.assertEqual(snapshot["overseas_stock_value_krw"], 20_000)
        self.assertEqual(len(dashboard["holdings"]), 2)
        self.assertEqual(len(history), 1)

    def test_overseas_account_client_maps_balance_request(self) -> None:
        transport = _RecordingTransport()
        with tempfile.TemporaryDirectory() as tmp:
            client = KisDevelopersApiClient(
                app_key="paper-app",
                app_secret="paper-secret",
                account_no="12345678-01",
                paper=True,
                enabled=True,
                transport=transport,
                token_cache_path=Path(tmp) / "token.json",
            )
            overseas = KisOverseasAccountClient(client)

            overseas.inquire_overseas_balance("NASD", "USD")

        call = transport.calls[-1]
        self.assertTrue(call["url"].endswith("/uapi/overseas-stock/v1/trading/inquire-balance"))
        self.assertEqual(call["headers"]["tr_id"], "VTTS3012R")
        self.assertEqual(call["params"]["OVRS_EXCG_CD"], "NASD")
        self.assertEqual(call["params"]["TR_CRCY_CD"], "USD")


class _RecordingTransport:
    def __init__(self) -> None:
        self.calls = []

    def request(self, method, url, headers, body=None, params=None, timeout=10.0):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "body": dict(body or {}),
                "params": dict(params or {}),
            }
        )
        if url.endswith("/oauth2/tokenP"):
            return {"access_token": "token", "expires_in": 86400}
        return {"rt_cd": "0", "output": {}, "output1": [], "output2": []}


if __name__ == "__main__":
    unittest.main()

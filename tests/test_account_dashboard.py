from __future__ import annotations

import sys
import tempfile
import unittest
import sqlite3
import gc
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

    def test_dashboard_uses_actual_equity_when_equity_key_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = AccountDashboardService(
                status_provider=lambda: {
                    "basis_source": "kis_live_account",
                    "account_checked": True,
                    "updated_at": "2026-07-01T00:00:00+00:00",
                    "actual_deposit": 22_440,
                    "foreign_cash_krw": 85_468.747,
                    "cash_equivalent_krw": 108_384.165,
                    "actual_equity": 124_628,
                    "cash_by_currency": {"KRW": 22_440, "USD": 55.51},
                    "positions": [
                        {
                            "ticker": "LCFYW",
                            "market": "NASD",
                            "currency": "USD",
                            "quantity": 1,
                            "average_price": 10.0,
                            "last_price": 10.55,
                            "market_value_krw": 16_243.835,
                        }
                    ],
                },
                logs_provider=lambda: {"collection_log": [], "last_error": None},
                store=AccountSnapshotStore(Path(tmp) / "account.sqlite3"),
            )

            dashboard = service.build_dashboard()

        snapshot = dashboard["snapshot"]
        self.assertEqual(snapshot["total_asset_krw"], 124_628)
        self.assertEqual(snapshot["krw_cash"], 22_440)
        self.assertEqual(snapshot["cash_equivalent_krw"], 108_384.165)
        self.assertEqual(len(dashboard["holdings"]), 1)

    def test_asset_history_rolls_up_to_one_point_per_minute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AccountSnapshotStore(Path(tmp) / "account.sqlite3")
            first = {
                "snapshot": {
                    "created_at": "2026-07-01T00:00:05+00:00",
                    "source": "kis_live_account",
                    "total_asset_krw": 100_000,
                    "net_asset_krw": 100_000,
                },
                "holdings": [],
                "cash": [],
                "trades": [],
            }
            second = {
                "snapshot": {
                    "created_at": "2026-07-01T00:00:55+00:00",
                    "source": "kis_live_account",
                    "total_asset_krw": 101_000,
                    "net_asset_krw": 101_000,
                },
                "holdings": [],
                "cash": [],
                "trades": [],
            }
            third = {
                "snapshot": {
                    "created_at": "2026-07-01T00:01:01+00:00",
                    "source": "kis_live_account",
                    "total_asset_krw": 102_000,
                    "net_asset_krw": 102_000,
                },
                "holdings": [],
                "cash": [],
                "trades": [],
            }

            first_id = store.save_dashboard(first)
            second_id = store.save_dashboard(second)
            third_id = store.save_dashboard(third)
            history = store.asset_history("1W")

        self.assertEqual(first_id, second_id)
        self.assertNotEqual(second_id, third_id)
        self.assertEqual([row["total_asset_krw"] for row in history], [101_000, 102_000])

    def test_trade_events_are_persisted_when_snapshot_updates_same_minute(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db_path = Path(tmp) / "account.sqlite3"
            store = AccountSnapshotStore(db_path)
            base = {
                "snapshot": {
                    "created_at": "2026-07-01T00:00:05+00:00",
                    "source": "kis_live_account",
                    "total_asset_krw": 100_000,
                    "net_asset_krw": 100_000,
                },
                "holdings": [],
                "cash": [],
                "trades": [],
            }
            updated = {
                **base,
                "snapshot": {**base["snapshot"], "created_at": "2026-07-01T00:00:55+00:00"},
                "trades": [
                    {
                        "occurred_at": "2026-07-01T00:00:40+00:00",
                        "ticker": "005930",
                        "market_group": "domestic",
                        "market": "KR",
                        "side": "BUY",
                        "order_id": "0001",
                        "order_status": "ACCEPTED",
                        "ordered_quantity": 1,
                        "filled_quantity": 0,
                        "average_fill_price": 70000,
                        "amount_krw": 70000,
                        "currency": "KRW",
                        "source": "live_order_submitted",
                    }
                ],
            }

            first_id = store.save_dashboard(base)
            second_id = store.save_dashboard(updated)
            store.save_dashboard(updated)
            with sqlite3.connect(db_path) as conn:
                count = conn.execute("select count(*) from trade_events").fetchone()[0]
                conn.execute("pragma wal_checkpoint(TRUNCATE)").fetchall()
            del store
            gc.collect()

        self.assertEqual(first_id, second_id)
        self.assertEqual(count, 1)

    def test_dashboard_maps_live_order_journal_to_trade_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = AccountDashboardService(
                status_provider=lambda: {
                    "basis_source": "kis_live_account",
                    "account_checked": True,
                    "updated_at": "2026-07-01T00:00:00+00:00",
                    "equity": 100_000,
                },
                logs_provider=lambda: {
                    "live_order_journal": {
                        "submitted_orders": [
                            {
                                "event_type": "live_order_submitted",
                                "recorded_at": "2026-07-01T00:00:10+00:00",
                                "ticker": "005930",
                                "market": "KR",
                                "side": "BUY",
                                "quantity": 1,
                                "limit_price": 70000,
                                "broker_order_id": "0001",
                                "status": "ACCEPTED",
                                "currency": "KRW",
                            }
                        ]
                    }
                },
                store=AccountSnapshotStore(Path(tmp) / "account.sqlite3"),
            )

            dashboard = service.build_dashboard()

        self.assertEqual(len(dashboard["trades"]), 1)
        self.assertEqual(dashboard["trades"][0]["order_id"], "0001")
        self.assertEqual(dashboard["trades"][0]["order_status"], "ACCEPTED")

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

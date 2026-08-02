from __future__ import annotations

import sys
import tempfile
import unittest
import sqlite3
import gc
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.account_dashboard import AccountDashboardService
from app.account_snapshot_store import AccountSnapshotStore
from app.execution.kis_overseas import KisOverseasAccountClient
from app.execution.kis_real import KisDevelopersApiClient


class AccountDashboardTest(unittest.TestCase):
    def test_cached_asset_summary_prefers_last_verified_live_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AccountSnapshotStore(Path(tmp) / "account.sqlite3")
            store.save_dashboard(
                {
                    "snapshot": {
                        "created_at": "2026-07-01T00:00:00+00:00",
                        "updated_at": "2026-07-01T00:00:00+00:00",
                        "source": "kis_live_account",
                        "is_live": True,
                        "total_asset_krw": 217_584,
                    },
                    "holdings": [{"ticker": "AAPL"}],
                }
            )
            store.save_dashboard(
                {
                    "snapshot": {
                        "created_at": "2026-07-01T00:01:00+00:00",
                        "updated_at": "2026-07-01T00:01:00+00:00",
                        "source": "analysis_context_fallback",
                        "is_live": False,
                        "total_asset_krw": 1_000_000,
                    },
                    "holdings": [],
                }
            )
            service = AccountDashboardService(store=store)

            summary = service.cached_asset_summary()

        self.assertEqual(summary["status"], "last_known")
        self.assertFalse(summary["authoritative"])
        self.assertEqual(summary["current_source"], "analysis_context_fallback")
        self.assertEqual(summary["snapshot"]["total_asset_krw"], 217_584)
        self.assertEqual(summary["holdings"][0]["ticker"], "AAPL")

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

    def test_live_zero_balance_does_not_reuse_fallback_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AccountSnapshotStore(Path(tmp) / "account.sqlite3")
            store.save_dashboard(
                {
                    "snapshot": {
                        "created_at": "2026-07-01T00:00:00+00:00",
                        "source": "analysis_context_fallback",
                        "total_asset_krw": 1_000_000,
                        "cash_equivalent_krw": 1_000_000,
                        "krw_cash": 1_000_000,
                    }
                }
            )
            service = AccountDashboardService(
                status_provider=lambda: {
                    "basis_source": "kis_live_account",
                    "account_checked": True,
                    "updated_at": "2026-07-01T00:01:00+00:00",
                    "equity": 0,
                    "krw_cash": 0,
                    "foreign_cash_krw": 0,
                    "cash_equivalent_krw": 0,
                    "positions": [],
                },
                store=store,
            )

            snapshot = service.build_dashboard(persist=False)["snapshot"]

        self.assertTrue(snapshot["is_live"])
        self.assertEqual(snapshot["source"], "kis_live_account")
        self.assertEqual(snapshot["total_asset_krw"], 0)
        self.assertEqual(snapshot["krw_cash"], 0)
        self.assertEqual(snapshot["foreign_cash_krw"], 0)

    def test_asset_history_rolls_up_to_one_point_per_minute(self) -> None:
        # Timestamps are relative to now. They used to be hardcoded to 2026-07-01,
        # which sat inside the "1M" lookback until the calendar rolled past 30 days
        # and then silently fell out of the window — the test began failing on a
        # date change rather than on a code change. The behaviour under test is the
        # per-minute roll-up, not the window boundary.
        base = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(
            days=2
        )

        def _stamp(offset_seconds: int) -> str:
            return (base + timedelta(seconds=offset_seconds)).isoformat()

        with tempfile.TemporaryDirectory() as tmp:
            store = AccountSnapshotStore(Path(tmp) / "account.sqlite3")
            first = {
                "snapshot": {
                    "created_at": _stamp(5),
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
                    "created_at": _stamp(55),
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
                    "created_at": _stamp(61),
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
            history = store.asset_history("1M")

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

    def test_dashboard_maps_live_order_journal_to_holding_order_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = AccountDashboardService(
                status_provider=lambda: {
                    "basis_source": "kis_live_account",
                    "account_checked": True,
                    "updated_at": "2026-07-01T00:00:00+00:00",
                    "equity": 100_000,
                    "positions": [
                        {
                            "ticker": "005930",
                            "market": "KRX",
                            "currency": "KRW",
                            "quantity": 1,
                            "average_price": 70_000,
                            "last_price": 71_000,
                            "market_value_krw": 71_000,
                        },
                        {
                            "ticker": "AAPL",
                            "market": "NASDAQ",
                            "currency": "USD",
                            "quantity": 1,
                            "average_price": 100,
                            "last_price": 102,
                            "market_value_krw": 140_000,
                        },
                    ],
                },
                logs_provider=lambda: {
                    "live_order_journal": {
                        "recent_orders": [
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
                            },
                            {
                                "event_type": "live_order_amended",
                                "recorded_at": "2026-07-01T00:00:20+00:00",
                                "ticker": "005930",
                                "market": "KR",
                                "side": "SELL",
                                "quantity": 1,
                                "limit_price": 71000,
                                "broker_order_id": "0002",
                                "status": "ACCEPTED",
                                "currency": "KRW",
                            }
                        ]
                    }
                },
                store=AccountSnapshotStore(Path(tmp) / "account.sqlite3"),
            )

            dashboard = service.build_dashboard()

        self.assertEqual(len(dashboard["holding_orders"]), 2)
        self.assertEqual(dashboard["holding_orders"][0]["ticker"], "005930")
        self.assertEqual(dashboard["holding_orders"][0]["average_price"], 70_000)
        self.assertEqual(dashboard["holding_orders"][0]["order_state"], "정정")
        self.assertEqual(dashboard["holding_orders"][0]["order_id"], "0002")
        self.assertEqual(dashboard["holding_orders"][0]["order_price"], 71_000)
        self.assertEqual(dashboard["holding_orders"][1]["ticker"], "AAPL")
        self.assertEqual(dashboard["holding_orders"][1]["order_state"], "주문 없음")

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
        self.assertEqual(call["headers"]["tr_id"], "TTTS3012R")
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

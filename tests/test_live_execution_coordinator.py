from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.config import load_order_execution_config
from app.execution.idempotency_store import IdempotencyStore
from app.execution.kis_errors import LiveExecutionBlocked
from app.execution.kis_real import KisDevelopersApiClient
from app.execution.live_execution_coordinator import LiveExecutionCoordinator
from app.execution.live_order_journal import LiveOrderJournal
from app.schemas.domain import FinalOrder, OrderSide, OrderType
from app.trading.live_runtime_guard import create_arming_file


class OrderTransport:
    def __init__(self) -> None:
        self.order_count = 0
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
        if url.endswith("/oauth2/Approval"):
            return {"approval_key": "approval"}
        if url.endswith("/uapi/hashkey"):
            return {"HASH": "hash"}
        if url.endswith("/uapi/domestic-stock/v1/trading/inquire-balance"):
            return {"rt_cd": "0", "output1": [], "output2": [{"dnca_tot_amt": "1000000"}]}
        if url.endswith("/uapi/domestic-stock/v1/trading/inquire-psbl-order"):
            return {"rt_cd": "0", "output": {"ord_psbl_cash": "1000000"}}
        if url.endswith("/uapi/overseas-stock/v1/trading/inquire-present-balance"):
            return {"rt_cd": "0", "output1": [], "output2": []}
        if url.endswith("/uapi/domestic-stock/v1/trading/order-cash"):
            self.order_count += 1
            return {"rt_cd": "0", "msg1": "accepted", "output": {"ODNO": "0000000010"}}
        if url.endswith("/uapi/overseas-stock/v1/trading/order"):
            self.order_count += 1
            return {"rt_cd": "0", "msg1": "overseas accepted", "output": {"ODNO": "OVRS000010"}}
        if url.endswith("/uapi/overseas-stock/v1/trading/daytime-order"):
            self.order_count += 1
            return {"rt_cd": "0", "msg1": "daytime accepted", "output": {"ODNO": "DAY000010"}}
        if url.endswith("/uapi/overseas-stock/v1/trading/inquire-ccnl"):
            return {
                "rt_cd": "0",
                "output": [
                    {
                        "odno": "OVRS000010",
                        "pdno": "SOXX",
                        "sll_buy_dvsn_cd": "02",
                        "ft_ord_qty": "1",
                        "ft_ccld_qty": "0",
                        "nccs_qty": "1",
                        "ft_ord_unpr3": "624.71000000",
                        "ft_ccld_unpr3": "0.00000000",
                    }
                ],
            }
        raise AssertionError(f"unexpected request: {url}")


class LiveExecutionCoordinatorTest(unittest.TestCase):
    def test_default_runtime_flags_block_submission_before_broker_order(self) -> None:
        transport = OrderTransport()
        with tempfile.TemporaryDirectory() as tmp:
            coordinator = self._coordinator(tmp, transport)
            env = {
                "LIVE_TRADING_ENABLED": "false",
                "KIS_LIVE_ENABLED": "false",
                "KIS_PAPER_TRADING": "true",
                "LIVE_ORDER_SUBMIT_ENABLED": "false",
                "KILL_SWITCH_ENABLED": "true",
            }
            with patch.dict("os.environ", env, clear=True):
                with self.assertRaises(LiveExecutionBlocked) as raised:
                    coordinator.submit_final_order(_order(), idempotency_key="unit")

        self.assertIn("LIVE_TRADING_ENABLED_NOT_TRUE", raised.exception.reason_codes)
        self.assertEqual(transport.order_count, 0)

    def test_live_submission_is_idempotent_when_all_gates_pass(self) -> None:
        transport = OrderTransport()
        with tempfile.TemporaryDirectory() as tmp:
            arming_path = Path("config/secrets/live_trading_armed.json")
            create_arming_file(arming_path, ttl_seconds=60)
            coordinator = self._coordinator(tmp, transport)
            env = {
                "LIVE_TRADING_ENABLED": "true",
                "KIS_LIVE_ENABLED": "true",
                "KIS_PAPER_TRADING": "false",
                "LIVE_ORDER_SUBMIT_ENABLED": "true",
                "KILL_SWITCH_ENABLED": "false",
            }
            try:
                with patch.dict("os.environ", env, clear=True):
                    first = coordinator.submit_final_order(_order(), idempotency_key="same-key")
                    second = coordinator.submit_final_order(_order(), idempotency_key="same-key")
            finally:
                try:
                    arming_path.unlink()
                except FileNotFoundError:
                    pass

        self.assertEqual(first.broker_order_id, "0000000010")
        self.assertEqual(second.broker_order_id, "0000000010")
        self.assertEqual(transport.order_count, 1)

    def test_live_submission_allows_overseas_limit_order(self) -> None:
        transport = OrderTransport()
        with tempfile.TemporaryDirectory() as tmp:
            arming_path = Path("config/secrets/live_trading_armed.json")
            create_arming_file(arming_path, ttl_seconds=60)
            coordinator = self._coordinator(tmp, transport)
            env = {
                "LIVE_TRADING_ENABLED": "true",
                "KIS_LIVE_ENABLED": "true",
                "KIS_PAPER_TRADING": "false",
                "LIVE_ORDER_SUBMIT_ENABLED": "true",
                "KILL_SWITCH_ENABLED": "false",
            }
            try:
                with patch.dict("os.environ", env, clear=True):
                    submitted = coordinator.submit_final_order(_overseas_order(), idempotency_key="us-order")
            finally:
                try:
                    arming_path.unlink()
                except FileNotFoundError:
                    pass

        order_call = next(call for call in transport.calls if call["url"].endswith("/overseas-stock/v1/trading/order"))
        self.assertEqual(order_call["headers"]["tr_id"], "TTTT1002U")
        self.assertEqual(order_call["body"]["OVRS_EXCG_CD"], "NASD")
        self.assertEqual(order_call["body"]["PDNO"], "SOXX")
        self.assertEqual(order_call["body"]["ORD_QTY"], "1")
        self.assertEqual(order_call["body"]["OVRS_ORD_UNPR"], "624.71")
        self.assertEqual(submitted.broker_order_id, "OVRS000010")

    def test_overseas_order_status_uses_overseas_ccnl_and_reports_open_quantity(self) -> None:
        transport = OrderTransport()
        with tempfile.TemporaryDirectory() as tmp:
            client = KisDevelopersApiClient(
                app_key="app",
                app_secret="secret",
                account_no="12345678-01",
                base_url="https://openapi.koreainvestment.com:9443",
                paper=False,
                enabled=True,
                transport=transport,
                token_cache_path=Path(tmp) / "token.json",
            )
            client._orders["OVRS000010"] = _overseas_order()

            execution = client.get_order_status("OVRS000010")

        status_call = next(call for call in transport.calls if call["url"].endswith("/overseas-stock/v1/trading/inquire-ccnl"))
        self.assertEqual(status_call["headers"]["tr_id"], "TTTS3035R")
        self.assertEqual(status_call["params"]["SLL_BUY_DVSN"], "02")
        self.assertEqual(status_call["params"]["CCLD_NCCS_DVSN"], "00")
        self.assertEqual(execution.ticker, "SOXX")
        self.assertEqual(execution.status, "OPEN")
        self.assertEqual(execution.quantity, 0)

    def test_live_submission_routes_us_daytime_order_to_daytime_api(self) -> None:
        transport = OrderTransport()
        with tempfile.TemporaryDirectory() as tmp:
            arming_path = Path("config/secrets/live_trading_armed.json")
            create_arming_file(arming_path, ttl_seconds=60)
            coordinator = self._coordinator(tmp, transport)
            env = {
                "LIVE_TRADING_ENABLED": "true",
                "KIS_LIVE_ENABLED": "true",
                "KIS_PAPER_TRADING": "false",
                "LIVE_ORDER_SUBMIT_ENABLED": "true",
                "KILL_SWITCH_ENABLED": "false",
                "KIS_FORCE_OVERSEAS_DAYTIME_ORDER": "true",
            }
            try:
                with patch.dict("os.environ", env, clear=True):
                    submitted = coordinator.submit_final_order(_overseas_order(), idempotency_key="us-daytime-order")
            finally:
                try:
                    arming_path.unlink()
                except FileNotFoundError:
                    pass

        order_call = next(call for call in transport.calls if call["url"].endswith("/overseas-stock/v1/trading/daytime-order"))
        self.assertEqual(order_call["headers"]["tr_id"], "TTTS6036U")
        self.assertEqual(order_call["body"]["OVRS_EXCG_CD"], "NASD")
        self.assertEqual(submitted.broker_order_id, "DAY000010")

    def test_domestic_order_can_use_after_hours_order_division(self) -> None:
        transport = OrderTransport()
        with tempfile.TemporaryDirectory() as tmp:
            arming_path = Path("config/secrets/live_trading_armed.json")
            create_arming_file(arming_path, ttl_seconds=60)
            coordinator = self._coordinator(tmp, transport)
            env = {
                "LIVE_TRADING_ENABLED": "true",
                "KIS_LIVE_ENABLED": "true",
                "KIS_PAPER_TRADING": "false",
                "LIVE_ORDER_SUBMIT_ENABLED": "true",
                "KILL_SWITCH_ENABLED": "false",
                "KIS_DOMESTIC_ORD_DVSN": "07",
            }
            try:
                with patch.dict("os.environ", env, clear=True):
                    coordinator.submit_final_order(_order(), idempotency_key="krx-after-hours")
            finally:
                try:
                    arming_path.unlink()
                except FileNotFoundError:
                    pass

        order_call = next(call for call in transport.calls if call["url"].endswith("/domestic-stock/v1/trading/order-cash"))
        self.assertEqual(order_call["body"]["ORD_DVSN"], "07")

    def _coordinator(self, tmp: str, transport: OrderTransport) -> LiveExecutionCoordinator:
        client = KisDevelopersApiClient(
            app_key="app",
            app_secret="secret",
            account_no="12345678-01",
            base_url="https://openapi.koreainvestment.com:9443",
            paper=False,
            enabled=True,
            transport=transport,
            token_cache_path=Path(tmp) / "token.json",
        )
        return LiveExecutionCoordinator(
            client,
            idempotency_store=IdempotencyStore(Path(tmp) / "idempotency.jsonl"),
            journal=LiveOrderJournal(Path(tmp) / "live-orders.jsonl"),
            execution_config=load_order_execution_config("config/order_execution.json"),
        )


def _order() -> FinalOrder:
    return FinalOrder(
        ticker="005930",
        market="KR",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=1,
        limit_price=70000,
    )


def _overseas_order() -> FinalOrder:
    return FinalOrder(
        ticker="SOXX",
        market="US-LISTED",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=1,
        limit_price=624.71,
    )


if __name__ == "__main__":
    unittest.main()

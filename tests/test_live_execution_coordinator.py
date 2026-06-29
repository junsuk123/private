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

    def request(self, method, url, headers, body=None, params=None, timeout=10.0):
        if url.endswith("/oauth2/tokenP"):
            return {"access_token": "token", "expires_in": 86400}
        if url.endswith("/oauth2/Approval"):
            return {"approval_key": "approval"}
        if url.endswith("/uapi/hashkey"):
            return {"HASH": "hash"}
        if url.endswith("/uapi/domestic-stock/v1/trading/inquire-balance"):
            return {"rt_cd": "0", "output1": [], "output2": [{"dnca_tot_amt": "1000000"}]}
        if url.endswith("/uapi/domestic-stock/v1/trading/order-cash"):
            self.order_count += 1
            return {"rt_cd": "0", "msg1": "accepted", "output": {"ODNO": "0000000010"}}
        raise AssertionError(f"unexpected request: {url}")


class LiveExecutionCoordinatorTest(unittest.TestCase):
    def test_default_runtime_flags_block_submission_before_broker_order(self) -> None:
        transport = OrderTransport()
        with tempfile.TemporaryDirectory() as tmp:
            coordinator = self._coordinator(tmp, transport)
            with patch.dict("os.environ", {}, clear=True):
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
            execution_config=load_order_execution_config("config/order_execution.example.json"),
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


if __name__ == "__main__":
    unittest.main()

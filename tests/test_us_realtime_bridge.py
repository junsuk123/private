from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.trading import us_realtime_bridge
from app.trading.us_realtime_bridge import _extract_price_book, fetch_overseas_volume_surge_symbols


class UsRealtimeBridgeTest(unittest.TestCase):
    def test_kis_get_uses_shared_cached_client(self) -> None:
        calls = []

        class FakeClient:
            def _get(self, path: str, tr_id: str, params: dict[str, str]) -> dict[str, object]:
                calls.append((path, tr_id, params))
                return {"rt_cd": "0", "output": {"ok": "1"}}

        original = us_realtime_bridge.build_kis_client
        try:
            us_realtime_bridge.build_kis_client = lambda *, enabled: FakeClient()
            data = us_realtime_bridge._kis_get("/path", "TRID", {"A": "B"})
        finally:
            us_realtime_bridge.build_kis_client = original

        self.assertEqual(data["output"], {"ok": "1"})
        self.assertEqual(calls, [("/path", "TRID", {"A": "B"})])

    def test_volume_surge_uses_official_required_params(self) -> None:
        calls = []

        def fake_get(path: str, tr_id: str, params: dict[str, str]) -> dict[str, object]:
            calls.append((path, tr_id, params))
            return {"rt_cd": "0", "output2": [{"symb": "AAPL"}]}

        original = us_realtime_bridge._kis_get
        try:
            us_realtime_bridge._kis_get = fake_get
            result = fetch_overseas_volume_surge_symbols(exchanges=("NAS",), max_symbols=5)
        finally:
            us_realtime_bridge._kis_get = original

        self.assertEqual(result["symbols"], ("AAPL",))
        self.assertEqual(calls[0][0], "/uapi/overseas-stock/v1/ranking/volume-surge")
        self.assertEqual(calls[0][1], "HHDFS76270000")
        self.assertEqual(calls[0][2], {"AUTH": "", "EXCD": "NAS", "MINX": "0", "VOL_RANG": "0", "KEYB": ""})

    def test_extract_price_book_rejects_zero_bid_ask(self) -> None:
        payload = {
            "output": {
                "last": "10.83",
                "pbid1": "0",
                "pask1": "0",
                "vbid1": "1",
                "vask1": "1",
            }
        }

        with self.assertRaisesRegex(RuntimeError, "INVALID_BID_ASK_FIELDS"):
            _extract_price_book(payload)


if __name__ == "__main__":
    unittest.main()

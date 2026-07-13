from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.trading.domestic_realtime_bridge import fetch_domestic_ranking_symbols


class DomesticRealtimeBridgeTest(unittest.TestCase):
    def test_fetch_domestic_ranking_symbols_combines_official_ranking_outputs(self) -> None:
        calls: list[tuple[str, str]] = []

        def fake_get(path: str, tr_id: str, params: dict[str, str]) -> dict[str, object]:
            calls.append((path, tr_id))
            if tr_id == "FHPST01710000":
                return {
                    "output": [
                        {"mksc_shrn_iscd": "005930"},
                        {"mksc_shrn_iscd": "Q530036"},
                    ]
                }
            if tr_id == "FHPST01700000":
                return {"output1": [{"stck_shrn_iscd": "000660"}, {"stck_shrn_iscd": "005930"}]}
            return {"output2": [{"stck_shrn_iscd": "035420"}]}

        with patch("app.trading.domestic_realtime_bridge._kis_get", side_effect=fake_get):
            result = fetch_domestic_ranking_symbols(
                sources=("volume_rank", "fluctuation", "volume_power"),
                max_symbols=10,
            )

        self.assertEqual(
            calls,
            [
                ("/uapi/domestic-stock/v1/quotations/volume-rank", "FHPST01710000"),
                ("/uapi/domestic-stock/v1/ranking/fluctuation", "FHPST01700000"),
                ("/uapi/domestic-stock/v1/ranking/volume-power", "FHPST01680000"),
            ],
        )
        self.assertEqual(result["symbols"], ("005930", "000660", "035420"))
        self.assertTrue(result["ok"])


if __name__ == "__main__":
    unittest.main()

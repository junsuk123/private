from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.execution.kis_auth import issue_websocket_approval_key, run_kis_health_check, validate_live_secret_file
from app.execution.kis_errors import KisModeMismatchError
from app.execution.kis_real import KisDevelopersApiClient
from app.execution.kis_types import KisMode
from app.execution.kis_auth import validate_kis_mode


class RecordingKisTransport:
    def __init__(self) -> None:
        self.calls = []

    def request(self, method, url, headers, body=None, params=None, timeout=10.0):
        self.calls.append({"method": method, "url": url, "headers": dict(headers), "body": dict(body or {})})
        if url.endswith("/oauth2/tokenP"):
            return {"access_token": "token", "expires_in": 86400}
        if url.endswith("/oauth2/Approval"):
            return {"approval_key": "approval"}
        if url.endswith("/uapi/domestic-stock/v1/trading/inquire-balance"):
            return {"rt_cd": "0", "output1": [], "output2": [{"dnca_tot_amt": "1000000"}]}
        raise AssertionError(f"unexpected request: {url}")


class KisAuthAndModeTest(unittest.TestCase):
    def test_live_secret_validation_requires_only_used_live_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kis_api_keys.env"
            path.write_text(
                "\n".join(
                    [
                        "KIS_APP_KEY=app",
                        "KIS_APP_SECRET=secret",
                        "KIS_ACCOUNT_NO=12345678",
                        "KIS_ACCOUNT_PRODUCT_CODE=01",
                    ]
                ),
                encoding="utf-8",
            )

            result = validate_live_secret_file(path)

        self.assertTrue(result["file_exists"])
        self.assertTrue(result["KIS_APP_KEY"])
        self.assertTrue(result["KIS_APP_SECRET"])
        self.assertTrue(result["KIS_ACCOUNT_NO"])
        self.assertTrue(result["KIS_ACCOUNT_PRODUCT_CODE"])

    def test_mode_mismatch_is_blocked(self) -> None:
        with self.assertRaises(KisModeMismatchError):
            validate_kis_mode(KisMode(paper=True, live_enabled=False, base_url="https://openapi.koreainvestment.com:9443"))

    def test_websocket_approval_key_uses_redactable_contract(self) -> None:
        transport = RecordingKisTransport()
        with tempfile.TemporaryDirectory() as tmp:
            client = KisDevelopersApiClient(
                app_key="app",
                app_secret="secret",
                account_no="12345678-01",
                paper=True,
                enabled=True,
                transport=transport,
                token_cache_path=Path(tmp) / "token.json",
            )

            key = issue_websocket_approval_key(client)

        self.assertEqual(key, "approval")
        approval_call = next(call for call in transport.calls if call["url"].endswith("/oauth2/Approval"))
        self.assertEqual(approval_call["body"]["grant_type"], "client_credentials")

    def test_health_check_covers_token_account_and_websocket(self) -> None:
        transport = RecordingKisTransport()
        with tempfile.TemporaryDirectory() as tmp:
            client = KisDevelopersApiClient(
                app_key="app",
                app_secret="secret",
                account_no="12345678-01",
                paper=True,
                enabled=True,
                transport=transport,
                token_cache_path=Path(tmp) / "token.json",
            )

            health = run_kis_health_check(client)

        self.assertTrue(health.ok, health.failures)
        self.assertEqual(health.mode, "paper")
        self.assertTrue(health.gates["websocket_approval_key"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.audit.logger import AuditLogger, REDACTED


@dataclass(frozen=True)
class BrokerPayload:
    account_no: str
    nested: dict[str, object]


class AuditLoggerTest(unittest.TestCase):
    def test_audit_logger_redacts_sensitive_fields_recursively(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            logger = AuditLogger(path)
            logger.record(
                "broker_response",
                BrokerPayload(
                    account_no="12345678",
                    nested={
                        "access_token": "token-value",
                        "items": [{"CANO": "9999"}, {"safe": "visible"}],
                    },
                ),
            )

            event = json.loads(path.read_text(encoding="utf-8").strip())

        payload = event["payload"]
        self.assertEqual(payload["account_no"], REDACTED)
        self.assertEqual(payload["nested"]["access_token"], REDACTED)
        self.assertEqual(payload["nested"]["items"][0]["CANO"], REDACTED)
        self.assertEqual(payload["nested"]["items"][1]["safe"], "visible")


if __name__ == "__main__":
    unittest.main()

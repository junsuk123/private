from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app import run as run_module


class RunStartupTest(unittest.TestCase):
    def test_main_starts_server_without_blocking_on_startup_checks(self) -> None:
        with (
            patch.object(sys, "argv", ["run.py", "--port", "8019", "--strict-port"]),
            patch("app.run._run_startup_checks_in_background") as background_checks,
            patch("app.run._stop_existing_app_servers") as stop_existing,
            patch("app.run.run_startup_checks") as startup_checks,
            patch("app.run.uvicorn.run") as uvicorn_run,
        ):
            run_module.main()

        stop_existing.assert_called_once_with("127.0.0.1", 8019)
        background_checks.assert_called_once()
        startup_checks.assert_not_called()
        uvicorn_run.assert_called_once()

    def test_existing_server_command_detection_is_scoped_to_app_servers(self) -> None:
        workspace = Path(__file__).resolve().parents[1]

        self.assertTrue(
            run_module._is_existing_app_server_command(
                f"python {workspace / 'run.py'} --port 8000",
                workspace,
            )
        )
        self.assertTrue(
            run_module._is_existing_app_server_command(
                f"python -m uvicorn app.web:app --app-dir {workspace / 'src'}",
                workspace,
            )
        )
        self.assertFalse(run_module._is_existing_app_server_command("python unrelated.py", workspace))


if __name__ == "__main__":
    unittest.main()

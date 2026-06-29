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
            patch("app.run.run_startup_checks") as startup_checks,
            patch("app.run.uvicorn.run") as uvicorn_run,
        ):
            run_module.main()

        background_checks.assert_called_once()
        startup_checks.assert_not_called()
        uvicorn_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()

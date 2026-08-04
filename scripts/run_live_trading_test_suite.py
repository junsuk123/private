from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    fixture_symbol = os.getenv("LIVE_TEST_FIXTURE_SYMBOL", "900001").strip() or "900001"
    fixture = _write_realtime_fixture(fixture_symbol)
    db_path = Path(tempfile.gettempdir()) / "codex_realtime_market_data_check.sqlite3"
    commands = [
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        [sys.executable, "scripts/live_readiness_check.py", "--dry-run"],
        [sys.executable, "scripts/live_order_dry_run.py", "--symbols", fixture_symbol, "--no-submit"],
        [
            sys.executable,
            "scripts/check_realtime_market_data.py",
            "--symbols",
            fixture_symbol,
            "--fixture",
            str(fixture),
            "--db-path",
            str(db_path),
        ],
    ]
    try:
        for command in commands:
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                return completed.returncode
        return 0
    finally:
        _remove_quietly(fixture)
        _remove_quietly(db_path)
        _remove_quietly(Path(f"{db_path}-wal"))
        _remove_quietly(Path(f"{db_path}-shm"))


def _write_realtime_fixture(symbol: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".kis.txt")
    with handle:
        handle.write(f"0|H0STCNT0|001|{symbol}^093000^70000^100^BUY^suite-seq-1\n")
        handle.write(f"0|H0STASP0|001|{symbol}^093000^70100^70000^1000^1200\n")
    return Path(handle.name)


def _remove_quietly(path: Path) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        return


if __name__ == "__main__":
    raise SystemExit(main())

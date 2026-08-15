#!/usr/bin/env bash
# Queue one UTF-8 WORK_ANALYSIS_REPORT for the local Codex automation worker.
# Python owns parsing, duplicate protection and retry so manual delivery and the
# Windows scheduled worker use exactly the same implementation.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [[ -x "$SCRIPT_DIR/../.venv/Scripts/python.exe" ]]; then
  PYTHON_BIN="$SCRIPT_DIR/../.venv/Scripts/python.exe"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  PYTHON_BIN=python3
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/work_report_receiver.py" "$@"

#!/usr/bin/env bash
# Optional lightweight launcher for a Raspberry Pi MONITOR / execution-guard node.
#
# The Pi is NOT a primary NPU/ontology/training node. This launcher loads the
# lightweight profile (no Intel NPU / no local LLM / small universe / low refresh) and
# starts the server for a monitoring dashboard. Live order submission stays OFF unless
# you deliberately arm it (scripts/arm_live_trading.py) — which is the live-trading
# safety switch, NOT "ARM CPU support".
#
# Usage:  ./scripts/run_raspberrypi.sh [--port 8010]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROFILE="config/runtime_profiles/raspberrypi.env"
if [[ -f "$PROFILE" ]]; then
  echo "[run_raspberrypi] loading profile: $PROFILE"
  set -a
  # shellcheck disable=SC1090
  . "$PROFILE"
  set +a
else
  echo "[run_raspberrypi] WARNING: $PROFILE not found; running with process defaults" >&2
fi

export PYTHONPATH="${PYTHONPATH:-src}"
PORT="${APP_PORT:-8010}"
if [[ "${1:-}" == "--port" && -n "${2:-}" ]]; then
  PORT="$2"
fi

PYTHON_BIN="python3"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN="python"

echo "[run_raspberrypi] starting monitor server on port ${PORT} (live submit: ${LIVE_ORDER_SUBMIT_ENABLED:-false})"
exec "$PYTHON_BIN" run.py --skip-startup-checks --port "$PORT" --strict-port

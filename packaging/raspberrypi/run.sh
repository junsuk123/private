#!/usr/bin/env bash
#
# Raspberry Pi launcher (headless, CPU-only) for the investment system.
#
# The Linux counterpart of run.ps1: it applies the NPU-free CPU defaults, reuses
# the data already on disk under data/, and starts the FastAPI/uvicorn server.
# No managed browser is opened (a Pi is typically headless) — point any device
# on the LAN at http://<pi-ip>:<port>/account.
#
# Usage:
#   bash packaging/raspberrypi/run.sh                 # foreground
#   APP_PORT=9000 bash packaging/raspberrypi/run.sh   # override a default
#   bash packaging/raspberrypi/run.sh --port 9000     # same, via flag
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-pi}"

cd "${REPO_ROOT}"

# ---- Load user overrides (pi.env), then apply CPU-only defaults --------------
if [ -f "${SCRIPT_DIR}/pi.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "${SCRIPT_DIR}/pi.env"
  set +a
fi

# setdefault: only set if not already provided by pi.env or the shell.
setdefault() { eval ": \${$1:=\$2}"; export "$1"; }

setdefault PYTHONPATH "src"
setdefault APP_ENV "local"
setdefault APP_HOST "0.0.0.0"
setdefault APP_PORT "8010"
setdefault DATA_ENV "realtime"
setdefault DATA_ROOT "data"
setdefault REALTIME_STORE_ROOT "data/store"

# NPU-free: force deterministic CPU everywhere.
setdefault ONTOLOGY_ACCELERATOR "CPU"
setdefault REALTIME_LATENCY_PROFILE "balanced"
setdefault ONTOLOGY_NPU_ENABLED "false"
setdefault ONTOLOGY_FILTER1_NATIVE "auto"
unset OPENVINO_DEVICE OPENVINO_HINT_PERFORMANCE_MODE OPENVINO_ENABLE_CPU_PINNING 2>/dev/null || true

# News/event sentiment: local LLM (Ollama) with automatic keyword fallback.
# The app loads the shared config/local_llm.env and probes Ollama at startup:
# if it is reachable the LLM sentiment path turns on, otherwise it falls back to
# deterministic keyword rules — no torch/transformers needed (HTTP only).
# EVENT_CLASSIFIER_PROVIDER stays "keyword" (a separate CPU-only scorer; the LLM
# news path is layered on top via the LLM_EVENT_* variables).
# To force pure keyword (no LLM), set LLM_EVENT_CLASSIFIER_ENABLED=false in pi.env.
setdefault LLM_EVENT_PROVIDER "local"
setdefault EVENT_CLASSIFIER_PROVIDER "keyword"

# Safe trading defaults (read-only). Override in pi.env to go live.
setdefault TRADING_MODE "read_only"
setdefault LIVE_TRADING_ENABLED "false"
setdefault KIS_LIVE_ENABLED "false"
setdefault KIS_PAPER_TRADING "true"
setdefault LIVE_ORDER_SUBMIT_ENABLED "false"
setdefault AUTO_START_REALTIME_TRADING "false"
setdefault AUTO_START_LIVE_TRAINING "true"
setdefault LIVE_TRAINING_INTERVAL_SECONDS "120"
setdefault LIVE_REFRESH_SECONDS "15"
setdefault RESEARCH_RETENTION_DAYS "30"

# ---- Parse minimal CLI flags -------------------------------------------------
EXTRA_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --port)  APP_PORT="$2"; shift 2 ;;
    --host)  APP_HOST="$2"; shift 2 ;;
    *)       EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# ---- Select interpreter ------------------------------------------------------
if [ -x "${VENV_DIR}/bin/python" ]; then
  PY="${VENV_DIR}/bin/python"
else
  echo "[warn] ${VENV_DIR} not found; run bootstrap.sh first. Falling back to system python3." >&2
  PY="python3"
fi

mkdir -p "${REPO_ROOT}/logs"

echo "Starting investment system (CPU-only, NPU disabled)"
echo "  interpreter : ${PY}"
echo "  data root   : ${DATA_ROOT}  (store: ${REALTIME_STORE_ROOT})"
echo "  web UI      : http://${APP_HOST}:${APP_PORT}/account"
echo "  trading     : mode=${TRADING_MODE} live=${LIVE_TRADING_ENABLED}"
echo

exec "${PY}" "${REPO_ROOT}/run.py" \
  --skip-startup-checks \
  --host "${APP_HOST}" \
  --port "${APP_PORT}" \
  --strict-port \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

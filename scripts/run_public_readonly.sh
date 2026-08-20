#!/usr/bin/env bash
#
# A second OBAITS instance intended to sit behind Tailscale Funnel: read-only,
# token-guarded, and unable to submit an order. Linux counterpart of
# scripts/run_public_readonly.ps1.
#
# The live instance (run.py --port 8010, started by run.ps1/run.sh) forces
# TRADING_MODE=live_trading, LIVE_ORDER_SUBMIT_ENABLED=true and
# REQUIRE_MANUAL_ARMING=false so they cannot be overridden. An instance that is
# safe to publish therefore cannot be a variant of it; it has to be its own
# process with its own posture, which is what this script starts.
#
# Four decisions here exist because this port is published to the internet:
#
# 1. NO ORDER PATH. TRADING_MODE=read_only, LIVE_ORDER_SUBMIT_ENABLED=false,
#    REQUIRE_MANUAL_ARMING=true, and every AUTO_START_* worker off. The 22
#    state-changing POST endpoints still EXIST -- this is not a hardened build
#    -- but the ones that reach a broker have nothing live behind them.
#
# 2. BINDS THE TAILNET ADDRESS, NOT LOOPBACK. This is the part that is easy to
#    get wrong and expensive to get wrong. AccessGuardMiddleware decides whether
#    to demand a token from request.client.host alone; it does not read
#    X-Forwarded-For. Funnel proxies inbound requests, so if this server
#    listened on 127.0.0.1 and Funnel pointed there, every request from the
#    public internet would arrive looking like a loopback client and the guard
#    would wave it through -- a public, unauthenticated dashboard over a live
#    brokerage account. Pointing Funnel at the tailnet address instead makes
#    request.client.host non-loopback, so the token is enforced on exactly the
#    traffic that arrives from outside. Do not "simplify" this to 127.0.0.1.
#
# 3. --keep-existing-servers IS REQUIRED, NOT OPTIONAL. run.py's
#    _stop_existing_app_servers matches other instances by command line and
#    workspace path, NOT by port: any python process in this workspace whose
#    command mentions run.py or app.web:app is killed on startup. Without this
#    flag, launching the published instance terminates the live trading engine.
#    Choosing a port outside the launcher's sweep range is not sufficient.
#
# 4. PORT OUTSIDE 8000-8050. The live launcher stops every run.py listener in
#    that range when it starts, so a second instance inside it would be killed
#    by the next ordinary live launch.
#
# The token is printed once. Pass --token to keep a published link valid across
# a restart; otherwise a restart issues a new one and the old URL stops working.
#
# Usage:
#   ./scripts/run_public_readonly.sh                 # reuse the stored token
#   ./scripts/run_public_readonly.sh --show-token    # print it without starting
#   ./scripts/run_public_readonly.sh --rotate-token  # revoke and reissue
#   ./scripts/run_public_readonly.sh --token <>=16 chars>  # adopt a specific one
#   ./scripts/run_public_readonly.sh --no-save       # ephemeral, nothing written

set -euo pipefail

PORT=8110
TOKEN=""
ROTATE=0
SAVE=1
SHOW_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)         PORT="${2:?--port needs a value}"; shift 2 ;;
    --token)        TOKEN="${2:?--token needs a value}"; shift 2 ;;
    --rotate-token) ROTATE=1; shift ;;
    --no-save)      SAVE=0; shift ;;
    --show-token)   SHOW_ONLY=1; shift ;;
    -h|--help) sed -n '2,60p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv-linux/bin/python"
[[ -x "$PYTHON" ]] || { echo "No virtual environment at $PYTHON. Run setup first." >&2; exit 1; }

command -v tailscale >/dev/null 2>&1 \
  || { echo "tailscale is not installed; this script exists to serve a tailnet/Funnel address." >&2; exit 1; }

# --- The address Funnel must point at ----------------------------------------
BIND_ADDRESS="$(tailscale ip -4 2>/dev/null | grep -oE '^100\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$' | head -1 || true)"
[[ -n "$BIND_ADDRESS" ]] \
  || { echo "Tailscale is not up (no tailnet IPv4 address). Start it, then re-run." >&2; exit 1; }

DNS_NAME="$(tailscale status --json 2>/dev/null \
  | python3 -c 'import json,sys; print((json.load(sys.stdin).get("Self") or {}).get("DNSName","").rstrip("."))' 2>/dev/null || true)"
[[ -n "$DNS_NAME" ]] || DNS_NAME="$BIND_ADDRESS"

# --- Token --------------------------------------------------------------------
# Persisted, because the published URL contains it. A token regenerated on every
# restart silently invalidates the link already handed out -- and the restart
# that does so is often not deliberate (a live relaunch, a reboot), so the first
# sign would be someone reporting a 401. The file lives under config/secrets/,
# which .gitignore excludes wholesale, and is written 0600.
TOKEN_FILE="${OBAITS_PUBLIC_TOKEN_FILE:-$ROOT/config/secrets/public_readonly_token}"

generate_token() { python3 -c 'import secrets; print(secrets.token_urlsafe(24))'; }

save_token() {
  mkdir -p "$(dirname "$TOKEN_FILE")"
  ( umask 077; printf '%s\n' "$1" > "$TOKEN_FILE" )
  chmod 600 "$TOKEN_FILE" 2>/dev/null || true
}

read_stored_token() {
  [[ -f "$TOKEN_FILE" ]] || return 1
  local stored
  stored="$(tr -d '[:space:]' < "$TOKEN_FILE" 2>/dev/null || true)"
  [[ ${#stored} -ge 16 ]] || return 1
  printf '%s' "$stored"
}

# Precedence: explicit flag, then environment, then the stored token, then a new
# one. --rotate-token jumps the queue so a leaked link can be revoked.
if [[ $ROTATE -eq 1 ]]; then
  TOKEN="$(generate_token)"; TOKEN_SOURCE="rotated (previous link is now dead)"
elif [[ -n "$TOKEN" ]]; then
  TOKEN_SOURCE="--token"
elif [[ -n "${APP_PUBLIC_READONLY_TOKEN:-}" ]]; then
  TOKEN="$APP_PUBLIC_READONLY_TOKEN"; TOKEN_SOURCE="APP_PUBLIC_READONLY_TOKEN"
elif TOKEN="$(read_stored_token)"; then
  TOKEN_SOURCE="stored ($TOKEN_FILE)"
else
  TOKEN="$(generate_token)"; TOKEN_SOURCE="generated (first run)"
fi

if [[ ${#TOKEN} -lt 16 ]]; then
  echo "Token must be at least 16 characters (the server enforces this too)." >&2
  exit 1
fi

if [[ $SAVE -eq 1 ]]; then
  save_token "$TOKEN"
else
  TOKEN_SOURCE="$TOKEN_SOURCE, not saved"
fi

if [[ $SHOW_ONLY -eq 1 ]]; then
  cat <<SHOW
token  : $TOKEN
source : $TOKEN_SOURCE
file   : $TOKEN_FILE
tailnet: http://${DNS_NAME}:${PORT}/account?token=${TOKEN}
public : https://${DNS_NAME}/account?token=${TOKEN}   (only if Funnel is on)
SHOW
  exit 0
fi

# --- Posture -----------------------------------------------------------------
# Exported unconditionally: every one of these is a value this instance must
# have regardless of what is already in the environment.
export APP_ENV=local
export APP_HOST="$BIND_ADDRESS"
export APP_PORT="$PORT"
export APP_ACCESS_TOKEN="$TOKEN"
# Set, not appended: an inherited PYTHONPATH (ROS 2's, for one) breaks imports.
export PYTHONPATH=src

# No order can be constructed, let alone submitted.
export TRADING_MODE=read_only
export LIVE_TRADING_ENABLED=false
export LIVE_ORDER_SUBMIT_ENABLED=false
export KIS_PAPER_TRADING=false
export REQUIRE_MANUAL_ARMING=true
export REALTIME_BUY_ENABLED=false
# Defence in depth: ExecutionGuard blocks every order while this is on, so even
# if some path re-enabled submission the guard still refuses. Held in place by
# TRADING_POLICY_KEYS -- the secrets file cannot override an explicit value.
export KILL_SWITCH_ENABLED=true

# Account reads stay on: a dashboard with no balances is not the thing being
# published. This is the deliberate trade -- holdings and cash ARE visible to
# anyone holding the token.
export KIS_LIVE_ENABLED=true
export DATA_ENV=realtime

# Nothing that writes. The live instance owns the stores; running the same
# writers twice against the same SQLite files is a corruption risk taken for no
# benefit.
export AUTO_START_REALTIME_TRADING=false
export AUTO_START_LIVE_WORKER=false
export AUTO_START_LIVE_TRAINING=false
export AUTO_START_WEEKEND_BRIEF=false
export AUTO_START_INVESTOR_FLOW_REFRESH=false
export LIVE_SIGNAL_MODEL_INFERENCE_ENABLED=false
export LLM_EVENT_CLASSIFIER_ENABLED=false

cat <<INFO

READ-ONLY INSTANCE
  bind        : ${BIND_ADDRESS}:${PORT}  (tailnet address, so the token is enforced)
  trading     : read_only, order submission disabled, workers off
  account data: VISIBLE to anyone holding the token below

token       : ${TOKEN_SOURCE}

On the tailnet:
  http://${DNS_NAME}:${PORT}/account?token=${TOKEN}

To publish it on the public internet, as root (Funnel needs it):
  sudo tailscale funnel --bg --https=443 http://${BIND_ADDRESS}:${PORT}
  # then: https://${DNS_NAME}/account?token=${TOKEN}
  # point it at ${BIND_ADDRESS}, NEVER 127.0.0.1 - see this script's header.
  # off again: sudo tailscale funnel --https=443 off

INFO

exec "$PYTHON" run.py \
  --skip-startup-checks \
  --keep-existing-servers \
  --host "$BIND_ADDRESS" \
  --port "$PORT" \
  --strict-port

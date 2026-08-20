#!/usr/bin/env bash
#
# Publish the LIVE dashboard read-only on the tailnet address, for Tailscale Funnel.
#
# This replaces the second application instance that used to serve this port. One
# process owns the workers and the SQLite stores; this only relays its pages, so
# the published site cannot drift from the local one -- it is the same page.
#
# The security argument lives in src/app/public_proxy.py and is worth reading
# before changing anything here: this proxy reaches the upstream over loopback,
# so the upstream's own access guard does not challenge it, and the upstream can
# place real orders. The token check and the read-only allowlist in that module
# are the entire boundary.
#
# Usage:
#   ./scripts/run_public_proxy.sh                # reuse the stored token
#   ./scripts/run_public_proxy.sh --show-token   # print the link, start nothing
#   ./scripts/run_public_proxy.sh --rotate-token # revoke and reissue
#   ./scripts/run_public_proxy.sh --no-auth      # NO token: every read is world-readable
#   ./scripts/run_public_proxy.sh --port 8110 --upstream http://127.0.0.1:8010

set -euo pipefail

PORT=8110
UPSTREAM="http://127.0.0.1:8010"
TOKEN=""
ROTATE=0
SAVE=1
SHOW_ONLY=0
ANONYMOUS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)         PORT="${2:?--port needs a value}"; shift 2 ;;
    --upstream)     UPSTREAM="${2:?--upstream needs a value}"; shift 2 ;;
    --token)        TOKEN="${2:?--token needs a value}"; shift 2 ;;
    --rotate-token) ROTATE=1; shift ;;
    --no-save)      SAVE=0; shift ;;
    --show-token)   SHOW_ONLY=1; shift ;;
    --no-auth)      ANONYMOUS=1; shift ;;
    -h|--help)      sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv-linux/bin/python"
[[ -x "$PYTHON" ]] || { echo "No virtual environment at $PYTHON." >&2; exit 1; }

command -v tailscale >/dev/null 2>&1 \
  || { echo "tailscale is not installed; this script exists to serve a tailnet/Funnel address." >&2; exit 1; }

BIND_ADDRESS="$(tailscale ip -4 2>/dev/null | grep -oE '^100\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$' | head -1 || true)"
[[ -n "$BIND_ADDRESS" ]] \
  || { echo "Tailscale is not up (no tailnet IPv4 address)." >&2; exit 1; }

DNS_NAME="$(tailscale status --json 2>/dev/null \
  | python3 -c 'import json,sys; print((json.load(sys.stdin).get("Self") or {}).get("DNSName","").rstrip("."))' 2>/dev/null || true)"
[[ -n "$DNS_NAME" ]] || DNS_NAME="$BIND_ADDRESS"

# --- Token --------------------------------------------------------------------
# Shared with the previous implementation on purpose: the published URL should
# survive the switch from a second instance to a proxy.
TOKEN_FILE="${OBAITS_PUBLIC_TOKEN_FILE:-$ROOT/config/secrets/public_readonly_token}"

generate_token() { python3 -c 'import secrets; print(secrets.token_urlsafe(24))'; }
save_token() {
  mkdir -p "$(dirname "$TOKEN_FILE")"
  ( umask 077; printf '%s\n' "$1" > "$TOKEN_FILE" )
  chmod 600 "$TOKEN_FILE" 2>/dev/null || true
}
read_stored_token() {
  [[ -f "$TOKEN_FILE" ]] || return 1
  local stored; stored="$(tr -d '[:space:]' < "$TOKEN_FILE" 2>/dev/null || true)"
  [[ ${#stored} -ge 16 ]] || return 1
  printf '%s' "$stored"
}

if [[ $ROTATE -eq 1 ]]; then
  TOKEN="$(generate_token)"; SOURCE="rotated (previous link is now dead)"
elif [[ -n "$TOKEN" ]]; then
  SOURCE="--token"
elif [[ -n "${APP_PUBLIC_READONLY_TOKEN:-}" ]]; then
  TOKEN="$APP_PUBLIC_READONLY_TOKEN"; SOURCE="APP_PUBLIC_READONLY_TOKEN"
elif TOKEN="$(read_stored_token)"; then
  SOURCE="stored ($TOKEN_FILE)"
else
  TOKEN="$(generate_token)"; SOURCE="generated (first run)"
fi
[[ ${#TOKEN} -ge 16 ]] || { echo "Token must be at least 16 characters." >&2; exit 1; }
[[ $SAVE -eq 1 ]] && save_token "$TOKEN" || SOURCE="$SOURCE, not saved"

if [[ $SHOW_ONLY -eq 1 ]]; then
  cat <<SHOW
token  : $TOKEN
source : $SOURCE
file   : $TOKEN_FILE
tailnet: http://${DNS_NAME}:${PORT}/account?token=${TOKEN}
public : https://${DNS_NAME}/account?token=${TOKEN}   (only if Funnel is on)
SHOW
  exit 0
fi

export PUBLIC_PROXY_TOKEN="$TOKEN"
export PUBLIC_PROXY_UPSTREAM="$UPSTREAM"
if [[ $ANONYMOUS -eq 1 ]]; then
  export PUBLIC_PROXY_ALLOW_ANONYMOUS=true
  ACCESS_LINE="ANONYMOUS -- no token required, every published read is world-readable"
else
  ACCESS_LINE="token required (${SOURCE})"
fi
# Set, not appended: an inherited PYTHONPATH (ROS 2's, for one) breaks imports.
export PYTHONPATH=src

cat <<INFO

PUBLIC READ-ONLY PROXY
  bind        : ${BIND_ADDRESS}:${PORT}
  upstream    : ${UPSTREAM}   (the live server -- one set of workers, one owner of the stores)
  forwards    : GET/HEAD on a read allowlist only; every mutating route is refused here
  access      : ${ACCESS_LINE}
  account data: balance, holdings, PnL and every /api read are exposed to whoever can reach this

On the tailnet:
  http://${DNS_NAME}:${PORT}/account?token=${TOKEN}

To publish it on the public internet, as root (Funnel needs it):
  sudo tailscale funnel --bg --https=443 http://${BIND_ADDRESS}:${PORT}
  # then: https://${DNS_NAME}/account?token=${TOKEN}
  # off again: sudo tailscale funnel --https=443 off

INFO

exec "$PYTHON" -m uvicorn app.public_proxy:app \
  --host "$BIND_ADDRESS" --port "$PORT" --app-dir src --access-log --no-server-header

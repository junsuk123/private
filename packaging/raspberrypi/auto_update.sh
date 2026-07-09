#!/usr/bin/env bash
#
# Repo auto-updater for the Raspberry Pi deployment.
#
# Polls origin/<branch> and, when it has advanced past the local checkout,
# mirrors the checkout to it and restarts the app service so the new code loads.
# Intended to run every ~2 minutes via repo-autoupdate.timer (see that unit).
#
# Design:
# - `git reset --hard origin/<branch>` mirrors origin EXACTLY (the established
#   Pi sync method). Runtime-modified TRACKED files (e.g. *.latest.json, the
#   research cursor) are reverted and regenerate at runtime; UNTRACKED artifacts,
#   secrets, and pi.env (gitignored) are preserved.
# - Trigger is commit-level (HEAD vs origin/<branch>), so the app writing to
#   tracked files between ticks does NOT trigger an update — only real pushes do.
# - Restart is immediate on any update (operator choice). A flock guard prevents
#   overlapping timer ticks from colliding during a long fetch/restart.
#
# Env overrides (optional, e.g. from pi.env or the unit): AUTOUPDATE_BRANCH,
# AUTOUPDATE_SERVICE, AUTOUPDATE_ENABLED (set to 0/false to pause).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Load pi.env if present so operators can pause/retarget without editing units.
if [ -f "${SCRIPT_DIR}/pi.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "${SCRIPT_DIR}/pi.env"
  set +a
fi

BRANCH="${AUTOUPDATE_BRANCH:-main}"
SERVICE="${AUTOUPDATE_SERVICE:-personal-investment.service}"
ENABLED="${AUTOUPDATE_ENABLED:-1}"
LOCK="/tmp/pi-repo-autoupdate.lock"

log() { echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] repo-autoupdate: $*"; }

case "$(printf '%s' "${ENABLED}" | tr '[:upper:]' '[:lower:]')" in
  0|false|no|off) log "disabled via AUTOUPDATE_ENABLED; skipping"; exit 0 ;;
esac

cd "${REPO_ROOT}" || { log "repo root not found: ${REPO_ROOT}"; exit 1; }

# Single-flight: skip if a previous run is still active (long fetch / restart).
exec 9>"${LOCK}" 2>/dev/null || exit 0
if ! flock -n 9; then
  log "previous run still holds the lock; skipping this tick"
  exit 0
fi

# Fetch quietly; a network hiccup must never crash the timer.
if ! git fetch --quiet origin "${BRANCH}" 2>/dev/null; then
  log "git fetch failed (network?); will retry next tick"
  exit 0
fi

LOCAL="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
REMOTE="$(git rev-parse "origin/${BRANCH}" 2>/dev/null || echo unknown)"

if [ "${REMOTE}" = "unknown" ]; then
  log "cannot resolve origin/${BRANCH}; skipping"
  exit 0
fi
if [ "${LOCAL}" = "${REMOTE}" ]; then
  exit 0  # up to date — nothing to do.
fi

log "update detected: ${LOCAL:0:9} -> ${REMOTE:0:9} on origin/${BRANCH}; applying"

# Ensure we are on the tracked branch, then mirror origin exactly.
git checkout -q "${BRANCH}" 2>/dev/null || true
if ! git reset --hard "origin/${BRANCH}"; then
  log "git reset --hard failed; leaving checkout unchanged"
  exit 1
fi
log "checkout now at $(git rev-parse --short HEAD)"

# Restart the app so the new code is loaded (operator chose: immediate).
if sudo -n systemctl restart "${SERVICE}" 2>/dev/null; then
  log "restarted ${SERVICE}"
else
  log "WARN: could not restart ${SERVICE} — install the NOPASSWD sudoers rule (install_autoupdate.sh). Code is updated; restart pending."
fi

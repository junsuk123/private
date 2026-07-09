#!/usr/bin/env bash
#
# Installer for the repo auto-updater (run ONCE, as root):
#
#     sudo bash packaging/raspberrypi/install_autoupdate.sh
#
# It:
#   1. Installs a narrowly-scoped NOPASSWD sudoers rule so the timer (running as
#      the app user) may restart ONLY personal-investment.service.
#   2. Copies repo-autoupdate.service / .timer into /etc/systemd/system/.
#   3. daemon-reload + enables --now the timer (starts polling immediately).
#
# Idempotent: safe to re-run after pulling a new version of the units.
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "Please run as root:  sudo bash $0" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

APP_USER="${AUTOUPDATE_USER:-doingobject}"
SERVICE="${AUTOUPDATE_SERVICE:-personal-investment.service}"
SYSTEMCTL="$(command -v systemctl || echo /usr/bin/systemctl)"

echo "==> app user     : ${APP_USER}"
echo "==> app service  : ${SERVICE}"
echo "==> systemctl at : ${SYSTEMCTL}"

# 1) sudoers rule (scoped to exactly one command), validated before install.
SUDOERS_TMP="$(mktemp)"
cat > "${SUDOERS_TMP}" <<EOF
# Installed by install_autoupdate.sh — lets the repo auto-updater timer restart
# ONLY the investment app service without a password. Scoped on purpose.
${APP_USER} ALL=(root) NOPASSWD: ${SYSTEMCTL} restart ${SERVICE}
EOF
if visudo -cf "${SUDOERS_TMP}" >/dev/null; then
  install -m 0440 "${SUDOERS_TMP}" /etc/sudoers.d/repo-autoupdate
  echo "==> installed /etc/sudoers.d/repo-autoupdate"
else
  echo "ERROR: generated sudoers file failed validation; not installing." >&2
  rm -f "${SUDOERS_TMP}"
  exit 1
fi
rm -f "${SUDOERS_TMP}"

# 2) systemd units.
install -m 0644 "${SCRIPT_DIR}/repo-autoupdate.service" /etc/systemd/system/repo-autoupdate.service
install -m 0644 "${SCRIPT_DIR}/repo-autoupdate.timer"   /etc/systemd/system/repo-autoupdate.timer
echo "==> installed systemd units"

# 3) enable + start the timer.
"${SYSTEMCTL}" daemon-reload
"${SYSTEMCTL}" enable --now repo-autoupdate.timer
echo "==> repo-autoupdate.timer enabled and started"
echo
echo "Done. Useful commands:"
echo "  systemctl list-timers repo-autoupdate.timer"
echo "  systemctl status repo-autoupdate.service"
echo "  journalctl -u repo-autoupdate.service -f      # watch update/restart activity"
echo "  # pause without uninstalling: set AUTOUPDATE_ENABLED=0 in packaging/raspberrypi/pi.env"

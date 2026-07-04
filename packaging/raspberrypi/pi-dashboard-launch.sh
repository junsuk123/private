#!/usr/bin/env bash
#
# Local LCD kiosk launcher for the Raspberry Pi.
# Shows the human-readable trade-reason board on the attached display.

set -euo pipefail

URL="${PI_DASHBOARD_URL:-http://127.0.0.1:8010/display}"
READY_URL="${PI_DASHBOARD_READY_URL:-http://127.0.0.1:8010/api/trade-explanations}"

sudo systemctl start personal-investment.service 2>/dev/null || true
for _ in $(seq 1 60); do
  curl -sf -m 2 "${READY_URL}" >/dev/null 2>&1 && break
  sleep 1
done

# Keep the small LCD awake (X11; harmless under Wayland).
xset s off 2>/dev/null || true
xset -dpms 2>/dev/null || true
xset s noblank 2>/dev/null || true

FLAGS="--kiosk --incognito --noerrdialogs --disable-infobars --disable-session-crashed-bubble --disable-features=TranslateUI --check-for-update-interval=31536000 --overscroll-history-navigation=0 --disable-pinch"
for browser in chromium-browser chromium; do
  if command -v "${browser}" >/dev/null 2>&1; then
    exec "${browser}" ${FLAGS} --app="${URL}"
  fi
done

exec xdg-open "${URL}"

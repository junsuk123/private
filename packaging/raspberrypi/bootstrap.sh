#!/usr/bin/env bash
#
# One-command Raspberry Pi bootstrap for the Personal Multi-Agent Ontology-Based
# Automated Stock Investment System (NPU-free / CPU-only profile).
#
# It performs, in order:
#   1. Install OS packages (python3, venv, build tools) via apt   [skippable]
#   2. Create an isolated Python virtual environment
#   3. Install the CPU-only Python dependencies + this project (editable)
#   4. Build the optional Rust `screening_core` native accelerator [if toolchain present]
#   5. Verify the CPU-only runtime end-to-end
#
# This never installs openvino / torch / transformers, and never touches the
# existing data under data/. The current Windows system is left untouched.
#
# Usage:
#   bash packaging/raspberrypi/bootstrap.sh                 # full install + build
#   bash packaging/raspberrypi/bootstrap.sh --no-apt        # skip system packages
#   bash packaging/raspberrypi/bootstrap.sh --with-rust     # force-install Rust toolchain, then build native core
#   bash packaging/raspberrypi/bootstrap.sh --run           # install/build, then launch the app
#
set -euo pipefail

# ---- Resolve paths -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-pi}"
PY="${VENV_DIR}/bin/python"

# ---- Flags -------------------------------------------------------------------
DO_APT=1
FORCE_RUST=0
DO_RUN=0
for arg in "$@"; do
  case "$arg" in
    --no-apt)    DO_APT=0 ;;
    --with-rust) FORCE_RUST=1 ;;
    --run)       DO_RUN=1 ;;
    -h|--help)
      grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }

cd "${REPO_ROOT}"

# ---- 0. Sanity: architecture note -------------------------------------------
ARCH="$(uname -m || true)"
log "Bootstrapping on ${ARCH:-unknown} in ${REPO_ROOT}"
case "${ARCH}" in
  aarch64|armv7l|armv6l) : ;;  # expected Raspberry Pi
  *) warn "Architecture '${ARCH}' is not a typical Raspberry Pi; continuing anyway." ;;
esac

# ---- 1. OS packages ----------------------------------------------------------
if [ "${DO_APT}" -eq 1 ]; then
  if command -v apt-get >/dev/null 2>&1; then
    SUDO=""
    [ "$(id -u)" -ne 0 ] && SUDO="sudo"
    log "Installing OS packages (python3, venv, build tools)"
    ${SUDO} apt-get update
    # libopenblas0: required at runtime by the piwheels numpy build used on
    # 32-bit (armhf) Raspberry Pi OS. Harmless on 64-bit (arm64) where the PyPI
    # numpy wheel bundles its own BLAS. Without it, `import numpy` fails with
    # "libopenblas.so.0: cannot open shared object file".
    ${SUDO} apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev python3-pip \
      build-essential pkg-config ca-certificates curl tzdata \
      libopenblas0
  else
    warn "apt-get not found; skipping OS package install. Ensure python3 + venv + a C compiler are present."
  fi
else
  log "Skipping OS package install (--no-apt)"
fi

# ---- 2. Virtual environment --------------------------------------------------
if [ ! -x "${PY}" ]; then
  log "Creating virtual environment at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
else
  log "Reusing existing virtual environment at ${VENV_DIR}"
fi

log "Upgrading pip / setuptools / wheel"
"${PY}" -m pip install --upgrade pip setuptools wheel

# ---- 3. Python dependencies + project (CPU-only, NPU-free) -------------------
log "Installing CPU-only dependencies (no openvino / torch / transformers)"
"${PY}" -m pip install -r "${SCRIPT_DIR}/requirements-pi.txt"

log "Installing project in editable mode (core dependencies only)"
"${PY}" -m pip install -e "${REPO_ROOT}"

# ---- 4. Optional native Rust accelerator ------------------------------------
NATIVE_DIR="${REPO_ROOT}/native/screening_core"
build_native() {
  log "Building optional Rust native screening core"
  "${PY}" -m pip install "maturin>=1.5,<2"
  # maturin develop installs into the active venv, which it reads from
  # VIRTUAL_ENV; set it explicitly so the build targets .venv-pi unambiguously.
  ( cd "${NATIVE_DIR}" && VIRTUAL_ENV="${VENV_DIR}" PATH="${VENV_DIR}/bin:${PATH}" "${VENV_DIR}/bin/maturin" develop --release )
}

if [ -d "${NATIVE_DIR}" ]; then
  if [ "${FORCE_RUST}" -eq 1 ] && ! command -v cargo >/dev/null 2>&1; then
    log "Installing Rust toolchain via rustup (--with-rust)"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck disable=SC1090
    . "${HOME}/.cargo/env"
  fi
  if command -v cargo >/dev/null 2>&1; then
    if build_native; then
      log "Native screening core built and installed."
    else
      warn "Native build failed; the pure-Python screening fallback will be used automatically."
    fi
  else
    warn "Rust/cargo not found; skipping native build. Pure-Python screening fallback will be used."
    warn "To build the native accelerator later, re-run with --with-rust."
  fi
else
  warn "native/screening_core not present; using pure-Python screening."
fi

# ---- 5. Verification ---------------------------------------------------------
log "Verifying CPU-only runtime"
PYTHONPATH="${REPO_ROOT}/src" \
ONTOLOGY_ACCELERATOR=CPU \
LLM_EVENT_CLASSIFIER_ENABLED=false \
"${PY}" "${SCRIPT_DIR}/verify_pi.py"

log "Bootstrap complete."
echo
echo "  Launch the app with:"
echo "      bash packaging/raspberrypi/run.sh"
echo
echo "  Or install it as a background service (auto-start on boot):"
echo "      see packaging/raspberrypi/personal-investment.service"
echo

if [ "${DO_RUN}" -eq 1 ]; then
  log "Launching app (--run)"
  exec bash "${SCRIPT_DIR}/run.sh"
fi

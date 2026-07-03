# Raspberry Pi Deployment (CPU-only, NPU-free)

This guide describes the **Raspberry Pi package** under
[`packaging/raspberrypi/`](../packaging/raspberrypi/). It runs the entire
system — realtime collection, ontology reasoning, indicators, strategy, risk,
principal protection, KIS broker adapter, short-horizon learning, paper/live
trading, and the web dashboard — on an ARM CPU, with **no NPU and no OpenVINO**.

The existing Windows system, its `run.ps1`/`run.bat` launchers, and the data
under `data/` are **left completely untouched**. The Pi package only *adds*
files; it changes nothing in the running system.

## Why this works without an NPU

The NPU was never a hard dependency. Every acceleration path in the codebase is
optional and already degrades to a deterministic CPU/NumPy implementation:

| Component | NPU/accelerated path | CPU fallback (used on Pi) |
| --- | --- | --- |
| Ontology accelerator selection ([`app/graph/runtime.py`](../src/app/graph/runtime.py)) | OpenVINO `NPU` device | `ONTOLOGY_ACCELERATOR=CPU` → deterministic `python-rules` |
| Signal inference ([`app/models/inference_backend.py`](../src/app/models/inference_backend.py)) | `OpenVinoNpuSignalModel` | `CpuSignalModel` (NumPy matmul) |
| NPU runtime manager ([`app/npu/runtime_manager.py`](../src/app/npu/runtime_manager.py)) | OpenVINO compiled linear graphs | `_NumpyLinearModel` |
| Realtime acceleration policy ([`app/realtime/acceleration.py`](../src/app/realtime/acceleration.py)) | requests NPU hints | reports CPU, trading logic unaffected |
| Candidate screening ([`app/native/screening.py`](../src/app/native/screening.py)) | Rust `screening_core` (pyo3) | pure-Python vectorized NumPy |
| Event classification | OpenVINO / torch / transformers LLM | deterministic keyword rules |

Trading logic, graph rules, and risk checks are pure Python and **always** run
on CPU — acceleration only ever touched numeric scoring. So dropping the NPU
changes throughput, not behavior or safety.

**Excluded on Pi:** `openvino`, `optimum-intel`, `torch`, `transformers`,
`accelerate`, `sentencepiece`. The Pi install pulls only the core dependencies,
all of which ship prebuilt `aarch64` wheels or are pure Python — so the core
install needs no compilation.

## Requirements

- Raspberry Pi 4 / 5 (or any `aarch64`/`armv7` board) with Raspberry Pi OS
  (Debian **Bookworm** recommended) or Ubuntu.
- Python **3.11+** (Bookworm ships 3.11).
- ~1 GB free disk for the venv; 2 GB+ RAM recommended.
- Network access to KIS / data sources if running live or realtime.

## One-command install + build

Copy the repository (including the accumulated `data/` directory) to the Pi,
then from the repo root run **one command**:

```bash
bash packaging/raspberrypi/bootstrap.sh
```

That single command:

1. Installs OS packages (`python3`, `python3-venv`, `python3-dev`,
   `build-essential`, …) via `apt`.
2. Creates an isolated virtualenv at `.venv-pi/` (separate from the Windows
   `.venv/`, so a shared/synced folder never clobbers either one).
3. Installs the CPU-only Python dependencies and the project itself
   (`pip install -e .`, core dependencies only — **no** NPU/LLM extras).
4. Builds the optional Rust `screening_core` accelerator **if** a Rust
   toolchain is present (non-fatal — the Python fallback is used otherwise).
5. Runs an end-to-end CPU verification and fails loudly if anything is wrong.

Equivalent via make:

```bash
make -C packaging/raspberrypi install
```

Useful variants:

```bash
bash packaging/raspberrypi/bootstrap.sh --no-apt     # system packages already installed
bash packaging/raspberrypi/bootstrap.sh --with-rust  # install Rust toolchain + build native core
bash packaging/raspberrypi/bootstrap.sh --run        # install/build, then launch immediately
```

## Running

```bash
bash packaging/raspberrypi/run.sh
```

The launcher is the Linux counterpart of `run.ps1`: it applies the NPU-free CPU
defaults, reuses the on-disk `data/` and `data/store/` produced by the main
system, and starts uvicorn. It is **headless** (no managed browser) and binds to
`0.0.0.0:8010` by default — open `http://<pi-ip>:8010/account` from any device
on the LAN.

Override defaults per run:

```bash
APP_PORT=9000 bash packaging/raspberrypi/run.sh
bash packaging/raspberrypi/run.sh --port 9000 --host 127.0.0.1
```

Or persist overrides by copying `pi.env.example` → `pi.env` (auto-sourced):

```bash
cp packaging/raspberrypi/pi.env.example packaging/raspberrypi/pi.env
# edit pi.env, then:
bash packaging/raspberrypi/run.sh
```

## Data: reuse what you already have

The Pi package **does not migrate or regenerate data**. `run.sh` keeps
`DATA_ROOT=data` and `REALTIME_STORE_ROOT=data/store`, the same layout the
Windows system uses, so:

- Copy the `data/` directory over as-is and the Pi reads/continues it directly.
- Model artifacts under `data/models/…` (e.g. `live_short_horizon`,
  `realtime_supervised`) are picked up unchanged; new artifacts append normally.
- Nothing in `data/` is deleted or rewritten by the install.

## Secrets and live trading

Secrets work exactly as on the main system. Copy your KIS keys to the ignored
file and complete the readiness check before enabling live orders:

```bash
cp config/secrets/kis_api_keys.env.example config/secrets/kis_api_keys.env
# fill in real values, then verify:
.venv-pi/bin/python scripts/check_kis_connection.py --account
.venv-pi/bin/python scripts/live_readiness_check.py
```

To go live, flip the trading flags in `pi.env`
(`TRADING_MODE=live_trading`, `LIVE_TRADING_ENABLED=true`,
`KIS_LIVE_ENABLED=true`, `KIS_PAPER_TRADING=false`,
`LIVE_ORDER_SUBMIT_ENABLED=true`, `AUTO_START_REALTIME_TRADING=true`). Defaults
ship **read-only** for safety.

## Run at boot (systemd)

```bash
# edit User= / WorkingDirectory= in the unit first
make -C packaging/raspberrypi service-install
# or manually:
sudo cp packaging/raspberrypi/personal-investment.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now personal-investment.service
journalctl -u personal-investment.service -f
```

## Verifying the CPU-only runtime

`bootstrap.sh` runs this automatically, but you can re-run it any time:

```bash
PYTHONPATH=src ONTOLOGY_ACCELERATOR=CPU .venv-pi/bin/python packaging/raspberrypi/verify_pi.py
```

Expected tail:

```text
VERIFICATION_OK — the system runs fully on CPU without NPU.
```

It asserts the ontology runtime is on CPU (`uses_npu=False`), CPU inference is
numerically correct, the screening fallback is available, the realtime policy
does not report NPU, and the FastAPI app imports and constructs.

## Optional native accelerator (Rust)

The `screening_core` Rust extension is a *speed* optimization for first-stage
candidate screening, not a requirement. Build it with:

```bash
bash packaging/raspberrypi/bootstrap.sh --no-apt --with-rust
# or, if cargo is already installed:
.venv-pi/bin/pip install "maturin>=1.5,<2"
( cd native/screening_core && ../../.venv-pi/bin/maturin develop --release )
```

Without it, `ONTOLOGY_FILTER1_NATIVE=auto` transparently uses the pure-Python
NumPy screening path.

## Performance notes

- The desktop NPU profile (`ONTOLOGY_NPU_BATCH_SIZE=4096`, hundreds of thousands
  of rows/s) does not apply. On Pi, keep universes smaller —
  `run.sh`/`pi.env` default to `ANALYSIS_MARKET_LIMIT=150`,
  `SIM_STREAMING_UNIVERSE_LIMIT=80`.
- Prefer `REALTIME_LATENCY_PROFILE=balanced` (the Pi default) over
  `low_latency`.
- The Rust native screening core gives the biggest single CPU speedup for wide
  candidate sets; build it if you screen large universes.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `python3 -m venv` fails | `sudo apt-get install python3-venv` (bootstrap does this) |
| `import numpy` → `libopenblas.so.0: cannot open shared object file` | 32-bit (armhf) Pi OS: `sudo apt-get install libopenblas0` (bootstrap now does this). The piwheels numpy build links the system OpenBLAS; 64-bit Pi OS bundles it in the wheel. |
| `numpy`/wheel builds from source | ensure Bookworm + pip ≥ 23; bootstrap upgrades pip first |
| Dashboard not reachable from LAN | confirm `APP_HOST=0.0.0.0` and the Pi's firewall allows the port |
| Native build errors | ignore — Python fallback is automatic; or re-run with `--with-rust` |
| Want to confirm CPU-only | run `verify_pi.py`; check `GET /api/ontology/runtime` shows `uses_npu:false` |

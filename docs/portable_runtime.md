# Portable runtime

The repository may be synchronized between Windows and Linux machines. Source,
configuration, and model artifacts are shared, but virtual environments are
not: Windows uses `.venv`, while Linux uses `.venv-linux`.

## Setup

Run `./setup.ps1` from either OS. On x64 PCs it installs OpenVINO automatically;
when `nvidia-smi` is visible it also installs the local-LLM/PyTorch extra. Use
`-BaseOnly` only when a minimal CPU installation is intentional. Explicit
`-WithNpu`, `-WithLocalLlm`, `-All`, and `-CudaWheels cu128` options remain
available.

## Placement and fallback

At process start the application probes OpenVINO CPU/GPU/NPU devices and
PyTorch CUDA/MPS devices. Each workload has its own compatibility ladder:

- deterministic order, risk, ontology-rule, and decision inference stays on CPU;
- OpenVINO graphs use NPU, then compatible OpenVINO GPU, then CPU;
- embedded event classification may use NPU, OpenVINO GPU, CUDA/MPS, or CPU;
- an unavailable or failed accelerator falls back without preventing startup.

Operator-provided `DEVICE_PLAN_*`, `OPENVINO_DEVICE`, `ONTOLOGY_ACCELERATOR`,
and `LLM_EVENT_DEVICE` values remain explicit overrides.

## Paths

`run.py`, `run.ps1`, and `run.bat` derive the project root from their own file
location. Python startup changes to that root before any service constructs a
relative `config/`, `data/`, or `logs/` path, so launching by absolute path from
another working directory still uses this codespace. `OBAITS_PROJECT_ROOT` is
published for integrations that need the resolved absolute root.

SQLite files that are written continuously cannot safely be opened or synced by
a cloud-drive client. In a Dropbox, Google Drive, OneDrive, or SynologyDrive
workspace, research and investor-flow runtime databases therefore use a
per-project directory under `%LOCALAPPDATA%/OBAITS` on Windows or
`$XDG_STATE_HOME/OBAITS` on Linux. This is the one deliberate exception to
codespace-relative paths: SQLite atomicity cannot be guaranteed inside a synced
tree. The location is derived automatically on every PC, so no code edit or
machine-specific configuration is required. Non-synced workspaces continue to
use `data/store`; `REALTIME_STORE_ROOT` remains an explicit override.

These small derived databases also use a single-file rollback journal in synced
trees. If SQLite reports definitive corruption, the original database and any
sidecars are moved to a timestamped `.corrupt.*` file before a clean schema is
created; account and order ledgers are never auto-reset by this mechanism.

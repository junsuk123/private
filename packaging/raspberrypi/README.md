# Raspberry Pi package (CPU-only, NPU-free)

Runs the **entire** system on a Raspberry Pi with no NPU and no OpenVINO. The
existing Windows system and the accumulated `data/` are left untouched — this
directory only *adds* files.

## One command: install dependencies + build

From the repository root:

```bash
bash packaging/raspberrypi/bootstrap.sh
```

Installs OS packages → creates `.venv-pi/` → installs CPU-only Python deps and
the project → builds the optional Rust accelerator (if a toolchain is present)
→ verifies the CPU runtime end-to-end.

## Run

```bash
bash packaging/raspberrypi/run.sh
```

Headless; open `http://<pi-ip>:8010/account` from any LAN device.

## Files

| File | Purpose |
| --- | --- |
| `bootstrap.sh` | One-command install + build + verify |
| `run.sh` | Headless CPU-only launcher (Linux counterpart of `run.ps1`) |
| `verify_pi.py` | End-to-end CPU-only runtime check |
| `requirements-pi.txt` | Core dependencies only (no openvino/torch/transformers) |
| `pi.env.example` | Copy to `pi.env` for persistent overrides (auto-sourced by `run.sh`) |
| `personal-investment.service` | systemd unit for auto-start on boot |
| `Makefile` | `make install` / `make run` / `make verify` / `make service-install` |

Full guide: [`docs/raspberry_pi_deployment.md`](../../docs/raspberry_pi_deployment.md).

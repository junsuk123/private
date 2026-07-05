# Raspberry Pi Package

Raspberry Pi에서 전체 시스템을 CPU-only로 실행하기 위한 패키지입니다. OpenVINO/NPU 없이 동작하며, 기본값은 read-only라서 설치 직후에는 실주문이 제출되지 않습니다.

## Install

저장소 루트에서:

```bash
bash packaging/raspberrypi/bootstrap.sh
```

수행 내용:

- OS package 설치
- `.venv-pi/` 생성
- Pi용 core dependency 설치
- 프로젝트 editable install
- 선택적 Rust `screening_core` 빌드
- CPU-only runtime verification 실행

## Run

```bash
bash packaging/raspberrypi/run.sh
```

기본 주소:

- LAN 브라우저: `http://<pi-ip>:8010/account`
- Pi 내부: `http://127.0.0.1:8010/account`

`run.sh`는 headless 런처입니다. Windows `run.ps1`처럼 브라우저를 관리하지 않고, 서버만 foreground로 실행합니다.

## LCD/Kiosk GUI

Attached LCD에서 trade-reason board를 띄울 때:

```bash
bash packaging/raspberrypi/pi-dashboard-launch.sh
```

동작:

- `personal-investment.service` 시작 시도
- `/api/trade-explanations` 준비 대기
- Chromium을 kiosk 모드로 `/display`에 연결
- X11 화면 절전 비활성화

온톨로지 화면을 LCD에 띄우려면:

```bash
PI_DASHBOARD_URL=http://127.0.0.1:8010/display/ontology \
PI_DASHBOARD_READY_URL=http://127.0.0.1:8010/api/ontology/graph \
bash packaging/raspberrypi/pi-dashboard-launch.sh
```

## Defaults

`run.sh`가 적용하는 Pi 기본값:

- `APP_HOST=0.0.0.0`
- `APP_PORT=8010`
- `DATA_ROOT=data`
- `REALTIME_STORE_ROOT=data/store`
- `ONTOLOGY_ACCELERATOR=CPU`
- `ONTOLOGY_NPU_ENABLED=false`
- `REALTIME_LATENCY_PROFILE=balanced`
- `TRADING_MODE=read_only`
- `LIVE_TRADING_ENABLED=false`
- `KIS_LIVE_ENABLED=false`
- `KIS_PAPER_TRADING=true`
- `LIVE_ORDER_SUBMIT_ENABLED=false`
- `AUTO_START_REALTIME_TRADING=false`
- `AUTO_START_LIVE_TRAINING=true`

지속 설정은 `packaging/raspberrypi/pi.env.example`을 `packaging/raspberrypi/pi.env`로 복사해서 수정합니다. KIS secrets는 `pi.env`가 아니라 `config/secrets/kis_api_keys.env`에 둡니다.

## Files

| File | Purpose |
| --- | --- |
| `bootstrap.sh` | one-command install, build, verify |
| `run.sh` | headless CPU-only launcher |
| `verify_pi.py` | CPU-only runtime verification |
| `requirements-pi.txt` | Pi core dependencies, no OpenVINO/torch/transformers |
| `pi.env.example` | persistent Pi runtime overrides |
| `pi-dashboard-launch.sh` | attached-LCD kiosk launcher for `/display` or `/display/ontology` |
| `manage_llm_opportunistic.py` | local LLM process manager for idle/off-market windows |
| `llm-opportunistic.service` / `.timer` | systemd units for opportunistic local LLM management |
| `personal-investment.service` | systemd auto-start unit |
| `Makefile` | `make install`, `make run`, `make verify`, `make service-install` |

Full guide: [../../docs/raspberry_pi_deployment.md](../../docs/raspberry_pi_deployment.md).

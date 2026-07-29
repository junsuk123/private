# Raspberry Pi Deployment

이 문서는 `packaging/raspberrypi/` 패키지로 Raspberry Pi에서 시스템을 실행하는 방법을 설명합니다. Pi 런타임은 CPU-only, NPU-free, headless 기본값을 사용합니다. 설치 직후에는 read-only라서 실주문이 제출되지 않습니다.

## 왜 NPU 없이 동작하는가

이 코드베이스에서 NPU/OpenVINO는 속도 최적화입니다. 주문 승인 권한이 아닙니다.

| Component | Desktop accelerated path | Pi fallback |
| --- | --- | --- |
| Ontology runtime | OpenVINO `NPU` | `ONTOLOGY_ACCELERATOR=CPU` |
| Ontology candidate scorer | OpenVINO linear scorer | CPU/NumPy scorer |
| Signal inference | OpenVINO backend | `CpuSignalModel` |
| Candidate screening | optional Rust/PyO3 core | pure Python/NumPy |
| Event/news sentiment | local OpenAI-compatible LLM when available | deterministic keyword fallback |

거래 제어 경로는 항상 CPU에서 동작합니다. `SharedLiveDecisionEngine`, `TradingCostEngine`, `PrincipalProtectionEngine`, `RiskManager`, idempotency, broker submission은 NPU에 의존하지 않습니다.

## Requirements

- Raspberry Pi 4/5 또는 aarch64/armv7 Linux board
- Raspberry Pi OS Bookworm 또는 Ubuntu 권장
- Python 3.11+
- 네트워크 연결
- live/realtime 운용 시 KIS 접근 가능 네트워크와 secrets

## Install

저장소 전체를 Pi로 복사합니다. 기존 `data/`를 함께 복사하면 학습/실시간 store/model artifact를 그대로 이어서 쓸 수 있습니다.

저장소 루트에서:

```bash
bash packaging/raspberrypi/bootstrap.sh
```

옵션:

```bash
bash packaging/raspberrypi/bootstrap.sh --no-apt
bash packaging/raspberrypi/bootstrap.sh --with-rust
bash packaging/raspberrypi/bootstrap.sh --run
```

Makefile 방식:

```bash
make -C packaging/raspberrypi install
make -C packaging/raspberrypi verify
make -C packaging/raspberrypi run
```

## Run

```bash
bash packaging/raspberrypi/run.sh
```

기본 접속 주소:

```text
http://<pi-ip>:8010/account
```

`run.sh`는 Windows `run.ps1`의 Linux/Pi counterpart이지만 브라우저를 직접 열지 않습니다. 기본 환경은 다음과 같습니다.

```text
APP_HOST=0.0.0.0
APP_PORT=8010
DATA_ENV=realtime
DATA_ROOT=data
REALTIME_STORE_ROOT=data/store
ONTOLOGY_ACCELERATOR=CPU
ONTOLOGY_NPU_ENABLED=false
REALTIME_LATENCY_PROFILE=balanced
TRADING_MODE=read_only
LIVE_TRADING_ENABLED=false
KIS_LIVE_ENABLED=false
KIS_PAPER_TRADING=true
LIVE_ORDER_SUBMIT_ENABLED=false
AUTO_START_REALTIME_TRADING=false
AUTO_START_LIVE_TRAINING=true
```

포트/호스트 override:

```bash
APP_PORT=9000 bash packaging/raspberrypi/run.sh
bash packaging/raspberrypi/run.sh --port 9000 --host 127.0.0.1
```

지속 override:

```bash
cp packaging/raspberrypi/pi.env.example packaging/raspberrypi/pi.env
nano packaging/raspberrypi/pi.env
bash packaging/raspberrypi/run.sh
```

## Web GUI on Pi

Pi 서버가 켜지면 같은 FastAPI GUI를 LAN에서 사용할 수 있습니다.

- `http://<pi-ip>:8010/account`: 계좌/자산/현금/보유종목/자동거래 판단 대시보드
- `http://<pi-ip>:8010/`: 연구/진단/수동 operation mode 화면
- `http://<pi-ip>:8010/display`: trade-reason board
- `http://<pi-ip>:8010/display/ontology`: 온톨로지 그래프 전체 화면

Pi는 기본적으로 read-only이므로 `/account`는 모니터링/진단용입니다. live 자동거래를 켜려면 아래 live 전환 단계를 별도로 수행해야 합니다.

## Attached LCD Kiosk

LCD가 연결된 Pi에서 키오스크 화면을 띄우려면:

```bash
bash packaging/raspberrypi/pi-dashboard-launch.sh
```

기본 화면은 `/display`입니다. 이 화면은 `/api/trade-explanations`를 polling해서 최근 판단 이유 카드를 보여줍니다.

온톨로지 화면을 LCD에 띄우려면:

```bash
PI_DASHBOARD_URL=http://127.0.0.1:8010/display/ontology \
PI_DASHBOARD_READY_URL=http://127.0.0.1:8010/api/ontology/graph \
bash packaging/raspberrypi/pi-dashboard-launch.sh
```

키오스크 런처는 Chromium을 찾으면 `--kiosk --incognito --app=<url>`로 실행하고, 없으면 `xdg-open`으로 fallback합니다.

## Data Reuse

Pi 패키지는 데이터를 migration하지 않습니다.

- `DATA_ROOT=data`
- `REALTIME_STORE_ROOT=data/store`
- model artifact: `data/models/<family>/`
- report: `data/reports/`
- audit/log: `logs/`

Windows에서 생성한 `data/`를 그대로 복사하면 Pi가 같은 SQLite store와 model artifact를 읽고 이어서 기록합니다.

## Secrets and Live Trading

KIS secrets는 Windows와 같은 위치를 사용합니다.

```bash
cp config/secrets/kis_api_keys.env.example config/secrets/kis_api_keys.env
nano config/secrets/kis_api_keys.env
.venv-pi/bin/python scripts/check_kis_connection.py --account
.venv-pi/bin/python scripts/live_readiness_check.py
```

실거래를 켜려면 `packaging/raspberrypi/pi.env`에서 명시적으로 바꿉니다.

```text
TRADING_MODE=live_trading
LIVE_TRADING_ENABLED=true
KIS_LIVE_ENABLED=true
KIS_PAPER_TRADING=false
LIVE_ORDER_SUBMIT_ENABLED=true
AUTO_START_REALTIME_TRADING=true
```

권장 절차:

1. read-only로 부팅해서 `/account`의 계좌/현금/보유종목이 맞는지 확인합니다.
2. `scripts/live_readiness_check.py`를 통과시킵니다.
3. `pi.env`에서 live flag를 켭니다.
4. 작은 universe와 긴 interval로 먼저 운용합니다.
5. `/api/realtime-trading/status`와 `/account`의 rejection reason을 확인합니다.

## Systemd

unit 파일의 `User=`와 `WorkingDirectory=`를 실제 Pi 경로에 맞춘 뒤:

```bash
sudo cp packaging/raspberrypi/personal-investment.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now personal-investment.service
journalctl -u personal-investment.service -f
```

Makefile 방식:

```bash
make -C packaging/raspberrypi service-install
```

## Optional Local LLM

Pi에서는 `torch`/`transformers`를 설치하지 않습니다. news/event sentiment는 OpenAI-compatible local server가 있을 때 HTTP로만 호출하고, 실패하면 keyword fallback을 사용합니다.

64-bit Pi OS에서 Ollama 예시:

```bash
# install Ollama via its official installer (see the Ollama project docs)
ollama pull qwen2.5:1.5b-instruct
cp config/local_llm.env.example config/local_llm.env
```

Pi 4/5에서는 작은 모델과 opportunistic 실행을 권장합니다. `pi.env.example`의 기본값은 market hours에는 LLM을 피하고, CPU load가 낮을 때만 사용하도록 잡혀 있습니다.

강제로 keyword-only로 두려면:

```text
LLM_EVENT_CLASSIFIER_ENABLED=false
```

## Verification

`bootstrap.sh`가 자동으로 실행하지만, 언제든 재검증할 수 있습니다.

```bash
PYTHONPATH=src ONTOLOGY_ACCELERATOR=CPU .venv-pi/bin/python packaging/raspberrypi/verify_pi.py
```

기대 결과:

```text
VERIFICATION_OK - the system runs fully on CPU without NPU.
```

## Troubleshooting

- `/account` 접속 불가: `journalctl -u personal-investment.service -f` 또는 foreground `run.sh` 로그를 확인합니다.
- LAN에서 접속 불가: `APP_HOST=0.0.0.0`, 방화벽, Pi IP를 확인합니다.
- KIS 계좌가 안 보임: `config/secrets/kis_api_keys.env`, token cache, `scripts/check_kis_connection.py --account`를 확인합니다.
- CPU가 높음: `ANALYSIS_MARKET_LIMIT`, `SIM_STRATEGY_CANDIDATES`, `LIVE_TRAINING_INTERVAL_SECONDS`, local LLM 설정을 낮춥니다.
- live 주문이 안 나감: `/api/realtime-trading/status`의 rejection reason, `/api/live-flags/status`, readiness report, safety gate 문서를 확인합니다.

관련 문서:

- [../README.md](../README.md)
- [README.md](README.md)
- [live_trading.md](live_trading.md) — 게이트, 운영 절차, 비상 정지, 런타임 프로파일
- [architecture.md](architecture.md) — 모듈 지도와 가속 경계

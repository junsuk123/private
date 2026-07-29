# Personal Investment Agent

KIS 실시간 데이터, 온톨로지 기반 근거 추론, 단기 학습 모델, 결정론적 리스크 게이트를 묶은 개인용 자동 투자 분석/운영 시스템입니다. 대표 실행 경로는 Windows의 `run.ps1`이고, Raspberry Pi에서는 CPU-only 런타임과 LCD 키오스크 GUI를 별도로 제공합니다.

> **핵심 원칙:** LLM, NPU, ML, 온톨로지, GNN은 분류·랭킹·설명·마스킹·보조 점수만 제공합니다. 실제 주문은 `RiskManager`와 비용/원금보호/신선도/중복주문/KIS 런타임 게이트를 모두 통과한 `FinalOrder`만 제출합니다.

![Current runtime architecture](docs/diagrams/system_overview.png)

## 온톨로지와 GNN 레이어

지식 표현(왼쪽)과 학습 기반 전략 효용 추정(오른쪽)이 어떻게 쌓여 있고, 어디서 결정론적 권한으로 넘어가는지를 보여줍니다. OWL은 open-world라 거래를 허가하지도 금지하지도 않고, closed-world 운영 게이트가 전략 허용 집합을 만들며, GNN은 그 마스크 안에서만 효용을 계산합니다.

![Ontology and GNN layers](docs/diagrams/ontology_gnn_layers.svg)

## 데이터가 어떤 판단에 쓰이는가

어떤 입력이 어떤 처리 단계를 거쳐, 실제로 어떤 판단에 영향을 줄 수 있는지의 지도입니다. 마지막 밴드가 "무엇이 무엇을 결정하는가"를 명시합니다.

![Data to decision map](docs/diagrams/data_to_decision_flow.svg)

두 그림은 스크립트로 생성됩니다. 구조가 바뀌면 `python scripts/gen_docs_diagrams.py`로 재생성하세요.

## 현재 런타임 요약

`run.ps1`은 로컬 Windows 운영용 표준 런처입니다. 기본 포트는 `8010`이고, 서버 준비 후 `http://127.0.0.1:8010/account`를 관리 브라우저 창으로 엽니다. 이 창을 닫으면 서버도 함께 종료됩니다.

시작 시 기본 수행:

- KIS 실계좌 읽기, 잔고/현금/보유종목 스냅샷 갱신
- KIS 실시간 체결가/호가 수집과 브로커 quote 갱신
- 실시간 feature frame 생성과 live short-horizon 모델 주기 학습
- 독립 실시간 자동거래 루프 시작
- SELL/REDUCE 평가를 BUY보다 먼저 실행
- 기존 미체결 SELL 주문은 의미 있는 가격 변화가 있을 때만 정정, 아니면 유지
- BUY는 현금, spread, 유동성, quote freshness, 온톨로지/런타임 근거, 순수익 게이트, 리스크 게이트를 모두 통과해야 제출

**Windows 기본값은 live-capable입니다.** `run.ps1`은 `TRADING_MODE=live_trading`, `LIVE_TRADING_ENABLED=true`, `KIS_LIVE_ENABLED=true`, `LIVE_ORDER_SUBMIT_ENABLED=true`, `REQUIRE_MANUAL_ARMING=false`, `AUTO_START_REALTIME_TRADING=true`를 프로세스 환경에 설정합니다. 안전성은 live flag를 끄는 방식이 아니라 계좌/시장 데이터 신뢰도와 최종 게이트에서 확보합니다. 끄는 방법은 [docs/live_trading.md](docs/live_trading.md)에 있습니다.

## 빠른 실행

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
.\run.ps1
```

수동 서버 실행:

```powershell
python .\run.py --host 127.0.0.1 --port 8010
```

KIS 실계좌 연동 전에는 `config/secrets/kis_api_keys.env.example`을 `config/secrets/kis_api_keys.env`로 복사한 뒤 값을 채웁니다.

```powershell
python scripts/check_kis_connection.py --account
python scripts/live_readiness_check.py
```

## 웹 GUI

운영 중심 화면은 `/account`입니다. 루트 화면(`/`)에는 연구/진단/수동 실행 기능이 남아 있습니다.

- `/account` — 온톨로지 전략 트레이딩 터미널. 온톨로지 후보와 선택 알고리즘, 실시간 1분봉·MA5·MA20·VWAP·거래량, 전략 stop/target, 호가 불균형을 표시합니다. 주문 흐름은 `Ontology → StrategyInstance → OrderIntent → RiskVerdict → KIS → Fill`을 인과 저널 기준으로 보여주고, 전략이 선택되지 않았거나 게이트가 차단하면 `NoTrade`를 그대로 표시합니다. 종료 버튼은 먼저 신규 BUY를 막고, 라이브 게이트를 통과하는 청산 SELL을 제출한 뒤 서버 종료를 예약합니다.
- `/display/ontology` — 온톨로지 지식 그래프와 추론 상태 전체 화면
- `/display` — Raspberry Pi LCD용 trade-reason 카드 보드

주요 API:

```text
GET  /api/account/dashboard                        GET  /api/account/technical
GET  /api/account/asset-history?range=1D|1W|1M|3M  GET  /api/account/macro-micro
GET  /api/realtime-trading/status                  GET  /api/refactor/dashboard
GET  /api/refactor/market-view?symbol=005930       GET  /api/trade-explanations
GET  /api/ontology/graph                           GET  /api/realtime/runtime
GET  /api/npu/runtime                              POST /api/live-trading/terminate?shutdown=true
```

## Raspberry Pi

Pi 패키지는 `packaging/raspberrypi/`에 있습니다. OpenVINO/NPU 없이 CPU-only로 실행되고 기본값은 read-only입니다.

```bash
bash packaging/raspberrypi/bootstrap.sh
bash packaging/raspberrypi/run.sh
bash packaging/raspberrypi/pi-dashboard-launch.sh   # LCD 키오스크
```

`APP_HOST=0.0.0.0`, `APP_PORT=8010`으로 LAN에서 `http://<pi-ip>:8010/account`에 접속하고, `ONTOLOGY_ACCELERATOR=CPU`, `TRADING_MODE=read_only`, `LIVE_ORDER_SUBMIT_ENABLED=false`가 기본입니다. 기존 `data/`를 그대로 재사용하며 `pi.env`로 지속 override합니다. 자세한 내용은 [docs/raspberry_pi_deployment.md](docs/raspberry_pi_deployment.md)와 [packaging/raspberrypi/README.md](packaging/raspberrypi/README.md)에 있습니다.

## 안전 모델

- LLM 또는 LLM-like 컴포넌트는 주문을 실행하지 않습니다.
- NPU/OpenVINO 출력은 숫자 근거 점수일 뿐 주문 승인 권한이 아닙니다.
- live short-horizon 모델은 기본 advisory-only입니다. `REALTIME_MODEL_AUXILIARY_ONLY=true`면 모델 단독 BUY는 거부됩니다.
- strategy-utility R-GCN은 shadow 관측 전용이며 체크포인트 provenance가 맞지 않으면 fail-closed로 `NO_TRADE`를 냅니다.
- synthetic/sample/hash 파생 데이터는 오프라인 fixture에서만 허용하고 paper/live 판단 근거로 쓰지 않습니다.
- margin, leverage, derivatives, short selling, credit loan, leveraged ETF는 거부 대상입니다.
- live 주문은 `LiveExecutionCoordinator`를 통해 limit order로만 제출합니다.
- audit log는 credential, token, 계좌번호, broker secret을 재귀적으로 마스킹합니다.

이 저장소의 어떤 코드도 수익이나 자본 보전을 보장하지 않습니다. 소액 계좌 단기 단타는 왕복 비용을 고려하면 구조적으로 음의 기대값에 가깝습니다 — [docs/validation.md](docs/validation.md)의 실측 지표를 먼저 보세요.

## 주요 디렉터리

| 경로 | 역할 |
| --- | --- |
| `src/app/web.py`, `web_account_routes.py`, `account_dashboard.py` | FastAPI 앱, `/account` 라우트와 payload |
| `src/app/data/` | KIS 실시간 수집, 이벤트 파이프라인, realtime store, source policy, 세션 판정 |
| `src/app/features/`, `technical/` | 지표, live feature frame, 근거 기반 기술적 예측 레이어 |
| `src/app/graph/`, `ontology/` | KnowledgeGraph, FactTable, RDF/OWL/SHACL, 거시–미시 추론, closed-world 게이트 |
| `src/app/models/`, `routing/`, `strategy/` | 학습·추론 backend, strategy-utility R-GCN, 라우터, 7개 전략 expert |
| `src/app/cost/`, `risk/`, `trading/` | ProfitabilityGate, 사이징, 원금보호, RiskManager, 동적 청산, 실시간 엔진 |
| `src/app/execution/` | KIS adapter, 주문 가격 정책, 거래소 라우팅, live coordinator, 저널 |
| `config/`, `packaging/raspberrypi/`, `scripts/` | 정책·프로파일 설정, Pi 패키지, 점검·학습·리플레이·벤치마크 스크립트 |

## 문서

| 문서 | 내용 |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | 런타임 구조, 모듈 지도, API, 운영 모드, 가속 경계, 알려진 한계 |
| [docs/ontology_and_gnn.md](docs/ontology_and_gnn.md) | 온톨로지 계층과 전략 효용 GNN, NPU/CPU 분할 |
| [docs/decision_and_risk.md](docs/decision_and_risk.md) | 데이터→판단 매핑, 순수익 게이트, 동적 청산, 리스크와 실행 |
| [docs/live_trading.md](docs/live_trading.md) | 설치, 게이트, 운영, 비상 정지, arming, 런타임 프로파일 |
| [docs/raspberry_pi_deployment.md](docs/raspberry_pi_deployment.md) | Pi CPU-only 배포와 키오스크 |
| [docs/validation.md](docs/validation.md) | 벤치마크, 리플레이 지표, 승격 판정 |

## 테스트

```powershell
python -m pytest
python -m pytest tests/test_web_live_flags.py tests/test_web_graph_payload.py tests/test_account_dashboard.py tests/test_kiosk_display_overview.py
```

live 주문을 제출하는 테스트는 없습니다. 환경에 따라 OpenVINO/NPU, KIS 실계좌, 로컬 LLM 관련 테스트는 optional dependency나 secrets 상태의 영향을 받습니다.

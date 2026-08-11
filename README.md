# Personal Investment Agent

KIS 실시간 데이터, 온톨로지 기반 근거 추론, 단기 학습 모델, 결정론적 리스크 게이트를 묶은 개인용 자동 투자 분석/운영 시스템입니다. 대표 실행 경로는 Windows의 `run.ps1`이고, Raspberry Pi에서는 CPU-only 런타임과 LCD 키오스크 GUI를 별도로 제공합니다.

> **핵심 원칙:** 온톨로지는 허용 관계와 전략 집합을 정의하고, GNN은 그 관계의 데이터 기반 가중치와 전략 효용을 학습해 실제 전략 선택에 사용됩니다. 단, GNN의 모델 보정 신뢰도와 전략별 양수 순효율이 실시간 데이터로 검증되어야 진입 권한이 생깁니다. LLM/NPU는 주문 권한이 없고, 실제 주문은 제비용·원금보호·신선도·중복주문·KIS 런타임 게이트를 모두 통과한 `FinalOrder`만 제출합니다.

![Current runtime architecture](docs/diagrams/system_overview.png)

## 온톨로지와 GNN 레이어

지식 표현(왼쪽)과 학습 기반 전략 효용 추정(오른쪽)이 하나의 판단 파이프라인으로 연결되는 구조입니다. OWL은 open-world 지식을 확장하고, closed-world 운영 게이트가 현재 사실로 허용 가능한 전략 집합을 만들며, GNN은 그 마스크 안에서 관계 가중치와 비용 차감 기대효용을 계산합니다. 모델 보정과 전략별 진입 권한은 별도 실시간 검증 단계입니다.

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
- 온톨로지 허용 전략 안에서 strategy-utility R-GCN 후보 평가 (롱 17 + 숏 3, 숏은 shadow 평가만)
- 전략별 실시간 forward 검증과 `calibrated`/`entry_authorized` 권한 분리
- 독립 실시간 자동거래 루프 시작
- SELL/REDUCE 평가를 BUY보다 먼저 실행
- 기존 미체결 SELL 주문은 의미 있는 가격 변화가 있을 때만 정정, 아니면 유지
- BUY는 현금, spread, 유동성, quote freshness, 온톨로지/런타임 근거, 순수익 게이트, 리스크 게이트를 모두 통과해야 제출
- **신규 진입은 정규장에서만** — 시간외 호가(예: 미국 애프터마켓의 분당 2주 · 스프레드 33bp)로는 진입하지 않습니다. 청산은 시간외에도 계속 동작합니다.
- **전략 선택은 보수적 하단값 기준** — `no_trade`가 실제 선택지이며, 측정된 순엣지 하단이 양수인 전략이 없으면 거래하지 않습니다.
- **시장 상태 공백은 연구 대상으로 승격** — 6축 coverage가 반복되는
  `STRATEGY_COVERAGE_GAP`을 집계합니다. 현재 1초/5초 체결 표본이 부족한 세션을 위해 완성된
  1분봉 추세·VWAP·지속성·상대거래량을 쓰는 `bar_trend_continuation`을 추가했지만, 이 전략도
  SHADOW에서 양의 순수익을 입증하기 전에는 주문할 수 없습니다. “어떤 상태든 평가”와 “근거 없이
  항상 거래”는 같은 요구가 아닙니다.
- **LONG / SHORT / NO_TRADE를 같은 선택 공간에서 비교**하되, 숏 arm은 배포 상태가 live 칸이 아니면
  순위만 매겨지고 실행 후보가 되지 않습니다. 롱과 숏 posterior는 절대 합산하지 않습니다.
- **항별로 분해된 선택기(V2)가 초기 SHADOW로 병행 실행**됩니다. Windows `run.ps1`은 이를
  활성화하고 자동 승격을 켭니다. 같은 evidence로
  `MarketContext → eligibility mask → proposal → 효용 − 비용 − downside − uncertainty + 온톨로지
  + bandit → NO_TRADE 비교`를 계산해 forward 반사실 증거를 쌓습니다. 보수적 하단·비용 스트레스·
  안정성·regret 기준을 3회 연속 통과하면 `SHADOW → LIVE_PROBE(10%)`, 실체결 증거까지 통과하면
  `LIVE(100%)`로 **자동 전환**합니다. 기준 미달이나 악화 시 자동 강등되며 수동 변경은 필요하지
  않습니다 — [docs/strategy_selection_v2.md](docs/strategy_selection_v2.md).

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
- `/account` 상단 **ENTRY BLOCKADE 패널** — "왜 신규 진입이 없는가"를 순서 있는 체인(엔진 → 라이브 무장 → 시장 세션 → 매수 후보 → 마이크로 전략 → 전략 선택 → 포지션)으로 보여줍니다. **처음 막힌 단계가 실제 원인**이며, 그 뒤 단계는 도달하지 않았음을 나타내기 위해 흐리게 표시됩니다. 이전에는 마지막으로 실패한 계층의 사유 코드 하나만 노출되어, 실제 원인이 시장 세션인데 11,614 사이클 동안 `NO_POSITIVE_NET_GNN_EDGE`(GNN 탓)로 표시되는 문제가 있었습니다.
- `/display/ontology` — 온톨로지 지식 그래프와 추론 상태 전체 화면
- `/display` — Raspberry Pi LCD용 trade-reason 카드 보드

주요 API:

```text
GET  /api/account/dashboard                        GET  /api/account/technical
GET  /api/account/asset-history?range=1D|1W|1M|3M  GET  /api/account/macro-micro
GET  /api/realtime-trading/status                  GET  /api/refactor/dashboard
GET  /api/refactor/market-view?symbol={symbol}     GET  /api/trade-explanations
GET  /api/ontology/graph                           GET  /api/realtime/runtime
GET  /api/npu/runtime                              GET  /api/gnn/realtime-trust
GET  /api/system-diagnostics                       GET  /api/auto-reliability/status
GET  /api/realtime-trading/entry-blockade          # 진입 차단 지점 진단 체인
POST /api/live-trading/terminate?shutdown=true
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
- strategy-utility R-GCN은 실제 전략 선택에 직접 참여합니다. 체크포인트 provenance, 온톨로지 허용 관계, 모델 보정 신뢰도, 전략별 실현 양수 순효율 검증 중 하나라도 실패하면 신규 진입 권한은 생기지 않습니다.
- `GNN_REALTIME_MODEL_TRUST_PASSED`는 모델 보정 통과이고, `GNN_REALTIME_TRUST_PASSED`는 선택 전략의 실거래 진입 권한까지 통과했다는 뜻입니다.
- synthetic/sample/hash 파생 데이터는 오프라인 fixture에서만 허용하고 paper/live 판단 근거로 쓰지 않습니다.
- margin, derivatives, leveraged ETF는 실행 경로가 없어 전면 거부 대상입니다.
- **short selling / credit loan은 실행 경로가 생겼지만 기본 거부입니다.** `RiskRules`의
  `short_selling_allowed` / `credit_loan_allowed`는 기본 `False`이며, 켜더라도 해당 arm의
  **배포 상태가 live 칸**이어야 주문이 만들어집니다. 두 조건은 독립이고, 하나만으로는 부족합니다.
- 숏 전략 3개(`market_intraday_momentum_short`, `opening_range_breakdown`,
  `residual_relative_weakness`)는 카탈로그에 있으나 **전량 `SHADOW`이며 실주문 권한이 0개**입니다.
  `SHADOW → LIVE_FULL` 직행 전이는 어떤 환경변수·설정·수동 조작으로도 존재하지 않습니다
  ([docs/short_selling_deployment.md](docs/short_selling_deployment.md)).
- 숏 **청산**은 계좌 플래그나 리스크 한도로 막지 않습니다. 손실 중인 숏의 상환을 거부하면 무제한
  손실 포지션이 갇힙니다.
- live 주문은 `LiveExecutionCoordinator`를 통해 limit order로만 제출합니다.
- audit log는 credential, token, 계좌번호, broker secret을 재귀적으로 마스킹합니다.
- **전략 선택 계층(V2)은 실행 계층을 import할 수 없습니다.** `app.execution`, `app.risk`,
  `app.cost.profitability_gate`, realtime engine, shared decision engine 어느 것도 import하지
  않으며 AST 테스트로 강제됩니다. import 경로가 없는 코드는 리스크 게이트를 우회할 수 없습니다.
- 반사실(shadow) 포지션은 산술일 뿐이고 broker 경로가 없습니다. V2가 권한을 얻어 선택한 전략의
  성과는 실체결에서 별도로 받아 `LIVE_PROBE → LIVE` 증거로 연결합니다. V2의 순수 선택 모듈은
  주문을 만들지 못하며, 세션 계층은 독립적으로 실주문 승인된 proposal과 일치할 때만 선택을
  채택합니다 — 시뮬레이션을 같은 arm의 posterior에 두 번 넣지 않기 위한 구조입니다.

이 저장소의 어떤 코드도 수익이나 자본 보전을 보장하지 않습니다. 소액 계좌 단기 단타는 왕복 비용을 고려하면 구조적으로 음의 기대값에 가깝습니다 — [docs/validation.md](docs/validation.md)의 실측 지표를 먼저 보세요.

> **2026-08-11 측정, 먼저 읽을 것.** 모든 전략을 처음으로 같은 evaluator에 통과시킨 결과, LIVE
> 권한을 가진 전략 13개 중 **9개가 관측 0건**이고 표본이 있는 유일한 전략
> (`liquidity_shock_reversal`, 약 740건이며 계속 증가)은 평균 순 **약 −120bps**로 손실 중이며 walk-forward 창 중
> 양수가 0개입니다. 그 행 전부가 shadow·US 이므로 시뮬레이션 증거이며, 그래서 audit은 RETIRE가
> 아니라 RESEARCH로 분류합니다. 권고는 기록됐고 적용은 별도 migration flag를 요구합니다 —
> [docs/validation.md](docs/validation.md) §1.1. 현재 값은
> `python scripts/report_strategy_selection_v2.py`.

## 주요 디렉터리

| 경로 | 역할 |
| --- | --- |
| `src/app/web.py`, `web_account_routes.py`, `account_dashboard.py` | FastAPI 앱, `/account` 라우트와 payload |
| `src/app/data/` | KIS 실시간 수집, 이벤트 파이프라인, realtime store, source policy, 세션 판정 |
| `src/app/features/`, `technical/` | 지표, live feature frame, 근거 기반 기술적 예측 레이어 |
| `src/app/context/` | 통합 `MarketContext` 단일 생성 권한 — (종목, 사이클)당 스냅샷 1개 + `context_id` |
| `src/app/graph/`, `ontology/` | KnowledgeGraph, FactTable, RDF/OWL/SHACL, 거시–미시 추론, closed-world 게이트, 전략 eligibility(hard mask + soft score) |
| `src/app/models/`, `routing/`, `strategy/` | 학습·추론 backend, strategy-utility R-GCN, 실시간 신뢰도 평가, 라우터, `StrategySelectorV2`, 전략 spec/registry/proposal/coverage, 전략 expert(롱 17 + 숏 3, `STRATEGY_IDS` 기준) |
| `src/app/evaluation/`, `monitoring/`, `strategy_validation/` | 반사실 shadow 포지션·selector regret·증거원 분리, drift 모니터, 단일 audit runner·purged CV·비용 스트레스·lifecycle 원장 |
| `src/app/cost/`, `risk/`, `trading/` | ProfitabilityGate, 사이징, 원금보호, RiskManager, 동적 청산, 실시간 엔진 |
| `src/app/trading/directional.py`, `borrow.py`, `directional_shadow.py`, `short_strategy_promotion.py` | 방향 계약, 대주 point-in-time 저널, forward shadow 평가, 배포 사다리 authority |
| `src/app/execution/` | KIS adapter, 주문 가격 정책, 거래소 라우팅, live coordinator, 저널 |
| `config/`, `packaging/raspberrypi/`, `scripts/` | 정책·프로파일 설정, Pi 패키지, 점검·학습·리플레이·벤치마크 스크립트 |

## 문서

| 문서 | 내용 |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | 런타임 구조, 모듈 지도, API, 운영 모드, 가속 경계, 알려진 한계 |
| [docs/ontology_and_gnn.md](docs/ontology_and_gnn.md) | 온톨로지 계층, hard/soft eligibility 분리, 전략 효용 GNN, NPU/CPU 분할 |
| [docs/decision_and_risk.md](docs/decision_and_risk.md) | 데이터→판단 매핑, 순수익 게이트, 동적 청산, 리스크와 실행, 컴포넌트별 단일 질문(§7.1), 레짐 변화 대응(§9), 진입 차단 진단(§10) |
| [docs/strategy_selection_v2.md](docs/strategy_selection_v2.md) | 전략 선택 V2: 효용 수식, 자동 `SHADOW → LIVE_PROBE → LIVE` 권한 사다리, 현재 상태, audit와 기술 부채 |
| [docs/live_trading.md](docs/live_trading.md) | 설치, 게이트, 운영, 비상 정지, arming, 런타임 프로파일 |
| [docs/short_selling_deployment.md](docs/short_selling_deployment.md) | 숏 배포 사다리: 자동 승격/강등, 대주 fail-closed 규칙, forward shadow 누수 방어 |
| [docs/raspberry_pi_deployment.md](docs/raspberry_pi_deployment.md) | Pi CPU-only 배포와 키오스크 |
| [docs/validation.md](docs/validation.md) | 벤치마크, 리플레이 지표, 검증 프레임워크, 승격 판정 |

## 테스트

```powershell
python -m pytest
python -m pytest tests/test_web_live_flags.py tests/test_web_graph_payload.py tests/test_account_dashboard.py tests/test_kiosk_display_overview.py
python -m pytest tests/test_directional_short_ladder.py   # 숏 사다리 안전 속성
python -m pytest tests/test_strategy_selector_v2.py       # 전략 선택 V2 + 실행 계층 격리
python -m pytest tests/test_selector_v2_auto_promotion.py # 자동 승격·강등·영속화·안전 기본값
```

전략·계좌 audit 리포트(읽기 전용, 주문 없음):

```powershell
python scripts/report_strategy_selection_v2.py            # 전략 audit 표 + coverage
python scripts/diagnose_strategy_selection.py             # 선출 진단
```

live 주문을 제출하는 테스트는 없습니다. 환경에 따라 OpenVINO/NPU, KIS 실계좌, 로컬 LLM 관련 테스트는 optional dependency나 secrets 상태의 영향을 받습니다.

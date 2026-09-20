# Ontology Based AI Trading System (OBAITS)

2026-09-20 변경: [다중시장·경량 추론 리팩터링과 검증](docs/refactor_2026_09_20.md). 기본 화면은 `/account`, 기존 상세 화면은 `/account/advanced`입니다.

추가 개편: [온톨로지 기반 동적 리스크와 청산](docs/dynamic_ontology_risk.md), [정식 OWL/SHACL 구성](docs/formal_policy_ontology.md), [시간 인식 R-GCN 선택·NPU 실측](docs/temporal_relational_graph_model.md), [시장별 전략 성과 적응](docs/strategy_performance_adaptation.md).

최신 판단 구조: [온톨로지 위험 판단 통합](docs/ontology_risk_authority.md). 위험·비용·수량을 한 번 결정하고, 승인 이후에는 유효기간·허용 가격·실제 주문 가능 여부만 확인합니다.

OBAITS는 KIS 실시간 데이터, 온톨로지 기반 근거 추론, live feature frame, 동적 위험 판단과 주문 실행 검증을 결합한 로컬 자동 투자 운영 시스템입니다. 현재 코드 기준으로는 `run.ps1`과 `run.py`가 표준 런처이며, `src/app` 아래에서 FastAPI UI, 실시간 수집, 전략 선택, 위험 판단, 실행 모듈이 함께 동작합니다.

> 핵심 원칙: 실시간 판단은 `selection`과 `execution`을 구분해 다룹니다. LLM이나 가속기 결과는 주문 권한이 아니며, 실제 주문은 `FinalOrder`를 통과한 경우만 `LiveExecutionCoordinator`를 통해 제출됩니다. 전략 선택 V2는 SHADOW에서 시작하고 증거가 누적되면 자동 승격/강등이 일어납니다.

![Current runtime architecture](docs/diagrams/system_overview.png)

## 현재 코드에서 실제로 무엇이 동작하나

현재 저장소는 다음 흐름으로 동작합니다.

- `setup.ps1`이 OS별 가상환경을 만들고 의존성을 설치한 뒤, `app.web` import와 device probe를 검증합니다.
- `run.py`는 `src` 경로를 추가하고, 외부 바인드 시 `APP_ACCESS_TOKEN` 요구 여부를 점검하며, 필요하면 포트 충돌 시 다음 사용 가능한 포트로 자동 전환합니다.
- `run.ps1`는 관리 브라우저와 서버 lifecycle을 함께 다루며, 기본 환경 변수로 `APP_PORT=8010`과 live-trading 플래그를 세팅합니다.
- 서버 시작 후 백그라운드에서 `ResearchService`와 demo startup checks가 실행되고, `/account` 및 관련 API가 준비됩니다.
- 선택 계층과 실행 계층은 분리되어 있으며, `strategy_selection_v2`는 실행 계층을 직접 import할 수 없도록 설계되어 있습니다.

이 구조는 코드 수준에서 명시되어 있으므로, 문서는 특정 제품 설명보다 실제 런타임 경로와 안전장치 중심으로 유지하는 것이 맞습니다.

## 빠른 시작

```powershell
# Windows: run.bat 더블클릭 또는 아래 명령
.\run.bat

# 설치·서버 시작·주문 없이 실행 환경만 확인
.\run.bat -CheckOnly
```

```bash
# Linux + PowerShell 7
./run.ps1
./run.ps1 -CheckOnly
```

일반 실행은 해당 OS의 가상환경과 필수 의존성을 먼저 검사하고, 누락 시 `setup.ps1`을 실행한 뒤 다시 검사합니다. 최초 준비에는 Python 3.11 이상 또는 `uv`, 패키지 다운로드 연결이 필요합니다. 이미 준비된 환경에서는 설치를 반복하지 않습니다. 자동 설치를 생략하려면 `-SkipSetup`, 브라우저 없이 실행하려면 `-Headless`를 사용합니다. 별도 구성은 `setup.ps1`의 옵션으로 지정할 수 있습니다.

다른 PC에서 동기화된 가상환경의 Python 실행 경로가 깨진 경우에는 해당 PC에서 `setup.ps1 -Recreate`로 가상환경을 다시 만든 뒤 실행합니다.

`run.bat`은 PowerShell 7을 우선 사용하고 없으면 Windows PowerShell 5.1로 실행합니다. 실행 위치와 무관하게 프로젝트를 찾으며 한글·공백 경로와 인자를 보존합니다. 시작 시 Python 버전, 실제 감지 장치, 기기별 DB·모델 경로를 표시합니다. Synology 폴더에서 실행할 때 모델, OpenVINO 캐시, 브라우저 프로필과 새 런처 로그는 기기별 로컬 경로를 사용하며, 명시적인 경로 설정은 유지합니다.

일반 실행은 기존 실계좌 운영 설정을 사용합니다. 명시적으로 지정한 `LIVE_ORDER_SUBMIT_ENABLED=false` 또는 `REQUIRE_MANUAL_ARMING=true`를 덮어쓰지 않습니다. 기존 서버의 종료 안전성을 확인할 수 없거나 관리 중인 포지션이 있으면 재시작을 중단합니다. 브라우저를 닫아도 안전한 종료가 거절되면 서버를 유지하고 주소를 표시합니다. 의도적인 강제 복구에만 `-ForceRestart` 또는 `-HardKill`을 사용합니다.

수동 실행:

```bash
.venv-linux/bin/python run.py --host 127.0.0.1 --port 8010
# Windows: .\.venv\Scripts\python.exe run.py --host 127.0.0.1 --port 8010
```

KIS 실계좌 연동 전에는 로컬 비밀 파일을 만들어야 합니다.

```bash
cp config/secrets/kis_api_keys.env.example config/secrets/kis_api_keys.env
cp config/principal_protection.example.json config/principal_protection.json
cp config/trading_costs.example.json config/trading_costs.json
cp config/live_trading_safety.example.json config/live_trading_safety.json
cp config/order_execution.example.json config/order_execution.json
```

점검 명령:

```bash
python scripts/check_kis_connection.py --account
python scripts/live_readiness_check.py --dry-run
```

## 실제 런타임 요약

- `run.ps1`은 기본적으로 포트 `8010`에서 운영용 서버를 띄우고, 로컬 브라우저를 열어 `/account` 화면에 연결합니다.
- `run.py`의 기본값은 `8000`이지만, 런처가 `APP_PORT`를 `8010`로 설정하고 충돌 시 다른 포트로 이동할 수 있습니다.
- 외부 바인드가 필요한 경우 `APP_ACCESS_TOKEN`이나 `-External` 정책을 통해 접근 제어가 요구됩니다.
- `run.py`는 startup checks를 백그라운드로 수행하여 `ResearchService`, demo pipeline, 그래프 저장/검증 등을 바로 실행합니다.
- 운영 온톨로지 경로는 시장 근거와 전략 예측을 `ontology.risk_authority`에서 비용·위험·수량으로 통합하고, 승인 기록이 있는 `TradePlan`을 `LiveExecutionCoordinator`에 전달합니다. 이후 별도의 시장 위험·수익성 심사를 반복하지 않습니다.
- 현재 운영 경로의 신규 진입은 현금 롱 주문만 허용합니다. 숏·대주 모듈은 연구와 기존 상태 처리용으로 남겨 두었습니다.
- LLM 또는 NPU 결과는 의사결정 보조 입력일 뿐이고, 주문 승인 권한은 없습니다.

## 안전 모델과 운영 포지처

- `run.ps1` 기본 실행은 platform-agnostic live-capable 상태를 전제로 합니다.
- 실행 계층은 계좌 상태, 시세 유효성, 중복 주문, 실제 주문 가능 현금·수량과 사용자 중지 설정을 확인합니다. 기존 리스크 API의 정책 미주입 경로는 과거 재현과 호환을 위해 유지합니다.
- `StrategySelectorV2`는 SHADOW → LIVE_PROBE → LIVE로 자동 전환할 수 있지만, 이를 위해 필요한 증거와 통계 게이트는 설정 파일과 promotion controller가 담당합니다.
- 전략의 승격 상태와 별개로 운영 계좌의 현금 롱 주문 계약을 지켜야 합니다.
- `run.py`는 시작 전에 `require_token_for_external_bind`를 수행하여, 외부 접근이 필요한 경우 안전하게 거절하거나 토큰을 요구합니다.

## 주요 디렉터리

| 경로 | 역할 |
| --- | --- |
| `src/app/web.py` | FastAPI 앱, 루트 UI, API, live/runtime orchestration |
| `src/app/run.py` | 실제 서버 기동 및 startup check 관리 |
| `src/app/cli.py` | demo, research, simulation CLI |
| `src/app/data/` | KIS realtime 수집, source trust, session capability, market data store |
| `src/app/context/` | `MarketContext` 생성과 temporal/session metadata |
| `src/app/ontology/` | RDF/OWL/SHACL, closed-world gate, strategy eligibility |
| `src/app/graph/` | knowledge graph, fact table, macro/micro reasoning |
| `src/app/models/` | live training, inference backend, strategy-utility model |
| `src/app/routing/` | `StrategySelectorV2`, proposal engine, promotion logic |
| `src/app/cost/`, `src/app/risk/`, `src/app/trading/` | profitability, sizing, risk, execution policy |
| `src/app/execution/` | KIS adapter, order pricing, order state machine, live coordinator |
| `config/` | 정책/setting YAML, runtime profiles, selector promotion config |
| `scripts/` | readiness, training, replay, diagnostics, benchmark utilities |
| `docs/` | 운영 문서, 아키텍처 설명, validation/strategy selection 문서 |

## 문서 인덱스

| 문서 | 내용 |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | 런타임 구조와 실제 엔트리포인트 |
| [docs/ontology_and_gnn.md](docs/ontology_and_gnn.md) | 온톨로지와 GNN/eligibility 경계 |
| [docs/decision_and_risk.md](docs/decision_and_risk.md) | data → decision → profitability/risk 흐름 |
| [docs/strategy_selection_v2.md](docs/strategy_selection_v2.md) | selector V2, SHADOW/LIVE 자동 promotion |
| [docs/live_trading.md](docs/live_trading.md) | 설치, 점검, live gate, emergency stop |
| [docs/short_selling_deployment.md](docs/short_selling_deployment.md) | 숏 배포 ladder와 fail-closed 규칙 |
| [docs/raspberry_pi_deployment.md](docs/raspberry_pi_deployment.md) | Pi CPU-only deployment |
| [docs/validation.md](docs/validation.md) | replay, validation, 승격 기준 |

## 테스트

```bash
python -m pytest
python -m pytest tests/test_strategy_selector_v2.py
python -m pytest tests/test_selector_v2_auto_promotion.py
python -m pytest tests/test_final_trade_gate.py
python -m pytest tests/test_directional_short_ladder.py
```

## 주의사항

이 저장소는 개인용/실험용 로컬 운영 코드이며, 어떤 코드도 수익 보장을 하지 않습니다. 실제 주문 제출 전에는 반드시 KIS 인증, 계좌 상태, live readiness, 보수적 게이트, 수량/비용 검증을 확인해야 합니다.

문서와 코드가 어긋날 수 있으므로, 최근 변경 사항은 이 README와 [docs/README.md](docs/README.md)를 기준으로 보세요. 새로 추가된 기능이나 정책 변경이 있으면 관련 문서도 함께 갱신하는 것을 권장합니다.

# Ontology Based AI Trading System (OBAITS)

OBAITS는 KIS 실시간 데이터, 온톨로지 기반 근거 추론, live feature frame, 그리고 결정론적 리스크/주문 게이트를 결합한 로컬 자동 투자 운영 시스템입니다. 현재 코드 기준으로는 `run.ps1`과 `run.py`가 표준 런처이며, `src/app` 아래에서 FastAPI UI, 실시간 수집, 전략 선택, risk/gate, 실행 모듈이 함께 동작합니다.

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
# Windows
.\setup.ps1 -All
.\run.ps1
```

```bash
# Linux + PowerShell 7
./setup.ps1 -All -CudaWheels cu128
./run.ps1
```

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
- 실시간 엔진은 `MarketContext`, `StrategySelectorV2`, 비용·리스크 게이트, `FinalTradeGate`, `LiveExecutionCoordinator`를 순서대로 통과해야만 실제 주문을 만들 수 있습니다.
- 숏 전략은 기본적으로 `SHADOW` 상태이며, 실제 실주문 권한은 별도 자동 승격 조건을 충족해야 생깁니다.
- LLM 또는 NPU 결과는 의사결정 보조 입력일 뿐이고, 주문 승인 권한은 없습니다.

## 안전 모델과 운영 포지처

- `run.ps1` 기본 실행은 platform-agnostic live-capable 상태를 전제로 합니다.
- 안전성은 "플래그를 낮게 끄는 것"이 아니라, 계좌 신뢰도, 시장 신선도, 비용·리스크 게이트, 주문 상태와 final approval를 모두 통과해야 하도록 구현되어 있습니다.
- `StrategySelectorV2`는 SHADOW → LIVE_PROBE → LIVE로 자동 전환할 수 있지만, 이를 위해 필요한 증거와 통계 게이트는 설정 파일과 promotion controller가 담당합니다.
- 숏/대주 관련 로직은 fail-closed 규칙이 적용되며, 계좌 정책 허용과 arm 배포 상태가 둘 다 충족되어야 실제 주문이 생성됩니다.
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

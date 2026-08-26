# Documentation

OBAITS의 현재 코드 기준 문서 허브입니다. 최상위 개요는 [../README.md](../README.md)를 먼저 보고, 이 문서에서는 `docs/` 안의 모든 문서를 역할별로 나눠서 안내합니다. 새 문서를 추가할 때는 기존 문서와 역할이 겹치지 않는지 먼저 확인하세요.

## 핵심 운영 문서

| 문서 | 내용 |
| --- | --- |
| [architecture.md](architecture.md) | 실제 런타임 진입점, startup flow, 모듈 지도, API 표면, 운영 모드, acceleration 경계 |
| [execution_authority.md](execution_authority.md) | 선출 이전/이후 권한 분리, TradePlan, strategy-owned fast tick executor, ExecutionGuard |
| [context_hierarchy.md](context_hierarchy.md) | 세션/캘린더 → 글로벌 → 국내 → 섹터 → 종목 컨텍스트, seasonality, FinalTradeGate, 주문 상태기계 |
| [ontology_and_gnn.md](ontology_and_gnn.md) | 온톨로지 레이어, closed-world gate, hard/soft eligibility, GNN이 선택에 참여하는 방식 |
| [decision_and_risk.md](decision_and_risk.md) | 데이터 → 판단 → cost/risk → exit 흐름, no-trade와 entry blockade의 원인 해석 |
| [strategy_selection_v2.md](strategy_selection_v2.md) | `MarketContext → eligibility → proposal → utility → cost → bandit → NO_TRADE` 경로와 자동 promotion ladder |
| [live_trading.md](live_trading.md) | 설치, readiness checks, live gate, emergency stop, 운영 절차 |
| [short_selling_deployment.md](short_selling_deployment.md) | 숏 전략의 SHADOW/LIVE 권한과 대주/borrow fail-closed 규칙 |
| [raspberry_pi_deployment.md](raspberry_pi_deployment.md) | Pi CPU-only 배포, 키오스크, LAN 접근 구성 |
| [validation.md](validation.md) | replay, benchmark, validation 기준, 승격/거부 사유 |

## 보조 운영/연구 문서

| 문서 | 내용 |
| --- | --- |
| [extended_hours_live_trading.md](extended_hours_live_trading.md) | 국내/미국 장외 시간 주문 규칙, route, 세션 충돌, fail-closed 운영 |
| [kis_market_session_capability_matrix.md](kis_market_session_capability_matrix.md) | KIS 공식 문서 기준 세션·TR·주문 가능성 검증표 |
| [dynamic_minute_bar_warmup.md](dynamic_minute_bar_warmup.md) | dynamic universe의 minute-bar warmup과 reconciled cache flow |
| [realtime_session_gap_analysis.md](realtime_session_gap_analysis.md) | 세션 gap 분석 감사 결과와 리팩터 배경 |
| [gpt_work_report_automation.md](gpt_work_report_automation.md) | ChatGPT Work → Codex 수신/검증/배포 안전 파이프라인 |
| [gs_quant_reference_layer.md](gs_quant_reference_layer.md) | GS Quant reference layer 비교 검증과 비-live mathematical reference |
| [open_source_strategy_review.md](open_source_strategy_review.md) | 공개 전략 후보 검토 결과와 채택/배제 기준 |
| [long_only_bear_market_strategy.md](long_only_bear_market_strategy.md) | `residual_relative_strength`의 defensive bear submode |

## 문서의 기준

이 문서 집합은 설계 백서보다 현재 코드와 일치하는 운영 문서를 목표로 합니다. 최근 코드에서 특히 중요한 사실은 다음과 같습니다.

- `run.py`는 startup checks를 백그라운드로 수행하고, 외부 바인드 시 token 검증을 먼저 요구합니다.
- `run.ps1`은 `.venv`/`.venv-linux`와 운영 환경을 함께 관리하고, `APP_PORT=8010`을 기본으로 설정합니다.
- `StrategySelectorV2`는 실행 계층을 직접 import하지 못하도록 설계되어 있고, 증거 기반 promotion이 별도 controller를 통합니다.
- 숏 arm은 기본적으로 SHADOW이며, 실제 실주문 권한은 별도 배포 상태와 fail-closed gate를 충족해야만 생깁니다.
- 가속기 장치는 보조 evidence만 제공하며, 주문 권한은 `FinalOrder`/`RiskManager`/`LiveExecutionCoordinator`가 모두 통과한 경우에만 발생합니다.

## 읽는 순서

1. 먼저 [../README.md](../README.md)와 [architecture.md](architecture.md)를 읽습니다.
2. 왜 신규 진입이 막히는지 알고 싶다면 [decision_and_risk.md](decision_and_risk.md)와 [live_trading.md](live_trading.md)를 함께 봅니다.
3. 전략이 어떤 이유로 선택되었는지 알고 싶다면 [strategy_selection_v2.md](strategy_selection_v2.md)로 이동합니다.
4. 숏 전략의 권한과 대주 규칙이 궁금하면 [short_selling_deployment.md](short_selling_deployment.md)를 봅니다.
5. 장외 시간, 세션, warmup, 외부 보고 파이프라인 같은 특수 주제가 필요하면 보조 문서들을 확인합니다.
6. 실측 기반 판단 수준을 확인하려면 [validation.md](validation.md)를 봅니다.

## 코드 옆에 사는 문서

| 위치 | 내용 |
| --- | --- |
| [../src/app/ontology/README.md](../src/app/ontology/README.md) | TTL 구성, reasoner boundary, eligibility 계층, ontology safety 확장 규칙 |
| [../packaging/raspberrypi/README.md](../packaging/raspberrypi/README.md) | Pi 패키지 파일 맵과 빠른 명령 |
| [../config/secrets/README.md](../config/secrets/README.md) | secrets 파일 규약 |

## 설정 파일이 문서인 경우

이 프로젝트는 값 자체보다 그 값의 이유를 설정 파일에서 설명하는 편입니다. 튜닝 전에는 아래 파일을 우선 확인하세요.

| 파일 | 내용 |
| --- | --- |
| [../config/no_trade_policy.yaml](../config/no_trade_policy.yaml) | 시장/시간대별 no-trade threshold와 하단 규칙 |
| [../config/strategy_selector_v2.yaml](../config/strategy_selector_v2.yaml) | utility weight와 V2 선택 가중치 |
| [../config/selector_v2_promotion.yaml](../config/selector_v2_promotion.yaml) | SHADOW → LIVE_PROBE → LIVE promotion/ demotion |
| [../config/bandit_adapter.yaml](../config/bandit_adapter.yaml) | bandit 경계와 context 설계 이유 |
| [../config/strategy_validation.yaml](../config/strategy_validation.yaml) | validation weight와 promotion gate |
| [../config/strategy_registry.yaml](../config/strategy_registry.yaml) | lifecycle와 권고 audit 기록 |
| [../config/short_strategy_deployment.yaml](../config/short_strategy_deployment.yaml) | 숏 arm 배포 상태와 fail-closed 기준 |
| [../config/market_sessions.yaml](../config/market_sessions.yaml) | 세션 시각창, 캘린더 스냅샷, live_order_authorized 정책 |

## 다이어그램

`docs/diagrams/` 안의 SVG는 script로 재생성됩니다. 구조가 바뀌면 아래 명령으로 다시 생성하세요.

```powershell
python scripts/gen_docs_diagrams.py
python scripts/gen_profitability_diagrams.py
```

| 파일 | 쓰이는 곳 |
| --- | --- |
| `system_overview.png` / `.svg` | README, architecture.md |
| `ontology_gnn_layers.svg` | README, ontology_and_gnn.md |
| `data_to_decision_flow.svg` | README, architecture.md, decision_and_risk.md |
| `entry_blockade_chain.svg` | decision_and_risk.md |

문서와 코드가 어긋나면 가장 먼저 README와 이 인덱스를 다시 확인하고, 수정이 필요한 문서만 좁게 업데이트하는 것이 안전합니다.

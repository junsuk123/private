# Documentation

현재 코드 기준의 운영 문서입니다. 최상위 개요는 [../README.md](../README.md)를 먼저 보세요.

문서는 7개로 유지합니다. 새 문서를 만들기 전에 기존 문서를 갱신할 수 있는지 먼저 확인하세요.

| 문서 | 내용 |
| --- | --- |
| [architecture.md](architecture.md) | 런타임 진입점, 모듈 지도, 저장소 레이아웃, API 표면, 운영 모드, 가속 경계, 알려진 한계 |
| [ontology_and_gnn.md](ontology_and_gnn.md) | 온톨로지 L0–L6 계층, closed-world 전략 마스크, 8전략 R-GCN 조건부 기대값, 실시간 모델 보정과 전략별 진입 권한 |
| [decision_and_risk.md](decision_and_risk.md) | 데이터→판단, gross/net 단위 계약, 제비용 게이트, 전략 소유권, 동적 청산, 리스크와 복구 |
| [live_trading.md](live_trading.md) | 설치, 점검, 자동 신뢰도 전환, GNN trust 진단, 제출 게이트, 종료/비상 정지, 로그 |
| [short_selling_deployment.md](short_selling_deployment.md) | 숏 전략 배포 사다리: SHADOW→LIVE_FULL 자동 승격/강등, 대주 fail-closed 규칙, forward shadow 검증의 누수 방어 |
| [raspberry_pi_deployment.md](raspberry_pi_deployment.md) | Pi CPU-only 설치/실행/서비스/키오스크 |
| [validation.md](validation.md) | 측정 결과, 벤치마크, 리플레이 지표, 승격 판정과 거부 사유 |

7번째 문서를 추가한 이유: 숏 배포 사다리는 상태 머신·승격 게이트·대주 회계·누수 방어가 서로
맞물린 하나의 서브시스템이고, 위 6개 문서 어디에 넣어도 **"왜 숏 전략이 코드에는 있는데
거래하지 않는가"**라는 단일 질문의 답이 4개 문서로 흩어집니다. 각 문서에는 해당 절에서 필요한
사실과 링크만 남겼습니다.

## 다이어그램

`docs/diagrams/`의 SVG는 스크립트로 생성됩니다. 구조가 바뀌면 스크립트를 고치고 재생성하세요.

```powershell
python scripts/gen_docs_diagrams.py         # ontology_gnn_layers, data_to_decision_flow
python scripts/gen_profitability_diagrams.py # profitability_* 4종
```

| 파일 | 쓰이는 곳 |
| --- | --- |
| `system_overview.png` / `.svg` | 최상위 README, architecture.md |
| `ontology_gnn_layers.svg` | 최상위 README, ontology_and_gnn.md |
| `data_to_decision_flow.svg` | 최상위 README, architecture.md, decision_and_risk.md |
| `ontology_layered_architecture.*`, `ontology_reasoning_boundary.*` | ontology_and_gnn.md, `src/app/ontology/README.md` |
| `profitability_architecture.svg`, `profitability_decision_flow.svg`, `profitability_dynamic_exit.svg`, `profitability_before_after.svg` | decision_and_risk.md, validation.md |
| `entry_blockade_chain.svg` | decision_and_risk.md §10 (`scripts/gen_entry_blockade_diagram.py`) |

## 코드 옆에 사는 문서

| 위치 | 내용 |
| --- | --- |
| [../src/app/ontology/README.md](../src/app/ontology/README.md) | TTL 파일 구성과 온톨로지 안전 확장 규칙 |
| [../packaging/raspberrypi/README.md](../packaging/raspberrypi/README.md) | Pi 패키지 파일 맵과 빠른 명령 |
| [../config/secrets/README.md](../config/secrets/README.md) | secrets 파일 규약 |

## 읽는 순서

1. 처음이라면 [../README.md](../README.md) → [architecture.md](architecture.md)
2. 왜 거래가 안 나가는지 알고 싶다면 **`/account` 상단 ENTRY BLOCKADE 패널** (또는 `GET /api/realtime-trading/entry-blockade`)을 먼저 보세요. 처음 막힌 단계가 원인입니다 → 배경은 [decision_and_risk.md](decision_and_risk.md) §10 → 세부는 [live_trading.md](live_trading.md)의 "조용한 사이클 읽기"
3. 추론 구조를 이해하려면 [ontology_and_gnn.md](ontology_and_gnn.md)
4. **숏 전략이 왜 거래하지 않는지** 알고 싶다면 `GET /api/short-strategies/status`를 먼저 보고,
   배경은 [short_selling_deployment.md](short_selling_deployment.md). 짧은 답: 3개 전량 `SHADOW`이며
   그것이 정상 상태입니다.
5. 어떤 주장이 증거를 갖고 있는지 확인하려면 [validation.md](validation.md)

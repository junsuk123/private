# Documentation Index

이 디렉터리는 현재 코드 기준의 운영 문서입니다. 최상위 개요는 [../README.md](../README.md)를 먼저 보고, 세부 운영/설계는 아래 문서를 따라가면 됩니다.

![Current runtime architecture](diagrams/system_overview.svg)

## 현재 런타임 계약

`run.ps1` 런타임은 KIS live-capable realtime 시스템입니다. 서버 시작 시 KIS 실시간 수집, read-only 실계좌 확인, 주기적 live short-horizon 학습, 독립 자동거래 루프가 자동으로 시작될 수 있습니다. OpenVINO/NPU는 숫자 evidence scoring에만 사용되며, 실패하거나 장치가 없으면 CPU로 fallback합니다.

최종 주문 권한은 항상 CPU 제어 경로에 남아 있습니다.

- `SharedLiveDecisionEngine`: SELL/REDUCE를 BUY보다 먼저 평가
- `TradingCostEngine`: 수수료, 세금, slippage, spread, net return 검사
- `PrincipalProtectionEngine`: 원금 보호와 drawdown 예산
- `RiskManager` / FinalTradeGate: 최종 승인/거부
- `LiveExecutionCoordinator`: KIS limit order 제출, open order keep/amend, idempotency

## GUI 문서

- Web 운영 화면: `GET /account`
  - 계좌, 현금, 외화, 보유종목, 실현/평가손익, 자산 히스토리, 자산 배분
  - 자동거래 cycle, SELL/BUY 평가 흐름, 제출/정정/차단 현황, 최근 보류 사유
  - 종료 버튼은 신규 BUY를 먼저 비활성화하고, 가능한 profit-seeking SELL 주문을 제출한 뒤 종료를 예약
- Root 연구 화면: `GET /`
  - 연구 refresh, operation mode, paper/mock trading, live snapshot, diagnostics, ontology graph
- Pi/LCD 화면: `GET /display`
  - 사람이 읽기 쉬운 trade-reason 카드 보드
- 온톨로지 전체 화면: `GET /display/ontology`
  - ontology graph payload를 시각화하는 키오스크/전체화면용 화면

관련 API:

- `GET /api/account/dashboard`
- `GET /api/account/asset-history`
- `GET /api/realtime-trading/status`
- `POST /api/live-trading/terminate`
- `GET /api/trade-explanations`
- `GET /api/ontology/graph`
- `GET /api/realtime/runtime`
- `GET /api/npu/runtime`

## Core Architecture

- [architecture.md](architecture.md): 런타임 모듈, API surface, operation mode, deterministic safety boundary
- [system_algorithm_analysis.md](system_algorithm_analysis.md): `src/app` 알고리즘별 구현 맵
- [data_environment_separation.md](data_environment_separation.md): realtime-only data layout, synthetic-data rejection
- [realtime_short_horizon_policy.md](realtime_short_horizon_policy.md): 실시간 학습/추론/준비도 정책

## Live Trading

- [live_trading_setup.md](live_trading_setup.md): 설치, secrets/config, readiness dry-run, arming
- [live_trading_runbook.md](live_trading_runbook.md): 운영 절차, 대시보드, stall 진단, 종료, emergency stop
- [live_trading_safety_gates.md](live_trading_safety_gates.md): 제출/BUY/SELL 게이트와 rejection code
- [small_account_loss_sell_fix_report.md](small_account_loss_sell_fix_report.md): small-account loss churn guard와 현금 분해
- [live_short_horizon_model_decision.md](live_short_horizon_model_decision.md): live ML 모델은 advisory-only라는 결정 기록

## Strategy And Features

- [short_term_trading_strategy_design.md](short_term_trading_strategy_design.md): 단기 전략 연구/게이트 설계
- [semantic_feature_engine.md](semantic_feature_engine.md): semantic feature, indicator routing, LLM classification, no-lookahead
- [semantic_feature_codebase_analysis.md](semantic_feature_codebase_analysis.md): semantic feature 통합 분석
- [theory_aware_ontology_voting.md](theory_aware_ontology_voting.md): ontology triple을 theory vote로 변환하는 경로

## Ontology

- [ontology_standardization_report.md](ontology_standardization_report.md): RDF/RDFS/OWL + SHACL additive layer
- [ontology_migration_audit.md](ontology_migration_audit.md): custom graph에서 표준 온톨로지로의 mapping
- Diagrams:
  - [diagrams/ontology_framework.svg](diagrams/ontology_framework.svg)
  - [diagrams/ontology_layered_architecture.svg](diagrams/ontology_layered_architecture.svg)
  - [diagrams/ontology_reasoning_boundary.svg](diagrams/ontology_reasoning_boundary.svg)
  - [diagrams/ontology_standardization_components.svg](diagrams/ontology_standardization_components.svg)
  - [diagrams/ontology_migration_beforeafter.svg](diagrams/ontology_migration_beforeafter.svg)

## Acceleration

- [npu_runtime_architecture.md](npu_runtime_architecture.md): CPU/NPU split, environment controls, fallback behavior
- [npu_optimization_audit.md](npu_optimization_audit.md): vectorized screening, Rust/PyO3 native core, rolling cache
- [npu_benchmark_results.md](npu_benchmark_results.md): CPU vs NPU benchmark snapshots

## Raspberry Pi

- [raspberry_pi_deployment.md](raspberry_pi_deployment.md): CPU-only Pi install/run/service/kiosk guide
- [../packaging/raspberrypi/README.md](../packaging/raspberrypi/README.md): Pi package file map and quick commands

Pi 기본값은 read-only, CPU-only입니다. `/account`는 LAN 브라우저용이고, `pi-dashboard-launch.sh`는 attached LCD용 `/display` 키오스크를 띄웁니다.

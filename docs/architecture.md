# Architecture

현재 코드 기준의 런타임 구조 문서입니다. 개념 설계가 아니라 `src/app` 아래에서 실제로 실행되는 경로를 기술합니다.

![Current runtime architecture](diagrams/system_overview.png)

## 1. 원칙

시스템은 **확률적 추론**과 **결정론적 통제**를 분리합니다.

- 온톨로지는 도메인 관계와 허용 전략을 결정하고, strategy-utility GNN은 그 관계 가중치와 전략 효용을 학습하여 실제 전략 선택에 직접 참여합니다.
- GNN 선택은 실시간 모델 보정과 전략별 양수 순효율 검증을 통과한 경우에만 신규 진입 권한을 얻습니다. LLM과 NPU 장치 자체에는 주문 권한이 없습니다.
- 주문을 만들 수 있는 것은 `RiskManager`를 통과한 `FinalOrder`뿐이고, 실제 제출은 `LiveExecutionCoordinator`의 limit order 경로만 가능합니다.
- OpenVINO/NPU는 숫자 evidence 가속일 뿐이며, 실패하면 동일 스키마로 CPU fallback합니다. NPU 출력은 주문 권한이 아닙니다.

데이터가 어떤 단계를 거쳐 어떤 판단에 쓰이는지는 [data → process → decision map](diagrams/data_to_decision_flow.svg)에, 온톨로지/GNN 계층 구조는 [ontology and GNN layers](diagrams/ontology_gnn_layers.svg)에 정리되어 있습니다.

## 2. 실행 진입점

| 진입점 | 역할 |
| --- | --- |
| `run.ps1` | Windows 표준 런처. 포트 `8010` 고정, live 프로세스 플래그 설정, 관리 브라우저 창 연결, 창 종료 시 서버 종료 |
| `run.py` → `app.run:main` | `src`를 `sys.path`에 넣고 startup 점검 후 `uvicorn app.web:app` 기동 |
| `packaging/raspberrypi/run.sh` | Pi CPU-only 런처 (read-only 기본값) |
| `scripts/*.py` | readiness 점검, dry-run, 학습, 리플레이, 벤치마크, arming/disarming |

`run.ps1`이 프로세스 환경에 **강제로** 설정하는 값(`Set-RunEnv`):

```text
TRADING_MODE=live_trading      LIVE_TRADING_ENABLED=true
KIS_LIVE_ENABLED=true          KIS_PAPER_TRADING=false
LIVE_ORDER_SUBMIT_ENABLED=true REQUIRE_MANUAL_ARMING=false
REALTIME_BUY_ENABLED=true      LIVE_TERMINATION_SELL_ONLY_ON_START=false
```

즉 **Windows 기본 실행은 실주문 제출이 가능한 상태**입니다. 안전성은 flag를 끄는 방식이 아니라 계좌/시세 신뢰도, 비용·리스크 게이트, KIS health check에서 확보합니다. 자세한 조작은 [live_trading.md](live_trading.md)를 참고하세요.

기타 startup 서비스(`Set-DefaultEnv`이므로 override 가능):

- `AUTO_START_KIS_REALTIME_COLLECTOR` — KIS 실시간 체결/호가 수집
- `AUTO_START_REALTIME_TRADING` — 독립 실시간 자동거래 루프
- `AUTO_START_LIVE_TRAINING` — live short-horizon 모델 주기 재학습 (`LIVE_TRAINING_INTERVAL_SECONDS=60`)
- `AUTO_START_LIVE_WORKER` — live refresh 워커 (`LIVE_REFRESH_SECONDS=15`)

## 3. 런타임 흐름

```text
KIS 실시간 체결/호가 (WebSocket) + KIS REST 계좌·주문·해외시세 + 브로커 quote/FX
  + 공개 시장/거시/뉴스/공시 데이터
  → source_policy 신뢰도·신선도·provenance 판정, synthetic/sample 차단
  → RealtimeMarketDataStore / LocalResearchStore / AccountSnapshotStore (SQLite)
  → IndicatorEngine · microstructure feature · LiveFeatureFrame(live_short_horizon_v2)
  → [자문] TechnicalPredictionEngine (regime + 방법론 + 보수적 expected exit price)
  → [자문] 거시–미시 온톨로지: MacroMarketReasoner → MicroSymbolReasoner(병렬)
            → OntologyCoordinator → GlobalTradeArbiter (SELL/REDUCE를 BUY보다 먼저 랭킹)
  → [자문] RDF/RDFS/OWL semantic label + SHACL 검증 + SemanticPolicyScorer
  → [자문] live short-horizon 모델 (CPU/OpenVINO, auxiliary-only)
  → ClosedWorldOntologyGate(허용 전략 마스크)
  → strategy-utility R-GCN(조건부 승/패 기대값과 제비용 차감 순효율)
  → GnnRealtimeTrustEvaluator
        1) 모델 보정(calibrated) 검증
        2) 전략별 forward 실현 순효율(entry_authorized) 검증
  → StrategySessionManager(종목·전략 단일 소유권 잠금)
  → SharedLiveDecisionEngine
        1) SELL/REDUCE 평가
        2) BUY 평가 (현금·신선도·spread·유동성·근거 충족 시에만)
  → TradingCostEngine + ProfitabilityGate + DynamicExitPolicy
  → PositionSizer + PrincipalProtectionEngine + RiskManager + FinalTradeGate
  → order_pricing_policy + exchange_resolver → LiveExecutionCoordinator
  → KIS limit order · order journal · status polling · 계좌 재동기화
  → /account · /display 대시보드 · audit log · 재학습 artifact
```

## 4. 모듈 지도

| 경로 | 역할 |
| --- | --- |
| `src/app/web.py` | FastAPI 앱, 루트 GUI, API, realtime/live runtime orchestration |
| `src/app/web_account_routes.py`, `account_dashboard.py` | `/account` 라우트와 payload 구성 |
| `src/app/data/` | KIS 실시간 수집, 이벤트 파이프라인, realtime store, source policy, **세션·capability 단일 권한 (`market_capabilities.py`)**, REST fallback |
| `src/app/features/` | indicator engine, live feature frame, short-horizon/semantic feature, provenance |
| `src/app/technical/` | 근거 기반 기술적 예측 레이어 (regime, 방법론, 예측, 리플레이) — 자문 전용 |
| `src/app/graph/` | custom KnowledgeGraph, FactTable, RDF/OWL/SHACL 레이어, 거시–미시 추론, theory vote, NPU evidence scorer |
| `src/app/ontology/` | TTL 스키마와 closed-world 운영 게이트, trading domain reasoner |
| `src/app/models/` | 라벨링, 학습 파이프라인, artifact registry, CPU/OpenVINO backend, strategy-utility R-GCN |
| `src/app/strategy/`, `routing/` | 카탈로그 전략 expert(롱 13 + 숏 3), 소유권 가드, strategy router, GNN 실시간 신뢰도, shadow comparison |
| `src/app/cost/`, `risk/` | TradingCostEngine, ProfitabilityGate, position sizing, principal protection, RiskManager |
| `src/app/trading/` | realtime engine, shared decision engine, dynamic exit policy, execution policy, runtime guard |
| `src/app/trading/directional.py` | 방향 축 계약: `PositionDirection`/`PositionEffect`/`ExecutionProduct`/`StrategyDeploymentState`, `DirectionalStrategyKey`, 방향별 exit geometry, 전이 화이트리스트 |
| `src/app/trading/borrow.py` | 대주 locate의 point-in-time 저널과 공유 admissibility 규칙(`evaluate_borrow`), 연율→거래별 안분 |
| `src/app/trading/borrow_source.py` | 대주 가용성 소스 인터페이스. 기본값은 명시적 "소스 없음"; 파일/callable 구현 |
| `src/app/trading/borrow_polling.py` | 수요 기반·예산 제한 폴링으로 저널을 채움 |
| `src/app/trading/shadow_evaluation_service.py` | shadow plan을 심볼별로 채점해 승격 증거로 전환 |
| `src/app/features/short_indicators.py` | 계산 가능한 숏 보조 지표 + 소스 없는 지표의 명시적 보고 |
| `src/app/trading/directional_shadow.py` | forward shadow 평가: plan 동결, bid/ask 체결 시뮬레이터, 3중 누수 방어 |
| `src/app/trading/short_strategy_promotion.py` | 배포 사다리 authority: confidence score, hard gate, 자동 승격/강등, 원자적 상태+audit 기록 |
| `src/app/ontology/short_rules.py` | 숏 closed-world 사실 평가와 레짐별 방향 허용 마스크 |
| `src/app/execution/` | KIS adapter(국내/해외), 주문 가격 정책, 거래소 라우팅, live coordinator, 저널, idempotency |
| `src/app/backtesting/`, `evaluation/` | 이벤트 시뮬레이터, purged walk-forward, reality check, counterfactual 평가 |
| `src/app/storage/` | local store, lifecycle store, model store, 마이그레이션 |
| `config/` | 수익성·청산·사이징·비용·리스크·거시미시·기술예측 정책과 런타임 프로파일 |
| `packaging/raspberrypi/` | Pi CPU-only 설치/실행/서비스/키오스크 스크립트 |

## 5. 저장소 레이아웃

```text
data/store/realtime_market_data.sqlite3   실시간 체결·호가·분봉
data/store/research.sqlite3               정규화된 리서치 레코드
data/store/account_dashboard.sqlite3      계좌/자산 히스토리
data/store/trading-lifecycle.sqlite3      전략 인스턴스·포지션·TradePlan
data/store/strategy-performance.sqlite3   실현 outcome (v2: direction/execution_product/evaluation_source 컬럼)
data/store/strategy-deployment.sqlite3    arm별 배포 상태 + promotion_audit (한 트랜잭션)
data/store/borrow-snapshots.sqlite3       대주 locate append-only 저널 (과거 시점 조회용)
data/store/directional-shadow.sqlite3     shadow plan과 채점된 forward outcome
data/store/causal-order-journal.jsonl     intent → verdict → order 인과 저널
data/models/<family>/                     버전 artifact + <model>.latest.json
data/reports/                             readiness, 벤치마크, counterfactual 리포트
logs/                                     서버 로그, live-orders.jsonl, feature frame 저널
```

`LocalResearchStore`는 `(kind, record_key)` 기반 dedup과 `RESEARCH_RETENTION_DAYS` 정리를 수행하고, synthetic/simulated 레코드를 거부합니다. `ModelArtifactStore`도 simulated artifact를 거부합니다. audit logger는 credential/token/계좌번호/broker secret을 재귀적으로 마스킹합니다.

## 6. API 표면

운영에서 실제로 보는 것:

- `GET /account` — 운영 대시보드 (계좌, 현금, 보유, 손익, 자산 히스토리, 판단 흐름, 거부 사유)
- `GET /display` — Pi LCD용 trade-reason 카드 보드
- `GET /display/ontology` — 온톨로지 그래프 전체 화면
- `GET /` — 연구/진단/수동 실행 화면

주요 API:

| 엔드포인트 | 내용 |
| --- | --- |
| `GET /api/account/dashboard` | 계좌/현금/보유/손익 payload |
| `GET /api/account/asset-history?range=1D\|1W\|1M\|3M` | 분 단위 총자산 히스토리 |
| `GET /api/account/technical` | 기술적 예측 패널 (자문) |
| `GET /api/account/macro-micro` | 거시–미시 온톨로지 패널 (자문) |
| `GET /api/realtime-trading/status` | 실시간 엔진 상태, 최근 이벤트, 거부 사유 분포 |
| `GET /api/refactor/dashboard` | 전략 소유 경로의 프로파일·게이트·shadow 상태 |
| `GET /api/refactor/market-view?symbol={symbol}&limit=180` | 로컬 차트/마켓 뷰 |
| `POST /api/live-trading/terminate?shutdown=true` | BUY 차단 → 청산 SELL 제출 → 서버 종료 예약 |
| `GET /api/trade-explanations` | `/display`용 사람이 읽는 판단 카드 |
| `GET /api/ontology/graph`, `/api/ontology/runtime` | 그래프 payload, 온톨로지 런타임 상태 |
| `GET /api/realtime/runtime`, `/api/npu/runtime` | 가속·모델·NPU 모듈 상태와 fallback 사유 |
| `GET /api/ai/validation`, `/api/live-training/status` | 이벤트 LLM/모델/학습 검증 상태 |
| `GET /api/gnn/realtime-trust` | GNN 전체/전략별 보정 점수, forward 표본, 진입 권한 |
| `GET /api/system-diagnostics` | 서버·계좌·시장 데이터·온톨로지·학습·거래 엔진 통합 진단 |
| `GET /api/auto-reliability/status` | `learning` ↔ `live_trading` 자동 전환 점수와 사유 |
| `GET /api/short-strategies/status` | 숏 arm별 배포 상태, 실주문 권한, confidence, 대주 건강도 |
| `GET /api/short-strategies/{id}/validation` | 전체 지표·임계값과 **실패한 hard gate 전량** (승격까지 남은 조건) |
| `GET /api/short-strategies/{id}/deployment-history` | 자동 승격/강등 audit 이력 (변경 당시 지표 포함) |
| `GET /api/short-strategies/{id}/shadow-outcomes` | forward shadow outcome (실행 불가 신호 포함) |
| `GET /api/short-strategies/entry-blockade` | 숏 진입을 **가장 먼저** 막은 단계 |
| `GET /api/borrow/{symbol}/availability`, `/api/borrow/health` | 대주 locate와 데스크 건강도 |
| `GET /api/directional-bandit/evaluations` | LONG vs SHORT vs NO_TRADE 보수적 엣지 비교 |
| `POST /api/short-strategies/{id}/suspend` | 수동 중단. **승격 endpoint는 존재하지 않음** |

`/api/refactor/market-view`의 LIVE DECISION ONTOLOGY source 노드는 고정 플래그가 아니다.
`event_feed`는 뉴스 저장소 조회 건수, `cross_section`은 shadow 판단에 보존된
`sector_rank_table`과 대상 종목 순위, `session_context`는 시계 사실과 가격 사실을 각각
측정한다. 시계는 계산 가능하지만 30분 가격 구간이나 과거 변동성 표본이 부족하면
`session_context.status=PARTIAL`이며, UI도 이를 `DATA MISSING`과 구분한다.

KRX 세션 가격 사실의 단일 계산 권한은 `app.features.session_structure`이다. 라이브 frame과
stored counterfactual은 모두 이 모듈의 30분 시초 범위, 전일 종가 기준 첫 30분 수익률,
과거 세션만을 사용한 변동성 percentile을 호출한다. 15:20~15:30 종가 단일가 매매는
연속매매 구간으로 취급하지 않는다.

## 7. 운영 모드

`src/app/realtime/mode_manager.py`:

| 모드 | 의미 |
| --- | --- |
| `learning` | 실시간 수집 + 실현손익 라벨 artifact 갱신 |
| `testing` | 하위 호환 legacy paper replay |
| `paper_trading` / `paper_trading_test` | KIS 모의 API 점검 + 로컬 paper 흐름 |
| `live_readiness` / `live_trading_test` | KIS 인증/계좌 read-only 점검, 주문 없음 |
| `live_trading` | 실시간 KIS 자동거래 루프 (모든 게이트 필수) |

모든 모드가 동일한 realtime store와 model root를 사용하며 synthetic 데이터는 입력으로 허용되지 않습니다.

별도로 `config/refactor_profile.json`이 비교/진단 경로의 posture를 선언합니다. 실행 시 실제 주문 가능 여부는 이 파일 하나가 아니라 `run.ps1`의 live 플래그, 자동 신뢰도 컨트롤러, GNN 전략별 진입 권한, 전략 세션, 최종 비용·리스크·KIS 게이트의 합성 결과로 결정됩니다. legacy/ontology/GNN 비교 로그는 계속 shadow comparison으로 남지만, 검증된 GNN 판단은 `StrategySessionManager`를 통해 실거래 전략 선택에 연결됩니다.

런타임 플래그는 `RefactorFeatureFlags.from_env()`가 환경변수에서 읽습니다. 기본값은 legacy 경로만 활성이며, 잘못된 조합(예: ontology router 없이 GNN rerank)은 로드 시점에 실패합니다.

## 8. 가속 경계

| 단계 | 장치 | 비고 |
| --- | --- | --- |
| 데이터 수집·저장 | CPU | broker/news/공시/거시/저장소 |
| 하드 필터 | CPU | 거래정지·관리종목·유동성·데이터 무효 결정론적 거부 |
| 후보 evidence 스코어링 | NPU + CPU fallback | `OntologyNpuLinearScorer`, `trading_pipeline._rank_accepted_with_npu` |
| theory vote / conflict / evidence cluster | NPU + CPU fallback | `graph/npu_*` 모듈, `/api/npu/runtime`에 상태 노출 |
| short-horizon 예측 | CPU 기본, OpenVINO 선택 | 모델은 auxiliary-only |
| strategy-utility R-GCN | OpenVINO CPU(실시간 검증) / NPU(미승격) | 온톨로지 마스크 안에서 전략 선택; 진입 권한은 CPU 실시간 forward 검증 |
| 그래프 탐색·설명·행동 선택 | CPU | 분기 많고 설명 가능성 필수 |
| 비용·리스크·주문 게이트·브로커 제출 | CPU | 안전 임계 경로 |

Windows 성능 카운터는 이 워크로드의 NPU 사용을 별도 엔진으로 표시하지 않을 수 있습니다. OpenVINO device discovery와 `/api/*/runtime` 상태를 기준으로 판단하세요. 세부는 [ontology_and_gnn.md](ontology_and_gnn.md)에 있습니다.

## 9. 안전 모델

- LLM 또는 LLM-like 컴포넌트는 주문을 실행하지 않습니다.
- NPU/OpenVINO 출력은 숫자 근거 점수이며 주문 승인 권한이 아닙니다.
- live short-horizon 모델은 기본 advisory-only입니다. `REALTIME_MODEL_AUXILIARY_ONLY=true`면 모델 단독 BUY는 거부됩니다.
- synthetic/sample/hash 파생 데이터는 오프라인 fixture 전용이며 paper/live 판단 근거로 쓰이지 않습니다.
- margin, leverage, derivatives, short selling, credit loan, leveraged ETF는 거부 대상입니다.
- live 주문은 `LiveExecutionCoordinator`를 통해 limit order로만 제출됩니다.
- audit log는 credential/token/계좌번호/broker secret을 재귀적으로 마스킹합니다.

## 10. 알려진 구조적 한계

정직하게 기록해 둡니다.

1. `app.web`이 UI·orchestration·수집·계좌 접근·live 실행을 함께 들고 있어 latency 격리와 재시작 복구가 어렵습니다.
2. 해외 종목은 REST polling(기본 12초)에 의존하므로 fast-path 불변식을 만족하지 않습니다.
3. legacy 저널과 causal strategy-owned 저널이 함께 남아 있어 복구 시 두 경로의 상태를 함께 대조해야 합니다.
4. decision/feature JSONL 로그가 수백 MB 규모까지 자라며 문서화된 retention/compaction 정책이 없습니다.
5. 소액 계좌 단타는 왕복 비용을 고려하면 구조적으로 기대값이 음수에 가깝습니다. 게이트는 엔지니어링 통제일 뿐 수익을 보장하지 않습니다.
6. 이벤트·횡단면 상대강도·갭 전략은 해당 point-in-time 사실이 실시간 feature schema에 없으면 closed-world로 차단됩니다. 결측 사실을 합성해 0이 아닌 적합도를 만들지 않습니다.
7. 숏 전략 3개는 코드에 존재하지만 **전량 `SHADOW`이며 실주문 권한이 없습니다.** 승격에는 최소 20 거래일의 forward 표본이 필요하므로 정상 상태입니다. 또한 KIS 대주 조회 TR id와 응답 필드명은 실계좌 응답으로 검증되지 않았고(read-only 경로), GNN 방향별 utility head는 미구현입니다 — 숏 arm은 현재 realized posterior와 규칙 신호로만 평가됩니다. 상세는 [short_selling_deployment.md](short_selling_deployment.md) §15.

관련 문서: [ontology_and_gnn.md](ontology_and_gnn.md) · [decision_and_risk.md](decision_and_risk.md) · [live_trading.md](live_trading.md) · [short_selling_deployment.md](short_selling_deployment.md) · [validation.md](validation.md)


## 세션·거래소 capability 계층 (국내·미국 전 세션)

세션 판정의 **유일한 source of truth** 는 `src/app/data/market_capabilities.py` 의
`MarketSessionService` 다. 이전에는 같은 판정이 `market_session.py`, `web.py`(2곳),
`realtime_trading_engine.py`, `kis_real.py`, `kis_realtime.py`, `llm_classifier.py` 에
독립적으로 존재하고 경계값이 서로 달랐다 (`docs/realtime_session_gap_analysis.md` §3).

정보는 두 층으로 분리된다:

| 층 | 위치 | 내용 | 변경 가능성 |
|---|---|---|---|
| 검증된 API 사실 | `market_capabilities.py` 의 `_VERIFIED_*` 테이블 | TR ID, 엔드포인트, `EXCG_ID_DVSN_CD`, `ORD_DVSN` 허용값, subscription key 형식 | **설정으로 덮어쓸 수 없다** |
| 거래소 규정 / 우리 정책 | `config/market_sessions.yaml` | 세션 시각창, 휴장·조기폐장 스냅샷(버전 기록), 세션별 정책 임계값 | 설정 |

검증 근거는 `research_notes/한국투자증권_오픈API_전체문서_*.xlsx` 이고 정리본은
`docs/kis_market_session_capability_matrix.md` 다.

주요 구성 요소:

* `MarketCapability` — (market, venue, session) 조합의 완전한 능력 서술.
  `data_available` / `trade_available` / `new_entry_allowed` / `exit_allowed` 를 **분리**한다.
* `KisSessionOrderRouter` (`src/app/execution/kis_session_order_router.py`) — 주문 route
  결정의 유일한 지점. 정정·취소는 원주문 route family 를 유지한다.
* `FeedMetadata` (`realtime_types.py`) — 모든 실시간 이벤트에 market/venue/session/
  feed_scope/TR/subscription_key 를 부착. `stream_id` 가 분 bar identity 와 `record_id` 에
  들어가 KRX·NXT·통합 피드가 구조적으로 섞이지 않는다.
* 실시간 저장소 스키마 v2 — `schema_version` / `schema_migrations` 테이블과 무손실
  versioned migration. v1 시절 행은 `metadata_inferred=1` 로 표시되어 신규 진입 근거에서 제외된다.

`market_session.py` 는 PRE/REGULAR/AFTER/CLOSED 4단계를 쓰는 기존 호출자를 위한
backward-compatible wrapper 로 축소됐다. 신규 코드는 `SessionId` 와 `MarketCapability` 를 직접 쓴다.

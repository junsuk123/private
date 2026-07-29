# Ontology and GNN

지식 표현(온톨로지)과 학습 기반 전략 효용 추정(GNN)이 어떻게 쌓여 있고, 어디까지 권한을 갖는지에 대한 문서입니다.

![Ontology and GNN layers](diagrams/ontology_gnn_layers.svg)

## 1. 왜 하이브리드인가

![Reasoning boundary](diagrams/ontology_reasoning_boundary.svg)

```text
OWL / RDFS  → 클래스·속성 계층, domain/range 타이핑, semantic 분류, disjoint 일관성
SHACL       → closed-world 데이터 품질/라이브 준비도 검증
Python      → support/contradiction/risk/confidence 점수, 랭킹, 임계값
Engines     → TradingCostEngine, PrincipalProtectionEngine, position sizing
RiskManager → 유일한 최종 실행 게이트
```

- **OWL은 open-world**입니다. 사실이 없다는 것은 *거짓*이 아니라 *모름*입니다. 따라서 OWL은 거래를 막지도 허가하지도 않습니다. risk assertion이 없다는 사실이 안전을 뜻하지 않습니다.
- **SHACL은 closed-world**입니다. "없으면 무효"가 맞는 곳(필수 필드, live 후보의 stale/synthetic 데이터)에만 씁니다.
- **숫자는 전부 Python이 소유**합니다. 점수·가중치·비용·세금·슬리피지·원금보호 금액을 OWL 공리로 인코딩하지 않습니다.
- 추론된 `tr:TradeEligibleAsset`이나 `tr:BuyCandidate`는 *semantic label*이며, 실행 판단은 여전히 RiskManager가 합니다.

## 2. 온톨로지 계층

### L0 — 사실과 근거

KIS 실시간 체결/호가/세션, 계좌·현금 스냅샷, 브로커 quote, 뉴스/RSS/DART 이벤트, 거시·섹터 스냅샷이 입력입니다. 모든 assertion은 `ev:{evidence_id}` `tr:EvidenceItem`에 연결되며 source name, source type, timestamp, data-quality score, synthetic flag, stale flag, confidence, analysis-cycle id를 갖습니다. RDF reification이나 RDF-star는 쓰지 않습니다 (OWL RL 친화성 유지).

관련: `app.data.source_policy`, `app.features.feature_provenance`

### L1 — 작업용 트리플 스토어

`app.graph.knowledge_graph.KnowledgeGraph`는 `(subject, predicate, object, evidence_id)` 문자열 튜플의 in-memory 스토어입니다. 핫 경로 조회는 `app.graph.fact_table.FactTable`이 담당합니다. 문자열 대신 정수 id(`FactDictionary`)와 유효 구간·플래그 비트를 쓰는 compact 표현이며, subject/predicate 인덱스로 조회를 가속합니다. 결과는 문자열 그래프와 동일하고 `to_human_readable()`로 되돌릴 수 있습니다.

### L2 — RDF assertion graph

`app.graph.rdf_graph` / `rdf_adapter`가 내부 레코드와 custom 트리플을 rdflib으로 투영합니다.

| Prefix | IRI | 용도 |
| --- | --- | --- |
| `tr:` | `https://example.com/ontology/trading#` | 스키마 용어 (클래스, 속성) |
| `res:` | `https://example.com/resource/` | 런타임 인스턴스 (종목, 스냅샷, 후보, 판단) |
| `ev:` | `https://example.com/evidence/` | provenance를 담는 `tr:EvidenceItem` |

인스턴스 IRI는 티커/id를 슬러그화한 **결정론적** 값이라 사이클이 바뀌어도 같은 엔티티는 같은 IRI를 갖습니다. 사이클마다 `rdflib.Dataset`의 named graph로 스코프됩니다.

### L3 — RDFS / OWL 2 RL 클로저 (open world)

- `src/app/ontology/trading_core.ttl` — 클래스와 `rdfs:subClassOf` 계층, object/data property, `rdfs:domain`/`rdfs:range`, `rdfs:subPropertyOf`, `owl:disjointWith` (Approved↔Rejected, Buy↔Sell intent, Eligible↔Forbidden).
- `src/app/ontology/trading_rules.ttl` — `owl:hasValue` restriction 클래스를 target 클래스의 subclass로 선언합니다. OWL RL 규칙 `cls-hv2`로 `x p v` → `x a <restriction>`, `cax-sco`로 `x a <target>`이 도출됩니다.

```turtle
[ a owl:Restriction ; owl:onProperty tr:increasesRiskOf ; owl:hasValue tr:Risk_TradeForbidden ]
    rdfs:subClassOf tr:TradeForbiddenAsset .
```

`app.graph.owl_reasoner`가 스키마 그래프를 파일 mtime 기준으로 캐시해 병합하고 `owlrl` 클로저를 돌립니다. `app.graph.semantic_materializer`가 추론된 트리플을 asserted 트리플과 분리해 반환합니다. `ONTOLOGY_REASONING_PROFILE=rdfs`는 더 싼 클로저, `ONTOLOGY_RDF_LAYER=0`은 레이어 전체 비활성화입니다.

거시/미시 어휘는 `macro_market_ontology.ttl` / `micro_symbol_ontology.ttl`이 `trading_core.ttl`을 import해서 확장합니다.

### L4 — SHACL 검증 (closed world)

`trading_shapes.ttl` + `app.graph.shacl_validator`가 필수 필드, 양수 broker price, live 후보의 stale/synthetic 차단, 계좌/주문 구조, approved-and-rejected 충돌, final-order 전제조건을 검사합니다. `mode="live"`는 차단, `mode="paper"`는 경고입니다. shapes 그래프는 lru-cache됩니다.

### L5 — 운영 전략 게이트 (closed world, fail-closed)

`app.ontology.operational_gate.ClosedWorldOntologyGate`가 거래 허가의 실제 기준입니다. OWL의 open-world 부재가 거래 허가로 둔갑하지 않도록, 검증된 point-in-time 스냅샷을 별도로 materialize합니다.

```python
OperationalFact(name, value, observed_at, valid_from, valid_until, source, confidence)
StrategyGateRule(strategy_id, required_true=(...), ...)
```

게이트 출력:

- ontology snapshot id와 유효 구간
- 허용 전략 ID 집합
- 전략별 hard-block 사유 코드
- soft compatibility score
- source가 연결된 explanation path

**필수 사실이 없으면 그 전략은 hard block**입니다. stale, not-yet-valid, 낮은 confidence, boolean 요구 미충족은 각각 결정론적 사유 코드를 만듭니다.

### L6 — 거시 → 미시 추론

계층형 추론 레이어(`app.graph.*`)는 가격을 직접 예측하지 않습니다. 예측·리스크 신호를 **구조화**하고 **전략을 선택**합니다.

```text
Common Trading Ontology (trading_core.ttl)
        ↓
MacroMarketReasoner   regime, risk level, sector ranking, candidate symbols,
                      allowed / blocked strategies
        ↓  (macro가 BLOCK_BUY면 신규 BUY 미시추론은 생략, 보유 종목은 항상 평가)
ParallelMicroReasoningPool → MicroSymbolReasoner per candidate
                      exit deterioration → freshness gate → technical composite
                      → micro regime → macro 전략 허가 게이트 → execution quality
                      → 양(+)의 expected net return이 있어야 BUY_CANDIDATE
        ↓
OntologyCoordinator   bounded ThreadPoolExecutor, worker timeout, 실패 격리
        ↓
GlobalTradeArbiter    SELL/REDUCE를 BUY보다 먼저 랭킹 (자본 보호 우선)
        ↓
SharedLiveDecisionEngine.consume_bundle  ← 어댑터, 기존 게이트 흐름 그대로
```

| 모듈 | 역할 |
| --- | --- |
| `graph/macro_micro_common.py` | 공유 enum (MarketRegime, MacroRiskLevel, MicroRegime, SelectedStrategy, ExecutionQuality, IntentType)과 사유 코드 |
| `graph/macro_reasoner.py` | 규칙 기반 regime/risk/sector/candidate/전략 허가 |
| `graph/micro_reasoner.py` | 종목별 진입·청산·실행품질·기대 순수익 |
| `graph/ontology_coordinator.py` | macro-first, bounded-parallel micro, timeout/예외 격리 |
| `graph/global_trade_arbiter.py` | 자문 전용 `RankedTradeIntent` 랭킹 |
| `graph/macro_micro_config.py` | `config/macro_micro_ontology.yaml` 로드 (env override, 보수적 fallback) |
| `graph/rdf_adapter.py` | `attach_macro_result_rdf` / `attach_micro_result_rdf` |
| `graph/macro_micro_replay.py` | no-look-ahead 리플레이 검증 |

macro 루프는 느리게(기본 60초), micro는 빠르게(기본 5초) 돕니다. 잘못된 설정값은 보수적 기본값으로 clamp되고 `diagnostics.config_fallbacks`에 기록됩니다.

**macro `BLOCK_BUY`는 게이트/브로커 호출 이전에 BUY를 rejected로 단락시킵니다. 막을 수만 있고 허가할 수는 없습니다.**

### 성능 고려

스키마 그래프는 mtime 캐시, SHACL shapes는 lru-cache, materialization은 현재 후보 유니버스로 스코프됩니다. 사이클당 결과는 한 번 계산되어 frozen `AnalysisContext`에 저장되고, live 런타임에서는 백그라운드 refresh 스레드가 만들기 때문에 API 요청은 캐시된 payload만 읽습니다. 전체 온톨로지에 대한 OWL RL 클로저는 사이클당 수백 ms가 들 수 있어 가장 촘촘한 tick 루프에서는 의도적으로 제외되어 있습니다.

## 3. 전략 효용 GNN

7개 전략 expert(`app.strategy.experts`) 각각에 대해 **종목 × 전략** 단위의 비용 조정 기대효용을 추정하는 fixed-shape temporal R-GCN입니다.

| 전략 ID | 논지 |
| --- | --- |
| `intraday_momentum` | 지속적 주문 흐름 + 시장/섹터 정렬 + 추세 상태 → 지속 |
| `breakout_volume` | 인과적으로 알려진 저항 돌파 + 비정상 거래량 + 우호적 흐름 |
| `vwap_mean_reversion` | 비추세·유동적 구간의 일시적 이탈은 VWAP으로 회귀 |
| `liquidity_shock_reversal` | 신규 정보 없는 기계적 급락은 유동성 정상화 시 부분 회복 |
| `event_momentum` | 신선하고 검증된 중대 정보의 과소반응 |
| `cross_sectional_relative_strength` | 공통 요인 제거 후 섹터 내 최강 종목 |
| `gap_context` | 갭의 지속/메움을 촉매·확인·유동성으로 구분 (한쪽 서브모드만 선택) |

### 텐서 계약

```text
X            [B, T, N_max, F]        node features
A            [B, T, R, N_max, N_max] relation-wise adjacency
node_mask    [B, T, N_max]
strategy_mask[B, N_max, S]
```

토폴로지 구성(스파스화, top-k 이웃, 엣지 히스테리시스)은 CPU에서 이뤄지고 모델 그래프 밖에 남습니다. 그래프는 바뀌어도 **텐서 rank/shape은 바뀌지 않습니다.** 런타임 shadow 체크포인트는 `B1 T1 N1 F12 R1 S7`, OpenVINO 벤치마크 형상은 `B1 T4 N16 F12 R4 S7`입니다.

### 순전파

```text
H_t = ReLU( X_t · W_self  +  Σ_r  A_{r,t} · X_t · W_r ),   H_t ×= node_mask
Z   = Σ_t  w_t · H_t                (인과 시간 풀링, 결정 시점 이후 관측 없음)
raw = Z · strategy_heads            → [B, N, S, 8]
```

`scatter`, `gather-by-index`, 동적 `nonzero/unique`, shape 의존 제어 흐름을 쓰지 않습니다. 내보내는 연산은 MatMul / Add / ReLU / Multiply / Concat / Squeeze로 제한됩니다.

### 헤드와 효용

```text
p_success  = sigmoid(raw0)          gross_bps = raw1 × 25
cost_bps   = softplus(raw2) × 10    MAE_bps   = softplus(raw3) × 15
MFE_bps    = softplus(raw4) × 20    p_fill    = sigmoid(raw5)
holding_s  = softplus(raw6) × 60    uncertainty = softplus(raw7)

net       = gross_bps − cost_bps
utility   = p_success·net − (1 − p_success)·MAE − uncertainty + 0.1·p_fill·MFE
utility   = −∞  where (node_mask ∧ strategy_mask) = 0
no_trade  = sigmoid(no_trade_head · Z),  마스킹된 노드는 1.0
```

`NoTrade`는 오류나 결측 예측이 아니라 **일급 행동이자 학습 라벨**입니다.

### 라우팅

`app.routing.strategy_router.StrategyRouter`가 허용 전략 중 효용 최대를 고르고, 그렇지 않으면 `NO_TRADE`를 반환합니다. 출력은 `StrategyUtilityEvidence`로, ontology snapshot id·feature snapshot id·model version·explanation path를 포함합니다.

`app.routing.orchestrator.StrategyOrchestrator`는 선택된 전략만 활성화하고, 열린 포지션이 있는 심볼에는 새 인스턴스를 만들지 않습니다(`OwnershipGuard`). 포지션은 진입부터 청산까지 `origin_strategy_id`/`strategy_instance_id`가 durable하게 소유합니다.

### Shadow 서비스

`app.routing.shadow_intelligence.ShadowIntelligenceService`가 슬로우 경로를 묶습니다.

1. 이벤트 파이프라인의 in-memory 상태에서 12차원 microstructure 스냅샷을 만듭니다 (`realtime_microstructure_v1`).
2. `ClosedWorldOntologyGate`로 허용 전략을 계산합니다 (`data_fresh`, `tradable`, `allow:<strategy>` 요구).
3. OpenVINO CPU(그리고 선택적으로 NPU)로 효용을 추론합니다.
4. legacy / ontology-only / cpu_gnn / npu_gnn 판단을 `logs/refactor-shadow-comparison.jsonl`에 비교 기록합니다.

체크포인트 provenance 검사가 fail-closed입니다. `input_feature_schema`가 현재 프레임 스키마와 다르거나 `live_authorized`가 아니면 `MODEL_INPUT_SCHEMA_MISMATCH` / `UTILITY_MODEL_NOT_LIVE_AUTHORIZED`로 `NO_TRADE`를 내고 모델 출력을 쓰지 않습니다.

`app.data.event_runtime`은 이 서비스를 WebSocket 수집 루프에 붙이되, 별도 큐와 스레드에서 돌려 콜백을 막지 않습니다. `REFACTOR_ONTOLOGY_ROUTER` 또는 `REFACTOR_GNN_SHADOW`가 켜져 있을 때만 생성됩니다.

### 현재 체크포인트 상태

`data/models/strategy_utility/rgcn_shadow.json` 기준:

```text
method              causal_feature_encoder_plus_ridge_calibrated_heads
input_feature_schema realtime_microstructure_v1
rows / snapshots     38,668 / 5,524   (7개 전략 각 5,524)
config               B1 T1 N1 F12 R1 S7, hidden 16, seed 17
authorization_scope  shadow_inference_only
```

관측 전용입니다. 이 체크포인트는 주문 권한을 부여하지 않으며, 실거래 승격에는 [validation.md](validation.md)의 조건이 필요합니다.

## 4. NPU / CPU 경계

| 단계 | 장치 | 비고 |
| --- | --- | --- |
| 데이터 수집 | CPU | broker, 뉴스, 공시, 거시, 저장 |
| 하드 필터 | CPU | 거래정지, 관리종목, 유동성, 무효 데이터 |
| 후보 evidence 스코어링 | NPU + CPU fallback | `OntologyNpuLinearScorer`, `_rank_accepted_with_npu` |
| evidence cluster 압축 | NPU + CPU fallback | 상관 지표를 투표 전에 압축 |
| theory vote 스코어링 | NPU + CPU fallback | BUY/SELL/HOLD/REDUCE/WATCH 벡터 |
| conflict penalty | NPU + CPU fallback | 라벨은 CPU에 남음 |
| short-horizon 예측 | NPU + CPU fallback | 단기 수익·순양수 확률·불확실성 |
| execution edge 스코어링 | NPU + CPU fallback | 체결/슬리피지/역선택 배치 추정 |
| strategy-utility R-GCN | OpenVINO CPU 검증 / NPU 미승격 | shadow 전용 |
| 그래프 탐색·설명 | CPU | 분기 많음, 설명 가능성 필수 |
| 최종 행동 결정·브로커 실행 | CPU | 안전 임계 |

거시–미시 추론 레이어는 **CPU-only이며 Raspberry Pi 호환**입니다. 재사용하는 기술 예측 엔진 내부의 학습 모델만 OpenVINO/NPU를 쓸 수 있고, 동일 스키마 CPU fallback(`models/inference_backend.py`)이 항상 있습니다.

### 환경 변수

| 변수 | 의미 |
| --- | --- |
| `OPENVINO_DEVICE` | 요청 OpenVINO 장치 (`run.ps1`은 `NPU`) |
| `ONTOLOGY_ACCELERATOR` | 온톨로지 런타임 요청 장치 |
| `ONTOLOGY_NPU_ENABLED` | 후보 스코어링 evidence 경로 활성 (기본 true) |
| `ONTOLOGY_NPU_BATCH_SIZE` | `auto` 또는 512/1024/2048/4096 (`run.ps1`은 4096) |
| `ONTOLOGY_NPU_TOP_K` | 그래프 추론으로 넘길 후보 상한 (기본 50) |
| `NPU_DEVICE_PREFERENCE`, `NPU_MIN_BATCH_FOR_NPU` | 공유 `NpuRuntimeManager` 모듈 설정 |
| `EVENT_CLASSIFIER_PROVIDER` / `_DEVICE` | `keyword`(기본), `openvino`, `llm` |
| `SHORT_HORIZON_PREDICTOR_ENABLED` / `_DEVICE` | 선택적 short-horizon evidence provider |
| `ONTOLOGY_GRAPH_SCOPE` | `candidate_only`, `candidate_and_holdings`, `full_debug` |
| `REFACTOR_GNN_CHECKPOINT` | strategy-utility 체크포인트 경로 |

### Fallback 보고

OpenVINO나 NPU가 없으면 동일 스키마로 NumPy 또는 OpenVINO CPU 스코어링으로 떨어집니다. 이벤트/short-horizon 모델 파일이 없으면 결정론적 keyword/linear baseline으로 떨어집니다. 상태는 명시적으로 노출됩니다.

- `/api/ontology/runtime` — 요청/활성 backend, 사용 가능 장치, fallback 사유
- `/api/realtime/runtime` — 가속 요약 + 온톨로지 NPU 상태 + live 모델 상태
- `/api/npu/runtime` — candidate/theory-vote/conflict/short-horizon/execution-edge 모듈 상태

### 벤치마크

```powershell
python scripts/benchmark_npu_scoring.py --device CPU
python scripts/benchmark_realtime_pipeline.py --device CPU
python scripts/benchmark_npu_theory_voting.py --device NPU
python scripts/benchmark_npu_full_decision_pipeline.py --device NPU
python scripts/benchmark_strategy_utility_openvino.py --iterations 30
python scripts/benchmark_fact_table.py
```

측정 결과와 승격 판정은 [validation.md](validation.md)에 있습니다.

## 5. 온톨로지를 안전하게 확장하기

1. 새 클래스/속성은 `trading_core.ttl`에 `rdfs:subClassOf` / `rdfs:subPropertyOf`와 `rdfs:domain`/`rdfs:range`를 붙여 추가합니다. 가능하면 기존 super-property(`tr:hasSemanticEvidence` 등)를 재사용합니다.
2. semantic 분류는 `trading_rules.ttl`의 `owl:hasValue` restriction subclass 공리로 추가합니다. OWL 2 RL 범위를 유지하고 cardinality나 복잡한 DL은 피합니다.
3. closed-world/데이터 품질 검사는 OWL이 아니라 `trading_shapes.ttl`의 SHACL shape로 추가합니다.
4. 수치 임계값·점수·거래 허가를 OWL에 인코딩하지 않습니다. 그것은 Python 정책 스코어러와 결정론적 엔진의 영역입니다.
5. 새로 emit하는 predicate/object 문자열은 `app.graph.rdf_adapter`에 매핑하고 `tests/test_ontology_*.py`에 테스트를 추가합니다.

TTL 파싱 점검:

```bash
python -c "import rdflib; [rdflib.Graph().parse(f, format='turtle') for f in \
  ['src/app/ontology/trading_core.ttl','src/app/ontology/trading_rules.ttl','src/app/ontology/trading_shapes.ttl']]; print('ok')"
```

런타임에서는 `app.graph.rdf_graph.RdfTradingGraph.serialize(format="turtle" | "json-ld")`가 현재 assertion 그래프를 덤프하고, `app.graph.semantic_materializer`가 추론 트리플을 asserted 트리플과 분리해 반환합니다.

# Strategy Selection V2

전략 선택 경로를 `MarketContext → Ontology Eligibility → Proposal → Utility → Cost →
Bandit → Ranking → NO_TRADE` 로 재편한 리팩터의 문서입니다. **관측 상태(2026-08-11
23:09:33 KST): `SHADOW`, 자동 승격 활성, 실효 live 권한 없음.** Windows `run.ps1`은 V2를
활성화하되 설정상 `shadow_only=true`를 유지하고, 영속 증거 컨트롤러가 검증 결과에 따라
`SHADOW → LIVE_PROBE → LIVE` 권한을 자동 전환합니다. 이 시각 영속 context는 66개이고 실제
전략을 선택한 context는 0개라 아직 legacy 경로가 실주문 선택 권한을 유지합니다. 수치는 live
cycle마다 변하므로 최신값은 §5.2의 API에서 확인합니다.

이 문서의 §1은 리팩터 착수 시점(2026-08-11)에 `main` 의 **실코드**를 import·생성자 주입·호출
사슬로 추적한 결과입니다. README나 생성된 아키텍처 문서를 근거로 쓴 문장은 없습니다.

---

## 1. 리팩터 이전의 실제 선출 경로

```text
run.py
└─ app.run  →  app.web (waitress/Flask)
   └─ web._build_realtime_trading_engine()                      src/app/web.py:8603
      ├─ RealtimeMarketDataStore()
      ├─ SharedLiveDecisionEngine(store, RiskManager(rules), market_refresher)
      ├─ LiveExecutionCoordinator(KisDevelopersApiClient(paper=False, enabled=True))
      ├─ StrategySessionManager(selection_evidence_provider=
      │                         web._strategy_session_selection_evidence)   web.py:8642
      └─ RealtimeTradingEngine(...)                trading/realtime_trading_engine.py:260

RealtimeTradingEngine.run_once()                    realtime_trading_engine.py:480
 ├─ session_open_provider()                         → MARKET_SESSION_CLOSED
 ├─ account_provider()                              → NO_ACCOUNT_SNAPSHOT
 ├─ candidate_symbols_provider()                    → cycle_buy_candidates
 ├─ macro_micro_observer(...)                       → bundle{macro_result, micro_results,
 │                                                            ranked_trade_intents}
 ├─ strategy_session_manager.evaluate(...)   ← 선출                    :637
 ├─ strategy_session_manager.allowed_buy_candidates(...)
 ├─ strategy_supervisor → SOFT/HARD halt
 ├─ SELL 루프  → decision_engine.evaluate_exit_for_holding → _prepare_order_for_execution
 │               → _submit → LiveExecutionCoordinator → KIS
 └─ BUY 루프   → decision_engine.evaluate_buy → _prepare_order_for_execution
                 → _submit → LiveExecutionCoordinator → KIS
```

선출 내부 (`trading/strategy_session.py`):

```text
StrategySessionManager.evaluate                                    :796
└─ _select(candidates, bundle, now)                                :1301
   ├─ evidence = selection_evidence_provider(candidates)
   │    └─ web._strategy_session_selection_evidence                web.py:8642
   │       ├─ LiveFeatureFrameBuilder.build(symbol)          (후보별)
   │       ├─ default_mechanical_shadow_collector().collect(...)
   │       ├─ _refresh_live_candidate_shadow → ShadowIntelligenceService.evaluate
   │       │     ├─ ClosedWorldOntologyGate.evaluate(snapshot, rules)
   │       │     └─ FixedShapeStrategyUtilityModel.infer  (R-GCN, head당 11채널)
   │       ├─ build_refactor_dashboard()["shadow"]["latest_by_symbol"]  ← 저장된 행
   │       └─ row에 mark_price / technical_features / rvgi_box_context 주입
   ├─ _new_entry_session_report(candidates, now)      → 세션 hard gate
   ├─ _intent_proposals(...)     ← 온톨로지 ranked BUY intent,
   │                               catalog.resolve_strategy_id 경유 (METHODOLOGY ALIAS)
   ├─ _evidence_proposals(...)   ← full GNN vector 행 (path=cpu_gnn_validation)
   │    └─ _mechanical_entry_verdict → strategy_algorithms.get_algorithm(id).entry(...)
   ├─ _deduplicate_joint_proposals
   ├─ _journal_shadow_proposals(...)                  ← SHADOW plan
   ├─ 선택기:  _gnn_direct_choice   (STRATEGY_SESSION_GNN_DIRECT_ELECTION=1)
   │           _bandit_choice       (기본값: ConservativeStrategyBandit.select)
   │           executable[0]        (bandit 비활성)
   ├─ _journal_shadow_proposals(counterfactual=True, exclude=winner)
   └─ _arm(winner)                                    → phase=ARMED
```

### 1.1 착수 시점 카탈로그

`STRATEGY_IDS` (`strategy/catalog.py`, append-only, 19개, 인덱스 순서가 계약):

| # | strategy_id | 방향 | 알고리즘 클래스 | `live_authorized` | geometry stop/tp/hold |
|--:|---|---|---|---|---|
| 0 | intraday_momentum | LONG | IntradayMomentumAlgorithm | (게이트 없음 → live) | 60/160/3600 |
| 1 | breakout_volume | LONG | BreakoutVolumeAlgorithm | (게이트 없음 → live) | 60/160/4500 |
| 2 | vwap_mean_reversion | LONG | VwapMeanReversionAlgorithm | (게이트 없음 → live) | 60/160/3600 |
| 3 | liquidity_shock_reversal | LONG | LiquidityShockReversalAlgorithm | 1.0 | 65/170/2400 |
| 4 | event_momentum | LONG | EventMomentumAlgorithm | (게이트 없음 → live) | 75/185/5400 |
| 5 | cross_sectional_relative_strength | LONG | CrossSectionalRelativeStrengthAlgorithm | (게이트 없음 → live) | 60/160/5400 |
| 6 | gap_context | LONG | GapContextAlgorithm | (게이트 없음 → live) | 70/175/4500 |
| 7 | rvgi_box_breakout | LONG | RvgiBoxBreakoutAlgorithm | 1.0 | 60/160/3600 |
| 8 | residual_relative_strength | LONG | ResidualRelativeStrengthAlgorithm | 1.0 | 65/170/5400 |
| 9 | adaptive_anchored_vwap_reversion | LONG | AdaptiveAnchoredVwapReversionAlgorithm | 1.0 | 75/185/3600 |
| 10 | ofi_microprice_exhaustion_reversal | LONG | OfiMicropriceExhaustionReversalAlgorithm | 1.0 | 60/160/1800 |
| 11 | opening_range_breakout | LONG | OpeningRangeBreakoutAlgorithm | 1.0 | 60/160/7200 |
| 12 | market_intraday_momentum | LONG | MarketIntradayMomentumAlgorithm | **0.0 (shadow)** | 60/160/1500 |
| 13 | market_intraday_momentum_short | SHORT | MarketIntradayMomentumShortAlgorithm | 1.0 (arm별 사다리) | 60/180/1500 |
| 14 | opening_range_breakdown | SHORT | OpeningRangeBreakdownAlgorithm | 1.0 (arm별 사다리) | 60/180/3600 |
| 15 | residual_relative_weakness | SHORT | ResidualRelativeWeaknessAlgorithm | 1.0 (arm별 사다리) | 65/190/2700 |
| 16 | bar_confirmed_vwap_recovery | LONG | BarConfirmedVwapRecoveryAlgorithm | 1.0 | 70/175/5400 |
| 17 | overnight_gap_carry | LONG | OvernightGapCarryAlgorithm | **0.0 (shadow)** | 90/205/64800 |
| 18 | range_support_reversion | LONG | RangeSupportReversionAlgorithm | 1.0 (운영자 결정, t=1.56) → **이후 false 로 변경** | 60/160/3600 |

숏 arm 은 `short_strategy_promotion.default_promotion_controller()` 로 상태를 해석하므로
`live_authorized: 1.0` 자체가 주문 권한이 아닙니다.

이 표는 **착수 시점 스냅샷**이고 그 뒤로 갱신하지 않습니다 — §2 이후의 서술이 무엇에 대한
것인지를 고정하는 기준선이기 때문입니다. 스냅샷 이후 이미 두 가지가 변했습니다:

* `range_support_reversion` 이 `config/strategy_algorithms.yaml` 에서 `live_authorized: false` 로
  바뀌어 현재 파생 lifecycle 은 SHADOW 입니다 (§6.3).
* `bar_trend_continuation` 이 index 19 로 **append** 됐습니다 (SHADOW). append-only 계약대로
  기존 인덱스는 하나도 움직이지 않았고, spec·family·soft 관계는 registry 와 ontology 에 함께
  등록됐습니다.

현재 값은 항상 `python scripts/report_strategy_selection_v2.py` 로 확인하세요 — 이 문서의 수치를
현재 상태로 읽지 마세요.

### 1.2 착수 시점의 6가지 구조적 결함

1. **통합 컨텍스트가 없음.** 한 사이클이 시장 상태를 네 번 계산했습니다 — evidence provider의
   `LiveFeatureFrame`, 거시–미시 observer의 `micro_results`, `_mechanical_entry_verdict` 의
   `TechnicalFeatureSet` 재구성, `_bandit_choice` 의 `BanditContext`. 공유 `context_id` 도
   field별 신선도도 없었습니다.
2. **온톨로지 어휘가 실행 어휘와 다름.** hard mask 가 generic methodology 이름에 걸려 있고
   alias 표로 연결됐습니다. 그 표의 주석 자체가 `mean_reversion → vwap_mean_reversion` 을
   "the loosest fit" 으로 기록합니다 — generic 논지는 밴드 중심선으로 회귀하고 카탈로그 전략은
   VWAP 으로 회귀하므로, 온톨로지 판정과 실행 논지가 서로 다른 가설이었습니다.
3. **효용과 비용이 모델 안에서 엉켜 있음.** `rgcn.output_from_raw` 가
   `cost = softplus(raw[...,2]) * 10` 을 예측하고 그것을 utility 에 접습니다. 즉 수수료·세금 정책을
   바꾸면 재학습이 필요하고, `TradingCostEngine` 은 선택 경로의 비용 권한이 아니었습니다.
4. **NO_TRADE 가 bandit 내부 결과.** 시장·horizon별 최소 엣지를 가진 1급 arm 이 아니었습니다.
5. **선택과 소유권이 같은 2,900줄 클래스.**
6. **완전히 없던 것:** coverage 분석, 전략 audit 프레임워크, selector regret, drift 모니터,
   (전략 × 시장 × 레짐)별 trust.

---

## 2. 책임 재배치 (old → new)

| 관심사 | 이전 | 현재 |
|---|---|---|
| 시장 상태 | 사이클당 4회 독립 계산, 공유 id 없음 | `app.context.MarketContext` — (symbol, cycle)당 1개 스냅샷, 모든 산출물에 `context_id`, field별 `FeatureSource` |
| 전략 요구사항 | 각 `entry()` 본문에 암묵적 | `app.strategy.spec.StrategySpec` — 선언적이고 평가가 저렴 |
| hard eligibility | evidence producer 내부 `ClosedWorldOntologyGate`, generic methodology + alias | `app.ontology.strategy_eligibility` — **실제 strategy id** 기준, hard/soft 관계 분리 |
| 전략 판정 | 세션 매니저 내부 `_mechanical_entry_verdict` | `app.strategy.proposal_engine` → `StrategyProposal` (수량·side·venue field 자체가 없음) |
| 효용 | `rgcn.output_from_raw` 가 비용과 효용을 모두 계산 | `app.routing.strategy_utility` — 모델은 gross/downside/duration/uncertainty만, 비용은 `TradingCostEngine`, net 은 항등식 |
| 비용 | 모델 채널 2 | `TradingCostAdapter`. fallback 시 `measured=False` 로 표시 |
| 최종 선택 | `ConservativeStrategyBandit.select` (또는 `_gnn_direct_choice`) | `app.routing.strategy_selector.StrategySelectorV2` — 항별 분해를 전부 저장 |
| bandit | 선택기 자체 | `app.routing.bandit_adapter` — ±20bps 유계 보정, clamp 여부 보고 |
| NO_TRADE | bandit 내부 결과 | `app.routing.no_trade_policy` — 시장·horizon별 최소 엣지, 양수 하단 규칙 |
| 소유권 | 선택과 혼재 | `StrategySessionManager`는 주문 가능한 proposal 소유권을 유지. V2가 권한을 얻어도 그 집합 안에서만 선택 |
| counterfactual | 주문 권한 없는 arm 만 shadow plan | `app.evaluation.counterfactual_engine` — eligible+ready 대안마다 가상 포지션, `context_id` 로 그룹 |
| selector 품질 | 측정 안 됨 | `app.evaluation.selector_regret` / `selector_evaluator` |
| 검증 | 전략별 임기응변, 각기 다른 증거 | `app.strategy_validation` — 단일 runner, purged CV, 비용 스트레스, 파라미터 안정성, 레짐 분해, lifecycle ledger |
| drift | 측정 안 됨 | `app.monitoring.{strategy,context,model}_drift` |

효용은 이제 문자 그대로 명세의 수식이고, **각 항은 그 항만 담당하는 컴포넌트가 산출**합니다:

```text
U_s = M_s * (mu_gross_s - C_s - lambda_d*D_s - lambda_u*sigma_s
             + lambda_o*O_s + lambda_b*B_s)
s*  = argmax(U_NO_TRADE, U_1, ..., U_N)
```

| 항 | 산출자 | 단위 |
|---|---|---|
| `M_s` | `OntologyStrategyMask` | 0 또는 1 |
| `mu_gross_s` | utility predictor | bps (**gross**, 절대 net 아님) |
| `C_s` | `TradingCostAdapter` → `TradingCostEngine` | bps |
| `D_s` | utility predictor | bps |
| `sigma_s` | utility predictor | bps |
| `O_s` | 온톨로지 soft 관계 | `[-1, 1]` → `lambda_o` 가 bps 로 환산 |
| `B_s` | `StrategyBanditAdapter` | bps, ±`max_correction_bps` 로 유계 |

`lambda_*` 는 `config/strategy_selector_v2.yaml` 에 있고 모델·optimizer 안에 hardcode 되지
않습니다.

---

## 3. 새 모듈 지도

| 패키지 | 파일 | 역할 |
|---|---|---|
| `app.context` | `market_context.py` | `MarketContext` + 9개 그룹 (identity/macro/cross_sectional/symbol/price_geometry/microstructure/temporal/event/data_quality), `FeatureSource` |
| | `context_builder.py` | 유일한 생성 지점. IO 없음 → 같은 입력이면 같은 출력 |
| | `macro_context.py`, `cross_sectional_context.py`, `symbol_context.py`, `microstructure_context.py` | 그룹별 빌더 |
| | `context_store.py` | 최근 컨텍스트 LRU (기본 512개 / 3600초). **SQLite 아님** — 실시간 writer 가 sqlite 를 소유 |
| `app.strategy` | `spec.py` | `StrategySpec`, `StrategyFamily`, `StrategyLifecycleState` |
| | `registry.py` | 실코드에서 spec **파생** (§4). lifecycle 권고와 migration flag |
| | `proposal.py` | `StrategyProposal` — 주문이 될 수 없는 구조 |
| | `proposal_engine.py` | mask 이후에만 알고리즘 실행. feature 객체 1개 공유 |
| | `coverage.py` | 6축 버킷팅, coverage gap, 연구 후보 집계 |
| `app.ontology` | `strategy_ontology.py` | hard 8종 / soft 4종 관계 타입, 실제 id 기준 soft 관계 표 |
| | `strategy_eligibility.py` | `M_s(x)` 와 `O_s(x)` |
| `app.routing` | `strategy_utility.py` | 예측 계약, `TradingCostAdapter`, GNN adapter, heuristic predictor |
| | `ontology_strategy_mask.py` | selector 가 온톨로지의 eligibility 이외에 손댈 수 없게 하는 얇은 층 |
| | `bandit_adapter.py` | 유계 posterior 보정 |
| | `no_trade_policy.py` | NO_TRADE 1급 action |
| | `strategy_selector.py` | `StrategySelectorV2`, `RankedStrategyCandidate`, `UtilityWeights` |
| | `selector_v2_shadow.py` | 런타임 관측·반사실 수집·실효 권한 공개 통합점 (§5) |
| | `selector_v2_promotion.py` | 영속 자동 권한 상태 머신, 승격·강등 통계 게이트 |
| `app.evaluation` | `shadow_position.py` | 가상 포지션. `app.execution` import 없음 |
| | `counterfactual_engine.py` | context별 대안 그룹 |
| | `outcome_resolver.py` | LIVE/LIVE_PROBE/SHADOW/BACKTEST 분리와 가중 |
| | `selector_regret.py` | `regret = best_available - selected` |
| | `selector_evaluator.py` | 전략 문제 vs selector 문제 분리 진단 |
| `app.monitoring` | `strategy_drift.py` | rolling 건강도, 1칸 강등 제안 |
| | `context_drift.py` | PSI, 고정 bin |
| | `model_drift.py` | (전략 × 시장 × 레짐)별 trust |
| `app.strategy_validation` | `metrics.py` | 전체 지표, 중첩 표본 할인 |
| | `purged_cv.py` | purge + embargo, CPCV |
| | `walk_forward.py` | anchored/rolling, out-of-sample 안정성 |
| | `cost_stress.py` | break-even cost multiple |
| | `parameter_stability.py` | 절벽 vs 평지 |
| | `regime_breakdown.py` | 구간이 실제로 구별되는지 |
| | `registry.py` | lifecycle 원장, 전이 화이트리스트, 승격 게이트 |
| | `audit_runner.py` | 단일 evaluator, KEEP/FIX/SHADOW_ONLY/RESEARCH/RETIRE |
| `app.config` | `selector_v2_flags.py` | feature flag, 안전 기본값 |

설정: `config/strategy_selector_v2.yaml`, `selector_v2_promotion.yaml`,
`no_trade_policy.yaml`, `bandit_adapter.yaml`, `strategy_validation.yaml`,
`context_features.yaml`, `strategy_registry.yaml`.

---

## 4. spec 은 선언이 아니라 파생이다

`app.strategy.registry` 의 거의 모든 값은 리터럴이 아닙니다:

| spec field | 출처 |
|---|---|
| `strategy_id`, 순서 | `app.strategy.catalog.STRATEGY_IDS` |
| `direction` | `strategy_algorithms.strategy_direction` |
| `horizon_seconds`, `min_liquidity_score`, `max_spread_bps` | 해석된 `AlgorithmConfig` (내장 기본값 < YAML < env) — YAML 수정이 spec 을 움직임 |
| `lifecycle_state` | `strategy_live_authorized` / `strategy_shadow_authorized` |
| `family` | `MACRO_FAMILY_BY_STRATEGY` 와 대조해 **주** 논지로 결정 |

리터럴로 선언된 것은 `required_features` 와 `required_election_inputs` 뿐이고, 이는 각 알고리즘의
`entry()` 본문이 실제로 참조하는 field 를 추출한 것입니다. Python 메서드는 실행하지 않고 자기
요구사항을 보고할 수 없고, mask 를 계산하려고 전 알고리즘을 실행하는 비용이 바로 이 층이 없애려는
비용이기 때문입니다. `tests/test_strategy_spec_registry.py` 가 선언된 모든 이름이
`TechnicalFeatureSet` / `ElectionContext` / `MarketContext` 의 실제 field 인지 검사하므로 rename 이
죽은 요구사항을 남길 수 없습니다.

### 4.1 lifecycle 권고 vs lifecycle 변경

저장된 증거가 현재 authorization 과 모순되는 경우, registry 는 **권고를 기록하고 운영 상태는
건드리지 않습니다.** 적용에는 `STRATEGY_LIFECYCLE_APPLY_RECOMMENDATIONS=1` 이 필요하고,
권고는 상태를 **낮추는 방향으로만** 작동합니다. 코드 읽기로 전략을 LIVE 에 넣거나 빼는 추측이
바로 금지 사항이며, flag 가 그 변경을 감사 가능한 운영자 행위로 만듭니다.

현재 등록된 권고 2건은 §6 에 측정치와 함께 있습니다.

### 4.2 숏 전략

`config/short_strategy_deployment.yaml` 이 `enabled: false` 이고 arm 별로도 전부 비활성입니다
(이 계좌는 대주/공매도를 거래할 수 없습니다 — 파생상품 계좌의 기본예탁금·사전 의무교육 미비).
registry 는 숏 spec 을 **보고하되** (coverage 분석이 "무엇이 없는지" 를 말할 수 있게)
알고리즘 기본값의 `live_authorized: 1.0` 과 무관하게 lifecycle 을 `RESEARCH` 로 고정합니다.

3중 잠금이며 어느 것도 이 문서의 대상이 아닙니다:

1. `short_strategy_deployment.yaml` `enabled: false` → promotion controller 자체가 정지
2. `app.strategy.registry` → 숏 lifecycle `RESEARCH` 고정
3. `app.ontology.strategy_eligibility` → `long_only` 동안 SHORT 방향 hard block

상세는 [short_selling_deployment.md](short_selling_deployment.md).

---

## 5. 런타임에서 V2 의 위치

선택 호출 지점은 **1곳**: `StrategySessionManager._observe_selector_v2`. legacy 경로가 주문 가능한
proposal 집합을 만든 뒤, legacy가 쓴 것과 **동일한** `evidence` mapping으로 호출됩니다. 동일
입력이어야 결과 차이가 "본 것의 차이"가 아니라 "선택기의 차이"가 됩니다. SHADOW에서는 비교
결과만 남기고, 실효 권한이 있으면 결과를 기존의 실주문 승인 proposal과 다시 대조합니다.

```text
_select(...)
 ├─ legacy 후보·proposal·선택 계산
 ├─ _observe_selector_v2(tradable_candidates, evidence, bundle, now, legacy_selected)
      └─ SelectorV2ShadowRunner.observe(...)
           ├─ MarketContextBuilder.build_cycle(...)        ← 사이클 전체가 같은 captured_at
           ├─ StrategySelectorV2.select(context, ...)      ← 심볼별
           ├─ StrategyCoverageAnalyzer.record_selection(...)
           └─ CounterfactualEngine.open_from_selection(...)
 ├─ SHADOW: legacy 선택 유지
 ├─ LIVE_PROBE/LIVE: _selector_v2_live_choice(...)         ← 승인된 proposal과만 매칭
 └─ _arm(winner) 또는 SELECTOR_V2_NO_TRADE
```

실시간 quote는 열린 반사실 포지션을 진행시키고, 해소된 context를 권한 컨트롤러가 주기적으로
평가·영속화합니다. V2가 소유한 실체결은 같은 `context_id`로 다시 연결되어 full-live 승격 증거와
shadow 결과가 섞이지 않습니다.

### 5.1 안전 속성

* Python 기본 posture는 무동작 — `STRATEGY_SELECTOR_V2_ENABLED=false`. Windows `run.ps1`은
  `enabled=true`, `shadow_only=true`, `auto_promote=true`로 시작합니다.
* `configured_shadow_only=true`는 운영자 강제 live 권한을 막는 초기 안전 설정입니다. 최상위
  `shadow_only`와 `live_authority`는 승격 컨트롤러까지 반영한 **실효 상태**입니다.
* `SelectorV2Flags.validate()`는 강제 live 또는 자동 승격을 허용할 때 mask·NO_TRADE·
  counterfactual이 모두 켜지지 않으면 기동을 거부합니다.
* **AST 검사로 강제**: `tests/test_strategy_selector_v2.py::test_selection_layer_cannot_reach_execution`
  가 선택 계층의 어떤 모듈도 `app.execution`, `app.risk`, `app.cost.profitability_gate`,
  realtime engine, shared decision engine 을 import 하지 않음을 확인합니다. **import 할 수 없는
  코드는 리스크 게이트를 우회할 수 없습니다.**
* 예외 가드 2중. belt-and-braces 가 아닙니다: 엔진의
  `strategy_session_manager.evaluate` 핸들러는 오류 시 **매수를 차단**하므로, telemetry 에서
  예외가 새면 거래가 멈춥니다.
* 작업량 유계 — `max_symbols_per_cycle` (기본 8). cap 은 telemetry 에 보고되며 조용히
  잘리지 않습니다. legacy 가 고른 심볼은 cap 과 무관하게 비교 집합에 남습니다.
* V2는 broker·execution 모듈을 import하지 않습니다. 승격된 결과도
  `StrategySessionManager`가 이미 `submits_orders=true`로 만든 proposal과 일치해야 하며, 이후
  ProfitabilityGate·RiskManager·FinalTradeGate·KIS 게이트를 모두 통과합니다.
* 개별 전략의 `live_authorized=false`, 숏 배포 `SHADOW`, 대주 불가 상태는 별도 전제조건입니다.
  선택기 승격은 이 권한을 새로 만들거나 우회하지 않습니다.
* 권한 상태 저장 실패는 승격 거부, 평가 예외는 `SUSPENDED`입니다. 영속 상태가 `SUSPENDED`인
  경우 clean restart는 SHADOW로 돌아가 보존된 증거를 다시 평가합니다.

### 5.2 켜는 방법과 읽는 곳

```powershell
# 수동 Python 실행용. run.ps1은 아래 세 값을 이미 설정합니다.
$env:STRATEGY_SELECTOR_V2_ENABLED = "1"
$env:STRATEGY_SELECTOR_V2_SHADOW_ONLY = "1"
$env:STRATEGY_SELECTOR_V2_AUTO_PROMOTE = "1"
# 선택: 기존 GNN 벡터가 gross/downside/uncertainty 를 공급
$env:STRATEGY_UTILITY_GNN_ENABLED = "1"
```

운영자가 표본을 보고 `shadow_only=false`로 바꿀 필요는 없습니다. 권장 경로는 설정상 shadow를
유지하고 자동 컨트롤러가 다음 사다리를 전환하게 하는 것입니다.

| 전이 | 필수 증거 |
|---|---|
| `SHADOW → LIVE_PROBE` | context 120+, 선택 거래 context 40+, 10일+, 시간순 4창 중 3창 양수, 순수익 95% 하단 ≥ +3bps, 추가 비용 5bps 후 하단 > 0, 평균 regret ≤ 15bps, 오레짐 거래율 ≤ 15%, top-1 hit ≥ 40%, 전부 3회 연속 통과 |
| `LIVE_PROBE → LIVE` | 실제 LIVE_PROBE context 30+, 10일+, 실현 순수익 95% 하단 ≥ 0bps를 3회 연속 통과. 이 단계는 shadow 결과만으로 통과할 수 없음 |
| `LIVE_PROBE/LIVE → SHADOW` | context 20+에서 순수익 하단 ≤ −5bps 또는 오레짐 거래율 > 25%면 즉시 강등 |

`LIVE_PROBE`는 기존 게이트가 승인한 수량의 최대 10%입니다. 양수 주문의 정수 수량 제약 때문에
원래 승인 수량이 1주인 경우에는 최소 1주입니다. `LIVE`는 100%이지만 기존 포지션 사이저와 모든
리스크 한도를 그대로 적용합니다. 증거·상태·최근 전이는
`data/store/selector-v2-promotion.json`에 최대 4,000 context까지 저장됩니다.

telemetry 는 기존 `strategy_session` payload 를 타고 나갑니다
(`snapshot()["selector_v2"]`). 권위 있는 live 조회는
`GET /api/realtime-trading/status`의 `strategy_session.selector_v2`이고,
`GET /api/refactor/market-view`에도 현재 session overlay가 실립니다. `/api/refactor/dashboard`는
정적 프로파일과 legacy/ontology/GNN shadow 비교용이며 V2 live 권한의 권위 소스가 아닙니다.

| 키 | 내용 |
|---|---|
| `enabled` / `configured_shadow_only` | 실행 여부와 설정상 초기 안전 posture |
| `shadow_only` / `live_authority` / `order_size_fraction` | 승격 상태를 반영한 실효 권한과 주문 비율 |
| `auto_promotion.state` | `SHADOW` / `LIVE_PROBE` / `LIVE` / `SUSPENDED` |
| `auto_promotion.metrics` / `evidence_rows` | 승격 기준별 현재 측정값과 영속 context 수 |
| `auto_promotion.recent_transitions` | 승격·강등·중단 이력과 당시 지표 |
| `latest_selection.context_id` | 이 판단이 본 스냅샷 |
| `latest_selection.blocked` | 온톨로지 hard block 사유 (전략별) |
| `latest_selection.proposals` | 모든 eligible proposal (`proposal_id` 포함) |
| `latest_selection.ranked_candidates[]` | 후보별 7개 항 전부 + `final_utility_bps` + `lower_confidence_bound_bps` + `reason_codes` |
| `latest_selection.no_trade` | NO_TRADE 임계값, 최고 후보, margin, 사유 |
| `comparisons[]` | legacy vs V2 (`SAME_STRATEGY` / `DIFFERENT_STRATEGY` / `V2_DECLINED` / `V2_TRADED` / `BOTH_NO_TRADE`) |
| `coverage`, `coverage_gaps` | 버킷 요약과 반복 gap |
| `counterfactual` | 열린/해소된 그룹 수 |

`SelectorV2ShadowRunner.regret_summary()` 는 counterfactual 그룹이 해소되면 selector regret 을
반환합니다.

### 5.3 실행 확인

플래그를 켜고 실제 `StrategySessionManager` 안에서 합성 evidence 1건으로 통과시킨 결과:

```text
runner attached: True
enabled True  shadow_only True  live_authority False
decision SELECT  selected intraday_momentum  utility 109.951
no_trade bar 10.0
  intraday_momentum         U= 109.951  gross=180.0  cost=35.811  src=UTILITY_SOURCE_GNN        ready=True
  liquidity_shock_reversal  U=-102.737  gross=  0.0  cost=35.811  src=UTILITY_SOURCE_HEURISTIC  ready=False
  breakout_volume           U=-122.813  gross=  0.0  cost=35.811  src=UTILITY_SOURCE_HEURISTIC  ready=False
blocked: 15   comparison: V2_TRADED
coverage: buckets_seen=1 / 4800   counterfactual: groups_open=1
```

`cost=35.811` 은 KRX 왕복 27.8bps + 측정된 8bps 스프레드입니다 — 모델 채널이 아니라
`TradingCostEngine` 에서 나온 값입니다.

자동 승격 연결 후 실제 런타임의 2026-08-11 23:09:33 KST 영속 스냅샷:

```text
configured_shadow_only=true  auto_promote=true
state=SHADOW  live_authority=false  order_size_fraction=0
context_count=66  traded_context_count=0  distinct_days=1
lower_bound_net_bps=0.0  cost_stressed_lower_bound_bps=-5.0
```

66개가 모두 NO_TRADE context라 양의 순수익 표본으로 세지 않습니다. 따라서 현재 수동 조작 없이
정상적으로 SHADOW를 유지하는 상태입니다.

---

## 6. 실제 저장 데이터에 대한 첫 audit

`python scripts/report_strategy_selection_v2.py`, 2026-08-11:

| 전략 | lifecycle | n | 증거 | net bps | 하단 | break-even cost × | OOS | 분류 |
|---|---|--:|---|--:|--:|--:|--:|---|
| liquidity_shock_reversal | LIVE | ~740 | shadow, US 전용 | **≈−120** | ≈−127 | −0.43 | 0.00 | **RESEARCH** |
| vwap_mean_reversion | LIVE | 2 | shadow | −143.7 | −168.9 | −1.98 | — | SHADOW_ONLY |
| breakout_volume | LIVE | 1 | shadow | −171.3 | — | −3.84 | — | SHADOW_ONLY |
| intraday_momentum | LIVE | 1 | **live** | +72.1 | — | 3.58 | — | SHADOW_ONLY |
| 나머지 9개 | **LIVE** | **0** | 없음 | — | — | — | — | INSUFFICIENT_DATA |
| SHADOW 4종 (market_intraday_momentum, overnight_gap_carry, range_support_reversion, bar_trend_continuation) | SHADOW | 0 | 없음 | — | — | — | — | INSUFFICIENT_DATA |
| 숏 3종 | RESEARCH | 0 | 없음 | — | — | — | — | INSUFFICIENT_DATA |

`liquidity_shock_reversal` 의 n 은 shadow 평가기가 계속 채점하므로 증가합니다 — 위 수치는 실행
시점값이고, 현재 값은 스크립트를 다시 돌려 확인하세요. §1.1 표의 `live_authorized` 열과 이 표의
lifecycle 이 `range_support_reversion` 에서 다른 이유는 §6.3 에 있습니다.

두 가지 발견이며, 둘 다 이전에는 한 곳에서 볼 수 없었습니다.

### 6.1 표본이 있는 유일한 전략이 손실 중

`liquidity_shock_reversal`: 약 740 관측, 평균 순 ≈−120bps, 중위 ≈−95, 단측 95% 하단 ≈−127bps,
walk-forward 창 중 양수가 **0개** (out-of-sample 안정성 0.00). break-even cost multiple 이
음수라는 것은 **gross 엣지도 음수**라는 뜻이므로 비용 문제가 아니라 전략 문제입니다. 현재
LIVE authorization 을 보유하고 있습니다.

**주장의 범위** — 결론을 제한하므로 명시합니다: 그 행 전부 `evaluation_source=shadow` 이고
전부 US 입니다. 즉 한 시장에 대한 큰 **시뮬레이션** 표본입니다. 이 논지의 거래를 멈출 근거로는
충분하고 파일을 닫을 근거로는 전혀 부족하므로, audit 은 RETIRE 가 아니라 RESEARCH 로
분류합니다. RETIRE 는 실질적으로 되돌릴 수 없어(아무도 돌리지 않는 전략은 복귀 증거를 쌓지
못함) audit runner 는 **shadow 전용 증거로는 retire 하지 않습니다.**

### 6.2 관측 0개로 LIVE 권한을 가진 전략이 9개

`event_momentum`, `cross_sectional_relative_strength`, `gap_context`, `rvgi_box_breakout`,
`residual_relative_strength`, `adaptive_anchored_vwap_reversion`,
`ofi_microprice_exhaustion_reversal`, `opening_range_breakout`, `bar_confirmed_vwap_recovery`.
**라이브 카탈로그 전체가 실체결 1건 위에 서 있습니다.**

### 6.3 무엇을 바꿨는가

audit 자체는 아무것도 바꾸지 않습니다. `LIFECYCLE_RECOMMENDATIONS` 에 측정치를 붙여 2건을
기록했고 **둘 다 `STRATEGY_LIFECYCLE_APPLY_RECOMMENDATIONS=1` 까지는 권고**입니다:

| 전략 | 권고 | 현재 상태 | 증거 |
|---|---|---|---|
| `range_support_reversion` | SHADOW | **SHADOW — 이미 충족** | 거래가능 유니버스 t=1.56 (하위기간 t=0.65). 유의성은 이 계좌가 주문할 수 없는 2X 인버스 ETF 가 상당 부분 지탱. 운영자 결정으로 등록 |
| `liquidity_shock_reversal` | SHADOW | **LIVE — 미충족** | §6.1 의 ~740행 |

`range_support_reversion` 은 §1.1 스냅샷 시점에 `live_authorized: 1.0` 이었고 권고는 LIVE_PROBE
였습니다. 그 뒤 `config/strategy_algorithms.yaml` 이 `live_authorized: false` 로 바뀌어
**권고보다 한 칸 더 내려갔으므로** 파생 lifecycle 은 이미 SHADOW 이고 권고를 적용해도 변화가
없습니다. 항목을 지우지 않고 남긴 이유는 지우면 config 가 왜 그렇게 되어 있는지가 사라지기
때문이고, 권고는 낮추기 전용이므로 되돌아가는 경로가 되지도 않습니다.

권고는 상태를 낮출 수만 있습니다. 올리는 것은 `app.strategy_validation.registry` 를 통해야
하고, 그 전이 그래프는 화이트리스트이며 승격에는 validation record 첨부가 필수입니다.

coverage와 반사실 표본은 Windows 런처에서 자동 누적됩니다. coverage의 현재 수치는
`GET /api/realtime-trading/status`의 `strategy_session.selector_v2.coverage`에서, 권한용 해소
context는 같은 payload의 `auto_promotion.evidence_rows`에서 확인합니다. 둘은 목적이 달라 coverage
셀 수만으로 권한을 승격하지 않습니다.

---

## 7. 코드 분류

**사용 중 (live 주문 경계):** `StrategySessionManager` (proposal 소유권 + 초기 legacy 선출 +
승격 후 V2 결과 매칭),
`SharedLiveDecisionEngine`, `RealtimeTradingEngine`, `LiveExecutionCoordinator`,
`RiskManager`, `ProfitabilityGate`, `ExecutionPricingPolicy`, `ExecutionQualityEngine`,
`ExchangeResolver`, `ConservativeStrategyBandit`, `StrategyPerformanceStore`,
`ShadowIntelligenceService`, `directional_shadow`, `short_strategy_promotion`.

**신규 선택·평가 계층:** §3의 모듈 전부. 선택 모듈은 execution import가 없고,
`selector_v2_promotion.py`가 실효 선택 권한만 공개합니다. 실제 주문 생성·승인은 기존 세션·비용·
리스크·실행 계층의 책임입니다.

**의도적으로 남긴 legacy:** `catalog.METHODOLOGY_STRATEGY_ALIASES` 와 `resolve_strategy_id`.
여전히 *legacy* fallback 선출 경로의 브리지이고, V2가 SHADOW이거나 자동 강등됐을 때 필요합니다.
V2 경로에는
없으며 그것이 완료 기준이 요구하는 것입니다 —
`tests/test_ontology_strategy_eligibility.py::test_generic_methodology_names_are_not_addressable`
가 고정합니다.

**V2 는 쓰지 않지만 legacy 경로에서 살아 있는 것:** `_gnn_direct_choice`
(운영자 posture, `STRATEGY_SESSION_GNN_DIRECT_ELECTION`), `_intent_proposals` 의 alias 해석.

---

## 8. Migration 단계별 상태

| 단계 | 상태 |
|---|---|
| 0 baseline | 완료 — §1 |
| 1 MarketContext + 계약 | 완료 |
| 2 전략 audit | 완료 — runner 존재, 첫 실행 결과 §6 |
| 3 실제 id 기준 온톨로지 eligibility | 완료 |
| 4 legacy LIVE 옆 SelectorV2 SHADOW | 완료. Windows 런처에서 활성, 초기 SHADOW |
| 5 counterfactual 수집 | 완료·영속 수집 중. 2026-08-11 23:09:33 KST context 66, 거래 선택 0 |
| 6 utility GNN | adapter 완료 (기존 벡터 소비, 비용 제거). **strategy-conditioned 학습 파이프라인은 미구현** |
| 7 bandit adapter | 완료 |
| 8 NO_TRADE 검증 | 정책·자동 승격 기준 완료. 캘리브레이션은 누적 데이터로 계속 검토 |
| 9 LIVE_PROBE | 코드·10% 수량 제한·실체결 연결 완료. **증거 미달로 아직 미진입** |
| 10 권한 cutover | 자동 `LIVE_PROBE → LIVE` 및 자동 강등 구현. **현재 SHADOW라 미발동** |

---

## 9. 남은 기술 부채

1. **6단계는 adapter 이고 재학습이 아닙니다.** R-GCN 은 여전히 비용 채널을 갖고 내부에서
   utility 를 계산하며, V2 는 둘 다 무시합니다. `context + strategy_id → outcome distribution`
   목표로 재학습하는 것이 진짜 6단계이고 5단계 데이터셋이 선행됩니다. 그때까지 대부분의
   컨텍스트는 `HeuristicUtilityPredictor` 가 담당하고, 모든 예측이
   `UTILITY_SOURCE_HEURISTIC` 로 그 사실을 밝힙니다.
2. ~~**`history_bar_count` 는 추론값이고 측정값이 아닙니다.**~~ **해소됨.**
   `web._strategy_session_selection_evidence` 가 frame 의 `slow_technical:bar_count` 를
   `row["history_bar_count"]` 로 publish 합니다. 그전까지 adapter 는 Donchian 컬럼 유무로 20 을
   추론했고, 그 하한이 `requiresHistory: 30` 인 전략을 `20<30` 으로 **거짓 차단**했습니다 —
   수백 봉을 가진 종목까지 포함해서, 장기 horizon coverage 전략 전부가 그렇게 막혔습니다.
   `_history_bars` 의 Donchian 추론은 구 row 를 위한 fallback 으로만 남아 있습니다.
3. **횡단면 컨텍스트가 얇습니다.** `dispersion` 과 `market_leadership_score` 는 peer return 이
   필요하고 라이브 evidence row 는 심볼별 peer 를 담지 않으므로 보통 `None` 입니다. 그러면 V2 는
   `cross_sectional_relative_strength` 와 `residual_relative_strength` 를 hard block 합니다.
   fail-closed 로 올바른 동작이지만, 그 두 family 는 현재 shadow 에서 커버 불가입니다.
4. **`StrategySessionManager`는 여전히 큰 통합 경계**이고 legacy 선출도 fallback으로 소유합니다.
   V2가 증거를 얻기 전과 자동 강등 후에 필요하므로 당장 제거할 수 없습니다.
5. **증거 가중치는 설계값입니다.** `config/strategy_validation.yaml` 의
   LIVE/PROBE/SHADOW/BACKTEST (1.0/0.7/0.3/0.0) 은 측정되지 않았습니다. shadow-vs-live 짝
   outcome 으로 재캘리브레이션해야 하고 그것이 5단계 산출물입니다.
6. **±20bps bandit 경계와 λ 가중치는 fit 되지 않았습니다.** 둘 다 근거를 적어 둔 config 상의
   시작값이며 데이터로 추정한 값이 아닙니다.
7. **coverage와 권한 증거는 JSON입니다.** 실시간 SQLite writer와 충돌하지 않는 의도적 선택이지만
   호스트별 상태이고 파일을 지우면 초기 SHADOW에서 다시 수집합니다. 특히
   `selector-v2-promotion.json` 저장 실패 시 권한을 부여하지 않습니다.

---

## 10. 테스트 표면

| 파일 | 건수 | 무엇을 고정하는가 |
|---|--:|---|
| `tests/test_market_context.py` | 11 | 결정론적 생성, 0 스프레드는 결측, field별 provenance, 그룹 간 이름 충돌 없음 |
| `tests/test_strategy_spec_registry.py` | 12 | append-only 순서 = `STRATEGY_IDS`, 선언된 요구사항이 모두 실제 field, 숏은 RESEARCH 고정, 권고는 낮추기만 |
| `tests/test_strategy_proposal_contract.py` | 12 | 주문 field 부재, 방향 인식 target/stop, 가격 없으면 `None`, 알고리즘 예외는 not-ready proposal |
| `tests/test_ontology_strategy_eligibility.py` | 15 | hard/soft 분할, generic methodology 이름 주소 불가, 결측 요구사항이 block, 미해석 레짐은 block 아님 |
| `tests/test_strategy_selector_v2.py` | 31 | 항 분해 합이 총합과 일치, NO_TRADE 정상 결과, 양수 평균도 하단 규칙으로 거부, coverage gap → NO_TRADE, **AST 로 실행 계층 import 금지** |
| `tests/test_bandit_adapter_bounds.py` | 9 | clamp 와 보고, 이력 없으면 보정 0, 얇은 표본 축소, 최근성 감쇠, context key 에 symbol 없음 |
| `tests/test_counterfactual_and_regret.py` | 17 | barrier 보수적 우선순위, 숏 부호 1회 적용, 신호 이전 quote 무효, 선택 전략은 sink 로 안 나감, shadow 는 LIVE 로 셀 수 없음, NO_TRADE 가 경쟁 |
| `tests/test_strategy_coverage_and_validation.py` | 26 | 버킷팅 결정성, gap 기록, 중첩 표본 할인, 2칸 승격 거부, 증거 없는 승격 거부, shadow 만으론 LIVE 불가, 1회 손실로 강등 안 함 |
| `tests/test_selector_v2_shadow_integration.py` | 12 | 라이브 evidence 모양으로 파이프라인 통과, 진단 payload 완비, 잘못된 입력에도 raise 안 함, flag 기본 무동작, live 권한은 안전 sub-flag 요구 |
| `tests/test_selector_v2_auto_promotion.py` | 5 | 자동 사다리·영속화, full-live 실체결 요구, 빠른 강등, 설정/실효 권한 분리, 단일 표본 안전성 |

표의 V2 중심 테스트는 **150건**입니다. 자동 권한 변경 후 세션·실시간 엔진·실행 안전까지 포함한
관련 회귀 묶음은 **286건 통과**했습니다(2026-08-11).

---

## 11. 관련 문서

[architecture.md](architecture.md) §3 런타임 흐름 · §4 모듈 지도 ·
[ontology_and_gnn.md](ontology_and_gnn.md) §2.5 eligibility 층 ·
[decision_and_risk.md](decision_and_risk.md) §7 무엇이 무엇을 결정하는가 ·
[validation.md](validation.md) §1 승격 상태 · §14 검증 프레임워크 ·
[short_selling_deployment.md](short_selling_deployment.md) 숏 3중 잠금

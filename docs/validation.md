# Validation and Evidence

측정된 것, 측정되지 않은 것, 그리고 무엇이 승격되지 않았는지에 대한 문서입니다. 좋아 보이는 숫자를 만드는 것이 목표가 아니라, 어떤 주장이 증거를 갖고 있는지 분리하는 것이 목표입니다.

## 1. 승격 상태 요약

| 항목 | 상태 | 근거 |
| --- | --- | --- |
| 순수익 게이트 / 동적 청산 리팩터 | 프로덕션 적용 | 아래 리플레이 지표, `tests/test_profitability_refactor_integration.py` |
| 온톨로지 가속 (indexed graph + FactTable) | 프로덕션 적용, 동작 동일 | `scripts/benchmark_fact_table.py`, `tests/test_fact_table.py` |
| 전략 소유 실행 경로 | live 연결, 모든 게이트 필수 | `StrategySessionManager`, causal journal, `tests/test_strategy_session.py` |
| 이벤트 시뮬레이터 + counterfactual 라벨 | 학습 적용, 표본 품질 필터 활성 | 연속 시계열·활동성·미래봉 검사, 전략별 목표/손절 |
| strategy-utility R-GCN | CPU live trust-gated 실행 | `authorization_scope=ontology_gnn_realtime_trust_gated_execution` |
| GNN 실시간 진입 권한 | 전략별 조건부 | `/api/gnn/realtime-trust`의 `trusted_strategy_ids`; 모델 보정과 진입 권한 분리 |
| NPU 추론 | 컴파일 성공, **승격 거부** | `data/reports/strategy_utility_openvino.json`, `promotion_eligible=false` |
| legacy vs ontology vs tabular vs R-GCN 비교 | **미완** | 필요한 point-in-time 데이터 부재 |
| 숏 전략 3개 (방향 계약·대주 회계·shadow 평가) | 코드 적용, **전량 `SHADOW`** | `tests/test_directional_short_ladder.py` (76건), 실주문 권한 0개 |
| 숏 배포 사다리 (자동 승격/강등 상태 머신) | 프로덕션 적용, 아직 승격 사례 없음 | `data/store/strategy-deployment.sqlite3`, `promotion_audit` |
| 숏 forward 성과 | **미측정** | forward 표본 0. 최소 20 거래일 필요 |
| KIS 대주 조회 TR | **실계좌 검증 완료 — 추측한 3개 전부 오류** | 융자 응답 / 없는 TR / 404. `BorrowDataSource`로 교체 |
| 대주 잔고 조회 | **검증된 경로로 이전** | 프로덕션이 이미 쓰는 `inquire-balance` 재사용 |
| GNN 대주 채널 (8→11) | 구현, **미학습** | 라벨이 forward 표본에서 나와야 함 |
| 스퀴즈 필터 | **비활성** | 공매도 잔고 소스 없음. 대주 게이트가 fail-closed 담당 |
| KIS 대주 주문 경로 | mock transport 검증만 | 실주문 미제출 |
| GNN 방향별 utility head | **미구현** | 숏 arm은 realized posterior + 규칙 신호로만 평가 |
| feature schema v5 (정체성 컬럼 8개 제거) | 프로덕션 적용, 승격 확인 | 아래 §12. 홀드아웃 1 split + 라이브 재학습 |
| 단계적 모델 강등 fallback | 프로덕션 적용 | `AUTO_RELIABILITY_MODEL_DEGRADED_FALLBACK`, `tests/test_auto_reliability_mode.py` |
| 스키마 인식 승격 | 프로덕션 적용 | `OBSOLETE_SCHEMA_INCUMBENT_REPLACED`, `tests/test_model_training_artifacts.py` |
| KIS 구독 티어링 (depth / trade-only) | 코드 적용, **KRX 미검증** | KR 세션을 아직 만나지 않음. §12.3 |
| KR 단기 모델 라이브 적격 | **미달** | precision@k 0.333 vs 임계 0.35 |
| 전략 선택 V2 (context/eligibility/utility/NO_TRADE 분리) | 프로덕션 초기 `SHADOW`, **자동 승격 활성** | §14, `selector_v2_promotion.py`, 안전·회귀 묶음 286건 통과 |
| V2 자동 권한 사다리 | 코드 적용, 아직 승격 전 | `SHADOW → LIVE_PROBE(10%) → LIVE`, 영속화·빠른 강등·오류 중단 |
| V2 반사실 표본 (selector regret) | **자동 수집 중, 66 context 관측·거래 선택 0** | 2026-08-11 23:09:33 KST 영속 스냅샷. 최소 120 context·선택 거래 40·10일 전에는 권한 없음 |
| 전략 audit 프레임워크 (단일 evaluator) | 코드 적용, 첫 실행 완료 | §1.1, `scripts/report_strategy_selection_v2.py` |
| **카탈로그 라이브 권한의 증거** | **거의 없음 — 아래 §1.1** | 실체결 1건. LIVE 권한 전략 9개가 관측 0건 |
| strategy-conditioned utility GNN 재학습 | **미착수** | 현재는 기존 벡터의 adapter. §14.2 |

### 1.1 라이브 카탈로그가 실체결 1건 위에 서 있다

`scripts/report_strategy_selection_v2.py`, 2026-08-11. 저장된 성과 데이터 전량에 대해 **모든 전략을
같은 evaluator 로** 처음 통과시킨 결과입니다. 이 문서의 다른 어떤 수치보다 먼저 읽어야 합니다.

| 전략 | lifecycle | n | 증거원 | net bps | 하단 | break-even cost × | OOS 안정성 | 분류 |
|---|---|--:|---|--:|--:|--:|--:|---|
| liquidity_shock_reversal | LIVE | ~740 | shadow, US 전용 | **≈−120** | ≈−127 | −0.43 | 0.00 | **RESEARCH** |
| vwap_mean_reversion | LIVE | 2 | shadow | −143.7 | −168.9 | −1.98 | — | SHADOW_ONLY |
| breakout_volume | LIVE | 1 | shadow | −171.3 | — | −3.84 | — | SHADOW_ONLY |
| intraday_momentum | LIVE | 1 | **live** | +72.1 | — | 3.58 | — | SHADOW_ONLY |
| 나머지 9개 | **LIVE** | **0** | 없음 | — | — | — | — | INSUFFICIENT_DATA |
| SHADOW 4종 (market_intraday_momentum, overnight_gap_carry, range_support_reversion, bar_trend_continuation) | SHADOW | 0 | 없음 | — | — | — | — | INSUFFICIENT_DATA |
| 숏 3종 | RESEARCH | 0 | 없음 | — | — | — | — | INSUFFICIENT_DATA |

`liquidity_shock_reversal` 의 n 은 shadow 평가기가 계속 채점하므로 늘어납니다. 위 수치는
2026-08-11 실행값이고, 현재 값은 스크립트를 다시 돌려 확인하세요.

**표본이 있는 유일한 전략이 손실 중입니다.** 약 740관측, 평균 순 ≈−120bps, 중위 ≈−95, 단측 95%
하단 ≈−127bps, walk-forward 창 중 양수 **0개**. break-even cost multiple 이 음수라는 것은 gross
엣지도 음수라는 뜻이므로 **비용 문제가 아니라 전략 문제**입니다.

주장의 범위: 그 행 전부 `evaluation_source=shadow` 이고 전부 US 입니다. 즉 한 시장에 대한 큰
**시뮬레이션** 표본입니다. 이 논지의 거래를 멈출 근거로는 충분하고 파일을 닫을 근거로는 부족하므로
audit 은 RETIRE 가 아니라 RESEARCH 로 분류합니다 — RETIRE 는 실질적으로 되돌릴 수 없고
(아무도 돌리지 않는 전략은 복귀 증거를 못 쌓음), runner 는 **shadow 전용 증거로 retire 하지
않습니다.**

**관측 0건으로 LIVE 권한을 가진 전략이 9개입니다:** `event_momentum`,
`cross_sectional_relative_strength`, `gap_context`, `rvgi_box_breakout`,
`residual_relative_strength`, `adaptive_anchored_vwap_reversion`,
`ofi_microprice_exhaustion_reversal`, `opening_range_breakout`, `bar_confirmed_vwap_recovery`.

**아무것도 바꾸지 않았습니다.** 권고 2건을 측정치와 함께 기록했고 둘 다
`STRATEGY_LIFECYCLE_APPLY_RECOMMENDATIONS=1` 까지는 권고입니다.

| 전략 | 권고 | 현재 상태 | 증거 |
|---|---|---|---|
| `range_support_reversion` | SHADOW | **SHADOW — 이미 충족** | 거래가능 유니버스 t=1.56 (하위기간 t=0.65). 유의성을 이 계좌가 주문할 수 없는 2X 인버스 ETF 가 상당 부분 지탱. `config/strategy_algorithms.yaml` 이 `live_authorized: false` 로 바뀌어 권고보다 더 나아갔음 |
| `liquidity_shock_reversal` | SHADOW | **LIVE — 미충족** | 위 ~740행 |

권고는 상태를 **낮추는 방향으로만** 작동합니다 — 이미 충족된 항목을 지우지 않고 남기는 이유는
지우면 config 가 왜 그렇게 되어 있는지가 사라지기 때문이고, 낮추기 전용이므로 되돌아가는 경로가
되지도 않습니다. 올리는 것은 §14.3 의 전이 화이트리스트와 승격 게이트를 통과해야 합니다.

재실행:

```powershell
python scripts/report_strategy_selection_v2.py
python scripts/report_strategy_selection_v2.py --json data/reports/strategy_audit.json
```

## 2. 수익성 리팩터 리플레이

![Before/after net-profitability gate](diagrams/profitability_before_after.svg)

```powershell
$env:PYTHONPATH="src"; python scripts/profitability_replay_report.py
```

`logs/live-orders.jsonl`에서 주문 흐름 결과, 거부 사유 분포, 비용 반영 실현 지표를 뽑습니다. 실현 PnL은 limit 가격을 체결가 proxy로 쓰므로 **근사값**이며, net PnL/수익률은 live `TradingCostEngine`을 적용합니다.

리팩터 이전 저널(181 round-trip) 기준선:

| 지표 | 값 |
| --- | --- |
| round_trips | 181 |
| gross_pnl | +1,342.45 |
| net_pnl | **−1,701.37** |
| win_rate | 0.442 |
| avg_win_net | +0.01689 |
| avg_loss_net | −0.01547 |
| payoff_ratio | 1.092 |
| expectancy_net | **−0.00117** |
| negative_net_trades | 101 / 181 |
| max_drawdown_net_pnl | −7,237.15 |

주문 흐름에서 드러난 병리: `live_order_amend_attempt` 3,897건 중 `live_order_amend_error` 3,861건, `live_order_submission_attempt` 1,343건 중 `live_order_submission_error` 957건. 차단 사유는 `LIVE_TRADING_ENABLED_NOT_TRUE` 237건이 대부분이었습니다.

**핵심 진단: gross는 양수인데 net은 음수.** 방향성 예측 자체가 아니라 비용이 문제였고, 이것이 단일 `ProfitabilityGate`와 단일 `DynamicExitPolicy`로 통합한 이유입니다. 성공 기준은 거래 수 증가가 아니라 **NET expectancy 개선과 net-negative 거래 감소**입니다.

리팩터 이후 값을 얻으려면 같은 스크립트를 다시 실행해 비교하세요. 리포트를 파일로 남기려면 `--markdown data/reports/profitability_replay.md`를 씁니다.

## 3. 온톨로지 가속 벤치마크

```powershell
python scripts/benchmark_fact_table.py
```

문자열 트리플 조회: 리팩터 전 `KnowledgeGraph`는 호출마다 전체 트리플 리스트를 스캔했고(O(전체)), 인덱스 버전은 O(매칭)입니다. 두 구현이 동일한 행을 반환함을 확인한 뒤 측정합니다.

| symbols | triples | linear ms | indexed ms | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6 | 5.21 | 2.84 | 1.8× |
| 10 | 60 | 16.82 | 3.26 | 5.2× |
| 50 | 300 | 62.89 | 3.22 | 19.5× |
| 100 | 600 | 118.01 | 3.23 | 36.5× |
| 500 | 3,000 | 565.30 | 3.09 | **183.1×** |

linear 비용은 그래프 크기에 비례해 오르고, indexed 비용은 매칭 수에만 의존하므로 ~3 ms로 평평합니다. 500 심볼에서 183×이며, 이것이 감사에서 지적한 per-tick 영역입니다.

FactTable: 3,000-fact 그래프에서 build 6.24 ms, `get_facts_by_subject` 500회 1.80 ms (쿼리당 ~3.6 µs).

**바꾸지 않은 것:** 판단 로직, 게이트 순서, 거래 동작. 인덱스 그래프는 동작상 이전과 동일하고 FactTable은 additive이며 아직 live 판단 경로에 연결되어 있지 않습니다.

## 4. 실시간 파이프라인 CPU 기준선

```powershell
python scripts/benchmark_realtime_pipeline.py --device CPU --output data/reports/refactor_p0_cpu_benchmark.md
```

| 합성 입력 | hard filter 통과 | top K | scoring ms | 전체 파이프라인 ms | peak MB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 80 | 50 | 585.23 | 597.921 | 6.285 |
| 1,024 | 644 | 50 | 22.514 | 60.553 | 7.120 |
| 4,096 | 2,647 | 50 | 29.010 | 155.238 | 10.171 |
| 10,000 | 6,460 | 50 | 28.322 | 318.851 | 15.943 |

첫 시나리오는 모델/런타임 warm-up을 포함하므로 warm한 큰 배치와 비교할 수 없습니다. legacy telemetry가 `npu_enabled=1`을 내보내더라도 요청·보고 장치는 `CPU`였습니다. **이 결과를 NPU 가속이라고 서술하면 안 됩니다.**

이 벤치마크는 합성 유니버스에 대한 배치 후보 필터링을 측정하며, WebSocket-to-order p50/p95/p99, CPU 사용률, OpenVINO 연산자 커버리지, CPU/NPU 판단 랭킹 parity는 측정하지 **않습니다.** 그 값들은 전용 계측과 대표성 있는 리플레이가 생기기 전까지 "측정되지 않음"으로 남아야 합니다.

## 5. OpenVINO CPU / NPU 벤치마크

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_strategy_utility_openvino.py --iterations 30
```

아래 값은 이전 `S=7` 체크포인트로 수행한 **역사적 NPU 승격 실험**입니다. 현재 `B1 T1 N16 F40 R3 S16` 체크포인트의 NPU 승격 근거로 재사용하지 않습니다. 당시 고정 형상은 `B=1, T=4, N=16, F=12, R=4, S=7`, FP32, 30 iterations였습니다.

| device | 실제 컴파일 장치 | compile ms | p50 ms | p95 ms | p99 ms | throughput/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CPU | CPU | 89.91 | 0.329 | 0.660 | 0.667 | 2,789 |
| NPU | NPU | 43.79 | 1.137 | 1.398 | 2.263 | 837 |

Parity: CPU/NPU top-1 전략 일치율 **100%**, 최대 utility 절대 오차 `0.030865`, NoTrade 확률 최대 오차 `7.66e-05`.

**NPU는 실제로 컴파일되었고 CPU fallback도 없었습니다.** 그럼에도 end-to-end 추론이 CPU보다 느렸고 utility 오차가 설정된 golden tolerance `0.001`을 초과했습니다. → `promotion_eligible=false`. CPU가 검증된 런타임으로 남습니다.

벤치마크된 모델 해시: `6acf71cd0718e3738b9725df0b616f5cf9bf21892b3592b7952340b050a99ed4`. 증거 JSON: `data/reports/strategy_utility_openvino.json`.

### NPU 승격 게이트

승격하려면 전부 만족해야 합니다: 주장하는 NPU 경로에 조용한 unsupported fallback이 없을 것, CPU와 NPU 판단이 설정된 tolerance 안에서 일치할 것(임계 근처 불일치는 교정 전까지 NoTrade로 처리), tensor-pack/transfer 오버헤드를 포함한 end-to-end p95가 실질적으로 우월할 것, NPU 실패 시 검증된 CPU 모델로 원자적 fallback하거나 신규 라우팅을 동결하되 활성 전략 소유권과 청산은 계속될 것, 그리고 실제 컴파일 장치·모델 해시·정밀도·추론 시간·큐 지연·fallback 횟수·마지막 성공 시각이 노출될 것.

## 6. 저장 데이터 counterfactual 평가

```powershell
python scripts/build_refactor_counterfactual_report.py
```

`data/store/realtime_market_data.sqlite3`를 평가해 `data/reports/refactor_counterfactual_evaluation.json`에 재현 가능한 결과를 씁니다.

현재 학습 모델 카드 기준:

```text
snapshots 1,462 · strategy labels 11,696 · 8개 전략
configuration: history 30봉, horizon 60봉, stride 5봉
markets: 심볼에서 KRX/US 실행 비용 정책을 구분
```

**누수·표본 품질 통제:** rolling quantile은 직전 slice만 사용, feature cutoff가 라벨 구간에 선행, purging과 embargo 활성. 120초보다 큰 봉 간격, 활동성이 10% 미만인 history, 연속 미래봉 5개 미만인 표본은 제외합니다. 전략이 실제로 발화하지 않은 상태는 분류 음성 표본으로만 쓰고 가상 체결 손실로 회귀 헤드에 넣지 않습니다.

**전략별 결과** (`rgcn_shadow.json`, triggered 기준 체결):

| 전략 | triggered | filled | 평균 net bps | positive net rate |
| --- | ---: | ---: | ---: | ---: |
| intraday_momentum | 337 | 295 | −62.84 | 5.42% |
| breakout_volume | 216 | 187 | −66.89 | 4.81% |
| vwap_mean_reversion | 53 | 44 | −54.56 | 6.82% |
| liquidity_shock_reversal | 232 | 211 | −62.84 | 2.84% |
| rvgi_box_breakout | 5 | 5 | −101.86 | 0.0% |
| event_momentum | 0 | 0 | — | — |
| cross_sectional_relative_strength | 0 | 0 | — | — |
| gap_context | 0 | 0 | — | — |

전체 trigger 평균이 음수라는 사실은 모델을 무조건 막는 단일 판정이 아닙니다. GNN은 point-in-time feature에서 희소한 양수 하위 구간을 학습하되, 그 양수 예측은 다시 실시간 forward 결과로 검증합니다. 현재 VWAP 필터에 거래량·유동성 확인을 추가하면서 체결 표본은 90개에서 44개로 줄고 양수 비율은 약 3.3%에서 6.8%로 개선됐습니다.

**필수 시스템 비교 상태:**

```text
legacy            비교 기록
ontology_only     closed-world 허용/차단
strategy_rgcn_cpu 실시간 판단 + 전략별 trust gate
strategy_rgcn_npu 미승격, CPU fallback
```

**한계 (리포트에 명시됨):**

- 로컬 분봉 스토어가 소수 US 세션에 집중되어 있고, 목표 시장인 KRX/NXT가 아닙니다.
- 분봉 OHLC로는 tick 수준 큐 포지션이나 봉 내부 barrier 순서를 복원할 수 없습니다.
- point-in-time 이벤트, 섹터 그래프, 권위 있는 세션 캘린더, 과거 legacy 판단이 없어 8개 전략 전체 비교가 불가능합니다.

오프라인 모델 카드의 `live_authorized=true`와 전략별 실거래 권한은 다릅니다. 전자는 스키마·커버리지·체크포인트 검사를 뜻하고, 후자는 실시간 검증의 `trusted_strategy_ids`로만 부여됩니다.

## 7. Strategy-utility 모델 카드

```powershell
python scripts/train_strategy_utility_rgcn.py
```

현재 배포본 `data/models/strategy_utility/rgcn_shadow.json` (v5 정렬 계약, 2026-08-09 승격):

```text
method                ontology_strategy_graph_rgcn_joint_gradient_calibration
input_feature_schema  realtime_strategy_graph_v5_aligned
feature_provenance    causal_minute_bar_microstructure_v2_aligned
rows                  57,552        snapshots 3,597
strategies            16개 각 3,597
config                B1 T1 N16 F40 R3 S16, hidden 16, seed 17
authorization_scope   ontology_gnn_realtime_trust_gated_execution
authorization_checks  row/snapshot/strategy 커버리지 및 런타임 스키마 일치 통과
```

`feature_provenance`의 `proxy_v1` → `v2_aligned` 변화가 이 재학습의 핵심입니다. v4는 학습 시
스프레드를 봉의 high-low range로, 호가 불균형을 봉 내 종가 위치로 **대리**하면서 서빙에서는 실제
값을 넣었습니다. `realtime_minute_bars`에 실제 컬럼이 이미 저장돼 있었고
`load_minute_microstructure`가 그걸 읽고 있었는데도, 그 값이 feature 벡터로 배선되지 않았을
뿐입니다. 상세는 [ontology_and_gnn.md](ontology_and_gnn.md#context-계약-realtime_strategy_graph_v5_aligned).

직전 v4 체크포인트는 `rgcn_shadow.pre-v5-aligned.{npz,json}`에 보존돼 있습니다. 롤백은 그 두
파일을 `rgcn_shadow.*`로 되돌리면 됩니다.

**승격이 곧 수익성 입증은 아닙니다.** 이 체크포인트의 선택 지표는
`selection_ranking_skill_established=true`(AUC 0.737, 순열 p=0.001)이지만
`selection_net_edge_established=false`(상위 10분위 순수익 CI가 0을 걸침, P(≤0)=0.48)입니다.
`live_authorized`는 스키마·표본 커버리지 요건이지 엣지 증명이 아니며, 실제 진입 권한은 여전히
`GnnRealtimeTrustEvaluator`의 전략별 실시간 검증을 통과해야 합니다.

런타임 그래프 연산은 고정 Gather, MatMul, Add, ReLU, Multiply, Concat, Squeeze입니다. 동적 sparse/scatter 연산은 없습니다. 하드 온톨로지 마스크는 학습 그래프 **밖에서** 적용되며 모델이 덮어쓸 수 없습니다.

출력은 종목×전략별 success, 제비용, 실패 조건부 손실, 성공 조건부 순이익, fill probability, holding time, uncertainty, utility, NoTrade probability입니다. 순효율은 `P(win)×E(net win) − P(loss)×E(net loss)`로 계산합니다.

체크포인트는 모델 보정 신뢰도만으로 주문하지 않습니다. `GnnRealtimeTrustEvaluator`가 전략별 최소 표본, Brier score, 불확실성, 순효율 부호 정확도, MAE를 먼저 검증하고, 양수 예측 표본의 실현 양수 비율과 평균 순효율까지 통과한 전략에만 `entry_authorized=true`를 부여합니다.

체크포인트 provenance는 런타임에서 fail-closed입니다. `input_feature_schema`가 현재 프레임과 다르거나 `live_authorized`가 아니면 shadow 서비스가 `MODEL_INPUT_SCHEMA_MISMATCH` / `UTILITY_MODEL_NOT_LIVE_AUTHORIZED`로 `NO_TRADE`를 내고 출력을 쓰지 않습니다.

## 8. 리플레이 하네스

| 하네스 | 명령 | 판정 기준 |
| --- | --- | --- |
| 기술적 예측 | `python scripts/replay_technical_prediction.py` | `data/models/technical_replay_reports/`, 실현 **net-after-cost** (gross hit rate 아님) |
| 거시–미시 온톨로지 | `python scripts/replay_macro_micro_ontology.py --from-bars <file>` | `data/models/macro_micro_replay_reports/`, 실현 net-after-cost와 `avg_edge_error_bps`, SELL/REDUCE 우선 확인 |
| 이벤트 시뮬레이터 | `app.backtesting.event_simulator` | 이벤트 시간 순서 limit 체결, 같은 봉 barrier는 보수적으로 stop 우선, 전략별 + NoTrade counterfactual |
| purged walk-forward | `app.evaluation.purged_walk_forward` | 라벨 horizon 중첩 purging + embargo |
| reality check | `app.evaluation.reality_check` | block bootstrap, fee-converted-loss ratio, cost-to-alpha ratio, Sharpe, drawdown |

시뮬레이터는 이벤트 시간 순서 limit 체결, 보수적 same-bar barrier 해소, 공유 비용 엔진을 통한 KRX 수수료·세금·spread·슬리피지·impact, fill/gross/net 수익·MAE/MFE·보유시간·청산사유, 그리고 **모든 공급 전략에 대한 counterfactual + NoTrade**를 산출합니다.

## 9. 과최적화 통제

- 시도한 전략·변형·파라미터 탐색·반복 평가 횟수를 추적합니다.
- 표본 구조가 허용하면 combinatorially symmetric cross-validation / PBO를 사용합니다.
- 최고 raw Sharpe 하나가 아니라 Deflated Sharpe Ratio 또는 동등한 다중검정 보정을 보고합니다.
- 거래 수가 적거나, 하나의 regime·하나의 종목·하나의 파라미터 지점에 의존하는 전략은 승격하지 않습니다.
- 신뢰구간은 block bootstrap 등으로 시계열 종속성을 반영해야 합니다.
- 승격에는 gross 개선이 아니라 통계적·경제적으로 유의한 **net** 개선이 필요합니다.
- 새 경로가 재시작 복구, 주문 idempotency, 포지션 재동기화, tail risk를 후퇴시키면 승격하지 않습니다.

## 10. 테스트 표면

```powershell
python -m pytest
```

`live` 주문을 제출하는 테스트는 없습니다. 모든 broker E2E는 mock, 기록된 이벤트, broker 시뮬레이션을 씁니다. 남아 있는 deprecation 경고는 기존 FastAPI `on_event`와 RDFLib `default_context` 관련입니다.

문서/설정 변경 후 빠른 확인:

```powershell
python -m pytest tests/test_web_live_flags.py tests/test_web_graph_payload.py tests/test_account_dashboard.py tests/test_kiosk_display_overview.py
python -m pytest tests/test_refactor_contracts.py tests/test_causal_order_journal.py tests/test_refactor_flags.py
python -m pytest tests/test_directional_short_ladder.py
```

### 숏 사다리 안전 속성 (`tests/test_directional_short_ladder.py`, 76건)

각 테스트는 위반 시 **미검증 숏에 실자금이 들어가는** 규칙을 고정합니다.

| 그룹 | 고정하는 것 |
| --- | --- |
| 방향 산술 | 숏 목표는 진입가 **아래**, 손절은 **위**. barrier 비교 부호. 워터마크는 최저가. 숏 목표가 롱보다 **크다**(비용이 높으므로) |
| 주문 의미 | 롱 청산과 숏 진입이 둘 다 SELL이지만 다른 주문. `TradePlan`이 모순된 side 거부. `position_effect` 추론이 롱 청산을 진입으로 재분류하지 않음 |
| 배포 게이팅 | 롱/숏 posterior 미합산. 이력 없는 숏이 롱 이력을 빌리지 않음. 실행 불가 신호가 승격 통계에 미포함. `SHADOW` arm은 +300bps 엣지에도 선택 불가 |
| 전이 합법성 | `SHADOW→LIVE_FULL` 등 9개 금지 전이 도달 불가. 승격은 정확히 한 칸. `SUSPENDED`는 `SHADOW`로만 복구. YAML `initial_state` clamp. 저장 경계에서 재검사 |
| 승격/강등 비대칭 | 연속 사이클 전량 필요. 실패 1회가 카운터 **리셋**(감소 아님). hard gate 하나로 차단. 강등은 첫 실패에 즉시. 즉시 중단 8종 |
| 누수 방어 | `signal_at` 이하 quote 거부. bid 진입/ask 청산. 양 barrier 동시 통과 시 **손절**. 미래 timestamp snapshot 거부. 대주 안분이 연율 |
| 회귀 | 숏은 append(인덱스 안정). `NO_TRADE` arm 유지. 롱 수익성 판정 불변. **롱 청산과 숏 상환 모두 게이트되지 않음** |

## 11. 아직 승격하지 않은 것 (명시)

- strategy-utility NPU 추론은 CPU 대비 지연과 golden tolerance 문제로 미승격입니다.
- 이벤트 모멘텀·횡단면 상대강도·갭 전략은 필요한 point-in-time 사실이 없으면 closed-world로 차단됩니다.
- `opening_range_breakout`과 `market_intraday_momentum`의 세션 가격 사실은 이제 라이브와
  counterfactual이 동일 계산 함수를 사용합니다. 다만 30분 구간 미완성, 전일 종가 부재,
  과거 변동성 표본 3개 미만이면 필드는 생략되고 네 개 세션 전략은 fail-closed입니다.
  이 배선 변경은 배포 권한을 올리지 않으며 해당 long 2개와 short 2개는 계속 SHADOW입니다.
  기존 counterfactual의 bar-count 근사와 수치가 달라질 수 있으므로 schema를
  `counterfactual_quantiles_v2_session_structure`로 올렸습니다. v1 label/model card/checkpoint는
  재사용하지 않고 label 재생성 및 재학습이 필요합니다.
- LIVE DECISION ONTOLOGY의 `event_feed`·`cross_section`·`session_context` 가용성은 입력에
  반응하는 테스트로 고정했습니다. 특히 세션 시계만 있고 가격 사실이 없으면 `PARTIAL`이며,
  `sector_rank_table`에 대상 종목의 실제 순위가 있으면 횡단면 source/indicator가 available입니다.
- `calibrated_strategy_ids`에 있다는 사실만으로 실거래 진입 권한을 주지 않습니다.
- 대표성 있는 시장·regime별 out-of-sample alpha와 다중검정 보정 우월성은 아직 주장하지 않습니다.
- **숏 전략 3개의 수익성은 어떤 형태로도 측정되지 않았습니다.** forward 표본이 0이며, 이것이
  정상 상태입니다. `SHADOW → LIVE_PROBE`는 실행 가능 신호 120건 / 체결 60건 / 거래일 20일 /
  종목 10개 / confidence 0.72 / 보수적 순엣지 8bps / cost coverage 1.7배 / holdout 3구간 /
  연속 5사이클을 요구하므로 최소 한 달 이상 걸립니다.
- 숏 백테스트 결과는 **의도적으로 만들지 않았습니다.** 과거 시점 대주 가능 여부 데이터가 없어
  백테스트는 당시 대주 불가였던 종목을 숏 친 것으로 계산하는데, 대주 불가 종목은 정확히 가장 많이
  하락하는 종목군이므로 편향이 크고 항상 유리한 방향입니다. 롱 결과의 부호 반전도 같은 이유로
  금지입니다(비용 비대칭).
- **추측한 KIS 대주 엔드포인트 3개를 실계좌 read-only로 전부 검증했고, 전부 틀렸습니다**
  (2026-08-02): `TTTC8909R`은 융자 매수가능금액(질문이 다름), `CTSC0271R`은 없는 TR,
  `CTRP6504R`은 404. 네 번째 추측 대신 가용성 조회를 `BorrowDataSource` 인터페이스 뒤로
  옮겼고 기본값은 명시적 "소스 없음"입니다. **따라서 대주 저널이 비어 있어 숏은 shadow 표본을
  쌓지 못합니다** — 사다리는 끝까지 배선되어 있으나 가동 중지 상태입니다.
  `get_borrow_balance`는 프로덕션이 이미 쓰는 `inquire-balance`로 이전해 검증되었습니다.
- **GNN 대주 채널(8→11)은 구현되었으나 학습되지 않았습니다.** 새 채널의 라벨(실현 대주 비용,
  실현 locate 성공률)이 forward 표본에서 나와야 하는데 그 표본이 0입니다.
  `model_calibrated=False`가 고정되어 모든 arm이 그 게이트에서 막힙니다.
- **스퀴즈 필터가 비활성입니다.** KRX 공매도 잔고를 수집하는 곳이 없고, 저장소 내 유일한
  `short_net_change`는 합성 데모 파이프라인 값이라 리스크 측정에 쓸 수 없습니다.
  `max_days_to_cover` / `max_short_interest_ratio` 게이트는 무조건 통과하며, 이 방어 심도
  감소는 `/api/short-strategies/status`의 `indicator_gaps`로 보고됩니다.
- `short_rescue_rate`는 `RealtimeTradingEngine._run_short_cycle`에서 집계·주입되도록 배선을
  완료했습니다. 값이 없으면 `None`이 되어 **승격을 차단**하므로 실패 방향은 안전합니다.
- 숏 보조 지표 중 `spread_bps` / `liquidity_score` / `market_alignment`는
  `app.features.short_indicators`가 계산합니다. `short_interest_ratio` / `days_to_cover`는
  소스가 없어 미측정이며, 그 사실이 명시적으로 보고됩니다.

## 13. feature schema v6 — 지표 계열 계층 (2026-08-06)

v6은 22개 컬럼을 **추가**합니다: 6개 계열 점수 + 5개 availability 플래그 + 11개 scale-free
스칼라. 40 → **62 컬럼**, hash `f3f8f330a8cf04969e8545fa`.

19개 원시 지표를 그대로 넣지 않은 이유는 대부분의 원시 값이 **종목별 수준(level)** 이고,
그것이 §12에서 제거한 바로 그 정체성 실패 모드이기 때문입니다. 추가된 컬럼은 전부 scale-free
이거나 0/1 플래그입니다.

**마이그레이션 비용 (우회하지 않음).** v4→v5는 부분집합 축소라 저장 행을 재스탬프할 수
있었지만, v6은 **추가**이므로 기존 10만 행에 새 컬럼이 존재하지 않아 재스탬프가 불가능합니다.
따라서:

- v6 모델은 신규 행을 새로 쌓아 학습해야 합니다(콜드 스타트).
- 그때까지 registry가 v5 아티팩트를 schema mismatch로 거부하고, §9.5의 단계적 강등이
  ontology/bandit 경로로 진입 평가를 유지합니다.
- `live_signal_predictor`가 `MODEL_FEATURE_SCHEMA_MISMATCH` / `MODEL_FEATURE_ORDER_MISMATCH`를
  가중치 적용 **전에** 던지므로, 구 아티팩트가 새 벡터를 채점하는 일은 구조적으로 불가능합니다.

구 모델을 새 스키마에 억지로 매핑하지 않은 이유: 모든 가중치가 잘못된 컬럼을 읽게 되며,
이는 모델 없이 거래하는 것보다 나쁩니다.

### 열린 문제 (2026-08-05 기준)

- **KR 단기 모델이 라이브 부적격입니다.** precision@k 0.333 vs 임계 0.35. v4의 0.222에서
  올랐고 top-K 순수익은 −20.43 → +7.30bps로 부호가 반전됐지만 임계 미달입니다. KR 학습
  데이터가 소수 종목 편중(단일 종목 26.7%)에서 나왔으므로 구독 폭이 넓어진 뒤 재평가가 필요합니다.
- **구독 티어링 · dwell · depth floor가 KRX에서 검증되지 않았습니다.** 코드와 단위 테스트는
  통과했고 라이브 설정 계산도 확인했으나(30종목 / 6 depth → 하한 적용 후 재계산), KRX 수집기가
  아직 한국장 세션을 만나지 않았습니다. 검증 지점은 다음 KR 개장 시 수집 로그의
  `depth_symbols` / `trade_only_symbols` / `registrations_spent`입니다.
- **v5 이전 프레임은 `book_quality`가 없어 fail-closed로 탈락합니다.** 의도된 안전 방향이지만,
  스키마 전환 직후 약 1시간 구간의 프레임은 학습에 쓰이지 않습니다. 재스탬프하지 않은 이유는
  저널이 provenance 기록이고 해당 행이 이미 materialized store에 있기 때문입니다.
- **KRX 세션 판정에 `SESSION_CALENDAR_SUSPECT`가 관측됩니다.** 마감 시간대에는 정상이지만
  개장 시각에도 남아 있으면 휴장일 캘린더 소스를 확인해야 합니다.
- **소액 계좌 · 통화 불일치.** 미국 후보만 열린 시간대에 USD 주문가능액이 $67.57 수준이면
  의미 있는 사이즈가 나오지 않습니다. 원화 잔고로는 미국 종목을 바로 살 수 없고, 환전은 이
  저장소가 처리하지 않습니다(브로커가 반영해 줄 때까지 스냅샷은 변하지 않습니다).
- **decision/feature JSONL 로그 성장이 여전히 통제되지 않습니다.** 관측값:
  `refactor-shadow-comparison.jsonl` 1.59GB, `decision-log.jsonl` 236MB,
  `live-feature-frames.jsonl` 73MB. 저널만 크기 기반 회전이 있고 나머지는 정책이 없습니다.
- **`tests/test_web_graph_payload.py::test_full_graph_payload_bypasses_ui_trimming`이 순서
  의존적입니다.** 단독 실행 시 통과, 전체 스위트에서 실패. 그래프 payload 경로의 공유 상태
  문제로 보이며 원인 미규명입니다.

CPU GNN은 실시간 판단에 연결되어 있지만, 각 전략의 신규 진입 권한은 live forward 증거에 따라 자동으로 부여되거나 회수됩니다. 숏 arm의 권한은 이와 **별도로** arm별 배포 상태에서 관리됩니다 — 상세는 [short_selling_deployment.md](short_selling_deployment.md).

## 12. 정체성 feature 제거 — schema v4 → v5 (2026-08-05)

### 12.1 증상과 오진 가능성

단기 모델이 671 사이클 연속 라이브 게이트에 실패했습니다. 표면 지표는 모순돼 보였습니다:
AUC 0.667–0.694(양호)인데 precision@top-1%는 0.148–0.343으로 **양성 base rate 0.395를 크게
밑돌았고**, top-K 순수익이 −16 ~ −51bps로 지속 음수였습니다.

"모델에 엣지가 없다"로 결론내면 오진입니다. 예측 확률 십분위로 쪼개면 랭킹은 정상입니다:

| 십분위 | label률 | 순수익 |
| --- | --- | --- |
| **D1 (최상위)** | 0.495 | **−0.66** ← 패턴 붕괴 |
| D2 | 0.685 | **+25.29** |
| D3 | 0.675 | +25.96 |
| D4 | 0.588 | +22.76 |
| D5 | 0.495 | +5.07 |
| D6–D10 | 0.30 → 0.12 | −13 → **−43** |

D2→D10이 완벽히 단조입니다. **최상단만 뒤집혀 있었습니다.**

### 12.2 원인 — feature가 시장 상태가 아니라 종목 정체성을 인코딩

top-53 중 **44개(83%)가 단일 종목**이었습니다. 종목간분산/총분산 비율(1.0 = 순수 정체성):

| feature | 비율 | 성질 |
| --- | --- | --- |
| `bid_depth` / `ask_depth` | 0.986 / 0.984 | 정규화 안 된 원시 주식 수 |
| `box_high` / `low` / `mid` / `previous_close` | 0.943 | 원시 가격 수준 |
| `liquidity_score` | 0.891 | 로그 거래량 절대값 |
| `realized_volatility_3m` | 0.779 | 종목 고유 변동성 수준 |
| `return_5s` / `volume_spike_ratio` (대조) | 0.004 / 0.001 | 순수 상태 |

LP 유동성을 받는 ETF 한 종목이 `ask_depth` z=+4.03(1,838만주 vs 평균 138만주)에 고정되고,
**그 종목의 최소 depth조차 대부분 종목의 최대치보다 높습니다**(z=+1.71). 모델이 이 feature에
양의 가중치를 주므로 최상단을 영구 점거합니다. 즉 precision@k는 모델 실력이 아니라 **그 한
종목의 홀드아웃 구간 운**을 측정하고 있었습니다. AUC는 전 pair 순위라 이 오염에 둔감합니다.

### 12.3 조치와 측정

8개를 모델 벡터에서 제거했습니다(40 컬럼). 각각의 scale-free 대응물이 **이미 feature set에
존재**해 정보 손실이 없습니다: `depth_ratio`(0.208), `box_position`(0.211), `box_width_pct`,
`breakout_distance_bps`, `orderbook_imbalance`(0.436).

동일 홀드아웃 split 재학습:

| 지표 | v4 (48) | v5 (40) |
| --- | --- | --- |
| AUC | 0.727 | **0.736** |
| precision@k | 0.148 | **0.444** (base rate 0.395 상회) |
| top-K 순수익 | **−50.96bps** | **+18.32bps** |
| top-K 단일종목 집중 | 44/54 | 13/54 |
| 라이브 게이트 | fail | **PASS** |

라이브 재학습(같은 사이클, 수 분 차이):

| registry | v4 | v5 |
| --- | --- | --- |
| combined | 부적격 · auc 0.665 · p@k 0.466 · **−9.97bps** | **적격** · auc 0.720 · p@k 0.621 · **+19.75bps** |
| KR | 부적격 · auc 0.685 · p@k 0.222 · **−20.43bps** | 부적격 · auc 0.700 · p@k **0.333** · **+7.30bps** |
| US | 적격 · auc 0.802 · p@k 1.000 · +28.58 | 적격 · auc 0.810 · p@k 1.000 · **+29.51** |

승격된 아티팩트 `live_short_horizon.20260805T140115987083Z` (auc 0.7471 / p@k 0.621 /
**+21.41bps**)로 게이트가 671 사이클 만에 처음 통과했습니다(`score 1.0/0.9`).

**증거의 한계 (명시).** 홀드아웃 split 1개와 라이브 재학습 소수 사이클입니다. 대표성 있는
out-of-sample alpha 주장이 아닙니다. **KR은 여전히 부적격**(p@k 0.333 vs 0.35)이며, KR 학습
데이터는 소수 종목 편중(단일 종목 26.7%) 상태에서 나온 것이라 구독 폭이 넓어진 뒤 재평가가
필요합니다.

### 12.4 스키마 변경이 조용히 깨뜨린 3곳

feature set 하나를 바꾸자 세 곳이 **모두 조용히** 실패했습니다. 방어가 얇다는 신호로 기록합니다.

| 위치 | 실패 방식 | 조치 |
| --- | --- | --- |
| `_row_matches_live_schema` | 저장된 해시 불일치로 학습 행 10만 개 전량 폐기(콜드 스타트) | 신규 set이 진부분집합임을 확인 후 백업·해시 재스탬프 |
| `_promotion_decision` | 스키마 무시 지표 비교로 죽은 incumbent가 교체를 차단 | `OBSOLETE_SCHEMA_INCUMBENT_REPLACED` |
| `_frame_passes_training_quality` / `_frame_passes_label_path_quality` | `values.get("bid_depth", 0.0)` → 0.0 → `>0` 실패로 **모든 v5 프레임 탈락, 행 유입 73분간 정지** | 깊이를 `book_quality` 블록으로 분리, 구 프레임은 `values` 폴백 |

세 번째가 가장 위험했습니다: 프레임 저널의 `values`는 **정확히 모델 feature 벡터**이므로,
품질 게이트가 거기서 값을 읽는 순간 "모델 입력 변경"이 "데이터 무결성 검사 무력화"가 됩니다.
깊이는 모델 입력이 아니라 "이 호가창이 정상인가" 검사이므로 별 필드로 옮겼습니다.

## 14. 검증 프레임워크 — `app.strategy_validation`

§1.1 의 표를 만든 도구입니다. 존재 이유는 하나입니다: **모든 전략이 같은 evaluator 를 통과해야
서로 비교 가능합니다.** 이전에는 각 전략의 증거가 있는 곳에서 왔습니다 — 하나는 저장된 분봉
스크리닝, 하나는 체크포인트의 시뮬레이션 체결, 하나는 아무것도 — 그리고 그것들이 같은 측정처럼
비교됐습니다.

**모든 지표는 데이터가 답할 수 없으면 `None` 을 반환합니다.** 빈 셀이 0으로 읽히는 검증
프레임워크는 측정한 적 없는 전략에 대해 자신 있는 판정을 내리고, 이 카탈로그에는 그런 전략이
여럿 있습니다.

### 14.1 유효 표본 수는 행 수가 아니다

시간이 겹치는 거래는 독립 관측이 아닙니다. 이 프로젝트는 그 오차의 크기를 이미 측정했습니다 —
`stride < horizon` 이 표본 수를 **56배** 뻥튀겼습니다. 그래서:

* `effective_sample_count` — 각 거래는 같은 심볼의 이전 거래가 덮지 않은 horizon 비율만 기여
* `purged_cv` — 테스트 폴드와 **겹치는** 학습 거래는 purge, 그 직후 구간은 embargo.
  embargo 기본값은 **표본의 중위 horizon** 입니다. 150초 스캘프와 64,800초 오버나이트 캐리에
  같은 상수를 쓰면 최소한 한쪽은 틀립니다
* `combinatorial_splits` — CPCV. 잘라낸 조합 수 상한(`max_paths`)에 걸리면 **더 적은 split 을
  반환**해서 부분 스윕이 전체 스윕처럼 보이지 않게 합니다
* 신뢰구간은 행 수가 아니라 유효 표본 수로 계산하고, 할인이 컸다면
  `METRICS_OVERLAPPING_TRADES_DISCOUNTED` 로 보고합니다

### 14.2 비용 스트레스와 파라미터 안정성

**`break_even_cost_multiple`** — 비용이 측정치보다 몇 배 나빠질 때까지 엣지가 살아남는가.
1.05배에서 죽는 엣지는 평균이 뭐라 하든 거래 불가입니다. 이 프로젝트는 US 왕복 비용이 KRX 기준
표의 28bps 가정 대비 중위 63.2bps (p90 125.2) 로 들어온 것을 이미 측정했으므로, 평균 측정 비용만
쓰는 검증은 거래 가능성에 대해 아무 말도 하지 않습니다.

스프레드 충격은 rate 배수가 아니라 **가산 bps** 로 넣습니다 — 충격은 요율을 스케일하지 않고
basis point 를 더하며, 하필 전략이 청산하려는 순간에 벌어집니다.

**파라미터 안정성** — 결과가 논지의 속성인지 임계값 하나의 속성인지. `range_support_reversion` 은
10bps 에서 t=3.01, 25bps 에서 t=2.91 (평지) 이지만 확인 필터를 넣자 −2.3bps → −35.8bps 로
무너지고 표본이 207 → 38 로 줄었습니다(절벽). 그 둘을 구분하는 것이 이 모듈이고,
**최적화는 일부러 하지 않습니다** — "최적값은 X" 를 보고하면 노브를 X 로 옮기라는 초대가 되고
그것이 단계만 늘린 curve fitting 입니다.

**레짐 분해** — 구간별 평균과 **하단**을 함께 내고, 최고/최저 구간의 신뢰구간이 겹치지 않을 때만
`discriminates=True` 입니다. 8건 표본 두 개로 "TREND_UP 에서는 되고 RANGE_BOUND 에서는 안 된다"는
이야기를 만들지 않기 위한 것입니다.

**증거원 가중** (`config/strategy_validation.yaml`) — LIVE 1.0 / LIVE_PROBE 0.7 / SHADOW 0.3 /
BACKTEST 0.0. **이 값들은 측정치가 아니라 초기 설계값입니다.** shadow-live 짝 outcome 이 쌓이면
재캘리브레이션해야 합니다. 가중 평균과 **live 전용** 평균을 나란히 보고하는 이유는 둘이 다른
질문에 답하기 때문입니다 — 가중 평균은 최선의 추정치, live 전용은 승격이 근거로 삼을 수 있는
유일한 값입니다.

### 14.3 lifecycle 원장 — 전이는 화이트리스트다

`app.strategy_validation.registry` 는 `app.trading.directional.ALLOWED_TRANSITIONS` 와 같은 정신입니다.
rank 비교가 아니라 화이트리스트인 이유: rank 비교는 `RESEARCH → LIVE` 를 "4칸 상승" 으로 조용히
허용하고, 그것이 이 서브시스템이 존재하는 이유 그 자체입니다.

```text
RESEARCH → VALIDATED → SHADOW → LIVE_PROBE → LIVE     (한 칸씩만)
DEGRADED → SHADOW                                      (drift 강등에서 복귀)
```

승격에는 `StrategyValidationRecord` 첨부가 **필수**입니다. record 없는 승격은 거부되므로
"괜찮을 것 같다" 가 권한이 될 수 없습니다. LIVE_PROBE / LIVE 로 가는 두 전이만 게이트를 받습니다.

| 게이트 | 기본 | 이유 |
|---|---|---|
| `minimum_samples` | 30 | — |
| `minimum_lower_bound_bps` | 0.0 | **1차 기준은 양수 평균이 아니라 양수 하단**입니다. 평균을 비용과 비교하면, 오차 폭이 넓고 평균이 살짝 양수인 전략이 계속 거래합니다 — 분포의 절반이 비용 아래인데 비용은 매번 지불됩니다 |
| `minimum_break_even_cost_multiple` | 1.25 | 측정 비용의 1.25배까지 살아남아야 함 |
| `minimum_out_of_sample_stability` | 0.6 | walk-forward 창 중 양수 비율. 예외적인 한 창이 전체 평균을 지탱하는 전략을 잡음 |
| `require_parameter_stability` | true | — |
| `require_live_evidence_for_live` | true | **shadow 증거로는 LIVE 에 갈 수 없습니다.** 하지 않은 거래로 승격할 수 없습니다 |

**강등은 승격보다 쉽습니다.** 하향 전이는 증거를 요구하지 않습니다 — 완성된 연구 없이는 강등을
거부하는 것이 망가진 전략이 계속 거래하는 방식입니다. drift 모니터의 자동 강등은 `actor` 를
기록하므로 사람이 한 것과 자동인 것을 구분할 수 있습니다.

### 14.4 drift 강등은 한 번의 손실로 발화하지 않는다

`app.monitoring.strategy_drift` 의 강등 제안은 **최소 표본 + rolling net EV 음수 + 하단 음수**를
동시에 요구합니다. 셋 중 하나만으로는 노이즈입니다 — −60bps 체결 하나가 12표본 평균을 5bps
움직이는데 그건 추정기 자체 오차 안입니다.

한 칸씩만 강등합니다 (`LIVE → DEGRADED → SHADOW`). `LIVE → SHADOW` 엣지는 없습니다. 두 칸
낙하는 "측정은 되지만 거래는 안 하는" 상태를 건너뛰고, 다음 판단의 증거가 바로 거기서 나옵니다.

비용/드로다운 플래그 단독은 **검토 신호이고 강등이 아닙니다** — 해법이 horizon 이나 거래 venue 일
수 있고 전략 자체가 아닐 수 있습니다.

모니터는 **제안만** 합니다 (`DemotionProposal`). 적용은 §14.3 의 원장을 통한 별개 행위입니다.
모니터가 lifecycle 을 조용히 다시 쓸 수 있으면 운영 설정을 검토할 수 없게 됩니다.

### 14.5 전략 문제와 selector 문제 분리

`app.evaluation.selector_evaluator` 가 명세의 진단 표를 구현합니다. 검사 **순서**가 임의가
아닙니다 — 비용/horizon 검사가 **먼저** 돕니다. gross 양수 net 음수 전략은 net 숫자만 보면
죽은 논지와 구별되지 않는데, 해법은 완전히 다릅니다(horizon 을 바꾸거나 그 venue 를 그만두는
것이지 논지를 버리는 것이 아님).

| oracle (적격·발화한 컨텍스트 전체) | router-selected (selector 가 실제로 고른 시점) | 판정 |
|---|---|---|
| gross 양수, net 음수 | — | `COST_OR_HORIZON_PROBLEM` |
| 컨텍스트별 부호가 뒤바뀜 | — | `CONTEXT_MODELLING_PROBLEM` |
| 음수 | 무관 | `STRATEGY_PROBLEM` |
| 양수 | 음수 | `SELECTOR_PROBLEM` |

`selector_regret = best_available_outcome − selected_outcome` 이고 **NO_TRADE 가 경쟁자**입니다.
selector 가 거절했을 때 실현 결과는 0.0 이며(노출 없음, 비용은 거래에만 발생), 모든 대안이 손실인
그룹에서 regret 은 **음수**가 됩니다 — 거절이 어떤 거래보다 나았다는 뜻이고, 그것이 표현
가능해야 거절이 보상받을 수 있습니다.

regret 은 **행 단위가 아니라 컨텍스트 단위**로 계산한 뒤 컨텍스트에 대해 평균합니다. 한 컨텍스트의
대안들은 같은 가격 경로를 다른 barrier 로 자른 것이라 독립이 아니고, 행 평균은 대안 9개인
컨텍스트를 대안 1개인 컨텍스트보다 9배 무겁게 셉니다.

**데이터 수집은 시작됐지만 승격 증거로는 아직 부족합니다.** 2026-08-11 23:09:33 KST 영속
스냅샷은 해소된 context 66개, 실제 전략 선택 context 0개, distinct day 1개입니다. 모두 NO_TRADE라 선택
순수익과 regret은 0이고, 이는 양의 수익 표본이 아닙니다. Windows 런처가 V2와 자동 승격을 켜므로
라이브 세션 동안 계속 누적됩니다. 최신값은 §14.6의 API에서 확인합니다.

### 14.6 커버리지 — 시장이 있는 상태에 전략이 존재하는가

`app.strategy.coverage` 가 컨텍스트를 6축 4,800 셀로 버킷하고 셀마다 적격 수 / 발화 수 /
**선택 가능(검증된) 수**를 셉니다. 세 번째가 중요한 숫자입니다 — 전략이 발화하지만 선택 가능한 것이
없는 셀은 시스템이 희망으로 거래하는 셀입니다.

`validated_positive_strategy_count == 0` 이면 `STRATEGY_COVERAGE_GAP` 을 기록하고 그 셀은
NO_TRADE 로 해소합니다. **가장 가까운 기존 전략을 강제 실행하지 않습니다** — 그것이
`forced_selection` 결함이고, 미검증 유사 전략으로 공백을 메우는 것은 전략 추가 정책의 금지
항목입니다. 반복되는 공백은 연구 후보로 집계되며, 그것이 카탈로그가 커지는 정당한 경로입니다.

이 경로로 추가된 첫 전략이 `bar_trend_continuation`입니다. 1초/5초 체결창이 준비되지 않은
세션에서 완성된 1분봉 추세, VWAP 위치, 추세 지속성, 상대거래량을 사용합니다. 현재 lifecycle은
SHADOW이며 다른 전략과 똑같이 비용 차감 후 양의 forward 하단을 입증해야 합니다. coverage gap
탐지는 새 전략의 **연구 필요성**을 자동 발견하지만, 검증되지 않은 코드를 자동 생성하거나 곧바로
실주문 권한을 주지는 않습니다.

따라서 시스템의 목표는 모든 시장·거래 상태를 분류하고 가장 나은 **검증된** action을 고르는
것입니다. 그 action에는 `NO_TRADE`도 포함됩니다. 현재 수익 가능한 알고리즘이 없다는 사실은 새
지표·horizon·venue 전략을 연구할 신호이지, 손실 예상 거래를 강제로 만드는 근거가 아닙니다.

coverage는 V2가 켜진 live cycle마다 자동 누적됩니다. 정확한 현재 수치는
`GET /api/realtime-trading/status`의 `strategy_session.selector_v2.coverage`에서 확인합니다. 권한 승격은
coverage 셀 수가 아니라 별도의 context별 수익·안정성 기준으로 결정됩니다.

### 14.7 테스트 표면

| 파일 | 건수 | 무엇을 고정하는가 |
|---|--:|---|
| `test_market_context.py` | 11 | 결정론적 생성, 0 스프레드는 결측(측정된 0 아님), field별 provenance |
| `test_strategy_spec_registry.py` | 12 | append-only 순서, 선언된 요구사항이 모두 실제 field, 숏 RESEARCH 고정, 권고는 낮추기만 |
| `test_strategy_proposal_contract.py` | 12 | 주문 field 부재, 방향 인식 target/stop, 알고리즘 예외는 not-ready proposal |
| `test_ontology_strategy_eligibility.py` | 15 | hard/soft 분할, generic methodology 주소 불가, 미해석 레짐은 block 아님 |
| `test_strategy_selector_v2.py` | 31 | 항 분해 합 = 총합, NO_TRADE 정상 결과, 양수 평균도 하단 규칙으로 거부, **AST 로 실행 계층 import 금지** |
| `test_bandit_adapter_bounds.py` | 9 | clamp 와 보고, 이력 0이면 보정 0, 얇은 표본 축소, context key 에 symbol 없음 |
| `test_counterfactual_and_regret.py` | 17 | barrier 보수적 우선순위, 선택 전략은 sink 로 안 나감, shadow 는 LIVE 로 셀 수 없음, NO_TRADE 경쟁 |
| `test_strategy_coverage_and_validation.py` | 26 | 중첩 표본 할인, 2칸 승격 거부, 증거 없는 승격 거부, shadow 만으론 LIVE 불가, 1회 손실로 강등 안 함 |
| `test_selector_v2_shadow_integration.py` | 12 | 라이브 evidence 모양으로 통과, 진단 payload 완비, 잘못된 입력에도 raise 안 함, live 권한은 안전 sub-flag 요구 |
| `test_selector_v2_auto_promotion.py` | 5 | 자동 단계 승격, full-live 실체결 요구, 빠른 강등, 영속화, 설정상 shadow와 실효 권한 분리, 단일 표본 안전성 |

자동 권한 변경 후 V2·세션·실시간 엔진·실행 안전 관련 회귀 묶음 **286건 통과**
(2026-08-11). 전체 저장소 테스트 수는 계속 변하므로 고정 총계로 승격 상태를 판단하지 않습니다.

상세 설계와 첫 audit 결과는 [strategy_selection_v2.md](strategy_selection_v2.md).

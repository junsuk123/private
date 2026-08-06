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

아래 값은 이전 `S=7` 체크포인트로 수행한 **역사적 NPU 승격 실험**입니다. 현재 `B1 T1 N8 F36 R3 S8` live 체크포인트의 NPU 승격 근거로 재사용하지 않습니다. 당시 고정 형상은 `B=1, T=4, N=16, F=12, R=4, S=7`, FP32, 30 iterations였습니다.

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

`data/models/strategy_utility/rgcn_shadow.json` 현재 상태:

```text
method                ontology_strategy_graph_rgcn_joint_gradient_calibration
input_feature_schema  realtime_strategy_graph_v4_market
feature_provenance    causal_minute_bar_microstructure_proxy_v1
rows                  11,696        snapshots 1,462
strategies            8개 각 1,462
config                B1 T1 N8 F36 R3 S8, hidden 16, seed 17
authorization_scope   ontology_gnn_realtime_trust_gated_execution
checkpoint_hash       9a3b4dbf1ced8052ae4a8ffa0705d1bb358ee07b6a3e0c368963f2be7dcef80b
authorization_checks  row/snapshot/strategy 커버리지 및 런타임 스키마 일치 통과
```

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

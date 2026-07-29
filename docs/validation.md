# Validation and Evidence

측정된 것, 측정되지 않은 것, 그리고 무엇이 승격되지 않았는지에 대한 문서입니다. 좋아 보이는 숫자를 만드는 것이 목표가 아니라, 어떤 주장이 증거를 갖고 있는지 분리하는 것이 목표입니다.

## 1. 승격 상태 요약

| 항목 | 상태 | 근거 |
| --- | --- | --- |
| 순수익 게이트 / 동적 청산 리팩터 | 프로덕션 적용 | 아래 리플레이 지표, `tests/test_profitability_refactor_integration.py` |
| 온톨로지 가속 (indexed graph + FactTable) | 프로덕션 적용, 동작 동일 | `scripts/benchmark_fact_table.py`, `tests/test_fact_table.py` |
| 전략 소유 실행 경로 | 구현 완료, live 미연결 | `config/refactor_profile.json` = `shadow`, `broker_submission_enabled=false` |
| 이벤트 시뮬레이터 + counterfactual 라벨 | 구현 완료, 성능 승격 거부 | `data/reports/refactor_counterfactual_evaluation.json` |
| strategy-utility R-GCN | shadow 추론 전용 | `data/models/strategy_utility/rgcn_shadow.json`, `authorization_scope=shadow_inference_only` |
| NPU 추론 | 컴파일 성공, **승격 거부** | `data/reports/strategy_utility_openvino.json`, `promotion_eligible=false` |
| legacy vs ontology vs tabular vs R-GCN 비교 | **미완** | 필요한 point-in-time 데이터 부재 |

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

고정 형상 `B=1, T=4, N=16, F=12, R=4, S=7`, FP32, 30 iterations.

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

**커버리지** (2026-06-29 ~ 2026-07-27):

```text
bars 27,884 · symbols 429 · distinct UTC dates 12 · 100봉 이상 symbols 22
snapshots 4,991 · strategy labels 34,937
configuration: US / NASD / overseas_stock, history 30봉, horizon 15봉, stride 5봉
```

**누수 통제:** rolling quantile은 직전 slice만 사용, feature cutoff가 라벨 구간에 선행, purging과 embargo 활성, walk-forward split 2개.

**전략별 결과** (triggered 기준 체결):

| 전략 | triggered | filled | fill rate | 평균 net bps | positive net rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| liquidity_shock_reversal | 3,768 | 3,617 | 0.960 | −72.23 | 0.0 |
| intraday_momentum | 3,059 | 2,913 | 0.952 | −72.20 | 0.0 |
| breakout_volume | 2,654 | 2,559 | 0.964 | −71.47 | 0.0 |
| vwap_mean_reversion | 199 | 153 | 0.769 | −72.73 | 0.0 |
| cross_sectional_relative_strength | 0 | — | — | — | — |
| event_momentum | 0 | — | — | — | — |
| gap_context | 0 | — | — | — | — |

**Tabular walk-forward baseline:** 학습 구간 전략 평균으로 양(+)의 기대 순효용만 선택하는 정책은 1,996개 test 관측에서 **0건**을 선택했습니다. 즉 일급 `NoTrade`를 올바르게 반환했습니다.

**필수 시스템 비교 상태:**

```text
legacy            UNAVAILABLE  (해당 스냅샷에 대한 legacy 판단이 저널링되지 않음)
ontology_only     NoTrade      (필수 event/sector/session 사실이 fail-closed)
tabular_baseline  평가됨
temporal_rgcn_cpu UNAVAILABLE_UNTRAINED
temporal_rgcn_npu UNAVAILABLE_UNTRAINED_AND_NPU_BENCHMARK_REJECTED
```

**한계 (리포트에 명시됨):**

- 로컬 분봉 스토어가 소수 US 세션에 집중되어 있고, 목표 시장인 KRX/NXT가 아닙니다.
- 분봉 OHLC로는 tick 수준 큐 포지션이나 봉 내부 barrier 순서를 복원할 수 없습니다.
- point-in-time 이벤트, 섹터 그래프, 권위 있는 세션 캘린더, 과거 legacy 판단이 없어 7개 전략 전체 비교가 불가능합니다.

`promotion_eligible=false`, `status=NOT_PROMOTED`. 리포트는 label-data / configuration / evaluation-code SHA-256 해시를 포함합니다. 유리해 보이는 합성 PnL을 만드는 것으로는 이 게이트를 통과할 수 없습니다.

## 7. Strategy-utility 모델 카드

```powershell
python scripts/train_strategy_utility_rgcn.py
```

`data/models/strategy_utility/rgcn_shadow.json` 현재 상태:

```text
method                causal_feature_encoder_plus_ridge_calibrated_heads
input_feature_schema  realtime_microstructure_v1
feature_provenance    causal_minute_bar_microstructure_proxy_v1
rows                  38,668        snapshots 5,524
strategies            7개 각 5,524
config                B1 T1 N1 F12 R1 S7, hidden 16, seed 17
authorization_scope   shadow_inference_only
authorization_checks  row/snapshot/strategy 커버리지 및 런타임 스키마 일치 통과
```

런타임 그래프 연산은 고정 Gather, MatMul, Add, ReLU, Multiply, Concat, Squeeze입니다. 동적 sparse/scatter 연산은 없습니다. 하드 온톨로지 마스크는 학습 그래프 **밖에서** 적용되며 모델이 덮어쓸 수 없습니다.

출력은 종목×전략별 success, gross return, cost, MAE, MFE, fill probability, holding time, uncertainty, utility, NoTrade probability입니다.

**out-of-sample alpha, 교정 품질, tabular baseline 대비 우월성은 주장하지 않습니다.** 실거래 승격에 여전히 필요한 것: 대표성 있는 KRX/NXT 커버리지, calibration curve, regime/symbol별 결과, block-bootstrap 신뢰구간, 다중검정 보정(Deflated Sharpe 또는 동등물).

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
python -m pytest              # 949 tests collected
```

`live` 주문을 제출하는 테스트는 없습니다. 모든 broker E2E는 mock, 기록된 이벤트, broker 시뮬레이션을 씁니다. 남아 있는 deprecation 경고는 기존 FastAPI `on_event`와 RDFLib `default_context` 관련입니다.

문서/설정 변경 후 빠른 확인:

```powershell
python -m pytest tests/test_web_live_flags.py tests/test_web_graph_payload.py tests/test_account_dashboard.py tests/test_kiosk_display_overview.py
python -m pytest tests/test_refactor_contracts.py tests/test_causal_order_journal.py tests/test_refactor_flags.py
```

## 11. 승격하지 않은 것 (명시)

- `REFACTOR_LIVE_ENABLED`, `strategy_owned_execution`, `gnn_rerank`, `npu_inference`는 비활성입니다.
- legacy 프로덕션 경로는 shadow/paper 수용 증거가 생길 때까지 유지됩니다. legacy 은퇴를 주장하지 않습니다.
- 새 historical/paper 증거 없이 canary/live를 켜는 것은 이 문서의 운영 규칙 위반입니다.

이것들은 구현 누락이 아니라 **승격 판정 결과**입니다.

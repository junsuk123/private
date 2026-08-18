# 컨텍스트 계층과 실거래 게이트

이 문서는 다음 체인을 다룬다.

```
Calendar/Session → GlobalContext → DomesticContext → SectorContext → StockMicroContext
  → OntologyGraph → TemporalHeteroGNN → Regime/TradeQuality → StrategySelector
  → RiskManager → FinalTradeGate → PositionSizing → KISExecution → FillReconciliation
```

각 단계는 독립적으로 테스트·재현 가능한 컴포넌트이고, `app.trading.context_decision_pipeline`
가 **순서와 기록**만 담당한다.

---

## 1. Calendar / Session — `app.context.session_phase`, `app.context.temporal_context`

세션 판정의 유일한 권한은 기존 `app.data.market_capabilities.MarketSessionService` 다.
이 계층은 두 번째 시계를 만들지 않고, 그 답을 **세션 궤적 위의 위치**로 투영한다.

| Phase | 의미 |
| --- | --- |
| `PRE_MARKET` | 장전/프리마켓/미국 주간거래 세션이 열려 있고 연속매매는 시작 전 |
| `OPEN_TRANSITION` | 연속매매 첫 `open_transition_minutes` — 시가 단일가 잔여 불균형이 해소되는 구간 |
| `OPENING` | `opening_minutes` 까지의 시가 레인지 |
| `MORNING_TREND` | 세션 길이의 `morning_end_fraction` 까지 |
| `MIDDAY` | `midday_end_fraction` 까지 |
| `AFTERNOON` | 종가 구간 직전까지 |
| `CLOSING` | 마지막 `closing_minutes` + 종가 단일가 세션 |
| `POST_MARKET` | 시간외/단일가/NXT 애프터 |
| `CLOSED` | 어떤 세션도 열려 있지 않음 |

경계값은 전부 `config/temporal_context.yaml` 이고, **캘린더가 해석한 개장/폐장 시각에
상대적인 분·비율**이다. 따라서 조기폐장·DST·설정 변경이 코드 수정 없이 반영된다.
(2026-11-27 미국 13:00 조기폐장에서 11:00 ET 의 `session_progress` 가 정상장보다 커지는
것을 `tests/test_temporal_context.py` 가 검증한다.)

`TemporalSnapshot` 은 위 phase 에 더해 `day_of_week`, `session_progress`,
`minutes_from_open`, `minutes_to_close`, `days_since_last_session`,
`holiday_adjacent`, `month_end`, `quarter_end`, `expiry_context` 를 제공한다.

* `holiday_adjacent` 는 **주말을 세지 않는다.** 달력일 기준으로 세면 모든 금요일과 월요일이
  휴일 인접으로 표시되어 `day_of_week` 의 재진술이 되어 버린다. Mon–Fri 중 휴장한 날만 센다.
* `session_progress` 는 연속매매 구간에서만 정의된다. 장 종료 후에도 1.0 을 보고하면 모델에
  포화된 상수를 주게 되고, 그 값으로 만든 seasonality 버킷은 종가와 새벽 3시를 함께 평균낸다.

### 요일·시간은 규칙이 될 수 없다

`day_of_week` 와 시계 위치는 **feature** 이며, BUY/SELL 로 가는 유일한 경로는
seasonality z-score 다. 고정 요일 규칙은 이미 끝난 구간에 적합된 상수이고, 레짐이 바뀌어도
같은 강도로 계속 발화한다.

---

## 2. Seasonality — `app.features.seasonality`

```
z = (x - mu[d, s, r]) / (sigma[d, s, r] + eps)
```

`d` 요일, `s` 세션 phase, `r` 레짐. 얇은 버킷은 metric 전역 baseline 으로 축소한다.

```
estimate = (n / (n + k)) * bucket + (k / (n + k)) * global      k = 30
```

억제하지 않고 축소한다 — sample_count 와 confidence 를 달고 있는 축소 추정치가, 없는 값보다
downstream 에서 유용하다. `confidence = n / (n + k)` 는 게이트와 사이징이 그대로 읽는다.

**누수 방지는 문서가 아니라 강제된다.**

* `observe()` 는 관측을 반영하기 **전** baseline 으로 점수를 매긴 뒤 병합한다. z-score 가
  자기 자신을 볼 수 없다.
* `last_observed_at` 보다 오래된 관측은 **거부**되고 버킷별로 카운트된다. 미래 관측을 흡수한
  baseline 은 되돌릴 수 없고, 그것을 조용히 받아들인 walk-forward 는 얻지 않은 점수를 보고한다.

롤링 윈도는 저장된 관측 링이 아니라 지수 감쇠다. 유효 표본수가 `baseline_window` 에서
포화하고 옛 레짐은 과거 재조회 없이 사라진다. `seasonality_baseline` 테이블에 영속화되어
재시작이 콜드 스타트가 되지 않는다.

---

## 3. Global → Domestic → Sector → Stock

### GlobalContext — `app.context.global_context`

`config/global_indicators.yaml` 의 8개 그룹(equity, semiconductor, risk, rates, fx,
commodity, asia, futures)에서 `direction, momentum, risk_sentiment, volatility,
rates_pressure, fx_pressure, global_alignment, confidence` 를 산출한다.

* 각 지표는 `tanh(change / scale)` 로 [-1, 1] 에 묶인다. 하드 클램프가 아니라 tanh 인 이유:
  클램프는 -3% 와 -8% 세션을 같은 점수로 만드는데, 그 구분이야말로 `risk_sentiment` 가
  존재하는 이유다.
* `global_alignment = mean / mean(|·|)` — 전부 같은 방향이면 ±1, 상쇄되면 0. 단순 평균은
  "전부 약하게 상승"과 "절반은 강한 상승, 절반은 강한 하락"을 구분하지 못한다.
* 관측 그룹 가중치가 `minimum_group_coverage` 미만이면 `direction=None` 이고 이유 코드가 붙는다.
  살아남은 한 계열로 방향을 만들지 않는다.

**GlobalContext 는 종목을 받지 않는다.** 구조적으로 한국 종목에 대한 의견을 가질 수 없다.

### DomesticContext — `app.context.domestic_context`

미국 약세는 국내 SELL 이 아니다. 이 규칙은 각 호출자에게 맡기지 않고 여기서 강제한다.

* `direction` 은 **국내** 가격·시장폭·수급에서만 측정한다.
* `global_agreement` / `global_confirmation` / `global_conflict` 가 글로벌과의 관계를 별도로 나른다.
* `confirms_global_weakness()` 가 글로벌 약세를 국내 결론으로 바꾸는 **유일한** 경로이며,
  direction·breadth·flow 세 개의 독립적인 국내 증인이 모두 음수여야 한다. 외국 시장은 국내
  조건의 증인이 아니다.

`venue_divergence` 는 KRX/NXT 통합 mid 괴리를 [0,1] 로 보고하고, **confidence 를 낮춘다.**
두 호가창이 어긋나면 그 위에 놓인 모든 가격이 덜 믿을 만하다.

### SectorContext — `app.context.sector_context`

`return, breadth, volume_z, volatility, relative_strength, foreign_flow,
leader_strength, leader_concentration, global_alignment`.

* `relative_strength = R_sector − beta · R_market`, beta 는 롤링 추정.
* `volume_z` 는 로그 공간 평균 — 한 종목이 평균 대비 40배를 찍어도 섹터 전체가 8배로 읽히지 않는다.
* 멤버 3개 미만이면 횡단면 통계(breadth, concentration)를 억제한다. 표본을 서술하는 값이지
  섹터를 서술하는 값이 아니다.

### Cross-market — `app.context.cross_market`

`RS = R_local − beta · R_reference`, beta 는 롤링 OLS. beta=1 가정은 저베타 방어주를 하락장에서
영구히 강해 보이게 하고 고베타 반도체를 영구히 약해 보이게 한다 — 종목선택으로 위장한 베타 노출이다.

윈도가 얇거나 reference 분산이 0이면 `beta=1.0` 으로 **표시된 채** fallback 한다.
`estimate_lead_lag` 는 유계 lag 범위에서 교차상관 최대점과 동시점 상관을 함께 보고한다.

---

## 4. OntologyGraph — `app.ontology.market_graph`, `config/market_graph_ontology.yaml`

노드 타입 11종, 관계 14종. 모든 엣지가 두 개의 가중치를 **따로** 나른다.

| 필드 | 출처 |
| --- | --- |
| `prior_strength` | 사람이 적은 전문가 사전확률 (YAML) |
| `learned_weight` | 학습이 측정한 값 (`ontology_edge` 테이블) |

둘을 하나로 합쳐 저장하면 이 그래프에 물어볼 가치가 있는 유일한 질문 — *모델은 어디서 전문가와
의견이 갈렸고, 갈릴 만했는가* — 이 사라진다. 모든 inference trace 에 둘 다 노출된다.

Prior 는 게이트가 아니라 **attention logit 의 soft bias** 로 들어간다. 온톨로지가 선언하지 않은
곳은 `-inf` 라 attention 이 관계를 발명할 수 없고, 선언된 엣지들 사이에서는 순위만 매긴다.

`BELONGS_TO`, `TRADED_ON`, `REQUIRES`, `INVALIDATES` 는 `learnable: false` — 가설이 아니라
정의와 하드 제약이다. 종목이 섹터에 75% 속한다거나, stale data 가 가끔만 논지를 무효화한다고
학습할 수 없어야 한다.

---

## 5. TemporalHeteroGNN — `app.models.temporal_hetero_gnn`

1. **노드 타입별 인코더.** `MacroFactor` 와 `Stock` 은 같은 종류의 객체가 아니고, 공유 인코더는
   둘을 한 basis 로 밀어 넣는다.
2. **관계별 attention + 온톨로지 prior bias.**
   `e[r,i,j] = LeakyReLU(a_r · [W_r h_i ‖ W_r h_j]) + scale · prior[r,i,j]`
3. **Causal TCN.** kernel 2, dilation (1,2,4). `y[t]` 는 `x[t]` 와 `x[t-d]` 만 읽는다 —
   미래 봉이 현재 예측에 닿을 수 없다는 것이 산술의 구조적 사실이지 호출자의 규율이 아니다.
4. **Heads.** `market_regime`(멀티라벨 sigmoid), `sector_strength`, `expected_return`(bps),
   `volatility`, `strategy_suitability`, `trade_quality`, `uncertainty`.

**모델은 주문할 수 없다.** `app.execution` / `app.risk` / 브로커 클라이언트로 가는 import 경로가
없고, 테스트가 그것을 검사한다.

NumPy 구현이다. Torch 는 이 프로젝트의 optional extra 이고 프로덕션 런타임에 없다.

### 런타임 상태 — `app.models.gnn_runtime`

| 상태 | 권한 |
| --- | --- |
| `HEALTHY` | 전량 사이징, 모델 증거 사용 |
| `DEGRADED` | 룰/컨텍스트 증거만, 사이즈 0.5배 |
| `OFFLINE` | **신규 진입 차단.** 기존 포지션 관리는 계속 |

비대칭이 이 클래스의 존재 이유다. 모델이 없는 것은 리스크를 열지 않을 이유이지, 닫지 못할
이유가 아니다.

### 학습 — `scripts/train_temporal_hetero_gnn.py`

```bash
python scripts/train_temporal_hetero_gnn.py --epochs 40
python scripts/train_temporal_hetero_gnn.py --relations-only   # 온톨로지 가중치만
```

저장된 결정과 해소된 결과에서 학습한다. 표본이 `MINIMUM_TRAINING_EXAMPLES` 미만이거나
손실이 시작점보다 개선되지 않으면 **체크포인트를 쓰지 않는다.** 미학습 가중치로 런타임을
HEALTHY 로 뒤집는 것이 OFFLINE 으로 남는 것보다 나쁘다.

---

## 6. Regime — `app.context.regime`

13개 라벨 각각이 **독립 확률**이다. 합이 1이 아니고 정규화하지 않는다. 시장은 `RISK_OFF` 이면서
`RANGE_HIGH_VOL` 이면서 `INDEX_UP_BREADTH_DOWN` 일 수 있고, 단일 라벨 분류기는 하나만 고르고
나머지를 버린다.

증거는 두 갈래로 유지된다: 결정론적 룰 점수(항상 사용 가능, 항목별 감사 가능)와 GNN head
(런타임이 HEALTHY 일 때만). `contributions` 에 양쪽이 남아, "모델은 BREAKDOWN 이라는데 룰은
아니다" 가 저장된 결정에서 답할 수 있는 질문이 된다.

어떤 라벨도 확률을 갖지 못하면 `dominant` 는 `UNKNOWN` 이다. 전부 0인 상태에서 `max()` 는
카탈로그 첫 라벨을 돌려주고, 대시보드에서는 확신에 찬 `TREND_UP` 으로 읽힌다.

---

## 7. StrategySelector — `app.routing.regime_strategy_selector`

```
score(f) = regime_term(f) + ontology_term(f) + model_term(f) + micro_term(f)
```

* `regime_term = Σ_r P(r) · (SUITABLE_FOR(r,f) − UNSUITABLE_FOR(r,f))`, 엣지의 **effective
  weight** 사용 (학습값 있으면 학습값, 없으면 prior — trace 에 둘 다 기록).
* `model_term` 은 런타임이 HEALTHY 일 때만. DEGRADED 는 낮은 가중치가 아니라 **0** 이다.
  출력이 의심스러운 모델은 덜 믿을 대상이 아니라 읽기를 멈출 대상이다.
* `micro_term` 은 패밀리별로 다른 계산이다. 하나의 강세 점수를 8번 재사용하면 micro 가
  regime 에 반대할 수 없게 된다.

활성 risk condition 의 `INVALIDATES` 엣지는 패밀리를 **완전히 제거**한다 — 점수도 순위도 없다.

`WAIT` 는 정상적인 성공 결과다. 최소 점수 미달, 상위 두 패밀리가 반대 방향으로 접전,
regime confidence 부족, 또는 `DEFENSIVE` 승리 시 반환된다. 8개 패밀리는 기존
`app.strategy.registry` 의 family map 에서 유도되어 두 분류체계가 어긋날 수 없다.

---

## 8. Data freshness — `app.data.freshness`, `config/data_freshness.yaml`

관측마다 세 개의 시각을 추적한다.

| 시각 | 답하는 질문 |
| --- | --- |
| `event_time` | 거래소가 말하는 발생 시각. `age = now − event_time` |
| `received_time` | 수신 시각. `received − event` 는 **피드** 지연 |
| `processed_time` | feature 화 완료 시각. `processed − received` 는 **우리** 지연 |

age 만으로는 "시장이 조용하다", "웹소켓이 멈췄다", "우리 파이프라인이 밀렸다"를 구분할 수
없는데 세 상황의 대응은 전혀 다르다.

`critical` 스트림이 `STALE` 이면 신규 주문이 차단된다(`STALE_DATA`). **청산은 차단하지 않는다.**
관측이 아예 없는 스트림은 부재가 아니라 `STALE` 이다 — "호가를 한 번도 못 받았다"와
"호가가 400ms 됐다"가 같은 결론이 되면 안 된다.

---

## 9. FinalTradeGate — `app.risk.final_trade_gate`, `config/final_trade_gate.yaml`

### Hard gate (설정 불가, 마스킹 불가, 모델이 무력화 불가)

`STALE_DATA`, `WS_DISCONNECTED`, `PRICE_FEED_CONFLICT`, `UNKNOWN_SESSION`,
`TRADING_HALT`, `ACCOUNT_RECONCILIATION_FAIL`, `UNKNOWN_ORDER_STATE`,
`DUPLICATE_ORDER_RISK`, `MODEL_INFERENCE_FAIL`, `RISK_ENGINE_FAIL`.

이들은 "나쁜 조건"이 아니라 **주문이 의존하는 무언가를 시스템이 모르는 상태**다. 입력이
누락되면 전부 차단으로 귀결된다 — 계좌 상태를 안 넘긴 호출자는 아무것도 없는 데서 계산된
승인 대신 거절을 받는다. `ACCOUNT_RECONCILIATION_FAIL` 을 끌 수 있는 임계값이 있다면 게이트는
권고에 불과해진다.

### Soft gate (사이즈 축소, 복합)

`HIGH_VOLATILITY`, `LOW_LIQUIDITY`, `GLOBAL_CONFLICT`, `SECTOR_CONFLICT`,
`LOW_MODEL_CONFIDENCE`, `OPENING_EXTREME_VOL`, `ABNORMAL_SPREAD`.

승수가 **곱**해진다. 가벼운 문제 둘이 하나보다 작은 포지션을 만든다. 복합 승수가
`block_below` 아래로 떨어지면 거절 — 자기 왕복비용도 못 넘는 포지션은 작은 버전의 트레이드가
아니라 지는 트레이드다.

### 사이징과 한계

```
size = base × model_confidence × regime_factor × liquidity_factor × risk_factor
```

그 다음 노출 한계가 적용된다. 한계는 **모델이 올릴 수 없는 천장**이다:
`min(model_requested, policy_permitted)`, 혼합도 스케일도 아니다. 모델이 확신을 갖는다고
`max_position_per_stock` 이 커지지 않는다.

### 청산은 다르게 게이팅된다

`evaluate_exit()` 는 **라우팅 불가**로 만드는 게이트만 적용한다(unknown session, unknown
order state, duplicate risk, halt). staleness·모델 건강·노출 한계는 적용하지 않는다. 피드가
멈췄다고 포지션을 못 닫는 것이 이 비대칭이 막는 실패다.

---

## 10. Execution — 상태 기계, 재조정, pre-submit guard

### OrderStateMachine — `app.execution.order_state_machine`

기존 `LiveExecutionCoordinator`(멱등 제출, 브로커 호출, 정정/취소, JSONL 저널)를 대체하지 않는다.
그것이 갖지 않은 것 — **재시작을 견디는 상태** — 을 더한다.

`UNKNOWN` 이 핵심이다. 타임아웃된 제출은 살아 있지도 죽지도 않았고, 어느 쪽으로 추측하든
중복 주문이나 미관리 포지션이 된다. 실제 상태로 존재하고, 게이트의 `UNKNOWN_ORDER_STATE` 로
해당 종목의 신규 주문을 막고, 재조정이 브로커에 물어 해소한다.

`filled_quantity` 는 누적이고 감소할 수 없다. 부분 체결 평균가는 수량 가중이다 — 단순 덮어쓰기는
마지막 부분체결 가격을 포지션 원가로 보고하는데, 그게 바로 부분체결이 존재하는 경우의 오답이다.

### Reconciliation — `app.execution.reconciliation`

| 클래스 | 잡는 실패 |
| --- | --- |
| `OrderReconciler` | 해소되지 않은 주문 상태, 미반영 부분체결, **우리가 모르는 브로커 주문** |
| `PositionReconciler` | 미관리 포지션, 유령 로컬 포지션, 수량 불일치 |
| `AccountReconciler` | 오래되거나 어긋난 equity/cash — 모든 사이징의 분모 |

브로커 조회가 예외를 던지면 `reconciled=False` 이지 빈-따라서-깨끗한 결과가 아니다.
"브로커가 답하지 않았다"와 "브로커가 보유 없다고 했다"는 순진한 호출자에게 같은 바이트이고
정반대 사실이다.

브로커의 **라벨보다 수량이 우선**한다. FILLED 라면서 부분 수량을 보고하는 브로커도, OPEN
이라면서 이미 일부 체결된 주문도 흔하다.

### PreSubmitGuard — `app.execution.pre_submit_guard`

결정 시점과 소켓 쓰기 사이에는 큐·가격결정·rate limiter, 최악의 경우 재시작이 있다. 승인
조건이 그 사이에 소멸할 수 있으므로, 모든 실주문이 지나는 한 지점에서 다시 확인한다:
세션, 주문 상태, 데이터 신선도, 계좌 재조정.

의도적으로 **작은** 집합이다 — 지금 이 순간 독립적으로 확인 가능한 것만. 게이트의 사이징·
soft gate·노출 한계를 재계산하지 않는다.

증거가 없을 때의 동작은 `strict` 가 정한다. 라이브는 `strict=True`(증거 없으면 주문 없음),
페이퍼/섀도는 False(부재를 이유 코드로 기록하되 차단하지 않음).

---

## 11. Storage, API, 대시보드

### `data/store/trading_state.sqlite3` — `app.storage.trading_state_store`

서로 한 트랜잭션 안에서 일치해야 하는 테이블들: 결정의 컨텍스트, 근거가 된 regime·모델 예측,
게이트 판정, 생성된 order intent, 뒤따른 실행. 이들을 스토어에 나눠 담는 것이 크래시 후 결정을
재구성 불가능하게 만든다 — intent 는 커밋되고 게이트 행은 아니면, 사후에 게이트가 무엇을
봤는지 말할 방법이 없다.

WAL + `synchronous=FULL` + 다중 테이블 쓰기마다 실제 트랜잭션. 기존
`realtime_market_data.sqlite3`(고빈도 피드)와 `account_dashboard.sqlite3`(운영자용 이력)은
건드리지 않는다. 여기의 `account_snapshot` 은 다른 것 — 재조정에 쓰는 결정 시점 브로커 진실이다.

### API — `app.web_context_routes`

`/api/session/current`, `/api/context/global`, `/api/context/domestic`,
`/api/context/sector/{sector}`, `/api/context/stock/{ticker}`, `/api/regime/current`,
`/api/candidates`, `/api/decision/{ticker}`, `/api/gate/{ticker}`, `/api/model/health`,
`/api/data/health`, `/api/context/dashboard`, `/api/orders/open`.

WebSocket 채널: `/ws/context`, `/ws/candidates`, `/ws/orders`, `/ws/health`.

**구조적으로 읽기 전용이다.** 주문을 여는 라우트가 없다 — 대시보드가 주문할 수 있으면
게이트가 HTTP 로 도달 가능해지고, 그것이 리스크 계층 전체가 막으려는 단 하나다.
사이클이 아직 없거나 종목이 후보가 아니면 빈 객체가 아니라
`{"available": false, "reason": ...}` 를 답한다. 둘을 빈 바디로 합치면 대시보드가 0을 관측값처럼
보여주게 된다.

### 대시보드 readiness

`readiness` 는 게이트가 읽는 것과 **같은** health 객체에서 계산된다. 따라서 준비도 100% 와
핵심 모듈 OFFLINE 이 동시에 표시될 수 없다. `tests/test_context_api.py` 와 dry-run 이 이 불변식을
검사한다.

---

## 12. 검증

### Walk-forward ablation — `app.evaluation.context_ablation`

`BASE, TIME, DAY, SEASONALITY, GLOBAL, DOMESTIC, SECTOR, ONTOLOGY, GNN, FULL` 을 같은 행,
같은 purged walk-forward split 에서 채점한다. 두 arm 의 차이는 해당 레이어이지 다른 무엇도 아니다.

`assert_no_future_leakage()` 는 주석이 아니라 실제로 실패하는 검사다. 학습 행의 라벨 윈도가
첫 테스트 행 이후에 끝나지 않는지 확인한다 — 경계 10분 전 봉의 30분 라벨은 테스트 구간
**안에서** 해소되므로 단순 시계열 분할이 잡지 못하는 실패다.

ML 지표: precision, recall, F1, Brier, calibration. 거래 지표: return, win rate,
profit factor, expectancy, Sharpe, Sortino, MDD, turnover, slippage.
슬라이스: day, session, regime, sector.

### 프로덕션 dry run

```bash
python scripts/context_pipeline_dry_run.py
```

실제 캘린더, 실제 freshness registry, 실제 그래프·모델 런타임, 실제 `FinalTradeGate`, 실제
상태 기계, 실제 재조정기를 라이브 스토어 데이터로 돌린다. **유일한 대체물은 브로커** —
KIS 처럼 답하고 아무것도 보내지 않는 recording client 다. 주문 여부를 결정하는 모든 것이
프로덕션 코드이고, 끝의 소켓만 아니다. `tests/test_context_pipeline_dry_run_script.py` 가
스위트의 일부로 실행한다.

---

## 13. 알려진 외부 제약

* **GNN 체크포인트.** `data/models/temporal_hetero_gnn/latest.npz` 가 없으면 런타임은
  `OFFLINE` 이고 신규 진입이 차단된다(청산은 계속). 저장된 결정이
  `MINIMUM_TRAINING_EXAMPLES` 만큼 쌓인 뒤 학습 스크립트를 돌려야 한다. 설계된 동작이다.
* **국내 지수 피드 없음.** `DomesticContext.kospi_return` 은 추적 유니버스의 비가중 평균이지
  KOSPI 프린트가 아니다. 한계를 숨기지 않고 `symbol_count`·breadth 와 함께 노출한다.
* **글로벌 지표 커버리지.** 현재 `config/research_sources.live.json` 의 FRED 계열은 VIX,
  US10Y, DXY 뿐이다. SP500/NASDAQ/SOX/선물/아시아 지수는 수집기가 붙기 전까지 관측되지
  않으며, GlobalContext 는 그 부재를 `GLOBAL_GROUP_MISSING` 과 낮은 confidence 로 보고한다
  (없는 값을 만들지 않는다).
* **KR 캘린더 완전성.** `config/market_sessions.yaml` 의 KR 휴장일은 고정일자만 담고 있어
  음력 기반·임시공휴일이 빠져 있다. `SESSION_CALENDAR_SUSPECT` 로 표시되고, 누락된 휴장일은
  freshness 게이트가 실제로 잡는다.

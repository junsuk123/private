# 실행 권한: 선출 이전과 이후

## 한 줄 요약

**모든 투자 판단은 전략 선출 이전에 끝난다. 선출 이후에는 선택된 전략이 실시간 틱으로
진입·청산을 직접 결정하고, 주문 직전 ExecutionGuard 는 기술적 주문 가능성만 검사한다.**

---

## 1. 무엇이 문제였나

리팩토링 이전 실제 호출 그래프는 이랬다.

```
선출 → SharedLiveDecisionEngine.evaluate_buy
         ├─ ProfitabilityGate.evaluate      ← 거부 가능
         ├─ PositionSizer.size              ← 사이즈 축소
         └─ RiskManager.validate            ← 거부 가능
     → RealtimeTradingEngine
         ├─ entry_order_size_fraction       ← 사이즈 재축소
         └─ ExecutionQuality.assess         ← spread/impact 수익성 거부
     → LiveExecutionCoordinator → 브로커
```

선출은 이미 "이 종목, 이 전략"을 결정했는데, 그 뒤로 **다섯 개의 권한**이 같은 질문을
다시 판단했다. 그 결과 세 가지 실패가 구조적으로 가능했다.

1. **지연.** 결정과 주문 사이에 무거운 재계산이 있었다.
2. **중복 veto.** 선출 시점 가격으로 통과한 후보가, 몇 초 뒤 더 시끄러운 시장에서 같은
   계산에 걸려 거부됐다.
3. **손절 차단.** `ExecutionQuality` 의 spread/impact 판정이 SELL 에도 적용되어,
   **손절을 유발한 스프레드 확대가 그 손절을 막을 수 있었다.**

---

## 2. 지금의 경로

```
live data → feature/context → weekday+time/session regime
  → account+portfolio+market+symbol+tick+orderbook+macro+flow+news+cost+risk
  → ontology reasoning → GNN utility/election
  ────────────────────── TradePlan freeze ──────────────────────
  → strategy-owned fast tick executor → minimal ExecutionGuard → KIS limit order
```

`/api/execution/authority-path` 가 이 표를 런타임에서 그대로 내보낸다. 화면과 코드가
어긋날 수 없도록 stage 목록은 코드에서 생성되고, `tests/test_execution_authority_api.py`
가 선출 이후 stage 가 투자 판단을 주장하지 않는지 검사한다.

---

## 3. 선출 이전 — `app.trading.trade_plan_builder`

`TradePlanBuilder.build()` 가 **같은 계산을 더 일찍** 수행한다. 삭제가 아니라 이동이다.

| 단계 | 컴포넌트 | 산출 |
| --- | --- | --- |
| 1 | `ProfitabilityGate` | all-in 비용, net edge. 못 넘으면 `NO_TRADE` |
| 2 | `PositionSizer` | position weight (fractional Kelly × edge × liquidity × drawdown) |
| 3 | `RiskManager.validate` | exposure/concentration/eligibility → 승인 수량 |

여기에 deployment/selector authority cap 도 **한 번만** 곱해진다. 이전에는 엔진이
downstream 에서 한 번 더 곱해서, 선출된 사이즈와 제출된 사이즈가 서로 다른 두 숫자였다.

출력은 정확히 하나다: 실행 가능한 `TradePlan` 하나, 또는 `NoTradeDecision`.
`NO_TRADE` 도 plan 과 동일한 cost/risk snapshot 을 들고 다닌다 — 숫자를 볼 수 없는
거부는 나중에 반박할 수 없다.

---

## 4. TradePlan — `app.trading.trade_plan`

선출의 durable·immutable 산출물.

| 필드 | 내용 |
| --- | --- |
| `plan_id`, `created_at`, `expires_at` | 정체성과 유효기간 |
| `symbol`, `market`, `direction`, `strategy_id` | 무엇을 어느 방향으로 |
| `quantity`, `max_notional` | 선출된 사이즈 |
| `entry_rule` (trigger + 가격 밴드) | 언제 들어가는가 |
| `exit_rules` (TP / SL / trailing / time / signal) | 언제 나오는가 |
| `cancel_rule` | 언제 포기하는가 |
| `expected_net_edge_bps`, `cost_snapshot`, `risk_snapshot` | 근거 |
| `weekday_time_context` | 요일·세션 phase·seasonality 버킷 |
| `source_ids` | provenance |

**동결되는 것:** `strategy_id`, risk budget 근거, 사이징 방법론.
`immutable_signature()` 가 변조를 감지하고, `decision_fingerprint()` 는 plan_id·시각을
제외해 "같은 증거로 두 번 선출하면 같은 결정"을 검증 가능하게 만든다.

**허용되는 유일한 수량 변경**은 `with_broker_clip()` 이다. 감소만 가능하고, 브로커가
실제로 받아줄 현금/매도가능수량에 맞추는 기술적 clip 이며, 그 사실이 기록된다.
증가는 position sizing 결정이므로 거부된다.

**만료는 힌트가 아니라 속성이다.** `expires_at` 을 넘긴 plan 은 실행될 수 없고, 전략은
새 시장에 대해 다시 선출되어야 한다. 재시작 후 5분 된 호가로 만든 plan 이 제출되는 것을
막는 것이 이 필드의 존재 이유다.

포지션을 보유한 plan 은 시계로 만료되지 않는다 — 포지션은 여전히 exit rule 이 필요하고,
고아로 만드는 편이 더 나쁘다.

---

## 5. 선출 이후 — `app.trading.strategy_fast_executor`

per-tick critical section. 고정된 plan 과 몇 개의 float 만 들고, 틱마다 한 가지만 한다:
plan 의 진입/청산 조건이 발동했는가.

```
WAIT_ENTRY --entry trigger--> ENTERING --fill--> POSITION_OPEN
     |                            |                    |
     |                     reject/cancel          exit trigger
     +--expiry/cancel--> CLOSED <--fill-- EXITING <-----+
```

**하지 않는 것:** ontology 재구축, GNN 추론, portfolio risk 재계산, position size 재계산,
profitability 재평가. 이 금지는 권고가 아니라 **구조적**이다 — 이 모듈은
`app.ontology` / `app.graph` / `app.models` / `app.cost` / `app.risk` 로 가는 import
경로가 없고, `tests/test_post_selection_authority.py` 가 그것을 검사한다. DB 읽기도,
네트워크 호출도, 모델 추론도 `on_tick` 안에 없다.

### 청산 우선순위

**손절이 먼저 평가된다.** take-profit, trailing, time exit, 전략 신호보다 앞이다.
스톱 레벨이 뚫린 순간 그 포지션은 지고 있고 속도만이 도움이 되는데, 수익 청산을 먼저
검사하면 스톱을 뚫은 틱이 평범한 trailing 갱신으로 처리될 수 있다.

**현재 수익성이 음수라는 이유로 청산이 억제되는 경로는 없다.** 수익성 판단을 선출 이전으로
옮긴 이유가 이것이다: 포지션이 존재하는 시점에 "이 거래가 할 만한가"는 이미 답이 나와
있고, 남은 질문은 언제 나오느냐뿐이다.

### 틱 소스

`app.data.kis_realtime` 의 `tick_observer` 훅이 파싱된 틱을 **영속화 이전에** 넘긴다.
SQLite 에서 다시 읽으면 쓰기·읽기·둘 사이의 경합이 매 결정마다 들어간다.

---

## 6. ExecutionGuard — `app.execution.execution_guard`

주문 직전 유일한 검사. **투자 품질은 절대 판단하지 않는다.**

### 검사하는 것 (전부 기술적)

* plan 존재·만료·terminal 여부
* price > 0, quantity > 0
* 지원되는 symbol / exchange / product
* 현재 주문 가능한 세션인가
* 주문을 구성할 만큼 신선한 호가·호가창인가
* BUY: 브로커가 요구할 금액만큼 현금이 아직 있는가
* SELL: 아직 매도 가능한 수량인가
* SHORT: 물리적 대주가 아직 있는가
* 중복/idempotency/미체결 주문 보호
* kill switch, KIS auth/account/order 엔드포인트 health

### 검사하지 않는 것

`FORBIDDEN_INVESTMENT_CHECKS` 로 코드에 선언되어 있고 테스트가 강제한다:
expected return, strategy confidence, portfolio concentration, sector weight,
model uncertainty, volatility threshold, liquidity threshold, ontology support,
strategy ranking, profitability.

### 청산 비대칭

청산은 **라우팅 불가**로 만드는 검사만 받는다. stale 피드, 미재조정 계좌, 닫힌 신규진입
창은 risk-reducing exit 을 막지 않는다. 포지션을 가둘 수 있는 guard 는 가장 비싼 종류의
안전장치다.

현금 부족은 BUY 를 **clip** 하거나(부분 가능) **block** 한다(전혀 불가). clip 은
포지션 재사이징이 아니라 브로커가 지금 받아줄 양으로의 축소이며, 그 사실이 기록된다.

`app.execution.pre_submit_guard` 는 삭제되지 않았다 — 세션/주문상태/신선도/계좌재조정
네 검사를 계속 소유하고, ExecutionGuard 가 그것을 조합한다.

---

## 7. 제거된 post-selection veto

| # | 위치 | 무엇이었나 | 지금 |
| --- | --- | --- | --- |
| 1 | `evaluate_buy` | `ProfitabilityGate.evaluate` | `TradePlanBuilder` (선출 이전) |
| 2 | `evaluate_buy` | `PositionSizer.size` | `TradePlanBuilder` (선출 이전) |
| 3 | `evaluate_buy` | `RiskManager.validate` | `TradePlanBuilder` (선출 이전) |
| 4 | `evaluate_buy` | ontology 재승인 | 이미 `strategy_locked` 에서 제외됨 |
| 5 | `RealtimeTradingEngine` | `entry_order_size_fraction` 재clip | 선출 시 한 번만 적용 |
| 6 | `_prepare_order_for_execution` | `ExecutionQuality.assess` 수익성 판정 | plan-owned 주문에서는 **가격 산출 전용** |

6번은 삭제가 아니라 축소다. plan-owned 주문에도 호가창 신선도는 여전히 필요하다 —
limit price 를 구성해야 하니까. 없어진 것은 spread/depth/impact 의 *수익성* 판정이다.

`legacy path 는 그대로 살아 있다.` plan 이 없으면(빌드 실패, 구버전 경로) 기존 gate 가
모두 동작한다. 저하될 뿐 무방비가 되지 않는다.

---

## 8. 지연 측정 — `app.monitoring.execution_latency`

```
tick_event → tick_received → strategy_decision → execution_guard → broker_submitted
            (feed lag)      (decision latency)   (guard)          (broker)
```

`decision_latency_ms` 가 제약의 대상이다. plan 이 이미 고정한 레벨과 가격을 비교하는 것
외에 할 일이 없으므로 sub-millisecond 여야 한다.

`tests/test_fast_loop_latency.py` 가 p99 < 5ms 를 강제한다. 관대한 절대값인 이유는
마이크로초를 단속하려는 것이 아니라 **질적으로 무거워진 단계**(DB 읽기, 모델 호출)를
잡으려는 것이기 때문이다. 로드된 CI 에서도 흔들리지 않는다.

`/api/execution/latency` 에서 percentile 요약과 최근 span 을 볼 수 있다.

---

## 9. 설정 — `config/execution_authority.yaml`

두 종류의 항목이 있고 다르게 동작한다.

**`invariants`** 는 토글이 아니다. `post_selection_profitability_veto: false` 를 `true`
로 바꿔도 켤 post-selection ProfitabilityGate 호출이 코드에 없다. 그래서 로더가
**거부한다** — config 는 계약을 문서화할 수 있지만 조용히 반박할 수는 없다.
누군가 플래그를 뒤집고 아무 일도 일어나지 않아 파일이 존재하지 않는 시스템을 서술하게
되는 상황을 막는다.

**`operational`** 은 평범한 설정이다: `max_buy_orders_per_cycle`,
`trade_plan_ttl_seconds`, `strict_affordability`, `manual_strategy_confirmation`.

`order_type` 은 `LIMIT` 외의 값을 거부한다. 이 계좌는 시장가를 승인받지 않았고 그것을
구성하는 코드 경로도 없다.

---

## 10. 재시작

기존 launcher/config 로 그대로 실거래 경로가 쓰인다. 코드 변경도, 새 환경변수도 필요 없다.

* `StrategySessionManager` 가 선출 시 plan 을 만들고 `trade_plan` 테이블에 영속화한다.
* `RealtimeTradingEngine` 이 plan 이 있으면 자동으로 plan-driven 경로를 탄다.
* `LiveExecutionCoordinator` 가 `ExecutionGuard` 를 항상 들고 있다.
* 재시작 후 `TradePlanStore.open_plans()` 로 포지션을 소유한 plan 을 복구하고,
  `expire_stale()` 이 미체결 만료 plan 만 정리한다.

---

## 11. 관련 테스트

| 파일 | 검사하는 것 |
| --- | --- |
| `tests/test_trade_plan.py` | plan 불변식, 만료, broker clip, 방향별 exit level, 영속성 |
| `tests/test_post_selection_authority.py` | 선출 이전 결정 완료, 선출 이후 authority 0개, guard 금지 항목 |
| `tests/test_plan_driven_decision_path.py` | 실제 엔진에 spy 를 걸어 gate 미호출 확인 |
| `tests/test_fast_loop_latency.py` | tick→guard→broker 경로, 단계별 timestamp, p99 지연 |
| `tests/test_execution_guard.py` 계열 | 기술적 검사만 수행, 청산 비대칭 |
| `tests/test_execution_authority_api.py` | UI/diagnostics 가 실제 경로를 서술 |
| `tests/test_execution_authority_config.py` | config 가 아키텍처를 반박할 수 없음 |

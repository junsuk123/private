# Decision and Risk

어떤 데이터가 어떤 판단에 쓰이고, 무엇이 실제로 주문을 승인하는지에 대한 문서입니다.

![Data to decision map](diagrams/data_to_decision_flow.svg)

핵심 규칙 두 가지:

- **모든 BUY는 단 하나의 순수익 게이트(`ProfitabilityGate`)로 판단합니다.**
- **모든 청산은 단 하나의 동적 청산 정책(`DynamicExitPolicy`)으로 판단합니다.**

![Profitability architecture](diagrams/profitability_architecture.svg)

## 1. 판단 순서

`RealtimeTradingEngine.run_once()`의 한 사이클:

1. 최신 KIS 계좌 스냅샷을 읽습니다.
2. 보유 종목의 **SELL/REDUCE를 먼저** 평가합니다.
3. 이미 열린 SELL 주문이 있고 대체 가격이 사실상 동일하면 `open_sell_kept`로 기록하고 중복 제출하지 않습니다.
4. `REALTIME_BUY_ENABLED=true`일 때만 BUY 후보를 평가합니다.
5. quote 신선도, 1주 매수 현금, spread/유동성, 온톨로지/런타임 근거, 리스크 승인을 요구합니다.
6. 승인된 `FinalOrder` limit 주문만 `LiveExecutionCoordinator`로 제출합니다.

`SharedLiveDecisionEngine.consume_bundle`은 거시–미시 번들(이미 SELL/REDUCE 우선 정렬됨)을 받아 같은 `evaluate_exit_for_holding` / `evaluate_buy` 경로로 라우팅하는 어댑터입니다. 게이트 흐름은 바뀌지 않습니다.

## 2. 자문 레이어: 기술적 예측

`src/app/technical/`는 근거 기반 단기 예측 레이어입니다. **산출물은 전부 자문 근거이며 주문을 만들거나 게이트를 완화하지 않습니다.**

| 모듈 | 역할 |
| --- | --- |
| `technical/indicators.py` | SMA/EMA/MACD/RSI/Bollinger/ATR/volume-spike는 `features/indicator_engine.py`에 위임(단일 진실원). VWAP, Donchian, rolling z-score, spread-bps, orderbook imbalance 추가. 순수 함수, NaN-safe, pandas 없음 |
| `technical/regime.py` | 규칙 기반 `TechnicalRegimeClassifier` → 8개 regime + confidence + 사유 + feature 기여도. 리스크 게이트 우선 |
| `technical/signals.py` | 방법론 provider + `CompositeTechnicalSignalEngine` (regime gating, VWAP/거래량 확인 필수, 단일 지표 BUY 금지) |
| `technical/feature_builder.py` | OHLCV(+호가) → `TechnicalFeatureSet`, live feature frame 매핑 |
| `technical/labels.py` | no-look-ahead 지도 라벨 + `TradingCostEngine` 기반 net-after-cost, synthetic source 가드 |
| `technical/prediction.py` | `TechnicalPredictionEngine`: 보수적 expected exit price / net return / downside. 약하면 `NO_TRADE` |
| `technical/replay.py` | walk-forward, no-look-ahead 리플레이 평가 |
| `technical/policy.py` | `config/technical_prediction_policy.yaml` 로드 (env override, 로깅) |
| `graph/technical_evidence.py` | 신호를 `KnowledgeGraph` 트리플 + RDF evidence로 투영 |

### 방법론

각 provider는 signed score `[-1,1]`, confidence `[0,1]`, **보수적** 기대 edge(bps; 측정된 변동성 × horizon에 1 미만의 capture fraction — **조작된 alpha 하한 없음**), horizon, 지지/반대 feature, 사유 코드를 냅니다.

1. **모멘텀 / 추세추종** (Jegadeesh–Titman, Brock–Lakonishok–LeBaron MA rules). EMA gap, MACD 히스토그램, 단기 수익률, 지속성. `TREND_UP` / `BREAKOUT_CANDIDATE`에서 활성, 하락추세에서는 SELL 근거로 기여.
2. **돌파 / 박스 상단 돌파**. Donchian 고점 + 거래량 + VWAP + false-breakout risk. **거래량 확인 없으면 차단.**
3. **평균회귀** (단기 reversal). RSI / Bollinger %b 극단, VWAP 이탈 거리. `RANGE_BOUND` / `MEAN_REVERSION_CANDIDATE`에서 활성, **강한 하락추세에서는 차단**(`MEAN_REVERSION_BLOCKED_BY_DOWNTREND`).
4. **VWAP / 거래량 / 유동성** — **필수 확인 레이어**. 모멘텀/돌파 BUY는 VWAP 위 + 우호적 흐름을 요구하고, 아니면 `VWAP_BREAKDOWN`으로 BUY 근거가 사라집니다.
5. **변동성 밴드 / regime** — 리스크 지향. regime 분류기와 동적 청산 악화 신호에 기여하며, `HIGH_VOLATILITY_RISK` regime은 BUY를 차단합니다.

`HIGH_VOLATILITY_RISK` / `LOW_LIQUIDITY_RISK` / `NO_TRADE` regime → **BLOCK_BUY**. 그 외에는 regime이 선호 방법론을 고르고, 비적합 방법론은 조용히 버려지지 않고 가중치가 낮아집니다.

## 3. BUY 권한: ProfitabilityGate

- 코드: `src/app/cost/profitability_gate.py`
- 설정: `config/profitability_policy.yaml`
- 비용 수식: `src/app/cost/trading_cost_engine.py` (단일 비용 모델)

![BUY decision flow](diagrams/profitability_decision_flow.svg)

### 판정 규칙

```text
allow_buy = (
    expected_exit_price     >= break_even_exit_price × (1 + min_net_profit_buffer_rate)
    and net_expected_return >= required_min_net_return
    and cost_to_alpha_ratio <= max_cost_to_alpha_ratio
    and spread_rate         <= max_spread_rate
    and spread_alpha_ratio  <= max_spread_alpha_ratio
    and liquidity_score     >= min_liquidity_score
    and expected_slippage_rate <= max_slippage_rate
)
```

### 수식

```text
mid                   = (bid + ask) / 2
spread_rate           = (ask − bid) / mid
gross_expected_return = (expected_exit_price − entry_price) / entry_price
all_in_cost_rate      = buy_fee + sell_fee + sell_tax + entry_slippage + exit_slippage
                        + market_impact + spread_cost + safety_margin
net_expected_return   = gross_expected_return − all_in_cost_rate
cost_to_alpha_ratio   = all_in_cost_rate / max(|gross_expected_return|, eps)
spread_alpha_ratio    = spread_rate       / max(|gross_expected_return|, eps)

required_min_net_return = max(
    min_required_net_return[market],                     # 시장별 하한 (KR/US)
    min_net_profit_buffer_rate
      + volatility_buffer_k  × realized_volatility
      + liquidity_buffer_max × (1 − liquidity_score)
      + account_buffer(소액 계좌인 경우)
)
```

전략별 `target_net_return`은 요구치를 **더 조일 수만** 있고 시장 하한 아래로 내릴 수 없습니다.

### 기술 예측 연동

`SharedLiveDecisionEngine`은 자문 기술 레이어가 tradable BUY를 낼 때 **보수적 기술 expected exit price**를 게이트에 넣습니다. 게이트의 기대 청산가는 기술 net edge를 **선호**하지만 정직한 모델 추정치 위로 절대 부풀리지 않습니다(`min` 규칙). 양(+)의 기술 신호가 있어도 비용 차감 후 순수익이 동적 최소 edge 아래면 **거부**됩니다.

매 BUY 판단마다(성공/거부 모두) `technical_prediction`, `technical_methodology`, `technical_regime`, `profitability_decision`이 진단에 남습니다.

### 거부 사유 코드

| 코드 | 의미 |
| --- | --- |
| `MISSING_EXPECTED_EXIT_PRICE` | 예측 청산가 없음/무효 |
| `BELOW_BREAK_EVEN_WITH_MARGIN` | 손익분기 + 버퍼 미달 |
| `BELOW_TARGET_NET_RETURN_AFTER_COST` | 동적 최소 순수익 미달 |
| `COST_BURDEN_HIGH` | 비용이 alpha를 지배 |
| `SPREAD_TOO_WIDE` | 절대 spread 상한 초과 |
| `SPREAD_CONSUMES_ALPHA` | alpha 대비 spread 과다 |
| `LIQUIDITY_TOO_LOW` | 유동성 점수 하한 미달 |
| `SLIPPAGE_RISK_HIGH` | 기대 슬리피지 상한 초과 |
| `PROFITABILITY_GATE_REJECTED` | 엔진이 붙이는 상위 코드 |

### 설정 우선순위

기동 시 1회 로깅됩니다. 높은 것이 이깁니다.

1. 환경변수 — `REALTIME_MIN_BUY_NET_RETURN_KR/_US`, `REALTIME_MIN_NET_PROFIT_BUFFER_RATE`
2. `config/profitability_policy.yaml`
3. `profitability_gate.py` 내장 기본값

`config/profitability_policy.yaml`의 **현재** 값:

```yaml
min_required_net_return:  default 0.0   KR 0.0   US 0.0003
min_net_profit_buffer_rate: 0.0
max_spread_rate: 0.003          max_slippage_rate: 0.003
max_spread_alpha_ratio: 0.35    max_cost_to_alpha_ratio: 1.05
min_liquidity_score: 0.3
volatility_buffer_k: 0.05       liquidity_buffer_max: 0.001
account_buffer: small_account_equity_krw 200000, small_account_extra_net 0.0
```

`run.ps1`은 `REALTIME_MIN_BUY_NET_RETURN_KR/_US=0.0005`, `REALTIME_MIN_NET_PROFIT_BUFFER_RATE=0.0`으로 pin하므로 실제 런타임 하한은 KR/US 모두 5 bps입니다.

> **완화 이력 (2026-07).** 초기 하한(KR 0.008 / US 0.012)과 초기 버퍼는 ~20만원 계좌에서 BUY를 100% 차단했습니다. 이후 단계적으로 완화되어 현재 값에 도달했습니다: 시장 하한은 사실상 "비용 차감 후 양(+)"까지 내려갔고, `volatility_buffer_k`는 0.5 → 0.05, `liquidity_buffer_max`는 0.001, `max_cost_to_alpha_ratio`는 0.5 → 1.05, `small_account_extra_net`은 0.002 → 0.0이 되었습니다.
>
> **이것은 튜닝된 최적값이 아니라 거래가 발생하도록 낮춘 값입니다.** 특히 `max_cost_to_alpha_ratio: 1.05`는 비용이 기대 alpha와 거의 같은 거래도 통과시킨다는 뜻이라 실질 보호가 얇습니다. 실현 손익([validation.md](validation.md))으로 재조정해야 하며, net expectancy가 음수로 유지되면 되돌리는 것이 맞습니다.

### 적용 지점

- `risk/manager.py` — BUY 비용 검사는 `ProfitabilityGate.evaluate` 한 번의 호출. 실패 시 `approved=False`, `metadata.profitability_decision` 기록
- `trading/shared_decision_engine.py::evaluate_buy` — 모델 예측 edge(또는 보수적 fallback 추정)에서 **실제** 기대 청산가를 유도한 뒤 주문 구성 전에 게이트 호출
- `strategy/candidate_factory.py` — 자체 검사 대신 동일 게이트 사용

## 4. 청산 권한: DynamicExitPolicy

- 코드: `src/app/trading/dynamic_exit_policy.py`
- 설정: `config/dynamic_exit_policy.yaml`
- 소비: `SharedLiveDecisionEngine.evaluate_exit_for_holding`

![Dynamic exit policy](diagrams/profitability_dynamic_exit.svg)

감사에서 발견된 ~12개 분산 소스(RealtimeTradingConfig, adaptive_exit_policy, 인라인 `REALTIME_*` 읽기 ~15곳, ExecutionPolicy 기본값)를 하나의 객체로 통합했고, 유효 정책이 항상 감사 가능하도록 **1회 로깅**합니다.

### 동적 레벨

```text
take_profit_rate    = max(min_take_profit_rate,
                          all_in_cost_rate + min_net_profit_buffer
                          + k_vol_take_profit × realized_volatility
                          + liquidity_take_profit_buffer × (1 − liquidity_score)
                          + spread_take_profit_buffer_k × spread_rate)
trailing_giveback   = max(min_trailing_giveback, k_trail_volatility × realized_volatility)
soft_stop_rate      = max(min_soft_stop_rate,
                          k_downside_soft_stop × predicted_downside_risk + realized_volatility)
hard_stop_rate      = hard_stop_loss_rate        # 자본 서킷브레이커
emergency_stop_rate = emergency_stop_loss_rate   # 자본 서킷브레이커
```

익절·이익 잠금·트레일링·소프트 스톱은 비용/변동성/유동성/spread를 반영해 동적으로 움직이고, 하드/긴급 스톱은 설정 상수 근처에 고정됩니다.

`config/dynamic_exit_policy.yaml`의 현재 값과 `run.ps1` pin:

| 키 | YAML | run.ps1 pin | env |
| --- | ---: | ---: | --- |
| `min_take_profit_rate` | 0.008 | 0.014 | `REALTIME_QUICK_TAKE_PROFIT_NET` |
| `min_net_profit_buffer` | 0.004 | 0.008 | `REALTIME_MIN_NET_PROFIT_EXIT` |
| `profit_lock_arm_net` | 0.010 | 0.012 | `REALTIME_PROFIT_LOCK_ARM_NET` |
| `min_trailing_giveback` | 0.35 | 0.30 | `REALTIME_PROFIT_LOCK_GIVEBACK` |
| `stop_loss_net` | 0.0 (off) | 0.008 | `REALTIME_STOP_LOSS_NET` |
| `min_soft_stop_rate` | 0.006 | — | — |
| `hard_stop_loss_rate` | 0.08 | **0.02** | `REALTIME_HARD_STOP_LOSS` |
| `emergency_stop_loss_rate` | 0.05 | 0.05 | `REALTIME_EMERGENCY_STOP_LOSS` |
| `allow_loss_exit` | false | **true** | `REALTIME_ALLOW_LOSS_EXIT` |
| `block_sell_below_breakeven` | true | **false** | `REALTIME_BLOCK_SELL_BELOW_BREAKEVEN` |
| `noise_band_loss_rate` | 0.004 | — | — |
| `ontology_sell_dominance` | −0.55 | — | — |
| `strong_negative_forecast_bps` | 8.0 | — | — |

YAML 기본값은 하위 호환용(이전 인라인 기본값과 동일)이고, **실제 프로덕션 동작은 `run.ps1` pin이 결정합니다.** 특히 하드 스톱이 8% → 2%로 조여져 있고 손실 청산이 허용되어 있습니다. 이는 손실을 −3%까지 끌고 가던 이전 동작(1주 손실 차단이 패자를 붙잡던 문제)에 대한 대응입니다.

### 손실 청산 거버넌스

`loss_exit_decision(levels, evidence)` → `(allowed, reason)`

- **항상 허용** (`allow_loss_exit`와 무관한 자본 서킷브레이커): 하드 스톱 돌파, 긴급 스톱 돌파, net tight-stop 돌파
- **차단**: `allow_loss_exit`가 false이거나 손실이 노이즈 밴드 내(`|loss| <= noise_band_loss_rate`)
- **강한 악화 근거가 있으면 허용**: 온톨로지 SELL/REDUCE 우세, 강한 음(−)의 모델 전망(`predicted_net_return_bps <= −strong_negative_forecast_bps`), 급격한 유동성/spread 악화, 고위험 시장 regime, 일일 손실 예산 임박, 소프트 스톱 돌파

이것이 "물린 포지션" 문제에 대한 답입니다. 손실 포지션을 넓은 하드 스톱까지 무조건 들고 가지도, 무조건 던지지도 않습니다. 근거가 강하면 나가고, 노이즈면 버팁니다.

사유 코드: `hard_stop_loss`, `emergency_stop_loss`, `stop_loss_net`, `LOSS_EXIT_DISABLED`, `LOSS_WITHIN_NOISE_BAND`, `ontology_sell_dominance`, `strong_negative_forecast`, `liquidity_deterioration`, `market_regime_high_risk`, `daily_loss_budget_near_breach`, `soft_stop_loss`, `HOLD_INSUFFICIENT_DETERIORATION_EVIDENCE`.

해결된 레벨은 각 청산 판단의 `strategy_metadata.resolved_exit_policy`에 붙어 GUI와 감사 로그로 갑니다.

### 기술적 악화 근거

`evaluate_exit_for_holding`은 `CompositeTechnicalSignalEngine.evaluate_exit_deterioration`을 조회합니다. VWAP 붕괴, 모멘텀 상실(음의 MACD 히스토그램), 변동성 확대, 높은 false-breakout risk, 유동성 악화가 `TECHNICAL_EXIT_DETERIORATION`, `VWAP_BREAKDOWN`, `MOMENTUM_WEAKENED` 등의 코드로 나옵니다.

강한 악화는 유효 온톨로지 점수에 **상한 있는 페널티(≤ 0.5)** 를 적용해 **이익 중인** 포지션을 기존 `invalid_signal_exit` 분기로 더 빨리 보낼 수 있습니다. **손실 청산을 강제하지는 않습니다** — 하드/긴급 스톱과 `REALTIME_ALLOW_LOSS_EXIT` 게이트가 손익분기 아래 청산의 유일한 권한으로 남습니다.

## 5. 사이징과 리스크

| 컴포넌트 | 역할 |
| --- | --- |
| `risk/position_sizing.py` | edge/confidence/유동성/드로다운 기반 fractional-Kelly. 음의 기대값에는 사이즈를 주지 않음 |
| `risk/principal_protection.py` | 원금 보호선과 drawdown 예산 |
| `risk/manager.py` | 최종 검증. live 비활성, 행동/타입 규칙, 일일 손실, 거래 횟수, 유동성, 변동성, 중복 주문, 데이터 무결성, 제한 상품, 단일 종목, 인트라데이, 섹터, 현금, 예수금 검사 |
| `execution/execution_quality.py` | spread/슬리피지가 alpha를 먹는 매수 거부, no-chase 가드, 실현 슬리피지 저장 |

수량 제약:

```text
Q_final = max(0, min(Q_proposed, Q_risk, Q_cash, Q_liquidity,
                     Q_position_limit, Q_portfolio_limit))
```

사이징은 음수이거나 교정되지 않은 edge를 구제할 수 없습니다. 기대 순효용이 임계 아래면 수량은 0입니다.

`config/live_trading_safety.json`의 현재 한도:

```text
max_quote_age_ms 15000        max_orderbook_age_ms 15000
minimum_source_quality_score 0.8   minimum_model_confidence 0.52
minimum_probability_success 0.51   minimum_expected_net_return_bps 10
maximum_spread_bps 40         maximum_volatility_5m_bps 1000
maximum_single_order_pct_of_cash 0.8   maximum_position_pct_of_equity 0.8
maximum_orders_per_day 60     maximum_orders_per_symbol_per_day 8
order_cooldown_seconds 20     market_orders_allowed false
require_trained_model true    allow_heuristic_fallback_in_live false
require_principal_protection true
require_recent_readiness_report true (max age 1800s)
require_manual_arming false   arming_ttl_seconds 900
```

## 6. 실행

### 주문 가격 정책

`execution/order_pricing_policy.py`가 호가와 틱 규칙에서 limit 가격을 정합니다. `run.ps1` 기본값:

```text
EXEC_REQUIRE_ORDERBOOK_FOR_BUY=true         EXEC_REQUIRE_FRESH_ORDERBOOK_FOR_BUY=true
EXEC_MAX_ORDERBOOK_AGE_SEC=3.0              EXEC_UNKNOWN_SPREAD_PENALTY_RATE=0.006
EXEC_BUY_MAX_CHASE_BPS=20                   EXEC_ALLOW_NO_ORDERBOOK_EMERGENCY_SELL=true
EXEC_SELL_EMERGENCY_OFFSET_TICKS=1          EXEC_SELL_STOP_OFFSET_TICKS=1
EXEC_SELL_EMERGENCY_FALLBACK_OFFSET_RATE=0.003
```

BUY는 신선한 호가를 요구하지만, **긴급 SELL은 호가가 없어도 fallback offset으로 나갈 수 있습니다.** 진입 차단과 청산 능력을 분리한 것입니다.

### 거래소 라우팅

`execution/exchange_resolver.py`가 심볼/계좌로 거래소를 결정합니다. `KIS_US_EXCHANGE_STRICT=true`, `KIS_ALLOW_DEFAULT_US_EXCHANGE_IN_LIVE=false`이므로 live에서 미국 거래소를 추측으로 채우지 않습니다. KRX-only 신호 가격을 설명되지 않은 통합/SOR 체결 가격과 비교하지 않습니다.

### 제출 경계

`LiveExecutionCoordinator`가 유일한 KIS 제출 경계입니다. 필수 조건은 [live_trading.md](live_trading.md)에 정리되어 있습니다.

### 전략 소유 실행 경로 (shadow)

새 경로의 인과 순서:

```text
StrategyInstance → OrderIntent 영속화 → RiskGate → RiskVerdict 영속화
→ ExecutionEngine → KIS gateway → BrokerOrder/OrderUpdate/Fill 영속화
→ 계좌 재동기화 → origin StrategyInstance가 소유하는 Position
```

- `CausalOrderJournal`이 중요한 레코드를 fsync합니다.
- idempotency store는 broker 제출 **전에** 키를 예약하고 fsync합니다. 동일 intent로 키를 재사용하면 no-op, 다른 payload면 fail-closed.
- intent가 없으면 risk verdict를 저장할 수 없습니다.
- `LifecycleStore` 마이그레이션 v1/v2가 전략 인스턴스·포지션·TradePlan을 durable하게 저장합니다. 재시작 테스트가 소유자를 복원하고 원래 plan의 청산 로직을 실행합니다.

이 경로는 아직 live 제출에 연결되어 있지 않습니다. 연결 전에 어댑터가 증명해야 하는 것: 승인 수량이 요청 수량과 각 컴포넌트 한도를 넘지 않을 것, 거부된 intent의 승인 수량이 0일 것, 하나의 idempotency 키가 최대 하나의 broker 주문만 만들 것, 정정/취소가 부모 intent와 소유자를 유지할 것, 애매한 timeout이 재시도 전에 broker 주문 상태를 조회할 것, 체결이 broker 확인 수량을 넘지 못할 것, 재동기화된 포지션이 `origin_strategy_id`/`strategy_instance_id`를 유지할 것.

### 복구 순서

1. 신규 진입 비활성화
2. 인과 lifecycle 저널 로드, 레코드 해시/스키마 버전 검증
3. KIS 주문·체결·잔고·현금·포지션 조회
4. idempotency/correlation과 broker 주문 상태로 애매한 제출 해소
5. 전략 소유권과 포지션 상태 재구성
6. 소유자가 없거나 중복 소유된 포지션은 fail-closed, 설정된 긴급 관리만 허용
7. 활성 포지션의 청산/관리 재개
8. 차단 불일치가 없을 때만 신규 라우팅 활성화

legacy 저널과 idempotency 파일은 롤백/비교를 위해 손대지 않습니다.

## 7. 무엇이 무엇을 결정하는가

| 데이터 | 영향을 주는 판단 |
| --- | --- |
| 호가·spread | BUY 게이트(`SPREAD_TOO_WIDE`, `SPREAD_CONSUMES_ALPHA`), limit 가격과 no-chase 상한, 실행 품질 거부 |
| 체결 신선도·세션 | NoTrade와 `MODEL_FEATURE_UNAVAILABLE`, 온톨로지 사실 유효 구간(stale = hard block), 장 마감 시 REST 스냅샷 fallback |
| 현금·보유·실현손익 | 포지션 사이징과 `INSUFFICIENT_CASH_FOR_ONE_SHARE`, 일일 손실 예산과 BUY 정지, 집중도/드로다운 축소 청산 |
| 변동성·유동성 | BUY 최소 순수익 요구치, 익절/트레일링/소프트스톱 레벨, 고변동성 regime의 macro `BLOCK_BUY` |
| 뉴스·공시 이벤트 | macro risk level과 시장 전체 BUY 차단, event-momentum 적격성과 TTL 만료, reasoning path의 semantic 리스크 근거 |
| 모델·온톨로지 출력 | 기대 청산가(정직한 추정치 위로 부풀리지 않음), 전략 적격 마스크와 compatibility score, 이익 포지션에 대한 상한 있는 악화 페널티 |

## 8. Before / after

![Before/after net-profitability gate](diagrams/profitability_before_after.svg)

리팩터 전에는 방향성 신호 강도만으로 BUY가 나갈 수 있었고, 비용 검사가 여러 곳에 흩어져 gross는 양수인데 net은 음수인 거래가 반복됐습니다. 리팩터 후에는 하나의 순수익 게이트와 하나의 청산 정책이 그 판단을 독점합니다. 리플레이 지표는 [validation.md](validation.md)에 있습니다.

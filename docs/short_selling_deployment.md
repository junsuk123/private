# 숏 전략 배포 사다리 (Short Strategy Deployment Ladder)

숏 전략을 **추가하는 것**과 숏 전략으로 **실거래하는 것**을 완전히 분리한 문서입니다. 코드에
숏 전략 3개가 존재하지만, 이 문서를 쓰는 시점에 실주문 권한을 가진 숏 arm은 **0개**입니다.
그것이 버그가 아니라 설계입니다.

## 1. 한 문장 요약

> 숏 전략은 코드에 추가된 직후 반드시 `SHADOW` 상태이며, 실시간 forward 데이터로
> 비용·대주·체결을 모두 반영한 보수적 순엣지 하단값을 증명한 뒤에만, 운영자 개입 없이
> 한 칸씩 자동 승격된다. `SHADOW → LIVE_FULL` 직행 경로는 **어떤 환경변수·설정·수동
> 조작으로도 존재하지 않는다.**

## 2. 왜 숏은 롱의 부호 반전이 아닌가

이 시스템에서 숏을 "롱에 -1을 곱한 것"으로 다루지 않는 이유가 설계 전체를 결정합니다.

| 항목 | 롱 | 숏 |
| --- | --- | --- |
| 최대 손실 | -100% (유계) | **무제한**, 그리고 역방향으로 갈수록 포지션이 커져 손실이 **가속** |
| 진입 체결 | ask를 매수 | bid를 매도 |
| 청산 체결 | bid를 매도 | **ask를 매수** (압박 상황에서의 매수 = 이 시스템 최악의 체결) |
| 왕복 비용 | 수수료 + 매도세 | 동일 + **대주 이용료 (보유시간 비례 누적)** |
| 청산 시점 결정권 | 전략 | 전략 **또는 대주 상환 요구(recall) 시 대주자** |
| 실행 전제 | 현금 | **대주 가능 수량 확보(locate)** — 없으면 거래 자체가 불가 |

따라서 다음 두 가지는 명시적으로 금지됩니다.

- **롱 결과의 부호 반전으로 숏 라벨을 만들지 않는다.** 비용 구조가 비대칭이므로 부호 반전은
  대주 비용과 매수 상환 슬리피지를 0으로 가정한 것과 같습니다.
- **저장된 과거 봉으로 백테스트하지 않는다.** 과거 시점의 대주 가능 여부 데이터가 없기 때문에,
  백테스트는 "당시 대주 불가였던 종목"을 숏 친 것으로 계산합니다. 그런데 대주 불가 종목은
  **정확히 가장 많이 하락하는 종목군**이므로, 이 편향은 크고 항상 유리한 방향입니다.

### 2.1 청산 목표는 좁아지지 않고 넓어진다

초기 구현에서 "숏은 위험하니 목표를 좁게" 잡았는데 이는 산술적으로 틀렸습니다. 비용은 익절과
손절 **양쪽 barrier에 더해지므로**(`net_reward_risk_ratio`), 비용이 오르는데 목표를 줄이면
순 reward:risk가 양쪽에서 압축됩니다. 60bps 손절에 130bps 목표는 net R:R 1.16으로, 테이블
전체가 기준으로 삼는 1.5 아래입니다 — 위험을 더 지고 **더 나쁜 보상**을 받는 구조입니다.

비용 하단이 올라갔을 때의 올바른 대응은 **목표를 키우는 것**입니다.

```
SHORT_REFERENCE_ROUND_TRIP_COST_BPS = 28 (KRX 왕복) + 8 (장중 대주) = 36bps
목표 = 36 + 1.5 x (60 + 36) ≈ 180bps
```

숏에서 실제로 좁아지는 것은 **시간**입니다. 대주 이용료는 보유하는 동안 계속 누적되고 recall은
예고가 없으므로, 보유시간은 공짜 옵션이 아니라 비용이자 위험입니다. 모든 숏의 horizon은 롱
counterpart보다 짧고(`test_no_short_thesis_outlives_its_long_counterpart`), 오버나이트는
`LIVE_FULL`에서도 금지입니다.

## 3. 추가된 숏 전략 3개 (13개 전량 복제 아님)

기존 13개 롱 전략을 대칭 복제하지 않았습니다. 하락 레짐에서 **구조적으로 필요한** 3개만,
각각 독립 가설로 추가했습니다.

| strategy_id | 논지 | 롱 counterpart | 비고 |
| --- | --- | --- | --- |
| `market_intraday_momentum_short` | 장 초반 30분의 **음의** 수익률이 장 후반 30분까지 이어짐 | `market_intraday_momentum` | Gao/Han/Li/Zhou (JFE 2018)는 **양방향** 결과. 롱 전략 docstring이 "long-only라 양의 leg만 거래 가능"이라 적어둔 나머지 절반 |
| `opening_range_breakdown` | 상대거래량 높은 종목이 시초 범위 **하단** 이탈 후 하락 지속 | `opening_range_breakout` | 상대거래량 게이트는 논지의 일부(무제한 이탈은 수익성 없음) |
| `residual_relative_weakness` | 시장·업종 베타 제거 후에도 남는 **음의** 잔차 | `residual_relative_strength` | **유일하게 지수 하락이 필요 없는** 숏. 상승장에서도 유효 |

보류한 것: 단순 VWAP 상단 숏, 유동성 급등 직후 숏, 뉴스 직후 즉시 숏, RVGI 단순 반전,
저유동성 소형주 숏.

`residual_relative_weakness`는 롱 전략의 `residual_return_*_bps`를 부호 반전해 쓰지 않고
**별도 필드** `residual_short_bps` / `residual_long_bps`를 소비합니다. 롱 측정값을 재사용하면
이 논지의 라벨이 다른 논지 라벨의 결정론적 함수가 되어 두 arm이 절대 서로 다른 결론을 낼 수
없게 되는데, 그러면 별도 arm으로 운영하는 이유 자체가 사라집니다.

## 4. 상태 머신

```
DISABLED ──► SHADOW ──► LIVE_PROBE ──► LIVE_LIMITED ──► LIVE_FULL
              ▲   ▲          │               │              │
              │   └──────────┘◄──────────────┘◄─────────────┘   (강등)
              │
       SUSPENDED ◄── (ANY: 즉시 중단 조건)
```

| 상태 | 실주문 | 의미 |
| --- | --- | --- |
| `DISABLED` | ✗ | 생성·평가 안 함 |
| `SHADOW` | ✗ | 실시간 신호와 가상 주문을 생성·기록하지만 **브로커 주문 없음** |
| `LIVE_PROBE` | ✓ | 최소 수량(1주), 1일 1회, 30분 이내, 당일 청산 |
| `LIVE_LIMITED` | ✓ | 정상 수량의 일부, 종목/횟수/보유시간 제한 |
| `LIVE_FULL` | ✓ | 정책 허용 범위 내 정상 거래 (오버나이트는 여전히 금지) |
| `SUSPENDED` | ✗ | 성능 저하·대주 오류·상태 불일치. 사다리의 칸이 아니라 **고장 상태** |

이 arm별 숏 배포 사다리와 `StrategySelectorV2`의 전역 선택 권한 사다리는 독립입니다. V2가
`LIVE_PROBE` 또는 `LIVE`로 자동 승격돼도, 숏 arm 자체가 이 표의 live 상태가 아니거나 fresh locate가
없으면 세션 계층에 실주문 승인 proposal이 존재하지 않습니다. 따라서 V2 승격으로 숏 `SHADOW`나
대주 fail-closed 규칙을 우회할 수 없습니다.

### 4.1 금지 전이는 화이트리스트로 불가능하게 만든다

`ALLOWED_TRANSITIONS`는 랭크 비교가 아니라 **명시적 화이트리스트**입니다. 랭크 비교
(`target.rank > current.rank`)를 썼다면 `SHADOW → LIVE_FULL`이 "3칸 상승"으로 조용히 허용됩니다 —
이 서브시스템이 존재하는 이유가 바로 그 전이를 막는 것입니다.

불가능한 전이(테스트로 고정, `test_forbidden_transitions_are_unreachable`):

```
DISABLED → LIVE_PROBE / LIVE_LIMITED / LIVE_FULL
SHADOW   → LIVE_LIMITED / LIVE_FULL
LIVE_PROBE → LIVE_FULL
SUSPENDED  → 모든 live 상태
```

방어는 3중입니다.

1. `config/short_strategy_deployment.yaml`의 `initial_state`는 **clamp**됩니다. `LIVE_FULL`을
   써도 경고 로그를 남기고 `SHADOW`가 됩니다.
2. `ShortStrategyPromotionController.decide`는 한 사이클에 **한 칸만** 이동합니다.
3. `DeploymentStateStore.apply` / `force_state`가 **저장 경계에서 다시** 화이트리스트를
   검사합니다. 미래에 누군가 손으로 `PromotionDecision`을 만들어도 기록되지 않습니다.

수동 override(`force_state`)도 화이트리스트에 묶입니다. 운영자가 할 수 있는 최대치는
**합법적인 한 칸**이며, 항상 actor가 붙은 audit event가 남습니다.

## 5. 승격보다 강등이 빠르다

이 비대칭이 안전성 논증의 핵심입니다.

| | 조건 |
| --- | --- |
| 승격 | **연속** 5~10 사이클 전체 hard gate 통과 |
| 강등 | 첫 실패 사이클에 즉시 (일부는 2회 연속) |
| 즉시 중단 | 유예 없음, 해당 사이클에 `SUSPENDED` |

이유: 승격이 느려서 생기는 비용은 **놓친 기회 한 번**이고, 강등이 느려서 생기는 비용은
**무제한 하방에서 복리로 커지는 손실**입니다.

실패 사이클은 연속 카운터를 **0으로 리셋**하며 감소시키지 않습니다. "5회 통과, 1회 실패, 1회
통과"는 연속 5회가 아닙니다. 감소 방식이면 60% 확률로 통과하는 arm이 결국 누적으로 live에
도달합니다.

강등 임계값은 같은 칸의 승격 임계값보다 **항상 느슨합니다**(hysteresis). 없으면 경계에 앉은
arm이 매 사이클 오르내리고, 그 진동 하나하나가 실제 포지션 한도 변경입니다.

### 5.1 즉시 중단 조건 (유예 없음)

모두 "내부 상태와 브로커 상태가 불일치" 또는 "이 논지가 검증된 시장 조건이 아님"을 뜻합니다.
여러 사이클에 걸쳐 평균 낼 대상이 아닙니다 — 우리가 보유하고 있다고 믿는 포지션이 실제
포지션과 다르면, 한 사이클 더 거래하는 것은 상황을 악화시킬 뿐입니다.

- broker position direction ↔ 내부 direction 불일치
- `loan_date`(대출일) 누락 → 매수상환 계약 생성 불가
- 대주 상환 주문 계약 생성 실패 / 대주 가능 수량 초과 주문 시도
- 실시간 데이터 신선도 hard fail, `HIGH_VOL_DISLOCATED`
- `change_point_probability >= 0.7`
- 일일 손실 한도 초과, 손절 주문 제출 실패
- 브로커 잔고 복원 실패, 비정상 가격 / 중복 신규 숏 주문

복구는 `SUSPENDED → SHADOW`만 가능하고, live로 직행하는 경로는 없습니다.

## 6. Confidence score와 hard gate의 관계

`confidence_score`는 7개 구성요소의 **가중 혼합**입니다.

```
0.25 edge_quality + 0.20 sample_reliability + 0.15 calibration_quality
+ 0.15 execution_quality + 0.10 borrow_reliability + 0.10 regime_coverage
+ 0.05 stability_quality
```

**hard gate가 confidence score보다 우선합니다.** 혼합이므로 한 구성요소가 파멸적이어도 총점은
높을 수 있습니다 — 나머지가 훌륭하면 대주 가능률 0.2로도 0.85가 나옵니다. 혼합은 **순위
매기기**에 맞는 형태이고 **권한 부여**에는 틀린 형태이므로, 모든 hard gate는 독립적으로
평가되고 하나라도 실패하면 점수와 무관하게 승격이 차단됩니다.

측정되지 않은 구성요소는 **0.0**이며 중립 0.5가 아닙니다. 보정을 한 번도 측정하지 않았다면 그
arm은 보정을 입증하지 않은 것이고, 0.5는 "증거의 부재"에 "증거의 절반" 무게를 주는 셈입니다.

측정 불가한 profit factor(손실 거래가 아직 없음)는 **통과가 아니라 실패**입니다. 3거래 전승
표본은 전형적인 소표본 우연이고, 무한 profit factor로 취급하면 바로 그것으로 승격됩니다.

## 7. Forward shadow 검증과 3중 누수 방어

`ShadowTradePlan`은 신호가 발생한 **그 순간** 기록되며, 진입 기준가·barrier·대주 관측을
plan 안에 동결합니다. 채점은 그 이후에 도착한 데이터만으로 합니다.

1. **시간 누수 방어** — `ShadowFillSimulator.observe`는 `signal_at` **이하**의 모든 quote를
   거부합니다. 같은 timestamp는 "매우 신선한 것"이 아니라 같은 순간이므로 거부합니다.
   따라서 barrier walk는 신호를 만든 봉을 볼 수 없습니다.
2. **대주 누수 방어** — 실행 가능성은 plan에 **값으로 embed된** 대주 snapshot으로만 판정합니다.
   채점 시점에 새 locate를 조회하는 경로가 물리적으로 없습니다. recall wave는 이 전략들이 숏
   치려는 하락과 상관되어 있으므로, 이 누수는 shadow 성과를 live에서 재현 불가능하게 만듭니다.
3. **가격 누수 방어** — 진입은 bid, 청산은 ask. **mid 체결은 금지**입니다. mid는 양쪽 leg에서
   스프레드 절반씩을 조용히 선물하는데, 20bps KRX 스프레드에서 왕복당 20bps의 순수한 허구이고
   목표가 ~180bps인 전략에서는 결정적입니다.

한 quote가 익절과 손절 barrier를 **동시에** 지나면 **손절**로 처리합니다. 하나의 관측 안에서는
어느 쪽이 먼저였는지 알 수 없고, 익절을 가정하면 변동성 큰 거래를 체계적으로 미화하는데 이
전략들이 사는 곳이 바로 그 모집단입니다.

`return_deadline`이 **없는 것**은 정상(개방형 대주)이며 즉시 상환으로 취급하지 않습니다. "마감이
없음"을 "지금 마감"으로 읽으면 모든 숏이 즉시 청산됩니다.

### 7.1 실행 불가 신호는 기록되지만 채점되지 않는다

대주가 없어 실행 불가였던 신호는 `SIGNAL_VALID_BUT_UNEXECUTABLE`로 남습니다.

- **포함**: 신호 품질 분석, `borrow_availability_rate` 분모
- **제외**: 모든 승격 통계 — **취할 수 없었던 거래로 전략을 승격할 수 없습니다**

## 8. short_rescue_rate — 숏이 실제로 무엇을 사왔는가

승격 게이트 중 유일하게 "이 전략이 좋은가"가 아니라 **"이 전략을 추가해서 얻은 것이 있는가"**를
묻습니다.

```
short_rescue_rate = (best_long_edge <= 0 이고 best_short_edge > 0인 평가 스냅숏 비율)
```

최소 3%를 요구합니다. 이것이 없으면 어떤 arm이 다른 모든 게이트를 통과하면서도 **항상 더 좋은
롱과 함께만** 발동할 수 있는데, 그렇다면 추가로 얻은 것은 없고 노출만 늘어난 것입니다.

`BanditSelection`은 숏이 선택 가능했는지와 **무관하게** 매 사이클 LONG/SHORT/NO_TRADE 비교를
기록합니다. `SHADOW` arm도 순위가 매겨지고 `shadow_arms`에 보고되므로, "숏이 있었으면
도움이 됐을까?"를 사후에 답할 수 있습니다 — 그것이 사다리를 올라갈 증거입니다.

`BOTH_DIRECTIONS_NEGATIVE`는 "양방향을 봤고 둘 다 수익성이 없다"는 **발견**이고,
방향 하나만 있었던 **커버리지 공백**과 구분됩니다.

## 9. 대주(borrow) 취급

### 9.1 단위 — 연율 bps

모든 대주 이용료는 **연율 basis point**입니다(브로커가 그렇게 고시). 필드 이름에 단위가
붙어 있습니다: `borrow_fee_bps_annualised`.

단위 혼동은 반올림 오차가 아니라 양방향 약 10,000배 오차입니다.

- 연율 800bps를 거래당 40bps 상한과 비교 → 모든 종목 거부. **즉시 보임**(아무것도 거래 안 됨)
- 거래당 값을 연율로 안분 → 8%/yr 대주가 0.0009bps, 즉 **무료**. **보이지 않음**, 그리고
  음의 기대값 숏이 비용 게이트를 통과함

두 번째가 위험한 쪽이므로 이름에 단위를 박았습니다. 안분은 `borrow_cost_bps()` 한 곳에서만,
**365일 기준**(장이 닫힌 날도 누적 — 금요일 오후 숏은 주말 값을 냅니다)으로 합니다.

### 9.2 fail-closed 규칙

| 상황 | 처리 |
| --- | --- |
| 관측 없음 | 거래 불가. `available=False`가 아니라 **snapshot 부재**로 표현 |
| 이용료 미상 | 거래 불가. **0이 아님** |
| 수량 미상 | 거래 불가. "가능: 예, 수량: 미상"은 사이징할 수 있는 locate가 아님 |
| snapshot 30초 초과 | 거래 불가. 오래된 locate는 locate가 아님 |
| 신호 **이후** timestamp | **거부**, 0으로 clamp하지 않음. 결정 대비 미래 정보 = 누수 |
| 조회 실패(예외) | `available=False`로 바꾸지 **않음**. "브로커가 없다고 함"과 "물어보지 못함"은 운영 대응이 다름 |

추정·보간·유사 종목 전이는 없습니다. "아마 대주 가능"이라는 상태는 존재하지 않습니다.

`BorrowSnapshotStore`는 캐시가 아니라 **append-only 저널**입니다. 과거 시점 locate 조회와
승격 구간 `borrow_availability_rate`가 모두 이력에 대한 질의이므로, 덮어쓰는 캐시로는 둘 다
지원할 수 없습니다.

## 10. 주문 계약 — SELL의 모호성 제거

`SELL`은 두 가지를 뜻할 수 있고, 두 결과는 계좌가 flat이 되는지 **주식을 빚지는지**로
갈립니다.

| 의미 | direction | effect | product | broker side |
| --- | --- | --- | --- | --- |
| 롱 진입 | LONG | OPEN | CASH | BUY |
| 롱 청산 | LONG | CLOSE | CASH | SELL |
| **숏 진입** | SHORT | OPEN | CREDIT_BORROW | **SELL** |
| **숏 청산** | SHORT | CLOSE | CREDIT_BORROW | **BUY** |

수량은 **항상 양의 절댓값**이고 방향은 별도 필드입니다. 방향을 수량 부호로 인코딩하면 리팩터
한 번을 살아남은 뒤 어딘가에서 조용히 매수로 바뀝니다.

`position_effect`의 기본값은 `"OPEN"`이 **아니라** 빈 문자열(추론)입니다. 이것은 구현 중 실제로
발생한 회귀입니다: `"OPEN"` 기본값은 shorts 이전에 만들어진 **모든 롱 SELL/REDUCE 청산 주문을
진입으로 재분류**했고, 그 주문들은 자기 broker side와 모순되므로 계약 일치 검사가 시스템의
**모든 위험 축소 주문을 거부**했습니다. 추론은 방향 인식적입니다 — BUY는 롱을 열지만 숏을
닫습니다.

`_require_credit_contract`는 direction·effect·product·broker side **4개 전부**를 검사합니다.
가장 위험한 경우는 `SHORT/CLOSE` 계약이 `side=SELL`을 들고 오는 것으로, 대주를 상환하는 대신
보유하지 않은 주식을 매도합니다.

`loan_date`(대출일)는 숏 청산에 **필수이며 fallback이 없습니다**. 없이 상환을 보내면 실패하거나 —
브로커 기본값에 따라 — **다른 lot을 상환**해서 한 lot은 두 배가 되고 다른 lot은 열린 채
남습니다. 둘 다 주문을 보내지 않는 것보다 나쁩니다. 대출일이 다른 대주는 브로커에게 별개
포지션이므로 별개 lot으로 관리합니다.

## 11. 실행 게이트 순서 (entry blockade)

대시보드는 **가장 먼저 막은 단계**를 보고합니다. 마지막에 나온 불만이 아닙니다 — 배포 권한
차단은 "아직 학습 중, 고칠 것 없음"이고, 승인된 arm의 대주 preflight 차단은 운영 문제인데,
마지막 이유를 보고하면 이 둘이 뒤섞입니다.

```
directional_candidates → short_signal → shadow_validation → deployment_authorization
→ borrow_preflight → profitability → short_risk → credit_order_contract → broker_execution
```

`evaluate_borrow`는 shadow simulator·live preflight·RiskManager **세 곳이 공유**합니다. 세
구현이 갈라지면 shadow 실행 가능성이 live 실행 가능성을 예측하지 못하고, 사다리 전체가 허구를
측정하게 됩니다.

## 12. 레짐별 방향 허용

`HIGH_VOL_DISLOCATED`는 **양방향 모두** 신규 진입을 막고 청산만 허용합니다. 붕괴된 호가창은 숏
기회가 아니라 **가격이 정보가 아닌 시장**이고, "테이프를 읽을 수 없다"에 대한 올바른 대응은
"그러니 반대로 걸겠다"가 아닙니다.

`TREND_DOWN`은 롱(`vwap_mean_reversion`)을 여전히 허용합니다. 지수 하락이 구조적 숏 전용의
근거는 아니며 — 평균회귀는 정확히 하락 추세에서 살아남는 롱 논지입니다 — 제거하면 방향 필터가
방향 베팅으로 변합니다.

알 수 없는 레짐 이름은 **빈 집합**(fail closed)입니다. 레짐 분류기의 오타가 조용히 양방향 모든
arm을 열면 안 됩니다.

## 13. 운영 — 확인해야 할 것

```powershell
# 모든 숏 arm의 현재 상태와 실주문 권한
curl http://127.0.0.1:8010/api/short-strategies/status

# 특정 전략이 승격까지 남긴 조건 (전체 목록)
curl http://127.0.0.1:8010/api/short-strategies/opening_range_breakdown/validation

# 자동 승격·강등 이력 (변경 당시 지표 포함)
curl http://127.0.0.1:8010/api/short-strategies/opening_range_breakdown/deployment-history

# 대주 데스크 건강도
curl http://127.0.0.1:8010/api/borrow/health

# LONG vs SHORT vs NO_TRADE 비교
curl http://127.0.0.1:8010/api/directional-bandit/evaluations

# 수동 중단 (유일한 변경 endpoint)
curl -X POST "http://127.0.0.1:8010/api/short-strategies/opening_range_breakdown/suspend" `
     -H "Content-Type: application/json" -d '{"actor":"operator","reason":"..."}'
```

**승격 endpoint는 없습니다.** 운영자는 언제나 시스템을 더 안전하게 만들 수 있지만, 덜 안전하게
만드는 것은 증거 사다리를 통과해야 합니다. 그것을 우회하는 HTTP 호출이 있으면 메커니즘 전체가
무의미해집니다.

### 13.1 계좌 레벨 마스터 스위치

`RiskRules.short_selling_allowed` / `credit_loan_allowed`는 **기본 False**입니다. 이 둘이 꺼져
있으면 배포 상태와 무관하게 모든 숏 진입이 거부됩니다. 즉 실거래에는 **두 개의 독립 조건**이
필요합니다.

1. 계좌 레벨 정책 허용 (운영자 결정)
2. 해당 arm의 배포 상태가 live 칸 (증거 기반 자동 결정)

숏 **청산**은 계좌 레벨 스위치로 막히지 **않습니다**. 플래그가 꺼졌다고 상환을 거부하면 무제한
손실 포지션이 갇힙니다.

## 14. 현재 승격 상태

| arm | 상태 | 실주문 | 막고 있는 것 |
| --- | --- | --- | --- |
| `market_intraday_momentum_short:SHORT:KR:CREDIT_BORROW` | `SHADOW` | ✗ | 표본 0, forward 데이터 미수집 |
| `opening_range_breakdown:SHORT:KR:CREDIT_BORROW` | `SHADOW` | ✗ | 동일 |
| `residual_relative_weakness:SHORT:KR:CREDIT_BORROW` | `SHADOW` | ✗ | 동일 |

`SHADOW → LIVE_PROBE`에 필요한 것: 실행 가능 신호 120건, 체결 60건, 거래일 20일, 종목 10개,
confidence 0.72, 보수적 순엣지 8bps, cost coverage 1.7배, holdout 3구간, 연속 5사이클.
**최소 20 거래일**이 걸리므로, 이 표가 바뀌기까지 한 달 이상입니다.

## 15. 배선 상태와 남은 작업

### 15.1 배선 완료 (사다리가 진행 가능한 상태)

| 구성요소 | 모듈 | 호출 지점 |
| --- | --- | --- |
| 대주 폴링 | `trading/borrow_polling.py` | `RealtimeTradingEngine._run_short_cycle` |
| 숏 election context | `strategy_session._short_election_context` | `_election_context` |
| shadow 채점 루프 | `trading/shadow_evaluation_service.py` | `_run_short_cycle` (심볼별 라우팅) |
| 승격 컨트롤러 주기 실행 | `short_strategy_promotion` | `_maybe_evaluate_promotions` (300s) |
| `short_rescue_rate` 주입 | `ShadowEvaluationService` | `RuntimeHealth`로 전달 |
| 약세 랭킹 | `SectorRankTable.weakness_rank_for` | 강세 랭킹에서 파생(`size-rank+1`) |

`_run_short_cycle`은 모든 주문 판단이 끝난 **뒤에** 실행되고 전체가 try/except로 감싸여
있습니다. 주문을 만들지 않는 서브시스템의 부기(簿記)가 라이브 롱 경로를 멈추게 할 수는 없습니다.

### 15.2 실계좌 read-only 검증 결과 (2026-08-02) — 대주 엔드포인트 3개 모두 오류

추측한 대주 엔드포인트 3개를 **실계좌 read-only 호출**로 전부 검증했고, 전부 틀렸습니다.

| TR / path | 응답 | 판정 |
| --- | --- | --- |
| `TTTC8909R` `/trading/inquire-credit-psamount` | `조회종목은 신용종목이 아닙니다.(융자신규매수)` | TR은 존재하나 **융자(margin 매수) 가능금액** 조회. 질문 자체가 다름 |
| `CTSC0271R` `/quotations/credit-by-company` | `잘못된 TR 코드 입니다` | **TR id가 존재하지 않음** |
| `CTRP6504R` `/trading/inquire-credit-balance` | HTTP 404 | **경로가 존재하지 않음** |

셋 다 제가 **추측한** 값이었고, 네 번째를 추측하는 것은 같은 실수의 반복입니다. 게다가
"성공했지만 의미가 다른" 응답은 실패보다 나쁩니다 — 저널에 다른 의미의 숫자가 채워지고
하위 게이트 전부가 그것을 신뢰하게 됩니다.

**대응:** 가용성 조회를 KIS 메서드에서 떼어내
`app.trading.borrow_source.BorrowDataSource` 인터페이스 뒤로 옮겼습니다.

| 구현 | 용도 |
| --- | --- |
| `NullBorrowSource` (기본값) | 소스 미설정을 **명시적으로 보고**. `available=False`를 반환하지 않음 — 그러면 "빌릴 게 없는 정상 시장"과 구분되지 않음 |
| `FileBorrowSource` | 운영자 관리 JSON (`config/borrow_availability.json`). 소매 계좌는 어차피 브로커 웹 UI에서 확인하므로 임시방편이 아니라 정직한 1차 소스 |
| `CallableBorrowSource` | 올바른 브로커 조회가 확정되면 한 줄로 연결하는 어댑터 |

`FileBorrowSource`의 `observed_at`은 **필수**이며 파일 전체에 적용됩니다. 이것이 일반 신선도
규칙을 그대로 작동시킵니다 — 어제 만든 파일은 라이브 조회와 **동일한 시계**로 stale 처리되며,
로컬 파일이라고 신뢰받지 않습니다. `observed_at`이 없는 파일은 거부됩니다: 날짜 없는 locate는
시점 평가가 불가능하고, "지금"으로 찍는 것은 shadow 평가가 막으려는 바로 그 look-ahead 누수입니다.

기본 staleness 상한이 30초이므로 **손으로 관리하는 파일은 거의 즉시 stale이 됩니다.** 이것은
올바른 동작입니다 — 손으로 관리한 locate는 실제 숏 주문을 낼 근거로 충분하지 않다는 뜻입니다.
shadow 평가용으로 상한을 올릴 수는 있지만, 그만큼 locate가 사라졌을 수 있는 창을 넓히는
것임을 알고 해야 합니다.

**검증된 것도 있습니다.** `get_borrow_balance`는 이제 별도 엔드포인트가 아니라 포트폴리오
경로가 매 사이클 프로덕션에서 이미 쓰는 **동일한 `inquire-balance` / `TTTC8434R`** 응답을
읽습니다. 대주 lot은 credit 메타데이터(`loan_dt` / `loan_amt` / `crdt_type`)를 가진 행일
뿐이며, `crdt_type=05`만 SHORT이고 `01`(융자)은 **레버리지 롱**입니다.

### 15.3 GNN 방향별 head

명세는 "전략마다 LONG/SHORT head"를 요구했지만, 이 카탈로그에서 숏은 **별개의 strategy_id**
입니다(`opening_range_breakdown`은 `opening_range_breakout`의 방향이 아님). 따라서
**전략별 head가 곧 방향별 head**이며, 별도 방향 축을 추가하면 head가 2배가 되고 그 절반은
의미가 없습니다 — 숏 전용 논지의 LONG head는 학습할 것이 없습니다.

숏이 롱과 달리 실제로 필요한 것은 **대주 leg**이므로 head를 8채널에서 11채널로 넓혔습니다.

| 채널 | 내용 |
| --- | --- |
| 8 | `expected_borrow_cost_bps` |
| 9 | `borrow_probability` — 발동 시점에 locate가 존재할 확률 |
| 10 | `epistemic_uncertainty` — 모델의 무지. aleatoric(시장 노이즈)과 분리 |

두 가지가 구조적으로 강제됩니다.

- **롱은 대주 비용을 가질 수 없습니다.** 마스크가 카탈로그에서 생성되므로, 어떤 학습
  데이터로도 롱 head가 대주 비용을 청구하거나 — 더 나쁘게 — 할인하도록 배울 수 없습니다.
- **utility에 `borrow_probability`가 곱해집니다.** 빌릴 수 없는 종목에서만 존재하는 엣지는
  엣지가 아닙니다.

head 폭 변경은 기존 체크포인트를 **의도적으로 무효화**합니다 — 텐서 shape이 맞지 않아
`load_checkpoint`가 raise하고 런타임은 no-GNN으로 fallback합니다. 스키마 변경에 요구되는
fail-closed 동작입니다.

`RuntimeHealth.model_calibrated`는 여전히 `False`로 고정되어 있습니다. head는 존재하지만
**학습되지 않았습니다** — 새 채널을 학습할 라벨(실현 대주 비용, 실현 locate 성공률)이 forward
표본에서 나와야 하고, 그 표본이 아직 0입니다. 모든 arm은 `SHORT_MODEL_NOT_CALIBRATED`에서
막히며 이것이 정직한 상태입니다.

### 15.4 숏 보조 지표 — 있는 것과 없는 것

`app.features.short_indicators`가 실제 데이터로 계산 가능한 것만 계산합니다.

| 지표 | 소스 | 상태 |
| --- | --- | --- |
| `spread_bps` | 호가창 | 계산됨 |
| `liquidity_score` | ADTV (RiskManager와 동일 매핑) | 계산됨 |
| `market_alignment` | 종목 수익률 vs 시장 수익률 | 계산됨 |
| `short_interest_ratio` | **없음** | 미측정 |
| `days_to_cover` | **없음** | 미측정 |

KRX는 공매도 잔고를 매일 공시하지만 이 저장소에서 수집하는 곳이 없습니다. 저장소 내 유일한
`short_net_change`는 **합성 데모 파이프라인**(`app.trading_pipeline`)이 만드는 값이고, source
policy가 라이브 판단 근거로 명시적으로 거부하는 데이터입니다. 그것으로 혼잡도 지표를 만드는
것은 샘플 데이터로 리스크 측정치를 날조하는 것이며, 없는 것보다 나쁩니다 — 아무것도 거르지
않으면서 작동하는 스퀴즈 필터처럼 보이게 됩니다.

**결과적으로 스퀴즈 필터가 비활성입니다.** `max_days_to_cover` / `max_short_interest_ratio`
게이트는 무조건 통과하고, fail-closed 부담 전체가 대주 게이트로 넘어갑니다. 이는 실질적인
방어 심도 감소이며, `short_indicator_gaps`가 `GET /api/short-strategies/status`의
`indicator_gaps`로 보고하므로 대시보드에서 "미측정"으로 보입니다 — 키가 없는 것을 보고
"통과했다"고 오해하지 않도록.

### 15.5 남은 작업

세션 기반 숏 2개(`opening_range_breakdown`, `market_intraday_momentum_short`)는 long 경로와
같은 `app.features.session_structure` 산출물을 받도록 배선되었다. 이는 영구 무력 상태를
해제한 것일 뿐 실거래 승격이 아니다. 둘 다 아래 대주 데이터 선행 조건과 별도 배포 사다리를
그대로 적용받으며 현재 상태는 전량 `SHADOW`, 주문 권한은 0이다.

1. **올바른 대주 조회 확정** — 이것이 첫 단추입니다. KIS 문서에서 대주 TR을 확인하거나,
   `config/borrow_availability.json`을 운영자가 유지합니다. 그전까지 저널은 비어 있고
   숏은 shadow 표본을 쌓지 못합니다. **사다리는 끝까지 배선되어 있으나 가동 중지입니다.**
2. **공매도 잔고 수집** — 스퀴즈 필터 활성화용. 없어도 대주 게이트가 fail-closed를 담당합니다.
3. **GNN 새 채널 학습** — forward 표본이 생긴 뒤. 1번이 선행 조건입니다.
4. **Phase 5/6 실주문** — 실계좌 대주 주문은 한 건도 제출되지 않았습니다.

## 16. 관련 문서

- [decision_and_risk.md](decision_and_risk.md) — 비용 게이트, 원금보호, 밴딧 선택 규칙
- [live_trading.md](live_trading.md) — 실주문 경로와 arming
- [ontology_and_gnn.md](ontology_and_gnn.md) — 온톨로지 허용 arm과 GNN 권한 분리
- [validation.md](validation.md) — 무엇이 측정되었고 무엇이 승격되지 않았는지

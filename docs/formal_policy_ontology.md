# 시장 근거와 동적 리스크 온톨로지

`src/app/ontology/policy_ontology.ttl`은 기존 거래 온톨로지를 확장한 RDF/OWL 문서다. 새로운 정책은 숫자가 붙은 태그 목록이 아니라, 시장·관측·출처·분석·정책 사이의 관계와 검증 조건을 가진다. 실제 숫자 계산은 `risk/ontology_thresholds.py`에서 수행하며, OWL 추론이나 신경망 출력만으로 주문을 허가하지 않는다.

| 구성 | 실제 정의 |
|---|---|
| Class | `Market`, `Session`, `Instrument`, `Observation`, `Regime`, `Strategy`, `Position`, `RiskAssessment`, `ThresholdPolicy`, `Threshold`, `EvidenceProjection`, `StrategyPerformanceObservation` |
| Instance | `KR`, `US`, 각 시장의 세션 종류, 실제 분석 시각별 관측·시장 국면·리스크 평가·정책·임계치 인스턴스 |
| Object Property | `appliesToMarket`, `originMarket`, `providedBy`, `observedMetric`, `usesEvidence`, `rejectedEvidence`, `assessedFrom`, `generatesPolicy`, `hasThreshold`, `observedRegime`, `observedStrategy`, `hasPerformance` 등 |
| Data Property | 실제 값·단위·관측 시각·최대 허용 경과 시간·변동성 측정 구간·근거 ID·정책 상태·실현 순수익·예측 순수익·표본 수 등 |
| Axiom | KR과 US의 구별, 승인/거부 관측의 상호 배타성, 현금 롱/숏 전략의 상호 배타성, 속성의 domain/range·역관계·하위 관계. 필수 값·동일 시장·미래/오래된 관측·변동성 구간·적응 정책의 근거 완전성은 별도 SHACL 제약으로 검증 |

네임스페이스는 `https://obaits.local/ontology/policy#`다. TTL에 적은 시장과 세션 인스턴스는 식별자다. 실제 개장 여부나 주문 권한을 정적 인스턴스만으로 주장하지 않는다.

## 실시간 계산과 감사 그래프 분리

1. `OntologyPolicyRuntime.resolve()`는 로컬 실시간 저장소와 이미 계산된 시장별 맥락을 읽는다. 브로커 호출, 모델 학습, 데이터 수집 시작을 수행하지 않는다.
2. `project_market_evidence()`는 가벼운 Python 자료형으로 시장, 출처, 시각, 단위, 범위, 측정 구간을 검사한다. 누락을 0으로 채우지 않는다. `.values`는 검증된 값의 읽기 전용 사전이다.
3. `resolve_ontology_policy()`가 이 근거로 손절·추적 손절·이익 목표·보유 시간·조기 탈출 확인 횟수·비중·거래/일일 손실 예산 등을 계산한다. 계산식의 계수는 연구 가정이며 수익성 검증 결과가 아니다. 별도 설정의 안전 상한은 동적 계산 결과가 넘을 수 없다.
4. `snapshot()`은 메모리에 저장된 최근 정책만 반환한다. `evidence_graph()` 또는 HTTP 감사 경로를 명시적으로 호출할 때 RDF 그래프를 만든다. SHACL 검증도 감사/테스트 단계에서 수행하며 호가 수신 경로에서 실행하지 않는다.

`project_market_evidence(..., required_metrics=...)`는 호출자가 필요한 근거를 명시하도록 한다. 실시간 정책에서는 종목 변동성, 호가 간격, 유동성, 시세 경과 시간, 시장 분석 신뢰도가 필수다. 필수 근거가 없으면 신규 진입용 정책은 성립하지 않으며, 보유 포지션의 보호적 청산에 필요한 별도 보수적 경로는 남는다.

## 측정과 출처의 의미

- 변동성은 최대 64개의 완료된 분봉에서 계산한다. 최소 6개 연속 분봉, 즉 5개 이상 60초 수익률이 필요하다. 진행 중인 봉·미래 봉·시간 간격 누락·서로 다른 시장/세션/스트림·충돌하는 중복 봉은 변동성의 연속 구간에 섞지 않는다. 원시 틱 간 표준편차에 임의로 60초 구간을 붙이지 않는다.
- 현재 호가는 실제 KIS WebSocket 출처와 거래 가능한 메타데이터, 양방향 가격을 확인한다. 거래소 시각과 수신 시각 중 오래된 값을 기준으로 경과 시간을 계산한다. 늦게 수신한 오래된 이벤트가 새 시세로 바뀌지 않는다.
- 유동성 지수는 현재 최우선 양방향 호가 중 작은 잔량을 최근 완료 분봉 거래량 중앙값과 비교한다: `depth / (depth + median_minute_volume)`. 고정 거래량 상수나 KRW/USD 환산값을 사용하지 않는다. 이 수치는 해당 호가창의 관측이며, 미국 전체 시장의 통합 최우선 호가나 실제 체결 가능 수량을 보장하지 않는다.
- 국장과 미장의 분석은 `latest_by_market()`의 각 사이클에서 가져온다. 국내 지표가 미국의 현지 변동성을 대체할 수 없다. `global_*` 자료만 기존 `indicator_graph.py`의 출처·적용 시장·최대 경과 시간 계약을 재확인해 연결한다. 일 단위 거시 지표는 원래 발표/관측 시간을 보존하며 실시간 종목 지표로 재분류하지 않는다.
- 검증된 시간 인식 R-GCN 보조 출력은 체크포인트 해시, 온톨로지 스냅샷 ID, 시장, 생성 시각, 만료 시각과 `ontology-risk-v1-entry-frozen-shadow` 검증 계약을 확인한 경우에만 포함한다. 현재 실제 정책에 연결하는 값은 예측 불확실성과 양수 크기로 표현한 조건부 순손실 예측(`expected_downside_net_bps`)이다. 이 값으로 시장 관측만으로 만든 기본 정책보다 손실 한도나 비중을 늘리거나 새 주문 권한을 만들지 못한다. 진입 시 고정한 가상 청산 기준의 순수익 예측을 이후 실제로 조정되는 청산 정책의 검증된 수익으로 취급하지 않는다.
- 전략 성과는 LIVE와 SHADOW 관측을 서로 다른 인스턴스로 내보낸다. `realizedNetBps`와 `expectedNetBps`를 별도 속성에 저장하고 없는 실제 수익을 예측으로 채우지 않는다.
- 계좌 현금 입출금을 보정한 고점 장부가 없으므로, 잔고 변화만으로 실제 낙폭을 발명하지 않는다. 명시적으로 측정된 `account.drawdown_rate`가 제공되었을 때만 계좌 낙폭 근거를 넣는다.

정책 만료는 모든 사용 근거의 유효 기간과 동적으로 정한 시세 허용 경과 시간을 반영한다. 이미 4초 지난 호가에 10초 기준을 적용했다면 정책이 앞으로 10초 더 유효하다고 표시하지 않는다. 분봉 캐시와 정책 캐시는 각각 기본 256개 종목으로 제한한다. 분봉은 같은 분 안에서 최대 10초 캐시하고 호가는 매 정책 계산에서 다시 읽는다.

## 확인 경로

- `/account`: 국내·미국별 최근/선택 종목의 목표 순수익률, 강제 손절, 추적 손절, 최대 보유 시간, 비중, 하루 손실 예산, 신뢰도, 조기 탈출 확인 횟수, 만료 상태와 근거 ID.
- `GET /api/ontology/policy`: 이미 계산된 정책 스냅샷. 호출만으로 정책을 새로 만들거나 브로커를 조회하지 않는다.
- `GET /api/ontology/policy/schema`: Class / Instance / Object Property / Data Property / Axiom 정의의 Turtle 파일.
- `GET /api/ontology/policy/{KR|US}/{symbol}`: 해당 종목의 최근 검증/거부 관측과 정책을 연결한 Turtle 감사 파일. FastAPI 작업 스레드에서 생성한다.

종목별 감사 파일에는 실제 R-GCN 인접 텐서에서 사용하는 전략 인스턴스와 `sameMethodologyFamily`, `confirmsStrategy`, `contrastsStrategy` 관계도 함께 담는다. 단순히 같은 이름의 스키마만 선언하지 않고, 해당 시장의 실제 비영(非零) 관계가 RDF 삼중항과 대응하는지 테스트한다.

스키마: `src/app/ontology/policy_ontology.ttl` 및 `policy_shapes.ttl`.
검증: `tests/test_policy_evidence.py`, `test_ontology_policy_runtime.py`, `test_policy_routes.py`, `test_operations_dashboard.py`.

원본 분봉 테이블에는 봉 자체의 최초 수신 시각/수정 이력 전체가 없다. 이 브리지는 현재 의사결정 시점에 알려진 완료 봉만 사용하지만, 과거 시점으로 돌리는 완전한 point-in-time 재생은 별도 원시 이벤트 저널로 확인해야 한다. 저장된 현재 봉을 과거 시점에도 알고 있었다고 가정한 백테스트 성과를 이 기능의 검증 결과로 취급해서는 안 된다.

실제 전방 관측 학습을 위한 `GraphPolicyContextCache`는 라이브 생성 시점의 원시 그래프 입력 49개를 최대 256종목까지 보관한다. 모델 입력 72개 중 나머지 23개는 전략 식별 열이므로 원시 맥락에 섞지 않는다. 입력 스키마, 시장 표시, 유한수, 출처 ID, 최대 5초 유효 기간과 `관측 시각 ≤ 포착 시각 ≤ 의사결정 시각 ≤ 만료 시각`을 확인한 뒤 전달한다. 원본 출처 ID를 별도로 유지하면서 내용 해시를 함께 기록해 저장 후 변경을 감지한다. 이 해시는 전자서명이 아니며, 과거 분봉을 뒤늦게 복원한 특징을 실제 과거 관측으로 인증하지 않는다. 아직 그래프 학습/서빙의 시장 분류기가 지원하지 않는 숫자로 시작하는 6자리 영숫자 KRX 코드는 이 학습 캐시에서 명시적으로 제외한다.

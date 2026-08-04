# 장외 시간 실거래 운영 가이드 (국내·미국 전 세션)

이 문서는 **정규장 밖에서** 데이터 수집·판단·주문을 운영하는 절차와 안전 장치를 설명한다.
API 근거는 `docs/kis_market_session_capability_matrix.md`, 리팩터링 배경은
`docs/realtime_session_gap_analysis.md`.

---

## 1. 세 가지 상태를 절대 섞지 않는다

운영 화면과 코드가 구분해야 하는 값이 셋이다. 예전에는 이 셋이 "장이 열렸는가" 하나로
뭉개져 있었고, 그 결과 브로커가 거부할 시각에 주문을 보내고 거부되지 않는 시각에는
보내지 않았다.

| 질문 | 대답하는 것 | 예 |
|---|---|---|
| 데이터가 오는가? | `MarketCapability.data_available` | NXT 19:00 → **예** |
| 공식 주문 route 가 있는가? | `trade_available` | NXT 19:00 → **예** (`EXCG_ID_DVSN_CD=NXT`) |
| 신규 진입해도 되는가? | `new_entry_allowed` | NXT 19:00 → **아니오** (세션 정책 + 실주문 미승인) |
| 청산할 수 있는가? | `exit_allowed` | NXT 19:00 → **예** |

`GET /api/system-diagnostics` 의 `market_session_capabilities` 가 이 네 값을 세션·venue별로
그대로 내려준다. `GET /api/realtime/runtime` 과 `market_session.new_entry_session_report()` 도
`data_available` / `trade_available` / `exit_allowed` / `new_entry_block_reasons` 를 포함한다.

**"데이터 수신 가능"을 "주문 가능"으로 표시하면 안 된다.** NXT처럼 시세는 오지만 세션별
실주문 승인이 없는 상태는 `is_data_only` 로 구별된다.

## 2. 실주문이 나가기 위한 3중 조건

```
공식 route 검증 (코드, 설정으로 우회 불가)
   AND readiness 통과 (신선한 체결·호가, 스프레드, 계좌 동기화, 리스크 게이트)
   AND 세션별 live_order_authorized (config/market_sessions.yaml)
```

세 번째가 없으면 `EXTENDED_LIVE_ORDER_NOT_AUTHORIZED` 로 fail-closed 된다.
기본 설정에서 `live_order_authorized: true` 인 세션은 **`KRX_REGULAR`, `KRX_CLOSING_AUCTION`,
`US_REGULAR` 뿐**이다. 장외 세션은 전부 false 로 출발한다.

기존 `TRADING_ALLOW_EXTENDED_HOURS_ENTRY` 는 backward-compatible alias 로 남아 있으나
**장외 세션에만** 적용되고, 그것만으로는 실주문이 열리지 않는다.

## 3. 세션별 운영 요약

### 국내

| 세션 | 시각(KST) | 데이터 TR | ORD_DVSN | 주의 |
|---|---|---|---|---|
| `KRX_PREOPEN` | 08:30–08:40 | `H0STOUP0`/`H0STOAA0` | `05` | **종가매매** — 임의 지정가는 거부된다 |
| `KRX_OPENING_AUCTION` | 08:40–09:00 | `H0ST*` | `00` | 단일가 접수, 09:00 체결 |
| `KRX_REGULAR` | 09:00–15:20 | `H0STCNT0`/`H0STASP0` | `00` | 신규 진입 기본 허용 |
| `KRX_CLOSING_AUCTION` | 15:20–15:30 | `H0ST*` | `00` | 신규 진입 차단, 청산 허용 |
| `KRX_AFTER_CLOSE` | 15:40–16:00 | `H0STOUP0`/`H0STOAA0` | `06` | **종가매매** |
| `KRX_AFTER_SINGLE_PRICE` | 16:00–18:00 | `H0STOUP0`/`H0STOAA0` | `07` | 시간외 단일가 |
| `NXT_PRE` / `NXT_REGULAR` / `NXT_POST` | 08:00–08:50 / 09:00–15:20 / 15:30–20:00 | `H0NXCNT0`/`H0NXASP0` | `00` | `EXCG_ID_DVSN_CD=NXT`. 시간외 3종(05/06/07) **사용 불가** |

`05`/`06` (장전·장후 시간외 종가)는 종가로 체결되는 주문유형이다. 라우터의
`validate_limit_price()` 가 참조 종가와 다르면 `CLOSING_PRICE_ORDER_TYPE` 로 차단한다.
참조 종가를 모르는 상태에서도 차단한다 — 임의 단가를 보내면 거부되기 때문이다.

### 미국

| 세션 | 시각 | subscription key | 주문 endpoint | 주의 |
|---|---|---|---|---|
| `US_DAYTIME` | 10:00–18:00 KST (**시세는 10:00–16:00**) | `R`+`BAQ/BAY/BAA` | `daytime-order` (`TTTS6036U`/`TTTS6037U`) | 지정가만. 모의투자 미지원 |
| `US_PREMARKET` | 04:00–09:30 ET | `D`+`NAS/NYS/AMS` | `order` (`TTTT1002U`/`TTTT1006U`) | 공식적으로 주문 가능 |
| `US_REGULAR` | 09:30–16:00 ET | `D`+`NAS/NYS/AMS` | `order` | 신규 진입 기본 허용 |
| `US_AFTERMARKET` | 16:00–**17:00 EST / 18:00 EDT** | `D`+`NAS/NYS/AMS` | `order` | 이전 코드의 20:00 ET 는 오류였다 |

주간거래는 **한국시간 고정창**이므로 요일 판정도 한국시간으로 한다. 토요일 12:00 KST 는
뉴욕 기준 금요일 밤이지만 주간거래는 열리지 않는다.

## 4. 두 개의 구조적 함정

### 4.1 서머타임 주간·프리마켓 중첩 (17:00–18:00 KST)

여름에는 주간거래(10:00–18:00 KST)와 프리마켓(17:00–22:30 KST)이 한 시간 겹치고,
**두 route 모두 공식적으로 열려 있다.** 자동으로 하나를 고르면 원주문/정정 불일치가
생기므로:

* 신규 진입 → `SESSION_ROUTE_AMBIGUOUS` 로 **차단**
* 청산 → `us_overlap_order_precedence` (기본 `night`) 로 결정론적 선택

### 4.2 주간거래 주문창 > 시세창 (16:00–18:00 KST)

공식 문서가 주문은 18:00까지, 시세는 16:00까지로 명시한다. 그 2시간 구간은
`data_available=False` + `DAYTIME_QUOTE_WINDOW_ENDED` 가 되어 신규 진입은 불가하고
청산 route 만 살아 있다.

## 5. 정정·취소는 원주문 route 를 따른다

`KisSessionOrderRouter.resolve_revise_cancel()` 은 **원주문 저널의 route family** 로만
라우팅한다. 이전 구현은 정정 시점의 시각으로 daytime/regular 를 다시 판정했고, 그래서
주간거래로 접수한 주문을 세션 경계 이후 정정하면 일반 `order-rvsecncl` 로 전송됐다.

* 원주문 route 기록이 없으면 → `RECONCILIATION_REQUIRED` 로 fail-closed
* 국내 정정은 원주문 `ORD_DVSN` 을 유지한다 (07로 접수한 주문을 00으로 정정하면 거부)
* 세션이 바뀌어 다른 venue 로 옮겨야 하면 **원주문 취소 + 신규 주문** 이라는 명시적 전이만 허용

## 6. 데이터 신뢰도

* 통합 피드(`H0UN*`)는 `is_consolidated=True`, `is_tradeable=False` — 시세 전용.
* 미국 무료 호가는 매수/매도 각 **10호가**지만 **나스닥 마켓센터 단일 시장**이다(NBBO 아님).
  `SINGLE_MARKET_CENTER_DEPTH` 사유코드가 항상 붙는다.
* 휴장 REST 스냅샷은 `feed_scope=REST_SNAPSHOT` 이며 `REST_SNAPSHOT_ONLY` 로
  실시간 신규매수와 forward label 에서 배제된다. 평가금액·화면 갱신에만 쓴다.
* v1 스키마 시절 행은 `metadata_inferred=1` 이며 신규 진입 근거가 될 수 없다.

## 6.1 배포 순서 — 반드시 이 순서로

**실행 중인 서버 프로세스는 예전 코드를 메모리에 들고 있다.** 스키마만 먼저 올리면 그
프로세스가 새 컬럼을 모르는 INSERT 를 계속 보내고, 새 NOT NULL 컬럼에 DEFAULT 가 없으면
수집 주기마다 죽는다. 실제로 그렇게 터졌다:

```
KIS overseas realtime collector failed:
IntegrityError: NOT NULL constraint failed: realtime_minute_bars.stream_id
```

두 가지로 방어한다:

1. **모든 신규 NOT NULL 컬럼에 DEFAULT 를 준다** (스키마 v3 이 `stream_id` 에 `DEFAULT ''`
   를 부여한다). 컬럼을 모르는 writer 의 INSERT 가 그대로 성공한다.
2. **배포 순서를 지킨다.**

```text
1) data/store/realtime_market_data.sqlite3 백업
2) 코드 배포 (아직 재시작하지 않음)
3) 서버 재시작   ← 이 단계에서 마이그레이션이 적용되고 새 코드가 메모리에 올라온다
4) scripts/check_market_session_readiness.py 로 schema_version 과 stream_inventory 확인
```

**재시작 전에는 metadata 가 채워지지 않는다.** 마이그레이션만 적용된 상태에서 들어오는
행은 `stream_id=''`, `tr_id=''`, `metadata_inferred=1` 이 되어 신규 진입 근거와
high-trust 학습 표본에서 제외된다 — 안전하지만 새 기능이 동작하지 않는 상태다.
`stream_inventory_1h` 에 `stream_id` 가 채워진 스트림이 보이면 정상 전환된 것이다.

## 7. 단계적 활성화 절차

각 단계 사이에 최소 1거래일 관찰한다.

1. **데이터만** — `data_enabled: true`, 나머지 false.
   `python scripts/check_market_session_readiness.py` 로 `stream_inventory_1h` 에 해당
   세션 스트림이 잡히는지, `quote_freshness` 가 임계값 안인지 확인.
2. **청산만** — `exit_enabled: true`. 기존 포지션 축소가 정상 라우팅되는지 저널로 확인.
3. **SHADOW 진입** — `TRADING_ALLOW_ENTRY_<SESSION>=true` 로 열되
   `live_order_authorized: false` 유지 → 의도만 생성되고 주문은 나가지 않는다.
   차단 사유가 `EXTENDED_LIVE_ORDER_NOT_AUTHORIZED` 하나로 깔끔한지 확인.
4. **소액 실주문** — `live_order_authorized: true` + `maximum_position_weight` 를 0.05 이하로.
5. **정상 운영** — 세션별 성능이 정규장 대비 하한을 넘으면 가중치 상향.

## 8. Rollback 절차

| 되돌릴 대상 | 방법 | 영향 |
|---|---|---|
| 특정 세션 실주문 | `config/market_sessions.yaml` 의 `live_order_authorized: false` | 즉시 fail-closed, 데이터 수집 유지 |
| 특정 세션 신규 진입 | `TRADING_ALLOW_ENTRY_<SESSION>=false` | 청산은 계속 가능 |
| 모든 장외 진입 | `TRADING_ALLOW_EXTENDED_HOURS_ENTRY=false` | 정규장만 진입 |
| NXT 라우팅 | 주문에 `execution_venue` 를 비워 둔다 → 문서상 기본값 KRX | NXT 시세 수집은 유지 |
| 국내 피드 축소 | `KIS_REALTIME_FEED=krx` | 08:00–20:00 커버리지 → 09:00–15:30 |
| 저장소 스키마 | **되돌리지 않는다.** v2 는 additive + PK 확장이며 v1 리더는 새 컬럼을 무시한다. 파일 백업에서 복원하는 것만이 되돌리기다 |
| 세션 도메인 모델 | `market_session.py` wrapper 를 통해 legacy 4단계 호출자는 그대로 동작 | — |

저장소 마이그레이션은 파괴적이지 않지만 **PK 재작성** 을 포함한다(테이블 재생성 후 rename).
되돌리기가 필요하면 마이그레이션 전 파일 백업이 유일한 경로이므로, 운영 전에
`data/store/realtime_market_data.sqlite3` 를 복사해 둔다.

## 9. 자주 보는 사유코드

| 코드 | 의미 | 조치 |
|---|---|---|
| `EXTENDED_ENTRY_DISABLED` | 세션 정책상 신규 진입 비활성 | 의도된 상태. GNN/전략 실패가 아니다 |
| `EXTENDED_LIVE_ORDER_NOT_AUTHORIZED` | 진입은 열렸으나 실주문 미승인 | 3단계(SHADOW) 정상 상태 |
| `SESSION_ROUTE_AMBIGUOUS` | 미국 주간·프리마켓 중첩 | 의도된 fail-closed |
| `DAYTIME_QUOTE_WINDOW_ENDED` | 주간 16:00–18:00 KST | 청산만 가능 |
| `EXCHANGE_CODE_UNRESOLVED` | 통합 피드 기반 주문 시도 / 거래소 미해석 | `execution_venue` 를 KRX 또는 NXT 로 확정 |
| `SESSION_CALENDAR_STALE` | 날짜가 캘린더 커버리지 밖 | `config/market_sessions.yaml` 캘린더 갱신 |
| `SESSION_CALENDAR_SUSPECT` | 캘린더가 불완전 (KR 음력 휴장일 등) | 차단 사유 아님. 캘린더 보강 권장 |
| `SINGLE_MARKET_CENTER_DEPTH` | 미국 호가가 나스닥 마켓센터 단일 시장 | 정보성. NBBO 가정 전략은 금지 |
| `CLOSING_PRICE_ORDER_TYPE` | 시간외 종가매매에 임의 단가 | 참조 종가를 넘겨라 |
| `PAPER_VENUE_UNSUPPORTED` / `PAPER_DAYTIME_UNSUPPORTED` | 모의투자 제약 | 실전 계정에서만 가능 |

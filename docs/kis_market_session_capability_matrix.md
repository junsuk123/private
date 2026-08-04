# KIS 시장·세션 Capability Matrix (공식 문서 검증본)

**검증 기준 자료 (1차, 최우선):**
`research_notes/한국투자증권_오픈API_전체문서_20260625_030000.xlsx`
— 한국투자증권 Open API 전체문서, 스냅샷 일자 2026-06-25, 339개 API 시트.
본 문서의 모든 TR ID / 엔드포인트 / 필드 허용값은 **이 파일에서 직접 인용**한 것이며,
추측하거나 기억에서 채운 값은 없다. 인용 위치는 각 절의 `출처` 항목에 시트명으로 기록한다.

**검증 일자:** 2026-08-04
**verification_source 식별자:** `KIS_OPENAPI_WORKBOOK_20260625`
**미검증 항목 표기:** `UNVERIFIED` — 데이터 수집만 허용(DATA_ONLY), 신규 주문 fail-closed.

---

## 0. 요약 — 프롬프트 가정과 공식 문서가 다른 3가지

작업 지시서의 일부 전제가 공식 문서와 불일치했다. 문서를 근거로 아래와 같이 정정하여 구현했다.

| # | 지시서 전제 | 공식 문서 사실 | 출처 시트 | 구현 반영 |
|---|---|---|---|---|
| 1 | "NXT 주문이 공식 확인되지 않으면 `NXT_ORDER_ROUTE_UNVERIFIED`로 차단" | **NXT 주문은 공식 문서에 존재한다.** `order-cash` / `order-rvsecncl`의 `EXCG_ID_DVSN_CD`에 `KRX / NXT / SOR` 3값이 명시되고, NXT 전용 `ORD_DVSN` 허용값 표가 별도로 제공된다. | `주식주문(현금)`, `주식주문(정정취소)` | NXT/SOR route를 **VERIFIED**로 모델링. 단 `live_order_authorized`는 세션별 기본 false → 운영자가 명시 승인해야 실주문. |
| 2 | "미국 HDFSASP0은 단일 depth", "미국 무료 호가의 단일 depth를 다단계처럼 쓰지 말라" | **미국 무료 시세가 매수/매도 각 10호가**다. 유료가 오히려 1호가이며 OpenAPI 미제공. | `해외주식 실시간호가` | `depth_level_count=10`, `is_consolidated=False`. 품질 제약은 depth가 아니라 **나스닥 마켓센터 단일 시장 호가(NBBO 아님)** 라는 점 → `SINGLE_MARKET_CENTER_DEPTH` 사유코드. |
| 3 | 미국 애프터마켓 = ET 16:00–20:00 (현재 코드 `market_session.py`) | KIS 공식 주문 가능 시간: **애프터마켓 06:00~07:00 KST (Summer Time 05:00~07:00)** → ET 16:00–17:00(EST) / 16:00–18:00(EDT). | `해외주식 주문`, `해외주식 정정취소주문` | 애프터마켓 종료를 DST에 따라 17:00/18:00 ET로 정정. 기존 코드의 20:00 ET는 **2~3시간 과대**였다. |

---

## 1. 국내주식 — 실시간 시세 (WebSocket)

`ws://ops.koreainvestment.com:21000` (실전) / `:31000` (모의)
출처 시트: 각 행의 시트명 그대로.

| TR ID | 이름 | 실전 | 모의 | 상태 |
|---|---|---|---|---|
| `H0STCNT0` | 국내주식 실시간체결가 (KRX) | O | O | VERIFIED |
| `H0STASP0` | 국내주식 실시간호가 (KRX) | O | O | VERIFIED |
| `H0UNCNT0` | 국내주식 실시간체결가 (통합) | O | 미지원 | VERIFIED |
| `H0UNASP0` | 국내주식 실시간호가 (통합) | O | 미지원 | VERIFIED |
| `H0NXCNT0` | 국내주식 실시간체결가 (NXT) | O | 미지원 | VERIFIED |
| `H0NXASP0` | 국내주식 실시간호가 (NXT) | O | 미지원 | VERIFIED |
| `H0STOUP0` | 국내주식 **시간외** 실시간체결가 (KRX) | O | 미지원 | VERIFIED |
| `H0STOAA0` | 국내주식 **시간외** 실시간호가 (KRX) | O | 미지원 | VERIFIED — 신규 발견, 기존 코드 미사용 |
| `H0STOAC0` | 국내주식 시간외 실시간예상체결 (KRX) | O | 미지원 | VERIFIED — 미사용 |
| `H0STMKO0` | 국내주식 **장운영정보** (KRX) | O | 미지원 | VERIFIED — 미사용. 세션 판정 보강에 활용 가치 있음 |
| `H0STCNI0` / `H0STCNI9` | 국내주식 실시간체결통보 | O | O | VERIFIED |

`장운영정보`는 (KRX) / (NXT) / (통합) 3종이 모두 존재한다.

**필드 개수 (positional 파싱용, 기존 코드에서 이미 검증됨 — 재확인 완료):**
체결 `H0STCNT0` / `H0UNCNT0` / `H0NXCNT0` = 46, `H0STOUP0` = 43.
호가 `H0STASP0` = 59, `H0UNASP0` / `H0NXASP0` = 65.

## 2. 국내주식 — 시간외 REST 보조 시세

| TR ID | 엔드포인트 | 이름 | 모의 | 상태 |
|---|---|---|---|---|
| `FHPST02300000` | `/uapi/domestic-stock/v1/quotations/inquire-overtime-price` | 국내주식 시간외현재가 | 미지원 | VERIFIED |
| `FHPST02300400` | `/uapi/domestic-stock/v1/quotations/inquire-overtime-asking-price` | 국내주식 시간외호가 | 미지원 | VERIFIED |
| `FHPST02310000` | `/uapi/domestic-stock/v1/quotations/inquire-time-overtimeconclusion` | 주식현재가 시간외시간별체결 | 지원 | VERIFIED |

용도 제한: WebSocket 보조/보완 전용. `feed_scope = REST_SNAPSHOT` 이므로 실시간 신규매수 적격 판정과
forward label 생성에는 사용하지 않는다.

## 3. 국내주식 — 주문 (핵심)

**출처 시트: `주식주문(현금)`, `주식주문(정정취소)` — 원문 인용**

| 항목 | 값 |
|---|---|
| 신규 주문 | `POST /uapi/domestic-stock/v1/trading/order-cash` |
| 실전 TR | 매도 `TTTC0011U` / 매수 `TTTC0012U` |
| 모의 TR | 매도 `VTTC0011U` / 매수 `VTTC0012U` |
| 정정·취소 | `POST /uapi/domestic-stock/v1/trading/order-rvsecncl` |
| 실전 / 모의 TR | `TTTC0013U` / `VTTC0013U` |
| `RVSE_CNCL_DVSN_CD` | `01` 정정 / `02` 취소 |

### 3.1 `EXCG_ID_DVSN_CD` (거래소ID구분코드, string(3), 필수 아님)

문서 원문: *"한국거래소 : KRX / 대체거래소 (넥스트레이드) : NXT / SOR (Smart Order Routing) : SOR
→ 미입력시 KRX로 진행되며, **모의투자는 KRX만 가능**"*

→ `KRX` | `NXT` | `SOR` 모두 **VERIFIED**. 모의투자 계정에서는 KRX만.

### 3.2 `ORD_DVSN` (주문구분) — 거래소별 공식 허용값

문서에 거래소별로 **서로 다른 표**가 제공된다. 아래는 원문 그대로.

| 코드 | 의미 | KRX | NXT | SOR |
|---|---|:--:|:--:|:--:|
| `00` | 지정가 | O | O | O |
| `01` | 시장가 | O | **X** | O |
| `02` | 조건부지정가 | O | X | X |
| `03` | 최유리지정가 | O | O | O |
| `04` | 최우선지정가 | O | O | O |
| `05` | **장전 시간외** | O | **X** | **X** |
| `06` | **장후 시간외** | O | **X** | **X** |
| `07` | **시간외 단일가** | O | **X** | **X** |
| `11`/`12` | IOC/FOK 지정가 | O | O | O |
| `13`/`14` | IOC/FOK 시장가 | O | O | O |
| `15`/`16` | IOC/FOK 최유리 | O | O | O |
| `21` | 중간가 | O | O | X |
| `22` | 스톱지정가 | O | O | X |
| `23`/`24` | 중간가 IOC/FOK | O | O | X |

**핵심 결론:** 국내 시간외 주문(`05`/`06`/`07`)은 **KRX 전용**이다. NXT·SOR로 시간외 주문을 보내면 안 된다.
`SLL_TYPE`(매도 시): `01` 일반매도 / `02` 임의매매 / `05` 대차매도, 미입력시 01.

### 3.3 세션별 주문 가능 시각 — `UNVERIFIED_BY_API_DOC`

`05`/`06`/`07` 각각이 **어느 시각 구간에서** 접수되는지는 이 API 문서에 없다(거래소 업무규정 영역).
따라서 세션 시각창은 코드 하드코딩이 아니라 **버전이 기록된 로컬 캘린더 스냅샷**
(`config/market_sessions.yaml`, `calendar_version`)으로 관리하고, 스냅샷 커버리지를 벗어난 날짜는
신규 진입 fail-closed(`SESSION_CALENDAR_STALE`)로 처리한다.

## 4. 해외주식 — 실시간 시세

| TR ID | 이름 | 상태 |
|---|---|---|
| `HDFSCNT0` | 해외주식 실시간**지연**체결가 (미국은 0분 지연 = 실시간 무료) | VERIFIED |
| `HDFSASP0` | 해외주식 실시간호가 — **미국 매수/매도 각 10호가 무료** | VERIFIED |
| `HDFSASP1` | 해외주식 지연호가(아시아) | VERIFIED (미사용) |
| `H0GSCNI0` / `H0GSCNI9` | 해외주식 실시간체결통보 | VERIFIED |

### 4.1 `tr_key` (subscription key) — 세션별 전환 규칙 (원문 인용)

| 세션 | 형식 | 예 | 시장구분 |
|---|---|---|---|
| 미국 야간거래(프리·정규·애프터) 무료 | `D` + 시장구분(3) + 종목 | `DNASAAPL` | `NYS` 뉴욕, `NAS` 나스닥, `AMS` 아멕스 |
| 미국 **주간거래** | `R` + 시장구분(3) + 종목 | `RBAQAAPL` | `BAY` 뉴욕(주간), `BAQ` 나스닥(주간), `BAA` 아멕스(주간) |
| 아시아 유료 | `R` + 시장구분(3) + 종목 | `RHKS00003` | — |

미국 유료시세는 OpenAPI 미제공 → 미국은 항상 `D`(야간) / `R`+`BA*`(주간).

### 4.2 미국 시세 품질 (원문 인용, 그대로 반영)

- 무료 실시간(나스닥 토탈뷰), 별도 신청 없이 제공. 유료 신청해도 OpenAPI는 무료 시세만.
- 무료 = 매수/매도 각 **10호가**, **나스닥 마켓센터에서 거래되는 호가 및 잔량 정보**.
- *"무료 실시간 시세 서비스는 유료 대비 평균 50% 수준"*, 시·고·저·종가 상이 가능,
  과거 데이터는 장 종료 후(오후 12시경) 갱신.
- 미국 당일 시가는 장중 상이할 수 있고 **익일 정정 표시**된다.

→ 도메인 반영: `is_consolidated = False`(NBBO 아님), `depth_level_count = 10`,
`source_quality`는 국내 다단계 호가보다 낮은 프로파일, 그리고 **당일 시가·전일 종가 파생 피처는
정정 위험이 있으므로 label/피처에서 revision-aware 처리**.

## 5. 해외주식 — 주문

| 용도 | 엔드포인트 | 실전 TR | 모의 TR | ORD_DVSN |
|---|---|---|---|---|
| 미국 일반(프리·정규·애프터) | `POST /uapi/overseas-stock/v1/trading/order` | 매수 `TTTT1002U` / 매도 `TTTT1006U` | `VTTT1002U` / `VTTT1001U` | 매수 `00,32,34,35,36` / 매도 `00,31,32,33,34,35,36` (모의는 `00`만) |
| 미국 일반 정정·취소 | `POST /uapi/overseas-stock/v1/trading/order-rvsecncl` | `TTTT1004U` | `VTTT1004U` | — |
| 미국 **주간거래** | `POST /uapi/overseas-stock/v1/trading/daytime-order` | 매수 `TTTS6036U` / 매도 `TTTS6037U` | **미지원** | **`00` 지정가만** (원문: "주간거래는 지정가만 가능") |
| 미국 주간 정정·취소 | `POST /uapi/overseas-stock/v1/trading/daytime-order-rvsecncl` | `TTTS6038U` | **미지원** | — |

`OVRS_EXCG_CD`: 일반·주간 공통 `NASD`(나스닥) / `NYSE`(뉴욕) / `AMEX`(아멕스).
일반 주문은 추가로 `SEHK, SHAA, SZAA, TKSE, HASE, VNSE` 지원(본 시스템 범위 외).

### 5.1 미국 거래시간 (KIS 공식, 한국시간 기준) — 원문 인용

`해외주식 주문` / `해외주식 정정취소주문` 시트:
> 1) 미국 : 23:30 ~ 06:00 (썸머타임 적용 시 22:30 ~ 05:00)
>    \* **프리마켓(18:00 ~ 23:30, Summer Time : 17:00 ~ 22:30), 애프터마켓(06:00 ~ 07:00, Summer Time : 05:00 ~ 07:00) 시간대에도 주문 가능**

`해외주식 미국주간주문` / `미국주간정정취소` 시트:
> \* 주간거래(장전거래)(한국시간 기준) : **10:00 ~ 18:00 (Summer Time 동일)**

`해외주식 실시간호가` / `실시간지연체결가` 시트:
> 해당 API로 **미국주간거래(10:00~16:00) 시세 조회**도 가능합니다.

**→ 환산 결과 (America/New_York 기준):**

| 세션 | KST(공문) | ET 환산 | 비고 |
|---|---|---|---|
| `US_PREMARKET` | 18:00–23:30 / ST 17:00–22:30 | **04:00–09:30** (양쪽 동일) | 주문 가능 (공식 명시) |
| `US_REGULAR` | 23:30–06:00 / ST 22:30–05:00 | **09:30–16:00** (양쪽 동일) | — |
| `US_AFTERMARKET` | 06:00–07:00 / ST 05:00–07:00 | **16:00–17:00 (EST)** / **16:00–18:00 (EDT)** | 주문 가능 (공식 명시) |
| `US_DAYTIME` | 10:00–18:00 (ST 동일) | KST 고정창 → ET로는 이동창 | **KST로 모델링해야 함** |

**중요 2점:**
1. **주문창 ≠ 시세창.** 주간거래 주문은 10:00–18:00 KST, 문서상 시세 조회는 10:00–16:00 KST.
   16:00–18:00 KST 구간은 "주문은 되지만 공식 실시간 시세 근거가 없음" → 신규 진입 fail-closed,
   청산만 허용(`DAYTIME_QUOTE_WINDOW_ENDED`).
2. **서머타임에 주간거래와 프리마켓이 17:00–18:00 KST에서 1시간 겹친다.**
   두 route 모두 공식 개방 상태이므로 자동 선택은 위험 → 결정론적 우선순위 규칙 + 신규 진입은
   모호성 fail-closed(`SESSION_ROUTE_AMBIGUOUS`).

### 5.2 주간거래 가능 종목 검증

문서 전체에 **"미국주간가능종목조회" 전용 API는 없다.**
현재 구현은 `해외주식 상품기본정보`(`CTPF1702R`, `/uapi/overseas-price/v1/quotations/search-info`)의
`dtm_tr_psbl_yn` 필드를 사용한다 — 엔드포인트·TR ID는 VERIFIED, 필드 의미는 문서에서
"주간거래 가능 여부"로 직접 확인되지 않아 `FIELD_SEMANTICS_UNVERIFIED`.
정책: 조회 실패/미지정(`None`)은 차단하지 않고(참조데이터 장애로 전체 거래 중단 방지),
명시적 `N`일 때만 차단. 문서 원문도 *"모든 미국 종목 매매가 지원되지 않습니다"* 라고만 명시.

## 6. 최종 Capability Matrix

`data` = 실시간 데이터 수집 가능 / `route` = 공식 주문 route 존재 /
`entry`·`exit` = 세션 정책 기본값(설정으로 상향 가능, 실주문은 `live_order_authorized` 추가 필요).

### 국내 (KR)

| 세션 | venue | 시각(KST) | data | route | ORD_DVSN | entry(기본) | exit(기본) |
|---|---|---|---|---|---|:--:|:--:|
| `KRX_PREOPEN` | KRX | 08:30–08:40 | `H0STOUP0`,`H0STOAA0` | order-cash | `05` | X | O |
| `KRX_OPENING_AUCTION` | KRX | 08:40–09:00 | `H0ST*`/`H0UN*` | order-cash | `00` | X | O |
| `KRX_REGULAR` | KRX | 09:00–15:20 | `H0STCNT0`,`H0STASP0` | order-cash | `00` | **O** | O |
| `KRX_CLOSING_AUCTION` | KRX | 15:20–15:30 | `H0ST*` | order-cash | `00` | X | O |
| `KRX_AFTER_CLOSE` | KRX | 15:40–16:00 | `H0STOUP0`,`H0STOAA0` | order-cash | `06` | X | O |
| `KRX_AFTER_SINGLE_PRICE` | KRX | 16:00–18:00 | `H0STOUP0`,`H0STOAA0` | order-cash | `07` | X | O |
| `NXT_PRE` | NXT | 08:00–08:50 | `H0NXCNT0`,`H0NXASP0` | order-cash `EXCG=NXT` | `00` | X | O |
| `NXT_REGULAR` | NXT | 09:00–15:20 | `H0NXCNT0`,`H0NXASP0` | order-cash `EXCG=NXT` | `00` | X | O |
| `NXT_POST` | NXT | 15:30–20:00 | `H0NXCNT0`,`H0NXASP0` | order-cash `EXCG=NXT` | `00` | X | O |
| (통합 피드) | KRX_NXT_UNIFIED | 08:00–20:00 | `H0UNCNT0`,`H0UNASP0` | **주문 route 없음** | — | X | X |
| `KR_CLOSED` | — | 그 외 | REST snapshot only | 없음 | — | X | X |

통합 피드(`H0UN*`)는 **시세 전용**이다. `EXCG_ID_DVSN_CD`에 "통합"에 해당하는 값이 없으므로
통합 피드로 관측한 체결을 근거로 주문할 때는 반드시 KRX 또는 NXT/SOR 중 하나로 route를 확정해야 하며,
확정 불가 시 `EXCHANGE_CODE_UNRESOLVED`로 차단한다.

### 미국 (US)

| 세션 | venue | 시각 | data (tr_key) | route | ORD_DVSN | entry | exit |
|---|---|---|---|---|---|:--:|:--:|
| `US_DAYTIME` | US_DAYTIME_VENUE | 10:00–18:00 KST (시세 10:00–16:00) | `R`+`BAQ/BAY/BAA` | `daytime-order` | `00`만 | X | O |
| `US_PREMARKET` | NASDAQ/NYSE/AMEX | 04:00–09:30 ET | `D`+`NAS/NYS/AMS` | `order` | `00` | X | O |
| `US_REGULAR` | NASDAQ/NYSE/AMEX | 09:30–16:00 ET | `D`+`NAS/NYS/AMS` | `order` | `00` | **O** | O |
| `US_AFTERMARKET` | NASDAQ/NYSE/AMEX | 16:00–17:00 EST / 16:00–18:00 EDT | `D`+`NAS/NYS/AMS` | `order` | `00` | X | O |
| `US_CLOSED` | — | 그 외 | REST snapshot only | 없음 | — | X | X |

정정·취소는 **원주문 route family를 그대로 따른다**: `daytime-order` → `daytime-order-rvsecncl`(`TTTS6038U`),
`order` → `order-rvsecncl`(`TTTT1004U`). 세션이 바뀌었다고 다른 family로 넘기지 않는다.

## 7. DATA_ONLY / fail-closed 로 남긴 항목

| 항목 | 사유코드 | 근거 |
|---|---|---|
| 통합 피드(`H0UN*`) 기반 직접 주문 | `EXCHANGE_CODE_UNRESOLVED` | `EXCG_ID_DVSN_CD`에 통합값 없음 |
| NXT 시간외 주문(`05`/`06`/`07` on NXT) | `SESSION_ORDER_TYPE_UNVERIFIED` | NXT ORD_DVSN 표에 05/06/07 없음 |
| 모의투자 NXT/SOR 주문 | `PAPER_VENUE_UNSUPPORTED` | "모의투자는 KRX만 가능" |
| 모의투자 주간거래 | `PAPER_DAYTIME_UNSUPPORTED` | daytime-order 모의 미지원 |
| 세션별 `05`/`06`/`07` 접수 시각 | `SESSION_CALENDAR_STALE`(스냅샷 만료 시) | API 문서 범위 밖 |
| `dtm_tr_psbl_yn` 필드 의미 | `FIELD_SEMANTICS_UNVERIFIED` | 문서에서 필드 설명 미확인 |
| 미국 주간 16:00–18:00 KST 시세 | `DAYTIME_QUOTE_WINDOW_ENDED` | 시세 문서는 10:00–16:00만 명시 |
| 서머타임 주간·프리마켓 중첩 구간 | `SESSION_ROUTE_AMBIGUOUS` | 두 route 모두 공식 개방 |
| KRX/NXT 세션 시각창·휴장일 | 로컬 캘린더 스냅샷 + 버전 | 거래소 규정, API 문서 밖 |

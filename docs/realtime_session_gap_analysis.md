# 실시간 세션 Gap 분석 (감사 결과)

**감사 기준 HEAD:** `142f73012ae181df084375ad07364d674c606046` (branch `main`)
작업 시작 시 확인한 HEAD가 지시서의 `analysis_baseline_commit`과 동일했다. working tree에는
`data/models/**` 자동 생성 artifact 다수와 소스 12개 파일의 미커밋 수정이 있었으나 본 리팩터링
대상 파일과는 겹치지 않았다.

**API 검증 근거:** `docs/kis_market_session_capability_matrix.md`
(원본: `research_notes/한국투자증권_오픈API_전체문서_20260625_030000.xlsx`)

---

## 1. 현재 데이터 수집 경로

| 경로 | 파일 | 상태 |
|---|---|---|
| 국내 WebSocket (KRX / NXT / 통합) | `src/app/data/kis_realtime.py:32-63` | 동작. TR ID 및 positional 필드 개수가 공식 문서와 일치함을 재확인. |
| 국내 피드 선택 | `kis_realtime.py:66-84` `_domestic_subscription_tr_ids()` | `KIS_REALTIME_FEED` 환경변수로 `unified`(기본) / `krx` / `nxt` **택1**. 동시 수집 불가. |
| 미국 WebSocket | `kis_realtime.py` (`HDFSCNT0` / `HDFSASP0`) | 동작. `DNAS*` ↔ `RBAQ*` 세션별 키 전환 기능 존재. |
| REST 휴장 fallback | `src/app/data/rest_snapshot_fallback.py` (92행) | `KIS_REST_SNAPSHOT_SOURCE` 별도 source 문자열로 실시간과 구분 (`realtime_types.py:10-14`). 설계 의도는 올바름. |
| 이벤트 파이프라인 | `src/app/data/event_pipeline.py` (430행) | 수집 → store → feature 경로. |
| 저장소 | `src/app/data/realtime_store.py:29-123` | `realtime_ticks` / `realtime_orderbook` / `realtime_minute_bars` / `market_data_health` / `data_source_events`. |

## 2. 현재 주문 경로

| 경로 | 파일 | 상태 |
|---|---|---|
| 국내 신규 | `kis_real.py:1578-1589` `_order_body` | `EXCG_ID_DVSN_CD` **하드코딩 `"KRX"`**. |
| 국내 정정·취소 | `kis_real.py:1606-1619` `_revise_cancel_body` | 동일하게 `"KRX"` 하드코딩. |
| 국내 신용 | `kis_real.py:614-636` `_credit_order_body` | 동일. |
| 국내 주문조회 | `kis_real.py:1670-1688` `_order_status_params` | 동일. NXT 주문은 조회에서 누락될 수 있음. |
| ORD_DVSN 산출 | `kis_real.py:2256-2275` `_domestic_order_division_code` | KST 시각으로 `05`/`06`/`07`/`00` 자동 선택. venue를 고려하지 않음. |
| 미국 일반 | `kis_real.py:680-717` `_place_overseas_limit_order` | `order` / `daytime-order` 분기. TR ID는 문서와 일치. |
| 미국 정정·취소 | `kis_real.py:748-770` | `order-rvsecncl` / `daytime-order-rvsecncl` 분기. |
| 미국 주간 세션 판정 | `kis_real.py:2292-2310` `_is_us_daytime_order_session` | **KST 09:00–16:50**. |
| TR ID 집합 | `kis_real.py:155-268` `KisEndpointSet` | `TTTC0011U/0012U/0013U`, `TTTT1002U/1006U/1004U`, `TTTS6036U/6037U/6038U` — **전부 공식 문서와 일치**. |

## 3. 세션 판정이 중복된 위치 (핵심 결함)

canonical 모듈(`market_session.py`)이 존재하는데도 **독립 구현이 7곳** 더 있고, KRX 정규장 경계조차
서로 다르다.

| # | 위치 | 판정 내용 | canonical과의 불일치 |
|---|---|---|---|
| 1 | `src/app/data/market_session.py:90-111` | KRX 09:00–15:30 / US pre 04:00 regular 09:30–16:00 after 16:00–**20:00** ET | (canonical) US 애프터마켓이 공식 문서보다 2~3h 과대 |
| 2 | `src/app/web.py:11793-11860` `_is_live_market_core_open` / `_is_live_market_extended_open` / `_is_us_market_holiday` | KRX core 09:00–15:30, KRX extended 09:00–**16:50**, US extended 04:00–20:00 ET | KRX extended가 NXT(08:00–20:00)·시간외단일가(→18:00) 미포함, 휴장일 집합 **중복 정의** |
| 3 | `src/app/web.py:5573-5615` | 대시보드용 인라인 시각창 재구현 | 위 두 곳과 또 다름 |
| 4 | `src/app/trading/realtime_trading_engine.py:68-73` `_is_krx_core_buy_session` | KRX 09:00–**15:20** | **세 번째** KRX 정규장 경계 |
| 5 | `src/app/execution/kis_real.py:2292-2310` `_is_us_daytime_order_session` | KST **09:00–16:50** | 공식 주간거래는 **10:00–18:00 KST** |
| 6 | `src/app/data/llm_classifier.py:438-441` | KR 09:00–15:30 / US 09:30–16:00 | 휴장일 미고려 |
| 7 | `src/app/features/live_feature_frame.py:549` | `session_open = local.replace(hour=9, ...)` 하드코딩 | 미국·시간외 세션에서 의미 없음 |

## 4. 확인된 실제 결함 (동작에 영향)

| ID | 결함 | 근거 | 영향 |
|---|---|---|---|
| D1 | 미국 주간거래 주문창이 **09:00–16:50 KST**로 잘못 설정 | 공식 10:00–18:00 KST | 09:00–10:00 KST에 `daytime-order` 호출 → KIS 오류. 16:50–18:00 KST에는 주간거래 대신 일반 `order`로 라우팅 → 그 시각 미국 정규장·프리마켓 아님(겨울) → 거부. |
| D2 | 미국 애프터마켓 창 ET 16:00–**20:00** | 공식 16:00–17:00(EST)/16:00–18:00(EDT) | 18:00–20:00 ET 주문 시도 → 거부. 또한 그 구간을 "거래 가능"으로 표시. |
| D3 | 정정·취소가 **현재 시각으로 route family를 재판정** | `kis_real.py:753`, `:765` | 주간거래로 접수한 주문을 세션 경계 이후 정정하면 일반 `order-rvsecncl`로 전송 → 원주문 불일치. 반대 방향도 동일. |
| D4 | `EXCG_ID_DVSN_CD` 하드코딩 `"KRX"` | `kis_real.py:1586,1618,623,1685` | NXT/SOR 주문 불가. 통합 피드로 NXT 체결을 관측해도 KRX로만 주문 → NXT 유동성 시각에 거부. |
| D5 | `ORD_DVSN`이 venue를 무시 | `kis_real.py:2256-2275` | `EXCG_ID_DVSN_CD`가 NXT가 되는 순간 `05`/`06`/`07`은 문서상 **비허용** → 조용한 거부. |
| D6 | 시간외 `05`/`06`에 임의 지정가 전송 | `_order_body`가 `ORD_UNPR = limit_price` | 장전·장후 시간외 종가매매는 종가로 체결되는 주문유형. 임의 단가 → 거부 위험. |
| D7 | 학습 백필이 **6자리 숫자 종목만** | `live_training_pipeline.py:224-225` `length(symbol) = 6 and symbol not glob '*[^0-9]*'` | 미국 데이터가 materialized training row로 전혀 들어가지 않음. |
| D8 | `realtime_minute_bars` PK = `(symbol, minute_start)` | `realtime_store.py:92` | KRX·NXT·통합 피드가 같은 행을 다투며 거래량이 뒤섞임/덮어씀. |
| D9 | `realtime_ticks` / `realtime_orderbook`이 symbol 기준 혼재 | `realtime_store.py:34-74` | venue/session 컬럼 없음. 동일 종목의 KRX·NXT 체결 구분 불가. |
| D10 | `record_id`가 venue/session/TR 미포함 | `realtime_types.py:31-33, 86-88` | 통합 피드와 venue 피드가 같은 체결을 서로 다른 record로, 또는 반대로 충돌 가능. |
| D11 | 스키마 버전·마이그레이션 이력 없음 | `realtime_store.py` 전체 | 컬럼 추가 시 기존 파일 처리 경로 부재. |
| D12 | 미국 호가 depth를 국내와 동일 신뢰도로 사용 | `realtime_types.py:70-83` `imbalance` | 미국 호가는 **나스닥 마켓센터 단일 시장**(NBBO 아님). depth 수(10)는 국내와 유사하나 커버리지 의미가 다르다. |
| D13 | **미국 REST 시세가 WebSocket 체결로 위장돼 있다** | `src/app/trading/us_realtime_bridge.py:489` | `_make_records` 는 KIS 해외 **REST** 시세(`sequence_key`가 `us-kis-rest:`, 거래량이 세션 누적값)로 이벤트를 만들면서 `source=KIS_REALTIME_SOURCE`(웹소켓)로 태깅한다. live-buy 판정 3곳이 `source == KIS_REALTIME_SOURCE` 를 직접 비교하므로, REST 스냅샷이 실시간 체결과 동일하게 신규매수 근거로 통과한다. |

### 4.1 D13 실측 (metadata 도입 후 관측)

metadata 를 붙이자마자 두 경로의 비율이 드러났다. 2026-08-04 13:15-13:17 UTC(미국 프리마켓)
2분 표본:

| stream_id | feed_scope | is_tradeable | 호가 건수 |
|---|---|:--:|---:|
| `US:NASDAQ:FREE_REALTIME:HDFSASP0` | FREE_REALTIME | 1 | 13 |
| `US:NASDAQ:REST_SNAPSHOT:UNKNOWN` | REST_SNAPSHOT | 0 | **11** |
| `US:NYSE:FREE_REALTIME:HDFSASP0` | FREE_REALTIME | 1 | 7 |
| `US:NYSE:REST_SNAPSHOT:UNKNOWN` | REST_SNAPSHOT | 0 | 1 |

즉 미국 호가의 **약 40%가 REST 스냅샷**이었고, `source` 문자열만으로는 구별할 수 없었다.

**현 조치 (완료):** 브릿지 이벤트에 `feed_scope=REST_SNAPSHOT`, `is_tradeable=False` metadata 를
부착했다. 저장소·진단에서 두 경로가 분리되고 `is_live_buy_eligible()` 은
`REST_SNAPSHOT_ONLY` / `NON_TRADEABLE_FEED` 로 거부한다.

**남은 조치 (미완):** live-buy 판정 3곳(`live_feature_frame.py:258,266`,
`market_data_health.py:37,46`, `realtime_trading_engine.py:1291`)이 아직 `source` 문자열을
비교한다. 이것을 `FeedMetadata.is_live_buy_eligible()` 로 전환하는 것이 Phase 6 이며,
**전환하는 순간 미국 신규매수 근거의 약 40%가 사라진다.** 실거래 동작이 크게 바뀌므로
`source` 문자열 변경과 함께 별도 단계로, 운영자 승인 아래 진행해야 한다.

## 5. 데이터 중복 및 학습 누락 위험

- **중복 합산:** 통합(`H0UN*`)과 venue별(`H0ST*`/`H0NX*`) 피드를 동시에 켜면 동일 체결이 두 번
  저장될 수 있고, `build_latest_minute_bar`가 거래량을 이중 계산한다. 현재는 `KIS_REALTIME_FEED`가
  택1이라 우연히 회피되고 있을 뿐, venue-specific 분석을 켜는 순간 발생한다.
- **학습 누락:** D7로 미국 표본 0. artifact metadata에 market 구분만 있고 session/venue/feed 분포가
  없어(`live_training_pipeline.py` `_split_by_market_enabled`) 어떤 세션 데이터로 학습했는지 사후 확인 불가.
- **label 누수 위험:** forward label이 세션 경계·휴장 구간을 무조건 통과하면, 예를 들어 15:20
  관측의 600초 label이 시간외 단일가 체결로 계산된다. 세션이 다르면 체결 메커니즘도 다르므로
  전략 exit geometry와 어긋난다.
- **REST fallback 혼입:** source 문자열로는 구분되지만, 저장 스키마에 `feed_scope`가 없어
  SQL 레벨에서 "REST snapshot 제외"를 강제하기 어렵다.

## 6. 국내·미국·세션별 지원 여부

`docs/kis_market_session_capability_matrix.md` §6 표 참조. 요약:
- 국내 데이터: KRX 정규 + 시간외(`H0STOUP0`,`H0STOAA0`) + NXT 08:00–20:00 + 통합 — **전 세션 수집 가능**.
- 국내 주문: KRX 전 세션(`00/05/06/07`) + NXT/SOR(`00` 등, 시간외 제외) — **공식 지원 확인**.
- 미국 데이터: 주간(`R`+`BA*`) + 프리·정규·애프터(`D`+`NAS/NYS/AMS`) — **전 세션 수집 가능**.
- 미국 주문: 주간 `daytime-order`, 프리·정규·애프터 `order` — **전 세션 공식 지원 확인**.

## 7. 변경 대상 파일과 변경 이유

| 파일 | 변경 이유 |
|---|---|
| `src/app/data/market_capabilities.py` (신규) | SessionId/Venue/FeedScope/MarketCapability + `MarketSessionService` = 세션·capability 단일 source of truth. |
| `config/market_sessions.yaml` (신규) | 세션 시각창·휴장일 스냅샷(`calendar_version`) + 세션별 정책(data/entry/exit/live_order_authorized/임계값). API 문서 밖 정보를 코드에서 분리. |
| `src/app/data/market_session.py` | 신규 서비스로 위임하는 backward-compatible wrapper로 축소. D2 수정. |
| `src/app/data/realtime_types.py` | 이벤트 metadata 필드 추가(D10), `record_id`에 stream identity 포함. |
| `src/app/data/realtime_store.py` | versioned migration + venue/session/feed 컬럼 + minute bar stream identity(D8/D9/D11), venue↔통합 dedup. |
| `src/app/data/kis_realtime.py` | 파싱 시 metadata 부착, 세션 기반 subscription/TR 계산, in-place resubscribe 유지. |
| `src/app/execution/kis_session_order_router.py` (신규) | 단일 주문 라우터. D1/D3/D4/D5/D6 수정. |
| `src/app/execution/kis_real.py` | 라우터에 위임, 중복 세션 헬퍼 제거. |
| `src/app/schemas/domain.py` | `FinalOrder`에 `market_session`/`execution_venue`/`exchange_code`/`order_condition` 선택 필드. |
| `src/app/trading/realtime_trading_engine.py` | `_is_krx_core_buy_session` 제거 → canonical service 위임. |
| `src/app/features/live_feature_frame.py` | 세션 인식 피처(session-local VWAP, gap, seconds_from/to session), D12 품질 프로파일. |
| `src/app/models/live_training_pipeline.py` | D7 수정, session/venue 분포 기록, label 세션 경계 정책, market-session fallback. |
| `src/app/web.py` | 중복 세션 헬퍼 위임, `/api/realtime/runtime`·`/api/live-training/status`·`/api/system-diagnostics` 확장. |
| `scripts/check_market_session_readiness.py` (신규) | 실주문 없는 read-only readiness 점검. |
| `.env.example`, `config/secrets/kis_api_keys.env.example`, `run.ps1`, Pi profile | 세션별 설정 노출. |
| `docs/*` | capability matrix, extended hours 운영 문서, architecture/live_trading/decision_and_risk 갱신. |

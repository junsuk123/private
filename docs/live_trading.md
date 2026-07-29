# Live Trading

설치, 게이트, 운영 절차, 비상 정지, 런타임 프로파일을 하나로 묶은 문서입니다.

> **현재 posture:** `run.ps1`로 띄운 Windows 런타임은 **실주문을 제출할 수 있는 상태**입니다. `TRADING_MODE=live_trading`, `LIVE_TRADING_ENABLED=true`, `KIS_LIVE_ENABLED=true`, `LIVE_ORDER_SUBMIT_ENABLED=true`, `REQUIRE_MANUAL_ARMING=false`가 프로세스에 강제 설정됩니다. read-only 상태가 아닙니다.

## 1. 설치

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

로컬 전용(ignored) 설정 파일 생성:

```powershell
copy config\secrets\kis_api_keys.env.example config\secrets\kis_api_keys.env
copy config\principal_protection.example.json config\principal_protection.json
copy config\trading_costs.example.json config\trading_costs.json
copy config\live_trading_safety.example.json config\live_trading_safety.json
copy config\order_execution.example.json config\order_execution.json
```

채워야 하는 값: KIS app key/secret, 계좌번호와 상품코드, HTS ID와 고객유형, 초기 원금과 보호 하한, **직접 확인한** KIS 수수료율과 국내 증권거래세율. 이 파일들은 커밋하지 않습니다.

## 2. 점검

```powershell
python scripts/live_readiness_check.py --dry-run
python scripts/live_readiness_check.py --check kis-auth,kis-account --no-orders
python scripts/check_kis_connection.py --account
python scripts/live_order_dry_run.py --symbols 005930,000660 --no-submit
python scripts/check_realtime_market_data.py --symbols 005930 --fixture path\to\kis_fixture.txt
```

리포트는 `data/reports/live_readiness_*.json`, `data/reports/live_order_dry_run_*.json`에 남습니다. `require_recent_readiness_report=true`이므로 30분 이내의 readiness 리포트가 없으면 제출이 막힙니다.

실제 JSONL 데이터셋으로 학습:

```powershell
python scripts/train_live_short_horizon_models.py --dataset data\training\live_short_horizon.jsonl
```

`--demo-fixture`는 코드 경로 검증 전용이며 항상 live-ineligible로 표시됩니다.

## 3. 시작

```powershell
.\run.ps1
```

런처가 하는 일: 포트 `8010`의 기존 서버 종료 → live 프로세스 플래그 설정 → 자동 시작 서비스 설정 → `http://127.0.0.1:8010/account` 열기 → 관리 창이 닫히면 서버 종료.

수동 실행:

```powershell
python .\run.py --host 127.0.0.1 --port 8010
```

주요 화면:

- 계좌 대시보드 `http://127.0.0.1:8010/account`
- 실시간 거래 상태 `http://127.0.0.1:8010/api/realtime-trading/status`
- 계좌 payload `http://127.0.0.1:8010/api/account/dashboard`
- AI/모델 검증 `http://127.0.0.1:8010/api/ai/validation`
- live 학습 상태 `http://127.0.0.1:8010/api/live-training/status`
- 전략 소유 경로 상태 `http://127.0.0.1:8010/api/refactor/dashboard`

## 4. 제출 게이트

`LiveExecutionCoordinator`와 KIS 어댑터는 live 주문 전에 **전부** 요구합니다.

- 입력 객체가 `FinalOrder`
- 주문 유형이 `LIMIT`
- 해당 시장에서 KIS가 지원하는 side
- 수량과 지정가가 양수
- 국내 라우팅 시 유효한 6자리 KRX 심볼
- `LIVE_TRADING_ENABLED=true`
- `KIS_LIVE_ENABLED=true`
- `KIS_PAPER_TRADING=false`
- `LIVE_ORDER_SUBMIT_ENABLED=true`
- `KILL_SWITCH_ENABLED=false`
- 런타임 가드가 요구하면 manual arming 통과
- KIS credential 검증, 토큰 발급/로드, 계좌 잔고 조회, 필요 시 WebSocket approval key 발급
- idempotency 키가 다른 주문 payload로 사용된 적 없음

### BUY 추가 요건

`REALTIME_BUY_ENABLED=true`, 해당 통화의 1주 매수 현금 + 버퍼, 신선한 broker quote 또는 KIS 실시간 체결/호가 근거, 적응형 `max_spread_bps` 이내의 spread, 충분한 유동성, 적응형 매수 임계 이상의 fallback/온톨로지/런타임 점수, `REALTIME_MODEL_AUXILIARY_ONLY=true`일 때 모델은 보조 입력으로만, 그리고 결정론적 `RiskManager` 승인.

자주 보이는 BUY 거부 코드: `MODEL_FEATURE_UNAVAILABLE:...`, `WIDE_SPREAD:x>ybps`, `LOW_LIQUIDITY`, `FALLBACK_SCORE_BELOW_THRESHOLD:x<y`, `ONTOLOGY_REQUIRED_FOR_MODEL_FALLBACK`, `MODEL_AUXILIARY_ONLY_NEEDS_CONFIRMATION`, `INSUFFICIENT_CASH_FOR_ONE_SHARE`. 순수익 게이트 코드는 [decision_and_risk.md](decision_and_risk.md)에 있습니다.

### SELL / REDUCE

BUY보다 먼저 평가됩니다. 승인 경로: 비용 차감 후 이익 목표 도달, `REALTIME_ALLOW_LOSS_EXIT=true`일 때 트레일링/손실 청산, 국내 드로다운 축소, 국내 긴급 청산, 국내 집중도 축소, 시간/시세 기반 청산 정책.

이미 열린 SELL이 있고 대체 가격이 사실상 동일하면 `open_sell_kept`를 기록하고 중복 제출하지 않습니다.

## 5. 조용한 사이클 읽기

조용한 사이클이 곧 고장은 아닙니다. `/api/realtime-trading/status`를 보세요.

| 신호 | 의미 |
| --- | --- |
| `open_sell_kept` | 같은 유효 가격의 SELL이 이미 열려 있음 |
| `HOLD_BELOW_PROFIT_TARGET` | SELL 평가됐지만 비용 차감 후 목표가 미달 |
| `MODEL_FEATURE_UNAVAILABLE:...QUOTE_STALE,ORDERBOOK_STALE` | 실시간 입력이 stale/결측이라 모델이 채점 불가 |
| `WIDE_SPREAD:x>ybps` | 적응형 정책 대비 spread 과다 |
| `LOW_LIQUIDITY` | 후보 유동성 부족 |
| `FALLBACK_SCORE_BELOW_THRESHOLD` | 규칙/온톨로지 fallback 점수 미달 |
| `INSUFFICIENT_CASH_FOR_ONE_SHARE` | 해당 통화 가용 현금이 1주 가격 + 버퍼 미만 |

`blocked=0`이고 `errors=0`이면 엔진은 정상 동작 중이며 저품질 거래를 의도적으로 거부하고 있는 것입니다.

**모든 필드가 비어 있고 AUC가 얼어붙고 거래가 0이면 대개 원인은 하나입니다 — 장이 닫혀 ticks=0.** `MarketPhase` 분류기와 REST 스냅샷 fallback이 이 상태를 오류가 아니라 상태로 표시합니다.

**"이벤트 LLM 대기/timeout"** 은 Ollama가 실행 중이 아니라는 뜻입니다. LLM 자동 감지는 기동 시 1회이므로 Ollama를 켠 뒤 앱을 재시작해야 합니다.

## 6. 종료 버튼

`/account`의 종료 버튼 순서:

1. `REALTIME_BUY_ENABLED=false` 설정
2. 엔진의 BUY 비활성화 제어 호출
3. 실시간 거래 루프 정지
4. live 게이트를 통과하는 범위에서 보유 종목에 profit-seeking limit SELL 제출
5. 서버 종료 예약
6. `run.ps1`이 관리 브라우저를 닫고 로컬 프로세스 정리

profit-seeking 종료 가격은 기본 `average_price × 1.0025` 이상입니다. 하드 손실 임계를 넘은 보유 종목은 노출을 더 키우지 않도록 현재 broker 시세를 씁니다. **시장가 패닉 버튼이 아닙니다.** 기본 주문 유형은 여전히 LIMIT입니다.

## 7. 비상 정지

```powershell
python scripts/disarm_live_trading.py
$env:KILL_SWITCH_ENABLED="true"
```

`KILL_SWITCH_ENABLED=true`는 `LiveExecutionCoordinator`의 신규 live 제출을 차단합니다. 청산은 막지 않고 신규 매수만 막으려면:

```powershell
$env:REALTIME_BUY_ENABLED="false"
```

하드 스톱 정리:

- `KILL_SWITCH_ENABLED=true` — 신규 live 제출 차단
- `REALTIME_BUY_ENABLED=false` — 신규 BUY 평가/제출 차단, 청산 관리는 유지
- KIS credential 누락, 런타임 가드 실패, idempotency 실패, 계좌/토큰 검사 실패 — live 제출 차단

## 8. Arming은 ARM/라즈베리파이가 아닙니다

`scripts/arm_live_trading.py` / `disarm_live_trading.py`는 **live 주문 제출 안전 스위치**입니다. `config/secrets/live_trading_armed.json`(TTL ~900초)을 쓰고 지웁니다. 여기서 "arm"은 *live 제출을 무장한다*는 뜻이며 ARM CPU 아키텍처나 Raspberry Pi와 무관합니다.

`require_manual_arming=true`일 때 arming 전에는 `submit`이 `LiveExecutionBlocked`를 던지고 엔진은 `blocked`만 기록합니다(실주문 없음). 현재 `config/live_trading_safety.json`은 `require_manual_arming: false`이고 `run.ps1`도 `REQUIRE_MANUAL_ARMING=false`를 강제하므로, **manual arming이 강제되기를 원한다면 두 곳 모두 `true`로 바꿔야 합니다.**

```powershell
python scripts/arm_live_trading.py
python scripts/disarm_live_trading.py
```

## 9. 런타임 프로파일

거래 정책 설정(`config/profitability_policy.yaml`, `config/dynamic_exit_policy.yaml`, `config/position_sizing_policy.yaml`)은 프로파일 간에 공유됩니다. 프로파일은 런타임 posture(가속기, 유니버스 크기, 갱신 주기, 안전 플래그)만 바꿉니다.

| 프로파일 | 파일 | 역할 |
| --- | --- | --- |
| Windows / Intel NPU (기본) | `run.ps1` env 기본값 | 전체 노드: OpenVINO NPU, 온톨로지, 학습, GUI |
| 소액 계좌 | `config/runtime_profiles/small_account.env` | 보수적 net-edge 하한 + 사이징 |
| Raspberry Pi (선택) | `config/runtime_profiles/raspberrypi.env` + `scripts/run_raspberrypi.sh` | 저전력 모니터 / 실행 가드 노드 |

`run.ps1`은 Windows 전용(`Get-NetTCPConnection`, `Get-CimInstance`)이며 Intel NPU + OpenVINO를 가정합니다.

Pi 프로파일은 Intel NPU / OpenVINO / 로컬 LLM / 대규모 유니버스 스캔 / 고빈도 갱신을 끄고, `LIVE_ORDER_SUBMIT_ENABLED=false` + `REQUIRE_MANUAL_ARMING=true`를 유지합니다.

```bash
set -a; . config/runtime_profiles/raspberrypi.env; set +a
./scripts/run_raspberrypi.sh --port 8010
```

Pi 권장 역할: KIS 계좌/주문 상태 모니터, kill-switch/disarm watchdog, 경량 대시보드, REST/WebSocket 헬스 모니터. **권장하지 않음**: 대규모 유니버스 스캔, 무거운 온톨로지 추론, OpenVINO NPU 추론, 모델 학습, 대형 차트. 이미지 패키징용 별도 런처는 [raspberry_pi_deployment.md](raspberry_pi_deployment.md)를 보세요.

## 10. 로그와 재동기화

| 위치 | 내용 |
| --- | --- |
| `logs/run-server.out.log`, `run-server.err.log` | 서버 stdout/stderr |
| `logs/live-orders.jsonl` | live 주문 저널 |
| `logs/live-feature-frames.jsonl` | feature frame 저널 |
| `logs/refactor-shadow-comparison.jsonl` | legacy/ontology/CPU/NPU 판단 비교 |
| `data/store/account_dashboard.sqlite3` | 계좌 대시보드 스토어 |
| `data/store/realtime_market_data.sqlite3` | 실시간 시장 스토어 |
| `data/store/causal-order-journal.jsonl` | intent → verdict → order 인과 저널 |
| `data/models/live_short_horizon/` | live 모델 artifact |
| `data/reports/live_readiness_*.json` | readiness 리포트 |

알 수 없는 네트워크/브로커 오류 이후 재시작 전에:

1. KIS live 계좌 잔고를 읽습니다.
2. 제출된 broker order id의 상태를 확인합니다.
3. 열린 SELL 주문이 pending/filled/canceled/amendable 중 무엇인지 확인합니다.
4. **브로커 상태를 재동기화하기 전에는 결과가 불명확한 주문을 재시도하지 않습니다.**

## 11. 리스크 자세

엔진은 의도적으로 보수적입니다. 넓은 spread, 얕은 유동성, stale 실시간 입력, 현금 부족, 온톨로지/런타임 근거 없음, 모델 단독 승인은 모두 무거래로 이어져야 합니다.

이 저장소의 어떤 코드도 수익이나 자본 보전을 보장하지 않습니다. 통제 장치는 엔지니어링 게이트일 뿐입니다. 소액 계좌 단기 국내 단타는 왕복 비용을 고려하면 구조적으로 음의 기대값에 가깝습니다. 실현 손익으로 검증하세요 — [validation.md](validation.md).

## 12. 테스트

```powershell
python -m pytest
python -m pytest tests/test_web_live_flags.py tests/test_web_graph_payload.py tests/test_account_dashboard.py tests/test_kiosk_display_overview.py
python scripts/run_live_trading_test_suite.py
```

**live 주문 테스트는 금지입니다.** 모든 broker E2E 테스트는 mock, 기록된 이벤트, broker 시뮬레이션, 또는 명시적으로 설정된 paper 환경을 씁니다. 환경에 따라 OpenVINO/NPU, KIS 실계좌, 로컬 LLM 관련 테스트는 optional dependency나 secrets 상태의 영향을 받습니다.

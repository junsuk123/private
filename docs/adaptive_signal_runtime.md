# 경량 적응형 모델과 장치 실행

실제 타임스탬프가 있는 학습 행은 기본적으로 `adaptive_relu` 모델을 사용한다. 기존 선형 아티팩트도 계속 읽는다. `LIVE_MODEL_FAMILY=linear`은 기존 학습 경로를 선택한다. 타임스탬프가 없는 과거 합성 테스트 자료는 기존 학습 경로를 유지한다.

## 실시간 경로

`LiveSignalPredictor`는 시장별 KR/US 레지스트리를 우선 사용한다. 모델은 표준화 → 범위 제한 → 고정 ReLU 은닉층(기본 64개) → 성공확률/비용 차감 수익률 두 헤드로 구성한다. 모든 장치가 동일한 학습 파라미터를 사용하며, 휴리스틱을 학습 모델로 표시하지 않는다.

장치 검색과 OpenVINO 그래프 컴파일은 별도 단일 작업 스레드에서 수행한다. 준비되는 동안 NumPy가 같은 모델로 예측한다. 기본 우선순위는 Intel NPU → NVIDIA CUDA → Intel GPU → CPU이며, `NPU_DEVICE_PREFERENCE=CPU`는 CPU를 명시적으로 선택한다. OpenVINO GPU는 NVIDIA GPU를 뜻하지 않는다. CUDA는 설치된 PyTorch와 드라이버가 실제 사용 가능하다고 보고할 때만 선택한다. Intel 이외 NPU는 이 구현에서 자동 지원하지 않으며 CPU로 내려간다.

NPU 입력은 고정 배치 크기를 사용하고 큰 배치는 여러 조각으로 나눈다. 캐시는 가중치 내용과 입력 크기를 포함하므로 온라인 갱신 후 옛 모델을 재사용하지 않는다. 컴파일/실행 실패의 실제 장치와 사유를 유지한다. 가속기 수치 검사를 통과한 모델만 사용하고 주문 승인 경계 근처에서는 CPU 결과로 다시 확인한다. 수익률 계산, 전략 선택, 리스크 및 주문 권한은 기존 CPU 규칙을 따른다.

`LiveSignalPredictor.status()`와 `NpuRuntimeManager.snapshot_status()`는 마지막 상태만 반환한다. GUI 조회로 학습, DB 전체 검색, 장치 검색 또는 컴파일을 시작하지 않는다.

## 제한된 온라인 학습과 승격

- DB에서 공급된 최근 행은 기본 최대 8,192개(설정 상한 32,768개)만 사용한다. 미래/미성숙 레이블, 잘못된 숫자와 시각, 종목·시각 중복을 제외한다.
- 전 종목을 같은 UTC 시각으로 분할하고 각 전략/주문 계획의 실제 레이블 구간(`label_horizon_seconds`)과 엠바고를 제거한다. 10분 기본값보다 긴 레이블도 실제 구간 전체를 제거한다. 제거 후 학습 표본이 없으면 승격하지 않는다.
- 표준화와 은닉층은 학습 구간에서만 만들며, 증분 갱신은 호환되는 이전 변환을 유지한다. 이전 모델 학습 시각이 검증 구간에 닿으면 처음부터 학습한다.
- 출력 헤드만 학습한다. 확률 헤드의 기본 반복은 초기 96회/증분 16회, 설정 상한 128회다. 수익률 헤드는 작은 정규화 회귀 문제를 푼다. 증분 재생은 최대 1,024행이다.
- 검증은 실제 확률·순수익·불확실성 승인 조건을 모두 적용한다. 시장 대비 초과수익(alpha)과 현금 손익을 구분하여 `raw_forward_net_return_bps`가 있으면 실제 비용 차감 수익으로 두 헤드를 학습·평가한다. 검증 수익에는 학습용 이상치 제한을 적용하지 않는다. 양의 순수익 하한 및 기존 검증 표본·종목 수·승격 기준을 통과해야 활성화한다.
- 검증 이후 전체 자료를 재학습해 미검증 가중치를 배포하지 않는다. 통과한 파라미터 그대로 원자적으로 게시한다. 약한 후보가 기존 모델을 자동 교체하지 않는 기존 정책을 유지한다.

훈련은 기존 수집/주기 학습 작업에서 수행하며 틱 예측 함수는 학습하지 않는다. KR/US 데이터셋 지문이 바뀌지 않으면 해당 시장 학습을 생략한다.

## 기기별 저장소

기본 모델 레지스트리는 `runtime_database_path("models/live_short_horizon")`, 훈련 행 DB는 `runtime_database_path("live_training_rows.sqlite3")`, OpenVINO 바이너리 캐시는 `runtime_database_path("openvino_cache")`에 둔다. Synology 경로에서는 기기 로컬 저장소를 사용한다. 소스 코드는 동기화해도 활성 모델/SQLite/드라이버별 바이너리는 공유하지 않는다.

`LIVE_MODEL_ARTIFACT_ROOT`로 모델 경로를 명시할 수 있다. 기존 동기화 폴더의 `data/models/live_short_horizon`은 자동 이동하거나 삭제하지 않는다. 새 기기 로컬 레지스트리는 해당 기기의 데이터로 검증된 후보를 생성할 때 채워진다. 이전 모델을 사용할 경우 검증된 모델 폴더를 기기 로컬 경로로 별도 복사해 설정한다.

검증 명령: `python -m pytest -q tests/test_adaptive_signal_runtime.py tests/test_live_signal_predictor.py tests/test_live_training_pipeline.py tests/test_model_training_artifacts.py tests/test_npu_expanded_modules.py tests/test_portable_runtime.py tests/test_inference_backend.py tests/test_model_staleness.py`

자동 테스트의 비선형 XOR 자료는 구현 동작을 검증하기 위한 합성 자료다. 실시장 수익률 증거가 아니다.

## 실제 장치 측정 (2026-09-20)

`python scripts/benchmark_adaptive_runtime.py --iterations 100 --json-output docs/validation/adaptive_runtime_benchmark_2026-09-20.json`으로 재현한다. 스크립트는 합성 파라미터와 임시 컴파일 캐시만 사용하며, 거래소 API/모델 레지스트리/승격/주문을 사용하지 않는다.

현재 장치는 Intel Core Ultra 7 155H, Intel Arc iGPU, Intel AI Boost NPU로 확인했다. OpenVINO 2026.3.1에서 실제 NPU로 4,286개 파라미터 그래프를 실행했다. 최초 컴파일 547.349ms, 준비 후 100회 추론 중앙값 0.620ms, p95 0.719ms였다. NumPy와 출력 최대 절대 차이는 약 0.0001695였다. 최초 컴파일은 거래 틱이 아닌 전용 작업 스레드에서 수행한다.

원본 결과: [합성 장치 벤치마크 JSON](validation/adaptive_runtime_benchmark_2026-09-20.json). 이 수치는 합성 모델 추론만 측정한다. 피드 수신·전략·리스크·주문 왕복 지연과 실제 수익률을 포함하지 않는다. RTX 5060/4060 경로는 이 장치에서 실측하지 않았다.

구현 근거: [OpenVINO NPU 장치와 모델 캐시](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html), [OpenVINO NPU 고정 입력 및 지연 최적화 예제](https://docs.openvino.ai/2024/notebooks/hello-npu-with-output.html), [PyTorch CUDA 지원 확인](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available.html).

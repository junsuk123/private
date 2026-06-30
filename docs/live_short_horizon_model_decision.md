# 라이브 단기 모델 — 매수 신호 결정 기록 (2026-07-01)

실시간 단타 엔진의 매수 신호를 어디서 얻을지에 대한 결정과 근거를 남긴다.

## 결론 / 권장

- **매수는 당분간 온톨로지 경로로 나간다(단기 조치).** 라이브 단기 모델(`LiveSignalPredictor`)은 신호가 생길 때까지 **보조**로만 둔다.
- **모델이 실제로 쓸모 있어지려면 피처·라벨·호라이즌 개선이라는 연구성 작업이 필요하다(학습 트릭만으로는 불가).** AUC ≈ 0.29가 그 증거다.
- 모든 임계값은 env로 조정 가능하지만, **비예측 모델을 강제로 라이브 승급(`live_eligible`)시키는 것은 권하지 않는다** — 노이즈로 매매하는 셈이다.

## 배경 / 근거

- 증상: 새 실시간 엔진의 매수가 0건. 원인은 매수가 오직 라이브 모델에만 의존했는데, 모델이 **전 종목 확률 ≈ 0.0**(`PROBABILITY_BELOW_THRESHOLD`)을 출력.
- 1차 원인(수정됨): `_fit_logistic`이 클래스 가중치 없는 평이한 SGD라, positive 라벨 ~1% 불균형에서 전부 음성으로 붕괴. → 클래스 가중치 + L2 + 라벨 완화로 수정.
- 재학습(실데이터 11,234행) 결과:
  - 확률 분포 정상화(전부 0 → p90 ≈ 0.06, 약 8%가 0.51 돌파), positive 1.1% → 2.2%.
  - 그러나 **AUC ≈ 0.29** — 현재 피처가 단기 수익을 예측하지 못함(랜덤 이하). 따라서 모델은 **정상적으로 `live_eligible=False` 유지**.
- 해석: 학습 메커니즘은 고쳤으나, 현재 피처/라벨에는 단타 수익을 예측할 실제 신호가 없다. AUC가 0.5 미만(0.29)이라는 점은 신호 부재뿐 아니라 **부호 반전(모멘텀 vs 평균회귀) 가능성**도 시사한다.

## 조정 가능한 env (참고)

- `LIVE_LABEL_MIN_NET_RETURN_BPS` (기본 5) — positive 라벨 임계(비용 차감 후 bps). 기존 20은 과도하게 빡빡.
- `LIVE_MODEL_L2` (기본 0.001) — 로지스틱·선형 회귀 L2 정규화.
- `LIVE_MODEL_MIN_AUC` (0.55) / `LIVE_MODEL_MIN_PRECISION_AT_K` (0.35) / `LIVE_MODEL_MIN_AVG_RETURN_BPS` (0) — 라이브 적격 임계.
- 매수 온톨로지 경로: `REALTIME_ONTOLOGY_BUY_SCORE` (0.20), `REALTIME_ONTOLOGY_BUY_TARGET` (0.012).

## 후속 작업 (연구성)

1. 피처 예측력 점검 — AUC<0.5의 부호 반전 여부 확인, 모멘텀/평균회귀 피처 재정의.
2. 라벨·호라이즌 재설계 — 단타 타깃에 맞는 전방 구간/순수익 정의.
3. 데이터량·다양성 확보, 클래스 균형 재검토.
4. 모델이 진짜 예측력(AUC, precision@k, top-k 수익 > 0)을 보일 때만 `live_eligible` 승급.

## 안전

- 라이브 주문은 기존 안전 게이트(`evaluate_live_runtime_gates` + 수동 무장)가 최종 스위치다.
- RiskManager 현금·한도·신선도 게이트는 온톨로지/모델 어느 경로든 항상 적용된다.

관련 코드: `src/app/models/live_model_trainer.py`, `src/app/models/live_training_pipeline.py`, `src/app/trading/shared_decision_engine.py`(`evaluate_buy` 온톨로지 결합), `src/app/trading/realtime_trading_engine.py`.

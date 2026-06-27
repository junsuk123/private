# 단기매매 전략 설계 및 통합 문서

## 개요

이 문서는 논문 기반 단기매매 전략을 현재 시스템에 연결한 구조를 설명한다. 구현된 전략은 직접 주문을 실행하지 않고 `StrategyCandidate`를 만든다. 후보는 `StrategyCandidateFactory`, `TradingCostEngine`, 온톨로지 추론, `RealityCheckValidator`, `RiskManager`를 통과해야 하며 기본 실행 모드는 paper trading 또는 dry run이다.

실거래 자동 실행은 기본 비활성화되어 있다. 이 시스템은 연구 및 검증용 인프라이며 수익을 보장하지 않는다.

## 반영한 5개 논문

- Jegadeesh (1990): 단기 수익률 예측 가능성과 단기 반전
- Gao, Han, Li, Zhou (2018): 장 초반 수익률 기반 장중 모멘텀
- Brock, Lakonishok, LeBaron (1992): 이동평균 및 trading range breakout 규칙
- Gatev, Goetzmann, Rouwenhorst (2006): 가격 경로 유사 페어의 평균회귀
- Sullivan, Timmermann, White (1999): 데이터 스누핑과 과최적화 방지

## 전략별 시스템 적용 방식

`src/app/strategy/short_horizon.py`에는 세 가지 단기 전략이 있다.

- `ShortTermReversalEngine`: 최근 5분 하락폭이 실현 변동성 대비 충분히 큰 경우 보수적 반등 후보를 만든다.
- `IntradayMomentumEngine`: `ret_open_30m` 기본값으로 장 초반 수익률, 거래량, 시장 정렬을 확인한다.
- `TechnicalRuleEngine`: 기존 `sma`를 재사용해 MA crossover를 계산하고, 현재 이후 데이터를 쓰지 않는 rolling high breakout을 확인한다.

`src/app/strategy/pairs_relative_value.py`에는 long-only 페어 상대가치 전략이 있다.

- `PairUniverseBuilder`: sector/theme/market beta와 normalized price path distance로 유사 페어를 고른다.
- `PairRelativeValueEngine`: underperformer만 매수 후보로 만들며 공매도, 레버리지, 파생상품은 구현하지 않는다.

## ShortHorizonFeatureBuilder

`src/app/features/short_horizon_features.py`는 분봉/일봉 기반 피처를 만든다.

- 시간대별 수익률: `ret_1m`, `ret_3m`, `ret_5m`, `ret_15m`, `ret_30m`, `ret_1d`
- 장중 수익률: `ret_open_10m`, `ret_open_30m`, `ret_preclose_30m`
- 리스크/유동성: `realized_volatility_5m`, `realized_volatility_30m`, `volume_zscore`, `spread_rate`, `orderbook_depth_score`, `liquidity_score`
- 시장/시간: `market_alignment_score`, `time_of_day_weight`

`as_of` 이후 데이터는 필터링해 look-ahead bias를 피한다. 데이터 부족은 `None`, `missing_fields`, `is_valid=False`로 드러낸다.

## StrategyCandidateFactory

`src/app/strategy/candidate_factory.py`는 구현된 전략 엔진을 통합 호출한다.

Factory는 각 후보 생성 직후 `TradingCostEngine`을 호출하고 다음 조건을 통과한 후보만 남긴다.

- `expected_exit_price > 0`
- `net_expected_return > target_net_return`
- `gross_expected_return > break_even_return + safety_margin`
- `cost_to_alpha_ratio < max_cost_to_alpha_ratio`
- `spread_rate < max_spread_rate`
- `liquidity_score > min_liquidity_score`

랭킹은 gross 수익률만 쓰지 않는다.

```text
excess_return_after_cost = net_expected_return - target_net_return
ranking_score = excess_return_after_cost
                * confidence
                * ontology_score
                * liquidity_score
                * risk_adjustment
```

## TradingCostEngine 연결

`TradingCostEngine`은 매수/매도 수수료, 매도세, 슬리피지, 스프레드, 시장충격, safety margin을 반영한다. 전략 후보와 RiskManager 모두 이 비용 모델을 사용한다.

후보 단계에서 비용을 통과해도 주문으로 바로 이어지지 않는다. 최종 주문 후보는 다시 `RiskManager`에서 같은 비용 구조와 계좌/포지션 제한, live 안전장치를 검증받는다.

## 온톨로지 추론 연결

`src/app/graph/trading_strategy_semantics.py`는 `StrategyCandidate`와 `RankedStrategyCandidate`의 `ontology_tags`를 `SemanticFeatureRecord`로 변환한다.

긍정 신호:

- `ShortTermReversalBuy`
- `IntradayMomentumBuy`
- `PairMeanReversionBuy`
- `TechnicalBreakoutBuy`
- `CostEfficientTrade`
- `RealityCheckPassed`

위험 신호:

- `BidAskBounceRisk`
- `FalseBreakoutRisk`
- `SpreadTooWide`
- `SlippageRiskHigh`
- `CostBurdenHigh`
- `DataSnoopingRisk`
- `NoOutOfSampleValidation`

`TradeForbidden`은 `RiskManager`의 `ontology_tags` 차단 흐름으로 전달될 수 있다.

## RiskManager 및 FinalTradeGate

`RiskManager`는 deterministic final gate다. 다음 조건은 후보를 reject한다.

- `expected_exit_price` 누락
- 비용 차감 후 목표 순수익률 미달
- break-even 및 safety margin 미달
- 비용 부담 과다
- 스프레드/슬리피지 위험
- live 모드에서 검증 ID 또는 reality check 부재
- 온톨로지 `TradeForbidden`

이 통합은 `RiskManager` 또는 FinalTradeGate를 우회하지 않는다.

## RealityCheck 및 백테스트 검증

`src/app/evaluation/reality_check.py`는 `RealityCheckValidator`를 제공한다.

검증 지표:

- gross/net total return
- gross/net win rate
- average cost per trade
- average net profit per trade
- break-even failure ratio
- fee-converted loss ratio
- cost-to-alpha ratio mean/median
- out-of-sample net return
- out-of-sample Sharpe
- max drawdown after cost
- block bootstrap p-value

검증을 통과한 report만 `RealityCheckPassed` 태그를 제공한다. 실패하면 `NoOutOfSampleValidation`, `DataSnoopingRisk`가 붙는다.

## 설정 파일 설명

설정 파일은 `config/short_horizon_strategies.yaml`이다.

주요 섹션:

- `short_term_reversal`
- `intraday_momentum`
- `technical_rule`
- `pair_relative_value`
- `strategy_candidate_factory`
- `reality_check`
- `execution`

기본값은 보수적이다. `execution.live_trading_enabled`는 `false`이고 `execution.default_mode`는 `paper_trading`이다.

## paper trading 실행 방법

기본 앱 실행:

```powershell
.\run.ps1
```

paper trading API:

```text
POST /api/paper-trading/start
POST /api/paper-trading/step
```

단기 전략 Factory는 `trading_pipeline.generate_short_horizon_strategy_candidates(...)`를 통해 paper/dry-run 모드로 호출할 수 있다. live trading 모드를 요청해도 설정에서 `live_trading_enabled: false`이면 후보를 반환하지 않는다.

## 한계와 주의사항

- 논문 결과는 한국 개별주에 그대로 확정 적용되는 값이 아니다.
- beta, threshold, spread, liquidity, target net return은 백테스트와 paper trading으로 재검증해야 한다.
- 과거 성과와 RealityCheck 통과는 미래 성과를 보장하지 않는다.
- live trading에는 별도 검증, 계좌/주문 제한, 수동 승인 정책이 필요하다.
- 데이터 품질, 지연, 호가 공백, 체결 가능성, 세금/수수료 정책 변경은 실전 결과에 큰 영향을 줄 수 있다.

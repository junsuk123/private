# 최신 main 기준 단기매매 전략 연결 감사

분석 기준: GitHub `main` 브랜치 최신 clone

- Repository: `https://github.com/junsuk123/private.git`
- Branch: `main`
- Commit: `06e39f7ac63b83a543e54787a459db79de5daf21`
- 분석 일시: 2026-06-27
- 작업 방식: 현재 작업트리를 보호하기 위해 `%TEMP%` 아래 별도 디렉터리에 최신 `main`을 clone한 뒤 파일 구조와 구현을 직접 확인했다.

## 현재 전체 아키텍처

현재 시스템은 "데이터 수집 -> 후보 필터 -> 온톨로지 그래프/추론 -> 전략 신호 -> OrderIntent -> RiskManager -> FinalOrder 또는 차단 -> paper/backtest/live-readiness" 흐름이다.

핵심 모듈은 다음과 같다.

- `src/app/research/service.py`: RSS, HTML, dynamic page, Stooq, Yahoo chart, Alpha Vantage, FRED, ECOS, OpenDART 수집을 `ResearchRunResult`로 묶는다.
- `src/app/pipeline.py`: 샘플/저장/실시간 수집 결과를 병합해 `AnalysisContext`를 만들고, 후보 필터, 그래프, 추론, 전략, 리스크 검증을 연결한다.
- `src/app/trading_pipeline.py`: 전체 유니버스를 가볍게 스크리닝하는 `ontology_filter_1`과 NPU/CPU 후보 랭킹을 담당한다.
- `src/app/graph/builders.py`, `src/app/graph/reasoner.py`: 시장/지표/이벤트/수급 데이터를 온톨로지 triple로 만들고 `BuyCandidate`, `RiskAdjustedSizing` 등을 추론한다.
- `src/app/strategy/rule_based.py`, `src/app/strategy/goal_directed.py`: 온톨로지, 지표, 목표수익률을 점수화해 `StrategySignal`과 `OrderIntent`를 생성한다.
- `src/app/risk/manager.py`: 모든 주문 후보의 최종 결정 게이트다.
- `src/app/cost/trading_cost_engine.py`: 수수료, 세금, 슬리피지, 스프레드, 시장충격, 순기대수익을 계산한다.
- `src/app/backtesting/*`: 가속/스트리밍 paper trading 시뮬레이션과 비용 반영 체결 로그를 담당한다.
- `src/app/execution/*`: paper executor, disabled live executor, mock KIS, real KIS REST adapter가 분리되어 있다.

## 시장 데이터 흐름

현재 시장 데이터 수집은 `ResearchService.run()` 중심이다.

1. 설정 파일(`config/research_sources.*.json`)을 읽는다.
2. 상장 유니버스를 로드하고, 필요하면 `known_tickers`에 병합한다.
3. RSS/HTML/dynamic page는 텍스트와 이벤트로 분류된다.
4. Stooq/Yahoo/Alpha Vantage는 `MarketSnapshot`을 만든다.
5. FRED/ECOS는 `MacroMetricRecord`, OpenDART는 공시 이벤트를 만든다.
6. 결과는 `ResearchRunResult(events, raw_records, market_snapshots, macro_metrics, diagnostics)`로 반환된다.
7. 웹 런타임은 이 결과를 저장소와 병합하고 `build_analysis_context()`로 넘긴다.

`trading_pipeline.py`의 `build_lightweight_market_snapshots_from_markets()`는 이미 수집된 `MarketSnapshot`을 저비용 후보 필터용 `LightweightMarketSnapshot`으로 변환한다. 다만 실전 호가/체결 기반의 초단기 피처는 아직 이 흐름에 명시적으로 연결되어 있지 않고, 일부 momentum/volume 값은 stable hash 기반 추정값으로 채워진다.

## 전략 신호 생성 흐름

현재 전략은 두 계층이다.

1. `rule_based.py`
   - `IndicatorSnapshot`의 성장성, 수익성, PER, macro risk, volatility를 점수화한다.
   - 온톨로지 수급 triple(`InformedOrderFlowImbalance`, `ForeignInstitutionJointBuying` 등)을 추가 가중치로 반영한다.
   - 점수 1.8 이상이면 BUY, 아니면 HOLD가 기본이다.
   - BUY 신호만 `OrderIntent`로 변환한다.

2. `goal_directed.py`
   - 목표수익률과 기간으로 연환산 요구수익률을 계산한다.
   - 온톨로지 support/risk/contradiction, RSI, volume, 실적, valuation, volatility, macro risk, 수급을 점수화한다.
   - score와 목표 가능성에 따라 BUY/SELL/REDUCE/HOLD를 만든다.
   - compounding mode 여부에 따라 position weight 상한을 다르게 적용한다.

별도 단기매매 관련 모듈도 있다.

- `src/app/realtime/short_horizon.py`: `expected_return`, `downside_risk`, `confidence`를 받아 BUY/REDUCE/HOLD로 분류한다.
- `src/app/realtime/short_horizon_npu_predictor.py`: 5초/15초/60초 기대수익, downside risk, confidence를 예측하는 OpenVINO 또는 linear fallback 구조가 있다.

하지만 이 short-horizon 예측은 현재 `OrderIntent` 스키마와 `RiskManager` 비용 게이트에 직접 연결되어 있지 않다.

## 온톨로지 추론 흐름

온톨로지는 다음 순서로 구성된다.

1. `build_market_graph()`가 회사/종목/시장/섹터 triple을 추가한다.
2. `MarketSnapshot.investor_flow`가 있으면 수급 모델 triple을 추가한다.
3. NPU/CPU 점수로 `EarningsGrowth`, `ProfitabilityQuality`, `MacroRateRisk`, `VolatilityRisk`, `NpuCompositeMomentum`, `LiquiditySupport` 등을 추가한다.
4. 이벤트는 `event_mapper`를 통해 그래프에 병합된다.
5. `pipeline.py`는 시간 프레임, 파이프라인 단계, 튜닝 파라미터, 비용 모델 노드도 추가한다.
6. `OntologyReasoner.infer()`가 `BuyCandidate`, `AggressiveBuy` 상충, `OrderFlowConfirmedBuyCandidate`, `RiskAdjustedSizing` 등을 추가 추론한다.
7. `build_reasoning_paths()`는 support/contradiction/risk weight를 합산해 confidence와 conclusion을 만든다.

최신 main에는 비용 온톨로지 노드도 들어가 있다.

- `OntologyFilter3:FinalRiskApproval --usesCostModel--> TradingCost`
- `TradingCost --contains--> BrokerageFee/SellTax/Slippage/BidAskSpread/MarketImpact`
- `TradingCost --produces--> BreakEvenReturn/RequiredExitPrice/NetExpectedReturn`
- `NetExpectedReturn --supportsSignal--> NetProfitability`
- `NetProfitability --requiresApprovalFrom--> FinalTradeGate`

다만 이 비용 triple은 현재 설명/시각화 메타데이터 성격이며, 종목별 실제 `CostBreakdown` 값이 그래프 노드로 들어가지는 않는다.

## TradingCostEngine 분석

`TradingCostEngine`은 `config/trading_costs.json`을 기본 설정으로 읽고, 없거나 깨진 경우 내장 기본값을 쓴다.

계산 필드:

- 입력 식별자: `symbol`, `market`, `venue`, `instrument_type`
- 가격/수량: `entry_price`, `expected_exit_price`, `quantity`
- gross: `gross_expected_profit`, `gross_expected_return`
- 비용: `buy_fee`, `sell_fee`, `sell_tax`, `slippage_cost`, `spread_cost`, `market_impact_cost`, `total_cost`, `total_cost_rate`
- 손익분기: `break_even_return`, `break_even_exit_price`, `required_exit_price`
- net: `net_expected_profit`, `net_expected_return`
- 게이트 보조값: `cost_to_alpha_ratio`, `excess_return_after_cost`, `tradable`, `reject_reason`

현재 기본 설정:

- 국내 주식 KRX: 매수/매도 수수료 `0.000140527`, 매도세 `0.002`
- 국내 주식 NXT: 매수/매도 수수료 `0.000130527`, 매도세 `0.002`
- ETF/ETN/ELW: 기본 수수료 `0.000146527`, 매도세 `0`
- 기본 슬리피지 `0.0005`
- 안전마진 `0.001`
- 최대 비용/알파 비율 `0.5`

orderbook이 있으면 bid/ask spread를 계산해 `spread_rate`와 최소 slippage를 동적으로 올린다. `average_daily_trading_value`가 있으면 참여율 기반 market impact를 최대 1%까지 반영한다.

## RiskManager 분석

`RiskManager.validate()`는 다음 검사를 순차적으로 수행한다.

- LLM 직접 주문 차단
- live trading disabled 기본 게이트
- 허용 action, limit order 여부
- 일일 손실, 일일 거래 횟수
- 유동성, 변동성
- 중복 주문
- source data 존재 및 가격 양수
- 제한 상품 차단: margin, short, derivatives, leverage ETF, credit loan
- live mode일 때 source trust, data quality, synthetic, quote freshness, model uncertainty, unknown source
- 단일 종목/섹터/장중 포지션 한도
- 예수금 및 현금 비중

BUY가 기본 검사를 통과하면 비용 검사를 추가한다.

```text
expected_exit_price = market.last_price * (1 + max(0.0, intent.confidence * 0.012))
```

즉, 현재 코드에는 confidence 기반 expected_exit_price 임의 생성 로직이 존재한다. 이 값은 실제 short-horizon 예측수익률, 목표가, 호가, ATR, stop/take-profit 모델에서 온 것이 아니라 `confidence * 1.2%`를 기대상승률처럼 쓰는 방식이다.

이 로직은 단기매매 전략 적용 시 가장 먼저 수정해야 한다. `OrderIntent`에 명시적인 `expected_return`, `expected_exit_price`, `horizon_seconds`, `downside_risk`, `target_net_return` 또는 별도 `TradeForecast`를 추가하고, RiskManager는 confidence가 아니라 해당 예측값을 사용해야 한다.

## 백테스트 구조 분석

백테스트는 두 종류다.

1. `accelerated_demo.py`
   - 상장 유니버스 로드
   - 후보 필터
   - synthetic chart 생성
   - `build_goal_execution_plan()`
   - `RiskManager.validate()`
   - 승인 주문을 cash/holdings에 반영
   - 마지막에 강제 청산

2. `streaming_demo.py`
   - 분 단위 step 실행
   - 후보군/NPU 분류/goal-directed 전략/RiskManager
   - 일반 주문, 빠른 익절/손절, 마지막 청산, 원금보호 현금 확보 매도

최신 main 기준으로 수수료/세금/슬리피지는 반영되어 있다.

- `SimulatedTrade`에 `trading_cost`, `net_value` 필드가 있다.
- `_simulated_trade_cost()`가 `TradingCostEngine.policy_for()`를 사용한다.
- BUY는 `value + trading_cost`만큼 현금을 차감한다.
- SELL은 `value - trading_cost`만큼 현금을 더한다.
- final liquidation, fast take-profit/stop-loss, principal protection reserve에도 비용이 반영된다.

남은 한계:

- backtest 체결 비용은 `policy_for()`의 단순 비용을 사용하고, `TradingCostEngine.estimate()`의 순수익성 판단과 완전히 같은 경로는 아니다.
- short-horizon expected return 예측이 백테스트 주문 생성의 명시 입력으로 연결되어 있지 않다.
- 비용 포함 성과 평가 지표가 `evaluation`에 별도 정리되어 있지 않다.

## 실거래 안전장치 분석

안전장치는 여러 층으로 구현되어 있다.

- README와 `OperationModeManager`는 paper/live-readiness/live-trading을 분리한다.
- `OperationModeManager.start()`는 `LIVE_TRADING`일 때만 `live_orders_allowed=True`를 반환한다.
- paper trading과 live readiness는 live broker order를 제출하지 않는 guardrail을 가진다.
- `PaperOrderExecutor`는 "brokerage API called 없음" 메시지만 기록한다.
- `DisabledLiveOrderExecutor`는 submit 호출 시 RuntimeError를 발생시킨다.
- `KisDevelopersApiClient`는 기본 `KIS_LIVE_ENABLED=false`이며, `_ensure_enabled()`가 false일 때 주문/잔고 API 호출을 막는다.
- KIS live readiness는 인증/잔고 확인만 수행하고 주문 제출은 하지 않는 웹 흐름으로 분리되어 있다.
- `FinalOrder.manual_approval_required=True`가 기본이다.
- `RiskRules.live_trading_enabled=False`, `llm_direct_order_execution_allowed=False`, synthetic live data 불허가 기본값이다.

주의점:

- `OperationModeManager` 자체는 `LIVE_TRADING` 선택 시 `live_orders_allowed=True`를 만들 수 있다. 실제 KIS 호출은 `KIS_LIVE_ENABLED`가 false면 막히지만, 향후 live execution 연결 시 두 게이트가 동시에 만족되어야 주문이 나가도록 확인해야 한다.
- KIS real adapter의 `_ensure_enabled()` 메시지는 trading disabled라고 되어 있지만 `get_portfolio()`에도 적용된다. live-readiness 잔고 조회를 허용하려면 현재 웹 쪽처럼 별도 probe 설정이 필요하다.

## 단기매매 전략 적용을 위한 수정 대상 파일

우선 수정 대상:

- `src/app/schemas/domain.py`
  - `OrderIntent`에 `expected_return`, `expected_exit_price`, `horizon_seconds`, `downside_risk`, `target_net_return`, `forecast_source` 같은 필드를 추가하거나 별도 dataclass를 연결한다.

- `src/app/realtime/short_horizon.py`
  - `ShortHorizonSignal`을 `OrderIntent`로 변환하는 명시적 adapter를 추가한다.
  - confidence는 확률/품질 지표로 유지하고, 기대수익은 별도 필드로 전달한다.

- `src/app/realtime/short_horizon_npu_predictor.py`
  - 5초/15초/60초 출력 중 어떤 horizon을 주문에 사용할지 정책화한다.
  - predictor output을 `TradeForecast` 또는 확장 `OrderIntent`에 매핑한다.

- `src/app/risk/manager.py`
  - `expected_exit_price = last_price * (1 + confidence * 0.012)` 제거.
  - forecast 기반 expected exit/return을 사용하도록 변경.
  - 비용 검사는 `TradingCostEngine.estimate()`를 계속 사용하되, horizon별 최소 순수익률과 downside risk 조건을 함께 본다.

- `src/app/cost/trading_cost_engine.py`
  - 단기매매용 왕복비용, 최소 tick, 호가단위, bid/ask, 주문유형별 슬리피지, maker/taker 또는 venue별 정책을 확장한다.
  - `estimate()`에 `horizon_seconds`, `side`, `entry_order_type`, `exit_order_type` 같은 입력을 추가할지 검토한다.

- `src/app/backtesting/accelerated_demo.py`
  - short-horizon forecast를 생성/주입하는 경로를 추가한다.
  - 비용 포함 hit-rate, average net return, turnover, max drawdown, cost drag를 산출한다.

- `src/app/backtesting/streaming_demo.py`
  - streaming step에서 short-horizon predictor를 호출하고, 기존 goal-directed intent와 합성 또는 대체한다.
  - 빠른 익절/손절 로직을 forecast/RiskManager 기준과 일관되게 만든다.

- `src/app/evaluation/walk_forward.py`
  - 단순 split만 있으므로 비용 포함 단기매매 평가 리포트를 추가한다.

- `src/app/graph/semantic_builder.py`, `src/app/graph/reasoning_rules.py`, `src/app/graph/builders.py`
  - `ShortHorizonPositiveEdge`, `CostAdjustedAlpha`, `DownsideRisk`, `SpreadRisk`, `NetProfitability` 같은 종목별 triple을 추가한다.

- `src/app/trading_pipeline.py`
  - 실시간 호가/체결/수급 기반 lightweight snapshot을 추가하고 synthetic/hash 기반 필드와 명확히 분리한다.

보조 수정 대상:

- `src/app/features/indicator_engine.py`: 초단기 피처(1m/3m return, VWAP distance, micro volume imbalance 등) 추가.
- `src/app/features/semantic_feature_engine.py`: 단기매매용 의미 피처 매핑 추가.
- `src/app/web.py`: 비용 조정 기대수익/거절 사유/단기 forecast 표시.
- `tests/test_trading_cost_engine.py`, `tests/test_risk_manager.py`, `tests/test_streaming_demo_timing.py`: forecast 기반 비용 게이트 회귀 테스트 추가.

## 다음 태스크에서 우선 수정해야 할 항목

1. `OrderIntent` 또는 새 `TradeForecast` 스키마 확장
   - confidence와 expected return을 분리한다.
   - `expected_exit_price`, `expected_return`, `horizon_seconds`, `downside_risk`, `prediction_confidence`를 명시한다.

2. `RiskManager`의 confidence 기반 expected_exit_price 제거
   - 현재 가장 큰 논리적 결함이다.
   - forecast가 없으면 BUY 비용 게이트를 실패시키거나 보수적 fallback을 적용한다.

3. short-horizon predictor를 전략 흐름에 연결
   - `ShortHorizonNpuPredictor.predict()` -> `ShortHorizonRiskPolicy.classify()` -> `OrderIntent/TradeForecast` 흐름을 만든다.

4. 비용 포함 단기매매 게이트 강화
   - `net_expected_return > 0`뿐 아니라 `net_expected_return >= min_required_net_return(horizon)` 기준을 둔다.
   - cost-to-alpha ratio, spread/slippage risk, downside risk를 함께 차단한다.

5. 백테스트 평가 리포트 확장
   - gross PnL, net PnL, total fees, tax, slippage, cost drag, turnover, win rate, average holding horizon을 기록한다.

6. 온톨로지에 종목별 비용/알파 triple 추가
   - 현재는 비용 개념 노드만 있다.
   - 실제 후보별 `CostAdjustedAlpha`, `BreakEvenReturn:<ticker>`, `NetExpectedReturn:<ticker>`를 넣으면 설명 가능성이 좋아진다.

7. 실전 안전장치 통합 점검
   - `OperationModeState.live_orders_allowed`, `RiskRules.live_trading_enabled`, `KIS_LIVE_ENABLED`, `manual_approval_required`가 모두 만족될 때만 실주문 가능하게 end-to-end 테스트를 추가한다.

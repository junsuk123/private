import os, sys, time, json, math
from pathlib import Path
from datetime import datetime, timedelta, timezone
sys.path.insert(0, 'src')
os.environ.setdefault('PYTHONPATH', 'src')
os.environ.setdefault('DATA_ENV', 'realtime')
os.environ.setdefault('ONTOLOGY_ACCELERATOR', 'NPU')
os.environ.setdefault('OPENVINO_DEVICE', 'NPU')
os.environ.setdefault('ONTOLOGY_NPU_BATCH_SIZE', '2048')
from app.schemas.domain import MarketSnapshot, IndicatorSnapshot, SourceMetadata, RealtimeQuote, StrategySignal, OrderAction
from app.graph.npu_classifier import get_ontology_npu_classifier
from app.time_series import build_time_synchronized_frames
from app.realtime.learning import build_realtime_supervised_examples, run_hypothetical_realtime_test, update_realtime_model_artifacts
from app.storage import ModelArtifactStore

now = datetime.now(timezone.utc)
source = SourceMetadata(source_name='resource_probe', retrieved_at=now, raw_url='local://resource-probe', source_id='resource-probe')
markets = []
indicators = {}
quotes = []
signals = []
for i in range(4096):
    ticker = f'RP{i:04d}'
    price = 50.0 + (i % 500) * 0.7
    market = MarketSnapshot(
        ticker=ticker,
        market='US',
        company_name=ticker,
        sector=('Technology', 'Healthcare', 'Finance', 'Industrial')[i % 4],
        last_price=price,
        average_daily_trading_value=1_500_000_000 + (i % 100) * 10_000_000,
        volatility_20d=0.02 + (i % 70) / 1000.0,
        source=source,
    )
    markets.append(market)
    indicators[ticker] = IndicatorSnapshot(
        ticker=ticker,
        revenue_growth=0.03 + (i % 40) / 500.0,
        operating_income_growth=0.05 + (i % 45) / 400.0,
        operating_margin=0.08 + (i % 35) / 300.0,
        roe=0.07,
        debt_ratio=0.45,
        per=10.0 + (i % 30),
        pbr=1.2,
        rsi_14d=35.0 + (i % 50),
        volume_ratio=0.7 + (i % 70) / 50.0,
        macro_risk_score=(i % 60) / 100.0,
        source_ids=('resource-probe',),
    )
    if i < 300:
        quotes.append(RealtimeQuote(ticker=ticker, market='US', observed_at=now - timedelta(minutes=15), last_price=price, change_rate=0.001, source=source))
        quotes.append(RealtimeQuote(ticker=ticker, market='US', observed_at=now, last_price=price * (1.0 + ((i % 9) - 4) / 1000.0), change_rate=((i % 9) - 4) / 1000.0, source=source))
        action = OrderAction.BUY if i % 3 == 0 else OrderAction.REDUCE if i % 3 == 1 else OrderAction.HOLD
        signals.append(StrategySignal(ticker=ticker, action=action, confidence=0.55 + (i % 20) / 100.0, score=((i % 40) - 10) / 10.0, supporting_factors=('resource_probe',), contradicting_factors=(), reasoning_path_ids=()))

markets = tuple(markets)
quotes = tuple(quotes)
signals = tuple(signals)
classifier = get_ontology_npu_classifier()
summary = {'iterations': [], 'market_count': len(markets), 'learning_signal_count': len(signals)}
start = time.perf_counter()
# warm-up/compile
classifier.classify(markets, indicators)
for iteration in range(30):
    t0 = time.perf_counter()
    score_count = 0
    for _ in range(30):
        scores = classifier.classify(markets, indicators)
        score_count += len(scores)
    frames = build_time_synchronized_frames(markets=markets[:300], realtime_quotes=quotes)
    examples = build_realtime_supervised_examples(frames, signals)
    test_result = run_hypothetical_realtime_test(frames, signals)
    paths = update_realtime_model_artifacts(ModelArtifactStore(), examples, test_result) if iteration in {0, 29} else {}
    status = classifier.status()
    summary['iterations'].append({
        'iteration': iteration + 1,
        'elapsed_ms': round((time.perf_counter() - t0) * 1000, 1),
        'npu_scores_produced': score_count,
        'frames': len(frames),
        'examples': len(examples),
        'hypothetical_trades': test_result.get('trade_count', 0),
        'model_artifacts_written': paths,
        'npu': status.__dict__,
    })
summary['total_elapsed_s'] = round(time.perf_counter() - start, 3)
print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


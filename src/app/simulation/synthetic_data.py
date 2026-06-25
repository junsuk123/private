from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from app.features import OHLCVBar
from app.runtime import DataEnvironment
from app.schemas.domain import ClassifiedEvent, EventType, SentimentDirection, SourceMetadata


@dataclass(frozen=True)
class MarketSession:
    name: str
    start_time: time
    end_time: time


@dataclass(frozen=True)
class MarketCalendar:
    name: str = "US"
    timezone_name: str = "America/New_York"
    open_time: time = time(9, 30)
    close_time: time = time(16, 0)
    sessions: tuple[MarketSession, ...] = (
        MarketSession("regular", time(9, 30), time(16, 0)),
    )

    @classmethod
    def us(cls) -> "MarketCalendar":
        return cls(
            name="US",
            timezone_name="America/New_York",
            open_time=time(9, 30),
            close_time=time(16, 0),
            sessions=(
                MarketSession("premarket", time(4, 0), time(9, 30)),
                MarketSession("regular", time(9, 30), time(16, 0)),
                MarketSession("aftermarket", time(16, 0), time(20, 0)),
                MarketSession("day_market", time(20, 0), time(4, 0)),
            ),
        )

    @classmethod
    def krx(cls) -> "MarketCalendar":
        return cls(
            name="KRX",
            timezone_name="Asia/Seoul",
            open_time=time(9, 0),
            close_time=time(15, 30),
            sessions=(
                MarketSession("premarket", time(8, 0), time(9, 0)),
                MarketSession("day_market", time(9, 0), time(15, 30)),
                MarketSession("aftermarket", time(15, 40), time(18, 0)),
            ),
        )


@dataclass(frozen=True)
class SyntheticDataBundle:
    root: Path
    ohlcv_path: Path
    news_path: Path
    manifest_path: Path
    tickers: tuple[str, ...]
    bars_count: int
    news_count: int
    calendar: MarketCalendar


@dataclass(frozen=True)
class SyntheticCorpus:
    root: Path
    manifest_path: Path
    bundles: tuple[SyntheticDataBundle, ...]
    total_bars: int
    total_news: int


@dataclass(frozen=True)
class SyntheticScenarioConfig:
    randomness_scale: float = 1.0
    shock_probability: float = 0.015
    volume_spike_probability: float = 0.025
    regime_switch_probability: float = 0.08
    news_events_per_ticker: int = 8
    intraday_seasonality: bool = True


def is_market_open(moment: datetime, calendar: MarketCalendar) -> bool:
    return market_session_for(moment, calendar) is not None


def market_session_for(moment: datetime, calendar: MarketCalendar) -> str | None:
    local = moment.astimezone(ZoneInfo(calendar.timezone_name))
    local_time = local.time()
    for session in calendar.sessions:
        if _session_contains_time(local_time, session):
            session_day = _session_start_day(local, session)
            if _session_is_trading_day(session_day, calendar):
                return session.name
    return None


def generate_synthetic_training_bundle(
    environment: DataEnvironment | None = None,
    *,
    tickers: tuple[str, ...] | None = ("SIM_A", "SIM_B", "SIM_C", "SIM_D", "SIM_E"),
    ticker_count: int | None = None,
    start: datetime | None = None,
    trading_days: int = 15,
    interval_minutes: int = 5,
    calendar: MarketCalendar | None = None,
    seed: int = 20260613,
    scenario_name: str | None = None,
    scenario_config: SyntheticScenarioConfig | None = None,
) -> SyntheticDataBundle:
    env = environment or DataEnvironment.simulation()
    if env.mode == "live":
        raise ValueError("Synthetic training data must be generated only in the simulation environment.")
    env.ensure_layout()
    calendar = calendar or MarketCalendar.us()
    scenario_config = scenario_config or SyntheticScenarioConfig()
    tickers = tickers or _synthetic_tickers(ticker_count or 20)
    rng = random.Random(seed)
    start_local = (start or datetime(2026, 1, 5, tzinfo=ZoneInfo(calendar.timezone_name))).astimezone(
        ZoneInfo(calendar.timezone_name)
    )
    bars = _generate_bars(tickers, start_local, trading_days, interval_minutes, calendar, rng, scenario_config)
    news = _generate_news(tickers, bars, rng, scenario_config)

    bundle_name = scenario_name or f"bundle_{seed}"
    root = env.synthetic_dir / bundle_name
    root.mkdir(parents=True, exist_ok=True)
    ohlcv_path = root / "synthetic_ohlcv.csv"
    news_path = root / "synthetic_news.jsonl"
    manifest_path = root / "manifest.json"
    _write_ohlcv(ohlcv_path, bars)
    _write_news(news_path, news)

    manifest = {
        "synthetic": True,
        "environment": env.mode,
        "calendar": asdict(calendar),
        "tickers": tickers,
        "scenario_config": asdict(scenario_config),
        "bars_count": len(bars),
        "news_count": len(news),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "warning": "Synthetic data is for simulation, model tests, and offline training only. Never mix with live store.",
    }
    manifest_path.write_text(json.dumps(_to_jsonable(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return SyntheticDataBundle(
        root=root,
        ohlcv_path=ohlcv_path,
        news_path=news_path,
        manifest_path=manifest_path,
        tickers=tickers,
        bars_count=len(bars),
        news_count=len(news),
        calendar=calendar,
    )


def generate_synthetic_training_corpus(
    environment: DataEnvironment | None = None,
    *,
    scenarios: int = 5,
    ticker_count: int = 30,
    start: datetime | None = None,
    trading_days: int = 30,
    interval_minutes: int = 5,
    calendar: MarketCalendar | None = None,
    seed: int = 20260613,
    randomness_scale: float = 1.25,
) -> SyntheticCorpus:
    env = environment or DataEnvironment.simulation()
    if env.mode == "live":
        raise ValueError("Synthetic training corpus must be generated only in the simulation environment.")
    env.ensure_layout()
    root = env.synthetic_dir / f"corpus_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    bundles: list[SyntheticDataBundle] = []
    for scenario_index in range(max(1, scenarios)):
        scenario_seed = seed + scenario_index * 9973
        intensity = randomness_scale * (0.75 + scenario_index / max(1, scenarios - 1) * 0.75)
        config = SyntheticScenarioConfig(
            randomness_scale=intensity,
            shock_probability=min(0.12, 0.01 + scenario_index * 0.006),
            volume_spike_probability=min(0.16, 0.02 + scenario_index * 0.008),
            regime_switch_probability=min(0.25, 0.06 + scenario_index * 0.01),
            news_events_per_ticker=6 + scenario_index % 5,
            intraday_seasonality=True,
        )
        bundles.append(
            generate_synthetic_training_bundle(
                env,
                tickers=_synthetic_tickers(ticker_count),
                start=start,
                trading_days=trading_days,
                interval_minutes=interval_minutes,
                calendar=calendar,
                seed=scenario_seed,
                scenario_name=f"corpus_{seed}/scenario_{scenario_index:03d}",
                scenario_config=config,
            )
        )
    manifest_path = root / "manifest.json"
    manifest = {
        "synthetic": True,
        "environment": env.mode,
        "scenarios": scenarios,
        "ticker_count": ticker_count,
        "trading_days": trading_days,
        "interval_minutes": interval_minutes,
        "seed": seed,
        "total_bars": sum(bundle.bars_count for bundle in bundles),
        "total_news": sum(bundle.news_count for bundle in bundles),
        "bundle_roots": [str(bundle.root) for bundle in bundles],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(_to_jsonable(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return SyntheticCorpus(
        root=root,
        manifest_path=manifest_path,
        bundles=tuple(bundles),
        total_bars=int(manifest["total_bars"]),
        total_news=int(manifest["total_news"]),
    )


def _generate_bars(
    tickers: tuple[str, ...],
    start_local: datetime,
    trading_days: int,
    interval_minutes: int,
    calendar: MarketCalendar,
    rng: random.Random,
    config: SyntheticScenarioConfig,
) -> tuple[OHLCVBar, ...]:
    rows: list[OHLCVBar] = []
    tz = ZoneInfo(calendar.timezone_name)
    prices = {ticker: rng.uniform(20, 500) for ticker in tickers}
    regimes = {ticker: rng.choice(("uptrend", "downtrend", "range", "volatile", "panic", "meltup")) for ticker in tickers}
    day = start_local.date()
    generated_days = 0
    while generated_days < trading_days:
        if _session_is_trading_day(datetime.combine(day, time(0, 0), tzinfo=tz), calendar):
            step = 0
            for session in calendar.sessions:
                session_open, session_close = _session_bounds(day, session, tz)
                current = session_open
                while current < session_close:
                    local_session = session.name
                    session_volume_factor = _session_volume_factor(local_session)
                    session_volatility_factor = _session_volatility_factor(local_session)
                    session_drift_bias = _session_drift_bias(local_session)
                    for index, ticker in enumerate(tickers):
                        if rng.random() < config.regime_switch_probability:
                            regimes[ticker] = rng.choice(("uptrend", "downtrend", "range", "volatile", "panic", "meltup"))
                        regime = regimes[ticker]
                        drift = _regime_drift(regime) + session_drift_bias
                        base_volatility = _regime_volatility(regime) * config.randomness_scale * session_volatility_factor
                        seasonal = math.sin((step + index * 3) / 12) * 0.0009 if config.intraday_seasonality else 0.0
                        jump = _event_jump(regime, rng, config.shock_probability) * config.randomness_scale
                        shock = rng.gauss(drift + seasonal, base_volatility) + jump
                        previous = prices[ticker]
                        close = max(1.0, previous * (1 + shock))
                        wick_scale = abs(rng.gauss(0.0015, 0.0007)) * config.randomness_scale * session_volatility_factor
                        high = max(previous, close) * (1 + wick_scale)
                        low = min(previous, close) * (1 - wick_scale)
                        volume_multiplier = rng.uniform(0.65, 1.55) * session_volume_factor
                        if rng.random() < config.volume_spike_probability or abs(jump) > 0:
                            volume_multiplier *= rng.uniform(2.5, 9.0)
                        volume = int((80_000 + index * 15_000 + abs(shock) * 30_000_000) * volume_multiplier)
                        prices[ticker] = close
                        rows.append(
                            OHLCVBar(
                                ticker=ticker,
                                as_of=current.astimezone(timezone.utc),
                                open=round(previous, 4),
                                high=round(high, 4),
                                low=round(low, 4),
                                close=round(close, 4),
                                volume=float(max(1, volume)),
                            )
                        )
                    current += timedelta(minutes=interval_minutes)
                    step += 1
            generated_days += 1
        day += timedelta(days=1)
    return tuple(rows)


def _generate_news(
    tickers: tuple[str, ...],
    bars: tuple[OHLCVBar, ...],
    rng: random.Random,
    config: SyntheticScenarioConfig,
) -> tuple[ClassifiedEvent, ...]:
    labels = (
        ("MajorSupplyContract", SentimentDirection.POSITIVE, "won a material supply contract"),
        ("AnalystUpgrade", SentimentDirection.POSITIVE, "received an analyst upgrade"),
        ("GuidanceLowered", SentimentDirection.NEGATIVE, "lowered guidance due to demand weakness"),
        ("RegulatoryPenaltyNegative", SentimentDirection.NEGATIVE, "faces a simulated regulatory penalty"),
        ("ProductLaunchPositive", SentimentDirection.POSITIVE, "announced a simulated product launch"),
        ("RumorRisk", SentimentDirection.NEUTRAL, "is affected by an unverified market rumor"),
        ("LiquidityRiskHigh", SentimentDirection.NEGATIVE, "shows a simulated liquidity risk warning"),
        ("EarningsSurprisePositive", SentimentDirection.POSITIVE, "reported a simulated earnings surprise"),
        ("ContractCancellation", SentimentDirection.NEGATIVE, "had a simulated contract cancellation"),
    )
    by_ticker = {ticker: [bar for bar in bars if bar.ticker == ticker] for ticker in tickers}
    events: list[ClassifiedEvent] = []
    for ticker in tickers:
        event_count = max(1, config.news_events_per_ticker)
        sample_bars = rng.sample(by_ticker[ticker], k=min(event_count, len(by_ticker[ticker])))
        sample_bars = sorted(sample_bars, key=lambda bar: bar.as_of)
        for index, bar in enumerate(sample_bars):
            label, sentiment, fact = rng.choice(labels)
            source = SourceMetadata(
                source_name="synthetic_news",
                retrieved_at=bar.as_of,
                raw_url=f"local://synthetic/news/{ticker}/{index}",
                source_id=f"synthetic:news:{ticker}:{index}",
            )
            events.append(
                ClassifiedEvent(
                    event_id=f"synthetic-{ticker}-{index}",
                    event_type=EventType.NEWS,
                    title=f"{ticker} synthetic event: {label}",
                    summary=f"{ticker} {fact}.",
                    companies=(f"Synthetic {ticker}",),
                    tickers=(ticker,),
                    sectors=("SyntheticSector",),
                    sentiment=sentiment,
                    event_date=bar.as_of,
                    source=source,
                    key_facts=(fact,),
                    event_labels=(label,),
                    classification_confidence=round(rng.uniform(0.65, 0.95), 3),
                    classification_model="synthetic_event_generator_v1",
                )
            )
    return tuple(events)


def _synthetic_tickers(count: int) -> tuple[str, ...]:
    return tuple(f"SIM_{index:04d}" for index in range(1, max(1, count) + 1))


def _session_contains_time(local_time: time, session: MarketSession) -> bool:
    if session.start_time <= session.end_time:
        return session.start_time <= local_time < session.end_time
    return local_time >= session.start_time or local_time < session.end_time


def _session_start_day(local: datetime, session: MarketSession) -> datetime:
    if session.start_time > session.end_time and local.time() < session.end_time:
        return local - timedelta(days=1)
    return local


def _session_is_trading_day(local: datetime, calendar: MarketCalendar) -> bool:
    if calendar.name.upper() == "US":
        if local.weekday() == 6:
            return any(session.start_time > session.end_time for session in calendar.sessions)
        return local.weekday() < 5
    return local.weekday() < 5


def _session_bounds(day: Any, session: MarketSession, tz: ZoneInfo) -> tuple[datetime, datetime]:
    session_open = datetime.combine(day, session.start_time, tzinfo=tz)
    session_close = datetime.combine(day, session.end_time, tzinfo=tz)
    if session.end_time <= session.start_time:
        session_close += timedelta(days=1)
    return session_open, session_close


def _session_volume_factor(session_name: str) -> float:
    return {
        "day_market": 0.35,
        "premarket": 0.55,
        "regular": 1.0,
        "aftermarket": 0.45,
    }.get(session_name, 1.0)


def _session_volatility_factor(session_name: str) -> float:
    return {
        "day_market": 1.25,
        "premarket": 1.20,
        "regular": 1.0,
        "aftermarket": 1.15,
    }.get(session_name, 1.0)


def _session_drift_bias(session_name: str) -> float:
    return {
        "day_market": 0.00005,
        "premarket": 0.00008,
        "regular": 0.0,
        "aftermarket": -0.00002,
    }.get(session_name, 0.0)


def _regime_drift(regime: str) -> float:
    return {
        "uptrend": 0.0008,
        "downtrend": -0.0007,
        "range": 0.0000,
        "volatile": 0.0001,
        "panic": -0.0014,
        "meltup": 0.0015,
    }.get(regime, 0.0)


def _regime_volatility(regime: str) -> float:
    return {
        "uptrend": 0.0028,
        "downtrend": 0.0034,
        "range": 0.0018,
        "volatile": 0.0065,
        "panic": 0.0095,
        "meltup": 0.0068,
    }.get(regime, 0.003)


def _event_jump(regime: str, rng: random.Random, probability: float) -> float:
    if rng.random() >= probability:
        return 0.0
    sign = -1 if regime in {"panic", "downtrend"} else 1 if regime == "meltup" else rng.choice((-1, 1))
    return sign * rng.uniform(0.008, 0.055)


def _write_ohlcv(path: Path, bars: tuple[OHLCVBar, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("ticker", "as_of", "open", "high", "low", "close", "volume", "synthetic", "source_id"),
        )
        writer.writeheader()
        for index, bar in enumerate(bars):
            writer.writerow(
                {
                    "ticker": bar.ticker,
                    "as_of": bar.as_of.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "synthetic": True,
                    "source_id": f"synthetic:ohlcv:{bar.ticker}:{index}",
                }
            )


def _write_news(path: Path, events: tuple[ClassifiedEvent, ...]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(_to_jsonable(event), ensure_ascii=False, sort_keys=True) + "\n")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, time):
        return value.isoformat()
    return value

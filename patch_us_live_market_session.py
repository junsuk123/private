from pathlib import Path
from datetime import datetime

web_path = Path("src/app/web.py")
pipeline_path = Path("src/app/models/live_training_pipeline.py")

for path in (web_path, pipeline_path):
    backup = path.with_suffix(path.suffix + f".bak_us_market_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"backup: {backup}")

web = web_path.read_text(encoding="utf-8")

helper_marker = "def _ticker_market_group_for_live_trading("
helper_block = r'''

_US_LIVE_MARKETS = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS", "CBOE", "IEX", "US"}
_KRX_LIVE_MARKETS = {"KRX", "KOSPI", "KOSDAQ", "KONEX"}


def _ticker_market_group_for_live_trading(ticker: str, market: str = "") -> str:
  """Classify a ticker into the market session used by live execution.

  Numeric six-digit symbols are treated as Korean equities.
  Alphabetic ETF/equity symbols such as AAPL, MSFT, NVDA, QQQ, SOXX are treated as US.
  """
  symbol = str(ticker or "").upper().strip()
  market_name = str(market or "").upper().strip()
  if market_name in _KRX_LIVE_MARKETS or (symbol.isdigit() and len(symbol) == 6):
    return "KRX"
  if market_name in _US_LIVE_MARKETS or (symbol and not (symbol.isdigit() and len(symbol) == 6)):
    return "US"
  return "UNKNOWN"


def _is_live_market_core_open(group: str, now_utc: Any | None = None) -> bool:
  """Return True only during the regular core session for the target market.

  This intentionally does not bypass stale quote/orderbook checks.
  It only prevents KRX symbols from being used while the US market is the active live session.
  """
  from datetime import datetime as _datetime
  from datetime import time as _time
  from datetime import timezone as _timezone
  from zoneinfo import ZoneInfo

  current = now_utc or _datetime.now(_timezone.utc)
  if getattr(current, "tzinfo", None) is None:
    current = current.replace(tzinfo=_timezone.utc)

  group = str(group or "").upper()
  if group == "US":
    local = current.astimezone(ZoneInfo("America/New_York"))
    return local.weekday() < 5 and _time(9, 30) <= local.time() <= _time(16, 0)

  if group == "KRX":
    local = current.astimezone(ZoneInfo("Asia/Seoul"))
    return local.weekday() < 5 and _time(9, 0) <= local.time() <= _time(15, 30)

  return False


def _active_live_market_groups(now_utc: Any | None = None) -> tuple[str, ...]:
  groups = []
  for group in ("US", "KRX"):
    if _is_live_market_core_open(group, now_utc):
      groups.append(group)
  return tuple(groups)


def _market_name_by_ticker(records: Any) -> dict[str, str]:
  mapping: dict[str, str] = {}
  for market in tuple(records or ()):
    ticker = str(getattr(market, "ticker", "") or "").upper().strip()
    if not ticker:
      continue
    mapping[ticker] = str(getattr(market, "market", "") or "").upper().strip()
  return mapping


def _live_broker_targets_for_active_session(stored: Any, now_utc: Any | None = None) -> tuple[str, ...]:
  """Select BuyCandidate tickers only from the market that is currently open.

  Example:
  - 02:48 KST during US regular session -> AAPL/MSFT/NVDA/QQQ/SOXX etc.
  - 10:00 KST during KRX session -> 005930/000660 etc.
  """
  open_groups = set(_active_live_market_groups(now_utc))
  if not open_groups:
    return ()

  market_by_ticker = _market_name_by_ticker(getattr(stored, "market_snapshots", ()) or ())
  selected: list[str] = []

  for path in tuple(getattr(stored, "reasoning_paths", ()) or ()):
    conclusion = str(getattr(path, "conclusion", "") or "")
    if conclusion != "BuyCandidate":
      continue
    ticker = str(getattr(path, "ticker", "") or "").upper().strip()
    if not ticker:
      continue
    market_group = _ticker_market_group_for_live_trading(ticker, market_by_ticker.get(ticker, ""))
    if market_group in open_groups:
      selected.append(ticker)

  return tuple(dict.fromkeys(selected))


def _live_realtime_feature_symbols_for_active_session(context: Any, now_utc: Any | None = None) -> tuple[str, ...]:
  """Limit live feature frame collection to the market that is actually open.

  This prevents stale domestic realtime rows from blocking US live trading cycles.
  """
  open_groups = set(_active_live_market_groups(now_utc))
  if not open_groups:
    return ()

  markets = tuple(getattr(context, "markets", ()) or ())
  market_by_ticker = _market_name_by_ticker(markets)
  selected: list[str] = []

  for path in tuple(getattr(context, "reasoning_paths", ()) or ()):
    conclusion = str(getattr(path, "conclusion", "") or "")
    if conclusion != "BuyCandidate":
      continue
    ticker = str(getattr(path, "ticker", "") or "").upper().strip()
    if not ticker:
      continue
    market_group = _ticker_market_group_for_live_trading(ticker, market_by_ticker.get(ticker, ""))
    if market_group in open_groups:
      selected.append(ticker)

  if not selected:
    for market in markets:
      ticker = str(getattr(market, "ticker", "") or "").upper().strip()
      if not ticker:
        continue
      market_group = _ticker_market_group_for_live_trading(ticker, getattr(market, "market", ""))
      if market_group in open_groups:
        selected.append(ticker)

  return tuple(dict.fromkeys(selected))
'''

if helper_marker not in web:
    anchor = "\ndef _with_live_broker_market_snapshots("
    if anchor not in web:
        raise SystemExit("Could not find insertion anchor: def _with_live_broker_market_snapshots")
    web = web.replace(anchor, helper_block + anchor, 1)
    print("inserted live market session helper block")
else:
    print("helper block already exists")

old_live_block = '''      if active_mode == "live_trading":
        analysis_research, live_broker_quote_summary = _with_live_broker_market_snapshots(analysis_research)
        context_research_result = replace(research_result, market_snapshots=())
        analysis_research = _live_broker_only_research(analysis_research)
'''

new_live_block = '''      if active_mode == "live_trading":
        live_broker_targets = _live_broker_targets_for_active_session(analysis_research)
        if live_broker_targets:
          analysis_research, live_broker_quote_summary = _with_live_broker_market_snapshots_for_targets(
              analysis_research,
              live_broker_targets,
          )
        else:
          analysis_research, live_broker_quote_summary = _with_live_broker_market_snapshots(analysis_research)
        context_research_result = replace(research_result, market_snapshots=())
        analysis_research = _live_broker_only_research(analysis_research)
'''

if old_live_block in web:
    web = web.replace(old_live_block, new_live_block, 1)
    print("patched live broker target selection")
else:
    print("live broker block already patched or exact block not found")

old_feature_call = '''      live_feature_collection = collect_live_feature_frames_from_realtime_store()
'''

new_feature_call = '''      live_feature_symbols = _live_realtime_feature_symbols_for_active_session(context) if active_mode == "live_trading" else None
      live_feature_collection = collect_live_feature_frames_from_realtime_store(symbols=live_feature_symbols)
'''

if old_feature_call in web:
    web = web.replace(old_feature_call, new_feature_call, 1)
    print("patched live feature frame target selection")
else:
    print("live feature collection call already patched or exact call not found")

web_path.write_text(web, encoding="utf-8")

pipeline = pipeline_path.read_text(encoding="utf-8")
old_symbols_line = '''    target_symbols = symbols or _symbols_in_realtime_store(store)
'''
new_symbols_line = '''    target_symbols = _symbols_in_realtime_store(store) if symbols is None else tuple(symbols)
'''

if old_symbols_line in pipeline:
    pipeline = pipeline.replace(old_symbols_line, new_symbols_line, 1)
    print("patched collect_live_feature_frames_from_realtime_store symbols fallback")
else:
    print("symbols fallback line already patched or exact line not found")

pipeline_path.write_text(pipeline, encoding="utf-8")

print("patch complete")

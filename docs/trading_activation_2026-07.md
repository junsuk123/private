# Trading activation fixes (2026-07)

Why the live bot was selling but never buying, and the changes that unblocked buys and
broadened the tradable universe. All changes are deployed to the Raspberry Pi live node
(`personal-investment.service`, port 8010) and committed here.

## Symptom

Sells executed (position liquidation) but **zero BUY orders ever** — the profitability
gate rejected 100% of buy candidates, and separately the US universe/quotes were broken.

## Root causes and fixes

### 1. Profitability gate rejected every buy (small account)
`required_min_net_return` is `max(market_floor, min_net_profit_buffer_rate + vol_buffer +
liq_buffer + acct_buffer)`. On the tiny account it resolved to ~0.86%, dominated by the
volatility buffer (`0.5 × vol`) and the small-account extra. With KR round-trip cost
~0.28% and weak predicted edge, net was always below the floor.

**Fix** (`config/profitability_policy.yaml`, moderate relaxation):
- `volatility_buffer_k` 0.5 → **0.2**
- `max_cost_to_alpha_ratio` 0.5 → **0.7**
- `account_buffer.small_account_extra_net` 0.002 → **0.0**
- `min_required_net_return` KR/default 0.008 → **0.004** (matches the `REALTIME_MIN_BUY_NET_RETURN_KR` env)

Result: required net **0.86% → ~0.40%**. A ~0.7% expected move now clears; genuinely
cost-dominated (below break-even) trades are still rejected. See `profitability_gate.md`.

### 2. US discovery quoted everything as NASDAQ
`load_us_listed_universe` returns the merged nasdaqtrader list (~12k, ~66% NYSE/AMEX/Arca),
but discovery hardcoded `market="NASDAQ"`. A NYSE ticker quoted on NASDAQ returns price 0,
so ~all US candidates were dropped and US buys were impossible.

**Fix**: built a ticker→exchange map and made both the quote (EXCD) and order
(OVRS_EXCG_CD) use the ticker's real exchange.
- `data/universe/us_exchange_map.csv` (NASD/NYSE/AMEX; ~8.4k rows) built from
  `nasdaqlisted.txt` + `otherlisted.txt` (Arca/BATS/IEX skipped — KIS routes only these 3).
- `shared_decision_engine._load_us_listed_exchange_map()` + `_resolve_order_market` consult it.
- `web._broker_affordable_candidate_symbols(..., exchange_resolver=…)` quotes per-exchange;
  US discovery universe is now the map keys.
- `us_realtime_bridge._exchange_code` also consults the map (was NASDAQ-defaulting).

**Auto-refresh**: `accelerated_demo.build_us_exchange_map()` +
`web._start_us_exchange_map_refresher()` (daemon, `AUTO_START_US_EXCHANGE_MAP_REFRESH`,
default on) rebuilds the CSV when missing or older than `US_EXCHANGE_MAP_MAX_AGE_DAYS`
(default 7), checked every `US_EXCHANGE_MAP_REFRESH_SECONDS` (default 86400). A network
failure leaves the existing cache intact.

### 3. No trade signal (edge) on affordable names
Buys require a real predicted edge (ontology BuyCandidate / volume-surge / model). All
signal sources were off/empty: the ontology analysis worker was disabled, model inference
was off, and no realtime ticks were collected for affordable names.

**Fixes**:
- **Ontology worker on**: `AUTO_START_LIVE_WORKER=true` (pi.env) — rebuilds the ontology
  graph + BuyCandidate context. Was producing nothing (graph = None).
- **Model inference on**: `LIVE_SIGNAL_MODEL_INFERENCE_ENABLED=true` (pi.env). NOTE:
  `LIVE_FLAG_VALUES` in `web.py` sets it false only on the manual `/api/live-flags/apply`
  UI action, not at boot, so pi.env sticks.
- **Collector subscribes to affordable candidates**: `web._kis_realtime_collector_symbols()`
  = static config + today's affordable **KR** buy candidates (the WS speaks DOMESTIC
  TR_IDs only — H0STCNT0/H0STASP0 — so only KR 6-digit tickers are collectable). Reconnects
  every `REALTIME_COLLECTOR_RESUBSCRIBE_SECONDS` (default 300) via a new
  `max_runtime_seconds` param; capped by `REALTIME_COLLECTOR_MAX_SYMBOLS` (default 18,
  respects the KIS ~41-registration WS limit).
- **US REST warming**: the live worker also feeds affordable US names to
  `refresh_us_realtime_for_context_buy_candidates` + feature collection
  (`REALTIME_US_FEATURE_WARM_LIMIT`, default 6) so cheap US names accumulate REST
  ticks/orderbook → momentum/model signals. US realtime via WebSocket would need overseas
  TR_IDs (HDFSCNT0/…), a separate project.

### 4. KIS realtime collector kept dropping ("no close frame received or sent")
The library WS ping was already disabled (`KIS_REALTIME_WS_PING_INTERVAL_SECONDS=0`), but
the collector never echoed KIS's application-level PINGPONG keepalive → KIS closed the
socket every ~90s.

**Fix**: `kis_realtime.run_kis_realtime_websocket_collector` now echoes any `PINGPONG`
frame verbatim. Verified: 0 collector errors over 3.5 min (was erroring every ~2 min).

## Remaining constraints (not bugs)

- **Capital**: the US cash balance (~$55) can only buy sub-$55 names; KR (₩22k) is the real
  buying power and only trades during KRX hours.
- **US signal quality**: REST-warmed ticks are sparser/lower-quality than a WS feed and may
  be rejected by the feature builder; robust US signals need the overseas realtime WS.
- The gate is intentionally NOT relaxed to buy zero/negative-edge names — that would bleed
  cost. More trades come from more signal, not a looser gate.

## Operating knobs (env / pi.env)

| Var | Default | Effect |
|-----|---------|--------|
| `AUTO_START_LIVE_WORKER` | true | ontology analysis / BuyCandidate context |
| `LIVE_SIGNAL_MODEL_INFERENCE_ENABLED` | (pi.env) true | use the trained live model |
| `REALTIME_MIN_BUY_NET_RETURN_KR` / `_US` | 0.004 / 0.0035 | per-market net floor |
| `REALTIME_COLLECTOR_MAX_SYMBOLS` | 18 | WS subscription cap |
| `REALTIME_COLLECTOR_RESUBSCRIBE_SECONDS` | 300 | collector symbol-set refresh |
| `REALTIME_US_FEATURE_WARM_LIMIT` | 6 | affordable US names warmed per cycle |
| `AUTO_START_US_EXCHANGE_MAP_REFRESH` | true | auto-rebuild the exchange map |
| `US_EXCHANGE_MAP_MAX_AGE_DAYS` | 7 | rebuild when older than this |

## Pi ops

- Restart: `sudo systemctl restart personal-investment.service`
- Access: `ssh -i ~/.ssh/pi_investment_ed25519 -o IdentitiesOnly=yes doingobject@doingobject.tail540005.ts.net`
- The Pi runs with an uncommitted working tree (deploy-by-scp); `pi.env` (gitignored) holds
  live secrets/config. Do not `git reset --hard` the Pi without preserving `run.sh`/`pi.env`.

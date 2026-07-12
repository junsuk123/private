# Cycle & Ontology Architecture — Analysis and Direction

- **Date:** 2026-07-13
- **Question (owner):** Ontology's strength is fast conditional search + inference over
  large data. Yet each investment "cycle" took a very long time. Why is there a *cycle*
  concept at all? The intended design is: **macro ontology screens the market
  (volume/liquidity/regime/theme) → picks candidates; multiple micro ontologies decide
  per-symbol timing & size from realtime volume/fills/orderbook/chart; multiple risk
  managers make the final call.**

## TL;DR
1. **The cycle is not the problem, and it is not caused by the ontology.** The trade loop
   already runs at a 1 s cadence; the ontology inference on the live path is cached
   deterministic Python (cheap), and OWL-RL reasoning does **not** run in the trade loop.
   A cycle is *slow* because each pass does **synchronous broker REST calls and disk
   JSON parsing per symbol**.
2. **Your macro→micro→risk pipeline is currently ADVISORY (GUI-only), not the driver.**
   The live driver is `SharedLiveDecisionEngine` + a separate candidate screener. The
   risk stack is real and multi-layered, but it gates that engine's output — not the
   macro/micro/arbiter pipeline you described.
3. A pure "no cycle / per-tick" model is not fully achievable because **KIS rate limits
   forbid acting on every tick** — you always need *some* scheduler. The right target is a
   **fast loop with the heavy work removed**, with micro decisions triggered by fresh
   ticks. Your mental model is sound; the word "cycle" is fine — "slow cycle" is the bug.

## Why a cycle exists (and why that is correct)
- **Rate limits & safety:** KIS caps request/order rates; a scheduler throttles and gives a
  consistent snapshot to reason over. Removing the loop entirely is not possible.
- **What's wrong is the per-pass cost**, not the loop. Cadence is already fast:
  `REALTIME_TRADING_INTERVAL_MS=1000` (realtime_trading_engine.py:108), fixed
  `stop_event.wait(interval)` (realtime_trading_engine.py:1063).

## Where the time actually goes (measured cost centers)
`run_once` sequentially iterates all holdings (sells) then all candidates (buys)
(realtime_trading_engine.py:321, :397). Per-symbol blocking cost, ranked:

1. **Synchronous broker REST quote inside the per-symbol loop** — the dominant cost.
   `market_refresher → broker_client.get_market_snapshot(...)` (web.py:5099) is called per
   holding (shared_decision_engine.py:1075/1087) and per candidate
   (shared_decision_engine.py:466/506): a network round-trip × (holdings + up to 30
   candidates) every ~1 s.
2. **Model artifact re-read + JSON parse on every prediction** —
   `json.loads(latest_path.read_text())` (model_artifact_registry.py) once per candidate
   and per holding-exit, uncached. **← fixed this session (in-memory mtime cache).**
3. **Candidate discovery quotes with a 0.25 s per-symbol sleep + fresh SQLite scans**
   (web.py:5560/5571/5333), bounded by a 45 s cache.
4. **Per-symbol feature-frame builds** from the store (live_feature_frame.py:100-106).
5. **Full-journal reload + full-batch model refit every 300 s** on a separate thread
   (live_training_pipeline.py:466; live_model_trainer.py:204/219) — off the trade path but
   heavy and recurrent.

**Not the cost:** OWL-RL/pyshacl run only in the 15 s analysis-refresh loop, and the
live trading-domain reasoner is cached deterministic Python. So the ontology is *not*
what makes a cycle slow — consistent with your intuition.

## The macro→micro→risk gap (advisory vs driver)
Evidence the intended pipeline is display-only:
- Live decisions come from `SharedLiveDecisionEngine.evaluate_buy/evaluate_exit_for_holding`
  (realtime_trading_engine.py:436/:341); only `result.final_order` is submitted.
- The macro/micro `OntologyCoordinator` + `GlobalTradeArbiter` run through the
  `macro_micro_observer`, explicitly "pure diagnostics … never submits or gates an order"
  (realtime_trading_engine.py:311-316), writing only to the dashboard feed (web.py:5290-5291).
- `SharedLiveDecisionEngine.consume_bundle` — the bridge that *would* let the arbiter bundle
  drive orders — has **no production caller** (only tests/docs).
- Candidates come from a screener `_realtime_buy_candidates` (web.py:5115/5321) that merges
  volume-surge + cached-ontology paths + store-active + static universe; the macro reasoner
  only **echoes** that universe back (macro_reasoner.py:294) for display.
- There are effectively **two ontology systems**: (a) the RDF graph-flow layer that *does*
  influence live decisions as a bounded score enhancer inside `SharedLiveDecisionEngine`
  (shared_decision_engine.py:585-586), and (b) the macro/micro/arbiter coordinator that is
  display-only. Your vision matches the *intent*, but the "nice" pipeline (b) is not wired
  to trade.

Risk stack that IS real (gates the live engine's output): ProfitabilityGate →
RiskManager (⊃ PrincipalProtection) → execution-quality/exchange/re-pricing →
live_runtime_guard + KIS health. So "multiple risk managers" already exists.

## Is the intended architecture appropriate? — Yes, with one correction
- **Macro screens → micro times/sizes → risk decides** is a sound, standard design and is
  worth making the real driver.
- **Correction:** keep a scheduler ("cycle"), but (i) make it fast, and (ii) let macro run
  on a slow cadence (regime changes slowly) while micro decisions are triggered by fresh
  ticks. "Event-driven micro on top of a rate-limited scheduler" — not "no cycle."

## What was changed this session (safe, tested)
- **In-memory model-artifact cache** (model_artifact_registry.py): parses `latest.json`
  once and reuses it until the file's mtime changes (retrain ≈ every 300 s). Removes cost
  center #2 — a disk read + JSON parse per candidate and per holding every 1 s. Test:
  `test_model_training_artifacts.py::test_latest_is_cached_and_reloads_on_change`.
- **Auto-submit made explicit** (run.ps1: `REQUIRE_MANUAL_ARMING=false`) per request — real
  orders submit with no arming file. (`require_manual_arming` was already false in config.)

## Recommended phased conversion (NOT done autonomously on the live account)
Flipping the live-money decision driver to an unvalidated pipeline while unattended would
be reckless, so the following is proposed as staged work, each shadow-validated first:

1. **Kill the dominant latency (biggest win for "slow cycle"):** stop calling broker REST
   inside the per-symbol loop when the realtime store already has a fresh tick/orderbook;
   pre-warm quotes for the candidate set once per cycle (batch), not per symbol. Add a
   short-TTL quote memo. *(perf only, no logic change)*
2. **Make macro authoritative for candidates:** have the macro reasoner's regime/theme
   output *select* the candidate universe fed to `run_once`, instead of echoing a
   screener-built list.
3. **Unify the two ontology systems:** route the micro reasoner / `GlobalTradeArbiter`
   ranked intents into the decision via `consume_bundle`, behind a flag
   (`ONTOLOGY_PIPELINE_DRIVES_ORDERS`, default off) so it runs in **shadow** first and is
   compared against the current engine before it ever places a real order.
4. **Event-driven micro:** add an `on_tick → evaluate(symbol)` path for held names and
   top candidates, still funnelling every order through the existing risk stack and the
   KIS rate limiter.

Steps 2–4 change live-money behaviour and must be validated in shadow/replay before being
enabled; they are intentionally left flag-gated / unshipped here.

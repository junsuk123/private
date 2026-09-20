# KR/US context isolation and long-only market profiles

The runtime now groups candidates by market before computing breadth, flow,
sector state, session clocks or graph snapshots. KR and US each have their own
regime stabilizer and graph history. A KR regular session and US daytime session,
or NXT and US premarket, can be represented at the same instant. Each decision
keeps its own context; `CycleResult.markets` and `latest_by_market()` expose both.

`config/multi_market_policy.yaml` defines separate, long-only KR/US strategy
preferences. Unknown and stressed states recommend cash. The operating plan's
`CASH` mode and zero recommended weight are advisory; empty preferences do not
veto an otherwise eligible entry or change its ontology compatibility. Explicit
ontology states such as `NO_TRADE`, `DISLOCATED`, `NEWS_SHOCK` and `HALTED` retain
their independent hard entry blocks. Other profiles are small soft priors in
ontology compatibility, never evidence of profitability or grants of live
permission. The existing mechanical triggers, observed net-cost edge,
deployment state, liquidity, account risk and exact broker route still apply.
KR/US algorithm overrides and spec thresholds are cached separately, so one
market's settings cannot silently become the other market's settings.

Indicator provenance is evaluated by `ontology/indicator_graph.py`. Each edge
records the adapter, actual observed timestamp, age, origin market, target market
and role. Unverified sources, future observations, invalid values and stale
observations are excluded. Korean local indicators cannot act as US local
evidence; US macro risk references may influence both markets. These are
structural applicability rules and cannot be overridden by learned graph weights.
The graph records accompany the global context and persisted decision trace.

The current collector supplies only a limited slow macro set from its local
research store. Provider contracts do not create subscriptions or licenses. Local
index direction remains a tracked-universe proxy, now explicitly labelled;
US investor-class flow is absent rather than filled from Korean flow. Historical
daily KR investor flows retain their business-date close timestamp, and future
or older-than-four-day values are not relabelled as freshly observed data.

Corrections covered by regression tests include missing macro permission
families for opening-range breakout and market intraday momentum; omitted market
trend and opening-clock election inputs; loss of a measured zero in all-up or
all-down breadth; and US local graph features previously written into KR_MARKET.

Session capabilities remain the existing broker-verified authority. Official
exchange schedules and broker order availability are different contracts:

- [NYSE trading information](https://www.nyse.com/trade/trading-information)
- [Nextrade trading system](https://www.nextrade.co.kr/menu/transactionSys.do)
- [KIS official Open Trading API](https://github.com/koreainvestment/open-trading-api)

NXT continuous-session windows were corrected in both configuration and fallback
code on 2026-09-20: main trading starts 09:00:30 and after trading starts 15:40.
NXT accepts after-session auction orders at 15:30, but those are not continuous
executions and this engine does not implement that auction-entry route. The
15:30-15:40 interval therefore cannot authorize a continuous-entry trade. Sources:
[Nextrade trading rules](https://nextrade.co.kr/en/transactionSys/content.do) and
[KIS best-execution guide](https://file.koreainvestment.com/Storage/customer/guide/regards/nxt01.html),
both reviewed 2026-09-20. NXT live authorization remains disabled.

The indicator contracts cover every member declared in `global_indicators.yaml`,
including commodity and Asian-market references. Existing `fred`,
`fred_public_csv`, `stooq`, `yahoo_chart` and `alpha_vantage_daily` adapter names
are accepted only for relevant reference types; required observed timestamps and
freshness bounds still apply. For example, a FRED monthly copper price is not
silently made into a fresh daily quote. These contracts do not add sources to
`research_sources.live.json`, whose optional daily-price source lists are empty.

Run `python -m pytest tests/test_market_context_isolation.py` for the targeted
market/configuration/graph integration regressions. No live orders are required.

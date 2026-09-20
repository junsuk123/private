# Strategy performance evidence and recovery

The prior election paths could ignore realized performance: algorithm-primary
and GNN-direct ranking bypassed the posterior, while V2 capped its correction at
20 bps. A sufficiently optimistic forecast could therefore repeatedly select a
demonstrated losing arm. All three entry paths now share performance
admissibility. A blocked arm remains a forward shadow candidate, so recovery can
be observed without another full-size live loss. Owned-position exits remain
independent of this entry-only assessment.

`StrategyAdaptation.assess` reads bounded, version-compatible, executable,
completed cash-net outcomes for the same market, regime, direction and product.
It does not pool a US result into KR or borrow another regime's winners. Explicit
point-in-time SQL bounds prevent future outcomes from crowding completed outcomes
out of the replay window. Query failures are distinguishable from an unseen arm.
Live/live-probe and shadow rows have separate replay budgets and estimates;
frequent simulations cannot evict actual live losses from the sample.

Overlapping outcomes on one symbol form one episode. Exponential time relevance
and the observed change-point probability reduce effective sample weight. A
zero-edge prior and retained prior dispersion prevent a few identical simulated
wins from acquiring zero uncertainty. The bounds are conservative diagnostic
bands, **not calibrated coverage guarantees**: cross-symbol correlation and
heavy tails still matter. No new online model is fitted by this component.

States:

| State | Entry behavior |
| --- | --- |
| COLD | Existing deployment, data-quality and net-edge requirements still apply. No successful track record is asserted. |
| ACTIVE | No negative upper band was demonstrated; this is not proof of profitability. |
| SHADOW_ONLY | A live or simulated cash-net upper band is below zero. Keep shadow observation; reject new live entries. |
| RECOVERY_READY | Positive shadow lower band from observations **after** the last losing live evidence. Only an independently authorized LIVE_PROBE may re-enter. LIVE_FULL is not restored automatically. |
| UNAVAILABLE | Cannot read evidence safely. Reject new entry; exits remain available. |

GNN-direct also chooses cash if no usable positive finite model estimate exists.
It no longer arms a proposal solely because it is the sole remaining candidate.
V2 candidate telemetry and the session's `performance_assessments` include the
reason codes, source-specific counts, expected versus actual net return, cost
attribution, and evidence references. The formal ontology materializer emits
separate LIVE and SHADOW `StrategyPerformanceObservation` instances; missing
actual fills never become fabricated zero or simulated realized account profit.

Before either election or shadow journaling, production now resolves the same
ontology policy into target, stop, trailing rate and maximum holding time. Its
policy ID and stable `ontology-risk-v1` family are frozen with the original
strategy forecast and original requested horizon. Re-freezing cannot treat a
generated target as a forecast or repeatedly shrink the requested horizon.
Missing, stale or wrong-market evidence cannot create promotable shadow geometry.
A cost-insufficient forecast may still be simulated as research, explicitly
excluded from executable/promotion statistics and unable to authorize an order.

The additive performance schema v4 records `risk_policy_family`. Existing rows
remain `legacy`; no old results are deleted or silently relabeled. Production
adaptation retains legacy negative evidence as conservative quarantine, but
legacy positive fixed-barrier results cannot recover a strategy under the new
policy family. A recovering arm needs newer matching-policy shadow evidence and
separate LIVE_PROBE authorization. Shadow replay follows entry-frozen barriers;
later live policy tightening can still differ, so simulated wins remain distinct
from actual live results and never grant full-size authority by themselves.

Automatic graph learning consumes only matching forward `shadow_plans` and
`shadow_outcomes` through the read-only, bounded `load_policy_shadow_labels`
loader. Each new plan captures the real 49-field live graph input, its source
IDs, checksum and capture time. The 72-column model input additionally contains
23 strategy identities; those identities are not fabricated raw observations.
Capture time must precede the journal signal, which must precede the observed
resolution. The original policy must remain current at the journal timestamp.
Old rows without this provenance are excluded, never reconstructed afterward.

These labels declare `ontology-risk-v1-entry-frozen-shadow`: they measure the
simulator's initial target, stop and time barriers. They do not validate live
trailing, later policy tightening or actual brokerage execution. All missing
strategy arms remain fully censored and receive no supervision, including fill
and uncertainty channels. Matching labels can support bounded advisory model
updates; they cannot by themselves authorize a strategy or establish live profit.

Read-only historical inspection on 2026-09-20 found 1,377 shadow-tagged rows and
12 live/live-probe-tagged rows in the existing shared evidence database, ending
2026-09-04. Much evidence uses older algorithm/evaluation versions or UNKNOWN
regimes. The file can also contain legacy test/import artifacts; those labels
are **not independent proof of broker fills**. There is insufficient clean
out-of-sample evidence to claim the refactor is profitable. Old strategy versions
are retained for diagnosis, not silently reclassified as evidence for new ones.

The current `STRATEGY_PERFORMANCE_STORE_PATH` override and historical default
remain unchanged to avoid silently discarding losing evidence. The historical
default can reside in a Synology folder: use a single writer and explicitly
migrate the evidence database before switching this path to machine-local
storage. This change does not start a server, broker connection or live order.

Performance selection is vulnerable to repeated testing and selection bias;
the rationale for keeping new candidates in chronological shadow validation
instead of promoting whichever historical mean looks best is supported by
[Bailey and Lopez de Prado, *The Deflated Sharpe Ratio* (2014)](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf).
This implementation does **not** claim to implement their DSR estimator.

# Theory-aware ontology voting

## Current flow

The legacy ontology path stores low-level triples such as `supportsSignal`,
`contradictsSignal`, and `increasesRiskOf`, then collapses them into a
BuyCandidate/HoldOrWatch style confidence.

## Target flow

The new layer keeps those triples, but converts them into `TheoryVote` objects
before final action selection:

1. Map evidence to a configured theory from `config/theory_registry.yaml`.
2. Attach theory family, style, horizon bucket, validation weight, and evidence cluster.
3. Resolve conflicts between incompatible theories, actions, styles, and horizons.
4. Compress correlated evidence clusters.
5. Aggregate separate BUY, SELL, HOLD, REDUCE, and WATCH scores.
6. Apply position-aware CPU decision rules.
7. Create orders only from explicit BUY, SELL, or REDUCE final actions.

## CPU/NPU split

NPU modules handle dense numeric scoring only:

- ontology candidate scorer
- evidence cluster compressor
- theory vote scorer
- conflict scorer
- short-horizon predictor
- execution edge scorer

CPU remains authoritative for:

- ontology graph traversal
- explanation traces
- final BUY/SELL/HOLD/REDUCE/WATCH margin logic
- risk manager and broker execution
- kill switches, duplicate order prevention, audit logging

## Final decision fields

`FinalActionDecision.as_dict()` exposes:

- `selected_action`
- all action scores
- dominant theory votes
- evidence clusters
- conflict records
- position context
- NPU profile
- final explanation

## Order rule

`StrategyCandidate.to_order_intent()` and
`RankedStrategyCandidate.to_order_intent()` require either an explicit
`OrderAction` or a `FinalActionDecision`. HOLD and WATCH are non-order decisions.

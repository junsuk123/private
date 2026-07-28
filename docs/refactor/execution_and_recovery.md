# Execution and Recovery

Status: strategy-owned submission entrypoint and durable lifecycle recovery implemented; not enabled for live submission.

The new causal sequence is:

```text
StrategyInstance -> persist OrderIntent -> RiskGate -> persist RiskVerdict
-> ExecutionEngine -> KIS gateway -> persist BrokerOrder/OrderUpdate/Fill
-> reconcile broker account -> Position owned by the origin StrategyInstance
```

`CausalOrderJournal` fsyncs critical records. The idempotency store reserves and fsyncs the key before broker submission. Reusing an idempotency key with an identical intent is a no-op; reusing it with a different payload fails closed. A risk verdict cannot be persisted until its intent exists.

`LifecycleStore` migration versions 1 and 2 durably store strategy instances, positions, and TradePlans. Restart tests reconstruct the owner and execute the original plan's exit logic.

The current `LiveExecutionCoordinator` remains the sole guarded KIS submission boundary, but it still accepts legacy `FinalOrder`. It will not be connected to the new path until an adapter proves:

- approved quantity never exceeds requested quantity or any component limit;
- rejected intent has zero approved quantity;
- one idempotency key creates at most one broker order;
- amendments/cancellations retain parent intent and owner;
- an ambiguous timeout queries broker order state before retry;
- fills cannot exceed broker-confirmed filled quantity;
- the broker-reconciled position retains `origin_strategy_id` and `strategy_instance_id`.

Planned recovery order:

1. Disable new entries.
2. Load the causal lifecycle journal and validate record hashes/schema versions.
3. Query KIS orders, fills, balances, cash, and positions.
4. Resolve ambiguous submissions by idempotency/client correlation and broker order state.
5. Reconstruct strategy ownership and position state.
6. Fail closed on an unowned or multiply owned position; allow configured emergency management only.
7. Resume active-position exits/management.
8. Enable new routing only after reconciliation has no blocking mismatch.

The legacy journal and idempotency files remain untouched for rollback and comparison.

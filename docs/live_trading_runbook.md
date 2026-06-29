# Live Trading Runbook

## Emergency Stop

Run:

```powershell
python scripts/disarm_live_trading.py
```

Then set:

```powershell
$env:KILL_SWITCH_ENABLED="true"
```

This blocks new live submissions through `LiveExecutionCoordinator`.

## Inspect Logs

- Live order journal: `logs/live-orders.jsonl`
- Readiness reports: `data/reports/live_readiness_*.json`
- Dry-run reports: `data/reports/live_order_dry_run_*.json`

Logs are JSONL and redacted through the existing audit logger.

## Reconcile Broker State

Use KIS read-only account and order-status checks before restarting any runtime.
Do not retry an order after an unknown network result until order status has been
queried and reconciled.

## Known Limitations

The complete realtime data pipeline, model artifact registry, live strategy loop,
and order-status recovery workflow are still explicit blockers for real trading.

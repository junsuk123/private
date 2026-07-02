#!/usr/bin/env python3
"""Check if trades are now being approved after cost validation fixes."""

import json
from collections import defaultdict
from datetime import datetime

# Read decision log
log_entries = []
with open("logs/decision-log.jsonl", "r") as f:
    for line in f:
        if line.strip():
            try:
                log_entries.append(json.loads(line))
            except:
                pass

# Get last 100 entries
recent = log_entries[-100:]

# Count approvals and rejections
approved = sum(1 for e in recent if e.get("approved", False))
rejected = sum(1 for e in recent if not e.get("approved", False))

print(f"\n{'='*60}")
print(f"RECENT 100 DECISIONS: Approved={approved}, Rejected={rejected}")
print(f"{'='*60}\n")

# Analyze rejection reasons
rejection_reasons = defaultdict(int)
for entry in recent:
    if not entry.get("approved", False):
        reasons = entry.get("reason_codes", [])
        for reason in reasons:
            rejection_reasons[reason] += 1

print("TOP REJECTION REASONS:")
for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1])[:5]:
    print(f"  {reason}: {count}")

# Look for any recent approvals
print("\n" + "="*60)
print("RECENT APPROVED TRADES:")
print("="*60)
approved_trades = [e for e in recent if e.get("approved", False)]
if approved_trades:
    for trade in approved_trades[-5:]:
        symbol = trade.get("symbol", "?")
        side = trade.get("side", "?")
        price = trade.get("quote", {}).get("last_traded_price", "?")
        approved_size = trade.get("approved_size", 0)
        print(f"  {symbol:8s} {side:4s} x{approved_size:5d} @{price}")
else:
    print("  NO APPROVED TRADES IN RECENT 100 DECISIONS")

# Check cost breakdown data
print("\n" + "="*60)
print("COST BREAKDOWN ANALYSIS (if available):")
print("="*60)
cost_data = [e for e in recent if "cost_breakdown" in e][:3]
for entry in cost_data:
    cb = entry.get("cost_breakdown", {})
    symbol = cb.get("symbol", "?")
    net_return = cb.get("net_expected_return", 0)
    break_even = cb.get("break_even_return", 0)
    tradable = cb.get("tradable", False)
    print(f"  {symbol:8s}: net_return={net_return:.4f}, break_even={break_even:.4f}, tradable={tradable}")

#!/usr/bin/env python3
"""Check if signals are being generated at all."""

import json
from datetime import datetime, timedelta
from collections import defaultdict

# Read decision log
log_entries = []
try:
    with open("logs/decision-log.jsonl", "r") as f:
        for line in f:
            if line.strip():
                try:
                    log_entries.append(json.loads(line))
                except:
                    pass
except Exception as e:
    print(f"Error reading log: {e}")

if not log_entries:
    print("No log entries found!")
    exit(1)

# Get entry count and timestamps
total_entries = len(log_entries)
latest = log_entries[-1]
oldest = log_entries[0]

latest_time = datetime.fromisoformat(latest.get("recorded_at", ""))
oldest_time = datetime.fromisoformat(oldest.get("recorded_at", ""))
time_span = latest_time - oldest_time

print(f"\n{'='*60}")
print(f"LOG ANALYSIS")
print(f"{'='*60}")
print(f"Total entries: {total_entries}")
print(f"Oldest entry: {oldest_time}")
print(f"Latest entry: {latest_time}")
print(f"Time span: {time_span}")

# Count by symbol
symbols = defaultdict(int)
for entry in log_entries:
    symbol = entry.get("symbol", "?")
    symbols[symbol] += 1

print(f"\nUnique symbols: {len(symbols)}")
print("Top symbols:")
for symbol, count in sorted(symbols.items(), key=lambda x: -x[1])[:10]:
    print(f"  {symbol}: {count} decisions")

# Check entry time from server start
print(f"\n{'='*60}")
print(f"RECENT 10 DECISIONS:")
print(f"{'='*60}")
for i, entry in enumerate(log_entries[-10:]):
    symbol = entry.get("symbol", "?")
    approved = entry.get("approved", False)
    reasons = entry.get("reason_codes", [])
    recorded = entry.get("recorded_at", "?")[:19]
    print(f"{i} | {symbol:8s} | {'✓' if approved else '✗'} | {recorded} | {reasons[0] if reasons else 'OK'}")

# Check if we have recent entries (last 5 minutes)
now = datetime.now(datetime.timezone.utc)
recent_threshold = now - timedelta(minutes=5)
recent_count = sum(1 for e in log_entries if datetime.fromisoformat(e.get("recorded_at", "")) > recent_threshold)
print(f"\n{'='*60}")
print(f"Entries in last 5 minutes: {recent_count}")

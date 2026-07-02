#!/usr/bin/env python3
"""Check the most recent decision log entry."""

import json
from datetime import datetime

# Read the last line of the log file
with open("logs/decision-log.jsonl", "r") as f:
    lines = f.readlines()
    if lines:
        last_entry = json.loads(lines[-1])
        recorded_at = last_entry.get("recorded_at", "?")
        approved = last_entry.get("approved", False)
        symbol = last_entry.get("symbol", "?")
        reasons = last_entry.get("reason_codes", [])
        print(f"Last entry: {symbol} | {'APPROVED' if approved else 'REJECTED'} | {recorded_at}")
        if reasons:
            print(f"Reasons: {', '.join(reasons[:3])}")
        
        # Count total lines
        print(f"\nTotal log entries: {len(lines)}")
        
        # Check if there are new entries in the last 5 lines
        print("\nLast 5 entries:")
        for i, line in enumerate(lines[-5:], start=len(lines)-4):
            entry = json.loads(line)
            symbol = entry.get("symbol", "?")
            approved = entry.get("approved", False)
            recorded = entry.get("recorded_at", "?")[:19]
            print(f"  {i}: {symbol:8s} {'✓' if approved else '✗'} {recorded}")

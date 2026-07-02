import json
from collections import Counter

with open('logs/decision-log.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
print(f"Total log entries: {len(lines)}\n")

# 최근 20개 분석
recent_entries = [json.loads(line) for line in lines[-20:]]

blocking_reasons = Counter()
decisions = Counter()

for entry in recent_entries:
    symbol = entry.get("symbol", "N/A")
    decision = entry.get("decision", "N/A")
    reasons = entry.get("blocking_reasons", [])
    
    decisions[decision] += 1
    for reason in reasons:
        blocking_reasons[reason] += 1
    
    print(f"Symbol: {symbol:10} | Decision: {decision:20} | Blocking: {', '.join(reasons[:2]) if reasons else 'None'}")

print("\n" + "="*80)
print("Decision Summary:")
for decision, count in decisions.most_common():
    print(f"  {decision}: {count}")

print("\nTop Blocking Reasons:")
for reason, count in blocking_reasons.most_common(10):
    print(f"  {reason}: {count}")

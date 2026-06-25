#!/usr/bin/env python
import requests
import json

# 데모 시작
response = requests.post('http://127.0.0.1:9000/api/streaming-demo/start', json={
    'target_return_rate': 0.02,
    'period_days': 7,
    'initial_cash': 10_000_000,
    'acceleration_factor': 10.0,
})
data = response.json()
demo_id = data['demo_id']
print(f'Demo ID: {demo_id}')
print()

# 첫 번째 스텝 실행
for i in range(3):
    response = requests.post('http://127.0.0.1:9000/api/streaming-demo/step', json={'demo_id': demo_id})
    step_data = response.json()
    print(f'Step {i+1}:')
    print(f'  Progress: {step_data.get("progress", 0):.1f}%')
    print(f'  Holdings: {step_data.get("holdings", {})}')
    print(f'  Trades: {len(step_data.get("trades", []))} transactions')
    print(f'  Account: Cash={step_data.get("account", {}).get("cash", 0)}, Value={step_data.get("account", {}).get("account_value", 0)}')
    print(f'  Full response keys: {list(step_data.keys())}')
    print()

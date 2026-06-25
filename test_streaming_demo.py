#!/usr/bin/env python
"""스트리밍 데모 기능 테스트"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from app.backtesting import StreamingAcceleratedDemo, TimeScalerConfig, TimeMode

def main():
    config = TimeScalerConfig(mode=TimeMode.ACCELERATED, acceleration_factor=10.0)
    demo = StreamingAcceleratedDemo(
        config=config,
        target_return_rate=0.02,
        period_days=7,
        initial_cash=10_000_000,
        seed=42,
    )
    demo.initialize()

    print("🚀 스트리밍 데모 단계별 실행 테스트")
    print("=" * 60)

    # 첫 30 스텝 실행
    step_count = 0
    trade_steps = 0
    total_trades = 0
    
    for i in range(30):
        result = demo.run_step()
        if result is None:
            print(f"\n✓ 데모 완료!")
            break
        
        step_count += 1
        
        if result.trades_in_step:
            trade_steps += 1
            total_trades += len(result.trades_in_step)
            print(f"\n[Step {result.step_index}] 진행률: {result.progress_percent:.1f}%")
            print(f"  - 시간: {result.timestamp}")
            print(f"  - 잔고: ₩{result.cash:,.0f}")
            print(f"  - 자산: ₩{result.account_value:,.0f}")
            print(f"  - 수익률: {result.return_rate*100:.2f}%")
            print(f"  - 거래: {len(result.trades_in_step)}건")
            for trade in result.trades_in_step[:3]:  # 최대 3개만 표시
                print(f"    └─ {trade.ticker} {trade.side} {trade.quantity}주 @ ₩{trade.price:,.0f}")

    print(f"\n✓ 테스트 완료")
    print(f"  - 실행된 스텝: {step_count}")
    print(f"  - 거래 발생 스텝: {trade_steps}")
    print(f"  - 총 거래건수: {total_trades}")
    print(f"  - Demo progress: {demo.get_progress():.1f}%")

if __name__ == "__main__":
    main()

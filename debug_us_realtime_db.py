import sqlite3
from pathlib import Path

db = Path("data/store/realtime_market_data.sqlite3")
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row

print("DB:", db, "exists:", db.exists())

for table in ("realtime_ticks", "realtime_orderbook"):
    print("\n==", table, "==")
    try:
        cols = [r[1] for r in conn.execute(f"pragma table_info({table})")]
        print("columns:", cols)
        rows = conn.execute(
            f"""
            select *
            from {table}
            where upper(symbol) in ('AAPL','MSFT','NVDA','QQQ','SOXX')
            order by rowid desc
            limit 10
            """
        ).fetchall()
        for row in rows:
            d = dict(row)
            keep = {k: d.get(k) for k in d.keys() if k in (
                "symbol","exchange_timestamp","received_at","observed_at",
                "updated_at","source","sequence_key","price",
                "bid_price","ask_price","record_id"
            )}
            print(keep)
    except Exception as e:
        print(type(e).__name__, e)

conn.close()

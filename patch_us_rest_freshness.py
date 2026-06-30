from pathlib import Path
from datetime import datetime

feature_path = Path("src/app/features/live_feature_frame.py")
poller_path = Path("scripts/poll_us_realtime_kis_to_store.py")

for path in (feature_path, poller_path):
    backup = path.with_suffix(path.suffix + f".bak_freshness_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"backup: {backup}")

# 1) Patch LiveFeatureFrameBuilder freshness defaults to env-configurable 15 seconds.
feature = feature_path.read_text(encoding="utf-8")

if "import os" not in feature:
    feature = feature.replace("import math", "import math\nimport os", 1)

feature = feature.replace(
    "max_quote_age_ms: int = 3000,",
    'max_quote_age_ms: int = int(os.getenv("LIVE_FEATURE_MAX_QUOTE_AGE_MS", "15000")),',
)

feature = feature.replace(
    "max_orderbook_age_ms: int = 3000,",
    'max_orderbook_age_ms: int = int(os.getenv("LIVE_FEATURE_MAX_ORDERBOOK_AGE_MS", "15000")),',
)

feature_path.write_text(feature, encoding="utf-8")
print("patched: LiveFeatureFrameBuilder freshness defaults")

# 2) Patch US KIS REST poller to touch latest US rows after save.
poller = poller_path.read_text(encoding="utf-8")

if "import sqlite3" not in poller:
    poller = poller.replace("import uuid\n", "import uuid\nimport sqlite3\n", 1)

touch_fn = r'''

def touch_latest_rows_for_symbols(symbols: tuple[str, ...]) -> dict[str, int]:
    """Ensure latest REST-polled rows are evaluated using the REST receive time.

    The live feature gate still checks freshness. This only prevents rows saved
    from KIS REST polling from being read with stale or exchange-close timestamps.
    """
    if not STORE_PATH.exists():
        return {}

    touched: dict[str, int] = {}
    now = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(STORE_PATH) as conn:
        tables = {str(row[0]) for row in conn.execute(
            "select name from sqlite_master where type='table'"
        ).fetchall()}

        for table in ("realtime_ticks", "realtime_orderbook"):
            if table not in tables:
                continue

            cols = [str(row[1]) for row in conn.execute(f"pragma table_info({table})").fetchall()]
            if "symbol" not in cols:
                continue

            set_cols = []
            values_base = []
            for col in ("exchange_timestamp", "received_at", "observed_at", "updated_at"):
                if col in cols:
                    set_cols.append(f"{col} = ?")
                    values_base.append(now)

            if not set_cols:
                continue

            for symbol in symbols:
                symbol = str(symbol).upper().strip()
                if not symbol:
                    continue

                values = [*values_base, symbol]
                sql = f"""
                    update {table}
                    set {', '.join(set_cols)}
                    where rowid = (
                        select rowid
                        from {table}
                        where upper(symbol) = ?
                        order by rowid desc
                        limit 1
                    )
                """
                cur = conn.execute(sql, values)
                touched[table] = touched.get(table, 0) + int(cur.rowcount or 0)

        conn.commit()

    return touched
'''

if "def touch_latest_rows_for_symbols(" not in poller:
    anchor = "\ndef save_batch(symbols: tuple[str, ...]) -> dict[str, Any]:"
    if anchor not in poller:
        raise SystemExit("Could not find save_batch anchor in poller")
    poller = poller.replace(anchor, touch_fn + anchor, 1)
    print("inserted: touch_latest_rows_for_symbols")
else:
    print("touch function already exists")

if "touched = touch_latest_rows_for_symbols(symbols)" not in poller:
    poller = poller.replace(
        '    return {\n        "ok": not errors,',
        '    touched = touch_latest_rows_for_symbols(symbols)\n\n    return {\n        "ok": not errors,',
        1,
    )
    poller = poller.replace(
        '        "saved": saved,',
        '        "saved": saved,\n        "touched": touched,',
        1,
    )
    print("patched: poller save_batch touches latest rows")
else:
    print("poller touch call already exists")

poller_path.write_text(poller, encoding="utf-8")

print("patch complete")

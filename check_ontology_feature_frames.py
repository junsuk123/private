import json
from urllib.request import urlopen
from app.models.live_training_pipeline import collect_live_feature_frames_from_realtime_store

diag = json.loads(urlopen("http://127.0.0.1:8000/api/research/diagnostics", timeout=10).read().decode("utf-8"))

symbols = []
for path in diag.get("reasoning_paths") or []:
    if path.get("conclusion") != "BuyCandidate":
        continue
    ticker = str(path.get("ticker") or "").upper().strip()
    if ticker and not (ticker.isdigit() and len(ticker) == 6):
        symbols.append(ticker)

symbols = tuple(dict.fromkeys(symbols))
print("ontology_us_buy_candidates =", symbols)

result = collect_live_feature_frames_from_realtime_store(symbols=symbols)
print(json.dumps(result, indent=2, ensure_ascii=False))

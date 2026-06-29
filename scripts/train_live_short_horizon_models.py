from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.features.feature_schema import LIVE_SHORT_HORIZON_SCHEMA
from app.models.live_model_trainer import train_live_short_horizon_model
from app.models.model_artifact_registry import ModelArtifactRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description="Train fitted live short-horizon models.")
    parser.add_argument("--dataset", type=Path, default=None, help="JSONL rows with features, label, forward_net_return_bps.")
    parser.add_argument("--demo-fixture", action="store_true", help="Generate deterministic fitted fixture data for dry-run validation.")
    parser.add_argument("--model-dir", type=Path, default=Path("data/models/live_short_horizon"))
    args = parser.parse_args()

    rows = _demo_rows() if args.demo_fixture else _load_rows(args.dataset)
    if args.demo_fixture and args.model_dir == Path("data/models/live_short_horizon"):
        args.model_dir = Path("data/models/live_short_horizon_demo")
    artifact = train_live_short_horizon_model(
        rows,
        registry=ModelArtifactRegistry(args.model_dir),
        force_live_ineligible_reason="DEMO_FIXTURE_NOT_LIVE_ELIGIBLE" if args.demo_fixture else None,
    )
    print(json.dumps({"live_eligible": artifact["live_eligible"], "artifact_id": artifact["artifact_id"], "reason_codes": artifact["reason_codes"]}, indent=2))
    return 0 if artifact["live_eligible"] else 1


def _load_rows(path: Path | None) -> list[dict]:
    if path is None:
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _demo_rows() -> list[dict]:
    rows = []
    names = LIVE_SHORT_HORIZON_SCHEMA.feature_names
    for index in range(80):
        signal = (index % 20) / 20.0
        positive = index % 4 != 0
        features = {name: 0.0 for name in names}
        features["return_30s"] = 0.003 + signal * 0.002 if positive else -0.002 - signal * 0.001
        features["return_1m"] = 0.004 + signal * 0.002 if positive else -0.003
        features["return_3m"] = 0.006 + signal * 0.004 if positive else -0.004
        features["distance_from_vwap"] = 0.001 if positive else -0.002
        features["spread_bps"] = 8 if positive else 35
        features["orderbook_imbalance"] = 0.2 if positive else -0.25
        features["bid_depth"] = 200_000 + index * 1000
        features["ask_depth"] = 150_000
        features["depth_ratio"] = features["bid_depth"] / features["ask_depth"]
        features["liquidity_score"] = 0.9 if positive else 0.25
        features["realized_volatility_3m"] = 0.001 + signal * 0.0002
        features["max_drop_3m"] = -0.001 if positive else -0.01
        features["cost_to_volatility_ratio"] = 0.2 if positive else 2.5
        features["principal_cushion_ratio"] = 1.0
        rows.append(
            {
                "features": features,
                "label": 1 if positive else 0,
                "forward_net_return_bps": 45 + signal * 30 if positive else -25,
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())

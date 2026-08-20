#!/usr/bin/env python
"""Train the temporal hetero GNN from stored decisions and write a checkpoint.

Without a checkpoint the runtime reports ``OFFLINE`` and the gate blocks every new entry
— by design. This script is how that state is left: it reads the decisions and outcomes
already in ``data/store/trading_state.sqlite3``, fits the model, and writes
``data/models/temporal_hetero_gnn/latest.npz`` atomically.

    python scripts/train_temporal_hetero_gnn.py --epochs 40

Refuses to write a checkpoint when the training set is too small or the fit did not
improve on its starting point. Both are real failure modes: a checkpoint trained on
fourteen rows would flip the runtime to HEALTHY and let a random-weight model size
positions, which is worse than staying OFFLINE.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from app.models.gnn_runtime import DEFAULT_CHECKPOINT_PATH  # noqa: E402
from app.models.graph_snapshot import FEATURE_DIM  # noqa: E402
from app.models.temporal_hetero_gnn import TemporalHeteroGnnConfig  # noqa: E402
from app.models.temporal_hetero_gnn_training import (  # noqa: E402
    build_relation_outcomes,
    evaluate_promotion,
    fit_relation_weights,
    label_counts,
    persist_relation_weights,
    train_temporal_hetero_gnn,
)
from app.ontology.market_graph import load_market_graph  # noqa: E402
from app.storage.trading_state_store import default_trading_state_store  # noqa: E402
from app.trading.context_runtime import GRAPH_MAX_NODES, GRAPH_TIME_STEPS  # noqa: E402
from app.trading.training_examples import (  # noqa: E402
    MINIMUM_TRAINING_EXAMPLES,
    build_training_examples,
)


@contextlib.contextmanager
def _single_writer(target: Path):
    """Hold an exclusive lock for the duration of the fit, or yield ``False``.

    ``run.ps1`` force-kills the app server, which skips every teardown path, so a trainer
    started by the previous server can still be running when the next one boots and trains
    again on startup. An advisory ``flock`` is the right guard because the kernel drops it
    when the holder dies — including on SIGKILL, where no cleanup code of ours would run.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows has no flock; run unguarded.
        yield True
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / f".{target.stem}.train.lock"
    handle = lock_path.open("w")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        yield True
    finally:
        handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Cap on replayed decisions. Loss cost is linear in this.",
    )
    parser.add_argument(
        "--horizon-minutes",
        type=int,
        default=30,
        help="Outcome window. Decisions newer than this are excluded as unresolved.",
    )
    parser.add_argument("--output", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Publish even if the deployed checkpoint scores better on the same examples. "
            "Without it a challenger that loses is discarded and the champion stands."
        ),
    )
    parser.add_argument(
        "--minimum-examples",
        type=int,
        default=MINIMUM_TRAINING_EXAMPLES,
        help="Refuse to write a checkpoint below this many resolved decisions.",
    )
    parser.add_argument(
        "--minimum-head-labels",
        type=int,
        default=MINIMUM_TRAINING_EXAMPLES,
        help=(
            "A head with fewer labels than this is treated as unsupervised: its weights "
            "are published as untrained, which holds the runtime at DEGRADED."
        ),
    )
    parser.add_argument(
        "--relations-only",
        action="store_true",
        help="Fit only the ontology relation weights; leave the GNN checkpoint alone.",
    )
    args = parser.parse_args()

    store = default_trading_state_store()
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=max(1, args.lookback_days))

    graph = load_market_graph()
    outcomes = build_relation_outcomes(
        store, horizon_minutes=args.horizon_minutes, now=now
    )
    learned = fit_relation_weights(graph, outcomes)
    applied = persist_relation_weights(graph, learned, store=store, updated_at=now)
    report: dict[str, object] = {
        "relation_outcomes": len(outcomes),
        "relation_weights_learned": len(learned),
        "relation_weights_applied": applied,
    }

    if args.relations_only:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    config = TemporalHeteroGnnConfig(
        max_nodes=GRAPH_MAX_NODES, feature_dim=FEATURE_DIM, time_steps=GRAPH_TIME_STEPS
    )
    examples = build_training_examples(
        store,
        config=config,
        since=since,
        horizon_minutes=args.horizon_minutes,
        now=now,
        limit=args.limit,
    )
    report["training_examples"] = len(examples)
    if len(examples) < max(1, args.minimum_examples):
        report["checkpoint"] = None
        report["refused"] = (
            f"only {len(examples)} resolved decisions; "
            f"{args.minimum_examples} required. The runtime stays OFFLINE, which blocks "
            "new entries and leaves exits working."
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    # Example count is not supervision. Decisions that never became fills carry no
    # trade_quality or expected_return label, so those heads would come out of a fit
    # exactly as they went in — and a checkpoint that does not say so reads as HEALTHY
    # and lets initialisation noise size live positions.
    counts = label_counts(examples)
    supervised = tuple(
        head for head, count in counts.items() if count >= max(1, args.minimum_head_labels)
    )
    report["head_label_counts"] = counts
    report["supervised_heads"] = list(supervised)
    if "regime" not in supervised:
        report["checkpoint"] = None
        report["refused"] = (
            f"no head reached {args.minimum_head_labels} labels "
            f"(counts: {counts}); there is nothing to fit. The runtime stays OFFLINE."
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1
    report["expected_health"] = (
        "HEALTHY" if len(supervised) == len(counts) else "DEGRADED (GNN_PARTIAL_TRAINING)"
    )

    with _single_writer(Path(args.output)) as acquired:
        if not acquired:
            report["checkpoint"] = None
            report["promoted"] = False
            report["refused"] = (
                "another training run already holds the checkpoint lock; skipping so "
                "two fits cannot compete for the machine and publish over each other."
            )
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0
        return _fit_and_publish(args, config, examples, supervised, report)


def _fit_and_publish(
    args: argparse.Namespace,
    config: TemporalHeteroGnnConfig,
    examples: Sequence[Any],
    supervised: tuple[str, ...],
    report: dict[str, Any],
) -> int:
    # Fit once and publish the same model, rather than fitting to test the improvement
    # and fitting again to keep it: identical arguments and seed make the second run a
    # reproduction of the first at full cost.
    model, training = train_temporal_hetero_gnn(
        examples,
        config=config,
        epochs=args.epochs,
        population=args.population,
        sigma=args.sigma,
        learning_rate=args.learning_rate,
        seed=args.seed,
        checkpoint_path=None,
        progress=lambda epoch, loss: print(
            f"epoch {epoch + 1}/{args.epochs} loss={loss:.6f}", file=sys.stderr
        ),
    )
    report["training"] = training.as_dict()
    if not training.improved:
        report["checkpoint"] = None
        report["refused"] = (
            "the fit did not improve on its starting point; refusing to publish a "
            "checkpoint that would flip the runtime to HEALTHY on untrained weights."
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    # Champion/challenger. Scheduled retraining publishes without anyone reading the
    # report, so a fit that improved on its own starting point but is still worse than
    # what is already deployed must not be able to replace it.
    promotion = evaluate_promotion(
        model, examples, incumbent_path=Path(args.output)
    )
    report["promotion"] = promotion
    if not promotion["promote"] and not args.force:
        report["checkpoint"] = None
        report["promoted"] = False
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    report["checkpoint"] = str(
        model.save_checkpoint(Path(args.output), trained_heads=supervised)
    )
    report["promoted"] = True
    report["parameter_count"] = model.parameter_count()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

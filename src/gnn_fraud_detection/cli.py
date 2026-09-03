"""CLI entry point.

Commands:
    prepare                          build processed graph from raw CSVs
    train   --model gat [--epochs N] single-model training run
    compare                        train all 5 architectures, write comparison table
    explain --model gat            feature attribution with Captum
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .data_loader import build_graph, load_processed, save_processed
from .evaluate import (
    evaluate_model,
    metrics_row,
    plot_roc_pr,
    plot_training_curves,
)
from .models import build_model
from .splits import build_masks
from .train import train_model

SEED = 42


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cmd_prepare(args) -> None:
    cfg = load_config()
    t0 = time.time()
    print("Building graph from raw CSVs...")
    data = build_graph(cfg)
    out = save_processed(data, cfg)
    n_lab = int((~torch.isnan(data.y)).sum())
    print(
        f"nodes={data.num_nodes:,} edges={data.edge_index.size(1):,} "
        f"labeled={n_lab:,} ({n_lab / data.num_nodes:.1%})"
    )
    print(f"illicit={int((data.y == 1).sum()):,} licit={int((data.y == 0).sum()):,}")
    print(f"saved -> {out}  ({time.time() - t0:.1f}s)")


def _train_one(model_name: str, overrides: dict | None = None, tag: str | None = None) -> dict:
    """Shared training pipeline; returns a row for the comparison table."""
    overrides = dict(overrides or {})
    overrides.setdefault("model.name", model_name)
    cfg = load_config(overrides)
    _set_seed(cfg.training.seed)
    data = load_processed(cfg)
    from .preprocessing import scale_features

    scale_features(data, cfg)
    masks = build_masks(data, cfg)
    model = build_model(cfg.model.name, data.x.size(1), cfg.model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {cfg.model.name.upper()} | params={n_params:,} ===")

    mlf = None
    if cfg.mlflow.enabled:
        from .mlflow_utils import flatten_params, setup_mlflow

        mlf = setup_mlflow(cfg)
        if mlf is not None:
            mlf.start_run(run_name=tag or cfg.model.name)
            mlf.log_params(flatten_params(cfg))

    history, artifacts = train_model(model, data, masks, cfg)

    metrics = evaluate_model(model, data, masks, artifacts["threshold"], device=cfg.training.device)
    row = metrics_row(
        cfg.model.name,
        metrics,
        extra={
            "params": n_params,
            "best_epoch": artifacts["best_epoch"],
            "train_s": round(artifacts["train_seconds"], 1),
        },
    )

    # plots
    plot_training_curves(
        history, cfg.reports_dir / f"curves_{cfg.model.name}.png", cfg.model.name.upper()
    )
    tm = masks["test"].cpu().numpy()
    y_true = data.y.cpu().numpy()[tm]
    dev = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    with torch.no_grad():
        probs = torch.sigmoid(model(data.x.to(dev), data.edge_index.to(dev))).cpu().numpy()
    plot_roc_pr(
        y_true, probs[tm], cfg.reports_dir / f"roc_pr_{cfg.model.name}.png", cfg.model.name.upper()
    )

    ckpt = Path(cfg.checkpoint_dir) / f"{cfg.model.name}.pt"
    torch.save(model.state_dict(), ckpt)

    if mlf is not None:
        mlf.log_metrics({k: v for k, v in row.items() if isinstance(v, (int, float))})
        try:
            mlf.log_artifact(str(ckpt))
            mlf.log_artifacts(str(cfg.reports_dir))
        except Exception as e:  # artifact logging is best-effort
            print(f"  [mlflow] artifact logging skipped: {e}")
        mlf.end_run()

    print(
        f"  test ROC-AUC {row['roc_auc']:.4f} | PR-AUC {row['pr_auc']:.4f} | "
        f"F1 {row['f1']:.4f} | P {row['precision']:.4f} | R {row['recall']:.4f} "
        f"| thr {row['threshold']:.3f}"
    )
    return row


def cmd_train(args) -> None:
    overrides = {"model.name": args.model}
    if args.epochs:
        overrides["training.epochs"] = args.epochs
    if args.lr:
        overrides["training.lr"] = args.lr
    if args.hidden:
        overrides["model.hidden_dim"] = args.hidden
    if args.focal_gamma is not None:
        overrides["training.focal_gamma"] = args.focal_gamma
    row = _train_one(args.model, overrides, tag=args.tag)
    print(json.dumps(row, indent=2))


def cmd_compare(args) -> None:
    out_csv = Path("reports") / "model_comparison.csv"
    rows = []
    for name in args.models:
        try:
            rows.append(_train_one(name, {}, tag=f"compare-{name}"))
        except Exception as e:
            print(f"!! {name} failed: {e}", file=sys.stderr)
            raise
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("\n=== MODEL COMPARISON (test split, t42..t49) ===")
    print(df.to_string(index=False))
    print(f"\nsaved -> {out_csv}")


def cmd_explain(args) -> None:
    cfg = load_config({"model.name": args.model})
    data = load_processed(cfg)
    masks = build_masks(data, cfg)
    model = build_model(cfg.model.name, data.x.size(1), cfg.model)
    ckpt = Path(cfg.checkpoint_dir) / f"{cfg.model.name}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt}. Run train first.")
    model.load_state_dict(torch.load(ckpt, weights_only=True))

    # pick a few illicit + licit test nodes for contrast
    test_idx = torch.nonzero(masks["test"]).squeeze(-1).numpy()
    y_test = data.y.numpy()[test_idx]
    illicit = test_idx[y_test == 1][: args.n]
    licit = test_idx[y_test == 0][: args.n]
    targets = np.concatenate([illicit, licit])

    from .explain import aggregate_importance, explain_nodes, top_features

    res = explain_nodes(model, data, targets, cfg, method=args.method)
    attrs = res["attributions"]

    print(
        f"\nExplained {len(targets)} nodes ({len(illicit)} illicit / "
        f"{len(licit)} licit) with {args.method}"
    )
    for i, node in enumerate(targets):
        top = top_features(attrs[i], k=5)
        label = "illicit" if i < len(illicit) else "licit"
        tops = ", ".join(f"{n}={v:+.3f}" for n, v in top)
        print(f"  node {node} [{label}]: {tops}")

    imp = aggregate_importance(attrs)
    order = np.argsort(-imp)[: args.n]
    print("\nGlobal top features (mean |attribution|):")
    print(", ".join(f"feature_{i + 1}={imp[i]:.4f}" for i in order))

    out = cfg.reports_dir / f"explanation_{args.model}_{args.method}.npz"
    np.savez_compressed(out, attributions=attrs, node_idx=targets, importance=imp)
    print(f"\nsaved -> {out}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gnn-fraud", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="build processed graph from raw CSVs")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="train a single model")
    p.add_argument("--model", default="gat", choices=["gcn", "gat", "graphsage", "gin", "mlp"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--focal-gamma", type=float, default=None, dest="focal_gamma")
    p.add_argument("--tag", default=None, help="run name tag for MLflow")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("compare", help="train all models and compare")
    p.add_argument("--models", nargs="+", default=["mlp", "gcn", "gat", "graphsage", "gin"])
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("explain", help="explain predictions (Captum)")
    p.add_argument("--model", default="gat")
    p.add_argument(
        "--method", default="integrated_gradients", choices=["integrated_gradients", "saliency"]
    )
    p.add_argument("--n", type=int, default=5, help="nodes per class to explain")
    p.set_defaults(func=cmd_explain)

    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())

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
from .data_loader import load_dataset, load_processed, save_processed
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
    overrides = {}
    if args.dataset:
        overrides["data.dataset"] = args.dataset
    if args.mode:
        overrides["data.pp_mode"] = args.mode
    cfg = load_config(overrides)
    t0 = time.time()
    print(f"Building graph: dataset={cfg.data.dataset} mode={cfg.data.pp_mode}")
    data = load_dataset(cfg)
    if hasattr(data, "node_types"):  # HeteroData: report per node type
        for nt in data.node_types:
            n_lab = int((~torch.isnan(data[nt].y)).sum())
            print(
                f"{nt}: nodes={data[nt].num_nodes:,} labeled={n_lab:,} "
                f"illicit={int((data[nt].y == 1).sum()):,}"
            )
        n_edges = sum(data[rel].edge_index.size(1) for rel in data.edge_types)
        print(f"edges={n_edges:,} across {len(data.edge_types)} relations")
    else:
        n_lab = int((~torch.isnan(data.y)).sum())
        print(
            f"nodes={data.num_nodes:,} edges={data.edge_index.size(1):,} "
            f"labeled={n_lab:,} ({n_lab / data.num_nodes:.1%})"
        )
        print(f"illicit={int((data.y == 1).sum()):,} licit={int((data.y == 0).sum()):,}")
    out = save_processed(data, cfg)
    print(f"saved -> {out}  ({time.time() - t0:.1f}s)")


def _train_one_hetero(
    model_name: str, overrides: dict | None = None, tag: str | None = None
) -> dict:
    """Merged-mode training pipeline (HeteroGNN, actor classification)."""
    del model_name  # hetero mode trains the SAGE-based HeteroGNN regardless
    overrides = dict(overrides or {})
    overrides.setdefault("data.dataset", "ellipticpp")
    overrides.setdefault("data.pp_mode", "merged")
    cfg = load_config(overrides)
    _set_seed(cfg.training.seed)

    data = load_processed(cfg)

    # scale each node type independently, fit on train-period rows
    t_tx, t_ac = data["tx"].t, data["actor"].t
    for nt, t in (("tx", t_tx), ("actor", t_ac)):
        m = t <= cfg.split.train_max_t
        if int(m.sum()) < 2:
            m = torch.ones(data[nt].num_nodes, dtype=torch.bool)
        mu, sd = data[nt].x[m].mean(dim=0), data[nt].x[m].std(dim=0).clamp_min(1e-8)
        data[nt].x = (data[nt].x - mu) / sd
        data[nt].x = torch.nan_to_num(data[nt].x, nan=0.0)

    from .train_hetero import evaluate_hetero, train_hetero

    history, artifacts = train_hetero(data, cfg)
    metrics = evaluate_hetero(artifacts["model"], data, cfg)
    n_params = sum(p.numel() for p in artifacts["model"].parameters())
    row = {
        "model": "heterosage",
        "roc_auc": round(metrics["roc_auc"], 4),
        "pr_auc": round(metrics["pr_auc"], 4),
        "f1": round(metrics["f1"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "threshold": round(float(artifacts["threshold"]), 4),
        "params": n_params,
        "best_epoch": artifacts["best_epoch"],
        "train_s": round(artifacts["train_seconds"], 1),
    }
    ckpt = Path(cfg.checkpoint_dir) / "heterosage_merged.pt"
    torch.save(artifacts["model"].state_dict(), ckpt)
    print(f"\n=== HETERO-SAGE (merged tx+actors) | params={n_params:,} ===")
    print(
        f"  test ROC-AUC {row['roc_auc']:.4f} | PR-AUC {row['pr_auc']:.4f} "
        f"| F1 {row['f1']:.4f} | P {row['precision']:.4f} | R {row['recall']:.4f}"
    )
    return row


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
    if args.dataset:
        overrides["data.dataset"] = args.dataset
    if args.mode:
        overrides["data.pp_mode"] = args.mode
    if args.epochs:
        overrides["training.epochs"] = args.epochs
    if args.lr:
        overrides["training.lr"] = args.lr
    if args.hidden:
        overrides["model.hidden_dim"] = args.hidden
    if args.focal_gamma is not None:
        overrides["training.focal_gamma"] = args.focal_gamma
    if overrides.get("data.pp_mode") == "merged":
        row = _train_one_hetero(args.model, overrides, tag=args.tag)
    else:
        row = _train_one(args.model, overrides, tag=args.tag)
    print(json.dumps(row, indent=2))


def cmd_compare(args) -> None:
    out_csv = Path("reports") / "model_comparison.csv"
    overrides = {}
    if args.dataset:
        overrides["data.dataset"] = args.dataset
    if args.mode:
        overrides["data.pp_mode"] = args.mode
    if args.focal_gamma is not None:
        overrides["training.focal_gamma"] = args.focal_gamma
    seeds = list(args.seeds) if args.seeds else [None]
    rows = []
    for name in args.models:
        for seed in seeds:
            ov = dict(overrides)
            tag = f"compare-{name}" if seed is None else f"compare-{name}-s{seed}"
            if seed is not None:
                ov["training.seed"] = seed
            try:
                rows.append(_train_one(name, ov, tag=tag))
            except Exception as e:
                print(f"!! {name} failed: {e}", file=sys.stderr)
                raise
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("\n=== MODEL COMPARISON (test split, t42..t49) ===")
    print(df.to_string(index=False))
    if len(seeds) > 1:
        agg = df.groupby("model")[["roc_auc", "pr_auc", "f1"]].agg(["mean", "std"])
        print("\n=== AGGREGATED OVER SEEDS ===")
        print(agg.round(4).to_string())
        agg_csv = Path("reports") / "model_comparison_aggregated.csv"
        agg.round(4).to_csv(agg_csv)
        print(f"saved -> {agg_csv}")
    print(f"\nsaved -> {out_csv}")


def cmd_temporal(args) -> None:
    """Tier-2 temporal evaluation: static per-step metrics + incremental FT."""
    from .preprocessing import scale_features
    from .temporal import (
        incremental_finetune,
        plot_temporal_comparison,
        static_per_timestep,
    )

    overrides = {"model.name": args.model}
    if args.dataset:
        overrides["data.dataset"] = args.dataset
    if args.mode:
        overrides["data.pp_mode"] = args.mode
    cfg = load_config(overrides)
    _set_seed(cfg.training.seed)
    data = load_processed(cfg)
    scale_features(data, cfg)
    masks = build_masks(data, cfg)
    model = build_model(cfg.model.name, data.x.size(1), cfg.model)
    ckpt = Path(cfg.checkpoint_dir) / f"{cfg.model.name}.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        print(f"loaded checkpoint {ckpt}")
    else:
        print(f"no checkpoint for {cfg.model.name}; training one now")
        history, artifacts = train_model(model, data, masks, cfg)
        torch.save(model.state_dict(), ckpt)

    static = static_per_timestep(model, data, cfg)
    print("\n=== STATIC per test time step ===")
    for s, m in sorted(static.items()):
        print(f"  t={s}: n={m['n']:5d}  ROC-AUC {m['roc_auc']:.4f}  PR-AUC {m['pr_auc']:.4f}")

    incr = (
        incremental_finetune(data, cfg, model_name=cfg.model.name) if not args.static_only else None
    )
    if incr:
        print("\n=== INCREMENTAL FT per test time step ===")
        for s, m in sorted(incr["per_timestep"].items()):
            print(f"  t={s}: ROC-AUC {m['roc_auc']:.4f}  PR-AUC {m['pr_auc']:.4f}")
        print(f"  mean: ROC-AUC {incr['mean_roc_auc']:.4f} PR-AUC {incr['mean_pr_auc']:.4f}")

    out_png = cfg.reports_dir / f"temporal_{cfg.model.name}.png"
    plot_temporal_comparison(
        static, incr["per_timestep"] if incr else None, out_png, cfg.model.name.upper()
    )
    import json as _json

    out_json = cfg.reports_dir / f"temporal_{cfg.model.name}.json"
    out_json.write_text(_json.dumps({"static": static, "incremental": incr}, indent=2, default=str))
    print(f"\nsaved -> {out_png}\nsaved -> {out_json}")


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
    p.add_argument("--dataset", default=None, choices=["elliptic", "ellipticpp"])
    p.add_argument("--mode", default=None, choices=["tx", "actors", "merged"])
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="train a single model")
    p.add_argument(
        "--model", default="gat", choices=["gcn", "gat", "graphsage", "gin", "mlp", "heterosage"]
    )
    p.add_argument("--dataset", default=None, choices=["elliptic", "ellipticpp"])
    p.add_argument("--mode", default=None, choices=["tx", "actors", "merged"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--focal-gamma", type=float, default=None, dest="focal_gamma")
    p.add_argument("--tag", default=None, help="run name tag for MLflow")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("compare", help="train all models and compare")
    p.add_argument("--models", nargs="+", default=["mlp", "gcn", "gat", "graphsage", "gin"])
    p.add_argument("--dataset", default=None, choices=["elliptic", "ellipticpp"])
    p.add_argument("--mode", default=None, choices=["tx", "actors", "merged"])
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="multi-seed mode: mean/std aggregation over these seeds",
    )
    p.add_argument("--focal-gamma", type=float, default=None, dest="focal_gamma")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("explain", help="explain predictions (Captum)")
    p.add_argument("--model", default="gat")
    p.add_argument(
        "--method", default="integrated_gradients", choices=["integrated_gradients", "saliency"]
    )
    p.add_argument("--n", type=int, default=5, help="nodes per class to explain")
    p.add_argument("--dataset", default=None, choices=["elliptic", "ellipticpp"])
    p.add_argument("--mode", default=None, choices=["tx", "actors", "merged"])
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("temporal", help="per-time-step metrics + incremental FT baseline")
    p.add_argument("--model", default="gcn")
    p.add_argument("--dataset", default=None, choices=["elliptic", "ellipticpp"])
    p.add_argument("--mode", default=None, choices=["tx", "actors", "merged"])
    p.add_argument("--static-only", action="store_true", dest="static_only")
    p.set_defaults(func=cmd_temporal)

    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())

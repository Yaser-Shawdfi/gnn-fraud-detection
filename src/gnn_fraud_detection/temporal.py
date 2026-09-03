"""Temporal evaluation: per-time-step test metrics + incremental baseline.

The test window (t42..t49) spans 8 consecutive two-week periods. A static
model degrades as the graph drifts; periodically fine-tuning on newly
labeled data adapts. This module quantifies both:

  1. static_per_timestep(): one trained model, metrics per time step
  2. incremental_finetune(): train on t<=33, then repeatedly fine-tune
     with each newly available labeled period before predicting the next
     (realistic deployed-system schedule: retrain as labels arrive)
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import Config
from .models import build_model
from .splits import compute_pos_weight
from .train import make_loss_fn


@torch.no_grad()
def _probs_all(model, data, device) -> np.ndarray:
    model.eval()
    logits = model(data.x.to(device), data.edge_index.to(device))
    return torch.sigmoid(logits).cpu().numpy()


def static_per_timestep(model, data, cfg: Config) -> dict[int, dict]:
    """Metrics for each test time step t42..t49 from a single trained model."""
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    probs = _probs_all(model, data, device)
    y = data.y.numpy()
    t = data.t.numpy()

    out: dict[int, dict] = {}
    for step in range(cfg.split.val_max_t + 1, 50):
        m = (t == step) & ~np.isnan(y)
        if m.sum() < 10 or len(np.unique(y[m])) < 2:
            continue
        out[step] = {
            "n": int(m.sum()),
            "roc_auc": float(roc_auc_score(y[m], probs[m])),
            "pr_auc": float(average_precision_score(y[m], probs[m])),
        }
    return out


def incremental_finetune(data, cfg: Config, model_name: str = "gcn") -> dict:
    """Simulate periodic retraining as new labeled periods arrive.

    Schedule:
      - train on labeled nodes with t <= 33
      - for each test step s in 42..49: predict s, then fine-tune with all
        labeled data up to s (inclusive), i.e. predictions for period s are
        made by a model that has seen labeled periods up to s-1.

    Returns {"per_timestep": {...}, "mean_roc_auc": float, "mean_pr_auc": float}
    """
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    data_gpu = data.to(device)
    t_cpu = data.t.cpu().clone()  # guaranteed-CPU copy for numpy conversion
    model = build_model(model_name, data.x.size(1), cfg.model).to(device)

    def fit(upto_t: int, epochs: int, lr: float):
        labeled = ~torch.isnan(data_gpu.y)
        m = labeled & (data_gpu.t <= upto_t)
        ytr = data_gpu.y[m].float()
        pos_w = compute_pos_weight(data_gpu, cfg, m).to(device)
        loss_fn = make_loss_fn(cfg, pos_w)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.training.weight_decay)
        for ep in range(epochs):
            model.train()
            opt.zero_grad()
            logits = model(data_gpu.x, data_gpu.edge_index)
            loss = loss_fn(logits[m], ytr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            opt.step()
        return float(loss.item())

    print("  [incremental] initial training t<=33 ...", flush=True)
    fit(cfg.split.train_max_t, cfg.training.epochs, cfg.training.lr)

    y_np = data_gpu.y.cpu().numpy()
    t_np = t_cpu.numpy()
    per_t: dict[int, dict] = {}

    # fine-tune with val-period labels (t34..41) before the first test step
    val_last = cfg.split.val_max_t
    print(f"  [incremental] fine-tune with labeled t<= {val_last} ...", flush=True)
    fit(val_last, max(20, cfg.training.epochs // 5), cfg.training.lr / 2)

    for step in range(cfg.split.val_max_t + 1, 50):
        probs = _probs_all(model, data_gpu, device)
        m = (t_np == step) & ~np.isnan(y_np)
        if m.sum() < 10 or len(np.unique(y_np[m])) < 2:
            continue
        per_t[step] = {
            "n": int(m.sum()),
            "roc_auc": float(roc_auc_score(y_np[m], probs[m])),
            "pr_auc": float(average_precision_score(y_np[m], probs[m])),
        }
        print(
            f"  [incremental] t={step}: AUC {per_t[step]['roc_auc']:.4f} "
            f"AP {per_t[step]['pr_auc']:.4f}",
            flush=True,
        )
        # after predicting step s, absorb its labels for the next step
        if step < 49:
            print(f"  [incremental] fine-tune with labeled t<= {step} ...", flush=True)
            fit(step, max(20, cfg.training.epochs // 5), cfg.training.lr / 2)

    mean_roc = float(np.mean([v["roc_auc"] for v in per_t.values()]))
    mean_ap = float(np.mean([v["pr_auc"] for v in per_t.values()]))
    return {"per_timestep": per_t, "mean_roc_auc": mean_roc, "mean_pr_auc": mean_ap}


def plot_temporal_comparison(static_per_t: dict, incr_per_t: dict | None, out_path, title: str):
    """Static vs incremental AUC/AP across the test time steps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = sorted(static_per_t)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metric, ax, lab in (("roc_auc", axes[0], "ROC-AUC"), ("pr_auc", axes[1], "PR-AUC")):
        ax.plot(
            steps, [static_per_t[s][metric] for s in steps], "o-", color="#004488", label="static"
        )
        if incr_per_t:
            is_steps = sorted(incr_per_t)
            ax.plot(
                is_steps,
                [incr_per_t[s][metric] for s in is_steps],
                "s--",
                color="#cc3311",
                label="incremental FT",
            )
        ax.set_xlabel("test time step")
        ax.set_ylabel(lab)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower left")
        ax.set_title(f"{title} - {lab} per time step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out_path

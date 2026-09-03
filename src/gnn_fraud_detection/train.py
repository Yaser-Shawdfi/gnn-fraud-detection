"""Training loop, loss functions, early stopping, checkpointing."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import Config
from .splits import compute_pos_weight


class FocalLoss(nn.Module):
    """Binary focal loss on logits (Lin et al. 2017) with optional pos_weight."""

    def __init__(
        self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None, reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)
        loss = (1 - p_t) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def make_loss_fn(cfg: Config, pos_weight: torch.Tensor) -> nn.Module:
    if cfg.training.focal_gamma is not None:
        return FocalLoss(gamma=cfg.training.focal_gamma, pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


@torch.no_grad()
def _eval_probs(model, data, mask) -> np.ndarray:
    model.eval()
    logits = model(data.x, data.edge_index)
    return torch.sigmoid(logits[mask]).cpu().numpy()


def _composite_score(auc: float, ap: float) -> float:
    """Early-stopping metric: mean of ROC-AUC and Average Precision."""
    return 0.5 * (auc + ap)


def train_model(
    model: nn.Module,
    data,
    masks: dict[str, torch.Tensor],
    cfg: Config,
    log_callback=None,
) -> tuple[dict, dict]:
    """Full-batch training loop with early stopping.

    Returns:
        (history, artifacts)
        history: dict of per-epoch metric lists
        artifacts: {"best_state": state_dict, "best_epoch": int,
                    "threshold": float, "best_val_score": float}
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)

    train_mask = masks["train"].to(device)
    val_mask = masks["val"].to(device)

    pos_weight = compute_pos_weight(data, cfg, train_mask).to(device)
    loss_fn = make_loss_fn(cfg, pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )

    y_train = data.y[train_mask].float()
    y_val = data.y[val_mask].cpu().numpy()

    history: dict[str, list] = {
        "epoch": [],
        "loss": [],
        "train_auc": [],
        "val_auc": [],
        "val_ap": [],
        "val_score": [],
    }
    best_score, best_state, best_epoch = -1.0, None, -1
    patience_left = cfg.training.patience

    t0 = time.time()
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = loss_fn(logits[train_mask], y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
        optimizer.step()

        # --- validation metrics every epoch (cheap on 200k nodes) ---
        p_tr = _eval_probs(model, data, train_mask)
        p_val = _eval_probs(model, data, val_mask)
        try:
            tr_auc = roc_auc_score(y_train.cpu().numpy(), p_tr)
        except ValueError:
            tr_auc = float("nan")
        try:
            va_auc = roc_auc_score(y_val, p_val)
            va_ap = average_precision_score(y_val, p_val)
        except ValueError:
            # degenerate validation split (single class) - neutral score
            va_auc, va_ap = 0.5, 0.5

        score = _composite_score(va_auc, va_ap)
        history["epoch"].append(epoch)
        history["loss"].append(float(loss.item()))
        history["train_auc"].append(tr_auc)
        history["val_auc"].append(va_auc)
        history["val_ap"].append(va_ap)
        history["val_score"].append(score)

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_left = cfg.training.patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  epoch {epoch:3d} | loss {loss.item():.4f} | "
                f"train AUC {tr_auc:.4f} | val AUC {va_auc:.4f} | "
                f"val AP {va_ap:.4f} | best {best_score:.4f}",
                flush=True,
            )

    elapsed = time.time() - t0

    # Restore best weights and pick threshold on validation
    model.load_state_dict(best_state)
    p_val = _eval_probs(model, data, val_mask)
    from .splits import pick_threshold

    threshold = pick_threshold(y_val, p_val, cfg.evaluation.threshold_strategy)

    artifacts = {
        "best_state": best_state,
        "best_epoch": best_epoch,
        "threshold": threshold,
        "best_val_score": best_score,
        "train_seconds": elapsed,
    }
    return history, artifacts


def save_checkpoint(model, cfg: Config, model_name: str) -> Path:
    out = cfg.checkpoint_dir / f"{model_name}.pt"
    torch.save(model.state_dict(), out)
    return out

"""Training loop for the heterogeneous merged graph (actor classification)."""

from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import Config
from .hetero import HeteroGNN, hetero_masks
from .splits import pick_threshold


def _actor_probs(model, data, cfg: Config) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(
            {"tx": data["tx"].x, "actor": data["actor"].x},
            {rel: data[rel].edge_index for rel in data.edge_types},
        )
    return torch.sigmoid(logits).cpu().numpy()


def train_hetero(data, cfg: Config) -> tuple[dict, dict]:
    """Full-batch hetero training with early stopping on val (AUC+AP)/2."""
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    model = HeteroGNN(
        data["tx"].x.size(1),
        data["actor"].x.size(1),
        cfg.model.hidden_dim,
        cfg.model.num_layers,
        cfg.model.dropout,
    ).to(device)

    data_t = data.to(device, non_blocking=True)
    masks = hetero_masks(data, cfg)
    train_mask = masks["train"].to(device)
    val_mask = masks["val"].to(device)

    y_tr = data_t["actor"].y[train_mask]
    n_pos = float((y_tr == 1).sum())
    n_neg = float((y_tr == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos if n_pos else 1.0], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    y_val = data_t["actor"].y[val_mask].cpu().numpy()

    history = {
        "epoch": [],
        "loss": [],
        "train_auc": [],
        "val_auc": [],
        "val_ap": [],
        "val_score": [],
    }
    best_score, best_state, best_epoch = -1.0, None, -1
    patience = cfg.training.patience
    t0 = time.time()

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(
            {"tx": data_t["tx"].x, "actor": data_t["actor"].x},
            {rel: data_t[rel].edge_index for rel in data_t.edge_types},
        )
        loss = loss_fn(logits[train_mask], y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
        optimizer.step()

        probs = _actor_probs(model, data_t, cfg)
        ytr_np = y_tr.cpu().numpy()
        try:
            tr_auc = roc_auc_score(ytr_np, probs[train_mask.cpu().numpy()])
        except ValueError:
            tr_auc = float("nan")
        try:
            va_auc = roc_auc_score(y_val, probs[val_mask.cpu().numpy()])
            va_ap = average_precision_score(y_val, probs[val_mask.cpu().numpy()])
        except ValueError:
            va_auc, va_ap = 0.5, 0.5
        score = 0.5 * (va_auc + va_ap)

        history["epoch"].append(epoch)
        history["loss"].append(float(loss.item()))
        history["train_auc"].append(tr_auc)
        history["val_auc"].append(va_auc)
        history["val_ap"].append(va_ap)
        history["val_score"].append(score)

        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = cfg.training.patience
        else:
            patience -= 1
            if patience == 0:
                break

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  epoch {epoch:3d} | loss {loss.item():.4f} | trAUC {tr_auc:.4f} "
                f"| valAUC {va_auc:.4f} | valAP {va_ap:.4f} | best {best_score:.4f}",
                flush=True,
            )

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    probs = _actor_probs(model, data_t, cfg)
    thr = pick_threshold(y_val, probs[val_mask.cpu().numpy()], cfg.evaluation.threshold_strategy)
    return history, {
        "best_state": best_state,
        "best_epoch": best_epoch,
        "threshold": thr,
        "best_val_score": best_score,
        "train_seconds": elapsed,
        "model": model,
    }


def evaluate_hetero(model, data, cfg: Config) -> dict:
    """Test metrics on actor nodes (t42..t49)."""
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    data_t = data.to(device)
    probs = _actor_probs(model, data_t, cfg)
    masks = hetero_masks(data, cfg)
    tm = masks["test"].cpu().numpy()
    y_true = data_t["actor"].y.cpu().numpy()[tm]
    y_prob = probs[tm]
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "n_test": int(len(y_true)),
        "n_illicit": int((y_true == 1).sum()),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

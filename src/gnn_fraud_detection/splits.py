"""Splits, masks, class weights, thresholding.

Elliptic semi-supervised protocol (Weber et al. 2019, Elliptic++ papers):
  - train: labeled nodes with time-step in [1, 33]
  - val:   labeled nodes with time-step in [34, 41]
  - test:  labeled nodes with time-step in [42, 49]
Unknown-class nodes are never in any mask.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import Config


def build_masks(data, cfg: Config) -> dict[str, torch.Tensor]:
    """Boolean masks over all nodes (False for unknown/unlabeled nodes)."""
    t = data.t
    labeled = ~torch.isnan(data.y)

    train_mask = labeled & (t <= cfg.split.train_max_t)
    val_mask = labeled & (t > cfg.split.train_max_t) & (t <= cfg.split.val_max_t)
    test_mask = labeled & (t > cfg.split.val_max_t)

    return {"train": train_mask, "val": val_mask, "test": test_mask}


def compute_pos_weight(data, cfg: Config, train_mask: torch.Tensor) -> torch.Tensor:
    """Weight for the positive (illicit) class in BCEWithLogitsLoss.

    Auto value = n_neg / n_pos on the labeled training nodes, unless the
    config pins it via split.pos_weight.
    """
    if cfg.split.pos_weight is not None:
        return torch.tensor([float(cfg.split.pos_weight)])
    ytr = data.y[train_mask]
    n_pos = float((ytr == 1).sum())
    n_neg = float((ytr == 0).sum())
    if n_pos == 0:
        return torch.tensor([1.0])
    return torch.tensor([n_neg / n_pos])


def pick_threshold(y_val: np.ndarray, p_val: np.ndarray, strategy: str) -> float:
    """Choose a decision threshold on validation probabilities.

    strategies:
      f1     : maximize F1 for the illicit class
      youden : maximize Youden's J statistic (TPR - FPR)
    Returns the threshold; falls back to 0.5 if validation is degenerate.
    """
    if len(y_val) == 0 or len(np.unique(y_val)) < 2:
        return 0.5

    thresholds = np.unique(np.round(p_val, 4))
    best_t, best_score = 0.5, -1.0

    if strategy == "youden":
        for thr in thresholds:
            pred = (p_val >= thr).astype(int)
            tp = float(((pred == 1) & (y_val == 1)).sum())
            fp = float(((pred == 1) & (y_val == 0)).sum())
            fn = float(((pred == 0) & (y_val == 1)).sum())
            tn = float(((pred == 0) & (y_val == 0)).sum())
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            score = tpr - fpr
            if score > best_score:
                best_t, best_score = float(thr), score
    else:  # f1
        for thr in thresholds:
            pred = (p_val >= thr).astype(int)
            tp = float(((pred == 1) & (y_val == 1)).sum())
            fp = float(((pred == 1) & (y_val == 0)).sum())
            fn = float(((pred == 0) & (y_val == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            if f1 > best_score:
                best_t, best_score = float(thr), f1

    return best_t

"""Evaluation: test metrics, per-class reports, confusion matrix, plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model,
    data,
    masks: dict[str, torch.Tensor],
    threshold: float,
    device: str = "cuda",
) -> dict:
    """Full evaluation on the test split (t42..t49 labeled nodes).

    Returns a metrics dict (JSON-serializable scalars only).
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    data = data.to(dev)
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.sigmoid(logits).cpu().numpy()

    test_mask = masks["test"].cpu().numpy()
    y_true = data.y.cpu().numpy()[test_mask]
    y_prob = probs[test_mask]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "threshold": float(threshold),
        "n_test": int(len(y_true)),
        "n_illicit": int((y_true == 1).sum()),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) else 0.0
    return metrics


def metrics_row(model_name: str, metrics: dict, extra: dict | None = None) -> dict:
    """Flatten metrics into one row for the comparison table."""
    row = {
        "model": model_name,
        "roc_auc": round(metrics["roc_auc"], 4),
        "pr_auc": round(metrics["pr_auc"], 4),
        "f1": round(metrics["f1"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "threshold": round(metrics["threshold"], 4),
        "fpr": round(metrics["fpr"], 4),
    }
    if extra:
        row.update(extra)
    return row


def plot_training_curves(history: dict, out_path: Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = history["epoch"]

    axes[0].plot(epochs, history["loss"], color="#cc3311")
    axes[0].set_title(f"{title} - training loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE loss")

    axes[1].plot(epochs, history["train_auc"], label="train AUC", color="#004488")
    axes[1].plot(epochs, history["val_auc"], label="val AUC", color="#33bbee")
    axes[1].plot(epochs, history["val_ap"], label="val AP", color="#cc3311", ls="--")
    axes[1].set_title(f"{title} - AUC / AP")
    axes[1].set_xlabel("epoch")
    axes[1].legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, color="#004488", label=f"ROC (AUC={auc(fpr, tpr):.4f})")
    axes[0].plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title(f"{title} - ROC")
    axes[0].legend(loc="lower right")

    axes[1].plot(
        rec, prec, color="#cc3311", label=f"PR (AP={average_precision_score(y_true, y_prob):.4f})"
    )
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title(f"{title} - Precision-Recall")
    axes[1].legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

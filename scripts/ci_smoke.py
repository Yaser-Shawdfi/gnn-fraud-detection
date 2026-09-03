"""CI smoke: train + evaluate a tiny GCN on a synthetic graph (CPU, seconds).

Runs the full pipeline - config, preprocessing, splits, training, eval -
without needing the real dataset or a GPU.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch_geometric.data import Data

from gnn_fraud_detection.config import load_config
from gnn_fraud_detection.evaluate import evaluate_model
from gnn_fraud_detection.models import build_model
from gnn_fraud_detection.preprocessing import scale_features
from gnn_fraud_detection.splits import build_masks
from gnn_fraud_detection.train import train_model


def main() -> int:
    g = torch.Generator().manual_seed(0)
    n, in_dim = 200, 165
    x = torch.randn(n, in_dim, generator=g)
    x[50:100] += 0.8  # illicit block, linearly separable signal
    ei = torch.randint(0, n, (2, 1200), generator=g)

    y = torch.zeros(n)
    y[50:100] = 1  # train illicit
    y[125:150] = 1  # val illicit
    y[175:200] = 1  # test illicit

    t = torch.empty(n, dtype=torch.long)
    t[:100] = torch.randint(1, 34, (100,), generator=g)  # train range
    t[100:150] = torch.randint(34, 42, (50,), generator=g)  # val range
    t[150:] = torch.randint(42, 50, (50,), generator=g)  # test range

    data = Data(x=x, edge_index=ei, y=y, t=t)
    data.num_nodes = n

    cfg = load_config(
        {"training.epochs": 5, "training.patience": 5, "mlflow.enabled": False}
    )
    scale_features(data, cfg)
    masks = build_masks(data, cfg)
    model = build_model("gcn", in_dim, cfg.model)
    history, artifacts = train_model(model, data, masks, cfg)
    metrics = evaluate_model(model, data, masks, artifacts["threshold"], device="cpu")
    print("smoke ROC-AUC:", round(metrics["roc_auc"], 4))
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    print("CI smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""DGraphFin loader (FinVolution fintech fraud graph, 3.7M nodes).

npz keys (official release):
  x      [N, 17] float32 node features
  y      [N] int64 labels: 0..2 non-fraud classes, 3 = fraud
  edge_index / edge arrays
  time   [N] int64 day index (1..821)

Label convention here: fraud = 3 -> y=1, everything else -> y=0.
We ignore the official random split and apply our temporal protocol for
consistency with the rest of the repo, splitting by day quantiles:
  train = first 60% of days, val = next 15%, test = last 25%.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from .config import Config


def build_dgraphfin(cfg: Config) -> Data:
    path = cfg.raw_dir / "dgraphfin.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Download DGraphFin and place it there "
            "(direct: https://dgraph.xinye.com/dataset/DGraphFin.zip or the "
            "HF mirror cryoushiwo/DGraph)."
        )
    z = np.load(path)
    print("npz keys:", list(z.keys()))
    x = torch.as_tensor(np.asarray(z["x"], dtype=np.float32))
    y_raw = z["y"].astype(np.int64)
    y = torch.as_tensor((y_raw == 3).astype(np.float32))

    # time: the npz has no per-node 'time' key -> derive from edges is complex;
    # official release stores edge_timestamp. Node time = max timestamp of
    # incident edges is a faithful proxy for "when the node was active".
    if "time" in z:
        t = torch.as_tensor(z["time"].astype(np.int64))
    else:
        ei_np = z["edge_index"]
        et = z["edge_timestamp"].astype(np.int64)
        t_np = np.full(x.size(0), 0, dtype=np.int64)
        np.maximum.at(t_np, ei_np[:, 0], et)
        np.maximum.at(t_np, ei_np[:, 1], et)
        t = torch.as_tensor(t_np)

    # edge array: this npz stores [E, 2] (rows = edges) + separate edge_type
    arr = np.asarray(z["edge_index"])
    if arr.ndim == 2 and arr.shape[1] >= 2:
        ei_t = torch.as_tensor(arr[:, :2].T, dtype=torch.int64)  # -> [2, E]
    else:
        raise ValueError(f"unexpected edge_index shape {arr.shape}")

    # temporal boundaries: 60/75th percentiles of the day distribution
    q60, q75 = np.quantile(t.numpy(), [0.60, 0.75])
    data = Data(x=x, edge_index=torch.unique(ei_t, dim=1), y=y, t=t)
    data.num_nodes = x.size(0)
    data.train_max_t = int(q60)
    data.val_max_t = int(q75)
    return data


def dgraphfin_masks(data: Data, cfg: Config) -> dict[str, torch.Tensor]:
    """Temporal masks using the dataset's own day quantiles."""
    t = data.t
    labeled = ~torch.isnan(data.y)
    train_max = getattr(data, "train_max_t", cfg.split.train_max_t)
    val_max = getattr(data, "val_max_t", cfg.split.val_max_t)
    return {
        "train": labeled & (t <= train_max),
        "val": labeled & (t > train_max) & (t <= val_max),
        "test": labeled & (t > val_max),
    }

"""Dataset loading: raw CSVs -> PyG Data object.

Elliptic bitcoin dataset layout:
  - elliptic_txs_features.csv : no header. Columns: txId, time-step, feature1..feature166
  - elliptic_txs_classes.csv  : header txId,class. class in {1, 2, unknown}
  - elliptic_txs_edgelist.csv : header txId1,txId2. Directed money flow txId1 -> txId2

Label mapping used throughout the project:
  y = 1 -> illicit (class 1)
  y = 0 -> licit   (class 2)
  y = NaN -> unknown (excluded from loss/metrics)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .config import Config

# Column names of the features file (no header in the raw CSV).
# The official release ships 166 features; some mirrors ship 165 (last column
# dropped). We infer the count from the file itself at load time.
FEATURE_COLS = ["txId", "time_step"] + [f"feature_{i}" for i in range(1, 167)]


def _feature_columns(n_total_cols: int) -> list[str]:
    """Names for a features CSV with n_total_cols columns (txId + time + feats)."""
    return FEATURE_COLS[:n_total_cols]


def load_raw_frames(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three raw CSVs with correct headers/limits."""
    # Peek at the first row only to learn the real column count
    n_total = pd.read_csv(cfg.raw_dir / cfg.data.features_file, header=None, nrows=1).shape[1]
    names = _feature_columns(n_total)
    feats = pd.read_csv(cfg.raw_dir / cfg.data.features_file, header=None, names=names)
    classes = pd.read_csv(cfg.raw_dir / cfg.data.classes_file)
    edges = pd.read_csv(cfg.raw_dir / cfg.data.edgelist_file)
    return feats, classes, edges


def build_graph(cfg: Config) -> Data:
    """Load CSVs and assemble a single PyG Data object.

    Node order = row order of the features file. Labels are aligned by txId.
    Time-step is stored as an int64 node attribute `t`.
    """
    feats, classes, edges = load_raw_frames(cfg)

    feats = feats.sort_values("time_step").reset_index(drop=True)
    txid2idx = {tx: i for i, tx in enumerate(feats["txId"].astype(np.int64))}

    n_feat = len(feats.columns) - 2
    feat_names = [f"feature_{i}" for i in range(1, n_feat + 1)]
    x = torch.as_tensor(feats[feat_names].to_numpy(dtype=np.float32))
    t = torch.as_tensor(feats["time_step"].to_numpy(dtype=np.int64))

    class_map = {"1": 1, "2": 0}
    y = (
        classes.set_index(classes["txId"].astype(np.int64))["class"]
        .map(class_map)
        .reindex(feats["txId"].astype(np.int64))
        .to_numpy()
    )
    # unknown -> NaN (float tensor with NaN = unlabeled)
    y_t = torch.as_tensor(y, dtype=torch.float32)

    src = edges["txId1"].map(txid2idx)
    dst = edges["txId2"].map(txid2idx)

    n_unmapped_src = int(src.isna().sum())
    n_unmapped_dst = int(dst.isna().sum())
    if n_unmapped_src or n_unmapped_dst:
        # Keep only edges whose endpoints exist in the features table
        keep = src.notna() & dst.notna()
        src, dst = src[keep], dst[keep]

    src_t = torch.as_tensor(src.to_numpy(dtype=np.int64))
    dst_t = torch.as_tensor(dst.to_numpy(dtype=np.int64))

    if not cfg.graph.directed:
        # Symmetrize: add reverse edges
        ei = torch.stack([torch.cat([src_t, dst_t]), torch.cat([dst_t, src_t])], dim=0)
    else:
        ei = torch.stack([src_t, dst_t], dim=0)

    data = Data(x=x, edge_index=ei, y=y_t, t=t)
    data.txids = feats["txId"].astype(np.int64).to_numpy()
    data.num_nodes = x.size(0)
    return data


def save_processed(data: Data, cfg: Config, name: str | None = None) -> Path:
    if name is None:
        name = processed_name(cfg)
    out = cfg.processed_dir / name
    torch.save(data, out)
    return out


def load_processed(cfg: Config, name: str | None = None) -> Data:
    if name is None:
        name = processed_name(cfg)
    return torch.load(cfg.processed_dir / name, weights_only=False)


def processed_name(cfg: Config) -> str:
    """Processed-file name derived from the dataset selection."""
    if cfg.data.dataset == "ellipticpp":
        return f"ellipticpp_{cfg.data.pp_mode}.pt"
    return "elliptic.pt"


def load_dataset(cfg: Config):
    """Dispatch to the configured dataset/graph-mode builder."""
    if cfg.data.dataset == "ellipticpp":
        if cfg.data.pp_mode == "merged":
            from .hetero import build_merged_hetero

            return build_merged_hetero(cfg)
        from .data_loader_pp import build_actor_graph, build_tx_graph_pp

        mode = cfg.data.pp_mode
        if mode == "tx":
            return build_tx_graph_pp(cfg)
        if mode == "actors":
            return build_actor_graph(cfg)
        raise ValueError(f"Unknown pp_mode: {mode} (tx | actors | merged)")
    return build_graph(cfg)

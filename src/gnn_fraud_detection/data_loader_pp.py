"""Elliptic++ loaders: transactions graph + actors (wallet addresses) graph.

Source: git-disl/EllipticPlusPlus (Elmougy & Liu, KDD 2023).
Files (data/raw/ellipticpp/):
  Transactions: txs_features.csv, txs_classes.csv, txs_edgelist.csv
  Actors:       wallets_features.csv, wallets_classes.csv,
                AddrAddr_edgelist.csv, AddrTx_edgelist.csv, TxAddr_edgelist.csv

Label mapping everywhere: 1 = illicit, 0 = licit, NaN = unknown (class 3).

Three graph modes:
  - "tx"      : transaction graph (183 features vs 165 in the old release)
  - "actors"  : wallet-address graph (illicit-actor detection task)
  - "merged"  : tx nodes + actor nodes in ONE graph with typed edges
                (0 = tx-tx, 1 = addr-addr, 2 = tx-addr); task = illicit
                actor detection with transaction context.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .config import Config


def _load_pp_csv(cfg: Config, name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(cfg.raw_dir / "ellipticpp" / name, **kw)


def _symmetrize(src_t: torch.Tensor, dst_t: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cat([src_t, dst_t]), torch.cat([dst_t, src_t])], dim=0)


def build_tx_graph_pp(cfg: Config) -> Data:
    """Elliptic++ transactions graph."""
    feats = _load_pp_csv(cfg, "txs_features.csv")
    classes = _load_pp_csv(cfg, "txs_classes.csv")
    edges = _load_pp_csv(cfg, "txs_edgelist.csv")

    feats = feats.sort_values("Time step").reset_index(drop=True)
    txids = feats["txId"].astype(np.int64)
    txid2idx = {tx: i for i, tx in enumerate(txids)}

    feat_cols = [c for c in feats.columns if c not in ("txId", "Time step")]
    x = torch.as_tensor(feats[feat_cols].to_numpy(dtype=np.float32))
    t = torch.as_tensor(feats["Time step"].to_numpy(dtype=np.int64))

    y = (
        classes.set_index(classes["txId"].astype(np.int64))["class"]
        .astype(str)
        .map({"1": 1, "2": 0})
        .reindex(txids)
        .to_numpy(dtype=np.float32)
    )

    src = edges["txId1"].map(txid2idx)
    dst = edges["txId2"].map(txid2idx)
    keep = src.notna() & dst.notna()
    src_t = torch.as_tensor(src[keep].to_numpy(dtype=np.int64))
    dst_t = torch.as_tensor(dst[keep].to_numpy(dtype=np.int64))

    data = Data(x=x, edge_index=_symmetrize(src_t, dst_t), y=torch.as_tensor(y), t=t)
    data.num_nodes = x.size(0)
    data.node_ids = txids.to_numpy()
    return data


def build_actor_graph(cfg: Config) -> Data:
    """Wallet-address graph: 822,942 nodes, 56 features, addr-addr edges."""
    feats = _load_pp_csv(cfg, "wallets_features.csv")
    classes = _load_pp_csv(cfg, "wallets_classes.csv")
    edges = _load_pp_csv(cfg, "AddrAddr_edgelist.csv")

    feats = feats.sort_values("Time step").reset_index(drop=True)
    addrs = feats["address"].astype(str).str.strip()
    addr2idx = {a: i for i, a in enumerate(addrs)}

    feat_cols = [c for c in feats.columns if c not in ("address", "Time step")]
    x = torch.as_tensor(feats[feat_cols].to_numpy(dtype=np.float32))
    t = torch.as_tensor(feats["Time step"].to_numpy(dtype=np.int64))

    y = (
        classes.set_index(classes["address"].astype(str).str.strip())["class"]
        .astype(str)
        .map({"1": 1, "2": 0})
        .reindex(addrs)
        .to_numpy(dtype=np.float32)
    )

    src = edges.iloc[:, 0].astype(str).str.strip().map(addr2idx)
    dst = edges.iloc[:, 1].astype(str).str.strip().map(addr2idx)
    keep = src.notna() & dst.notna()
    src_t = torch.as_tensor(src[keep].to_numpy(dtype=np.int64))
    dst_t = torch.as_tensor(dst[keep].to_numpy(dtype=np.int64))

    data = Data(x=x, edge_index=_symmetrize(src_t, dst_t), y=torch.as_tensor(y), t=t)
    data.num_nodes = x.size(0)
    data.node_ids = addrs.to_numpy()
    return data

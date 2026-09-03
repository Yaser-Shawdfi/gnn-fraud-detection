"""Heterogeneous merged graph (tx + actors) and its model.

Node types: "tx" (182 features) and "actor" (55 features).
Edge relations: ("tx","money","tx"), ("actor","interacts","actor"),
                ("tx","flows","actor")  [TxAddr, symmetrized]
Task: classify ACTOR nodes (illicit vs licit).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from .config import Config


def _sym(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)


def build_merged_hetero(cfg: Config) -> HeteroData:
    from .data_loader_pp import _load_pp_csv

    data = HeteroData()

    # ---------------- transaction nodes ----------------
    feats = _load_pp_csv(cfg, "txs_features.csv")
    classes = _load_pp_csv(cfg, "txs_classes.csv")
    txedges = _load_pp_csv(cfg, "txs_edgelist.csv")
    txaddr = _load_pp_csv(cfg, "TxAddr_edgelist.csv")

    feats = feats.sort_values("Time step").reset_index(drop=True)
    txids = feats["txId"].astype(np.int64)
    txid2idx = {tx: i for i, tx in enumerate(txids)}
    tx_feat_cols = [c for c in feats.columns if c not in ("txId", "Time step")]
    data["tx"].x = torch.as_tensor(feats[tx_feat_cols].to_numpy(dtype=np.float32))
    data["tx"].t = torch.as_tensor(feats["Time step"].to_numpy(dtype=np.int64))
    tx_y = (
        classes.set_index(classes["txId"].astype(np.int64))["class"]
        .astype(str)
        .map({"1": 1, "2": 0})
        .reindex(txids)
        .to_numpy(dtype=np.float32)
    )
    data["tx"].y = torch.as_tensor(tx_y)
    n_tx = len(txids)
    data["tx"].num_nodes = n_tx

    # ---------------- actor nodes ----------------
    afeats = _load_pp_csv(cfg, "wallets_features.csv")
    aclasses = _load_pp_csv(cfg, "wallets_classes.csv")
    afeats = afeats.sort_values("Time step").reset_index(drop=True)
    addrs = afeats["address"].astype(str).str.strip()
    addr2idx = {a: i for i, a in enumerate(addrs)}
    a_feat_cols = [c for c in afeats.columns if c not in ("address", "Time step")]
    data["actor"].x = torch.as_tensor(afeats[a_feat_cols].to_numpy(dtype=np.float32))
    data["actor"].t = torch.as_tensor(afeats["Time step"].to_numpy(dtype=np.int64))
    ay = (
        aclasses.set_index(aclasses["address"].astype(str).str.strip())["class"]
        .astype(str)
        .map({"1": 1, "2": 0})
        .reindex(addrs)
        .to_numpy(dtype=np.float32)
    )
    data["actor"].y = torch.as_tensor(ay)
    data["actor"].num_nodes = len(addrs)

    # ---------------- edges ----------------
    s = txedges["txId1"].map(txid2idx)
    d = txedges["txId2"].map(txid2idx)
    k = s.notna() & d.notna()
    data["tx", "money", "tx"].edge_index = _sym(
        torch.as_tensor(s[k].to_numpy(dtype=np.int64)),
        torch.as_tensor(d[k].to_numpy(dtype=np.int64)),
    )

    aa = _load_pp_csv(cfg, "AddrAddr_edgelist.csv")
    s = aa["input_address"].astype(str).str.strip().map(addr2idx)
    d = aa["output_address"].astype(str).str.strip().map(addr2idx)
    k = s.notna() & d.notna()
    data["actor", "interacts", "actor"].edge_index = _sym(
        torch.as_tensor(s[k].to_numpy(dtype=np.int64)),
        torch.as_tensor(d[k].to_numpy(dtype=np.int64)),
    )

    s = txaddr["txId"].map(txid2idx)
    d = txaddr["output_address"].astype(str).str.strip().map(addr2idx)
    k = s.notna() & d.notna()
    # Bipartite relation: keep DIRECTIONAL (tx -> actor). Symmetrizing would
    # put actor-range indices into a tx-typed conv and crash with OOB asserts.
    data["tx", "flows", "actor"].edge_index = torch.stack(
        [
            torch.as_tensor(s[k].to_numpy(dtype=np.int64)),
            torch.as_tensor(d[k].to_numpy(dtype=np.int64)),
        ],
        dim=0,
    )

    return data


class HeteroGNN(torch.nn.Module):
    """Per-type input encoders -> HeteroConv (SAGE) layers -> actor logits."""

    def __init__(
        self, tx_in_dim: int, actor_in_dim: int, hidden_dim: int, num_layers: int, dropout: float
    ):
        super().__init__()
        self.tx_enc = torch.nn.Linear(tx_in_dim, hidden_dim)
        self.actor_enc = torch.nn.Linear(actor_in_dim, hidden_dim)
        rels = [("tx", "money", "tx"), ("actor", "interacts", "actor"), ("tx", "flows", "actor")]
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({rel: SAGEConv(hidden_dim, hidden_dim) for rel in rels})
            self.convs.append(conv)
        self.head = torch.nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        h = {
            "tx": F.relu(self.tx_enc(x_dict["tx"])),
            "actor": F.relu(self.actor_enc(x_dict["actor"])),
        }
        for conv in self.convs:
            h = conv(h, edge_index_dict)
            h = {
                k: F.dropout(F.relu(v), p=self.dropout, training=self.training)
                for k, v in h.items()
            }
        return self.head(h["actor"]).squeeze(-1)


def hetero_masks(data: HeteroData, cfg: Config) -> dict[str, torch.Tensor]:
    """Temporal masks on the ACTOR node type (the classification target)."""

    # reuse the same temporal logic on the actor node type
    t = data["actor"].t
    labeled = ~torch.isnan(data["actor"].y)
    return {
        "train": labeled & (t <= cfg.split.train_max_t),
        "val": labeled & (t > cfg.split.train_max_t) & (t <= cfg.split.val_max_t),
        "test": labeled & (t > cfg.split.val_max_t),
    }

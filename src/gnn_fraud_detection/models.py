"""Model definitions: MLP baseline + 4 GNN architectures.

All models share the same interface: forward(x, edge_index) -> logits [N, 1].
The MLP ignores edge_index, giving an honest "no-graph" baseline.

Design (standard for Elliptic benchmarks):
  input(166) -> [Conv -> ReLU -> Dropout] x L -> Conv(->1 logit)
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv


class MLP(nn.Module):
    """2-3 layer MLP on raw node features. Ignores graph structure."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x, edge_index=None):
        return self.net(x).squeeze(-1)


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, 1))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index).squeeze(-1)


class SAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, 1))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index).squeeze(-1)


class GAT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        heads: int = 4,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        # Hidden layers use heads * (hidden_dim // heads) so concat == hidden_dim
        h = hidden_dim // heads
        self.convs.append(GATConv(in_dim, h, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(h * heads, h, heads=heads, dropout=dropout))
        # Final layer: averaged (not concatenated) multi-head output, 1 logit
        self.convs.append(GATConv(h * heads, 1, heads=heads, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index).squeeze(-1)


class GIN(nn.Module):
    """GIN with sum aggregation and epsilon learning (gin_train_eps)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        train_eps: bool = True,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(
            GINConv(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                train_eps=train_eps,
            )
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    ),
                    train_eps=train_eps,
                )
            )
        self.convs.append(
            GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                ),
                train_eps=train_eps,
            )
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index).squeeze(-1)


def build_model(name: str, in_dim: int, mc) -> nn.Module:
    """Factory used by training/eval code. `mc` = ModelConfig."""
    if name == "mlp":
        return MLP(in_dim, mc.hidden_dim, mc.num_layers, mc.dropout)
    if name == "gcn":
        return GCN(in_dim, mc.hidden_dim, mc.num_layers, mc.dropout)
    if name == "graphsage":
        return SAGE(in_dim, mc.hidden_dim, mc.num_layers, mc.dropout)
    if name == "gat":
        return GAT(in_dim, mc.hidden_dim, mc.num_layers, mc.dropout, heads=mc.heads)
    if name == "gin":
        return GIN(in_dim, mc.hidden_dim, mc.num_layers, mc.dropout, train_eps=mc.gin_train_eps)
    raise ValueError(f"Unknown model: {name}")

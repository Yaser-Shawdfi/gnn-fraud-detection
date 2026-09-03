"""Shared fixtures: tiny synthetic graph shaped like Elliptic (166 features)."""

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gnn_fraud_detection.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def tiny_graph():
    """200-node graph, 166 features, ~25% illicit, 3 time steps."""
    g = torch.Generator().manual_seed(0)
    n, in_dim = 200, 166
    x = torch.randn(n, in_dim, generator=g)

    # illicit node blocks per split so every labeled split has both classes
    illicit = torch.zeros(n, dtype=torch.bool)
    illicit[50:100] = True  # train illicit
    illicit[125:150] = True  # val illicit
    illicit[175:200] = True  # test illicit
    x[illicit] += 0.8  # shifted means so a linear model CAN fit
    ei = torch.randint(0, n, (2, 1200), generator=g)
    ei = torch.unique(ei, dim=1)

    y = torch.zeros(n)
    y[illicit] = 1.0

    t = torch.empty(n, dtype=torch.long)
    t[:100] = torch.randint(1, 34, (100,), generator=g)  # train range
    t[100:150] = torch.randint(34, 42, (50,), generator=g)  # val range
    t[150:] = torch.randint(42, 50, (50,), generator=g)  # test range

    from torch_geometric.data import Data

    data = Data(x=x, edge_index=ei, y=y, t=t)
    data.num_nodes = n
    return data


@pytest.fixture(scope="session")
def cfg():
    c = load_config({"training.epochs": 5, "training.patience": 5, "mlflow.enabled": False})
    # redirect writes into a temp reports dir during tests
    return c


@pytest.fixture(scope="session")
def masks(tiny_graph, cfg):
    from gnn_fraud_detection.splits import build_masks

    return build_masks(tiny_graph, cfg)

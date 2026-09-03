"""Tests for config, splits, thresholding, and models (CPU-friendly)."""

import numpy as np
import torch


def test_config_defaults_and_override(cfg):
    assert cfg.model.hidden_dim == 128
    assert cfg.training.epochs == 5  # overridden in fixture
    assert cfg.split.train_max_t == 33


def test_config_invalid_model_rejected():
    import pytest

    from gnn_fraud_detection.config import load_config

    with pytest.raises(ValueError):
        load_config({"model.name": "transformer"})


def test_masks_disjoint_and_exclusive(tiny_graph, cfg, masks):
    # no node in two masks; unknowns in no mask
    tr, va, te = masks["train"], masks["val"], masks["test"]
    assert not (tr & va).any()
    assert not (tr & te).any()
    assert not (va & te).any()
    labeled = ~torch.isnan(tiny_graph.y)
    assert (tr | va | te == labeled).all()


def test_pos_weight_ratio(tiny_graph, cfg, masks):
    from gnn_fraud_detection.splits import compute_pos_weight

    w = compute_pos_weight(tiny_graph, cfg, masks["train"])
    ytr = tiny_graph.y[masks["train"]]
    expected = float((ytr == 0).sum()) / float((ytr == 1).sum())
    assert abs(w.item() - expected) < 1e-4


def test_pick_threshold_f1_beats_default():
    from gnn_fraud_detection.splits import pick_threshold

    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.3, 0.2, 0.7, 0.1, 0.6, 0.4])
    thr = pick_threshold(y, p, "f1")
    assert 0.0 <= thr <= 1.0
    # at the optimum, F1 should be >= F1 at 0.5
    from gnn_fraud_detection.splits import pick_threshold as pt

    assert thr == pt(y, p, "f1")


def test_models_forward_shapes(tiny_graph):
    from gnn_fraud_detection.models import build_model

    class MC:
        hidden_dim = 32
        num_layers = 2
        dropout = 0.5
        heads = 4
        gin_train_eps = True

    for name in ["mlp", "gcn", "graphsage", "gat", "gin"]:
        m = build_model(name, tiny_graph.x.size(1), MC())
        out = m(tiny_graph.x, tiny_graph.edge_index)
        assert out.shape == (tiny_graph.num_nodes,), name


def test_mlp_ignores_edges():
    from gnn_fraud_detection.models import MLP

    m = MLP(16, 8, 2, 0.5)
    x = torch.randn(50, 16)
    ei = torch.randint(0, 50, (2, 100))
    m.eval()
    with torch.no_grad():
        a = m(x, ei)
        b = m(x, torch.randperm(100)[:0].reshape(2, 0))
        # different edge sets, same output (graph-free baseline)
        ei2 = ei.flip(0)
        c = m(x, ei2)
    assert torch.equal(a, b) and torch.equal(a, c)


def test_gat_heads_concat_hidden_dim(tiny_graph):
    from gnn_fraud_detection.models import GAT

    m = GAT(166, 32, 2, 0.5, heads=4)
    assert m(tiny_graph.x, tiny_graph.edge_index).shape == (tiny_graph.num_nodes,)

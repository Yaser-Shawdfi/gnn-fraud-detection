"""End-to-end training smoke test on the tiny synthetic graph (CPU, seconds)."""

import torch


def test_train_and_eval_e2e(tiny_graph, cfg, masks):
    from gnn_fraud_detection.evaluate import evaluate_model
    from gnn_fraud_detection.models import build_model
    from gnn_fraud_detection.train import train_model

    torch.manual_seed(0)
    model = build_model("gcn", tiny_graph.x.size(1), cfg.model)
    history, artifacts = train_model(model, tiny_graph, masks, cfg)

    assert len(history["epoch"]) >= 1
    assert artifacts["best_state"] is not None
    assert 0.0 <= artifacts["threshold"] <= 1.0
    assert artifacts["best_epoch"] > 0

    metrics = evaluate_model(model, tiny_graph, masks, artifacts["threshold"], device="cpu")
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert metrics["n_test"] == 50
    assert set(metrics["confusion"]) == {"tn", "fp", "fn", "tp"}


def test_focal_loss_runs():
    from gnn_fraud_detection.train import FocalLoss

    logits = torch.randn(64)
    targets = torch.randint(0, 2, (64,)).float()
    loss = FocalLoss(gamma=2.0, pos_weight=torch.tensor([3.0]))(logits, targets)
    assert loss.dim() == 0 and torch.isfinite(loss)

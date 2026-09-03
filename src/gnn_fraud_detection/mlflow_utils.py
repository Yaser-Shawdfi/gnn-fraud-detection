"""MLflow experiment tracking (optional, on by default).

MLflow 3.x notes baked in:
  - sqlite tracking URI (file backend is deprecated)
  - model logging via runs:/ URI when registering
"""

from __future__ import annotations

import mlflow

from .config import Config


def setup_mlflow(cfg: Config):
    """Returns an active-run context helper or None if disabled."""
    if not cfg.mlflow.enabled:
        return None
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    return mlflow


def flatten_params(cfg: Config) -> dict:
    """Config -> flat param dict for mlflow.log_params (bounded lengths)."""
    return {
        "model": cfg.model.name,
        "hidden_dim": cfg.model.hidden_dim,
        "num_layers": cfg.model.num_layers,
        "dropout": cfg.model.dropout,
        "heads": cfg.model.heads,
        "epochs": cfg.training.epochs,
        "lr": cfg.training.lr,
        "weight_decay": cfg.training.weight_decay,
        "patience": cfg.training.patience,
        "seed": cfg.training.seed,
        "focal_gamma": cfg.training.focal_gamma,
        "directed": cfg.graph.directed,
        "pos_weight_cfg": cfg.split.pos_weight,
        "threshold_strategy": cfg.evaluation.threshold_strategy,
    }

"""Config loader: YAML defaults + CLI/env overrides.

Usage:
    from gnn_fraud_detection.config import load_config
    cfg = load_config()                     # defaults only
    cfg = load_config(model_name="gcn")     # CLI-style overrides
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"

VALID_MODELS = ("gcn", "gat", "graphsage", "gin", "mlp", "heterosage")


@dataclass
class DataConfig:
    raw_dir: Path
    processed_dir: Path
    checkpoint_dir: Path
    reports_dir: Path
    dataset: str = "elliptic"
    pp_mode: str = "actors"
    features_file: str = "elliptic_txs_features.csv"
    edgelist_file: str = "elliptic_txs_edgelist.csv"
    classes_file: str = "elliptic_txs_classes.csv"


@dataclass
class GraphConfig:
    directed: bool = False
    num_timesteps: int = 49


@dataclass
class SplitConfig:
    train_max_t: int = 33
    val_max_t: int = 41
    pos_weight: Optional[float] = None


@dataclass
class ModelConfig:
    name: str = "gat"
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.6
    heads: int = 4
    gin_train_eps: bool = True


@dataclass
class TrainingConfig:
    seed: int = 42
    device: str = "cuda"
    epochs: int = 200
    lr: float = 0.005
    weight_decay: float = 0.0005
    batch_size: Optional[int] = None
    patience: int = 30
    max_grad_norm: float = 5.0
    focal_gamma: Optional[float] = None


@dataclass
class MinibatchConfig:
    batch_size: int = 2048
    num_neighbors: str = "15,10"  # 1-hop, 2-hop sampling budgets
    num_workers: int = 0


@dataclass
class EvaluationConfig:
    threshold_strategy: str = "f1"


@dataclass
class MLflowConfig:
    enabled: bool = True
    tracking_uri: str = "sqlite:///./data/mlflow.db"
    experiment_name: str = "gnn-fraud-detection"


@dataclass
class Config:
    data: DataConfig
    graph: GraphConfig
    split: SplitConfig
    model: ModelConfig
    training: TrainingConfig
    minibatch: MinibatchConfig
    evaluation: EvaluationConfig
    mlflow: MLflowConfig
    root: Path = PROJECT_ROOT

    # Convenience accessors -------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.root / self.data.raw_dir

    @property
    def processed_dir(self) -> Path:
        return self.root / self.data.processed_dir

    @property
    def checkpoint_dir(self) -> Path:
        return self.root / self.data.checkpoint_dir

    @property
    def reports_dir(self) -> Path:
        return self.root / self.data.reports_dir

    def makedirs(self) -> None:
        for d in (self.processed_dir, self.checkpoint_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)


def _apply_override(cfg_dict: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg_dict
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def _deep_set_from_env(cfg_dict: dict) -> None:
    """Env overrides: GFD_MODEL=gcn GFD_TRAIN_EPOCHS=50 -> model.name / training.epochs"""
    prefix = "GFD_"
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        dotted = env_key[len(prefix) :].lower().replace("_", ".")
        # coerce common literals
        if env_val.lower() in ("true", "false"):
            val: Any = env_val.lower() == "true"
        else:
            try:
                val = int(env_val)
            except ValueError:
                try:
                    val = float(env_val)
                except ValueError:
                    if env_val.lower() == "null" or env_val == "":
                        val = None
                    else:
                        val = env_val
        _apply_override(cfg_dict, dotted, val)


def load_config(overrides: Optional[dict[str, Any]] = None) -> Config:
    """Load YAML config and apply overrides (dict of dotted keys)."""
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw = copy.deepcopy(raw)
    _deep_set_from_env(raw)
    if overrides:
        for dotted_key, value in overrides.items():
            _apply_override(raw, dotted_key, value)

    model_name = str(raw.get("model", {}).get("name", "gat")).lower()
    if model_name not in VALID_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Valid: {VALID_MODELS}")

    cfg = Config(
        data=DataConfig(**raw["data"]),
        graph=GraphConfig(**raw["graph"]),
        split=SplitConfig(**raw["split"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        minibatch=MinibatchConfig(**(raw.get("minibatch") or {})),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        mlflow=MLflowConfig(**raw["mlflow"]),
        root=PROJECT_ROOT,
    )
    cfg.makedirs()
    return cfg

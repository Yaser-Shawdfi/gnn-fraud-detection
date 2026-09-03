"""Feature preprocessing: standardization fit on the train period only.

The Elliptic raw features contain extreme outlier values (BTC amounts up to
~1e19 in the aggregated neighborhood features). Without standardization,
full-batch training immediately produces NaN losses.

Standard practice for this dataset (Weber et al. 2019, Elliptic++ papers):
z-score features using statistics from the training time window. We use ALL
nodes with t <= train_max_t (labeled + unknown) - mild transductive use of
feature values only, never labels.
"""
from __future__ import annotations

import torch

from .config import Config


def scale_features(data, cfg: Config) -> None:
    """In-place standardization of data.x using train-period statistics.

    Adds `x_mean` / `x_std` attributes to the Data object for provenance.
    """
    t = data.t
    train_period = t <= cfg.split.train_max_t
    if train_period.sum() < 2:
        train_period = torch.ones(data.num_nodes, dtype=torch.bool)

    x_tr = data.x[train_period]
    mean = x_tr.mean(dim=0)
    std = x_tr.std(dim=0).clamp_min(1e-8)

    data.x = (data.x - mean) / std
    data.x_mean = mean
    data.x_std = std


def verify_finite(data) -> bool:
    return bool(torch.isfinite(data.x).all())
"""Model explanation with Captum (Integrated Gradients + saliency).

Attribution runs on the k-hop SUBGRAPH around each explained node, not the
full 203k-node graph: IntegratedGradients batches its interpolation steps
through the leading dimension, which would otherwise corrupt edge_index and
explode memory. A 2-hop subgraph is the local receptive field of a 2-layer
GNN, so attributions on it are exact for the node's prediction.
"""

from __future__ import annotations

import numpy as np
import torch
from captum.attr import IntegratedGradients, Saliency
from torch_geometric.utils import k_hop_subgraph

from .config import Config


def explain_nodes(
    model,
    data,
    node_idx: int | np.ndarray,
    cfg: Config,
    method: str = "integrated_gradients",
    n_steps: int = 32,
    num_hops: int = 2,
) -> dict:
    """Feature attribution for one or more nodes (k-hop subgraph IG/saliency).

    Args:
        node_idx: single index or array of indices into data.x rows.
        method: 'integrated_gradients' or 'saliency'.

    Returns:
        {"node_idx": [...], "attributions": np.ndarray [n_nodes, F],
         "method": str, "n_steps": int}
    """
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    ei_full = data.edge_index.to(device)

    targets = np.atleast_1d(node_idx)
    all_attrs = []

    for node in targets:
        subset, ei_sub, mapping, _ = k_hop_subgraph(
            int(node), num_hops, ei_full, relabel_nodes=True
        )
        pos = int(mapping[0])  # position of the target node inside the subgraph
        subset = subset.to(data.x.device)
        x_sub = data.x[subset].to(device).detach()
        ei_sub = ei_sub.to(device)

        def fwd(x_batched, _ei=ei_sub, _pos=pos):
            # Captum may batch interpolation steps along dim 0:
            # [B, n_sub, F] (IG) or [1, n_sub, F] (saliency).
            if x_batched.dim() == 2:
                x_batched = x_batched.unsqueeze(0)
            outs = []
            for i in range(x_batched.size(0)):
                logits = model(x_batched[i], _ei)
                outs.append(logits[_pos])
            return torch.stack(outs)

        x_in = x_sub.unsqueeze(0).requires_grad_(True)  # [1, n_sub, F] leaf

        if method == "integrated_gradients":
            algo = IntegratedGradients(fwd)
            attrs = algo.attribute(x_in, target=None, n_steps=n_steps)
        elif method == "saliency":
            algo = Saliency(fwd)
            attrs = algo.attribute(x_in, target=None)
        else:
            raise ValueError(f"Unknown attribution method: {method}")

        all_attrs.append(attrs[0, pos].detach().cpu().numpy())

    return {
        "node_idx": targets.tolist(),
        "attributions": np.stack(all_attrs, axis=0),
        "method": method,
        "n_steps": n_steps,
    }


def top_features(attributions: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
    """Return top-k (feature_name, signed_attribution) pairs for one node."""
    names = [f"feature_{i}" for i in range(1, attributions.shape[-1] + 1)]
    order = np.argsort(-np.abs(attributions))[:k]
    return [(names[i], float(attributions[i])) for i in order]


def aggregate_importance(attributions: np.ndarray) -> np.ndarray:
    """Global importance = mean |attribution| across explained nodes."""
    return np.abs(attributions).mean(axis=0)

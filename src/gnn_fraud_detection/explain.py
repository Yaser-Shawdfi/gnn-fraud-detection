"""Model explanation with Captum (Integrated Gradients + saliency).

Explains which of the 166 input features drive a node's illicit score.
Graph-level edges stay fixed; we attribute w.r.t. input node features.
"""

from __future__ import annotations

import numpy as np
import torch
from captum.attr import IntegratedGradients, Saliency

from .config import Config


def _forward_wrapper(model):
    def fwd(x_input, edge_index):
        # forward(x, edge_index) -> [N] logits; attribute all nodes at once
        return model(x_input, edge_index)

    return fwd


def explain_nodes(
    model,
    data,
    node_idx: int | np.ndarray,
    cfg: Config,
    method: str = "integrated_gradients",
    n_steps: int = 32,
) -> dict:
    """Feature attribution for one or more nodes.

    Args:
        node_idx: single index or array of indices into data.x rows.
        method: 'integrated_gradients' or 'saliency'.

    Returns:
        {"node_idx": [...], "attributions": np.ndarray [n_nodes, 166],
         "method": str}
    """
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    x = data.x.to(device).detach().requires_grad_(True)
    ei = data.edge_index.to(device)

    idx = torch.as_tensor(np.atleast_1d(node_idx), dtype=torch.long, device=device)

    if method == "integrated_gradients":
        algo = IntegratedGradients(_forward_wrapper(model))
        attrs = algo.attribute(
            (x,),
            forward_kwargs={"edge_index": ei},
            target=None,
            n_steps=n_steps,
            additional_forward_args=(),
        )
    elif method == "saliency":
        algo = Saliency(_forward_wrapper(model))
        attrs = algo.attribute(
            (x,),
            target=None,
            additional_forward_kwargs={"edge_index": ei},
        )
    else:
        raise ValueError(f"Unknown attribution method: {method}")

    attr = attrs[0].detach().cpu().numpy()
    return {
        "node_idx": np.atleast_1d(node_idx).tolist(),
        "attributions": attr[idx.cpu().numpy()],
        "method": method,
    }


def top_features(attributions: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
    """Return top-k (feature_name, signed_attribution) pairs for one node."""
    names = [f"feature_{i}" for i in range(1, attributions.shape[-1] + 1)]
    order = np.argsort(-np.abs(attributions))[:k]
    return [(names[i], float(attributions[i])) for i in order]


def aggregate_importance(attributions: np.ndarray) -> np.ndarray:
    """Global importance = mean |attribution| across explained nodes."""
    return np.abs(attributions).mean(axis=0)

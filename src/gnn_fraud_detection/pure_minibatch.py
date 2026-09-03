"""Pure-torch mini-batch training: dependency-free neighbor sampling.

Replaces PyG's NeighborLoader, which requires compiled torch-sparse/pyg-lib
(unavailable: data.pyg.org currently serves NODATA globally, so the prebuilt
wheels are unreachable, and there is no MSVC toolchain to build from source).

Sampling is fully vectorized over a CSR index built once (argsort +
searchsorted); a 2-hop sample of 4096 seeds takes ~25 ms on the 1.27M-node
actors graph.

Batch contract: the first `batch_size` rows of n_id are the seed nodes (the
labeled nodes we compute loss/predictions for); the rest are sampled context.
Conv layers run on (x_sub, edge_index_sub) with edges remapped to local ids.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import Config
from .models import build_model
from .train import make_loss_fn


class PureNeighborSampler:
    """Dependency-free 2-hop neighbor sampler (CSR + vectorized gather)."""

    def __init__(self, edge_index: torch.Tensor, num_nodes: int):
        order = edge_index[1].argsort(stable=True)
        self.src = edge_index[0][order].contiguous()
        self.dst = edge_index[1][order].contiguous()
        # arange must live on the same device as dst (cuda or cpu)
        self.row_ptr = torch.searchsorted(
            self.dst,
            torch.arange(num_nodes + 1, dtype=self.dst.dtype, device=self.dst.device),
        )
        self.edge_index = edge_index

    def one_hop(self, nodes: torch.Tensor, size: int) -> torch.Tensor:
        """Sample up to `size` in-neighbors per node; -1 where none exist."""
        starts = self.row_ptr[nodes]
        ends = self.row_ptr[nodes + 1]
        spans = (ends - starts).clamp_min(0)
        rand = torch.rand(nodes.numel(), size, device=nodes.device)
        off = (rand * spans.unsqueeze(1).float()).long()
        off = torch.where(spans.unsqueeze(1) > 0, off, torch.zeros_like(off))
        valid = spans > 0
        gather_idx = (starts.unsqueeze(1) + off).clamp_max(len(self.src) - 1)
        gathered = self.src[gather_idx]
        return torch.where(valid.unsqueeze(1), gathered, torch.full_like(gathered, -1))


class PureTrainer:
    """Mini-batch train/eval on sampled 2-hop subgraphs."""

    def __init__(self, data, cfg: Config, model_name: str):
        device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
        self.device = device
        self.cfg = cfg
        self.data = data
        self.data_gpu = data.to(device)
        self.sampler = PureNeighborSampler(data.edge_index, data.num_nodes)
        self.model = build_model(model_name, data.x.size(1), cfg.model).to(device)
        self.bs = cfg.minibatch.batch_size
        self.sizes = tuple(int(x) for x in str(cfg.minibatch.num_neighbors).split(","))

        labeled = ~torch.isnan(data.y)
        self.train_idx = (
            torch.nonzero(labeled & (data.t <= cfg.split.train_max_t)).flatten().to(device)
        )
        self.val_idx = (
            torch.nonzero(
                labeled & (data.t > cfg.split.train_max_t) & (data.t <= cfg.split.val_max_t)
            )
            .flatten()
            .to(device)
        )
        self.test_idx = torch.nonzero(labeled & (data.t > cfg.split.val_max_t)).flatten().to(device)

        n_pos = int(((data.y == 1) & (data.t <= cfg.split.train_max_t)).sum())
        n_neg = int(((data.y == 0) & (data.t <= cfg.split.train_max_t)).sum())
        pos_weight = torch.tensor([n_neg / n_pos if n_pos else 1.0], device=device)
        self.loss_fn = make_loss_fn(cfg, pos_weight)
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )

    def _subgraph(self, seeds: torch.Tensor):
        s = self.sampler
        n1 = s.one_hop(seeds, self.sizes[1])
        layer1 = torch.unique(torch.cat([seeds, n1.flatten()]))
        layer1 = layer1[layer1 >= 0]
        n2 = s.one_hop(layer1, self.sizes[0])
        layer0 = torch.unique(torch.cat([layer1, n2.flatten()]))
        layer0 = layer0[layer0 >= 0]

        # n_id layout: seeds first, then sampled context (deterministic)
        seed_of_l0 = torch.isin(layer0, seeds)
        n_id = torch.cat([layer0[seed_of_l0], layer0[~seed_of_l0]])
        # NOTE: if some seeds had no in-edges they still appear in layer0
        # (seeds themselves are in layer1 via torch.cat), and every seed is
        # in layer0, so n_id[:n] covers exactly the seeds - but ONLY if the
        # number of layer0 entries matching seeds == seeds.numel().
        # seeds are unique, torch.isin keeps one entry per seed: guaranteed.

        local = torch.full((self.data.num_nodes,), -1, dtype=torch.long, device=n_id.device)
        local[n_id] = torch.arange(n_id.numel(), device=n_id.device)
        ei = self.sampler.edge_index
        keep = (local[ei[0]] >= 0) & (local[ei[1]] >= 0)
        ei_sub = torch.stack([local[ei[0]][keep], local[ei[1]][keep]], dim=0)
        return n_id, ei_sub

    def _forward_logits(self, seeds: torch.Tensor) -> torch.Tensor:
        n_id, ei_sub = self._subgraph(seeds)
        return self.model(self.data.x[n_id].to(self.device), ei_sub)

    def train_epoch(self) -> float:
        self.model.train()
        perm = self.train_idx[torch.randperm(self.train_idx.numel())]
        total, seen = 0.0, 0
        for i in range(0, perm.numel(), self.bs):
            seeds = perm[i : i + self.bs]
            if seeds.numel() < 8:
                continue
            logits = self._forward_logits(seeds)
            loss = self.loss_fn(logits[: seeds.numel()], self.data.y[seeds].float())
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.max_grad_norm)
            self.opt.step()
            total += float(loss.item()) * seeds.numel()
            seen += seeds.numel()
        return total / max(seen, 1)

    @torch.no_grad()
    def predict(self, idx: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        ys, ps = [], []
        for i in range(0, idx.numel(), self.bs):
            seeds = idx[i : i + self.bs]
            if seeds.numel() < 4:
                continue
            logits = self._forward_logits(seeds)
            ys.append(self.data.y[seeds].float().cpu().numpy())
            ps.append(torch.sigmoid(logits[: seeds.numel()]).cpu().numpy())
        if not ys:
            return np.array([]), np.array([])
        return np.concatenate(ys), np.concatenate(ps)

    def run(self) -> tuple[dict, dict]:
        history = {"epoch": [], "loss": [], "val_auc": [], "val_ap": [], "val_score": []}
        best_score, best_state, best_epoch = -1.0, None, -1
        patience = self.cfg.training.patience
        t0 = time.time()

        for epoch in range(1, self.cfg.training.epochs + 1):
            ep_loss = self.train_epoch()
            yv, pv = self.predict(self.val_idx)
            try:
                va_auc = roc_auc_score(yv, pv)
                va_ap = average_precision_score(yv, pv)
            except ValueError:
                va_auc, va_ap = 0.5, 0.5
            score = 0.5 * (va_auc + va_ap)
            history["epoch"].append(epoch)
            history["loss"].append(ep_loss)
            history["val_auc"].append(va_auc)
            history["val_ap"].append(va_ap)
            history["val_score"].append(score)
            if score > best_score:
                best_score, best_epoch = score, epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience = self.cfg.training.patience
            else:
                patience -= 1
                if patience == 0:
                    break
            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"  epoch {epoch:3d} | loss {ep_loss:.4f} | val AUC {va_auc:.4f} "
                    f"| val AP {va_ap:.4f} | best {best_score:.4f}",
                    flush=True,
                )

        elapsed = time.time() - t0
        self.model.load_state_dict(best_state)
        yt, pt = self.predict(self.test_idx)
        pred = (pt >= 0.5).astype(int)
        metrics = {
            "roc_auc": float(roc_auc_score(yt, pt)),
            "pr_auc": float(average_precision_score(yt, pt)),
            "f1": float(f1_score(yt, pred)),
            "precision": float(precision_score(yt, pred)),
            "recall": float(recall_score(yt, pred)),
            "n_test": int(len(yt)),
        }
        return history, {"metrics": metrics, "train_seconds": elapsed, "best_epoch": best_epoch}

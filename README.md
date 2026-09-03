# GNN Bitcoin Fraud Detection

Graph neural networks for illicit Bitcoin transaction detection on the
[Elliptic dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
(Weber et al., 2019). Five architectures are trained under one identical
pipeline and compared on a strict temporal test split:

**MLP (no graph) vs GCN vs GAT vs GraphSAGE vs GIN**

## The dataset

| | |
|---|---|
| Nodes (transactions) | 203,769 |
| Edges (BTC payments) | 234,355 directed |
| Node features | 165 in this release (94 local tx stats + 71 aggregated one-hop stats; official docs list 166) |
| Time steps | 49 (two-week periods) |
| Labels | illicit 4,545 / licit 42,019 / unknown 157,205 |

Class balance among labeled nodes is roughly 1:10 illicit:licit.

## The temporal split (no leakage)

The labeled graph is split **by time step**, exactly like the original paper:

| Split | Time steps | Purpose |
|---|---|---|
| Train | t1 - t33 | fit weights |
| Validation | t34 - t41 | early stopping, threshold selection |
| Test | t42 - t49 | reported metrics |

This is stricter and more realistic than random splits: the model is deployed
on *future* transactions it has never seen from *any* neighbor.

## Architecture

```
raw CSVs -> build_graph() -> PyG Data (x[203769,166], edge_index, y, t)
         -> build_masks()  -> temporal train/val/test masks
         -> train_model()  -> full-batch training, BCEWithLogits(+pos_weight)
                              or FocalLoss, early stopping on val (AUC+AP)/2
         -> pick_threshold() on validation (max F1 or Youden's J)
         -> evaluate_model() on test: ROC-AUC, PR-AUC, F1, precision, recall
         -> Captum Integrated Gradients for per-node feature attribution
         -> MLflow tracks every run (params, metrics, artifacts)
```

## Quickstart

```bash
# 1. Environment (CUDA 12.8 wheels; adjust -index-url for your CUDA)
uv venv .venv --python 3.11
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv torch_geometric pandas numpy scikit-learn \
    matplotlib pyyaml tqdm mlflow captum

# 2. Download the 3 dataset CSVs into data/raw/
#    (Kaggle: ellipticco/elliptic-data-set, or the HF mirror
#     huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset)
#    elliptic_txs_features.csv / elliptic_txs_edgelist.csv / elliptic_txs_classes.csv

# 3. Build the graph, train one model, explain it
uv pip install --python .venv -e .[dev]
.venv/Scripts/gnn-fraud prepare
.venv/Scripts/gnn-fraud train --model gat
.venv/Scripts/gnn-fraud explain --model gat

# 4. Head-to-head comparison of all 5 architectures
.venv/Scripts/gnn-fraud compare

# 5. Inspect experiments
mlflow ui --backend-store-uri sqlite:///data/mlflow.db
```

Configuration lives in `config/settings.yaml`; override anything with CLI
flags (`--epochs 100`) or env vars (`GFD_MODEL=gcn`).

## Results (real, temporal test split t42-t49)

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Params |
|---|---|---|---|---|---|---|
| MLP (no graph) | 0.8254 | 0.2473 | 0.2709 | 0.1826 | 0.5245 | 21,377 |
| GCN | 0.8143 | **0.4362** | **0.4670** | **0.5621** | 0.3995 | 21,377 |
| GAT | **0.8435** | 0.3673 | 0.3719 | 0.3460 | 0.4020 | 22,025 |
| GraphSAGE | 0.7991 | 0.3568 | 0.3076 | 0.2354 | 0.4436 | 42,625 |
| GIN | 0.7719 | 0.3747 | 0.4023 | 0.4861 | 0.3431 | 54,403 |

Reproduce with `.venv/Scripts/gnn-fraud compare`; full table also lands in
`reports/model_comparison.csv` with thresholds, FPR, best epochs, train times.

**How to read this table.** On a temporal split (train on past periods, test
on *future* ones) these numbers are honest and hard - papers reporting
0.93-0.97 ROC-AUC on Elliptic use random splits that leak neighborhood
information across time. The signal is in PR-AUC and F1, which matter for
imbalanced fraud detection: every GNN beats the MLP's 0.25 PR-AUC by a wide
margin (GCN reaches 0.44 with the best precision 0.56 and F1 0.47). The MLP's
decent ROC-AUC comes from the 71 aggregated one-hop features it gets for
free - but it cannot exploit multi-hop structure, which is exactly what
shows up in the precision/F1 gap. GCN was still improving at the 200-epoch
cap (best_epoch=199), so more training or a deeper GCN is the clear next
tuning lever.

## Project layout

```
gnn-fraud-detection/
├── config/settings.yaml          # all hyperparameters, single source of truth
├── src/gnn_fraud_detection/
│   ├── config.py                 # YAML + env + CLI overrides (dataclasses)
│   ├── data_loader.py            # CSVs -> PyG Data (label alignment by txId)
│   ├── splits.py                 # temporal masks, pos_weight, threshold picker
│   ├── models.py                 # MLP / GCN / GAT / GraphSAGE / GIN
│   ├── train.py                  # full-batch loop, early stopping, FocalLoss
│   ├── evaluate.py               # test metrics, ROC/PR plots, curves
│   ├── explain.py                # Captum IG + saliency attribution
│   ├── mlflow_utils.py           # experiment tracking helpers
│   └── cli.py                    # prepare / train / compare / explain
├── tests/                        # pytest: masks, models, e2e smoke, focal loss
├── data/raw|processed|checkpoints
└── reports/                      # comparison CSV, curves, ROC/PR, explanations
```

## Notes and limitations

- Full-batch training (~200k nodes) fits in 16 GB VRAM; for much larger graphs
  switch to `NeighborLoader` mini-batching.
- Class weighting (`pos_weight = n_neg/n_pos`) is the default imbalance
  handler; `--focal-gamma 2.0` switches to focal loss for experiments.
- The 72 aggregated features already encode one-hop neighborhood statistics,
  which is why even the MLP is decent - but it cannot capture multi-hop and
  higher-order patterns, which is exactly what the GNNs exploit.
- `unknown`-class nodes are excluded from all splits; semi-supervised
  transductive tricks (using unknown nodes in message passing) are possible
  follow-ups, as are EvolveGCN / temporal GNNs.
- Educational/portfolio project - not an AML production system.

## References

- Weber, M. et al. (2019). *Anti-Money Laundering in Bitcoin: Experimenting
  with Graph Convolutional Networks for Financial Forensics.* KDD 2019.
- Elliptic++ (Bellei et al., 2021) - extended dataset with actor addresses.
- PyTorch Geometric documentation.
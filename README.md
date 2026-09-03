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

# 2. Datasets into data/raw/
#    Elliptic  : 3 CSVs (Kaggle ellipticco/elliptic-data-set or the HF mirror
#                huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset)
#    Elliptic++: 8 CSVs from github.com/git-disl/EllipticPlusPlus
#                (Transactions Dataset/ and Actors Dataset/ folders ->
#                 put flat into data/raw/ellipticpp/)

# 3. Build graphs, train, compare (classic dataset)
uv pip install --python .venv -e .[dev]
gnn-fraud prepare
gnn-fraud train --model gat
gnn-fraud compare
gnn-fraud explain --model gat

# 4. Elliptic++ (KDD'23): actors task + heterogeneous merged graph
gnn-fraud prepare --dataset ellipticpp --mode actors
gnn-fraud compare --dataset ellipticpp --mode actors
gnn-fraud prepare --dataset ellipticpp --mode merged
gnn-fraud train --model heterosage --dataset ellipticpp --mode merged

# 5. Inspect experiments
mlflow ui --backend-store-uri sqlite:///data/mlflow.db
```

Configuration lives in `config/settings.yaml`; override anything with CLI
flags (`--epochs 100`) or env vars (`GFD_MODEL=gcn`).

## Results (real, temporal test split t42-t49)

### Elliptic (classic release, 165-feature transaction graph)

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Params |
|---|---|---|---|---|---|---|
| MLP (no graph) | 0.8254 | 0.2473 | 0.2709 | 0.1826 | 0.5245 | 21,377 |
| GCN | 0.8143 | **0.4362** | **0.4670** | **0.5621** | 0.3995 | 21,377 |
| GAT | **0.8435** | 0.3673 | 0.3719 | 0.3460 | 0.4020 | 22,025 |
| GraphSAGE | 0.7991 | 0.3568 | 0.3076 | 0.2354 | 0.4436 | 42,625 |
| GIN | 0.7719 | 0.3747 | 0.4023 | 0.4861 | 0.3431 | 54,403 |

### Elliptic++ actors graph (wallet-address detection, 1.27M interaction nodes)

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| MLP | 0.7247 | 0.1347 | 0.1337 | 0.1462 | 0.1232 |
| GCN | 0.6825 | 0.1273 | 0.0883 | 0.0937 | 0.0834 |
| GAT | 0.7090 | 0.1252 | 0.2012 | 0.1654 | 0.2570 |
| **GraphSAGE** | **0.8484** | **0.3476** | **0.3111** | **0.2175** | **0.5462** |
| GIN | 0.7151 | 0.1069 | 0.0056 | 0.0149 | 0.0035 |
| Hetero-SAGE (merged tx+actors, 7.0M typed edges) | 0.8448 | 0.2348 | 0.3347 | 0.2709 | 0.4379 |

**How to read these tables.** On a temporal split (train on past periods, test
on *future* ones) these numbers are honest and hard - papers reporting
0.93-0.97 ROC-AUC on Elliptic use random splits that leak neighborhood
information across time. The signal is in PR-AUC and F1, which matter for
imbalanced fraud detection:

- **Transactions graph**: every GNN beats the MLP's 0.25 PR-AUC (GCN reaches
  0.44 with the best precision 0.56 and F1 0.47).
- **Architecture-task interaction**: GCN/GAT win the transactions graph, but
  **GraphSAGE dominates the actors graph** (0.85 ROC / 0.35 PR vs <0.71 ROC
  for GCN/GAT there) - mean-aggregation generalizes better on the actor
  graph's noisy duplicated-interaction structure where attention overfits.
- **Heterogeneous merged graph** (tx context + actors, HeteroConv SAGE):
  matches actors-only ROC but does not beat it on PR-AUC - the wallet
  features already carry most of the signal; tx context helps recall.
- The MLP's decent ROC-AUC comes from the aggregated neighborhood features
  it gets for free - the precision/F1 gap is where graph structure pays off.

### Robustness and temporal analysis

**Focal loss (gamma=2.0) vs weighted BCE, tx graph:**

| Model | BCE PR-AUC | Focal PR-AUC | BCE F1 | Focal F1 |
|---|---|---|---|---|
| MLP | 0.2473 | **0.3780** | 0.2709 | **0.3863** |
| GCN | **0.4362** | 0.3229 | 0.4670 | 0.4050 |
| GAT | 0.3673 | 0.2903 | 0.3719 | 0.3590 |
| GraphSAGE | 0.3568 | 0.3735 | 0.3076 | 0.3943 |
| GIN | 0.3747 | **0.4387** | 0.4023 | **0.4315** |

Focal loss redistributes the wins: MLP and GIN jump substantially (GIN becomes
the tx-graph runner-up), GCN's early-stopping advantage under BCE shrinks.
Loss choice is a real hyperparameter on this dataset, not a footnote.

**Multi-seed (3 seeds: 42/7/123), Elliptic++ actors graph, mean +/- std:**

| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| GraphSAGE | 0.8301 +/- 0.0166 | 0.2713 +/- 0.0667 | 0.2720 +/- 0.0347 |
| MLP | 0.7229 +/- 0.0040 | 0.1378 +/- 0.0113 | 0.1444 +/- 0.0164 |
| GAT | 0.7177 +/- 0.0156 | 0.1298 +/- 0.0082 | 0.2091 +/- 0.0157 |
| GIN | 0.7043 +/- 0.0086 | 0.1047 +/- 0.0074 | 0.0352 +/- 0.0444 |
| GCN | 0.6845 +/- 0.0021 | 0.1311 +/- 0.0037 | 0.0904 +/- 0.0027 |

GraphSAGE's dominance is real but high-variance on PR-AUC (0.27 +/- 0.07) -
it wins every seed, by how much varies. GCN/GAT genuinely collapse on this
graph (tight stds well below SAGE).

**GCN at 400 epochs, tx graph:** 0.8064 ROC / 0.3446 PR - worse than the
200-epoch run (0.8143/0.4362). The 200-epoch result was already past the
sweet spot; longer training overfits the train window on a temporal split.

**Temporal analysis (GCN, Elliptic++ actors, per test time step):**

| t | Static ROC | Incremental FT ROC | Static AP | Incr FT AP |
|---|---|---|---|---|
| 42 | 0.860 | **0.903** | 0.276 | 0.372 |
| 43 | 0.738 | 0.752 | 0.037 | 0.040 |
| 44 | 0.892 | 0.862 | 0.297 | 0.162 |
| 45 | 0.704 | 0.801 | 0.007 | 0.012 |
| 46 | **0.219** | 0.228 | 0.230 | 0.208 |
| 47 | 0.746 | 0.737 | 0.054 | 0.055 |
| 48 | 0.834 | **0.879** | 0.128 | 0.196 |
| 49 | 0.778 | **0.858** | 0.317 | **0.427** |
| mean | 0.723 | **0.753** | 0.166 | **0.184** |

Static models swing wildly between adjacent periods (t44: 0.89 vs t46: 0.22)
- per-period fraud behavior shifts fast. Periodic fine-tuning as labels
arrive lifts mean ROC +0.03 and mean PR-AUC +0.02, and wins or ties 5 of 8
steps - the deployed-system lesson: retrain often, the graph drifts. Full
per-step JSON + plot in `reports/temporal_gcn.{json,png}`.

Commands: `compare --focal-gamma 2.0`, `compare --seeds 42 7 123 ...`,
`temporal --model gcn --dataset ellipticpp --mode actors`.

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
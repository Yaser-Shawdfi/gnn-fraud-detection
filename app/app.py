"""Streamlit dashboard for GNN fraud detection.

Pages: Overview, Model Comparison, Predict & Explain, Dataset Explorer.
Run: .venv/Scripts/streamlit run app/app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st
import torch

st.set_page_config(page_title="GNN Bitcoin Fraud Detection", page_icon=None, layout="wide")

from gnn_fraud_detection.config import load_config  # noqa: E402

MODELS = ["mlp", "gcn", "gat", "graphsage", "gin"]
COMPARISON_CSV = PROJECT_ROOT / "reports" / "model_comparison.csv"
COMPARISON_CSV_PP = PROJECT_ROOT / "reports" / "model_comparison_ellipticpp_actors.csv"
# (label, processed file, comparison csv)
DATASETS = {
    "Elliptic (transactions)": ("elliptic.pt", COMPARISON_CSV),
    "Elliptic++ (actors)": ("ellipticpp_actors.pt", COMPARISON_CSV_PP),
    "Elliptic++ (tx)": ("ellipticpp_tx.pt", None),
}


@st.cache_resource
def load_hetero_bundle():
    """Merged hetero graph + Hetero-SAGE actor probabilities (cached)."""
    cfg = load_config()
    merged = cfg.processed_dir / "ellipticpp_merged.pt"
    ckpt = PROJECT_ROOT / "data" / "checkpoints" / "heterosage_merged.pt"
    if not merged.exists() or not ckpt.exists():
        return None
    from gnn_fraud_detection.hetero import HeteroGNN

    data = torch.load(merged, weights_only=False)
    # per-type z-score fit on train period (identical to the training path)
    for nt in ("tx", "actor"):
        m = data[nt].t <= cfg.split.train_max_t
        mu = data[nt].x[m].mean(dim=0)
        sd = data[nt].x[m].std(dim=0).clamp_min(1e-8)
        data[nt].x = torch.nan_to_num((data[nt].x - mu) / sd, nan=0.0)

    model = HeteroGNN(
        data["tx"].x.size(1),
        data["actor"].x.size(1),
        cfg.model.hidden_dim,
        cfg.model.num_layers,
        cfg.model.dropout,
    )
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    data_gpu = data.to(dev)
    with torch.no_grad():
        logits = model(
            {"tx": data_gpu["tx"].x, "actor": data_gpu["actor"].x},
            {rel: data_gpu[rel].edge_index for rel in data_gpu.edge_types},
        )
    probs = torch.sigmoid(logits).cpu().numpy()
    return data, probs  # graph kept on CPU for cheap subgraph slicing


def hetero_subgraph_figure(data_h, probs, actor_idx: int, max_nbrs: int = 30):
    """1-hop typed-edge neighborhood of one actor as a plotly star graph."""
    import plotly.graph_objects as go

    aa = data_h["actor", "interacts", "actor"].edge_index
    fa = data_h["tx", "flows", "actor"].edge_index
    a = actor_idx

    nbr_a = torch.cat([aa[0][aa[1] == a], aa[1][aa[0] == a]]).unique()
    nbr_t = fa[0][fa[1] == a].unique()
    if nbr_a.numel() > max_nbrs:
        nbr_a = nbr_a[torch.randperm(nbr_a.numel())[:max_nbrs]]
    if nbr_t.numel() > max_nbrs:
        nbr_t = nbr_t[torch.randperm(nbr_t.numel())[:max_nbrs]]
    na, ntx = nbr_a.tolist(), nbr_t.tolist()

    pos = {a: (0.0, 0.0)}
    for k, i in enumerate(na):
        th = 2 * np.pi * k / max(len(na), 1)
        pos[i] = (float(np.cos(th)), float(np.sin(th)))
    for k, i in enumerate(ntx):
        th = 2 * np.pi * (k + 0.5) / max(len(ntx), 1)
        pos[i] = (float(1.9 * np.cos(th)), float(1.9 * np.sin(th)))

    def edge_trace(pairs, color, name):
        xs, ys = [], []
        for s, d in pairs:
            x0, y0 = pos[s]
            x1, y1 = pos[d]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
        return go.Scatter(
            x=xs, y=ys, mode="lines", line=dict(width=1, color=color), name=name, hoverinfo="skip"
        )

    def _lab(v):
        return "illicit" if v == 1 else ("licit" if v == 0 else "unknown")

    def actor_nodes_trace(nodes):
        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]
        colors = [
            "#cc3311"
            if _lab(float(data_h["actor"].y[n])) == "illicit"
            else ("#228833" if _lab(float(data_h["actor"].y[n])) == "licit" else "#999999")
            for n in nodes
        ]
        sizes = [16 if n == actor_idx else 10 for n in nodes]
        text = [
            f"actor #{n} | t={int(data_h['actor'].t[n])} | "
            f"{_lab(float(data_h['actor'].y[n]))} | p={probs[n]:.3f}"
            for n in nodes
        ]
        return go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            name="actors",
            marker=dict(size=sizes, symbol="circle", color=colors),
            text=text,
            hoverinfo="text",
        )

    def tx_nodes_trace(nodes):
        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]
        text = [
            f"tx #{n} | t={int(data_h['tx'].t[n])} | {_lab(float(data_h['tx'].y[n]))}"
            for n in nodes
        ]
        return go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            name="transactions",
            marker=dict(size=9, symbol="square", color="#ee7733"),
            text=text,
            hoverinfo="text",
        )

    fig = go.Figure()
    fig.add_trace(edge_trace([(a, i) for i in na], "#33bbee", "actor-actor"))
    fig.add_trace(edge_trace([(a, i) for i in ntx], "#cc3311", "tx-actor"))
    fig.add_trace(actor_nodes_trace([a] + na))
    fig.add_trace(tx_nodes_trace(ntx))
    fig.update_layout(
        height=460,
        showlegend=True,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


@st.cache_data
def load_processed_graph(processed_file: str):
    from gnn_fraud_detection.preprocessing import scale_features

    cfg = load_config()
    data = torch.load(cfg.processed_dir / processed_file, weights_only=False)
    scale_features(data, cfg)
    return data, cfg


@st.cache_data
def load_comparison(csv_path: Path) -> pd.DataFrame | None:
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@st.cache_resource
def load_model(_data, model_name: str):
    from gnn_fraud_detection.models import build_model

    cfg = load_config()
    model = build_model(model_name, _data.x.size(1), cfg.model)
    ckpt = PROJECT_ROOT / "data" / "checkpoints" / f"{model_name}.pt"
    if not ckpt.exists():
        return None
    state = torch.load(ckpt, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    st.title("GNN Bitcoin Fraud Detection")
    st.caption(
        "Elliptic + Elliptic++ | temporal test split t42-t49 | "
        "MLP vs GCN vs GAT vs GraphSAGE vs GIN vs Hetero-SAGE"
    )

    ds_label = st.sidebar.selectbox("Dataset / graph", list(DATASETS.keys()), index=0)
    processed_file, comp_csv = DATASETS[ds_label]
    data, cfg = load_processed_graph(processed_file)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "Model Comparison", "Predict & Explain", "Dataset", "Hetero Explorer"]
    )

    # ------------------------------------------------------------------ #
    # Tab 1: Overview
    # ------------------------------------------------------------------ #
    with tab1:
        from gnn_fraud_detection.splits import build_masks

        masks = build_masks(data, cfg)
        y = data.y
        labeled = (~torch.isnan(y)).sum().item()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nodes", f"{data.num_nodes:,}")
        c2.metric("Edges (symmetrized)", f"{data.edge_index.size(1):,}")
        c3.metric("Labeled", f"{labeled:,}")
        c4.metric("Illicit", f"{int((y == 1).sum()):,}")

        st.markdown(
            "**Temporal protocol**: train t1-t33, validation t34-t41, "
            "test t42-t49. Unknown-class nodes are never in any split."
        )

        comp = load_comparison(comp_csv)
        if comp is not None:
            st.subheader("Test-split results")
            st.dataframe(comp, use_container_width=True, hide_index=True)

        curves = PROJECT_ROOT / "reports"
        for name in MODELS:
            p = curves / f"curves_{name}.png"
            if p.exists():
                with st.expander(f"{name.upper()} training curves"):
                    st.image(str(p))

    # ------------------------------------------------------------------ #
    # Tab 2: Model comparison bar chart
    # ------------------------------------------------------------------ #
    with tab2:
        comp = load_comparison(comp_csv)
        if comp is None:
            st.info(
                "Run the corresponding `gnn-fraud compare` command first to "
                "generate results for this dataset."
            )
        else:
            import plotly.express as px

            fig = px.bar(
                comp,
                x="model",
                y=["roc_auc", "pr_auc"],
                barmode="group",
                title="Test-split ROC-AUC and PR-AUC",
            )
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                "The MLP-vs-GNN gap is the evidence that graph structure adds "
                "signal beyond the raw features (which already include 72 "
                "one-hop neighborhood aggregates)."
            )

    # ------------------------------------------------------------------ #
    # Tab 3: Predict & Explain
    # ------------------------------------------------------------------ #
    with tab3:
        model_name = st.selectbox("Model", MODELS, index=2)
        model = load_model(data, model_name)
        if model is None:
            st.warning(f"No checkpoint for {model_name}. Train it first.")
        else:
            from gnn_fraud_detection.splits import build_masks

            masks = build_masks(data, cfg)
            test_idx = torch.nonzero(masks["test"]).flatten()
            y_test = data.y[test_idx]

            with torch.no_grad():
                logits = model(data.x, data.edge_index)
                probs = torch.sigmoid(logits)

            col_a, col_b = st.columns([2, 3])
            with col_a:
                label_filter = st.radio(
                    "Show", ["illicit", "licit", "all"], index=0, horizontal=True
                )
                k = st.slider("Top-k features", 3, 20, 8)
            with col_b:
                if label_filter == "illicit":
                    pool = test_idx[(y_test == 1).cpu().numpy()]
                elif label_filter == "licit":
                    pool = test_idx[(y_test == 0).cpu().numpy()]
                else:
                    pool = test_idx.cpu().numpy()
                node_id = st.selectbox(
                    "Transaction (test split)",
                    pool.tolist(),
                    format_func=lambda i: f"tx #{i}",
                )

            p = float(probs[node_id])
            y_true = int(data.y[node_id])
            st.metric(
                "Illicit probability",
                f"{p:.3f}",
                delta=f"actual: {'illicit' if y_true == 1 else 'licit'}",
            )

            if st.button("Explain this prediction (Captum IG)"):
                from gnn_fraud_detection.explain import explain_nodes, top_features

                with st.spinner("Integrated Gradients (32 steps)..."):
                    res = explain_nodes(
                        model, data, int(node_id), cfg, method="integrated_gradients"
                    )
                attr = res["attributions"][0]
                top = top_features(attr, k=k)
                names = [n for n, _ in top][::-1]
                vals = [v for _, v in top][::-1]

                import plotly.graph_objects as go

                fig = go.Figure(
                    go.Bar(
                        x=vals,
                        y=names,
                        orientation="h",
                        marker_color=["#cc3311" if v > 0 else "#004488" for v in vals],
                    )
                )
                fig.update_layout(
                    title=f"Top-{k} feature attributions (IG) for tx #{node_id}",
                    xaxis_title="attribution",
                    height=max(300, 28 * k + 80),
                )
                st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------ #
    # Tab 4: Dataset
    # ------------------------------------------------------------------ #
    with tab4:
        from gnn_fraud_detection.splits import build_masks

        masks = build_masks(data, cfg)
        y = data.y
        t = data.t

        st.markdown("### Label and time-step distribution")
        df_stats = pd.DataFrame(
            {
                "time step": range(1, 50),
                "illicit": [int(((y == 1) & (t == s)).sum()) for s in range(1, 50)],
                "licit": [int(((y == 0) & (t == s)).sum()) for s in range(1, 50)],
                "unknown": [int((torch.isnan(y) & (t == s)).sum()) for s in range(1, 50)],
            }
        )
        import plotly.express as px

        fig = px.bar(
            df_stats,
            x="time step",
            y=["illicit", "licit", "unknown"],
            barmode="stack",
            title="Class distribution across the 49 two-week time steps",
        )
        # shade the split boundaries
        fig.add_vrect(
            x0=33.5,
            x1=34.5,
            line_width=0,
            fillcolor="#33bbee",
            opacity=0.15,
            annotation_text="train | val",
        )
        fig.add_vrect(
            x0=41.5,
            x1=42.5,
            line_width=0,
            fillcolor="#cc3311",
            opacity=0.15,
            annotation_text="val | test",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "165 features per transaction in this release: 94 local (fee, version, "
            "R/B types, input/output counts, BTC amounts) + 71 aggregated one-hop "
            "neighborhood statistics. The official release documents 166; this "
            "mirror ships 165."
        )

    # ------------------------------------------------------------------ #
    # Tab 5: Hetero Explorer (merged tx+actors graph, Hetero-SAGE)
    # ------------------------------------------------------------------ #
    with tab5:
        st.markdown(
            "**Merged heterogeneous graph**: transaction nodes + wallet-actor "
            "nodes with typed edges (tx-tx money flow, actor-actor interaction, "
            "tx-actor flow). Predictions from the trained HeteroConv GraphSAGE "
            "(`heterosage_merged.pt`)."
        )
        bundle = load_hetero_bundle()
        if bundle is None:
            st.info(
                "Run `gnn-fraud prepare --dataset ellipticpp --mode merged` and "
                "`gnn-fraud train --model heterosage --dataset ellipticpp "
                "--mode merged` first."
            )
        else:
            data_h, probs = bundle
            y_actor = data_h["actor"].y
            t_actor = data_h["actor"].t

            test_mask = (~torch.isnan(y_actor)) & (t_actor > cfg.split.val_max_t)
            test_pool = torch.nonzero(test_mask).flatten()

            col_a, col_b = st.columns([2, 3])
            with col_a:
                label_filter = st.radio("Show actors", ["illicit", "licit", "all"], horizontal=True)
                max_nbrs = st.slider("Max neighbors per type", 5, 60, 25)
            with col_b:
                if label_filter == "illicit":
                    pool = test_pool[(y_actor[test_pool] == 1).cpu().numpy()]
                elif label_filter == "licit":
                    pool = test_pool[(y_actor[test_pool] == 0).cpu().numpy()]
                else:
                    pool = test_pool
                actor_id = st.selectbox(
                    "Actor (test split)",
                    pool.tolist(),
                    format_func=lambda i: f"actor #{i} (t={int(t_actor[i])})",
                )

            y_true = int(y_actor[actor_id])
            p = float(probs[actor_id])
            m1, m2, m3 = st.columns(3)
            m1.metric("Illicit probability", f"{p:.3f}")
            m2.metric(
                "Actual",
                "illicit" if y_true == 1 else "licit",
            )
            m3.metric(
                "Verdict",
                "correct" if (p >= 0.5) == (y_true == 1) else "miss",
            )

            aa = data_h["actor", "interacts", "actor"].edge_index
            fa = data_h["tx", "flows", "actor"].edge_index
            n_aa = int((aa[0] == actor_id).sum() + (aa[1] == actor_id).sum())
            n_fa = int((fa[1] == actor_id).sum())
            st.caption(
                f"Actor #{actor_id}: {n_aa} actor-actor edges, {n_fa} "
                f"tx-actor edges in the full graph (showing up to {max_nbrs} each)."
            )

            st.plotly_chart(
                hetero_subgraph_figure(data_h, probs, actor_id, max_nbrs=max_nbrs),
                use_container_width=True,
            )
            st.caption(
                "Center = selected actor. Inner ring: actor-actor interactions "
                "(blue edges). Outer ring: transactions flowing to the actor "
                "(red edges, squares). Colors: red=illicit, green=licit, "
                "grey=unknown ground truth."
            )


if __name__ == "__main__":
    main()

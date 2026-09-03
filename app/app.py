"""Streamlit dashboard for GNN fraud detection.

Pages: Overview, Model Comparison, Predict & Explain, Dataset Explorer.
Run: .venv/Scripts/streamlit run app/app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import streamlit as st
import torch

st.set_page_config(page_title="GNN Bitcoin Fraud Detection", page_icon=None, layout="wide")

from gnn_fraud_detection.config import load_config  # noqa: E402

MODELS = ["mlp", "gcn", "gat", "graphsage", "gin"]
COMPARISON_CSV = PROJECT_ROOT / "reports" / "model_comparison.csv"


@st.cache_data
def load_processed_graph():
    from gnn_fraud_detection.preprocessing import scale_features

    cfg = load_config()
    data = torch.load(cfg.processed_dir / "elliptic.pt", weights_only=False)
    scale_features(data, cfg)
    return data, cfg


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


@st.cache_data
def load_comparison() -> pd.DataFrame | None:
    if COMPARISON_CSV.exists():
        return pd.read_csv(COMPARISON_CSV)
    return None


def main():
    st.title("GNN Bitcoin Fraud Detection")
    st.caption(
        "Elliptic dataset | 203,769 transactions | temporal test split t42-t49 | "
        "MLP vs GCN vs GAT vs GraphSAGE vs GIN"
    )

    data, cfg = load_processed_graph()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Model Comparison", "Predict & Explain", "Dataset"]
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
        c1.metric("Transactions", f"{data.num_nodes:,}")
        c2.metric("Edges (symmetrized)", f"{data.edge_index.size(1):,}")
        c3.metric("Labeled", f"{labeled:,}")
        c4.metric("Illicit", f"{int((y == 1).sum()):,}")

        st.markdown(
            "**Temporal protocol**: train t1-t33, validation t34-t41, "
            "test t42-t49. Unknown-class nodes are never in any split."
        )

        comp = load_comparison()
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
        comp = load_comparison()
        if comp is None:
            st.info("Run `.venv/Scripts/gnn-fraud compare` first to generate results.")
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


if __name__ == "__main__":
    main()

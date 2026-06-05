import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dimensionality Reduction | Module 2",
    page_icon="📉",
    layout="wide",
)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("📉 Module 2: Dimensionality Reduction")
st.markdown("Interactive explorer for **PCA** and **t-SNE** on the Iris dataset.")
st.divider()

# ── Load & preprocess data ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    feature_names = iris.feature_names
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, y, target_names, feature_names

X, X_scaled, y, target_names, feature_names = load_data()
COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
method = st.sidebar.radio("Reduction Method", ["PCA", "t-SNE", "Both (Side by Side)"])

if method in ["t-SNE", "Both (Side by Side)"]:
    perplexity = st.sidebar.slider("t-SNE Perplexity", 5, 50, 30, step=5)
    n_iter = st.sidebar.slider("t-SNE Iterations", 250, 2000, 1000, step=250)

show_loadings = st.sidebar.checkbox("Show PCA Loadings Chart", value=True)
show_scree = st.sidebar.checkbox("Show Scree Plot", value=True)

st.sidebar.divider()
st.sidebar.markdown("**Dataset:** Iris (150 samples, 4 features, 3 classes)")

# ── Compute reductions ────────────────────────────────────────────────────────
@st.cache_data
def run_pca(X_scaled):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_full = PCA(n_components=4).fit(X_scaled)
    return X_pca, pca, pca_full

@st.cache_data
def run_tsne(X_scaled, perplexity, n_iter):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=n_iter)
    return tsne.fit_transform(X_scaled)

X_pca, pca_2d, pca_full = run_pca(X_scaled)

# ── Helper: scatter plot ──────────────────────────────────────────────────────
def scatter(ax, data, title, xlabel="Dim 1", ylabel="Dim 2"):
    for cls, color, name in zip(range(3), COLORS, target_names):
        mask = y == cls
        ax.scatter(data[mask, 0], data[mask, 1],
                   c=color, label=name, s=70, edgecolors="k", linewidths=0.4)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

# ── Main plots ────────────────────────────────────────────────────────────────
if method == "PCA":
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter(ax, X_pca, "PCA – Iris Dataset",
            f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)",
            f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
    st.pyplot(fig)

elif method == "t-SNE":
    with st.spinner("Running t-SNE… this may take a moment."):
        X_tsne = run_tsne(X_scaled, perplexity, n_iter)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter(ax, X_tsne, f"t-SNE – Iris Dataset (perplexity={perplexity})")
    st.pyplot(fig)

else:  # Both
    with st.spinner("Running t-SNE…"):
        X_tsne = run_tsne(X_scaled, perplexity, n_iter)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    scatter(ax1, X_pca, "PCA",
            f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)",
            f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
    scatter(ax2, X_tsne, f"t-SNE (perplexity={perplexity})")
    plt.tight_layout()
    st.pyplot(fig)

st.divider()

# ── Scree plot ────────────────────────────────────────────────────────────────
if show_scree and method in ["PCA", "Both (Side by Side)"]:
    st.subheader("📊 Scree Plot (PCA)")
    ev = pca_full.explained_variance_ratio_
    cum = np.cumsum(ev)

    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.bar(range(1, 5), ev, color="steelblue", edgecolor="black")
    a1.set_xlabel("Principal Component")
    a1.set_ylabel("Explained Variance Ratio")
    a1.set_title("Individual Explained Variance")
    a1.set_xticks(range(1, 5))

    a2.plot(range(1, 5), cum, marker="o", color="coral")
    a2.axhline(0.95, linestyle="--", color="gray", label="95% threshold")
    a2.set_xlabel("Number of Components")
    a2.set_ylabel("Cumulative Explained Variance")
    a2.set_title("Cumulative Explained Variance")
    a2.legend()

    plt.tight_layout()
    st.pyplot(fig2)

# ── Loadings chart ────────────────────────────────────────────────────────────
if show_loadings and method in ["PCA", "Both (Side by Side)"]:
    st.subheader("🔍 PCA Feature Loadings")
    loadings = pd.DataFrame(
        pca_2d.components_.T, index=feature_names, columns=["PC1", "PC2"]
    )
    st.dataframe(loadings.style.background_gradient(cmap="coolwarm", axis=None).format("{:.3f}"))

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    loadings.plot(kind="bar", ax=ax3, edgecolor="black")
    ax3.set_title("Feature Loadings on PC1 & PC2")
    ax3.set_ylabel("Loading Score")
    ax3.set_xticklabels(feature_names, rotation=25, ha="right")
    plt.tight_layout()
    st.pyplot(fig3)

# ── Dataset preview ───────────────────────────────────────────────────────────
with st.expander("📋 View Raw Iris Dataset"):
    df = pd.DataFrame(X, columns=feature_names)
    df["species"] = [target_names[i] for i in y]
    st.dataframe(df, use_container_width=True)

st.caption("Module 2 · Dimensionality Reduction · Iris Dataset")

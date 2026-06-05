import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Module 1 — Clustering", page_icon="🧩", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.algo-badge { display:inline-block; padding:0.3rem 1rem; border-radius:20px; font-size:0.72rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.8rem; }
.badge-km  { background:#1e3a5f; color:#7dd3fc; border:1px solid #38bdf8; }
.badge-hc  { background:#1a3a2a; color:#6ee7b7; border:1px solid #34d399; }
.badge-db  { background:#3b1f1f; color:#fca5a5; border:1px solid #f87171; }
.badge-gm  { background:#2d1f40; color:#c4b5fd; border:1px solid #a78bfa; }
.theory-box { background:linear-gradient(135deg,#1a1f35,#1e2540); border-left:4px solid #38bdf8; border-radius:0 10px 10px 0; padding:1rem 1.4rem; margin:0.8rem 0; color:#cbd5e1; line-height:1.7; font-size:0.9rem; }
.mrow { display:flex; gap:0.8rem; margin:0.8rem 0; }
.mchip { background:#1e2130; border:1px solid #3a3f5c; border-radius:10px; padding:0.7rem 1rem; text-align:center; flex:1; }
.mval  { font-family:'DM Mono',monospace; font-size:1.4rem; color:#7dd3fc; }
.mlbl  { font-size:0.7rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; }
</style>
""", unsafe_allow_html=True)

DARK_FIG, DARK_AX, GRID_CLR, TEXT_CLR = '#0d1117', '#161b27', '#2a2f45', '#cbd5e1'

def dark_fig(w=10, h=5, ncols=1, nrows=1):
    fig, ax = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor(DARK_FIG)
    axes = np.array([ax]).flatten()
    for a in axes:
        a.set_facecolor(DARK_AX)
        for sp in a.spines.values(): sp.set_edgecolor(GRID_CLR)
        a.tick_params(colors=TEXT_CLR, labelsize=9)
        a.xaxis.label.set_color(TEXT_CLR); a.yaxis.label.set_color(TEXT_CLR)
        a.title.set_color('#e2e8f0'); a.grid(True, alpha=0.15, color=GRID_CLR)
    return fig, ax

def draw_ellipse(pos, cov, ax, **kw):
    if cov.shape == (2, 2):
        U, s, _ = np.linalg.svd(cov)
        angle = np.degrees(np.arctan2(U[1,0], U[0,0]))
        w2, h2 = 2*np.sqrt(s)
    else:
        angle, w2, h2 = 0, 2*np.sqrt(cov), 2*np.sqrt(cov)
    for ns in range(1, 4):
        ax.add_patch(Ellipse(pos, ns*w2, ns*h2, angle=angle, **kw))

def get_data(name, n):
    if name == "Blobs (4 clusters)":    return make_blobs(n_samples=n, centers=4, cluster_std=0.9, random_state=42)
    elif name == "Moons":               return make_moons(n_samples=n, noise=0.08, random_state=42)
    elif name == "Circles":             return make_circles(n_samples=n, factor=0.5, noise=0.06, random_state=42)
    else:                               return make_blobs(n_samples=n, centers=3, cluster_std=[0.8,1.3,0.5], random_state=42)

def metrics_html(labels, X):
    nc = len(set(labels)) - (1 if -1 in labels else 0)
    noise = int((labels == -1).sum())
    mask = labels != -1
    if nc >= 2 and mask.sum() > nc:
        sil = silhouette_score(X[mask], labels[mask])
        db  = davies_bouldin_score(X[mask], labels[mask])
    else:
        sil = db = float('nan')
    st.markdown(f"""<div class="mrow">
      <div class="mchip"><div class="mval">{nc}</div><div class="mlbl">Clusters</div></div>
      <div class="mchip"><div class="mval">{sil:.3f}</div><div class="mlbl">Silhouette ↑</div></div>
      <div class="mchip"><div class="mval">{db:.3f}</div><div class="mlbl">Davies-Bouldin ↓</div></div>
      <div class="mchip"><div class="mval">{noise}</div><div class="mlbl">Noise Points</div></div>
    </div>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧩 Module 1\n### Clustering Algorithms")
    st.markdown("---")
    algo = st.radio("Algorithm", ["🔵 K-Means","🌿 Hierarchical","🔴 DBSCAN","🟣 GMM"])
    st.markdown("---")
    ds_name  = st.selectbox("Dataset", ["Blobs (4 clusters)","Moons","Circles","Blobs (3 clusters)"])
    n_samples = st.slider("Samples", 100, 1000, 300, 50)

X_raw, y_raw = get_data(ds_name, n_samples)
X_sc = StandardScaler().fit_transform(X_raw)

# ══════════════════════════ K-MEANS ══════════════════════════
if algo == "🔵 K-Means":
    st.markdown('<span class="algo-badge badge-km">K-Means Clustering</span>', unsafe_allow_html=True)
    st.title("K-Means Clustering")
    st.markdown('<div class="theory-box">Partitions data into <b>K clusters</b> minimising within-cluster sum of squares (Inertia). Assigns each point to the nearest centroid. Uses <b>k-means++</b> seeding to avoid bad initialisations.</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🔍 Result", "📐 Elbow + Silhouette", "📊 K Comparison"])

    with tab1:
        c1, c2 = st.columns([1,3])
        with c1:
            k       = st.slider("Clusters K", 2, 10, 4)
            init    = st.selectbox("Init", ["k-means++","random"])
            max_itr = st.slider("Max Iter", 50, 500, 300)
            n_init  = st.slider("n_init", 1, 20, 10)
        with c2:
            km = KMeans(n_clusters=k, init=init, max_iter=max_itr, n_init=n_init, random_state=42)
            labels = km.fit_predict(X_sc)
            metrics_html(labels, X_sc)
            fig, axes = dark_fig(14, 5, 2)
            axes = axes.flatten()
            axes[0].scatter(X_sc[:,0], X_sc[:,1], c=labels, cmap='tab10', s=18, alpha=0.7)
            axes[0].scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], c='white', s=180, marker='X', zorder=5, label='Centroids')
            axes[0].set_title(f'K-Means K={k}  Inertia={km.inertia_:.1f}')
            axes[0].legend(labelcolor='white', facecolor=DARK_AX, edgecolor=GRID_CLR)
            axes[1].scatter(X_sc[:,0], X_sc[:,1], c=y_raw, cmap='tab10', s=18, alpha=0.7)
            axes[1].set_title('Ground Truth')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        inertias, sils = [], []
        for kk in range(1,13):
            m = KMeans(n_clusters=kk, n_init=10, random_state=42)
            lbl = m.fit_predict(X_sc)
            inertias.append(m.inertia_)
            if kk >= 2: sils.append(silhouette_score(X_sc, lbl))
        fig, axes = dark_fig(14, 5, 2)
        axes = axes.flatten()
        axes[0].plot(range(1,13), inertias, 'o-', color='#7dd3fc', lw=2)
        axes[0].fill_between(range(1,13), inertias, alpha=0.1, color='#7dd3fc')
        axes[0].set_title('Elbow — Inertia vs K'); axes[0].set_xlabel('K'); axes[0].set_ylabel('Inertia')
        axes[1].plot(range(2,13), sils, 's-', color='#4ade80', lw=2)
        axes[1].set_title('Silhouette Score vs K'); axes[1].set_xlabel('K'); axes[1].set_ylabel('Silhouette')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        max_k = st.slider("Max K to compare", 4, 9, 6)
        ks = list(range(2, max_k+1))[:6]
        nrows = (len(ks)+2)//3; ncols = min(3, len(ks))
        fig, axes = dark_fig(18, nrows*4, ncols, nrows)
        axes = np.array(axes).flatten()
        for i, kk in enumerate(ks):
            m = KMeans(n_clusters=kk, n_init=10, random_state=42)
            lbl = m.fit_predict(X_sc)
            sil = silhouette_score(X_sc, lbl)
            axes[i].scatter(X_sc[:,0], X_sc[:,1], c=lbl, cmap='tab10', s=15, alpha=0.7)
            axes[i].scatter(m.cluster_centers_[:,0], m.cluster_centers_[:,1], c='white', s=100, marker='X', zorder=5)
            axes[i].set_title(f'K={kk}  Sil={sil:.3f}')
        plt.suptitle('K-Means — K Comparison', color='#e2e8f0', fontsize=14)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════ HIERARCHICAL ══════════════════════════
elif algo == "🌿 Hierarchical":
    st.markdown('<span class="algo-badge badge-hc">Hierarchical Clustering</span>', unsafe_allow_html=True)
    st.title("Hierarchical Clustering")
    st.markdown('<div class="theory-box">Builds a <b>tree of clusters (dendrogram)</b> bottom-up. No need to pre-specify K — cut the tree at the right height. <b>Ward linkage</b> minimises within-cluster variance and usually produces the most balanced clusters.</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🌳 Dendrogram","🔍 Result","🔗 Linkage Comparison"])

    with tab1:
        lm = st.selectbox("Linkage", ["ward","complete","average","single"])
        Z  = linkage(X_sc, method=lm)
        fig, ax = dark_fig(16, 7)
        dendrogram(Z, ax=ax, truncate_mode='lastp', p=25, leaf_rotation=90, leaf_font_size=8,
                   color_threshold=Z[-4,2], above_threshold_color='gray')
        ax.axhline(Z[-4,2], color='#f87171', linestyle='--', lw=1.5, label='Cut → K≈4')
        ax.set_title(f'Dendrogram — {lm.capitalize()} Linkage')
        ax.set_xlabel('Cluster / Sample', color=TEXT_CLR); ax.set_ylabel('Distance', color=TEXT_CLR)
        ax.legend(labelcolor='white', facecolor=DARK_AX, edgecolor=GRID_CLR)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        c1, c2 = st.columns([1,3])
        with c1:
            k   = st.slider("Clusters", 2, 10, 4)
            lm2 = st.selectbox("Linkage", ["ward","complete","average","single"], key='hc2')
        with c2:
            model = AgglomerativeClustering(n_clusters=k, linkage=lm2)
            labels = model.fit_predict(X_sc)
            metrics_html(labels, X_sc)
            fig, axes = dark_fig(14, 5, 2)
            axes = axes.flatten()
            axes[0].scatter(X_sc[:,0], X_sc[:,1], c=labels, cmap='tab10', s=18, alpha=0.7)
            axes[0].set_title(f'Hierarchical ({lm2}, K={k})')
            axes[1].scatter(X_sc[:,0], X_sc[:,1], c=y_raw, cmap='tab10', s=18, alpha=0.7)
            axes[1].set_title('Ground Truth')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        linkages_all = ["ward","complete","average","single"]
        fig, axes = dark_fig(18, 10, 4, 2)
        axes = np.array(axes).flatten()
        for i, lnk in enumerate(linkages_all):
            Z2 = linkage(X_sc, method=lnk)
            dendrogram(Z2, ax=axes[i], truncate_mode='lastp', p=12, no_labels=True, leaf_font_size=7, color_threshold=Z2[-4,2])
            axes[i].set_facecolor(DARK_AX); axes[i].set_title(f'{lnk.capitalize()}', color='#e2e8f0'); axes[i].tick_params(colors=TEXT_CLR)
            m2 = AgglomerativeClustering(n_clusters=4, linkage=lnk)
            lbl2 = m2.fit_predict(X_sc)
            sil2  = silhouette_score(X_sc, lbl2)
            axes[i+4].scatter(X_sc[:,0], X_sc[:,1], c=lbl2, cmap='tab10', s=15, alpha=0.7)
            axes[i+4].set_facecolor(DARK_AX); axes[i+4].set_title(f'{lnk} | Sil={sil2:.3f}', color='#e2e8f0'); axes[i+4].tick_params(colors=TEXT_CLR)
        plt.suptitle('Linkage Method Comparison', color='#e2e8f0', fontsize=14)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════ DBSCAN ══════════════════════════
elif algo == "🔴 DBSCAN":
    st.markdown('<span class="algo-badge badge-db">DBSCAN Clustering</span>', unsafe_allow_html=True)
    st.title("DBSCAN Clustering")
    st.markdown('<div class="theory-box"><b>Density-Based Spatial Clustering of Applications with Noise.</b> Discovers arbitrary-shaped clusters, automatically finds K, and labels outliers as <b>noise (−1)</b>. Two key params: <b>eps</b> (neighborhood radius) and <b>min_samples</b> (density threshold).</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🔍 Result","📏 K-Distance Graph","⚙️ Sensitivity"])

    with tab1:
        c1, c2 = st.columns([1,3])
        with c1:
            eps     = st.slider("eps", 0.05, 2.0, 0.3, 0.05)
            min_smp = st.slider("min_samples", 2, 20, 5)
        with c2:
            db = DBSCAN(eps=eps, min_samples=min_smp)
            labels = db.fit_predict(X_sc)
            metrics_html(labels, X_sc)
            core_mask  = np.zeros(len(X_sc), dtype=bool)
            core_mask[db.core_sample_indices_] = True
            noise_mask = labels == -1
            bord_mask  = ~core_mask & ~noise_mask
            fig, axes = dark_fig(14, 5, 2)
            axes = axes.flatten()
            axes[0].scatter(X_sc[:,0], X_sc[:,1], c=labels, cmap='tab10', s=18, alpha=0.7)
            axes[0].set_title(f'DBSCAN eps={eps} min_samples={min_smp}')
            axes[1].scatter(X_sc[core_mask,0],  X_sc[core_mask,1],  c='#7dd3fc', s=25, alpha=0.8, label=f'Core ({core_mask.sum()})')
            axes[1].scatter(X_sc[bord_mask,0],  X_sc[bord_mask,1],  c='#fbbf24', s=25, alpha=0.8, label=f'Border ({bord_mask.sum()})')
            axes[1].scatter(X_sc[noise_mask,0], X_sc[noise_mask,1], c='#f87171', s=50, marker='x', lw=1.5, label=f'Noise ({noise_mask.sum()})')
            axes[1].set_title('Core / Border / Noise')
            axes[1].legend(labelcolor='white', facecolor=DARK_AX, edgecolor=GRID_CLR, fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        ms_k = st.slider("min_samples for K-distance", 2, 15, 5)
        nbrs = NearestNeighbors(n_neighbors=ms_k).fit(X_sc)
        dists, _ = nbrs.kneighbors(X_sc)
        dsts = np.sort(dists[:, ms_k-1])
        fig, ax = dark_fig(12, 5)
        ax.plot(dsts, lw=2, color='#7dd3fc')
        knee = np.percentile(dsts, 90)
        ax.axhline(knee, color='#f87171', linestyle='--', lw=1.5, label=f'90th pct ≈ {knee:.2f}  (suggested eps)')
        ax.set_title(f'{ms_k}-NN Distance — Elbow = optimal eps')
        ax.set_xlabel('Points sorted by distance'); ax.set_ylabel(f'{ms_k}-NN distance')
        ax.legend(labelcolor='white', facecolor=DARK_AX, edgecolor=GRID_CLR)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        eps_vals = [0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
        fig, axes = dark_fig(18, 9, 3, 2)
        axes = np.array(axes).flatten()
        for i, e in enumerate(eps_vals):
            db2 = DBSCAN(eps=e, min_samples=5)
            lbl = db2.fit_predict(X_sc)
            nc  = len(set(lbl)) - (1 if -1 in lbl else 0)
            axes[i].scatter(X_sc[:,0], X_sc[:,1], c=lbl, cmap='tab10', s=12, alpha=0.7)
            axes[i].set_facecolor(DARK_AX); axes[i].tick_params(colors=TEXT_CLR)
            axes[i].set_title(f'eps={e} | K={nc} | Noise={(lbl==-1).sum()}', color='#e2e8f0')
        plt.suptitle('DBSCAN — eps Sensitivity (min_samples=5)', color='#e2e8f0', fontsize=14)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════ GMM ══════════════════════════
elif algo == "🟣 GMM":
    st.markdown('<span class="algo-badge badge-gm">Gaussian Mixture Model</span>', unsafe_allow_html=True)
    st.title("Gaussian Mixture Model (GMM)")
    st.markdown('<div class="theory-box">A <b>probabilistic model</b> assuming data comes from a mixture of K Gaussians. Gives <b>soft membership probabilities</b> per cluster. Fitted with <b>EM algorithm</b>. Use <b>BIC/AIC</b> to select the number of components.</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🔍 Result","📐 BIC / AIC","🎯 Covariance Types"])

    with tab1:
        c1, c2 = st.columns([1,3])
        with c1:
            n_comp   = st.slider("Components K", 2, 8, 3)
            cov_type = st.selectbox("Covariance Type", ["full","tied","diag","spherical"])
            n_init_g = st.slider("n_init", 1, 10, 3)
        with c2:
            gmm = GaussianMixture(n_components=n_comp, covariance_type=cov_type, n_init=n_init_g, random_state=42)
            gmm.fit(X_sc)
            labels = gmm.predict(X_sc)
            probs  = gmm.predict_proba(X_sc)
            metrics_html(labels, X_sc)
            fig, axes = dark_fig(14, 5, 2)
            axes = axes.flatten()
            clrs = plt.cm.tab10(np.linspace(0, 0.9, n_comp))
            for i in range(n_comp):
                axes[0].scatter(X_sc[labels==i,0], X_sc[labels==i,1], color=clrs[i], s=18, alpha=0.6)
                try:
                    if   cov_type == 'full':   cm = gmm.covariances_[i]
                    elif cov_type == 'tied':   cm = gmm.covariances_
                    elif cov_type == 'diag':   cm = np.diag(gmm.covariances_[i])
                    else:                      cm = np.eye(2)*gmm.covariances_[i]
                    if cm.shape == (2,2): draw_ellipse(gmm.means_[i], cm, axes[0], alpha=0.12, color=clrs[i])
                except Exception: pass
            axes[0].scatter(gmm.means_[:,0], gmm.means_[:,1], c='white', s=180, marker='*', zorder=5)
            axes[0].set_title(f'GMM K={n_comp} cov={cov_type}')
            sc = axes[1].scatter(X_sc[:,0], X_sc[:,1], c=probs.max(axis=1), cmap='RdYlGn', s=18, alpha=0.8, vmin=0.5, vmax=1)
            plt.colorbar(sc, ax=axes[1], label='Confidence')
            axes[1].set_title('Soft Assignment Confidence')
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.caption(f"BIC: **{gmm.bic(X_sc):.2f}** | AIC: **{gmm.aic(X_sc):.2f}**")

    with tab2:
        max_k = st.slider("Max components", 4, 12, 8)
        bics, aics = [], []
        for n in range(1, max_k+1):
            g = GaussianMixture(n_components=n, n_init=3, random_state=42).fit(X_sc)
            bics.append(g.bic(X_sc)); aics.append(g.aic(X_sc))
        bb, ab = np.argmin(bics)+1, np.argmin(aics)+1
        fig, ax = dark_fig(12, 5)
        ax.plot(range(1,max_k+1), bics, 'o-', color='#7dd3fc', lw=2, markersize=7, label='BIC')
        ax.plot(range(1,max_k+1), aics, 's-', color='#f472b6', lw=2, markersize=7, label='AIC')
        ax.axvline(bb, color='#7dd3fc', linestyle='--', lw=1.5, alpha=0.7, label=f'Best BIC K={bb}')
        ax.axvline(ab, color='#f472b6', linestyle='--', lw=1.5, alpha=0.7, label=f'Best AIC K={ab}')
        ax.set_xlabel('Components'); ax.set_ylabel('Score (lower = better)')
        ax.set_title('BIC / AIC Component Selection')
        ax.legend(labelcolor='white', facecolor=DARK_AX, edgecolor=GRID_CLR)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        col1, col2 = st.columns(2)
        col1.metric("Optimal K (BIC)", bb); col2.metric("Optimal K (AIC)", ab)

    with tab3:
        ctypes = ['full','tied','diag','spherical']
        fig, axes = dark_fig(18, 9, 2, 2)
        axes = np.array(axes).flatten()
        for i, ct in enumerate(ctypes):
            g = GaussianMixture(n_components=3, covariance_type=ct, n_init=3, random_state=42).fit(X_sc)
            lbl = g.predict(X_sc); sil = silhouette_score(X_sc, lbl); bic = g.bic(X_sc)
            clrs2 = plt.cm.Set1(np.linspace(0, 0.7, 3))
            for j in range(3):
                axes[i].scatter(X_sc[lbl==j,0], X_sc[lbl==j,1], color=clrs2[j], s=18, alpha=0.6)
                try:
                    if ct=='full':   cm=g.covariances_[j]
                    elif ct=='tied': cm=g.covariances_
                    elif ct=='diag': cm=np.diag(g.covariances_[j])
                    else:            cm=np.eye(2)*g.covariances_[j]
                    if cm.shape==(2,2): draw_ellipse(g.means_[j], cm, axes[i], alpha=0.15, color=clrs2[j])
                except Exception: pass
            axes[i].scatter(g.means_[:,0], g.means_[:,1], c='white', s=150, marker='*', zorder=5)
            axes[i].set_facecolor(DARK_AX); axes[i].tick_params(colors=TEXT_CLR)
            axes[i].set_title(f'"{ct}" | Sil={sil:.3f} | BIC={bic:.0f}', color='#e2e8f0')
        plt.suptitle('GMM — Covariance Type Comparison', color='#e2e8f0', fontsize=14)
        plt.tight_layout(); st.pyplot(fig); plt.close()

st.markdown("---")
st.markdown("<center style='color:#4a5568;font-size:0.78rem'>Module 1 — Clustering Algorithms &nbsp;|&nbsp; K-Means · Hierarchical · DBSCAN · GMM</center>", unsafe_allow_html=True)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anomaly Detection | Module 3",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Module 3: Anomaly Detection — Isolation Forest")
st.markdown("Upload your own CSV, configure the model, and explore anomaly detection results interactively.")
st.divider()

COLORS = {"normal": "#377eb8", "anomaly": "#e41a1c"}

# ── CSV Upload ────────────────────────────────────────────────────────────────
st.subheader("📂 Upload CSV File")
uploaded_file = st.file_uploader(
    "Upload a CSV file with numeric columns. Optionally include a 'label' column (values: normal / anomaly) for ground-truth evaluation.",
    type=["csv"]
)

if uploaded_file is None:
    st.info("👆 Please upload a CSV file to get started. You can use `data/iris_with_anomalies.csv` from the notebook as a sample.")
    st.stop()

# ── Load & validate CSV ───────────────────────────────────────────────────────
try:
    df_raw = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.success(f"File loaded — **{df_raw.shape[0]} rows × {df_raw.shape[1]} columns**")

with st.expander("👀 Preview Raw Data"):
    st.dataframe(df_raw.head(20), use_container_width=True)

# ── Column selection ──────────────────────────────────────────────────────────
st.subheader("🛠️ Configure Columns")

all_cols = df_raw.columns.tolist()
numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

has_label_col = "label" in df_raw.columns

col_a, col_b = st.columns(2)
with col_a:
    feature_cols = st.multiselect(
        "Select feature columns (numeric only)",
        options=numeric_cols,
        default=numeric_cols[:min(len(numeric_cols), 6)]
    )
with col_b:
    if has_label_col:
        st.info("✅ 'label' column detected — confusion matrix & classification report will be shown.")
        label_col = "label"
    else:
        label_col = st.selectbox(
            "Optional: select a label column (normal/anomaly) for evaluation",
            options=["None"] + all_cols
        )
        label_col = None if label_col == "None" else label_col

if not feature_cols:
    st.warning("Please select at least one feature column.")
    st.stop()

# ── Sidebar model settings ────────────────────────────────────────────────────
st.sidebar.header("⚙️ Model Settings")
n_estimators  = st.sidebar.slider("n_estimators (trees)", 50, 300, 100, step=50)
contamination = st.sidebar.slider("Contamination", 0.01, 0.40, 0.10, step=0.01)
random_state  = st.sidebar.number_input("Random State", value=42, step=1)
st.sidebar.divider()
st.sidebar.markdown(f"**Features selected:** {len(feature_cols)}")
st.sidebar.markdown(f"**Samples:** {df_raw.shape[0]}")

# ── Prepare data ──────────────────────────────────────────────────────────────
df_model = df_raw[feature_cols].dropna()
X = df_model.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_components = min(2, X.shape[1])
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# ── Run Isolation Forest ──────────────────────────────────────────────────────
@st.cache_data
def run_model(X_scaled_bytes, n_estimators, contamination, random_state):
    X_arr = np.frombuffer(X_scaled_bytes).reshape(-1, X_scaled.shape[1])
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=int(random_state)
    )
    model.fit(X_arr)
    return model.predict(X_arr), model.decision_function(X_arr)

X_bytes = X_scaled.tobytes()
y_pred, scores = run_model(X_bytes, n_estimators, contamination, int(random_state))
labels_pred = ["anomaly" if v == -1 else "normal" for v in y_pred]

# ── Metrics row ───────────────────────────────────────────────────────────────
st.divider()
n_detected = int(np.sum(y_pred == -1))

c1, c2, c3 = st.columns(3)
c1.metric("Total Samples", len(X))
c2.metric("Flagged as Anomaly", n_detected)
c3.metric("Flagged as Normal", len(X) - n_detected)

# ── Ground truth evaluation (if label column provided) ────────────────────────
y_true = None
if label_col:
    label_series = df_raw.loc[df_model.index, label_col].str.strip().str.lower()
    y_true = np.where(label_series == "anomaly", -1, 1)

    from sklearn.metrics import classification_report, confusion_matrix
    detected     = int(np.sum((y_pred == -1) & (y_true == -1)))
    false_pos    = int(np.sum((y_pred == -1) & (y_true ==  1)))
    total_true   = int(np.sum(y_true == -1))

    st.divider()
    st.subheader("📊 Evaluation (Ground Truth Available)")

    e1, e2, e3 = st.columns(3)
    e1.metric("True Anomalies", total_true)
    e2.metric("Correctly Detected", detected)
    e3.metric("False Positives", false_pos)

    col_cm, col_rep = st.columns([1, 1])
    with col_cm:
        cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                    xticklabels=["Pred: Anomaly", "Pred: Normal"],
                    yticklabels=["True: Anomaly", "True: Normal"])
        ax_cm.set_title("Confusion Matrix")
        plt.tight_layout()
        st.pyplot(fig_cm)
    with col_rep:
        report = classification_report(y_true, y_pred, target_names=["Anomaly", "Normal"])
        st.code(report)

# ── PCA Scatter ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("📍 PCA Scatter — Predictions" + (" vs Ground Truth" if y_true is not None else ""))

if y_true is not None:
    labels_true = ["anomaly" if v == -1 else "normal" for v in y_true]
    fig_pca, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for label in ["normal", "anomaly"]:
        mask = np.array(labels_true) == label
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c=COLORS[label],
                    label=label, s=60, edgecolors="k", linewidths=0.4)
    ax1.set_title("Ground Truth", fontsize=13, fontweight="bold")
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2"); ax1.legend()

    for label in ["normal", "anomaly"]:
        mask = np.array(labels_pred) == label
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], c=COLORS[label],
                    label=label, s=60, edgecolors="k", linewidths=0.4)
    ax2.set_title("Isolation Forest Predictions", fontsize=13, fontweight="bold")
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2"); ax2.legend()

else:
    fig_pca, ax1 = plt.subplots(figsize=(8, 5))
    for label in ["normal", "anomaly"]:
        mask = np.array(labels_pred) == label
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c=COLORS[label],
                    label=label, s=60, edgecolors="k", linewidths=0.4)
    ax1.set_title("Isolation Forest Predictions (PCA view)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2"); ax1.legend()

plt.tight_layout()
st.pyplot(fig_pca)

# ── Anomaly score distribution ────────────────────────────────────────────────
st.divider()
st.subheader("📊 Anomaly Score Distribution")

fig_score, ax_s = plt.subplots(figsize=(9, 4))
if y_true is not None:
    for label, color in [("normal", "#377eb8"), ("anomaly", "#e41a1c")]:
        mask = np.array(labels_true) == label
        ax_s.hist(scores[mask], bins=25, alpha=0.65, color=color, label=f"true {label}", edgecolor="black")
else:
    for label, color in [("normal", "#377eb8"), ("anomaly", "#e41a1c")]:
        mask = np.array(labels_pred) == label
        ax_s.hist(scores[mask], bins=25, alpha=0.65, color=color, label=f"pred {label}", edgecolor="black")

ax_s.axvline(0, linestyle="--", color="black", label="Decision boundary (0)")
ax_s.set_xlabel("Anomaly Score")
ax_s.set_ylabel("Count")
ax_s.set_title("Score Distribution (lower = more anomalous)")
ax_s.legend()
plt.tight_layout()
st.pyplot(fig_score)

# ── Feature distributions ─────────────────────────────────────────────────────
st.divider()
st.subheader("📈 Feature Distributions: Normal vs Anomaly (Predicted)")

n_feats = len(feature_cols)
ncols = min(3, n_feats)
nrows = (n_feats + ncols - 1) // ncols
fig_feat, axes_feat = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
axes_feat = np.array(axes_feat).flatten() if n_feats > 1 else [axes_feat]

df_plot = df_model.copy()
df_plot["predicted"] = labels_pred

for i, feat in enumerate(feature_cols):
    ax = axes_feat[i]
    for label, color in [("normal", "#377eb8"), ("anomaly", "#e41a1c")]:
        subset = df_plot[df_plot["predicted"] == label][feat]
        ax.hist(subset, bins=20, alpha=0.6, color=color, label=label, edgecolor="black")
    ax.set_title(feat); ax.set_xlabel("Value"); ax.set_ylabel("Count"); ax.legend()

for j in range(i + 1, len(axes_feat)):
    axes_feat[j].set_visible(False)

plt.suptitle("Feature Distributions by Predicted Label", fontsize=13, fontweight="bold")
plt.tight_layout()
st.pyplot(fig_feat)

# ── Download results ──────────────────────────────────────────────────────────
st.divider()
st.subheader("⬇️ Download Results")

df_result = df_raw.loc[df_model.index].copy()
df_result["predicted"] = labels_pred
df_result["anomaly_score"] = scores.round(4)

csv_out = df_result.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download predictions as CSV",
    data=csv_out,
    file_name="anomaly_predictions.csv",
    mime="text/csv"
)

# ── Data preview ──────────────────────────────────────────────────────────────
with st.expander("📋 View Full Dataset with Predictions"):
    st.dataframe(
        df_result.style.map(
            lambda v: "background-color: #fdd;" if v == "anomaly" else "",
            subset=["predicted"]
        ),
        use_container_width=True
    )

st.caption("Module 3 · Anomaly Detection · Isolation Forest · CSV Upload")

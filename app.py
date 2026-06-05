import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="🔍",
    layout="wide",
)

st.title("Anomaly Detection")
st.markdown("Interactive explorer for **Isolation Forest** on the Iris dataset with injected anomalies.")
st.divider()

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

n_anomalies = st.sidebar.slider("Number of Injected Anomalies", 5, 50, 20, step=5)
n_estimators = st.sidebar.slider("n_estimators (trees)", 50, 300, 100, step=50)
contamination = st.sidebar.slider("Contamination", 0.01, 0.40, round(n_anomalies / (150 + n_anomalies), 2), step=0.01)
random_state = st.sidebar.number_input("Random State", value=42, step=1)

st.sidebar.divider()
st.sidebar.markdown("**Dataset:** Iris (150 normal) + injected anomalies")

# ── Load & prepare data ───────────────────────────────────────────────────────
@st.cache_data
def load_data(n_anomalies, random_state):
    iris = load_iris()
    X_normal = iris.data
    feature_names = iris.feature_names

    rng = np.random.RandomState(random_state)
    X_anomalies = rng.uniform(low=-2, high=10, size=(n_anomalies, X_normal.shape[1]))

    X = np.vstack([X_normal, X_anomalies])
    y_true = np.array([1] * len(X_normal) + [-1] * n_anomalies)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    return X, X_scaled, X_pca, y_true, feature_names

X, X_scaled, X_pca, y_true, feature_names = load_data(n_anomalies, int(random_state))

# ── Run Isolation Forest ──────────────────────────────────────────────────────
@st.cache_data
def run_model(X_scaled, n_estimators, contamination, random_state):
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state
    )
    model.fit(X_scaled)
    y_pred = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)
    return y_pred, scores

y_pred, scores = run_model(X_scaled, n_estimators, contamination, int(random_state))

labels_true = ["anomaly" if v == -1 else "normal" for v in y_true]
labels_pred = ["anomaly" if v == -1 else "normal" for v in y_pred]

# ── Metrics row ───────────────────────────────────────────────────────────────
detected = int(np.sum((y_pred == -1) & (y_true == -1)))
total_anomalies = int(np.sum(y_true == -1))
false_positives = int(np.sum((y_pred == -1) & (y_true == 1)))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Samples", len(X))
col2.metric("True Anomalies", total_anomalies)
col3.metric("Anomalies Detected", detected, delta=f"{detected - total_anomalies} vs truth")
col4.metric("False Positives", false_positives)

st.divider()

# ── PCA scatter: Ground Truth vs Predictions ──────────────────────────────────
st.subheader("📍 PCA View — Ground Truth vs Predictions")
COLORS = {"normal": "#377eb8", "anomaly": "#e41a1c"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for label in ["normal", "anomaly"]:
    mask = np.array(labels_true) == label
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c=COLORS[label],
                label=label, s=60, edgecolors="k", linewidths=0.4)
ax1.set_title("Ground Truth", fontsize=13, fontweight="bold")
ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
ax1.legend()

for label in ["normal", "anomaly"]:
    mask = np.array(labels_pred) == label
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], c=COLORS[label],
                label=label, s=60, edgecolors="k", linewidths=0.4)
ax2.set_title("Isolation Forest Predictions", fontsize=13, fontweight="bold")
ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
ax2.legend()

plt.tight_layout()
st.pyplot(fig)

st.divider()

# ── Anomaly score distribution ────────────────────────────────────────────────
st.subheader("📊 Anomaly Score Distribution")

fig2, ax = plt.subplots(figsize=(9, 4))
for label, color in [("normal", "#377eb8"), ("anomaly", "#e41a1c")]:
    mask = np.array(labels_true) == label
    ax.hist(scores[mask], bins=25, alpha=0.65, color=color, label=label, edgecolor="black")
ax.axvline(0, linestyle="--", color="black", label="Decision boundary (0)")
ax.set_xlabel("Anomaly Score")
ax.set_ylabel("Count")
ax.set_title("Score Distribution (lower = more anomalous)")
ax.legend()
plt.tight_layout()
st.pyplot(fig2)

st.divider()

# ── Confusion matrix ──────────────────────────────────────────────────────────
st.subheader("🔲 Confusion Matrix")
cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
fig3, ax3 = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3,
            xticklabels=["Pred: Anomaly", "Pred: Normal"],
            yticklabels=["True: Anomaly", "True: Normal"])
ax3.set_title("Confusion Matrix")
plt.tight_layout()
st.pyplot(fig3)

# ── Classification report ─────────────────────────────────────────────────────
with st.expander("📋 Full Classification Report"):
    report = classification_report(y_true, y_pred, target_names=["Anomaly", "Normal"])
    st.code(report)

st.divider()

# ── Feature distributions ─────────────────────────────────────────────────────
st.subheader("📈 Feature Distributions: Normal vs Anomaly")
df = pd.DataFrame(X, columns=feature_names)
df["label"] = labels_true

fig4, axes = plt.subplots(2, 2, figsize=(12, 7))
axes = axes.flatten()
for i, feat in enumerate(feature_names):
    for label, color in [("normal", "#377eb8"), ("anomaly", "#e41a1c")]:
        subset = df[df["label"] == label][feat]
        axes[i].hist(subset, bins=20, alpha=0.6, color=color, label=label, edgecolor="black")
    axes[i].set_title(feat)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Count")
    axes[i].legend()
plt.suptitle("Feature Distributions", fontsize=13, fontweight="bold")
plt.tight_layout()
st.pyplot(fig4)

# ── Data preview ──────────────────────────────────────────────────────────────
with st.expander("📋 View Dataset with Predictions"):
    df["predicted"] = labels_pred
    df["anomaly_score"] = scores.round(4)
    st.dataframe(df.style.applymap(
        lambda v: "background-color: #fdd;" if v == "anomaly" else "",
        subset=["label", "predicted"]
    ), use_container_width=True)

st.caption(" Anomaly Detection · Isolation Forest · Iris Dataset")

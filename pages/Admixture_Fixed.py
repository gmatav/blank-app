import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar

st.set_page_config(layout="wide")

# -----------------------------
# Title
# -----------------------------
st.title("🧪 2-Way Admixture (Advanced)")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Settings")

dataset_type = st.sidebar.radio(
    "Source dataset",
    ["MODERN", "ANCIENTS", "BOTH"]
)

# 🔥 NEW: Preselection toggle
use_preselection = st.sidebar.checkbox(
    "Use preselection (Top K closest)",
    value=True
)

top_k = st.sidebar.slider(
    "Preselect top populations",
    10,
    200,
    50,
    disabled=not use_preselection
)

top_n = st.sidebar.slider("Top results", 5, 50, 20)

min_prop = st.sidebar.slider("Minimum component %", 0, 20, 5) / 100

metrics = st.sidebar.multiselect(
    "Metrics",
    ["euclidean", "weighted", "manhattan", "cosine", "correlation"],
    default=["euclidean"]
)

decay = st.sidebar.slider("Weight decay (weighted)", 0.5, 2.0, 1.0)

st.sidebar.markdown("---")

# -----------------------------
# Load datasets
# -----------------------------
@st.cache_data
def load_modern():
    df = pd.read_csv("data/G25_WestEurasian_only.csv", header=None)
    df = df.rename(columns={0: "Label"})
    labels = df["Label"].astype(str).values
    X = df.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce").values
    return labels, X


@st.cache_data
def load_ancients():
    df = pd.read_csv("data/ancients_combined.csv", header=None)
    df = df.rename(columns={0: "Label"})
    labels = df["Label"].astype(str).values
    X = df.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce").values
    return labels, X


labels_mod, X_mod = load_modern()
labels_anc, X_anc = load_ancients()

# weights
n_dims = X_mod.shape[1]
weights = 1 / ((np.arange(n_dims) + 1) ** decay)

# -----------------------------
# Target
# -----------------------------
target_mode = st.radio(
    "Target mode",
    ["Use dataset", "Paste G25 coordinates"],
    horizontal=True
)

if target_mode == "Use dataset":

    target_label = st.selectbox("Target (modern)", labels_mod)
    t = X_mod[np.where(labels_mod == target_label)[0][0]]

    st.session_state["t"] = t
    st.session_state["target_label"] = target_label

else:
    coord_text = st.text_area("Paste G25 row", placeholder="Sample, 0.1, ...")

    if st.button("Load coordinates"):
        try:
            parts = [x.strip() for x in coord_text.split(",")]
            target_label = parts[0]
            t = np.array([float(x) for x in parts[1:]])

            if len(t) != X_mod.shape[1]:
                st.error("Wrong dimensions")
            else:
                st.session_state["t"] = t
                st.session_state["target_label"] = target_label
                st.success(f"Loaded {target_label}")

        except:
            st.error("Invalid format")

    if "t" not in st.session_state:
        st.stop()

    t = st.session_state["t"]
    target_label = st.session_state["target_label"]

st.info(f"Target: {target_label}")

# -----------------------------
# Build pool
# -----------------------------
if dataset_type == "MODERN":
    labels_pool, X_pool = labels_mod, X_mod
elif dataset_type == "ANCIENTS":
    labels_pool, X_pool = labels_anc, X_anc
else:
    labels_pool = np.concatenate([labels_mod, labels_anc])
    X_pool = np.vstack([X_mod, X_anc])

mask = labels_pool != target_label
labels_pool = labels_pool[mask]
X_pool = X_pool[mask]

# -----------------------------
# Distance functions
# -----------------------------
def dist_euclidean(a, b): return np.linalg.norm(a - b)
def dist_weighted(a, b): return np.sqrt(np.sum(weights * (a - b)**2))
def dist_manhattan(a, b): return np.sum(np.abs(a - b))

def dist_cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 1 - (a @ b)/(na*nb) if na and nb else np.inf

def dist_corr(a, b):
    a0, b0 = a-a.mean(), b-b.mean()
    return 1 - (a0 @ b0)/(np.linalg.norm(a0)*np.linalg.norm(b0))

metric_map = {
    "euclidean": dist_euclidean,
    "weighted": dist_weighted,
    "manhattan": dist_manhattan,
    "cosine": dist_cosine,
    "correlation": dist_corr
}

# -----------------------------
# Candidate pool selection
# -----------------------------
if use_preselection:

    dists = np.sqrt(np.sum((X_pool - t) ** 2, axis=1))
    top_idx = np.argsort(dists)[:top_k]

    labels_sub = labels_pool[top_idx]
    X_sub = X_pool[top_idx]

    st.info(f"Using top {top_k} closest populations")

else:
    labels_sub = labels_pool
    X_sub = X_pool

    if len(X_pool) > 500:
        st.warning("⚠️ Large dataset without preselection may be slow")

# -----------------------------
# Alpha optimization
# -----------------------------
def best_alpha_euclidean(x, y, t):
    d = x - y
    denom = d @ d
    if denom == 0:
        return 0.5
    a = ((t - y) @ d) / denom
    return float(np.clip(a, min_prop, 1 - min_prop))


def best_alpha_other(x, y, t, metric_func):
    def f(a):
        return metric_func(t, a*x + (1-a)*y)

    res = minimize_scalar(
        f,
        bounds=(min_prop, 1-min_prop),
        method="bounded"
    )
    return float(res.x), float(res.fun)

# -----------------------------
# Run
# -----------------------------
if st.button("Run 2-Way Admixture"):

    for m in metrics:

        st.subheader(m.upper())

        results = []

        for i in range(len(X_sub)):
            for j in range(i+1, len(X_sub)):

                x, y = X_sub[i], X_sub[j]

                if m in ["euclidean", "weighted"]:
                    a = best_alpha_euclidean(x, y, t)
                    d = metric_map[m](t, a*x + (1-a)*y)
                else:
                    a, d = best_alpha_other(x, y, t, metric_map[m])

                results.append((d, labels_sub[i], labels_sub[j], a, 1-a))

        df = pd.DataFrame(results, columns=["Distance","Pop1","Pop2","Prop1","Prop2"])

        df = df[
            (df["Prop1"] > min_prop) &
            (df["Prop2"] > min_prop)
        ]

        df = df.sort_values("Distance").head(top_n).reset_index(drop=True)

        df["Prop1"] = (df["Prop1"] * 100).round(2)
        df["Prop2"] = (df["Prop2"] * 100).round(2)

        styled = (
            df.style
            .format({
                "Distance":"{:.6f}",
                "Prop1":"{:.2f}%",
                "Prop2":"{:.2f}%"
            })
            .background_gradient(subset=["Distance"], cmap="RdYlGn_r")
            .hide(axis="index")
        )

        st.table(styled)
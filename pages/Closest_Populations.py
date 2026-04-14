import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

# -----------------------------
# Title
# -----------------------------
st.title("📍 Closest Populations")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Settings")

dataset_type = st.sidebar.radio(
    "Source dataset",
    ["MODERN", "ANCIENTS", "BOTH"]
)

top_n = st.sidebar.slider("Top N", 5, 50, 15)

metrics = st.sidebar.multiselect(
    "Distance metrics",
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
# Target mode
# -----------------------------
target_mode = st.radio(
    "Target mode",
    ["Use dataset", "Paste G25 coordinates"],
    horizontal=True
)

# -----------------------------
# Target logic (session safe)
# -----------------------------
if target_mode == "Use dataset":

    target_label = st.selectbox("Target (modern)", labels_mod)
    t = X_mod[np.where(labels_mod == target_label)[0][0]]

    st.session_state["t"] = t
    st.session_state["target_label"] = target_label

else:
    coord_text = st.text_area(
        "Paste G25 row",
        placeholder="Sample, 0.1, 0.2, ..."
    )

    if st.button("Load coordinates"):

        try:
            parts = [x.strip() for x in coord_text.split(",")]

            target_label = parts[0]
            t = np.array([float(x) for x in parts[1:]])

            if len(t) != X_mod.shape[1]:
                st.error("Wrong number of dimensions")
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

st.info(f"Target: {target_label} | Source: {dataset_type}")

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

# remove target if exists
mask = labels_pool != target_label
labels_pool = labels_pool[mask]
X_pool = X_pool[mask]

# -----------------------------
# Distance functions
# -----------------------------
def dist_euclidean(A, b):
    return np.sqrt(np.sum((A - b) ** 2, axis=1))

def dist_weighted(A, b):
    return np.sqrt(np.sum(weights * (A - b) ** 2, axis=1))

def dist_manhattan(A, b):
    return np.sum(np.abs(A - b), axis=1)

def dist_cosine(A, b):
    denom = (np.linalg.norm(A, axis=1) * np.linalg.norm(b))
    denom[denom == 0] = np.nan
    return 1 - (A @ b) / denom

def dist_correlation(A, b):
    Ab = A - A.mean(axis=1, keepdims=True)
    bb = b - b.mean()
    denom = (np.linalg.norm(Ab, axis=1) * np.linalg.norm(bb))
    denom[denom == 0] = np.nan
    return 1 - (Ab @ bb)

dist_funcs = {
    "euclidean": dist_euclidean,
    "weighted": dist_weighted,
    "manhattan": dist_manhattan,
    "cosine": dist_cosine,
    "correlation": dist_correlation
}

# -----------------------------
# Compute
# -----------------------------
if st.button("Compute distances"):

    # clean data
    valid_mask = ~np.isnan(X_pool).any(axis=1)
    labels_v = labels_pool[valid_mask]
    X_v = X_pool[valid_mask]

    # compute
    for m in metrics:

        d = dist_funcs[m](X_v, t)
        d = np.where(np.isfinite(d), d, np.inf)

        order = np.argsort(d)
        top_idx = order[:top_n]

        df_res = pd.DataFrame({
            "Rank": np.arange(1, len(top_idx) + 1),
            "Population": labels_v[top_idx],
            "Distance": d[top_idx]
        })

        st.subheader(m.upper())

        styled = (
            df_res.style
            .format({"Distance": "{:.6f}"})
            .background_gradient(subset=["Distance"], cmap="RdYlGn_r")
            .hide(axis="index")
        )

        st.table(styled)
import streamlit as st
import pandas as pd
import numpy as np

st.title("📍 Closest Populations")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/G25_WestEurasian_only.csv", header=None)
    df = df.rename(columns={0: "Label"})
    labels = df["Label"].astype(str).str.strip()
    X = df.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce")
    return labels, X

labels, X = load_data()

# -----------------------------
# UI — Target selection
# -----------------------------
st.subheader("Select Target Population")

target_label = st.selectbox(
    "Search or select a population:",
    labels,
    index=0
)

top_n = st.slider("Top N results", 5, 50, 15)

metrics = st.multiselect(
    "Distance metrics:",
    ["euclidean", "manhattan", "cosine", "correlation"],
    default=["euclidean"]
)

# -----------------------------
# Distance functions
# -----------------------------
def dist_euclidean(A, b):
    return np.sqrt(np.sum((A - b) ** 2, axis=1))

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
    return 1 - (Ab @ bb) / denom

dist_funcs = {
    "euclidean": dist_euclidean,
    "manhattan": dist_manhattan,
    "cosine": dist_cosine,
    "correlation": dist_correlation
}

# -----------------------------
# Compute
# -----------------------------
if st.button("Compute distances"):

    # clean data
    valid_mask = X.notna().all(axis=1)
    labels_v = labels[valid_mask].to_numpy()
    X_v = X[valid_mask].to_numpy(float)

    # target
    tpos = np.where(labels_v == target_label)[0][0]
    x0 = X_v[tpos]

    # remove target
    X_ref = np.delete(X_v, tpos, axis=0)
    labels_ref = np.delete(labels_v, tpos, axis=0)

    # -----------------------------
    # Results
    # -----------------------------
    for m in metrics:
        d = dist_funcs[m](X_ref, x0)
        d = np.where(np.isfinite(d), d, np.inf)

        order = np.argsort(d)
        top_idx = order[:top_n]

        df_res = pd.DataFrame({
            "Rank": np.arange(1, len(top_idx) + 1),
            "Population": labels_ref[top_idx],
            "Distance": d[top_idx]
        })

        st.subheader(f"{m.upper()}")

        # 🎨 Heat-style table
        styled = (
            df_res.style
            .format({"Distance": "{:.6f}"})
            .background_gradient(subset=["Distance"], cmap="RdYlGn_r")
        )

        st.dataframe(styled, use_container_width=True)
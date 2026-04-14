import streamlit as st
import pandas as pd
import numpy as np
from numba import njit

st.set_page_config(layout="wide")

# -----------------------------
# Title
# -----------------------------
st.title("🧪 2-Way Admixture (Numba Fast Engine)")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Settings")

dataset_type = st.sidebar.radio(
    "Source dataset",
    ["MODERN", "ANCIENTS", "BOTH"]
)

top_n = st.sidebar.slider("Top results", 5, 50, 20)

min_prop = st.sidebar.slider("Minimum component %", 0, 20, 2) / 100

metric = st.sidebar.selectbox(
    "Metric",
    ["euclidean", "weighted"]
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

# -----------------------------
# Target input
# -----------------------------
target_mode = st.radio(
    "Target mode",
    ["Use dataset", "Paste G25 coordinates"],
    horizontal=True
)

if target_mode == "Use dataset":
    target_label = st.selectbox("Target (modern)", labels_mod)
    t = X_mod[np.where(labels_mod == target_label)[0][0]]

else:
    coord_text = st.text_area("Paste G25 row")

    if st.button("Load coordinates"):
        try:
            parts = [x.strip() for x in coord_text.split(",")]
            target_label = parts[0]
            t = np.array([float(x) for x in parts[1:]])

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

# remove target if exists
mask = labels_pool != target_label
labels_pool = labels_pool[mask]
X_pool = X_pool[mask]

# -----------------------------
# NUMBA ENGINE
# -----------------------------
@njit
def compute_two_way(X, t, min_prop, weighted, decay):

    n, d = X.shape

    # weights
    weights = np.zeros(d)
    for i in range(d):
        weights[i] = 1.0 / ((i + 1) ** decay)

    # target distances
    d_target = np.zeros(n)

    for i in range(n):
        s = 0.0
        for k in range(d):
            diff = t[k] - X[i, k]
            if weighted:
                s += weights[k] * diff * diff
            else:
                s += diff * diff
        d_target[i] = s

    # store results
    max_results = n * n // 2
    out_dist = np.empty(max_results)
    out_i = np.empty(max_results, dtype=np.int32)
    out_j = np.empty(max_results, dtype=np.int32)
    out_a = np.empty(max_results)

    count = 0

    for i in range(n - 1):
        for j in range(i + 1, n):

            d12 = 0.0
            for k in range(d):
                diff = X[i, k] - X[j, k]
                if weighted:
                    d12 += weights[k] * diff * diff
                else:
                    d12 += diff * diff

            if d12 == 0:
                continue

            d1 = d_target[i]
            d2 = d_target[j]

            a = (d12 + d2 - d1) / (2 * d12)

            if a < min_prop or a > 1 - min_prop:
                continue

            val = d2 - a * a * d12
            if val <= 0:
                continue

            out_dist[count] = np.sqrt(val)
            out_i[count] = i
            out_j[count] = j
            out_a[count] = a

            count += 1

    return out_dist[:count], out_i[:count], out_j[:count], out_a[:count]

# -----------------------------
# Run
# -----------------------------
if st.button("Run 2-Way Admixture"):

    weighted = (metric == "weighted")

    with st.spinner("Computing... (first run compiles Numba)"):
        dist, ii, jj, aa = compute_two_way(
            X_pool.astype(np.float64),
            t.astype(np.float64),
            min_prop,
            weighted,
            decay
        )

    # sort
    idx = np.argsort(dist)[:top_n]

    rows = []
    for k in idx:
        i = ii[k]
        j = jj[k]
        a = aa[k]

        rows.append((
            dist[k],
            labels_pool[i],
            labels_pool[j],
            a * 100,
            (1 - a) * 100
        ))

    df = pd.DataFrame(rows, columns=["Distance","Pop1","Pop2","Prop1","Prop2"])

    df["Prop1"] = df["Prop1"].round(2)
    df["Prop2"] = df["Prop2"].round(2)

    styled = (
    df.style
    .format({
        "Distance": "{:.6f}",
        "Prop1": "{:.2f}%",
        "Prop2": "{:.2f}%"
    })
    .background_gradient(subset=["Distance"], cmap="RdYlGn_r")
    .hide(axis="index")
)

    st.table(styled)

    # -----------------------------
    # Bars
    # -----------------------------
    st.subheader("📊 Admixture Breakdown")

    for _, row in df.iterrows():
        st.markdown(f"**{row['Pop1']} + {row['Pop2']}**")
        st.progress(int(row["Prop1"]))
        st.caption(f"{row['Pop1']} ({row['Prop1']:.2f}%)")
        st.progress(int(row["Prop2"]))
        st.caption(f"{row['Pop2']} ({row['Prop2']:.2f}%)")
        st.markdown("---")
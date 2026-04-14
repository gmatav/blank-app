import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar

# -----------------------------
# Page title
# -----------------------------
st.title("🧪 2-Way Admixture")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("⚙️ 2-Way Settings")

top_n = st.sidebar.slider("Top results", 5, 50, 20, key="2way_top")

num_pairs = st.sidebar.slider("Number of pairs to test", 1000, 50000, 10000, step=1000, key="2way_pairs")

metrics = st.sidebar.multiselect(
    "Distance metrics",
    ["euclidean", "manhattan", "cosine", "correlation"],
    default=["euclidean"],
    key="2way_metrics"
)

st.sidebar.markdown("---")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/G25_WestEurasian_only.csv", header=None)
    df = df.rename(columns={0: "Label"})
    labels = df["Label"].astype(str).values
    X = df.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce").values
    return labels, X

labels, X = load_data()

# -----------------------------
# Target selection
# -----------------------------
target_label = st.selectbox(
    "Select target population",
    labels,
    key="2way_target"
)

# -----------------------------
# Distance functions
# -----------------------------
def dist_euclidean(a, b): return np.linalg.norm(a - b)
def dist_manhattan(a, b): return np.sum(np.abs(a - b))

def dist_cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return np.inf if na == 0 or nb == 0 else 1 - (a @ b) / (na * nb)

def dist_correlation(a, b):
    a0, b0 = a - a.mean(), b - b.mean()
    na, nb = np.linalg.norm(a0), np.linalg.norm(b0)
    return np.inf if na == 0 or nb == 0 else 1 - (a0 @ b0) / (na * nb)

metric_map = {
    "euclidean": dist_euclidean,
    "manhattan": dist_manhattan,
    "cosine": dist_cosine,
    "correlation": dist_correlation
}

# -----------------------------
# Alpha optimization
# -----------------------------
def best_alpha_euclidean(x, y, t):
    d = x - y
    denom = d @ d
    if denom == 0:
        return 0.5
    a_star = ((t - y) @ d) / denom
    return float(np.clip(a_star, 0.0, 1.0))


def best_alpha_nongeuclidean(x, y, t, metric_func):
    def f(a):
        combo = a * x + (1 - a) * y
        return metric_func(t, combo)

    res = minimize_scalar(f, bounds=(0, 1), method="bounded")
    return float(res.x), float(res.fun)

# -----------------------------
# Core computation
# -----------------------------
def run_two_way(metric, top_n, num_pairs, rng, labels_pool, X_pool, t):

    metric_func = metric_map[metric]
    results = {}
    n_pool = len(labels_pool)

    for _ in range(num_pairs):
        i, j = rng.choice(n_pool, size=2, replace=False)
        p1, p2 = labels_pool[i], labels_pool[j]
        key = tuple(sorted((p1, p2)))
        x, y = X_pool[i], X_pool[j]

        if metric == "euclidean":
            a = best_alpha_euclidean(x, y, t)
            combo = a * x + (1 - a) * y
            d = dist_euclidean(t, combo)
        else:
            a, d = best_alpha_nongeuclidean(x, y, t, metric_func)

        if key not in results or d < results[key][0]:
            results[key] = (d, p1, p2, a, 1 - a)

    best = sorted(results.values(), key=lambda z: z[0])[:top_n]

    df_out = pd.DataFrame(best, columns=["Distance", "Pop1", "Pop2", "Prop1", "Prop2"])

    # format %
    df_out["Prop1"] = (df_out["Prop1"] * 100).map(lambda x: f"{x:.2f}%")
    df_out["Prop2"] = (df_out["Prop2"] * 100).map(lambda x: f"{x:.2f}%")

    return df_out.reset_index(drop=True)

# -----------------------------
# Run button
# -----------------------------
if st.button("Run 2-Way Admixture"):

    rng = np.random.default_rng(42)

    # target
    t_idx = np.where(labels == target_label)[0][0]
    t = X[t_idx]

    # pool
    mask = np.ones(len(labels), dtype=bool)
    mask[t_idx] = False
    labels_pool = labels[mask]
    X_pool = X[mask]

    for m in metrics:

        st.subheader(f"{m.upper()}")

        with st.spinner(f"Running {m}..."):

            df_res = run_two_way(m, top_n, num_pairs, rng, labels_pool, X_pool, t)

            styled = (
                df_res.style
                .format({"Distance": "{:.6f}"})
                .background_gradient(subset=["Distance"], cmap="RdYlGn_r")
            )

            st.table(styled)
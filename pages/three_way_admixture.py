import streamlit as st
import pandas as pd
import numpy as np

# optional numba
try:
    from numba import njit
    NUMBA = True
except:
    NUMBA = False

st.set_page_config(layout="wide")

# -----------------------------
# Title
# -----------------------------
st.title("🧪 3-Way Admixture (Exact Solver)")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Settings")

dataset_type = st.sidebar.radio(
    "Source dataset",
    ["MODERN", "ANCIENTS", "BOTH"]
)

top_n = st.sidebar.slider("Top results", 5, 50, 20)

min_prop = st.sidebar.slider("Minimum component %", 0, 20, 5) / 100

use_preselect = st.sidebar.checkbox("⚡ Use preselection", value=False)
top_k = st.sidebar.slider("Top K (if preselect)", 50, 500, 150)

# -----------------------------
# Load data
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
# Target
# -----------------------------
target_mode = st.radio(
    "Target mode",
    ["Use dataset", "Paste G25 coordinates"],
    horizontal=True
)

if target_mode == "Use dataset":
    target_label = st.selectbox("Target", labels_mod)
    t = X_mod[np.where(labels_mod == target_label)[0][0]]

else:
    coord_text = st.text_area("Paste G25 row")

    if st.button("Load coordinates"):
        parts = [x.strip() for x in coord_text.split(",")]
        st.session_state["t"] = np.array([float(x) for x in parts[1:]])
        st.session_state["target_label"] = parts[0]

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
# Preselect
# -----------------------------
if use_preselect:
    st.warning("⚠️ Preselection ON")
    dists = np.linalg.norm(X_pool - t, axis=1)
    idx = np.argsort(dists)[:top_k]
    X_pool = X_pool[idx]
    labels_pool = labels_pool[idx]

# -----------------------------
# NUMBA CORE
# -----------------------------
if NUMBA:

    @njit
    def solve_3way(X, t, min_prop):
        n, d = X.shape
        results = []

        for i in range(n - 2):
            A = X[i]
            for j in range(i + 1, n - 1):
                B = X[j]
                for k in range(j + 1, n):
                    C = X[k]

                    u = A - C
                    v = B - C
                    w = t - C

                    uu = (u * u).sum()
                    uv = (u * v).sum()
                    vv = (v * v).sum()
                    wu = (w * u).sum()
                    wv = (w * v).sum()

                    det = uu * vv - uv * uv

                    if det == 0:
                        continue

                    a = (wu * vv - wv * uv) / det
                    b = (wv * uu - wu * uv) / det
                    c = 1.0 - a - b

                    if a < min_prop or b < min_prop or c < min_prop:
                        continue

                    # distance
                    mix = a * A + b * B + c * C
                    dist = np.sqrt(((mix - t) ** 2).sum())

                    results.append((dist, i, j, k, a, b, c))

        return results

else:
    def solve_3way(X, t, min_prop):
        n = len(X)
        results = []

        for i in range(n - 2):
            A = X[i]
            for j in range(i + 1, n - 1):
                B = X[j]
                for k in range(j + 1, n):
                    C = X[k]

                    u = A - C
                    v = B - C
                    w = t - C

                    uu = np.dot(u, u)
                    uv = np.dot(u, v)
                    vv = np.dot(v, v)
                    wu = np.dot(w, u)
                    wv = np.dot(w, v)

                    det = uu * vv - uv * uv
                    if det == 0:
                        continue

                    a = (wu * vv - wv * uv) / det
                    b = (wv * uu - wu * uv) / det
                    c = 1 - a - b

                    if a < min_prop or b < min_prop or c < min_prop:
                        continue

                    mix = a*A + b*B + c*C
                    dist = np.linalg.norm(mix - t)

                    results.append((dist, i, j, k, a, b, c))

        return results

# -----------------------------
# RUN
# -----------------------------
if st.button("Run 3-Way Admixture"):

    with st.spinner("Computing..."):
        results = solve_3way(X_pool, t, min_prop)

    if len(results) == 0:
        st.warning("No valid combinations found")
        st.stop()

    # sort
    results.sort(key=lambda x: x[0])
    results = results[:top_n]

    rows = []
    for d, i, j, k, a, b, c in results:
        rows.append({
            "Distance": d,
            "Pop1": labels_pool[i],
            "Pop2": labels_pool[j],
            "Pop3": labels_pool[k],
            "Prop1": a * 100,
            "Prop2": b * 100,
            "Prop3": c * 100
        })

    df = pd.DataFrame(rows)

    st.dataframe(df)

    # -----------------------------
    # Bars
    # -----------------------------
    st.subheader("📊 Breakdown")

    for _, row in df.iterrows():

        st.markdown(f"**{row['Pop1']} + {row['Pop2']} + {row['Pop3']}**")

        st.progress(int(row["Prop1"]))
        st.caption(f"{row['Pop1']} ({row['Prop1']:.2f}%)")

        st.progress(int(row["Prop2"]))
        st.caption(f"{row['Pop2']} ({row['Prop2']:.2f}%)")

        st.progress(int(row["Prop3"]))
        st.caption(f"{row['Pop3']} ({row['Prop3']:.2f}%)")

        st.markdown("---")
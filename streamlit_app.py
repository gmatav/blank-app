import streamlit as st

import streamlit as st

# -----------------------------
# Page config (must be first)
# -----------------------------
st.set_page_config(
    page_title="Genetic Analysis Toolkit",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Title / Header
# -----------------------------
st.title("🧬 Genetic Analysis Toolkit")

st.markdown("""
Welcome to your genetic analysis environment.

Use the sidebar to navigate between tools, or jump directly below:
""")

# -----------------------------
# Navigation buttons (nice UX)
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("📍 Closest Populations"):
        st.switch_page("pages/Closest_Populations.py")

    if st.button("🧪 2-Way Admixture"):
        st.switch_page("pages/2-way-admixture.py")

with col2:
    if st.button("🧪 3-Way Admixture"):
        st.switch_page("pages/3_Admixture_3_Way.py")

    if st.button("⚙️ Custom Admixture"):
        st.switch_page("pages/4_Custom_Admixture.py")


# -----------------------------
# Info / Footer
# -----------------------------
st.markdown("---")

st.markdown("""
### 🧠 About this app

- Explore genetic distances
- Compute admixture models
- Build custom ancestry profiles

⚠️ Always remember to save your work (git commit + push)
""")